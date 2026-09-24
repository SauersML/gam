//! The smoothing-corrected posterior of a constrained fit: the θ-mixture of its
//! cone-truncated laws (gam#3229).
//!
//! # The target
//!
//! A smoothing-corrected covariance reports the θ-mixture variance of the
//! coefficients, which to first order in `V_θ = Var(θ̂)` is
//!
//! ```text
//!   Var(β | y) = E_θ[V(θ)] + Var_θ(m(θ))
//!              ≈ V(θ̂) + J_m V_θ J_mᵀ + ½ Σ_jk V_θ[j,k] ∂²V/∂θ_j∂θ_k,
//! ```
//!
//! with `m(θ)` and `V(θ)` the REPORTED posterior mean and covariance at `θ`. At a
//! constrained mode those are the moments of the cone-truncated Gaussian
//! `TN(c(θ), Σ(θ))`, and both derivatives are cumulant objects of that law: `J_m`
//! and `∂²V` need its moments to order six once more than one row is retained,
//! and the orthant cubature serves order two.
//!
//! # The object, rather than its expansion
//!
//! The expansion is the second-order Taylor polynomial of `E_θ[g(θ)]` for
//! `g = V + (m − m̄)(m − m̄)ᵀ`, and a node rule that integrates every polynomial of
//! degree three in `δ = θ − θ̂` exactly against `N(0, V_θ)` reproduces it to the
//! same order. The symmetric rule does, with no tuning constant: with
//! `V_θ = Σ_m σ_m v_m v_mᵀ` over its `r` resolved directions, the `2r` nodes
//! `δ = ±√(r σ_m) v_m` at weight `1/(2r)` have mean zero, second moment
//! `(1/2r)·2·Σ_m r σ_m v_m v_mᵀ = V_θ`, and vanishing odd moments. Each node is
//! one truncated law, whose moments the cubature already serves for any number
//! of retained rows. So the mixture over the nodes IS the smoothing-corrected
//! posterior, at every `q`, and it is one object: the fit publishes its
//! covariance and a predictor reads its law, so the two cannot describe
//! different posteriors (the divergence gam#3229 opened on).
//!
//! # Each node's law
//!
//! The ambient Gaussian at `θ` is the quadratic model of the penalized objective
//! at the mode, with the data curvature held at the mode and the penalties
//! carrying `θ` exactly. Along the ρ block `Ṁ_k = λ_k S̃_k = D_k` and
//! `M(θ) = M̂ + Σ_k (e^{δ_k} − 1) D_k`, the same identity
//! `crate::estimate::smoothing_curvature` differentiates at an interior mode;
//! the model's minimizer is `c(θ) = M(θ)⁻¹ M̂ ĉ`, since its gradient
//! `M̂(β − ĉ) + Σ_k (λ_k − λ̂_k) S̃_k β` vanishes there. The node's ambient
//! covariance is `φ·M(θ)⁻¹`. Every `M(θ)` is positive definite whenever `M̂`
//! is: `e^{δ_k} − 1 > −1` and each `D_k ⪰ 0`, so
//! `M(θ) ⪰ min(1, min_k e^{δ_k})·M̂`.
//!
//! At an interior mode the truncation is the identity and the expansion of
//! the mixture is `Σ̂ + J V_θ Jᵀ + ½ Σ V_θ ∂²Σ`, exactly the first-order
//! correction and the curvature term the interior routes carry; as a bound
//! activates, the same object moves continuously onto the cone.
//!
//! A design (ψ) coordinate has no `D_k`: its `M̈` is the family's own second
//! design derivative, published by no family, so a mixture over one is not
//! built and the caller reports the typed absence.

//! # What is stored, and what it costs
//!
//! The mixture is persisted as what rebuilds it, not as its node covariances:
//! the ambient centre `ĉ`, the scale `φ`, the drifts in root form
//! `D_k = R_kᵀR_k` (`rank_k × p` each, the shape the penalty roots already
//! have), and the `2r` node offsets. A node's precision is
//! `M̂ + Σ_k expm1(δ_k) R_kᵀR_k` against the fit's own penalized Hessian `M̂`, so
//! a dense route factors it directly and a factorized route factors it with the
//! machinery that already factors `M̂`; both describe one law. Beside those it
//! stores the mixture's mean, and its covariance on a route that forms dense
//! covariances. A factorized route publishes the mixture's standard errors from
//! the same node truncations ([`SmoothingMixture::node_truncations`]) in its own
//! published frame.
//!
//! The cost is the node count: `2r` factorizations of a `p × p` precision, with
//! `r ≤ K` the resolved directions of `V_θ`, and on a factorized route `2r`
//! passes of the inverse-diagonal solve the conditional standard errors already
//! take. Each build logs `r` and the node count.

use super::{
    ConstrainedPosteriorCorrection, ConstrainedPosteriorGeometry, ConstrainedProjectionLaw,
    MixtureQuantile, NormalNodes, constrained_posterior_correction,
    constrained_posterior_correction_from_covariance, normal_cdf_and_pdf, normal_logsf,
    scalar_truncated_moments, standard_normal_quantile,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// The smoothing-corrected posterior of a constrained fit, as the equally
/// weighted mixture of its node laws (see the module doc), in the constrained
/// posterior's own coefficient frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmoothingMixture {
    /// `ĉ`, the ambient centre at the mode.
    center: Array1<f64>,
    /// `φ`, the scale each node's covariance `φ·M_i⁻¹` carries.
    covariance_scale: f64,
    /// `R_k` with `D_k = λ̂_k S̃_k = R_kᵀR_k`, one per θ coordinate.
    drift_roots: Vec<Array2<f64>>,
    /// `δ_i`, one per node, each over the θ coordinates.
    offsets: Vec<Array1<f64>>,
    /// `Σ_i w_i m_i`.
    mean: Array1<f64>,
    /// `Σ_i w_i [V_i + (m_i − m̄)(m_i − m̄)ᵀ]` on a route that forms dense
    /// covariances, the covariance it publishes; `None` on a factorized one,
    /// which publishes the standard errors of the same law in its own frame.
    covariance: Option<Array2<f64>>,
}

/// What [`SmoothingMixture::build`] reads, all in the constrained posterior's
/// coefficient frame.
pub struct SmoothingMixtureInputs<'a> {
    /// `ĉ`, the ambient centre at the mode.
    pub center: ArrayView1<'a, f64>,
    /// `M̂ĉ`, the right-hand side every node's centre `M_i⁻¹M̂ĉ` solves.
    pub precision_center: ArrayView1<'a, f64>,
    /// `φ`, the scale `Σ = φ·M⁻¹` carries.
    pub covariance_scale: f64,
    /// `R_k` with `D_k = R_kᵀR_k`, one per θ coordinate, in the order of
    /// `rho_covariance` ([`drift_root`] forms one from a dense drift).
    pub drift_roots: &'a [Array2<f64>],
    /// `V_θ`, positive semidefinite; a direction it does not resolve carries no
    /// θ variance and contributes no node.
    pub rho_covariance: ArrayView2<'a, f64>,
    /// The fit's inequality system `Aβ ≥ b`.
    pub constraints: &'a LinearInequalityConstraints,
}

/// One node's precision `M̂ + Σ_k w_k R_kᵀR_k`, factored by whoever owns its
/// frame, as far as a truncation reads it: solves.
pub trait NodeSolve {
    /// `M_i⁻¹·rhs`.
    fn solve(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String>;
}

/// A node precision a fit publishes from: its solves, and its inverse on a
/// route that forms dense covariances.
pub trait NodePrecision: NodeSolve {
    /// `M_i⁻¹` on a route that forms dense covariances, `None` on a factorized
    /// one.
    fn dense_inverse(&self) -> Option<Result<Array2<f64>, String>>;
}

/// A node precision formed and factored densely.
pub struct DenseNodePrecision {
    inverse: Array2<f64>,
}

impl DenseNodePrecision {
    /// `M̂ + Σ_k weights[k]·R_kᵀR_k`, factored by Cholesky and inverted. Every
    /// node precision is positive definite whenever `M̂` is (module doc), so a
    /// failed factor is a malformed input and is reported as one.
    pub fn factor(
        precision: ArrayView2<'_, f64>,
        drift_roots: &[Array2<f64>],
        weights: &[f64],
    ) -> Result<Self, String> {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let p = precision.nrows();
        if drift_roots.len() != weights.len() {
            return Err(format!(
                "smoothing mixture node: {} drift root(s) for {} weight(s)",
                drift_roots.len(),
                weights.len()
            ));
        }
        let mut node_precision = precision.to_owned();
        for (root, &weight) in drift_roots.iter().zip(weights) {
            if root.ncols() != p {
                return Err(format!(
                    "smoothing mixture node: a drift root has {} columns in a {p}-coefficient \
                     frame",
                    root.ncols()
                ));
            }
            node_precision.scaled_add(weight, &root.t().dot(root));
        }
        gam_linalg::matrix::symmetrize_in_place(&mut node_precision);
        let factor = node_precision.cholesky(Side::Lower).map_err(|error| {
            format!("smoothing mixture node precision is not positive definite: {error:?}")
        })?;
        let mut inverse = factor.solve_mat(&Array2::<f64>::eye(p));
        gam_linalg::matrix::symmetrize_in_place(&mut inverse);
        Ok(Self { inverse })
    }
}

impl NodeSolve for DenseNodePrecision {
    fn solve(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        if rhs.nrows() != self.inverse.nrows() {
            return Err(format!(
                "smoothing mixture node: a {}-row right-hand side in a {}-coefficient frame",
                rhs.nrows(),
                self.inverse.nrows()
            ));
        }
        Ok(self.inverse.dot(rhs))
    }
}

impl NodePrecision for DenseNodePrecision {
    fn dense_inverse(&self) -> Option<Result<Array2<f64>, String>> {
        Some(Ok(self.inverse.clone()))
    }
}

/// `R` with `RᵀR = D` for a symmetric positive semidefinite drift `D`, read
/// off `D`'s own eigensystem with the eigenvalues its decomposition resolves
/// (`gam_linalg::roundoff::resolved_eigenvalue_band`).
///
/// A drift is a sum of admitted penalties `Σ λ_k S_k`, so it is refused as not positive
/// semidefinite by the rule those penalties were admitted by
/// ([`gam_problem::penalty_matrix::psd_admission_band`]), not by the eigensolver's band alone:
/// that band is a hundred times tighter, and a binomial `flexible(loglog)` link-wiggle drift was
/// refused for an eigenvalue of `−9.35e−13` against `8.94e−13`, curvature its penalties carried
/// when they were admitted (#2155).
pub fn drift_root(drift: &Array2<f64>) -> Result<Array2<f64>, String> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let p = drift.nrows();
    if drift.ncols() != p {
        return Err(format!(
            "smoothing mixture drift is {:?}, not square",
            drift.dim()
        ));
    }
    let mut symmetric = drift.clone();
    gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
    let (eigenvalues, eigenvectors) = symmetric
        .eigh(Side::Lower)
        .map_err(|error| format!("smoothing mixture drift eigendecomposition: {error:?}"))?;
    let eigenvalues = eigenvalues.to_vec();
    let band = gam_linalg::roundoff::resolved_eigenvalue_band(&eigenvalues, 0.0);
    let max_abs = eigenvalues.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let admitted = gam_problem::penalty_matrix::psd_admission_band(p, max_abs);
    if let Some(&negative) = eigenvalues.iter().find(|&&value| value < -admitted) {
        return Err(format!(
            "smoothing mixture drift has eigenvalue {negative:.6e} below the negative curvature a \
             penalty is admitted with, {admitted:.6e}"
        ));
    }
    let kept: Vec<usize> = (0..p).filter(|&index| eigenvalues[index] > band).collect();
    let mut root = Array2::<f64>::zeros((kept.len(), p));
    for (row, &index) in kept.iter().enumerate() {
        let scale = eigenvalues[index].sqrt();
        for column in 0..p {
            root[[row, column]] = scale * eigenvectors[[column, index]];
        }
    }
    Ok(root)
}

/// One node's truncation as a factorized route reads it: its centre
/// `M_i⁻¹M̂ĉ`, its correction from `Σ_iAᵀ` by solves, and its truncated mean.
pub struct NodeTruncation {
    pub center: Array1<f64>,
    pub correction: Option<ConstrainedPosteriorCorrection>,
    pub mean: Array1<f64>,
}

/// One node's truncated moments.
struct NodeMoments {
    mean: Array1<f64>,
    covariance: Option<Array2<f64>>,
}

impl SmoothingMixture {
    /// Build the mixture of a constrained fit. `factor_node` factors the node
    /// precision `M̂ + Σ_k w_k R_kᵀR_k` for the weights it is handed, in the
    /// route's own way ([`DenseNodePrecision`] on a dense route). Fails when an
    /// input is malformed, when `V_θ` is materially indefinite, or when a node's
    /// truncated moments cannot be computed; the caller reports the typed
    /// absence, never a frozen-θ substitute.
    pub fn build<P: NodePrecision>(
        inputs: SmoothingMixtureInputs<'_>,
        factor_node: impl Fn(&[f64]) -> Result<P, String>,
    ) -> Result<Self, String> {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let SmoothingMixtureInputs {
            center,
            precision_center,
            covariance_scale,
            drift_roots,
            rho_covariance,
            constraints,
        } = inputs;
        let p = center.len();
        let k = drift_roots.len();
        if precision_center.len() != p || constraints.a.ncols() != p {
            return Err(format!(
                "smoothing mixture: a centre of length {p}, a precision-centre of length {} and \
                 a constraint system of {} columns do not share one coefficient frame",
                precision_center.len(),
                constraints.a.ncols()
            ));
        }
        if rho_covariance.dim() != (k, k) {
            return Err(format!(
                "smoothing mixture: {k} drift root(s) against a {:?} theta covariance",
                rho_covariance.dim()
            ));
        }
        if let Some(index) = drift_roots.iter().position(|root| root.ncols() != p) {
            return Err(format!(
                "smoothing mixture: drift root {index} has {} columns in a {p}-coefficient frame",
                drift_roots[index].ncols()
            ));
        }
        if !(covariance_scale.is_finite() && covariance_scale > 0.0) {
            return Err(format!(
                "smoothing mixture: the covariance scale must be positive and finite, got \
                 {covariance_scale}"
            ));
        }
        if center
            .iter()
            .chain(precision_center.iter())
            .chain(rho_covariance.iter())
            .chain(drift_roots.iter().flat_map(|root| root.iter()))
            .any(|value| !value.is_finite())
        {
            return Err("smoothing mixture: an input is not finite".to_string());
        }

        let mut theta_covariance = rho_covariance.to_owned();
        gam_linalg::matrix::symmetrize_in_place(&mut theta_covariance);
        let mut offsets = Vec::new();
        let mut resolved_rank = 0usize;
        if k > 0 {
            let (eigenvalues, eigenvectors) = theta_covariance
                .eigh(Side::Lower)
                .map_err(|error| format!("smoothing mixture: theta covariance: {error:?}"))?;
            let eigenvalues = eigenvalues.to_vec();
            let band = gam_linalg::roundoff::resolved_eigenvalue_band(&eigenvalues, 0.0);
            if let Some(&negative) = eigenvalues.iter().find(|&&value| value < -band) {
                return Err(format!(
                    "smoothing mixture: the theta covariance has eigenvalue {negative:.6e} below \
                     its own rounding band {band:.6e}"
                ));
            }
            let resolved: Vec<usize> = (0..k).filter(|&index| eigenvalues[index] > band).collect();
            resolved_rank = resolved.len();
            let rank = resolved_rank as f64;
            for &index in &resolved {
                let step = eigenvectors
                    .column(index)
                    .mapv(|value| value * (rank * eigenvalues[index]).sqrt());
                offsets.push(step.clone());
                offsets.push(step.mapv(|value| -value));
            }
        }
        if offsets.is_empty() {
            offsets.push(Array1::zeros(k));
        }
        log::debug!(
            "[smoothing-mixture] theta_dimension={k} resolved_directions={resolved_rank} nodes={} \
             (gam#3229): one node-precision factorization per node",
            offsets.len()
        );

        let precision_center = precision_center.to_owned();
        let constraints_transpose = constraints.a.t().to_owned();
        let mut moments = Vec::with_capacity(offsets.len());
        for (index, offset) in offsets.iter().enumerate() {
            let weights: Vec<f64> = offset.iter().map(|delta| delta.exp_m1()).collect();
            let node = factor_node(&weights)
                .and_then(|node| {
                    node_moments(
                        &node,
                        &precision_center,
                        &constraints_transpose,
                        covariance_scale,
                        constraints,
                    )
                })
                .map_err(|reason| format!("smoothing mixture node {index}: {reason}"))?;
            moments.push(node);
        }

        let weight = 1.0 / moments.len() as f64;
        let mut mean = Array1::<f64>::zeros(p);
        for node in &moments {
            mean.scaled_add(weight, &node.mean);
        }
        let covariance = if moments.iter().all(|node| node.covariance.is_some()) {
            let mut covariance = Array2::<f64>::zeros((p, p));
            for node in &moments {
                if let Some(node_covariance) = node.covariance.as_ref() {
                    covariance.scaled_add(weight, node_covariance);
                }
                let spread = &node.mean - &mean;
                for a in 0..p {
                    for b in 0..=a {
                        covariance[[a, b]] += weight * spread[a] * spread[b];
                    }
                }
            }
            for a in 0..p {
                for b in 0..a {
                    covariance[[b, a]] = covariance[[a, b]];
                }
            }
            Some(covariance)
        } else {
            None
        };
        Ok(Self {
            center: center.to_owned(),
            covariance_scale,
            drift_roots: drift_roots.to_vec(),
            offsets,
            mean,
            covariance,
        })
    }

    /// The number of nodes.
    pub fn node_count(&self) -> usize {
        self.offsets.len()
    }

    /// The mixture's mean.
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// The mixture's covariance, on a route that formed it densely.
    pub fn covariance(&self) -> Option<&Array2<f64>> {
        self.covariance.as_ref()
    }

    /// Each node's law, rebuilt densely against `precision` (the fit's `M̂`) as
    /// its ambient covariance beside a constrained posterior on `base`'s cone:
    /// the form a projection or a sampler reads one truncated law in.
    pub fn node_laws(
        &self,
        precision: ArrayView2<'_, f64>,
        base: &ConstrainedPosteriorGeometry,
    ) -> Result<Vec<(Array2<f64>, ConstrainedPosteriorGeometry)>, String> {
        let p = self.center.len();
        if precision.dim() != (p, p) {
            return Err(format!(
                "smoothing mixture: a {:?} precision for a {p}-coefficient mixture",
                precision.dim()
            ));
        }
        let precision_center = precision.dot(&self.center);
        self.offsets
            .iter()
            .map(|offset| {
                let weights: Vec<f64> = offset.iter().map(|delta| delta.exp_m1()).collect();
                let node = DenseNodePrecision::factor(precision, &self.drift_roots, &weights)?;
                let (covariance, center, correction) = dense_node_law(
                    &node.inverse,
                    &precision_center,
                    self.covariance_scale,
                    &base.constraints,
                )?;
                let geometry = ConstrainedPosteriorGeometry::with_moments(
                    base.constraints.clone(),
                    base.mode.clone(),
                    center,
                    correction,
                );
                Ok((covariance, geometry))
            })
            .collect()
    }

    /// The scale each node's covariance `φ·M_i⁻¹` carries.
    pub fn covariance_scale(&self) -> f64 {
        self.covariance_scale
    }

    /// Node `index`'s precision `M̂ + Σ_k expm1(δ_k) R_kᵀR_k` against the fit's
    /// `precision`, formed densely for a caller that factors it itself.
    pub fn node_precision(
        &self,
        precision: ArrayView2<'_, f64>,
        index: usize,
    ) -> Result<Array2<f64>, String> {
        let offset = self.offsets.get(index).ok_or_else(|| {
            format!(
                "smoothing mixture has {} node(s), not a node {index}",
                self.offsets.len()
            )
        })?;
        let p = self.center.len();
        if precision.dim() != (p, p) {
            return Err(format!(
                "smoothing mixture: a {:?} precision for a {p}-coefficient mixture",
                precision.dim()
            ));
        }
        let mut node_precision = precision.to_owned();
        for (root, delta) in self.drift_roots.iter().zip(offset.iter()) {
            node_precision.scaled_add(delta.exp_m1(), &root.t().dot(root));
        }
        gam_linalg::matrix::symmetrize_in_place(&mut node_precision);
        Ok(node_precision)
    }

    /// Every node's truncation as a factorized route reads it, beside the node
    /// solver `factor_node` built for node `index` from
    /// [`Self::node_precision`]: what a predictor serving a factorized fit's
    /// mixture composes its per-node backends from. `precision` is the fit's
    /// `M̂`, which forms `M̂ĉ`.
    pub fn node_truncations<S: NodeSolve>(
        &self,
        precision: ArrayView2<'_, f64>,
        constraints: &LinearInequalityConstraints,
        factor_node: impl Fn(usize) -> Result<S, String>,
    ) -> Result<Vec<(S, NodeTruncation)>, String> {
        let p = self.center.len();
        if precision.dim() != (p, p) || constraints.a.ncols() != p {
            return Err(format!(
                "smoothing mixture: a {:?} precision and {} constraint columns for a \
                 {p}-coefficient mixture",
                precision.dim(),
                constraints.a.ncols()
            ));
        }
        let precision_center = precision.dot(&self.center);
        let constraints_transpose = constraints.a.t().to_owned();
        (0..self.offsets.len())
            .map(|index| {
                let node = factor_node(index)?;
                let truncation = solved_node_truncation(
                    &node,
                    &precision_center,
                    &constraints_transpose,
                    self.covariance_scale,
                    constraints,
                )
                .map_err(|reason| format!("smoothing mixture node {index}: {reason}"))?;
                Ok((node, truncation))
            })
            .collect()
    }

    /// Structural validation against the coefficient dimension it is read in.
    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), String> {
        let k = self.drift_roots.len();
        if self.offsets.is_empty() {
            return Err("smoothing mixture carries no node".to_string());
        }
        if self.center.len() != dimension
            || self.mean.len() != dimension
            || self
                .covariance
                .as_ref()
                .is_some_and(|covariance| covariance.dim() != (dimension, dimension))
        {
            return Err(format!(
                "smoothing mixture moments are not all in a {dimension}-coefficient frame"
            ));
        }
        if self
            .drift_roots
            .iter()
            .any(|root| root.ncols() != dimension)
            || self.offsets.iter().any(|offset| offset.len() != k)
        {
            return Err(format!(
                "smoothing mixture drift roots and node offsets do not share {k} theta \
                 coordinate(s) in a {dimension}-coefficient frame"
            ));
        }
        if !(self.covariance_scale.is_finite() && self.covariance_scale > 0.0)
            || self
                .center
                .iter()
                .chain(self.mean.iter())
                .chain(self.drift_roots.iter().flat_map(|root| root.iter()))
                .chain(self.offsets.iter().flat_map(|offset| offset.iter()))
                .chain(
                    self.covariance
                        .iter()
                        .flat_map(|covariance| covariance.iter()),
                )
                .any(|value| !value.is_finite())
        {
            return Err("smoothing mixture contains a non-finite value".to_string());
        }
        Ok(())
    }
}

/// One node's dense law: its ambient covariance `φM_i⁻¹`, its centre
/// `M_i⁻¹M̂ĉ`, and its truncation. The fit that publishes the mixture and the
/// predictor that reads it both form a dense node here, so they form one law.
fn dense_node_law(
    inverse: &Array2<f64>,
    precision_center: &Array1<f64>,
    covariance_scale: f64,
    constraints: &LinearInequalityConstraints,
) -> Result<
    (
        Array2<f64>,
        Array1<f64>,
        Option<ConstrainedPosteriorCorrection>,
    ),
    String,
> {
    let ambient = inverse.mapv(|value| covariance_scale * value);
    let center = inverse.dot(precision_center);
    let correction =
        constrained_posterior_correction_from_covariance(&ambient, &center, constraints)?;
    Ok((ambient, center, correction))
}

/// The truncated moments of one node's law. A dense node is
/// [`dense_node_law`] with its truncated covariance in the sum-of-Grams form. A
/// factorized node reads its centre and `Σ_iAᵀ` by solves, all the truncation
/// reads, and contributes its mean; its variances are published by its route
/// in the route's own frame.
fn node_moments(
    node: &impl NodePrecision,
    precision_center: &Array1<f64>,
    constraints_transpose: &Array2<f64>,
    covariance_scale: f64,
    constraints: &LinearInequalityConstraints,
) -> Result<NodeMoments, String> {
    if let Some(inverse) = node.dense_inverse() {
        let (ambient, center, correction) =
            dense_node_law(&inverse?, precision_center, covariance_scale, constraints)?;
        let (mean, covariance) = match correction.as_ref() {
            Some(correction) => (
                correction.posterior_mean(&center),
                correction.truncated_covariance_psd(&ambient, constraints)?,
            ),
            None => (center, ambient),
        };
        return Ok(NodeMoments {
            mean,
            covariance: Some(covariance),
        });
    }
    let truncation = solved_node_truncation(
        node,
        precision_center,
        constraints_transpose,
        covariance_scale,
        constraints,
    )?;
    Ok(NodeMoments {
        mean: truncation.mean,
        covariance: None,
    })
}

/// A node's truncation read through its solves: the path a factorized fit
/// publishes by and a factorized predictor rebuilds by, so the two read one
/// construction.
fn solved_node_truncation(
    node: &impl NodeSolve,
    precision_center: &Array1<f64>,
    constraints_transpose: &Array2<f64>,
    covariance_scale: f64,
    constraints: &LinearInequalityConstraints,
) -> Result<NodeTruncation, String> {
    let center = node
        .solve(&precision_center.clone().insert_axis(ndarray::Axis(1)))?
        .column(0)
        .to_owned();
    let sigma_at = node
        .solve(constraints_transpose)?
        .mapv(|value| covariance_scale * value);
    let correction = constrained_posterior_correction(sigma_at.view(), &center, constraints)?;
    let mean = correction.as_ref().map_or_else(
        || center.clone(),
        |correction| correction.posterior_mean(&center),
    );
    Ok(NodeTruncation {
        center,
        correction,
        mean,
    })
}

/// Every scalar projection `cᵀβ` of an equally weighted mixture of truncated
/// laws: the smoothing-corrected law of a constrained fit, one
/// [`ConstrainedProjectionLaw`] per node.
///
/// The projection's CDF is the mean of the nodes' CDFs, and each node's CDF is
/// the one its own law already defines (the ambient normal, the closed-form
/// scalar truncation, or its certified orthant nodes with or without the tangent
/// residual). The equal-tailed quantiles are settled on that mean by the same
/// safeguarded Newton the single-law route uses, at the resolution it states.
pub struct ConstrainedMixtureProjectionLaw<'a> {
    components: Vec<ConstrainedProjectionLaw<'a>>,
}

/// One node's projection law at one contrast.
enum ComponentProjection<'n> {
    /// `N(mean, sd²)`; `sd = 0` is a point mass.
    Normal { mean: f64, sd: f64 },
    /// `ambient_mean + lift·(u − center)`, `u ~ TN(center, variance)` on
    /// `[0, upper]`.
    Scalar {
        ambient_mean: f64,
        lift: f64,
        center: f64,
        variance: f64,
        upper: f64,
    },
    /// The orthant nodes' conditional means with weights, convolved with an
    /// independent `N(0, residual_sd²)`; a zero residual is the step law.
    Nodes {
        means: Array1<f64>,
        weights: &'n Array1<f64>,
        residual_sd: f64,
    },
}

impl ComponentProjection<'_> {
    /// `(F(t), f(t))`, with `f = 0` where the law has no density.
    fn cdf_and_density(&self, point: f64) -> (f64, f64) {
        match *self {
            Self::Normal { mean, sd } => {
                if sd > 0.0 {
                    let (cdf, density) = normal_cdf_and_pdf((point - mean) / sd);
                    (cdf, density / sd)
                } else {
                    (if point >= mean { 1.0 } else { 0.0 }, 0.0)
                }
            }
            Self::Scalar {
                ambient_mean,
                lift,
                center,
                variance,
                upper,
            } => {
                let value = center + (point - ambient_mean) / lift;
                let (cdf, density) =
                    scalar_truncated_cdf_and_density(center, variance, upper, value);
                if lift > 0.0 {
                    (cdf, density / lift)
                } else {
                    (1.0 - cdf, density / -lift)
                }
            }
            Self::Nodes {
                ref means,
                weights,
                residual_sd,
            } => {
                let mut cdf = 0.0;
                let mut density = 0.0;
                for (&mean, &weight) in means.iter().zip(weights.iter()) {
                    if residual_sd > 0.0 {
                        let (node_cdf, node_density) =
                            normal_cdf_and_pdf((point - mean) / residual_sd);
                        cdf += weight * node_cdf;
                        density += weight * node_density / residual_sd;
                    } else if point >= mean {
                        cdf += weight;
                    }
                }
                (cdf, density)
            }
        }
    }
}

/// `(F(v), f(v))` of `N(center, variance)` restricted to `[0, upper]`, in
/// survival space so a deeply truncated face never forms `1 − Φ` of a tail.
fn scalar_truncated_cdf_and_density(
    center: f64,
    variance: f64,
    upper: f64,
    value: f64,
) -> (f64, f64) {
    if value <= 0.0 {
        return (0.0, 0.0);
    }
    if value >= upper {
        return (1.0, 0.0);
    }
    let sd = variance.sqrt();
    let wall = -center / sd;
    let at = (value - center) / sd;
    let log_mass_above_wall = normal_logsf(wall);
    // `P(wall ≤ Z ≤ at) / P(wall ≤ Z ≤ top)`, each a difference of tail masses
    // formed as `Φ̄(wall)·(1 − e^{ln Φ̄(x) − ln Φ̄(wall)})`.
    let retained_to = -(normal_logsf(at) - log_mass_above_wall).exp_m1();
    let retained_total = if upper.is_finite() {
        -(normal_logsf((upper - center) / sd) - log_mass_above_wall).exp_m1()
    } else {
        1.0
    };
    let density_at = normal_cdf_and_pdf(at).1;
    let density = (density_at.ln() - log_mass_above_wall).exp() / (retained_total * sd);
    (retained_to / retained_total, density)
}

impl<'a> ConstrainedMixtureProjectionLaw<'a> {
    /// The mixture of `components`, equally weighted. At least one component.
    pub fn new(components: Vec<ConstrainedProjectionLaw<'a>>) -> Result<Self, String> {
        if components.is_empty() {
            return Err("a mixture projection law needs at least one component".to_string());
        }
        Ok(Self { components })
    }

    /// Equal-tailed intervals for the projections `contrasts.row(r)ᵀβ`, one per
    /// row, at `level`. A one-component mixture is that law, and reads it.
    pub fn equal_tailed_intervals(
        &self,
        contrasts: ArrayView2<'_, f64>,
        level: f64,
    ) -> Result<Vec<(f64, f64)>, String> {
        if let [single] = self.components.as_slice() {
            return single.equal_tailed_intervals(contrasts, level);
        }
        if !(level.is_finite() && level > 0.0 && level < 1.0) {
            return Err(format!(
                "constrained projection interval level must lie in (0, 1), got {level}"
            ));
        }
        let alpha = 0.5 * (1.0 - level);
        let z = standard_normal_quantile(1.0 - alpha)
            .map_err(|error| format!("constrained projection normal quantile: {error}"))?;
        let weight = 1.0 / self.components.len() as f64;
        let mut intervals = Vec::with_capacity(contrasts.nrows());
        for contrast in contrasts.outer_iter() {
            let mut pieces = Vec::with_capacity(self.components.len());
            let mut mean_sum = 0.0;
            let mut second_sum = 0.0;
            for component in &self.components {
                let (piece, mean, variance) = component.component_projection(contrast)?;
                mean_sum += weight * mean;
                second_sum += weight * (variance + mean * mean);
                pieces.push(piece);
            }
            // The moment-matched normal's quantiles start the iteration, as they
            // do for one law; its spread sets the resolution.
            let spread = (second_sum - mean_sum * mean_sum).max(0.0).sqrt();
            if spread == 0.0 {
                intervals.push((mean_sum, mean_sum));
                continue;
            }
            let mut quantiles = [
                MixtureQuantile::new(alpha, mean_sum - z * spread, spread),
                MixtureQuantile::new(1.0 - alpha, mean_sum + z * spread, spread),
            ];
            for quantile in quantiles.iter_mut() {
                while quantile.value.is_none() {
                    let mut cdf = 0.0;
                    let mut density = 0.0;
                    for piece in &pieces {
                        let (piece_cdf, piece_density) = piece.cdf_and_density(quantile.point);
                        cdf += weight * piece_cdf;
                        density += weight * piece_density;
                    }
                    quantile.advance(cdf, density)?;
                }
            }
            let [lower, upper] = &quantiles;
            match (lower.value, upper.value) {
                (Some(lower), Some(upper)) => intervals.push((lower, upper)),
                _ => {
                    return Err(
                        "mixture projection quantile iteration ended without a value".to_string(),
                    );
                }
            }
        }
        Ok(intervals)
    }
}

impl<'a> ConstrainedProjectionLaw<'a> {
    /// This law's projection at `contrast`, with its mean and variance.
    fn component_projection(
        &self,
        contrast: ArrayView1<'_, f64>,
    ) -> Result<(ComponentProjection<'_>, f64, f64), String> {
        let decomposition = self.decompose(contrast)?;
        let Some(truncated) = decomposition.truncated else {
            return Ok((
                ComponentProjection::Normal {
                    mean: decomposition.ambient_mean,
                    sd: decomposition.ambient_variance.sqrt(),
                },
                decomposition.ambient_mean,
                decomposition.ambient_variance,
            ));
        };
        let truncation = self.truncation.as_ref().ok_or_else(|| {
            "a truncated decomposition came from a law with no truncation".to_string()
        })?;
        if truncation.normal_center.len() == 1
            && truncated.residual_variance == 0.0
            && truncated.projection_lift[0] != 0.0
        {
            let center = truncation.normal_center[0];
            let variance = truncation.normal_covariance[[0, 0]];
            let upper = truncation.upper_limits[0];
            let lift = truncated.projection_lift[0];
            let (moments_mean, moments_covariance) =
                scalar_truncated_moments(center, variance, upper)?;
            return Ok((
                ComponentProjection::Scalar {
                    ambient_mean: decomposition.ambient_mean,
                    lift,
                    center,
                    variance,
                    upper,
                },
                decomposition.ambient_mean + lift * (moments_mean[0] - center),
                lift * lift * moments_covariance[[0, 0]],
            ));
        }
        let nodes: &NormalNodes = truncation.certified_nodes()?;
        let means = nodes.conditional_means(
            0..nodes.weight.len(),
            &truncated.projection_lift,
            decomposition.ambient_mean,
        );
        let variance = truncated
            .projection_lift
            .dot(&nodes.covariance.dot(&truncated.projection_lift))
            + truncated.residual_variance;
        Ok((
            ComponentProjection::Nodes {
                means,
                weights: &nodes.weight,
                residual_sd: truncated.residual_variance.sqrt(),
            },
            truncated.posterior_mean,
            variance,
        ))
    }
}

#[cfg(test)]
mod tests {
    //! The smallest fixture where the two readings of gam#3229 differ: one
    //! coefficient on `β ≥ 0` whose ambient posterior at `ρ` is
    //! `N(c(ρ), s²(ρ))`, `c = −1/(1+λ)`, `s² = 1/(1+λ)`, `λ = e^ρ` — the quadratic
    //! model with unit data curvature, a unit penalty and a unit score, so the
    //! mixture's fixed-curvature model is exact here.
    use super::*;
    use gam_linalg::roundoff::accumulation_band;
    use ndarray::array;

    fn bound() -> LinearInequalityConstraints {
        LinearInequalityConstraints::new(array![[1.0]], array![0.0]).expect("one nonnegativity row")
    }

    /// The mixture a dense route builds: drifts to roots, `M̂ĉ` from the dense
    /// precision, and each node factored by [`DenseNodePrecision`].
    fn dense_mixture(
        precision: &Array2<f64>,
        center: &Array1<f64>,
        drifts: &[Array2<f64>],
        rho_covariance: &Array2<f64>,
        constraints: &LinearInequalityConstraints,
    ) -> SmoothingMixture {
        let roots = drifts
            .iter()
            .map(drift_root)
            .collect::<Result<Vec<_>, String>>()
            .expect("positive semidefinite drifts");
        let precision_center = precision.dot(center);
        SmoothingMixture::build(
            SmoothingMixtureInputs {
                center: center.view(),
                precision_center: precision_center.view(),
                covariance_scale: 1.0,
                drift_roots: &roots,
                rho_covariance: rho_covariance.view(),
                constraints,
            },
            |weights| DenseNodePrecision::factor(precision.view(), &roots, weights),
        )
        .expect("the dense mixture builds")
    }

    /// The mixture's published variance at `ρ0` with `V_ρ = t`.
    fn mixture_variance(rho0: f64, t: f64) -> f64 {
        let lambda = rho0.exp();
        dense_mixture(
            &array![[1.0 + lambda]],
            &array![-1.0 / (1.0 + lambda)],
            &[array![[lambda]]],
            &array![[t]],
            &bound(),
        )
        .covariance()
        .expect("a dense route forms the covariance")[[0, 0]]
    }

    /// The q = 1 closed form of the target's first-order coefficient,
    /// `m′² + ½K″`, where `m` and `K` are the truncated mean and variance and `′`
    /// is `d/dρ` (the hand-off on gam#3229). With `z = −μ/s` and the inverse
    /// Mills ratio `λ(z) = φ(z)/Φ̄(z)`, `λ′ = λ(λ − z)`, and
    /// `m = μ + sλ`, `K = s²κ`, `κ = 1 + zλ − λ²`; `μ` and `s` carry `ρ` through the
    /// fixture. Returns `(m′² + ½K″, m′², ∂K/∂W·μ′²)`: the target, the part a
    /// mint without the curvature publishes, and the first-order coefficient of
    /// the old reading `trunc(Σ + J_c V_ρ J_cᵀ)`.
    fn closed_form_coefficients(rho0: f64) -> (f64, f64, f64) {
        let e = rho0.exp();
        let d = 1.0 + e;
        let mu = -1.0 / d;
        let w = 1.0 / d;
        let s = w.sqrt();
        let z = -mu / s;
        let log_density = -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let l = (log_density - normal_logsf(z)).exp();
        let l1 = l * (l - z);
        let l2 = l1 * (l - z) + l * (l1 - 1.0);
        let k = 1.0 + z * l - l * l;
        let k1 = l + z * l1 - 2.0 * l * l1;
        let k2 = 2.0 * l1 + z * l2 - 2.0 * l1 * l1 - 2.0 * l * l2;
        let m_mu = 1.0 - l1;
        let m_s = l - z * l1;
        let k_mu = -s * k1;
        let k_s = s * (2.0 * k - z * k1);
        let k_mu_mu = k2;
        let k_mu_s = -k1 + z * k2;
        let k_s_s = 2.0 * k - 2.0 * z * k1 + z * z * k2;
        let mu1 = e / (d * d);
        let mu2 = e * (1.0 - e) / (d * d * d);
        let w1 = -e / (d * d);
        let w2 = -e * (1.0 - e) / (d * d * d);
        let s1 = w1 / (2.0 * s);
        let s2 = w2 / (2.0 * s) - w1 * w1 / (4.0 * s * s * s);
        let m1 = m_mu * mu1 + m_s * s1;
        let k_second =
            k_mu_mu * mu1 * mu1 + 2.0 * k_mu_s * mu1 * s1 + k_s_s * s1 * s1 + k_mu * mu2 + k_s * s2;
        (
            m1 * m1 + 0.5 * k_second,
            m1 * m1,
            k_s / (2.0 * s) * mu1 * mu1,
        )
    }

    /// The fixture's truncated `(m, K)` at `ρ`, from the production closed form
    /// the cubature routes one retained row through.
    fn truncated_moments(rho: f64) -> (f64, f64) {
        let lambda = rho.exp();
        let (mean, covariance) =
            scalar_truncated_moments(-1.0 / (1.0 + lambda), 1.0 / (1.0 + lambda), f64::INFINITY)
                .expect("the fixture's truncated moments");
        (mean[0], covariance[[0, 0]])
    }

    /// A forward first difference in `t` at zero, Richardson-extrapolated, with
    /// its measured bar: `D(h) = (f(h) − f(0))/h = a + bh + O(h²)`, so
    /// `R(h) = 2D(h) − D(2h) = a + O(h²)`; `R(h) − R(2h)` is three times `R(h)`'s
    /// own truncation, and is the bar, beside the roundoff `5e/h` of the
    /// combination (`D(h)` carries `2e/h`, `D(2h)` `e/h`).
    fn forward_richardson(f: impl Fn(f64) -> f64, h: f64, evaluation_error: f64) -> (f64, f64) {
        let origin = f(0.0);
        let difference = |width: f64| (f(width) - origin) / width;
        let extrapolated = |width: f64| 2.0 * difference(width) - difference(2.0 * width);
        let fine = extrapolated(h);
        let coarse = extrapolated(2.0 * h);
        (fine, (fine - coarse).abs() + 5.0 * evaluation_error / h)
    }

    /// gam#3229's pin. At `ρ0 ∈ {0, ±1}` the published mixture's first-order
    /// term in `V_ρ` is the target `m′² + ½K″`, and the target is the second
    /// derivative of the exact θ-mixture: `E_δ[g(ρ0 + δ)]` with
    /// `g(ρ) = K(ρ) + (m(ρ) − m(ρ0))²` has first-order term `½g″ = m′² + ½K″`.
    ///
    /// Three computations that share no line: the closed form, a Richardson
    /// second difference of the production truncated moments (which pins the
    /// closed form), and the mixture itself. Neither alternative reading may
    /// pass: `V + J_m V_ρ J_mᵀ` alone (the curvature dropped) and
    /// `trunc(Σ + J_c V_ρ J_cᵀ)` (the ambient inflation) must each sit outside
    /// the mixture's bar, which is what makes this the fixture where the
    /// readings differ.
    #[test]
    fn the_mixture_carries_the_theta_mixture_variance_at_a_bound_3229() {
        for rho0 in [0.0_f64, 1.0, -1.0] {
            let (target, without_curvature, ambient_reading) = closed_form_coefficients(rho0);

            // The closed form against a second difference of the exact moments.
            let (m0, k0) = truncated_moments(rho0);
            let g = |rho: f64| {
                let (m, k) = truncated_moments(rho);
                k + (m - m0) * (m - m0)
            };
            // `h` balances a central second difference's `h²` truncation against
            // its `4e/h²` roundoff, the fourth root of the unit roundoff.
            let h = f64::EPSILON.powf(0.25);
            let central =
                |width: f64| (g(rho0 + width) - 2.0 * k0 + g(rho0 - width)) / (width * width);
            let fine = (4.0 * central(h) - central(2.0 * h)) / 3.0;
            let coarse = (4.0 * central(2.0 * h) - central(4.0 * h)) / 3.0;
            // One evaluation of `g` is the closed-form moments (two roundings for
            // the mean, four for the variance, one for the Mills ratio's log-space
            // quotient) and the square added to it (three): ten.
            let g_error = accumulation_band(10, k0.abs());
            let second_bar = 4.0 * (fine - coarse).abs() + (17.0 / 3.0) * g_error / (h * h);
            assert!(
                (0.5 * fine - target).abs() <= 0.5 * second_bar,
                "rho0={rho0}: closed form {target:e} against the exact moments' second \
                 difference {:e} (bar {:e})",
                0.5 * fine,
                0.5 * second_bar
            );

            // The mixture against the target.
            // A node is its precision, factor and two solves (five roundings),
            // the truncated moments (seven) and its share of the mixture's
            // sums (four): sixteen per node, and there are two.
            let mixture_error = accumulation_band(32, k0.abs());
            let (published, bar) = forward_richardson(
                |t| mixture_variance(rho0, t),
                f64::EPSILON.cbrt(),
                mixture_error,
            );
            assert!(
                (published - target).abs() <= bar,
                "rho0={rho0}: the mixture's first-order term {published:e} against the target \
                 {target:e} (bar {bar:e})"
            );
            for (reading, value) in [
                ("V + J_m V_rho J_m^T", without_curvature),
                ("trunc(Sigma + J_c V_rho J_c^T)", ambient_reading),
            ] {
                assert!(
                    (value - target).abs() > bar,
                    "rho0={rho0}: the reading {reading} ({value:e}) is inside the mixture's bar \
                     {bar:e} of the target {target:e}, so this fixture cannot tell them apart"
                );
            }
        }
    }

    /// Away from every bound the truncation is the identity, and the mixture's
    /// first-order term is exactly what the interior routes publish: the
    /// first-order correction `J V_ρ Jᵀ`, `J[:, k] = −Σ̂ D_k ĉ`, plus the
    /// curvature term `crate::estimate::laplace_covariance_curvature_term`
    /// carries. That is the continuity claim: as a bound activates, the
    /// published object moves onto the cone rather than switching formula.
    #[test]
    fn at_an_interior_mode_the_mixture_is_the_interior_correction_3229() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let precision = array![[2.0, 0.3], [0.3, 1.5]];
        let drifts = vec![
            array![[0.8, 0.0], [0.0, 0.0]],
            array![[0.2, 0.1], [0.1, 0.4]],
        ];
        let center = array![0.7, -0.4];
        let rho_covariance = array![[0.6, 0.15], [0.15, 0.3]];
        // Rows forty standard deviations below the centre: the truncation is
        // invisible at f64 resolution at every node.
        let constraints =
            LinearInequalityConstraints::new(array![[1.0, 0.0], [0.0, 1.0]], array![-40.0, -40.0])
                .expect("two slack rows");
        let covariance_at = |t: f64| {
            dense_mixture(
                &precision,
                &center,
                &drifts,
                &rho_covariance.mapv(|value| t * value),
                &constraints,
            )
            .covariance()
            .expect("a dense route forms the covariance")
            .clone()
        };
        let factor = precision.cholesky(Side::Lower).expect("SPD precision");
        let sigma = factor.solve_mat(&Array2::eye(2));
        let jacobian =
            Array2::from_shape_fn((2, 2), |(row, k)| -sigma.dot(&drifts[k].dot(&center))[row]);
        let first_order = jacobian.dot(&rho_covariance).dot(&jacobian.t());
        let curvature = crate::estimate::laplace_covariance_curvature_term(
            sigma.view(),
            &drifts,
            rho_covariance.view(),
        )
        .expect("the curvature term");
        let expected = &first_order + &curvature;
        let scale = sigma
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        // Per node: the precision update (two), factor and solve (four in two
        // dimensions), the centre (four) and its mixture share (four): fourteen,
        // over four nodes.
        let evaluation_error = accumulation_band(56, scale);
        let mut curvature_resolved = false;
        for a in 0..2 {
            for b in 0..2 {
                let (published, bar) = forward_richardson(
                    |t| covariance_at(t)[[a, b]],
                    f64::EPSILON.cbrt(),
                    evaluation_error,
                );
                assert!(
                    (published - expected[[a, b]]).abs() <= bar,
                    "entry ({a},{b}): mixture {published:e} against J V Jt + curvature {:e} \
                     (bar {bar:e})",
                    expected[[a, b]]
                );
                curvature_resolved |= curvature[[a, b]].abs() > bar;
            }
        }
        assert!(
            curvature_resolved,
            "no entry of the curvature term clears the bar, so the interior pin cannot tell the \
             mixture from J V Jt alone"
        );
    }

    /// With no resolved θ direction the mixture is the one law at the mode: the
    /// node law the predictor rebuilds is the one the fit published.
    #[test]
    fn a_zero_theta_covariance_is_the_conditional_law_3229() {
        let constraints = bound();
        let center = array![-0.5];
        let precision = array![[2.0]];
        let mixture = dense_mixture(
            &precision,
            &center,
            &[array![[1.0]]],
            &array![[0.0]],
            &constraints,
        );
        assert_eq!(mixture.node_count(), 1);
        let base = ConstrainedPosteriorGeometry::with_moments(
            constraints.clone(),
            array![0.0],
            center.clone(),
            None,
        );
        let laws = mixture
            .node_laws(precision.view(), &base)
            .expect("node laws");
        let (ambient, geometry) = &laws[0];
        // `N(ĉ, M̂⁻¹)` to the roundings of a one-by-one factor and its solves.
        let band = accumulation_band(4, 0.5);
        assert!(
            (ambient[[0, 0]] - 0.5).abs() <= band,
            "{:e}",
            ambient[[0, 0]]
        );
        let node_center = geometry.unconstrained_center().expect("centre");
        assert!((node_center[0] + 0.5).abs() <= band, "{:e}", node_center[0]);
        let correction = geometry
            .correction()
            .expect("moments")
            .expect("the bound is visible at the node");
        assert_eq!(mixture.mean(), &correction.posterior_mean(node_center));
        assert_eq!(
            mixture.covariance().expect("dense"),
            &correction
                .truncated_covariance_psd(ambient, &constraints)
                .expect("the node's truncated covariance")
        );
    }

    /// A node factored by a route that forms no dense covariance, here the
    /// dense factor with its inverse withheld, so it is served through solves
    /// alone.
    struct SolvesOnly(DenseNodePrecision);

    impl NodeSolve for SolvesOnly {
        fn solve(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
            self.0.solve(rhs)
        }
    }

    impl NodePrecision for SolvesOnly {
        fn dense_inverse(&self) -> Option<Result<Array2<f64>, String>> {
            None
        }
    }

    /// A factorized route builds the same mixture as a dense one (gam#3229,
    /// the factorized branch): the same nodes and the same mean, with no dense
    /// covariance formed. The two read each node's truncation from the same
    /// `Σ_iAᵀ` and centre, formed once as a product with a dense inverse and
    /// once as solves against it, so they differ by the roundings of those
    /// products: eight on the largest node's ambient scale bound them.
    #[test]
    fn a_factorized_route_publishes_the_dense_mixture_3229() {
        let precision = array![[1.5, 0.2], [0.2, 1.1]];
        let center = array![-0.4, 0.3];
        let drifts = vec![
            array![[0.5, 0.0], [0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.3]],
        ];
        let rho_covariance = array![[0.4, 0.1], [0.1, 0.2]];
        let constraints = LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0])
            .expect("one nonnegativity row");
        let dense = dense_mixture(&precision, &center, &drifts, &rho_covariance, &constraints);
        let roots = drifts
            .iter()
            .map(drift_root)
            .collect::<Result<Vec<_>, String>>()
            .expect("roots");
        let precision_center = precision.dot(&center);
        let factorized = SmoothingMixture::build(
            SmoothingMixtureInputs {
                center: center.view(),
                precision_center: precision_center.view(),
                covariance_scale: 1.0,
                drift_roots: &roots,
                rho_covariance: rho_covariance.view(),
                constraints: &constraints,
            },
            |weights| DenseNodePrecision::factor(precision.view(), &roots, weights).map(SolvesOnly),
        )
        .expect("the factorized mixture builds");
        assert!(factorized.covariance().is_none());
        assert_eq!(factorized.node_count(), dense.node_count());
        let base = ConstrainedPosteriorGeometry::with_moments(
            constraints.clone(),
            array![0.0, 0.3],
            center.clone(),
            None,
        );
        let scale = dense
            .node_laws(precision.view(), &base)
            .expect("node laws")
            .iter()
            .flat_map(|(ambient, _)| ambient.diag().to_vec())
            .fold(0.0_f64, f64::max);
        for index in 0..2 {
            let bar = accumulation_band(8, scale) + accumulation_band(8, dense.mean()[index].abs());
            assert!(
                (factorized.mean()[index] - dense.mean()[index]).abs() <= bar,
                "mean {index}: {} against {}",
                factorized.mean()[index],
                dense.mean()[index]
            );
        }
    }

    /// The mixture's projection interval is the equal-tailed interval of the
    /// MIXTURE law: at each endpoint the mean of the nodes' CDFs is the tail
    /// probability, to the resolution `√ε·spread` the quantile is stated at. The
    /// node CDFs here are the direct `(Φ(z) − Φ(wall))/(1 − Φ(wall))`, a
    /// different evaluation from the survival-space one the law uses. A
    /// one-node mixture is that node's law and returns its interval exactly.
    #[test]
    fn mixture_intervals_are_the_mixture_laws_quantiles_3229() {
        let lambda = 1.0_f64;
        let constraints = bound();
        let precision = array![[1.0 + lambda]];
        let center = array![-1.0 / (1.0 + lambda)];
        let mixture = dense_mixture(
            &precision,
            &center,
            &[array![[lambda]]],
            &array![[0.5]],
            &constraints,
        );
        let base = ConstrainedPosteriorGeometry::with_moments(
            constraints.clone(),
            array![0.0],
            center.clone(),
            None,
        );
        let laws = mixture
            .node_laws(precision.view(), &base)
            .expect("node laws");
        let projections = laws
            .iter()
            .map(|(ambient, geometry)| ConstrainedProjectionLaw::new(ambient, geometry))
            .collect::<Result<Vec<_>, String>>()
            .expect("node projections");
        let level = 0.9;
        let contrast = array![[1.0]];
        let single = projections[0]
            .equal_tailed_intervals(contrast.view(), level)
            .expect("one node's interval");
        let law = ConstrainedMixtureProjectionLaw::new(projections).expect("the mixture law");
        let (lower, upper) = law
            .equal_tailed_intervals(contrast.view(), level)
            .expect("the mixture interval")[0];
        let node_cdf = |ambient: &Array2<f64>, node_center: f64, x: f64| {
            let sd = ambient[[0, 0]].sqrt();
            let wall_cdf = normal_cdf_and_pdf(-node_center / sd).0;
            let (at_cdf, at_density) = normal_cdf_and_pdf((x - node_center) / sd);
            let mass = 1.0 - wall_cdf;
            ((at_cdf - wall_cdf) / mass, at_density / (sd * mass))
        };
        let spread = mixture.covariance().expect("dense")[[0, 0]].sqrt();
        let weight = 1.0 / laws.len() as f64;
        for (endpoint, probability) in [(lower, 0.05), (upper, 0.95)] {
            let (cdf, density) = laws.iter().fold((0.0, 0.0), |acc, (ambient, geometry)| {
                let node_center = geometry.unconstrained_center().expect("centre")[0];
                let (node_cdf_value, node_density) = node_cdf(ambient, node_center, endpoint);
                (
                    acc.0 + weight * node_cdf_value,
                    acc.1 + weight * node_density,
                )
            });
            // The quantile is settled to `√ε·spread` in value, which is
            // `density·√ε·spread` in probability; each CDF above is a handful of
            // roundings on a quotient by the retained mass.
            let bar = density * f64::EPSILON.sqrt() * spread + accumulation_band(8, 1.0);
            assert!(
                (cdf - probability).abs() <= bar,
                "endpoint {endpoint}: mixture CDF {cdf} against {probability} (bar {bar:e})"
            );
        }
        let one_node = ConstrainedMixtureProjectionLaw::new(vec![
            ConstrainedProjectionLaw::new(&laws[0].0, &laws[0].1).expect("node zero"),
        ])
        .expect("a one-node mixture");
        assert_eq!(
            one_node
                .equal_tailed_intervals(contrast.view(), level)
                .expect("the one-node interval"),
            single
        );
    }
}
