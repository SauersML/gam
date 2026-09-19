//! Post-fit inference on the penalized Hessian's identified subspace (#2901 V22).
//!
//! `H = XᵀWX + S_λ` with both terms PSD is singular exactly on
//! `null(X) ∩ null(S_λ)`: coefficient directions no observation and no penalty
//! pins down, such as an intercept column that every partition-of-unity spline
//! block reproduces. PIRLS solves such an `H` min-norm, and the criterion scores
//! `½log|H|₊` over the eigenvectors that
//! `RemlState::intrinsic_hessian_pseudo_logdet_parts_from_eigensystem` keeps.
//! When the strict Cholesky of `H` refuses, post-fit inference takes its traces,
//! covariances, influence and IFT solves against `H⁺ = U·M⁻¹·Uᵀ` over that same
//! kept set, so a fit is summarized on the subspace it was scored on.
//!
//! That kept set is a step function of ρ: `½log|H|₊` jumps by `½ln σ` where a
//! direction crosses the rounding band. [`certify_fitted_identified_rank`]
//! certifies at finalization that the rank is the same over the outer
//! certificate's own Newton step, so the derivative certificate at ρ̂ describes
//! a criterion that is smooth where the certificate applies.

use super::EstimationError;
use super::reml::RemlState;
use super::reml::reml_outer_engine::DenseSpectralOperator;
use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::utils::{CertifiedSymmetricSolveError, certify_linear_system_residual};
use ndarray::{Array1, Array2, ArrayView1, s};

/// `H⁺` in spectral form over the identified subspace of a dense penalized Hessian.
pub(crate) struct IdentifiedHessianInverse {
    /// The kept eigenvectors `U`, one column per identified direction.
    basis: Array2<f64>,
    /// `M⁻¹ = diag(1/σ_a)` over the kept eigenvalues `σ_a`.
    reduced_inverse: Array2<f64>,
}

impl IdentifiedHessianInverse {
    /// Decompose `hessian` and keep exactly the directions the criterion keeps
    /// for a penalty of rank `penalty_rank`. An eigenvalue below `−p·ε·‖H‖₂`,
    /// PIRLS's own rounding band, is material indefiniteness that `XᵀWX + S_λ`
    /// cannot have, so it is refused rather than dropped.
    pub(crate) fn from_dense(
        hessian: &Array2<f64>,
        penalty_rank: usize,
    ) -> Result<Self, EstimationError> {
        let mut symmetric = hessian.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
        let (eigenvalues, eigenvectors) = symmetric
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let eigenvalues = eigenvalues.to_vec();
        let rounding_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&eigenvalues);
        let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if !(min_eigenvalue >= -rounding_band) {
            return Err(EstimationError::HessianNotPositiveDefinite { min_eigenvalue });
        }
        let kept = RemlState::intrinsic_hessian_pseudo_logdet_parts_from_eigensystem(
            &eigenvalues,
            &eigenvectors,
            penalty_rank,
        )?
        .1
        .ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "the penalized Hessian has no positive curvature, so no coefficient direction \
                 is identified"
                    .to_string(),
            )
        })?;
        Ok(Self {
            basis: kept.u_s,
            reduced_inverse: kept.h_proj_inverse,
        })
    }

    /// The number of identified directions.
    pub(crate) fn rank(&self) -> usize {
        self.reduced_inverse.nrows()
    }

    /// The same inverse for `Qs·H·Qsᵀ` with orthogonal `Qs`: the eigenvalues are
    /// unchanged and every kept eigenvector maps through `Qs`.
    pub(crate) fn rotated(&self, qs: &Array2<f64>) -> Self {
        Self {
            basis: qs.dot(&self.basis),
            reduced_inverse: self.reduced_inverse.clone(),
        }
    }

    /// The IFT sensitivity operator `U·M⁻¹·Uᵀ`.
    pub(crate) fn sensitivity(&self) -> crate::sensitivity::FitSensitivity<'_> {
        crate::sensitivity::FitSensitivity::from_projected(&self.basis, &self.reduced_inverse)
    }

    /// `H⁺·B`.
    pub(crate) fn apply(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut coordinates = self.basis.t().dot(rhs);
        for (mut row, &inverse) in coordinates
            .rows_mut()
            .into_iter()
            .zip(self.reduced_inverse.diag().iter())
        {
            row *= inverse;
        }
        self.basis.dot(&coordinates)
    }

    /// `U·Uᵀ·B`, the part of `B` that lies in the identified subspace.
    pub(crate) fn project(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.basis.dot(&self.basis.t().dot(rhs))
    }

    /// Solve `H·X = U·Uᵀ·B` min-norm and certify the residual against `U·Uᵀ·B`,
    /// the right-hand side a solve on the identified subspace can represent.
    pub(crate) fn certified_solve(
        &self,
        hessian: &Array2<f64>,
        rhs: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>, CertifiedSymmetricSolveError> {
        let solution = self.apply(rhs);
        let projected = self.project(rhs);
        let residual = hessian.dot(&solution) - &projected;
        certify_linear_system_residual(
            hessian.nrows(),
            max_abs_entry(hessian),
            &projected,
            &solution,
            &residual,
            label,
        )?;
        Ok(solution)
    }

    /// `H⁺` itself, certified by `H·H⁺ = U·Uᵀ`.
    pub(crate) fn certified_inverse(
        &self,
        hessian: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>, CertifiedSymmetricSolveError> {
        let identity = Array2::<f64>::eye(hessian.nrows());
        let mut inverse = self.apply(&identity);
        gam_linalg::matrix::symmetrize_in_place(&mut inverse);
        let projector = self.project(&identity);
        let residual = hessian.dot(&inverse) - &projector;
        certify_linear_system_residual(
            hessian.nrows(),
            max_abs_entry(hessian),
            &projector,
            &inverse,
            &residual,
            label,
        )?;
        Ok(inverse)
    }
}

fn max_abs_entry(matrix: &Array2<f64>) -> f64 {
    matrix
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// How far the curvature weights can move over a smoothing-parameter step
/// (#2901 V22).
///
/// The weights move through `β̂` by at most `ΔW_i` per row, so a row with
/// `W_i > 0` moves by at most `m·W_i`, `m = max_{W_i > 0} ΔW_i/W_i`. The rows
/// are split by sign. With `G = XᵀWX` and
/// `V = Σ_{W_i ≤ 0} (m·|W_i| + ΔW_i)·x_i x_iᵀ`, in Loewner order
///
/// ```text
///   (1 − m)·G − V ⪯ XᵀW'X ⪯ (1 + m)·G + V,
/// ```
///
/// row by row and exactly, whatever the weights' signs: a positive row is
/// scaled by `1 ∓ m`, and on a row whose weight is not positive, scaling `G` by
/// `1 ∓ m` moves `W_i` by `m·|W_i|` the wrong way and the row's own motion adds
/// `ΔW_i`, both of which `V` returns, so that row is bounded by `W_i ∓ ΔW_i`. A
/// non-canonical link or observed information can make `G` indefinite, and the
/// bound does not ask it to be semidefinite.
#[derive(Clone, Debug)]
pub(crate) struct HessianSpectrumMotion {
    /// `m`.
    relative_weight_motion: f64,
    /// `V` in the Hessian's basis, when some row whose weight is not positive
    /// carries mass.
    nonpositive_rows: Option<Array2<f64>>,
    /// Whether every weight is positive or still at zero, with `m ≤ 1`: then
    /// `XᵀW'X ⪰ 0` everywhere the step reaches, so `H' ⪰ S'_λ`.
    weights_stay_nonnegative: bool,
}

impl HessianSpectrumMotion {
    /// The motion of curvature `weights` whose first-order motion over the step
    /// is at most `weight_motion`, row by row. `nonpositive_row_gram(v)` is
    /// `Xᵀdiag(v)X` in the Hessian's basis, for a nonnegative `v` that vanishes on
    /// every row with a positive weight; it is asked only when some other row
    /// carries mass.
    pub(crate) fn over_weights(
        weights: ArrayView1<'_, f64>,
        weight_motion: ArrayView1<'_, f64>,
        nonpositive_row_gram: impl FnOnce(&Array1<f64>) -> Result<Array2<f64>, EstimationError>,
    ) -> Result<Self, EstimationError> {
        if weights.len() != weight_motion.len() {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum motion needs one weight motion per row: {} weights, {} motions",
                weights.len(),
                weight_motion.len()
            )));
        }
        let mut relative_motion = 0.0_f64;
        let mut still_or_positive = true;
        for (&weight, &motion) in weights.iter().zip(weight_motion.iter()) {
            if !(weight.is_finite() && motion.is_finite() && motion >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Hessian spectrum motion needs finite weights and finite nonnegative weight \
                     motions: weight {weight:.4e}, motion {motion:.4e}"
                )));
            }
            if weight > 0.0 {
                relative_motion = relative_motion.max(motion / weight);
            } else {
                still_or_positive &= weight == 0.0 && motion == 0.0;
            }
        }
        let mass: Array1<f64> = weights
            .iter()
            .zip(weight_motion.iter())
            .map(|(&weight, &motion)| {
                if weight > 0.0 {
                    0.0
                } else {
                    relative_motion * weight.abs() + motion
                }
            })
            .collect();
        let nonpositive_rows = if mass.iter().any(|&value| value > 0.0) {
            let gram = nonpositive_row_gram(&mass)?;
            if !gram.iter().all(|value| value.is_finite()) {
                return Err(EstimationError::InvalidInput(
                    "Hessian spectrum motion: the nonpositive-weight rows' Gram is not finite"
                        .to_string(),
                ));
            }
            Some(gram)
        } else {
            None
        };
        Ok(Self {
            relative_weight_motion: relative_motion,
            nonpositive_rows,
            weights_stay_nonnegative: still_or_positive && relative_motion <= 1.0,
        })
    }
}

/// Loewner bounds on the penalized Hessian's spectrum over a smoothing-parameter
/// step (#2901 V22).
///
/// Along `ρ ↦ ρ + δρ` with `|δρ_k| ≤ t_k` the criterion's Hessian is
/// `H' = XᵀW'X + Σ_k e^{δρ_k}·λ_k S̃_k`. Every `λ_k S̃_k ⪰ 0` is scaled by a factor
/// in `[e^{−t_k}, e^{t_k}]`, exactly, for any finite step, and the curvature
/// weights move as [`HessianSpectrumMotion`] bounds them. With `G = XᵀWX`,
///
/// ```text
///   H₋ ⪯ H' ⪯ H₊,
///   H₋ = (1 − m)·G − V + Σ_k e^{−t_k}·λ_k S̃_k,   H₊ = (1 + m)·G + V + Σ_k e^{t_k}·λ_k S̃_k,
/// ```
///
/// so by Weyl's monotonicity theorem `σ_i(H₋) ≤ σ_i(H') ≤ σ_i(H₊)` for every
/// `i`. Each coordinate's step is charged to its own penalty, and each row's
/// motion to its own row, so only to the directions they reach. The premise is
/// that every `λ_k S̃_k` is positive semidefinite and that the step moves only the
/// `λ_k`: every penalty registered here is a root product `λ_k·RᵀR`.
///
/// The penalty PIRLS put in `H` is the engine's `S̃ = EᵀE`, built on the
/// λ-invariant penalized block, so it is exactly zero on the structural null
/// space the engine declares. The blocks are the engine's own projections
/// `S̃_k = Π S_k Π` onto that block
/// ([`gam_terms::construction::ReparamResult::applied_penalties`]): then `Σ_k λ_k S̃_k = S̃` in
/// exact arithmetic, and `Π` does not move with ρ. A root rotated into the
/// transformed frame keeps a relative leakage onto the null coordinates, which
/// the raw `S_k` carry. On `y ~ s(x) + s(x, g, bs='fs')` (n=120, seed 0) that
/// leakage reached 8.9e4 at λ = 1.98e12, H's smallest eigenvalue was 0.996, and
/// the raw blocks let it "fall to −99".
///
/// `G` is taken as `Ĝ = H − B` with `B = Σ_k λ_k S̃_k`, and whatever the
/// arithmetic leaves of that identity is charged rather than assumed. With
/// `R = S̃ − B`, `Ĝ = G + R`, so `(1 − m)·G ⪰ (1 − m)·Ĝ − |1 − m|·‖R‖₂·I` and
/// `(1 + m)·G ⪯ (1 + m)·Ĝ + (1 + m)·‖R‖₂·I`. The charge is `‖R‖_F ≥ ‖R‖₂`, and a
/// shift by a multiple of `I` moves every eigenvalue by exactly that multiple. On
/// the fs fit `R`'s largest entry was 9.8e-4, rounding at ‖S̃‖ = 1.9e12. An engine
/// penalty the blocks do not reproduce, such as a floored range eigenvalue or a
/// reordered split, is charged its whole residual, so the certificate refuses
/// rather than vouching for a spectrum it did not bound.
///
/// Scaling all of `H` by `e^{±max_k t_k}` bounds the same motion, but it charges
/// one coordinate's step to every eigenvalue and to the band together. On a
/// gamma-log Matérn fit whose largest step, 5.48, fell on a penalty the smallest
/// identified direction does not see (Rayleigh quotient 9.4e-7 against
/// σ_r = 132.36), that uniform scale let σ_r fall to 0.552 and the band rise to
/// 3.15 and refused the fit. The Hessian re-solved at the Newton point, at both
/// ends of every axis and at every corner of the step kept its rank, its
/// smallest eigenvalue (133.57) and its band (1.32e-2) to five digits.
pub(crate) struct HessianSpectrumBounds {
    /// `σ_i(H₋)`, descending.
    lower: Vec<f64>,
    /// `σ_i(H₊)`, descending.
    upper: Vec<f64>,
    /// See [`HessianSpectrumMotion`].
    weights_stay_nonnegative: bool,
}

impl HessianSpectrumBounds {
    /// The bounds for the penalized `hessian`, which carries the engine's
    /// `penalty` `S̃`, over a step of at most `step[k]` in coordinate `k`, for the
    /// curvature weights' `motion` over that step. `penalties` yields one
    /// `(range, block)` per coordinate, in coordinate order: `block` is
    /// `λ_k S̃_k` on `hessian`'s rows and columns `range`, zero elsewhere.
    pub(crate) fn over_step(
        hessian: &Array2<f64>,
        penalty: &Array2<f64>,
        penalties: impl IntoIterator<Item = (std::ops::Range<usize>, Array2<f64>)>,
        step: ArrayView1<'_, f64>,
        motion: HessianSpectrumMotion,
    ) -> Result<Self, EstimationError> {
        let dimension = hessian.nrows();
        if hessian.ncols() != dimension
            || penalty.dim() != (dimension, dimension)
            || step.iter().any(|&radius| !(radius.is_finite() && radius >= 0.0))
        {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum bounds need a square Hessian, a penalty of its shape and \
                 finite nonnegative steps: {}x{} Hessian, {}x{} penalty, steps {step}",
                hessian.nrows(),
                hessian.ncols(),
                penalty.nrows(),
                penalty.ncols()
            )));
        }
        let m = motion.relative_weight_motion;
        let mut total = Array2::<f64>::zeros((dimension, dimension));
        let mut shrunk = Array2::<f64>::zeros((dimension, dimension));
        let mut grown = Array2::<f64>::zeros((dimension, dimension));
        let mut upper_unbounded = false;
        let mut coordinates = 0usize;
        for (range, block) in penalties {
            let Some(&radius) = step.get(coordinates) else {
                return Err(EstimationError::InvalidInput(format!(
                    "Hessian spectrum bounds: more penalties than the {} step coordinates",
                    step.len()
                )));
            };
            if range.end > dimension || block.dim() != (range.len(), range.len()) {
                return Err(EstimationError::InvalidInput(format!(
                    "Hessian spectrum bounds: penalty {coordinates} is a {}x{} block on columns \
                     {}..{} of a {dimension}x{dimension} Hessian",
                    block.nrows(),
                    block.ncols(),
                    range.start,
                    range.end
                )));
            }
            total
                .slice_mut(s![range.clone(), range.clone()])
                .scaled_add(1.0, &block);
            shrunk
                .slice_mut(s![range.clone(), range.clone()])
                .scaled_add((-radius).exp(), &block);
            let growth = radius.exp();
            if growth.is_finite() {
                grown
                    .slice_mut(s![range.clone(), range])
                    .scaled_add(growth, &block);
            } else {
                upper_unbounded = true;
            }
            coordinates += 1;
        }
        if coordinates != step.len() {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum bounds: {coordinates} penalties for {} step coordinates",
                step.len()
            )));
        }
        let difference = penalty - &total;
        let scale = difference
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let residual = if scale > 0.0 {
            scale
                * difference
                    .iter()
                    .fold(0.0_f64, |acc, value| acc + (value / scale).powi(2))
                    .sqrt()
        } else {
            0.0
        };
        if !residual.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Hessian spectrum bounds: the engine penalty's residual against the per-coordinate \
                 blocks is not finite: {residual:.4e}"
            )));
        }
        let data = hessian - &total;
        let mut lower = &data * (1.0 - m) + &shrunk;
        let mut upper = &data * (1.0 + m) + &grown;
        if let Some(rows) = motion.nonpositive_rows.as_ref() {
            if rows.dim() != (dimension, dimension) {
                return Err(EstimationError::InvalidInput(format!(
                    "Hessian spectrum bounds: a {}x{} nonpositive-row Gram for a \
                     {dimension}x{dimension} Hessian",
                    rows.nrows(),
                    rows.ncols()
                )));
            }
            lower -= rows;
            upper += rows;
        }
        let lower_charge = (1.0 - m).abs() * residual;
        let upper_charge = (1.0 + m) * residual;
        Ok(Self {
            lower: descending_spectrum(lower)?
                .into_iter()
                .map(|value| value - lower_charge)
                .collect(),
            upper: if upper_unbounded {
                vec![f64::INFINITY; dimension]
            } else {
                descending_spectrum(upper)?
                    .into_iter()
                    .map(|value| value + upper_charge)
                    .collect()
            },
            weights_stay_nonnegative: motion.weights_stay_nonnegative,
        })
    }
}

fn descending_spectrum(mut matrix: Array2<f64>) -> Result<Vec<f64>, EstimationError> {
    gam_linalg::matrix::symmetrize_in_place(&mut matrix);
    let mut values = matrix
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?
        .0
        .to_vec();
    values.sort_by(|left, right| right.total_cmp(left));
    Ok(values)
}

/// The outer certificate's own Newton displacement `|H_ρ⁺·g|` at ρ̂, coordinate
/// by coordinate (#2901 V22): how far from ρ̂ the stationary point it vouches for
/// can lie. A coordinate certified on a rail is certified by its tail law, not
/// by a derivative at ρ̂, so it moves nothing here. The inverse is taken over the
/// interior block's eigenvalues outside that block's own rounding band: a
/// direction whose curvature the arithmetic does not resolve localizes nothing.
pub(crate) fn certificate_newton_displacement(
    hessian_rho: &Array2<f64>,
    gradient: &Array1<f64>,
    railed: &[usize],
) -> Result<Array1<f64>, EstimationError> {
    let dimension = gradient.len();
    if hessian_rho.dim() != (dimension, dimension) {
        return Err(EstimationError::InvalidInput(format!(
            "certificate Newton displacement: a {}x{} outer Hessian against {dimension} gradient \
             coordinates",
            hessian_rho.nrows(),
            hessian_rho.ncols()
        )));
    }
    let interior: Vec<usize> = (0..dimension)
        .filter(|index| !railed.contains(index))
        .collect();
    let mut displacement = Array1::<f64>::zeros(dimension);
    if interior.is_empty() {
        return Ok(displacement);
    }
    let mut block = Array2::from_shape_fn((interior.len(), interior.len()), |(row, column)| {
        hessian_rho[[interior[row], interior[column]]]
    });
    gam_linalg::matrix::symmetrize_in_place(&mut block);
    let (eigenvalues, eigenvectors) = block
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let rounding_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&eigenvalues.to_vec());
    let interior_gradient: Array1<f64> = interior.iter().map(|&index| gradient[index]).collect();
    let coordinates = eigenvectors.t().dot(&interior_gradient);
    let mut interior_step = Array1::<f64>::zeros(interior.len());
    for (column, (&sigma, &coordinate)) in eigenvalues.iter().zip(coordinates.iter()).enumerate() {
        if sigma.abs() > rounding_band {
            interior_step.scaled_add(coordinate / sigma, &eigenvectors.column(column));
        }
    }
    for (&index, &step) in interior.iter().zip(interior_step.iter()) {
        displacement[index] = step.abs();
    }
    if displacement.iter().all(|value| value.is_finite()) {
        Ok(displacement)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "certificate Newton displacement is not finite: {displacement}"
        )))
    }
}

/// The identified rank of a penalized Hessian, certified constant over a step,
/// with the spectrum numbers it was judged at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdentifiedRankCertificate {
    /// The number of identified coefficient directions.
    pub(crate) rank: usize,
    /// The smallest identified eigenvalue `σ_r`.
    pub(crate) smallest_identified: f64,
    /// The largest unidentified eigenvalue `σ_{r+1}`; `None` at full rank.
    pub(crate) largest_unidentified: Option<f64>,
    /// `H`'s rounding band `p·ε·‖H‖₂`.
    pub(crate) band: f64,
}

/// Certify that the identified rank of a penalized Hessian with `eigenvalues`,
/// for a penalty of rank `penalty_rank`, is the same at every point `bounds`
/// reaches (#2901 V22).
///
/// The criterion prices `½log|H|₊` over the top
/// [`DenseSpectralOperator::identified_rank`] eigenvalues: those above
/// `p·ε·‖H‖₂`, and never fewer than `rank(S_λ)`. Anywhere the step reaches, the
/// smallest kept eigenvalue is at least `σ_r(H₋)` and the largest dropped one at
/// most `σ_{r+1}(H₊)`, while the spectral radius lies between
/// `max(σ_1(H₋), −σ_p(H₊), 0)` and `max(σ_1(H₊), −σ_p(H₋))`, so the band lies
/// between `p·ε` times those. The rank is constant when each stays
/// on its own side of the band. At the floor `rank = rank(S_λ)` the kept set
/// cannot shrink while the curvature weights stay nonnegative: `H' ⪰ S'_λ` gives
/// `σ_{rank(S_λ)}(H') ≥ σ_{rank(S_λ)}(S'_λ) > 0`, and `rank(S_λ)` does not depend
/// on ρ.
pub(crate) fn certify_identified_rank_locally_constant(
    eigenvalues: &[f64],
    rank: usize,
    penalty_rank: usize,
    bounds: &HessianSpectrumBounds,
) -> Result<IdentifiedRankCertificate, EstimationError> {
    let coefficients = eigenvalues.len();
    if bounds.lower.len() != coefficients || bounds.upper.len() != coefficients {
        return Err(EstimationError::InvalidInput(format!(
            "identified-rank certificate: {coefficients} eigenvalues against bounds of {} and {}",
            bounds.lower.len(),
            bounds.upper.len()
        )));
    }
    let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(eigenvalues);
    let mut descending = eigenvalues.to_vec();
    descending.sort_by(|left, right| right.total_cmp(left));
    let Some(&smallest_identified) = rank.checked_sub(1).and_then(|index| descending.get(index))
    else {
        return Err(EstimationError::RemlOptimizationFailed(
            "the penalized Hessian has no positive curvature, so no coefficient direction is \
             identified"
                .to_string(),
        ));
    };
    let largest_unidentified = descending.get(rank).copied();
    let resolution = coefficients as f64 * f64::EPSILON;
    let reachable_band = (
        resolution
            * bounds.lower[0]
                .max(-bounds.upper[coefficients - 1])
                .max(0.0),
        resolution * bounds.upper[0].max(-bounds.lower[coefficients - 1]),
    );
    let reachable_smallest_identified = bounds.lower[rank - 1];
    let reachable_largest_unidentified = largest_unidentified.map(|_| bounds.upper[rank]);
    let kept_set_holds = (rank == penalty_rank && bounds.weights_stay_nonnegative)
        || reachable_smallest_identified > reachable_band.1;
    let dropped_set_holds = match reachable_largest_unidentified {
        Some(sigma) => sigma <= reachable_band.0,
        None => true,
    };
    if kept_set_holds && dropped_set_holds {
        Ok(IdentifiedRankCertificate {
            rank,
            smallest_identified,
            largest_unidentified,
            band,
        })
    } else {
        Err(EstimationError::IdentifiedRankNotLocallyConstant {
            rank,
            coefficients,
            smallest_identified,
            largest_unidentified,
            band,
            reachable_smallest_identified,
            reachable_largest_unidentified,
            reachable_band,
        })
    }
}

/// The fitted penalized Hessian's spectrum and the identified rank the criterion
/// priced it at (#2901 V22).
pub(crate) struct FittedHessianSpectrum {
    /// `H` itself, symmetrized.
    hessian: Array2<f64>,
    /// Every eigenvalue of `H`, in the eigensolver's order.
    eigenvalues: Vec<f64>,
    /// The matching eigenvectors, one column each, in `H`'s basis.
    eigenvectors: Array2<f64>,
    /// `rank(S_λ)` in the same basis.
    penalty_rank: usize,
    /// [`DenseSpectralOperator::identified_rank`] of `eigenvalues`.
    rank: usize,
}

impl FittedHessianSpectrum {
    /// Decompose PIRLS's dense penalized `hessian`, for a penalty of rank
    /// `penalty_rank`.
    pub(crate) fn of(hessian: &Array2<f64>, penalty_rank: usize) -> Result<Self, EstimationError> {
        let mut symmetric = hessian.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
        let (eigenvalues, eigenvectors) = symmetric
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let eigenvalues = eigenvalues.to_vec();
        let rank = DenseSpectralOperator::identified_rank(&eigenvalues, penalty_rank);
        Ok(Self::from_eigensystem(
            symmetric,
            eigenvalues,
            eigenvectors,
            penalty_rank,
            rank,
        ))
    }

    /// The spectrum a criterion builder already decomposed `hessian` into and
    /// priced at `rank`, for a penalty of rank `penalty_rank` (#2959 D1). The
    /// certificate then judges the eigenpairs and the rank the criterion priced,
    /// not a second decomposition and a second call to the predicate.
    pub(crate) fn from_eigensystem(
        hessian: Array2<f64>,
        eigenvalues: Vec<f64>,
        eigenvectors: Array2<f64>,
        penalty_rank: usize,
        rank: usize,
    ) -> Self {
        Self {
            hessian,
            eigenvalues,
            eigenvectors,
            penalty_rank,
            rank,
        }
    }

    /// The number of identified coefficient directions.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// An orthonormal basis, `p × (p − rank)`, of the directions the criterion
    /// dropped: the eigenvectors of the `p − rank` smallest eigenvalues, the
    /// complement of the kept set
    /// `RemlState::intrinsic_hessian_pseudo_logdet_parts_from_eigensystem` scores.
    pub(crate) fn unidentified_basis(&self) -> Array2<f64> {
        let mut order: Vec<usize> = (0..self.eigenvalues.len()).collect();
        order.sort_by(|&left, &right| self.eigenvalues[right].total_cmp(&self.eigenvalues[left]));
        let dropped = &order[self.rank..];
        Array2::from_shape_fn((self.eigenvectors.nrows(), dropped.len()), |(row, column)| {
            self.eigenvectors[[row, dropped[column]]]
        })
    }
}

/// Certify at finalization that the fitted Hessian's identified rank is constant
/// over the outer certificate's own Newton step (#2901 V22), and return that
/// certificate with the step's largest coordinate.
///
/// `spectrum` is PIRLS's dense penalized Hessian's, in its transformed basis;
/// `hessian_rho` and `gradient` are the outer certificate's curvature and
/// gradient at ρ̂, and `railed` lists the coordinates it certified on a rail. The
/// curvature weights move through `β̂`: `ΔW_i = c_i·x_iᵀΔβ` with
/// `Δβ = Σ_k δρ_k·∂β̂/∂ρ_k` and `∂β̂/∂ρ_k = −H⁺λ_kS_kβ̂` over the identified
/// subspace the criterion priced, so row `i` moves by at most
/// `|c_i|·Σ_k |δρ_k|·|x_iᵀ∂β̂/∂ρ_k|`. Rows whose weight is not positive are
/// charged by their own weighted rank-one sum, in PIRLS's transformed basis. The
/// bounds are taken at this certified state, the one the criterion priced. Both
/// channels read the same penalties: the engine's projections `S̃_k = Π S_k Π`
/// that sum to the `S̃` in `H` ([`HessianSpectrumBounds`]).
pub(crate) fn certify_fitted_identified_rank(
    pirls: &crate::pirls::PirlsResult,
    spectrum: &FittedHessianSpectrum,
    lambdas: &Array1<f64>,
    design: &gam_linalg::matrix::DesignMatrix,
    hessian_rho: &Array2<f64>,
    gradient: &Array1<f64>,
    railed: &[usize],
) -> Result<(IdentifiedRankCertificate, f64), EstimationError> {
    let displacement = certificate_newton_displacement(hessian_rho, gradient, railed)?;
    let step_radius = displacement.iter().fold(0.0_f64, |acc, value| acc.max(*value));
    let penalties = pirls.reparam_result.applied_penalties().map_err(|error| {
        EstimationError::LayoutError(format!(
            "projecting the rank certificate's penalty blocks onto the reparameterization's \
             penalized subspace failed: {error}"
        ))
    })?;
    let eigenvalues = &spectrum.eigenvalues;
    let eigenvectors = &spectrum.eigenvectors;
    let penalty_rank = spectrum.penalty_rank;
    let rank = spectrum.rank;
    let rows = design.nrows();
    let weight_motion = if pirls.solve_c_nontrivial && step_radius > 0.0 {
        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&left, &right| eigenvalues[right].total_cmp(&eigenvalues[left]));
        let beta: &Array1<f64> = pirls.beta_transformed.as_ref();
        let qs = &pirls.reparam_result.qs;
        let mut eta_motion = Array1::<f64>::zeros(rows);
        for ((penalty, &lambda), &step) in penalties
            .iter()
            .zip(lambdas.iter())
            .zip(displacement.iter())
        {
            if step == 0.0 {
                continue;
            }
            let range = penalty.col_range.clone();
            let root_beta = penalty.root.dot(&beta.slice(s![range.clone()]));
            let mut penalty_gradient = Array1::<f64>::zeros(beta.len());
            penalty_gradient
                .slice_mut(s![range])
                .assign(&penalty.root.t().dot(&root_beta));
            let mut response = Array1::<f64>::zeros(beta.len());
            for &column in order.iter().take(rank) {
                let direction = eigenvectors.column(column);
                response.scaled_add(
                    -lambda * direction.dot(&penalty_gradient) / eigenvalues[column],
                    &direction,
                );
            }
            let row_response = design.apply_view(qs.dot(&response).view());
            eta_motion.zip_mut_with(&row_response, |motion, value| {
                *motion += step * value.abs();
            });
        }
        pirls
            .solve_c_array
            .iter()
            .zip(eta_motion.iter())
            .map(|(&derivative, &motion)| derivative.abs() * motion)
            .collect()
    } else {
        Array1::<f64>::zeros(rows)
    };
    let motion = HessianSpectrumMotion::over_weights(
        pirls.finalweights.view(),
        weight_motion.view(),
        |mass| {
            let columns = design.ncols();
            let mut gram = Array2::<f64>::zeros((columns, columns));
            for (row, &value) in mass.iter().enumerate() {
                if value > 0.0 {
                    design
                        .syr_row_into(row, value, &mut gram)
                        .map_err(EstimationError::InvalidInput)?;
                }
            }
            gam_linalg::matrix::symmetrize_in_place(&mut gram);
            let qs = &pirls.reparam_result.qs;
            Ok(qs.t().dot(&gram).dot(qs))
        },
    )?;
    let bounds = HessianSpectrumBounds::over_step(
        &spectrum.hessian,
        &pirls.reparam_result.s_transformed,
        penalties.iter().zip(lambdas.iter()).map(|(penalty, &lambda)| {
            (
                penalty.col_range.clone(),
                penalty.root.t().dot(&penalty.root) * lambda,
            )
        }),
        displacement.view(),
        motion,
    )?;
    certify_identified_rank_locally_constant(eigenvalues, rank, penalty_rank, &bounds)
        .map(|certificate| (certificate, step_radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The certificate at the rank the identified-subspace predicate prices.
    fn certify_at_identified_rank(
        eigenvalues: &[f64],
        penalty_rank: usize,
        bounds: &HessianSpectrumBounds,
    ) -> Result<IdentifiedRankCertificate, EstimationError> {
        certify_identified_rank_locally_constant(
            eigenvalues,
            DenseSpectralOperator::identified_rank(eigenvalues, penalty_rank),
            penalty_rank,
            bounds,
        )
    }

    /// A rank-2 PSD Hessian in three dimensions, rotated off the axes: the
    /// identified inverse is its Moore–Penrose pseudo-inverse, and a right-hand
    /// side with a null-space component solves to the min-norm answer.
    #[test]
    fn identified_inverse_is_the_pseudo_inverse_of_a_singular_hessian() {
        let s = 0.5_f64.sqrt();
        let q = array![[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let spectrum = array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let pseudo_spectrum = array![[1.0 / 3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let hessian = q.dot(&spectrum).dot(&q.t());
        let pseudo_inverse = q.dot(&pseudo_spectrum).dot(&q.t());

        let inverse = IdentifiedHessianInverse::from_dense(&hessian, 2).unwrap();
        assert_eq!(inverse.rank(), 2);
        let computed = inverse
            .certified_inverse(&hessian, "identified inverse test")
            .unwrap();
        for (value, expected) in computed.iter().zip(pseudo_inverse.iter()) {
            assert!((value - expected).abs() < 1e-12, "{computed:?} vs {pseudo_inverse:?}");
        }

        let rhs = array![[1.0], [2.0], [5.0]];
        let solution = inverse
            .certified_solve(&hessian, &rhs, "identified solve test")
            .unwrap();
        let expected = pseudo_inverse.dot(&rhs);
        for (value, reference) in solution.iter().zip(expected.iter()) {
            assert!((value - reference).abs() < 1e-12, "{solution:?} vs {expected:?}");
        }
    }

    /// Positive curvature weights that do not move.
    fn still_weights() -> HessianSpectrumMotion {
        let weights = Array1::from_elem(4, 1.0);
        let weight_motion = Array1::<f64>::zeros(4);
        HessianSpectrumMotion::over_weights(weights.view(), weight_motion.view(), |mass| {
            Ok(Array2::from_diag(mass))
        })
        .unwrap()
    }

    fn diagonal(values: &[f64]) -> Array2<f64> {
        Array2::from_diag(&Array1::from(values.to_vec()))
    }

    /// The bounds for `diag(hessian)` carrying the penalties `diag(penalty)`,
    /// each over its own step, with still weights.
    fn diagonal_bounds(hessian: &[f64], penalties: &[(&[f64], f64)]) -> HessianSpectrumBounds {
        let steps = Array1::from_iter(penalties.iter().map(|entry| entry.1));
        let engine = penalties.iter().fold(
            Array2::<f64>::zeros((hessian.len(), hessian.len())),
            |sum, entry| sum + diagonal(entry.0),
        );
        HessianSpectrumBounds::over_step(
            &diagonal(hessian),
            &engine,
            penalties
                .iter()
                .map(|entry| (0..hessian.len(), diagonal(entry.0))),
            steps.view(),
            still_weights(),
        )
        .unwrap()
    }

    /// A structural null beside well-resolved directions: the rank is the same
    /// everywhere a tenth of a log-unit reaches.
    #[test]
    fn a_resolved_spectrum_certifies_its_rank_over_the_step() {
        let spectrum = [3.0, 1.0, 0.5, 0.0];
        let bounds = diagonal_bounds(&spectrum, &[(&[0.0, 1.0, 0.5, 0.0], 0.1)]);
        let certificate = certify_at_identified_rank(&spectrum, 2, &bounds).unwrap();
        assert_eq!(certificate.rank, 3);
        assert_eq!(certificate.largest_unidentified, Some(0.0));
    }

    /// A penalized direction just under the band is certified at a zero step,
    /// and refused once its penalty's step can lift it over the band.
    #[test]
    fn an_unidentified_direction_straddling_the_band_refuses() {
        let band = 3.0 * f64::EPSILON;
        let spectrum = [1.0, 0.3, 0.6 * band];
        let penalty: &[f64] = &[0.0, 0.3, 0.6 * band];
        let pointwise = certify_at_identified_rank(
            &spectrum,
            2,
            &diagonal_bounds(&spectrum, &[(penalty, 0.0)]),
        )
        .unwrap();
        assert_eq!(pointwise.rank, 2);
        let refusal = certify_at_identified_rank(
            &spectrum,
            2,
            &diagonal_bounds(&spectrum, &[(penalty, 1.0)]),
        )
        .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// The certificate judges the rank it is handed, the one the criterion priced,
    /// not the identified-subspace predicate's (#2959 D1). The predicate drops a
    /// direction under the band, and the dropped set is certified. The same
    /// spectrum priced at full rank, as the root prices it, keeps that direction,
    /// and bounds judged at the assembled band cannot hold it above that band.
    #[test]
    fn the_certificate_judges_the_rank_the_criterion_priced_2959() {
        let band = 3.0 * f64::EPSILON;
        let spectrum = [1.0, 0.3, 0.6 * band];
        let penalty: &[f64] = &[0.0, 0.3, 0.6 * band];
        let bounds = diagonal_bounds(&spectrum, &[(penalty, 0.0)]);
        assert_eq!(DenseSpectralOperator::identified_rank(&spectrum, 2), 2);
        let dropped = certify_identified_rank_locally_constant(&spectrum, 2, 2, &bounds).unwrap();
        assert_eq!(dropped.rank, 2);
        assert_eq!(dropped.largest_unidentified, Some(0.6 * band));
        let priced_in_full =
            certify_identified_rank_locally_constant(&spectrum, 3, 2, &bounds).unwrap_err();
        assert!(
            matches!(
                priced_in_full,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 3, .. }
            ),
            "{priced_in_full}"
        );
    }

    /// A penalized direction just over the band, above the penalty-rank floor,
    /// is certified at a zero step and refused once its penalty's step can push
    /// it under the band.
    #[test]
    fn an_identified_direction_straddling_the_band_refuses() {
        let band = 3.0 * f64::EPSILON;
        let spectrum = [1.0, 1.5 * band, 0.0];
        let penalty: &[f64] = &[0.0, 1.5 * band, 0.0];
        let pointwise = certify_at_identified_rank(
            &spectrum,
            1,
            &diagonal_bounds(&spectrum, &[(penalty, 0.0)]),
        )
        .unwrap();
        assert_eq!(pointwise.rank, 2);
        let refusal = certify_at_identified_rank(
            &spectrum,
            1,
            &diagonal_bounds(&spectrum, &[(penalty, 1.0)]),
        )
        .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// A penalty railed near λ = 10¹⁵ dominates the spectrum and the band, but a
    /// data direction it does not reach stays far above that band and a
    /// structural null stays at zero, so the rank is certified.
    #[test]
    fn a_railed_penalty_does_not_charge_the_directions_it_does_not_reach() {
        let spectrum = [1.0e15, 5.0e14, 1.0e3, 0.0];
        let bounds = diagonal_bounds(&spectrum, &[(&[1.0e15, 5.0e14, 0.0, 0.0], 0.05)]);
        let certificate = certify_at_identified_rank(&spectrum, 2, &bounds).unwrap();
        assert_eq!(certificate.rank, 3);
    }

    /// #2901 V22: a coordinate's step is charged to its own penalty, so a large
    /// step on a penalty the smallest identified direction does not see moves
    /// that direction by nothing. This is the gamma-log Matérn refusal's shape: a
    /// data direction sets `‖H‖₂ = 4.9713e12`, the smallest identified direction
    /// (132.36) is data too, and the step of 5.48 falls on a penalty of a third
    /// direction. Scaling all of `H` by `e^{±5.48}` let σ_r fall to 0.552 under a
    /// band of 0.79 and refused. The control: a penalty that does carry the
    /// smallest identified direction, over a step that takes it under the band,
    /// still refuses.
    #[test]
    fn a_step_is_charged_only_to_the_directions_its_penalty_reaches_2901() {
        let spectrum = [4.9713e12, 1.0e5, 132.36];
        let flat: (&[f64], f64) = (&[0.0, 4.68e3, 0.0], 5.48);
        let certificate =
            certify_at_identified_rank(&spectrum, 1, &diagonal_bounds(&spectrum, &[flat]))
                .unwrap();
        assert_eq!(certificate.rank, 3);
        let reaching: (&[f64], f64) = (&[0.0, 0.0, 132.36], 14.66);
        let refusal = certify_at_identified_rank(
            &spectrum,
            2,
            &diagonal_bounds(&spectrum, &[flat, reaching]),
        )
        .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 3, .. }
            ),
            "{refusal}"
        );
    }

    /// #2901 V22: a rotated penalty root can leak onto the structural null
    /// coordinates, where the engine's penalty is exactly zero. With null
    /// coordinate 1, root `r = (1, 1e-4)` and `λ = 1e8`, the raw block `λ·rᵀr`
    /// carries 1e4 onto the null coordinate. Projected through the engine's
    /// primitive, the block is the engine's penalty `diag(1e8, 0)` and a unit step
    /// certifies. The raw block over the same step refuses a Hessian whose rank
    /// cannot change: it sends the data part indefinite and leaves a residual of
    /// 1.4e4 against the engine's penalty.
    #[test]
    fn a_leaking_penalty_root_is_restricted_to_the_penalized_block_2901() {
        let spectrum = [1.0e8 + 1.0, 1.0];
        let hessian = diagonal(&spectrum);
        let engine = diagonal(&[1.0e8, 0.0]);
        let leaking =
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[1.0, 1.0e-4]], 2);
        let projected = leaking
            .project_out_null_directions(array![[0.0], [1.0]].view())
            .unwrap();
        let bounds_for = |penalty: &gam_terms::construction::CanonicalPenalty| {
            HessianSpectrumBounds::over_step(
                &hessian,
                &engine,
                [(
                    penalty.col_range.clone(),
                    penalty.root.t().dot(&penalty.root) * 1.0e8,
                )],
                array![1.0].view(),
                still_weights(),
            )
            .unwrap()
        };
        let certificate =
            certify_at_identified_rank(&spectrum, 1, &bounds_for(&projected)).unwrap();
        assert_eq!(certificate.rank, 2);
        let refusal =
            certify_at_identified_rank(&spectrum, 1, &bounds_for(&leaking)).unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// #2901 V22: what the per-coordinate blocks do not reproduce of the engine's
    /// penalty is charged to every eigenvalue. With `H = diag(1, 0.3)`, blocks
    /// `diag(0, 0.2)` and an engine penalty `diag(0, 0.6)`, `R = diag(0, 0.4)`, so a
    /// still step bounds the spectrum by `σ_i ∓ 0.4` and the identified direction
    /// at 0.3 can reach −0.1. The same blocks against the engine penalty they
    /// reproduce certify: the charge, not the spectrum, is what refuses.
    #[test]
    fn an_engine_penalty_the_blocks_do_not_reproduce_is_charged_its_residual_2901() {
        let spectrum = [1.0, 0.3];
        let bounds_for = |engine: &[f64]| {
            HessianSpectrumBounds::over_step(
                &diagonal(&spectrum),
                &diagonal(engine),
                [(0..2, diagonal(&[0.0, 0.2]))],
                array![0.0].view(),
                still_weights(),
            )
            .unwrap()
        };
        let matching = certify_at_identified_rank(&spectrum, 1, &bounds_for(&[0.0, 0.2]))
            .unwrap();
        assert_eq!(matching.rank, 2);
        let charged = bounds_for(&[0.0, 0.6]);
        let charge = 0.6 - 0.2;
        let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&charged.upper);
        for (index, &value) in spectrum.iter().enumerate() {
            assert!(
                (charged.lower[index] - (value - charge)).abs() <= band
                    && (charged.upper[index] - (value + charge)).abs() <= band,
                "eigenvalue {index} ({value}) is bounded by [{:.6e}, {:.6e}], not by ±{charge}",
                charged.lower[index],
                charged.upper[index]
            );
        }
        let refusal =
            certify_at_identified_rank(&spectrum, 1, &charged).unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }

    /// #2901 V22: the certificate charges a step against the engine's own split.
    /// On a fixture with a structural null (first differences on coefficients
    /// 0..12 and second differences on 8..20 leave only the constant), the
    /// engine's `S̃` is exactly zero on the coordinate its null basis spans, and
    /// the blocks projected through its split reproduce `S̃` to rounding. The two
    /// sides are two formations of one exact matrix. The engine takes `S̃ = EᵀE`
    /// from a backward-stable decomposition of the stacked roots, and a Gram of a
    /// factor perturbed by `η` relative is perturbed by `2η + η²`, so `S̃` carries
    /// twice the band `p·ε·‖·‖₂` of a `p`-dimensional decomposition. The block sum
    /// carries its own band. The control is a basis that is not the engine's, the
    /// same basis with its coordinates reversed: `S̃` does not vanish there, and
    /// blocks projected through it miss `S̃` far outside that rounding. An engine
    /// that reorders its split fails here, where the certificate would only charge
    /// the residual and refuse.
    #[test]
    fn the_engines_split_is_where_its_penalty_vanishes_2901() {
        use gam_terms::construction::{
            CanonicalPenalty, precompute_reparam_invariant_from_canonical,
            stable_reparameterizationwith_invariant,
        };
        let p = 20;
        let difference_root = |start: usize, end: usize, stencil: &[f64]| {
            let rows = end - start + 1 - stencil.len();
            let mut root = Array2::<f64>::zeros((rows, p));
            for row in 0..rows {
                for (offset, &weight) in stencil.iter().enumerate() {
                    root[[row, start + row + offset]] = weight;
                }
            }
            CanonicalPenalty::from_dense_root(root, p)
        };
        let canonical = vec![
            difference_root(0, 12, &[-1.0, 1.0]),
            difference_root(8, 20, &[1.0, -2.0, 1.0]),
        ];
        let lambdas = [1.0e6, 3.0];
        let invariant = precompute_reparam_invariant_from_canonical(&canonical, p).unwrap();
        let reparam =
            stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &invariant).unwrap();
        assert_eq!(reparam.null_split().declared_null_dim(), 1);
        let engine = &reparam.s_transformed;
        let engine_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(
            &descending_spectrum(engine.clone()).unwrap(),
        );
        // The residual's spectral norm, against the rounding both formations carry.
        let residual_for = |penalties: &[CanonicalPenalty]| {
            let mut sum = Array2::<f64>::zeros(engine.dim());
            for (penalty, &lambda) in penalties.iter().zip(lambdas.iter()) {
                sum.slice_mut(s![penalty.col_range.clone(), penalty.col_range.clone()])
                    .scaled_add(lambda, &penalty.root.t().dot(&penalty.root));
            }
            let sum_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(
                &descending_spectrum(sum.clone()).unwrap(),
            );
            let residual = descending_spectrum(engine - &sum)
                .unwrap()
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            (residual, 2.0 * engine_band + sum_band)
        };
        // An orthonormal basis of m columns carries total mass m over its rows; a
        // basis aligned with coordinates puts one unit on each of m rows and only
        // rounding elsewhere, so the rows holding more than half a unit are its
        // coordinates.
        let spanned_coordinates = |basis: &Array2<f64>| -> Vec<usize> {
            (0..basis.nrows())
                .filter(|&row| basis.row(row).dot(&basis.row(row)) > 0.5)
                .collect()
        };
        let null_coordinates = spanned_coordinates(&reparam.u_truncated);
        assert_eq!(null_coordinates.len(), 1, "{}", reparam.u_truncated);
        for &coordinate in &null_coordinates {
            assert!(
                engine
                    .row(coordinate)
                    .iter()
                    .chain(engine.column(coordinate).iter())
                    .all(|&value| value == 0.0),
                "the engine's penalty does not vanish on its null coordinate {coordinate}"
            );
        }
        let applied = reparam.applied_penalties().unwrap();
        let (residual, band) = residual_for(&applied);
        assert!(
            residual <= band,
            "the engine's split: blocks miss S̃ by {residual:.4e}, rounding {band:.4e}"
        );

        let reversed = reparam.u_truncated.slice(s![..;-1, ..]).to_owned();
        let reversed_coordinates = spanned_coordinates(&reversed);
        assert!(
            reversed_coordinates
                .iter()
                .any(|&coordinate| engine.row(coordinate).iter().any(|&value| value != 0.0)),
            "the reversed basis must name a penalized coordinate"
        );
        let misprojected = reparam
            .canonical_transformed
            .iter()
            .map(|penalty| penalty.project_out_null_directions(reversed.view()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let (misprojected_residual, misprojected_band) = residual_for(&misprojected);
        assert!(
            misprojected_residual > misprojected_band,
            "a basis that is not the engine's: blocks miss S̃ by {misprojected_residual:.4e}, \
             rounding {misprojected_band:.4e}"
        );
    }

    /// The bounds hold the spectrum wherever the step reaches: a rotated data
    /// block and two overlapping penalties, re-assembled at every corner of the
    /// step and at interior points, keep each eigenvalue inside
    /// `[σ_i(H₋), σ_i(H₊)]`.
    #[test]
    fn the_bounds_hold_the_spectrum_everywhere_the_step_reaches() {
        let s = 0.5_f64.sqrt();
        let rotation = array![[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let data = rotation.dot(&diagonal(&[2.0, 0.5, 0.25])).dot(&rotation.t());
        let first = array![[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.3]];
        let second = rotation.dot(&diagonal(&[0.0, 3.0, 1.0])).dot(&rotation.t());
        let hessian = &data + &first + &second;
        let steps = array![1.3, 0.4];
        let bounds = HessianSpectrumBounds::over_step(
            &hessian,
            &(&first + &second),
            [(0..3, first.clone()), (0..3, second.clone())],
            steps.view(),
            still_weights(),
        )
        .unwrap();
        let tolerance = 16.0 * f64::EPSILON * bounds.upper[0];
        for (first_step, second_step) in [
            (1.3, 0.4),
            (-1.3, 0.4),
            (1.3, -0.4),
            (-1.3, -0.4),
            (0.7, -0.1),
            (-0.2, 0.3),
        ] {
            let displaced = &data + &(&first * f64::exp(first_step)) + &(&second * f64::exp(second_step));
            let values = descending_spectrum(displaced).unwrap();
            for (index, &value) in values.iter().enumerate() {
                assert!(
                    bounds.lower[index] - tolerance <= value && value <= bounds.upper[index] + tolerance,
                    "eigenvalue {index} at steps ({first_step}, {second_step}) is {value:.6e}, outside \
                     [{:.6e}, {:.6e}]",
                    bounds.lower[index],
                    bounds.upper[index]
                );
            }
        }
    }

    /// A coordinate certified on a rail is certified by its tail, so only the
    /// interior Newton displacement enters the step.
    #[test]
    fn a_railed_coordinate_moves_nothing_in_the_certificate_step() {
        let hessian_rho = array![[2.0, 0.0], [0.0, 1.0e-3]];
        let gradient = array![1.0e-4, 5.0e-2];
        let railed = certificate_newton_displacement(&hessian_rho, &gradient, &[1]).unwrap();
        assert!((railed[0] - 5.0e-5).abs() < 1e-18, "{railed}");
        assert_eq!(railed[1], 0.0);
        let interior = certificate_newton_displacement(&hessian_rho, &gradient, &[]).unwrap();
        assert!((interior[1] - 50.0).abs() < 1e-10, "{interior}");
    }

    /// A row whose curvature weight is negative is charged by its own row mass
    /// `m·|W_i| + ΔW_i`, and voids the penalty-rank floor.
    #[test]
    fn a_negative_weight_row_is_charged_its_own_mass() {
        let weights = array![1.0, -0.5];
        let weight_motion = array![0.1, 0.2];
        let motion =
            HessianSpectrumMotion::over_weights(weights.view(), weight_motion.view(), |mass| {
                Ok(Array2::from_diag(mass))
            })
            .unwrap();
        assert!((motion.relative_weight_motion - 0.1).abs() < 1e-15);
        let rows = motion.nonpositive_rows.as_ref().expect("the negative row carries mass");
        assert_eq!(rows[[0, 0]], 0.0);
        assert!((rows[[1, 1]] - 0.25).abs() < 1e-15);
        assert!(!motion.weights_stay_nonnegative);
    }

    /// #2901 V22: split by sign, a row whose weight is negative is bounded by
    /// `W_i ∓ ΔW_i`, not scaled with the positive rows. With `X = I`,
    /// `W = [1, −0.4]` and a penalty of 0.5 on the second direction, `H =
    /// diag(1, 0.1)` has rank 2. Moving the negative row by 0.1 takes `H'` to
    /// `diag(1, 0)`, rank 1, so the rank is not constant and the certificate must
    /// refuse. The same spectrum with that row still certifies: the charge, not
    /// the spectrum, is what refuses.
    #[test]
    fn a_negative_weight_rows_motion_refuses_the_direction_it_can_empty_2901() {
        let spectrum = [1.0, 0.1];
        let penalty = (0..2, diagonal(&[0.0, 0.5]));
        let weights = array![1.0, -0.4];
        let bounds_for = |weight_motion: Array1<f64>| {
            let motion =
                HessianSpectrumMotion::over_weights(weights.view(), weight_motion.view(), |mass| {
                    Ok(Array2::from_diag(mass))
                })
                .unwrap();
            HessianSpectrumBounds::over_step(
                &diagonal(&spectrum),
                &diagonal(&[0.0, 0.5]),
                [penalty.clone()],
                array![0.0].view(),
                motion,
            )
            .unwrap()
        };
        let still = certify_at_identified_rank(&spectrum, 1, &bounds_for(array![0.0, 0.0]))
            .unwrap();
        assert_eq!(still.rank, 2);
        let displaced = [1.0, -0.4 - 0.1 + 0.5];
        assert_eq!(DenseSpectralOperator::identified_rank(&displaced, 1), 1);
        let refusal =
            certify_at_identified_rank(&spectrum, 1, &bounds_for(array![0.0, 0.1]))
                .unwrap_err();
        assert!(
            matches!(
                refusal,
                EstimationError::IdentifiedRankNotLocallyConstant { rank: 2, .. }
            ),
            "{refusal}"
        );
    }
}
