//! Posterior-predictive class probabilities for the penalized multinomial
//! logit, computed as a RATIO OF NORMALISING CONSTANTS.
//!
//! # Why this exists rather than integrating the Gaussian posterior
//!
//! The published estimand is the posterior mean probability
//! `E[softmax(x'β) | data]`.  The obvious implementation — approximate the
//! posterior of `β` by the Laplace Gaussian `N(β̂, H⁻¹)` and integrate `softmax`
//! against it — is **not a valid approximation of that estimand**, and the
//! failure is not small.
//!
//! Write the posterior mean of any positive functional `g(β)` as a ratio of
//! integrals.  Laplace applied SEPARATELY to numerator and denominator (the
//! "fully exponential" form of Tierney and Kadane) has its `O(n⁻¹)` errors
//! cancel between the two, leaving `O(n⁻²)`.  Integrating `g` against the
//! Gaussian instead keeps only the CURVATURE half of the `O(n⁻¹)` correction
//! (`½ tr(H⁻¹ ∇²g)`) and silently drops the SKEWNESS half, which comes from the
//! third derivative of the log-posterior.  On a well-conditioned fit the two
//! halves are both small and nobody notices.  On a (quasi-)separated
//! multinomial they are not small and they have opposite signs: the likelihood
//! is flat toward more separation and steep away from it, so the true posterior
//! is strongly skewed toward LARGER `|η|`, while the symmetric Gaussian puts
//! half of its mass on the side the likelihood has already excluded.  `softmax`
//! is concave along the winning coordinate, so that misplaced mass converts
//! directly into under-confidence: right argmax, flattened probabilities.
//!
//! For `g = p_c(x) = P(new row at x is class c | β)` the ratio is not merely a
//! device — it is exactly the posterior predictive, because the extra row's
//! likelihood factor IS the functional being averaged:
//!
//! ```text
//!     E[p_c(x) | D]  =  Z(D ∪ {(x, c)}) / Z(D)
//! ```
//!
//! with `Z` the posterior normalising constant.  Approximating each `Z` by
//! Laplace at its own mode gives
//!
//! ```text
//!     E[p_c(x)] ≈ exp( L⁺(β̂⁺) − L(β̂) ) · sqrt( det H / det H⁺ )
//! ```
//!
//! where `L` is the penalized log-posterior, `β̂⁺` the mode with the extra row
//! present, and `H`, `H⁺` the corresponding negative Hessians.  The `(2π)^{d/2}`
//! factors cancel exactly (same dimension on both sides).
//!
//! The identity `Σ_c E[p_c(x)] = 1` is exact for the true integrals, so the
//! deviation of the computed `Σ_c` from one is a MEASURED accuracy statement
//! about this approximation, available at every prediction row and requiring no
//! reference. Every row publishes its renormalised mean together with that
//! defect (see [`MultinomialPredictiveMoments::mass_defect`]), and the decisions
//! it drives are per row: a `(1 − α)` interval is declined exactly on the rows
//! whose missing mass exceeds `α`.
//!
//! The same machinery supplies the second moments the standard-error surface
//! consumes, with two extra rows instead of one:
//!
//! ```text
//!     E[p_c(x) · p_d(x)]  =  Z(D ∪ {(x, c), (x, d)}) / Z(D)
//! ```
//!
//! # Cost
//!
//! One warm-started Newton solve per (row, class), plus `K(K+1)/2` more per row
//! when second moments are requested. Every objective is expanded around the
//! published mode `β̂` as `ℓ(θ) + s(β̂)ᵀo − ½ oᵀ Q o`, with `o = θ − β̂` and
//! `s = −∇ℓ`, so its curvature is `XᵀW(θ)X + Q` plus the extra rows' (see
//! [`MultinomialPredictiveModel::terminal_precision`] for `Q`). Where `Q` is
//! positive definite every augmented objective is strictly concave, so Newton
//! with backtracking is unconditionally safe. A fitted extra term brings the
//! second-order completion `−½ tr(K H''[e_a, e_b])`, which is indefinite in
//! general, so `Q` need not be positive definite, and where it is not the
//! second-order posterior is improper. `predictive_moments` therefore certifies
//! `Q` against its rounding band before any solve and integrates it only where it
//! is positive definite (see `terminal_precision` for what it does otherwise), so
//! the Cholesky refusal in `augmented_mode` is a numerical exit only. Each Newton
//! iteration is `O(n·M²·P²)` for the curvature (as `M(M+1)/2` GEMMs) and
//! `O(d³)` for the factorisation, so the whole predictive is
//! `O(R·K·iters·(n M² P² + d³))`. The `R` prediction rows are independent and
//! run in parallel.
//!
//! On the fixture this exists for that is a large improvement, not a cost: the
//! Smolyak integrator it replaces spent ~930 s on one penguins prediction
//! block, because its level requirement grows with exactly the posterior width
//! that makes the Gaussian wrong in the first place, while the same block here
//! is `n = 228`, `P = 37`, `M = 2` — under a second. The scaling is different
//! in kind, though, and worth stating plainly: this method's cost grows with
//! the TRAINING size, which the Gaussian route's did not, because evaluating a
//! posterior away from its mode is what the Gaussian route was avoiding by
//! being wrong.

use crate::model_types::EstimationError;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// The training data and penalty a saved multinomial model needs in order to
/// evaluate its own log-posterior away from the mode.
///
/// A Laplace SUMMARY (`β̂`, `H⁻¹`) is not enough to compute a posterior mean:
/// the summary is precisely the quadratic model whose inadequacy is the defect.
/// The predictive therefore needs the likelihood itself, which means the rows.
#[derive(Debug, Clone, Copy)]
pub struct MultinomialPredictiveModel<'a> {
    /// Training design in the SAME (raw) basis as the saved coefficients and as
    /// the design rebuilt for prediction, shape `(n, P)`.
    pub training_design: ArrayView2<'a, f64>,
    /// Training class index per row, values in `0..K`, aligned to
    /// `class_levels`.
    pub training_class_index: &'a [u32],
    /// Training row weights, length `n`.
    pub training_weights: ArrayView1<'a, f64>,
    /// The joint penalty `S_λ` at the selected smoothing parameters, in the
    /// stacked class-major coefficient order `θ[a·P + i] = β[i, a]`, shape
    /// `(P·M, P·M)` with `M = K − 1`.
    pub joint_penalty: ArrayView2<'a, f64>,
    /// Total class count `K` (the reference class `K − 1` carries `η ≡ 0`).
    pub n_classes: usize,
    /// The fitted objective's own terminal posterior precision at the published
    /// mode, in the same stacked order, shape `(P·M, P·M)`. On a fit that armed a
    /// proper prior it is `T = H + S_λ + H_Φ + completion`, the matrix the saved
    /// covariance inverts. `None` when the fitted objective is `ℓ − ½θ'S_λθ`
    /// itself.
    ///
    /// What the objective carries beyond the likelihood has curvature
    /// `Q = T − Xᵀ W(β̂) X` at the mode: `S_λ` plus the extra term's `−∇²Φ(β̂)`.
    /// Around `β̂`, `−½θ'S_λθ + Φ(θ)` is `s(β̂)ᵀo − ½ oᵀ Q o` to second order and up
    /// to a constant, where `o = θ − β̂` and `s(β̂) = −∇ℓ(β̂)` is what the fit's
    /// stationarity balances. So the predictive integrates `ℓ(θ) + s(β̂)ᵀo − ½ oᵀ Q o`,
    /// and each ratio integrates the posterior the published coefficients belong to
    /// (#1082). Without a fitted term `Q = S_λ`, and that objective is
    /// `ℓ − ½θ'S_λθ` up to a constant.
    ///
    /// `Q` is formed once, as `T` less the likelihood curvature. It is never
    /// rebuilt as `S_λ + (T − XᵀWX − S_λ)`: where `λ` is large `S_λ` dwarfs both
    /// other terms, and that round trip leaves rounding at the scale of `S_λ` in
    /// every precision the predictive factors.
    ///
    /// `Q` is integrated only where its spectrum certifies it positive definite
    /// against its rounding band `p·ε·‖Q‖₂`. A negative direction makes the
    /// second-order posterior improper, because `ℓ ≤ 0` while `−½ oᵀ Q o` grows
    /// along it, and an unresolved one cannot be certified either way. Such a fit
    /// has no second-order spread to publish: second moments decline with
    /// `EstimationError::PredictiveIntervalsDeclined`, carrying `Q`'s inertia. The
    /// point moments never need `T`, and integrate `ℓ − ½θ'S_λθ` with the measured
    /// tilt, the published posterior with its extra term carried to first order.
    pub terminal_precision: Option<ArrayView2<'a, f64>>,
}

/// Posterior-predictive moments at a block of prediction rows.
#[derive(Debug, Clone)]
pub struct MultinomialPredictiveMoments {
    /// `E[p_c(x)]`, shape `(R, K)`, rows summing to one.
    pub class_mean: Array2<f64>,
    /// `E[p_c(x) · p_d(x)]`, shape `(R, K, K)`, present only when second
    /// moments were requested.
    pub class_second_moment: Option<Array3<f64>>,
    /// Per-row `|Σ_c E[p_c] − 1|` BEFORE renormalisation — the approximation's
    /// own measured error at that row, published beside the renormalised mean.
    ///
    /// It over-states the error that survives renormalisation. Two independent
    /// fixtures measured by how much:
    ///
    /// ```text
    ///   K = 3, p = 10, asymmetric quasi-separated, MCMC truth
    ///       worst-row defect 1.37e-2   worst-row error after normalising 4.4e-3   (3.1x)
    ///   K = 2, p = 2,  quasi-separated, exact 2-D quadrature truth
    ///       worst-row defect 4.93e-3   worst-row error after normalising 5.2e-4   (9.5x)
    /// ```
    ///
    /// No block of rows is refused on it. A `(1 − α)` interval cannot be certified
    /// on a row whose missing mass exceeds `α`, so that row's interval is declined
    /// and every other row's is published (#1082).
    pub mass_defect: Array1<f64>,
}

/// One extra observation appended to the training data: a design row and the
/// class it is assigned.
#[derive(Debug, Clone, Copy)]
struct ExtraRow<'a> {
    design: ArrayView1<'a, f64>,
    class: usize,
}

/// The fitted quadratic `Q = T − XᵀW(β̂)X` and its inertia against its rounding
/// band. See [`MultinomialPredictiveModel::terminal_precision`].
struct FittedQuadratic {
    quadratic: Array2<f64>,
    positive: usize,
    negative: usize,
    unresolved: usize,
    lowest_eigenvalue: f64,
    rounding_band: f64,
}

impl<'a> MultinomialPredictiveModel<'a> {
    fn active_classes(&self) -> usize {
        self.n_classes.saturating_sub(1)
    }

    fn coefficient_dim(&self) -> usize {
        self.training_design.ncols() * self.active_classes()
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let n = self.training_design.nrows();
        let p = self.training_design.ncols();
        let m = self.active_classes();
        if self.n_classes < 2 {
            crate::bail_invalid_estim!(
                "multinomial predictive requires K >= 2 classes, got {}",
                self.n_classes
            );
        }
        if self.training_class_index.len() != n || self.training_weights.len() != n {
            crate::bail_invalid_estim!(
                "multinomial predictive training frame has {n} design rows, {} labels and {} \
                 weights",
                self.training_class_index.len(),
                self.training_weights.len(),
            );
        }
        if let Some(bad) = self
            .training_class_index
            .iter()
            .find(|&&c| c as usize >= self.n_classes)
        {
            crate::bail_invalid_estim!(
                "multinomial predictive training label {bad} is outside 0..{}",
                self.n_classes
            );
        }
        if self.training_weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            crate::bail_invalid_estim!(
                "multinomial predictive training weights must be finite and non-negative"
            );
        }
        if self.training_design.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_estim!("multinomial predictive training design must be finite");
        }
        let d = p * m;
        if self.joint_penalty.dim() != (d, d) {
            crate::bail_invalid_estim!(
                "multinomial predictive joint penalty is {}x{}, expected {d}x{d}",
                self.joint_penalty.nrows(),
                self.joint_penalty.ncols(),
            );
        }
        if self.joint_penalty.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_estim!("multinomial predictive joint penalty must be finite");
        }
        if let Some(terminal) = self.terminal_precision {
            if terminal.dim() != (d, d) {
                crate::bail_invalid_estim!(
                    "multinomial predictive terminal precision is {}x{}, expected {d}x{d}",
                    terminal.nrows(),
                    terminal.ncols(),
                );
            }
            if terminal.iter().any(|v| !v.is_finite()) {
                crate::bail_invalid_estim!(
                    "multinomial predictive terminal precision must be finite"
                );
            }
        }
        Ok(())
    }

    /// Softmax probabilities of one row's active logits, with the reference
    /// class pinned at `η = 0`. Written with the max subtracted so a saturated
    /// logit cannot overflow.
    fn row_probabilities(&self, eta: &[f64], out: &mut [f64]) {
        let shift = eta.iter().copied().fold(0.0_f64, f64::max);
        let mut total = (-shift).exp();
        for (a, &value) in eta.iter().enumerate() {
            let e = (value - shift).exp();
            out[a] = e;
            total += e;
        }
        out[self.n_classes - 1] = (-shift).exp();
        for value in out.iter_mut() {
            *value /= total;
        }
    }

    /// `log Σ_k exp(η_k)` over the active logits plus the pinned reference `0`.
    fn row_log_partition(eta: &[f64]) -> f64 {
        let shift = eta.iter().copied().fold(0.0_f64, f64::max);
        let mut total = (-shift).exp();
        for &value in eta {
            total += (value - shift).exp();
        }
        shift + total.ln()
    }

    fn row_eta(&self, design_row: ArrayView1<'_, f64>, theta: &[f64], eta: &mut [f64]) {
        let p = self.training_design.ncols();
        for (a, slot) in eta.iter_mut().enumerate() {
            let block = &theta[a * p..(a + 1) * p];
            *slot = design_row
                .iter()
                .zip(block.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>();
        }
    }

    /// The objective this predictive integrates, expanded around `anchor = β̂`:
    /// `ℓ(θ) + tiltᵀo − ½ oᵀ Q o` with `o = θ − β̂` and `Q = self.joint_penalty`,
    /// optionally with extra observations appended (see `terminal_precision`).
    ///
    /// See [`Self::stationarity_tilt`] for what the tilt is and why it is measured
    /// rather than assumed.
    fn log_posterior(
        &self,
        theta: &[f64],
        extra: &[ExtraRow<'_>],
        anchor: &[f64],
        tilt: &[f64],
    ) -> f64 {
        let m = self.active_classes();
        let mut eta = vec![0.0_f64; m];
        let mut total = 0.0_f64;
        for (row, &label) in self.training_class_index.iter().enumerate() {
            let weight = self.training_weights[row];
            if weight == 0.0 {
                continue;
            }
            self.row_eta(self.training_design.row(row), theta, &mut eta);
            let picked = if (label as usize) < m {
                eta[label as usize]
            } else {
                0.0
            };
            total += weight * (picked - Self::row_log_partition(&eta));
        }
        for row in extra {
            self.row_eta(row.design, theta, &mut eta);
            let picked = if row.class < m { eta[row.class] } else { 0.0 };
            total += picked - Self::row_log_partition(&eta);
        }
        let offset: Vec<f64> = theta
            .iter()
            .zip(anchor.iter())
            .map(|(t, a)| t - a)
            .collect();
        let mut quadratic = 0.0_f64;
        for (i, &oi) in offset.iter().enumerate() {
            let mut acc = 0.0_f64;
            for (j, &oj) in offset.iter().enumerate() {
                acc += self.joint_penalty[[i, j]] * oj;
            }
            quadratic += oi * acc;
        }
        let linear: f64 = offset.iter().zip(tilt.iter()).map(|(o, c)| o * c).sum();
        total - 0.5 * quadratic + linear
    }

    /// The training rows' part of the NEGATIVE log-posterior: their gradient
    /// `Xᵀ(p(θ) − y)`, returned, and their curvature `XᵀW(θ)X`, added into
    /// `curvature`, both in the stacked class-major order.
    ///
    /// The `(a, b)` curvature block is `Xᵀ diag(w_ab) X` with
    /// `w_ab[row] = weight · p_a (δ_ab − p_b)`, which is a GEMM. Accumulating it
    /// row-by-row instead would be the same flops with none of the locality,
    /// and this is the inner loop of every augmented mode: one per (prediction
    /// row, class).
    fn add_training_rows(&self, theta: &[f64], curvature: &mut Array2<f64>) -> Array1<f64> {
        let n = self.training_design.nrows();
        let p = self.training_design.ncols();
        let m = self.active_classes();
        let mut gradient = Array1::<f64>::zeros(p * m);
        let mut eta = vec![0.0_f64; m];
        let mut probs = vec![0.0_f64; self.n_classes];
        // `curvature_weights[(row, a * m + b)]` is the row's contribution to the
        // `(a, b)` block, kept as a column so each block is one GEMM.
        let mut curvature_weights = Array2::<f64>::zeros((n, m * m));

        for (row, &label) in self.training_class_index.iter().enumerate() {
            let weight = self.training_weights[row];
            if weight == 0.0 {
                continue;
            }
            let design_row = self.training_design.row(row);
            self.row_eta(design_row, theta, &mut eta);
            self.row_probabilities(&eta, &mut probs);
            let label = label as usize;
            for a in 0..m {
                let residual = weight * (probs[a] - if label == a { 1.0 } else { 0.0 });
                for (i, &xi) in design_row.iter().enumerate() {
                    gradient[a * p + i] += residual * xi;
                }
                for b in 0..m {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    curvature_weights[[row, a * m + b]] = weight * probs[a] * (delta - probs[b]);
                }
            }
        }

        for a in 0..m {
            for b in a..m {
                let column = curvature_weights.column(a * m + b);
                let mut scaled = self.training_design.to_owned();
                for (mut design_row, &w) in scaled.rows_mut().into_iter().zip(column.iter()) {
                    design_row.map_inplace(|value| *value *= w);
                }
                let block = self.training_design.t().dot(&scaled);
                for i in 0..p {
                    for j in 0..p {
                        curvature[[a * p + i, b * p + j]] += block[[i, j]];
                        if a != b {
                            // `w_ab = w·p_a(δ_ab − p_b)` is symmetric in `(a, b)`,
                            // so the mirrored block is the transpose and does not
                            // need its own GEMM.
                            curvature[[b * p + j, a * p + i]] += block[[i, j]];
                        }
                    }
                }
            }
        }
        gradient
    }

    /// The training likelihood's gradient `∇ℓ(θ) = Xᵀ(y − p(θ))` and its Fisher
    /// information `XᵀW(θ)X`, both in the stacked class-major order. The softmax
    /// link is canonical, so the information is also the observed curvature.
    pub(crate) fn likelihood_gradient_and_information(
        &self,
        theta: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let d = self.coefficient_dim();
        let mut information = Array2::<f64>::zeros((d, d));
        let negative_gradient = self.add_training_rows(theta, &mut information);
        (-negative_gradient, information)
    }

    /// Gradient of the NEGATIVE of [`Self::log_posterior`] and its Hessian, both
    /// in the stacked class-major order.
    fn gradient_and_precision(
        &self,
        theta: &[f64],
        extra: &[ExtraRow<'_>],
        anchor: &[f64],
        tilt: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let training = self.training_state(theta);
        self.finish_state(theta, extra, anchor, tilt, training)
    }

    /// The `θ`-dependent part of [`Self::gradient_and_precision`] that does not
    /// involve the extra rows: the training rows' gradient and `Q + XᵀW(θ)X`.
    /// It is the `O(n·M²·P²)` part, and every augmented solve starts at the same
    /// base mode, so it is computed there once and handed to each solve's first
    /// iteration rather than rebuilt per (row, class).
    fn training_state(&self, theta: &[f64]) -> (Array1<f64>, Array2<f64>) {
        let mut precision = self.joint_penalty.to_owned();
        let gradient = self.add_training_rows(theta, &mut precision);
        (gradient, precision)
    }

    /// Completes a [`Self::training_state`] at the same `θ` with the extra rows,
    /// the quadratic's gradient and the tilt, in the order the full assembly
    /// always used, so a cached training state gives a bit-identical result.
    fn finish_state(
        &self,
        theta: &[f64],
        extra: &[ExtraRow<'_>],
        anchor: &[f64],
        tilt: &[f64],
        training: (Array1<f64>, Array2<f64>),
    ) -> (Array1<f64>, Array2<f64>) {
        let p = self.training_design.ncols();
        let m = self.active_classes();
        let d = p * m;
        let (mut gradient, mut precision) = training;
        let mut eta = vec![0.0_f64; m];
        let mut probs = vec![0.0_f64; self.n_classes];

        // The extra observations are rank-`m` and there are at most two of them,
        // so they are accumulated directly rather than through another GEMM.
        for row in extra {
            self.row_eta(row.design, theta, &mut eta);
            self.row_probabilities(&eta, &mut probs);
            for a in 0..m {
                let residual = probs[a] - if row.class == a { 1.0 } else { 0.0 };
                for (i, &xi) in row.design.iter().enumerate() {
                    gradient[a * p + i] += residual * xi;
                }
            }
            for a in 0..m {
                for b in 0..m {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    let w = probs[a] * (delta - probs[b]);
                    if w == 0.0 {
                        continue;
                    }
                    for (i, &xi) in row.design.iter().enumerate() {
                        let scaled = w * xi;
                        if scaled == 0.0 {
                            continue;
                        }
                        for (j, &xj) in row.design.iter().enumerate() {
                            precision[[a * p + i, b * p + j]] += scaled * xj;
                        }
                    }
                }
            }
        }

        // The quadratic's contribution to the gradient of the NEGATIVE
        // log-posterior is `Q(θ − β̂)`; the tilt's is `−tilt`. Neither touches the
        // curvature — `Q` is already in `precision` and a linear term has no
        // second derivative.
        for i in 0..d {
            let mut acc = 0.0_f64;
            for j in 0..d {
                acc += self.joint_penalty[[i, j]] * (theta[j] - anchor[j]);
            }
            gradient[i] += acc - tilt[i];
        }
        (gradient, precision)
    }

    /// `Q = T − XᵀW(β̂)X` at `anchor`, symmetrized, with its inertia against its
    /// own rounding band (see `terminal_precision`).
    fn fitted_quadratic(
        &self,
        anchor: &[f64],
        terminal: ArrayView2<'_, f64>,
    ) -> Result<FittedQuadratic, EstimationError> {
        let d = self.coefficient_dim();
        let mut likelihood_curvature = Array2::<f64>::zeros((d, d));
        self.add_training_rows(anchor, &mut likelihood_curvature);
        let mut quadratic = terminal.to_owned();
        quadratic -= &likelihood_curvature;
        for i in 0..d {
            for j in (i + 1)..d {
                let average = 0.5 * (quadratic[[i, j]] + quadratic[[j, i]]);
                quadratic[[i, j]] = average;
                quadratic[[j, i]] = average;
            }
        }
        let eigenvalues = quadratic
            .eigh(faer::Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?
            .0
            .to_vec();
        let rounding_band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&eigenvalues);
        let positive = eigenvalues
            .iter()
            .filter(|value| **value > rounding_band)
            .count();
        let negative = eigenvalues
            .iter()
            .filter(|value| **value < -rounding_band)
            .count();
        let lowest_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(FittedQuadratic {
            quadratic,
            positive,
            negative,
            unresolved: eigenvalues.len() - positive - negative,
            lowest_eigenvalue,
            rounding_band,
        })
    }

    /// The linear term that makes the SUPPLIED coefficients an exact stationary
    /// point of the objective this predictive integrates.
    ///
    /// # What it is measuring
    ///
    /// A Laplace ratio is only a Laplace ratio if its denominator is expanded at
    /// a mode. The published coefficients are the mode of the objective the FIT
    /// maximised, and that objective is not always `ℓ − ½θ'S_λθ`: the multinomial
    /// formula path arms a Jeffreys/Firth term `Φ` on separation evidence, and
    /// on the geometry where it arms, `Φ`'s `O(1)` gradient is what pins a
    /// direction the penalized likelihood leaves at `O(λ)`. Integrating the
    /// bare penalized likelihood against a mode that belongs to `ℓ − ½θ'S_λθ + Φ`
    /// would take every ratio against a posterior the published coefficients do
    /// not live in.
    ///
    /// The first-order content of everything the objective carries beyond the
    /// likelihood is `−S_λβ̂ + ∇Φ(β̂)`, and by stationarity of the fit's own mode
    /// it is EXACTLY `s(β̂) = −∇ℓ(β̂)`, the likelihood's part of the gradient at the
    /// published point. So it is *measured* here rather than reconstructed.
    /// Carrying it as a linear term is free: it moves no curvature (a linear
    /// function has no second derivative), so the augmented modes and their
    /// log-determinants stay the ones the ratio needs.
    ///
    /// When the fit carried no extra term, the tilt differs from `−S_λβ̂` by the
    /// inner solve's own residual gradient — orders below anything that matters,
    /// and absorbing it makes the base exactly stationary rather than nearly so,
    /// which is a strict improvement on using the raw mode.
    ///
    /// The extra term's CURVATURE cannot be measured from the gradient, and it is
    /// not second-order in the ratio where the term arms: there `Φ`'s curvature is
    /// `O(1)` along directions the penalized likelihood leaves at `O(λ)`, so it
    /// sets how far each augmented mode moves. It is carried in the quadratic `Q`
    /// read from the fit's own terminal precision (see `terminal_precision`). The
    /// mass-defect identity below cannot stand in for it: `Σ_c E[p_c] = 1` holds
    /// for whichever posterior the ratios integrate, so it measures the Laplace
    /// expansion, not whether the right posterior was integrated (#1082).
    fn stationarity_tilt(&self, mode: &[f64]) -> Array1<f64> {
        let zero = vec![0.0_f64; mode.len()];
        let (gradient, _) = self.gradient_and_precision(mode, &[], mode, &zero);
        gradient
    }

    /// Newton with backtracking on the negative log-posterior, warm-started at
    /// `start`. Its Hessian is `XᵀW(θ)X + Q` plus the extra rows' curvature, which
    /// is positive definite everywhere when `Q` is, so the objective is then
    /// strictly convex. A precision that is not positive definite is refused rather
    /// than stepped on (see the module doc's Cost section).
    ///
    /// Returns the mode, the objective value there, and the log-determinant of
    /// the precision at that point — the three quantities a Laplace ratio needs
    /// from one side of it.
    ///
    /// `start_training`, when given, is [`Self::training_state`] at `start`; the
    /// first iteration then only adds the extra rows to it.
    ///
    /// Both exits are derived rather than budgeted: the Newton decrement falls
    /// inside the objective's rounding band, or no representable step along the
    /// Newton direction gains more than that band. Every step that continues
    /// gains more than the band and the log-posterior is bounded above, so the
    /// iteration ends on its own.
    fn augmented_mode(
        &self,
        start: &[f64],
        start_training: Option<&(Array1<f64>, Array2<f64>)>,
        extra: &[ExtraRow<'_>],
        anchor: &[f64],
        tilt: &[f64],
    ) -> Result<(Vec<f64>, f64, f64), EstimationError> {
        let d = self.coefficient_dim();
        let mut theta = start.to_vec();
        let mut value = self.log_posterior(&theta, extra, anchor, tilt);
        let mut cached_training = start_training;
        loop {
            let (gradient, precision) = match cached_training.take() {
                Some(training) => self.finish_state(&theta, extra, anchor, tilt, training.clone()),
                None => self.gradient_and_precision(&theta, extra, anchor, tilt),
            };
            let factor = precision.cholesky(faer::Side::Lower).map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "multinomial predictive: augmented posterior precision is not positive \
                     definite ({error}); the fit's own posterior is not Laplace-describable at \
                     this prediction row"
                ))
            })?;
            let logdet = factor.diag().iter().map(|v| v.abs().ln()).sum::<f64>() * 2.0;
            let step = factor.solvevec(&(-&gradient));
            if step.iter().any(|v| !v.is_finite()) {
                crate::bail_invalid_estim!(
                    "multinomial predictive: augmented Newton step is not finite"
                );
            }
            // `½ gᵀH⁻¹g = −½ gᵀ·step`, the quadratic model's predicted gain.
            let decrement = -0.5
                * gradient
                    .iter()
                    .zip(step.iter())
                    .map(|(g, s)| g * s)
                    .sum::<f64>();
            // The stop is the Newton decrement, not a gradient norm: every ratio this
            // module publishes is `exp(L⁺ − L)`, so the residual error in `L` is what
            // has to vanish and the decrement bounds exactly that, while a gradient
            // norm is `O(n)` and not the currency of the answer. `L` accumulates the
            // weighted rows, the extra rows, `d²` quadratic products and `d` tilt
            // products; a predicted gain inside that accumulation's rounding band
            // `γ·|L|` is one the objective cannot represent.
            let terms = self.training_class_index.len() + extra.len() + d * d + d;
            let growth = gam_linalg::roundoff::accumulation_growth(terms);
            let objective_band = growth * value.abs();
            if decrement <= objective_band {
                return Ok((theta, value, logdet));
            }
            // Halve along the Newton direction until a trial gains more than the two
            // evaluations' rounding bands, or until the trial no longer moves `θ` in
            // floating point, which exhausts every representable step length.
            let mut length = 1.0_f64;
            let accepted = loop {
                let mut trial = vec![0.0_f64; d];
                for i in 0..d {
                    trial[i] = theta[i] + length * step[i];
                }
                if trial
                    .iter()
                    .zip(theta.iter())
                    .all(|(new, old)| new.to_bits() == old.to_bits())
                {
                    break false;
                }
                let trial_value = self.log_posterior(&trial, extra, anchor, tilt);
                // A step is accepted only when the objective rises by more than it
                // can represent. A trial that leaves the value bit-identical, or
                // moves it inside the rounding bands, is not progress, and accepting
                // it (`>=`) let a solve whose remaining gain sat under the
                // objective's own round-off — a strong penalty makes `L` large and
                // its resolution larger than the decrement target — burn every
                // iteration on steps that changed nothing and then report
                // non-convergence.
                if trial_value.is_finite()
                    && trial_value - value > objective_band + growth * trial_value.abs()
                {
                    theta = trial;
                    value = trial_value;
                    break true;
                }
                length *= 0.5;
            };
            if !accepted {
                // A convex objective whose Newton direction admits no gain beyond
                // the rounding bands at any representable step length is at its
                // optimum to floating-point resolution; the decrement test above has
                // not fired only because the remaining gain is below what the
                // objective can represent, which is the same statement.
                return Ok((theta, value, logdet));
            }
        }
    }

    /// Posterior-predictive moments at each row of `x_new`.
    ///
    /// `mode` is the un-augmented posterior mode in the same stacked class-major
    /// order; it is the warm start for every augmented solve and the base of
    /// every ratio.
    pub fn predictive_moments(
        &self,
        mode: ArrayView1<'_, f64>,
        x_new: ArrayView2<'_, f64>,
        want_second_moments: bool,
    ) -> Result<MultinomialPredictiveMoments, EstimationError> {
        self.validate()?;
        let d = self.coefficient_dim();
        if mode.len() != d {
            crate::bail_invalid_estim!(
                "multinomial predictive mode has {} entries, expected {d}",
                mode.len()
            );
        }
        if x_new.ncols() != self.training_design.ncols() {
            crate::bail_invalid_estim!(
                "multinomial predictive design has {} columns, training design has {}",
                x_new.ncols(),
                self.training_design.ncols(),
            );
        }
        let base_theta: Vec<f64> = mode.iter().copied().collect();
        // Everything the objective carries beyond the likelihood, as the quadratic
        // `Q` the ratios integrate (see `terminal_precision`): the fit's own terminal
        // precision less the likelihood curvature at the published mode, used only
        // where its spectrum certifies it positive definite. Otherwise the point
        // moments integrate `S_λ` with the measured tilt, and a spread declines.
        let fitted_quadratic = match self.terminal_precision {
            Some(terminal) => Some(self.fitted_quadratic(&base_theta, terminal)?),
            None => None,
        };
        let certified_quadratic = match fitted_quadratic {
            Some(fitted) if fitted.negative == 0 && fitted.unresolved == 0 => {
                Some(fitted.quadratic)
            }
            Some(fitted) => {
                if want_second_moments {
                    return Err(EstimationError::PredictiveIntervalsDeclined {
                        positive: fitted.positive,
                        negative: fitted.negative,
                        unresolved: fitted.unresolved,
                        lowest_eigenvalue: fitted.lowest_eigenvalue,
                        rounding_band: fitted.rounding_band,
                    });
                }
                None
            }
            None => None,
        };
        let model = MultinomialPredictiveModel {
            training_design: self.training_design,
            training_class_index: self.training_class_index,
            training_weights: self.training_weights,
            joint_penalty: match certified_quadratic.as_ref() {
                Some(quadratic) => quadratic.view(),
                None => self.joint_penalty,
            },
            n_classes: self.n_classes,
            terminal_precision: None,
        };
        // The base mode and its log-determinant are recomputed here rather than
        // read from the saved covariance ON PURPOSE: numerator and denominator
        // of every ratio must come from the same assembly, or the difference of
        // two log-determinants inherits whatever the two paths disagree about.
        let tilt = model.stationarity_tilt(&base_theta);
        let tilt = tilt.as_slice().expect("owned gradient is contiguous");
        let (base_mode, base_value, base_logdet) =
            model.augmented_mode(&base_theta, None, &[], &base_theta, tilt)?;
        // ... and with the stationarity tilt in place the polish must be a
        // NO-OP: the tilt was measured so that the supplied coefficients ARE the
        // stationary point of this objective, so the tilted gradient there is only
        // the rounding residual of `score − tilt`, and its Newton decrement is
        // that residual squared — far inside the objective's rounding band, where
        // `augmented_mode` returns its start unchanged. A polish that moves the
        // coefficients at all is therefore a statement about this module, not about
        // the fit — the tilt and the gradient it was built from have come apart —
        // and it is checked rather than assumed, because every ratio below is
        // expanded at this point and a base that is not a mode makes each of them
        // something other than a Laplace approximation.
        let drift = base_mode
            .iter()
            .zip(base_theta.iter())
            .fold(0.0_f64, |acc, (polished, supplied)| {
                acc.max((polished - supplied).abs())
            });
        if drift > 0.0 {
            let scale = base_theta
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            crate::bail_invalid_estim!(
                "multinomial predictive: the stationarity-tilted base is not stationary — \
                 polishing the supplied coefficients moved them by {drift:e} (largest \
                 coefficient magnitude {scale:e}), so the tilt and the gradient it was measured \
                 from disagree and every ratio below would be expanded somewhere other than a \
                 mode"
            );
        }

        // Every augmented solve is warm-started at `base_mode`, so its first
        // iteration's training curvature is this one.
        let base_training = model.training_state(&base_mode);
        let base = PredictiveBase {
            mode: &base_mode,
            training: &base_training,
            anchor: &base_theta,
            tilt,
            value: base_value,
            logdet: base_logdet,
        };

        let rows = x_new.nrows();
        let k = self.n_classes;
        // Every prediction row is its own set of augmented solves against the same
        // base, sharing nothing but read-only views, so the rows run in parallel.
        // Each row is still computed serially and in the same order, so the result
        // is bit-identical to a serial sweep; errors are reported for the first
        // failing row in row order, as the serial sweep would.
        let per_row: Vec<Result<PredictiveRow, EstimationError>> = (0..rows)
            .into_par_iter()
            .map(|row| model.predictive_row(row, x_new.row(row), &base, want_second_moments))
            .collect();
        let mut class_mean = Array2::<f64>::zeros((rows, k));
        let mut mass_defect = Array1::<f64>::zeros(rows);
        let mut second = if want_second_moments {
            Some(Array3::<f64>::zeros((rows, k, k)))
        } else {
            None
        };
        for (row, outcome) in per_row.into_iter().enumerate() {
            let outcome = outcome?;
            mass_defect[row] = outcome.mass_defect;
            class_mean.row_mut(row).assign(&outcome.class_mean);
            if let (Some(second), Some(row_second)) = (second.as_mut(), outcome.second_moment) {
                second
                    .index_axis_mut(ndarray::Axis(0), row)
                    .assign(&row_second);
            }
        }

        Ok(MultinomialPredictiveMoments {
            class_mean,
            class_second_moment: second,
            mass_defect,
        })
    }

    /// One prediction row's renormalised class means, mass defect and (when
    /// requested) renormalised second moments: `K` augmented solves for the
    /// means and `K(K+1)/2` more for the second moments.
    fn predictive_row(
        &self,
        row: usize,
        design_row: ArrayView1<'_, f64>,
        base: &PredictiveBase<'_>,
        want_second_moments: bool,
    ) -> Result<PredictiveRow, EstimationError> {
        let k = self.n_classes;
        let mut raw = vec![0.0_f64; k];
        for class in 0..k {
            let extra = [ExtraRow {
                design: design_row,
                class,
            }];
            let (_, value, logdet) =
                self.augmented_mode(base.mode, Some(base.training), &extra, base.anchor, base.tilt)?;
            raw[class] = (value - base.value + 0.5 * (base.logdet - logdet)).exp();
        }
        let total: f64 = raw.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            crate::bail_invalid_estim!(
                "multinomial predictive: row {row} produced a non-positive total predictive \
                 mass {total}"
            );
        }
        let mass_defect = (total - 1.0).abs();
        let class_mean = Array1::from_iter(raw.iter().map(|value| value / total));

        let mut second_moment = None;
        if want_second_moments {
            let mut second = Array2::<f64>::zeros((k, k));
            let mut raw_second = vec![0.0_f64; k * k];
            for c in 0..k {
                for dd in c..k {
                    let extra = [
                        ExtraRow {
                            design: design_row,
                            class: c,
                        },
                        ExtraRow {
                            design: design_row,
                            class: dd,
                        },
                    ];
                    let (_, value, logdet) =
                        self.augmented_mode(base.mode, Some(base.training), &extra, base.anchor, base.tilt)?;
                    let entry = (value - base.value + 0.5 * (base.logdet - logdet)).exp();
                    raw_second[c * k + dd] = entry;
                    raw_second[dd * k + c] = entry;
                }
            }
            let second_total: f64 = raw_second.iter().sum();
            if !second_total.is_finite() || second_total <= 0.0 {
                crate::bail_invalid_estim!(
                    "multinomial predictive: row {row} produced a non-positive second-moment \
                     mass {second_total}"
                );
            }
            // `Σ_{c,d} E[p_c p_d] = E[(Σ_c p_c)²] = 1` is the same exact
            // identity one order up, so the same normalisation applies.
            for c in 0..k {
                for dd in 0..k {
                    second[[c, dd]] = raw_second[c * k + dd] / second_total;
                }
            }
            second_moment = Some(second);
        }
        Ok(PredictiveRow {
            class_mean,
            mass_defect,
            second_moment,
        })
    }
}

/// The un-augmented side every ratio shares: the base mode, its cached
/// [`MultinomialPredictiveModel::training_state`], the expansion anchor and
/// tilt, and the base objective value and log-determinant.
struct PredictiveBase<'b> {
    mode: &'b [f64],
    training: &'b (Array1<f64>, Array2<f64>),
    anchor: &'b [f64],
    tilt: &'b [f64],
    value: f64,
    logdet: f64,
}

/// One prediction row's share of [`MultinomialPredictiveMoments`].
struct PredictiveRow {
    class_mean: Array1<f64>,
    mass_defect: f64,
    second_moment: Option<Array2<f64>>,
}

/// Per-class posterior standard deviation of the probability, from the moments
/// above: `sd(p_c) = sqrt(E[p_c²] − E[p_c]²)`, with the typed reason at each row
/// that has none.
///
/// A materially negative variance is declined rather than clamped: `E[p_c²]` and
/// `E[p_c]` come from two different ratios, so a negative difference means the
/// two expansions disagree by more than the quantity being reported, which is
/// exactly the situation in which a clamped `sd = 0` would be a lie. The decision
/// is the row's own: that row returns its decline, and every other row returns
/// its standard deviations.
pub(crate) fn predictive_standard_deviation(
    moments: &MultinomialPredictiveMoments,
) -> Result<Vec<Result<Array1<f64>, crate::multinomial::MultinomialSpreadDecline>>, EstimationError>
{
    let second = moments.class_second_moment.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial predictive standard deviation requires second moments".to_string(),
        )
    })?;
    let (rows, k) = moments.class_mean.dim();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut deviation = Array1::<f64>::zeros(k);
        let mut declined = None;
        for class in 0..k {
            let mean = moments.class_mean[[row, class]];
            let variance = second[[row, class, class]] - mean * mean;
            // `E[p²]` and `E[p]²` are two different ratios, so their difference
            // is only resolvable down to the accuracy of the ratios themselves —
            // and that accuracy is MEASURED at this row by the mass defect, not
            // guessed. A probability's variance is bounded by `mean(1 − mean)`,
            // so the envelope is that scale times the row's own measured error,
            // floored at round-off. Anything more negative than that is not
            // cancellation: it is the two expansions disagreeing by more than the
            // quantity being reported, which is exactly the case where a clamped
            // `sd = 0` would be a lie.
            let bound = (mean * (1.0 - mean)).max(f64::EPSILON);
            let envelope =
                bound * moments.mass_defect[row].max(16.0 * f64::EPSILON);
            if variance < -envelope {
                declined = Some(crate::multinomial::MultinomialSpreadDecline {
                    class,
                    variance,
                    envelope,
                });
                break;
            }
            deviation[class] = variance.max(0.0).sqrt();
        }
        out.push(match declined {
            Some(decline) => Err(decline),
            None => Ok(deviation),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #1082: a row whose second moments disagree with its mean beyond its own
    /// envelope declines alone, and the rows around it publish their standard
    /// deviations.
    #[test]
    fn a_negative_variance_row_declines_only_its_own_spread_1082() {
        let class_mean = ndarray::array![[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]];
        let mut second = Array3::<f64>::zeros((3, 2, 2));
        second[[0, 0, 0]] = 0.3 * 0.3 + 0.01;
        second[[0, 1, 1]] = 0.7 * 0.7 + 0.01;
        second[[1, 0, 0]] = 0.5 * 0.5 - 0.02;
        second[[1, 1, 1]] = 0.5 * 0.5 + 0.01;
        second[[2, 0, 0]] = 0.9 * 0.9 + 4.0e-4;
        second[[2, 1, 1]] = 0.1 * 0.1 + 4.0e-4;
        let moments = MultinomialPredictiveMoments {
            class_mean,
            class_second_moment: Some(second),
            mass_defect: ndarray::array![1.0e-4, 1.0e-3, 1.0e-4],
        };
        let rows = predictive_standard_deviation(&moments).unwrap();
        assert_eq!(rows.len(), 3);
        let first = rows[0].as_ref().unwrap();
        assert!((first[0] - 0.1).abs() < 1e-12 && (first[1] - 0.1).abs() < 1e-12, "{first}");
        let decline = rows[1].as_ref().unwrap_err();
        assert_eq!(decline.class, 0);
        assert!((decline.variance + 0.02).abs() < 1e-12, "{decline}");
        assert!((decline.envelope - 0.25 * 1.0e-3).abs() < 1e-18, "{decline}");
        let third = rows[2].as_ref().unwrap();
        assert!((third[0] - 0.02).abs() < 1e-12 && (third[1] - 0.02).abs() < 1e-12, "{third}");
    }
}
