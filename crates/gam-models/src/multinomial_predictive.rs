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
//! reference. [`MultinomialPredictiveModel`] refuses rather than publishing a
//! row whose mass defect exceeds [`PREDICTIVE_MASS_DEFECT_TOLERANCE`].
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
//! One warm-started Newton solve per (row, class) — the augmented objective is
//! strictly convex, so Newton with backtracking is unconditionally safe — plus
//! `K(K+1)/2` more per row when second moments are requested.  Each Newton
//! iteration is `O(n·M²·P²)` for the curvature (as `M(M+1)/2` GEMMs) and
//! `O(d³)` for the factorisation, so the whole predictive is
//! `O(R·K·iters·(n M² P² + d³))`.
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
use gam_linalg::faer_ndarray::FaerCholesky;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

/// Largest tolerated deviation of `Σ_c E[p_c(x)]` from one before a prediction
/// row is refused.
///
/// This is not a fudge factor on the answer: the sum is an EXACT identity of the
/// estimand, so its deviation is the approximation's own error, measured at the
/// row being published.
///
/// The value is set from what the defect is worth as a predictor of the error
/// that survives renormalisation, which is measured rather than assumed. Two
/// independent fixtures:
///
/// ```text
///   K = 3, p = 10, asymmetric quasi-separated, MCMC truth
///       worst-row defect 1.37e-2   worst-row error after normalising 4.4e-3   (3.1x)
///   K = 2, p = 2,  quasi-separated, exact 2-D quadrature truth
///       worst-row defect 4.93e-3   worst-row error after normalising 5.2e-4   (9.5x)
/// ```
///
/// So the defect OVER-states the published error by roughly 3-10x, and a bar at
/// `5e-2` refuses where the published probability would be wrong by more than
/// about `1e-2` — the second decimal of a probability, which is the right place
/// to stop publishing one. A tighter bar would refuse rows whose answers are
/// good to four decimals; a looser one would publish a probability wrong in its
/// first. A refusal here is a real statement — the posterior at that row is not
/// described by either Laplace expansion well enough — and it is louder and more
/// useful than a number nobody can bound.
pub const PREDICTIVE_MASS_DEFECT_TOLERANCE: f64 = 5.0e-2;

/// Convergence target for the augmented-mode Newton solve, stated as the NEWTON
/// DECREMENT `½ gᵀH⁻¹g` — the quadratic model's own bound on how much
/// log-posterior is left to gain.
///
/// A gradient-norm target would be the wrong currency here twice over. It is
/// not scale-free (the gradient of an `n`-row log-likelihood is `O(n)`, so the
/// same threshold means different things on different fixtures), and it is not
/// the quantity the answer depends on: every ratio this module publishes is
/// `exp(L⁺ − L)`, so what has to be small is the residual error in `L`, which
/// is exactly what the decrement bounds. At `1e-10` the ratio is converged to
/// `1e-10` relative — five orders below the `1e-2` mass defect the estimator's
/// own identity is checked at, so the solve is never the binding error.
const AUGMENTED_MODE_DECREMENT_TOLERANCE: f64 = 1.0e-10;

/// How far the base-mode polish may move the supplied coefficients, relative to
/// their own largest magnitude, before the predictive refuses.
///
/// The polish exists so the base of every ratio is exactly a stationary point of
/// the objective the ratios are taken against; on a fit whose mode was found
/// under that same objective it moves the coefficients by the solver's own
/// residual, which is orders below this. A LARGE move means the supplied mode
/// belongs to a different objective, and the bar is set where "solver slack" and
/// "different model" cannot be confused: an inner solve certified at a scaled
/// KKT residual of `1e-5` leaves a mode displacement far under `1e-3` of the
/// coefficient scale, while a first-order objective difference in a
/// near-unpenalized direction moves it by `O(1)`.
const BASE_MODE_POLISH_TOLERANCE: f64 = 1.0e-3;

/// Maximum Newton iterations for one augmented mode. The objective is strictly
/// convex and the start point is the un-augmented mode, one observation away, so
/// this bound is never approached on a well-posed fit; it exists so a
/// pathological row fails loudly instead of spinning.
const AUGMENTED_MODE_MAX_ITERATIONS: usize = 100;

/// Backtracking line-search contraction factor and its iteration bound.
const LINE_SEARCH_CONTRACTION: f64 = 0.5;
const LINE_SEARCH_MAX_STEPS: usize = 60;

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
    /// own measured error at that row.
    pub mass_defect: Array1<f64>,
}

/// One extra observation appended to the training data: a design row and the
/// class it is assigned.
#[derive(Debug, Clone, Copy)]
struct ExtraRow<'a> {
    design: ArrayView1<'a, f64>,
    class: usize,
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

    /// The objective this predictive integrates:
    /// `ℓ(θ) − ½ θ' S_λ θ + cᵀθ`, optionally with extra observations appended.
    ///
    /// See [`Self::stationarity_tilt`] for what `c` is and why it is measured
    /// rather than assumed.
    fn log_posterior(&self, theta: &[f64], extra: &[ExtraRow<'_>], tilt: &[f64]) -> f64 {
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
        let mut quadratic = 0.0_f64;
        for (i, &ti) in theta.iter().enumerate() {
            let mut acc = 0.0_f64;
            for (j, &tj) in theta.iter().enumerate() {
                acc += self.joint_penalty[[i, j]] * tj;
            }
            quadratic += ti * acc;
        }
        let linear: f64 = theta.iter().zip(tilt.iter()).map(|(t, c)| t * c).sum();
        total - 0.5 * quadratic + linear
    }

    /// Gradient of the NEGATIVE penalized log-posterior and its Hessian, both
    /// in the stacked class-major order.
    ///
    /// The `(a, b)` curvature block is `Xᵀ diag(w_ab) X` with
    /// `w_ab[row] = weight · p_a (δ_ab − p_b)`, which is a GEMM. Accumulating it
    /// row-by-row instead would be the same flops with none of the locality,
    /// and this is the inner loop of every augmented mode: one per (prediction
    /// row, class).
    fn gradient_and_precision(
        &self,
        theta: &[f64],
        extra: &[ExtraRow<'_>],
        tilt: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let n = self.training_design.nrows();
        let p = self.training_design.ncols();
        let m = self.active_classes();
        let d = p * m;
        let mut gradient = Array1::<f64>::zeros(d);
        let mut precision = self.joint_penalty.to_owned();
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
                        precision[[a * p + i, b * p + j]] += block[[i, j]];
                        if a != b {
                            // `w_ab = w·p_a(δ_ab − p_b)` is symmetric in `(a, b)`,
                            // so the mirrored block is the transpose and does not
                            // need its own GEMM.
                            precision[[b * p + j, a * p + i]] += block[[i, j]];
                        }
                    }
                }
            }
        }

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

        // The penalty's own contribution to the gradient of the NEGATIVE
        // log-posterior is `S_λ θ`; the tilt's is `−c`. Neither touches the
        // curvature — the penalty is already in `precision` and a linear term
        // has no second derivative.
        for i in 0..d {
            let mut acc = 0.0_f64;
            for j in 0..d {
                acc += self.joint_penalty[[i, j]] * theta[j];
            }
            gradient[i] += acc - tilt[i];
        }
        (gradient, precision)
    }

    /// The linear term `c` that makes the SUPPLIED coefficients an exact
    /// stationary point of the objective this predictive integrates.
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
    /// `c = ∇(−[ℓ − ½θ'S_λθ])(β̂)` is EXACTLY `∇Φ(β̂)`, by stationarity of the
    /// fit's own mode — so the first-order content of whatever extra term the
    /// objective carried is *measurable* at the published point rather than
    /// something this module has to reconstruct. Carrying it as a linear term is
    /// free: it moves no curvature (a linear function has no second derivative),
    /// so the augmented modes and their log-determinants stay the ones the ratio
    /// needs.
    ///
    /// When the fit carried no extra term, `c` is the inner solve's own residual
    /// gradient — orders below anything that matters, and absorbing it makes the
    /// base exactly stationary rather than nearly so, which is a strict
    /// improvement on using the raw mode.
    ///
    /// What is NOT carried is the extra term's CURVATURE. That is second-order
    /// in the ratio: `Φ`'s Hessian appears in the numerator's and the
    /// denominator's log-determinants alike, one observation apart, and the
    /// residual is what the mass-defect identity below measures at every row.
    fn stationarity_tilt(&self, mode: &[f64]) -> Array1<f64> {
        let zero = vec![0.0_f64; mode.len()];
        let (gradient, _) = self.gradient_and_precision(mode, &[], &zero);
        gradient
    }

    /// Newton with backtracking on the strictly convex negative penalized
    /// log-posterior, warm-started at `start`.
    ///
    /// Returns the mode, the objective value there, and the log-determinant of
    /// the precision at that point — the three quantities a Laplace ratio needs
    /// from one side of it.
    fn augmented_mode(
        &self,
        start: &[f64],
        extra: &[ExtraRow<'_>],
        tilt: &[f64],
    ) -> Result<(Vec<f64>, f64, f64), EstimationError> {
        let d = self.coefficient_dim();
        let mut theta = start.to_vec();
        let mut value = self.log_posterior(&theta, extra, tilt);
        let mut logdet;
        for _iteration in 0..AUGMENTED_MODE_MAX_ITERATIONS {
            let (gradient, precision) = self.gradient_and_precision(&theta, extra, tilt);
            let factor = precision.cholesky(faer::Side::Lower).map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "multinomial predictive: augmented posterior precision is not positive \
                     definite ({error}); the fit's own posterior is not Laplace-describable at \
                     this prediction row"
                ))
            })?;
            logdet = factor.diag().iter().map(|v| v.abs().ln()).sum::<f64>() * 2.0;
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
            if decrement <= AUGMENTED_MODE_DECREMENT_TOLERANCE {
                return Ok((theta, value, logdet));
            }
            let mut accepted = false;
            let mut length = 1.0_f64;
            for _attempt in 0..LINE_SEARCH_MAX_STEPS {
                let mut trial = vec![0.0_f64; d];
                for i in 0..d {
                    trial[i] = theta[i] + length * step[i];
                }
                let trial_value = self.log_posterior(&trial, extra, tilt);
                // A step is accepted only when the objective actually rises.
                // A trial that leaves the value bit-identical is not progress,
                // and accepting it (`>=`) let a solve whose remaining gain sat
                // under the objective's own round-off — a strong penalty makes
                // `L` large and its resolution `ε|L|` larger than the
                // decrement target — burn every iteration on steps that changed
                // nothing and then report non-convergence.
                if trial_value.is_finite() && trial_value > value {
                    theta = trial;
                    value = trial_value;
                    accepted = true;
                    break;
                }
                length *= LINE_SEARCH_CONTRACTION;
            }
            if !accepted {
                // A convex objective whose Newton direction admits no ascent at
                // any step length is at its optimum to floating-point
                // resolution; the decrement test above has not fired only
                // because the remaining gain is below what the objective can
                // represent, which is the same statement.
                return Ok((theta, value, logdet));
            }
        }
        Err(EstimationError::InvalidInput(format!(
            "multinomial predictive: augmented mode did not converge in \
             {AUGMENTED_MODE_MAX_ITERATIONS} Newton iterations"
        )))
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
        // The base mode and its log-determinant are recomputed here rather than
        // read from the saved covariance ON PURPOSE: numerator and denominator
        // of every ratio must come from the same assembly, or the difference of
        // two log-determinants inherits whatever the two paths disagree about.
        let tilt = self.stationarity_tilt(&base_theta);
        let tilt = tilt.as_slice().expect("owned gradient is contiguous");
        let (base_mode, base_value, base_logdet) = self.augmented_mode(&base_theta, &[], tilt)?;
        // ... and with the stationarity tilt in place the polish must be a
        // NO-OP: `c` was measured so that the supplied coefficients ARE the
        // stationary point of this objective. A polish that moves anywhere is
        // therefore a statement about this module, not about the fit — the tilt
        // and the gradient it was built from have come apart — and it is checked
        // rather than assumed, because every ratio below is expanded at this
        // point and a base that is not a mode makes each of them something other
        // than a Laplace approximation.
        let scale = base_theta
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let drift = base_mode
            .iter()
            .zip(base_theta.iter())
            .fold(0.0_f64, |acc, (polished, supplied)| {
                acc.max((polished - supplied).abs())
            });
        if drift > BASE_MODE_POLISH_TOLERANCE * scale {
            crate::bail_invalid_estim!(
                "multinomial predictive: the stationarity-tilted base is not stationary — \
                 polishing the supplied coefficients moved them by {drift:e} against a \
                 coefficient scale of {scale:e} (relative {relative:e} > {tol:e}), so the tilt \
                 and the gradient it was measured from disagree and every ratio below would be \
                 expanded somewhere other than a mode",
                relative = drift / scale,
                tol = BASE_MODE_POLISH_TOLERANCE,
            );
        }

        let rows = x_new.nrows();
        let k = self.n_classes;
        let mut class_mean = Array2::<f64>::zeros((rows, k));
        let mut mass_defect = Array1::<f64>::zeros(rows);
        let mut second = if want_second_moments {
            Some(Array3::<f64>::zeros((rows, k, k)))
        } else {
            None
        };
        // `(row, mass, defect)` of the worst row over tolerance, and how many
        // rows are over it. The refusal below is raised from these rather than
        // from the first row that trips, so it states the estimand's accuracy
        // over the whole block instead of naming one witness.
        let mut worst_defect: Option<(usize, f64, f64)> = None;
        let mut over_tolerance_rows = 0usize;

        for row in 0..rows {
            let design_row = x_new.row(row);
            let mut raw = vec![0.0_f64; k];
            for class in 0..k {
                let extra = [ExtraRow {
                    design: design_row,
                    class,
                }];
                let (_, value, logdet) = self.augmented_mode(&base_mode, &extra, tilt)?;
                raw[class] = (value - base_value + 0.5 * (base_logdet - logdet)).exp();
            }
            let total: f64 = raw.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                crate::bail_invalid_estim!(
                    "multinomial predictive: row {row} produced a non-positive total predictive \
                     mass {total}"
                );
            }
            mass_defect[row] = (total - 1.0).abs();
            if mass_defect[row] > PREDICTIVE_MASS_DEFECT_TOLERANCE {
                // Recorded, not raised — see the refusal after the loop. Stopping
                // here would report an EXAMPLE where the estimand's own accuracy
                // statement is a MEASUREMENT: "row 46 is bad" and "3 of 86 rows
                // are bad, the worst at 7.2e-2, the 90th percentile at 4e-3" are
                // different findings, and the first cannot be told from the
                // second by a caller who only ever sees the first bad row.
                if worst_defect.is_none_or(|(_, _, defect)| defect < mass_defect[row]) {
                    worst_defect = Some((row, total, mass_defect[row]));
                }
                over_tolerance_rows += 1;
            }
            for class in 0..k {
                class_mean[[row, class]] = raw[class] / total;
            }

            if let Some(second) = second.as_mut() {
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
                        let (_, value, logdet) = self.augmented_mode(&base_mode, &extra, tilt)?;
                        let entry = (value - base_value + 0.5 * (base_logdet - logdet)).exp();
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
                        second[[row, c, dd]] = raw_second[c * k + dd] / second_total;
                    }
                }
            }
        }

        if let Some((row, mass, defect)) = worst_defect {
            // The distribution, not just the extreme: a block where one row in
            // eighty-six is over and the rest are at `1e-4` is a statement about
            // that row, while a block where a third of the rows are over is a
            // statement about the fit. Those need different repairs and used to
            // print identically.
            let mut sorted: Vec<f64> = mass_defect.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let quantile = |q: f64| -> f64 {
                let index = ((sorted.len() - 1) as f64 * q).round() as usize;
                sorted[index]
            };
            crate::bail_invalid_estim!(
                "multinomial predictive: {over_tolerance_rows} of {rows} row(s) exceed the \
                 predictive mass-defect tolerance {tol:e}; the posterior at those rows is not \
                 described by either Laplace expansion well enough to publish a probability. \
                 Worst row {row} has predictive mass {mass} (|Σ_c E[p_c] − 1| = {defect:e}). \
                 Defect over the block: median {median:e}, 90th percentile {p90:e}, max {max:e}",
                tol = PREDICTIVE_MASS_DEFECT_TOLERANCE,
                median = quantile(0.5),
                p90 = quantile(0.9),
                max = quantile(1.0),
            );
        }

        Ok(MultinomialPredictiveMoments {
            class_mean,
            class_second_moment: second,
            mass_defect,
        })
    }
}

/// Per-class posterior standard deviation of the probability, from the moments
/// above: `sd(p_c) = sqrt(E[p_c²] − E[p_c]²)`.
///
/// A materially negative variance is refused rather than clamped: `E[p_c²]` and
/// `E[p_c]` come from two different ratios, so a negative difference means the
/// two expansions disagree by more than the quantity being reported, which is
/// exactly the situation in which a clamped `sd = 0` would be a lie.
pub fn predictive_standard_deviation(
    moments: &MultinomialPredictiveMoments,
) -> Result<Array2<f64>, EstimationError> {
    let second = moments.class_second_moment.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial predictive standard deviation requires second moments".to_string(),
        )
    })?;
    let (rows, k) = moments.class_mean.dim();
    let mut sd = Array2::<f64>::zeros((rows, k));
    for row in 0..rows {
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
                crate::bail_invalid_estim!(
                    "multinomial predictive: row {row} class {class} has negative probability \
                     variance {variance:e} (mean {mean}, backward-error envelope {envelope:e})"
                );
            }
            sd[[row, class]] = variance.max(0.0).sqrt();
        }
    }
    Ok(sd)
}
