use super::*;

/// Result of the unified REML/LAML evaluation.
#[derive(Debug)]
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Additive scalar decomposition of `cost`, retained through outer
    /// correction atoms so structured finite-difference audits can compare
    /// each analytic gradient atom with the derivative of the scalar it owns.
    pub criterion_components: RemlCriterionComponents,
    /// Newton-decrement energy `½ rᵀH⁻¹r` of the converged inner KKT
    /// residual at this `ρ`, where `r = ∇_β L(β̂, ρ)` and `H` is the inner
    /// Hessian. Bounds the inner sub-optimality `|V(β̂) − V(β*)| ≤
    /// ½ rᵀH⁻¹r` to first order, and is consumed by the trust-energy gate in
    /// the outer strategy, which shrinks the trust radius when this energy
    /// exceeds `TRUST_ENERGY_FACTOR × |predicted_decrease|`.
    ///
    /// `None` when the inner solve did not compute an energy estimate
    /// (e.g., projected-pseudo-inverse paths that lack a full-H solve).
    pub ift_residual_energy: Option<f64>,
    /// One-Newton-step inner polish vector `w = H⁻¹ r`, populated only
    /// when the evaluator solves against the full inner Hessian `H` (not
    /// the projected pseudo-inverse used on rank-deficient paths).
    ///
    /// Applied by the runtime as a *free* refinement of the warm-start β
    /// at the next outer iteration: `β_warm ← β̂ + w` short-circuits one
    /// PIRLS step, exploiting the Hessian factorization already paid for
    /// during the cost-side IFT correction. `None` whenever the polish
    /// step was not produced (projected-pseudo-inverse path, value-only
    /// evaluation, etc.).
    pub inner_polish_step: Option<Array1<f64>>,
    /// Gradient ∂V/∂ρ (present if mode ≥ ValueAndGradient).
    pub gradient: Option<Array1<f64>>,
    /// Outer Hessian ∂²V/∂ρ² (present if mode = ValueGradientHessian).
    pub hessian: gam_problem::HessianValue,
    /// Rho-coordinate mode responses, one `K · g_j` vector per column, when
    /// they were already built for derivative corrections. Consumed by the
    /// runtime IFT mode-response cache for joint-IFT warm starts.
    pub rho_mode_response_cols: Option<Array2<f64>>,
    /// Extended-coordinate mode responses, one `K · g_j` vector per column,
    /// when extended derivative coordinates required them.
    pub ext_mode_response_cols: Option<Array2<f64>>,
    /// The inner mode's fold record along its softest direction ([`InnerModeFold`], gam#2765,
    /// gam#3173), on the path that VALIDATED.
    ///
    /// A refused evaluation carries the same record through [`RemlLamlError::InnerModeFold`],
    /// which is the only place it used to go. That left the record unreadable exactly where it is
    /// load-bearing: a mode whose Laplace series' leading correction is not below the term it
    /// corrects is admitted, and it is admitted BECAUSE its curvature is still resolved — but it
    /// sits close enough to the saddle bounding its basin that another basin is within one solve.
    /// `None` when the mode response names no span to grade.
    pub inner_mode_fold: Option<InnerModeFold>,
    /// WHY this evaluation carries no outer Hessian.
    ///
    /// `Some` exactly when `hessian` is `HessianValue::Unavailable`, and set
    /// beside it so the two cannot drift. `HessianValue` is `opt`'s enum and
    /// has no room for a reason, but the reasons are not interchangeable: one
    /// of them is a DECLARATION by the criterion that the matrix does not
    /// exist for this model, and a consumer that reads it as a failure refuses
    /// a fit that is fine (gam#3234, gam#1561).
    pub hessian_absence: Option<OuterHessianAbsence>,
}

/// Why a [`RemlLamlResult`] carries no outer Hessian.
///
/// The three are different findings and only the criterion can tell them
/// apart, so it says which one it is rather than leaving every consumer to
/// read `HessianValue::Unavailable` as the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterHessianAbsence {
    /// The evaluation mode did not ask for one. Nothing is wrong and nothing
    /// was attempted.
    NotRequested,
    /// THE CRITERION DECLARES IT HAS NONE, and this is a property of the model.
    ///
    /// The second derivative of a term priced on a PROFILED posterior carries
    /// the scale's own second-order channel, which is assembled a function
    /// away, so publishing the fixed-scale Hessian in its place would hand a
    /// caller a matrix that is not the second derivative of the value the fit
    /// certifies against. The criterion declares no Hessian instead and the
    /// outer plan reads that declaration before the search starts (gam#3234).
    ///
    /// A consumer whose own output is DEFINED without the Hessian -- the
    /// smoothing correction, whose absence leaves the uncorrected covariance --
    /// records that it was declined and carries on. A consumer that needs the
    /// matrix refuses, naming this.
    ProfiledCriterionDeclares,
    /// The envelope-gradient tripwire suppressed this evaluation's outputs, so
    /// no Hessian was assembled for a point whose gradient is already invalid
    /// as a descent direction. A numerical event at this trial, not a property
    /// of the model.
    EnvelopeSuppressed,
}

impl OuterHessianAbsence {
    /// Whether the absence is the criterion's own declaration rather than a
    /// failure or a suppression at this trial.
    pub fn is_declared_by_criterion(self) -> bool {
        matches!(self, Self::ProfiledCriterionDeclares)
    }
}


impl RemlLamlResult {
    /// The outer gradient an evaluation in `mode` hands its caller.
    ///
    /// `gradient` is `None` in exactly two situations. A value-only evaluation computes
    /// no gradient, and its caller never reads the gradient slot, so it gets zeros of
    /// length `dim`. A derivative-bearing evaluation has no gradient only when the
    /// envelope-gradient tripwire suppressed an invalid descent direction. There is no
    /// gradient at that θ, and zeros would say the point is stationary. On the #979
    /// 160×6 survival repro they did: BFGS reported "Converged by gradient ||g||=0" and
    /// the screening certificate read |g| = 0 as stationary (job 1131465). That
    /// evaluation is refused instead, so the outer search retreats from the trial or
    /// rejects the seed, which is what the tripwire exists to cause.
    pub fn gradient_for_mode(&mut self, mode: EvalMode, dim: usize) -> Result<Array1<f64>, String> {
        match (self.gradient.take(), mode) {
            (Some(gradient), _) => Ok(gradient),
            (None, EvalMode::ValueOnly) => Ok(Array1::zeros(dim)),
            (None, mode) => Err(format!(
                "the {mode:?} evaluation has no outer gradient: the envelope-gradient tripwire \
                 suppressed it at this point, so the trial is refused instead of being handed to \
                 the outer optimizer as a zero gradient"
            )),
        }
    }
}

/// The Laplace record of an inner mode along its softest direction (gam#2765, gam#979).
///
/// The criterion's normalizer is the Gaussian integral of the quadratic model of the inner
/// objective `f` about its mode. Along the softest eigenpair `(σ, v)` of the operator the mode
/// response inverts, `f(β̂ + s·v) = f̂ + ½σs² + (t₃/6)s³ + (t₄/24)s⁴ + …`, and the one-dimensional
/// Laplace series is `log ∫ e^{−f} ds = −f̂ + ½log(2π/σ) + c + O(c²)` with leading correction
/// `c = 5t₃²/(24σ³) − t₄/(8σ²)`.
///
/// The one refusal is the rounding band: at or below the span spectrum's band `σ` is not resolved
/// from zero, so the normalizer has no curvature to integrate and the trial point is refused
/// before anything is priced. The cubic share `5t₃²/(24σ³)` is a RECORD, not a refusal. It
/// measures how non-Gaussian the posterior is along `v` at one point, which does not tell a mode
/// folding along the search path (`σ → 0`) from a well-conditioned mode with a large third
/// derivative. Refusing on it was measured wrong (gate job 1219877): a mode at `σ = 2`, `t₃ = 8.5`
/// (share 1.875) was refused; on the #2894 repro 122 of 387 graded evaluations refused, 80 of them
/// at `σ ≥ 0.1`; and six custom-family pins went red. A fold test compares the fold distance
/// `σ/|t₃|` with the inner mode's motion along `v`, and is its own derivation.
///
/// `t₃ = vᵀ D_β M[v] v` prices the complete operator the mode response inverts: the drift of the
/// log-determinant operator (`HessianDerivativeProvider::hessian_derivative_correction`) plus the
/// motion of the stationarity operator's difference from it
/// (`HessianDerivativeProvider::mode_response_rhs_correction`), which that hook's contract omits
/// exactly when the difference is constant. Where the difference moves without its derivatives the
/// record says so ([`CompletionShare::NotSupplied`]). The quartic share is not priced. Grading the
/// softest eigenpair alone describes one direction: a stiffer direction with a much larger `t₃`,
/// and mixed third derivatives coupling `v` to stiff directions, also enter the multivariate
/// correction. Every quantity is un-scaled by the operator's uniform curvature scale, and the share
/// is invariant to rescaling `v`, so the record describes the objective.
#[derive(Clone, Debug)]
pub struct InnerModeFold {
    /// The softest eigenvalue `σ` on the span the mode response inverts, un-scaled.
    pub sigma: f64,
    /// The span spectrum's rounding band, un-scaled.
    pub rounding_band: f64,
    /// The unit softest eigenvector `v` of that span, in the coefficient frame the operator acts
    /// on. It is the direction every other field is measured ALONG, and the direction a start past
    /// the saddle is displaced in ([`InnerModeFold::saddle_crossing_displacement`]).
    pub softest_direction: Array1<f64>,
    /// `t₃ = vᵀ D_β M[v] v` along the softest eigenvector, un-scaled; `None` when the rounding band
    /// refused before it was priced, or when the grading priced no record (the unified evaluator's
    /// verdict reads `σ` alone, #979).
    pub third_derivative: Option<f64>,
    /// The cubic share `5t₃²/(24σ³)` of the Laplace series' leading correction, priced with `t₃`.
    pub cubic_correction: Option<f64>,
    /// The quartic share `|t₄|/(8σ²)` of the leading correction.
    pub quartic_correction: QuarticShare,
    /// Whether `t₃` carries the stationarity difference's motion, priced with `t₃`.
    pub completion: Option<CompletionShare>,
}

/// Whether a fold record's `t₃` carries the motion of the stationarity operator's difference from
/// the log-determinant operator (gam#2765).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionShare {
    /// `t₃` prices the complete operator: the difference is constant, or its motion was priced.
    Priced,
    /// The difference moves but the provider supplies no derivative of it, so `t₃` is the
    /// log-determinant operator's share alone.
    NotSupplied,
}

/// Whether the quartic share of a fold verdict's leading correction was priced (gam#2765).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuarticShare {
    /// Not priced by design: the verdict refuses on the cubic share and the rounding band, which is
    /// sufficient for invalidity, and pricing `t₄` would add a second directional pass to every
    /// healthy evaluation with no provider declaring that hook.
    NotPriced,
}

impl InnerModeFold {
    /// Whether the Laplace series' leading correction along the softest direction is below the
    /// term it corrects.
    pub fn is_valid(&self) -> bool {
        self.sigma > self.rounding_band
    }

    /// Whether the Laplace series' leading correction along `v` is NOT below the term it corrects:
    /// the cubic share `5t₃²/(24σ³)` at or above one (gam#3173).
    ///
    /// The share equals `5/(36·ΔF)` for the barrier `ΔF = (2/3)σ³/t₃²` the cubic model puts
    /// between this minimum and the saddle beyond it, so the condition reads "the barrier is below
    /// `5/36` in log-likelihood units". It has no constant of its own: a correction at or above
    /// the term it corrects is not a correction, and that is the whole of it.
    ///
    /// This is a DISCOVERY condition and never a refusal. Refusing on the same share was measured
    /// wrong (gate job 1219877: a mode at `σ = 2`, `t₃ = 8.5` refused; 122 of 387 graded
    /// evaluations refused on the #2894 repro, 80 of them at `σ ≥ 0.1`; six custom-family pins
    /// red), and that measurement stands — a well-conditioned mode with a large third derivative
    /// is not at a fold. What the share does say is where looking for another basin is worth one
    /// solve. `false` where `t₃` was not priced.
    pub fn barrier_is_below_its_own_correction(&self) -> bool {
        self.cubic_correction.is_some_and(|share| share >= 1.0)
    }

    /// The displacement from this mode that lands past the saddle bounding its basin along `v`,
    /// `2s*·v` with `s* = −2σ/t₃` (gam#3173).
    ///
    /// The cubic model along `v` is `f(β̂ + s·v) ≈ f̂ + ½σs² + t₃s³/6`, whose other stationary
    /// point is the saddle at `s* = −2σ/t₃`. Twice that overshoots it, so an inner solve started
    /// there descends into the neighbouring basin where one exists and returns to this mode where
    /// it does not: at a saddle-node fold the vanishing minimum and the saddle coincide, and the
    /// mountain-pass inequality then puts the rival basin below this one. `σ` and `t₃` are both
    /// un-scaled by the operator's curvature scale, so their ratio is scale-free and the
    /// displacement is in coefficient units.
    ///
    /// `None` when `t₃` was not priced, or is zero or not finite: a vanishing cubic term puts the
    /// saddle at infinity, which names no crossing.
    pub fn saddle_crossing_displacement(&self) -> Option<Array1<f64>> {
        let third = self.third_derivative?;
        if !third.is_finite() || third == 0.0 || !self.sigma.is_finite() {
            return None;
        }
        let step = -4.0 * self.sigma / third;
        step.is_finite().then(|| &self.softest_direction * step)
    }
}

impl std::fmt::Display for InnerModeFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.third_derivative, self.cubic_correction, self.completion) {
            (Some(third), Some(correction), Some(completion)) => write!(
                f,
                "the inner mode's softest curvature is {:.3e} (rounding band {:.3e}) with third \
                 directional derivative {third:.3e} (completion {completion:?}), recording a \
                 cubic share of {correction:.3e} of the Laplace series' leading correction; the \
                 quartic share is not priced (gam#2765, gam#979)",
                self.sigma, self.rounding_band,
            ),
            _ if self.is_valid() => write!(
                f,
                "the inner mode's softest curvature is {:.3e}, resolved above its rounding band \
                 {:.3e}; its Laplace series' third-order share was not priced (gam#2765, gam#979)",
                self.sigma, self.rounding_band,
            ),
            _ => write!(
                f,
                "the inner mode's softest curvature {:.3e} is at or below its rounding band \
                 {:.3e}, so the Laplace normalizer has no resolved curvature to integrate \
                 (gam#2765, gam#979)",
                self.sigma, self.rounding_band,
            ),
        }
    }
}

/// Why the unified evaluator published no evaluation (gam#2765).
#[derive(Clone, Debug)]
pub enum RemlLamlError {
    /// The inner mode is at a fold: its Laplace normalizer approximates no integral, so the trial
    /// point is refused, with no value or derivative standing in for one.
    InnerModeFold(InnerModeFold),
    /// The constrained Laplace term could not be formed at this trial point (gam#2765), so it
    /// is refused with no value or derivative standing in for one.
    ConeNormalizer(crate::constrained_posterior::ConeLaplaceRefusal),
    /// Any other failure, with its diagnostic.
    Failed(String),
}

impl std::fmt::Display for RemlLamlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InnerModeFold(fold) => write!(f, "{fold}"),
            Self::ConeNormalizer(refusal) => write!(f, "{refusal}"),
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

impl From<String> for RemlLamlError {
    fn from(reason: String) -> Self {
        Self::Failed(reason)
    }
}

impl From<RemlError> for RemlLamlError {
    fn from(error: RemlError) -> Self {
        Self::Failed(error.into())
    }
}

/// Four additive scalar atoms of the unified criterion.
///
/// `fixed_beta` owns every scalar other than the two determinant terms and the
/// accepted-inner-mode correction. This includes configured priors, barriers,
/// Firth, Tierney–Kadane, and sampled-block corrections. The four values always
/// sum to [`RemlLamlResult::cost`].
#[derive(Clone, Copy, Debug)]
pub struct RemlCriterionComponents {
    pub fixed_beta: f64,
    pub logdet_h: f64,
    pub logdet_s: f64,
    pub kkt: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::estimate::smooth_floor_dp;

/// Residual degrees of freedom `ν = n − M_p` of the profiled-Gaussian scale.
///
/// `n` is the positive-weight observation count and `M_p = p − rank(S_λ)` the
/// number of UNPENALIZED coefficient directions, so `ν` is a difference of two
/// integer counts: it is either `≥ 1` or `≤ 0`, never in between. Both
/// consumers — the profiled scale `φ̂ = D_p/ν` and the `(ν/2)·log(2πφ̂)`
/// REML term — are undefined at `ν ≤ 0`: a design whose unpenalized directions
/// already exhaust the observations carries no residual information from which
/// to estimate a scale. This refuses there, which is what the rest of the
/// codebase does with exactly this condition
/// (`estimate/optimizer.rs`, `gaussian_reml.rs` × 3,
/// `fit_orchestration/drivers/design_construction.rs`).
///
/// It replaces a `.max(1e-8)` clamp (#2669). Because `ν` is integer-valued that
/// clamp could never interpolate: it was exactly `if ν ≤ 0 { 1e-8 }`, and what
/// it produced there was `φ̂ = D_p/1e-8`, which collapses the data-fit term
/// `D_p/(2φ̂) = ν/2` to `5e-9` INDEPENDENTLY of the response and leaves the
/// outer optimizer selecting λ against a bare `½(log|H| − log|S|)` determinant
/// ratio. Fabricating a finite criterion for a structurally invalid fit is
/// worse than refusing it (SPEC: a fit object must only ever come from a
/// converged optimization).
///
/// Takes the two scalars rather than the whole `InnerSolution` so the refusal
/// is directly exercisable — see `profiled_gaussian_residual_dof_tests`.
pub(crate) fn profiled_gaussian_residual_dof(
    n_observations: usize,
    nullspace_dim: f64,
) -> Result<f64, String> {
    let dof = n_observations as f64 - nullspace_dim;
    if dof > 0.0 {
        Ok(dof)
    } else {
        Err(format!(
            "profiled Gaussian residual degrees of freedom must be positive; got \
             n({n_observations}) − M_p({nullspace_dim}) = {dof}. Every unpenalized \
             coefficient direction consumes one observation, so this design leaves \
             nothing to estimate the Gaussian scale from: penalize the offending \
             directions or drop them."
        ))
    }
}

/// The profiled Gaussian scale `φ̂ = D_p/ν` at an inner mode, with the pieces the criterion's
/// value and derivatives read.
///
/// `D_p = −2ℓ(β̂) + β̂ᵀS_λβ̂` is the penalized deviance, floored smoothly and relative to the
/// response's own deviance scale ([`crate::estimate::smooth_floor_dp`]), and `ν = n − M_p` is the
/// residual degrees of freedom the scale is estimated from.
pub(crate) struct ProfiledGaussianScale {
    /// `D_p` before the floor, which the ρ audit reports beside the floored value.
    pub(crate) raw_deviance: f64,
    /// The floored `D_p` and the floor's first and second derivatives in the raw deviance.
    pub(crate) deviance: f64,
    pub(crate) deviance_gradient: f64,
    pub(crate) deviance_curvature: f64,
    /// `ν = n − M_p`.
    pub(crate) residual_dof: f64,
    /// `φ̂ = D_p/ν`.
    pub(crate) scale: f64,
}

/// The profiled Gaussian scale at an inner mode, from the four quantities it is a function of.
///
/// This is the ONE rule for `φ̂`. The criterion reads it in [`reml_laml_evaluate`], and the
/// constrained Laplace term reads it again when it prices its integral on the posterior precision
/// `H/φ̂` (gam#2765): two spellings of `φ̂` would price two posteriors for one mode, and the
/// term's gradient would then not be the derivative of the criterion's value.
pub(crate) fn profiled_gaussian_scale(
    log_likelihood: f64,
    penalty_quadratic: f64,
    n_observations: usize,
    nullspace_dim: f64,
    deviance_floor_scale: f64,
) -> Result<ProfiledGaussianScale, String> {
    // `penalty_quadratic` is the FULL `β̂ᵀSλβ̂`; the criterion halves it itself.
    let raw_deviance = -2.0 * log_likelihood + penalty_quadratic;
    let (deviance, deviance_gradient, deviance_curvature) =
        crate::estimate::smooth_floor_dp(raw_deviance, deviance_floor_scale);
    let residual_dof = profiled_gaussian_residual_dof(n_observations, nullspace_dim)?;
    Ok(ProfiledGaussianScale {
        raw_deviance,
        deviance,
        deviance_gradient,
        deviance_curvature,
        residual_dof,
        scale: deviance / residual_dof,
    })
}

/// Apply the curvature-conditioning scale `s = rho_curvature_scale` to a
/// raw ρ-coordinate `λ_k = exp(ρ_k)`.
///
/// Returns `s · λ_k`, which is the per-coordinate drift coefficient
/// `∂H_op/∂ρ_k = s · λ_k · S_k` under the convention documented on
/// [`InnerSolution::rho_curvature_scale`].  The matching
/// `hessian_logdet_correction = −p · log(s)` (additive in ρ, derivative
/// zero) cancels the `p · log(s)` term in `log|H_op|` so that the cost
/// the evaluator reports and the trace `tr(K · s·λ_k·S_k)` (with
/// `K = H_op⁻¹ = (1/s) · H_orig⁻¹`) both correspond to the SAME unscaled
/// `log|H_orig|` and its analytic derivative `tr(H_orig⁻¹ · λ_k S_k)`.
///
/// If you change this scaling, you MUST also update the corresponding
/// `hessian_logdet_correction` in every caller that sets
/// `rho_curvature_scale ≠ 1`, or the cost and gradient will disagree by
/// a factor `s` — see issue #200 for the failure mode.
#[inline]
pub(crate) fn rho_curvature_lambda(solution: &InnerSolution<'_>, lambda: f64) -> f64 {
    solution.rho_curvature_scale * lambda
}

pub(crate) fn penalty_coord_to_operator(
    coord: PenaltyCoordinate,
    scale: f64,
) -> Arc<dyn HyperOperator> {
    use gam_linalg::faer_ndarray::{fast_ab, fast_ata, fast_atb};
    use ndarray::s;

    struct OwnedPenaltyHyperOperator {
        pub(crate) coord: PenaltyCoordinate,
        pub(crate) scale: f64,
    }

    impl HyperOperator for OwnedPenaltyHyperOperator {
        fn dim(&self) -> usize {
            self.coord.dim()
        }

        fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(v.len());
            self.mul_vec_into(v.view(), out.view_mut());
            out
        }

        fn as_any(&self) -> &(dyn std::any::Any + 'static) {
            self
        }

        fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(v.len());
            self.mul_vec_into(v, out.view_mut());
            out
        }

        fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
            self.coord.apply_penalty_view_into(v, self.scale, out);
        }

        fn scaled_add_mul_vec(
            &self,
            v: ArrayView1<'_, f64>,
            scale: f64,
            out: ArrayViewMut1<'_, f64>,
        ) {
            if scale == 0.0 {
                return;
            }
            self.coord
                .scaled_add_penalty_view(v, scale * self.scale, out);
        }

        // `B = scale · RᵀR` on the root's column block, so every factor
        // contraction is two GEMMs through the root `R` (rank × block) rather
        // than the trait default's two matvecs per factor column.
        fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
            let (root, start, end) = self.root_block();
            let root_factor = fast_ab(root, &factor.slice(s![start..end, ..]));
            let mut out = Array2::<f64>::zeros(factor.dim());
            let mut block = out.slice_mut(s![start..end, ..]);
            block.assign(&fast_atb(root, &root_factor));
            block *= self.scale;
            out
        }

        fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
            let (root, start, end) = self.root_block();
            let root_factor = fast_ab(root, &factor.slice(s![start..end, ..]));
            self.scale * root_factor.iter().map(|&value| value * value).sum::<f64>()
        }

        fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
            let (root, start, end) = self.root_block();
            let root_factor = fast_ab(root, &factor.slice(s![start..end, ..]));
            let mut projected = fast_ata(&root_factor);
            projected *= self.scale;
            projected
        }

        fn to_dense(&self) -> Array2<f64> {
            self.coord.scaled_dense_matrix(self.scale)
        }

        fn is_implicit(&self) -> bool {
            false
        }
    }

    impl OwnedPenaltyHyperOperator {
        fn root_block(&self) -> (&Array2<f64>, usize, usize) {
            self.coord
                .block_local_root()
                .expect("every penalty coordinate carries its root in a block chart")
        }
    }

    Arc::new(OwnedPenaltyHyperOperator { coord, scale })
}

pub(crate) fn penalty_total_drift_result(
    coord: &PenaltyCoordinate,
    scale: f64,
    correction: Option<&DriftDerivResult>,
) -> DriftDerivResult {
    match correction {
        Some(DriftDerivResult::Dense(corr)) => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: Some(corr.clone()),
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                let mut dense = coord.scaled_dense_matrix(scale);
                dense += corr;
                DriftDerivResult::Dense(dense)
            }
        }
        Some(DriftDerivResult::Operator(corr_op)) => {
            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                dense: if coord.uses_operator_fast_path() {
                    None
                } else {
                    Some(coord.scaled_dense_matrix(scale))
                },
                operators: {
                    let mut ops = vec![Arc::clone(corr_op)];
                    if coord.uses_operator_fast_path() {
                        ops.push(penalty_coord_to_operator(coord.clone(), scale));
                    }
                    ops
                },
                dim_hint: coord.dim(),
            }))
        }
        None => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: None,
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                DriftDerivResult::Dense(coord.scaled_dense_matrix(scale))
            }
        }
    }
}

pub(crate) fn hyper_coord_drift_operators(drift: &HyperCoordDrift) -> Vec<Arc<dyn HyperOperator>> {
    let mut operators: Vec<Arc<dyn HyperOperator>> = Vec::new();
    if let Some(block_local) = drift.block_local.as_ref() {
        operators.push(Arc::new(block_local.clone()));
    }
    if let Some(operator) = drift.operator.as_ref() {
        operators.push(Arc::clone(operator));
    }
    operators
}

pub(crate) fn hyper_coord_drift_operator_arc(
    drift: &HyperCoordDrift,
    dim_hint: usize,
) -> Option<Arc<dyn HyperOperator>> {
    let mut operators = hyper_coord_drift_operators(drift);
    if operators.is_empty() {
        return None;
    }

    if drift.dense.is_none() && operators.len() == 1 {
        return Some(operators.pop().expect("single operator drift"));
    }

    Some(Arc::new(CompositeHyperOperator {
        dense: drift.dense.clone(),
        operators,
        dim_hint,
    }))
}

pub(crate) fn drift_parts_into_result(
    dense: Option<Array2<f64>>,
    mut operators: Vec<Arc<dyn HyperOperator>>,
    dim_hint: usize,
) -> DriftDerivResult {
    if operators.is_empty() {
        DriftDerivResult::Dense(dense.unwrap_or_else(|| Array2::<f64>::zeros((dim_hint, dim_hint))))
    } else if dense.is_none() && operators.len() == 1 {
        DriftDerivResult::Operator(operators.pop().expect("single operator drift"))
    } else {
        DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
            dense,
            operators,
            dim_hint,
        }))
    }
}

pub(crate) fn hyper_coord_total_drift_parts(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
) -> (Option<Array2<f64>>, Vec<Arc<dyn HyperOperator>>) {
    let mut dense = drift.dense.clone();
    let mut operators = hyper_coord_drift_operators(drift);
    if let Some(correction) = correction {
        match correction {
            DriftDerivResult::Dense(matrix) => {
                if let Some(existing) = dense.as_mut() {
                    *existing += matrix;
                } else {
                    dense = Some(matrix.clone());
                }
            }
            DriftDerivResult::Operator(operator) => operators.push(Arc::clone(operator)),
        }
    }
    (dense, operators)
}

pub(crate) fn hyper_coord_total_drift_result(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
    dim_hint: usize,
) -> DriftDerivResult {
    let (dense, operators) = hyper_coord_total_drift_parts(drift, correction);
    drift_parts_into_result(dense, operators, dim_hint)
}

// ─── EFS multiplicative-update helpers ───────────────────────────────────
//
// The Wood–Fasiolo Extended Fellner–Schall update is multiplicative in the
// smoothing parameter. Writing it in log coordinates `ρ = log λ`,
//
//   Δρ = log( target / q_eff )
//      = log( ( d − t ) / q_eff )
//
// where:
//   • q_eff is the penalty-quadratic contribution to the *gradient*,
//     scaled exactly the way `outer_gradient_entry` scales it. For Fixed
//     dispersion, q_eff = β̂ᵀ B β̂ = 2 a_i. For ProfiledGaussian, it picks
//     up the smooth-floor factor `dp_cgrad / φ̂` so EFS and the gradient
//     share the same stationarity equation.
//   • d = ∂ log|S_λ|₊/∂ρ_i = tr(S_λ⁺ B_i). For ρ-coords this is
//     `solution.penalty_logdet.first[idx]`; for τ-coords it is
//     `coord.ld_s`.
//   • t = tr(K · B_i) where K is the *cost's* logdet kernel — `G_ε(H)` in
//     ordinary SPD/smooth-spectral mode, or the projected
//     `U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ` under the rank-deficient LAML fix.
//
// The previous implementation used `Δρ = (2a − tr(H⁻¹B)) / tr(H⁻¹BH⁻¹B)`,
// which (a) silently dropped the `tr(S_λ⁺ B)` term, (b) used a different
// kernel from the gradient, and (c) used the Frobenius/Gram trace as a
// curvature proxy instead of the canonical EFS denominator. As a concrete
// counterexample, the scalar Gaussian/Laplace model with z = 2, λ = 1/3 is
// at the exact REML optimum (gradient = 0) but the old formula returned
// step `+8` (clamped to `+5`) — see the unit test in this module.
//
// Exactness depends on the likelihood curvature. For Gaussian/quadratic
// likelihoods, `H_obs` is beta-independent, so `C[v_k] = 0` and the
// classical explicit trace fixed point with `Ḣ_k = λ_k S_k` is exact. For
// non-Gaussian families (Cox/survival/binomial), `H_obs` depends on beta;
// the exact logdet gradient uses the total Hessian drift
// `Ḣ_k = λ_k S_k + C[v_k]`. A pure MacKay/Tipping/Wood-Fasiolo explicit
// trace update that uses only `λ_k S_k` is therefore an approximation.
//
// This code path does not use that pure explicit-trace surrogate. EFS is
// expressed in terms of the full outer gradient from `reml_laml_evaluate`;
// that gradient builds `rho_corrections`, threads them through
// `penalty_total_drift_result`, and traces the corrected `Ḣ_k`.

/// `q_eff = 2 · penalty_term` matching `outer_gradient_entry`.
#[inline]
pub(crate) fn efs_q_eff(a_i: f64, dispersion: &DispersionHandling, dp_cgrad: f64, phi: f64) -> f64 {
    match dispersion {
        DispersionHandling::ProfiledGaussian => 2.0 * dp_cgrad * a_i / phi,
        DispersionHandling::Fixed { .. } => 2.0 * a_i,
    }
}

pub(crate) fn gamma_precision_rate_for_rho(
    prior: &gam_problem::RhoPrior,
    idx: usize,
) -> Option<f64> {
    match prior {
        gam_problem::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
        gam_problem::RhoPrior::Independent(priors) => {
            priors.get(idx).and_then(|prior| match prior {
                gam_problem::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
                _ => None,
            })
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn efs_q_eff_with_gamma_rate(
    base_q_eff: f64,
    lambda: f64,
    prior: &gam_problem::RhoPrior,
    idx: usize,
) -> f64 {
    match gamma_precision_rate_for_rho(prior, idx) {
        Some(rate) if rate.is_finite() && rate > 0.0 => base_q_eff + 2.0 * rate * lambda,
        _ => base_q_eff,
    }
}

/// EFS step expressed in terms of the *full* outer gradient
/// `g_full = ∂V_total/∂ρ_i` and the penalty-quadratic curvature scale
/// `q_eff`:
///
/// ```text
///   Δρ = log(1 − 2·g_full / q_eff).
/// ```
///
/// This is the universal-form Wood–Fasiolo update: when the cost is base
/// REML/LAML, the canonical `g_base = (q_eff + t − d)/2` gives
/// `1 − 2·g_base/q_eff = (d − t)/q_eff` (the classical pseudoinverse-and-
/// trace form); when out-of-band terms — Tierney–Kadane corrections,
/// smoothing-parameter priors, Firth bias-reduction, monotonicity
/// barriers — enter `g_full = g_base + g_extra`,
/// the multiplicative target shifts by exactly the right amount,
/// `1 − 2·g_full/q_eff = (d − t − 2·g_extra)/q_eff`. No per-augmentation
/// post-correction is needed in `compute_efs_update` /
/// `compute_hybrid_efs_update`. The line search in the outer
/// fixed-point bridge handles the only thing this formula can't —
/// non-PSD penalty derivatives that flip the descent direction.
///
/// The update solves the multiplicative model of the gradient along `ρ_i`
/// that holds every trace fixed and moves only the penalty quadratic, which is
/// linear in `λ_i`:
///
/// ```text
///   g(ρ + Δ) ≈ g_full + q_eff·(e^Δ − 1)/2.
/// ```
///
/// Three regimes:
/// - **Stable (`q_eff > 0`, `2·g_full < q_eff`)**: the model's root,
///   `Δ = log(1 − 2·g_full/q_eff)`, taken whole. Its length is the outer
///   fixed-point bridge's to decide: it clips the step to the outer domain and
///   contracts it by the cost line search. A per-coordinate box on it (#2902)
///   slowed every walk whose root lay farther away than the box, with no
///   property of the problem behind the box's width.
/// - **Over-correction (`q_eff > 0`, `2·g_full ≥ q_eff`)**: the model has no
///   root (its infimum `g_full − q_eff/2` is still non-negative), so the step
///   is the model's Newton step from `Δ = 0`, `−2·g_full/q_eff`, along the
///   model's own curvature `q_eff/2 > 0`: a downhill step whose length the
///   same line search decides, after which the next evaluation re-linearizes.
/// - **Pathological (`q_eff ≤ 0` or non-finite)**: returns `None` so the
///   caller leaves the step at zero for that coordinate.
#[inline]
pub(crate) fn efs_log_step_from_grad(q_eff: f64, g_full: f64) -> Option<f64> {
    if !q_eff.is_finite() || q_eff <= 0.0 || !g_full.is_finite() {
        return None;
    }
    let relative = -2.0 * g_full / q_eff;
    if relative > -1.0 {
        Some(relative.ln_1p())
    } else {
        Some(relative)
    }
}

/// EFS profiling factors (`profiled_scale`, `dp_cgrad`) matched to the
/// gradient assembly. For Fixed dispersion both are unused; we return
/// `(phi, 0.0)` so that `efs_q_eff` simply uses `2·a_i`.
#[inline]
pub(crate) fn efs_profiling(solution: &InnerSolution<'_>) -> Result<(f64, f64), String> {
    match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, _) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
            let denom =
                profiled_gaussian_residual_dof(solution.n_observations, solution.nullspace_dim)?;
            Ok((dp_c / denom, dp_cgrad))
        }
        DispersionHandling::Fixed { phi, .. } => Ok((*phi, 0.0)),
    }
}

pub(crate) fn trace_hinv_cached_drift_cross(
    hop: &dyn HessianFactorization,
    left_dense: Option<&Array2<f64>>,
    left_op: Option<&dyn HyperOperator>,
    right_dense: Option<&Array2<f64>>,
    right_op: Option<&dyn HyperOperator>,
) -> f64 {
    match (left_op, right_op) {
        (Some(left), Some(right)) => hop.trace_hinv_operator_cross(left, right),
        (Some(left), None) => hop.trace_hinv_matrix_operator_cross(
            right_dense.expect("right dense drift should be cached"),
            left,
        ),
        (None, Some(right)) => hop.trace_hinv_matrix_operator_cross(
            left_dense.expect("left dense drift should be cached"),
            right,
        ),
        (None, None) => hop.trace_hinv_product_cross(
            left_dense.expect("left dense drift should be cached"),
            right_dense.expect("right dense drift should be cached"),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared outer-derivative formulas
// ═══════════════════════════════════════════════════════════════════════════
//
// These helpers implement the analytic identities ONCE so that all
// coordinate types (ρ, τ, ψ) and all pair types (ρ-ρ, ρ-ext, ext-ext)
// go through the same formula. Any chain-rule or transformed-parameter
// fix automatically applies to every code path.

/// Compute one entry of the outer gradient.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂V/∂θ_i = a_i_scaled + ½ tr(G_ε Ḣ_i) − ½ ∂_i log|S|₊
/// ```
///
/// where:
/// - `a_i` is the fixed-β cost derivative (0.5 × β̂ᵀAₖβ̂ for ρ, coord.a for ext)
/// - `trace_logdet_i` is tr(G_ε(H) Ḣ_i) (logdet gradient operator applied to
///   the total Hessian drift including IFT correction)
/// - `ld_s_i` is ∂_i log|S|₊ (penalty pseudo-logdet derivative)
///
/// The dispersion handling scales the penalty term:
/// - Profiled Gaussian: dp_cgrad × a_i / φ̂
/// - Fixed dispersion: a_i
#[inline]
pub(crate) fn outer_gradient_entry(
    a_i: f64,
    trace_logdet_i: f64,
    ld_s_i: f64,
    dispersion: &DispersionHandling,
    dp_cgrad: f64,
    profiled_scale: f64,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let penalty_term = match dispersion {
        DispersionHandling::ProfiledGaussian => dp_cgrad * a_i / profiled_scale,
        DispersionHandling::Fixed { .. } => a_i,
    };
    let trace_term = if incl_logdet_h {
        0.5 * trace_logdet_i
    } else {
        0.0
    };
    let det_term = if incl_logdet_s { 0.5 * ld_s_i } else { 0.0 };
    penalty_term + trace_term - det_term
}

/// The profiled Gaussian criterion's data-fit channel at second order,
/// `∂²[D_p/(2φ̂)]` through the floored penalized deviance `D_p = f(D_raw)`:
/// with `q_raw = pair_a − g_iᵀv_j = ½D̈_raw` and `Ḋ_raw,c = 2a_c`,
/// `f′q_raw/φ̂ + 2(f″νφ̂ − f′²)a_ia_j/(νφ̂²)`.
///
/// It is also `½ν·ℓ_ij` for the scale rate `ℓ_c = φ̂̇_c/φ̂` and `ℓ_ij = ∂ℓ_i/∂θ_j`, which is how
/// the constrained Laplace term's profiled Hessian reads `φ̂̈` (gam#3234): one formula, so the
/// criterion and the term cannot disagree about how the scale moves.
pub(crate) fn profiled_data_fit_second_derivative(
    a_i: f64,
    a_j: f64,
    g_i_dot_v_j: f64,
    pair_a: f64,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
) -> f64 {
    let q_raw = pair_a - g_i_dot_v_j;
    profiled_dp_cgrad * q_raw / profiled_phi
        + 2.0
            * (profiled_dp_cgrad2 * profiled_nu * profiled_phi
                - profiled_dp_cgrad * profiled_dp_cgrad)
            * a_i
            * a_j
            / (profiled_nu * profiled_phi * profiled_phi)
}

/// Compute one entry of the outer Hessian.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂²V/∂θ_i∂θ_j = Q_ij + L_ij + P_ij
/// ```
///
/// where:
/// - Q_ij = pair_a − g_i·v_j  (penalty quadratic second derivative, with
///   profiled Gaussian chain-rule terms from the smooth deviance floor)
/// - L_ij = ½ (cross_trace + h2_trace) (logdet Hessian)
/// - P_ij = −½ pair_ld_s  (penalty logdet second derivative)
///
/// The `cross_trace` is the exact logdet spectral cross term. For ordinary
/// SPD backends this is `−tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i)`; for smooth spectral logdet
/// regularization it is the divided-difference contraction of
/// `log r_ε(σ)`. The `h2_trace` is tr(G_ε Ḧ_ij) from the second Hessian
/// drift including IFT and fourth-derivative corrections.
#[inline]
pub(crate) fn outer_hessian_entry(
    a_i: f64,
    a_j: f64,
    g_i_dot_v_j: f64,
    pair_a: f64,
    cross_trace: f64,
    h2_trace: f64,
    pair_ld_s: f64,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let q = if is_profiled {
        profiled_data_fit_second_derivative(
            a_i,
            a_j,
            g_i_dot_v_j,
            pair_a,
            profiled_phi,
            profiled_nu,
            profiled_dp_cgrad,
            profiled_dp_cgrad2,
        )
    } else {
        pair_a - g_i_dot_v_j
    };
    let l = if incl_logdet_h {
        0.5 * (cross_trace + h2_trace)
    } else {
        0.0
    };
    let p = if incl_logdet_s { -0.5 * pair_ld_s } else { 0.0 };
    q + l + p
}

// ═══════════════════════════════════════════════════════════════════════════
//  Constraint-tangent-space projection
// ═══════════════════════════════════════════════════════════════════════════
//
// When the inner solver converges at a constrained-stationary point with a
// non-empty active inequality-constraint set `A_act β = b_act` (k_act rows),
// the Laplace approximation lives on the tangent manifold `T = β̂ + null(A_act)`.
// With orthonormal basis `Z ∈ ℝ^{p × m}` for null(A_act) (m = p − k_act), the
// principled outer LAML objective is
//
//   V_T(ρ) = -ℓ(β̂) + ½ β̂ᵀ S(λ) β̂ + ½ log|ZᵀHZ| − ½ log|Zᵀ S(λ) Z|_+ + …
//
// (β̂-quadratic terms stay in p-space; β̂ doesn't change under projection.)
// The gradient is the envelope-theorem derivative at fixed β̂:
//
//   ∂_ρ_k V_T = ½ λ_k β̂ᵀ S_k β̂ + ½ tr((ZᵀHZ)⁻¹ Zᵀ(λ_k S_k) Z)
//             − ½ λ_k tr((ZᵀSZ)⁺ ZᵀS_kZ)
//
// Refs: Wood 2011; Wood–Pya–Säfken 2016 §3; Marra–Wood 2012 §2.
//
// The implementation strategy: wrap the inner Hessian operator in a
// tangent-projected adapter that transforms its trace/solve/logdet APIs
// from p-space to tangent space, recompute `PenaltyLogdetDerivs` for
// `ZᵀS(λ)Z`, then recurse into the regular `reml_laml_evaluate` with
// `active_constraints = None`. This routes the entire downstream pipeline
// (gradient, Hessian, IFT corrections) through the projected operator
// without duplicating cost/gradient formulas.

/// Authoritative coefficient geometry of a non-empty active constraint face.
pub enum ActiveConstraintTangentGeometry {
    /// The active rows span coefficient space, so the mode is fully pinned.
    FullyPinned,
    /// Orthonormal basis `Z` for the non-empty tangent `null(A_act)`.
    Tangent(Array2<f64>),
}

/// Row-normalize a non-empty active constraint block, returning the normalized
/// rows alongside the norms that were divided out.
///
/// Constraint feasibility, working-face membership, and every active-set KKT
/// gate are defined in scaled slack `(a·beta-b)/‖a‖`; rank, tangent and affine
/// solutions must be invariant to multiplying an inequality by an arbitrary
/// positive constant as well. Factoring raw Khatri–Rao rows lets large-norm
/// rows numerically erase independent small-norm rows, producing a direction
/// that is tangent only in an unscaled least-squares aggregate and leaves the
/// solver's actual face. A zero or non-finite row is left alone with a unit
/// norm so the caller's own row validity check — not this scaling — is what
/// reports it.
fn normalize_active_face_rows(a_act: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut normalized = a_act.clone();
    let mut row_norms = Array1::<f64>::ones(a_act.nrows());
    for (index, mut row) in normalized.rows_mut().into_iter().enumerate() {
        let norm = row.dot(&row).sqrt();
        if norm.is_finite() && norm > 0.0 {
            row /= norm;
            row_norms[index] = norm;
        }
    }
    (normalized, row_norms)
}

/// Numerical row rank and tangent geometry of an already-normalized active
/// block, read off a factorization the caller already paid for.
///
/// Splitting this out is what lets the tangent, the rank and the affine
/// particular solution of one face be answered from ONE factorization instead
/// of from three that can disagree (gam#2600).
fn tangent_from_row_factorization(
    normalized: &Array2<f64>,
    singular: &Array1<f64>,
    vt: &Array2<f64>,
) -> Result<(usize, ActiveConstraintTangentGeometry), String> {
    let p = normalized.ncols();
    // The row rank is read at the SVD's own rounding band `max(rows, p)·ε·σ_max`
    // (`svd_rank_band`), with no extra factor: a singular value above it is
    // resolved by the factorization that produced it (#2469).
    let rank = crate::active_set::svd_rank(singular, normalized.nrows(), p);
    if rank == 0 {
        return Err("non-empty active constraint block has zero numerical row rank".to_string());
    }
    if rank == p {
        return Ok((rank, ActiveConstraintTangentGeometry::FullyPinned));
    }

    // The orthonormal complement of the leading `rank` right singular vectors.
    // Pivoted Gram–Schmidt on the coordinate axes always takes the longest
    // remaining residual, whose squared norm is at least `(p − rank − j)/p`
    // after `j` columns, so no cutoff on residual length is needed (#2469).
    let null_count = p - rank;
    let z = crate::active_set::null_space_complement(vt, rank).ok_or_else(|| {
        format!(
            "active constraint tangent complement construction produced no {null_count}-column basis \
             for row rank {rank} over {} right singular vectors",
            vt.nrows()
        )
    })?;

    // This routine is the shared authority for both optimization and LAML
    // projection. Refuse geometry that cannot preserve the solver's working-
    // face contract instead of silently returning a numerically false tangent.
    for row in normalized.rows() {
        let row_norm = row.dot(&row).sqrt();
        if row_norm == 0.0 {
            continue;
        }
        for tangent in z.columns() {
            let relative_leakage = row.dot(&tangent).abs() / row_norm;
            if relative_leakage > crate::active_set::ACTIVE_SET_WORKING_FACE_TOL {
                return Err(format!(
                    "active constraint tangent leaks through its working face: relative leakage {relative_leakage:.3e}"
                ));
            }
        }
    }
    Ok((rank, ActiveConstraintTangentGeometry::Tangent(z)))
}

/// Compute the coefficient geometry of a non-empty active constraint face.
///
/// This is shared by the terminal inner determinant and the outer evaluator so
/// the value cannot classify curvature in one coefficient space while its
/// derivatives use another. A non-empty row block with zero numerical rank is
/// invalid active-set evidence, not an unconstrained fallback.
pub fn active_constraint_tangent_geometry(
    a_act: &Array2<f64>,
) -> Result<ActiveConstraintTangentGeometry, String> {
    if a_act.nrows() == 0 {
        return Err("active constraint tangent geometry requires at least one row".to_string());
    }
    let p = a_act.ncols();
    if p == 0 {
        return Ok(ActiveConstraintTangentGeometry::FullyPinned);
    }
    let (normalized, _row_norms) = normalize_active_face_rows(a_act);

    // Factor the rectangular normalized row block directly. Forming `A_actᵀ A_act`
    // squares its condition number and can erase independent active rows near
    // machine precision. That produces a tangent which leaks in a constraint-
    // normal direction: the next accepted point then falls off the working
    // face and discards the entire warm active set. The thin SVD gives the row
    // rank without normal equations; complete its right-singular row basis to
    // an orthonormal null basis by pivoted Gram–Schmidt on the coordinate axes.
    //
    // Left singular vectors are NOT requested here: this entry point answers
    // only the tangent question, and the callers that also need the affine
    // particular solution go through `active_constraint_face_geometry`, which
    // pays for `U` once and shares this exact rank rule and completion.
    let (_u, singular, vt) = normalized
        .svd(false, true)
        .map_err(|error| format!("active constraint tangent SVD failed: {error}"))?;
    let vt = vt.ok_or_else(|| "active constraint tangent SVD omitted Vᵀ".to_string())?;
    tangent_from_row_factorization(&normalized, &singular, &vt).map(|(_rank, geometry)| geometry)
}

/// Minimum-norm particular solution of an active face's affine system,
/// together with the residual the numerically null directions leave behind.
pub struct ParticularFaceSolution {
    /// The step `δ` of least Euclidean norm over the numerically identified
    /// row space, satisfying `A δ = rhs` up to `residual_inf`.
    pub delta: Array1<f64>,
    /// `‖A δ − rhs‖∞`, in scaled-slack units — the same units feasibility,
    /// working-face membership and every active-set gate are stated in.
    pub residual_inf: f64,
    /// Largest residual a *consistent* face can leave at this scale. Anything
    /// above it means the equalities cannot be met simultaneously: the face is
    /// wrong, not the arithmetic.
    pub residual_tolerance: f64,
}

/// Rank-revealing affine geometry of a non-empty active constraint face.
///
/// A reduced-face solve needs three things from one face: its numerical row
/// rank, an orthonormal basis of `null(A)`, and a particular solution of the
/// affine system `A δ = rhs`. Deciding those from three different
/// factorizations — which is what the physical reduced face used to do, taking
/// the rank from a Gram–Schmidt scan of `A`, the tangent from an SVD of `A`,
/// and the particular solution from an SVD of the trust-whitened `A D^{-1/2}`
/// — lets one face be simultaneously full rank and singular. That is exactly
/// how gam#2600 refused: `A` was accepted as 39 independent rows while the
/// whitened block reported `σ_min = 4.8e-18` against a `1.1e-13` floor, and
/// the trust metric's dynamic range, not the geometry, decided it.
///
/// This type carries ONE factorization of the row-normalized block and answers
/// all three from it. The trust metric is not involved: it selects *which*
/// solution of an underdetermined face is smallest, which is a separate
/// projection the caller applies afterwards, and it must not be allowed to
/// decide whether the face has a solution at all.
pub struct ActiveConstraintFaceGeometry {
    rank: usize,
    tangent: ActiveConstraintTangentGeometry,
    normalized: Array2<f64>,
    row_norms: Array1<f64>,
    left: Array2<f64>,
    singular: Array1<f64>,
    right_transposed: Array2<f64>,
}

impl ActiveConstraintFaceGeometry {
    /// Numerical row rank of the row-normalized face.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Largest singular value of the row-normalized face.
    pub fn largest_singular_value(&self) -> f64 {
        self.singular.iter().fold(0.0_f64, |largest, &s| largest.max(s))
    }

    /// Smallest singular value retained by the rank decision. `0.0` for an
    /// empty rank, which `active_constraint_face_geometry` already refuses.
    pub fn smallest_retained_singular_value(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.singular[self.rank - 1]
        }
    }

    /// Tangent geometry of the face, decided by the same rank rule.
    pub fn tangent(&self) -> &ActiveConstraintTangentGeometry {
        &self.tangent
    }

    /// Consume the geometry for its tangent basis.
    pub fn into_tangent(self) -> ActiveConstraintTangentGeometry {
        self.tangent
    }

    /// Minimum-Euclidean-norm solution of `A δ = rhs` over the numerically
    /// identified row space.
    ///
    /// `rhs` is stated against the ORIGINAL (unnormalized) rows and is scaled
    /// here by the same row norms the factorization used, so a face and its
    /// positively rescaled twin produce the identical `δ`.
    ///
    /// Truncating the directions below the rank floor is what makes this
    /// total: those directions cannot be resolved by any step of bounded norm,
    /// so the honest answer is the solution that ignores them plus the
    /// residual they leave — reported, not swallowed.
    pub fn minimum_norm_particular(
        &self,
        rhs: &Array1<f64>,
    ) -> Result<ParticularFaceSolution, String> {
        let rows = self.normalized.nrows();
        let cols = self.normalized.ncols();
        if rhs.len() != rows {
            return Err(format!(
                "active constraint face affine system needs one right-hand side per face row \
                 (rows={rows}, rhs={})",
                rhs.len()
            ));
        }
        let scaled_rhs: Array1<f64> = rhs
            .iter()
            .zip(self.row_norms.iter())
            .map(|(value, norm)| value / norm)
            .collect();
        if scaled_rhs.iter().any(|value| !value.is_finite()) {
            return Err(
                "active constraint face right-hand side is non-finite in scaled-slack units"
                    .to_string(),
            );
        }
        let mut delta = Array1::<f64>::zeros(cols);
        for mode in 0..self.rank {
            let coefficient = self.left.column(mode).dot(&scaled_rhs) / self.singular[mode];
            delta.scaled_add(coefficient, &self.right_transposed.row(mode));
        }
        if delta.iter().any(|value| !value.is_finite()) {
            return Err("active constraint face particular solution is non-finite".to_string());
        }
        let residual = self.normalized.dot(&delta) - &scaled_rhs;
        let residual_inf = residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let delta_norm = delta.dot(&delta).sqrt();
        let rhs_norm = scaled_rhs.dot(&scaled_rhs).sqrt();
        // Backward-error bound for a truncated-SVD least-squares solve. The
        // rows are unit-normalized, so `‖A‖₂ ≤ √rows`, and a CONSISTENT face
        // leaves at most `O(eps)·(‖A‖‖δ‖ + ‖rhs‖)`. Same `100·eps·max(k,p)`
        // factor the rank floor above uses, and — like it — scale-covariant
        // with no absolute floor, so a face stated in tiny units is not
        // declared inconsistent for being small.
        let residual_tolerance = 100.0
            * f64::EPSILON
            * (rows.max(cols).max(1) as f64)
            * ((rows as f64).sqrt() * delta_norm + rhs_norm);
        Ok(ParticularFaceSolution {
            delta,
            residual_inf,
            residual_tolerance,
        })
    }
}

/// Factor a non-empty active constraint face once, for every affine question
/// a reduced-face solve asks of it. See [`ActiveConstraintFaceGeometry`].
pub fn active_constraint_face_geometry(
    a_act: &Array2<f64>,
) -> Result<ActiveConstraintFaceGeometry, String> {
    if a_act.nrows() == 0 {
        return Err("active constraint face geometry requires at least one row".to_string());
    }
    let p = a_act.ncols();
    if p == 0 {
        return Err("active constraint face geometry requires at least one coefficient".to_string());
    }
    let (normalized, row_norms) = normalize_active_face_rows(a_act);
    let (u, singular, vt) = normalized
        .svd(true, true)
        .map_err(|error| format!("active constraint face SVD failed: {error}"))?;
    let left = u.ok_or_else(|| "active constraint face SVD omitted U".to_string())?;
    let right_transposed = vt.ok_or_else(|| "active constraint face SVD omitted Vᵀ".to_string())?;
    let (rank, tangent) = tangent_from_row_factorization(&normalized, &singular, &right_transposed)?;
    Ok(ActiveConstraintFaceGeometry {
        rank,
        tangent,
        normalized,
        row_norms,
        left,
        singular,
        right_transposed,
    })
}

#[cfg(test)]
mod active_constraint_tangent_geometry_tests {
    use super::*;

    #[test]
    fn direct_rectangular_factorization_preserves_ill_conditioned_row_rank() {
        // The two independent rows have condition number O(1e8), so forming
        // AᵀA pushes their squared condition to machine precision. The true
        // tangent is exactly the third coordinate and must remain one-dimensional.
        let a = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-7, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("direct SVD geometry")
        else {
            panic!("ill-conditioned rank-two face must have a tangent");
        };
        assert_eq!(z.dim(), (3, 1));
        assert!((z.column(0).dot(&z.column(0)) - 1.0).abs() < 1e-12);
        for row in a.rows() {
            assert!(row.dot(&z.column(0)).abs() / row.dot(&row).sqrt() < 1e-12);
        }
    }

    #[test]
    fn dependent_rows_produce_the_full_exact_null_space() {
        let a = ndarray::array![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("rank-one geometry")
        else {
            panic!("rank-one face in three dimensions must have a tangent");
        };
        assert_eq!(z.dim(), (3, 2));
        let gram = z.t().dot(&z);
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((gram[[i, j]] - target).abs() < 1e-12);
            }
        }
        assert!(a.dot(&z).iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn tangent_geometry_is_invariant_to_constraint_row_scaling() {
        let a = ndarray::array![[1e12, 1e12, 0.0], [1e-12, 1e-12 + 1e-19, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("row-scaled geometry")
        else {
            panic!("two independent normalized rows must leave one tangent dimension");
        };
        assert_eq!(z.dim(), (3, 1));
        for row in a.rows() {
            assert!(row.dot(&z.column(0)).abs() / row.dot(&row).sqrt() < 1e-12);
        }
    }

    /// #2469: the row rank is read at the SVD's own band `max(rows, p)·ε·σ_max`.
    /// Two unit rows at angle `θ ≈ 1.33e-14` have `σ₂ ≈ θ/√2 ≈ 9.4e-15`, ten
    /// times that band (`3·ε·√2 ≈ 9.4e-16`) and a tenth of the `100×` cutoff it
    /// replaced. The face is rank two, and its tangent is the one untouched axis.
    #[test]
    fn row_rank_is_read_at_the_svd_rounding_band_2469() {
        let theta = 1.33e-14_f64;
        let a = ndarray::array![[1.0, 0.0, 0.0], [1.0, theta, 0.0]];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        let smax = geometry.singular.iter().fold(0.0_f64, |acc, &s| acc.max(s));
        let band = 3.0 * f64::EPSILON * smax;
        let smallest = geometry.singular.iter().fold(f64::INFINITY, |acc, &s| acc.min(s));
        assert!(
            smallest > band && smallest < 100.0 * band,
            "fixture premise: the second singular value {smallest:.3e} sits between the band \
             {band:.3e} and the replaced cutoff {:.3e}",
            100.0 * band
        );
        assert_eq!(geometry.rank(), 2);
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("tangent geometry")
        else {
            panic!("a rank-two face in three dimensions has a tangent");
        };
        assert_eq!(z.dim(), (3, 1));
        assert!((z[[2, 0]].abs() - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn face_geometry_and_tangent_geometry_cannot_disagree_about_rank_2600() {
        // Two rows whose normalized independence is O(1e-8) and a third
        // coordinate nobody touches. Both entry points must report the same
        // rank and the same tangent dimension, because they now read the same
        // rule off the same factorization.
        let a = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-8, 0.0]];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        assert_eq!(geometry.rank(), 2);
        let ActiveConstraintTangentGeometry::Tangent(direct) =
            active_constraint_tangent_geometry(&a).expect("tangent geometry")
        else {
            panic!("rank-two face in three dimensions must have a tangent");
        };
        let ActiveConstraintTangentGeometry::Tangent(shared) = geometry.into_tangent() else {
            panic!("face geometry must agree that the face has a tangent");
        };
        assert_eq!(direct.dim(), shared.dim());
        // Both bases are built by the same pivoted completion
        // from the same rank, so they span the same line; assert the span
        // rather than the bits, since only one of the two SVD calls also asks
        // for `U` and the factorization is free to differ in the last digits.
        for row in a.rows() {
            for column in shared.columns() {
                assert!(row.dot(&column).abs() / row.dot(&row).sqrt() < 1e-12);
            }
        }
    }

    #[test]
    fn minimum_norm_particular_solves_a_consistent_face_and_reports_its_residual_2600() {
        // Rank-one face in three dimensions: `x = 2` with a redundant restated
        // copy at ten times the scale. The minimum-norm solution is the axis
        // point, and it must not depend on the redundancy or the row scaling.
        let a = ndarray::array![[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let rhs = ndarray::array![2.0, 20.0];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        assert_eq!(geometry.rank(), 1);
        let particular = geometry
            .minimum_norm_particular(&rhs)
            .expect("consistent rank-one face");
        assert!((particular.delta[0] - 2.0).abs() < 1e-12);
        assert!(particular.delta[1].abs() < 1e-12);
        assert!(particular.delta[2].abs() < 1e-12);
        assert!(
            particular.residual_inf <= particular.residual_tolerance,
            "a consistent face must not be reported inconsistent \
             (residual={:.6e}, tolerance={:.6e})",
            particular.residual_inf,
            particular.residual_tolerance
        );
    }

    #[test]
    fn minimum_norm_particular_names_an_inconsistent_face_instead_of_solving_it_2600() {
        // The same two parallel rows now demand contradictory offsets. No step
        // satisfies both, and that has to surface as a residual above the
        // backward-error bound rather than as a plausible-looking `delta`.
        let a = ndarray::array![[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let rhs = ndarray::array![2.0, 30.0];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        let particular = geometry
            .minimum_norm_particular(&rhs)
            .expect("least-squares answer still exists");
        assert!(
            particular.residual_inf > particular.residual_tolerance,
            "a contradictory face must exceed its own consistency bound \
             (residual={:.6e}, tolerance={:.6e})",
            particular.residual_inf,
            particular.residual_tolerance
        );
    }

    #[test]
    fn minimum_norm_particular_is_invariant_to_positive_row_rescaling_2600() {
        let a = ndarray::array![[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let rhs = ndarray::array![3.0, 5.0];
        let scale = ndarray::array![[1e9, 1e9, 0.0], [0.0, 1e-9, 1e-9]];
        let scaled_rhs = ndarray::array![3.0e9, 5.0e-9];
        let plain = active_constraint_face_geometry(&a)
            .expect("plain geometry")
            .minimum_norm_particular(&rhs)
            .expect("plain particular");
        let rescaled = active_constraint_face_geometry(&scale)
            .expect("rescaled geometry")
            .minimum_norm_particular(&scaled_rhs)
            .expect("rescaled particular");
        for (left, right) in plain.delta.iter().zip(rescaled.delta.iter()) {
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "row rescaling moved the particular solution: {left:.17e} vs {right:.17e}"
            );
        }
    }
}

/// Reconstruct the *raw* Hessian `H = V · diag(σ) · Vᵀ` (pre-regularization)
/// from a `DenseSpectralOperator`. The operator stores
/// `r_ε(σ) = ½(σ + √(σ² + 4ε²))`; invert via `σ = r − ε²/r` so the tangent
/// projection `ZᵀHZ` sees the un-regularized data. The `from_symmetric`
/// call applied to that projection then performs a *single* tangent-space
/// regularization, matching `log|ZᵀHZ|` with one consistent `r_ε` instead
/// of double-regularizing (`r_ε(ZᵀV·r_ε(σ)·VᵀZ)`).
///
/// Per the math review (codex), projecting an already-regularized H_reg
/// and re-regularizing in tangent space is not exactly `log|ZᵀHZ|`; it is
/// a modified smoothed objective. Inverting `r_ε` first restores the
/// principled single-regularization identity.
pub(crate) fn assemble_h_raw_dense(op: &DenseSpectralOperator) -> Array2<f64> {
    let p = op.n_dim;
    // `ε = √ε_mach · p`. Same `spectral_epsilon` formula as the operator's
    // own construction; depends only on dim.
    let epsilon = f64::EPSILON.sqrt() * (p as f64).max(1.0);
    let eps_sq = epsilon * epsilon;
    if p == 0 {
        return Array2::<f64>::zeros((0, 0));
    }
    // Express `H = V · diag(σ_raw) · Vᵀ` as two BLAS3 matmuls (faer's
    // `fast_ab` / `fast_atb` are already parallelized internally),
    // replacing the previous triple-nested O(p³) loop.
    //
    //   sigma_j = r_j − ε²/r_j  for active, nonzero `r`; else 0.
    //   VS = V · diag(sigma)    (scale columns of V by sigma)
    //   H  = VS · Vᵀ            (= fast_abt(VS, V))
    let mut vs = op.eigenvectors.clone();
    for j in 0..p {
        let sigma = if op.active_mask[j] {
            let r = op.reg_eigenvalues[j];
            if r == 0.0 { 0.0 } else { r - eps_sq / r }
        } else {
            0.0
        };
        if sigma != 1.0 {
            let mut col = vs.column_mut(j);
            if sigma == 0.0 {
                col.fill(0.0);
            } else {
                col.mapv_inplace(|v| v * sigma);
            }
        }
    }
    // H = VS · Vᵀ without materializing Vᵀ.
    gam_linalg::faer_ndarray::fast_abt(&vs, &op.eigenvectors)
}

/// Tangent-projected `HessianFactorization` adapter. Wraps an `m × m`
/// `H_T = ZᵀHZ` operator and exposes the `p × p` interface needed by the
/// existing evaluator pipeline. All p-space inputs are projected via `Z`
/// before being passed to the tangent operator; outputs are lifted back
/// via `Z`. By construction this is the constraint-aware pseudo-inverse
/// `H⁺_T = Z (ZᵀHZ)⁻¹ Zᵀ`, which is bounded independent of σ_min(H)
/// when σ_min(ZᵀHZ) is bounded.
pub(crate) struct TangentProjectedHessianOperator {
    /// Orthonormal basis for null(A_act), `p × m`.
    pub(crate) z: Array2<f64>,
    /// `H_T = ZᵀHZ`, re-eigendecomposed with its own `r_ε` regularization.
    pub(crate) h_t_op: DenseSpectralOperator,
}

impl HessianFactorization for TangentProjectedHessianOperator {
    fn active_rank(&self) -> usize {
        self.h_t_op.active_rank()
    }

    /// `Z · U_T` over the tangent operator's active eigenpairs: the span `solve` lifts through `Z`
    /// (gam#2765).
    fn inverted_span(&self) -> Option<InvertedSpan> {
        let tangent = InvertedSpan::from_dense_spectral(&self.h_t_op)?;
        Some(InvertedSpan {
            basis: self.z.dot(&tangent.basis),
            eigenvalues: tangent.eigenvalues,
        })
    }

    fn dim(&self) -> usize {
        self.z.nrows()
    }
    fn logdet(&self) -> f64 {
        self.h_t_op.logdet()
    }
    /// `logdet` is `log|ZᵀHZ|` read off the tangent factor, so its forward error
    /// is that factor's (#3321).
    fn logdet_forward_error(&self) -> Option<f64> {
        self.h_t_op.logdet_forward_error()
    }
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve(&r_t);
        self.z.dot(&q_t)
    }
    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve_multi(&r_t);
        self.z.dot(&q_t)
    }
    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(Z H_T⁻¹ Zᵀ · A) = tr(H_T⁻¹ · ZᵀAZ) (cyclic permutation).
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_hinv_product(&zaz)
    }
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) · A) where H is the wrapped tangent operator.
        // d log|ZᵀHZ|/dt = tr((ZᵀHZ)⁻¹ · Zᵀ Ḣ Z) → use H_T's logdet kernel
        // applied to ZᵀḢZ.
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_logdet_gradient(&zaz)
    }
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        // Matrix-free tangent projection of an operator-backed Hessian drift.
        //
        // The `HessianFactorization` trait default densifies `op` (`op.to_dense()`,
        // p forward HVPs + a p×p transient) and then evaluates
        // `trace_logdet_gradient`, which internally forms `Zᵀ Bdense Z`, so it
        // unconditionally hits the warn-and-materialize branch — the dominant
        // source of `trace_logdet_operator: materializing implicit
        // HyperOperator` spam (and O(p²) work per outer eval per penalty) on
        // every shape-constrained REML fit.
        //
        // `Zᵀ B Z` is exactly `op.projected_matrix(Z) = Zᵀ (B·Z)`, where `B·Z`
        // is `op.mul_mat(Z)` — the operator's own matrix-free action, only
        // m ≤ p HVPs and no dense p×p B. The two are algebraically identical
        // (both have entry `z_iᵀ B z_j`), so this override changes only the
        // arithmetic path, never the value: `tr(G_ε(H_T) · ZᵀBZ)` via the
        // wrapped spectral logdet kernel.
        let zbz = op.projected_matrix(&self.z);
        self.h_t_op.trace_logdet_gradient(&zbz)
    }
    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        // tr(Z H_T⁻¹ Zᵀ · B) = tr(H_T⁻¹ · ZᵀBZ) (cyclic permutation), with `ZᵀBZ`
        // taken through the operator's own action exactly as in
        // `trace_logdet_operator`: m ≤ p HVPs and no dense p×p B. The trait default
        // would densify B; the value and logdet traces of one drift share H_T's
        // exact factor.
        let zbz = op.projected_matrix(&self.z);
        self.h_t_op.trace_hinv_product(&zbz)
    }
    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        // tr(H⁺_T A H⁺_T B) = tr(H_T⁻¹ · ZᵀAZ · H_T⁻¹ · ZᵀBZ) (cyclic permutation).
        let zaz = self.z.t().dot(matrix).dot(&self.z);
        let zbz = op.projected_matrix(&self.z);
        self.h_t_op.trace_hinv_product_cross(&zaz, &zbz)
    }
    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let zaz = left.projected_matrix(&self.z);
        if std::ptr::addr_eq(left, right) {
            return self.h_t_op.trace_hinv_product_cross(&zaz, &zaz);
        }
        let zbz = right.projected_matrix(&self.z);
        self.h_t_op.trace_hinv_product_cross(&zaz, &zbz)
    }
    fn is_dense(&self) -> bool {
        self.h_t_op.is_dense()
    }
    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        self.h_t_op.logdet_traces_match_hinv_kernel()
    }
    // Deliberately keep `as_dense_spectral` and `as_exact_dense_spectral`
    // at default `None`: their consumers expect a p-space spectral basis,
    // whereas the wrapped operator lives in m-dimensional tangent space.
    // Surfacing the tangent operator there would silently let downstream
    // code mix p- and m-dim eigenvectors.
}

/// Borrowing adapter that lets the constrained-response `InnerSolution` reuse
/// the original `HessianDerivativeProvider` without taking ownership. The
/// provider's drift matrices stay in the full coefficient space because the
/// LAML value and trace kernel stay there; only the mode-response vectors fed
/// into those drifts are tangent-restricted.
pub(crate) struct BorrowedDerivProvider<'a>(&'a dyn HessianDerivativeProvider);

impl<'a> HessianDerivativeProvider for BorrowedDerivProvider<'a> {
    fn mode_response_rhs_correction(&self) -> Option<ModeResponseRhsCorrectionFn> {
        self.0.mode_response_rhs_correction()
    }
    fn mode_response_rhs_correction_supplied(&self) -> bool {
        self.0.mode_response_rhs_correction_supplied()
    }
    fn hessian_derivative_correction(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_derivative_correction(v)
    }
    fn hessian_derivative_correction_result(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0.hessian_derivative_correction_result(v)
    }
    fn hessian_derivative_corrections_result(
        &self,
        vs: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_derivative_corrections_result(vs)
    }
    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_derivative_corrections()
    }
    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_second_derivative_correction(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_second_derivative_corrections_result(triples)
    }
    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_second_derivative_corrections()
    }
    fn has_corrections(&self) -> bool {
        self.0.has_corrections()
    }
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.0.outer_hessian_derivative_kernel()
    }
    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        self.0.family_outer_hessian_operator()
    }
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        self.0.scalar_glm_ingredients()
    }
}

/// A zero-dimensional inverse on the tangent of a fully pinned mode.
///
/// This operator is installed only as InnerSolution::mode_response_op; the
/// full-space Hessian remains the sole owner of the LAML value and traces.
/// Every response of a mode with null(A_act) = {0} is exactly zero.
struct FullyPinnedModeResponse {
    dimension: usize,
}

impl HessianFactorization for FullyPinnedModeResponse {
    fn logdet(&self) -> f64 {
        0.0
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), (self.dimension, self.dimension));
        0.0
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_eq!(rhs.len(), self.dimension);
        Array1::zeros(self.dimension)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        assert_eq!(rhs.nrows(), self.dimension);
        Array2::zeros(rhs.raw_dim())
    }

    fn dim(&self) -> usize {
        self.dimension
    }
    fn active_rank(&self) -> usize {
        0
    }
}

/// If the inner solution carries a non-empty active inequality-constraint
/// set, restrict the implicit response of its constrained mode to the face and
/// keep whatever value/trace geometry the solution's kernel carries.
///
/// A producer that installs a `penalty_subspace_trace` owns the criterion's
/// geometry. The custom-family projected route builds that kernel on the active
/// face, pricing 1/2 log|Z' M_true Z|+ with u_s = Z V (gam#2894, option A). That
/// supersedes 4c3c7f960's full-space value for constrained families: a
/// full-space pseudo-determinant drops an eigenvalue crossing zero off the face,
/// and its gradient never vanishes there. A solution without a kernel keeps the
/// full-space operator determinant.
///
/// Active geometry enters through the derivative of the constrained mode. With
/// Z an orthonormal basis of null(A_act),
///
/// d beta_hat / d theta = -Z (Z' M_true Z)^-1 Z' d g / d theta,
///
/// where M_true is the inner stationarity system (which may deliberately
/// differ from the log-determinant operator; #2612). The borrowed solution
/// therefore retains the solution's value/trace objects and installs only this
/// tangent-restricted mode-response operator. Clearing active_constraints on
/// it prevents recursion; the constraint's first-order effect is already
/// represented by the installed operator.
///
/// Returns Ok(None) when no active constraints are present and Ok(Some(result))
/// after evaluating the full-space criterion with the constrained response. A
/// backend that cannot materialize the true response curvature returns a named
/// error rather than silently differentiating through the log-determinant
/// curvature.
pub(crate) fn try_tangent_projected_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<Option<RemlLamlResult>, RemlLamlError> {
    let block = match solution.active_constraints.as_ref() {
        Some(block) if block.a.nrows() > 0 => block,
        _ => return Ok(None),
    };
    let p = solution.beta.len();
    if block.a.ncols() != p {
        return Err(RemlLamlError::Failed(format!(
            "active_constraints.a has {} columns but beta is {}-dim",
            block.a.ncols(),
            p
        )));
    }

    // The KKT gradient's motion the constrained Laplace term reads (gam#2765): on a face
    // `ġ = M_true β̂̇ + ∂_θ∇F` with the same stationarity curvature the mode response solves.
    let (constrained_mode_response, gradient_motion): (
        Arc<dyn HessianFactorization>,
        ConeGradientMotion,
    ) = match active_constraint_tangent_geometry(&block.a)? {
            ActiveConstraintTangentGeometry::FullyPinned => (
                Arc::new(FullyPinnedModeResponse { dimension: p }),
                ConeGradientMotion::Pinned,
            ),
            ActiveConstraintTangentGeometry::Tangent(z) => {
                // Differentiate the stationarity system the inner solve
                // actually used, not the operator that owns the Laplace
                // log-determinant. The two differ under a Jeffreys completion
                // (#2612).
                let response_full = solution
                    .mode_response_operator()
                    .assemble_h_dense_for_tangent_projection()
                    .map_err(|error| {
                        format!(
                            "active-constraint mode response needs a dense stationarity \
                             curvature: {error}"
                        )
                    })?;
                // #979: locate the smallest eigenvalue the criterion's
                // pseudo-log-determinant keeps relative to this face. A kept
                // direction normal to the face prices curvature the constrained
                // mode never explores; one inside the tangent disagrees with the
                // certified tangent curvature there.
                if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                    let inverse = &kernel.h_proj_inverse;
                    let rank = inverse.nrows();
                    let diagonal = (0..rank)
                        .all(|i| (0..rank).all(|j| i == j || inverse[[i, j]] == 0.0));
                    let shaped = inverse.ncols() == rank
                        && kernel.u_s.ncols() == rank
                        && kernel.u_s.nrows() == z.nrows();
                    let smallest = (0..rank).max_by(|&left, &right| {
                        inverse[[left, left]].total_cmp(&inverse[[right, right]])
                    });
                    if let (true, true, Some(column)) = (diagonal, shaped, smallest) {
                        let direction = kernel.u_s.column(column);
                        let tangent_part = z.t().dot(&direction);
                        // The value prices the log-determinant operator; the inner
                        // certificate prices the stationarity curvature. Report both
                        // spectra on the full space and on the face.
                        let smallest_eigenvalue = |matrix: &Array2<f64>| {
                            match DenseSpectralOperator::from_symmetric(matrix) {
                                Ok(operator) => Some(
                                    operator.raw_spectrum().iter().copied().fold(f64::INFINITY, f64::min),
                                ),
                                Err(error) => {
                                    log::debug!("[979-FACE-LOGDET] spectrum unavailable: {error}");
                                    None
                                }
                            }
                        };
                        let (value_min, value_tangent_min) =
                            match solution.hessian_op.assemble_h_dense_for_tangent_projection() {
                                Ok(matrix) => (
                                    smallest_eigenvalue(&matrix),
                                    smallest_eigenvalue(&z.t().dot(&matrix).dot(&z)),
                                ),
                                Err(error) => {
                                    log::debug!(
                                        "[979-FACE-LOGDET] log-determinant operator has no dense \
                                         assembly: {error}"
                                    );
                                    (None, None)
                                }
                            };
                        let true_min = smallest_eigenvalue(&response_full);
                        let true_tangent_min = smallest_eigenvalue(&z.t().dot(&response_full).dot(&z));
                        log::debug!(
                            "[979-FACE-LOGDET] kept_rank={rank}/{} tangent_dim={} \
                             sigma_min_kept={:.6e} normal_fraction={:.3e} \
                             value_min={value_min:?} value_tangent_min={value_tangent_min:?} \
                             true_min={true_min:?} true_tangent_min={true_tangent_min:?}",
                            z.nrows(),
                            z.ncols(),
                            1.0 / inverse[[column, column]],
                            direction.dot(&direction) - tangent_part.dot(&tangent_part),
                        );
                    }
                }
                let response_tangent = z.t().dot(&response_full).dot(&z);
                let response_tangent_op =
                    DenseSpectralOperator::from_symmetric(&response_tangent).map_err(|error| {
                        format!(
                            "constrained mode-response eigendecomposition failed: {error}"
                        )
                    })?;
                (
                    Arc::new(TangentProjectedHessianOperator {
                        z,
                        h_t_op: response_tangent_op,
                    }),
                    ConeGradientMotion::OnFace(Arc::new(response_full)),
                )
            }
        };

    let constrained = InnerSolution {
        log_likelihood: solution.log_likelihood,
        penalty_quadratic: solution.penalty_quadratic,
        // Value and trace geometry are the kernel's: on the face for the
        // custom-family projected route (gam#2894). Only the constrained mode
        // response is installed here.
        hessian_op: Arc::clone(&solution.hessian_op),
        mode_response_op: Some(constrained_mode_response),
        beta: solution.beta.clone(),
        penalty_coords: solution.penalty_coords.clone(),
        penalty_logdet: solution.penalty_logdet.clone(),
        deriv_provider: Box::new(BorrowedDerivProvider(solution.deriv_provider.as_ref())),
        firth: solution.firth.clone(),
        hessian_logdet_correction: solution.hessian_logdet_correction,
        penalty_subspace_trace: solution.penalty_subspace_trace.clone(),
        rho_curvature_scale: solution.rho_curvature_scale,
        rho_prior: solution.rho_prior.clone(),
        n_observations: solution.n_observations,
        nullspace_dim: solution.nullspace_dim,
        gaussian_weight_log_sum_half: solution.gaussian_weight_log_sum_half,
        dp_floor_scale: solution.dp_floor_scale,
        dispersion: solution.dispersion.clone(),
        ext_coords: solution.ext_coords.clone(),
        ext_coord_pair_fn: solution.ext_coord_pair_fn.clone(),
        rho_ext_pair_fn: solution.rho_ext_pair_fn.clone(),
        contracted_psi_second_order: solution.contracted_psi_second_order.clone(),
        fixed_drift_deriv: solution.fixed_drift_deriv.clone(),
        barrier_config: solution.barrier_config.clone(),
        kkt_residual: solution.kkt_residual.clone(),
        // Prevent recursive constrained-response installation. The operator
        // above already carries the active geometry.
        active_constraints: None,
        // The term is already priced; only the motion of the KKT gradient it reads is a
        // property of the active geometry resolved here (gam#2765).
        cone_normalizer: solution.cone_normalizer.as_ref().map(|term| {
            Arc::new(ConeNormalizerTerm {
                gradient_motion,
                laplace: term.laplace.clone(),
                profiled: term.profiled.clone(),
                // The constraint rows' own psi motion belongs to the cone, not to the
                // response geometry re-resolved here, so it rides unchanged (gam#3171).
                constraint_motion: term.constraint_motion.clone(),
            })
        }),
    };
    reml_laml_evaluate(&constrained, rho, mode, prior_cost_gradient).map(Some)
}

#[cfg(test)]
mod profiled_gaussian_residual_dof_tests {
    use super::profiled_gaussian_residual_dof;

    /// The positive control: the refusal this replaced `.max(DENOM_RIDGE)`
    /// with must actually fire, at the step of one in an integer where the old
    /// clamp used to take over.
    #[test]
    fn refuses_at_and_below_zero_residual_dof_and_accepts_one() {
        assert_eq!(
            profiled_gaussian_residual_dof(8, 7.0).expect("nu = 1 is a fittable model"),
            1.0
        );
        for nullspace_dim in [8.0, 9.0, 23.0] {
            let refusal = profiled_gaussian_residual_dof(8, nullspace_dim)
                .expect_err("nu <= 0 has no profiled scale and must refuse");
            assert!(
                refusal.contains("residual degrees of freedom must be positive"),
                "refusal must name the condition, got: {refusal}"
            );
        }
    }

    /// The old clamp's whole justification was "denominator safety", and the
    /// value it delivered at `nu <= 0` was `1e-8`, eight orders of magnitude
    /// below the `nu = 1` it neighbours across a step of one in an integer.
    /// Nothing may return a positive number in that regime again.
    #[test]
    fn no_fabricated_positive_denominator_below_one() {
        assert!(profiled_gaussian_residual_dof(0, 0.0).is_err());
        assert!(profiled_gaussian_residual_dof(3, 3.0).is_err());
        assert_eq!(
            profiled_gaussian_residual_dof(3, 2.0).expect("nu = 1"),
            1.0
        );
    }
}

#[cfg(test)]
mod penalty_hyper_operator_factor_tests {
    use super::penalty_coord_to_operator;
    use gam_problem::PenaltyCoordinate;
    use ndarray::{Array2, s};

    /// The penalty drift operator contracts a factor through its root with two
    /// GEMMs; each contraction must equal the dense `scale · S` it stands for,
    /// on a block root placed inside a wider coefficient vector (rows outside
    /// the block must come back exactly zero) and on a full-width root.
    #[test]
    fn root_factor_contractions_match_the_dense_penalty() {
        let (p, start, end, k) = (9usize, 2usize, 7usize, 4usize);
        let root = Array2::from_shape_fn((3, end - start), |(i, j)| {
            ((i * 7 + j * 3) % 5) as f64 - 1.5 + 0.1 * j as f64
        });
        let factor =
            Array2::from_shape_fn((p, k), |(i, j)| ((i * 5 + j * 11) % 7) as f64 * 0.3 - 0.8);
        let scale = 2.75;
        let block = PenaltyCoordinate::from_block_root(root.clone(), start, end, p);
        let mut full_root = Array2::<f64>::zeros((3, p));
        full_root.slice_mut(s![.., start..end]).assign(&root);
        let dense_coord = PenaltyCoordinate::from_dense_root(full_root);
        for coord in [block, dense_coord] {
            let dense = coord.scaled_dense_matrix(scale);
            let op = penalty_coord_to_operator(coord, scale);
            let expected_mul = dense.dot(&factor);
            let expected_projected = factor.t().dot(&expected_mul);
            let expected_trace: f64 = (0..k).map(|c| expected_projected[[c, c]]).sum();
            let tol = 1e-12 * dense.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            let mul = op.mul_mat(&factor);
            for (got, want) in mul.iter().zip(expected_mul.iter()) {
                assert!((got - want).abs() <= tol * 10.0, "mul_mat {got} vs {want}");
            }
            for row in (0..start).chain(end..p) {
                assert!(mul.row(row).iter().all(|&v| v == 0.0));
            }
            let projected = op.projected_matrix(&factor);
            for (got, want) in projected.iter().zip(expected_projected.iter()) {
                assert!((got - want).abs() <= tol * 100.0, "projected {got} vs {want}");
            }
            let trace = op.trace_projected_factor(&factor);
            assert!(
                (trace - expected_trace).abs() <= tol * 100.0,
                "trace {trace} vs {expected_trace}"
            );
        }
    }
}
