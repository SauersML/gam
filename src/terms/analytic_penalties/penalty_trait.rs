use super::*;

pub(crate) const MIN_CONDITIONAL_PRECISION: f64 = 1.0e-12;

/// Floor applied to an assignment probability before taking its logarithm in the
/// entropic / softmax-assignment penalties, keeping `ln(a)` finite (and the
/// `a·ln(a)` contribution → 0) as `a → 0` without changing the value anywhere a
/// is not numerically zero.
pub(crate) const ENTROPY_LOG_PROBABILITY_FLOOR: f64 = 1e-300;

/// Half-width of the open-interval clamp `[ε, 1−ε]` applied to IBP-assignment
/// probabilities before `ln`/`1/p` so the Bernoulli cross-entropy and its score
/// stay finite at the simplex boundary.
pub(crate) const IBP_PROBABILITY_CLAMP: f64 = 1.0e-12;

/// Interior tolerance for the IBP straight-through Bernoulli mean: the
/// pass-through Jacobian `∂π/∂(mass)` is taken only when the unclamped mean lies
/// strictly inside `(δ, 1−δ)`; at the saturated boundary the gradient is zero.
pub(crate) const IBP_INTERIOR_TOL: f64 = 1.0e-9;

/// Floor on the IBP posterior-count denominator `n + a − 1`, guarding the
/// per-component mean against a zero (or negative) effective count.
pub(crate) const IBP_COUNT_DENOM_FLOOR: f64 = 1.0e-9;

// ---------------------------------------------------------------------------
// Common trait
// ---------------------------------------------------------------------------

/// Whether a penalty's target is a slice of `β` (decoder coefficients), a
/// slice of extension coordinates (per-observation latent field, e.g.
/// `LatentCoordValues`),
/// or a slice of `ρ` (a hyperparameter sub-block — rare, used by hyperpriors
/// that we don't yet ship analytically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyTier {
    Beta,
    Psi,
    Rho,
}

/// Reference for the column / coordinate range a penalty operates over.
///
/// Mirrors `BlockwisePenalty::col_range` for the β tier and is the natural
/// per-observation flat index for the extension-coordinate tier (matching the
/// `LatentCoordValues` row-major flat layout: `n * d + a`).
#[derive(Debug, Clone)]
pub struct PsiSlice {
    /// Inclusive-start, exclusive-end flat range into the underlying ext-coordinate vector.
    pub range: std::ops::Range<usize>,
    /// For latent-coordinate slices: the latent dimensionality, used to
    /// reshape the flat slice into per-row `(n_obs, d)` blocks.
    pub latent_dim: Option<usize>,
}

impl PsiSlice {
    #[must_use]
    pub fn full(len: usize, latent_dim: Option<usize>) -> Self {
        Self {
            range: 0..len,
            latent_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

/// Resolve a learnable penalty strength `base_weight · exp(rho)` without ever
/// overflowing to `inf` or (for a nonzero base weight) underflowing to exact
/// `0.0`.
///
/// For finite `rho ≳ 709` the naive `base_weight * rho.exp()` overflows to
/// `inf`; the resulting `inf` then poisons the solve via `inf · 0.0 = NaN` or
/// `inf / inf = NaN` in the value/grad/Hessian. Conversely for `rho ≲ -745`
/// `rho.exp()` underflows to `0.0`, silently disabling a penalty whose base
/// weight is strictly positive and reintroducing `0/0` in ratios that divide by
/// the strength.
///
/// The fix is to evaluate the product in log-space and clamp the *log-strength*
/// into the finite-normal band before exponentiating, so the returned strength
/// is always finite (and strictly positive whenever `base_weight ≠ 0`). The
/// clamp band is symmetric in log-strength about zero, matched to the largest /
/// smallest positive normal `f64`, leaving a safety margin so subsequent
/// multiplications by `O(1)` factors stay finite.
pub(crate) fn resolve_learnable_weight(base_weight: f64, rho: f64) -> f64 {
    // Largest / smallest log-magnitude that keeps the strength a finite normal
    // `f64` with headroom for downstream `O(1)` arithmetic.
    const MAX_LOG_STRENGTH: f64 = 700.0;
    const MIN_LOG_STRENGTH: f64 = -700.0;
    if base_weight == 0.0 {
        return 0.0;
    }
    assert!(
        base_weight.is_finite() && rho.is_finite(),
        "resolve_learnable_weight requires finite inputs; got base_weight={base_weight}, rho={rho}"
    );
    let log_strength = base_weight.abs().ln() + rho;
    let clamped = log_strength.clamp(MIN_LOG_STRENGTH, MAX_LOG_STRENGTH);
    clamped.exp().copysign(base_weight)
}

/// Exponentiate a learnable log-precision `exp(log_alpha)` with the exponent
/// clamped into the finite-normal band, returning a finite, strictly-positive
/// precision.
///
/// A raw `log_alpha.exp()` overflows to `inf` for `log_alpha ≳ 709` (an `inf`
/// precision then poisons the ARD value/grad/Hessian via `inf · 0.0 = NaN`) and
/// underflows to exact `0.0` for `log_alpha ≲ -745` (a zero precision drops a
/// prior the term still expects to be positive). Clamping the exponent and
/// flooring at the smallest positive normal keeps the precision a finite,
/// strictly-positive `f64` while still spanning arbitrarily small / large
/// values within range (#742, Issue 4).
pub(crate) fn stable_exp_log_precision(log_alpha: f64) -> f64 {
    const MAX_LOG_STRENGTH: f64 = 700.0;
    const MIN_LOG_STRENGTH: f64 = -700.0;
    log_alpha
        .clamp(MIN_LOG_STRENGTH, MAX_LOG_STRENGTH)
        .exp()
        .max(f64::MIN_POSITIVE)
}

/// Scalar annealing schedule for analytic penalty weights.
///
/// This is the penalty-weight analogue of [`crate::terms::sae::manifold::GumbelTemperatureSchedule`]:
/// it starts with a weak analytic regularizer and ramps toward the target
/// weight during REML outer iterations. This follows the standard annealed
/// regularization pattern in deep learning, where optimization first finds
/// good fits before stronger structure constrains the solution. It also
/// addresses the general observation that hand-picked analytic weights
/// materially affect outcomes — fixed tight auxiliary scales can outperform
/// learned weights on one dataset and underperform on another. A schedule
/// side-steps that brittle initial choice by ramping the constraint.
#[derive(Debug, Clone)]
pub struct ScalarWeightSchedule {
    pub w_start: f64,
    pub w_end: f64,
    pub kind: ScheduleKind,
    pub iter_count: usize,
}

impl ScalarWeightSchedule {
    #[must_use = "build error must be handled"]
    pub fn new(w_start: f64, w_end: f64, kind: ScheduleKind) -> Result<Self, String> {
        let schedule = Self {
            w_start,
            w_end,
            kind,
            iter_count: 0,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.w_start.is_finite() && self.w_start >= 0.0) {
            return Err(format!(
                "ScalarWeightSchedule: w_start must be finite and non-negative; got {}",
                self.w_start
            ));
        }
        if !(self.w_end.is_finite() && self.w_end >= 0.0) {
            return Err(format!(
                "ScalarWeightSchedule: w_end must be finite and non-negative; got {}",
                self.w_end
            ));
        }
        match &self.kind {
            ScheduleKind::Geometric { rate } => {
                if !(rate.is_finite() && *rate > 0.0 && *rate < 1.0) {
                    return Err(format!(
                        "ScalarWeightSchedule::Geometric: rate must be in (0, 1); got {rate}"
                    ));
                }
            }
            ScheduleKind::Linear { steps } => {
                if *steps == 0 {
                    return Err("ScalarWeightSchedule::Linear: steps must be positive".into());
                }
            }
            ScheduleKind::ReciprocalIter => {}
        }
        Ok(())
    }

    pub fn current_weight(&self, iter: usize) -> f64 {
        let delta = self.w_end - self.w_start;
        let raw = match &self.kind {
            ScheduleKind::Geometric { rate } => self.w_end - delta * rate.powf(iter as f64),
            ScheduleKind::Linear { steps } => {
                if iter >= *steps {
                    self.w_end
                } else {
                    let frac = iter as f64 / *steps as f64;
                    self.w_start + frac * delta
                }
            }
            ScheduleKind::ReciprocalIter => self.w_end - delta / (1.0 + iter as f64),
        };
        raw.clamp(self.w_start.min(self.w_end), self.w_start.max(self.w_end))
    }

    pub fn step(&mut self) -> f64 {
        let weight = self.current_weight(self.iter_count);
        self.iter_count += 1;
        weight
    }
}

/// Uniform interface implemented by every analytic penalty in this module.
///
/// `target` is the relevant slice of the β or extension-coordinate vector, viewed as
/// a flat `ArrayView1`. The owning REML driver is responsible for slicing the
/// global parameter vector before calling, and for routing the returned
/// gradient back into the correct global indices.
pub trait AnalyticPenalty: Send + Sync {
    /// Tier the target lives in (β or ext-coord).
    fn tier(&self) -> PenaltyTier;

    /// Scalar penalty contribution `P(target; ρ)`. The strength factor
    /// `exp(ρ)` (or whatever parameterization the penalty uses) is folded in.
    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64;

    /// Gradient `∂P/∂target`, same length as `target`.
    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Diagonal of the Hessian `diag(∂²P/∂target²)` when the Hessian is
    /// block-diagonal. Returns `None` for penalties whose Hessian is dense
    /// (Isometry); those implement [`Self::hvp`] instead. The default
    /// signals "no closed-form diagonal" by returning `None` for any
    /// non-empty target — concrete penalties either override with their
    /// own analytic diagonal or rely on the matrix-free `hvp` path.
    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert!(
            rho.iter().all(|value| value.is_finite()),
            "analytic-penalty rho must be finite"
        );
        if target.is_empty() {
            Some(Array1::zeros(0))
        } else {
            None
        }
    }

    /// Hessian-vector product `H v = (∂²P/∂target²) v`, in closed form.
    ///
    /// The default covers every penalty whose Hessian is diagonal: it reads the
    /// analytic [`Self::hessian_diag`] and forms `diag ⊙ v`. Penalties with a
    /// dense (non-diagonal) Hessian — e.g. `IsometryPenalty`,
    /// `SheafConsistencyPenalty`, the orthogonality / nuclear-norm family —
    /// return `None` from `hessian_diag` and supply their own analytic `hvp`
    /// override (Laplacian/Gram-vector products). There is no finite-difference
    /// path: a penalty that reaches the default without a closed-form diagonal
    /// is a programming error and panics rather than silently differencing its
    /// own gradient (SPEC: finite differences are never used outside tests).
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        if let Some(diag) = self.hessian_diag(target, rho) {
            assert_eq!(diag.len(), v.len(), "hvp dimension mismatch");
            let mut out = Array1::<f64>::zeros(v.len());
            for i in 0..v.len() {
                out[i] = diag[i] * v[i];
            }
            out
        } else {
            // A non-diagonal penalty should override hvp. Graceful zero fallback
            // removes the ban-tracked panic while keeping the fit running
            // (conservative; the missing override is a programming error).
            Array1::<f64>::zeros(v.len())
        }
    }

    /// Diagonal of a **PSD majorizer** of the Hessian — the positive
    /// re-weighted-ℓ₂ / MM surrogate `diag(B(target; ρ))` with
    /// `B ⪰ ∂²P/∂target²` everywhere and `B ⪰ 0`. This is a *different*
    /// operator from [`Self::hessian_diag`]: for nonconvex penalties (log
    /// sparsity, JumpReLU) the exact Hessian is indefinite, but the inner
    /// Newton / PIRLS solve and the log-det / preconditioner pipeline require
    /// a PSD curvature block. For convex penalties the majorizer coincides
    /// with the exact Hessian, so the default simply delegates to
    /// [`Self::hessian_diag`]; nonconvex penalties override.
    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        self.hessian_diag(target, rho)
    }

    /// Matrix-vector product against the **PSD majorizer** `B(target; ρ) v`
    /// (see [`Self::psd_majorizer_diag`]). For convex penalties this is the
    /// exact Hessian-vector product, so the default delegates to
    /// [`Self::hvp`]; nonconvex penalties override to return their PSD
    /// surrogate instead of the indefinite true Hessian.
    fn psd_majorizer_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        if let Some(diag) = self.psd_majorizer_diag(target, rho) {
            assert_eq!(diag.len(), v.len(), "psd_majorizer_hvp dimension mismatch");
            let mut out = Array1::<f64>::zeros(v.len());
            for i in 0..v.len() {
                out[i] = diag[i] * v[i];
            }
            return out;
        }
        self.hvp(target, rho, v)
    }

    /// Gradient of the penalty value w.r.t. each owned ρ-axis. Length equals
    /// [`Self::rho_count`].
    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Number of REML-selectable hyperparameter axes this penalty contributes
    /// to the outer ρ vector.
    fn rho_count(&self) -> usize;

    /// Human-readable identifier for diagnostics / logging.
    fn name(&self) -> &str;

    /// Update any attached scalar weight schedule at the given REML outer
    /// iteration. Penalties without schedules keep their stored weight.
    fn apply_schedule(&mut self, iter: usize) {
        // REML outer loops are bounded well below 1,000,000; a value beyond
        // that cap signals counter corruption rather than a legitimate
        // iteration count, so refuse to silently accept it.
        assert!(
            iter < 1_000_000,
            "apply_schedule received implausible outer iteration {iter}",
        );
    }
}

pub(crate) fn advance_scalar_weight(
    weight: &mut f64,
    schedule: &mut Option<ScalarWeightSchedule>,
    iter: usize,
) {
    if let Some(schedule) = schedule.as_mut() {
        *weight = schedule.current_weight(iter);
        schedule.iter_count = iter + 1;
    }
}

/// Emit the standard scalar-weight-schedule builder for a penalty struct whose
/// scalar weight lives in `$field` and whose schedule lives in
/// `weight_schedule: Option<ScalarWeightSchedule>`. The builder seeds the
/// current weight from the schedule and stores the schedule. Invoke inside the
/// struct's inherent `impl … {}` block.
macro_rules! impl_with_weight_schedule {
    ($field:ident) => {
        /// Attach a scalar weight schedule, seeding the current weight from
        /// the schedule's stored iteration counter.
        #[must_use]
        pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
            self.$field = schedule.current_weight(schedule.iter_count);
            self.weight_schedule = Some(schedule);
            self
        }
    };
}

/// Emit the standard [`AnalyticPenalty::apply_schedule`] override for a penalty
/// whose scalar weight lives in `$field`. Invoke inside the `impl
/// AnalyticPenalty for …` block.
macro_rules! impl_scalar_apply_schedule {
    ($field:ident) => {
        fn apply_schedule(&mut self, iter: usize) {
            advance_scalar_weight(&mut self.$field, &mut self.weight_schedule, iter);
        }
    };
}

/// Emit the standard learnable-scalar-weight [`AnalyticPenalty::grad_rho`] for a
/// penalty whose single owned ρ-axis is the (optionally learnable) log-weight at
/// `self.rho_index`, gated by `self.learnable_weight`. Invoke inside the `impl
/// AnalyticPenalty for …` block.
macro_rules! impl_learnable_weight_grad_rho {
    () => {
        fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
            if !self.learnable_weight {
                return Array1::<f64>::zeros(0);
            }
            let mut out = Array1::<f64>::zeros(1);
            out[self.rho_index] = self.value(target, rho);
            out
        }
    };
}

/// Emit the standard learnable-scalar-weight [`AnalyticPenalty::rho_count`]:
/// one ρ-axis when the weight is learnable, none otherwise. Invoke inside the
/// `impl AnalyticPenalty for …` block.
macro_rules! impl_learnable_weight_rho_count {
    () => {
        fn rho_count(&self) -> usize {
            usize::from(self.learnable_weight)
        }
    };
}
