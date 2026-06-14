use faer::Side;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, CowArray, Ix2, Ix3};

use std::sync::{Arc, RwLock};


use crate::linalg::faer_ndarray::{FaerEigh, FaerSvd};

use crate::linalg::lanczos::{
    SymmetricLanczosOptions, symmetric_lanczos_eigenpairs, symmetric_lanczos_log_quadrature,
};

use crate::terms::basis::{BasisError, DuchonNullspaceOrder, radial_basis_cartesian_derivative};

use crate::terms::penalties::PenaltyManifest;

use crate::terms::penalty_op::PenaltyOp;

use crate::terms::sae_manifold::{GumbelTemperatureSchedule, ScheduleKind};

use crate::terms::sheaf::SheafConsistencyPenalty;

use crate::terms::smooth::BlockwisePenalty;


const MIN_CONDITIONAL_PRECISION: f64 = 1.0e-12;


/// Floor applied to an assignment probability before taking its logarithm in the
/// entropic / softmax-assignment penalties, keeping `ln(a)` finite (and the
/// `a·ln(a)` contribution → 0) as `a → 0` without changing the value anywhere a
/// is not numerically zero.
const ENTROPY_LOG_PROBABILITY_FLOOR: f64 = 1e-300;


/// Half-width of the open-interval clamp `[ε, 1−ε]` applied to IBP-assignment
/// probabilities before `ln`/`1/p` so the Bernoulli cross-entropy and its score
/// stay finite at the simplex boundary.
const IBP_PROBABILITY_CLAMP: f64 = 1.0e-12;


/// Interior tolerance for the IBP straight-through Bernoulli mean: the
/// pass-through Jacobian `∂π/∂(mass)` is taken only when the unclamped mean lies
/// strictly inside `(δ, 1−δ)`; at the saturated boundary the gradient is zero.
const IBP_INTERIOR_TOL: f64 = 1.0e-9;


/// Floor on the IBP posterior-count denominator `n + a − 1`, guarding the
/// per-component mean against a zero (or negative) effective count.
const IBP_COUNT_DENOM_FLOOR: f64 = 1.0e-9;


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
/// This is the penalty-weight analogue of [`crate::terms::sae_manifold::GumbelTemperatureSchedule`]:
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
        let diag = self.hessian_diag(target, rho).unwrap_or_else(|| {
            // SAFETY: programming-error invariant, never a runtime/data condition.
            // A penalty whose Hessian is non-diagonal MUST override `hvp` with its
            // closed-form Hessian-vector product; reaching this default means the
            // impl is missing that override. SPEC forbids a finite-difference
            // fallback outside tests, so there is no recoverable path — failing
            // loud here is the contract.
            panic!(
                "AnalyticPenalty::hvp default reached for `{}`, whose Hessian is \
                 not diagonal (hessian_diag returned None). Such a penalty must \
                 override `hvp` with its closed-form Hessian-vector product; the \
                 default never finite-differences.",
                self.name()
            )
        });
        assert_eq!(diag.len(), v.len(), "hvp dimension mismatch");
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
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


fn advance_scalar_weight(
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


// ---------------------------------------------------------------------------
// Isometry penalty
// ---------------------------------------------------------------------------

/// Choice of reference Riemannian metric `g^ref(t)` on the latent manifold.
///
/// `Euclidean` is the natural default: the reference metric is `I_d`, so the
/// penalty pulls the decoder toward locally-isometric (length-preserving)
/// behavior. `UserSupplied` lets the caller hand in a `(n_obs, d, d)` jet of
/// per-row reference metrics (useful for warm-starting from a chart of a
/// pre-fit GP-LVM).
#[derive(Clone)]
pub enum IsometryReference {
    Euclidean,
    UserSupplied(Arc<Array2<f64>>), // (n_obs, d*d) row-major flattened
}


impl std::fmt::Debug for IsometryReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsometryReference::Euclidean => f.write_str("Euclidean"),
            IsometryReference::UserSupplied(a) => f
                .debug_tuple("UserSupplied")
                .field(&format_args!("{}×{}", a.nrows(), a.ncols()))
                .finish(),
        }
    }
}


/// Per-observation behavioral-metric field `W_n ∈ ℝ^{p × p}`, stored in
/// **low-rank factored form** `W_n = U_n U_n^T` with `U_n ∈ ℝ^{p × r_n}`.
///
/// The canonical coordinate is the one where one unit of motion in `t` is one
/// unit of behavioral change in the output space, so the `W_n` weighting is
/// load-bearing: the pullback metric is `g_n = J_n^T W_n J_n`. Storing as
/// `U_n` lets every contraction in this module run in
/// `(J^T U_n)(U_n^T J)` order, which is `O(p · r · d + r · d²)` per row — we
/// **never** materialize the `p × p` `W_n`, which is essential when `p`
/// (number of observation channels) is large but rank is small (e.g. one or
/// two behavioral dimensions per latent observation).
///
/// `Identity` is the gauge-fix default and corresponds to `U_n = I_p` so the
/// pullback reduces to the standard `J_n^T J_n`. `Factored` stores the
/// per-row `U_n` blocks contiguously: every row's factor is `p × rank`, and
/// rows may share the same rank (uniform-rank case) or vary if the field is
/// data-driven. For the uniform-rank case the storage is
/// `(n_obs, p * rank)` row-major.
#[derive(Clone)]
pub enum WeightField {
    /// `W_n = I_p` for every `n`. Reduces to the bare pullback `J^T J`.
    Identity,
    /// Per-row low-rank factor `U_n ∈ ℝ^{p × rank}`. Storage layout: a
    /// `(n_obs, p * rank)` row-major matrix where row `n` packs `U_n` in
    /// column-major-within-row order `U_n[i, k] = u[n, i * rank + k]`.
    Factored {
        u: Arc<Array2<f64>>,
        rank: usize,
        p_out: usize,
    },
}


/// Radial Duchon decoder metadata used to materialize
/// `∂J_n[i, a] / ∂t_{n, c}` from `φ'(r)` and `φ''(r)` on demand.
///
/// `radial_coefficients[k, i]` is the decoder coefficient that maps radial
/// basis column `k` into output channel `i`. Polynomial-tail columns are not
/// represented here; callers whose decoder contains a non-linear polynomial
/// tail should provide `jacobian_second_cache` directly.
#[derive(Debug, Clone)]
pub struct IsometryDuchonRadialSource {
    pub centers: Arc<Array2<f64>>,
    pub radial_coefficients: Arc<Array2<f64>>,
    pub length_scale: Option<f64>,
    pub nullspace_order: DuchonNullspaceOrder,
    /// Forward hybrid spectral order `s = spec.power`. The Cartesian
    /// derivative engine must resolve the same `(p, s, κ)` the forward
    /// `build_duchon_basis` used, so it differentiates the exact resolved
    /// hybrid Green's function `φ_{p,s,κ}` rather than a hard-coded `s = 0`
    /// surrogate (issue #440).
    pub power: usize,
}


impl std::fmt::Debug for WeightField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightField::Identity => f.write_str("Identity"),
            WeightField::Factored { u, rank, p_out } => f
                .debug_struct("Factored")
                .field("shape", &format_args!("{}×{}", u.nrows(), u.ncols()))
                .field("rank", rank)
                .field("p_out", p_out)
                .finish(),
        }
    }
}


impl WeightField {
    /// Apply `U_n^T J_n` for a specific row, given both the row's `J_n` flat
    /// `(p * d)` slice and the row's `U_n` flat `(p * rank)` slice. Returns
    /// the `(rank × d)` matrix and its row count.
    fn project_jac_row_with_u(
        u_row: &[f64],
        jac_row: &[f64],
        p: usize,
        rank: usize,
        d: usize,
    ) -> Array2<f64> {
        // M[k, a] = Σ_i U[i, k] · J[i, a].
        let mut m = Array2::<f64>::zeros((rank, d));
        for k in 0..rank {
            for a in 0..d {
                let mut s = 0.0;
                for i in 0..p {
                    s += u_row[i * rank + k] * jac_row[i * d + a];
                }
                m[[k, a]] = s;
            }
        }
        m
    }
}


/// Isometry-to-reference penalty (canonical-coordinate gauge term).
///
/// Lives on ext-coords: the target slice is a row of the `LatentCoordValues` flat
/// vector (row-major `n_obs × d`). Owns one ρ-axis (`log μ_iso`).
///
/// Penalizes `½ μ Σ_n ‖g_n(t) − g^ref(t_n)‖²_F`, where the pullback metric
/// at row `n` is
///
/// ```text
///   g_n = J_n^T W_n J_n,    J_n ∈ ℝ^{p × d}
/// ```
///
/// and `W_n` is a per-row low-rank PSD behavioral metric stored as
/// `W_n = U_n U_n^T` with `U_n ∈ ℝ^{p × r}`. The canonical-coordinate
/// statement is "one unit of motion in `t` ↦ one unit of behavioral change",
/// so the `W_n` weighting is load-bearing.
///
/// In the SAE objective this is the extension-coordinate gauge fix: it prevents
/// the latent chart from absorbing arbitrary smooth reparameterizations of the
/// decoder manifold. ARD, sparsity, or rank penalties can then select axes or
/// structure in a chart whose metric scale is pinned.
///
/// **Contraction order invariant.** Every place this struct touches `W_n`,
/// the contraction is `(J^T U_n)(U_n^T J)` — never `J^T W_n J` with `W_n`
/// materialized as `p × p`. Concretely we form `M_n = U_n^T J_n ∈ ℝ^{r × d}`
/// once and then `g_n = M_n^T M_n` (`d × d`). Cost per row:
/// `O(p · r · d + r · d²)`, independent of `p²`.
///
/// **When to use.** Whenever a `LatentCoord` block is in play without an
/// auxiliary variable (`AuxPrior`) to break the diffeomorphism gauge. Fixes
/// the audit finding that ARD is not a standalone gauge fix. With a Euclidean
/// reference, the penalty pulls the decoder toward a local isometry, which is
/// enough to make the inner Hessian on `t` full-rank and the IFT well-defined.
///
/// **Math.** Let `J_n ∈ ℝ^{p × d}` be the local decoder Jacobian. Then
/// `g_n = J_n^T W_n J_n` and the penalty is
/// `½ μ Σ_n ‖J_n^T W_n J_n − g^ref_n‖²_F`. Analytic gradient w.r.t. `t_n`:
///
/// ```text
///   ∂P/∂t_{n,c}
///     = μ Σ_{a,b} (g_n − g^ref_n)_{ab}
///         [ H_{n,:,a,c}^T W_n J_{n,:,b}
///           + J_{n,:,a}^T W_n H_{n,:,b,c} ],
///   H_{n,i,a,c} = ∂J_{n,i,a}/∂t_{n,c}.
/// ```
///
/// Gotchas:
///
/// * The value path returns the configured missing-cache default when the
///   first-jet cache is absent; gradient/HVP paths need the first and second
///   decoder jets and return zeros when the analytic jet source is unavailable.
/// * The exact Hessian includes a residual-curvature term requiring the third
///   decoder jet. REML/PIRLS curvature should prefer the Gauss-Newton PSD
///   majorizer when a positive curvature block is required.
/// * `W_n` is a metric weight, not a scalar confidence. Changing it changes the
///   canonical units of latent motion.
///
/// The per-row Jacobian `J_n` is exactly the radial-derivative jet
/// `design_gradient_wrt_t` already computes for `LatentCoordValues`; the
/// second derivative `∂J/∂t` is built by the shared
/// [`crate::terms::basis::radial_basis_cartesian_derivative`] engine from the
/// radial Hessian identity. A finite-difference oracle for the docstring is
/// to central-difference `value(t ± h e_j)` against `grad_target(t)[j]`;
/// the analytic value follows the oracle until finite-difference
/// cancellation dominates. No autograd needed.
///
/// `μ = exp(ρ_iso)` is REML-selectable as one extra ρ axis.
///
/// `jacobian_cache_slot` and `jacobian_second_cache_slot` are interior-mutable
/// (`RwLock<Option<Arc<…>>>`) so the SAE outer loop can refresh them in place
/// each step without needing `&mut self` on the registry-held penalty (see
/// `refresh_caches` and [`crate::terms::sae_manifold::refresh_isometry_caches_from_atom`]).
/// Readers go through the [`Self::jacobian_cache`] / [`Self::jacobian_second_cache`]
/// accessors, which take the read lock briefly and clone the inner `Arc`
/// (refcount bump — no payload copy). Writers go through [`Self::refresh_caches`].
#[derive(Debug)]
pub struct IsometryPenalty {
    pub target: PsiSlice,
    pub reference: IsometryReference,
    /// Index of this penalty's strength `log μ_iso` inside the *local* rho
    /// view this penalty receives. Always `0` for now (single owned axis).
    pub rho_index: usize,
    /// Cached Jacobian `J ∈ ℝ^{n_obs × p × d}`, flattened row-major
    /// `(n_obs, p*d)`. The owning driver refreshes this each IFT outer step
    /// before invoking `value` / `grad_target`; in operator-only call sites
    /// (Hessian-vector products) the cache must be live. Access through
    /// [`Self::jacobian_cache`] / [`Self::set_jacobian_cache`].
    pub jacobian_cache_slot: RwLock<Option<Arc<Array2<f64>>>>,
    /// Optional cached per-row Jacobian *second derivative*
    /// `H_n ∈ ℝ^{p × d × d}`, flattened row-major as `(n_obs, p*d*d)`.
    /// `H_n[i, a, c] = ∂J_n[i, a] / ∂t_{n, c}`. Either this cache or
    /// `duchon_radial_source` must be present for exact isometry
    /// gradient/HVP calls. Access through [`Self::jacobian_second_cache`] /
    /// [`Self::set_jacobian_second_cache`].
    pub jacobian_second_cache_slot: RwLock<Option<Arc<Array2<f64>>>>,
    /// Optional radial-Duchon source used to build `jacobian_second_cache`
    /// analytically from `φ'(r)` and the public `φ''(r)` jet helper. This is
    /// the exact chain-rule path for callers that do not pre-cache `∂J/∂t`.
    pub duchon_radial_source: Option<Arc<IsometryDuchonRadialSource>>,
    /// Optional cached per-row Jacobian *third derivative*
    /// `K_n ∈ ℝ^{p × d × d × d}`, stored as an `Array3` with shape
    /// `(n_obs, p, d * d * d)` where the third axis packs `(a, c, d)` in
    /// row-major order `((a * d) + c) * d + dd`. `hvp` uses the full
    /// residual-curvature Hessian (proposal §4(b)):
    ///   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
    ///             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd}.
    /// Either this cache or `duchon_radial_source` must be present for
    /// analytic `hvp` calls. Interior-mutable (mirrors
    /// `jacobian_second_cache_slot`) so the SAE outer loop can refresh `K` in
    /// place each step. Access through [`Self::third_decoder_derivative`] /
    /// [`Self::set_third_decoder_derivative`].
    pub third_decoder_derivative_slot: RwLock<Option<Arc<ndarray::Array3<f64>>>>,
    /// Output dimensionality `p` (column count of each per-row Jacobian).
    pub p_out: usize,
    /// Per-row behavioral metric in low-rank factored form. Defaults to
    /// `Identity` (the unweighted `J^T J` pullback). When `Factored`, all
    /// `g_n` contractions are done via `M_n = U_n^T J_n` (`r × d`), keeping
    /// memory and FLOPs scaling at `O(p · r · d)` per row instead of
    /// `O(p²)` per row.
    pub weight: WeightField,
    pub scalar_weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


struct IsometryHvpState<'a> {
    d: usize,
    n_obs: usize,
    p: usize,
    jac2: CowArray<'a, f64, Ix2>,
    jac3: CowArray<'a, f64, Ix3>,
    metric: IsometryMetricState,
    wj_rows: Vec<Array2<f64>>,
}


#[derive(Debug, Clone)]
struct IsometryMetricState {
    g: Array2<f64>,
    residual: Array2<f64>,
    metric_grad: Array2<f64>,
    normalizer: f64,
    trace_denominator: f64,
    residual_dot_g: f64,
}


impl IsometryMetricState {
    fn residual_direction(&self, delta_g: ArrayView2<'_, f64>, d: usize) -> (Array2<f64>, f64) {
        let n_obs = self.g.nrows();
        let dd = d * d;
        let mut delta_trace_sum = 0.0;
        for n in 0..n_obs {
            for a in 0..d {
                delta_trace_sum += delta_g[[n, a * d + a]];
            }
        }
        let delta_normalizer = delta_trace_sum / self.trace_denominator;
        let inv_norm = 1.0 / self.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        let mut delta_residual = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for k in 0..dd {
                delta_residual[[n, k]] =
                    delta_g[[n, k]] * inv_norm - self.g[[n, k]] * delta_normalizer * inv_norm_sq;
            }
        }
        (delta_residual, delta_normalizer)
    }

    fn metric_grad_direction(&self, delta_g: ArrayView2<'_, f64>, d: usize) -> Array2<f64> {
        let n_obs = self.g.nrows();
        let dd = d * d;
        let (delta_residual, delta_normalizer) = self.residual_direction(delta_g, d);
        let mut delta_residual_dot_g = 0.0;
        for n in 0..n_obs {
            for k in 0..dd {
                delta_residual_dot_g += delta_residual[[n, k]] * self.g[[n, k]];
                delta_residual_dot_g += self.residual[[n, k]] * delta_g[[n, k]];
            }
        }
        let inv_norm = 1.0 / self.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        let delta_trace_coeff = delta_residual_dot_g * inv_norm_sq / self.trace_denominator
            - 2.0 * self.residual_dot_g * delta_normalizer * inv_norm_sq * inv_norm
                / self.trace_denominator;
        let mut out = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for a in 0..d {
                for b in 0..d {
                    let k = a * d + b;
                    let mut value = delta_residual[[n, k]] * inv_norm
                        - self.residual[[n, k]] * delta_normalizer * inv_norm_sq;
                    if a == b {
                        value -= delta_trace_coeff;
                    }
                    out[[n, k]] = value;
                }
            }
        }
        out
    }
}


fn isometry_dg_entry(
    jac2: ArrayView2<'_, f64>,
    wj: ArrayView2<'_, f64>,
    n: usize,
    d: usize,
    p: usize,
    a: usize,
    b: usize,
    c: usize,
) -> f64 {
    let mut s = 0.0;
    for i in 0..p {
        s += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
        s += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
    }
    s
}


fn isometry_row_delta_g(
    jac2: ArrayView2<'_, f64>,
    wj: ArrayView2<'_, f64>,
    v: ArrayView1<'_, f64>,
    n: usize,
    d: usize,
    p: usize,
) -> Array2<f64> {
    let mut delta_g = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            let mut s = 0.0;
            for c in 0..d {
                s += isometry_dg_entry(jac2, wj, n, d, p, a, b, c) * v[n * d + c];
            }
            delta_g[[a, b]] = s;
        }
    }
    delta_g
}


impl IsometryPenalty {
    pub const DEFAULT_VALUE_ON_MISSING_CACHE: f64 = 0.0;

    #[must_use]
    pub fn new_euclidean(target: PsiSlice, p_out: usize) -> Self {
        Self {
            target,
            reference: IsometryReference::Euclidean,
            rho_index: 0,
            jacobian_cache_slot: RwLock::new(None),
            jacobian_second_cache_slot: RwLock::new(None),
            duchon_radial_source: None,
            third_decoder_derivative_slot: RwLock::new(None),
            p_out,
            weight: WeightField::Identity,
            scalar_weight: 1.0,
            weight_schedule: None,
        }
    }

    /// Read-side accessor: takes the read lock briefly and clones the inner
    /// `Arc` (refcount bump only; no payload copy). Returns `None` when the
    /// cache has not been refreshed yet. Internally panics on poisoned lock
    /// — the lock only wraps an `Option<Arc<…>>`, so the write side cannot
    /// leave it in an invariant-violating state.
    #[must_use]
    pub fn jacobian_cache(&self) -> Option<Arc<Array2<f64>>> {
        self.jacobian_cache_slot
            .read()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned")
            .clone()
    }

    /// Read-side accessor for the per-row Jacobian second derivative.
    /// Mirrors [`Self::jacobian_cache`].
    #[must_use]
    pub fn jacobian_second_cache(&self) -> Option<Arc<Array2<f64>>> {
        self.jacobian_second_cache_slot
            .read()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned")
            .clone()
    }

    /// Per-step refresh entry point. Takes `&self` (no `&mut`) so the SAE
    /// outer loop can install fresh caches on an `Arc<IsometryPenalty>` held
    /// in the analytic-penalty registry without disturbing the surrounding
    /// dispatcher. Pass `None` for either argument to clear that cache (the
    /// dispatcher will then either fall back to the Duchon radial source if
    /// available, or return the zero safe default).
    pub fn refresh_caches(&self, jac: Option<Arc<Array2<f64>>>, jac2: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned") = jac;
        *self
            .jacobian_second_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned") = jac2;
    }

    /// In-place writer for just the Jacobian cache (used by callers that
    /// already own the radial Duchon source and only want to refresh `J`).
    pub fn set_jacobian_cache(&self, jac: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_cache_slot poisoned") = jac;
    }

    /// In-place writer for just the Jacobian second-derivative cache.
    pub fn set_jacobian_second_cache(&self, jac2: Option<Arc<Array2<f64>>>) {
        *self
            .jacobian_second_cache_slot
            .write()
            .expect("IsometryPenalty::jacobian_second_cache_slot poisoned") = jac2;
    }

    /// Read-side accessor for the per-row Jacobian third derivative `K`.
    /// Mirrors [`Self::jacobian_second_cache`].
    #[must_use]
    pub fn third_decoder_derivative(&self) -> Option<Arc<ndarray::Array3<f64>>> {
        self.third_decoder_derivative_slot
            .read()
            .expect("IsometryPenalty::third_decoder_derivative_slot poisoned")
            .clone()
    }

    /// In-place writer for just the Jacobian third-derivative cache `K`.
    pub fn set_third_decoder_derivative(&self, jac3: Option<Arc<ndarray::Array3<f64>>>) {
        *self
            .third_decoder_derivative_slot
            .write()
            .expect("IsometryPenalty::third_decoder_derivative_slot poisoned") = jac3;
    }
}


impl Clone for IsometryPenalty {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            reference: self.reference.clone(),
            rho_index: self.rho_index,
            jacobian_cache_slot: RwLock::new(self.jacobian_cache()),
            jacobian_second_cache_slot: RwLock::new(self.jacobian_second_cache()),
            duchon_radial_source: self.duchon_radial_source.clone(),
            third_decoder_derivative_slot: RwLock::new(self.third_decoder_derivative()),
            p_out: self.p_out,
            weight: self.weight.clone(),
            scalar_weight: self.scalar_weight,
            weight_schedule: self.weight_schedule.clone(),
        }
    }
}


impl IsometryPenalty {
    /// Attach a cached third decoder derivative
    /// `K_n[i, a, c, d] = ∂²J_n[i, a] / ∂t_{n, c} ∂t_{n, d}`, flattened
    /// row-major as `(n_obs, p * d * d * d)`. The Hessian-vector product
    /// uses the full residual-curvature term in addition to the metric
    /// Gauss-Newton piece.
    #[must_use]
    pub fn with_third_decoder_derivative(self, k: Arc<ndarray::Array3<f64>>) -> Self {
        self.set_third_decoder_derivative(Some(k));
        self
    }

    #[must_use]
    pub fn with_reference(mut self, reference: IsometryReference) -> Self {
        self.reference = reference;
        self
    }

    #[must_use]
    pub fn with_jacobian_cache(self, j: Arc<Array2<f64>>) -> Self {
        self.set_jacobian_cache(Some(j));
        self
    }

    #[must_use]
    pub fn with_jacobian_second_cache(self, h: Arc<Array2<f64>>) -> Self {
        self.set_jacobian_second_cache(Some(h));
        self
    }

    /// Attach radial Duchon decoder metadata so the exact `∂J/∂t` tensor can
    /// be rebuilt from the current target coordinates. A doc-test oracle for
    /// this path is: build `J(t)` from `duchon_radial_first_derivative_nd`,
    /// evaluate `grad_target(t)`, then central-difference `value(t ± h e_j)`;
    /// the analytic component should agree to finite-difference tolerance as
    /// `h` is refined before cancellation dominates.
    #[must_use]
    pub fn with_duchon_radial_source(mut self, source: Arc<IsometryDuchonRadialSource>) -> Self {
        self.duchon_radial_source = Some(source);
        self
    }

    /// Attach the gauge metric **from the single
    /// [`RowMetric`](crate::inference::row_metric::RowMetric)** that also drives
    /// the reconstruction likelihood. This is the only way an `IsometryPenalty`
    /// acquires a non-identity behavioral metric: the independent
    /// `WeightField` setter has been removed so a gauge-metric ≠
    /// likelihood-metric state is structurally unrepresentable. The
    /// contraction-order invariant (`M_n = U_n^T J_n`, never materializing the
    /// `p × p` `W_n`) is preserved by the [`WeightField::Factored`] layout the
    /// metric emits.
    ///
    /// `p_out` is taken from the metric so the gauge's output dimension is
    /// pinned to the metric's.
    #[must_use]
    pub fn with_row_metric(mut self, metric: &crate::inference::row_metric::RowMetric) -> Self {
        // Only a metric that drives the gauge installs a non-identity pullback
        // weight. A Euclidean metric reduces the gauge pullback to the bare
        // `J_nᵀ J_n`, so its `to_weight_field()` is `Identity` and the existing
        // (default-Identity) weight is left exactly as is — bit-for-bit the
        // pre-metric isotropic gauge. The output dimension is pinned to the
        // metric's regardless, so the gauge and likelihood agree on `p_out`.
        if metric.drives_gauge() {
            self.weight = metric.to_weight_field();
        }
        self.p_out = metric.p_out();
        self
    }

    impl_with_weight_schedule!(scalar_weight);

    fn missing_cache_default(&self, method: &str, detail: &str) {
        log::warn!(
            "IsometryPenalty::{method} missing required derivative state: {detail}; \
             returning the zero safe default"
        );
    }

    fn has_jacobian_cache(&self, method: &str) -> bool {
        if self.jacobian_cache().is_some() {
            true
        } else {
            self.missing_cache_default(method, "jacobian_cache is None");
            false
        }
    }

    fn has_jacobian_second_source(&self, method: &str) -> bool {
        if self.jacobian_second_cache().is_some() || self.duchon_radial_source.is_some() {
            true
        } else {
            self.missing_cache_default(
                method,
                "both jacobian_second_cache and duchon_radial_source are None",
            );
            false
        }
    }

    fn has_jacobian_third_source(&self, method: &str) -> bool {
        if self.third_decoder_derivative().is_some() || self.duchon_radial_source.is_some() {
            true
        } else {
            self.missing_cache_default(
                method,
                "both third_decoder_derivative cache and duchon_radial_source are None",
            );
            false
        }
    }

    /// Build `M_n = U_n^T J_n ∈ ℝ^{r_n × d}` for row `n`. For
    /// `WeightField::Identity`, `r_n = p` and `M_n = J_n`.
    ///
    /// This is the single contraction site where `W_n` (or its `U_n` factor)
    /// is consumed. Every value/grad/hvp path funnels through here, so the
    /// `(J^T U)(U^T J)` ordering invariant cannot be violated by accident.
    fn projected_jacobian_row(&self, n: usize, d: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("projected_jacobian_row", "jacobian_cache is None");
            return None;
        };
        let jac_row = jac.row(n);
        let jac_slice = jac_row
            .as_slice()
            .expect("jacobian cache must be in standard row-major layout");
        match &self.weight {
            WeightField::Identity => {
                let p = self.p_out;
                let mut m = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        m[[i, a]] = jac_slice[i * d + a];
                    }
                }
                Some(m)
            }
            WeightField::Factored { u, rank, p_out } => {
                let u_row = u.row(n);
                let u_slice = u_row
                    .as_slice()
                    .expect("weight factor U must be in standard row-major layout");
                Some(WeightField::project_jac_row_with_u(
                    u_slice, jac_slice, *p_out, *rank, d,
                ))
            }
        }
    }

    /// Form `W_n J_n` without materializing `W_n`.
    fn weighted_jacobian_row(&self, n: usize, d: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("weighted_jacobian_row", "jacobian_cache is None");
            return None;
        };
        let p = self.p_out;
        match &self.weight {
            WeightField::Identity => {
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        out[[i, a]] = jac[[n, i * d + a]];
                    }
                }
                Some(out)
            }
            WeightField::Factored { u, rank, p_out } => {
                assert_eq!(p, *p_out);
                let r = *rank;
                let m_n = self.projected_jacobian_row(n, d)?;
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        let mut s = 0.0;
                        for k in 0..r {
                            s += u[[n, i * r + k]] * m_n[[k, a]];
                        }
                        out[[i, a]] = s;
                    }
                }
                Some(out)
            }
        }
    }

    fn weighted_dot_decoder_vectors<F, G>(&self, n: usize, p: usize, x: F, y: G) -> f64
    where
        F: Fn(usize) -> f64,
        G: Fn(usize) -> f64,
    {
        match &self.weight {
            WeightField::Identity => {
                let mut s = 0.0;
                for i in 0..p {
                    s += x(i) * y(i);
                }
                s
            }
            WeightField::Factored { u, rank, p_out } => {
                assert_eq!(p, *p_out);
                let r = *rank;
                let mut s = 0.0;
                for k in 0..r {
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    for i in 0..p {
                        let uik = u[[n, i * r + k]];
                        ux += uik * x(i);
                        uy += uik * y(i);
                    }
                    s += ux * uy;
                }
                s
            }
        }
    }

    fn target_matrix(target: ArrayView1<'_, f64>, n_obs: usize, d: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for a in 0..d {
                out[[n, a]] = target[n * d + a];
            }
        }
        out
    }

    /// Second-order input-location derivative tensor of the Duchon decoder,
    /// flattened to `(n_obs, p_out · d²)` with column layout
    /// `i·d² + (a·d + c)`.
    ///
    /// Thin adapter over the shared [`radial_basis_cartesian_derivative`]
    /// engine: it owns the radial-jet evaluation and the radial→Cartesian map;
    /// here we only forward the source geometry.
    fn duchon_radial_jacobian_second(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<Array2<f64>, BasisError> {
        assert_eq!(source.centers.ncols(), d);
        assert_eq!(source.radial_coefficients.nrows(), source.centers.nrows());
        assert_eq!(source.radial_coefficients.ncols(), self.p_out);
        let t = Self::target_matrix(target, n_obs, d);
        radial_basis_cartesian_derivative(
            2,
            t.view(),
            source.centers.view(),
            source.radial_coefficients.view(),
            source.length_scale,
            source.nullspace_order,
            source.power,
        )
    }

    /// Third-order input-location derivative tensor of the Duchon decoder,
    /// shaped `(n_obs, p_out, d³)` with last-axis layout `(a·d + c)·d + e`.
    ///
    /// Thin adapter over the shared [`radial_basis_cartesian_derivative`]
    /// engine; the flat `(n_obs, p_out · d³)` result is reshaped to the
    /// `Array3` consumed by the HVP path (row-major flatten of `(p_out, d³)`
    /// is exactly `i·d³ + idx`).
    fn duchon_radial_jacobian_third(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<ndarray::Array3<f64>, BasisError> {
        assert_eq!(source.centers.ncols(), d);
        assert_eq!(source.radial_coefficients.nrows(), source.centers.nrows());
        assert_eq!(source.radial_coefficients.ncols(), self.p_out);
        let t = Self::target_matrix(target, n_obs, d);
        let flat = radial_basis_cartesian_derivative(
            3,
            t.view(),
            source.centers.view(),
            source.radial_coefficients.view(),
            source.length_scale,
            source.nullspace_order,
            source.power,
        )?;
        Ok(flat
            .into_shape_with_order((n_obs, self.p_out, d * d * d))
            .expect("radial_basis_cartesian_derivative order-3 output reshapes to (n_obs, p, d³)"))
    }

    fn jacobian_second<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<CowArray<'a, f64, Ix2>> {
        if let Some(jac2) = self.jacobian_second_cache() {
            // Clone the underlying Array2 to detach from the Arc — the
            // CowArray needs to outlive the temporary Arc returned by the
            // accessor. The clone is `n_obs × p·d²` floats, paid once per
            // grad_target / hvp_state invocation; same per-step cost as the
            // pre-refactor code path which also took ownership via
            // `jac2.view().to_owned()` semantics implicitly.
            return Some(CowArray::from((*jac2).clone()));
        }
        let source = self.duchon_radial_source.as_ref()?;
        match self.duchon_radial_jacobian_second(target, n_obs, d, source) {
            Ok(jac2) => Some(CowArray::from(jac2)),
            Err(err) => {
                self.missing_cache_default(
                    "jacobian_second",
                    &format!("failed to materialize Duchon radial second derivative: {err}"),
                );
                None
            }
        }
    }

    fn jacobian_third<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<CowArray<'a, f64, Ix3>> {
        if let Some(jac3) = self.third_decoder_derivative() {
            return Some(CowArray::from(jac3.as_ref().clone()));
        }
        let source = self.duchon_radial_source.as_ref()?;
        match self.duchon_radial_jacobian_third(target, n_obs, d, source) {
            Ok(jac3) => Some(CowArray::from(jac3)),
            Err(err) => {
                self.missing_cache_default(
                    "jacobian_third",
                    &format!("failed to materialize Duchon radial third derivative: {err}"),
                );
                None
            }
        }
    }

    fn hvp_state<'a>(&'a self, target: ArrayView1<'_, f64>) -> Option<IsometryHvpState<'a>> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("hvp")
            || !self.has_jacobian_second_source("hvp")
            || !self.has_jacobian_third_source("hvp")
        {
            return None;
        }
        let p = self.p_out;
        let jac2 = self.jacobian_second(target.view(), n_obs, d)?;
        let jac3 = self.jacobian_third(target.view(), n_obs, d)?;
        let g = self.pullback_metric(d)?;
        let metric = self.normalized_metric_state(g, n_obs, d)?;
        let mut wj_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            wj_rows.push(self.weighted_jacobian_row(n, d)?);
        }
        Some(IsometryHvpState {
            d,
            n_obs,
            p,
            jac2,
            jac3,
            metric,
            wj_rows,
        })
    }

    fn hvp_with_precomputed_state(
        &self,
        state: &IsometryHvpState<'_>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let d = state.d;
        let n_obs = state.n_obs;
        let p = state.p;
        let jac2 = &state.jac2;
        let jac3 = &state.jac3;
        let metric = &state.metric;
        let mut out = Array1::<f64>::zeros(v.len());
        let mut delta_g = Array2::<f64>::zeros((n_obs, d * d));
        for n in 0..n_obs {
            let wj = &state.wj_rows[n];
            let row_delta = isometry_row_delta_g(jac2.view(), wj.view(), v, n, d, p);
            for a in 0..d {
                for b in 0..d {
                    delta_g[[n, a * d + b]] = row_delta[[a, b]];
                }
            }
        }
        let delta_metric_grad = metric.metric_grad_direction(delta_g.view(), d);

        for n in 0..n_obs {
            let wj = &state.wj_rows[n];
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let dg = isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, b, c);
                        acc += dg * delta_metric_grad[[n, a * d + b]];
                    }
                }
                out[n * d + c] = mu * acc;
            }

            for c in 0..d {
                let mut acc_res = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let metric_grad = metric.metric_grad[[n, a * d + b]];
                        if metric_grad == 0.0 {
                            continue;
                        }
                        let mut bv = 0.0;
                        for dd in 0..d {
                            let vd = v[n * d + dd];
                            if vd == 0.0 {
                                continue;
                            }
                            let mut k_a_cd_w_j_b = 0.0;
                            for i in 0..p {
                                k_a_cd_w_j_b += jac3[[n, i, ((a * d) + c) * d + dd]] * wj[[i, b]];
                            }
                            let h_a_c_w_h_b_d = self.weighted_dot_decoder_vectors(
                                n,
                                p,
                                |i| jac2[[n, (i * d + a) * d + c]],
                                |i| jac2[[n, (i * d + b) * d + dd]],
                            );
                            let h_a_d_w_h_b_c = self.weighted_dot_decoder_vectors(
                                n,
                                p,
                                |i| jac2[[n, (i * d + a) * d + dd]],
                                |i| jac2[[n, (i * d + b) * d + c]],
                            );
                            let mut j_a_w_k_b_cd = 0.0;
                            for i in 0..p {
                                j_a_w_k_b_cd += wj[[i, a]] * jac3[[n, i, ((b * d) + c) * d + dd]];
                            }
                            bv +=
                                (k_a_cd_w_j_b + h_a_c_w_h_b_d + h_a_d_w_h_b_c + j_a_w_k_b_cd) * vd;
                        }
                        acc_res += metric_grad * bv;
                    }
                }
                out[n * d + c] += mu * acc_res;
            }
        }
        out
    }

    /// Per-row pullback metric `g_n = J_n^T W_n J_n = M_n^T M_n` with
    /// `M_n = U_n^T J_n ∈ ℝ^{r_n × d}`. Returns `(n_obs, d, d)` flattened
    /// row-major as `(n_obs, d*d)`.
    ///
    /// Cost per row: `O(p · r · d)` for the `M_n` build (single pass over
    /// `U_n` and `J_n`) plus `O(r · d²)` for `M_n^T M_n`. The `p × p` weight
    /// `W_n` is never materialized.
    pub(crate) fn pullback_metric(&self, latent_dim: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache() else {
            self.missing_cache_default("pullback_metric", "jacobian_cache is None");
            return None;
        };
        let n_obs = jac.nrows();
        let p = self.p_out;
        assert_eq!(jac.ncols(), p * latent_dim);
        let mut g_all = Array2::<f64>::zeros((n_obs, latent_dim * latent_dim));
        for n in 0..n_obs {
            // M_n = U_n^T J_n  (or J_n itself when W = I).
            let m = self.projected_jacobian_row(n, latent_dim)?;
            let r = m.nrows();
            // g_n = M_n^T M_n: (d × d) result, contracting r.
            for a in 0..latent_dim {
                for b in 0..latent_dim {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += m[[k, a]] * m[[k, b]];
                    }
                    g_all[[n, a * latent_dim + b]] = s;
                }
            }
        }
        Some(g_all)
    }

    /// Reference metric per row for the normalized pullback metric, `(n_obs, d*d)`.
    fn reference_metric(&self, n_obs: usize, d: usize) -> CowArray<'_, f64, Ix2> {
        match &self.reference {
            IsometryReference::Euclidean => {
                let mut out = Array2::<f64>::zeros((n_obs, d * d));
                for n in 0..n_obs {
                    for a in 0..d {
                        out[[n, a * d + a]] = 1.0;
                    }
                }
                CowArray::from(out)
            }
            IsometryReference::UserSupplied(a) => {
                assert_eq!(a.nrows(), n_obs);
                assert_eq!(a.ncols(), d * d);
                CowArray::from(a.view())
            }
        }
    }

    /// Shared normalized metric state for the scale-invariant isometry gauge.
    ///
    /// The residual is `R_n = g_n / gbar - g_ref,n`, with
    /// `gbar = (1 / (N d)) Σ_n tr(g_n)`. The metric-gradient is the exact
    /// derivative of `0.5 Σ ||R_n||²` with respect to the raw pullback metrics:
    ///
    /// `A_n = R_n / gbar - (Σ_l R_l:g_l) I / (gbar² N d)`.
    ///
    /// All value, gradient, and HVP paths consume this state so the global
    /// normalizer's derivative is never detached.
    fn normalized_metric_state(
        &self,
        g: Array2<f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<IsometryMetricState> {
        let dd = d * d;
        let trace_denominator = (n_obs * d) as f64;
        let mut trace_sum = 0.0;
        for n in 0..n_obs {
            for a in 0..d {
                trace_sum += g[[n, a * d + a]];
            }
        }
        let normalizer = trace_sum / trace_denominator;
        if !(normalizer.is_finite() && normalizer > f64::MIN_POSITIVE) {
            self.missing_cache_default(
                "normalized_metric_state",
                &format!(
                    "unit-average-speed normalizer is non-positive or non-finite: {normalizer}"
                ),
            );
            return None;
        }
        let g_ref = self.reference_metric(n_obs, d);
        let mut residual = Array2::<f64>::zeros((n_obs, dd));
        let inv_norm = 1.0 / normalizer;
        for n in 0..n_obs {
            for k in 0..dd {
                residual[[n, k]] = g[[n, k]] * inv_norm - g_ref[[n, k]];
            }
        }
        let mut residual_dot_g = 0.0;
        for n in 0..n_obs {
            for k in 0..dd {
                residual_dot_g += residual[[n, k]] * g[[n, k]];
            }
        }
        let trace_coeff = residual_dot_g / (normalizer * normalizer * trace_denominator);
        let mut metric_grad = Array2::<f64>::zeros((n_obs, dd));
        for n in 0..n_obs {
            for a in 0..d {
                for b in 0..d {
                    let k = a * d + b;
                    let mut value = residual[[n, k]] * inv_norm;
                    if a == b {
                        value -= trace_coeff;
                    }
                    metric_grad[[n, k]] = value;
                }
            }
        }
        Some(IsometryMetricState {
            g,
            residual,
            metric_grad,
            normalizer,
            trace_denominator,
            residual_dot_g,
        })
    }

    /// Exact closed-form gradient of the isometry penalty with respect to the
    /// cached decoder Jacobian `J ∈ ℝ^{n_obs × p × d}` (the autograd input that
    /// torch's `_IsometryPenaltyFn` differentiates). Returns the flattened
    /// `(n_obs, p*d)` layout that matches the Jacobian cache.
    ///
    /// Derivation (W-aware, reference-aware, weight-aware):
    ///
    ///   P        = ½ μ Σ_n ‖R_n‖²_F,
    ///   R_n      = g_n / gbar − g^ref_n,
    ///   gbar     = (1 / (N d)) Σ_n tr(g_n)
    ///   A_n      = ∂(P/μ)/∂g_n
    ///   ∂g_{ab}/∂J_{i,c}
    ///            = δ_{ca}(W J)_{i,b} + δ_{cb}(W J)_{i,a}   (W symmetric)
    ///   ∂P/∂J_{i,c}
    ///            = μ Σ_{a,b} A_{ab} ∂g_{ab}/∂J_{i,c}
    ///            = 2 μ Σ_b A_{cb} (W J)_{i,b}
    ///            = 2 μ ((W J) A)_{i,c}
    ///
    /// where `A` includes the exact derivative of the shared `gbar` normalizer.
    pub fn grad_jacobian(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        let p = self.p_out;
        let mut grad = Array2::<f64>::zeros((n_obs, p * d));
        if !self.has_jacobian_cache("grad_jacobian") {
            return grad;
        }
        let Some(g) = self.pullback_metric(d) else {
            return grad;
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return grad;
        };
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return Array2::<f64>::zeros((n_obs, p * d));
            };
            for i in 0..p {
                for c in 0..d {
                    let mut acc = 0.0;
                    for b in 0..d {
                        acc += metric.metric_grad[[n, c * d + b]] * wj[[i, b]];
                    }
                    grad[[n, i * d + c]] = 2.0 * mu * acc;
                }
            }
        }
        grad
    }
}


impl AnalyticPenalty for IsometryPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("value") {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        }
        let Some(g) = self.pullback_metric(d) else {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Self::DEFAULT_VALUE_ON_MISSING_CACHE;
        };
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut acc = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                let diff = metric.residual[[n, k]];
                acc += diff * diff;
            }
        }
        0.5 * mu * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Exact closed-form gradient, W-aware:
        //
        //   P     = ½ μ Σ_n ‖R_n‖²_F,   R_n = g_n / gbar − g^ref_n
        //   g_n   = J_n^T W_n J_n,      W_n = U_n U_n^T
        //   A_n   = ∂(P/μ)/∂g_n, including the exact derivative of
        //           gbar = (1 / (N d)) Σ_n tr(g_n)
        //   ∂g_{ab}/∂t_c
        //         = (H_{:,a,c})^T (W J)_{:,b}  +  (J_{:,a})^T W H_{:,b,c}
        //   ∂P/∂t_c
        //         = μ Σ_{a,b} A_{a,b} · ∂g_{ab}/∂t_c
        //
        // `H = ∂J/∂t` comes either from the live cache or from the radial
        // Duchon `φ''(r)` helper. The sign is positive: differentiating
        // `t - c` with respect to `t` contributes `+I`.
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("grad_target")
            || !self.has_jacobian_second_source("grad_target")
        {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(g) = self.pullback_metric(d) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = self.p_out;
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut grad = Array1::<f64>::zeros(target.len());
        let Some(jac2) = self.jacobian_second(target, n_obs, d) else {
            return grad;
        };
        assert_eq!(jac2.ncols(), p * d * d);

        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return grad;
            };
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let mut dg = 0.0;
                        for i in 0..p {
                            dg += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
                            dg += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
                        }
                        acc += metric.metric_grad[[n, a * d + b]] * dg;
                    }
                }
                grad[n * d + c] = mu * acc;
            }
        }
        grad
    }

    /// Fully analytic - wired through `radial_basis_cartesian_derivative`.
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Fully analytic isometry Hessian-vector product wired through the
        // shared `radial_basis_cartesian_derivative` engine when no
        // third-derivative cache is supplied.
        //
        // The full Hessian of P_iso = (μ/2) Σ_n ||J^T W J / gbar - G_ref||²_F
        // (per proposal §4(b)) is
        //   μ [Dgᵀ · ∂²(0.5||R||²)/∂g² · Dg + A · ∂²g],
        // where R = g/gbar - G_ref and A = ∂(0.5||R||²)/∂g includes the global
        // gbar derivative.
        //   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
        //             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd},
        // where K is the third decoder derivative and H is the second.
        let Some(state) = self.hvp_state(target) else {
            return Array1::<f64>::zeros(v.len());
        };
        self.hvp_with_precomputed_state(&state, rho, v)
    }

    /// PSD majorizer-vector product `B_GN(target; ρ) v` for the **nonconvex**
    /// isometry penalty.
    ///
    /// The Gauss-Newton block differentiates the normalized residual
    /// `R = g/gbar - G_ref` itself and returns `μ DRᵀ DR v`. This is PSD by
    /// construction and includes the shared-normalizer derivative exactly;
    /// using only `∂g` would reintroduce scale coupling and would not be the
    /// Gauss-Newton operator of the objective being minimized.
    fn psd_majorizer_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        if !self.has_jacobian_cache("psd_majorizer_hvp")
            || !self.has_jacobian_second_source("psd_majorizer_hvp")
        {
            return Array1::<f64>::zeros(v.len());
        }
        let Some(jac2) = self.jacobian_second(target, n_obs, d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let Some(g) = self.pullback_metric(d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let Some(metric) = self.normalized_metric_state(g, n_obs, d) else {
            return Array1::<f64>::zeros(v.len());
        };
        let p = self.p_out;
        let mu = resolve_learnable_weight(self.scalar_weight, rho[self.rho_index]);
        let mut out = Array1::<f64>::zeros(v.len());
        let mut wj_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return Array1::<f64>::zeros(v.len());
            };
            wj_rows.push(wj);
        }
        let mut delta_g = Array2::<f64>::zeros((n_obs, d * d));
        for n in 0..n_obs {
            let row_delta = isometry_row_delta_g(jac2.view(), wj_rows[n].view(), v, n, d, p);
            for a in 0..d {
                for b in 0..d {
                    delta_g[[n, a * d + b]] = row_delta[[a, b]];
                }
            }
        }
        let (delta_residual, _delta_normalizer) = metric.residual_direction(delta_g.view(), d);
        let mut g_dot_delta_residual = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                g_dot_delta_residual += metric.g[[n, k]] * delta_residual[[n, k]];
            }
        }
        let inv_norm = 1.0 / metric.normalizer;
        let inv_norm_sq = inv_norm * inv_norm;
        for n in 0..n_obs {
            let wj = &wj_rows[n];
            for c in 0..d {
                let mut trace_dg = 0.0;
                for a in 0..d {
                    trace_dg += isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, a, c);
                }
                let delta_normalizer_c = trace_dg / metric.trace_denominator;
                let mut acc = -delta_normalizer_c * inv_norm_sq * g_dot_delta_residual;
                for a in 0..d {
                    for b in 0..d {
                        let dg = isometry_dg_entry(jac2.view(), wj.view(), n, d, p, a, b, c);
                        acc += dg * inv_norm * delta_residual[[n, a * d + b]];
                    }
                }
                out[n * d + c] = mu * acc;
            }
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // P(ρ) = ½ μ · S, where S is the (ρ-independent) Frobenius sum and
        // μ = exp(ρ_iso). So ∂P/∂ρ_iso = P.
        let mut out = Array1::<f64>::zeros(self.rho_count());
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "isometry"
    }

    impl_scalar_apply_schedule!(scalar_weight);
}


// ---------------------------------------------------------------------------
// Sparsity penalty
// ---------------------------------------------------------------------------

/// Sparsifier kernel.
///
/// * `SmoothedL1 { eps }` — `Σ_i sqrt(x_i² + ε²)`. The smoothing scale `ε`
///   may be REML-selected (`eps_rho_index = Some(_)`), in which case the
///   shrink rate `ε → 0` is governed by the marginal likelihood (Occam keeps
///   `ε` large when the data don't demand sharpness).
/// * `Hoyer` — `(√n · ‖x‖_1 − ‖x‖_2) / (√n − 1)`. Scale-invariant; encourages
///   absolute sparsity even when the global scale of `x` drifts.
/// * `Log { delta }` — `Σ_i log(1 + x_i² / δ²)`. Strongly concave; aggressive
///   sparsifier suitable for active-set / iterative-reweighted paths.
#[derive(Debug, Clone, Copy)]
pub enum SparsityKind {
    SmoothedL1 { eps: f64 },
    Hoyer,
    Log { delta: f64 },
}


/// Sparsity penalty on a slice of β (SAE codes) or ext-coords (soft atom assignments).
///
/// The smoothed-L¹ default `Σ_i sqrt(x_i² + ε²)` is the simplest analytic
/// option. Its gradient is `x_i / sqrt(x_i² + ε²)` (a smooth sign function),
/// and its Hessian is diagonal with entries `ε² / (x_i² + ε²)^{3/2}` — so
/// `hvp` is cheap and the inner Newton step inherits a benign block-diagonal
/// regularizer.
///
/// When to use: any time a parameter block carries a "this should be sparse"
/// prior — SAE atom codes (β slice), soft-routing weights on a latent
/// ext-coordinate slice. For SAE codes specifically, smoothed-L¹ with REML-selected `ε`
/// gives the principled relaxation of the L¹ objective without giving up
/// differentiability.
#[derive(Debug, Clone)]
pub struct SparsityPenalty {
    pub target_tier: PenaltyTier,
    pub kind: SparsityKind,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Index of `log strength` inside this penalty's local ρ view.
    pub strength_rho_index: usize,
    /// If `Some`, the index of `log ε` (or `log δ`) inside this penalty's
    /// local ρ view. If `None`, `ε` / `δ` is held fixed at the value baked
    /// into [`SparsityKind`].
    pub eps_rho_index: Option<usize>,
}


/// Entropy sparsity over row-wise softmax assignment logits.
///
/// This is the SAE-manifold soft-assignment penalty. The target is a flat
/// row-major `(N, K)` logit matrix. Assignments are
/// `a_i = softmax(logits_i / temperature)`, and the penalty is
///
/// ```text
///   lambda_sparse * sum_i H(a_i)
///   H(a_i) = -sum_k a_ik log a_ik
/// ```
///
/// Minimizing entropy drives each row toward a small active support while the
/// softmax keeps `a_ik >= 0` and `sum_k a_ik = 1`. The exact Hessian is dense
/// in each row and can be indefinite because entropy is concave in assignment
/// space, so callers must use the HVP rather than a diagonal Hessian shortcut.
#[derive(Debug, Clone)]
pub struct SoftmaxAssignmentSparsityPenalty {
    pub k_atoms: usize,
    pub temperature: f64,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl SoftmaxAssignmentSparsityPenalty {
    #[must_use]
    pub fn new(k_atoms: usize, temperature: f64) -> Self {
        assert!(k_atoms > 0);
        assert!(temperature > 0.0);
        Self {
            k_atoms,
            temperature,
            weight: 1.0,
            weight_schedule: None,
        }
    }

    impl_with_weight_schedule!(weight);

    fn softmax_row(&self, row: &[f64]) -> Vec<f64> {
        let inv_tau = 1.0 / self.temperature;
        let mut max_logit = f64::NEG_INFINITY;
        for (idx, &v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "SoftmaxAssignmentSparsityPenalty: non-finite logit at atom {idx}: {v}"
            );
            max_logit = max_logit.max(v);
        }
        let mut out = vec![0.0; self.k_atoms];
        let mut sum = 0.0;
        for i in 0..self.k_atoms {
            let v = ((row[i] - max_logit) * inv_tau).exp();
            out[i] = v;
            sum += v;
        }
        assert!(
            sum.is_finite() && sum > 0.0,
            "SoftmaxAssignmentSparsityPenalty: non-finite softmax normalizer"
        );
        for v in out.iter_mut() {
            *v /= sum;
        }
        out
    }

    /// Absolute row sums of the exact per-row dense entropy Hessian, used as a
    /// Gershgorin / diagonal-dominance PSD majorizer.
    ///
    /// The exact per-row Hessian wrt logits (symmetric, dense) is
    ///
    /// ```text
    ///   H_kj = (λ/τ²)·a_k·[ δ_kj·(m − L_k − 1) + a_j·(L_k + L_j + 1 − 2m) ],
    ///   L_k = ln a_k + 1,   m = Σ_j a_j L_j,
    /// ```
    ///
    /// whose diagonal coincides with [`AnalyticPenalty::hessian_diag`]. Entropy
    /// is concave in assignment space, so this block is indefinite (negative on
    /// near-uniform rows). Setting `D_kk = Σ_j |H_kj|` makes `D − H` symmetric
    /// with nonnegative diagonal and diagonally dominant
    /// (`D_kk − H_kk = |H_kk| − H_kk + Σ_{j≠k}|H_kj| ≥ Σ_{j≠k}|(D−H)_kj|`),
    /// hence PSD: `D ⪰ H` and `D ⪰ 0` both hold. `D` is a genuine PSD diagonal
    /// operator that dominates the dense Hessian's quadratic form — unlike the
    /// raw indefinite diagonal, which is neither PSD nor a faithful stand-in for
    /// the dense operator.
    fn psd_majorizer_abs_row_sums(&self, row: &[f64], scale: f64) -> Vec<f64> {
        let a = self.softmax_row(row);
        let k = self.k_atoms;
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        let mut d = vec![0.0_f64; k];
        for kk in 0..k {
            // Diagonal entry H_kk.
            let h_kk = scale * a[kk] * ((m - l[kk] - 1.0) + a[kk] * (2.0 * l[kk] + 1.0 - 2.0 * m));
            let mut acc = h_kk.abs();
            // Off-diagonal entries H_kj, j ≠ k.
            for jj in 0..k {
                if jj == kk {
                    continue;
                }
                let h_kj = scale * a[kk] * a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m);
                acc += h_kj.abs();
            }
            d[kk] = acc;
        }
        d
    }

    /// Exact per-row dense softmax-entropy Hessian wrt the row's logits (#1038),
    /// scaled by `scale = λ/τ²`. Returns the symmetric `K×K` block
    ///
    /// ```text
    ///   H_kj = scale·a_k·[ δ_kj·(m − L_k − 1) + a_j·(L_k + L_j + 1 − 2m) ],
    ///   L_k = ln a_k + 1,   m = Σ_r a_r L_r,
    /// ```
    ///
    /// whose diagonal coincides with [`AnalyticPenalty::hessian_diag`] and whose
    /// quadratic form coincides with [`AnalyticPenalty::hvp`]. This is the dense
    /// block the Arrow-Schur row factor stores so the criterion's `log|H|` and
    /// the #1006 θ-adjoint differentiate the SAME operator (not just its
    /// diagonal). The entropy block alone is gauge-null (`H·𝟙 = 0`, softmax
    /// shift-invariance); callers must add it to the gauge-breaking data-fit
    /// row block before factoring — never factor it in isolation.
    #[must_use]
    pub fn row_dense_hessian(&self, row_logits: &[f64], scale: f64) -> Array2<f64> {
        let k = self.k_atoms;
        let a = self.softmax_row(row_logits);
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        let mut h = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                h[[kk, jj]] = scale
                    * a[kk]
                    * (indicator * (m - l[kk] - 1.0) + a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m));
            }
        }
        h
    }

    /// Derivative of the exact per-row dense entropy Hessian
    /// [`Self::row_dense_hessian`] with respect to a single row logit `z_w`,
    /// scaled by `scale = λ/τ²`. Returns the symmetric `K×K` block
    /// `∂H_kj/∂z_w`, the third-derivative tensor slice the #1006 θ-adjoint
    /// contracts against the row's selected inverse. Built from the SAME
    /// `(a, L, m)` as [`Self::row_dense_hessian`] (`∂a_r/∂z_w = a_r(δ_rw − a_w)/τ`),
    /// so value, logdet and adjoint stay on one branch.
    #[must_use]
    pub fn row_dense_hessian_logit_derivative(
        &self,
        row_logits: &[f64],
        scale: f64,
        w: usize,
    ) -> Array2<f64> {
        let k = self.k_atoms;
        let inv_tau = 1.0 / self.temperature;
        let a = self.softmax_row(row_logits);
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        // ∂a_r/∂z_w = a_r (δ_rw − a_w)/τ ; ∂L_r/∂z_w = (∂a_r/∂z_w)/a_r.
        let da: Vec<f64> = (0..k)
            .map(|r| a[r] * (if r == w { 1.0 } else { 0.0 } - a[w]) * inv_tau)
            .collect();
        let dl: Vec<f64> = (0..k)
            .map(|r| da[r] / a[r].max(ENTROPY_LOG_PROBABILITY_FLOOR))
            .collect();
        let dm: f64 = (0..k).map(|r| da[r] * l[r] + a[r] * dl[r]).sum();
        let mut dh = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                // bracket = δ_kj(m − L_k − 1) + a_j(L_k + L_j + 1 − 2m).
                let bracket =
                    indicator * (m - l[kk] - 1.0) + a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m);
                let dbracket = indicator * (dm - dl[kk])
                    + da[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m)
                    + a[jj] * (dl[kk] + dl[jj] - 2.0 * dm);
                dh[[kk, jj]] = scale * (da[kk] * bracket + a[kk] * dbracket);
            }
        }
        dh
    }
}


impl AnalyticPenalty for SoftmaxAssignmentSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut acc = 0.0;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            for v in a {
                if v > 0.0 {
                    acc += -v * v.ln();
                }
            }
        }
        lambda * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau = 1.0 / self.temperature;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut d_h_da = vec![0.0; self.k_atoms];
            let mut mean = 0.0;
            for k in 0..self.k_atoms {
                let ak = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR);
                d_h_da[k] = -lambda * (ak.ln() + 1.0);
                mean += a[k] * d_h_da[k];
            }
            for k in 0..self.k_atoms {
                out[start + k] = a[k] * (d_h_da[k] - mean) * inv_tau;
            }
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 1, "softmax entropy expects one rho parameter");
        assert!(
            rho.iter().all(|value| value.is_finite()),
            "softmax entropy rho must be finite"
        );
        assert_eq!(
            target.len() % self.k_atoms,
            0,
            "softmax entropy target length must be divisible by k_atoms"
        );
        // Closed-form diagonal of the softmax-entropy Hessian wrt logits.
        // Derived by probing the row-dense HVP with the unit vector e_k:
        // for a row with softmax weights a_k and L_k = ln a_k + 1,
        //   H_kk = (lambda / tau^2) * a_k *
        //          ((1 - 2 a_k) * (E_a[L] - L_k) + a_k - 1).
        // This matches `hvp(...) . e_k` analytically (see derivation in the
        // bug-fix comment on `hvp`) and gives Newton/Arrow-Schur callers a
        // principled diagonal surrogate without per-row dense factorization.
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut mean_log_plus_one = 0.0;
            for k in 0..self.k_atoms {
                mean_log_plus_one += a[k] * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
            }
            for k in 0..self.k_atoms {
                let log_plus_one = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0;
                let term = (1.0 - 2.0 * a[k]) * (mean_log_plus_one - log_plus_one) + a[k] - 1.0;
                out[start + k] = scale * a[k] * term;
            }
        }
        Some(out)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        /*
        Softmax entropy is not coordinate-separable in logits. The old
        `hessian_diag` returned λ p_k(1-p_k)/τ², which is only the softmax
        Jacobian diagonal and omits the entropy curvature and all cross-logit
        terms. For H(p(z)), p'=p*(v-E_p[v])/τ and
        (log p_k + 1)'=(v_k-E_p[v])/τ. Differentiating
        g_k=λ p_k(E_p[log p + 1]-(log p_k+1))/τ gives the row-dense product
        below. `hessian_diag` returns the analytic diagonal extracted from
        this HVP by setting v = e_k row-by-row.
        */
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut mean_log_plus_one = 0.0;
            let mut mean_v = 0.0;
            for k in 0..self.k_atoms {
                mean_log_plus_one += a[k] * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
                mean_v += a[k] * v[start + k];
            }
            let mut mean_centered_v_log_plus_one = 0.0;
            for k in 0..self.k_atoms {
                let centered_v = v[start + k] - mean_v;
                mean_centered_v_log_plus_one +=
                    a[k] * centered_v * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
            }
            for k in 0..self.k_atoms {
                let log_plus_one = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0;
                let centered_v = v[start + k] - mean_v;
                out[start + k] = scale
                    * a[k]
                    * (centered_v * (mean_log_plus_one - log_plus_one - 1.0)
                        + mean_centered_v_log_plus_one);
            }
        }
        out
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 1, "softmax entropy expects one rho parameter");
        assert_eq!(
            target.len() % self.k_atoms,
            0,
            "softmax entropy target length must be divisible by k_atoms"
        );
        // Entropy minimization is nonconvex: the exact per-row Hessian is dense
        // and indefinite, so the convex-only trait default (which returns the
        // raw indefinite `hessian_diag`) violates the `B ⪰ 0` contract and is a
        // diagonal masquerading as a dense operator. Replace it with the
        // Gershgorin / diagonal-dominance majorizer of the dense per-row block
        // (see `psd_majorizer_abs_row_sums`): a genuine PSD diagonal with
        // `D ⪰ H` and `D ⪰ 0`. Coordinate-indexed, so the inherited
        // `psd_majorizer_hvp` applies `D` as a diagonal operator consistently.
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_atoms;
            let d = self.psd_majorizer_abs_row_sums(&values[start..start + self.k_atoms], scale);
            for k in 0..self.k_atoms {
                out[start + k] = d[k];
            }
        }
        Some(out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        Array1::from_vec(vec![self.value(target, rho)])
    }

    fn rho_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "softmax_assignment_sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}


/// IBP-MAP active-set prior over SAE-manifold assignment logits.
///
/// Infinite GPFA / IBP-GPFA in neuroscience uses an Indian Buffet Process
/// prior over factor loadings to infer both a potentially unbounded factor
/// set and which factors contribute at each observation. The relevant
/// diagnosis carries over directly to SAE-manifold assignment: ordinary ARD
/// selects one global factor set for all observations, not a different set
/// for each observation. A per-row IBP active set is the established GPFA
/// remedy, adapted here to gamfit's REML/MAP engine with a finite truncation
/// and deterministic concrete relaxation.
///
/// The target is row-major `(N, K)` logits. For MAP we use a deterministic
/// binary-concrete score `z_ik = sigmoid(logit_ik / tau)`, with optional
/// Gumbel temperature annealing across outer iterations. Each column has
/// `pi_k ~ Beta(alpha / K, 1)` and `z_ik | pi_k ~ Bernoulli(pi_k)`. We plug in
/// the columnwise Beta-Bernoulli MAP `pi_k` from the relaxed active mass, so
/// the penalty is a gauge-fixing prior: it breaks the per-row
/// interchangeability of atom indices by making each row choose a sparse
/// binary-ish subset rather than assigning every atom a soft nonzero weight.
#[derive(Debug, Clone)]
pub struct IBPAssignmentPenalty {
    pub k_max: usize,
    pub alpha: f64,
    pub tau: f64,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub learnable_alpha: bool,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl IBPAssignmentPenalty {
    #[must_use]
    pub fn new(k_max: usize, alpha: f64, tau: f64, learnable_alpha: bool) -> Self {
        assert!(k_max > 0);
        assert!(alpha.is_finite() && alpha > 0.0);
        assert!(tau.is_finite() && tau > 0.0);
        Self {
            k_max,
            alpha,
            tau,
            temperature_schedule: None,
            learnable_alpha,
            weight: 1.0,
            weight_schedule: None,
        }
    }

    #[must_use]
    pub fn with_temperature_schedule(mut self, schedule: GumbelTemperatureSchedule) -> Self {
        self.tau = schedule.current_tau(schedule.iter_count);
        self.temperature_schedule = Some(schedule);
        self
    }

    impl_with_weight_schedule!(weight);

    fn resolved_alpha(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_alpha {
            resolve_learnable_weight(self.alpha, rho[0])
        } else {
            self.alpha
        }
    }

    fn concrete_temperature(&self) -> f64 {
        self.tau
    }

    fn concrete_logits(&self, target: ArrayView1<'_, f64>) -> Array1<f64> {
        let tau = self.concrete_temperature();
        let mut out = Array1::<f64>::zeros(target.len());
        for i in 0..target.len() {
            let x = target[i] / tau;
            out[i] = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
        }
        out
    }

    fn pi_map(&self, z: ArrayView1<'_, f64>, alpha: f64) -> Array1<f64> {
        let n = z.len() / self.k_max;
        let a = alpha / self.k_max as f64;
        let mut pi = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let mut active_mass = 0.0;
            for row in 0..n {
                active_mass += z[row * self.k_max + k];
            }
            let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let raw = (active_mass + a - 1.0) / denom;
            pi[k] = raw.clamp(IBP_INTERIOR_TOL, 1.0 - IBP_INTERIOR_TOL);
        }
        pi
    }

    /// Exact third-derivative channels of [`Self::hessian_diag`] with respect to
    /// the logits, for the SAE outer-ρ log-det adjoint Γ (#1006).
    ///
    /// `hessian_diag` returns, per row `i` and column `k`, the on-diagonal
    /// curvature
    ///
    /// ```text
    ///   H_ik = w · [ sd_k · J_ik²  +  score_k · c_ik ],
    /// ```
    ///
    /// with `J_ik = z(1−z)/τ` the logit→concrete jacobian, `c_ik =
    /// z(1−z)(1−2z)/τ²` the second jacobian, and the column scalars
    /// `score_k`, `sd_k = ∂score_k/∂M_k` exactly as assembled there
    /// (`M_k = Σ_i z_ik` is the column active mass, `π_k(M_k)` the plug-in
    /// stick-breaking MAP). Because `π_k` couples every row in column `k`, the
    /// logit derivative splits into a row-local direct-`z` channel and a global
    /// empirical-`M_k` channel:
    ///
    /// ```text
    ///   ∂H_ik/∂ℓ_wk = δ_iw · (∂_z H_ik)·J_ik   +   (∂_M H_ik) · J_wk,
    ///   ∂_z H_ik = w·J_ik·[ sd_k·2J_ik·(1−2z)/τ + score_k·(1−6z+6z²)/τ² ],
    ///   ∂_M H_ik = w·[ sdd_k · J_ik²  +  sd_k · c_ik ],
    ///   sdd_k = ∂sd_k/∂M_k = ∂²score_k/∂M_k².
    /// ```
    ///
    /// `local_logit_third[i*K+k] = (∂_z H_ik)·J_ik` is the row-diagonal third
    /// derivative; `m_channel[i*K+k] = ∂_M H_ik` and `z_jac[i*K+k] = J_ik` let
    /// the caller form, per column, `C_k = Σ_i (H⁻¹)_ik,ik · ∂_M H_ik` and
    /// distribute `C_k · J_wk` to every row `w` (the cross-row coupling the
    /// row-local primitive cannot see). All boundary clamps (`pi_jac = 0` at the
    /// `π_k` clamp) ride the same convention as `hessian_diag`, so the channels
    /// are zero exactly where the assembled curvature is constant in `M_k`.
    #[must_use]
    pub fn hessian_diag_logit_third_channels(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> IbpHessianDiagThirdChannels {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);

        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }

        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut score_second_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a - 1.0) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
            let direct_z_score = ((1.0 - pk) / pk).ln();
            let implicit_pi_score = pi_score * pi_jac;
            score[k] = direct_z_score + implicit_pi_score;
            let direct_z_score_derivative = pi_jac * (-1.0 / pk - 1.0 / (1.0 - pk));
            score_derivative[k] = direct_z_score_derivative + pi_score_derivative * pi_jac;

            // sdd_k = ∂score_derivative_k/∂M_k, holding the explicit per-row z
            // fixed (the same partial `hessian_diag` takes for score/sd). With
            // π_k = (M_k+a−1)/D clamped, ∂π_k/∂M_k = pi_jac (0 at the clamp):
            //   ∂(direct_z_score_derivative)/∂M = pi_jac²·(1/π² − 1/(1−π)²),
            //   ∂(pi_score_derivative)/∂M = pi_jac·[ 2/π² − 2(M+a−1)·pi_jac/π³
            //                                        − 2/(1−π)² + 2(n−M)·pi_jac/(1−π)³ ].
            let one_minus = 1.0 - pk;
            let ddzd = pi_jac * pi_jac * (1.0 / (pk * pk) - 1.0 / (one_minus * one_minus));
            let dpisd = 2.0 / (pk * pk)
                - 2.0 * (mass + a - 1.0) * pi_jac / (pk * pk * pk)
                - 2.0 / (one_minus * one_minus)
                + 2.0 * (n as f64 - mass) * pi_jac / (one_minus * one_minus * one_minus);
            score_second_derivative[k] = ddzd + dpisd * pi_jac;
        }

        let len = target.len();
        let mut z_jac = Array1::<f64>::zeros(len);
        let mut local_logit_third = Array1::<f64>::zeros(len);
        let mut m_channel = Array1::<f64>::zeros(len);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let c_ik = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                // ∂_z J = (1−2z)/τ, ∂_z c = (1−6z+6z²)/τ².
                let dz_j = (1.0 - 2.0 * zk) * inv_tau;
                let dz_c = (1.0 - 6.0 * zk + 6.0 * zk * zk) * inv_tau2;
                let dz_h = score_derivative[k] * 2.0 * jac * dz_j + score[k] * dz_c;
                z_jac[start + k] = jac;
                local_logit_third[start + k] = self.weight * jac * dz_h;
                m_channel[start + k] = self.weight
                    * (score_second_derivative[k] * jac * jac + score_derivative[k] * c_ik);
            }
        }

        // #1038 cross-row Woodbury: per column `k`, the EXACT IBP Hessian has the
        // rank-one cross-row block `H_(i,k),(j,k) += w·s'_k·z'_ik·z'_jk` (for all
        // `i,j`, including `i=j`). `cross_row_d[k] = w·s'_k = w·score_derivative_k`
        // is its scalar `D`-coefficient; `z_jac` already holds `u_k`'s entries
        // `z'_ik`. The consumer subtracts the `i=j` self term from `H₀` (the
        // assembled diagonal carries it) and adds the FULL rank-one via the
        // determinant lemma, so value/logdet/adjoint all differentiate one
        // operator. Built from the SAME `(score_derivative, z_jac)` source as the
        // diagonal `hessian_diag` and the `m_channel`/`local_logit_third` third
        // tensor — the issue's one-operator non-negotiable.
        let mut cross_row_d = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            cross_row_d[k] = self.weight * score_derivative[k];
        }

        IbpHessianDiagThirdChannels {
            k_max: self.k_max,
            z_jac,
            local_logit_third,
            m_channel,
            cross_row_d,
        }
    }

    /// Mixed derivative `∂/∂ℓ_ik [∂F/∂ρ_alpha]` for learnable-alpha IBP.
    ///
    /// This differentiates the implemented energy in [`Self::value`]. At the
    /// empirical-π interior, the BCE and `(a-1) log π` implicit-π terms cancel in
    /// `∂F/∂a`, leaving the normalized Beta(a,1) channel. At the probability
    /// clamp, the same zero-π-Jacobian convention as [`Self::grad_target`] and
    /// [`Self::hessian_diag`] applies.
    #[must_use]
    pub fn log_alpha_target_mixed_derivative(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        if !self.learnable_alpha {
            return out;
        }
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let raw = (active_mass[k] + a - 1.0) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) / tau;
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                out[start + k] = -self.weight * a * pi_jac[k] * z_jac / pk;
            }
        }
        out
    }

    /// `∂ hessian_diag / ∂ρ_alpha` for learnable-alpha IBP.
    ///
    /// The SAE log-det trace differentiates the diagonal returned by
    /// [`Self::hessian_diag`]. This channel differentiates that exact diagonal
    /// with respect to the learnable-alpha log-coordinate while holding logits
    /// fixed. IBP columns remain independent, so within-row off-diagonals are zero.
    #[must_use]
    pub fn hessian_diag_log_alpha_derivative(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        if !self.learnable_alpha {
            return out;
        }
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut d_score = Array1::<f64>::zeros(self.k_max);
        let mut d_score_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a - 1.0) / denom;
            if raw <= IBP_INTERIOR_TOL || raw >= 1.0 - IBP_INTERIOR_TOL {
                continue;
            }
            let one_minus = 1.0 - pk;
            let dpi_da = (n as f64 - mass) / (denom * denom);
            let dpi_drho = a * dpi_da;
            let d_score_dpi = -1.0 / pk - 1.0 / one_minus;
            d_score[k] = d_score_dpi * dpi_drho;

            let inv_p = 1.0 / pk;
            let inv_q = 1.0 / one_minus;
            let a_channel = inv_p + inv_q;
            let d_a_channel_da = dpi_da * (-inv_p * inv_p + inv_q * inv_q);
            let d_score_derivative_da = a_channel / (denom * denom) - d_a_channel_da / denom;
            d_score_derivative[k] = a * d_score_derivative_da;
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) * inv_tau;
                let z_second = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] =
                    self.weight * (d_score_derivative[k] * z_jac * z_jac + d_score[k] * z_second);
            }
        }
        out
    }
}


/// Exact logit third-derivative channels of [`IBPAssignmentPenalty::hessian_diag`]
/// for the SAE outer-ρ log-det adjoint Γ (#1006). Row-major `(N, K)` layout.
#[derive(Debug, Clone)]
pub struct IbpHessianDiagThirdChannels {
    /// Number of columns `K` (atoms) in the row-major logit layout.
    pub k_max: usize,
    /// `J_ik = z(1−z)/τ`, the per-logit concrete jacobian (row-major `N·K`).
    pub z_jac: Array1<f64>,
    /// `(∂_z H_ik)·J_ik`: the row-local direct-`z` third derivative of the
    /// assembled diagonal curvature `H_ik` (row-major `N·K`).
    pub local_logit_third: Array1<f64>,
    /// `∂_M H_ik`: the empirical-`M_k` channel of `H_ik`. Contract against the
    /// selected-inverse diagonal per column, then distribute `C_k·J_wk` to every
    /// row `w` (row-major `N·K`).
    pub m_channel: Array1<f64>,
    /// `cross_row_d[k] = w·s'_k`: the scalar `D`-coefficient of the per-column
    /// cross-row rank-one Hessian block `H_(i,k),(j,k) = w·s'_k·z'_ik·z'_jk`
    /// (#1038). Paired with `u_k = z_jac[·,k]` this is the exact column-`k`
    /// Woodbury update `d_k·u_k·u_kᵀ` (full outer product, `i=j` included).
    /// Length `K`.
    pub cross_row_d: Array1<f64>,
}


impl AnalyticPenalty for IBPAssignmentPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut acc = 0.0;
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                acc -= zk * pk.ln() + (1.0 - zk) * (1.0 - pk).ln();
            }
        }
        for k in 0..self.k_max {
            // Normalized Beta(a,1) density is a*pi^(a-1), so its negative
            // log contribution is -ln(a) - (a - 1) ln(pi). The normalizer is
            // constant only for fixed alpha; keep it in both modes so the energy
            // has one mathematical definition across configurations.
            acc -= a.ln();
            acc -= (a - 1.0) * pi[k].ln();
        }
        self.weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut out = Array1::<f64>::zeros(target.len());
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a - 1.0) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                out[start + k] =
                    self.weight * (direct_z_score + implicit_pi_score) * zk * (1.0 - zk) / tau;
            }
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (tau * tau);
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a - 1.0) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
            pi_score_derivative[k] = -1.0 / pk + (mass + a - 1.0) * pi_jac[k] / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac[k] / ((1.0 - pk) * (1.0 - pk));
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                let score = direct_z_score + implicit_pi_score;
                let direct_z_score_derivative = pi_jac[k] * (-1.0 / pk - 1.0 / (1.0 - pk));
                let score_derivative =
                    direct_z_score_derivative + pi_score_derivative[k] * pi_jac[k];
                let z_jac = zk * (1.0 - zk) / tau;
                out[start + k] = self.weight
                    * (score_derivative * z_jac * z_jac
                        + score * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2);
            }
        }
        Some(out)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(
            v.len(),
            target.len(),
            "IBPAssignmentPenalty::hvp dimension mismatch"
        );
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let denom = (n as f64 + a - 1.0).max(IBP_COUNT_DENOM_FLOOR);

        // Column aggregates (active_mass, pi_jac, pi_score, pi_score_derivative,
        // score, score_derivative). These are identical to hessian_diag and
        // share the same interior / boundary-clamp convention, so the on-row
        // diagonal returned by hvp(·, eⱼ) agrees with hessian_diag bit-for-bit.
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a - 1.0) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
            let direct_z_score = ((1.0 - pk) / pk).ln();
            let implicit_pi_score = pi_score * pi_jac;
            score[k] = direct_z_score + implicit_pi_score;
            let direct_z_score_derivative = pi_jac * (-1.0 / pk - 1.0 / (1.0 - pk));
            score_derivative[k] = direct_z_score_derivative + pi_score_derivative * pi_jac;
        }

        // Within-column block structure: pi[k] and active_mass[k] depend on
        // EVERY row in column k, so the per-column Hessian block is a rank-1
        // perturbation of a diagonal,
        //
        //   H[(j,k), (j',k)] = w · score_derivative[k] · z_jac[j,k] · z_jac[j',k]
        //                    + δ_{jj'} · w · score[k] · (1-2z[j,k]) · z(1-z) / τ²,
        //
        // where z_jac[j,k] = z(1-z)/τ at row j in column k. Different
        // columns are decoupled (pi[k] depends only on column k), so the
        // full Hessian is block-diagonal by column.
        //
        // For an input vector v, the rank-1 contribution collapses to a
        // single per-column scalar sₖ = Σⱼ z_jac[j,k] · v[j,k]:
        //
        //   (Hv)[j,k] = w · score_derivative[k] · z_jac[j,k] · sₖ
        //             + w · score[k] · (1-2z[j,k]) · z(1-z)/τ² · v[j,k].
        //
        // The default diagonal-only hvp drops the off-diagonal rank-1 piece,
        // which empirically carries ≈85% of the operator's Frobenius norm.
        let mut s_per_col = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let zjac = zk * (1.0 - zk) * inv_tau;
                s_per_col[k] += zjac * v[start + k];
            }
        }
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let zjac = zk * (1.0 - zk) * inv_tau;
                let rank1 = score_derivative[k] * zjac * s_per_col[k];
                let c_diag = score[k] * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] = self.weight * (rank1 + c_diag * v[start + k]);
            }
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_alpha {
            return Array1::<f64>::zeros(0);
        }
        let alpha = self.resolved_alpha(rho);
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let mut sum_log_pi = 0.0;
        for &pk in pi.iter() {
            sum_log_pi += pk
                .clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP)
                .ln();
        }
        Array1::from_vec(vec![
            -self.weight * (alpha * sum_log_pi / self.k_max as f64 + self.k_max as f64),
        ])
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_alpha)
    }

    fn name(&self) -> &str {
        "ibp_assignment_map"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.temperature_schedule.as_mut() {
            self.tau = schedule.current_tau(iter);
            schedule.iter_count = iter + 1;
        }
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}


impl SparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn smoothed_l1(target_tier: PenaltyTier, eps: f64) -> Result<Self, String> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!(
                "SparsityPenalty::smoothed_l1 requires eps > 0 \
                 (Hessian / gradient have a `1/sqrt(x² + eps²)` factor that needs eps > 0 \
                 for differentiability at x = 0); got eps = {eps}"
            ));
        }
        Ok(Self {
            target_tier,
            kind: SparsityKind::SmoothedL1 { eps },
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        })
    }

    #[must_use = "build error must be handled"]
    pub fn log(target_tier: PenaltyTier, delta: f64) -> Result<Self, String> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(format!(
                "SparsityPenalty::log requires delta > 0 \
                 (the log-sparsifier is log(1 + x²/δ²), undefined at δ = 0); \
                 got delta = {delta}"
            ));
        }
        Ok(Self {
            target_tier,
            kind: SparsityKind::Log { delta },
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        })
    }

    /// Hoyer scale-invariant sparsifier. Requires a target of length > 1
    /// because the normalized form divides by `sqrt(n) - 1`.
    #[must_use]
    pub fn hoyer(target_tier: PenaltyTier) -> Self {
        Self {
            target_tier,
            kind: SparsityKind::Hoyer,
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        }
    }

    impl_with_weight_schedule!(weight);

    #[must_use]
    pub fn with_eps_reml(mut self, eps_rho_index: usize) -> Self {
        self.eps_rho_index = Some(eps_rho_index);
        self
    }

    /// Resolve `(strength, eps_or_delta)` from the current ρ view.
    fn resolved(&self, rho: ArrayView1<'_, f64>) -> (f64, f64) {
        let strength = resolve_learnable_weight(self.weight, rho[self.strength_rho_index]);
        let smoothing = match (self.eps_rho_index, self.kind) {
            // A learnable smoothing `exp(rho)` underflows to exact `0.0` for
            // `rho ≲ -745`, which reintroduces a non-differentiable kink and a
            // `0/0` at `x = 0` in `sqrt(x² + ε²)` / the Log sparsifier. Floor it
            // at the smallest positive normal so the smoothing stays strictly
            // positive while still shrinking arbitrarily close to zero.
            (Some(idx), _) => rho[idx].exp().max(f64::MIN_POSITIVE),
            (None, SparsityKind::SmoothedL1 { eps }) => eps,
            (None, SparsityKind::Log { delta }) => delta,
            (None, SparsityKind::Hoyer) => 0.0,
        };
        (strength, smoothing)
    }
}


impl AnalyticPenalty for SparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        self.target_tier
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut acc = 0.0;
                for &x in target.iter() {
                    acc += (x * x + smooth * smooth).sqrt();
                }
                lam * acc
            }
            SparsityKind::Hoyer => {
                // Normalized anti-sparsity penalty
                //   P(x) = (||x||_1 / ||x||_2 - 1) / (sqrt(n) - 1)
                // maps [1, sqrt(n)] -> [0, 1]. A perfectly dense
                // equal-magnitude vector hits ||x||_1/||x||_2 = sqrt(n),
                // so P = 1; a 1-sparse vector has ratio 1, so P = 0
                // (sparse vectors minimize the penalty).
                let n = target.len() as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return 0.0;
                }
                let h = (l1 / l2 - 1.0) / (n.sqrt() - 1.0);
                lam * h
            }
            SparsityKind::Log { .. } => {
                let mut acc = 0.0;
                let d2 = smooth * smooth;
                for &x in target.iter() {
                    acc += (1.0 + x * x / d2).ln();
                }
                lam * acc
            }
        }
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let (lam, smooth) = self.resolved(rho);
        let mut g = Array1::<f64>::zeros(target.len());
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    g[i] = lam * x / (x * x + eps2).sqrt();
                }
            }
            SparsityKind::Hoyer => {
                // P(x) = A · (L1/L2 - 1), A = lam / (sqrt(n) - 1).
                // ∂P/∂x_i = A · (sign(x_i)/L2 - L1 · x_i / L2³).
                let n = target.len() as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return g;
                }
                let denom = n.sqrt() - 1.0;
                let a = lam / denom;
                let inv_l2 = 1.0 / l2;
                let inv_l2_cubed = inv_l2 * inv_l2 * inv_l2;
                for (i, &x) in target.iter().enumerate() {
                    let sgn = if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    g[i] = a * (sgn * inv_l2 - l1 * x * inv_l2_cubed);
                }
            }
            SparsityKind::Log { .. } => {
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    g[i] = lam * 2.0 * x / (d2 + x * x);
                }
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let r = (x * x + eps2).sqrt();
                    d[i] = lam * eps2 / (r * r * r);
                }
                Some(d)
            }
            SparsityKind::Log { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                // The EXACT second derivative of λ log(1 + x²/δ²):
                //   d/dx [ 2λx/(δ²+x²) ] = 2λ(δ² − x²)/(δ² + x²)²,
                // which is NEGATIVE for |x| > δ — Log is nonconvex. This is
                // the genuine Hessian diagonal and exactly differentiates
                // `grad_target`. PSD consumers (Newton block, preconditioner,
                // `log_det_plus_λI`, FrozenAnalyticPenaltyOp) must instead
                // route through `psd_majorizer_diag`/`psd_majorizer_hvp`,
                // which expose the IRLS/MM surrogate `2λ/(δ²+x²)`.
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    d[i] = lam * 2.0 * (d2 - x * x) / (denom * denom);
                }
                Some(d)
            }
            // Hoyer's Hessian is DENSE and NOT generally PSD (Hoyer is a
            // nonconvex sparsifier). We cannot return a meaningful diagonal
            // that would be safe to use as a preconditioner / Newton block
            // through the standard `hessian_diag` path, so we return `None`
            // and force callers through `hvp`. See `hvp` below for the exact
            // dense-Hessian-vector product.
            SparsityKind::Hoyer => None,
        }
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // For SmoothedL1/Log/Hoyer we route through the closed-form Hessian.
        // SmoothedL1 and Log have purely diagonal Hessians and would
        // ordinarily reach the diagonal branch of the default `hvp`; we
        // override here to also serve Hoyer (whose Hessian is dense
        // rank-1-plus-diagonal).
        let (lam, smooth) = self.resolved(rho);
        let n_target = target.len();
        assert_eq!(v.len(), n_target, "hvp dimension mismatch");
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut out = Array1::<f64>::zeros(n_target);
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let r = (x * x + eps2).sqrt();
                    out[i] = lam * eps2 / (r * r * r) * v[i];
                }
                out
            }
            SparsityKind::Log { .. } => {
                // EXACT Hessian-vector product: the Log Hessian is diagonal
                // with entries 2λ(δ²−x²)/(δ²+x²)², so (Hv)_i = h_i v_i. This
                // is the genuine second derivative (indefinite for |x|>δ).
                // PSD consumers use `psd_majorizer_hvp` for the IRLS/MM
                // surrogate 2λ/(δ²+x²) instead.
                let mut out = Array1::<f64>::zeros(n_target);
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    out[i] = lam * 2.0 * (d2 - x * x) / (denom * denom) * v[i];
                }
                out
            }
            SparsityKind::Hoyer => {
                // P(x) = A · (L1/L2 - 1), A = lam / (sqrt(n) - 1).
                // H_ij = A · [ -s_i x_j/L2³ - x_i s_j/L2³
                //              - L1 δ_ij/L2³ + 3 L1 x_i x_j/L2⁵ ]
                // (Hv)_i = A · [ -s_i (xᵀv)/L2³ - x_i (sᵀv)/L2³
                //                - L1 v_i/L2³ + 3 L1 x_i (xᵀv)/L2⁵ ]
                let n = n_target as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                let mut out = Array1::<f64>::zeros(n_target);
                if l2 == 0.0 {
                    return out;
                }
                let a = lam / (n.sqrt() - 1.0);
                let inv_l2_cubed = 1.0 / (l2 * l2 * l2);
                let inv_l2_5 = inv_l2_cubed / (l2 * l2);
                let mut x_dot_v = 0.0;
                let mut s_dot_v = 0.0;
                for i in 0..n_target {
                    let xi = target[i];
                    let si = if xi > 0.0 {
                        1.0
                    } else if xi < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    x_dot_v += xi * v[i];
                    s_dot_v += si * v[i];
                }
                for i in 0..n_target {
                    let xi = target[i];
                    let si = if xi > 0.0 {
                        1.0
                    } else if xi < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    out[i] = a
                        * (-si * x_dot_v * inv_l2_cubed
                            - xi * s_dot_v * inv_l2_cubed
                            - l1 * v[i] * inv_l2_cubed
                            + 3.0 * l1 * xi * x_dot_v * inv_l2_5);
                }
                out
            }
        }
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            // SmoothedL1 is convex: the majorizer equals the exact Hessian.
            SparsityKind::SmoothedL1 { .. } => self.hessian_diag(target, rho),
            // Log is nonconvex; expose the IRLS/MM re-weighted-ℓ₂ surrogate
            //   2λ/(δ²+x²) ⪰ 2λ(δ²−x²)/(δ²+x²)²,
            // strictly positive, agreeing with the exact Hessian at x = 0.
            SparsityKind::Log { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    d[i] = lam * 2.0 / (d2 + x * x);
                }
                Some(d)
            }
            // Hoyer's Hessian is dense; no diagonal majorizer. Callers fall
            // back to the exact dense `hvp` through `psd_majorizer_hvp`.
            SparsityKind::Hoyer => None,
        }
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Strength axis: ∂P/∂ρ_strength = P (chain rule through exp).
        // ε axis (if owned): ∂P/∂ρ_eps = ε · ∂P/∂ε.
        let n_rho = self.rho_count();
        let mut out = Array1::<f64>::zeros(n_rho);
        let p_val = self.value(target, rho);
        out[self.strength_rho_index] = p_val;
        if let Some(eps_idx) = self.eps_rho_index {
            let (lam, smooth) = self.resolved(rho);
            let mut dp_deps = 0.0;
            match self.kind {
                SparsityKind::SmoothedL1 { .. } => {
                    for &x in target.iter() {
                        dp_deps += smooth / (x * x + smooth * smooth).sqrt();
                    }
                    dp_deps *= lam;
                }
                SparsityKind::Log { .. } => {
                    // d/dδ log(1 + x²/δ²) = -2 x² / (δ (δ² + x²))
                    let d2 = smooth * smooth;
                    for &x in target.iter() {
                        dp_deps += -2.0 * x * x / (smooth * (d2 + x * x));
                    }
                    dp_deps *= lam;
                }
                SparsityKind::Hoyer => {}
            }
            // Chain through ρ_eps = log(ε)  ⇒  ∂ε/∂ρ_eps = ε.
            out[eps_idx] = smooth * dp_deps;
        }
        out
    }

    fn rho_count(&self) -> usize {
        1 + if self.eps_rho_index.is_some() { 1 } else { 0 }
    }

    fn name(&self) -> &str {
        "sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// ARD penalty
// ---------------------------------------------------------------------------

/// ARD (Automatic Relevance Determination) over latent axes.
///
/// One independent quadratic ridge penalty per latent axis, with one
/// REML-selectable log-precision per axis. Penalty contribution for axis `j`:
///
/// ```text
///   P_j(t; ρ) = ½ α_j · ‖t[:, j]‖² - (n_eff / 2) · log α_j,
///   α_j = weight · exp(ρ_j)
/// ```
///
/// summed over `j ∈ [0, d)` for the extension-coordinate target block
/// `T ∈ ℝ^{n_eff × d}`. In the SAE objective this is the latent-axis
/// dimension-selection prior: under REML, axis `j` whose data evidence is too
/// weak gets `ρ_j → +∞` (precision → ∞, coefficients → 0), so the latent
/// dimension is effectively pruned.
///
/// Because the penalty is quadratic and block-diagonal in latent axes, it
/// reduces to a [`BlockwisePenalty`] per axis and slots into the existing
/// canonical-penalty pipeline with zero extra wiring beyond appending `d`
/// hyperparameter axes to `ρ`.
///
/// Gotchas:
///
/// * ARD is not a standalone identifiability fix. The intrinsic dimensionality
///   is meaningful only after a separate gauge-fixing prior (`AuxPrior`,
///   `IsometryPenalty`, or an equivalent basis constraint) has fixed rotations
///   and reparameterizations.
/// * `n_eff` controls the Gaussian normalizer / Occam term. Override it only
///   when rows have been aggregated or otherwise represent a different
///   effective observation count than `target.len() / latent_dim`.
/// * The row-major `LatentCoordValues` layout means each per-axis ridge is
///   strided in memory; [`Self::as_blockwise`] expands it into scalar
///   `BlockwisePenalty` entries rather than pretending each axis is contiguous.
///
/// When to use: any [`crate::terms::latent_coord::LatentCoordValues`] block
/// where the intrinsic dimension is unknown. Compose with `IsometryPenalty`
/// for full gauge fixing.
#[derive(Debug, Clone)]
pub struct ARDPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Local ρ indices for the `d` per-axis log-precisions.
    pub rho_indices: Vec<usize>,
    /// Effective number of observations contributing to each latent axis.
    /// Enters the per-axis log-determinant Occam term in `grad_rho`:
    /// at an unused axis (Σ_n t_{n,j}² = 0) the gradient becomes
    /// `-n_eff / 2`, which under minimization pushes ρ_j → +∞ and prunes
    /// the axis. Default is the number of latent-row observations
    /// (`target.len() / latent_dim`).
    pub n_eff: f64,
}


impl ARDPenalty {
    #[must_use]
    pub fn new(target: PsiSlice, latent_dim: usize) -> Self {
        assert!(latent_dim > 0, "ARDPenalty requires latent_dim > 0");
        let n_obs = if latent_dim == 0 {
            0
        } else {
            target.len() / latent_dim
        };
        let rho_indices = (0..latent_dim).collect();
        Self {
            target,
            latent_dim,
            weight: 1.0,
            weight_schedule: None,
            rho_indices,
            n_eff: n_obs as f64,
        }
    }

    impl_with_weight_schedule!(weight);

    /// Override the effective observation count used in the Occam log-det
    /// term (default: `target.len() / latent_dim`). Pass the number of
    /// latent rows that actually contribute to axis `j` (uniform across
    /// axes for the current implementation).
    #[must_use = "build error must be handled"]
    pub fn with_n_eff(mut self, n_eff: f64) -> Result<Self, String> {
        if !(n_eff.is_finite() && n_eff >= 0.0) {
            return Err(format!(
                "ARDPenalty::with_n_eff requires a finite non-negative value, got {n_eff}"
            ));
        }
        self.n_eff = n_eff;
        Ok(self)
    }

    /// Build scalar [`BlockwisePenalty`] entries for each latent-axis row.
    /// Fixes the audit finding that the row-major `LatentCoordValues` layout
    /// (`n * d + j`) cannot be represented as one contiguous per-axis range.
    pub fn as_blockwise(&self, global_offset: usize) -> Vec<BlockwisePenalty> {
        let n_obs = self.target.len() / self.latent_dim;
        let mut out = Vec::with_capacity(n_obs * self.latent_dim);
        for j in 0..self.latent_dim {
            for n in 0..n_obs {
                let idx = global_offset + self.target.range.start + n * self.latent_dim + j;
                out.push(BlockwisePenalty::ridge(idx..idx + 1, 1.0).with_op(None));
            }
        }
        out
    }
}


impl AnalyticPenalty for ARDPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut acc = 0.0;
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            acc += 0.5 * lam_j * sq - 0.5 * self.n_eff * lam_j.ln();
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut g = Array1::<f64>::zeros(target.len());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            for n in 0..n_obs {
                g[n * d + j] = lam_j * target[n * d + j];
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            for n in 0..n_obs {
                diag[n * d + j] = lam_j;
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Uses the prior normalizer -0.5 * N_eff * log(weight * exp(rho_j)).
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(self.rho_count());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            out[self.rho_indices[j]] = 0.5 * lam_j * sq - 0.5 * self.n_eff;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.latent_dim
    }

    fn name(&self) -> &str {
        "ard"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// TopK activation penalty
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TopKActivationPenalty {
    pub target: PsiSlice,
    pub k: usize,
    pub latent_dim: usize,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl TopKActivationPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(target: PsiSlice, k: usize, weight: f64) -> Result<Self, String> {
        let latent_dim = target
            .latent_dim
            .ok_or_else(|| "TopKActivationPenalty::new requires target.latent_dim".to_string())?;
        if latent_dim == 0 {
            return Err("TopKActivationPenalty::new requires latent_dim > 0".to_string());
        }
        if k == 0 || k > latent_dim {
            return Err(format!(
                "TopKActivationPenalty::new requires 0 < k <= latent_dim; got k={k}, latent_dim={latent_dim}"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "TopKActivationPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        Ok(Self {
            target,
            k,
            latent_dim,
            weight,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn topk_mask_row(&self, target: ArrayView1<'_, f64>, row: usize, mask: &mut [bool]) {
        mask.fill(false);
        let d = self.latent_dim;
        let base = row * d;
        let mut order = (0..d).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            target[base + b]
                .abs()
                .total_cmp(&target[base + a].abs())
                .then_with(|| a.cmp(&b))
        });
        for &axis in order.iter().take(self.k) {
            mask[axis] = true;
        }
    }
}


impl AnalyticPenalty for TopKActivationPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut acc = 0.0;
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    let v = target[base + axis];
                    acc += 0.5 * self.weight * v * v;
                }
            }
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut grad = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    grad[base + axis] = self.weight * target[base + axis];
                }
            }
        }
        grad
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    diag[base + axis] = self.weight;
                }
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        assert_eq!(
            target.len() % self.latent_dim,
            0,
            "TopKActivationPenalty target length must be a multiple of latent_dim"
        );
        Array1::<f64>::zeros(0)
    }

    fn rho_count(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "topk_activation"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// JumpReLU penalty
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct JumpReLUPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    pub thresholds: Array1<f64>,
    pub weight: f64,
    pub smoothing_eps: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl JumpReLUPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        thresholds: Array1<f64>,
        weight: f64,
        smoothing_eps: f64,
    ) -> Result<Self, String> {
        let latent_dim = target
            .latent_dim
            .ok_or_else(|| "JumpReLUPenalty::new requires target.latent_dim".to_string())?;
        if latent_dim == 0 {
            return Err("JumpReLUPenalty::new requires latent_dim > 0".to_string());
        }
        if thresholds.len() != latent_dim {
            return Err(format!(
                "JumpReLUPenalty::new thresholds length {} does not match latent_dim {latent_dim}",
                thresholds.len()
            ));
        }
        for (idx, &tau) in thresholds.iter().enumerate() {
            if !(tau.is_finite() && tau > 0.0) {
                return Err(format!(
                    "JumpReLUPenalty::new thresholds[{idx}] must be finite and > 0, got {tau}"
                ));
            }
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "JumpReLUPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "JumpReLUPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            target,
            latent_dim,
            thresholds,
            weight,
            smoothing_eps,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn threshold(&self, axis: usize, rho: ArrayView1<'_, f64>) -> f64 {
        // A learnable threshold `θ·exp(rho)` overflows to `inf` for large `rho`;
        // the downstream gate `σ((l−θ)/τ)` then evaluates `inf·gate = NaN`. Clamp
        // the log-magnitude so the threshold stays a finite normal.
        resolve_learnable_weight(self.thresholds[axis], rho[axis])
    }

    fn sigmoid_gate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    fn true_hessian_diag_entry(&self, tau: f64, gate: f64) -> f64 {
        self.weight * tau * gate * (1.0 - gate) * (1.0 - 2.0 * gate)
            / (self.smoothing_eps * self.smoothing_eps)
    }

    fn psd_hessian_diag_entry(&self, tau: f64, gate: f64) -> f64 {
        // Genuine PSD majorizer of the indefinite exact diagonal Hessian
        //   h(g) = λτ·g(1−g)(1−2g)/ε².
        // The bare re-weighted-ℓ₂ surrogate λτ·[g(1−g)]²/ε² is ≥ 0 but only
        // dominates h in the concave region g > ½. For g < (3−√5)/2 ≈ 0.382 the
        // exact curvature is positive and strictly larger, so the square alone
        // is NOT an upper bound — the `B ⪰ ∂²P` contract is violated for exactly
        // the comfortably-below-threshold (inactive) coordinates JumpReLU is
        // meant to suppress, costing the MM step its monotone-decrease guarantee.
        //
        // Take the elementwise max of that surrogate and the absolute exact
        // Hessian |h| = λτ·g(1−g)|1−2g|/ε². Since |h| ≥ h everywhere and ≥ 0, the
        // max is a true PSD upper bound; it equals |h| in the wings (tight where
        // the bare square failed) and keeps the surrogate's strictly-positive
        // floor near the inflection g ≈ ½ (where h ≈ 0) so the curvature block
        // never collapses to zero.
        let slope = gate * (1.0 - gate);
        let reweighted_l2 = slope * slope;
        let abs_exact = slope * (1.0 - 2.0 * gate).abs();
        self.weight * tau * reweighted_l2.max(abs_exact) / (self.smoothing_eps * self.smoothing_eps)
    }
}


/// JumpReLU activation gate `φ(z) = z · 1[z > τ]` together with the
/// straight-through-estimator derivatives of its smooth surrogate
/// `φ̃(z) = z · σ((z − τ)/ε)`. The forward value is the hard gate; the backward
/// uses the surrogate's gradients so the activation has a usable subgradient in
/// the smoothing band `|z − τ| ≲ ε`:
///
///   g       = σ((z − τ)/ε)
///   φ        = z · 1[z > τ]                 (returned value)
///   ∂φ̃/∂z   = g + z · g (1 − g) / ε          (`dphi_dz`)
///   ∂φ̃/∂τ   = − z · g (1 − g) / ε            (`dphi_dtau`)
///
/// This is the single Rust source of truth that `gamfit.torch`'s
/// `_JumpReLUSTEFn` consumes so the torch activation gate's backward matches the
/// smoothed gate exactly instead of re-deriving it in Python.
#[must_use]
pub fn jumprelu_gate_value_grad(z: f64, tau: f64, smoothing_eps: f64) -> (f64, f64, f64) {
    let g = crate::linalg::utils::stable_logistic((z - tau) / smoothing_eps);
    let value = if z > tau { z } else { 0.0 };
    let slope = z * g * (1.0 - g) / smoothing_eps;
    let dphi_dz = g + slope;
    let dphi_dtau = -slope;
    (value, dphi_dz, dphi_dtau)
}


impl AnalyticPenalty for JumpReLUPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut acc = 0.0;
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                acc += self.weight * tau * gate;
            }
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut grad = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                grad[base + axis] = self.weight * tau * gate * (1.0 - gate) / self.smoothing_eps;
            }
        }
        grad
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                diag[base + axis] = self.true_hessian_diag_entry(tau, gate);
            }
        }
        Some(diag)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                out[base + axis] = self.true_hessian_diag_entry(tau, gate) * v[base + axis];
            }
        }
        out
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        // The smoothed JumpReLU surrogate's exact diagonal Hessian
        //   λτ·g(1−g)(1−2g)/ε²
        // is indefinite (negative once the gate passes the inflection
        // g = ½). The Newton / PIRLS pipeline needs a PSD curvature block, so
        // expose the re-weighted majorizer  λτ·[g(1−g)]²/ε² ⪰ 0.
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                diag[base + axis] = self.psd_hessian_diag_entry(tau, gate);
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(d);
        for axis in 0..d {
            let tau = self.threshold(axis, rho);
            let mut g_tau = 0.0;
            for row in 0..n_obs {
                let x = target[row * d + axis];
                let gate = self.sigmoid_gate((x - tau) / self.smoothing_eps);
                g_tau += gate - tau * gate * (1.0 - gate) / self.smoothing_eps;
            }
            out[axis] = self.weight * tau * g_tau;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.latent_dim
    }

    fn name(&self) -> &str {
        "jumprelu"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Total variation penalty
// ---------------------------------------------------------------------------

/// Shape of the first-difference operator used by [`TotalVariationPenalty`].
#[derive(Debug, Clone)]
pub enum DifferenceOpKind {
    /// Path graph with rows connected as `(0, 1), (1, 2), ...`.
    ForwardDiff1D,
    /// Explicit adjacency list; each edge row has `-1` at `from`, `+1` at `to`.
    GraphEdges(Vec<(usize, usize)>),
}


/// Coordinatewise/anisotropic smoothed-L¹ total variation on a row-major
/// `(n_eff, d)` latent block.
///
/// Uses the differentiable Huber-style kernel `φ(x)=sqrt(x²+ε²)-ε` separately
/// for each edge and latent axis. This is not vector-norm/isotropic edge TV:
/// the Hessian intentionally has no cross-axis terms. The difference operator
/// defines the prior shape: forward 1-D differences for ordered context
/// windows, or graph edges for adjacency-structured atoms. Pair TV with
/// Orthogonality when piecewise-constant atoms need a gauge-fixed basis.
#[derive(Debug, Clone)]
pub struct TotalVariationPenalty {
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major latent coefficient block.
    pub n_eff: usize,
    pub difference_op: DifferenceOpKind,
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl TotalVariationPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        weight: f64,
        n_eff: usize,
        difference_op: DifferenceOpKind,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("TotalVariationPenalty::new requires n_eff > 0".to_string());
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if let DifferenceOpKind::GraphEdges(edges) = &difference_op {
            if edges.is_empty() {
                return Err(
                    "TotalVariationPenalty::new GraphEdges requires at least one edge".to_string(),
                );
            }
            for &(a, b) in edges {
                if a >= n_eff || b >= n_eff {
                    return Err(format!(
                        "TotalVariationPenalty::new graph edge ({a}, {b}) exceeds n_eff {n_eff}"
                    ));
                }
                if a == b {
                    return Err(format!(
                        "TotalVariationPenalty::new graph edge ({a}, {b}) is self-referential"
                    ));
                }
            }
        }
        Ok(Self {
            weight,
            n_eff,
            difference_op,
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn edge_count(&self) -> usize {
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => self.n_eff.saturating_sub(1),
            DifferenceOpKind::GraphEdges(edges) => edges.len(),
        }
    }

    fn add_edge_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = eps2 / (r * r * r);
            let dv = v[ib] - v[ia];
            let h = weight * curvature * dv;
            out[ia] -= h;
            out[ib] += h;
        }
    }

    fn add_edge_grad(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let smooth_sign = diff / (diff * diff + eps2).sqrt();
            let g = weight * smooth_sign;
            out[ia] -= g;
            out[ib] += g;
        }
    }

    fn add_edge_diag(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = weight * eps2 / (r * r * r);
            out[ia] += curvature;
            out[ib] += curvature;
        }
    }

    fn add_edge_dense(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array2<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = weight * eps2 / (r * r * r);
            out[[ia, ia]] += curvature;
            out[[ib, ib]] += curvature;
            out[[ia, ib]] -= curvature;
            out[[ib, ia]] -= curvature;
        }
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_diag(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_diag(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    /// Materialize `Dᵀ diag(φ''(D T)) D` for diagnostics and small graph cases.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(d) = self.latent_dim(n) else {
            return Array2::<f64>::zeros((n, n));
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros((n, n));
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_dense(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_dense(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    pub fn log_det_plus_lambda_i_forward_1d(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !matches!(&self.difference_op, DifferenceOpKind::ForwardDiff1D) {
            return Err(
                "TotalVariationPenalty::log_det_plus_lambda_i_forward_1d requires ForwardDiff1D"
                    .to_string(),
            );
        }
        let Some(d) = self.latent_dim(target.len()) else {
            return Err(format!(
                "TotalVariationPenalty target length {} is not divisible by n_eff {}",
                target.len(),
                self.n_eff
            ));
        };
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::log_det_plus_lambda_i_forward_1d requires finite λ > 0; got {lambda}"
            ));
        }
        let n = self.n_eff;
        if n == 1 {
            return Ok((d as f64) * lambda.ln());
        }
        let weight = self.resolved_weight(rho);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        let mut total = 0.0;
        for j in 0..d {
            let mut edge_w = vec![0.0; n - 1];
            for a in 0..n - 1 {
                let diff = target[(a + 1) * d + j] - target[a * d + j];
                let r = (diff * diff + eps2).sqrt();
                edge_w[a] = weight * eps2 / (r * r * r);
            }

            let mut prev_pivot = lambda + edge_w[0];
            if !prev_pivot.is_finite() || prev_pivot <= 0.0 {
                return Err(format!(
                    "TotalVariationPenalty log-det encountered non-positive pivot {prev_pivot:.3e}"
                ));
            }
            total += prev_pivot.ln();
            for row in 1..n {
                let left = edge_w[row - 1];
                let right = if row + 1 < n { edge_w[row] } else { 0.0 };
                let diag = lambda + left + right;
                let pivot = diag - left * left / prev_pivot;
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(format!(
                        "TotalVariationPenalty log-det encountered non-positive pivot {pivot:.3e}"
                    ));
                }
                total += pivot.ln();
                prev_pivot = pivot;
            }
        }
        Ok(total)
    }
}


impl AnalyticPenalty for TotalVariationPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(d) = self.latent_dim(target.len()) else {
            return 0.0;
        };
        if self.edge_count() == 0 {
            return 0.0;
        }
        let weight = self.resolved_weight(rho);
        let eps = self.smoothing_eps;
        let eps2 = eps * eps;
        let mut acc = 0.0;
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    let b = a + 1;
                    for j in 0..d {
                        let diff = target[b * d + j] - target[a * d + j];
                        acc += (diff * diff + eps2).sqrt() - eps;
                    }
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    for j in 0..d {
                        let diff = target[b * d + j] - target[a * d + j];
                        acc += (diff * diff + eps2).sqrt() - eps;
                    }
                }
            }
        }
        weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_grad(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_grad(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_hvp(target, v, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_hvp(target, v, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "total_variation"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Monotonicity penalty (1D shape constraint)
// ---------------------------------------------------------------------------

/// Soft monotonicity penalty over a row-major `(n_eff, d)` latent block.
///
/// For each adjacent pair `(a, a+1)` along the leading axis and each output
/// column `j`, the penalty contribution is
///
///     softplus(-direction * (target[a+1, j] - target[a, j]) / smoothing_eps)
///     * smoothing_eps
///
/// which is the smoothed hinge that hits zero when the slope agrees with
/// `direction` (+1 ⇒ non-decreasing, -1 ⇒ non-increasing) and grows
/// approximately linearly when it disagrees. The Hessian is positive
/// semidefinite (softplus is convex) so the penalty composes cleanly with
/// PIRLS/REML.
///
/// `n_eff` is the number of latent rows along the constrained axis; the
/// remaining `target.len() / n_eff` columns are penalized independently and
/// summed.
#[derive(Debug, Clone)]
pub struct MonotonicityPenalty {
    pub weight: f64,
    pub n_eff: usize,
    /// `+1.0` for non-decreasing, `-1.0` for non-increasing along the leading axis.
    pub direction: f64,
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl MonotonicityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        weight: f64,
        n_eff: usize,
        direction: f64,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("MonotonicityPenalty::new requires n_eff > 0".to_string());
        }
        if !(direction.is_finite() && direction.abs() > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite non-zero direction (+1 or -1), got {direction}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            weight,
            n_eff,
            direction: direction.signum(),
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            return None;
        }
        Some(target_len / self.n_eff)
    }

    /// Smoothed-hinge contribution for a single edge `(a, b)` and column `j`.
    fn edge_value(&self, target: ArrayView1<'_, f64>, d: usize, a: usize, b: usize) -> f64 {
        let eps = self.smoothing_eps;
        let mut acc = 0.0;
        for j in 0..d {
            let slope = target[b * d + j] - target[a * d + j];
            let z = -self.direction * slope / eps;
            // softplus(z) * eps, computed in a numerically stable form.
            let sp = if z > 0.0 {
                z + (-z).exp().ln_1p()
            } else {
                z.exp().ln_1p()
            };
            acc += sp * eps;
        }
        acc
    }

    /// d softplus(-dir * slope / eps) * eps / d target = -dir * sigma(-dir*slope/eps).
    fn edge_grad(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps = self.smoothing_eps;
        for j in 0..d {
            let slope = target[b * d + j] - target[a * d + j];
            let z = -self.direction * slope / eps;
            // Stable sigmoid(z).
            let sigma = if z > 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            };
            let g = weight * (-self.direction) * sigma;
            out[a * d + j] -= g;
            out[b * d + j] += g;
        }
    }
}


impl AnalyticPenalty for MonotonicityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(d) = self.latent_dim(target.len()) else {
            return 0.0;
        };
        if self.n_eff < 2 {
            return 0.0;
        }
        let weight = self.resolved_weight(rho);
        let mut acc = 0.0;
        for a in 0..self.n_eff.saturating_sub(1) {
            acc += self.edge_value(target, d, a, a + 1);
        }
        weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for a in 0..self.n_eff.saturating_sub(1) {
            self.edge_grad(target, &mut out, d, a, a + 1, weight);
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let eps = self.smoothing_eps;
        let mut out = Array1::<f64>::zeros(target.len());
        for a in 0..self.n_eff.saturating_sub(1) {
            let b = a + 1;
            for j in 0..d {
                let slope = target[b * d + j] - target[a * d + j];
                let z = -self.direction * slope / eps;
                let sigma = if z > 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                // d²P/d(target_a)d(target_b) follows from the chain rule on
                // z = -dir * (target_b - target_a) / eps. The penalty value is
                // `softplus(z) * eps` (note the outer eps from `edge_value`).
                // softplus''(z) = sigma(z)(1 - sigma(z)) and the (dz/dtarget)²
                // factor is 1/eps², but the value's outer `* eps` cancels one of
                // those, leaving `sigma(1 - sigma) / eps` — exactly the eps power
                // that keeps `hvp` consistent with the finite difference of
                // `grad_target` (whose own eps already cancelled). Off-diagonal
                // entries carry an extra minus sign from the difference.
                let h = weight * sigma * (1.0 - sigma) / eps;
                let dv = v[b * d + j] - v[a * d + j];
                out[a * d + j] -= h * dv;
                out[b * d + j] += h * dv;
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "monotonicity"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Nuclear norm penalty
// ---------------------------------------------------------------------------

/// Basis-free low-rank penalty for a row-major `(n_eff, d)` latent block.
///
/// Lives on the extension-coordinate tier. The target is viewed as
/// `T ∈ ℝ^{n_eff × d}` and penalized by the smoothed nuclear norm
///
/// ```text
///   P(T) = w · Σ_{i < r} (sqrt(σ_i(T)^2 + ε^2) - ε),
/// ```
///
/// where `σ_i(T)` are singular values and `r` is either the full thin-SVD rank
/// or `max_rank` when a spectral cap is supplied. The penalty is basis-free:
/// it selects the rank of the decoder/latent embedding used by SAE wiring
/// without first committing to a canonical coordinate axis.
///
/// In the SAE objective this is the decoder embedding-rank selection lever
/// (#672): it shrinks unused singular directions of the matrix-valued latent
/// block while allowing the active subspace to rotate. It complements ARD
/// (axis-wise pruning after a gauge fix) and orthogonality/isometry terms
/// (basis and metric identification).
///
/// Gotchas:
///
/// * The Hessian is spectral and dense; callers should use the analytic HVP,
///   not a row-block diagonal shortcut.
/// * `max_rank` truncates the active singular spectrum. If the cutoff splits a
///   tied smoothed Gram eigenvalue, the HVP is undefined and the implementation
///   treats that as a caller contract violation.
/// * `ε > 0` smooths the zero singular-value kink. Very small `ε` makes rank
///   selection sharper but increases curvature around nearly-null directions.
#[derive(Debug, Clone)]
pub struct NuclearNormPenalty {
    pub target: PsiSlice,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub smoothing_eps: f64,
    /// Optional spectrum cap. The implementation computes faer's full thin SVD
    /// and retains the leading `max_rank` singular triplets when present.
    pub max_rank: Option<usize>,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


struct NuclearSvdCache {
    u: Array2<f64>,
    singular: Array1<f64>,
    vt: Array2<f64>,
}


impl NuclearNormPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        weight: f64,
        n_eff: usize,
        smoothing_eps: f64,
        max_rank: Option<usize>,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("NuclearNormPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "NuclearNormPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("NuclearNormPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "NuclearNormPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        if let Some(latent_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(latent_dim).ok_or_else(|| {
                "NuclearNormPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "NuclearNormPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    latent_dim
                ));
            }
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "NuclearNormPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if matches!(max_rank, Some(0)) {
            return Err("NuclearNormPenalty::new requires max_rank > 0".to_string());
        }
        Ok(Self {
            target,
            weight,
            n_eff,
            smoothing_eps,
            max_rank,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn rank_limit(&self, thin_rank: usize) -> usize {
        self.max_rank.unwrap_or(thin_rank).min(thin_rank)
    }

    /// PSD-floored squared smoothed singular value `max(σ² + ε², eig_floor)`,
    /// with `eig_floor = max(ε², 1e-15)`.
    ///
    /// This is the single regularized spectrum shared by `value`, `grad_target`
    /// and the HVP's right-Gram filter, so that the smoothed nuclear norm
    /// `Σ(√(σ²+ε²) − ε)`, its gradient `σ/√(σ²+ε²)`, and the Fréchet
    /// inverse-square-root filter `(σ²+ε²)^{-1/2}` are all evaluated on the
    /// *same* eigenvalue. Without the shared floor the value/gradient (which
    /// previously used the unfloored `σ²+ε²`) desync from the HVP (which floors
    /// the right-Gram eigenvalues) when `ε² < 1e-15`, breaking the
    /// value↔gradient↔Hessian consistency that REML evidence and the Newton
    /// curvature block rely on (#737). The floor itself was introduced for
    /// PSD-roundoff robustness (651d827e6); applying it everywhere preserves
    /// that protection without reintroducing the desync.
    fn regularized_sigma_sq(&self, sigma_sq: f64) -> f64 {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        let eig_floor = eps2.max(1.0e-15);
        (sigma_sq + eps2).max(eig_floor)
    }

    /// Number of leading right-Gram eigen-directions (top singular values) the
    /// HVP keeps active, identical to the rank `value`/`grad` sum over.
    ///
    /// The right Gram `TᵀT` is `d×d` but has at most `thin_rank = min(n_rows, d)`
    /// nonzero eigenvalues (the squared singular values); the remaining
    /// `d − thin_rank` are an exact, *tied* zero subspace. Capping the active
    /// count with the Gram width `d` (or any value `> thin_rank`) would push the
    /// active/inactive cutoff *inside* that tied zero subspace, where the split
    /// is ill-defined — for a wide block (`n_rows < d`) with
    /// `max_rank > thin_rank` this previously panicked. We therefore cap with the
    /// true SVD length `thin_rank`, matching `rank_limit`, so the cutoff always
    /// lands either between the zero subspace and the nonzero singular values, or
    /// between distinct nonzero singular values, never bisecting the zero block.
    fn right_filter_active_count(&self, n_rows: usize, n_cols: usize) -> usize {
        let thin_rank = n_rows.min(n_cols);
        match self.max_rank {
            // No cap: keep every right-Gram direction. The `d − thin_rank` exact
            // zero directions get a finite smoothed `(0+ε²)^(-1/2)` filter and
            // contribute nothing to `G(T)` (since `T` has no projection onto
            // them), so this is consistent with `value`/`grad`'s full sum.
            None => n_cols,
            // A cap that does not bite (`max_rank ≥ thin_rank`) is likewise a
            // no-op: keep every direction.
            Some(max_rank) if max_rank >= thin_rank => n_cols,
            // A genuine cap keeps only the top `max_rank` singular directions —
            // never more than `thin_rank`, so the active/inactive cutoff lands
            // strictly inside the nonzero singular block and never bisects the
            // tied zero subspace of the `d×d` Gram.
            Some(max_rank) => max_rank,
        }
    }

    /// Apply the right-spectral filter pair directly: returns `(V·R, T·dR[V])`,
    /// each `(n_rows, d)`, where `R = (TᵀT + ε²I)^{-1/2}` (regularized, active-
    /// windowed) and `dR[V]` is its Fréchet derivative along `V` — the two
    /// pieces [`Self::hvp`] sums.
    ///
    /// Cost structure: the right Gram `TᵀT` is `d×d` but `rank(G) ≤ n_rows`,
    /// and the tangent Gram `TᵀV + VᵀT` is supported on the joint row space
    /// `S = rowspace(T) ∪ rowspace(V)` with `dim S ≤ 2·n_rows`. Every pair of
    /// eigen-directions with either side in `S⊥` contributes `0` to `dR`
    /// (the tangent Gram annihilates them), and `R` acts on `S⊥` as the
    /// constant `f₀ = regularized(0)^{-1/2}` (a tied eigen-class, so the
    /// filter there is the basis-independent `f₀·(I − SSᵀ)` — active only
    /// when the window covers the full Gram, i.e. no biting `max_rank`).
    /// So the whole computation collapses to an `s×s` eigenproblem plus
    /// `O(n_rows·s·d)` products — replacing the former dense `d×d` eigh and
    /// two `O(d⁴)` basis rotations PER HVP CALL, which at decoder-block
    /// orientation (`d = p` in the thousands) was measured eating >99% of
    /// whole-fit wall time. The dense route is kept below for small `d`
    /// (no asymptotic win) and remains the defining oracle.
    fn right_spectral_filters_applied(
        &self,
        t: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let m = t.nrows();
        let d = t.ncols();
        if d <= 2 * m + 8 {
            let (rf, rfd) = self.right_spectral_inverse_sqrt_derivative(t, v)?;
            return Ok((v.dot(&rf), t.dot(&rfd)));
        }
        // Joint row-space basis S (d × s) by modified Gram-Schmidt over the
        // 2m stacked rows of T and V. Deterministic; relative drop tolerance.
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(2 * m);
        for source in [&t, &v] {
            for row in source.rows() {
                let scale = row.iter().fold(0.0_f64, |a, &x| a + x * x).sqrt();
                if scale <= 0.0 {
                    continue;
                }
                let mut r = row.to_owned();
                for b in &basis {
                    let proj = b.dot(&r);
                    r.scaled_add(-proj, b);
                }
                // Re-orthogonalize once (classical MGS twice-is-enough) so the
                // basis stays orthonormal to working precision.
                for b in &basis {
                    let proj = b.dot(&r);
                    r.scaled_add(-proj, b);
                }
                let norm = r.iter().fold(0.0_f64, |a, &x| a + x * x).sqrt();
                if norm > 1.0e-13 * scale {
                    basis.push(r / norm);
                }
            }
        }
        let s_dim = basis.len();
        if s_dim == 0 {
            // T = V = 0: R is the constant f₀ filter and dR vanishes.
            let active_count = self.right_filter_active_count(m, d);
            if active_count != d {
                // The full right-Gram spectrum is one d-fold tied zero class.
                // A biting max_rank would choose an arbitrary subspace of that
                // class, matching the dense oracle's tie-split rejection.
                return Err(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff (0.0e0, 0.0e0)"
                        .to_string(),
                );
            }
            let f0 = self.regularized_sigma_sq(0.0).powf(-0.5);
            let vr = v.to_owned() * f0;
            return Ok((vr, Array2::<f64>::zeros((m, d))));
        }
        let mut s = Array2::<f64>::zeros((d, s_dim));
        for (j, b) in basis.iter().enumerate() {
            s.column_mut(j).assign(b);
        }
        let ts = t.dot(&s); // m × s
        let vs = v.dot(&s); // m × s
        let gh = ts.t().dot(&ts); // Sᵀ G S
        let dgh = ts.t().dot(&vs) + vs.t().dot(&ts); // Sᵀ dG S
        let (evals, q) = gh.eigh(Side::Lower).map_err(|err| {
            format!("NuclearNormPenalty right-Gram eigendecomposition failed: {err}")
        })?;
        let trace_scale = evals
            .iter()
            .fold(0.0_f64, |acc, &lambda| acc.max(lambda.abs()))
            .max(1.0);
        let psd_tol = 1.0e-10 * trace_scale;
        let mut raw_evals = Array1::<f64>::zeros(s_dim);
        for i in 0..s_dim {
            let lambda = evals[i];
            if !lambda.is_finite() {
                return Err(format!(
                    "NuclearNormPenalty expected finite right-Gram eigenvalue; got {lambda}"
                ));
            }
            if lambda < -psd_tol {
                return Err(format!(
                    "NuclearNormPenalty expected PSD right Gram; eigenvalue {lambda:.3e} \
                     is below numerical tolerance {psd_tol:.3e}"
                ));
            }
            raw_evals[i] = lambda.max(0.0);
        }
        // Active window over the FULL ascending d-spectrum, which is the
        // (d − s)-fold tied zero class of S⊥ followed by `raw_evals`
        // (ascending). Mirrors the dense path's windowing and its tie-split
        // guard exactly.
        let active_count = self.right_filter_active_count(m, d);
        let zero_class_active = active_count == d;
        if !zero_class_active && active_count > s_dim {
            // The cutoff would bisect the tied zero class of S⊥ — the same
            // condition the dense path rejects via its adjacent-eigenvalue
            // guard (both neighbors are exact zeros).
            return Err(
                "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                 right-Gram eigenvalue at the active/inactive cutoff (0.0e0, 0.0e0)"
                    .to_string(),
            );
        }
        // Index of the first ACTIVE entry within the s-block.
        let active_start_s = s_dim.saturating_sub(active_count.min(s_dim));
        if self.max_rank.is_some() && !zero_class_active {
            // Tie guard at the cutoff, on RAW eigenvalues as in the dense path.
            // Left neighbor is inside the s-block when the window is strictly
            // interior; when the window covers the whole s-block the left
            // neighbor is the top of the S⊥ zero class.
            let (left, right) = if active_start_s > 0 {
                (evals[active_start_s - 1], evals[active_start_s])
            } else {
                (0.0, evals[0])
            };
            let scale = (left.abs() + right.abs()).max(1.0);
            if (right - left).abs() <= 1.0e-12 * scale {
                return Err(format!(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff \
                     ({left:.3e}, {right:.3e})"
                ));
            }
        }
        let mut regularized_evals = Array1::<f64>::zeros(s_dim);
        let mut f = Array1::<f64>::zeros(s_dim);
        let mut df = Array1::<f64>::zeros(s_dim);
        for i in 0..s_dim {
            regularized_evals[i] = self.regularized_sigma_sq(raw_evals[i]);
            if i >= active_start_s {
                let lambda = regularized_evals[i];
                f[i] = lambda.powf(-0.5);
                df[i] = -0.5 * lambda.powf(-1.5);
            }
        }
        // B̂ = Q̂ᵀ (Sᵀ dG S) Q̂, then the divided-difference Hadamard product —
        // identical pair rules to the dense path. All pairs touching S⊥ have
        // B = 0 (dG is supported on S), so they need no representation.
        let b_basis = q.t().dot(&dgh).dot(&q);
        let mut deriv_basis = Array2::<f64>::zeros((s_dim, s_dim));
        for i in 0..s_dim {
            for j in 0..s_dim {
                let denom = regularized_evals[i] - regularized_evals[j];
                let scale = (regularized_evals[i].abs() + regularized_evals[j].abs())
                    .max(f64::MIN_POSITIVE);
                let divided_difference = if denom.abs() <= 1.0e-12 * scale {
                    let i_active = i >= active_start_s;
                    let j_active = j >= active_start_s;
                    if i_active && j_active {
                        0.5 * (df[i] + df[j])
                    } else {
                        0.0
                    }
                } else {
                    (f[i] - f[j]) / denom
                };
                deriv_basis[[i, j]] = divided_difference * b_basis[[i, j]];
            }
        }
        // V·R = f₀·V·(I − SSᵀ) [zero-class active only] + (V S) Q̂ f̂ Q̂ᵀ Sᵀ.
        let qf = {
            let mut qf = q.clone();
            for i in 0..s_dim {
                let fi = f[i];
                qf.column_mut(i).mapv_inplace(|x| x * fi);
            }
            qf.dot(&q.t()) // Q̂ diag(f̂) Q̂ᵀ, s×s
        };
        let mut vr = vs.dot(&qf).dot(&s.t());
        if zero_class_active {
            let f0 = self.regularized_sigma_sq(0.0).powf(-0.5);
            // V − (V S) Sᵀ is V's S⊥ component.
            let v_perp = v.to_owned() - vs.dot(&s.t());
            vr += &(v_perp * f0);
        }
        // T·dR = (T S) Q̂ (Δf̂ ∘ B̂) Q̂ᵀ Sᵀ.
        let w = q.dot(&deriv_basis).dot(&q.t());
        let tdr = ts.dot(&w).dot(&s.t());
        Ok((vr, tdr))
    }

    fn compute_svd_cached(&self, t: ArrayView2<'_, f64>) -> NuclearSvdCache {
        // Existing faer wrapper calls `faer::linalg::svd::svd(..., Thin, Thin, ...)`.
        let owned = t.to_owned();
        let (u, singular, vt) = owned
            .svd(true, true)
            .expect("NuclearNormPenalty SVD failed to converge");
        NuclearSvdCache {
            u: u.expect("NuclearNormPenalty requested left singular vectors"),
            singular,
            vt: vt.expect("NuclearNormPenalty requested right singular vectors"),
        }
    }

    fn right_spectral_inverse_sqrt_derivative(
        &self,
        t: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        // HVP for spectral matrix functions (matrix-derivative-with-singular-values):
        // G(T)=T(TᵀT+ε²I)^(-1/2), so dG[V]=V R + T dR[V].
        // The Fréchet derivative dR uses divided differences in the right
        // singular-vector basis, avoiding any dense Hessian materialization.
        let d = t.ncols();
        let active_count = self.right_filter_active_count(t.nrows(), d);
        let active_start = d.saturating_sub(active_count);
        let mut gram = Array2::<f64>::zeros((d, d));
        let mut tangent_gram = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut g = 0.0;
                let mut dg = 0.0;
                for n in 0..t.nrows() {
                    g += t[[n, a]] * t[[n, b]];
                    dg += t[[n, a]] * v[[n, b]] + v[[n, a]] * t[[n, b]];
                }
                gram[[a, b]] = g;
                tangent_gram[[a, b]] = dg;
            }
        }

        let (evals, q) = gram.eigh(Side::Lower).map_err(|err| {
            format!("NuclearNormPenalty right-Gram eigendecomposition failed: {err}")
        })?;
        let trace_scale = evals
            .iter()
            .fold(0.0_f64, |acc, &lambda| acc.max(lambda.abs()))
            .max(1.0);
        let psd_tol = 1.0e-10 * trace_scale;
        let mut raw_evals = Array1::<f64>::zeros(d);
        for i in 0..d {
            let lambda = evals[i];
            if !lambda.is_finite() {
                return Err(format!(
                    "NuclearNormPenalty expected finite right-Gram eigenvalue; got {lambda}"
                ));
            }
            if lambda < -psd_tol {
                return Err(format!(
                    "NuclearNormPenalty expected PSD right Gram; eigenvalue {lambda:.3e} \
                     is below numerical tolerance {psd_tol:.3e}"
                ));
            }
            raw_evals[i] = lambda.max(0.0);
        }
        if self.max_rank.is_some() && active_count < d && active_start > 0 {
            let left = evals[active_start - 1];
            let right = evals[active_start];
            let scale = (left.abs() + right.abs()).max(1.0);
            if (right - left).abs() <= 1.0e-12 * scale {
                return Err(format!(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff \
                     ({left:.3e}, {right:.3e})"
                ));
            }
        }
        let mut regularized_evals = Array1::<f64>::zeros(d);
        let mut f = Array1::<f64>::zeros(d);
        let mut df = Array1::<f64>::zeros(d);
        for i in 0..d {
            // Same shared floor used by `value`/`grad_target` (#737): the
            // right-Gram eigenvalue `raw_evals[i]` is the squared singular value
            // `σ²`, so `regularized_sigma_sq(σ²) = max(σ²+ε², eig_floor)` keeps
            // the filter on the identical regularized spectrum.
            regularized_evals[i] = self.regularized_sigma_sq(raw_evals[i]);
            if i >= active_start {
                // Keep the value filter and Fréchet derivative on the same
                // regularized spectrum. This preserves the PSD-roundoff floor
                // without letting divided differences observe stale raw
                // eigenvalues near zero.
                let lambda = regularized_evals[i];
                f[i] = lambda.powf(-0.5);
                df[i] = -0.5 * lambda.powf(-1.5);
            }
        }

        let mut right_filter = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += q[[a, i]] * f[i] * q[[b, i]];
                }
                right_filter[[a, b]] = s;
            }
        }

        let mut b_basis = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        s += q[[a, i]] * tangent_gram[[a, b]] * q[[b, j]];
                    }
                }
                b_basis[[i, j]] = s;
            }
        }

        let mut derivative_basis = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let denom = regularized_evals[i] - regularized_evals[j];
                let scale = (regularized_evals[i].abs() + regularized_evals[j].abs())
                    .max(f64::MIN_POSITIVE);
                let divided_difference = if denom.abs() <= 1.0e-12 * scale {
                    let i_active = i >= active_start;
                    let j_active = j >= active_start;
                    if i_active && j_active {
                        0.5 * (df[i] + df[j])
                    } else {
                        0.0
                    }
                } else {
                    (f[i] - f[j]) / denom
                };
                derivative_basis[[i, j]] = divided_difference * b_basis[[i, j]];
            }
        }

        let mut right_filter_derivative = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        s += q[[a, i]] * derivative_basis[[i, j]] * q[[b, j]];
                    }
                }
                right_filter_derivative[[a, b]] = s;
            }
        }

        Ok((right_filter, right_filter_derivative))
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }
}


impl AnalyticPenalty for NuclearNormPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let svd = self.compute_svd_cached(t);
        let rank = self.rank_limit(svd.singular.len());
        let eps = self.smoothing_eps;
        let mut acc = 0.0;
        for i in 0..rank {
            let sigma = svd.singular[i];
            // Floored on the shared regularized spectrum so the value matches the
            // HVP's right-Gram filter (see `regularized_sigma_sq`).
            acc += self.regularized_sigma_sq(sigma * sigma).sqrt() - eps;
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let svd = self.compute_svd_cached(t);
        let rank = self.rank_limit(svd.singular.len());
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for i in 0..rank {
            let sigma = svd.singular[i];
            // d/dσ (√(σ²+ε²) − ε) = σ/√(σ²+ε²), floored on the shared regularized
            // spectrum so grad↔value↔HVP stay mutually consistent (#737).
            let spectral_grad = sigma / self.regularized_sigma_sq(sigma * sigma).sqrt();
            for n in 0..t.nrows() {
                for a in 0..t.ncols() {
                    grad[[n, a]] += weight * svd.u[[n, i]] * spectral_grad * svd.vt[[i, a]];
                }
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        // `AnalyticPenalty::hvp_target` has no Result channel; decomposition
        // or active-rank cutoff failures from the spectral helper are upstream
        // contract violations that must surface loudly.
        let (vr, tdr) = self
            .right_spectral_filters_applied(t.view(), v_mat.view())
            // SAFETY: error path is a caller contract violation; the upstream
            // helper already formatted a diagnostic message.
            .unwrap_or_else(|message| panic!("{}", message));
        let weight = self.resolved_weight(rho);
        let out = (vr + tdr) * weight;
        Self::flatten_matrix(&out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "nuclear_norm"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Block sparsity / group-lasso penalty
// ---------------------------------------------------------------------------

/// Group-lasso penalty over predefined latent-axis blocks.
///
/// This is structured L¹ on group L² norms: per-element L¹ can zero isolated
/// coefficients, ARD applies per-axis L² precision, while block sparsity
/// zeroes whole semantic axis groups together. It pairs directly with
/// `LatentIdMode::AuxPriorDimSelection` when the auxiliary prior supplies
/// class-specific active group subsets.
#[derive(Debug, Clone)]
pub struct BlockSparsityPenalty {
    pub target: PsiSlice,
    pub groups: Vec<Vec<usize>>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    /// Yuan-Lin group-size normalization uses the latent-axis group cardinality
    /// `sqrt(|g|)`; `n_eff` repeats each group across rows inside its norm and
    /// does not participate in this normalization.
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl BlockSparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        groups: Vec<Vec<usize>>,
        weight: f64,
        n_eff: usize,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("BlockSparsityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "BlockSparsityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("BlockSparsityPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "BlockSparsityPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "BlockSparsityPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "BlockSparsityPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "BlockSparsityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if groups.is_empty() {
            return Err("BlockSparsityPenalty::new requires at least one group".to_string());
        }
        let mut seen = vec![false; latent_dim];
        for (group_idx, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "BlockSparsityPenalty::new groups[{group_idx}] must not be empty"
                ));
            }
            for &axis in group {
                if axis >= latent_dim {
                    return Err(format!(
                        "BlockSparsityPenalty::new groups[{group_idx}] axis {axis} exceeds latent_dim {latent_dim}"
                    ));
                }
                if seen[axis] {
                    return Err(format!(
                        "BlockSparsityPenalty::new axis {axis} appears in more than one group"
                    ));
                }
                seen[axis] = true;
            }
        }
        for (axis, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "BlockSparsityPenalty::new groups must partition latent axes; missing axis {axis}"
                ));
            }
        }
        Ok(Self {
            target,
            groups,
            weight,
            n_eff,
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn group_size_factor(group: &[usize]) -> f64 {
        (group.len() as f64).sqrt()
    }

    fn group_norm(&self, t: ArrayView2<'_, f64>, group: &[usize]) -> f64 {
        let mut norm2 = 0.0;
        for n in 0..t.nrows() {
            for &axis in group {
                let x = t[[n, axis]];
                norm2 += x * x;
            }
        }
        (norm2 + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            for n in 0..t.nrows() {
                for &axis in group {
                    let x = t[[n, axis]];
                    out[n * t.ncols() + axis] = factor * (inv_s - x * x * inv_s3);
                }
            }
        }
        out
    }

    /// Materialize the group-lasso Hessian for small-block spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            for row1 in 0..t.nrows() {
                for &col1 in group {
                    let i = row1 * d + col1;
                    let x_i = t[[row1, col1]];
                    for row2 in 0..t.nrows() {
                        for &col2 in group {
                            let j = row2 * d + col2;
                            let mut entry = -x_i * t[[row2, col2]] * inv_s3;
                            if i == j {
                                entry += inv_s;
                            }
                            dense[[i, j]] = factor * entry;
                        }
                    }
                }
            }
        }
        dense
    }
}


impl AnalyticPenalty for BlockSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for group in &self.groups {
            acc += Self::group_size_factor(group) * self.group_norm(t.view(), group);
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for group in &self.groups {
            let s = self.group_norm(t.view(), group);
            let factor = weight * Self::group_size_factor(group) / s;
            for n in 0..t.nrows() {
                for &axis in group {
                    grad[[n, axis]] = factor * t[[n, axis]];
                }
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros(t.dim());
        for group in &self.groups {
            let factor = weight * Self::group_size_factor(group);
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            let mut inner = 0.0;
            for n in 0..t.nrows() {
                for &axis in group {
                    inner += t[[n, axis]] * v_mat[[n, axis]];
                }
            }
            for n in 0..t.nrows() {
                for &axis in group {
                    out[[n, axis]] =
                        factor * (v_mat[[n, axis]] * inv_s - t[[n, axis]] * inner * inv_s3);
                }
            }
        }
        Self::flatten_matrix(&out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "block_sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Mechanism-sparsity penalty
// ---------------------------------------------------------------------------

/// Per-latent group-lasso sparsity over decoder feature groups.
#[derive(Debug, Clone)]
pub struct MechanismSparsityPenalty {
    pub target: PsiSlice,
    pub feature_groups: Vec<Vec<usize>>,
    pub weight: f64,
    pub smoothing_eps: f64,
    pub n_eff: f64,
    pub weight_schedule: Option<Arc<ScalarWeightSchedule>>,
    pub learnable_weight: bool,
    pub rho_index: usize,
}


impl MechanismSparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        feature_groups: Vec<Vec<usize>>,
        weight: f64,
        smoothing_eps: f64,
        n_eff: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("MechanismSparsityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if !(n_eff.is_finite() && n_eff > 0.0) {
            return Err(format!(
                "MechanismSparsityPenalty::new requires finite n_eff > 0, got {n_eff}"
            ));
        }
        if feature_groups.is_empty() {
            return Err(
                "MechanismSparsityPenalty::new requires at least one feature group".to_string(),
            );
        }
        let latent_dim = target.latent_dim.ok_or_else(|| {
            "MechanismSparsityPenalty::new requires target.latent_dim".to_string()
        })?;
        if latent_dim == 0 {
            return Err("MechanismSparsityPenalty::new requires latent_dim > 0".to_string());
        }
        let p_features = Self::validate_feature_groups(&feature_groups)?;
        let expected_len = latent_dim.checked_mul(p_features).ok_or_else(|| {
            "MechanismSparsityPenalty::new target shape overflows usize".to_string()
        })?;
        if target.len() != expected_len {
            return Err(format!(
                "MechanismSparsityPenalty::new target length {} does not match latent_dim {} × feature_count {}",
                target.len(),
                latent_dim,
                p_features
            ));
        }
        Ok(Self {
            target,
            feature_groups,
            weight,
            smoothing_eps,
            n_eff,
            weight_schedule: None,
            learnable_weight,
            rho_index: 0,
        })
    }

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(Arc::new(schedule));
        self
    }

    fn validate_feature_groups(feature_groups: &[Vec<usize>]) -> Result<usize, String> {
        let mut max_feature = None::<usize>;
        for (group_idx, group) in feature_groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "MechanismSparsityPenalty::new feature_groups[{group_idx}] must not be empty"
                ));
            }
            for &feature in group {
                max_feature = Some(max_feature.map_or(feature, |current| current.max(feature)));
            }
        }
        let p_features = max_feature
            .and_then(|feature| feature.checked_add(1))
            .ok_or_else(|| {
                "MechanismSparsityPenalty::new feature shape overflows usize".to_string()
            })?;
        let mut seen = vec![false; p_features];
        for (group_idx, group) in feature_groups.iter().enumerate() {
            for &feature in group {
                if seen[feature] {
                    return Err(format!(
                        "MechanismSparsityPenalty::new feature {feature} appears in more than one group"
                    ));
                }
                seen[feature] = true;
            }
            for &feature in group {
                if feature >= p_features {
                    return Err(format!(
                        "MechanismSparsityPenalty::new feature_groups[{group_idx}] feature {feature} exceeds feature_count {p_features}"
                    ));
                }
            }
        }
        for (feature, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "MechanismSparsityPenalty::new feature_groups must partition features; missing feature {feature}"
                ));
            }
        }
        Ok(p_features)
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self) -> Option<usize> {
        self.target.latent_dim.filter(|&d| d > 0)
    }

    fn feature_count(&self) -> Option<usize> {
        let d = self.latent_dim()?;
        if !self.target.len().is_multiple_of(d) {
            return None;
        }
        Some(self.target.len() / d)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        if self.target.range.end > target.len() {
            return None;
        }
        let d = self.latent_dim()?;
        let p = self.feature_count()?;
        let local = target.slice_move(ndarray::s![self.target.range.start..self.target.range.end]);
        local.into_shape_with_order((d, p)).ok()
    }

    fn group_size_factor(group: &[usize]) -> f64 {
        (group.len() as f64).sqrt()
    }

    fn group_norm(&self, w: ArrayView2<'_, f64>, latent: usize, group: &[usize]) -> f64 {
        let mut norm2 = 0.0;
        for &feature in group {
            let x = w[[latent, feature]];
            norm2 += x * x;
        }
        (norm2 + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                for &feature in group {
                    let x = w[[latent, feature]];
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor * (inv_s - x * x * inv_s3);
                }
            }
        }
        out
    }

    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(w) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                for &feature_i in group {
                    let i = self.target.range.start + latent * p + feature_i;
                    let x_i = w[[latent, feature_i]];
                    for &feature_j in group {
                        let j = self.target.range.start + latent * p + feature_j;
                        let mut entry = -x_i * w[[latent, feature_j]] * inv_s3;
                        if i == j {
                            entry += inv_s;
                        }
                        dense[[i, j]] = factor * entry;
                    }
                }
            }
        }
        dense
    }
}


impl AnalyticPenalty for MechanismSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Beta
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(w) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                acc += Self::group_size_factor(group) * self.group_norm(w.view(), latent, group);
            }
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let s = self.group_norm(w.view(), latent, group);
                let factor = weight * Self::group_size_factor(group) / s;
                for &feature in group {
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor * w[[latent, feature]];
                }
            }
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(w) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let p = w.ncols();
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for latent in 0..w.nrows() {
            for group in &self.feature_groups {
                let factor = weight * Self::group_size_factor(group);
                let s = self.group_norm(w.view(), latent, group);
                let inv_s = 1.0 / s;
                let inv_s3 = inv_s * inv_s * inv_s;
                let mut inner = 0.0;
                for &feature in group {
                    inner += w[[latent, feature]] * v_mat[[latent, feature]];
                }
                for &feature in group {
                    let idx = self.target.range.start + latent * p + feature;
                    out[idx] = factor
                        * (v_mat[[latent, feature]] * inv_s
                            - w[[latent, feature]] * inner * inv_s3);
                }
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "mechanism_sparsity"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.weight_schedule.as_mut() {
            let schedule = Arc::make_mut(schedule);
            self.weight = schedule.current_weight(iter);
            schedule.iter_count = iter + 1;
        }
    }
}


// ---------------------------------------------------------------------------
// Row-precision prior penalty
// ---------------------------------------------------------------------------

/// Fixed zero-mean Gaussian row-precision prior on the latent block.
///
/// Evaluates the row-wise precision energy `½ μ Σ_n t_nᵀ Λ_n t_n`, with the
/// `ρ`-dependent Gaussian precision normalizer when `μ` is learnable. Callers
/// pass one positive-definite precision matrix per row. This is not the iVAE
/// conditional-mean gauge `½ μ ||t - h(u)||²`; use `LatentIdMode::AuxPrior`
/// for the ridge/linear projection-residual gauge.
#[derive(Debug, Clone)]
pub struct RowPrecisionPriorPenalty {
    pub lambda_per_row: Array3<f64>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl RowPrecisionPriorPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        lambda_per_row: Array3<f64>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("RowPrecisionPriorPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "RowPrecisionPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("RowPrecisionPriorPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "RowPrecisionPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "RowPrecisionPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (lambda_n, lambda_rows, lambda_cols) = lambda_per_row.dim();
        if lambda_n != n_eff || lambda_rows != latent_dim || lambda_cols != latent_dim {
            return Err(format!(
                "RowPrecisionPriorPenalty::new lambda_per_row shape must be ({n_eff}, {latent_dim}, {latent_dim}), got ({lambda_n}, {lambda_rows}, {lambda_cols})"
            ));
        }
        for n in 0..n_eff {
            let mut matrix = Array2::<f64>::zeros((latent_dim, latent_dim));
            for i in 0..latent_dim {
                for j in 0..latent_dim {
                    let value = lambda_per_row[[n, i, j]];
                    if !value.is_finite() {
                        return Err(format!(
                            "RowPrecisionPriorPenalty::new lambda_per_row[{n},{i},{j}] must be finite"
                        ));
                    }
                    let transpose = lambda_per_row[[n, j, i]];
                    if (value - transpose).abs() >= 1.0e-10 {
                        return Err(format!(
                            "RowPrecisionPriorPenalty::new lambda_per_row[{n}] must be symmetric; |Λ[{i},{j}] - Λ[{j},{i}]| = {:.3e}",
                            (value - transpose).abs()
                        ));
                    }
                    matrix[[i, j]] = value;
                }
            }
            let (evals, _) = matrix.eigh(Side::Lower).map_err(|err| {
                format!("RowPrecisionPriorPenalty::new lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            let min_eval = evals.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
            if !(min_eval.is_finite() && min_eval > 0.0) {
                return Err(format!(
                    "RowPrecisionPriorPenalty::new lambda_per_row[{n}] must be positive definite; minimum eigenvalue {min_eval:.3e}"
                ));
            }
        }
        Ok(Self {
            lambda_per_row,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                out[n * t.ncols() + i] = weight * self.lambda_per_row[[n, i, i]];
            }
        }
        out
    }

    /// Materialize the row-block-diagonal Hessian for exact spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n_total, n_total));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for n in 0..t.nrows() {
            for i in 0..d {
                let row = n * d + i;
                for j in 0..d {
                    dense[[row, n * d + j]] = weight * self.lambda_per_row[[n, i, j]];
                }
            }
        }
        dense
    }

    pub fn log_det_plus_lambda_i(
        &self,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "RowPrecisionPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let (n_obs, d, _) = self.lambda_per_row.dim();
        let weight = self.resolved_weight(rho);
        let mut sum = 0.0;
        for n in 0..n_obs {
            let mut matrix = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                for j in 0..d {
                    matrix[[i, j]] = self.lambda_per_row[[n, i, j]];
                }
            }
            let (evals, _) = matrix.eigh(Side::Lower).map_err(|err| {
                format!("RowPrecisionPriorPenalty::log_det_plus_lambda_i lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            for &eval in evals.iter() {
                let shifted = weight * eval + lambda;
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "RowPrecisionPriorPenalty::log_det_plus_lambda_i non-positive shifted eigenvalue {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}


impl AnalyticPenalty for RowPrecisionPriorPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut row_dot = 0.0;
                for j in 0..t.ncols() {
                    row_dot += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                acc += t[[n, i]] * row_dot;
            }
        }
        let weight = self.resolved_weight(rho);
        let log_weight_normalizer = -0.5 * target.len() as f64 * weight.ln();
        0.5 * weight * acc + log_weight_normalizer
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut acc = 0.0;
                for j in 0..t.ncols() {
                    acc += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                grad[[n, i]] = weight * acc;
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let Some(t) = self.target_matrix(target) else {
            return Some(Array1::<f64>::zeros(target.len()));
        };
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                for j in 0..t.ncols() {
                    if i != j && self.lambda_per_row[[n, i, j]] != 0.0 {
                        return None;
                    }
                }
            }
        }
        Some(self.diag_target(target, rho))
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros(t.dim());
        for n in 0..v_mat.nrows() {
            for i in 0..v_mat.ncols() {
                let mut acc = 0.0;
                for j in 0..v_mat.ncols() {
                    acc += self.lambda_per_row[[n, i, j]] * v_mat[[n, j]];
                }
                out[[n, i]] = weight * acc;
            }
        }
        Self::flatten_matrix(&out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(1);
        };
        let mut quad = 0.0;
        for n in 0..t.nrows() {
            for i in 0..t.ncols() {
                let mut row_dot = 0.0;
                for j in 0..t.ncols() {
                    row_dot += self.lambda_per_row[[n, i, j]] * t[[n, j]];
                }
                quad += t[[n, i]] * row_dot;
            }
        }
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = 0.5 * weight * quad - 0.5 * target.len() as f64;
        out
    }

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "row_precision_prior"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// iVAE ridge conditional-mean gauge penalty
// ---------------------------------------------------------------------------

/// iVAE conditional-mean gauge penalty on the latent block.
///
/// Khemakhem et al. (2020) identify nonlinear ICA/iVAE latent factors from
/// auxiliary-variable variation up to an affine transform under sufficient
/// variation in `u`. This penalty implements the conditional-mean side of that
/// signal as `0.5 * μ * ||t - U(UᵀU + εI)⁻¹Uᵀt||²`, penalizing only the
/// component of each latent axis not explained by a ridge linear fit to `u`.
#[derive(Debug, Clone)]
pub struct IvaeRidgeMeanGauge {
    pub aux: Array2<f64>,
    pub ridge_inv: Array2<f64>,
    pub ridge_eps: f64,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl IvaeRidgeMeanGauge {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        aux: Array2<f64>,
        ridge_eps: f64,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("IvaeRidgeMeanGauge::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(ridge_eps.is_finite() && ridge_eps > 0.0) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new requires finite ridge_eps > 0, got {ridge_eps}"
            ));
        }
        if n_eff == 0 {
            return Err("IvaeRidgeMeanGauge::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "IvaeRidgeMeanGauge::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "IvaeRidgeMeanGauge::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (aux_n, aux_dim) = aux.dim();
        if aux_n != n_eff {
            return Err(format!(
                "IvaeRidgeMeanGauge::new aux rows must equal n_eff {n_eff}, got {aux_n}"
            ));
        }
        if aux_dim == 0 {
            return Err("IvaeRidgeMeanGauge::new requires aux dimension > 0".to_string());
        }
        for (idx, &value) in aux.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("IvaeRidgeMeanGauge::new aux[{idx}] must be finite"));
            }
        }
        let mut gram = Array2::<f64>::zeros((aux_dim, aux_dim));
        for n in 0..n_eff {
            for i in 0..aux_dim {
                for j in 0..aux_dim {
                    gram[[i, j]] += aux[[n, i]] * aux[[n, j]];
                }
            }
        }
        for i in 0..aux_dim {
            gram[[i, i]] += ridge_eps;
        }
        let ridge_inv = Self::invert_spd_gram(gram)?;
        Ok(Self {
            aux,
            ridge_inv,
            ridge_eps,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn invert_spd_gram(gram: Array2<f64>) -> Result<Array2<f64>, String> {
        let q = gram.nrows();
        let (evals, evecs) = gram.eigh(Side::Lower).map_err(|err| {
            format!("IvaeRidgeMeanGauge::new ridge Gram eigendecomposition failed: {err}")
        })?;
        let mut inv = Array2::<f64>::zeros((q, q));
        for k in 0..q {
            let eval = evals[k];
            if !(eval.is_finite() && eval > 0.0) {
                return Err(format!(
                    "IvaeRidgeMeanGauge::new ridge Gram must be positive definite; eigenvalue {k} is {eval:.3e}"
                ));
            }
            let inv_eval = 1.0 / eval;
            for i in 0..q {
                for j in 0..q {
                    inv[[i, j]] += evecs[[i, k]] * evecs[[j, k]] * inv_eval;
                }
            }
        }
        Ok(inv)
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn projected_matrix(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let q = self.aux.ncols();
        let d = x.ncols();
        let mut u_t_x = Array2::<f64>::zeros((q, d));
        for n in 0..x.nrows() {
            for i in 0..q {
                let u_ni = self.aux[[n, i]];
                for a in 0..d {
                    u_t_x[[i, a]] += u_ni * x[[n, a]];
                }
            }
        }
        let mut coeff = Array2::<f64>::zeros((q, d));
        for i in 0..q {
            for j in 0..q {
                let inv_ij = self.ridge_inv[[i, j]];
                for a in 0..d {
                    coeff[[i, a]] += inv_ij * u_t_x[[j, a]];
                }
            }
        }
        let mut projected = Array2::<f64>::zeros(x.dim());
        for n in 0..x.nrows() {
            for i in 0..q {
                let u_ni = self.aux[[n, i]];
                for a in 0..d {
                    projected[[n, a]] += u_ni * coeff[[i, a]];
                }
            }
        }
        projected
    }

    fn residual_matrix(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        let projected = self.projected_matrix(x);
        let mut residual = Array2::<f64>::zeros(x.dim());
        for n in 0..x.nrows() {
            for a in 0..x.ncols() {
                residual[[n, a]] = x[[n, a]] - projected[[n, a]];
            }
        }
        residual
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            let mut p_nn = 0.0;
            for i in 0..self.aux.ncols() {
                for j in 0..self.aux.ncols() {
                    p_nn += self.aux[[n, i]] * self.ridge_inv[[i, j]] * self.aux[[n, j]];
                }
            }
            let diag = weight * (1.0 - p_nn);
            for a in 0..t.ncols() {
                out[n * t.ncols() + a] = diag;
            }
        }
        out
    }

    /// Materialize `μ(I - U(UᵀU + εI)⁻¹Uᵀ)` repeated per latent axis.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n_total, n_total));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for n in 0..t.nrows() {
            for m in 0..t.nrows() {
                let mut p_nm = 0.0;
                for i in 0..self.aux.ncols() {
                    for j in 0..self.aux.ncols() {
                        p_nm += self.aux[[n, i]] * self.ridge_inv[[i, j]] * self.aux[[m, j]];
                    }
                }
                let entry = weight * (if n == m { 1.0 } else { 0.0 } - p_nm);
                for a in 0..d {
                    dense[[n * d + a, m * d + a]] = entry;
                }
            }
        }
        dense
    }
}


impl AnalyticPenalty for IvaeRidgeMeanGauge {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let residual = self.residual_matrix(t.view());
        let mut acc = 0.0;
        for n in 0..t.nrows() {
            for a in 0..t.ncols() {
                acc += t[[n, a]] * residual[[n, a]];
            }
        }
        let weight = self.resolved_weight(rho);
        let mut value = 0.5 * weight * acc;
        if self.learnable_weight {
            value -= 0.5 * target.len() as f64 * weight.ln();
        }
        value
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = self.residual_matrix(t.view());
        for value in grad.iter_mut() {
            *value *= weight;
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut hv = self.residual_matrix(v_mat.view());
        for value in hv.iter_mut() {
            *value *= weight;
        }
        Self::flatten_matrix(&hv)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        if self.target_matrix(target).is_none() {
            return Array1::<f64>::zeros(1);
        }
        let mut out = Array1::<f64>::zeros(1);
        let weight = self.resolved_weight(rho);
        out[self.rho_index] =
            self.value(target, rho) + 0.5 * target.len() as f64 * (weight.ln() - 1.0);
        out
    }

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "ivae_ridge_mean_gauge"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Parametric row-precision prior penalty
// ---------------------------------------------------------------------------

/// Parametric zero-mean Gaussian row-precision prior on the latent block.
///
/// Uses a diagonal precision
/// `λ_k(u_n) = exp(log_alpha_k) + softplus(raw_beta_k) ||u_n - μ_k||²`.
/// REML may learn that conditional precision map, including the Gaussian
/// precision normalizer derivatives. This is not a learnable conditional
/// mean map and does not implement the iVAE projection-residual gauge.
#[derive(Debug, Clone)]
pub struct ParametricRowPrecisionPriorPenalty {
    pub aux: Array2<f64>,
    pub log_alpha: Array1<f64>,
    pub raw_beta: Array1<f64>,
    pub mu: Array2<f64>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[weight_rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub target: PsiSlice,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl ParametricRowPrecisionPriorPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        aux: Array2<f64>,
        log_alpha: Array1<f64>,
        raw_beta: Array1<f64>,
        mu: Array2<f64>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires a non-empty target".to_string(),
            );
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("ParametricRowPrecisionPriorPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if latent_dim == 0 {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires latent_dim > 0".to_string(),
            );
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "ParametricRowPrecisionPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (aux_n, aux_dim) = aux.dim();
        if aux_n != n_eff {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new aux rows must equal n_eff {n_eff}, got {aux_n}"
            ));
        }
        if aux_dim == 0 {
            return Err(
                "ParametricRowPrecisionPriorPenalty::new requires aux dimension > 0".to_string(),
            );
        }
        if log_alpha.len() != latent_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new log_alpha length must equal latent_dim {latent_dim}, got {}",
                log_alpha.len()
            ));
        }
        if raw_beta.len() != latent_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new raw_beta length must equal latent_dim {latent_dim}, got {}",
                raw_beta.len()
            ));
        }
        let (mu_rows, mu_cols) = mu.dim();
        if mu_rows != latent_dim || mu_cols != aux_dim {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::new mu shape must be ({latent_dim}, {aux_dim}), got ({mu_rows}, {mu_cols})"
            ));
        }
        for (idx, &value) in aux.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new aux[{idx}] must be finite"
                ));
            }
        }
        for k in 0..latent_dim {
            let log_alpha_k = log_alpha[k];
            if !log_alpha_k.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new log_alpha[{k}] must be finite"
                ));
            }
            let alpha_k = log_alpha_k.exp();
            if !(alpha_k.is_finite() && alpha_k > 0.0) {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new exp(log_alpha[{k}]) must be finite and > 0"
                ));
            }
            let raw_beta_k = raw_beta[k];
            if !raw_beta_k.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new raw_beta[{k}] must be finite"
                ));
            }
            let beta_k = crate::linalg::utils::stable_softplus(raw_beta_k);
            if !(beta_k.is_finite() && beta_k >= 0.0) {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new softplus(raw_beta[{k}]) must be finite and >= 0"
                ));
            }
        }
        for (idx, &value) in mu.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricRowPrecisionPriorPenalty::new mu[{idx}] must be finite"
                ));
            }
        }
        Ok(Self {
            aux,
            log_alpha,
            raw_beta,
            mu,
            weight,
            n_eff,
            learnable_weight,
            target,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn log_alpha_offset(&self) -> usize {
        0
    }

    fn raw_beta_offset(&self) -> usize {
        self.log_alpha.len()
    }

    fn mu_offset(&self) -> usize {
        self.log_alpha.len() + self.raw_beta.len()
    }

    fn weight_offset(&self) -> usize {
        self.mu_offset() + self.mu.len()
    }

    fn active_log_alpha(&self, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.log_alpha[k] + rho[self.log_alpha_offset() + k]
    }

    fn active_raw_beta(&self, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.raw_beta[k] + rho[self.raw_beta_offset() + k]
    }

    fn active_mu(&self, k: usize, a: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.mu[[k, a]] + rho[self.mu_offset() + k * self.aux.ncols() + a]
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.weight_offset()])
        } else {
            self.weight
        }
    }

    fn lambda_at(&self, n: usize, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = stable_exp_log_precision(self.active_log_alpha(k, rho));
        let beta = crate::linalg::utils::stable_softplus(self.active_raw_beta(k, rho));
        MIN_CONDITIONAL_PRECISION + alpha + beta * self.dist2(n, k, rho)
    }

    fn dist2(&self, n: usize, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        let mut r2 = 0.0;
        for a in 0..self.aux.ncols() {
            let delta = self.aux[[n, a]] - self.active_mu(k, a, rho);
            r2 += delta * delta;
        }
        r2
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                out[n * t.ncols() + k] = weight * self.lambda_at(n, k, rho);
            }
        }
        out
    }

    /// Materialize the row-block-diagonal Hessian for exact spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_total = target.len();
        let diag = self.diag_target(target, rho);
        let mut dense = Array2::<f64>::zeros((n_total, n_total));
        for i in 0..n_total {
            dense[[i, i]] = diag[i];
        }
        dense
    }

    pub fn log_det_plus_lambda_i(
        &self,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "ParametricRowPrecisionPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let weight = self.resolved_weight(rho);
        let mut sum = 0.0;
        for n in 0..self.n_eff {
            for k in 0..self.log_alpha.len() {
                let shifted = lambda + weight * self.lambda_at(n, k, rho);
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "ParametricRowPrecisionPriorPenalty::log_det_plus_lambda_i non-positive shifted diagonal {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}


impl AnalyticPenalty for ParametricRowPrecisionPriorPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let weight = self.resolved_weight(rho);
        let mut quadratic = 0.0;
        let mut log_det = 0.0;
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                let lambda = self.lambda_at(n, k, rho);
                quadratic += lambda * t[[n, k]] * t[[n, k]];
                log_det += (weight * lambda).ln();
            }
        }
        0.5 * weight * quadratic - 0.5 * log_det
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                grad[[n, k]] = weight * self.lambda_at(n, k, rho) * t[[n, k]];
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        Some(self.diag_target(target, rho))
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let diag = self.diag_target(target, rho);
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(self.rho_count());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(self.rho_count());
        let d = t.ncols();
        let du = self.aux.ncols();
        let mut grad_weight_direct = 0.0;
        for k in 0..d {
            let log_alpha = self.active_log_alpha(k, rho);
            let alpha = stable_exp_log_precision(log_alpha);
            let raw_beta = self.active_raw_beta(k, rho);
            let beta = crate::linalg::utils::stable_softplus(raw_beta);
            let beta_jac = crate::linalg::utils::stable_logistic(raw_beta);
            let mut grad_alpha_direct = 0.0;
            let mut grad_beta_direct = 0.0;
            let mut grad_mu_direct = vec![0.0_f64; du];
            for n in 0..t.nrows() {
                let tk = t[[n, k]];
                let sq = tk * tk;
                let r2 = self.dist2(n, k, rho);
                let lambda = alpha + beta * r2;
                let precision_score = 0.5 * weight * sq - 0.5 / lambda;
                grad_weight_direct += 0.5 * weight * lambda * sq;
                grad_alpha_direct += precision_score;
                grad_beta_direct += precision_score * r2;
                for a in 0..du {
                    let delta = self.aux[[n, a]] - self.active_mu(k, a, rho);
                    grad_mu_direct[a] += -2.0 * precision_score * beta * delta;
                }
            }
            out[self.log_alpha_offset() + k] = grad_alpha_direct * alpha;
            out[self.raw_beta_offset() + k] = grad_beta_direct * beta_jac;
            for a in 0..du {
                out[self.mu_offset() + k * du + a] = grad_mu_direct[a];
            }
        }
        if self.learnable_weight {
            out[self.weight_offset()] = grad_weight_direct - 0.5 * target.len() as f64;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.log_alpha.len()
            + self.raw_beta.len()
            + self.mu.len()
            + usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "parametric_row_precision_prior"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// SCAD / MCP concave sparsity penalty
// ---------------------------------------------------------------------------

/// Concave alternative to smoothed-L¹ sparsity with less bias on large signals.
/// MCP (Zhang 2010) and SCAD (Fan-Li 2001) keep strong shrinkage near zero but
/// taper the derivative to zero for large coefficients; `gamma` controls
/// concavity, and `gamma -> infinity` recovers the L¹ limit. Fan-Li recommend
/// `gamma = 3.7` for SCAD.
///
/// `SparsityPenalty` uses Huber-smoothed L¹, `Σ_j sqrt(t_j² + ε²)`, whose
/// gradient magnitude stays constant outside the Huber region. That constant
/// pull over-shrinks moderate true signals. SCAD/MCP flatten the gradient for
/// large coefficients, which gives less-biased estimates while still shrinking
/// near-zero noise; paired with row-precision priors, the precision field
/// anchors which axes are active and this penalty fits their magnitudes without
/// L¹'s constant pull.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyConcavity {
    Mcp,
    Scad,
}


/// Element-wise SCAD/MCP family penalty on a row-major latent block.
///
/// Lives on the extension-coordinate tier. The target is a row-major
/// `(n_eff, d)` latent block, but the penalty is coordinate-separable:
/// `P(T) = Σ_i p(r_i; λ, γ)` with `r_i = sqrt(T_i² + ε²)` and
/// `λ = weight · exp(ρ)` when the weight is learnable.
///
/// MCP uses
///
/// ```text
///   p_MCP(r) = λr - (r² - ε²)/(2γ),       r ≤ γλ
///            = γλ²/2 + ε²/(2γ),          r > γλ,
/// ```
///
/// so the shrinkage derivative tapers linearly to zero at `γλ`. SCAD uses the
/// Fan-Li three-region derivative: L¹ shrinkage up to `λ`, a linear taper on
/// `(λ, γλ]`, and zero derivative beyond `γλ`.
///
/// In the SAE objective this is the nonconvex sparsity prior for latent
/// extension-coordinate amplitudes: it still suppresses near-zero activations
/// but avoids L¹'s bias on large coefficients that should remain active.
///
/// Gotchas:
///
/// * The exact Hessian diagonal can be negative in the taper region. This is a
///   row-block-diagonal penalty, but not a PSD curvature source.
/// * `γ` must be `> 1` for MCP and `> 2` for SCAD. Larger values approach
///   smoothed L¹; smaller valid values make the taper more aggressive.
/// * `ε > 0` smooths the absolute-value cusp and slightly shifts the effective
///   cutoffs because the piecewise rules use `r = sqrt(t² + ε²)`.
#[derive(Debug, Clone)]
pub struct ScadMcpPenalty {
    pub target: PsiSlice,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    /// Concavity parameter. Larger values approach smoothed L¹.
    pub gamma: f64,
    pub smoothing_eps: f64,
    pub variant: PenaltyConcavity,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl ScadMcpPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        weight: f64,
        n_eff: usize,
        gamma: f64,
        smoothing_eps: f64,
        variant: PenaltyConcavity,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("ScadMcpPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("ScadMcpPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "ScadMcpPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff
                .checked_mul(expected_dim)
                .ok_or_else(|| "ScadMcpPenalty::new target shape overflows usize".to_string())?;
            if expected != target.len() {
                return Err(format!(
                    "ScadMcpPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        match variant {
            PenaltyConcavity::Mcp if !(gamma.is_finite() && gamma > 1.0) => {
                return Err(format!(
                    "ScadMcpPenalty::new MCP requires finite gamma > 1, got {gamma}"
                ));
            }
            PenaltyConcavity::Scad if !(gamma.is_finite() && gamma > 2.0) => {
                return Err(format!(
                    "ScadMcpPenalty::new SCAD requires finite gamma > 2, got {gamma}"
                ));
            }
            PenaltyConcavity::Mcp | PenaltyConcavity::Scad => {}
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            target,
            weight,
            n_eff,
            gamma,
            smoothing_eps,
            variant,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn smooth_abs(&self, t: f64) -> f64 {
        (t * t + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    fn value_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        match self.variant {
            PenaltyConcavity::Mcp => {
                let cutoff = self.gamma * weight;
                if r <= cutoff {
                    weight * r
                        - (r * r - self.smoothing_eps * self.smoothing_eps) / (2.0 * self.gamma)
                } else {
                    0.5 * self.gamma * weight * weight
                        + self.smoothing_eps * self.smoothing_eps / (2.0 * self.gamma)
                }
            }
            PenaltyConcavity::Scad => {
                let cutoff1 = weight;
                let cutoff2 = self.gamma * weight;
                if r <= cutoff1 {
                    weight * r
                } else if r <= cutoff2 {
                    (-r * r + 2.0 * self.gamma * weight * r - weight * weight)
                        / (2.0 * (self.gamma - 1.0))
                } else {
                    0.5 * (self.gamma + 1.0) * weight * weight
                }
            }
        }
    }

    fn grad_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * t / r - t / self.gamma
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * t / r
                } else if r <= self.gamma * weight {
                    self.gamma * weight * t / (denom * r) - t / denom
                } else {
                    0.0
                }
            }
        }
    }

    fn hess_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * eps2 / (r * r * r) - 1.0 / self.gamma
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * eps2 / (r * r * r)
                } else if r <= self.gamma * weight {
                    self.gamma * weight * eps2 / (denom * r * r * r) - 1.0 / denom
                } else {
                    0.0
                }
            }
        }
    }

    /// Diagonal of the **PSD majorizer** for a single coordinate.
    ///
    /// SCAD/MCP are nonconvex: within their active region the penalty splits
    /// into a convex smoothed-ℓ¹ part (`λr`, resp. `γλr/(γ−1)`) and a concave
    /// quadratic taper (`−t²/(2γ)`, resp. `−t²/(2(γ−1))`). The exact Hessian
    /// [`Self::hess_one`] adds the concave constant (`−1/γ`, `−1/(γ−1)`) and is
    /// therefore negative across most of the active region.
    ///
    /// The MM/LLA majorizer keeps the convex part's reweighted-ℓ² curvature and
    /// majorizes the concave quadratic by its tangent line (zero curvature):
    /// this is exactly [`Self::hess_one`] with the concave constant dropped.
    /// Beyond the active cutoff the penalty is flat (Hessian `0`), so the
    /// majorizer is `0`. The result satisfies both legs of the trait contract:
    ///
    /// * `B ⪰ 0`: `λε²/r³ ≥ 0` and the constant `0` branch are nonnegative.
    /// * `B ⪰ ∂²P`: it exceeds the exact Hessian by exactly the dropped
    ///   concave constant (`1/γ` for MCP, `1/(γ−1)` for SCAD's middle region)
    ///   and equals it in the convex first SCAD region and the flat tail.
    fn psd_majorizer_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * eps2 / (r * r * r)
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * eps2 / (r * r * r)
                } else if r <= self.gamma * weight {
                    self.gamma * weight * eps2 / (denom * r * r * r)
                } else {
                    0.0
                }
            }
        }
    }

    fn grad_log_weight_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let d_p_d_weight = match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    r
                } else {
                    self.gamma * weight
                }
            }
            PenaltyConcavity::Scad => {
                if r <= weight {
                    r
                } else if r <= self.gamma * weight {
                    (self.gamma * r - weight) / (self.gamma - 1.0)
                } else {
                    (self.gamma + 1.0) * weight
                }
            }
        };
        weight * d_p_d_weight
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.hess_one(t, weight);
        }
        out
    }

    pub fn log_det_plus_lambda_i(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let diag = self.diag_target(target, rho);
        let mut sum = 0.0;
        for &entry in diag.iter() {
            let shifted = lambda + entry;
            if !(shifted.is_finite() && shifted > 0.0) {
                return Err(format!(
                    "ScadMcpPenalty::log_det_plus_lambda_i non-positive shifted diagonal {shifted:.3e}"
                ));
            }
            sum += shifted.ln();
        }
        Ok(sum)
    }
}


impl AnalyticPenalty for ScadMcpPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let weight = self.resolved_weight(rho);
        let mut acc = 0.0;
        for &t in target.iter() {
            acc += self.value_one(t, weight);
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.grad_one(t, weight);
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        Some(self.diag_target(target, rho))
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let diag = self.diag_target(target, rho);
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
    }

    /// PSD majorizer diagonal (see [`Self::psd_majorizer_one`]). SCAD/MCP are
    /// nonconvex, so this overrides the convex-only trait default — which would
    /// otherwise return the exact, negative [`Self::hessian_diag`] — with the
    /// reweighted-ℓ² MM surrogate. Coordinate-separable, so the inherited
    /// [`AnalyticPenalty::psd_majorizer_hvp`] correctly applies this as a
    /// diagonal operator.
    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.psd_majorizer_one(t, weight);
        }
        Some(out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let weight = self.resolved_weight(rho);
        let mut grad = 0.0;
        for &t in target.iter() {
            grad += self.grad_log_weight_one(t, weight);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = grad;
        out
    }

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "scad_mcp"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Block-orthogonality penalty
// ---------------------------------------------------------------------------

/// Between-block-only orthogonality on a row-major matrix-valued latent
/// block.
///
/// Lives on the extension-coordinate tier. Penalizes the squared Frobenius
/// norm of the between-block Gram matrices, where `T` is the row-major
/// `n_eff × latent_dim` view of the target slice and `groups` partitions
/// the latent axes into disjoint subsets:
///
/// ```text
///   P(T) = ½ · w · Σ_{g < h} ‖ T[:, group_g]^T T[:, group_h] ‖²_F
/// ```
///
/// Within-block structure is unconstrained: this penalty only pushes different
/// groups into mutually orthogonal subspaces. In the SAE objective it is the
/// block-level separability / gauge term for latent decompositions where known
/// or supervised coordinates should not leak into free coordinates.
///
/// Typical use: gauge-fixing a latent decomposition where one block has been
/// supervised (e.g. anchored to known coordinates) and a free block needs to
/// inhabit the orthogonal complement of that supervision. Pair with per-block
/// ARD or sparsity when you also want within-block axis selection.
///
/// Gotchas:
///
/// * `groups` must be a true partition of all latent axes: every axis appears
///   exactly once, and at least two groups are required.
/// * The Hessian is dense across rows and axes even though an exact diagonal is
///   available for diagnostics/preconditioning. Use the HVP for the full
///   Newton curvature.
#[derive(Debug, Clone)]
pub struct BlockOrthogonalityPenalty {
    pub target: PsiSlice,
    pub groups: Vec<Vec<usize>>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl BlockOrthogonalityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        groups: Vec<Vec<usize>>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("BlockOrthogonalityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "BlockOrthogonalityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("BlockOrthogonalityPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "BlockOrthogonalityPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "BlockOrthogonalityPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        if groups.len() < 2 {
            return Err("BlockOrthogonalityPenalty::new requires at least two groups".to_string());
        }
        let mut seen = vec![false; latent_dim];
        for (group_idx, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new groups[{group_idx}] must not be empty"
                ));
            }
            for &axis in group {
                if axis >= latent_dim {
                    return Err(format!(
                        "BlockOrthogonalityPenalty::new groups[{group_idx}] axis {axis} exceeds latent_dim {latent_dim}"
                    ));
                }
                if seen[axis] {
                    return Err(format!(
                        "BlockOrthogonalityPenalty::new axis {axis} appears in more than one group"
                    ));
                }
                seen[axis] = true;
            }
        }
        for (axis, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new groups must partition latent axes; missing axis {axis}"
                ));
            }
        }
        Ok(Self {
            target,
            groups,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn cross_gram(t: ArrayView2<'_, f64>, left: &[usize], right: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((left.len(), right.len()));
        for (li, &a) in left.iter().enumerate() {
            for (ri, &b) in right.iter().enumerate() {
                let mut s = 0.0;
                for n in 0..t.nrows() {
                    s += t[[n, a]] * t[[n, b]];
                }
                out[[li, ri]] = s;
            }
        }
        out
    }

    /// `out[li, ri] = Σ_n a[n, left[li]] · b[n, right[ri]]` — two-argument
    /// cross-gram used to assemble the directional derivative of `C_{gh}` in
    /// direction `v`:  `∂_v C_{gh}[gi, hi] = Σ_n {v[n, axes_g[gi]] · t[n, axes_h[hi]] + t[n, axes_g[gi]] · v[n, axes_h[hi]]}`.
    /// The `cross_gram` helper (single-input self-product) was previously
    /// (mis)used for both terms, but `cross_gram(v, h, g) + cross_gram(t, h, g)`
    /// is the unrelated quantity `(v⊗v) + (t⊗t)`, not `(v⊗t) + (t⊗v)`.
    fn mixed_cross_gram(
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        left: &[usize],
        right: &[usize],
    ) -> Array2<f64> {
        assert_eq!(a.nrows(), b.nrows(), "mixed_cross_gram row mismatch");
        let mut out = Array2::<f64>::zeros((left.len(), right.len()));
        for (li, &al) in left.iter().enumerate() {
            for (ri, &br) in right.iter().enumerate() {
                let mut s = 0.0;
                for n in 0..a.nrows() {
                    s += a[[n, al]] * b[[n, br]];
                }
                out[[li, ri]] = s;
            }
        }
        out
    }

    fn add_right_times_cross(
        out: &mut Array2<f64>,
        right: ArrayView2<'_, f64>,
        left_axes: &[usize],
        right_axes: &[usize],
        cross_right_left: ArrayView2<'_, f64>,
        factor: f64,
    ) {
        assert_eq!(cross_right_left.dim(), (right_axes.len(), left_axes.len()));
        for n in 0..out.nrows() {
            for (li, &left_axis) in left_axes.iter().enumerate() {
                let mut s = 0.0;
                for (ri, &right_axis) in right_axes.iter().enumerate() {
                    s += right[[n, right_axis]] * cross_right_left[[ri, li]];
                }
                out[[n, left_axis]] += factor * s;
            }
        }
    }

    fn hvp_with_precomputed_cross(
        &self,
        t: ArrayView2<'_, f64>,
        cross: &[Vec<Option<Array2<f64>>>],
        v: ArrayView2<'_, f64>,
        weight: f64,
    ) -> Array2<f64> {
        assert_eq!(v.dim(), t.dim(), "hvp matrix dimension mismatch");
        if v.dim() != t.dim() {
            return Array2::<f64>::zeros(t.dim());
        }
        let mut out = Array2::<f64>::zeros(t.dim());
        for g in 0..self.groups.len() {
            let group_g = &self.groups[g];
            for h in 0..self.groups.len() {
                if g == h {
                    continue;
                }
                let group_h = &self.groups[h];
                let c_hg = cross[h][g]
                    .as_ref()
                    .expect("between-block cross Gram must be precomputed");
                // Linear contribution: w · Σ_b C_{g,h}[i,b] · v[n, axes_h[b]] —
                // the C-direct piece of d/dv (∂P/∂t).
                Self::add_right_times_cross(&mut out, v, group_g, group_h, c_hg.view(), weight);

                // Directional derivative of C_{hg} in direction v:
                //   ∂_v C_{hg}[hi, gi] = Σ_n {v[n, axes_h[hi]] · t[n, axes_g[gi]]
                //                            + t[n, axes_h[hi]] · v[n, axes_g[gi]]}
                // = MixedCross(v, t, h, g) + MixedCross(t, v, h, g).
                // The earlier formulation used `cross_gram(v, h, g) +
                // cross_gram(t, h, g)`, which is `(v⊗v) + (t⊗t)` — quadratic in v
                // (resp. independent of v) and unrelated to the JVP. The bug made
                // the Hessian non-symmetric (it added a fixed `(t⊗t)`-driven term
                // to every column), violated the gradient/Hessian consistency
                // check that REML's spectral solve relies on, and the sibling
                // `OrthogonalityPenalty::hvp_with_precomputed_m` already uses the
                // correct `v_c · t_b + t_c · v_b` mixed pattern.
                let dv_h_g = Self::mixed_cross_gram(v, t, group_h, group_g);
                let tv_h_g = Self::mixed_cross_gram(t, v, group_h, group_g);
                let mut d_c_hg = dv_h_g;
                d_c_hg += &tv_h_g;
                Self::add_right_times_cross(&mut out, t, group_g, group_h, d_c_hg.view(), weight);
            }
        }
        out
    }

    fn precompute_cross(&self, t: ArrayView2<'_, f64>) -> Vec<Vec<Option<Array2<f64>>>> {
        let mut cross = vec![vec![None; self.groups.len()]; self.groups.len()];
        for g in 0..self.groups.len() {
            for h in 0..self.groups.len() {
                if g != h {
                    cross[g][h] = Some(Self::cross_gram(t, &self.groups[g], &self.groups[h]));
                }
            }
        }
        cross
    }

    /// Materialize the between-block orthogonality Hessian for small-block
    /// spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let cross = self.precompute_cross(t.view());
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            let Some(e_mat) = self.target_matrix(e.view()) else {
                return Array2::<f64>::zeros((n, n));
            };
            let col = self.hvp_with_precomputed_cross(t.view(), &cross, e_mat, weight);
            for i in 0..n {
                dense[[i, j]] = col[[i / t.ncols(), i % t.ncols()]];
            }
            e[j] = 0.0;
        }
        dense
    }
}


impl AnalyticPenalty for BlockOrthogonalityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for g in 0..self.groups.len() {
            for h in (g + 1)..self.groups.len() {
                let c = Self::cross_gram(t.view(), &self.groups[g], &self.groups[h]);
                for &v in c.iter() {
                    acc += v * v;
                }
            }
        }
        0.5 * self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let cross = self.precompute_cross(t.view());
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for g in 0..self.groups.len() {
            for h in 0..self.groups.len() {
                if g == h {
                    continue;
                }
                let c_hg = cross[h][g]
                    .as_ref()
                    .expect("between-block cross Gram must be precomputed");
                Self::add_right_times_cross(
                    &mut grad,
                    t.view(),
                    &self.groups[g],
                    &self.groups[h],
                    c_hg.view(),
                    weight,
                );
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let cross = self.precompute_cross(t.view());
        let hv = self.hvp_with_precomputed_cross(
            t.view(),
            &cross,
            v_mat.view(),
            self.resolved_weight(rho),
        );
        Self::flatten_matrix(&hv)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let t = self.target_matrix(target)?;
        let n_obs = t.nrows();
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut group_of = vec![usize::MAX; d];
        for (gi, group) in self.groups.iter().enumerate() {
            for &axis in group {
                group_of[axis] = gi;
            }
        }
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            let mut row_sq = 0.0_f64;
            let mut group_sq = vec![0.0_f64; self.groups.len()];
            for b in 0..d {
                let v = t[[n, b]];
                let v2 = v * v;
                row_sq += v2;
                group_sq[group_of[b]] += v2;
            }
            for a in 0..d {
                let g = group_of[a];
                out[n * d + a] = weight * (row_sq - group_sq[g]);
            }
        }
        Some(out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "block_orthogonality"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Decoder column-space incoherence penalty
// ---------------------------------------------------------------------------

/// Cross-atom decoder column-space incoherence, restricted to co-activating
/// atom pairs (issue #671).
///
/// Lives on the β tier and targets the flat SAE decoder coefficient block. The
/// β layout concatenates the per-atom decoder blocks in atom order: atom `k`
/// owns `M_k · p_out` coefficients, stored as
/// `β[off_k + a·p_out + o]` for basis row `a` and output feature `o`.
/// The stored block is `B_k ∈ ℝ^{M_k × p_out}` with rows `B_k[a, :]`
/// representing decoder directions in output space.
///
/// The penalty is the co-activation-masked cross-column-space overlap
///
/// ```text
///   P = ½ · w · Σ_{j<k} W[j,k] · ‖B_j B_k^T‖²_F,
///   W[j,k] = ½ · (coactivation[j,k] + coactivation[k,j]).
/// ```
///
/// `coactivation[j,k]` is the mean over observations of
/// `gate[n,j] · gate[n,k]`; pairs that never co-fire (`W[j,k] = 0`) contribute
/// nothing. In the SAE objective this is the separability lever: atoms that
/// are active on the same examples are discouraged from spanning the same
/// decoder output directions, while unrelated atoms are not pushed apart just
/// because they both exist in the dictionary.
///
/// The Hessian used here is the Gauss-Newton (positive-semidefinite) curvature
/// of the Frobenius objective in `C`, dropping the indefinite second-order term
/// in `C`. This keeps the β-tier Newton / PIRLS curvature block PSD, matching
/// the other quadratic-on-Gram penalties.
///
/// Gotchas:
///
/// * `block_sizes` are decoder basis-row counts `M_k`, not output widths;
///   every atom shares the same `p_out`. Stored decoder blocks are
///   `(M_k, p_out)`, so `B_j B_k^T` is the cross-Gram of decoder directions in
///   output space and remains well-defined for heterogeneous `M_k`.
/// * The descriptor path builds a placeholder penalty; live SAE wiring replaces
///   the co-activation matrix with the current mean gate products.
/// * Offsets are interpreted against the vector passed to this penalty. In the
///   SAE decoder-incoherence path the registered target slice is zero-based;
///   callers using an already sliced target view must keep that convention.
#[derive(Debug, Clone)]
pub struct DecoderIncoherencePenalty {
    pub target: PsiSlice,
    /// Per-atom decoder basis-function counts `M_k`. The atom blocks are laid
    /// out contiguously in β order; `Σ_k M_k·p_out == target.len()`.
    pub block_sizes: Vec<usize>,
    /// Output / feature dimension `p_out` (decoder column count, shared by all
    /// atoms).
    pub p_out: usize,
    /// `(K, K)` joint gate activity. `coactivation[j,k]` is `mean_n gate[n,j]·gate[n,k]`.
    pub coactivation: Array2<f64>,
    /// Base strength. If `learnable_weight` is true the resolved strength is
    /// `weight·exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}
