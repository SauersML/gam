//! Analytic penalty primitives for the three-tier (beta / ext-coord / rho) engine.
//!
//! See `proposals/composition_engine.md` §3-§4 and `proposals/latent_coord.md`
//! §2.3 for the motivation. This module implements the structured
//! penalties identified as the minimal identifiability tools needed by an
//! SAE / principal-manifold / latent-coordinate workflow:
//!
//!   * [`IsometryPenalty`] — pulls the pullback metric of the decoder toward a
//!     reference metric on the latent manifold. Lives on the extension-coordinate tier
//!     (specifically on
//!     a [`crate::terms::latent_coord::LatentCoordValues`] slice). Breaks the
//!     diffeomorphism gauge so the inner Hessian on `t` is full-rank and the
//!     IFT is well-defined.
//!   * [`SparsityPenalty`] — smoothed L¹ (`sqrt(x² + ε²)`), Hoyer, or Log
//!     sparsifier. Applied to a `β` slice (SAE codes) or extension-coordinate
//!     slice (soft atom
//!     amplitudes). Differentiable everywhere; the smoothing parameter `ε` may
//!     itself live in `ρ` so REML shrinks it.
//!   * [`IBPAssignmentPenalty`] — deterministic continuous-relaxation
//!     Beta-Bernoulli/IBP prior over per-row SAE-manifold active sets.
//!   * [`ARDPenalty`] — one penalty parameter per latent axis. The marginal
//!     likelihood's Occam factor sends unused axes' precision to infinity,
//!     discovering intrinsic dimension only after a separate gauge fix
//!     (`AuxPrior` or `Isometry`) pins rotations / reparameterisations.
//!   * [`TotalVariationPenalty`] — smoothed L¹ on first differences of a
//!     latent coefficient block. This is coordinatewise/anisotropic TV: each
//!     latent axis is penalized independently on every edge. Promotes
//!     piecewise-constant atom maps.
//!   * [`NuclearNormPenalty`] — smoothed L¹ on singular values of a matrix
//!     latent block. Promotes low intrinsic rank without choosing a canonical
//!     axis basis.
//!   * [`BlockSparsityPenalty`] — group-lasso smoothed L¹ over predefined
//!     latent-axis blocks. Unlike per-element L¹ or per-axis L² ARD, it
//!     shrinks whole semantic groups together; pair with
//!     `LatentIdMode::AuxPriorDimSelection` when aux classes define the active
//!     group subset.
//!   * [`RowPrecisionPriorPenalty`] — zero-mean Gaussian row-precision
//!     prior on latent rows. This fixed-precomputed variant accepts one
//!     precision matrix per row. It is not an iVAE conditional-mean gauge;
//!     use `LatentIdMode::AuxPrior` for the ridge/linear projection residual.
//!   * [`IvaeRidgeMeanGauge`] — iVAE-style conditional-mean gauge fixing:
//!     penalizes the component of the latent field not explained by auxiliary
//!     covariates via the ridge projection `U(UᵀU + εI)⁻¹Uᵀ`.
//!   * [`ParametricRowPrecisionPriorPenalty`] — zero-mean Gaussian
//!     row-precision prior with a learnable distance-kernel map from auxiliary
//!     rows to diagonal per-row precision. It changes shrinkage strength, not
//!     the conditional mean.
//!   * [`OrthogonalityPenalty`] — fixes the rotation gauge inside a latent
//!     block by penalizing cross-axis correlations. Pair with ARD when
//!     intrinsic dimension should be identifiable.
//!   * [`BlockOrthogonalityPenalty`] — penalizes only between-block
//!     cross-products of latent axes, leaving within-block structure free.
//!
//! All shipped primitives are **analytic**: no autograd, no finite differencing. Each
//! exposes:
//!
//!   * `value(target, rho) -> f64`
//!   * `grad_target(target, rho) -> Array1<f64>`
//!   * `hessian_diag(target, rho) -> Array1<f64>` (when block-diagonal) or
//!     `hvp(target, rho, v) -> Array1<f64>` (when not)
//!   * `grad_rho(target, rho) -> Array1<f64>` (one entry per ρ-axis owned)
//!
//! The signatures are deliberately uniform with the existing smoothness path:
//! the quadratic ARD penalty produces a [`crate::terms::smooth::BlockwisePenalty`]
//! that slots directly into the canonical-penalty pipeline, while the
//! non-quadratic Sparsity, TV, NuclearNorm, Orthogonality, and Isometry
//! penalties produce [`AnalyticPenaltyOp`] handles that downstream PIRLS / REML consumers query
//! through the same `value / gradient / hvp` interface they already use for
//! smoothness.
//!
//! ## Registration with REML
//!
//! Each penalty owns a (possibly empty) sub-range of the global `ρ` vector.
//! See [`AnalyticPenaltyKind::rho_count`]. The outer REML loop concatenates
//! these onto the existing per-smooth `ρ`s, exactly the way anisotropic
//! kernel-shape paths append ext-coords. The IsometryPenalty owns one `ρ`; the
//! SparsityPenalty owns either zero (`ε` fixed) or one (`ε` REML-selected) plus
//! one strength; the ARDPenalty owns `d` (one per latent axis);
//! NuclearNorm, BlockSparsity, BlockOrthogonality, RowPrecisionPrior, and
//! Orthogonality each own one strength only when their weight is learnable.
//! IvaeRidgeMeanGauge owns one strength only when its weight is learnable.
//! ParametricRowPrecisionPrior owns its log-baseline precision, raw distance
//! sensitivity, and reference point coordinates, plus one strength axis when
//! requested.
//!
//! ## Three-tier landings
//!
//! | Penalty   | Target tier | ρ-axes owned         |
//! |-----------|-------------|----------------------|
//! | Isometry  | ext-coord (latent t) | 1 (log μ_iso)        |
//! | Sparsity  | β or ext-coord       | 1 (strength) [+1 ε]  |
//! | IBP       | ext-coord (logits)   | 0 or 1 (log α)       |
//! | ARD       | ext-coord (latent t) | d (one per axis)     |
//! | TV        | ext-coord (latent t) | 0 or 1 (log μ_tv)    |
//! | NuclearNorm | ext-coord (latent t) | 0 or 1 (log μ_nuc)  |
//! | BlockSparsity | ext-coord (latent t) | 0 or 1 (log μ_group) |
//! | MechanismSparsity | β (decoder W) | 0 or 1 (log μ_mech) |
//! | RowPrecisionPrior | ext-coord (latent t) | 0 or 1 (log μ_aux) |
//! | IvaeRidgeMeanGauge | ext-coord (latent t) | 0 or 1 (log μ_ivae_mean) |
//! | ParametricRowPrecisionPrior | ext-coord (latent t) | d + d + d·du [+1 log μ_aux] |
//! | Orthogonality | ext-coord (latent t) | 0 or 1 (log μ_orth) |
//! | BlockOrthogonality | ext-coord (latent t) | 0 or 1 (log μ_block_orth) |

use faer::Side;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, CowArray, Ix2, Ix3};
use std::sync::Arc;

use crate::linalg::faer_ndarray::{FaerEigh, FaerSvd};
use crate::terms::basis::{
    BasisError, DuchonNullspaceOrder, duchon_radial_first_derivative_nd,
    duchon_radial_second_derivative_nd, duchon_radial_third_derivative_nd,
};
use crate::terms::penalties::PenaltyManifest;
use crate::terms::penalty_op::PenaltyOp;
use crate::terms::sae_manifold::{GumbelTemperatureSchedule, ScheduleKind};
use crate::terms::smooth::BlockwisePenalty;

const MIN_CONDITIONAL_PRECISION: f64 = 1.0e-12;

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

fn stable_softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Scalar annealing schedule for analytic penalty weights.
///
/// This is the penalty-weight analogue of [`crate::terms::sae_manifold::GumbelTemperatureSchedule`]:
/// it starts with a weak analytic regularizer and ramps toward the target
/// weight during REML outer iterations. This follows the standard annealed
/// regularization pattern in deep learning, where optimization first finds
/// good fits before stronger structure constrains the solution. It also
/// addresses the auto_exp_30-37 finding that hand-picked analytic weights
/// materially affect outcomes: auto_exp_30 preferred `w_ortho = 0`, while
/// auto_exp_33's fixed tight `sigma_aux = 0.5` beat the learned-weight
/// auto_exp_34 result (`R²(hue)=0.70` versus `0.62`). A schedule side-steps
/// that brittle initial choice by ramping the constraint.
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
    /// (Isometry); those implement [`Self::hvp`] instead.
    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        std::hint::black_box(target);
        std::hint::black_box(rho);
        None
    }

    /// Hessian-vector product `H v = (∂²P/∂target²) v`. Default implementation
    /// uses `hessian_diag` when available; otherwise penalties must override.
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
            return out;
        }
        // Principled generic default: central finite-difference of `grad_target`
        // along direction `v`. The directional derivative of the gradient *is*
        // the Hessian-vector product, so this is exact up to FD truncation /
        // round-off and requires no closed-form Hessian. Cost is two extra
        // `grad_target` evaluations per call — the price of a derivative-free,
        // dense-Hessian-agnostic column extractor.
        let n = v.len();
        let mut v_inf: f64 = 0.0;
        for i in 0..n {
            let a = v[i].abs();
            if a > v_inf {
                v_inf = a;
            }
        }
        if v_inf == 0.0 {
            return Array1::<f64>::zeros(n);
        }
        let eps: f64 = 1e-7_f64.max(v_inf * 1e-7);
        let mut t_plus = target.to_owned();
        t_plus.scaled_add(eps, &v);
        let mut t_minus = target.to_owned();
        t_minus.scaled_add(-eps, &v);
        let g_plus = self.grad_target(t_plus.view(), rho);
        let g_minus = self.grad_target(t_minus.view(), rho);
        let mut out = Array1::<f64>::zeros(n);
        let inv_two_eps = 1.0 / (2.0 * eps);
        for i in 0..n {
            out[i] = (g_plus[i] - g_minus[i]) * inv_two_eps;
        }
        out
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
        std::hint::black_box(iter);
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
/// **Contraction order invariant.** Every place this struct touches `W_n`,
/// the contraction is `(J^T U_n)(U_n^T J)` — never `J^T W_n J` with `W_n`
/// materialized as `p × p`. Concretely we form `M_n = U_n^T J_n ∈ ℝ^{r × d}`
/// once and then `g_n = M_n^T M_n` (`d × d`). Cost per row:
/// `O(p · r · d + r · d²)`, independent of `p²`.
///
/// **When to use.** Whenever a `LatentCoord` block is in play without an
/// auxiliary variable (`AuxPrior`) to break the diffeomorphism gauge. Fixes
/// the audit finding that ARD is not a standalone gauge fix. With a Euclidean
/// reference, the penalty pulls the decoder toward a
/// local isometry, which is enough to make the inner Hessian on `t` full-rank
/// and the IFT well-defined.
///
/// **Math.** Let `J_n ∈ ℝ^{p × d}` be the local decoder Jacobian. Then
/// `g_n = J_n^T J_n` and the penalty is `½ μ Σ_n ‖J_n^T J_n − g^ref_n‖²_F`.
/// Analytic gradient w.r.t. `t_n`:
///
/// ```text
///   ∂P/∂t_{n,c}
///     = μ Σ_{a,b} (g_n − g^ref_n)_{ab}
///         [ H_{n,:,a,c}^T W_n J_{n,:,b}
///           + J_{n,:,a}^T W_n H_{n,:,b,c} ],
///   H_{n,i,a,c} = ∂J_{n,i,a}/∂t_{n,c}.
/// ```
///
/// The per-row Jacobian `J_n` is exactly the radial-derivative jet
/// `design_gradient_wrt_t` already computes for `LatentCoordValues`; the
/// second derivative `∂J/∂t` is rebuilt from
/// [`crate::terms::basis::duchon_radial_second_derivative_nd`] using the
/// radial Hessian identity. A finite-difference oracle for the docstring is
/// to central-difference `value(t ± h e_j)` against `grad_target(t)[j]`;
/// the analytic value follows the oracle until finite-difference
/// cancellation dominates. No autograd needed.
///
/// `μ = exp(ρ_iso)` is REML-selectable as one extra ρ axis.
#[derive(Debug, Clone)]
pub struct IsometryPenalty {
    pub target: PsiSlice,
    pub reference: IsometryReference,
    /// Index of this penalty's strength `log μ_iso` inside the *local* rho
    /// view this penalty receives. Always `0` for now (single owned axis).
    pub rho_index: usize,
    /// Cached Jacobian `J ∈ ℝ^{n_obs × p × d}`, flattened row-major
    /// `(n_obs, p*d)`. The owning driver refreshes this each IFT outer step
    /// before invoking `value` / `grad_target`; in operator-only call sites
    /// (Hessian-vector products) the cache must be live.
    pub jacobian_cache: Option<Arc<Array2<f64>>>,
    /// Optional cached per-row Jacobian *second derivative*
    /// `H_n ∈ ℝ^{p × d × d}`, flattened row-major as `(n_obs, p*d*d)`.
    /// `H_n[i, a, c] = ∂J_n[i, a] / ∂t_{n, c}`. Either this cache or
    /// `duchon_radial_source` must be present for exact isometry
    /// gradient/HVP calls.
    pub jacobian_second_cache: Option<Arc<Array2<f64>>>,
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
    /// analytic `hvp` calls.
    pub cache_third_decoder_derivative: Option<Arc<ndarray::Array3<f64>>>,
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
    g: Array2<f64>,
    g_ref: CowArray<'a, f64, Ix2>,
    wj_rows: Vec<Array2<f64>>,
}

impl IsometryPenalty {
    pub const DEFAULT_VALUE_ON_MISSING_CACHE: f64 = 0.0;

    #[must_use]
    pub fn new_euclidean(target: PsiSlice, p_out: usize) -> Self {
        Self {
            target,
            reference: IsometryReference::Euclidean,
            rho_index: 0,
            jacobian_cache: None,
            jacobian_second_cache: None,
            duchon_radial_source: None,
            cache_third_decoder_derivative: None,
            p_out,
            weight: WeightField::Identity,
            scalar_weight: 1.0,
            weight_schedule: None,
        }
    }

    /// Attach a cached third decoder derivative
    /// `K_n[i, a, c, d] = ∂²J_n[i, a] / ∂t_{n, c} ∂t_{n, d}`, flattened
    /// row-major as `(n_obs, p * d * d * d)`. The Hessian-vector product
    /// uses the full residual-curvature term in addition to the metric
    /// Gauss-Newton piece.
    #[must_use]
    pub fn with_third_decoder_derivative(mut self, k: Arc<ndarray::Array3<f64>>) -> Self {
        self.cache_third_decoder_derivative = Some(k);
        self
    }

    #[must_use]
    pub fn with_reference(mut self, reference: IsometryReference) -> Self {
        self.reference = reference;
        self
    }

    #[must_use]
    pub fn with_jacobian_cache(mut self, j: Arc<Array2<f64>>) -> Self {
        self.jacobian_cache = Some(j);
        self
    }

    #[must_use]
    pub fn with_jacobian_second_cache(mut self, h: Arc<Array2<f64>>) -> Self {
        self.jacobian_second_cache = Some(h);
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

    /// Attach a per-row behavioral metric in low-rank factored form
    /// (`W_n = U_n U_n^T`). The contraction-order invariant is enforced by
    /// the per-row builder `M_n = U_n^T J_n`; see [`WeightField`].
    #[must_use]
    pub fn with_weight(mut self, weight: WeightField) -> Self {
        self.weight = weight;
        self
    }

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.scalar_weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn missing_cache_default(&self, method: &str, detail: &str) {
        let has_required_cache = false;
        debug_assert!(
            has_required_cache,
            "IsometryPenalty::{method} missing required derivative state: {detail}"
        );
        log::warn!(
            "IsometryPenalty::{method} missing required derivative state: {detail}; \
             returning the zero safe default"
        );
    }

    fn has_jacobian_cache(&self, method: &str) -> bool {
        if self.jacobian_cache.is_some() {
            true
        } else {
            self.missing_cache_default(method, "jacobian_cache is None");
            false
        }
    }

    fn has_jacobian_second_source(&self, method: &str) -> bool {
        if self.jacobian_second_cache.is_some() || self.duchon_radial_source.is_some() {
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
        if self.cache_third_decoder_derivative.is_some() || self.duchon_radial_source.is_some() {
            true
        } else {
            self.missing_cache_default(
                method,
                "both cache_third_decoder_derivative and duchon_radial_source are None",
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
        let Some(jac) = self.jacobian_cache.as_ref() else {
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
        let Some(jac) = self.jacobian_cache.as_ref() else {
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
                debug_assert_eq!(p, *p_out);
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
                debug_assert_eq!(p, *p_out);
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

    fn duchon_radial_jacobian_second(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<Array2<f64>, BasisError> {
        let t = Self::target_matrix(target, n_obs, d);
        let phi_r = duchon_radial_first_derivative_nd(
            t.view(),
            source.centers.view(),
            source.length_scale,
            source.nullspace_order,
        )?;
        let phi_rr = duchon_radial_second_derivative_nd(
            t.view(),
            source.centers.view(),
            source.length_scale,
            source.nullspace_order,
        )?;
        let n_centers = source.centers.nrows();
        debug_assert_eq!(source.centers.ncols(), d);
        debug_assert_eq!(source.radial_coefficients.nrows(), n_centers);
        debug_assert_eq!(source.radial_coefficients.ncols(), self.p_out);

        let mut out = Array2::<f64>::zeros((n_obs, self.p_out * d * d));
        for n in 0..n_obs {
            for k in 0..n_centers {
                let mut r2 = 0.0_f64;
                for a in 0..d {
                    let delta = t[[n, a]] - source.centers[[k, a]];
                    r2 += delta * delta;
                }
                let r = r2.sqrt();
                for a in 0..d {
                    for c in 0..d {
                        let basis_hess = if r == 0.0 {
                            if a == c { phi_rr[[n, k]] } else { 0.0 }
                        } else {
                            let inv_r = 1.0 / r;
                            let u_a = (t[[n, a]] - source.centers[[k, a]]) * inv_r;
                            let u_c = (t[[n, c]] - source.centers[[k, c]]) * inv_r;
                            let q = phi_r[[n, k]] * inv_r;
                            let eye = if a == c { 1.0 } else { 0.0 };
                            q * eye + (phi_rr[[n, k]] - q) * u_a * u_c
                        };
                        if basis_hess == 0.0 {
                            continue;
                        }
                        for i in 0..self.p_out {
                            out[[n, (i * d + a) * d + c]] +=
                                source.radial_coefficients[[k, i]] * basis_hess;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn duchon_radial_jacobian_third(
        &self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
        source: &IsometryDuchonRadialSource,
    ) -> Result<ndarray::Array3<f64>, BasisError> {
        let t = Self::target_matrix(target, n_obs, d);
        let phi_r = duchon_radial_first_derivative_nd(
            t.view(),
            source.centers.view(),
            source.length_scale,
            source.nullspace_order,
        )?;
        let phi_rr = duchon_radial_second_derivative_nd(
            t.view(),
            source.centers.view(),
            source.length_scale,
            source.nullspace_order,
        )?;
        let phi_rrr = duchon_radial_third_derivative_nd(
            t.view(),
            source.centers.view(),
            source.length_scale,
            source.nullspace_order,
        )?;
        let n_centers = source.centers.nrows();
        debug_assert_eq!(source.centers.ncols(), d);
        debug_assert_eq!(source.radial_coefficients.nrows(), n_centers);
        debug_assert_eq!(source.radial_coefficients.ncols(), self.p_out);

        let mut out = ndarray::Array3::<f64>::zeros((n_obs, self.p_out, d * d * d));
        for n in 0..n_obs {
            for k in 0..n_centers {
                let mut r2 = 0.0_f64;
                for a in 0..d {
                    let delta = t[[n, a]] - source.centers[[k, a]];
                    r2 += delta * delta;
                }
                let r = r2.sqrt();
                if r == 0.0 {
                    continue;
                }
                let inv_r = 1.0 / r;
                let q = phi_r[[n, k]] * inv_r;
                let b_coef = (phi_rr[[n, k]] - q) * inv_r;
                let a_coef = phi_rrr[[n, k]] - 3.0 * b_coef;
                for a in 0..d {
                    let u_a = (t[[n, a]] - source.centers[[k, a]]) * inv_r;
                    for c in 0..d {
                        let u_c = (t[[n, c]] - source.centers[[k, c]]) * inv_r;
                        for dd in 0..d {
                            let u_d = (t[[n, dd]] - source.centers[[k, dd]]) * inv_r;
                            let eye_ac = if a == c { 1.0 } else { 0.0 };
                            let eye_ad = if a == dd { 1.0 } else { 0.0 };
                            let eye_cd = if c == dd { 1.0 } else { 0.0 };
                            let basis_third = a_coef * u_a * u_c * u_d
                                + b_coef * (eye_ac * u_d + eye_ad * u_c + eye_cd * u_a);
                            if basis_third == 0.0 {
                                continue;
                            }
                            let idx = ((a * d) + c) * d + dd;
                            for i in 0..self.p_out {
                                out[[n, i, idx]] +=
                                    source.radial_coefficients[[k, i]] * basis_third;
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn jacobian_second<'a>(
        &'a self,
        target: ArrayView1<'_, f64>,
        n_obs: usize,
        d: usize,
    ) -> Option<CowArray<'a, f64, Ix2>> {
        if let Some(jac2) = self.jacobian_second_cache.as_ref() {
            return Some(CowArray::from(jac2.view()));
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
        if let Some(jac3) = self.cache_third_decoder_derivative.as_ref() {
            return Some(CowArray::from(jac3.view()));
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
        let g_ref = self.reference_metric(n_obs, d);
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
            g,
            g_ref,
            wj_rows,
        })
    }

    fn hvp_with_precomputed_state(
        &self,
        state: &IsometryHvpState<'_>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mu = self.scalar_weight * rho[self.rho_index].exp();
        let d = state.d;
        let n_obs = state.n_obs;
        let p = state.p;
        let jac2 = &state.jac2;
        let jac3 = &state.jac3;
        let g = &state.g;
        let g_ref = &state.g_ref;
        let mut out = Array1::<f64>::zeros(v.len());

        for n in 0..n_obs {
            let wj = &state.wj_rows[n];
            let mut delta_g = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for b in 0..d {
                    let mut s = 0.0;
                    for c in 0..d {
                        let vc = v[n * d + c];
                        if vc == 0.0 {
                            continue;
                        }
                        for i in 0..p {
                            s += vc * jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
                            s += vc * wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
                        }
                    }
                    delta_g[[a, b]] = s;
                }
            }
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let mut dg_c = 0.0;
                        for i in 0..p {
                            dg_c += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
                            dg_c += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
                        }
                        acc += dg_c * delta_g[[a, b]];
                    }
                }
                out[n * d + c] = mu * acc;
            }

            for c in 0..d {
                let mut acc_res = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let diff = g[[n, a * d + b]] - g_ref[[n, a * d + b]];
                        if diff == 0.0 {
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
                        acc_res += diff * bv;
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
    fn pullback_metric(&self, latent_dim: usize) -> Option<Array2<f64>> {
        let Some(jac) = self.jacobian_cache.as_ref() else {
            self.missing_cache_default("pullback_metric", "jacobian_cache is None");
            return None;
        };
        let n_obs = jac.nrows();
        let p = self.p_out;
        debug_assert_eq!(jac.ncols(), p * latent_dim);
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

    /// Reference metric per row, `(n_obs, d*d)`.
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
                debug_assert_eq!(a.nrows(), n_obs);
                debug_assert_eq!(a.ncols(), d * d);
                CowArray::from(a.view())
            }
        }
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
        let g_ref = self.reference_metric(n_obs, d);
        let mu = self.scalar_weight * rho[self.rho_index].exp();
        let mut acc = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                let diff = g[[n, k]] - g_ref[[n, k]];
                acc += diff * diff;
            }
        }
        0.5 * mu * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Exact closed-form gradient, W-aware:
        //
        //   P     = ½ μ Σ_n ‖D_n‖²_F,   D_n = g_n − g^ref_n
        //   g_n   = J_n^T W_n J_n,      W_n = U_n U_n^T
        //   ∂g_{ab}/∂t_c
        //         = (H_{:,a,c})^T (W J)_{:,b}  +  (J_{:,a})^T W H_{:,b,c}
        //   ∂P/∂t_c
        //         = μ Σ_{a,b} D_{a,b} · ∂g_{ab}/∂t_c
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
        let g_ref = self.reference_metric(n_obs, d);
        let p = self.p_out;
        let mu = self.scalar_weight * rho[self.rho_index].exp();
        let mut grad = Array1::<f64>::zeros(target.len());
        let Some(jac2) = self.jacobian_second(target, n_obs, d) else {
            return grad;
        };
        debug_assert_eq!(jac2.ncols(), p * d * d);

        for n in 0..n_obs {
            let Some(wj) = self.weighted_jacobian_row(n, d) else {
                return grad;
            };
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        let diff = g[[n, a * d + b]] - g_ref[[n, a * d + b]];
                        let mut dg = 0.0;
                        for i in 0..p {
                            dg += jac2[[n, (i * d + a) * d + c]] * wj[[i, b]];
                            dg += wj[[i, a]] * jac2[[n, (i * d + b) * d + c]];
                        }
                        acc += diff * dg;
                    }
                }
                grad[n * d + c] = mu * acc;
            }
        }
        grad
    }

    /// Fully analytic - wired through `duchon_radial_third_derivative_nd`.
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Fully analytic isometry Hessian-vector product wired through
        // duchon_radial_third_derivative_nd when no third-derivative cache is
        // supplied.
        //
        // The full Hessian of P_iso = (μ/2) Σ_n ||J^T W J - G_ref||²_F
        // (per proposal §4(b)) is
        //   ∂²P/∂t_c ∂t_d = μ Σ_{a,b} [
        //       ∂g_{ab}/∂t_c · ∂g_{ab}/∂t_d                  (GN piece)
        //     + (g_{ab} - g^ref_{ab}) · B_{ab,cd}             (residual piece)
        //   ],
        //   B_{ab,cd} = K_{a,cd}^T W J_b + H_{a,c}^T W H_{b,d}
        //             + H_{a,d}^T W H_{b,c} + J_a^T W K_{b,cd},
        // where K is the third decoder derivative and H is the second.
        let Some(state) = self.hvp_state(target) else {
            return Array1::<f64>::zeros(v.len());
        };
        self.hvp_with_precomputed_state(&state, rho, v)
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.scalar_weight, &mut self.weight_schedule, iter);
    }
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
        debug_assert!(k_atoms > 0);
        debug_assert!(temperature > 0.0);
        Self {
            k_atoms,
            temperature,
            weight: 1.0,
            weight_schedule: None,
        }
    }

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

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
}

impl AnalyticPenalty for SoftmaxAssignmentSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let lambda = self.weight * rho[0].exp();
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
        let lambda = self.weight * rho[0].exp();
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
                let ak = a[k].max(1e-300);
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
        std::hint::black_box(target);
        std::hint::black_box(rho);
        None
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        /*
        Bug fix: softmax entropy is not coordinate-separable in logits.
        The old `hessian_diag` returned λ p_k(1-p_k)/τ², which is only the
        softmax Jacobian diagonal and omits the entropy curvature and all
        cross-logit terms. That made `hvp` and log-det callers consume a
        diagonal surrogate that is not the derivative of `grad_target`.
        For H(p(z)), p'=p*(v-E_p[v])/τ and
        (log p_k + 1)'=(v_k-E_p[v])/τ. Differentiating
        g_k=λ p_k(E_p[log p + 1]-(log p_k+1))/τ gives the row-dense product
        below. `hessian_diag` now returns `None`, so diagonal users recover
        the true diagonal by probing this exact symmetric HVP.
        */
        let lambda = self.weight * rho[0].exp();
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
                mean_log_plus_one += a[k] * (a[k].max(1e-300).ln() + 1.0);
                mean_v += a[k] * v[start + k];
            }
            let mut mean_centered_v_log_plus_one = 0.0;
            for k in 0..self.k_atoms {
                let centered_v = v[start + k] - mean_v;
                mean_centered_v_log_plus_one += a[k] * centered_v * (a[k].max(1e-300).ln() + 1.0);
            }
            for k in 0..self.k_atoms {
                let log_plus_one = a[k].max(1e-300).ln() + 1.0;
                let centered_v = v[start + k] - mean_v;
                out[start + k] = scale
                    * a[k]
                    * (centered_v * (mean_log_plus_one - log_plus_one - 1.0)
                        + mean_centered_v_log_plus_one);
            }
        }
        out
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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
        debug_assert!(k_max > 0);
        debug_assert!(alpha.is_finite() && alpha > 0.0);
        debug_assert!(tau.is_finite() && tau > 0.0);
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_alpha(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_alpha {
            self.alpha * rho[0].exp()
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
        let eps = 1.0e-9;
        let mut pi = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let mut active_mass = 0.0;
            for row in 0..n {
                active_mass += z[row * self.k_max + k];
            }
            let denom = (n as f64 + a - 1.0).max(eps);
            let raw = (active_mass + a - 1.0) / denom;
            pi[k] = raw.clamp(eps, 1.0 - eps);
        }
        pi
    }
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
                let zk = z[start + k].clamp(1.0e-12, 1.0 - 1.0e-12);
                let pk = pi[k].clamp(1.0e-12, 1.0 - 1.0e-12);
                acc -= zk * pk.ln() + (1.0 - zk) * (1.0 - pk).ln();
            }
        }
        for k in 0..self.k_max {
            // Beta(a,1) contributes -(a - 1) ln(pi), matching pi_map.
            acc -= (a - 1.0) * pi[k].ln();
        }
        self.weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(1.0e-12, 1.0 - 1.0e-12);
                let d_p_d_z = ((1.0 - pk) / pk).ln();
                out[start + k] = self.weight * d_p_d_z * zk * (1.0 - zk) / tau;
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
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (tau * tau);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(1.0e-12, 1.0 - 1.0e-12);
                let d_p_d_z = ((1.0 - pk) / pk).ln();
                out[start + k] =
                    self.weight * d_p_d_z * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
            }
        }
        Some(out)
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
            sum_log_pi += pk.clamp(1.0e-12, 1.0 - 1.0e-12).ln();
        }
        Array1::from_vec(vec![-self.weight * alpha * sum_log_pi / self.k_max as f64])
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    #[must_use]
    pub fn with_eps_reml(mut self, eps_rho_index: usize) -> Self {
        self.eps_rho_index = Some(eps_rho_index);
        self
    }

    /// Resolve `(strength, eps_or_delta)` from the current ρ view.
    fn resolved(&self, rho: ArrayView1<'_, f64>) -> (f64, f64) {
        let strength = self.weight * rho[self.strength_rho_index].exp();
        let smoothing = match (self.eps_rho_index, self.kind) {
            (Some(idx), _) => rho[idx].exp(),
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
                debug_assert!(n > 1.0, "Hoyer requires n > 1");
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
                debug_assert!(n > 1.0, "Hoyer requires n > 1");
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
        let mut d = Array1::<f64>::zeros(target.len());
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let r = (x * x + eps2).sqrt();
                    d[i] = lam * eps2 / (r * r * r);
                }
                Some(d)
            }
            SparsityKind::Log { .. } => {
                // The TRUE second derivative of λ log(1 + x²/δ²) is
                //   2λ(δ² − x²)/(δ² + x²)²
                // which is NEGATIVE for |x| > δ — i.e. Log is nonconvex.
                // We therefore expose the IRLS (MM) MAJORIZER
                //   2λ / (δ² + x²)
                // through `hessian_diag`. This is always strictly positive,
                // matches the true Hessian at |x| = 0, and is the standard
                // re-weighted ℓ₂ surrogate used by IRLS-based log-sparsity
                // solvers. PSD consumers (preconditioner, `log_det_plus_λI`,
                // FrozenAnalyticPenaltyOp routing) thus see a PSD operator
                // even though `value` and `grad_target` use the exact log.
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    d[i] = lam * 2.0 / denom;
                }
                Some(d)
            }
            // Hoyer's Hessian is DENSE and NOT generally PSD (Hoyer is a
            // nonconvex sparsifier). We cannot return a meaningful diagonal
            // that would be safe to use as a preconditioner / Newton block
            // through the standard `hessian_diag` path, so we return `None`
            // and force callers through `hvp`. See `hvp` below for the exact
            // dense-Hessian-vector product.
            SparsityKind::Hoyer => {
                drop(d);
                None
            }
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
                // PSD IRLS majorizer 2λ/(δ²+x²) — matches `hessian_diag`.
                // The true second derivative 2λ(δ²−x²)/(δ²+x²)² is not used
                // here because it is indefinite (negative for |x|>δ) and
                // would break the PSD contract `FrozenAnalyticPenaltyOp`
                // exposes through `matvec` to the canonical PIRLS / log-det
                // pipeline. IRLS / MM theory: the surrogate is a global
                // upper bound on the log-sparsity penalty and agrees with
                // the exact Hessian at x = 0.
                let mut out = Array1::<f64>::zeros(n_target);
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    out[i] = lam * 2.0 / denom * v[i];
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
                debug_assert!(n > 1.0, "Hoyer requires n > 1");
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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
/// summed over `j ∈ [0, d)`. Under REML, axis `j` whose data evidence is too
/// weak gets `ρ_j → +∞` (precision → ∞, coefficients → 0), so the latent
/// dimension is effectively pruned. The intrinsic dimensionality is read off
/// as the count of finite `ρ_j` at convergence, but only after a separate
/// gauge-fixing prior (AuxPrior or Isometry) has fixed the rotation gauge.
///
/// Because the penalty is quadratic and block-diagonal in latent axes, it
/// reduces to a [`BlockwisePenalty`] per axis and slots into the existing
/// canonical-penalty pipeline with zero extra wiring beyond appending `d`
/// hyperparameter axes to `ρ`.
///
/// When to use: any [`LatentCoordValues`] block where the intrinsic dimension
/// is unknown. Compose with `IsometryPenalty` for full gauge fixing.
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
        debug_assert!(latent_dim > 0, "ARDPenalty requires latent_dim > 0");
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

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
            let rho_j = rho[self.rho_indices[j]];
            let lam_j = self.weight * rho_j.exp();
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
            let lam_j = self.weight * rho[self.rho_indices[j]].exp();
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
            let lam_j = self.weight * rho[self.rho_indices[j]].exp();
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
            let lam_j = self.weight * rho[self.rho_indices[j]].exp();
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

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
        std::hint::black_box(rho);
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
        std::hint::black_box(rho);
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
        std::hint::black_box(rho);
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
        std::hint::black_box(target);
        std::hint::black_box(rho);
        Array1::<f64>::zeros(0)
    }

    fn rho_count(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "topk_activation"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn threshold(&self, axis: usize, rho: ArrayView1<'_, f64>) -> f64 {
        self.thresholds[axis] * rho[axis].exp()
    }

    fn sigmoid_gate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }
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
                diag[base + axis] = self.weight * tau * gate * (1.0 - gate) * (1.0 - 2.0 * gate)
                    / (self.smoothing_eps * self.smoothing_eps);
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "total_variation"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

// ---------------------------------------------------------------------------
// Nuclear norm penalty
// ---------------------------------------------------------------------------

/// Basis-free low-rank penalty for a row-major `(n_eff, d)` latent block.
///
/// The smoothed nuclear norm applies a Huber-style L¹ penalty to the singular
/// spectrum, encouraging low intrinsic rank even when useful axes are rotated
/// away from the canonical basis. It complements ARD's per-axis pruning and
/// Orthogonality's basis-fixing role.
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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

    fn rank_limit(&self, rank: usize) -> usize {
        self.max_rank.unwrap_or(rank).min(rank)
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
            gram[[a, a]] += self.smoothing_eps * self.smoothing_eps;
        }

        let (evals, q) = gram
            .eigh(Side::Lower)
            .map_err(|err| format!("NuclearNormPenalty Gram eigendecomposition failed: {err}"))?;
        let active_start = d.saturating_sub(self.rank_limit(d));
        if self.max_rank.is_some() && active_start > 0 && active_start < d {
            let left = evals[active_start - 1];
            let right = evals[active_start];
            let scale = (left.abs() + right.abs()).max(1.0);
            if (right - left).abs() <= 1.0e-12 * scale {
                return Err(format!(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     smoothed Gram eigenvalue at the active/inactive cutoff \
                     ({left:.3e}, {right:.3e})"
                ));
            }
        }
        let mut f = Array1::<f64>::zeros(d);
        let mut df = Array1::<f64>::zeros(d);
        for i in 0..d {
            let lambda = evals[i];
            assert!(
                lambda.is_finite() && lambda > 0.0,
                "NuclearNormPenalty expected positive smoothed Gram eigenvalue"
            );
            if i >= active_start {
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
                let denom = evals[i] - evals[j];
                let scale = (evals[i].abs() + evals[j].abs()).max(1.0);
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
            acc += (sigma * sigma + eps * eps).sqrt() - eps;
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
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        let mut grad = Array2::<f64>::zeros(t.dim());
        for i in 0..rank {
            let sigma = svd.singular[i];
            let spectral_grad = sigma / (sigma * sigma + eps2).sqrt();
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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
        // or active-rank cutoff failures from `right_spectral_inverse_sqrt_derivative`
        // are upstream contract violations that must surface loudly.
        let (right_filter, right_filter_derivative) = self
            .right_spectral_inverse_sqrt_derivative(t.view(), v_mat.view())
            // SAFETY: error path is a caller contract violation; the upstream
            // helper already formatted a diagnostic message.
            .unwrap_or_else(|message| panic!("{}", message));
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros(t.dim());
        for n in 0..t.nrows() {
            for a in 0..t.ncols() {
                let mut term = 0.0;
                for b in 0..t.ncols() {
                    term += v_mat[[n, b]] * right_filter[[b, a]]
                        + t[[n, b]] * right_filter_derivative[[b, a]];
                }
                out[[n, a]] = weight * term;
            }
        }
        Self::flatten_matrix(&out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "nuclear_norm"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "block_sparsity"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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
            self.weight * rho[self.rho_index].exp()
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "row_precision_prior"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

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
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "ivae_ridge_mean_gauge"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
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
            let beta_k = stable_softplus(raw_beta_k);
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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
            self.weight * rho[self.weight_offset()].exp()
        } else {
            self.weight
        }
    }

    fn lambda_at(&self, n: usize, k: usize, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.active_log_alpha(k, rho).exp();
        let beta = stable_softplus(self.active_raw_beta(k, rho));
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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
            let alpha = log_alpha.exp();
            let raw_beta = self.active_raw_beta(k, rho);
            let beta = stable_softplus(raw_beta);
            let beta_jac = logistic(raw_beta);
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

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

// ---------------------------------------------------------------------------
// SCAD / MCP concave sparsity penalty
// ---------------------------------------------------------------------------

/// Concave alternative to smoothed-L¹ sparsity with less bias on large signals.
/// MCP (Zhang 2010) and SCAD (Fan-Li 2001) keep strong shrinkage near zero but
/// flatten the gradient on large coefficients; `gamma` controls concavity, and
/// `gamma -> infinity` recovers the L¹ limit. Fan-Li recommend `gamma = 3.7`
/// for SCAD.
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "scad_mcp"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

// ---------------------------------------------------------------------------
// Block-orthogonality penalty
// ---------------------------------------------------------------------------

/// Between-block-only orthogonality on a row-major matrix-valued latent
/// block. Penalizes the squared Frobenius norm of the off-diagonal blocks
/// of the latent Gram matrix `Tᵀ T`, where `T` is the row-major
/// `n_eff × latent_dim` view of the target slice and `groups` partitions
/// the latent axes into disjoint subsets:
///
/// ```text
///   P(T) = ½ · w · Σ_{g ≠ h}  ‖ (Tᵀ T)_{group_g, group_h} ‖²_F
/// ```
///
/// Within-block structure is unconstrained — the penalty zeroes out only
/// off-diagonal blocks. Pair with a per-block ARD penalty when you also
/// want within-block axis sparsity / scale identification.
///
/// Typical use: gauge-fixing a latent decomposition where one block has
/// been supervised (e.g. anchored to known coordinates) and a free block
/// needs to inhabit the orthogonal complement of that supervision.
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

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            debug_assert_eq!(
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

    fn add_right_times_cross(
        out: &mut Array2<f64>,
        right: ArrayView2<'_, f64>,
        left_axes: &[usize],
        right_axes: &[usize],
        cross_right_left: ArrayView2<'_, f64>,
        factor: f64,
    ) {
        debug_assert_eq!(cross_right_left.dim(), (right_axes.len(), left_axes.len()));
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
        debug_assert_eq!(v.dim(), t.dim(), "hvp matrix dimension mismatch");
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
                Self::add_right_times_cross(&mut out, v, group_g, group_h, c_hg.view(), weight);

                let v_h_t_g = Self::cross_gram(v, group_h, group_g);
                let t_h_v_g = Self::cross_gram(t, group_h, group_g);
                let mut d_c_hg = v_h_t_g;
                d_c_hg += &t_h_v_g;
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
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

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "block_orthogonality"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

// ---------------------------------------------------------------------------
// Orthogonality penalty
// ---------------------------------------------------------------------------

/// Gauge-fixing penalty for latent-coordinate axes.
///
/// ARD alone is rotation-invariant — pair with Orthogonality to identify
/// intrinsic dim (auto_exp_21). This penalty locks a canonical orthonormal
/// basis first; ARD can then shrink axes after the rotation gauge has been
/// identified.
#[derive(Debug, Clone)]
pub struct OrthogonalityPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Effective observation count used to keep the Frobenius contribution on
    /// the same scale as per-axis latent priors.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl OrthogonalityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        latent_dim: usize,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("OrthogonalityPenalty::new requires latent_dim > 0".to_string());
        }
        if !target.len().is_multiple_of(latent_dim) {
            return Err(format!(
                "OrthogonalityPenalty::new target length {} is not divisible by latent_dim {}",
                target.len(),
                latent_dim
            ));
        }
        let n_obs = target.len() / latent_dim;
        if n_obs < latent_dim {
            return Err(format!(
                "OrthogonalityPenalty::new requires n_obs >= latent_dim for a feasible \
                 Stiefel target, got n_obs {n_obs} and latent_dim {latent_dim}"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "OrthogonalityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("OrthogonalityPenalty::new requires n_eff > 0".to_string());
        }
        if n_eff != n_obs {
            return Err(format!(
                "OrthogonalityPenalty::new requires n_eff to match target rows, got \
                 n_eff {n_eff} and target rows {n_obs}"
            ));
        }
        Ok(Self {
            target,
            latent_dim,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight = schedule.current_weight(schedule.iter_count);
        self.weight_schedule = Some(schedule);
        self
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn scale(&self, rho: ArrayView1<'_, f64>) -> f64 {
        self.resolved_weight(rho) / self.n_eff as f64
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim;
        if !target.len().is_multiple_of(d) {
            debug_assert_eq!(
                target.len() % d,
                0,
                "target length must be divisible by latent_dim"
            );
            return None;
        }
        let n_obs = target.len() / d;
        target.into_shape_with_order((n_obs, d)).ok()
    }

    fn gram_minus_identity(t: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        let mut gram = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for n in 0..n_obs {
                    s += t[[n, a]] * t[[n, b]];
                }
                gram[[a, b]] = s;
            }
            gram[[a, a]] -= 1.0;
        }
        gram
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

    fn hvp_with_precomputed_m(
        &self,
        t: ArrayView2<'_, f64>,
        m: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
        scale: f64,
    ) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        debug_assert_eq!(v.dim(), t.dim(), "hvp matrix dimension mismatch");
        debug_assert_eq!(m.dim(), (d, d), "precomputed gram dimension mismatch");
        if v.dim() != t.dim() {
            return Array2::<f64>::zeros((n_obs, d));
        }

        let mut vt_t_plus_tt_v = Array2::<f64>::zeros((d, d));
        for c in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for n in 0..n_obs {
                    s += v[[n, c]] * t[[n, b]] + t[[n, c]] * v[[n, b]];
                }
                vt_t_plus_tt_v[[c, b]] = s;
            }
        }

        let mut out = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for b in 0..d {
                let mut va = 0.0;
                let mut tb = 0.0;
                for c in 0..d {
                    va += v[[n, c]] * m[[c, b]];
                    tb += t[[n, c]] * vt_t_plus_tt_v[[c, b]];
                }
                out[[n, b]] = 2.0 * scale * (va + tb);
            }
        }
        out
    }

    fn as_dense_with_precomputed_m(
        &self,
        t: ArrayView2<'_, f64>,
        m: ArrayView2<'_, f64>,
        scale: f64,
    ) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        debug_assert_eq!(m.dim(), (d, d), "precomputed gram dimension mismatch");
        if m.dim() != (d, d) {
            return Array2::<f64>::zeros((n_obs * d, n_obs * d));
        }

        let mut dense = Array2::<f64>::zeros((n_obs * d, n_obs * d));
        let factor = 2.0 * scale;
        for row1 in 0..n_obs {
            for row2 in 0..n_obs {
                let mut row_dot = 0.0;
                for axis in 0..d {
                    row_dot += t[[row1, axis]] * t[[row2, axis]];
                }
                for col1 in 0..d {
                    let i = row1 * d + col1;
                    for col2 in 0..d {
                        let j = row2 * d + col2;
                        let mut entry = t[[row1, col2]] * t[[row2, col1]];
                        if row1 == row2 {
                            entry += m[[col2, col1]];
                        }
                        if col1 == col2 {
                            entry += row_dot;
                        }
                        dense[[i, j]] = factor * entry;
                    }
                }
            }
        }
        dense
    }

    /// Dense cross-axis Hessian; no blockwise reduction preserves the
    /// rotation-gauge term.
    pub fn as_blockwise(&self, global_offset: usize) -> Option<Vec<BlockwisePenalty>> {
        std::hint::black_box(global_offset);
        None
    }
}

impl AnalyticPenalty for OrthogonalityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let gram = Self::gram_minus_identity(t.view());
        let mut acc = 0.0;
        for &v in gram.iter() {
            acc += v * v;
        }
        0.5 * self.scale(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Matrix-calculus core:
        //   d/dT ½·scale·||TᵀT - I||²_F = 2·scale·T·(TᵀT - I),
        // because TᵀT - I is symmetric.
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let gram = Self::gram_minus_identity(t.view());
        let n_obs = t.nrows();
        let d = t.ncols();
        let factor = 2.0 * self.scale(rho);
        let mut grad = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for a in 0..d {
                let mut s = 0.0;
                for b in 0..d {
                    s += t[[n, b]] * gram[[b, a]];
                }
                grad[[n, a]] = factor * s;
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
        debug_assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let m = Self::gram_minus_identity(t.view());
        let hv = self.hvp_with_precomputed_m(t.view(), m.view(), v_mat.view(), self.scale(rho));
        Self::flatten_matrix(&hv)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = self.value(target, rho);
        out
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "orthogonality"
    }

    fn apply_schedule(&mut self, iter: usize) {
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

// ---------------------------------------------------------------------------
// Operator-form wrapper for the REML/PIRLS canonical pipeline
// ---------------------------------------------------------------------------

/// Wraps any [`AnalyticPenalty`] so the existing PIRLS / REML consumers
/// (which expect a `value + gradient + (hvp | hessian-diag)` quintuple) can
/// query it uniformly. The wrapper is `Send + Sync` and `Arc`-shared so the
/// outer loop can hand it to multiple workers.
pub struct AnalyticPenaltyOp {
    pub penalty: Arc<dyn AnalyticPenalty>,
}

impl AnalyticPenaltyOp {
    #[must_use]
    pub fn new(penalty: Arc<dyn AnalyticPenalty>) -> Self {
        Self { penalty }
    }
}

// ---------------------------------------------------------------------------
// Registration helper — collects penalty kinds for the outer REML driver
// ---------------------------------------------------------------------------

/// Tagged sum of the analytic penalty kinds, with enough metadata for the outer
/// REML driver to:
///
///   1. Concatenate each penalty's owned ρ-axes onto the global ρ vector.
///   2. Route the inner gradient `∂L/∂target` contribution back into the
///      correct β or ext-coordinate slice.
///   3. Build a Hessian-block stub for `RemlState` cache-key invalidation.
macro_rules! define_analytic_penalty_kind {
    ($(register!($variant:ident, $ty:ty);)*) => {
        #[derive(Clone)]
        pub enum AnalyticPenaltyKind {
            $($variant(Arc<$ty>),)*
        }

        impl AnalyticPenaltyKind {
            pub fn apply_schedule(&mut self, iter: usize) {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => Arc::make_mut(p).apply_schedule(iter),)*
                }
            }

            pub fn tier(&self) -> PenaltyTier {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.dispatch_tier(),)*
                }
            }

            pub fn rho_count(&self) -> usize {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.rho_count(),)*
                }
            }

            pub fn name(&self) -> &str {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.name(),)*
                }
            }

            pub fn kind_tag(&self) -> &'static str {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::KIND_TAG,)*
                }
            }

            pub fn python_wrapper_name(&self) -> &'static str {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::PYTHON_WRAPPER,)*
                }
            }

            pub fn is_row_block_diagonal(&self) -> bool {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::ROW_BLOCK_DIAGONAL,)*
                }
            }

            pub fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.value(target, rho),)*
                }
            }

            pub fn grad_target(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.grad_target(target, rho),)*
                }
            }

            pub fn grad_rho(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.grad_rho(target, rho),)*
                }
            }

            pub fn hessian_diag(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Option<Array1<f64>> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.hessian_diag(target, rho),)*
                }
            }

            pub fn hvp(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
                v: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                if let Some(diag) = self.hessian_diag(target, rho) {
                    let mut out = Array1::<f64>::zeros(v.len());
                    for i in 0..v.len() {
                        out[i] = diag[i] * v[i];
                    }
                    return out;
                }
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.hvp(target, rho, v),)*
                }
            }
        }
    };
}

crate::analytic_penalty_registry!(define_analytic_penalty_kind);

/// Registry of analytic penalties active in a single fit. The owning
/// `RemlState` builder concatenates the per-penalty ρ-axes onto its global
/// ρ vector in the order they appear here, so the rho-index bookkeeping
/// inside each penalty is interpreted relative to its local slice.
#[derive(Clone, Default)]
pub struct AnalyticPenaltyRegistry {
    pub penalties: Vec<AnalyticPenaltyKind>,
}

impl AnalyticPenaltyRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, p: AnalyticPenaltyKind) {
        self.penalties.push(p);
    }

    pub fn total_rho_count(&self) -> usize {
        self.penalties.iter().map(|p| p.rho_count()).sum()
    }

    pub fn apply_weight_schedules(&mut self, iter: usize) {
        for penalty in &mut self.penalties {
            penalty.apply_schedule(iter);
        }
    }

    /// Returns `(local_rho_slice, target_tier, name)` for each registered
    /// penalty so the outer driver can wire its ρ-views.
    pub fn rho_layout(&self) -> Vec<(std::ops::Range<usize>, PenaltyTier, &str)> {
        let mut out = Vec::with_capacity(self.penalties.len());
        let mut offset = 0usize;
        for p in &self.penalties {
            let n = p.rho_count();
            out.push((offset..offset + n, p.tier(), p.name()));
            offset += n;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// PenaltyOp integration
// ---------------------------------------------------------------------------
//
// The canonical PIRLS / REML pipeline consumes square symmetric operators
// through the `PenaltyOp` trait (see `terms::penalty_op`). The non-quadratic
// analytic penalties here are *not* linear in their target, but the inner
// Newton step only sees their **Hessian at the current iterate**. We therefore
// expose each penalty as a `PenaltyOp` by
// freezing `(target, rho)` and routing `matvec` to `hvp`. The solver re-builds
// the frozen op once per outer iteration (after PIRLS converges on `β`), in
// exactly the same place the existing closed-form operator is rebuilt when
// the extension-coordinate block advances.

/// `PenaltyOp` view of an [`AnalyticPenalty`] frozen at `(target, rho)`.
///
/// `as_dense()` materializes the frozen local Hessian via `n` matvecs against
/// the standard basis — `O(n²)` and intended only for spectral diagnostics;
/// the hot path uses `matvec` and `diag` directly.
pub struct FrozenAnalyticPenaltyOp {
    penalty: AnalyticPenaltyKind,
    target: Array1<f64>,
    rho: Array1<f64>,
}

const ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD: usize = 1024;
const HUTCHINSON_DIAG_SAMPLES: usize = 32;
const ORTHOGONALITY_LOGDET_SLQ_PROBES: usize = 16;
const ORTHOGONALITY_LOGDET_LANCZOS_STEPS: usize = 32;

impl FrozenAnalyticPenaltyOp {
    #[must_use]
    pub fn new(penalty: AnalyticPenaltyKind, target: Array1<f64>, rho: Array1<f64>) -> Self {
        Self {
            penalty,
            target,
            rho,
        }
    }

    /// Underlying penalty (read-only). Useful for the outer driver that needs
    /// to query `grad_rho` while still holding the frozen op.
    pub fn penalty(&self) -> &AnalyticPenaltyKind {
        &self.penalty
    }
}

impl PenaltyOp for FrozenAnalyticPenaltyOp {
    fn dim(&self) -> usize {
        self.target.len()
    }

    fn matvec(&self, w: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let h = self.penalty.hvp(self.target.view(), self.rho.view(), w);
        for i in 0..h.len() {
            out[i] = h[i];
        }
    }

    fn diag(&self) -> Array1<f64> {
        // Each diagonal penalty exposes `hessian_diag` directly (ARD,
        // smoothed-L¹, Log; Hoyer currently exposes its preconditioner
        // diagonal). Penalties whose exact diagonal is cheap or contractually
        // required use the analytic path even when the dense Hessian is large.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("ARD diag"),
            AnalyticPenaltyKind::TopKActivation(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("TopK activation diag"),
            AnalyticPenaltyKind::JumpReLU(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("JumpReLU diag"),
            AnalyticPenaltyKind::TotalVariation(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::BlockOrthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::BlockOrthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Orthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NuclearNorm(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::BlockSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::BlockSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::MechanismSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::MechanismSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ScadMcp(p) => p.diag_target(self.target.view(), self.rho.view()),
            AnalyticPenaltyKind::IBPAssignment(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("IBP assignment diag"),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Sparsity(p) => {
                if let Some(d) = p.hessian_diag(self.target.view(), self.rho.view()) {
                    d
                } else {
                    self.diag_via_matvec()
                }
            }
            AnalyticPenaltyKind::Isometry(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::Isometry(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NestedPrefix(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("NestedPrefix diag"),
        }
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        // For the diagonal-Hessian penalties (ARD, smoothed-L¹ and Log) the
        // closed form is `Σ_i log(d_i + λ)`. Forward-difference TV uses the
        // tridiagonal path-graph structure. Graph TV, NuclearNorm,
        // BlockSparsity, BlockOrthogonality, and IvaeRidgeMeanGauge keep the
        // exact dense eigensolve only below the small-block threshold; large
        // blocks use SLQ against the analytic HVP.
        // Orthogonality is excluded because its exact Hessian is indefinite.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::IBPAssignment(_)
            | AnalyticPenaltyKind::NestedPrefix(_) => {
                let d = self.diag();
                let mut s = 0.0;
                for &v in d.iter() {
                    let r = v + lambda;
                    if !r.is_finite() || r <= 0.0 {
                        return Err(format!(
                            "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i: \
                             non-positive entry {r:.3e} after λ shift"
                        ));
                    }
                    s += r.ln();
                }
                Ok(s)
            }
            AnalyticPenaltyKind::TotalVariation(p) => match &p.difference_op {
                DifferenceOpKind::ForwardDiff1D => {
                    p.log_det_plus_lambda_i_forward_1d(self.target.view(), self.rho.view(), lambda)
                }
                DifferenceOpKind::GraphEdges(_)
                    if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
                {
                    self.stochastic_log_det_plus_lambda_i(lambda)
                }
                DifferenceOpKind::GraphEdges(_) => {
                    let dense = p.as_dense(self.target.view(), self.rho.view());
                    <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
                }
            },
            AnalyticPenaltyKind::Orthogonality(_) => Err(
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i cannot treat \
                 OrthogonalityPenalty as PSD; its exact Hessian is indefinite"
                    .to_string(),
            ),
            AnalyticPenaltyKind::NuclearNorm(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                p.log_det_plus_lambda_i(self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                p.log_det_plus_lambda_i(self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::ScadMcp(p) => {
                p.log_det_plus_lambda_i(self.target.view(), self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::BlockSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::MechanismSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::BlockOrthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::Isometry(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
            AnalyticPenaltyKind::NuclearNorm(_)
            | AnalyticPenaltyKind::BlockSparsity(_)
            | AnalyticPenaltyKind::MechanismSparsity(_)
            | AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
        }
    }

    fn as_dense(&self) -> Array2<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::TotalVariation(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::BlockSparsity(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::MechanismSparsity(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::BlockOrthogonality(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array2::<f64>::zeros((n, n));
                };
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                return p.as_dense_with_precomputed_m(
                    t.view(),
                    gram.view(),
                    p.scale(self.rho.view()),
                );
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array2::<f64>::zeros((n, n));
                };
                let mut dense = Array2::<f64>::zeros((n, n));
                let mut e = Array1::<f64>::zeros(n);
                for j in 0..n {
                    e[j] = 1.0;
                    let col = p.hvp_with_precomputed_state(&state, self.rho.view(), e.view());
                    for i in 0..n {
                        dense[[i, j]] = col[i];
                    }
                    e[j] = 0.0;
                }
                return dense;
            }
            _ => {}
        }
        let n = self.target.len();
        let mut m = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            let col = self
                .penalty
                .hvp(self.target.view(), self.rho.view(), e.view());
            for i in 0..n {
                m[[i, j]] = col[i];
            }
            e[j] = 0.0;
        }
        m
    }
}

impl FrozenAnalyticPenaltyOp {
    fn diag_via_matvec(&self) -> Array1<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let latent_dim = t.ncols();
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                let scale = p.scale(self.rho.view());
                let factor = 2.0 * scale;
                let mut diag = Array1::<f64>::zeros(n);
                for row in 0..t.nrows() {
                    let mut row_norm_sq = 0.0;
                    for col in 0..latent_dim {
                        row_norm_sq += t[[row, col]] * t[[row, col]];
                    }
                    for col in 0..latent_dim {
                        let i = row * latent_dim + col;
                        diag[i] = factor
                            * (gram[[col, col]] + t[[row, col]] * t[[row, col]] + row_norm_sq);
                    }
                }
                return diag;
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let mut d = Array1::<f64>::zeros(n);
                let mut e = Array1::<f64>::zeros(n);
                for i in 0..n {
                    e[i] = 1.0;
                    let h = p.hvp_with_precomputed_state(&state, self.rho.view(), e.view());
                    d[i] = h[i];
                    e[i] = 0.0;
                }
                return d;
            }
            _ => {}
        }
        let n = self.target.len();
        let mut d = Array1::<f64>::zeros(n);
        let mut e = Array1::<f64>::zeros(n);
        for i in 0..n {
            e[i] = 1.0;
            let h = self
                .penalty
                .hvp(self.target.view(), self.rho.view(), e.view());
            d[i] = h[i];
            e[i] = 0.0;
        }
        d
    }

    fn stochastic_diag_via_matvec(&self) -> Array1<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                let scale = p.scale(self.rho.view());
                let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
                let mut diag = Array1::<f64>::zeros(n);
                let mut z = Array1::<f64>::zeros(n);
                for probe in 0..samples {
                    rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
                    let Some(z_mat) = p.target_matrix(z.view()) else {
                        return diag;
                    };
                    let hz = p.hvp_with_precomputed_m(t.view(), gram.view(), z_mat, scale);
                    for i in 0..n {
                        diag[i] += z[i] * hz[[i / t.ncols(), i % t.ncols()]];
                    }
                }
                let inv_samples = 1.0 / samples as f64;
                for i in 0..n {
                    diag[i] *= inv_samples;
                }
                return diag;
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
                let mut diag = Array1::<f64>::zeros(n);
                let mut z = Array1::<f64>::zeros(n);
                for probe in 0..samples {
                    rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
                    let hz = p.hvp_with_precomputed_state(&state, self.rho.view(), z.view());
                    for i in 0..n {
                        diag[i] += z[i] * hz[i];
                    }
                }
                let inv_samples = 1.0 / samples as f64;
                for i in 0..n {
                    diag[i] *= inv_samples;
                }
                return diag;
            }
            _ => {}
        }
        let n = self.target.len();
        let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
        let mut diag = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        let mut hz = Array1::<f64>::zeros(n);
        // Hutchinson-Hadamard diagonal estimator (Bekas et al., 2007):
        // Var[(z ⊙ Hz)_i] = Σ_{j≠i} H_ij², so averaging m probes leaves
        // variance equal to the off-diagonal row mass divided by m.
        // With m=32, diagonally dominant Frobenius/TV Hessians have ~16% relative SD.
        for probe in 0..samples {
            rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
            self.matvec(z.view(), hz.view_mut());
            for i in 0..n {
                diag[i] += z[i] * hz[i];
            }
        }
        let inv_samples = 1.0 / samples as f64;
        for i in 0..n {
            diag[i] *= inv_samples;
        }
        diag
    }

    fn stochastic_log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        let n = self.dim();
        if n == 0 {
            return Ok(0.0);
        }
        let probes = ORTHOGONALITY_LOGDET_SLQ_PROBES.max(1);
        let steps = ORTHOGONALITY_LOGDET_LANCZOS_STEPS.min(n).max(1);
        let inv_norm = 1.0 / (n as f64).sqrt();
        let mut estimate = 0.0;
        for probe in 0..probes {
            let mut q0 = Array1::<f64>::zeros(n);
            rademacher_unit_probe_into(q0.view_mut(), probe as u64, inv_norm);
            let quad = self.lanczos_log_quadrature(lambda, q0, steps)?;
            estimate += n as f64 * quad;
        }
        Ok(estimate / probes as f64)
    }

    fn lanczos_log_quadrature(
        &self,
        lambda: f64,
        mut q: Array1<f64>,
        max_steps: usize,
    ) -> Result<f64, String> {
        let n = self.dim();
        let mut q_prev = Array1::<f64>::zeros(n);
        let mut alphas = Vec::<f64>::with_capacity(max_steps);
        let mut betas = Vec::<f64>::with_capacity(max_steps.saturating_sub(1));
        let mut beta_prev = 0.0;
        let tol = 1e-12_f64;

        for step in 0..max_steps {
            let mut w = Array1::<f64>::zeros(n);
            self.matvec(q.view(), w.view_mut());
            for i in 0..n {
                w[i] += lambda * q[i];
                if step > 0 {
                    w[i] -= beta_prev * q_prev[i];
                }
            }
            let alpha = dot(&q, &w);
            if !alpha.is_finite() {
                return Err(
                    "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ produced non-finite alpha"
                        .to_string(),
                );
            }
            for i in 0..n {
                w[i] -= alpha * q[i];
            }
            let beta = norm2(&w);
            alphas.push(alpha);
            if step + 1 == max_steps || beta <= tol {
                break;
            }
            if !beta.is_finite() {
                return Err(
                    "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ produced non-finite beta"
                        .to_string(),
                );
            }
            betas.push(beta);
            q_prev = q;
            q = w;
            for i in 0..n {
                q[i] /= beta;
            }
            beta_prev = beta;
        }

        let k = alphas.len();
        let mut tri = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            tri[[i, i]] = alphas[i];
            if i + 1 < k {
                tri[[i, i + 1]] = betas[i];
                tri[[i + 1, i]] = betas[i];
            }
        }
        let (evals, evecs) = tri.eigh(Side::Lower).map_err(|e| {
            format!(
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ eigendecomposition failed: {e}"
            )
        })?;
        let mut quad = 0.0;
        for j in 0..k {
            let theta = evals[j];
            if !theta.is_finite() || theta <= 0.0 {
                return Err(format!(
                    "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i expected SPD S+λI, \
                     Lanczos Ritz value {j} is {theta:.3e}"
                ));
            }
            let weight = evecs[[0, j]] * evecs[[0, j]];
            quad += weight * theta.ln();
        }
        Ok(quad)
    }
}

#[inline]
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn norm2(a: &Array1<f64>) -> f64 {
    dot(a, a).sqrt()
}

fn rademacher_unit_probe_into(mut z: ArrayViewMut1<'_, f64>, probe: u64, scale: f64) {
    let mut state = 0x6A09E667F3BCC909_u64 ^ probe.wrapping_mul(0xD1B54A32D192ED03);
    let mut bits = 0_u64;
    let mut remaining_bits = 0_u32;
    for i in 0..z.len() {
        if remaining_bits == 0 {
            bits = splitmix64(&mut state);
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { scale } else { -scale };
        bits >>= 1;
        remaining_bits -= 1;
    }
}

#[inline]
const fn splitmix64(state: &mut u64) -> u64 {
    crate::linalg::utils::splitmix64(state)
}

impl AnalyticPenaltyKind {
    /// Freeze this kind at `(target, rho)` and return an `Arc<dyn PenaltyOp>`
    /// ready to slot into `BlockwisePenalty::with_op` or `PenaltyForm::Operator`.
    #[must_use]
    pub fn freeze(&self, target: Array1<f64>, rho: Array1<f64>) -> Arc<dyn PenaltyOp> {
        Arc::new(FrozenAnalyticPenaltyOp::new(self.clone(), target, rho))
    }
}

// ---------------------------------------------------------------------------
// NestedPrefixPenalty — Matryoshka SAE
// ---------------------------------------------------------------------------

/// Nested-prefix sparsity penalty used by the Matryoshka SAE
/// (Bussmann/Nabeshima/Karvonen/Nanda, ICML 2025, arXiv:2503.17547).
///
/// Given K nested prefix sizes `m_1 < m_2 < ... < m_K ≤ F` over the latent
/// dimension `F`, and per-shell weights `λ_k = w_k · exp(ρ_k)`, the penalty is
///
/// ```text
///   P(t; ρ) = Σ_k λ_k · Σ_{i=0}^{m_k - 1} sqrt(t_i² + ε²)
/// ```
///
/// summed over all rows of the latent target. Equivalently, coordinate `i`
/// contributes with effective weight `W_i = Σ_{k: m_k > i} λ_k`, so the
/// earliest atoms (small `i`) are penalized by every shell (= strongest L¹)
/// and the latest atoms only by the outermost shell. This is exactly the
/// mask-weighted sum-of-L¹ over K prefixes used to enforce shell-wise
/// reconstruction during Matryoshka training.
///
/// Closed forms (per row, summed across all rows):
///
/// ```text
///   ∂P/∂t_i      = W_i · t_i / sqrt(t_i² + ε²)
///   Hess_diag(i) = W_i · ε² / (t_i² + ε²)^{3/2}           (PSD)
///   ∂P/∂ρ_k      = λ_k · Σ_{i < m_k} sqrt(t_i² + ε²)
/// ```
///
/// `target` lays out `n_rows × latent_dim` in row-major order (`row * F + col`).
/// `latent_dim` is taken from `PsiSlice::latent_dim`; if absent we fall back to
/// the maximum prefix size, which is the standard Matryoshka convention.
#[derive(Debug, Clone)]
pub struct NestedPrefixPenalty {
    pub target: PsiSlice,
    pub target_tier: PenaltyTier,
    /// Sorted strictly-increasing prefix sizes `m_1 < m_2 < ... < m_K`.
    pub prefix_sizes: Vec<usize>,
    /// Per-shell base weights `w_k`. The effective strength is
    /// `λ_k = w_k · exp(ρ_k)`.
    pub shell_weights: Vec<f64>,
    /// Smoothing parameter ε > 0 for the smoothed-L¹ surrogate
    /// `sqrt(x² + ε²)`; the Hessian needs ε > 0 for differentiability at 0.
    pub eps: f64,
    /// Local ρ indices for the K per-shell log-strengths.
    pub rho_indices: Vec<usize>,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl NestedPrefixPenalty {
    /// Build a new nested-prefix penalty.
    ///
    /// Errors when:
    ///  * `prefix_sizes` is empty.
    ///  * `prefix_sizes` is not strictly increasing.
    ///  * any prefix exceeds the latent dimension (when known).
    ///  * `shell_weights.len() != prefix_sizes.len()`.
    ///  * `eps <= 0` (the smoothed-L¹ gradient `1/sqrt(x²+ε²)` and Hessian
    ///    `ε²/(x²+ε²)^{3/2}` both need ε > 0).
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        target_tier: PenaltyTier,
        prefix_sizes: Vec<usize>,
        shell_weights: Vec<f64>,
        eps: f64,
    ) -> Result<Self, String> {
        if prefix_sizes.is_empty() {
            return Err("NestedPrefixPenalty requires at least one prefix".into());
        }
        if shell_weights.len() != prefix_sizes.len() {
            return Err(format!(
                "NestedPrefixPenalty requires shell_weights.len() == prefix_sizes.len(); \
                 got {} weights for {} prefixes",
                shell_weights.len(),
                prefix_sizes.len()
            ));
        }
        for w in &shell_weights {
            if !w.is_finite() || *w < 0.0 {
                return Err(format!(
                    "NestedPrefixPenalty shell weights must be finite and ≥ 0; got {w}"
                ));
            }
        }
        for i in 0..prefix_sizes.len() {
            if prefix_sizes[i] == 0 {
                return Err("NestedPrefixPenalty prefixes must be > 0".into());
            }
            if i > 0 && prefix_sizes[i] <= prefix_sizes[i - 1] {
                return Err(format!(
                    "NestedPrefixPenalty prefixes must be strictly increasing; got {:?}",
                    prefix_sizes
                ));
            }
        }
        if let Some(d) = target.latent_dim {
            let max_prefix = *prefix_sizes.last().expect("non-empty");
            if max_prefix > d {
                return Err(format!(
                    "NestedPrefixPenalty largest prefix {max_prefix} exceeds latent_dim {d}"
                ));
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!(
                "NestedPrefixPenalty requires eps > 0 (1/sqrt(x²+ε²) singularity at 0); got {eps}"
            ));
        }
        let rho_indices = (0..prefix_sizes.len()).collect();
        Ok(Self {
            target,
            target_tier,
            prefix_sizes,
            shell_weights,
            eps,
            rho_indices,
            weight_schedule: None,
        })
    }

    /// Attach a global annealing schedule shared by all shell weights. The
    /// REML loop still picks per-shell ρ_k on top of this baseline.
    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight_schedule = Some(schedule);
        self
    }

    /// Latent dimension used to slice rows. Falls back to the largest prefix.
    fn latent_dim(&self) -> usize {
        self.target
            .latent_dim
            .unwrap_or_else(|| *self.prefix_sizes.last().expect("non-empty"))
    }

    /// Resolve per-shell effective weights `λ_k = w_k · exp(ρ_k)`.
    fn lambdas(&self, rho: ArrayView1<'_, f64>) -> Vec<f64> {
        self.prefix_sizes
            .iter()
            .enumerate()
            .map(|(k, _)| self.shell_weights[k] * rho[self.rho_indices[k]].exp())
            .collect()
    }

    /// Per-axis cumulative weight `W_i = Σ_{k: m_k > i} λ_k`. Length = F.
    /// Computed in `O(F + K)` by scanning prefixes from outer to inner.
    fn per_axis_weights(&self, lambdas: &[f64]) -> Vec<f64> {
        let f = self.latent_dim();
        let mut w = vec![0.0_f64; f];
        // For each shell k, every axis i ∈ [0, m_k) gets +λ_k.
        // Equivalent reverse-cumulative form, but the direct O(K·F) loop is
        // K≤8 in practice, so this is O(F) for the use cases we ship.
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let lam = lambdas[k];
            if lam == 0.0 {
                continue;
            }
            let end = m_k.min(f);
            for entry in w.iter_mut().take(end) {
                *entry += lam;
            }
        }
        w
    }
}

impl AnalyticPenalty for NestedPrefixPenalty {
    fn tier(&self) -> PenaltyTier {
        self.target_tier
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let f = self.latent_dim();
        debug_assert!(target.len().is_multiple_of(f), "target length must be n_rows · F");
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let eps2 = self.eps * self.eps;
        // Per-axis L¹ totals s_i = Σ_n sqrt(t_{n,i}² + ε²).
        let mut s_axis = vec![0.0_f64; f];
        for n in 0..n_rows {
            let row = &target.as_slice().expect("contiguous")[n * f..(n + 1) * f];
            for (i, &x) in row.iter().enumerate() {
                s_axis[i] += (x * x + eps2).sqrt();
            }
        }
        // Now P = Σ_k λ_k · Σ_{i<m_k} s_i.
        let mut total = 0.0;
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let end = m_k.min(f);
            let mut acc = 0.0;
            for &v in s_axis.iter().take(end) {
                acc += v;
            }
            total += lambdas[k] * acc;
        }
        total
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let w_per_axis = self.per_axis_weights(&lambdas);
        let eps2 = self.eps * self.eps;
        let src = target.as_slice().expect("contiguous");
        let mut g = Array1::<f64>::zeros(target.len());
        let g_slice = g.as_slice_mut().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let x = src[n * f + i];
                let w = w_per_axis[i];
                if w == 0.0 {
                    continue;
                }
                g_slice[n * f + i] = w * x / (x * x + eps2).sqrt();
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let w_per_axis = self.per_axis_weights(&lambdas);
        let eps2 = self.eps * self.eps;
        let src = target.as_slice().expect("contiguous");
        let mut d = Array1::<f64>::zeros(target.len());
        let d_slice = d.as_slice_mut().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let w = w_per_axis[i];
                if w == 0.0 {
                    continue;
                }
                let x = src[n * f + i];
                let r = (x * x + eps2).sqrt();
                d_slice[n * f + i] = w * eps2 / (r * r * r);
            }
        }
        Some(d)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let eps2 = self.eps * self.eps;
        // Same axis-wise reduction as `value`, but we need the per-shell
        // (not cumulative) sums for the ρ-gradient.
        let mut s_axis = vec![0.0_f64; f];
        let src = target.as_slice().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let x = src[n * f + i];
                s_axis[i] += (x * x + eps2).sqrt();
            }
        }
        let n_rho = self.rho_count();
        let mut out = Array1::<f64>::zeros(n_rho);
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let end = m_k.min(f);
            let mut shell_sum = 0.0;
            for &v in s_axis.iter().take(end) {
                shell_sum += v;
            }
            // ∂P/∂ρ_k = λ_k · shell_sum  because λ_k = w_k · exp(ρ_k).
            out[self.rho_indices[k]] = lambdas[k] * shell_sum;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.prefix_sizes.len()
    }

    fn name(&self) -> &str {
        "nested_prefix"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.weight_schedule.as_mut() {
            let prev = schedule.current_weight(schedule.iter_count);
            let next = schedule.current_weight(iter);
            if prev > 0.0 {
                let ratio = next / prev;
                for w in &mut self.shell_weights {
                    *w *= ratio;
                }
            }
            schedule.iter_count = iter + 1;
        }
    }
}

/// REML helper: given the K analytic per-shell penalty values at the current
/// fit, the model's MultiShell deviance, the number of effective parameters
/// per shell, and `n_eff` rows, compute the BIC for a candidate prefix
/// schedule. The outer driver enumerates a small list of schedules
/// (typically 4 in production: doubling, dyadic, golden-ratio, manual) and
/// picks the schedule with the lowest BIC. This is the "discrete prefix
/// boundaries" piece of the REML extension.
///
/// `bic = deviance + log(n_eff) · edf_total + 2 · (Σ_k log(1 + λ_k))`
///
/// The last term is a smooth proxy for the discrete shell-mask boundaries:
/// it is monotone-increasing in each `λ_k` so the IFT derivative w.r.t. a
/// boundary move (which only changes which shell a coordinate falls into,
/// i.e. swaps which `λ_k` contributes) is well-defined as a finite difference
/// across the discrete candidate set.
#[must_use]
pub fn nested_prefix_bic(
    multishell_deviance: f64,
    edf_total: f64,
    n_eff: f64,
    lambdas: &[f64],
) -> f64 {
    let shell_term: f64 = lambdas.iter().map(|l| 2.0 * (1.0 + l).ln()).sum();
    multishell_deviance + n_eff.max(1.0).ln() * edf_total + shell_term
}

/// REML helper: select the prefix schedule (boundary set) and per-shell
/// weights minimising `nested_prefix_bic` across a discrete candidate list.
/// Returns `(best_index, best_bic)`.
#[must_use]
pub fn select_nested_prefix_schedule(
    candidates: &[(Vec<usize>, Vec<f64>)],
    multishell_deviances: &[f64],
    edf_totals: &[f64],
    n_eff: f64,
) -> (usize, f64) {
    assert_eq!(candidates.len(), multishell_deviances.len());
    assert_eq!(candidates.len(), edf_totals.len());
    let mut best = (0usize, f64::INFINITY);
    for (i, (_prefixes, weights)) in candidates.iter().enumerate() {
        let bic = nested_prefix_bic(multishell_deviances[i], edf_totals[i], n_eff, weights);
        if bic < best.1 {
            best = (i, bic);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn ard_value_matches_quadratic_form() {
        let d = 2;
        let t = array![0.5_f64, 1.0, 2.0, -1.0, 0.0, 3.0];
        let target = PsiSlice::full(t.len(), Some(d));
        let ard = ARDPenalty::new(target, d);
        let rho = array![0.0_f64, 0.0]; // λ = 1 on both axes
        let v = ard.value(t.view(), rho.view());
        // Axis 0: 0.5² + 2.0² + 0.0² = 4.25 → ½·1·4.25
        // Axis 1: 1.0² + (-1)² + 3² = 11    → ½·1·11
        assert!((v - 0.5 * (4.25 + 11.0)).abs() < 1e-12);
    }

    #[test]
    fn smoothed_l1_grad_smoothes_signum_at_zero() {
        let p = SparsityPenalty::smoothed_l1(PenaltyTier::Beta, 1e-3)
            .expect("positive eps builds smoothed L1 penalty");
        let t = array![0.0_f64, 1.0, -2.0];
        let rho = array![0.0_f64];
        let g = p.grad_target(t.view(), rho.view());
        // At x=0, grad = 0 / sqrt(0 + ε²) = 0 (not ±1).
        assert!(g[0].abs() < 1e-9);
        // At x=1, grad ≈ 1/sqrt(1 + ε²) ≈ 1.
        assert!((g[1] - 1.0).abs() < 1e-3);
        assert!((g[2] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn softmax_assignment_hvp_matches_gradient_directional_derivative() {
        let pen = SoftmaxAssignmentSparsityPenalty::new(3, 0.7);
        let t = array![0.4_f64, -0.8, 1.3, -0.2, 0.9, 0.1];
        let rho = array![1.4_f64.ln()];
        let v = array![0.2_f64, -0.5, 0.7, -0.3, 0.4, 0.6];

        assert!(
            pen.hessian_diag(t.view(), rho.view()).is_none(),
            "softmax entropy has row-dense logit curvature, not a diagonal Hessian"
        );

        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let eps = 1e-6;
        let mut tp = t.clone();
        let mut tm = t.clone();
        for i in 0..t.len() {
            tp[i] += eps * v[i];
            tm[i] -= eps * v[i];
        }
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        for i in 0..t.len() {
            let fd = (gp[i] - gm[i]) / (2.0 * eps);
            assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-6);
        }
    }

    #[test]
    fn ard_grad_target_matches_lambda_t() {
        let d = 2;
        let t = array![0.5_f64, 1.0, 2.0, -1.0];
        let target = PsiSlice::full(t.len(), Some(d));
        let ard = ARDPenalty::new(target, d);
        // log-precisions: ρ0 = ln 2 (λ0 = 2), ρ1 = ln 3 (λ1 = 3).
        let rho = array![2.0_f64.ln(), 3.0_f64.ln()];
        let g = ard.grad_target(t.view(), rho.view());
        // Axis 0 entries (n*d + 0): indices 0, 2. λ0 · t at those slots.
        assert!((g[0] - 2.0 * 0.5).abs() < 1e-12);
        assert!((g[2] - 2.0 * 2.0).abs() < 1e-12);
        // Axis 1 entries (n*d + 1): indices 1, 3. λ1 · t.
        assert!((g[1] - 3.0 * 1.0).abs() < 1e-12);
        assert!((g[3] - 3.0 * (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn ard_hessian_diag_matches_lambda() {
        let d = 2;
        let t = array![0.5_f64, 1.0, 2.0, -1.0];
        let target = PsiSlice::full(t.len(), Some(d));
        let ard = ARDPenalty::new(target, d);
        let rho = array![2.0_f64.ln(), 3.0_f64.ln()];
        let h = ard
            .hessian_diag(t.view(), rho.view())
            .expect("ARD has a diagonal Hessian");
        assert!((h[0] - 2.0).abs() < 1e-12);
        assert!((h[2] - 2.0).abs() < 1e-12);
        assert!((h[1] - 3.0).abs() < 1e-12);
        assert!((h[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ard_rho_grad_includes_occam_log_det_term() {
        let d = 2;
        let t = array![1.0_f64, 0.0, 0.0, 2.0];
        let n_obs = t.len() / d; // 2
        let target = PsiSlice::full(t.len(), Some(d));
        let ard = ARDPenalty::new(target, d);
        assert!((ard.n_eff - n_obs as f64).abs() < 1e-12);
        let rho = array![0.0_f64, 0.0];
        let dr = ard.grad_rho(t.view(), rho.view());
        // ∂P_j/∂ρ_j = ½ λ_j Σ t² − N_eff/2.
        // Axis 0: ½·1·(1+0) − ½·2 = −0.5.
        // Axis 1: ½·1·(0+4) − ½·2 =  1.0.
        assert!((dr[0] - (-0.5)).abs() < 1e-12);
        assert!((dr[1] - 1.0).abs() < 1e-12);
    }

    // ----- BlockOrthogonalityPenalty tests -----

    fn block_ortho_test_target() -> Array1<f64> {
        // n_eff = 4, latent_dim = 4, row-major (n*d + a):
        // T = [[ 1, 0, 1,  0],
        //      [ 0, 1, 0,  1],
        //      [ 1, 1, 1, -1],
        //      [-1, 0, 1,  0]]
        // Groups = [[0,1], [2,3]]
        // Between-block cross Gram (T_L^T T_R) is 2x2:
        //   col0·col2 = 1*1 + 0*0 + 1*1 + (-1)*1 = 1
        //   col0·col3 = 1*0 + 0*1 + 1*(-1) + (-1)*0 = -1
        //   col1·col2 = 0*1 + 1*0 + 1*1 + 0*1 = 1
        //   col1·col3 = 0*0 + 1*1 + 1*(-1) + 0*0 = 0
        // ‖.‖_F² = 1 + 1 + 1 + 0 = 3
        array![
            1.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0
        ]
    }

    #[test]
    fn block_orthogonality_value_matches_offdiag_gram_frobenius() {
        let t = block_ortho_test_target();
        let target = PsiSlice::full(t.len(), Some(4));
        let pen = BlockOrthogonalityPenalty::new(
            target,
            vec![vec![0_usize, 1], vec![2, 3]],
            2.5,
            4,
            false,
        )
        .expect("valid block orthogonality penalty");
        let rho = array![0.0_f64];
        let v = pen.value(t.view(), rho.view());
        // value = 0.5 · w · 3.0 = 0.5 · 2.5 · 3.0 = 3.75
        assert!(v.is_finite(), "block-orthogonality value must be finite");
        assert_abs_diff_eq!(v, 3.75, epsilon = 1e-12);
    }

    #[test]
    fn block_orthogonality_grad_matches_finite_difference() {
        let t = block_ortho_test_target();
        let n = t.len();
        let target = PsiSlice::full(n, Some(4));
        let pen = BlockOrthogonalityPenalty::new(
            target,
            vec![vec![0_usize, 1], vec![2, 3]],
            1.25,
            4,
            false,
        )
        .expect("valid block orthogonality penalty");
        let rho = array![0.0_f64];
        let g = pen.grad_target(t.view(), rho.view());
        let eps = 1e-6;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[i] += eps;
            tm[i] -= eps;
            let fd =
                (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * eps);
            let err = (g[i] - fd).abs();
            if err > max_err {
                max_err = err;
            }
            assert_abs_diff_eq!(g[i], fd, epsilon = 1e-6);
        }
        assert!(max_err < 1e-6, "grad-FD max abs error = {max_err:.3e}");
    }

    #[test]
    fn block_orthogonality_hvp_matches_gradient_directional_derivative() {
        let t = block_ortho_test_target();
        let n = t.len();
        let target = PsiSlice::full(n, Some(4));
        let pen = BlockOrthogonalityPenalty::new(
            target,
            vec![vec![0_usize, 1], vec![2, 3]],
            0.75,
            4,
            false,
        )
        .expect("valid block orthogonality penalty");
        let rho = array![0.0_f64];
        // Pick a non-trivial probe direction.
        let v: Array1<f64> =
            Array1::from_vec((0..n).map(|i| 0.3 * ((i as f64) + 1.0).sin()).collect());
        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let eps = 1e-5;
        let mut tp = t.clone();
        let mut tm = t.clone();
        for i in 0..n {
            tp[i] += eps * v[i];
            tm[i] -= eps * v[i];
        }
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let fd = (gp[i] - gm[i]) / (2.0 * eps);
            let err = (hv[i] - fd).abs();
            if err > max_err {
                max_err = err;
            }
            assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-5);
        }
        assert!(max_err < 1e-5, "hvp-FD max abs error = {max_err:.3e}");
    }

    #[test]
    fn block_orthogonality_rejects_groups_missing_an_axis() {
        // latent_dim derived as len/n_eff = 16/4 = 4; groups cover only axes
        // 0,1,2 → axis 3 is missing.
        let t = block_ortho_test_target();
        let target = PsiSlice::full(t.len(), Some(4));
        let err =
            BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2]], 1.0, 4, false)
                .expect_err("groups missing axis 3 must error");
        assert!(
            err.contains("must partition latent axes") && err.contains("missing axis 3"),
            "unexpected error message: {err}"
        );
    }

    // ----- MechanismSparsityPenalty tests -----

    fn mech_sparsity_test_target() -> Array1<f64> {
        // latent_dim = 2, p_features = 3, row-major (latent*p + feature):
        // W = [[ 0.4, -0.3,  0.2],
        //      [-0.1,  0.6,  0.5]]
        array![0.4_f64, -0.3, 0.2, -0.1, 0.6, 0.5]
    }

    fn build_mech_sparsity(weight: f64) -> MechanismSparsityPenalty {
        let t = mech_sparsity_test_target();
        let target = PsiSlice::full(t.len(), Some(2));
        MechanismSparsityPenalty::new(
            target,
            vec![vec![0_usize, 1], vec![2]],
            weight,
            1e-2,
            4.0,
            false,
        )
        .expect("valid mechanism sparsity penalty")
    }

    #[test]
    fn mechanism_sparsity_value_matches_group_norm_sum() {
        let pen = build_mech_sparsity(1.5);
        let t = mech_sparsity_test_target();
        let rho = array![0.0_f64];
        let v = pen.value(t.view(), rho.view());
        // For each latent row, sum_g sqrt(|g|) · sqrt(Σ x² + ε²)
        let eps2 = 1e-2_f64 * 1e-2_f64;
        let sqrt2 = 2.0_f64.sqrt();
        // Latent 0: group{0,1} → sqrt(2)·sqrt(0.16+0.09+eps²); group{2} → 1·sqrt(0.04+eps²)
        let l0 = sqrt2 * (0.16_f64 + 0.09 + eps2).sqrt() + (0.04_f64 + eps2).sqrt();
        // Latent 1: group{0,1} → sqrt(2)·sqrt(0.01+0.36+eps²); group{2} → 1·sqrt(0.25+eps²)
        let l1 = sqrt2 * (0.01_f64 + 0.36 + eps2).sqrt() + (0.25_f64 + eps2).sqrt();
        let expected = 1.5 * (l0 + l1);
        assert!(v.is_finite(), "mechanism-sparsity value must be finite");
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    #[test]
    fn mechanism_sparsity_grad_matches_finite_difference() {
        let pen = build_mech_sparsity(0.8);
        let t = mech_sparsity_test_target();
        let n = t.len();
        let rho = array![0.0_f64];
        let g = pen.grad_target(t.view(), rho.view());
        let eps = 1e-6;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[i] += eps;
            tm[i] -= eps;
            let fd =
                (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * eps);
            let err = (g[i] - fd).abs();
            if err > max_err {
                max_err = err;
            }
            assert_abs_diff_eq!(g[i], fd, epsilon = 1e-6);
        }
        assert!(max_err < 1e-6, "grad-FD max abs error = {max_err:.3e}");
    }

    #[test]
    fn mechanism_sparsity_hvp_matches_gradient_directional_derivative() {
        let pen = build_mech_sparsity(0.5);
        let t = mech_sparsity_test_target();
        let n = t.len();
        let rho = array![0.0_f64];
        let v: Array1<f64> =
            Array1::from_vec((0..n).map(|i| 0.2 * ((i as f64) + 1.3).cos()).collect());
        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let eps = 1e-5;
        let mut tp = t.clone();
        let mut tm = t.clone();
        for i in 0..n {
            tp[i] += eps * v[i];
            tm[i] -= eps * v[i];
        }
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let fd = (gp[i] - gm[i]) / (2.0 * eps);
            let err = (hv[i] - fd).abs();
            if err > max_err {
                max_err = err;
            }
            assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-5);
        }
        assert!(max_err < 1e-5, "hvp-FD max abs error = {max_err:.3e}");
    }

    #[test]
    fn mechanism_sparsity_rejects_groups_missing_a_feature() {
        // groups cover features {0, 2} only → feature 1 missing.
        let t = mech_sparsity_test_target();
        let target = PsiSlice::full(t.len(), Some(2));
        let err = MechanismSparsityPenalty::new(
            target,
            vec![vec![0_usize], vec![2]],
            1.0,
            1e-2,
            4.0,
            false,
        )
        .expect_err("groups missing feature 1 must error");
        assert!(
            err.contains("must partition features") && err.contains("missing feature 1"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn mechanism_sparsity_rejects_overlapping_groups() {
        // Feature 1 appears in two groups → must error before reaching the
        // missing-feature path.
        let t = mech_sparsity_test_target();
        let target = PsiSlice::full(t.len(), Some(2));
        let err = MechanismSparsityPenalty::new(
            target,
            vec![vec![0_usize, 1], vec![1, 2]],
            1.0,
            1e-2,
            4.0,
            false,
        )
        .expect_err("overlapping feature must error");
        assert!(
            err.contains("feature 1 appears in more than one group"),
            "unexpected error message: {err}"
        );
    }

    // ----- NestedPrefixPenalty (Matryoshka SAE) tests -----

    fn nested_prefix_test_target() -> (Array1<f64>, usize, usize) {
        // n_rows = 3, latent_dim = 4. Row-major (n*F + i).
        let t = array![
            1.0_f64, 2.0, 3.0, 4.0, // row 0
            -1.0, 0.5, 0.0, 2.0, // row 1
            0.1, -0.2, 0.3, -0.4, // row 2
        ];
        (t, 3, 4)
    }

    #[test]
    fn nested_prefix_grad_matches_finite_difference() {
        let (t, _n, f) = nested_prefix_test_target();
        let target = PsiSlice::full(t.len(), Some(f));
        let pen = NestedPrefixPenalty::new(
            target,
            PenaltyTier::Psi,
            vec![1_usize, 2, 4],
            vec![0.7, 0.5, 0.3],
            1e-3,
        )
        .expect("valid nested-prefix penalty");
        let rho = array![0.0_f64, 0.0, 0.0];
        let g = pen.grad_target(t.view(), rho.view());
        let eps = 1e-6;
        let mut max_err = 0.0_f64;
        for i in 0..t.len() {
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[i] += eps;
            tm[i] -= eps;
            let fd =
                (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * eps);
            let err = (g[i] - fd).abs();
            if err > max_err {
                max_err = err;
            }
            assert_abs_diff_eq!(g[i], fd, epsilon = 1e-5);
        }
        assert!(max_err < 1e-5, "grad-FD max abs error = {max_err:.3e}");
    }

    #[test]
    fn nested_prefix_hessian_diag_is_psd() {
        let (t, _n, f) = nested_prefix_test_target();
        let target = PsiSlice::full(t.len(), Some(f));
        let pen = NestedPrefixPenalty::new(
            target,
            PenaltyTier::Psi,
            vec![2_usize, 3, 4],
            vec![1.0, 0.5, 0.25],
            1e-3,
        )
        .expect("valid nested-prefix penalty");
        let rho = array![0.0_f64, 0.0, 0.0];
        let h = pen
            .hessian_diag(t.view(), rho.view())
            .expect("nested-prefix Hessian is diagonal");
        for &v in h.iter() {
            assert!(
                v >= 0.0 && v.is_finite(),
                "Hessian diag must be finite and PSD; got {v}"
            );
        }
        assert!(h[0] > 0.0);
    }

    #[test]
    fn nested_prefix_mask_is_correct() {
        assert!(file!().ends_with(".rs"));
        let (t, n_rows, f) = nested_prefix_test_target();
        let target = PsiSlice::full(t.len(), Some(f));
        let prefixes = vec![1_usize, 3, 4];
        let weights = vec![2.0_f64, 1.0, 0.5];
        let eps = 0.5;
        let pen = NestedPrefixPenalty::new(target, PenaltyTier::Psi, prefixes, weights, eps)
            .expect("valid");
        let rho = Array1::<f64>::zeros(3);
        let v = pen.value(t.view(), rho.view());

        // W_i = Σ_{k: m_k > i} λ_k.
        //   axis 0: m_k ∈ {1,3,4} > 0 → 2+1+0.5 = 3.5
        //   axis 1: {3,4} > 1 → 1+0.5 = 1.5
        //   axis 2: {3,4} > 2 → 1+0.5 = 1.5
        //   axis 3: {4} > 3 → 0.5
        let w_axis = [3.5_f64, 1.5, 1.5, 0.5];
        let mut expected = 0.0;
        let eps2 = eps * eps;
        let src = t.as_slice().unwrap();
        for n in 0..n_rows {
            for i in 0..f {
                let x = src[n * f + i];
                expected += w_axis[i] * (x * x + eps2).sqrt();
            }
        }
        assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
    }

    #[test]
    fn nested_prefix_grad_rho_matches_finite_difference() {
        assert!(file!().ends_with(".rs"));
        let (t, _n, f) = nested_prefix_test_target();
        let target = PsiSlice::full(t.len(), Some(f));
        let pen = NestedPrefixPenalty::new(
            target,
            PenaltyTier::Psi,
            vec![1_usize, 2, 4],
            vec![0.7, 0.5, 0.3],
            1e-3,
        )
        .expect("valid");
        let rho = array![0.1_f64, -0.2, 0.3];
        let dr = pen.grad_rho(t.view(), rho.view());
        let eps = 1e-6;
        for k in 0..3 {
            let mut rp = rho.clone();
            let mut rm = rho.clone();
            rp[k] += eps;
            rm[k] -= eps;
            let fd =
                (pen.value(t.view(), rp.view()) - pen.value(t.view(), rm.view())) / (2.0 * eps);
            assert_abs_diff_eq!(dr[k], fd, epsilon = 1e-5);
        }
    }

    #[test]
    fn nested_prefix_rejects_non_monotone_prefixes() {
        let target = PsiSlice::full(12, Some(4));
        let err = NestedPrefixPenalty::new(
            target,
            PenaltyTier::Psi,
            vec![2_usize, 2, 4],
            vec![1.0, 1.0, 1.0],
            1e-3,
        )
        .expect_err("non-strictly-increasing prefixes must error");
        assert!(err.contains("strictly increasing"), "got: {err}");
    }

    #[test]
    fn nested_prefix_bic_picks_best_schedule() {
        let candidates = vec![
            (vec![64_usize, 256], vec![10.0_f64, 5.0]),
            (vec![64, 256], vec![1.0, 0.5]),
            (vec![64, 256], vec![0.01, 0.005]),
        ];
        let deviances = vec![100.0_f64, 100.0, 100.0];
        let edfs = vec![50.0_f64, 50.0, 50.0];
        let (best, _bic) = select_nested_prefix_schedule(&candidates, &deviances, &edfs, 1000.0);
        assert_eq!(best, 2, "smallest λ wins under equal deviance");

        let deviances2 = vec![10.0_f64, 100.0, 100.0];
        let (best2, _) = select_nested_prefix_schedule(&candidates, &deviances2, &edfs, 1000.0);
        assert_eq!(best2, 0, "low deviance overrides shell term");
    }
}
