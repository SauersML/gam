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
//!     latent coefficient block. Promotes piecewise-constant atom maps.
//!   * [`NuclearNormPenalty`] — smoothed L¹ on singular values of a matrix
//!     latent block. Promotes low intrinsic rank without choosing a canonical
//!     axis basis.
//!   * [`BlockSparsityPenalty`] — group-lasso smoothed L¹ over predefined
//!     latent-axis blocks. Unlike per-element L¹ or per-axis L² ARD, it
//!     shrinks whole semantic groups together; pair with
//!     `LatentIdMode::AuxPriorDimSelection` when aux classes define the active
//!     group subset.
//!   * [`AuxConditionalPriorPenalty`] — iVAE-style auxiliary-conditional
//!     prior on latent rows. This fixed-precomputed v1 accepts one precision
//!     matrix per row; when ARD/Ortho fail to break the gauge, it adds the
//!     missing supervision signal (memory `project_ard_gauge_fix_doesnt_help_cogito`,
//!     `proposals/composition_engine.md` §4(c)).
//!   * [`ParametricAuxConditionalPriorPenalty`] — iVAE-style aux-conditional
//!     prior with a learnable distance-kernel map from auxiliary rows to
//!     diagonal per-row precision.
//!   * [`OrthogonalityPenalty`] — fixes the rotation gauge inside a latent
//!     block by penalizing cross-axis correlations. Pair with ARD when
//!     intrinsic dimension should be identifiable.
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
//! NuclearNorm, BlockSparsity, AuxConditionalPrior, and Orthogonality each own
//! one strength only when their weight is learnable. ParametricAuxConditional
//! owns its log-baseline precision, raw distance sensitivity, and reference
//! point coordinates, plus one strength axis when requested.
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
//! | AuxConditionalPrior | ext-coord (latent t) | 0 or 1 (log μ_aux) |
//! | ParametricAuxConditionalPrior | ext-coord (latent t) | d + d + d·du [+1 log μ_aux] |
//! | Orthogonality | ext-coord (latent t) | 0 or 1 (log μ_orth) |

use faer::Side;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, CowArray, Ix2, Ix3,
};
use std::sync::Arc;

use crate::linalg::faer_ndarray::{FaerEigh, FaerSvd};
use crate::terms::basis::{
    BasisError, DuchonNullspaceOrder, duchon_radial_first_derivative_nd,
    duchon_radial_second_derivative_nd, duchon_radial_third_derivative_nd,
};
use crate::terms::penalty_op::PenaltyOp;
use crate::terms::smooth::BlockwisePenalty;

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
    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64>;

    /// Diagonal of the Hessian `diag(∂²P/∂target²)` when the Hessian is
    /// block-diagonal. Returns `None` for penalties whose Hessian is dense
    /// (Isometry); those implement [`Self::hvp`] instead.
    fn hessian_diag(
        &self,
        _target: ArrayView1<'_, f64>,
        _rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
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
        unimplemented!("hvp must be implemented for non-diagonal analytic penalties");
    }

    /// Gradient of the penalty value w.r.t. each owned ρ-axis. Length equals
    /// [`Self::rho_count`].
    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64>;

    /// Number of REML-selectable hyperparameter axes this penalty contributes
    /// to the outer ρ vector.
    fn rho_count(&self) -> usize;

    /// Human-readable identifier for diagnostics / logging.
    fn name(&self) -> &str;
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
    pub fn with_duchon_radial_source(
        mut self,
        source: Arc<IsometryDuchonRadialSource>,
    ) -> Self {
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
        let mu = rho[self.rho_index].exp();
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
                                k_a_cd_w_j_b +=
                                    jac3[[n, i, ((a * d) + c) * d + dd]] * wj[[i, b]];
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
                                j_a_w_k_b_cd +=
                                    wj[[i, a]] * jac3[[n, i, ((b * d) + c) * d + dd]];
                            }
                            bv += (k_a_cd_w_j_b
                                + h_a_c_w_h_b_d
                                + h_a_d_w_h_b_c
                                + j_a_w_k_b_cd)
                                * vd;
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
        let mu = rho[self.rho_index].exp();
        let mut acc = 0.0;
        for n in 0..n_obs {
            for k in 0..(d * d) {
                let diff = g[[n, k]] - g_ref[[n, k]];
                acc += diff * diff;
            }
        }
        0.5 * mu * acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
        let mu = rho[self.rho_index].exp();
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
/// and can be indefinite because entropy is concave in assignment space; the
/// diagonal returned here is the positive Gauss-Newton damping used by the
/// row-local arrow solve.
#[derive(Debug, Clone)]
pub struct SoftmaxAssignmentSparsityPenalty {
    pub k_atoms: usize,
    pub temperature: f64,
}

impl SoftmaxAssignmentSparsityPenalty {
    #[must_use]
    pub fn new(k_atoms: usize, temperature: f64) -> Self {
        debug_assert!(k_atoms > 0);
        debug_assert!(temperature > 0.0);
        Self {
            k_atoms,
            temperature,
        }
    }

    fn softmax_row(&self, row: &[f64]) -> Vec<f64> {
        let inv_tau = 1.0 / self.temperature;
        let mut max_scaled = f64::NEG_INFINITY;
        for &v in row {
            max_scaled = max_scaled.max(v * inv_tau);
        }
        let mut out = vec![0.0; self.k_atoms];
        let mut sum = 0.0;
        for i in 0..self.k_atoms {
            let v = (row[i] * inv_tau - max_scaled).exp();
            out[i] = v;
            sum += v;
        }
        if sum == 0.0 || !sum.is_finite() {
            out.fill(1.0 / self.k_atoms as f64);
        } else {
            for v in out.iter_mut() {
                *v /= sum;
            }
        }
        out
    }
}

impl AnalyticPenalty for SoftmaxAssignmentSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let lambda = rho[0].exp();
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

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let lambda = rho[0].exp();
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
        let lambda = rho[0].exp();
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (self.temperature * self.temperature);
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            for k in 0..self.k_atoms {
                out[start + k] = lambda * a[k] * (1.0 - a[k]) * inv_tau2;
            }
        }
        Some(out)
    }

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        Array1::from_vec(vec![self.value(target, rho)])
    }

    fn rho_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "softmax_assignment_sparsity"
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
/// The target is row-major `(N, K)` logits. For MAP we drop Gumbel noise and
/// use `z_ik = sigmoid(logit_ik / tau)`. Each column has
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
    pub learnable_alpha: bool,
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
            learnable_alpha,
        }
    }

    fn resolved_alpha(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_alpha {
            self.alpha * rho[0].exp()
        } else {
            self.alpha
        }
    }

    fn sigmoid_logits(&self, target: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        for i in 0..target.len() {
            let x = target[i] / self.tau;
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
        let z = self.sigmoid_logits(target);
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
        acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let z = self.sigmoid_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(1.0e-12, 1.0 - 1.0e-12);
                let d_p_d_z = ((1.0 - pk) / pk).ln();
                out[start + k] = d_p_d_z * zk * (1.0 - zk) / self.tau;
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
        let z = self.sigmoid_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (self.tau * self.tau);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(1.0e-12, 1.0 - 1.0e-12);
                let d_p_d_z = ((1.0 - pk) / pk).ln();
                out[start + k] = d_p_d_z * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
            }
        }
        Some(out)
    }

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        if !self.learnable_alpha {
            return Array1::<f64>::zeros(0);
        }
        let alpha = self.resolved_alpha(rho);
        let z = self.sigmoid_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let mut sum_log_pi = 0.0;
        for &pk in pi.iter() {
            sum_log_pi += pk.clamp(1.0e-12, 1.0 - 1.0e-12).ln();
        }
        Array1::from_vec(vec![-alpha * sum_log_pi / self.k_max as f64])
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_alpha)
    }

    fn name(&self) -> &str {
        "ibp_assignment_map"
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
            strength_rho_index: 0,
            eps_rho_index: None,
        }
    }

    #[must_use]
    pub fn with_eps_reml(mut self, eps_rho_index: usize) -> Self {
        self.eps_rho_index = Some(eps_rho_index);
        self
    }

    /// Resolve `(strength, eps_or_delta)` from the current ρ view.
    fn resolved(&self, rho: ArrayView1<'_, f64>) -> (f64, f64) {
        let strength = rho[self.strength_rho_index].exp();
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

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
                let _ = d;
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
///   P_j(t; ρ) = ½ exp(ρ_j) · ‖t[:, j]‖² - (n_eff / 2) · ρ_j
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
            rho_indices,
            n_eff: n_obs as f64,
        }
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
            let lam_j = rho_j.exp();
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            acc += 0.5 * lam_j * sq - 0.5 * self.n_eff * rho_j;
        }
        acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut g = Array1::<f64>::zeros(target.len());
        for j in 0..d {
            let lam_j = rho[self.rho_indices[j]].exp();
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
            let lam_j = rho[self.rho_indices[j]].exp();
            for n in 0..n_obs {
                diag[n * d + j] = lam_j;
            }
        }
        Some(diag)
    }

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Occam-corrected REML penalty contribution per axis:
        //
        //   P_j(ρ_j) = ½ exp(ρ_j) Σ_n t_{n,j}²  −  (N_eff/2) · ρ_j  + const
        //
        // (the −(N/2) ρ comes from the −½ log|S| Gaussian normalizing
        // constant under prior precision λ_j = exp(ρ_j)).
        //
        //   ∂P_j/∂ρ_j = ½ exp(ρ_j) Σ_n t_{n,j}²  −  N_eff/2.
        //
        // At an unused axis Σ_n t_{n,j}² = 0 the gradient is −N_eff/2 < 0;
        // minimising the (negative-log) marginal drives ρ_j → +∞ and
        // prunes the axis, recovering ARD's pruning behaviour.
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(self.rho_count());
        for j in 0..d {
            let lam_j = rho[self.rho_indices[j]].exp();
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

/// Smoothed-L¹ total variation on a row-major `(n_eff, d)` latent block.
///
/// Uses the differentiable Huber-style kernel `φ(x)=sqrt(x²+ε²)-ε`.
/// The difference operator defines the prior shape: forward 1-D differences
/// for ordered context windows, or graph edges for adjacency-structured atoms.
/// Pair TV with Orthogonality when piecewise-constant atoms need a gauge-fixed,
/// interpretable basis.
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
                    "TotalVariationPenalty::new GraphEdges requires at least one edge"
                        .to_string(),
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
        })
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || target_len % self.n_eff != 0 {
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
    pub fn as_dense(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
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

    fn grad_target(
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
        if target.len() % n_eff != 0 {
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
        })
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || target_len % self.n_eff != 0 {
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
    ) -> (Array2<f64>, Array2<f64>) {
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
            .expect("NuclearNormPenalty Gram eigendecomposition failed");
        let active_start = d.saturating_sub(self.rank_limit(d));
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

        (right_filter, right_filter_derivative)
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

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
        let (right_filter, right_filter_derivative) =
            self.right_spectral_inverse_sqrt_derivative(t.view(), v_mat.view());
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
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
        if target.len() % n_eff != 0 {
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
        })
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || target_len % self.n_eff != 0 {
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
            let s = self.group_norm(t.view(), group);
            let inv_s = 1.0 / s;
            let inv_s3 = inv_s * inv_s * inv_s;
            for n in 0..t.nrows() {
                for &axis in group {
                    let x = t[[n, axis]];
                    out[n * t.ncols() + axis] = weight * (inv_s - x * x * inv_s3);
                }
            }
        }
        out
    }

    /// Materialize the group-lasso Hessian for small-block spectral paths.
    pub fn as_dense(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let n = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        for group in &self.groups {
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
                            dense[[i, j]] = weight * entry;
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
            acc += self.group_norm(t.view(), group);
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for group in &self.groups {
            let s = self.group_norm(t.view(), group);
            let factor = weight / s;
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
                        weight * (v_mat[[n, axis]] * inv_s - t[[n, axis]] * inner * inv_s3);
                }
            }
        }
        Self::flatten_matrix(&out)
    }

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
}

// ---------------------------------------------------------------------------
// Aux-conditional prior penalty
// ---------------------------------------------------------------------------

/// iVAE-style auxiliary-conditional prior on the latent block.
///
/// Fixed-precomputed v1 of `p(t_n | u_n) ∝ exp(-½ t_nᵀ Λ_n t_n)`: callers pass
/// one positive-definite precision matrix per row. This is the missing sibling
/// to ARD/Ortho/sparsity from `proposals/composition_engine.md` §4(c); when
/// ARD/Ortho fail to break the gauge, aux-conditional precision adds the HSV
/// supervision signal identified in memory `project_ard_gauge_fix_doesnt_help_cogito`.
#[derive(Debug, Clone)]
pub struct AuxConditionalPriorPenalty {
    pub lambda_per_row: Array3<f64>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub target: PsiSlice,
}

impl AuxConditionalPriorPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        lambda_per_row: Array3<f64>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("AuxConditionalPriorPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "AuxConditionalPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("AuxConditionalPriorPenalty::new requires n_eff > 0".to_string());
        }
        if target.len() % n_eff != 0 {
            return Err(format!(
                "AuxConditionalPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "AuxConditionalPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "AuxConditionalPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "AuxConditionalPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (lambda_n, lambda_rows, lambda_cols) = lambda_per_row.dim();
        if lambda_n != n_eff || lambda_rows != latent_dim || lambda_cols != latent_dim {
            return Err(format!(
                "AuxConditionalPriorPenalty::new lambda_per_row shape must be ({n_eff}, {latent_dim}, {latent_dim}), got ({lambda_n}, {lambda_rows}, {lambda_cols})"
            ));
        }
        for n in 0..n_eff {
            let mut matrix = Array2::<f64>::zeros((latent_dim, latent_dim));
            for i in 0..latent_dim {
                for j in 0..latent_dim {
                    let value = lambda_per_row[[n, i, j]];
                    if !value.is_finite() {
                        return Err(format!(
                            "AuxConditionalPriorPenalty::new lambda_per_row[{n},{i},{j}] must be finite"
                        ));
                    }
                    let transpose = lambda_per_row[[n, j, i]];
                    if (value - transpose).abs() >= 1.0e-10 {
                        return Err(format!(
                            "AuxConditionalPriorPenalty::new lambda_per_row[{n}] must be symmetric; |Λ[{i},{j}] - Λ[{j},{i}]| = {:.3e}",
                            (value - transpose).abs()
                        ));
                    }
                    matrix[[i, j]] = value;
                }
            }
            let (evals, _) = matrix.eigh(Side::Lower).map_err(|err| {
                format!("AuxConditionalPriorPenalty::new lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            let min_eval = evals.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
            if !(min_eval.is_finite() && min_eval > 0.0) {
                return Err(format!(
                    "AuxConditionalPriorPenalty::new lambda_per_row[{n}] must be positive definite; minimum eigenvalue {min_eval:.3e}"
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
        })
    }

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            self.weight * rho[self.rho_index].exp()
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || target_len % self.n_eff != 0 {
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
    pub fn as_dense(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
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
                "AuxConditionalPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
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
                format!("AuxConditionalPriorPenalty::log_det_plus_lambda_i lambda_per_row[{n}] eigendecomposition failed: {err}")
            })?;
            for &eval in evals.iter() {
                let shifted = weight * eval + lambda;
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "AuxConditionalPriorPenalty::log_det_plus_lambda_i non-positive shifted eigenvalue {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}

impl AnalyticPenalty for AuxConditionalPriorPenalty {
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
        0.5 * self.resolved_weight(rho) * acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
        "aux_conditional_prior"
    }
}

// ---------------------------------------------------------------------------
// Parametric auxiliary-conditional prior penalty
// ---------------------------------------------------------------------------

/// Parametric iVAE auxiliary-conditional prior on the latent block.
///
/// Distance-kernel variant of `p(t_n | u_n) ∝ exp(-½ t_nᵀ Λ(u_n)t_n)` with
/// diagonal precision `λ_k(u_n) = exp(log_alpha_k) + softplus(raw_beta_k)
/// ||u_n - μ_k||²`. Use the FIXED variant when Λ is known per row; use this
/// PARAMETRIC variant when REML should learn the aux-to-Λ mapping.
///
/// Motivation: `examples/aux_conditional_prior_demo.py` (b31w36m23) showed
/// that AuxConditional + ARD recovers iVAE §4(c) identifiability on synthetic
/// data with `axes_kept = [4, 4, 2]`; ML workflows such as cogito steering and
/// gene-expression iVAE need learnable `Λ(u_n) = f(u_n; ψ)` rather than
/// externally precomputed row inverses.
#[derive(Debug, Clone)]
pub struct ParametricAuxConditionalPriorPenalty {
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
}

impl ParametricAuxConditionalPriorPenalty {
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
                "ParametricAuxConditionalPriorPenalty::new requires a non-empty target".to_string(),
            );
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err(
                "ParametricAuxConditionalPriorPenalty::new requires n_eff > 0".to_string(),
            );
        }
        if target.len() % n_eff != 0 {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if latent_dim == 0 {
            return Err(
                "ParametricAuxConditionalPriorPenalty::new requires latent_dim > 0".to_string(),
            );
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "ParametricAuxConditionalPriorPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
            if expected_dim != latent_dim {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new inferred latent_dim {latent_dim} does not match target latent_dim {expected_dim}"
                ));
            }
        }
        let (aux_n, aux_dim) = aux.dim();
        if aux_n != n_eff {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new aux rows must equal n_eff {n_eff}, got {aux_n}"
            ));
        }
        if aux_dim == 0 {
            return Err(
                "ParametricAuxConditionalPriorPenalty::new requires aux dimension > 0".to_string(),
            );
        }
        if log_alpha.len() != latent_dim {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new log_alpha length must equal latent_dim {latent_dim}, got {}",
                log_alpha.len()
            ));
        }
        if raw_beta.len() != latent_dim {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new raw_beta length must equal latent_dim {latent_dim}, got {}",
                raw_beta.len()
            ));
        }
        let (mu_rows, mu_cols) = mu.dim();
        if mu_rows != latent_dim || mu_cols != aux_dim {
            return Err(format!(
                "ParametricAuxConditionalPriorPenalty::new mu shape must be ({latent_dim}, {aux_dim}), got ({mu_rows}, {mu_cols})"
            ));
        }
        for (idx, &value) in aux.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new aux[{idx}] must be finite"
                ));
            }
        }
        for k in 0..latent_dim {
            let log_alpha_k = log_alpha[k];
            if !log_alpha_k.is_finite() {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new log_alpha[{k}] must be finite"
                ));
            }
            let alpha_k = log_alpha_k.exp();
            if !(alpha_k.is_finite() && alpha_k > 0.0) {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new exp(log_alpha[{k}]) must be finite and > 0"
                ));
            }
            let raw_beta_k = raw_beta[k];
            if !raw_beta_k.is_finite() {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new raw_beta[{k}] must be finite"
                ));
            }
            let beta_k = stable_softplus(raw_beta_k);
            if !(beta_k.is_finite() && beta_k >= 0.0) {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new softplus(raw_beta[{k}]) must be finite and >= 0"
                ));
            }
        }
        for (idx, &value) in mu.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "ParametricAuxConditionalPriorPenalty::new mu[{idx}] must be finite"
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
        })
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || target_len % self.n_eff != 0 {
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
        alpha + beta * self.dist2(n, k, rho)
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
    pub fn as_dense(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
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
                "ParametricAuxConditionalPriorPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let weight = self.resolved_weight(rho);
        let mut sum = 0.0;
        for n in 0..self.n_eff {
            for k in 0..self.log_alpha.len() {
                let shifted = lambda + weight * self.lambda_at(n, k, rho);
                if !(shifted.is_finite() && shifted > 0.0) {
                    return Err(format!(
                        "ParametricAuxConditionalPriorPenalty::log_det_plus_lambda_i non-positive shifted diagonal {shifted:.3e}"
                    ));
                }
                sum += shifted.ln();
            }
        }
        Ok(sum)
    }
}

impl AnalyticPenalty for ParametricAuxConditionalPriorPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for n in 0..t.nrows() {
            for k in 0..t.ncols() {
                acc += self.lambda_at(n, k, rho) * t[[n, k]] * t[[n, k]];
            }
        }
        0.5 * self.resolved_weight(rho) * acc
    }

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(self.rho_count());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(self.rho_count());
        let d = t.ncols();
        let du = self.aux.ncols();
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
                grad_alpha_direct += 0.5 * weight * sq;
                grad_beta_direct += 0.5 * weight * sq * r2;
                for a in 0..du {
                    let delta = self.aux[[n, a]] - self.active_mu(k, a, rho);
                    grad_mu_direct[a] += -weight * sq * beta * delta;
                }
            }
            out[self.log_alpha_offset() + k] = grad_alpha_direct * alpha;
            out[self.raw_beta_offset() + k] = grad_beta_direct * beta_jac;
            for a in 0..du {
                out[self.mu_offset() + k * du + a] = grad_mu_direct[a];
            }
        }
        if self.learnable_weight {
            out[self.weight_offset()] = self.value(target, rho);
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.log_alpha.len() + self.raw_beta.len() + self.mu.len() + usize::from(self.learnable_weight)
    }

    fn name(&self) -> &str {
        "parametric_aux_conditional_prior"
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
/// near-zero noise; paired with AuxConditional/Parametric priors, the aux prior
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
        if target.len() % n_eff != 0 {
            return Err(format!(
                "ScadMcpPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "ScadMcpPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "ScadMcpPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        if !(gamma.is_finite() && gamma > 1.0) {
            return Err(format!(
                "ScadMcpPenalty::new requires finite gamma > 1, got {gamma}"
            ));
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
        })
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
                    weight * r - (r * r - self.smoothing_eps * self.smoothing_eps) / (2.0 * self.gamma)
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
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    let concavity_grad = if t * t > eps2 {
                        t / self.gamma
                    } else {
                        0.0
                    };
                    weight * t / r - concavity_grad
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                if r <= weight {
                    weight * t / r
                } else if r <= self.gamma * weight {
                    let concavity_grad = if t * t > eps2 {
                        t / (self.gamma - 1.0)
                    } else {
                        0.0
                    };
                    self.gamma * weight * t / ((self.gamma - 1.0) * r) - concavity_grad
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
                    let concavity_hess = if t * t > eps2 {
                        1.0 / self.gamma
                    } else {
                        0.0
                    };
                    weight * eps2 / (r * r * r) - concavity_hess
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                if r <= weight {
                    weight * eps2 / (r * r * r)
                } else if r <= self.gamma * weight {
                    let concavity_hess = if t * t > eps2 {
                        1.0 / (self.gamma - 1.0)
                    } else {
                        0.0
                    };
                    self.gamma * weight * eps2 / ((self.gamma - 1.0) * r * r * r)
                        - concavity_hess
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

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
        if target.len() % latent_dim != 0 {
            return Err(format!(
                "OrthogonalityPenalty::new target length {} is not divisible by latent_dim {}",
                target.len(),
                latent_dim
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
        Ok(Self {
            target,
            latent_dim,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
        })
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
        if target.len() % d != 0 {
            debug_assert_eq!(target.len() % d, 0, "target length must be divisible by latent_dim");
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
    pub fn as_blockwise(&self, _global_offset: usize) -> Option<Vec<BlockwisePenalty>> {
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

    fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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

    fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
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
#[derive(Clone)]
pub enum AnalyticPenaltyKind {
    Isometry(Arc<IsometryPenalty>),
    Sparsity(Arc<SparsityPenalty>),
    SoftmaxAssignmentSparsity(Arc<SoftmaxAssignmentSparsityPenalty>),
    IBPAssignment(Arc<IBPAssignmentPenalty>),
    Ard(Arc<ARDPenalty>),
    TotalVariation(Arc<TotalVariationPenalty>),
    NuclearNorm(Arc<NuclearNormPenalty>),
    BlockSparsity(Arc<BlockSparsityPenalty>),
    AuxConditionalPrior(Arc<AuxConditionalPriorPenalty>),
    ParametricAuxConditionalPrior(Arc<ParametricAuxConditionalPriorPenalty>),
    ScadMcp(Arc<ScadMcpPenalty>),
    Orthogonality(Arc<OrthogonalityPenalty>),
}

impl AnalyticPenaltyKind {
    pub fn tier(&self) -> PenaltyTier {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.tier(),
            AnalyticPenaltyKind::Sparsity(p) => p.tier(),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.tier(),
            AnalyticPenaltyKind::IBPAssignment(p) => p.tier(),
            AnalyticPenaltyKind::Ard(p) => p.tier(),
            AnalyticPenaltyKind::TotalVariation(p) => p.tier(),
            AnalyticPenaltyKind::NuclearNorm(p) => p.tier(),
            AnalyticPenaltyKind::BlockSparsity(p) => p.tier(),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.tier(),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.tier(),
            AnalyticPenaltyKind::ScadMcp(p) => p.tier(),
            AnalyticPenaltyKind::Orthogonality(p) => p.tier(),
        }
    }

    pub fn rho_count(&self) -> usize {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.rho_count(),
            AnalyticPenaltyKind::Sparsity(p) => p.rho_count(),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.rho_count(),
            AnalyticPenaltyKind::IBPAssignment(p) => p.rho_count(),
            AnalyticPenaltyKind::Ard(p) => p.rho_count(),
            AnalyticPenaltyKind::TotalVariation(p) => p.rho_count(),
            AnalyticPenaltyKind::NuclearNorm(p) => p.rho_count(),
            AnalyticPenaltyKind::BlockSparsity(p) => p.rho_count(),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.rho_count(),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.rho_count(),
            AnalyticPenaltyKind::ScadMcp(p) => p.rho_count(),
            AnalyticPenaltyKind::Orthogonality(p) => p.rho_count(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.name(),
            AnalyticPenaltyKind::Sparsity(p) => p.name(),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.name(),
            AnalyticPenaltyKind::IBPAssignment(p) => p.name(),
            AnalyticPenaltyKind::Ard(p) => p.name(),
            AnalyticPenaltyKind::TotalVariation(p) => p.name(),
            AnalyticPenaltyKind::NuclearNorm(p) => p.name(),
            AnalyticPenaltyKind::BlockSparsity(p) => p.name(),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.name(),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.name(),
            AnalyticPenaltyKind::ScadMcp(p) => p.name(),
            AnalyticPenaltyKind::Orthogonality(p) => p.name(),
        }
    }

    pub fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.value(target, rho),
            AnalyticPenaltyKind::Sparsity(p) => p.value(target, rho),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.value(target, rho),
            AnalyticPenaltyKind::IBPAssignment(p) => p.value(target, rho),
            AnalyticPenaltyKind::Ard(p) => p.value(target, rho),
            AnalyticPenaltyKind::TotalVariation(p) => p.value(target, rho),
            AnalyticPenaltyKind::NuclearNorm(p) => p.value(target, rho),
            AnalyticPenaltyKind::BlockSparsity(p) => p.value(target, rho),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.value(target, rho),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.value(target, rho),
            AnalyticPenaltyKind::ScadMcp(p) => p.value(target, rho),
            AnalyticPenaltyKind::Orthogonality(p) => p.value(target, rho),
        }
    }

    pub fn grad_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::Sparsity(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::IBPAssignment(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::Ard(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::TotalVariation(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::NuclearNorm(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::BlockSparsity(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::ScadMcp(p) => p.grad_target(target, rho),
            AnalyticPenaltyKind::Orthogonality(p) => p.grad_target(target, rho),
        }
    }

    pub fn grad_rho(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::Sparsity(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::IBPAssignment(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::Ard(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::TotalVariation(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::NuclearNorm(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::BlockSparsity(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::ScadMcp(p) => p.grad_rho(target, rho),
            AnalyticPenaltyKind::Orthogonality(p) => p.grad_rho(target, rho),
        }
    }

    pub fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::Sparsity(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::IBPAssignment(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::Ard(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::TotalVariation(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::NuclearNorm(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::BlockSparsity(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::ScadMcp(p) => p.hessian_diag(target, rho),
            AnalyticPenaltyKind::Orthogonality(p) => p.hessian_diag(target, rho),
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
            AnalyticPenaltyKind::Isometry(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::Sparsity(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::IBPAssignment(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::Ard(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::TotalVariation(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::NuclearNorm(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::BlockSparsity(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::AuxConditionalPrior(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::ScadMcp(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::Orthogonality(p) => p.hvp(target, rho, v),
        }
    }
}

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
        // diagonal). Dense HVP-only penalties keep exact probing only below
        // the small-block threshold.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("ARD diag"),
            AnalyticPenaltyKind::TotalVariation(p) => match &p.difference_op {
                DifferenceOpKind::GraphEdges(_)
                    if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
                {
                    self.stochastic_diag_via_matvec()
                }
                DifferenceOpKind::GraphEdges(_) | DifferenceOpKind::ForwardDiff1D => {
                    p.diag_target(self.target.view(), self.rho.view())
                }
            },
            AnalyticPenaltyKind::Orthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::Orthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NuclearNorm(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::NuclearNorm(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::BlockSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::BlockSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::AuxConditionalPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ScadMcp(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::IBPAssignment(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("IBP assignment diag"),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("softmax assignment diag"),
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
        // BlockSparsity, and Orthogonality keep the exact dense eigensolve
        // only below the small-block threshold; large blocks use SLQ against
        // the analytic HVP.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::IBPAssignment(_) => {
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
                DifferenceOpKind::ForwardDiff1D => p.log_det_plus_lambda_i_forward_1d(
                    self.target.view(),
                    self.rho.view(),
                    lambda,
                ),
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
            AnalyticPenaltyKind::Orthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::NuclearNorm(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::AuxConditionalPrior(p) => {
                p.log_det_plus_lambda_i(self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => {
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
            AnalyticPenaltyKind::Isometry(_) | AnalyticPenaltyKind::Orthogonality(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
            AnalyticPenaltyKind::NuclearNorm(_) | AnalyticPenaltyKind::BlockSparsity(_) => {
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
            AnalyticPenaltyKind::AuxConditionalPrior(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::ParametricAuxConditionalPrior(p) => {
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
                let mut d = Array1::<f64>::zeros(n);
                let mut v = Array2::<f64>::zeros(t.dim());
                for i in 0..n {
                    let row = i / latent_dim;
                    let col = i % latent_dim;
                    v[[row, col]] = 1.0;
                    let h = p.hvp_with_precomputed_m(t.view(), gram.view(), v.view(), scale);
                    d[i] = h[[row, col]];
                    v[[row, col]] = 0.0;
                }
                return d;
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
            format!("FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ eigendecomposition failed: {e}")
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
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
}
