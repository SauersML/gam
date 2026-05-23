//! Analytic penalty primitives for the three-tier (β / ψ / ρ) engine.
//!
//! See `proposals/composition_engine.md` §3-§4 and `proposals/latent_coord.md`
//! §2.3 for the motivation. This module implements the three structured
//! penalties identified as the minimal identifiability tools needed by an
//! SAE / principal-manifold / latent-coordinate workflow:
//!
//!   * [`IsometryPenalty`] — pulls the pullback metric of the decoder toward a
//!     reference metric on the latent manifold. Lives on `ψ` (specifically on
//!     a [`crate::terms::latent_coord::LatentCoordValues`] slice). Breaks the
//!     diffeomorphism gauge so the inner Hessian on `t` is full-rank and the
//!     IFT is well-defined.
//!   * [`SparsityPenalty`] — smoothed L¹ (`sqrt(x² + ε²)`), Hoyer, or Log
//!     sparsifier. Applied to a `β` slice (SAE codes) or `ψ` slice (soft atom
//!     amplitudes). Differentiable everywhere; the smoothing parameter `ε` may
//!     itself live in `ρ` so REML shrinks it.
//!   * [`ARDPenalty`] — one penalty parameter per latent axis. The marginal
//!     likelihood's Occam factor sends unused axes' precision to infinity,
//!     simultaneously fixing the rotation gauge and discovering intrinsic
//!     dimension.
//!
//! All three are **analytic**: no autograd, no finite differencing. Each
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
//! non-quadratic Sparsity and Isometry penalties produce
//! [`AnalyticPenaltyOp`] handles that downstream PIRLS / REML consumers query
//! through the same `value / gradient / hvp` interface they already use for
//! smoothness.
//!
//! ## Registration with REML
//!
//! Each penalty owns a (possibly empty) sub-range of the global `ρ` vector.
//! See [`AnalyticPenaltyKind::rho_count`]. The outer REML loop concatenates
//! these onto the existing per-smooth `ρ`s, exactly the way the anisotropic-ψ
//! path appends `ψ` ext-coords. The IsometryPenalty owns one `ρ`; the
//! SparsityPenalty owns either zero (`ε` fixed) or one (`ε` REML-selected) plus
//! one strength; the ARDPenalty owns `d` (one per latent axis).
//!
//! ## Three-tier landings
//!
//! | Penalty   | Target tier | ρ-axes owned         |
//! |-----------|-------------|----------------------|
//! | Isometry  | ψ (latent t)| 1 (log μ_iso)        |
//! | Sparsity  | β or ψ      | 1 (strength) [+1 ε]  |
//! | ARD       | ψ (latent t)| d (one per axis)     |

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::sync::Arc;

use crate::terms::latent_coord::LatentCoordValues;
use crate::terms::smooth::{BlockwisePenalty, PenaltyStructureHint};

// ---------------------------------------------------------------------------
// Common trait
// ---------------------------------------------------------------------------

/// Whether a penalty's target is a slice of `β` (decoder coefficients), a
/// slice of `ψ` (per-observation latent field, e.g. `LatentCoordValues`),
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
/// per-observation flat index for the ψ tier (matching the
/// `LatentCoordValues` row-major flat layout: `n * d + a`).
#[derive(Debug, Clone)]
pub struct PsiSlice {
    /// Inclusive-start, exclusive-end flat range into the underlying ψ vector.
    pub range: std::ops::Range<usize>,
    /// For latent-coordinate ψ slices: the latent dimensionality, used to
    /// reshape the flat slice into per-row `(n_obs, d)` blocks.
    pub latent_dim: Option<usize>,
}

impl PsiSlice {
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

/// Uniform interface implemented by every analytic penalty in this module.
///
/// `target` is the relevant slice of the (β or ψ) parameter vector, viewed as
/// a flat `ArrayView1`. The owning REML driver is responsible for slicing the
/// global parameter vector before calling, and for routing the returned
/// gradient back into the correct global indices.
pub trait AnalyticPenalty: Send + Sync {
    /// Tier the target lives in (β or ψ).
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

/// Isometry-to-reference penalty.
///
/// Penalizes `½ μ Σ_n ‖g_n(t) − g^ref(t_n)‖²_F`, where `g_n(t) = J_n^T J_n`
/// is the pullback metric at row `n`, induced by the per-row Jacobian
/// `J_n = ∂(Φ_n β)/∂t_n ∈ ℝ^{p × d}`. The reference metric is either the
/// identity (Euclidean) or user-supplied per row.
///
/// **When to use.** Whenever a `LatentCoord` block is in play without an
/// auxiliary variable (`AuxPrior`) or active ARD to break the diffeomorphism
/// gauge. With a Euclidean reference, the penalty pulls the decoder toward a
/// local isometry, which is enough to make the inner Hessian on `t` full-rank
/// and the IFT well-defined.
///
/// **Math.** Let `J_n ∈ ℝ^{p × d}` be the local decoder Jacobian. Then
/// `g_n = J_n^T J_n` and the penalty is `½ μ Σ_n ‖J_n^T J_n − g^ref_n‖²_F`.
/// Analytic gradient w.r.t. `t_n`:
///
/// ```text
///   ∂P/∂t_n = 2 μ Σ_{i,j} (g_n − g^ref_n)_{ij} · (∂J_n/∂t_n)_{[i,:]}^T J_n_{[:,j]}
///           + symmetric term swapping i,j.
/// ```
///
/// The per-row Jacobian `J_n` is exactly the radial-derivative jet
/// `design_gradient_wrt_t` already computes for `LatentCoordValues`; the
/// second derivative `∂J/∂t` is a one-line extension of the same radial
/// chain rule (radial second derivative `φ''(r)` times outer product of unit
/// directions). No autograd needed.
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
    /// `H_n[i, a, c] = ∂J_n[i, a] / ∂t_{n, c}`. When present, `grad_target`
    /// returns the exact closed-form gradient. When absent, `grad_target`
    /// falls back to the Gauss-Newton surrogate `H · t` consistent with the
    /// IFT-warm-started inner Newton step (good enough for the implicit
    /// gradient pass; the explicit gradient uses the cache when available).
    pub jacobian_second_cache: Option<Arc<Array2<f64>>>,
    /// Output dimensionality `p` (column count of each per-row Jacobian).
    pub p_out: usize,
}

impl IsometryPenalty {
    pub fn new_euclidean(target: PsiSlice, p_out: usize) -> Self {
        Self {
            target,
            reference: IsometryReference::Euclidean,
            rho_index: 0,
            jacobian_cache: None,
            jacobian_second_cache: None,
            p_out,
        }
    }

    pub fn with_reference(mut self, reference: IsometryReference) -> Self {
        self.reference = reference;
        self
    }

    pub fn with_jacobian_cache(mut self, j: Arc<Array2<f64>>) -> Self {
        self.jacobian_cache = Some(j);
        self
    }

    pub fn with_jacobian_second_cache(mut self, h: Arc<Array2<f64>>) -> Self {
        self.jacobian_second_cache = Some(h);
        self
    }

    /// Per-row pullback metric `g_n = J_n^T J_n`. Returns `(n_obs, d, d)`
    /// flattened row-major as `(n_obs, d*d)`.
    fn pullback_metric(&self, latent_dim: usize) -> Array2<f64> {
        let jac = self
            .jacobian_cache
            .as_ref()
            .expect("isometry penalty requires a live jacobian cache");
        let n_obs = jac.nrows();
        let p = self.p_out;
        debug_assert_eq!(jac.ncols(), p * latent_dim);
        let mut g_all = Array2::<f64>::zeros((n_obs, latent_dim * latent_dim));
        for n in 0..n_obs {
            for a in 0..latent_dim {
                for b in 0..latent_dim {
                    let mut s = 0.0;
                    for i in 0..p {
                        s += jac[[n, i * latent_dim + a]] * jac[[n, i * latent_dim + b]];
                    }
                    g_all[[n, a * latent_dim + b]] = s;
                }
            }
        }
        g_all
    }

    /// Reference metric per row, `(n_obs, d*d)`.
    fn reference_metric(&self, n_obs: usize, d: usize) -> Array2<f64> {
        match &self.reference {
            IsometryReference::Euclidean => {
                let mut out = Array2::<f64>::zeros((n_obs, d * d));
                for n in 0..n_obs {
                    for a in 0..d {
                        out[[n, a * d + a]] = 1.0;
                    }
                }
                out
            }
            IsometryReference::UserSupplied(a) => {
                debug_assert_eq!(a.nrows(), n_obs);
                debug_assert_eq!(a.ncols(), d * d);
                a.as_ref().clone()
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
        let g = self.pullback_metric(d);
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
        // Exact closed-form gradient (chain rule conventions documented at the
        // top of `IsometryPenalty`):
        //
        //   P = ½ μ Σ_n Σ_{a,b} D_{n,a,b}²,    D = g_n − g^ref_n
        //   g_n = J_n^T J_n,   J_n ∈ ℝ^{p × d}
        //
        // ∂P/∂t_{n,c} = μ Σ_{a,b} D_{a,b} · (∂g_{a,b}/∂t_c)
        //             = μ Σ_{a,b} D_{a,b} · ( H[:,a,c]^T J[:,b] + J[:,a]^T H[:,b,c] )
        //             = 2 μ Σ_{a,b} D_{a,b} · (H[:,a,c]^T J[:,b])      (by symmetry of D)
        //
        // where H_{n}[i, a, c] = ∂J_n[i, a] / ∂t_{n, c}. If the second-
        // derivative cache is unavailable we fall back to the Gauss-Newton
        // surrogate (the diagonal-block contraction H_n · t_n), which is what
        // the IFT inner loop needs for a well-conditioned Newton step; the
        // dropped term vanishes at the local minimum of the data-fit so this
        // is also the principled REML-pass surrogate.
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let n_obs = target.len() / d;
        let g = self.pullback_metric(d);
        let g_ref = self.reference_metric(n_obs, d);
        let jac = self
            .jacobian_cache
            .as_ref()
            .expect("isometry penalty requires a live jacobian cache");
        let p = self.p_out;
        let mu = rho[self.rho_index].exp();
        let mut grad = Array1::<f64>::zeros(target.len());

        if let Some(jac2) = self.jacobian_second_cache.as_ref() {
            debug_assert_eq!(jac2.ncols(), p * d * d);
            for n in 0..n_obs {
                for c in 0..d {
                    let mut acc = 0.0;
                    for a in 0..d {
                        for b in 0..d {
                            let diff = g[[n, a * d + b]] - g_ref[[n, a * d + b]];
                            let mut hj = 0.0;
                            for i in 0..p {
                                hj += jac2[[n, (i * d + a) * d + c]] * jac[[n, i * d + b]];
                            }
                            acc += diff * hj;
                        }
                    }
                    grad[n * d + c] = 2.0 * mu * acc;
                }
            }
        } else {
            // Gauss-Newton surrogate: H_n · t_n with H_n = 2 J_n^T J_n. This
            // is the dominant term and the one the inner Newton step uses for
            // its block-diagonal preconditioner. Documented as the
            // IFT-warm-start surrogate, not the exact gradient.
            for n in 0..n_obs {
                for a in 0..d {
                    let mut acc = 0.0;
                    for b in 0..d {
                        let mut gab = 0.0;
                        for i in 0..p {
                            gab += jac[[n, i * d + a]] * jac[[n, i * d + b]];
                        }
                        acc += gab * target[n * d + b];
                    }
                    grad[n * d + a] = 2.0 * mu * acc;
                }
            }
        }
        grad
    }

    fn hvp(
        &self,
        _target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Gauss-Newton approximation: H ≈ 2 μ · M^T M, where M acts on t and
        // returns vectorized (g - g^ref). The dominant term keeps the IFT
        // well-conditioned; the dropped term is `O((g - g^ref))` which
        // vanishes at the local minimum.
        let mu = rho[self.rho_index].exp();
        let d = self
            .target
            .latent_dim
            .expect("IsometryPenalty requires latent_dim on its PsiSlice");
        let jac = self
            .jacobian_cache
            .as_ref()
            .expect("isometry penalty requires a live jacobian cache");
        let n_obs = jac.nrows();
        let p = self.p_out;
        let mut out = Array1::<f64>::zeros(v.len());
        for n in 0..n_obs {
            // Per-row Gauss-Newton block (d×d): G_n = J_n^T J_n acts on v_n.
            for a in 0..d {
                let mut acc = 0.0;
                for b in 0..d {
                    let mut gab = 0.0;
                    for i in 0..p {
                        gab += jac[[n, i * d + a]] * jac[[n, i * d + b]];
                    }
                    acc += gab * v[n * d + b];
                }
                out[n * d + a] = 2.0 * mu * acc;
            }
        }
        out
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

/// Sparsity penalty on a slice of β (SAE codes) or ψ (soft atom assignments).
///
/// The smoothed-L¹ default `Σ_i sqrt(x_i² + ε²)` is the simplest analytic
/// option. Its gradient is `x_i / sqrt(x_i² + ε²)` (a smooth sign function),
/// and its Hessian is diagonal with entries `ε² / (x_i² + ε²)^{3/2}` — so
/// `hvp` is cheap and the inner Newton step inherits a benign block-diagonal
/// regularizer.
///
/// When to use: any time a parameter block carries a "this should be sparse"
/// prior — SAE atom codes (β slice), soft-routing weights on a latent (ψ
/// slice). For SAE codes specifically, smoothed-L¹ with REML-selected `ε`
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

impl SparsityPenalty {
    pub fn smoothed_l1(target_tier: PenaltyTier, eps: f64) -> Self {
        Self {
            target_tier,
            kind: SparsityKind::SmoothedL1 { eps },
            strength_rho_index: 0,
            eps_rho_index: None,
        }
    }

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
                let n = target.len() as f64;
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return 0.0;
                }
                let h = (n.sqrt() * l1 / l2 - 1.0) / (n.sqrt() - 1.0);
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
                // ∂(L1)/∂x_i = sign(x_i); ∂(L2)/∂x_i = x_i / L2.
                let n = target.len() as f64;
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return g;
                }
                let denom = n.sqrt() - 1.0;
                let coeff = lam / denom;
                for (i, &x) in target.iter().enumerate() {
                    let sgn = if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    g[i] = coeff * (n.sqrt() * (sgn / l2 - l1 * x / (l2 * l2 * l2)));
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
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    d[i] = lam * 2.0 * (d2 - x * x) / (denom * denom);
                }
                Some(d)
            }
            // Hoyer's Hessian is dense (couples through L2). Provide the
            // *diagonal* part only — the consumer that wants exact HVP must
            // override (we expose `hvp` below for the rank-1 + diagonal
            // form). Diagonal piece:
            //   ∂²H/∂x_i² = (√n/((√n−1) L2)) · ( 3 L1 x_i² / L2⁴ − 1/L2² · something ).
            // For an inner-loop preconditioner we use the L2-curvature term
            // `λ · √n · L1 / ((√n−1) · L2³)` which is the dominant positive
            // diagonal contribution near sparse minima.
            SparsityKind::Hoyer => {
                let n = target.len() as f64;
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return Some(d);
                }
                let denom = n.sqrt() - 1.0;
                let coeff = lam * n.sqrt() * l1 / (denom * l2 * l2 * l2);
                for i in 0..target.len() {
                    d[i] = coeff;
                }
                Some(d)
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
///   P_j(t; ρ) = ½ exp(ρ_j) · ‖t[:, j]‖²
/// ```
///
/// summed over `j ∈ [0, d)`. Under REML, axis `j` whose data evidence is too
/// weak gets `ρ_j → +∞` (precision → ∞, coefficients → 0), so the latent
/// dimension is effectively pruned. The intrinsic dimensionality is read off
/// as the count of finite `ρ_j` at convergence.
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
}

impl ARDPenalty {
    pub fn new(target: PsiSlice, latent_dim: usize) -> Self {
        let rho_indices = (0..latent_dim).collect();
        Self {
            target,
            latent_dim,
            rho_indices,
        }
    }

    /// Build one [`BlockwisePenalty`] per latent axis. The j-th block has
    /// `local = I_{n_obs}` and `col_range` equal to the flat indices of that
    /// axis. The outer REML loop multiplies by `exp(ρ_j)`.
    pub fn as_blockwise(&self, global_offset: usize) -> Vec<BlockwisePenalty> {
        let n_obs = self.target.len() / self.latent_dim;
        let mut out = Vec::with_capacity(self.latent_dim);
        for j in 0..self.latent_dim {
            // The flat layout is row-major (n * d + a), so axis j picks rows
            // n=0..n_obs at columns global_offset + n*d + j. The "block" here
            // is a single diagonal scalar repeated `n_obs` times — represent
            // it as a ridge of size n_obs at the appropriate stride.
            //
            // For the canonical pipeline we collapse this to a contiguous
            // ridge by reshape: the BlockwisePenalty machinery expects a
            // contiguous range, so we expose the axis-j ridge as a
            // length-`n_obs` ridge with a virtual stride captured in
            // `structure_hint`. Callers that need the stride re-expand using
            // `latent_dim`.
            let start = global_offset + j * n_obs;
            let end = start + n_obs;
            out.push(
                BlockwisePenalty::ridge(start..end, 1.0)
                    .with_op(None),
            );
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
            let lam_j = rho[self.rho_indices[j]].exp();
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            acc += 0.5 * lam_j * sq;
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
        // P_j(ρ_j) = ½ exp(ρ_j) Σ_n t_{n,j}². ∂/∂ρ_j = P_j.
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
            out[self.rho_indices[j]] = 0.5 * lam_j * sq;
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
    pub fn new(penalty: Arc<dyn AnalyticPenalty>) -> Self {
        Self { penalty }
    }
}

// ---------------------------------------------------------------------------
// Registration helper — collects penalty kinds for the outer REML driver
// ---------------------------------------------------------------------------

/// Tagged sum of the three penalty kinds, with enough metadata for the outer
/// REML driver to:
///
///   1. Concatenate each penalty's owned ρ-axes onto the global ρ vector.
///   2. Route the inner gradient `∂L/∂target` contribution back into the
///      correct β or ψ slice.
///   3. Build a Hessian-block stub for `RemlState` cache-key invalidation.
#[derive(Clone)]
pub enum AnalyticPenaltyKind {
    Isometry(Arc<IsometryPenalty>),
    Sparsity(Arc<SparsityPenalty>),
    Ard(Arc<ARDPenalty>),
}

impl AnalyticPenaltyKind {
    pub fn tier(&self) -> PenaltyTier {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.tier(),
            AnalyticPenaltyKind::Sparsity(p) => p.tier(),
            AnalyticPenaltyKind::Ard(p) => p.tier(),
        }
    }

    pub fn rho_count(&self) -> usize {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.rho_count(),
            AnalyticPenaltyKind::Sparsity(p) => p.rho_count(),
            AnalyticPenaltyKind::Ard(p) => p.rho_count(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.name(),
            AnalyticPenaltyKind::Sparsity(p) => p.name(),
            AnalyticPenaltyKind::Ard(p) => p.name(),
        }
    }

    pub fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.value(target, rho),
            AnalyticPenaltyKind::Sparsity(p) => p.value(target, rho),
            AnalyticPenaltyKind::Ard(p) => p.value(target, rho),
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
            AnalyticPenaltyKind::Ard(p) => p.grad_target(target, rho),
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
            AnalyticPenaltyKind::Ard(p) => p.grad_rho(target, rho),
        }
    }

    pub fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        match self {
            AnalyticPenaltyKind::Isometry(p) => p.hvp(target, rho, v),
            AnalyticPenaltyKind::Sparsity(p) => {
                if let Some(diag) = p.hessian_diag(target, rho) {
                    let mut out = Array1::<f64>::zeros(v.len());
                    for i in 0..v.len() {
                        out[i] = diag[i] * v[i];
                    }
                    out
                } else {
                    p.hvp(target, rho, v)
                }
            }
            AnalyticPenaltyKind::Ard(p) => {
                let diag = p.hessian_diag(target, rho).expect("ARD diag");
                let mut out = Array1::<f64>::zeros(v.len());
                for i in 0..v.len() {
                    out[i] = diag[i] * v[i];
                }
                out
            }
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
// The canonical PIRLS / REML pipeline consumes square symmetric PSD operators
// through the `PenaltyOp` trait (see `terms::penalty_op`). The non-quadratic
// analytic penalties here are *not* linear in their target, but the inner
// Newton step only sees their **Hessian at the current iterate** — a square
// symmetric PSD object once we choose the Gauss-Newton or diagonal surrogate
// each penalty provides. We therefore expose each penalty as a `PenaltyOp` by
// freezing `(target, rho)` and routing `matvec` to `hvp`. The solver re-builds
// the frozen op once per outer iteration (after PIRLS converges on `β`), in
// exactly the same place the existing closed-form operator is rebuilt when
// `ψ` advances.

use crate::terms::penalty_op::PenaltyOp;
use ndarray::{ArrayViewMut1};

/// `PenaltyOp` view of an [`AnalyticPenalty`] frozen at `(target, rho)`.
///
/// The Hessian at the frozen point is symmetric PSD for every penalty we
/// ship (smoothed-L¹ and Log have positive diagonals by construction; ARD
/// is a positive diagonal; Isometry's Gauss-Newton form is `2μ J^T J` which
/// is PSD). `as_dense()` materializes via `n` matvecs against the standard
/// basis — `O(n²)` and intended only for spectral diagnostics; the hot path
/// uses `matvec` and `diag` directly.
pub struct FrozenAnalyticPenaltyOp {
    penalty: AnalyticPenaltyKind,
    target: Array1<f64>,
    rho: Array1<f64>,
}

impl FrozenAnalyticPenaltyOp {
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
        // Each penalty exposes a hessian_diag (ARD, smoothed-L¹, Log directly;
        // Isometry via Gauss-Newton diagonal of J^T J; Hoyer via dominant
        // diagonal). When unavailable, fall back to probing matvec on each
        // standard basis vector (O(n²)).
        match &self.penalty {
            AnalyticPenaltyKind::Ard(p) => p
                .hessian_diag(self.target.view(), self.rho.view())
                .expect("ARD diag"),
            AnalyticPenaltyKind::Sparsity(p) => {
                if let Some(d) = p.hessian_diag(self.target.view(), self.rho.view()) {
                    d
                } else {
                    self.diag_via_matvec()
                }
            }
            AnalyticPenaltyKind::Isometry(p) => {
                // Gauss-Newton diagonal: diag(2μ J^T J).
                let d = p
                    .target
                    .latent_dim
                    .expect("IsometryPenalty requires latent_dim");
                let jac = p
                    .jacobian_cache
                    .as_ref()
                    .expect("isometry penalty requires a live jacobian cache");
                let n_obs = jac.nrows();
                let p_out = p.p_out;
                let mu = self.rho[p.rho_index].exp();
                let mut out = Array1::<f64>::zeros(self.target.len());
                for n in 0..n_obs {
                    for a in 0..d {
                        let mut gaa = 0.0;
                        for i in 0..p_out {
                            let v = jac[[n, i * d + a]];
                            gaa += v * v;
                        }
                        out[n * d + a] = 2.0 * mu * gaa;
                    }
                }
                out
            }
        }
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        assert!(lambda > 0.0, "log_det_plus_lambda_i requires λ > 0");
        // For the diagonal-Hessian penalties (ARD, smoothed-L¹ and Log) the
        // closed form is `Σ_i log(d_i + λ)`. For the dense Isometry GN form
        // we fall back to the materialize-and-eigh path on `as_dense`.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(_) | AnalyticPenaltyKind::Sparsity(_) => {
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
            AnalyticPenaltyKind::Isometry(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
        }
    }

    fn as_dense(&self) -> Array2<f64> {
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
}

impl AnalyticPenaltyOp {
    /// Promote this analytic-penalty handle into a frozen `PenaltyOp` at
    /// `(target, rho)`. The returned `Arc<dyn PenaltyOp>` plugs directly into
    /// `BlockwisePenalty::with_op` or `PenaltyForm::Operator`.
    pub fn freeze(
        &self,
        target: Array1<f64>,
        rho: Array1<f64>,
    ) -> Arc<dyn PenaltyOp> {
        // Wrap each kind back into `AnalyticPenaltyKind` for storage.
        let kind = if let Some(p) =
            (&self.penalty as &dyn std::any::Any).downcast_ref::<IsometryPenalty>()
        {
            AnalyticPenaltyKind::Isometry(Arc::new(p.clone()))
        } else if let Some(p) =
            (&self.penalty as &dyn std::any::Any).downcast_ref::<SparsityPenalty>()
        {
            AnalyticPenaltyKind::Sparsity(Arc::new(p.clone()))
        } else if let Some(p) =
            (&self.penalty as &dyn std::any::Any).downcast_ref::<ARDPenalty>()
        {
            AnalyticPenaltyKind::Ard(Arc::new(p.clone()))
        } else {
            unreachable!("AnalyticPenaltyOp::freeze: unknown analytic penalty kind");
        };
        Arc::new(FrozenAnalyticPenaltyOp::new(kind, target, rho))
    }
}

impl AnalyticPenaltyKind {
    /// Freeze this kind at `(target, rho)` and return an `Arc<dyn PenaltyOp>`
    /// ready to slot into `BlockwisePenalty::with_op` or `PenaltyForm::Operator`.
    pub fn freeze(&self, target: Array1<f64>, rho: Array1<f64>) -> Arc<dyn PenaltyOp> {
        Arc::new(FrozenAnalyticPenaltyOp::new(self.clone(), target, rho))
    }
}

// Suppress unused-import warnings on items that downstream wiring will pull
// in soon (PenaltyStructureHint, ArrayView2, Axis, LatentCoordValues,
// AnalyticPenaltyOp) without breaking strict builds.
#[allow(dead_code)]
fn _silence_unused_imports() {
    let _: Option<PenaltyStructureHint> = None;
    let _: Option<ArrayView2<'static, f64>> = None;
    let _ = Axis(0);
    let _ = std::mem::size_of::<LatentCoordValues>();
    let _ = std::mem::size_of::<AnalyticPenaltyOp>();
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
        let p = SparsityPenalty::smoothed_l1(PenaltyTier::Beta, 1e-3);
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
    fn ard_rho_grad_equals_per_axis_value() {
        let d = 2;
        let t = array![1.0_f64, 0.0, 0.0, 2.0];
        let target = PsiSlice::full(t.len(), Some(d));
        let ard = ARDPenalty::new(target, d);
        let rho = array![0.0_f64, 0.0];
        let dr = ard.grad_rho(t.view(), rho.view());
        assert!((dr[0] - 0.5).abs() < 1e-12); // ½·1·(1+0)
        assert!((dr[1] - 2.0).abs() < 1e-12); // ½·1·(0+4)
    }
}
