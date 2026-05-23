//! Bundle-adjustment Schur solver for joint `(t, β)` inner systems.
//!
//! BIBLIOGRAPHY
//!
//! * Agarwal, Snavely, Seitz, Szeliski, "Bundle Adjustment in the Large",
//!   ECCV 2010 / University of Washington technical report: inexact-step
//!   Levenberg-Marquardt, reduced camera system, and PCG on the Schur system.
//! * Demmel, Gao, Gu, et al., "Square Root Bundle Adjustment for Large-Scale
//!   Reconstruction", CVPR 2021 / TheCVF: form Schur contributions through
//!   square-root per-point factors for improved numerical stability.
//! * Nocedal and Wright, "Numerical Optimization", 2nd ed.; Steihaug 1983:
//!   truncated conjugate gradients for trust-region subproblems, used by
//!   Ceres-style trust-region solvers.
//! * Ceres Solver documentation, "Solving Non-linear Least Squares":
//!   reduced camera systems, Schur preconditioners, and trust-region LM
//!   practice for BA.
//! * Liu et al., "MegBA: A GPU-Based Distributed Library for Large-Scale
//!   Bundle Adjustment", ECCV 2020: batched point-block solves and Schur
//!   reductions as GPU kernels.
//!
//! See `proposals/latent_coord.md` §4 (the plumbing change) and
//! `proposals/composition_engine.md` §7 (audit-revised complexity claim:
//! "cost is arrow-shaped, but the REML log|H| gradient carries a shared
//! Schur⁻¹ factor handled as one-time-per-outer-iteration setup plus N
//! rank-≤d per-row traces"). The math-audit revisions in those proposals
//! are the source of the explicit precondition story below.
//!
//! ## What this module does
//!
//! When a [`crate::terms::latent_coord::LatentCoordValues`] block is
//! registered with the design, each inner Gauss–Newton iteration must
//! solve the same normal equations that bundle adjustment solves:
//! per-3D-point blocks are our per-row latent coordinates `t_i`, and
//! per-camera shared parameters are our decoder coefficients `β`.
//!
//! ```text
//! [ H_tt   H_tβ ] [ Δt ]     [ -g_t ]
//! [ H_βt   H_ββ ] [ Δβ ]  =  [ -g_β ]
//! ```
//!
//! where:
//!
//! * `H_tt` is **block-diagonal in rows** — `N` independent `d × d`
//!   blocks `H_tt^(i)` (one per observation). This is the load-bearing
//!   structure exploited here.
//! * `H_tβ`, `H_βt = H_tβ^T` are row-local in `t` and dense in `β` —
//!   each row `i` contributes a `d × K` slab.
//! * `H_ββ` is the standard `K × K` penalized Hessian already handled by
//!   the existing PIRLS β-only path.
//!
//! BA's reduced camera system (RCS) eliminates `Δt` first and produces the
//! reduced `K × K` shared system
//!
//! ```text
//! S · Δβ = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i),   S = H_ββ - Σ_i H_βt^(i) (H_tt^(i))⁻¹ H_tβ^(i)
//! ```
//!
//! followed by row-local back-substitution
//!
//! ```text
//! Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
//! ```
//!
//! Per inner iteration: `O(N d³)` for the per-row Cholesky factors, the
//! Schur subtraction, and the back-substitution, plus one standard
//! `K × K` solve for `Δβ`. Memory is `O(N d²)` for the per-row factors
//! plus the existing `O(K²)` β workspace.
//!
//! ## Scope — what is and is not in this file
//!
//! **In scope.** The arrow-Schur elimination of `H_tt` *for the inner
//! Gauss–Newton step*. The block-diagonality of `H_tt` is the property
//! that makes per-row elimination cheap; this is correct as long as
//! penalty contributions to `H_tt` are themselves row-block-diagonal
//! (true for [`crate::terms::analytic_penalties::ARDPenalty`] — diagonal —
//! and for [`crate::terms::analytic_penalties::IsometryPenalty`] in its
//! metric-residual Gauss–Newton form — per-row `d × d` blocks through
//! `∂(J_n^T W_n J_n)/∂t_n`).
//!
//! **Out of scope (do not confuse).** The REML *outer-loop* gradient of
//! `log|H|` with respect to `t` carries a shared `Schur⁻¹` factor; only
//! row `i` of `Φ` moves with `t_i`, but `Schur⁻¹` itself is dense in all
//! `t`. That requires one dense `Schur⁻¹` formation per outer iteration
//! plus N rank-≤d per-row traces. It is **not** handled here — that's a
//! separate plumbing change owned by the REML driver. The two cost
//! analyses must not be conflated: the *inner* step is genuinely
//! O(N d³ + K³); the *outer* gradient is O(K³ + N · K d) once `Schur⁻¹`
//! is in scope.
//!
//! Future maintainers: this is BA. Solver improvements should first look
//! at Ceres/g2o/MegBA/Square-Root BA literature, not bespoke algebra. If you
//! find yourself extending `ArrowSchurSystem` with an outer-REML gradient
//! hook, please re-read the audit revisions in `proposals/latent_coord.md`
//! §7 and `proposals/composition_engine.md` §7 first.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use std::sync::Arc;

use crate::terms::analytic_penalties::{AnalyticPenaltyKind, AnalyticPenaltyRegistry, PenaltyTier};
use crate::terms::latent_coord::LatentCoordValues;

const DIRECT_SOLVE_MAX_K: usize = 2_000;
const DEFAULT_PCG_MAX_ITERATIONS: usize = 200;
const DEFAULT_PCG_RELATIVE_TOLERANCE: f64 = 1e-4;
const DEFAULT_TRUST_REGION_RADIUS: f64 = f64::INFINITY;

/// Matrix-free shared-block multiply for large BA/SAE Schur PCG.
///
/// The closure writes `out = H_ββ x` without the LM ridge. This is the hook
/// that lets SAE-manifold scale callers avoid materializing a dense `K × K`
/// shared block before Agarwal-style inexact Schur PCG.
pub type SharedBetaMatvec =
    Arc<dyn for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;
type MetricWeights = [f64];

/// BA Schur solve variant for the reduced shared `β` system.
///
/// * [`ArrowSolverMode::Direct`] is BA's dense reduced-camera-system solve:
///   eliminate the per-point/per-row blocks, form the reduced system, and
///   Cholesky factor it. This is the Ceres/g2o default for modest camera
///   counts and is appropriate here for `K <= 2000`.
/// * [`ArrowSolverMode::SqrtBA`] ports Square-Root BA (Demmel/Gao/Gu et al.,
///   CVPR 2021): Schur terms are formed as `(L_i^-1 H_tβ_i)^T
///   (L_i^-1 H_tβ_i)` from the per-row square-root factor `L_i`, avoiding
///   explicit `H_tt^-1 H_tβ` products. It is the preferred direct path when
///   single-precision assembly is introduced or when row blocks are poorly
///   conditioned.
/// * [`ArrowSolverMode::InexactPCG`] ports "Bundle Adjustment in the Large"
///   (Agarwal et al.): the Schur system is solved inexactly by PCG with a
///   Jacobi Schur preconditioner, avoiding dense `K × K` factorization for
///   SAE-manifold scale shared systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowSolverMode {
    Direct,
    SqrtBA,
    InexactPCG,
}

impl ArrowSolverMode {
    /// BA-size heuristic: dense RCS for modest `K`, inexact Schur PCG for
    /// large shared systems. This follows Agarwal et al.'s direct-vs-iterative
    /// split for large BA, mapped from cameras to decoder coefficients.
    pub fn automatic(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::Direct
        } else {
            Self::InexactPCG
        }
    }

    /// Square-Root BA is the direct-solve stability mode for future f32
    /// callers. Large `K` still routes to inexact PCG because dense Schur
    /// storage dominates precision concerns at that scale.
    pub fn automatic_for_single_precision(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::SqrtBA
        } else {
            Self::InexactPCG
        }
    }
}

/// PCG controls for BA's inexact reduced-camera-system solve.
///
/// The defaults mirror the loose inner tolerances used by inexact-step LM in
/// "Bundle Adjustment in the Large": solve the Schur system only accurately
/// enough for a useful trust-region step, then let the outer LM iteration
/// correct the remaining error.
#[derive(Debug, Clone)]
pub struct ArrowPcgOptions {
    pub max_iterations: usize,
    pub relative_tolerance: f64,
}

impl Default for ArrowPcgOptions {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_PCG_MAX_ITERATIONS,
            relative_tolerance: DEFAULT_PCG_RELATIVE_TOLERANCE,
        }
    }
}

/// Trust-region controls for Steihaug-CG on the reduced BA system.
///
/// This is the Ceres-style guard around LM: `ridge_t`/`ridge_beta` provide
/// Levenberg damping, while the trust radius bounds the reduced shared step
/// in Euclidean β coordinates using Steihaug's truncated-CG stopping rules for
/// boundary hits and negative curvature.
#[derive(Debug, Clone)]
pub struct ArrowTrustRegionOptions {
    pub radius: f64,
    pub steihaug_relative_tolerance: f64,
    pub max_iterations: usize,
}

impl Default for ArrowTrustRegionOptions {
    fn default() -> Self {
        Self {
            radius: DEFAULT_TRUST_REGION_RADIUS,
            steihaug_relative_tolerance: DEFAULT_PCG_RELATIVE_TOLERANCE,
            max_iterations: DEFAULT_PCG_MAX_ITERATIONS,
        }
    }
}

/// Complete BA Schur solve options.
///
/// Use [`ArrowSolveOptions::automatic`] for normal latent-coordinate fits;
/// use [`ArrowSolveOptions::sqrt_ba`] when the assembler has single-precision
/// row blocks or an ill-conditioned gauge; use [`ArrowSolveOptions::inexact_pcg`]
/// for SAE-manifold scale `K`.
#[derive(Debug, Clone)]
pub struct ArrowSolveOptions {
    pub mode: ArrowSolverMode,
    pub pcg: ArrowPcgOptions,
    pub trust_region: ArrowTrustRegionOptions,
    /// Use the Riemannian latent projection before the Schur reduction. The
    /// reduced Steihaug solve itself remains in Euclidean β coordinates.
    pub riemannian_trust_region: bool,
}

impl ArrowSolveOptions {
    /// Select Direct for `K <= 2000` and InexactPCG above, following BA RCS
    /// practice for dense-vs-iterative reduced systems.
    pub fn automatic(k: usize) -> Self {
        Self {
            mode: ArrowSolverMode::automatic(k),
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            riemannian_trust_region: false,
        }
    }

    /// Force dense reduced-camera-system Cholesky, the classic BA direct
    /// solve for small `K`.
    pub fn direct() -> Self {
        Self {
            mode: ArrowSolverMode::Direct,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            riemannian_trust_region: false,
        }
    }

    /// Force Square-Root BA Schur assembly for the direct reduced solve.
    pub fn sqrt_ba() -> Self {
        Self {
            mode: ArrowSolverMode::SqrtBA,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            riemannian_trust_region: false,
        }
    }

    /// Force inexact BA Schur PCG with Jacobi preconditioning.
    pub fn inexact_pcg() -> Self {
        Self {
            mode: ArrowSolverMode::InexactPCG,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            riemannian_trust_region: false,
        }
    }
}

/// CPU/GPU seam for BA point-block work.
///
/// BA systems spend most time in independent point-block factorizations,
/// triangular solves, and Schur block products. MegBA maps exactly these
/// operations to GPU kernels. This trait keeps that boundary explicit so a
/// CUDA/Ceres backend can replace [`CpuBatchedBlockSolver`] without changing
/// `ArrowSchurSystem` algebra.
pub trait BatchedBlockSolver {
    /// Factor every per-row point block `H_tt^(i) + ridge_t I`, as in BA's
    /// point elimination stage.
    fn factor_blocks(
        &self,
        rows: &[ArrowRowBlock],
        ridge_t: f64,
        d: usize,
    ) -> Result<Vec<Array2<f64>>, ArrowSchurError>;

    /// Solve one factored point block against a vector RHS.
    fn solve_block_vector(&self, factor: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64>;

    /// Solve one factored point block against a dense matrix RHS.
    fn solve_block_matrix(&self, factor: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64>;

    /// Apply the Square-Root BA lower-triangular solve `L_i^-1 rhs`.
    fn sqrt_solve_block_matrix(&self, factor: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64>;

    /// Subtract a row-local Schur product from the dense reduced system.
    fn block_gemm_subtract(&self, schur: &mut Array2<f64>, left: &Array2<f64>, right: &Array2<f64>);
}

/// Current CPU implementation of the BA batched block interface.
///
/// It is intentionally plain Rust loops because `d` is tiny. The trait shape,
/// not this implementation, is the load-bearing part for the future MegBA or
/// Ceres backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBatchedBlockSolver;

impl BatchedBlockSolver for CpuBatchedBlockSolver {
    fn factor_blocks(
        &self,
        rows: &[ArrowRowBlock],
        ridge_t: f64,
        d: usize,
    ) -> Result<Vec<Array2<f64>>, ArrowSchurError> {
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut block = row.htt.clone();
            for a in 0..d {
                block[[a, a]] += ridge_t;
            }
            out.push(
                cholesky_lower(&block).map_err(|e| ArrowSchurError::PerRowFactorFailed {
                    row: out.len(),
                    reason: e,
                })?,
            );
        }
        Ok(out)
    }

    fn solve_block_vector(&self, factor: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
        chol_solve_vector(factor, rhs)
    }

    fn solve_block_matrix(&self, factor: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
        chol_solve_matrix(factor, rhs)
    }

    fn sqrt_solve_block_matrix(&self, factor: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
        lower_triangular_solve_matrix(factor, rhs)
    }

    fn block_gemm_subtract(
        &self,
        schur: &mut Array2<f64>,
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) {
        let k = schur.nrows();
        let d = left.nrows();
        debug_assert_eq!(left.ncols(), k);
        debug_assert_eq!(right.ncols(), k);
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0;
                for c in 0..d {
                    acc += left[[c, a]] * right[[c, b]];
                }
                schur[[a, b]] -= acc;
            }
        }
    }
}

/// Per-row block data for the arrow-Schur system.
///
/// `htt` holds the `d × d` Gauss–Newton block for row `i` (including any
/// analytic-penalty contributions on that row); `htbeta` holds the
/// `d × K` cross-block `H_tβ^(i)`; `gt` is the `d`-length latent
/// gradient for row `i`.
#[derive(Debug, Clone)]
pub struct ArrowRowBlock {
    /// `H_tt^(i)`, shape `(d, d)`.
    pub htt: Array2<f64>,
    /// `H_tβ^(i)`, shape `(d, K)`.
    pub htbeta: Array2<f64>,
    /// `g_t^(i)`, shape `(d,)`.
    pub gt: Array1<f64>,
}

impl ArrowRowBlock {
    /// Allocate one BA point-block row: local latent Hessian, point-camera
    /// cross block, and point gradient.
    pub fn new(d: usize, k: usize) -> Self {
        Self {
            htt: Array2::<f64>::zeros((d, d)),
            htbeta: Array2::<f64>::zeros((d, k)),
            gt: Array1::<f64>::zeros(d),
        }
    }
}

/// Bordered (t, β) Newton system with arrow structure.
///
/// The β-block is held as a dense `K × K` Hessian `H_ββ` plus a `K`-length
/// gradient `g_β` for direct BA modes. Large-scale inexact BA callers may
/// additionally install a matrix-free `H_ββ x` operator and diagonal via
/// [`ArrowSchurSystem::set_shared_beta_operator`]; the InexactPCG mode then
/// avoids dense Schur formation/factorization.
/// The t-block is a `Vec<ArrowRowBlock>` of length `N`.
///
/// Construction is the driver's responsibility: the driver
///
///   1. evaluates Φ(t) and the radial jet `∂Φ/∂t` (the latter via
///      [`crate::terms::latent_coord::LatentCoordValues::design_gradient_wrt_t`]);
///   2. forms the working-weighted Gauss–Newton blocks
///      `H_tt^(i) += (g_i β)(g_i β)^T`, `H_tβ^(i) += (g_i β) ⊗ Φ_i`,
///      `H_ββ += Φ^T W Φ + Σ_k λ_k S_k`;
///   3. calls [`ArrowSchurSystem::add_analytic_penalty_contributions`] to
///      fold the registered analytic penalties (`IsometryPenalty`,
///      `ARDPenalty`, `SparsityPenalty`) into the appropriate `H_tt^(i)`
///      diagonal/dense block (Psi tier) and into `H_ββ` (Beta tier);
///   4. calls [`ArrowSchurSystem::solve`] to obtain `(Δt, Δβ)`.
pub struct ArrowSchurSystem {
    /// Per-row latent block (length `N`, each row `d × d` / `d × K` / `d`).
    pub rows: Vec<ArrowRowBlock>,
    /// `H_ββ`, shape `(K, K)` for direct BA modes; empty when constructed
    /// by [`ArrowSchurSystem::new_matrix_free_shared`] for PCG-only use.
    pub hbb: Array2<f64>,
    /// Optional matrix-free `H_ββ x` operator for large BA Schur PCG.
    ///
    /// Direct and Square-Root BA modes still require `hbb`; InexactPCG uses
    /// this operator when present, avoiding dense shared-block storage for
    /// SAE-manifold scale `K`.
    pub hbb_matvec: Option<SharedBetaMatvec>,
    /// Optional diagonal of the matrix-free shared block, used by the
    /// Schur-Jacobi preconditioner in the Agarwal-style PCG path.
    pub hbb_diag: Option<Array1<f64>>,
    /// `g_β`, shape `(K,)`.
    pub gb: Array1<f64>,
    /// Latent dimensionality `d`.
    pub d: usize,
    /// β dimensionality `K`.
    pub k: usize,
}

impl ArrowSchurSystem {
    /// Allocate an empty BA reduced-camera-system instance sized
    /// `(N point/latent rows × d, K shared decoder parameters)`.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        Self {
            rows,
            hbb: Array2::<f64>::zeros((k, k)),
            hbb_matvec: None,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            k,
        }
    }

    /// Allocate an arrow system whose shared `H_ββ` block is supplied only as
    /// a matrix-free operator for large BA InexactPCG.
    ///
    /// Direct and Square-Root BA modes require dense `hbb` and must not be
    /// used with this constructor. The row-local `H_tβ` slabs remain explicit;
    /// a future MegBA backend can replace those slab operations behind
    /// [`BatchedBlockSolver`].
    pub fn new_matrix_free_shared<F>(
        n: usize,
        d: usize,
        k: usize,
        matvec: F,
        diag: Array1<f64>,
    ) -> Self
    where
        F: for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        debug_assert_eq!(diag.len(), k);
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: Some(Arc::new(matvec)),
            hbb_diag: Some(diag),
            gb: Array1::<f64>::zeros(k),
            d,
            k,
        }
    }

    /// Number of BA point/latent rows `N`.
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Install a matrix-free shared-block operator for Agarwal-style
    /// inexact Schur PCG.
    ///
    /// `diag` must be the diagonal of the same `H_ββ` operator and is used
    /// for the Schur-Jacobi preconditioner. This is the BA "large camera
    /// system" path mapped to large decoder coefficient blocks.
    pub fn set_shared_beta_operator<F>(&mut self, matvec: F, diag: Array1<f64>)
    where
        F: for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        debug_assert_eq!(diag.len(), self.k);
        self.hbb_matvec = Some(Arc::new(matvec));
        self.hbb_diag = Some(diag);
    }

    /// Fold analytic-penalty contributions into the appropriate blocks.
    ///
    /// BA source mapping: these are extra prior/regularization normal-equation
    /// terms before point elimination, the same place Ceres/g2o attach robust
    /// priors or gauge-fixing constraints.
    ///
    /// **Composition path.** Each registered [`AnalyticPenaltyKind`] is
    /// queried for `grad_target` (added to `g_t` or `g_β`) and then for
    /// `hessian_diag` first. Diagonal penalties (ARD and the shipped
    /// sparsity kernels) are injected directly. Dense penalties fall back to
    /// `hvp` probes against the canonical basis vectors restricted to the
    /// per-row `d`-block (when `tier == Psi`) or against `β` (when `tier ==
    /// Beta`). The extension-coordinate contributions land on the per-row `H_tt^(i)`
    /// **only** — we rely on the analytic-penalty contract that Psi-tier
    /// penalties are row-block-diagonal in their Hessian.
    ///
    /// `target_t` is the full flat latent-coordinate vector (row-major, `N·d` entries)
    /// at the current iterate; `target_beta` is the current `β`. `rho`
    /// is the global ρ vector restricted to each penalty's local slice
    /// by [`AnalyticPenaltyRegistry::rho_layout`].
    pub fn add_analytic_penalty_contributions(
        &mut self,
        registry: &AnalyticPenaltyRegistry,
        target_t: ArrayView1<'_, f64>,
        target_beta: ArrayView1<'_, f64>,
        rho_global: ArrayView1<'_, f64>,
    ) {
        let layout = registry.rho_layout();
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(ndarray::s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    self.add_ext_coord_penalty(penalty, target_t, rho_local);
                }
                PenaltyTier::Beta => {
                    self.add_beta_penalty(penalty, target_beta, rho_local);
                }
                PenaltyTier::Rho => {
                    // Rho-tier hyperpriors do not contribute to the inner
                    // (t, β) Newton step; they enter only at the REML
                    // outer level.
                }
            }
        }
    }

    /// Convert row-local Euclidean latent blocks to Riemannian tangent blocks.
    ///
    /// This is the only arrow-Schur algebra change needed for manifold
    /// latents: `g_t`, `H_tt`, and each `H_tβ` column are projected to
    /// `T_{t_i}M`, while the shared β block and Schur structure remain
    /// untouched. Embedded constrained manifolds carry a pinned normal block
    /// so the existing ambient Cholesky factorization still works; all RHS
    /// terms live in the tangent space, so the solved update retracts cleanly.
    pub fn apply_riemannian_latent_geometry(&mut self, latent: &LatentCoordValues) {
        let manifold = latent.manifold();
        if manifold.is_euclidean() {
            return;
        }
        debug_assert_eq!(latent.n_obs(), self.rows.len());
        debug_assert_eq!(latent.latent_dim(), self.d);
        for (i, row) in self.rows.iter_mut().enumerate() {
            let t_i = ArrayView1::from(latent.row(i));
            let gt_e = row.gt.clone();
            let htt_e = row.htt.clone();
            let htbeta_e = row.htbeta.clone();
            row.gt = manifold.project_to_tangent(t_i.clone(), gt_e.view());
            row.htt = manifold.riemannian_hessian_matrix(t_i.clone(), gt_e.view(), htt_e.view());
            row.htbeta = manifold.project_matrix_columns_to_tangent(t_i, htbeta_e.view());
        }
    }

    fn add_ext_coord_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_t: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        debug_assert_eq!(target_t.len(), n * d);
        // Gradient: per-row `d`-slice added to `g_t^(i)`.
        let grad = penalty.grad_target(target_t, rho_local);
        for i in 0..n {
            for a in 0..d {
                self.rows[i].gt[a] += grad[i * d + a];
            }
        }
        // Hessian: inject diagonal penalties directly. This avoids O(d)
        // full-length HVP probes for ARD/sparsity on the Psi tier.
        if let Some(diag) = penalty.hessian_diag(target_t, rho_local) {
            debug_assert_eq!(diag.len(), n * d);
            for i in 0..n {
                for a in 0..d {
                    self.rows[i].htt[[a, a]] += diag[i * d + a];
                }
            }
            return;
        }

        // Dense Hessian: probe via HVP against each unit-`d`-vector for each row.
        // For row-block penalties (such as Isometry's metric-residual GN
        // operator) this yields the per-row `d × d` block. For penalties that
        // would couple rows the off-row entries are silently dropped — a
        // design choice consistent with the arrow-shape precondition. The
        // analytic-penalty contract documents that extension-coordinate penalties
        // *must* be row-block-diagonal in their Hessian.
        let mut probe = Array1::<f64>::zeros(n * d);
        for a in 0..d {
            // One probe per latent axis: set the `a`-th column of each
            // row simultaneously to 1, HVP once, extract the column-`a`
            // entries of `H_tt^(i)` for every row.
            probe.fill(0.0);
            for i in 0..n {
                probe[i * d + a] = 1.0;
            }
            let hv = penalty.hvp(target_t, rho_local, probe.view());
            for i in 0..n {
                for b in 0..d {
                    self.rows[i].htt[[b, a]] += hv[i * d + b];
                }
            }
        }
    }

    fn add_beta_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let k = self.k;
        debug_assert_eq!(target_beta.len(), k);
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            self.gb[j] += grad[j];
        }
        // Hessian: inject diagonal penalties directly. K may be large
        // (~100K for production SAE), so this is the hot path for β-tier
        // sparsity/ARD-style penalties.
        if let Some(diag) = penalty.hessian_diag(target_beta, rho_local) {
            debug_assert_eq!(diag.len(), k);
            for j in 0..k {
                if self.hbb.dim() == (k, k) {
                    self.hbb[[j, j]] += diag[j];
                }
                if let Some(hbb_diag) = self.hbb_diag.as_mut() {
                    hbb_diag[j] += diag[j];
                }
            }
            return;
        }

        // Dense Hessian: probe with unit β-vectors.
        if self.hbb.dim() != (k, k) {
            return;
        }
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                self.hbb[[i, j]] += hv[i];
            }
        }
    }

    /// Schur-eliminate the per-row latent block and solve for `(Δt, Δβ)`.
    ///
    /// This uses [`ArrowSolveOptions::automatic`]: BA dense RCS for
    /// `K <= 2000`, and Agarwal-style inexact Schur PCG above that size.
    /// Call [`ArrowSchurSystem::solve_with_options`] to force Square-Root BA
    /// or a specific inexact solve policy.
    ///
    /// Returns `(delta_t, delta_beta)` with `delta_t` flat row-major of
    /// length `N · d` and `delta_beta` of length `K`. The sign convention
    /// matches `solve_newton_direction_dense`: the returned increments
    /// satisfy the bordered system with RHS `[-g_t; -g_β]`, i.e. they are
    /// the *negated* solutions of the standard Newton-direction
    /// formulation.
    ///
    /// `ridge_t` and `ridge_beta` are nonnegative diagonal regularizers
    /// added to the latent and β blocks respectively before factorization
    /// — used by the LM damping outer wrapper to recover from near-singular
    /// inner steps. Pass `0.0` for both to obtain the unregularized
    /// Newton direction.
    pub fn solve(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let options = ArrowSolveOptions::automatic(self.k);
        solve_arrow_newton_step_core(self, ridge_t, ridge_beta, &options)
    }

    /// Solve with an explicit BA Schur mode.
    ///
    /// [`ArrowSolverMode::Direct`] is the classic dense reduced-camera-system
    /// Cholesky path; [`ArrowSolverMode::SqrtBA`] forms the same dense system
    /// through Square-Root BA factors; [`ArrowSolverMode::InexactPCG`] runs
    /// inexact-step LM on the reduced system with Jacobi-preconditioned
    /// Steihaug-CG.
    pub fn solve_with_options(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        solve_arrow_newton_step_core(self, ridge_t, ridge_beta, options)
    }
}

/// Per-row + Schur Cholesky factor cache produced by
/// [`solve_arrow_newton_step`]. Consumed downstream by the IFT warm-start
/// predictor in `crate::solver::persistent_warm_start`: when the outer
/// loop perturbs `(β, ρ)` by a small amount, the new Newton step can be
/// predicted by re-using these factors against a refreshed RHS, saving
/// the dominant `O(N d³ + K³)` factorization cost.
#[derive(Debug, Clone)]
pub struct ArrowFactorCache {
    /// Per-row lower-triangular Cholesky factors of `H_tt^(i) + ridge_t·I`.
    ///
    /// These are the *damped* factors used inside the Newton solve. The IFT
    /// predictor must NOT use them — see [`Self::htt_factors_undamped`].
    pub htt_factors: Vec<Array2<f64>>,
    /// Per-row lower-triangular Cholesky factors of the UNDAMPED
    /// `H_tt^(i)` (no `ridge_t` added).
    ///
    /// The IFT predictor formula
    /// `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) Δβ + δg_t^(i))` is derived from
    /// `∂g_t/∂t = H_tt` at the stationary point, with no LM damping term.
    /// Reusing the damped factors would bias the predicted shift toward zero
    /// in proportion to `ridge_t`. We pay one extra `O(N d³)` Cholesky per
    /// Newton solve — the same complexity class as the Newton solve itself —
    /// to make the IFT exact.
    pub htt_factors_undamped: Vec<Array2<f64>>,
    /// Lower-triangular Cholesky factor of the Schur complement when the
    /// selected BA mode formed/factored dense RCS. `None` for
    /// [`ArrowSolverMode::InexactPCG`], where Agarwal-style inexact LM avoids
    /// the dense `K × K` factor.
    pub schur_factor: Option<Array2<f64>>,
    /// BA mode used to create this cache.
    pub solver_mode: ArrowSolverMode,
    /// Ridge values used to build the cached factors (recorded so the
    /// warm-start predictor knows whether the cache is still valid for a
    /// requested ridge level).
    pub ridge_t: f64,
    pub ridge_beta: f64,
    /// Per-row cross-blocks `H_tβ^(i)` carried so the IFT warm-start
    /// predictor (see `crate::solver::persistent_warm_start::ift_warm_start_latent`)
    /// can rebuild the `β`-coupled RHS without revisiting the assembly
    /// path. Length `N`, each entry shape `(d, K)`.
    pub htbeta: Vec<Array2<f64>>,
    /// Latent dimensionality `d`.
    pub d: usize,
    /// β dimensionality `K`.
    pub k: usize,
}

impl ArrowFactorCache {
    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) · Δβ)` per row, returning
    /// the flat row-major `Δt` of length `N · d`.
    ///
    /// IFT first-order predictor for the latent field under a
    /// shape-coefficient perturbation `Δβ`. See
    /// `proposals/latent_coord.md` §2.2. BA analogue: back-substitution after
    /// reduced-camera-system solve.
    pub fn predict_delta_t_from_delta_beta(&self, delta_beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.htt_factors_undamped.len();
        let d = self.d;
        let k = self.k;
        debug_assert_eq!(delta_beta.len(), k);
        let mut out = Array1::<f64>::zeros(n * d);
        let mut rhs = Array1::<f64>::zeros(d);
        for i in 0..n {
            for c in 0..d {
                let mut acc = 0.0_f64;
                for a in 0..k {
                    acc += self.htbeta[i][[c, a]] * delta_beta[a];
                }
                rhs[c] = acc;
            }
            // Use UNDAMPED factor: IFT inverts H_tt, not H_tt + ridge_t·I.
            let v = chol_solve_vector(&self.htt_factors_undamped[i], &rhs);
            for c in 0..d {
                out[i * d + c] = -v[c];
            }
        }
        out
    }

    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · δg_t^(i)` per row.
    ///
    /// IFT first-order predictor for the latent field under a
    /// per-row gradient perturbation (typically `∂g_t/∂ρ · Δρ`
    /// resolved externally by the driver). BA analogue: reuse point-block
    /// factors for local point updates after shared parameters move.
    pub fn predict_delta_t_from_delta_gt(&self, delta_gt: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.htt_factors_undamped.len();
        let d = self.d;
        debug_assert_eq!(delta_gt.len(), n * d);
        let mut out = Array1::<f64>::zeros(n * d);
        let mut rhs = Array1::<f64>::zeros(d);
        for i in 0..n {
            for c in 0..d {
                rhs[c] = delta_gt[i * d + c];
            }
            // Use UNDAMPED factor: IFT inverts H_tt, not H_tt + ridge_t·I.
            let v = chol_solve_vector(&self.htt_factors_undamped[i], &rhs);
            for c in 0..d {
                out[i * d + c] = -v[c];
            }
        }
        out
    }
}

/// Schur-eliminate the per-row latent block and solve for `(Δt, Δβ)`,
/// returning the factor cache alongside the increments.
///
/// This cached entry point is for IFT warm-start consumers. Call
/// [`ArrowSchurSystem::solve`] or [`ArrowSchurSystem::solve_with_options`]
/// when the cache is not needed.
///
/// `ridge_t` and `ridge_beta` are nonnegative diagonal regularizers added
/// to the latent and β blocks respectively before factorization — used by
/// the LM damping outer wrapper to recover from near-singular inner steps.
/// Pass `0.0` for both to obtain the unregularized Newton direction. The
/// default mode selection follows Agarwal et al.'s dense-vs-inexact BA split.
pub fn solve_arrow_newton_step(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<(Array1<f64>, Array1<f64>, ArrowFactorCache), ArrowSchurError> {
    let options = ArrowSolveOptions::automatic(sys.k);
    solve_arrow_newton_step_with_options(sys, ridge_t, ridge_beta, &options)
}

/// Schur-eliminate the per-row latent block and solve with an explicit BA
/// mode, returning the factor cache alongside the increments.
///
/// This is the BA-grade entry point. Direct and Square-Root BA form the dense
/// reduced camera/shared system; InexactPCG applies the same Schur operator by
/// matvec and uses Jacobi-preconditioned Steihaug-CG, following Agarwal et al.
pub fn solve_arrow_newton_step_with_options(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, ArrowFactorCache), ArrowSchurError> {
    let step = solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)?;
    let backend = CpuBatchedBlockSolver;

    // Snapshot per-row cross-blocks so the IFT predictor can apply
    // the β-coupled sensitivity without re-running the assembler.
    let htbeta: Vec<Array2<f64>> = sys.rows.iter().map(|r| r.htbeta.clone()).collect();
    // Factor the UNDAMPED per-row blocks for the IFT predictor. When
    // ridge_t was zero the damped and undamped factors coincide and we
    // can reuse htt_factors directly; otherwise pay a second per-row
    // Cholesky (O(N d³), same complexity class as the Newton solve).
    let htt_factors_undamped: Vec<Array2<f64>> = if ridge_t == 0.0 {
        step.htt_factors.clone()
    } else {
        backend.factor_blocks(&sys.rows, 0.0, sys.d)?
    };
    let cache = ArrowFactorCache {
        htt_factors: step.htt_factors,
        htt_factors_undamped,
        schur_factor: step.schur_factor,
        solver_mode: options.mode,
        ridge_t,
        ridge_beta,
        htbeta,
        d: sys.d,
        k: sys.k,
    };
    Ok((step.delta_t, step.delta_beta, cache))
}

/// Schur-eliminate the per-row latent block and solve with explicit options,
/// returning only `(Δt, Δβ)`.
///
/// Use this entry point when the IFT factor cache is not consumed.
pub fn solve_arrow_newton_step_core(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)
        .map(|step| (step.delta_t, step.delta_beta))
}

struct ArrowNewtonStepArtifacts {
    delta_t: Array1<f64>,
    delta_beta: Array1<f64>,
    htt_factors: Vec<Array2<f64>>,
    schur_factor: Option<Array2<f64>>,
}

fn solve_arrow_newton_step_artifacts(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let backend = CpuBatchedBlockSolver;

    // 1. BA point elimination: per-row Cholesky factors of
    // (H_tt^(i) + ridge_t · I).
    let htt_factors = backend.factor_blocks(&sys.rows, ridge_t, d)?;

    // 2. Reduced RHS r_β = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i).
    let rhs_beta = reduced_rhs_beta(sys, &htt_factors, &backend);
    // The Schur solve is over the reduced β vector. Latent manifold metric
    // weights live on each d-dimensional t_i block, so the induced metric for
    // this β-only Steihaug problem is Euclidean.
    let trust_metric_weights = None;

    // 3. Solve reduced shared system using the selected BA mode.
    let (delta_beta, schur_factor) = match options.mode {
        ArrowSolverMode::Direct => {
            let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, &backend)?;
            solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?
        }
        ArrowSolverMode::SqrtBA => {
            let schur = build_dense_schur_sqrt_ba(sys, &htt_factors, ridge_beta, &backend)?;
            solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?
        }
        ArrowSolverMode::InexactPCG => {
            let preconditioner =
                JacobiPreconditioner::from_arrow_schur(sys, &htt_factors, ridge_beta, &backend);
            let delta = steihaug_pcg_reduced_system(
                sys,
                &htt_factors,
                ridge_beta,
                &rhs_beta,
                &preconditioner,
                &options.pcg,
                &options.trust_region,
                &backend,
                trust_metric_weights,
            )?;
            (delta, None)
        }
    };

    // 4. Back-substitute Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
    let mut delta_t = Array1::<f64>::zeros(n * d);
    for i in 0..n {
        let mut tmp = sys.rows[i].gt.clone();
        for c in 0..d {
            let mut acc = 0.0;
            for a in 0..k {
                acc += sys.rows[i].htbeta[[c, a]] * delta_beta[a];
            }
            tmp[c] += acc;
        }
        let dt_i = backend.solve_block_vector(&htt_factors[i], &tmp);
        for c in 0..d {
            delta_t[i * d + c] = -dt_i[c];
        }
    }

    Ok(ArrowNewtonStepArtifacts {
        delta_t,
        delta_beta,
        htt_factors,
        schur_factor,
    })
}

// ===========================================================================
// Riemannian retraction hook (additive — opt-in)
// ===========================================================================
//
// The existing Newton step above returns `delta_t` as a flat Euclidean
// increment to be added to the current latent field. When the caller has
// equipped each per-row latent coordinate `t_i` with a non-trivial manifold
// (e.g. S¹ for periodic, S² for sphere) the increment must be applied via a
// retraction rather than free addition. The helper below performs that
// post-processing in-place and is intentionally orthogonal to the existing
// solver: when no manifold is supplied — or every entry is
// `ManifoldKind::Euclidean(d)` — the result is bit-equivalent to
// `new_point[i] = point[i] + delta_t[i]`, matching the pre-Riemannian path.
//
// Callers integrating the Riemannian per-point step do:
//
//   let (delta_t, delta_beta, cache) = solve_arrow_newton_step_with_options(...);
//   let new_points = apply_per_row_retraction(&points_flat, &delta_t,
//                                              row_ambient_dim, manifolds);
//
// where `manifolds.get(i)` is the optional per-row `ManifoldKind`.

/// Apply a per-row retraction to a Euclidean tangent `delta_t`.
///
/// `point_flat` has length `N · d_ambient` (row-major). `delta_t` has the
/// same length. `manifolds.get(i)` supplies an optional per-row manifold; an
/// entry of `None` or a `ManifoldKind::Euclidean(_)` is treated as flat space
/// and produces `point + delta` bit-equivalently.
///
/// Returns the new flat point array; does not consume the cache or touch the
/// existing Newton arithmetic.
pub fn apply_per_row_retraction(
    point_flat: &Array1<f64>,
    delta_t: &Array1<f64>,
    row_ambient_dim: usize,
    manifolds: &[Option<crate::solver::riemannian::ManifoldKind>],
) -> Array1<f64> {
    let n = manifolds.len();
    debug_assert_eq!(point_flat.len(), n * row_ambient_dim);
    debug_assert_eq!(delta_t.len(), n * row_ambient_dim);
    let mut out = Array1::<f64>::zeros(n * row_ambient_dim);
    for i in 0..n {
        let start = i * row_ambient_dim;
        let end = start + row_ambient_dim;
        let p = point_flat.slice(ndarray::s![start..end]);
        let dx = delta_t.slice(ndarray::s![start..end]);
        match manifolds[i].as_ref() {
            None => {
                for c in 0..row_ambient_dim {
                    out[start + c] = p[c] + dx[c];
                }
            }
            Some(kind) if kind.is_euclidean() => {
                for c in 0..row_ambient_dim {
                    out[start + c] = p[c] + dx[c];
                }
            }
            Some(kind) => {
                let m = kind.build();
                let mut new_pt = Array1::<f64>::zeros(row_ambient_dim);
                crate::solver::riemannian::retract_euclidean_delta(
                    m.as_ref(),
                    p,
                    dx,
                    new_pt.view_mut(),
                );
                for c in 0..row_ambient_dim {
                    out[start + c] = new_pt[c];
                }
            }
        }
    }
    out
}

fn reduced_rhs_beta<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    backend: &B,
) -> Array1<f64> {
    let k = sys.k;
    let mut rhs_beta = Array1::<f64>::zeros(k);
    for (i, row) in sys.rows.iter().enumerate() {
        let v = backend.solve_block_vector(&htt_factors[i], &row.gt);
        for a in 0..k {
            let mut acc = 0.0;
            for c in 0..sys.d {
                acc += row.htbeta[[c, a]] * v[c];
            }
            rhs_beta[a] += acc;
        }
    }
    for j in 0..k {
        rhs_beta[j] -= sys.gb[j];
    }
    rhs_beta
}

fn build_dense_schur_direct<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    backend: &B,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    if sys.hbb.dim() != (k, k) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Direct BA requires a dense K×K shared H_ββ block".to_string(),
        });
    }
    let mut schur = sys.hbb.clone();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    for (i, row) in sys.rows.iter().enumerate() {
        let solved = backend.solve_block_matrix(&htt_factors[i], &row.htbeta);
        backend.block_gemm_subtract(&mut schur, &row.htbeta, &solved);
    }
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

fn build_dense_schur_sqrt_ba<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    backend: &B,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    if sys.hbb.dim() != (k, k) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Square-Root BA direct solve requires a dense K×K shared H_ββ block"
                .to_string(),
        });
    }
    let mut schur = sys.hbb.clone();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    for (i, row) in sys.rows.iter().enumerate() {
        // Square-Root BA: H_tβ^T H_tt^-1 H_tβ =
        // (L^-1 H_tβ)^T (L^-1 H_tβ), where H_tt = L L^T.
        let whitened = backend.sqrt_solve_block_matrix(&htt_factors[i], &row.htbeta);
        backend.block_gemm_subtract(&mut schur, &whitened, &whitened);
    }
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

fn solve_dense_reduced_system(
    schur: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, Option<Array2<f64>>), ArrowSchurError> {
    let factor =
        cholesky_lower(schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
    let direct = chol_solve_vector(&factor, rhs_beta);
    if step_inside_trust_region(direct.view(), options.trust_region.radius, metric_weights) {
        return Ok((direct, Some(factor)));
    }

    // Ceres-style trust-region correction: once the dense BA solve proposes a
    // step outside the trust ball, Steihaug-CG returns the boundary point
    // without requiring a second dense factorization.
    let identity = IdentityPreconditioner;
    let delta = steihaug_dense_system(
        schur,
        rhs_beta,
        &identity,
        &ArrowPcgOptions {
            max_iterations: options.trust_region.max_iterations,
            relative_tolerance: options.trust_region.steihaug_relative_tolerance,
        },
        &options.trust_region,
        metric_weights,
    )?;
    Ok((delta, Some(factor)))
}

fn step_inside_trust_region(
    step: ArrayView1<'_, f64>,
    radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> bool {
    !radius.is_finite() || metric_norm(step, metric_weights) <= radius
}

fn schur_matvec<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    backend: &B,
) {
    let k = sys.k;
    let d = sys.d;
    if let Some(hbb_matvec) = sys.hbb_matvec.as_ref() {
        hbb_matvec(x.view(), out);
        for a in 0..k {
            out[a] += ridge_beta * x[a];
        }
    } else {
        for a in 0..k {
            let mut acc = ridge_beta * x[a];
            for b in 0..k {
                acc += sys.hbb[[a, b]] * x[b];
            }
            out[a] = acc;
        }
    }
    let mut local = Array1::<f64>::zeros(d);
    for (i, row) in sys.rows.iter().enumerate() {
        for c in 0..d {
            let mut acc = 0.0;
            for a in 0..k {
                acc += row.htbeta[[c, a]] * x[a];
            }
            local[c] = acc;
        }
        let solved = backend.solve_block_vector(&htt_factors[i], &local);
        for a in 0..k {
            let mut acc = 0.0;
            for c in 0..d {
                acc += row.htbeta[[c, a]] * solved[c];
            }
            out[a] -= acc;
        }
    }
}

/// Jacobi Schur preconditioner for BA's inexact reduced-system PCG.
///
/// This is the block-diagonal Schur preconditioner specialized to scalar
/// decoder coefficients. When coefficient blocking metadata lands, this type
/// is the replacement point for Ceres-style block-Jacobi or cluster-Jacobi.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    inverse_diag: Array1<f64>,
}

impl JacobiPreconditioner {
    /// Build `diag(S)^-1` without materializing the dense Schur complement,
    /// following the Schur-Jacobi preconditioner used by large BA PCG.
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
    ) -> Self {
        let k = sys.k;
        let d = sys.d;
        let mut diag = Array1::<f64>::zeros(k);
        for a in 0..k {
            let base = match sys.hbb_diag.as_ref() {
                Some(hbb_diag) => hbb_diag[a],
                None => sys.hbb[[a, a]],
            };
            diag[a] = base + ridge_beta;
        }
        let mut col = Array1::<f64>::zeros(d);
        for (i, row) in sys.rows.iter().enumerate() {
            for a in 0..k {
                for c in 0..d {
                    col[c] = row.htbeta[[c, a]];
                }
                let solved = backend.solve_block_vector(&htt_factors[i], &col);
                let mut acc = 0.0;
                for c in 0..d {
                    acc += col[c] * solved[c];
                }
                diag[a] -= acc;
            }
        }
        let mut inverse_diag = Array1::<f64>::zeros(k);
        for a in 0..k {
            let v = diag[a];
            inverse_diag[a] = if v.is_finite() && v.abs() > 1e-18 {
                1.0 / v
            } else {
                1.0
            };
        }
        Self { inverse_diag }
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for i in 0..r.len() {
            out[i] = self.inverse_diag[i] * r[i];
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
struct IdentityPreconditioner;

impl IdentityPreconditioner {
    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        r.clone()
    }
}

fn steihaug_pcg_reduced_system<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    rhs: &Array1<f64>,
    preconditioner: &JacobiPreconditioner,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    metric_weights: Option<&MetricWeights>,
) -> Result<Array1<f64>, ArrowSchurError> {
    steihaug_cg(
        rhs,
        |p, out| schur_matvec(sys, htt_factors, ridge_beta, p, out, backend),
        |r| preconditioner.apply(r),
        pcg.max_iterations.min(trust.max_iterations),
        pcg.relative_tolerance
            .max(trust.steihaug_relative_tolerance),
        trust.radius,
        metric_weights,
    )
}

fn steihaug_dense_system(
    schur: &Array2<f64>,
    rhs: &Array1<f64>,
    preconditioner: &IdentityPreconditioner,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<Array1<f64>, ArrowSchurError> {
    steihaug_cg(
        rhs,
        |p, out| dense_matvec(schur, p, out),
        |r| preconditioner.apply(r),
        pcg.max_iterations,
        pcg.relative_tolerance,
        trust.radius,
        metric_weights,
    )
}

fn steihaug_cg<MatVec, ApplyPrec>(
    rhs: &Array1<f64>,
    mut matvec: MatVec,
    mut apply_preconditioner: ApplyPrec,
    max_iterations: usize,
    relative_tolerance: f64,
    trust_radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> Result<Array1<f64>, ArrowSchurError>
where
    MatVec: FnMut(&Array1<f64>, &mut Array1<f64>),
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let n = rhs.len();
    if let Some(weights) = metric_weights {
        assert_eq!(
            weights.len(),
            n,
            "Steihaug-CG metric weight length must match solve dimension"
        );
    }
    let radius = if trust_radius.is_finite() && trust_radius > 0.0 {
        trust_radius
    } else {
        f64::INFINITY
    };
    let rhs_norm = metric_norm(rhs.view(), metric_weights);
    if rhs_norm == 0.0 {
        return Ok(Array1::<f64>::zeros(n));
    }
    let tol = relative_tolerance.max(0.0) * rhs_norm;
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = apply_preconditioner(&r);
    let mut p = z.clone();
    let mut rz = metric_dot(&r, &z, metric_weights);
    if rz <= 0.0 || !rz.is_finite() {
        if radius.is_finite() {
            return Ok(step_to_trust_boundary(&x, &r, radius, metric_weights));
        }
        return Err(ArrowSchurError::PcgFailed {
            reason: "non-positive preconditioned residual in Schur PCG".to_string(),
        });
    }
    if rz.sqrt() <= tol {
        return Ok(x);
    }
    let mut ap = Array1::<f64>::zeros(n);
    for _ in 0..max_iterations {
        matvec(&p, &mut ap);
        let pap = metric_dot(&p, &ap, metric_weights);
        if pap <= 0.0 || !pap.is_finite() {
            if radius.is_finite() {
                return Ok(step_to_trust_boundary(&x, &p, radius, metric_weights));
            }
            return Err(ArrowSchurError::PcgFailed {
                reason: "negative curvature in unbounded Schur PCG".to_string(),
            });
        }
        let alpha = rz / pap;
        let mut candidate = x.clone();
        for i in 0..n {
            candidate[i] += alpha * p[i];
        }
        if radius.is_finite() && metric_norm(candidate.view(), metric_weights) >= radius {
            return Ok(step_to_trust_boundary(&x, &p, radius, metric_weights));
        }
        x = candidate;
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        if metric_norm(r.view(), metric_weights) <= tol {
            return Ok(x);
        }
        z = apply_preconditioner(&r);
        let rz_next = metric_dot(&r, &z, metric_weights);
        if rz_next <= 0.0 || !rz_next.is_finite() {
            return Err(ArrowSchurError::PcgFailed {
                reason: "non-positive or non-finite PCG residual".to_string(),
            });
        }
        let beta = rz_next / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_next;
    }
    Ok(x)
}

fn step_to_trust_boundary(
    x: &Array1<f64>,
    p: &Array1<f64>,
    radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> Array1<f64> {
    let pp = metric_dot(p, p, metric_weights);
    if pp == 0.0 {
        return x.clone();
    }
    let xp = metric_dot(x, p, metric_weights);
    let xx = metric_dot(x, x, metric_weights);
    let disc = (xp * xp + pp * (radius * radius - xx)).max(0.0);
    let tau = (-xp + disc.sqrt()) / pp;
    let mut out = x.clone();
    for i in 0..out.len() {
        out[i] += tau * p[i];
    }
    out
}

fn dense_matvec(a: &Array2<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
    let n = a.nrows();
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[[i, j]] * x[j];
        }
        out[i] = acc;
    }
}

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn metric_dot(a: &Array1<f64>, b: &Array1<f64>, metric_weights: Option<&MetricWeights>) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    match metric_weights {
        Some(weights) => {
            debug_assert_eq!(weights.len(), a.len());
            let mut acc = 0.0;
            for i in 0..a.len() {
                acc += weights[i] * a[i] * b[i];
            }
            acc
        }
        None => dot(a, b),
    }
}

fn metric_norm(v: ArrayView1<'_, f64>, metric_weights: Option<&MetricWeights>) -> f64 {
    let mut acc = 0.0;
    match metric_weights {
        Some(weights) => {
            debug_assert_eq!(weights.len(), v.len());
            for i in 0..v.len() {
                acc += weights[i] * v[i] * v[i];
            }
        }
        None => {
            for x in v.iter() {
                acc += x * x;
            }
        }
    }
    acc.sqrt()
}

fn symmetrize_upper_from_lower(a: &mut Array2<f64>) {
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
}

/// Errors raised by [`ArrowSchurSystem::solve`].
#[derive(Debug, Clone)]
pub enum ArrowSchurError {
    /// A per-row `H_tt^(i)` block was not positive-definite at the
    /// supplied ridge. Indicates an under-regularized latent block —
    /// typically a gauge-free fit without an identifiability penalty.
    PerRowFactorFailed { row: usize, reason: String },
    /// The Schur complement was not positive-definite. Indicates a
    /// near-collinear decoder or a degenerate weighting; the LM outer
    /// wrapper should escalate `ridge_beta` and retry.
    SchurFactorFailed { reason: String },
    /// The BA inexact-step PCG solve failed before producing a usable
    /// Steihaug trust-region step.
    PcgFailed { reason: String },
}

impl std::fmt::Display for ArrowSchurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowSchurError::PerRowFactorFailed { row, reason } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky failed: {reason}"
            ),
            ArrowSchurError::SchurFactorFailed { reason } => {
                write!(f, "arrow-Schur: Schur complement Cholesky failed: {reason}")
            }
            ArrowSchurError::PcgFailed { reason } => {
                write!(f, "arrow-Schur: Schur PCG failed: {reason}")
            }
        }
    }
}

impl std::error::Error for ArrowSchurError {}

// ---------------------------------------------------------------------------
// Cholesky helpers (kept local to avoid a new public-API dependency on the
// linalg crate. The systems here are tiny per-row (d × d, d ∈ {1..16}) and
// modest at the Schur level (K × K, K ∈ {basis size}). For production SAE
// scales the Schur factor should switch to faer; this module's `cholesky_lower`
// is the obvious replacement site.)
// ---------------------------------------------------------------------------

fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!("cholesky_lower: non-square {}×{}", n, a.ncols()));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

fn chol_solve_vector(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for kk in 0..i {
            sum -= l[[i, kk]] * y[kk];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for kk in (i + 1)..n {
            sum -= l[[kk, i]] * x[kk];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

fn chol_solve_matrix(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    let mut col = Array1::<f64>::zeros(n);
    for cidx in 0..m {
        for r in 0..n {
            col[r] = b[[r, cidx]];
        }
        let x = chol_solve_vector(l, &col);
        for r in 0..n {
            out[[r, cidx]] = x[r];
        }
    }
    out
}

fn lower_triangular_solve_matrix(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for cidx in 0..m {
        for i in 0..n {
            let mut sum = b[[i, cidx]];
            for kk in 0..i {
                sum -= l[[i, kk]] * out[[kk, cidx]];
            }
            out[[i, cidx]] = sum / l[[i, i]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Convenience: in-place writeback of the arrow-Schur Newton step into the
// global PIRLS direction buffer. The driver owns the layout: β occupies
// `[0, K)` and flat `t` occupies `[K, K + N·d)`.
// ---------------------------------------------------------------------------

/// Layout convention for the joint `(β, t)` direction buffer.
///
/// The β block occupies entries `[0, K)`; the flat per-row `t`
/// block occupies `[K, K + N·d)`. This matches the convention used
/// by [`crate::terms::analytic_penalties::PsiSlice`] and by the existing
/// `SpatialLogKappaCoords` extension to the outer ρ vector. BA analogue:
/// export the reduced-camera-system shared step followed by point
/// back-substitution increments.
pub fn write_arrow_direction(
    delta_t: &Array1<f64>,
    delta_beta: &Array1<f64>,
    out: &mut ArrayViewMut1<'_, f64>,
) {
    let k = delta_beta.len();
    let nt = delta_t.len();
    debug_assert_eq!(out.len(), k + nt);
    for j in 0..k {
        out[j] = delta_beta[j];
    }
    for i in 0..nt {
        out[k + i] = delta_t[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Verify the arrow-Schur solve against a small dense reference.
    /// Build the joint bordered system as a single dense (K + N·d)² matrix,
    /// solve it with the local cholesky_lower path, and compare to the
    /// arrow-Schur output.
    #[test]
    fn arrow_schur_matches_dense_reference_2x2() {
        // N = 2 rows, d = 2 latent, K = 3 β.
        let n = 2;
        let d = 2;
        let k = 3;
        let mut sys = ArrowSchurSystem::new(n, d, k);

        // Row 0: H_tt = [[2, 0.1],[0.1, 3]], H_tβ = [[1, 0, 0.5],[0.2, 1, 0]],
        //         g_t = [0.3, -0.2].
        sys.rows[0].htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.0, 0.5], [0.2, 1.0, 0.0]];
        sys.rows[0].gt = array![0.3_f64, -0.2];

        // Row 1.
        sys.rows[1].htt = array![[1.5_f64, -0.1], [-0.1, 2.0]];
        sys.rows[1].htbeta = array![[0.1_f64, 0.5, 0.0], [0.0, 0.3, 1.0]];
        sys.rows[1].gt = array![-0.1_f64, 0.4];

        // β-block.
        sys.hbb = array![[4.0_f64, 0.2, 0.0], [0.2, 5.0, 0.1], [0.0, 0.1, 6.0],];
        sys.gb = array![0.5_f64, -0.3, 0.2];

        let (delta_t, delta_beta) = sys.solve(0.0, 0.0).expect("arrow-schur solve");

        // Build dense reference: order is [β; t_0; t_1] = K + N·d entries.
        let total = k + n * d;
        let mut hjoint = Array2::<f64>::zeros((total, total));
        let mut gjoint = Array1::<f64>::zeros(total);
        // β-β block.
        for a in 0..k {
            for b in 0..k {
                hjoint[[a, b]] = sys.hbb[[a, b]];
            }
            gjoint[a] = sys.gb[a];
        }
        // t-blocks and cross-blocks.
        for i in 0..n {
            let toff = k + i * d;
            for a in 0..d {
                for b in 0..d {
                    hjoint[[toff + a, toff + b]] = sys.rows[i].htt[[a, b]];
                }
                gjoint[toff + a] = sys.rows[i].gt[a];
                for a2 in 0..k {
                    hjoint[[toff + a, a2]] = sys.rows[i].htbeta[[a, a2]];
                    hjoint[[a2, toff + a]] = sys.rows[i].htbeta[[a, a2]];
                }
            }
        }
        // Solve hjoint · x = -gjoint via cholesky.
        let lj = cholesky_lower(&hjoint).expect("dense ref PD");
        let neg_g = gjoint.mapv(|v| -v);
        let xref = chol_solve_vector(&lj, &neg_g);
        // Compare β.
        for a in 0..k {
            assert!(
                (xref[a] - delta_beta[a]).abs() < 1e-10,
                "β[{a}] mismatch: dense {} vs arrow {}",
                xref[a],
                delta_beta[a]
            );
        }
        // Compare t.
        for i in 0..n {
            for a in 0..d {
                let dense = xref[k + i * d + a];
                let arrow = delta_t[i * d + a];
                assert!(
                    (dense - arrow).abs() < 1e-10,
                    "t[{i},{a}] mismatch: dense {dense} vs arrow {arrow}"
                );
            }
        }
    }
}
