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

use ndarray::{Array1, Array2, ArrayView1};
use std::ops::Range;
use std::sync::Arc;

use crate::cache::Fingerprinter;
use crate::linalg::faer_ndarray::{FaerArrayView, FaerLlt};
use crate::solver::arrow_schur_beta_graph::BetaCouplingGraph;
use crate::terms::analytic_penalties::{AnalyticPenaltyKind, AnalyticPenaltyRegistry, PenaltyTier};
use crate::terms::latent_coord::{LatentCoordValues, LatentManifold};

const DIRECT_SOLVE_MAX_K: usize = 2_000;
const DEFAULT_PCG_MAX_ITERATIONS: usize = 200;
const DEFAULT_PCG_RELATIVE_TOLERANCE: f64 = 1e-4;
const DEFAULT_TRUST_REGION_RADIUS: f64 = f64::INFINITY;
pub const DEFAULT_PROXIMAL_INITIAL_RIDGE: f64 = 1e-8;
pub const DEFAULT_PROXIMAL_RIDGE_GROWTH: f64 = 10.0;
pub const DEFAULT_PROXIMAL_MAX_ATTEMPTS: usize = 16;
const DEFAULT_ARMIJO_C1: f64 = 1e-4;
const DEFAULT_GRADIENT_TOLERANCE: f64 = 1e-10;
const EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT: u64 = 0;
const ARROW_FACTOR_CACHE_HTBETA_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Matrix-free shared-block multiply for large BA/SAE Schur PCG.
///
/// The closure writes `out = H_ββ x` without the LM ridge. This is the hook
/// that lets SAE-manifold scale callers avoid materializing a dense `K × K`
/// shared block before Agarwal-style inexact Schur PCG.
pub type SharedBetaMatvec =
    Arc<dyn for<'a> Fn(ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;
pub type RowHtbetaMatvec =
    Arc<dyn for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync>;
pub type StreamingArrowRowBuilder =
    Arc<dyn Fn(usize) -> Result<ArrowRowBlock, ArrowSchurError> + Send + Sync>;

/// GPU-backed Schur matvec for CPU-driven PCG at K ≥ 5000.
///
/// The closure writes `out = S·x` where `S = H_ββ + ρ·I − Σ_i Y_i^T Y_i`
/// is the reduced shared system, with `Y_i = L_i^{-1} H_tβ^(i)` pre-computed
/// on device from the same forward kernel that Layer D uses for the dense Schur
/// build. The CPU-driven Steihaug-CG outer loop uploads `x` (K doubles),
/// receives `out` (K doubles), and handles the H_ββ contribution on the CPU side.
///
/// Constructed by `crate::gpu::arrow_schur::gpu_schur_matvec_backend` when
/// `cuda_selected()` and K ≥ 5000. The closure is `Send + Sync` so PCG callers
/// can hold it in an `Arc`.
pub type GpuSchurMatvec = Arc<dyn Fn(&Array1<f64>, &mut Array1<f64>) + Send + Sync>;

type MetricWeights = [f64];

// ---------------------------------------------------------------------------
// BetaPenaltyOp — matrix-free penalty-side H_ββ abstraction (#296)
// ---------------------------------------------------------------------------

/// Identifies one contiguous column block in the shared β vector for
/// block-Jacobi Schur pre-conditioning (#287).
///
/// A `BetaBlockId(i)` refers to the `i`-th range in
/// [`ArrowSchurSystem::block_offsets`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BetaBlockId(pub usize);

/// Matrix-free operator for the penalty side of `H_ββ`.
///
/// Callers must satisfy the additive convention: every method **adds** its
/// contribution to the output buffer (i.e. `y += P x`, not `y = P x`).
/// This matches the assembly pattern where multiple penalty terms are
/// accumulated into the same gradient / Hessian buffers.
pub trait BetaPenaltyOp: Send + Sync {
    /// Full dimension `K` of the β vector.
    fn dim(&self) -> usize;
    /// `y += P x` — penalty Hessian-vector product (length `K`).
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    /// Penalty gradient: `out += P β`.
    fn gradient(&self, beta: &[f64], out: &mut [f64]);
    /// `diag += diag(P)` — diagonal entries used by Jacobi preconditioner.
    fn diagonal(&self, diag: &mut [f64]);
    /// Add the `b×b` dense penalty sub-block for block `id` into `out`
    /// (row-major, block size `b = offsets[id.0].len()`).
    /// Used by the block-Jacobi Schur preconditioner (#287).
    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>);
    /// Materialize the full `K×K` dense penalty matrix (needed by
    /// Direct / SqrtBA modes that form the Schur complement explicitly).
    fn to_dense(&self) -> Array2<f64>;
}

/// Dense fallback: wraps the existing `K×K` `H_ββ` accumulator.
pub struct DensePenaltyOp(pub Array2<f64>);

impl BetaPenaltyOp for DensePenaltyOp {
    fn dim(&self) -> usize {
        self.0.nrows()
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let k = self.0.nrows();
        for a in 0..k {
            let mut acc = 0.0_f64;
            for b in 0..k {
                acc += self.0[[a, b]] * x[b];
            }
            y[a] += acc;
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        let k = self.0.nrows();
        for a in 0..k {
            let mut acc = 0.0_f64;
            for b in 0..k {
                acc += self.0[[a, b]] * beta[b];
            }
            out[a] += acc;
        }
    }

    fn diagonal(&self, diag: &mut [f64]) {
        let k = self.0.nrows().min(diag.len());
        for j in 0..k {
            diag[j] += self.0[[j, j]];
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        let range = &offsets[id.0];
        let b = range.end - range.start;
        for bi in 0..b {
            for bj in 0..b {
                out[[bi, bj]] += self.0[[range.start + bi, range.start + bj]];
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        self.0.clone()
    }
}

/// Block-local penalty operator: applies per-block penalty matrices
/// (matching `ParameterBlockSpec` boundaries) without materialising a
/// full `K×K` dense matrix.
///
/// Each entry is `(global_offset, local_matrix)` where `global_offset`
/// is the start of that block in the full β vector.
pub struct BlockPenaltyOp {
    /// Full β dimension `K`.
    pub k: usize,
    /// `(global_start, local_matrix)` for each atom/block.
    pub blocks: Vec<(usize, Array2<f64>)>,
}

impl BetaPenaltyOp for BlockPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (off, local) in &self.blocks {
            let b = local.nrows();
            for i in 0..b {
                let gi = off + i;
                let mut acc = 0.0_f64;
                for j in 0..b {
                    acc += local[[i, j]] * x[off + j];
                }
                y[gi] += acc;
            }
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        for (off, local) in &self.blocks {
            let b = local.nrows();
            for i in 0..b {
                let gi = off + i;
                let mut acc = 0.0_f64;
                for j in 0..b {
                    acc += local[[i, j]] * beta[off + j];
                }
                out[gi] += acc;
            }
        }
    }

    fn diagonal(&self, diag: &mut [f64]) {
        for (off, local) in &self.blocks {
            let b = local.nrows();
            for j in 0..b {
                diag[off + j] += local[[j, j]];
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        let range = &offsets[id.0];
        let b_out = range.end - range.start;
        for (off, local) in &self.blocks {
            let b = local.nrows();
            let block_end = off + b;
            if block_end <= range.start || *off >= range.end {
                continue;
            }
            for bi in 0..b_out {
                let gi = range.start + bi;
                if gi < *off || gi >= block_end {
                    continue;
                }
                let li = gi - off;
                for bj in 0..b_out {
                    let gj = range.start + bj;
                    if gj < *off || gj >= block_end {
                        continue;
                    }
                    let lj = gj - off;
                    out[[bi, bj]] += local[[li, lj]];
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for (off, local) in &self.blocks {
            let b = local.nrows();
            for i in 0..b {
                for j in 0..b {
                    out[[off + i, off + j]] += local[[i, j]];
                }
            }
        }
        out
    }
}

/// Kronecker-product penalty: `P = A ⊗ B` applied without materialising
/// the full `(p_a·p_b)×(p_a·p_b)` matrix.
pub struct KroneckerPenaltyOp {
    /// Left factor `A`, shape `(p_a, p_a)`.
    pub factor_a: Array2<f64>,
    /// Right factor `B`, shape `(p_b, p_b)`.
    pub factor_b: Array2<f64>,
    /// Global offset into the β vector where this block starts.
    pub global_offset: usize,
    /// Full β dimension `K`.
    pub k: usize,
}

impl BetaPenaltyOp for KroneckerPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let p_a = self.factor_a.nrows();
        let p_b = self.factor_b.nrows();
        let off = self.global_offset;
        // (A ⊗ B) vec(V) where V is (p_b, p_a) with Fortran/vec ordering.
        for i_a in 0..p_a {
            for i_b in 0..p_b {
                let gi = off + i_a * p_b + i_b;
                let mut acc = 0.0_f64;
                for j_a in 0..p_a {
                    let a_ij = self.factor_a[[i_a, j_a]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    for j_b in 0..p_b {
                        acc += a_ij * self.factor_b[[i_b, j_b]] * x[off + j_a * p_b + j_b];
                    }
                }
                y[gi] += acc;
            }
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        let p_a = self.factor_a.nrows();
        let p_b = self.factor_b.nrows();
        let off = self.global_offset;
        for i_a in 0..p_a {
            for i_b in 0..p_b {
                let gi = off + i_a * p_b + i_b;
                let mut acc = 0.0_f64;
                for j_a in 0..p_a {
                    let a_ij = self.factor_a[[i_a, j_a]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    for j_b in 0..p_b {
                        acc += a_ij * self.factor_b[[i_b, j_b]] * beta[off + j_a * p_b + j_b];
                    }
                }
                out[gi] += acc;
            }
        }
    }

    fn diagonal(&self, diag: &mut [f64]) {
        let p_a = self.factor_a.nrows();
        let p_b = self.factor_b.nrows();
        let off = self.global_offset;
        for i_a in 0..p_a {
            for i_b in 0..p_b {
                diag[off + i_a * p_b + i_b] +=
                    self.factor_a[[i_a, i_a]] * self.factor_b[[i_b, i_b]];
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        let range = &offsets[id.0];
        let b = range.end - range.start;
        let p_a = self.factor_a.nrows();
        let p_b = self.factor_b.nrows();
        let off = self.global_offset;
        let block_end = off + p_a * p_b;
        if block_end <= range.start || off >= range.end {
            return;
        }
        for bi in 0..b {
            let gi = range.start + bi;
            if gi < off || gi >= block_end {
                continue;
            }
            let li = gi - off;
            let i_a = li / p_b;
            let i_b = li % p_b;
            for bj in 0..b {
                let gj = range.start + bj;
                if gj < off || gj >= block_end {
                    continue;
                }
                let lj = gj - off;
                let j_a = lj / p_b;
                let j_b = lj % p_b;
                out[[bi, bj]] += self.factor_a[[i_a, j_a]] * self.factor_b[[i_b, j_b]];
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let p_a = self.factor_a.nrows();
        let p_b = self.factor_b.nrows();
        let off = self.global_offset;
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for i_a in 0..p_a {
            for i_b in 0..p_b {
                let gi = off + i_a * p_b + i_b;
                for j_a in 0..p_a {
                    let a_ij = self.factor_a[[i_a, j_a]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    for j_b in 0..p_b {
                        let gj = off + j_a * p_b + j_b;
                        out[[gi, gj]] += a_ij * self.factor_b[[i_b, j_b]];
                    }
                }
            }
        }
        out
    }
}

/// Composite penalty: sum of multiple `BetaPenaltyOp` operators.
pub struct CompositePenaltyOp {
    /// Full β dimension `K`.
    pub k: usize,
    /// Component operators, each contributing additively.
    pub ops: Vec<Arc<dyn BetaPenaltyOp>>,
}

impl BetaPenaltyOp for CompositePenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for op in &self.ops {
            op.matvec(x, y);
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        for op in &self.ops {
            op.gradient(beta, out);
        }
    }

    fn diagonal(&self, diag: &mut [f64]) {
        for op in &self.ops {
            op.diagonal(diag);
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        for op in &self.ops {
            op.block(id, offsets, out);
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for op in &self.ops {
            let dense = op.to_dense();
            out += &dense;
        }
        out
    }
}

/// BA Schur solve variant for the reduced shared `β` system.
///
/// * [`ArrowSolverMode::Direct`] is BA's dense reduced-camera-system solve:
///   eliminate the per-point/per-row blocks, form the reduced system, and
///   Cholesky factor it. This is the Ceres/g2o default for modest camera
///   counts and is appropriate here for `K <= 2000`.
///   **GPU support: ✓** — requires dense H_ββ and dense per-row H_tβ slabs.
///
/// * [`ArrowSolverMode::SqrtBA`] ports Square-Root BA (Demmel/Gao/Gu et al.,
///   CVPR 2021): Schur terms are formed as `(L_i^-1 H_tβ_i)^T
///   (L_i^-1 H_tβ_i)` from the per-row square-root factor `L_i`, avoiding
///   explicit `H_tt^-1 H_tβ` products. It is the preferred direct path when
///   single-precision assembly is introduced or when row blocks are poorly
///   conditioned.
///   **GPU support: ✓** — requires dense H_ββ and dense per-row H_tβ slabs.
///
/// * [`ArrowSolverMode::InexactPCG`] ports "Bundle Adjustment in the Large"
///   (Agarwal et al.): the Schur system is solved inexactly by PCG with a
///   Jacobi Schur preconditioner, avoiding dense `K × K` factorization for
///   SAE-manifold scale shared systems.
///   **GPU support: CPU only** until the row-procedural H_tβ GPU PCG path
///   (issue #288 Part B) is wired. The topology selector must not request
///   `InexactPCG` via the GPU entry point; `solve_arrow_newton_step` returns
///   `GpuRequiresDenseSystem` for matrix-free systems, and the wrapper in
///   `solver/gpu/arrow_schur_gpu.rs` routes those to CPU InexactPCG
///   automatically. At K ≥ 5000 the GPU PCG path will supersede the CPU path
///   once the row-procedural H_tβ kernel and boxed GPU matvec backend in
///   `steihaug_pcg_reduced_system` are wired.
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
    pub const fn automatic(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::Direct
        } else {
            Self::InexactPCG
        }
    }

    /// Square-Root BA is the direct-solve stability mode for future f32
    /// callers. Large `K` still routes to inexact PCG because dense Schur
    /// storage dominates precision concerns at that scale.
    pub const fn automatic_for_single_precision(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::SqrtBA
        } else {
            Self::InexactPCG
        }
    }
}

/// Reason the Steihaug-CG loop stopped.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PcgStopReason {
    /// Residual fell below the relative tolerance threshold.
    #[default]
    Converged,
    /// Loop exhausted max_iterations without converging.
    MaxIter,
    /// Step hit the trust-region boundary (Steihaug boundary projection).
    TrustRegion,
    /// Negative curvature detected in an unbounded solve.
    Indefinite,
    /// Non-positive or non-finite preconditioned residual after an update.
    Stagnation,
}

/// Per-solve instrumentation counters returned alongside the PCG solution.
///
/// All fields default to zero; callers that do not need diagnostics simply
/// ignore the value. The struct is Copy so passing it through return tuples
/// is zero-overhead.
#[derive(Debug, Default, Clone, Copy)]
pub struct PcgDiagnostics {
    /// Number of CG iterations executed.
    pub iterations: usize,
    /// Total calls to the Schur matvec A·p.
    pub matvec_calls: usize,
    /// Total calls to the preconditioner M^{-1}·r.
    pub precond_apply_calls: usize,
    /// Number of times the LM ridge was escalated before a successful factor.
    pub ridge_escalations: usize,
    /// Relative residual at termination; 0.0 when the RHS was zero.
    pub final_relative_residual: f64,
    /// Why the loop stopped.
    pub stopping_reason: PcgStopReason,
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
    /// Row chunk size for streaming direct/Square-Root Schur assembly.
    pub streaming_chunk_size: Option<usize>,
    /// Use the Riemannian latent projection before the Schur reduction. The
    /// reduced Steihaug solve itself remains in Euclidean β coordinates.
    pub riemannian_trust_region: bool,
    /// Optional GPU-backed Schur matvec for CPU-driven `InexactPCG` at K ≥ 5000.
    ///
    /// When set, `steihaug_pcg_reduced_system` delegates each `S·p` call to
    /// this closure instead of the CPU `schur_matvec`. Constructed by
    /// `crate::gpu::arrow_schur::gpu_schur_matvec_backend` when `cuda_selected()`
    /// and the system has dense per-row H_tβ slabs. `None` means CPU-only PCG.
    pub gpu_matvec: Option<GpuSchurMatvec>,
}

/// Globalization guard for non-convex arrow-Schur inner steps.
///
/// The raw Schur solve is exactly Newton. For non-convex analytic penalties,
/// full Newton can cycle. This controller adds a proximal LM shift `mu I` to
/// both blocks and accepts only Armijo-decreasing trial points.
#[derive(Debug, Clone)]
pub struct ArrowProximalCorrectionOptions {
    pub initial_ridge: f64,
    pub ridge_growth: f64,
    pub max_attempts: usize,
    pub armijo_c1: f64,
    pub gradient_tolerance: f64,
}

impl Default for ArrowProximalCorrectionOptions {
    fn default() -> Self {
        Self {
            initial_ridge: DEFAULT_PROXIMAL_INITIAL_RIDGE,
            ridge_growth: DEFAULT_PROXIMAL_RIDGE_GROWTH,
            max_attempts: DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            armijo_c1: DEFAULT_ARMIJO_C1,
            gradient_tolerance: DEFAULT_GRADIENT_TOLERANCE,
        }
    }
}

/// Accepted proximal arrow-Schur step and the damping that made it descent.
#[derive(Debug, Clone)]
pub struct ArrowAcceptedProximalStep {
    pub delta_t: Array1<f64>,
    pub delta_beta: Array1<f64>,
    pub ridge_t: f64,
    pub ridge_beta: f64,
    pub proximal_ridge: f64,
    pub objective_value: f64,
    pub trial_objective_value: f64,
    pub gradient_dot_step: f64,
    pub attempts: usize,
}

impl ArrowSolveOptions {
    /// Select Direct for `K <= 2000` and InexactPCG above, following BA RCS
    /// practice for dense-vs-iterative reduced systems.
    pub fn automatic(k: usize) -> Self {
        Self {
            mode: ArrowSolverMode::automatic(k),
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
        }
    }

    /// Force dense reduced-camera-system Cholesky, the classic BA direct
    /// solve for small `K`.
    pub fn direct() -> Self {
        Self {
            mode: ArrowSolverMode::Direct,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
        }
    }

    /// Force Square-Root BA Schur assembly for the direct reduced solve.
    pub fn sqrt_ba() -> Self {
        Self {
            mode: ArrowSolverMode::SqrtBA,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
        }
    }

    /// Force inexact BA Schur PCG with Jacobi preconditioning.
    pub fn inexact_pcg() -> Self {
        Self {
            mode: ArrowSolverMode::InexactPCG,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
        }
    }

    pub fn with_streaming_chunk_size(mut self, chunk_size: Option<usize>) -> Self {
        self.streaming_chunk_size = chunk_size.filter(|&chunk| chunk > 0);
        self
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
        for (row_idx, row) in rows.iter().enumerate() {
            out.push(factor_one_row(row, ridge_t, d, row_idx)?);
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
        // Performance: ndarray Array2 is row-major, so `right[[c, b]]` is
        // unit-strided in `b`. The canonical (a, b, c) order produced
        // strided reads of `left[[c, a]]` for every (a, b); reorder to
        // (c, a, b) so the inner `b`-loop is contiguous in `right` and
        // `left[[c, a]]` is hoisted out of the inner loop.
        let k = schur.nrows();
        let d = left.nrows();
        assert_eq!(left.ncols(), k);
        assert_eq!(right.ncols(), k);
        assert_eq!(schur.ncols(), k);
        for c in 0..d {
            for a in 0..k {
                let lca = left[[c, a]];
                if lca == 0.0 {
                    continue;
                }
                for b in 0..k {
                    schur[[a, b]] -= lca * right[[c, b]];
                }
            }
        }
    }
}

fn factor_one_row(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    row_idx: usize,
) -> Result<Array2<f64>, ArrowSchurError> {
    // Dimension mismatches in caller-supplied row blocks must surface as a
    // typed error rather than aborting the process. The BA/SAE assembler can
    // mis-size a row (for instance when latent_dim disagrees between the
    // design and the term that materialized the block), and downstream code
    // — including the LM outer loop — needs to recover by escalating ridge
    // or rebuilding the system, not by panicking.
    if row.htt.dim() != (d, d) {
        return Err(ArrowSchurError::PerRowFactorFailed {
            row: row_idx,
            reason: format!(
                "row {row_idx} H_tt shape {:?} does not match per_point_hessian_block dimension ({d}, {d})",
                row.htt.dim()
            ),
        });
    }
    if row.gt.len() != d {
        return Err(ArrowSchurError::PerRowFactorFailed {
            row: row_idx,
            reason: format!(
                "row {row_idx} g_t length {} does not match latent dimension {d}",
                row.gt.len()
            ),
        });
    }
    let mut block = row.htt.clone();
    for a in 0..d {
        block[[a, a]] += ridge_t;
    }
    let factor = cholesky_lower(&block).map_err(|e| ArrowSchurError::PerRowFactorFailed {
        row: row_idx,
        reason: format!(
            "row {row_idx} H_tt was non-PD at ridge_t={ridge_t}; \
             cholesky error: {e}"
        ),
    })?;
    // Cholesky succeeded, but barely-PD H_tt^(i) (pivots on the order of
    // ε·trace) yield an inverse with condition number ~1/ε. Plugging that
    // inverse into the Schur reduction
    //     S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
    // contaminates S by spectral terms scaled by κ_i, while still letting
    // the outer Cholesky on S succeed. Treat that case as functionally
    // equivalent to a PSD failure so LM escalation lifts ridge_t.
    //
    // Diagonal-ratio condition-number proxy: for a Cholesky factor L,
    //     κ(L Lᵀ) ≈ (max_i L_ii / min_i L_ii)².
    // (Golub & Van Loan, "Matrix Computations" 4th ed., §4.2.4 — the
    // ratio of diagonal entries of the Cholesky factor bounds the
    // 2-norm condition number of the SPD matrix.)
    //
    // Near-singularity threshold for double precision at dimension d:
    //     κ_max = 1 / (sqrt(DBL_EPS) · max(d, 1)).
    // This is the classic Higham (Higham, "Accuracy and Stability of
    // Numerical Algorithms" 2nd ed., §10.1) rule: a system is treated
    // as numerically rank-deficient once κ · ε approaches 1/sqrt(ε),
    // scaled by problem dimension.
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;
    for a in 0..d {
        let v = factor[[a, a]];
        if v < min_diag {
            min_diag = v;
        }
        if v > max_diag {
            max_diag = v;
        }
    }
    if min_diag > 0.0 && max_diag.is_finite() {
        let ratio = max_diag / min_diag;
        let kappa_est = ratio * ratio;
        let d_scale = (d as f64).max(1.0);
        let kappa_max = 1.0 / (f64::EPSILON.sqrt() * d_scale);
        if !kappa_est.is_finite() || kappa_est > kappa_max {
            return Err(ArrowSchurError::PerRowFactorIllConditioned {
                row: row_idx,
                kappa_estimate: kappa_est,
            });
        }
    } else {
        return Err(ArrowSchurError::PerRowFactorIllConditioned {
            row: row_idx,
            kappa_estimate: f64::INFINITY,
        });
    }
    Ok(factor)
}

fn manifold_mode_fingerprint(latent: &LatentCoordValues) -> u64 {
    let manifold = latent.manifold();
    if manifold.is_euclidean() {
        return EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT;
    }

    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-manifold-mode-v1");
    hasher.write_usize(latent.n_obs());
    hasher.write_usize(latent.latent_dim());
    write_latent_manifold(&mut hasher, manifold);
    let mut metric_weights = Vec::new();
    append_latent_metric_weights(&mut metric_weights, manifold);
    hasher.write_usize(metric_weights.len());
    for weight in metric_weights {
        hasher.write_f64(weight);
    }
    hasher.finish_u64()
}

fn row_hessian_fingerprint_for_system(sys: &ArrowSchurSystem) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-row-hessian-v2");
    hasher.write_usize(sys.rows.len());
    hasher.write_usize(sys.d);
    hasher.write_usize(sys.k);
    // When htbeta_matvec is installed (Kronecker / matrix-free path),
    // row.htbeta is a zero slab that does not capture the operator state.
    // Hash the Arc pointer address as a proxy: a new Arc is allocated per
    // assemble call, so the fingerprint is invalidated each time the system
    // is rebuilt with a fresh Kronecker operator.
    let htbeta_op_addr: Option<usize> = sys
        .htbeta_matvec
        .as_ref()
        .map(|op| Arc::as_ptr(op) as usize);
    for row in sys.rows.iter() {
        write_array2_fingerprint(&mut hasher, &row.htt);
        match htbeta_op_addr {
            Some(addr) => hasher.write_usize(addr),
            None => write_array2_fingerprint(&mut hasher, &row.htbeta),
        }
    }
    write_array2_fingerprint(&mut hasher, &sys.hbb);
    match sys.hbb_diag.as_ref() {
        Some(diag) => {
            hasher.write_bool(true);
            hasher.write_usize(diag.len());
            for &value in diag.iter() {
                hasher.write_f64(value);
            }
        }
        None => hasher.write_bool(false),
    }
    hasher.finish_u64()
}

fn combine_row_and_registry_fingerprints(row: u64, registry: u64) -> u64 {
    if registry == 0 {
        return row;
    }
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-row-hessian-with-penalties-v1");
    hasher.write_u64(row);
    hasher.write_u64(registry);
    hasher.finish_u64()
}

fn stable_softplus_for_fingerprint(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn write_array2_fingerprint(hasher: &mut Fingerprinter, values: &Array2<f64>) {
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    for &value in values.iter() {
        hasher.write_f64(value);
    }
}

fn analytic_penalty_row_hessian_fingerprint(
    penalty: &AnalyticPenaltyKind,
    target_t: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
) -> Option<u64> {
    if penalty.tier() != PenaltyTier::Psi || !analytic_penalty_is_row_block_diagonal(penalty) {
        return None;
    }

    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-analytic-row-hessian-v1");
    hasher.write_str(penalty.name());
    hasher.write_usize(target_t.len());
    hasher.write_usize(rho_local.len());
    for &rho in rho_local.iter() {
        hasher.write_f64(rho);
    }

    match penalty {
        AnalyticPenaltyKind::RowPrecisionPrior(p) => {
            let (n, rows, cols) = p.lambda_per_row.dim();
            hasher.write_str("row-precision-fixed");
            hasher.write_usize(n);
            hasher.write_usize(rows);
            hasher.write_usize(cols);
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            if p.learnable_weight {
                hasher.write_usize(p.rho_index);
                hasher.write_f64(p.weight * rho_local[p.rho_index].exp());
            }
            for &value in p.lambda_per_row.iter() {
                hasher.write_f64(value);
            }
        }
        AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
            let (aux_n, aux_dim) = p.aux.dim();
            let (mu_rows, mu_cols) = p.mu.dim();
            let weight_offset = p.log_alpha.len() + p.raw_beta.len() + p.mu.len();
            hasher.write_str("row-precision-parametric");
            hasher.write_usize(aux_n);
            hasher.write_usize(aux_dim);
            hasher.write_usize(mu_rows);
            hasher.write_usize(mu_cols);
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            for &value in p.aux.iter() {
                hasher.write_f64(value);
            }
            for k in 0..p.log_alpha.len() {
                let active_log_alpha = p.log_alpha[k] + rho_local[k];
                hasher.write_f64(p.log_alpha[k]);
                hasher.write_f64(active_log_alpha);
                hasher.write_f64(active_log_alpha.exp());
            }
            let raw_beta_offset = p.log_alpha.len();
            for k in 0..p.raw_beta.len() {
                let active_raw_beta = p.raw_beta[k] + rho_local[raw_beta_offset + k];
                hasher.write_f64(p.raw_beta[k]);
                hasher.write_f64(active_raw_beta);
                hasher.write_f64(stable_softplus_for_fingerprint(active_raw_beta));
            }
            let mu_offset = p.log_alpha.len() + p.raw_beta.len();
            for k in 0..p.mu.nrows() {
                for a in 0..p.mu.ncols() {
                    let idx = mu_offset + k * p.aux.ncols() + a;
                    hasher.write_f64(p.mu[[k, a]]);
                    hasher.write_f64(p.mu[[k, a]] + rho_local[idx]);
                }
            }
            if p.learnable_weight {
                hasher.write_usize(weight_offset);
                hasher.write_f64(p.weight * rho_local[weight_offset].exp());
            }
        }
        _ => {
            hasher.write_str("row-block-diagonal");
            if let Some(diag) = penalty.hessian_diag(target_t, rho_local) {
                hasher.write_usize(diag.len());
                for &value in diag.iter() {
                    hasher.write_f64(value);
                }
            } else {
                hasher.write_usize(0);
            }
        }
    }

    Some(hasher.finish_u64())
}

fn write_latent_manifold(hasher: &mut Fingerprinter, manifold: &LatentManifold) {
    match manifold {
        LatentManifold::Euclidean => {
            hasher.write_str("euclidean");
        }
        LatentManifold::Circle { period } => {
            hasher.write_str("circle");
            hasher.write_f64(*period);
        }
        LatentManifold::Sphere { dim } => {
            hasher.write_str("sphere");
            hasher.write_usize(*dim);
        }
        LatentManifold::Interval { lo, hi } => {
            hasher.write_str("interval");
            hasher.write_f64(*lo);
            hasher.write_f64(*hi);
        }
        LatentManifold::Product(parts) => {
            hasher.write_str("product");
            hasher.write_usize(parts.len());
            for part in parts {
                write_latent_manifold(hasher, part);
            }
        }
        LatentManifold::ProductWithMetric { manifolds, weights } => {
            hasher.write_str("product-with-metric");
            hasher.write_usize(manifolds.len());
            for part in manifolds {
                write_latent_manifold(hasher, part);
            }
            hasher.write_usize(weights.len());
            for weight in weights {
                hasher.write_f64(*weight);
            }
        }
    }
}

fn append_latent_metric_weights(out: &mut Vec<f64>, manifold: &LatentManifold) {
    match manifold {
        LatentManifold::Euclidean => out.push(1.0),
        LatentManifold::Circle { period } => {
            out.push(1.0 / (period * period));
        }
        LatentManifold::Sphere { dim } => {
            let scale = std::f64::consts::PI;
            for _ in 0..*dim {
                out.push(1.0 / (scale * scale));
            }
        }
        LatentManifold::Interval { lo, hi } => {
            let scale = hi - lo;
            out.push(1.0 / (scale * scale));
        }
        LatentManifold::Product(parts) => {
            for part in parts {
                append_latent_metric_weights(out, part);
            }
        }
        LatentManifold::ProductWithMetric {
            manifolds: _,
            weights,
        } => {
            out.extend(weights.iter().copied());
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
///      fold row-block Psi-tier analytic penalties (`ARDPenalty`,
///      `SparsityPenalty`) into `H_tt^(i)` and Beta-tier penalties into `H_ββ`;
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
    /// Optional row-local matrix-free multiply for `H_tβ^(i) x`.
    ///
    /// When present, all inner-Schur paths route through this operator instead
    /// of indexing the per-row `htbeta` dense slabs: `reduced_rhs_beta`,
    /// `schur_matvec` (PCG hot loop), back-substitution,
    /// `JacobiPreconditioner` construction, `build_dense_schur_direct`, and
    /// `build_dense_schur_sqrt_ba` all call `sys_htbeta_apply_row` or
    /// `sys_htbeta_materialize_row`.  Factor caches retain the operator for
    /// IFT/evidence consumers as before.
    pub htbeta_matvec: Option<RowHtbetaMatvec>,
    /// Optional diagonal of the matrix-free shared block, used by the
    /// Schur-Jacobi preconditioner in the Agarwal-style PCG path.
    pub hbb_diag: Option<Array1<f64>>,
    /// `g_β`, shape `(K,)`.
    pub gb: Array1<f64>,
    /// Maximum per-row latent dimensionality across all rows.
    ///
    /// For homogeneous systems (all rows have the same dim) this equals the
    /// common per-row `d`.  For heterogeneous systems (e.g. sparse SAE rows
    /// where JumpReLU / TopK / sparsemax active sets vary per observation)
    /// this is `max_i row_dims[i]`.  Per-row code should use
    /// `row.htt.nrows()` or `row_dims[i]`; `d` is an upper bound for
    /// scratch-buffer sizing.
    pub d: usize,
    /// Per-row latent dimensionality: `row_dims[i] == rows[i].htt.nrows()`.
    ///
    /// For homogeneous systems `row_dims[i] == d` for all `i`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets for the `delta_t` vector produced by
    /// [`Self::solve`] / [`solve_arrow_newton_step_core`].
    ///
    /// `row_offsets[i]` is the start index for row `i`'s slice in `delta_t`;
    /// `row_offsets[n]` is the total `delta_t` length.  For homogeneous
    /// systems `row_offsets[i] == i * d`.
    pub row_offsets: Arc<[usize]>,
    /// β dimensionality `K`.
    pub k: usize,
    /// Geometry tag for the row-local latent blocks after optional
    /// Riemannian projection. Euclidean/no-op geometry uses the sentinel.
    pub manifold_mode_fingerprint: u64,
    /// Structural/value tag for row-local Hessian factors and their Schur
    /// inputs. Stale caches must be rejected when row-dependent Hessian
    /// penalties or cross-blocks change.
    pub row_hessian_fingerprint: u64,
    /// Registry-side tag for row-dependent analytic-penalty Hessian inputs.
    /// Combined with the materialized row blocks in
    /// [`Self::current_row_hessian_fingerprint`].
    pub analytic_row_hessian_fingerprint: u64,
    /// Term-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Each entry `r` means that indices `r.start..r.end` belong to one
    /// coefficient block (a GAM term or a custom parameter family from
    /// `ParameterBlockSpec`). When populated via
    /// [`Self::set_block_offsets`], the Jacobi preconditioner inverts the
    /// full `b × b` Schur block for each term instead of only its diagonal.
    ///
    /// The default (empty slice) causes `JacobiPreconditioner` to fall back
    /// to pure scalar diagonal inversion, preserving the pre-#283 behaviour.
    pub block_offsets: Arc<[Range<usize>]>,
    /// Optional matrix-free penalty-side `H_ββ` operator (#296).
    ///
    /// When set, all hot paths (`schur_matvec`, `build_dense_schur_*`,
    /// `JacobiPreconditioner`, quadratic-form reduction) route through this
    /// operator instead of the dense `hbb` accumulator, enabling
    /// `BlockPenaltyOp` / `KroneckerPenaltyOp` to skip the `O(K²)` dense
    /// materialisation for structured smoothness penalties.
    ///
    /// When `None`, those paths fall back to wrapping `hbb` in a transient
    /// `DensePenaltyOp` — identical observable behaviour, no new allocation
    /// hot-path cost for callers that have not opted in.
    pub penalty_op: Option<Arc<dyn BetaPenaltyOp>>,
}

impl ArrowSchurSystem {
    /// Allocate an empty BA reduced-camera-system instance sized
    /// `(N point/latent rows × d, K shared decoder parameters)`.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((k, k)),
            hbb_matvec: None,
            htbeta_matvec: None,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
        };
        sys.refresh_row_hessian_fingerprint();
        sys
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
        assert_eq!(diag.len(), k);
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: Some(Arc::new(matvec)),
            htbeta_matvec: None,
            hbb_diag: Some(diag),
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
        };
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    /// Allocate a heterogeneous BA system where each row has its own latent
    /// dimensionality `per_row_dims[i]`.
    ///
    /// Used by sparse-assignment SAE paths (JumpReLU / TopK / sparsemax /
    /// hard-concrete) where the active-set size varies per observation.
    /// `sys.d` is set to `max(per_row_dims)` (or 0 for an empty system).
    pub fn new_with_per_row_dims(per_row_dims: Vec<usize>, k: usize) -> Self {
        let n = per_row_dims.len();
        let max_d = per_row_dims.iter().copied().max().unwrap_or(0);
        let row_dims: Arc<[usize]> = per_row_dims.iter().copied().collect::<Vec<_>>().into();
        let mut off_vec = Vec::with_capacity(n + 1);
        let mut cursor = 0usize;
        for &di in &per_row_dims {
            off_vec.push(cursor);
            cursor += di;
        }
        off_vec.push(cursor);
        let row_offsets: Arc<[usize]> = off_vec.into();
        let rows = per_row_dims
            .iter()
            .map(|&di| ArrowRowBlock::new(di, k))
            .collect();
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((k, k)),
            hbb_matvec: None,
            htbeta_matvec: None,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d: max_d,
            row_dims,
            row_offsets,
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
        };
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    /// Number of BA point/latent rows `N`.
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Recompute the row-system fingerprint from the currently materialized
    /// row blocks, cross-blocks, and shared-block diagonal.
    pub fn compute_row_hessian_fingerprint(&self) -> u64 {
        row_hessian_fingerprint_for_system(self)
    }

    /// Current effective row-system fingerprint, including the materialized
    /// row blocks and any registry metadata captured while folding analytic
    /// penalties into the system.
    pub fn current_row_hessian_fingerprint(&self) -> u64 {
        combine_row_and_registry_fingerprints(
            self.compute_row_hessian_fingerprint(),
            self.analytic_row_hessian_fingerprint,
        )
    }

    /// Store the current row-system fingerprint on the system.
    pub fn refresh_row_hessian_fingerprint(&mut self) {
        self.row_hessian_fingerprint = self.current_row_hessian_fingerprint();
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
        assert_eq!(diag.len(), self.k);
        self.hbb_matvec = Some(Arc::new(matvec));
        self.hbb_diag = Some(diag);
        self.refresh_row_hessian_fingerprint();
    }

    /// Install a matrix-free per-row cross-block operator.
    ///
    /// The closure must write `out = H_tβ^(row) x` for `out.len() == d` and
    /// `x.len() == K`.
    ///
    /// When installed, the operator is used both during the Newton solve
    /// (inside `reduced_rhs_beta`, `schur_matvec`, back-substitution, and
    /// `JacobiPreconditioner` construction) and afterwards by IFT/evidence
    /// predictors.  Per-row `htbeta` slabs in `ArrowRowBlock` may be left
    /// zero-sized when this operator is installed — all inner-Schur paths route
    /// through the matvec instead of indexing the dense block.
    pub fn set_row_htbeta_operator<F>(&mut self, matvec: F)
    where
        F: for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        self.htbeta_matvec = Some(Arc::new(matvec));
        self.refresh_row_hessian_fingerprint();
    }

    /// Register term-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// Each `Range<usize>` covers the columns of one GAM term (or custom
    /// parameter family) in the shared `β` vector. The ranges must be
    /// non-overlapping, sorted, and their union must cover `0..k`.
    ///
    /// Call this after building the system and before [`Self::solve`] /
    /// [`Self::solve_with_options`] whenever the solver will use
    /// [`ArrowSolverMode::InexactPCG`]. Absent a call, the preconditioner
    /// falls back to scalar diagonal Jacobi (the pre-#283 behaviour).
    ///
    /// The same plumbing is compatible with #287 (custom `ParameterBlockSpec`
    /// families): callers from that path simply supply ranges derived from
    /// their own block layout instead of `EngineLayout.terms[*].col_range`.
    pub fn set_block_offsets(&mut self, offsets: Arc<[Range<usize>]>) {
        self.block_offsets = offsets;
    }

    /// Install a matrix-free penalty-side `H_ββ` operator (#296).
    ///
    /// When set, all hot paths (`schur_matvec`, `build_dense_schur_*`,
    /// `JacobiPreconditioner`, quadratic-form reduction) route through this
    /// operator instead of the dense `hbb` accumulator, enabling
    /// `BlockPenaltyOp` / `KroneckerPenaltyOp` to avoid `O(K²)` allocation
    /// for structured smoothness penalties.
    pub fn set_penalty_op(&mut self, op: Arc<dyn BetaPenaltyOp>) {
        self.penalty_op = Some(op);
    }

    /// Return the effective penalty operator: the installed `penalty_op` if
    /// present, otherwise a `DensePenaltyOp` wrapping the current `hbb`.
    ///
    /// All hot paths call this once and dispatch through the trait, so there
    /// is no parallel dense branch — `DensePenaltyOp` IS the dense path.
    pub fn effective_penalty_op(&self) -> Arc<dyn BetaPenaltyOp> {
        match self.penalty_op.as_ref() {
            Some(op) => Arc::clone(op),
            None => Arc::new(DensePenaltyOp(self.hbb.clone())),
        }
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
    /// sparsity kernels) are injected directly. Psi-tier penalties with
    /// off-row Hessian blocks are rejected because the arrow representation
    /// has no place to store them. The supported row-block-only Psi-tier
    /// penalties are `ARDPenalty`, `SparsityPenalty`,
    /// `SoftmaxAssignmentSparsity`, `IBPAssignment`,
    /// `RowPrecisionPrior`, `ParametricRowPrecisionPrior`, and
    /// `ScadMcpPenalty`. Dense Beta-tier penalties still fall back to `hvp`
    /// probes against the canonical basis vectors for `β`.
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
    ) -> Result<(), ArrowSchurError> {
        let layout = registry.rho_layout();
        let mut penalty_fingerprints = Vec::new();
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(ndarray::s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    if !analytic_penalty_is_row_block_diagonal(penalty) {
                        return Err(ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "analytic penalty {name:?} couples latent rows; cross-row Hessian contributions are not yet supported on any production solver path. Consider using a row-block-only penalty (ARDPenalty, SparsityPenalty, SoftmaxAssignmentSparsity, IBPAssignment) or filing an issue requesting cross-row Hessian support."
                            ),
                        });
                    }
                    self.add_ext_coord_penalty(penalty, target_t, rho_local);
                    if let Some(fingerprint) =
                        analytic_penalty_row_hessian_fingerprint(penalty, target_t, rho_local)
                    {
                        penalty_fingerprints.push(fingerprint);
                    }
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
        self.analytic_row_hessian_fingerprint = if penalty_fingerprints.is_empty() {
            0
        } else {
            let mut hasher = Fingerprinter::new();
            hasher.write_str("arrow-schur-row-hessian-registry-v1");
            hasher.write_usize(penalty_fingerprints.len());
            for fingerprint in penalty_fingerprints {
                hasher.write_u64(fingerprint);
            }
            hasher.finish_u64()
        };
        self.refresh_row_hessian_fingerprint();
        Ok(())
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
        self.manifold_mode_fingerprint = manifold_mode_fingerprint(latent);
        if manifold.is_euclidean() {
            self.refresh_row_hessian_fingerprint();
            return;
        }
        assert_eq!(latent.n_obs(), self.rows.len());
        assert_eq!(latent.latent_dim(), self.d);
        for (i, row) in self.rows.iter_mut().enumerate() {
            let t_i = ArrayView1::from(latent.row(i));
            let gt_e = row.gt.clone();
            let htt_e = row.htt.clone();
            let htbeta_e = row.htbeta.clone();
            row.gt = manifold.project_to_tangent(t_i, gt_e.view());
            row.htt = manifold.riemannian_hessian_matrix(t_i, gt_e.view(), htt_e.view());
            row.htbeta = manifold.project_matrix_columns_to_tangent(t_i, htbeta_e.view());
        }
        self.refresh_row_hessian_fingerprint();
    }

    fn add_ext_coord_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_t: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        apply_analytic_penalty(
            penalty,
            target_t,
            rho_local,
            n * d,
            d,
            self,
            |sys, flat, value| sys.rows[flat / d].gt[flat % d] += value,
            |sys, flat, value| sys.rows[flat / d].htt[[flat % d, flat % d]] += value,
            |a, probe| {
                for i in 0..n {
                    probe[i * d + a] = 1.0;
                }
            },
            |sys, a, hv| {
                for i in 0..n {
                    for b in 0..d {
                        sys.rows[i].htt[[b, a]] += hv[i * d + b];
                    }
                }
            },
        );
    }

    fn add_beta_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let k = self.k;
        let hvp_columns = if self.hbb.dim() == (k, k) { k } else { 0 };
        apply_analytic_penalty(
            penalty,
            target_beta,
            rho_local,
            k,
            hvp_columns,
            self,
            |sys, j, value| sys.gb[j] += value,
            |sys, j, value| {
                if sys.hbb.dim() == (k, k) {
                    sys.hbb[[j, j]] += value;
                }
                if let Some(hbb_diag) = sys.hbb_diag.as_mut() {
                    hbb_diag[j] += value;
                }
            },
            |j, probe| probe[j] = 1.0,
            |sys, j, hv| {
                for i in 0..k {
                    sys.hbb[[i, j]] += hv[i];
                }
                // Keep `hbb_diag` consistent with the dense `hbb` Hessian when
                // both are populated (the dense-allocated path + a later
                // `set_shared_beta_operator` install). The HVP probe for
                // column `j` returns the full Hessian column, whose `j`-th
                // entry is the diagonal contribution of this penalty. Without
                // this mirror, the Jacobi Schur preconditioner — which prefers
                // `hbb_diag` over `hbb`'s diagonal — would silently use a
                // stale diagonal for any Beta-tier analytic penalty that
                // exposes only an HVP (no `hessian_diag`).
                if let Some(hbb_diag) = sys.hbb_diag.as_mut() {
                    hbb_diag[j] += hv[j];
                }
            },
        );
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

    /// Solve with the standard LM-style ridge escalation: if a per-row
    /// `H_tt + ridge_t·I` Cholesky pivot is non-PD, or the reduced Schur
    /// factor fails, geometrically grow both ridges and retry. This is the
    /// same Ceres-style proximal correction the Newton driver in
    /// `run_joint_fit_arrow_schur` performs around `solve`, lifted into the
    /// system itself so every entry point (predict OOS reconstruction,
    /// single-shot Newton refinement, …) is self-healing against the
    /// pathological per-row blocks produced by PCA-seeded latent
    /// coordinates on subset / new data — see #163 and #175.
    ///
    /// `ridge_t` / `ridge_beta` are the caller-nominal Tikhonov ridges; the
    /// escalation only adds extra damping on top of them when the factor
    /// fails. PCG / AdaptiveCorrection failures are left untouched because
    /// they are not factorization-recoverable.
    pub fn solve_with_lm_escalation(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let options = ArrowSolveOptions::automatic(self.k);
        solve_with_lm_escalation_inner(self, ridge_t, ridge_beta, &options)
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

/// Chunked Schur assembler that never retains all row cross-blocks.
pub struct StreamingArrowSchur {
    pub n_rows: usize,
    /// Maximum per-row latent dim (upper bound for scratch buffers).
    pub d: usize,
    /// Per-row latent dims `row_dims[i] == rows[i].htt.nrows()`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets: `row_offsets[i]` is the start of row `i` in
    /// `delta_t`; `row_offsets[n_rows]` is the total `delta_t` length.
    pub row_offsets: Arc<[usize]>,
    pub k: usize,
    pub chunk_size: usize,
    pub s_acc: Array2<f64>,
    rhs_acc: Array1<f64>,
    hbb: Array2<f64>,
    gb: Array1<f64>,
    row_builder: StreamingArrowRowBuilder,
}

impl std::fmt::Debug for StreamingArrowSchur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingArrowSchur")
            .field("n_rows", &self.n_rows)
            .field("d", &self.d)
            .field("k", &self.k)
            .field("chunk_size", &self.chunk_size)
            .finish_non_exhaustive()
    }
}

impl StreamingArrowSchur {
    #[must_use]
    pub fn new(
        n_rows: usize,
        d: usize,
        row_dims: Arc<[usize]>,
        row_offsets: Arc<[usize]>,
        k: usize,
        hbb: Array2<f64>,
        gb: Array1<f64>,
        row_builder: StreamingArrowRowBuilder,
        chunk_size: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        assert_eq!(gb.len(), k);
        Self {
            n_rows,
            d,
            row_dims,
            row_offsets,
            k,
            chunk_size: chunk_size.max(1),
            s_acc: Array2::<f64>::zeros((k, k)),
            rhs_acc: Array1::<f64>::zeros(k),
            hbb,
            gb,
            row_builder,
        }
    }

    #[must_use]
    pub fn from_system(sys: &ArrowSchurSystem, chunk_size: usize) -> Self {
        // When a Kronecker / matrix-free htbeta_matvec is installed, the dense
        // row.htbeta slabs may be zero-sized.  Materialize them now so the
        // streaming accumulator's validate_row + direct arithmetic can work.
        let rows: Vec<ArrowRowBlock> = if sys.htbeta_matvec.is_some() {
            sys.rows
                .iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    let htbeta = sys_htbeta_materialize_row(sys, row_idx, row);
                    ArrowRowBlock {
                        htt: row.htt.clone(),
                        htbeta,
                        gt: row.gt.clone(),
                    }
                })
                .collect()
        } else {
            sys.rows.clone()
        };
        let rows = Arc::new(rows);
        let row_builder: StreamingArrowRowBuilder = Arc::new(move |row| {
            rows.get(row)
                .cloned()
                .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                    reason: format!("streaming row {row} out of bounds"),
                })
        });
        Self::new(
            sys.rows.len(),
            sys.d,
            Arc::clone(&sys.row_dims),
            Arc::clone(&sys.row_offsets),
            sys.k,
            sys.hbb.clone(),
            sys.gb.clone(),
            row_builder,
            chunk_size,
        )
    }

    /// Reset the dense shared accumulator to `H_ββ + ridge_beta I`.
    pub fn reset_accumulator(&mut self, ridge_beta: f64) -> Result<(), ArrowSchurError> {
        if self.hbb.dim() != (self.k, self.k) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "streaming Arrow-Schur requires a dense beta block accumulator".to_string(),
            });
        }
        self.s_acc.assign(&self.hbb);
        for j in 0..self.k {
            self.s_acc[[j, j]] += ridge_beta;
            self.rhs_acc[j] = 0.0;
        }
        Ok(())
    }

    /// Accumulate rows `[start, end)` into the reduced RHS and Schur block.
    pub fn accumulate_chunk(
        &mut self,
        start: usize,
        end: usize,
        ridge_t: f64,
        mode: ArrowSolverMode,
    ) -> Result<(), ArrowSchurError> {
        if start > end || end > self.n_rows {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "streaming Arrow-Schur chunk [{start}, {end}) outside 0..{}",
                    self.n_rows
                ),
            });
        }
        let backend = CpuBatchedBlockSolver;
        for row_idx in start..end {
            let row = (self.row_builder)(row_idx)?;
            let di = row.htt.nrows();
            self.validate_row(row_idx, &row)?;
            let factor = factor_one_row(&row, ridge_t, di, row_idx)?;
            let v = backend.solve_block_vector(&factor, &row.gt);
            for c in 0..di {
                let vc = v[c];
                if vc == 0.0 {
                    continue;
                }
                for a in 0..self.k {
                    self.rhs_acc[a] += row.htbeta[[c, a]] * vc;
                }
            }
            match mode {
                ArrowSolverMode::Direct => {
                    let solved = backend.solve_block_matrix(&factor, &row.htbeta);
                    backend.block_gemm_subtract(&mut self.s_acc, &row.htbeta, &solved);
                }
                ArrowSolverMode::SqrtBA => {
                    let whitened = backend.sqrt_solve_block_matrix(&factor, &row.htbeta);
                    backend.block_gemm_subtract(&mut self.s_acc, &whitened, &whitened);
                }
                ArrowSolverMode::InexactPCG => {
                    return Err(ArrowSchurError::PcgFailed {
                        reason: "streaming Arrow-Schur accumulator is for dense direct modes; use matrix-free PCG without streaming_chunk_size".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn solve(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>, Option<Array2<f64>>), ArrowSchurError> {
        self.reset_accumulator(ridge_beta)?;
        for start in (0..self.n_rows).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(self.n_rows);
            self.accumulate_chunk(start, end, ridge_t, options.mode)?;
        }
        for j in 0..self.k {
            self.rhs_acc[j] -= self.gb[j];
        }
        symmetrize_upper_from_lower(&mut self.s_acc);
        let trust_metric_weights = None;
        let (delta_beta, schur_factor) =
            solve_dense_reduced_system(&self.s_acc, &self.rhs_acc, options, trust_metric_weights)?;
        let delta_t = self.back_substitute(ridge_t, delta_beta.view())?;
        Ok((delta_t, delta_beta, schur_factor))
    }

    fn back_substitute(
        &self,
        ridge_t: f64,
        delta_beta: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let backend = CpuBatchedBlockSolver;
        // Total delta_t length = row_offsets[n_rows].
        let total_len = self.row_offsets[self.n_rows];
        let mut delta_t = Array1::<f64>::zeros(total_len);
        let mut rhs = Array1::<f64>::zeros(self.d);
        for start in (0..self.n_rows).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(self.n_rows);
            for row_idx in start..end {
                let row = (self.row_builder)(row_idx)?;
                let di = row.htt.nrows();
                self.validate_row(row_idx, &row)?;
                let factor = factor_one_row(&row, ridge_t, di, row_idx)?;
                for c in 0..di {
                    let mut acc = row.gt[c];
                    for a in 0..self.k {
                        acc += row.htbeta[[c, a]] * delta_beta[a];
                    }
                    rhs[c] = acc;
                }
                let dt_i = backend.solve_block_vector(&factor, &rhs);
                let row_base = self.row_offsets[row_idx];
                for c in 0..di {
                    delta_t[row_base + c] = -dt_i[c];
                }
            }
        }
        Ok(delta_t)
    }

    fn validate_row(&self, row_idx: usize, row: &ArrowRowBlock) -> Result<(), ArrowSchurError> {
        let expected_di = if row_idx < self.row_dims.len() {
            self.row_dims[row_idx]
        } else {
            self.d
        };
        let actual_di = row.htt.nrows();
        if actual_di != expected_di || row.htt.ncols() != expected_di {
            return Err(ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason: format!(
                    "streaming row H_tt shape {:?} != ({expected_di}, {expected_di})",
                    row.htt.dim(),
                ),
            });
        }
        if row.htbeta.dim() != (expected_di, self.k) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "streaming row H_tβ shape {:?} != ({expected_di}, {})",
                    row.htbeta.dim(),
                    self.k
                ),
            });
        }
        if row.gt.len() != expected_di {
            return Err(ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason: format!(
                    "streaming row g_t length {} != {expected_di}",
                    row.gt.len()
                ),
            });
        }
        Ok::<(), _>(())
    }
}

fn apply_analytic_penalty<S, G, D, P, H>(
    penalty: &AnalyticPenaltyKind,
    target: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
    expected_target_len: usize,
    hvp_columns: usize,
    scatter_target: &mut S,
    mut grad_scatter: G,
    mut diag_scatter: D,
    seed_hvp_probe: P,
    mut hvp_column_scatter: H,
) where
    G: FnMut(&mut S, usize, f64),
    D: FnMut(&mut S, usize, f64),
    P: Fn(usize, &mut Array1<f64>),
    H: for<'a> FnMut(&mut S, usize, ArrayView1<'a, f64>),
{
    assert_eq!(target.len(), expected_target_len);

    let grad = penalty.grad_target(target, rho_local);
    for index in 0..expected_target_len {
        grad_scatter(scatter_target, index, grad[index]);
    }

    if let Some(diag) = penalty.hessian_diag(target, rho_local) {
        assert_eq!(diag.len(), expected_target_len);
        for index in 0..expected_target_len {
            diag_scatter(scatter_target, index, diag[index]);
        }
        return;
    }

    let mut probe = Array1::<f64>::zeros(expected_target_len);
    for column in 0..hvp_columns {
        probe.fill(0.0);
        seed_hvp_probe(column, &mut probe);
        let hv = penalty.hvp(target, rho_local, probe.view());
        hvp_column_scatter(scatter_target, column, hv.view());
    }
}

fn analytic_penalty_is_row_block_diagonal(penalty: &AnalyticPenaltyKind) -> bool {
    penalty.is_row_block_diagonal()
}

/// Per-row + Schur Cholesky factor cache produced by
/// [`solve_arrow_newton_step_with_options`]. Consumed downstream by the IFT warm-start
/// predictor in `crate::solver::persistent_warm_start`: when the outer
/// loop perturbs `(β, ρ)` by a small amount, the new Newton step can be
/// predicted by re-using these factors against a refreshed RHS, saving
/// the dominant `O(N d³ + K³)` factorization cost.
#[derive(Clone)]
pub enum ArrowUndampedFactors {
    SameAsDamped,
    Owned(Arc<[Array2<f64>]>),
}

impl std::fmt::Debug for ArrowUndampedFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SameAsDamped => f.write_str("SameAsDamped"),
            Self::Owned(factors) => f.debug_tuple("Owned").field(&factors.len()).finish(),
        }
    }
}

/// Apply `H_tβ^(row) · x` for one row, writing into `out` (length `d`).
///
/// Routes through `sys.htbeta_matvec` when present; otherwise indexes the dense
/// `row.htbeta` slab.  Panics when neither is available (zero-sized block and no
/// matvec) — callers must not invoke this when no cross-block is wired.
fn sys_htbeta_apply_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    x: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        op(row_idx, x, out);
    } else {
        // Per-row dim from the actual block shape (supports hetereogeneous systems).
        let di = row.htbeta.nrows();
        let k = sys.k;
        for c in 0..di {
            let mut acc = 0.0_f64;
            for a in 0..k {
                acc += row.htbeta[[c, a]] * x[a];
            }
            out[c] = acc;
        }
    }
}

/// Accumulate `H_βt^(row) · v` into `out` (length `k`).
///
/// `out[a] += Σ_c H_tβ^(row)[c, a] · v[c]`
///
/// Routes through `sys.htbeta_matvec` (column-probe) when present; otherwise
/// indexes the dense `row.htbeta` slab directly.
fn sys_htbeta_accumulate_transpose(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    v: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        let di = v.len();
        htbeta_probe_transpose(row_idx, op, v, out, di, sys.k);
    } else {
        // Per-row dim from actual block shape.
        let di = row.htbeta.nrows();
        let k = sys.k;
        for c in 0..di {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for a in 0..k {
                out[a] += row.htbeta[[c, a]] * vc;
            }
        }
    }
}

/// Materialize the dense `(di, k)` cross-block for one row.
///
/// When `sys.htbeta_matvec` is set and `row.htbeta` is zero-sized, probes each
/// of the `k` standard basis vectors to reconstruct the matrix.  When the dense
/// block is already present with the correct per-row shape, clones it.
fn sys_htbeta_materialize_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
) -> Array2<f64> {
    let di = sys.row_dims[row_idx];
    let k = sys.k;
    if row.htbeta.dim() == (di, k) {
        return row.htbeta.clone();
    }
    // Zero-sized or mismatched dense block: materialize via the matvec.
    let op = sys.htbeta_matvec.as_ref().unwrap_or_else(|| {
        panic!(
            "row {row_idx}: htbeta shape {:?} != ({di}, {k}) and no htbeta_matvec installed",
            row.htbeta.dim()
        )
    });
    let mut mat = Array2::<f64>::zeros((di, k));
    let mut e_a = Array1::<f64>::zeros(k);
    let mut col = Array1::<f64>::zeros(di);
    for a in 0..k {
        e_a.fill(0.0);
        e_a[a] = 1.0;
        col.fill(0.0);
        op(row_idx, e_a.view(), &mut col);
        for c in 0..d {
            mat[[c, a]] = col[c];
        }
    }
    mat
}

/// Probe each column of `H_tβ^(row)` by applying the operator to `e_a` and
/// dotting the result with `v`.  Accumulates into `out[a]` for all `a in 0..k`.
///
/// `out[a] += (H_tβ^(row) e_a) · v = H_βt^(row)[a, :] · v`
fn htbeta_probe_transpose(
    row: usize,
    op: &RowHtbetaMatvec,
    v: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
    d: usize,
    k: usize,
) {
    let mut e_a = Array1::<f64>::zeros(k);
    let mut col_a = Array1::<f64>::zeros(d);
    for a in 0..k {
        e_a.fill(0.0);
        e_a[a] = 1.0;
        col_a.fill(0.0);
        op(row, e_a.view(), &mut col_a);
        let mut acc = 0.0_f64;
        for c in 0..d {
            acc += col_a[c] * v[c];
        }
        out[a] += acc;
    }
}

#[derive(Clone)]
pub enum ArrowHtbetaCache {
    Dense {
        blocks: Arc<[Array2<f64>]>,
        estimated_bytes: usize,
    },
    Matvec {
        op: RowHtbetaMatvec,
        estimated_bytes: usize,
    },
    Disabled {
        estimated_bytes: usize,
    },
}

impl std::fmt::Debug for ArrowHtbetaCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense {
                blocks,
                estimated_bytes,
            } => f
                .debug_struct("Dense")
                .field("blocks", &blocks.len())
                .field("estimated_bytes", estimated_bytes)
                .finish(),
            Self::Matvec {
                estimated_bytes, ..
            } => f
                .debug_struct("Matvec")
                .field("estimated_bytes", estimated_bytes)
                .finish(),
            Self::Disabled { estimated_bytes } => f
                .debug_struct("Disabled")
                .field("estimated_bytes", estimated_bytes)
                .finish(),
        }
    }
}

impl ArrowHtbetaCache {
    fn is_available(&self) -> bool {
        !matches!(self, Self::Disabled { .. })
    }

    fn apply_row(
        &self,
        row: usize,
        delta_beta: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) -> bool {
        match self {
            Self::Dense { blocks, .. } => {
                let Some(block) = blocks.get(row) else {
                    return false;
                };
                if block.ncols() != delta_beta.len() || block.nrows() != out.len() {
                    return false;
                }
                for c in 0..block.nrows() {
                    let mut acc = 0.0_f64;
                    for a in 0..block.ncols() {
                        acc += block[[c, a]] * delta_beta[a];
                    }
                    out[c] = acc;
                }
                true
            }
            Self::Matvec { op, .. } => {
                op(row, delta_beta, out);
                true
            }
            Self::Disabled { .. } => false,
        }
    }

    /// Apply the transpose: `out[a] += H_βt^(row)[a, c] · v[c]` for all `a`.
    ///
    /// `v` has length `d`; `out` has length `k`. Accumulates (does NOT zero
    /// `out` first) so callers can sum contributions across rows into a shared
    /// accumulator.  Returns `false` when the cache is `Disabled` and no
    /// `fallback_op` is provided.
    fn apply_row_transpose_accumulate(
        &self,
        row: usize,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        k: usize,
        fallback_op: Option<&RowHtbetaMatvec>,
    ) -> bool {
        match self {
            Self::Dense { blocks, .. } => {
                let Some(block) = blocks.get(row) else {
                    return false;
                };
                if block.nrows() != v.len() || block.ncols() != out.len() {
                    return false;
                }
                // H_βt^(i) · v: outer-loop c hoists v[c], inner-loop a is
                // contiguous in row-major (d, k) layout.
                for c in 0..block.nrows() {
                    let vc = v[c];
                    if vc == 0.0 {
                        continue;
                    }
                    for a in 0..block.ncols() {
                        out[a] += block[[c, a]] * vc;
                    }
                }
                true
            }
            Self::Matvec { op, .. } => {
                // Probe column-by-column: H_tβ^(row) e_a is column a.  dot(col_a, v)
                // is entry a of H_βt^(row) v.
                htbeta_probe_transpose(row, op, v, out, d, k);
                true
            }
            Self::Disabled { .. } => {
                // No cached block.  Use the caller-supplied fallback op if present.
                if let Some(op) = fallback_op {
                    htbeta_probe_transpose(row, op, v, out, d, k);
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrowFactorCache {
    /// Per-row lower-triangular Cholesky factors of `H_tt^(i) + ridge_t·I`.
    ///
    /// These are the *damped* factors used inside the Newton solve. The IFT
    /// predictor must NOT use them — see [`Self::htt_factors_undamped`].
    pub htt_factors: Arc<[Array2<f64>]>,
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
    pub htt_factors_undamped: ArrowUndampedFactors,
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
    /// Per-row cross-block access for `H_tβ^(i) x`.
    ///
    /// Large caches retain a row matvec callback or disable β-coupled IFT
    /// prediction instead of cloning every dense `d × K` slab.
    pub htbeta: ArrowHtbetaCache,
    /// Maximum per-row latent dim (upper bound; matches `sys.d` at creation).
    pub d: usize,
    /// Per-row latent dims: `row_dims[i]` is the active dim for row `i`.
    pub row_dims: Arc<[usize]>,
    /// Flat-buffer row offsets for `delta_t` / IFT output vectors.
    /// `row_offsets[i]` is the start of row `i`; `row_offsets[n]` is the
    /// total length.
    pub row_offsets: Arc<[usize]>,
    /// β dimensionality `K`.
    pub k: usize,
    /// Geometry tag for the row-local factors and cross-blocks.
    pub manifold_mode_fingerprint: u64,
    /// Row-system tag for the cached per-row factors, cross-blocks, and
    /// shared-block diagonal used to build the Schur factor.
    pub row_hessian_fingerprint: u64,
    /// PCG instrumentation from the solve that produced this cache.
    ///
    /// Zero-valued (default) when the selected mode did not use PCG
    /// (i.e. `Direct` or `SqrtBA`).
    pub pcg_diagnostics: PcgDiagnostics,
}

impl ArrowFactorCache {
    pub fn n_rows(&self) -> usize {
        self.htt_factors.len()
    }

    pub fn htbeta_available(&self) -> bool {
        self.htbeta.is_available()
    }

    pub fn undamped_factor(&self, row: usize) -> &Array2<f64> {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => &self.htt_factors[row],
            ArrowUndampedFactors::Owned(factors) => &factors[row],
        }
    }

    pub fn undamped_factor_count(&self) -> usize {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => self.htt_factors.len(),
            ArrowUndampedFactors::Owned(factors) => factors.len(),
        }
    }

    pub fn undamped_factors_iter(&self) -> impl Iterator<Item = &Array2<f64>> {
        (0..self.undamped_factor_count()).map(|row| self.undamped_factor(row))
    }

    /// The total length of `delta_t` / IFT output vectors for this cache.
    pub fn delta_t_len(&self) -> usize {
        self.row_offsets[self.n_rows()]
    }

    pub fn apply_htbeta_row(
        &self,
        row: usize,
        delta_beta: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
    ) -> bool {
        let di = if row < self.row_dims.len() {
            self.row_dims[row]
        } else {
            self.d
        };
        if out.len() != di || delta_beta.len() != self.k {
            return false;
        }
        self.htbeta.apply_row(row, delta_beta, out)
    }

    /// Accumulate `out[a] += H_βt^(row)[a, :] · v` for all `a in 0..k`.
    ///
    /// `v` has length `row_dims[row]`; `out` has length `k`. The caller must
    /// zero `out` before the first call if it needs a fresh result.  Returns
    /// `false` when the cache is `Disabled` and no `fallback_op` is provided;
    /// callers must treat the accumulator as invalid in that case.
    pub fn apply_htbeta_row_transpose(
        &self,
        row: usize,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        fallback_op: Option<&RowHtbetaMatvec>,
    ) -> bool {
        let di = if row < self.row_dims.len() {
            self.row_dims[row]
        } else {
            self.d
        };
        if v.len() != di || out.len() != self.k {
            return false;
        }
        self.htbeta
            .apply_row_transpose_accumulate(row, v, out, di, self.k, fallback_op)
    }

    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) · Δβ)` per row, returning
    /// the flat `Δt` of total length `row_offsets[N]`.
    ///
    /// IFT first-order predictor for the latent field under a
    /// shape-coefficient perturbation `Δβ`. See
    /// `proposals/latent_coord.md` §2.2. BA analogue: back-substitution after
    /// reduced-camera-system solve.
    pub fn predict_delta_t_from_delta_beta(&self, delta_beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.undamped_factor_count();
        let total_len = self.delta_t_len();
        assert_eq!(delta_beta.len(), self.k);
        if !self.htbeta_available() {
            return Array1::<f64>::zeros(total_len);
        }
        let mut out = Array1::<f64>::zeros(total_len);
        let mut rhs = Array1::<f64>::zeros(self.d);
        for i in 0..n {
            let di = self.row_dims[i];
            rhs.fill(0.0);
            let rhs_i = rhs.slice_mut(ndarray::s![..di]);
            let mut rhs_slice = rhs_i.to_owned();
            if !self.apply_htbeta_row(i, delta_beta.view(), &mut rhs_slice) {
                return Array1::<f64>::zeros(total_len);
            }
            let v = chol_solve_vector(self.undamped_factor(i), &rhs_slice);
            let row_base = self.row_offsets[i];
            for c in 0..di {
                out[row_base + c] = -v[c];
            }
        }
        out
    }

    /// Apply the *combined* IFT predictor
    /// `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) Δβ + δg_t^(i))` per row.
    ///
    /// This is the canonical single-pass form of the IFT formula from
    /// `proposals/per_point_hessian.md` §4. Compared to the legacy split
    /// path (`predict_delta_t_from_delta_beta` + `predict_delta_t_from_delta_gt`),
    /// this routine performs *one* per-row Cholesky back-substitution
    /// instead of two — halving the IFT predictor cost for callers that
    /// have both a β perturbation and a per-row gradient perturbation.
    pub fn predict_delta_t_combined(
        &self,
        delta_beta: Option<ArrayView1<'_, f64>>,
        delta_gt: Option<ArrayView1<'_, f64>>,
    ) -> Array1<f64> {
        let n = self.undamped_factor_count();
        let total_len = self.delta_t_len();
        if let Some(db) = delta_beta.as_ref() {
            assert_eq!(db.len(), self.k);
        }
        if let Some(dg) = delta_gt.as_ref() {
            assert_eq!(dg.len(), total_len);
        }
        let mut out = Array1::<f64>::zeros(total_len);
        // Hoist per-row scratch outside the loop; sized to max_d.
        let mut rhs = Array1::<f64>::zeros(self.d);
        let mut htbeta_delta = Array1::<f64>::zeros(self.d);
        for i in 0..n {
            let di = self.row_dims[i];
            let row_base = self.row_offsets[i];
            for c in 0..di {
                rhs[c] = 0.0;
            }
            if let Some(db) = delta_beta.as_ref() {
                for c in 0..di {
                    htbeta_delta[c] = 0.0;
                }
                let mut htbeta_slice = htbeta_delta.slice_mut(ndarray::s![..di]).to_owned();
                if !self.apply_htbeta_row(i, db.view(), &mut htbeta_slice) {
                    return Array1::<f64>::zeros(total_len);
                }
                for c in 0..di {
                    rhs[c] += htbeta_slice[c];
                }
            }
            if let Some(dg) = delta_gt.as_ref() {
                for c in 0..di {
                    rhs[c] += dg[row_base + c];
                }
            }
            let rhs_slice = rhs.slice(ndarray::s![..di]).to_owned();
            let v = chol_solve_vector(self.undamped_factor(i), &rhs_slice);
            for c in 0..di {
                out[row_base + c] = -v[c];
            }
        }
        out
    }

    /// Arrow log-determinant
    /// `log|H| = Σ_i log|H_{t_i t_i}| + log|Schur_β|`
    /// using the cached (damped) factors.
    ///
    /// Returns `(log_det_tt_sum, log_det_schur)` so the caller can decide
    /// what to do with the Schur piece (e.g. REML evidence wants both;
    /// some diagnostics want only the per-row sum). `None` for the Schur
    /// piece signals that the cache was produced by an InexactPCG solve
    /// and never formed/factored the dense `K × K` reduced system.
    ///
    /// The log-determinant of a Cholesky factor `L` of `M` is
    /// `2 Σ log L_ii`.
    pub fn arrow_log_det(&self) -> (f64, Option<f64>) {
        let mut log_det_tt = 0.0_f64;
        for l in self.htt_factors.iter() {
            for i in 0..l.nrows() {
                log_det_tt += l[[i, i]].ln();
            }
        }
        log_det_tt *= 2.0;
        let log_det_schur = self.schur_factor.as_ref().map(|l| {
            let mut s = 0.0_f64;
            for i in 0..l.nrows() {
                s += l[[i, i]].ln();
            }
            2.0 * s
        });
        (log_det_tt, log_det_schur)
    }

    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · δg_t^(i)` per row.
    ///
    /// IFT first-order predictor for the latent field under a
    /// per-row gradient perturbation (typically `∂g_t/∂ρ · Δρ`
    /// resolved externally by the driver). BA analogue: reuse point-block
    /// factors for local point updates after shared parameters move.
    pub fn predict_delta_t_from_delta_gt(&self, delta_gt: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.undamped_factor_count();
        let total_len = self.delta_t_len();
        assert_eq!(delta_gt.len(), total_len);
        assert_eq!(
            self.undamped_factor_count(),
            n,
            "undamped factor cache and N must agree"
        );
        let mut out = Array1::<f64>::zeros(total_len);
        for i in 0..n {
            let di = self.row_dims[i];
            let row_base = self.row_offsets[i];
            let rhs = delta_gt.slice(ndarray::s![row_base..row_base + di]).to_owned();
            let v = chol_solve_vector(self.undamped_factor(i), &rhs);
            for c in 0..di {
                out[row_base + c] = -v[c];
            }
        }
        out
    }
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
    if options.streaming_chunk_size.is_some() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "streaming Arrow-Schur solve does not materialize the factor cache required by this entry point".to_string(),
        });
    }
    let step = solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)?;
    let backend = CpuBatchedBlockSolver;

    let htbeta_estimated_bytes =
        estimated_htbeta_bytes(sys.rows.len(), sys.d, sys.k).unwrap_or(usize::MAX);
    let htbeta = if let Some(op) = sys.htbeta_matvec.as_ref() {
        ArrowHtbetaCache::Matvec {
            op: Arc::clone(op),
            estimated_bytes: htbeta_estimated_bytes,
        }
    } else if htbeta_estimated_bytes <= ARROW_FACTOR_CACHE_HTBETA_BUDGET_BYTES {
        ArrowHtbetaCache::Dense {
            blocks: sys
                .rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>()
                .into(),
            estimated_bytes: htbeta_estimated_bytes,
        }
    } else {
        ArrowHtbetaCache::Disabled {
            estimated_bytes: htbeta_estimated_bytes,
        }
    };
    // Factor the UNDAMPED per-row blocks for the IFT predictor. When
    // ridge_t was zero the damped and undamped factors coincide and we
    // can alias htt_factors directly; otherwise pay a second per-row
    // Cholesky (O(N d³), same complexity class as the Newton solve).
    let htt_factors = Arc::<[Array2<f64>]>::from(step.htt_factors);
    let htt_factors_undamped = if ridge_t == 0.0 {
        ArrowUndampedFactors::SameAsDamped
    } else {
        ArrowUndampedFactors::Owned(backend.factor_blocks(&sys.rows, 0.0, sys.d)?.into())
    };
    let cache = ArrowFactorCache {
        htt_factors,
        htt_factors_undamped,
        schur_factor: step.schur_factor,
        solver_mode: options.mode,
        ridge_t,
        ridge_beta,
        htbeta,
        d: sys.d,
        row_dims: Arc::clone(&sys.row_dims),
        row_offsets: Arc::clone(&sys.row_offsets),
        k: sys.k,
        manifold_mode_fingerprint: sys.manifold_mode_fingerprint,
        row_hessian_fingerprint: sys.current_row_hessian_fingerprint(),
        pcg_diagnostics: step.pcg_diagnostics,
    };
    Ok((step.delta_t, step.delta_beta, cache))
}

fn estimated_htbeta_bytes(n: usize, d: usize, k: usize) -> Option<usize> {
    n.checked_mul(d)?
        .checked_mul(k)?
        .checked_mul(std::mem::size_of::<f64>())
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
    if let Some(chunk_size) = options.streaming_chunk_size {
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        return streaming
            .solve(ridge_t, ridge_beta, options)
            .map(|(delta_t, delta_beta, _)| (delta_t, delta_beta));
    }
    solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)
        .map(|step| (step.delta_t, step.delta_beta))
}

/// LM-style ridge escalation around `solve_arrow_newton_step_core`.
///
/// On `PerRowFactorFailed` / `PerRowFactorIllConditioned` /
/// `SchurFactorFailed` (the factorization-level failure modes triggered
/// when a per-row `H_tt + ridge_t·I` block is non-PD, barely-PD with a
/// condition estimate above the safe Schur threshold, or the reduced
/// Schur complement has a non-PD pivot at the nominal ridge),
/// geometrically grow a `proximal_ridge` on top of the caller-supplied
/// `ridge_t` / `ridge_beta` and retry, exactly as the Ceres-style proximal
/// correction the Newton driver in `run_joint_fit_arrow_schur` does around
/// `solve`. Non-factorization failures (PCG divergence, adaptive-correction
/// exhaustion) surface immediately because they are not recoverable by
/// shifting the diagonal.
///
/// Returns the same `(Δt, Δβ)` as `solve_arrow_newton_step_core`, computed
/// with the smallest escalated ridge that produced a successful factor.
pub fn solve_with_lm_escalation_inner(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    let mut proximal_ridge = 0.0_f64;
    let mut escalations: usize = 0;
    let mut last_err: Option<ArrowSchurError> = None;
    for attempt in 0..=DEFAULT_PROXIMAL_MAX_ATTEMPTS {
        let damped_ridge_t = ridge_t + proximal_ridge;
        let damped_ridge_beta = ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_artifacts(sys, damped_ridge_t, damped_ridge_beta, options) {
            Ok(mut step) => {
                step.pcg_diagnostics.ridge_escalations = escalations;
                return Ok((step.delta_t, step.delta_beta));
            }
            Err(err) => {
                let recoverable = matches!(
                    err,
                    ArrowSchurError::PerRowFactorFailed { .. }
                        | ArrowSchurError::PerRowFactorIllConditioned { .. }
                        | ArrowSchurError::SchurFactorFailed { .. }
                );
                last_err = Some(err);
                if !recoverable {
                    break;
                }
                if attempt == DEFAULT_PROXIMAL_MAX_ATTEMPTS {
                    break;
                }
                proximal_ridge = if proximal_ridge == 0.0 {
                    DEFAULT_PROXIMAL_INITIAL_RIDGE
                } else {
                    proximal_ridge * DEFAULT_PROXIMAL_RIDGE_GROWTH
                };
                escalations += 1;
            }
        }
    }
    Err(last_err.expect("escalation loop set last_err on failure"))
}

/// Solve a non-convex arrow-Schur step with adaptive proximal damping.
///
/// `trial_objective` receives the proposed `(delta_t, delta_beta)` and must
/// return the true nonlinear objective after applying that step. The function
/// increases a common proximal ridge until factorization succeeds, the
/// direction is descent, and Armijo decrease holds.
pub fn solve_arrow_newton_step_with_proximal_correction<F>(
    sys: &ArrowSchurSystem,
    base_ridge_t: f64,
    base_ridge_beta: f64,
    current_objective_value: f64,
    options: &ArrowSolveOptions,
    correction: &ArrowProximalCorrectionOptions,
    mut trial_objective: F,
) -> Result<ArrowAcceptedProximalStep, ArrowSchurError>
where
    F: for<'a, 'b> FnMut(ArrayView1<'a, f64>, ArrayView1<'b, f64>) -> f64,
{
    if !current_objective_value.is_finite() {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: "current objective is not finite".to_string(),
        });
    }
    if !(correction.ridge_growth.is_finite() && correction.ridge_growth > 1.0) {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: format!(
                "ridge_growth must be finite and > 1; got {}",
                correction.ridge_growth
            ),
        });
    }
    if !(correction.armijo_c1.is_finite()
        && correction.armijo_c1 > 0.0
        && correction.armijo_c1 < 1.0)
    {
        return Err(ArrowSchurError::AdaptiveCorrectionFailed {
            reason: format!("armijo_c1 must be in (0, 1); got {}", correction.armijo_c1),
        });
    }

    let grad_norm = arrow_gradient_norm(sys);
    if grad_norm <= correction.gradient_tolerance.max(0.0) {
        return Ok(ArrowAcceptedProximalStep {
            delta_t: Array1::<f64>::zeros(sys.row_offsets[sys.rows.len()]),
            delta_beta: Array1::<f64>::zeros(sys.k),
            ridge_t: base_ridge_t,
            ridge_beta: base_ridge_beta,
            proximal_ridge: 0.0,
            objective_value: current_objective_value,
            trial_objective_value: current_objective_value,
            gradient_dot_step: 0.0,
            attempts: 0,
        });
    }

    let mut proximal_ridge = correction.initial_ridge.max(0.0);
    let mut last_reason = String::from("no attempts were made");
    for attempt in 0..correction.max_attempts {
        let ridge_t = base_ridge_t + proximal_ridge;
        let ridge_beta = base_ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_core(sys, ridge_t, ridge_beta, options) {
            Ok((delta_t, delta_beta)) => {
                let g_dot_p = arrow_gradient_dot_step(sys, delta_t.view(), delta_beta.view());
                if !(g_dot_p.is_finite() && g_dot_p < 0.0) {
                    last_reason =
                        format!("candidate was not a finite descent direction: g·p={g_dot_p}");
                } else {
                    let trial_value = trial_objective(delta_t.view(), delta_beta.view());
                    let armijo_bound = current_objective_value + correction.armijo_c1 * g_dot_p;
                    if trial_value.is_finite() && trial_value <= armijo_bound {
                        return Ok(ArrowAcceptedProximalStep {
                            delta_t,
                            delta_beta,
                            ridge_t,
                            ridge_beta,
                            proximal_ridge,
                            objective_value: current_objective_value,
                            trial_objective_value: trial_value,
                            gradient_dot_step: g_dot_p,
                            attempts: attempt + 1,
                        });
                    }
                    last_reason = format!(
                        "Armijo rejected trial objective {trial_value}; bound {armijo_bound}"
                    );
                }
            }
            Err(err) => {
                last_reason = err.to_string();
            }
        }
        proximal_ridge = next_proximal_ridge(proximal_ridge, correction.ridge_growth);
    }

    Err(ArrowSchurError::AdaptiveCorrectionFailed {
        reason: format!(
            "failed after {} attempts; last rejection: {last_reason}",
            correction.max_attempts
        ),
    })
}

/// Predicted reduction of the *damped* joint Arrow-Schur quadratic model.
///
/// Includes the LM ridge terms in the quadratic:
///
/// `m(δ) - m(0) = gᵀδ + 0.5 δᵀ(H + ridge)δ`
///
/// Use this only for internal LM rejection logic that needs the damped model
/// (e.g. checking whether a candidate step satisfies a trust-region condition
/// against the augmented quadratic). For gain-ratio computations against the
/// bare penalized objective, use [`arrow_bare_quadratic_model_reduction`].
pub fn arrow_damped_quadratic_model_reduction(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<f64, ArrowSchurError> {
    let total_len = sys.row_offsets[sys.rows.len()];
    assert_eq!(delta_t.len(), total_len);
    assert_eq!(delta_beta.len(), sys.k);
    let mut lin = sys.gb.dot(&delta_beta);
    let mut quad = ridge_beta * delta_beta.dot(&delta_beta);

    let mut hbb_delta = Array1::<f64>::zeros(sys.k);
    if let Some(hbb_matvec) = sys.hbb_matvec.as_ref() {
        hbb_matvec(delta_beta, &mut hbb_delta);
    } else if sys.hbb.dim() == (sys.k, sys.k) {
        hbb_delta.assign(&sys.hbb.dot(&delta_beta));
    } else {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Arrow-Schur predicted reduction requires a dense H_ββ block or matrix-free H_ββ operator".to_string(),
        });
    }
    quad += delta_beta.dot(&hbb_delta);

    // Allocate scratch at max_d; per-row slice is ..di.
    let mut htbeta_x = Array1::<f64>::zeros(sys.d);
    for (i, row) in sys.rows.iter().enumerate() {
        let di = sys.row_dims[i];
        let row_base = sys.row_offsets[i];
        // H_tβ^(i) · Δβ via helper (routes through htbeta_matvec when present).
        let mut htbeta_x_i = htbeta_x.slice_mut(ndarray::s![..di]).to_owned();
        htbeta_x_i.fill(0.0);
        sys_htbeta_apply_row(sys, i, row, delta_beta, &mut htbeta_x_i);
        for c in 0..di {
            let dt_c = delta_t[row_base + c];
            lin += row.gt[c] * dt_c;
            quad += ridge_t * dt_c * dt_c;
            for r in 0..di {
                quad += dt_c * row.htt[[c, r]] * delta_t[row_base + r];
            }
            quad += 2.0 * dt_c * htbeta_x_i[c];
        }
    }

    Ok(-(lin + 0.5 * quad))
}

/// Predicted reduction of the *bare* joint Arrow-Schur quadratic model.
///
/// Drops the LM ridge contributions from the quadratic so the predicted
/// reduction is measured against the same bare penalized objective that the
/// actual reduction is measured against:
///
/// `m_bare(δ) - m_bare(0) = gᵀδ + 0.5 δᵀH δ`
///
/// Implemented as:
///   damped_quad − 0.5·(ridge_beta·‖δβ‖² + ridge_t·‖δt‖²)
///
/// When #282 lands and damping becomes diagonal (`λD²` instead of scalar `λI`),
/// replace the scalar `ridge_beta` / `ridge_t` correction with
/// `0.5 · δβᵀ(D_beta²)δβ` and `0.5 · δtᵀ(D_t²)δt` respectively — the
/// structure of this function already accepts per-scalar corrections; passing
/// a per-coordinate D² diagonal merely requires looping over coordinates
/// instead of multiplying by the squared norm.
///
/// Use this for PIRLS gain-ratio computations and any other place where the
/// accept/reject criterion compares against the bare (non-augmented) objective.
pub fn arrow_bare_quadratic_model_reduction(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<f64, ArrowSchurError> {
    // Compute the damped version first, then subtract the ridge contributions
    // to recover the bare-H quadratic.  This mirrors the beta-only PIRLS path:
    //     δ'(H+λI)δ − λ‖δ‖² = δ'Hδ
    let damped = arrow_damped_quadratic_model_reduction(
        sys, delta_t, delta_beta, ridge_t, ridge_beta,
    )?;
    // Subtract 0.5 * (ridge_beta * ‖δβ‖² + ridge_t * ‖δt‖²).
    // The sign convention: arrow_damped returns -(lin + 0.5*quad), so the
    // ridge terms enter with a negative sign there.  To remove them we add
    // back 0.5 * (ridge_beta * ‖δβ‖² + ridge_t * ‖δt‖²).
    let ridge_beta_contrib = 0.5 * ridge_beta * delta_beta.dot(&delta_beta);
    let ridge_t_contrib = {
        let mut acc = 0.0_f64;
        for v in delta_t.iter() {
            acc += v * v;
        }
        0.5 * ridge_t * acc
    };
    Ok(damped + ridge_beta_contrib + ridge_t_contrib)
}

fn next_proximal_ridge(current: f64, growth: f64) -> f64 {
    if current > 0.0 {
        current * growth
    } else {
        DEFAULT_PROXIMAL_INITIAL_RIDGE
    }
}

fn arrow_gradient_norm(sys: &ArrowSchurSystem) -> f64 {
    let mut sum = 0.0;
    for row in sys.rows.iter() {
        for &v in row.gt.iter() {
            sum += v * v;
        }
    }
    for &v in sys.gb.iter() {
        sum += v * v;
    }
    sum.sqrt()
}

fn arrow_gradient_dot_step(
    sys: &ArrowSchurSystem,
    delta_t: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(delta_t.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut out = 0.0;
    for (i, row) in sys.rows.iter().enumerate() {
        let di = sys.row_dims[i];
        let row_base = sys.row_offsets[i];
        for c in 0..di {
            out += row.gt[c] * delta_t[row_base + c];
        }
    }
    for a in 0..sys.k {
        out += sys.gb[a] * delta_beta[a];
    }
    out
}

struct ArrowNewtonStepArtifacts {
    delta_t: Array1<f64>,
    delta_beta: Array1<f64>,
    htt_factors: Vec<Array2<f64>>,
    schur_factor: Option<Array2<f64>>,
    pcg_diagnostics: PcgDiagnostics,
}

fn solve_arrow_newton_step_artifacts(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    if let Some(chunk_size) = options.streaming_chunk_size {
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        let (delta_t, delta_beta, schur_factor) = streaming.solve(ridge_t, ridge_beta, options)?;
        return Ok(ArrowNewtonStepArtifacts {
            delta_t,
            delta_beta,
            htt_factors: Vec::new(),
            schur_factor,
            pcg_diagnostics: PcgDiagnostics::default(),
        });
    }
    let n = sys.rows.len();
    let backend = CpuBatchedBlockSolver;

    // 1. BA point elimination: per-row Cholesky factors of
    // (H_tt^(i) + ridge_t · I).  `factor_blocks` reads the actual row
    // dimension from `row.htt.nrows()` so heterogeneous systems work.
    let htt_factors = backend.factor_blocks(&sys.rows, ridge_t, sys.d)?;

    // 2. Reduced RHS r_β = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i).
    let rhs_beta = reduced_rhs_beta(sys, &htt_factors, &backend);
    // The Schur solve is over the reduced β vector. Latent manifold metric
    // weights live on each d-dimensional t_i block, so the induced metric for
    // this β-only Steihaug problem is Euclidean.
    let trust_metric_weights = None;

    // 3. Solve reduced shared system using the selected BA mode.
    let (delta_beta, schur_factor, pcg_diagnostics) = match options.mode {
        ArrowSolverMode::Direct => {
            let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, &backend)?;
            let (db, sf, diag) = solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::SqrtBA => {
            let schur = build_dense_schur_sqrt_ba(sys, &htt_factors, ridge_beta, &backend)?;
            let (db, sf, diag) = solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::InexactPCG => {
            // Auto-select preconditioner level: starts with JacobiPreconditioner
            // (Diagonal / BetaBlockJacobi) and escalates to ClusterJacobi or
            // AdditiveSchwarz when K > 100 and PCG exhausts max_iterations.
            let (delta, diag) = steihaug_pcg_auto(
                sys,
                &htt_factors,
                ridge_beta,
                &rhs_beta,
                &options.pcg,
                &options.trust_region,
                &backend,
                options.gpu_matvec.as_ref(),
                trust_metric_weights,
            )?;
            (delta, None, diag)
        }
    };

    // 4. Back-substitute Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
    //
    // H_tβ^(i) · Δβ is routed through sys.htbeta_matvec when the dense slab
    // is absent; otherwise indexed directly from row.htbeta.
    // `row_offsets[n]` gives the total delta_t length for hetereogeneous dims.
    let total_dt_len = sys.row_offsets[n];
    let mut delta_t = Array1::<f64>::zeros(total_dt_len);
    // Allocate scratch at max_d so no per-row realloc.
    let mut rhs = Array1::<f64>::zeros(sys.d);
    let mut htbeta_delta = Array1::<f64>::zeros(sys.d);
    for i in 0..n {
        let di = sys.row_dims[i];
        let row_base = sys.row_offsets[i];
        assert_eq!(sys.rows[i].gt.len(), di);
        for c in 0..di {
            htbeta_delta[c] = 0.0;
        }
        let mut htbeta_slice = htbeta_delta.slice_mut(ndarray::s![..di]).to_owned();
        sys_htbeta_apply_row(sys, i, &sys.rows[i], delta_beta.view(), &mut htbeta_slice);
        let rhs_i = rhs.slice_mut(ndarray::s![..di]);
        for c in 0..di {
            rhs_i[c] = sys.rows[i].gt[c] + htbeta_slice[c];
        }
        let rhs_slice = rhs.slice(ndarray::s![..di]).to_owned();
        let dt_i = backend.solve_block_vector(&htt_factors[i], &rhs_slice);
        for c in 0..di {
            delta_t[row_base + c] = -dt_i[c];
        }
    }

    Ok(ArrowNewtonStepArtifacts {
        delta_t,
        delta_beta,
        htt_factors,
        schur_factor,
        pcg_diagnostics,
    })
}

fn reduced_rhs_beta<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    backend: &B,
) -> Array1<f64> {
    // Numerical invariant: each per-row `H_tt^(i)` factor must be PD
    // (already enforced by the adaptive-ridge `factor_blocks`).
    let k = sys.k;
    let mut rhs_beta = Array1::<f64>::zeros(k);
    for (i, row) in sys.rows.iter().enumerate() {
        let v = backend.solve_block_vector(&htt_factors[i], &row.gt);
        // H_βt^(i) · v accumulates into rhs_beta.  Routes through
        // sys.htbeta_matvec when the dense block is absent.
        sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut rhs_beta);
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
    // Materialise H_ββ via the BetaPenaltyOp trait (#296): DensePenaltyOp
    // for the legacy dense path, structured ops for SAE / Kronecker smooths.
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Direct BA requires a K×K shared H_ββ penalty operator".to_string(),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    for (i, row) in sys.rows.iter().enumerate() {
        // Materialize the (d, k) cross-block, probing via the matvec when
        // the dense slab is absent.
        let htbeta = sys_htbeta_materialize_row(sys, i, row);
        let solved = backend.solve_block_matrix(&htt_factors[i], &htbeta);
        backend.block_gemm_subtract(&mut schur, &htbeta, &solved);
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
    // Materialise H_ββ via the BetaPenaltyOp trait (#296).
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Square-Root BA direct solve requires a K×K shared H_ββ penalty operator"
                .to_string(),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    for (i, row) in sys.rows.iter().enumerate() {
        // Square-Root BA: H_tβ^T H_tt^-1 H_tβ =
        // (L^-1 H_tβ)^T (L^-1 H_tβ), where H_tt = L L^T.
        // Materialize the (d, k) cross-block, probing via the matvec when
        // the dense slab is absent.
        let htbeta = sys_htbeta_materialize_row(sys, i, row);
        let whitened = backend.sqrt_solve_block_matrix(&htt_factors[i], &htbeta);
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
) -> Result<(Array1<f64>, Option<Array2<f64>>, PcgDiagnostics), ArrowSchurError> {
    let factor =
        cholesky_lower(schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
    let direct = chol_solve_vector(&factor, rhs_beta);
    if step_inside_trust_region(direct.view(), options.trust_region.radius, metric_weights) {
        return Ok((direct, Some(factor), PcgDiagnostics::default()));
    }

    // Ceres-style trust-region correction: once the dense BA solve proposes a
    // step outside the trust ball, Steihaug-CG returns the boundary point
    // without requiring a second dense factorization.
    let identity = IdentityPreconditioner;
    let (delta, diag) = steihaug_dense_system(
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
    Ok((delta, Some(factor), diag))
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
    // Route the penalty-side H_ββ x product through the BetaPenaltyOp trait
    // (#296). effective_penalty_op() returns a DensePenaltyOp wrapping sys.hbb
    // when no structured op is installed, preserving existing behaviour.
    {
        let op = sys.effective_penalty_op();
        let x_slice = x.as_slice().expect("x must be contiguous");
        let out_slice = out.as_slice_mut().expect("out must be contiguous");
        op.matvec(x_slice, out_slice);
        for a in 0..k {
            out_slice[a] += ridge_beta * x_slice[a];
        }
    }
    // Allocate scratch at max_d; per-row slice is `..di`.
    let mut local = Array1::<f64>::zeros(sys.d);
    let mut neg_contrib = Array1::<f64>::zeros(k);
    for (i, row) in sys.rows.iter().enumerate() {
        let di = sys.row_dims[i];
        // H_tβ^(i) · x → local[..di], routed through sys.htbeta_matvec
        // when the dense block is absent.
        let mut local_i = local.slice_mut(ndarray::s![..di]).to_owned();
        local_i.fill(0.0);
        sys_htbeta_apply_row(sys, i, row, x.view(), &mut local_i);
        let solved = backend.solve_block_vector(&htt_factors[i], &local_i);
        // H_βt^(i) · solved accumulates into neg_contrib (length k), then
        // subtracted from out.  Routed through sys.htbeta_matvec when needed.
        neg_contrib.fill(0.0);
        sys_htbeta_accumulate_transpose(sys, i, row, solved.view(), &mut neg_contrib);
        for a in 0..k {
            out[a] -= neg_contrib[a];
        }
    }
}

/// One per-term block factor for the block-Jacobi Schur preconditioner.
///
/// Carries either a dense Cholesky factor (for PD blocks ≤ 256 columns) or
/// the scalar inverses for that block's diagonal as a fallback.
#[derive(Clone)]
enum BlockFactor {
    /// Cholesky L stored column-major via faer. `range` identifies the
    /// columns in the full K-vector this block covers.
    Chol { factor: FaerLlt<f64>, range: Range<usize> },
    /// Scalar fallback: per-element `1/s_aa` for each column in `range`.
    Scalar { inv: Array1<f64>, range: Range<usize> },
}

impl std::fmt::Debug for BlockFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFactor::Chol { range, .. } => {
                write!(f, "BlockFactor::Chol {{ range: {:?} }}", range)
            }
            BlockFactor::Scalar { inv, range } => {
                write!(
                    f,
                    "BlockFactor::Scalar {{ inv.len: {}, range: {:?} }}",
                    inv.len(),
                    range
                )
            }
        }
    }
}

/// Block-Jacobi Schur preconditioner for BA's inexact reduced-system PCG.
///
/// When [`ArrowSchurSystem::block_offsets`] is populated (via
/// [`ArrowSchurSystem::set_block_offsets`]) and the largest block has ≤ 256
/// columns, builds one small dense Schur block per term, factors it with
/// Cholesky (faer LLT), and applies the preconditioner as per-block
/// triangular solves.  Non-PD blocks fall back to scalar diagonal inversion
/// for that block only.  When `block_offsets` is empty or the largest block
/// exceeds 256 columns the preconditioner reduces to pure scalar-diagonal
/// Jacobi (pre-#283 behaviour), so callers that have not called
/// `set_block_offsets` are unaffected.
///
/// The `block_offsets` plumbing is compatible with issue #287 (custom
/// `ParameterBlockSpec` families): those callers supply ranges derived from
/// their own block layout rather than `EngineLayout.terms[*].col_range`.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    blocks: Vec<BlockFactor>,
}

/// Maximum block size for which we attempt dense block-Jacobi factorization.
const BLOCK_JACOBI_MAX_BLOCK: usize = 256;

impl JacobiPreconditioner {
    /// Build the block-Jacobi (or scalar fallback) preconditioner from the
    /// Arrow-Schur system without materializing the full dense Schur
    /// complement.
    ///
    /// When `sys.block_offsets` is non-empty and `max(block_size) ≤ 256`,
    /// each block gets a dense `b×b` Schur sub-matrix formed, factored, and
    /// stored.  Otherwise every column gets its own scalar entry.
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let use_block = !sys.block_offsets.is_empty()
            && sys
                .block_offsets
                .iter()
                .map(|r| r.end.saturating_sub(r.start))
                .max()
                .unwrap_or(0)
                <= BLOCK_JACOBI_MAX_BLOCK;
        if use_block {
            Self::build_block_jacobi(sys, htt_factors, ridge_beta, backend)
        } else {
            Self::build_scalar_jacobi(sys, htt_factors, ridge_beta, backend)
        }
    }

    /// Build scalar-diagonal Jacobi: one `BlockFactor::Scalar` of length 1
    /// per column.  Matches pre-#283 semantics.
    ///
    /// When `sys.htbeta_matvec` is set and per-row `htbeta` slabs are absent,
    /// each column is probed via the matvec (one call per column per row).
    fn build_scalar_jacobi<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        // Extract diagonal of H_ββ via BetaPenaltyOp trait (#296).
        let mut diag = Array1::<f64>::zeros(k);
        {
            let op = sys.effective_penalty_op();
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            op.diagonal(diag_slice);
        }
        for a in 0..k {
            diag[a] += ridge_beta;
        }
        // For each column a, extract H_tβ^(i) e_a via matvec probe when
        // dense slab is absent, then compute the scalar Schur diagonal.
        // Allocate scratch at max_d; per-row slice is ..di.
        let mut col = Array1::<f64>::zeros(sys.d);
        let mut e_a = Array1::<f64>::zeros(k);
        for (i, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[i];
            let mut col_i = col.slice_mut(ndarray::s![..di]).to_owned();
            for a in 0..k {
                if sys.htbeta_matvec.is_some() || row.htbeta.dim() != (di, k) {
                    // Kronecker / matrix-free path: probe column a.
                    e_a.fill(0.0);
                    e_a[a] = 1.0;
                    col_i.fill(0.0);
                    sys_htbeta_apply_row(sys, i, row, e_a.view(), &mut col_i);
                } else {
                    for c in 0..di {
                        col_i[c] = row.htbeta[[c, a]];
                    }
                }
                let solved = backend.solve_block_vector(&htt_factors[i], &col_i);
                let mut acc = 0.0;
                for c in 0..di {
                    acc += col_i[c] * solved[c];
                }
                diag[a] -= acc;
            }
        }
        let mut blocks = Vec::with_capacity(k);
        for a in 0..k {
            let v = diag[a];
            if !v.is_finite() || v <= 1e-18 {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "invalid Schur Jacobi diagonal at index {a}: {v}; \
                         operator regularization is required"
                    ),
                });
            }
            blocks.push(BlockFactor::Scalar {
                inv: Array1::from_elem(1, 1.0 / v),
                range: a..a + 1,
            });
        }
        Ok(Self { blocks })
    }

    /// Build term-block Jacobi: one dense `b×b` Schur block per term in
    /// `sys.block_offsets`.
    fn build_block_jacobi<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;
        let mut blocks = Vec::with_capacity(block_offsets.len());

        for range in block_offsets.iter() {
            let b = range.end - range.start;
            // Initialise the b×b Schur sub-block from H_ββ + ridge·I.
            let mut schur_block = Array2::<f64>::zeros((b, b));
            for bi in 0..b {
                for bj in 0..b {
                    let gi = range.start + bi;
                    let gj = range.start + bj;
                    let base = if sys.hbb.dim() == (sys.k, sys.k) {
                        sys.hbb[[gi, gj]]
                    } else {
                        // matrix-free path: only diagonal available
                        if bi == bj {
                            sys.hbb_diag.as_ref().map(|hd| hd[gi]).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    };
                    schur_block[[bi, bj]] = base + if bi == bj { ridge_beta } else { 0.0 };
                }
            }
            // Subtract Schur contributions:
            // S_kk -= H_βt_k^(i) (H_tt^(i))^{-1} H_tβ_k^(i)
            //
            // Materialize the per-row (di, k) cross-block once and slice out
            // the b-column submatrix.  sys_htbeta_materialize_row handles the
            // Kronecker / htbeta_matvec path transparently.
            for (i, row) in sys.rows.iter().enumerate() {
                let di = sys.row_dims[i];
                let htbeta_full = sys_htbeta_materialize_row(sys, i, row);
                let mut solved_cols = Array2::<f64>::zeros((di, b));
                for bj in 0..b {
                    let gj = range.start + bj;
                    let solved = backend.solve_block_vector(
                        &htt_factors[i],
                        &htbeta_full.column(gj).to_owned(),
                    );
                    for c in 0..di {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                for bi in 0..b {
                    let gi = range.start + bi;
                    for bj in 0..b {
                        let mut acc = 0.0;
                        for c in 0..di {
                            acc += htbeta_full[[c, gi]] * solved_cols[[c, bj]];
                        }
                        schur_block[[bi, bj]] -= acc;
                    }
                }
            }
            // Attempt Cholesky (LLT) factorization.
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(&schur_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                blocks.push(BlockFactor::Chol {
                    factor: llt,
                    range: range.clone(),
                });
            } else {
                // Non-PD block: fall back to scalar diagonal for this block.
                let mut inv = Array1::<f64>::zeros(b);
                for bi in 0..b {
                    let v = schur_block[[bi, bi]];
                    if !v.is_finite() || v <= 1e-18 {
                        return Err(ArrowSchurError::PcgFailed {
                            reason: format!(
                                "block Jacobi scalar fallback: non-PD diagonal at \
                                 global index {}: {v}; regularization required",
                                range.start + bi
                            ),
                        });
                    }
                    inv[bi] = 1.0 / v;
                }
                blocks.push(BlockFactor::Scalar {
                    inv,
                    range: range.clone(),
                });
            }
        }
        Ok(Self { blocks })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for block in &self.blocks {
            match block {
                BlockFactor::Scalar { inv, range } => {
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = inv[local] * r[gi];
                    }
                }
                BlockFactor::Chol { factor, range } => {
                    let b = range.end - range.start;
                    let mut rhs = Array1::<f64>::zeros(b);
                    for (local, gi) in range.clone().enumerate() {
                        rhs[local] = r[gi];
                    }
                    use faer::linalg::solvers::Solve;
                    let stride = rhs.strides()[0];
                    let len = rhs.len();
                    // SAFETY: rhs is a uniquely-borrowed contiguous Array1
                    // with positive stride (standard layout).
                    let rhs_mat = unsafe {
                        faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0)
                    };
                    let solved = factor.solve(rhs_mat);
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = solved[(local, 0)];
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Preconditioner ladder: SchurPreconditionerKind, ClusterJacobi,
// AdditiveSchwarz  (issue #299)
// ---------------------------------------------------------------------------

/// Which Schur preconditioner to use in the inexact-PCG path.
///
/// Ladder ordered by cost / effectiveness:
/// - `Diagonal`: scalar Jacobi (pre-#283 behaviour).
/// - `BetaBlockJacobi`: block-Jacobi per `block_offsets` term (#287).
/// - `ClusterJacobi`: one dense block per beta-graph connected component.
/// - `AdditiveSchwarz { overlap }`: component + `overlap`-hop expansion,
///   overlapping columns averaged by partition-of-unity weights.
///
/// ```text
/// Future variants (not yet wired, see #299):
///   DiagAssembledSchwarz { overlap: usize },
///   SparseIncompleteCholesky,
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurPreconditionerKind {
    Diagonal,
    BetaBlockJacobi,
    ClusterJacobi,
    AdditiveSchwarz { overlap: usize },
}

/// Escalate beyond BetaBlockJacobi only when K exceeds this value and PCG
/// exhausted `max_iterations`.
const PRECOND_ESCALATE_K_THRESHOLD: usize = 100;

/// Cholesky or scalar factor for one cluster of the beta-coefficient graph.
#[derive(Clone)]
enum ClusterFactor {
    Chol { cols: Vec<usize>, factor: FaerLlt<f64> },
    Scalar { cols: Vec<usize>, inv: Vec<f64> },
}

impl std::fmt::Debug for ClusterFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterFactor::Chol { cols, .. } => {
                write!(f, "ClusterFactor::Chol {{ cols.len: {} }}", cols.len())
            }
            ClusterFactor::Scalar { cols, inv } => write!(
                f,
                "ClusterFactor::Scalar {{ cols.len: {}, inv.len: {} }}",
                cols.len(),
                inv.len()
            ),
        }
    }
}

/// Maximum columns per cluster before scalar fallback.
const CLUSTER_JACOBI_MAX_CLUSTER: usize = 512;

/// Dense Schur block per connected component of the beta-coupling graph.
///
/// Nodes = beta blocks (`block_offsets`); edges = rows where two blocks
/// co-occur with nonzero `H_t_beta` entries. One Cholesky factor per
/// connected component; applied as a triangular solve.
#[derive(Debug, Clone)]
pub struct ClusterJacobiPreconditioner {
    clusters: Vec<ClusterFactor>,
}

impl ClusterJacobiPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            return Self::build_from_column_groups(
                sys, htt_factors, ridge_beta, backend, &[cols],
            );
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows.iter().map(|r| r.htbeta.clone()).collect::<Vec<_>>(),
        );
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|comp_blocks| {
                let mut cols: Vec<usize> = comp_blocks
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols
            })
            .collect();
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    fn build_from_column_groups<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
        let d = sys.d;
        let mut clusters = Vec::with_capacity(col_groups.len());
        for cols in col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            if b > CLUSTER_JACOBI_MAX_CLUSTER {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar { cols: cols.clone(), inv });
                continue;
            }
            let mut s_block = Array2::<f64>::zeros((b, b));
            for bi in 0..b {
                for bj in 0..b {
                    let gi = cols[bi];
                    let gj = cols[bj];
                    let base = if sys.hbb.dim() == (sys.k, sys.k) {
                        sys.hbb[[gi, gj]]
                    } else if bi == bj {
                        sys.hbb_diag.as_ref().map(|hd| hd[gi]).unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    s_block[[bi, bj]] = base + if bi == bj { ridge_beta } else { 0.0 };
                }
            }
            let mut col_vec = Array1::<f64>::zeros(d);
            let mut solved_cols = Array2::<f64>::zeros((d, b));
            for (row_idx, row) in sys.rows.iter().enumerate() {
                for bj in 0..b {
                    let gj = cols[bj];
                    for c in 0..d {
                        col_vec[c] = row.htbeta[[c, gj]];
                    }
                    let solved = backend.solve_block_vector(&htt_factors[row_idx], &col_vec);
                    for c in 0..d {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                for bi in 0..b {
                    let gi = cols[bi];
                    for bj in 0..b {
                        let mut acc = 0.0;
                        for c in 0..d {
                            acc += row.htbeta[[c, gi]] * solved_cols[[c, bj]];
                        }
                        s_block[[bi, bj]] -= acc;
                    }
                }
            }
            symmetrize_upper_from_lower(&mut s_block);
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(&s_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                clusters.push(ClusterFactor::Chol { cols: cols.clone(), factor: llt });
            } else {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar { cols: cols.clone(), inv });
            }
        }
        Ok(Self { clusters })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster_non_overlapping(cluster, r, &mut out);
        }
        out
    }
}

/// Additive Schwarz: base components expanded by `overlap` graph-hops;
/// overlapping columns averaged by partition-of-unity weights.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzPreconditioner {
    clusters: Vec<ClusterFactor>,
    weights: Vec<f64>,
}

impl AdditiveSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &[Array2<f64>],
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            let inner = ClusterJacobiPreconditioner::build_from_column_groups(
                sys, htt_factors, ridge_beta, backend, &[cols],
            )?;
            return Ok(Self { clusters: inner.clusters, weights: vec![1.0f64; sys.k] });
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows.iter().map(|r| r.htbeta.clone()).collect::<Vec<_>>(),
        );
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|seed| {
                let mut current = seed.clone();
                for _ in 0..overlap {
                    current = graph.expand_one_hop(&current);
                }
                let mut cols: Vec<usize> = current
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols.dedup();
                cols
            })
            .collect();
        let mut counts = vec![0u32; sys.k];
        for cols in &col_groups {
            for &gi in cols {
                counts[gi] += 1;
            }
        }
        let weights: Vec<f64> =
            counts.iter().map(|&c| if c == 0 { 1.0 } else { 1.0 / c as f64 }).collect();
        let inner = ClusterJacobiPreconditioner::build_from_column_groups(
            sys, htt_factors, ridge_beta, backend, &col_groups,
        )?;
        Ok(Self { clusters: inner.clusters, weights })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster_overlapping(cluster, r, &mut out, &self.weights);
        }
        out
    }
}

/// Apply a cluster factor (overwrite) for non-overlapping clusters.
fn apply_cluster_non_overlapping(
    cluster: &ClusterFactor,
    r: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    match cluster {
        ClusterFactor::Scalar { cols, inv } => {
            for (local, &gi) in cols.iter().enumerate() {
                out[gi] = inv[local] * r[gi];
            }
        }
        ClusterFactor::Chol { cols, factor } => {
            let b = cols.len();
            let mut rhs = Array1::<f64>::zeros(b);
            for (local, &gi) in cols.iter().enumerate() {
                rhs[local] = r[gi];
            }
            use faer::linalg::solvers::Solve;
            let stride = rhs.strides()[0];
            let len = rhs.len();
            // SAFETY: rhs is uniquely-borrowed contiguous Array1 with positive stride.
            let rhs_mat =
                unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
            let solved = factor.solve(rhs_mat);
            for (local, &gi) in cols.iter().enumerate() {
                out[gi] = solved[(local, 0)];
            }
        }
    }
}

/// Apply a cluster factor (accumulate with partition-of-unity weights)
/// for overlapping Schwarz clusters.
fn apply_cluster_overlapping(
    cluster: &ClusterFactor,
    r: &Array1<f64>,
    out: &mut Array1<f64>,
    weights: &[f64],
) {
    match cluster {
        ClusterFactor::Scalar { cols, inv } => {
            for (local, &gi) in cols.iter().enumerate() {
                out[gi] += weights[gi] * inv[local] * r[gi];
            }
        }
        ClusterFactor::Chol { cols, factor } => {
            let b = cols.len();
            let mut rhs = Array1::<f64>::zeros(b);
            for (local, &gi) in cols.iter().enumerate() {
                rhs[local] = r[gi];
            }
            use faer::linalg::solvers::Solve;
            let stride = rhs.strides()[0];
            let len = rhs.len();
            // SAFETY: rhs is uniquely-borrowed contiguous Array1 with positive stride.
            let rhs_mat =
                unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
            let solved = factor.solve(rhs_mat);
            for (local, &gi) in cols.iter().enumerate() {
                out[gi] += weights[gi] * solved[(local, 0)];
            }
        }
    }
}

/// Build scalar diagonal inverses for a set of global column indices.
///
/// Used when a cluster is non-PD or exceeds `CLUSTER_JACOBI_MAX_CLUSTER`.
fn build_schur_scalar_inv<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Result<Vec<f64>, ArrowSchurError> {
    let d = sys.d;
    let mut result = Vec::with_capacity(cols.len());
    let mut col_vec = Array1::<f64>::zeros(d);
    for &gi in cols {
        let base = match sys.hbb_diag.as_ref() {
            Some(hd) => hd[gi],
            None => {
                if sys.hbb.dim() == (sys.k, sys.k) {
                    sys.hbb[[gi, gi]]
                } else {
                    0.0
                }
            }
        };
        let mut s = base + ridge_beta;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            for c in 0..d {
                col_vec[c] = row.htbeta[[c, gi]];
            }
            let solved = backend.solve_block_vector(&htt_factors[row_idx], &col_vec);
            let mut acc = 0.0;
            for c in 0..d {
                acc += col_vec[c] * solved[c];
            }
            s -= acc;
        }
        if !s.is_finite() || s <= 1e-18 {
            return Err(ArrowSchurError::PcgFailed {
                reason: format!(
                    "cluster Schur scalar fallback: non-PD diagonal at index {gi}: {s}"
                ),
            });
        }
        result.push(1.0 / s);
    }
    Ok(result)
}

/// Inexact PCG with automatic preconditioner-ladder escalation.
///
/// Starts with `JacobiPreconditioner` (Diagonal or BetaBlockJacobi).
/// If PCG hits `MaxIter` and `k > PRECOND_ESCALATE_K_THRESHOLD`,
/// escalates to `ClusterJacobi`; if still `MaxIter`, escalates to
/// `AdditiveSchwarz { overlap: 1 }`.
fn steihaug_pcg_auto<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    rhs: &Array1<f64>,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    let jacobi =
        JacobiPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend)?;
    let (x0, diag0) = run_pcg_with_preconditioner(
        sys, htt_factors, ridge_beta, rhs, |r| jacobi.apply(r),
        pcg, trust, backend, gpu_matvec, metric_weights,
    )?;
    if sys.k <= PRECOND_ESCALATE_K_THRESHOLD
        || diag0.stopping_reason != PcgStopReason::MaxIter
    {
        return Ok((x0, diag0));
    }
    let cluster =
        ClusterJacobiPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend)?;
    let (x1, diag1) = run_pcg_with_preconditioner(
        sys, htt_factors, ridge_beta, rhs, |r| cluster.apply(r),
        pcg, trust, backend, gpu_matvec, metric_weights,
    )?;
    if diag1.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x1, diag1));
    }
    let schwarz = AdditiveSchwarzPreconditioner::from_arrow_schur(
        sys, htt_factors, ridge_beta, backend, 1,
    )?;
    run_pcg_with_preconditioner(
        sys, htt_factors, ridge_beta, rhs, |r| schwarz.apply(r),
        pcg, trust, backend, gpu_matvec, metric_weights,
    )
}

/// Run Steihaug-CG with a generic preconditioner closure.
/// Routes matvec through GPU when `gpu_matvec` is set.
fn run_pcg_with_preconditioner<ApplyPrec, B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    rhs: &Array1<f64>,
    apply_prec: ApplyPrec,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError>
where
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let max_iters = pcg.max_iterations.min(trust.max_iterations);
    let tol = pcg.relative_tolerance.max(trust.steihaug_relative_tolerance);
    if let Some(gpu_mv) = gpu_matvec {
        let gpu_mv = Arc::clone(gpu_mv);
        steihaug_cg(
            rhs,
            move |p, out| gpu_mv(p, out),
            apply_prec,
            max_iters,
            tol,
            trust.radius,
            metric_weights,
        )
    } else {
        steihaug_cg(
            rhs,
            |p, out| schur_matvec(sys, htt_factors, ridge_beta, p, out, backend),
            apply_prec,
            max_iters,
            tol,
            trust.radius,
            metric_weights,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct IdentityPreconditioner;

impl IdentityPreconditioner {
    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        r.clone()
    }
}

/// Steihaug-CG on the reduced BA Schur system.
///
/// When `gpu_matvec` is `Some`, each iteration delegates the full `S·p` product
/// to the GPU-backed closure instead of the CPU `schur_matvec`; this is the
/// Part-B (issue #288) hook that lets the CPU PCG outer loop call into a GPU
/// kernel for the dominant `Σ_i Y_i^T Y_i · p` accumulation at K ≥ 5000.
/// The closure is responsible for the complete `(H_ββ + ρ·I − Σ Y_i^T Y_i)·p`
/// product; constructors in `crate::gpu::arrow_schur` pre-compute the `Y_i`
/// factors on device and add the CPU-side `H_ββ` term inside the closure.
fn steihaug_pcg_reduced_system<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &[Array2<f64>],
    ridge_beta: f64,
    rhs: &Array1<f64>,
    preconditioner: &JacobiPreconditioner,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    if let Some(gpu_mv) = gpu_matvec {
        let gpu_mv = Arc::clone(gpu_mv);
        steihaug_cg(
            rhs,
            move |p, out| gpu_mv(p, out),
            |r| preconditioner.apply(r),
            pcg.max_iterations.min(trust.max_iterations),
            pcg.relative_tolerance
                .max(trust.steihaug_relative_tolerance),
            trust.radius,
            metric_weights,
        )
    } else {
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
}

fn steihaug_dense_system(
    schur: &Array2<f64>,
    rhs: &Array1<f64>,
    preconditioner: &IdentityPreconditioner,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
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
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError>
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
        return Ok((Array1::<f64>::zeros(n), PcgDiagnostics::default()));
    }
    let tol = relative_tolerance.max(0.0) * rhs_norm;
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = apply_preconditioner(&r);
    let mut diag = PcgDiagnostics {
        precond_apply_calls: 1,
        ..PcgDiagnostics::default()
    };
    let mut p = z.clone();
    let mut rz = metric_dot(&r, &z, metric_weights);
    if rz <= 0.0 || !rz.is_finite() {
        if radius.is_finite() {
            diag.final_relative_residual =
                metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &r, radius, metric_weights), diag));
        }
        return Err(ArrowSchurError::PcgFailed {
            reason: "non-positive preconditioned residual in Schur PCG".to_string(),
        });
    }
    if metric_norm(r.view(), metric_weights) <= tol {
        diag.final_relative_residual = 0.0;
        diag.stopping_reason = PcgStopReason::Converged;
        return Ok((x, diag));
    }
    let mut ap = Array1::<f64>::zeros(n);
    // Reused candidate scratch — avoid per-iteration clone of x.
    let mut candidate = Array1::<f64>::zeros(n);
    for _ in 0..max_iterations {
        matvec(&p, &mut ap);
        diag.matvec_calls += 1;
        diag.iterations += 1;
        let pap = metric_dot(&p, &ap, metric_weights);
        if pap <= 0.0 || !pap.is_finite() {
            if radius.is_finite() {
                diag.final_relative_residual =
                    metric_norm(r.view(), metric_weights) / rhs_norm;
                diag.stopping_reason = PcgStopReason::TrustRegion;
                return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
            }
            diag.final_relative_residual =
                metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Indefinite;
            return Err(ArrowSchurError::PcgFailed {
                reason: "negative curvature in unbounded Schur PCG".to_string(),
            });
        }
        let alpha = rz / pap;
        for i in 0..n {
            candidate[i] = x[i] + alpha * p[i];
        }
        if radius.is_finite() && metric_norm(candidate.view(), metric_weights) >= radius {
            diag.final_relative_residual =
                metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
        }
        x.assign(&candidate);
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        if metric_norm(r.view(), metric_weights) <= tol {
            diag.final_relative_residual =
                metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Converged;
            return Ok((x, diag));
        }
        z = apply_preconditioner(&r);
        diag.precond_apply_calls += 1;
        let rz_next = metric_dot(&r, &z, metric_weights);
        if rz_next <= 0.0 || !rz_next.is_finite() {
            diag.final_relative_residual =
                metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Stagnation;
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
    diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
    diag.stopping_reason = PcgStopReason::MaxIter;
    Ok((x, diag))
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
    assert_eq!(a.len(), b.len());
    match metric_weights {
        Some(weights) => {
            assert_eq!(weights.len(), a.len());
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
            assert_eq!(weights.len(), v.len());
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
    /// A per-row `H_tt^(i)` block factored, but the Cholesky factor's
    /// diagonal-ratio condition-number estimate exceeded the safe
    /// threshold for the Schur reduction. Cholesky technically
    /// succeeded, but the inverse used in
    /// `S = H_ββ − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)` is contaminated
    /// by spectral terms on the order of `κ_i`; functionally
    /// equivalent to a PSD-fail for Schur stability. The LM outer
    /// wrapper escalates `ridge_t` identically to `PerRowFactorFailed`.
    PerRowFactorIllConditioned { row: usize, kappa_estimate: f64 },
    /// The Schur complement was not positive-definite. Indicates a
    /// near-collinear decoder or a degenerate weighting; the LM outer
    /// wrapper should escalate `ridge_beta` and retry.
    SchurFactorFailed { reason: String },
    /// The BA inexact-step PCG solve failed before producing a usable
    /// Steihaug trust-region step.
    PcgFailed { reason: String },
    /// Adaptive proximal damping could not produce an Armijo-accepted
    /// nonlinear step.
    AdaptiveCorrectionFailed { reason: String },
}

impl std::fmt::Display for ArrowSchurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowSchurError::PerRowFactorFailed { row, reason } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky failed: {reason}"
            ),
            ArrowSchurError::PerRowFactorIllConditioned {
                row,
                kappa_estimate,
            } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky succeeded but is \
                 ill-conditioned (kappa_estimate={kappa_estimate:e}); Schur \
                 reduction would be numerically contaminated"
            ),
            ArrowSchurError::SchurFactorFailed { reason } => {
                write!(f, "arrow-Schur: Schur complement Cholesky failed: {reason}")
            }
            ArrowSchurError::PcgFailed { reason } => {
                write!(f, "arrow-Schur: Schur PCG failed: {reason}")
            }
            ArrowSchurError::AdaptiveCorrectionFailed { reason } => {
                write!(
                    f,
                    "arrow-Schur: adaptive proximal correction failed: {reason}"
                )
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
    if let Some((idx, _)) = a.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "cholesky_lower: non-finite entry at linear index {idx}"
        ));
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
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
        let streaming_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(1));
        let (delta_t_stream, delta_beta_stream) = sys
            .solve_with_options(0.0, 0.0, &streaming_options)
            .expect("streaming arrow-schur solve");
        assert_eq!(delta_beta, delta_beta_stream);
        assert_eq!(delta_t, delta_t_stream);

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

    fn quartic_counterexample_value(t: f64) -> f64 {
        0.25 * t.powi(4) - t * t + 2.0 * t
    }

    fn quartic_counterexample_system(t: f64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(1, 1, 0);
        sys.rows[0].gt = array![t.powi(3) - 2.0 * t + 2.0];
        sys.rows[0].htt = array![[3.0 * t * t - 2.0]];
        sys
    }

    #[test]
    fn proximal_correction_breaks_scalar_newton_cycle() {
        let options = ArrowSolveOptions::direct();
        let correction = ArrowProximalCorrectionOptions {
            initial_ridge: 1e-8,
            ridge_growth: 10.0,
            max_attempts: 16,
            armijo_c1: 1e-4,
            gradient_tolerance: 1e-12,
        };
        let mut t = 0.0_f64;
        let mut previous_value = quartic_counterexample_value(t);

        for _ in 0..32 {
            let sys = quartic_counterexample_system(t);
            let accepted = solve_arrow_newton_step_with_proximal_correction(
                &sys,
                0.0,
                0.0,
                previous_value,
                &options,
                &correction,
                |delta_t, _delta_beta| quartic_counterexample_value(t + delta_t[0]),
            )
            .expect("proximal correction should accept a descent step");
            assert!(
                accepted.trial_objective_value <= previous_value,
                "accepted step must not increase the objective"
            );
            t += accepted.delta_t[0];
            previous_value = accepted.trial_objective_value;
        }

        let final_grad = t.powi(3) - 2.0 * t + 2.0;
        assert!(
            final_grad.abs() < 1e-7,
            "corrected iteration should reach the scalar critical point; t={t}, g={final_grad}"
        );
    }

    /// Issue #195: a per-row block that is barely-PD (smallest pivot on
    /// the order of ε·trace) factors successfully but is unsafe to use in
    /// the Schur reduction. `factor_one_row` must detect this via the
    /// diagonal-ratio condition estimate and surface
    /// `PerRowFactorIllConditioned` rather than silently contaminating
    /// `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`.
    #[test]
    fn factor_one_row_rejects_barely_pd_block() {
        let d = 2;
        let k = 2;
        let mut row = ArrowRowBlock::new(d, k);
        // Matrix from the issue body: PD by an exact ε along the second
        // direction. Cholesky succeeds, but κ ≈ 1e14.
        row.htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
        row.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row.gt = array![0.0_f64, 0.0];

        let err = factor_one_row(&row, 0.0, d, 0)
            .expect_err("barely-PD H_tt must be rejected by the condition check");
        match err {
            ArrowSchurError::PerRowFactorIllConditioned {
                row: r,
                kappa_estimate,
            } => {
                assert_eq!(r, 0);
                assert!(
                    kappa_estimate > 1e10,
                    "kappa estimate should reflect the barely-PD block; got {kappa_estimate:e}"
                );
            }
            other => panic!("expected PerRowFactorIllConditioned, got {other:?}"),
        }

        // Sanity: a well-conditioned block at the same dimension still
        // factors successfully.
        let mut row_ok = ArrowRowBlock::new(d, k);
        row_ok.htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        row_ok.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_ok.gt = array![0.0_f64, 0.0];
        factor_one_row(&row_ok, 0.0, d, 0)
            .expect("well-conditioned block must still factor at ridge_t=0");
    }

    /// Issue #195 follow-up: when the per-row block is barely-PD at
    /// `ridge_t = 0`, `solve_with_lm_escalation_inner` must escalate
    /// `ridge_t` and produce a successful solve at a higher ridge.
    #[test]
    fn lm_escalation_recovers_from_ill_conditioned_row() {
        let n = 1;
        let d = 2;
        let k = 2;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        // Same barely-PD row as the issue body.
        sys.rows[0].htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        sys.rows[0].gt = array![0.1_f64, -0.2];
        sys.hbb = array![[4.0_f64, 0.2], [0.2, 5.0]];
        sys.gb = array![0.3_f64, -0.1];

        // Direct factor at ridge_t=0 must report ill-conditioning.
        let direct = factor_one_row(&sys.rows[0], 0.0, d, 0);
        assert!(matches!(
            direct,
            Err(ArrowSchurError::PerRowFactorIllConditioned { .. })
        ));

        // But the LM-escalating wrapper must recover by lifting ridge_t.
        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta) = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
            .expect("LM escalation must recover from PerRowFactorIllConditioned");
        for v in delta_t.iter().chain(delta_beta.iter()) {
            assert!(v.is_finite(), "recovered step must be finite: {v}");
        }
    }
}
