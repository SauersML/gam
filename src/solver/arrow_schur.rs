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
//! The cost is arrow-shaped, but the REML log|H| gradient carries a shared
//! Schur⁻¹ factor handled as one-time-per-outer-iteration setup plus N
//! rank-≤d per-row traces; that is the source of the explicit precondition
//! story below.
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
//! hook, re-read the inner/outer cost split documented above first.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::ops::Range;
use std::sync::Arc;

use crate::cache::Fingerprinter;
use crate::linalg::faer_ndarray::{FaerArrayView, FaerLlt};
use crate::linalg::triangular::{
    cholesky_solve_matrix, cholesky_solve_vector, forward_substitution_lower_matrix,
};
use crate::terms::analytic_penalties::{AnalyticPenaltyKind, AnalyticPenaltyRegistry, PenaltyTier};
use crate::terms::latent_coord::{LatentCoordValues, LatentManifold};

const DIRECT_SOLVE_MAX_K: usize = 2_000;
const DEFAULT_PCG_MAX_ITERATIONS: usize = 200;
const DEFAULT_PCG_RELATIVE_TOLERANCE: f64 = 1e-4;
/// Absolute floor on the Steihaug-CG residual stopping threshold.
///
/// The native PCG criterion is purely relative: `tol = rel_tol · ‖rhs‖`. When
/// `‖rhs‖` is tiny (degenerate / near-stationary reduced systems) this product
/// can fall below the roundoff resolution of `metric_norm` (~1e-15 for f64),
/// so the loop would "converge" on floating-point noise rather than a genuinely
/// accurate solution. Floor the threshold at 1e-14: above machine epsilon
/// (~2.2e-16) yet below any practical single-iteration residual reduction, so
/// well-scaled problems are unaffected while degenerate ones stop cleanly.
const PCG_ABSOLUTE_TOLERANCE_FLOOR: f64 = 1e-14;
const DEFAULT_TRUST_REGION_RADIUS: f64 = f64::INFINITY;
pub const DEFAULT_PROXIMAL_INITIAL_RIDGE: f64 = 1e-8;
const F32_UNIT_ROUNDOFF: f64 = (f32::EPSILON as f64) * 0.5;
const DEFAULT_MIXED_PRECISION_MAX_REFINEMENTS: usize = 6;
const DEFAULT_MIXED_PRECISION_CERTIFICATE_TOLERANCE: f64 = 1e-11;
const DEFAULT_MIXED_PRECISION_KAPPA_MARGIN: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BetaEdge {
    a: usize,
    b: usize,
}

#[derive(Debug, Clone)]
struct BetaCouplingGraph {
    num_blocks: usize,
    edges: Vec<BetaEdge>,
    adj_start: Vec<usize>,
    adj_targets: Vec<usize>,
}

impl BetaCouplingGraph {
    fn build(block_offsets: &[Range<usize>], htbeta_rows: &[Array2<f64>]) -> Self {
        let num_blocks = block_offsets.len();
        if num_blocks == 0 {
            return Self {
                num_blocks: 0,
                edges: Vec::new(),
                adj_start: vec![0],
                adj_targets: Vec::new(),
            };
        }

        let mut edge_set = Vec::<(usize, usize)>::new();
        for row in htbeta_rows {
            let mut active = Vec::<usize>::new();
            for (block, range) in block_offsets.iter().enumerate() {
                if range
                    .clone()
                    .any(|col| (0..row.nrows()).any(|axis| row[[axis, col]] != 0.0))
                {
                    active.push(block);
                }
            }
            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    edge_set.push((active[i].min(active[j]), active[i].max(active[j])));
                }
            }
        }
        edge_set.sort_unstable();
        edge_set.dedup();

        let edges: Vec<_> = edge_set.iter().map(|&(a, b)| BetaEdge { a, b }).collect();
        let mut degree = vec![0usize; num_blocks];
        for &BetaEdge { a, b } in &edges {
            degree[a] += 1;
            degree[b] += 1;
        }
        let mut adj_start = vec![0usize; num_blocks + 1];
        for block in 0..num_blocks {
            adj_start[block + 1] = adj_start[block] + degree[block];
        }
        let mut adj_targets = vec![0usize; adj_start[num_blocks]];
        let mut cursor = adj_start[..num_blocks].to_vec();
        for &BetaEdge { a, b } in &edges {
            adj_targets[cursor[a]] = b;
            cursor[a] += 1;
            adj_targets[cursor[b]] = a;
            cursor[b] += 1;
        }
        Self {
            num_blocks,
            edges,
            adj_start,
            adj_targets,
        }
    }

    fn neighbours(&self, node: usize) -> &[usize] {
        &self.adj_targets[self.adj_start[node]..self.adj_start[node + 1]]
    }

    fn component_partition(&self) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..self.num_blocks).collect();
        let mut rank = vec![0u8; self.num_blocks];

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for &BetaEdge { a, b } in &self.edges {
            let lhs = find(&mut parent, a);
            let rhs = find(&mut parent, b);
            if lhs != rhs {
                if rank[lhs] < rank[rhs] {
                    parent[lhs] = rhs;
                } else if rank[lhs] > rank[rhs] {
                    parent[rhs] = lhs;
                } else {
                    parent[rhs] = lhs;
                    rank[lhs] += 1;
                }
            }
        }

        let mut label_map = vec![usize::MAX; self.num_blocks];
        let mut parts = Vec::<Vec<usize>>::new();
        for block in 0..self.num_blocks {
            let root = find(&mut parent, block);
            let label = if label_map[root] == usize::MAX {
                label_map[root] = parts.len();
                parts.push(Vec::new());
                label_map[root]
            } else {
                label_map[root]
            };
            parts[label].push(block);
        }
        parts
    }

    fn expand_one_hop(&self, seed: &[usize]) -> Vec<usize> {
        let mut expanded = seed.to_vec();
        for &block in seed {
            expanded.extend_from_slice(self.neighbours(block));
        }
        expanded.sort_unstable();
        expanded.dedup();
        expanded
    }
}
pub const DEFAULT_PROXIMAL_RIDGE_GROWTH: f64 = 10.0;
/// Number of geometric proximal-ridge escalations the adaptive correction
/// attempts before giving up. Raised from 16 to 22 so the ridge can climb from
/// `1e-8` to `~1e14` (`1e-8 · 10^21`): when the penalised Hessian curvature
/// along the gradient exceeds `~1e9`, the damped Newton step at ridge `1e9`
/// still overshoots, and the extra decades let the step length collapse far
/// enough to either find descent or reach the near-stationary resolution floor
/// that triggers the convergence exit. The cost of the extra attempts is paid
/// only on configs that would otherwise have failed.
pub const DEFAULT_PROXIMAL_MAX_ATTEMPTS: usize = 22;
const DEFAULT_ARMIJO_C1: f64 = 1e-4;
const DEFAULT_GRADIENT_TOLERANCE: f64 = 1e-10;
/// Relative objective resolution for the proximal-correction convergence exit.
///
/// When the best achievable change in the penalised objective across all ridge
/// attempts is within `rel_tol · (|f| + 1)` of the incumbent value, the damped
/// Newton model has reached the floating-point resolution of the objective and
/// no further productive decrease exists. `8e-12` sits a few decades above the
/// `~2.2e-16` f64 epsilon (so genuine reductions of a well-scaled objective are
/// never swallowed) yet comfortably above the accumulated rounding of the
/// `O(N·M·p)` reductions that form the objective, so a truly stationary state
/// is recognised rather than chased into a spurious failure.
const DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL: f64 = 8e-12;
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
/// Row-local matrix-free transpose multiply `out += H_βt^(i) · v` (length `K`).
///
/// This is the adjoint of [`RowHtbetaMatvec`]: it scatters a per-row latent
/// vector `v` (length `d_i`) back into the shared β gradient, **adding** its
/// contribution to `out`. For the SAE Kronecker form this is the sparse
/// `scatter_jbeta_t` over the row's active atoms — `O(m_i · p)` per row, the
/// per-row sparse apply that replaces the `O(K)` column-probe in the GPU and
/// streaming Schur matvec.
pub type RowHtbetaTransposeMatvec =
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
    /// Mix the operator's defining state into `hasher` for cache-validity
    /// fingerprinting. Must change whenever `matvec` / `to_dense` would change,
    /// so the factorization / evidence cache (`cache_matches_system`) is
    /// invalidated when the β-block content changes. Implementations hash their
    /// own compact defining data (e.g. Kronecker factors, block matrices)
    /// rather than the full `K×K` dense form, which would defeat the structured
    /// operator's storage savings.
    fn fingerprint(&self, hasher: &mut Fingerprinter);
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

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("dense-penalty-op-v1");
        hasher.write_f64_array2(&self.0);
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

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("block-penalty-op-v1");
        hasher.write_usize(self.k);
        hasher.write_usize(self.blocks.len());
        for (off, local) in &self.blocks {
            hasher.write_usize(*off);
            hasher.write_f64_array2(local);
        }
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

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("kronecker-penalty-op-v1");
        hasher.write_usize(self.global_offset);
        hasher.write_usize(self.k);
        hasher.write_f64_array2(&self.factor_a);
        hasher.write_f64_array2(&self.factor_b);
    }
}

/// Kronecker-product penalty with an identity right factor:
/// `P = A ⊗ I_p`.
///
/// This is the hot SAE smoothness case. Storing `I_p` as a dense matrix costs
/// `O(p²)` memory per atom and makes every matvec pay an unnecessary right-factor
/// loop. This operator stores only the identity dimension and keeps the same
/// layout as [`KroneckerPenaltyOp`]: local index `i_a * p + i_b`.
pub struct IdentityRightKroneckerPenaltyOp {
    /// Left factor `A`, shape `(p_a, p_a)`.
    pub factor_a: Array2<f64>,
    /// Identity right-factor dimension `p`.
    pub p: usize,
    /// Global offset into the β vector where this block starts.
    pub global_offset: usize,
    /// Full β dimension `K`.
    pub k: usize,
}

impl BetaPenaltyOp for IdentityRightKroneckerPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let p_a = self.factor_a.nrows();
        let p = self.p;
        let off = self.global_offset;
        for i_a in 0..p_a {
            for i_b in 0..p {
                let gi = off + i_a * p + i_b;
                let mut acc = 0.0_f64;
                for j_a in 0..p_a {
                    let a_ij = self.factor_a[[i_a, j_a]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    acc += a_ij * x[off + j_a * p + i_b];
                }
                y[gi] += acc;
            }
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        let p_a = self.factor_a.nrows();
        let p = self.p;
        let off = self.global_offset;
        for i_a in 0..p_a {
            let a_ii = self.factor_a[[i_a, i_a]];
            for i_b in 0..p {
                diag[off + i_a * p + i_b] += a_ii;
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        let range = &offsets[id.0];
        let b = range.end - range.start;
        let p_a = self.factor_a.nrows();
        let p = self.p;
        let off = self.global_offset;
        let block_end = off + p_a * p;
        if block_end <= range.start || off >= range.end {
            return;
        }
        for bi in 0..b {
            let gi = range.start + bi;
            if gi < off || gi >= block_end {
                continue;
            }
            let li = gi - off;
            let i_a = li / p;
            let i_b = li % p;
            for bj in 0..b {
                let gj = range.start + bj;
                if gj < off || gj >= block_end {
                    continue;
                }
                let lj = gj - off;
                let j_a = lj / p;
                let j_b = lj % p;
                if i_b == j_b {
                    out[[bi, bj]] += self.factor_a[[i_a, j_a]];
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let p_a = self.factor_a.nrows();
        let p = self.p;
        let off = self.global_offset;
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for i_a in 0..p_a {
            for j_a in 0..p_a {
                let a_ij = self.factor_a[[i_a, j_a]];
                if a_ij == 0.0 {
                    continue;
                }
                for i_b in 0..p {
                    let gi = off + i_a * p + i_b;
                    let gj = off + j_a * p + i_b;
                    out[[gi, gj]] += a_ij;
                }
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("identity-right-kronecker-penalty-op-v1");
        hasher.write_usize(self.global_offset);
        hasher.write_usize(self.k);
        hasher.write_usize(self.p);
        hasher.write_f64_array2(&self.factor_a);
    }
}

/// One co-occurring atom-pair block of a block-sparse left factor `A`.
///
/// `data` is the dense `(m_i × m_j)` coupling between the basis columns of
/// atom `i` (rows, starting at left-factor offset `row_off`) and atom `j`
/// (columns, starting at `col_off`). Both offsets are in *left-factor* (`A`)
/// coordinates, i.e. `μ`-space, not β-space.
#[derive(Debug, Clone)]
pub struct SparseGBlock {
    /// Left-factor (`μ`-space) row offset = `beta_offset[atom_i] / p`.
    pub row_off: usize,
    /// Left-factor (`μ`-space) column offset = `beta_offset[atom_j] / p`.
    pub col_off: usize,
    /// Dense `(m_i × m_j)` coupling block.
    pub data: Array2<f64>,
}

/// Block-sparse Kronecker penalty `P = A ⊗ I_p` where the left factor `A`
/// (dimension `dim_a × dim_a` in `μ`-space) is stored only on its non-empty
/// co-occurring atom-pair blocks rather than as a dense `(dim_a × dim_a)`
/// matrix.
///
/// This is the sparse-atom (`K = 100K`) replacement for wrapping the dense
/// data-fit Gauss-Newton Gram `G` (`m_total × m_total`) in a
/// [`KroneckerPenaltyOp`]: with per-row active sets of size `k_active ≪ K`,
/// only the `(atom, atom')` pairs that co-occur in some row contribute a
/// non-zero `(m_i × m_j)` block, so the storage and every matvec/diagonal
/// pass cost `O(Σ_pairs m_i m_j · p)` instead of `O((m_total · p)²)`.
///
/// The β index of left-factor coordinate `μ` and output channel `oc` is
/// `μ · p + oc` (the same `μ`-major / `oc`-minor layout the dense
/// `KroneckerPenaltyOp { factor_b: I_p }` uses), so this op is a drop-in
/// structured replacement: with the full dense pair set it reproduces the
/// dense operator exactly.
pub struct SparseBlockKroneckerPenaltyOp {
    /// Right-factor identity dimension `p` (number of decoder output channels).
    pub p: usize,
    /// Left-factor dimension `dim_a` in `μ`-space (= `m_total`).
    pub dim_a: usize,
    /// Full β dimension `K = dim_a · p`.
    pub k: usize,
    /// Non-empty `(atom_i, atom_j)` coupling blocks of `A`.
    pub blocks: Vec<SparseGBlock>,
}

impl BetaPenaltyOp for SparseBlockKroneckerPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let p = self.p;
        for blk in &self.blocks {
            let (m_i, m_j) = blk.data.dim();
            for li in 0..m_i {
                let gi_base = (blk.row_off + li) * p;
                for lj in 0..m_j {
                    let a_ij = blk.data[[li, lj]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    let gj_base = (blk.col_off + lj) * p;
                    for oc in 0..p {
                        y[gi_base + oc] += a_ij * x[gj_base + oc];
                    }
                }
            }
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        let p = self.p;
        for blk in &self.blocks {
            // Only on-diagonal `A` blocks (row_off == col_off) carry diagonal
            // mass; their `(li, li)` entries map to `(row_off+li)·p + oc`.
            if blk.row_off != blk.col_off {
                continue;
            }
            let (m_i, m_j) = blk.data.dim();
            let m = m_i.min(m_j);
            for li in 0..m {
                let a_ii = blk.data[[li, li]];
                let gi_base = (blk.row_off + li) * p;
                for oc in 0..p {
                    diag[gi_base + oc] += a_ii;
                }
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        let range = &offsets[id.0];
        let b = range.end - range.start;
        let p = self.p;
        for blk in &self.blocks {
            let (m_i, m_j) = blk.data.dim();
            let row_start = blk.row_off * p;
            let row_end = (blk.row_off + m_i) * p;
            let col_start = blk.col_off * p;
            let col_end = (blk.col_off + m_j) * p;
            if row_end <= range.start
                || row_start >= range.end
                || col_end <= range.start
                || col_start >= range.end
            {
                continue;
            }
            for bi in 0..b {
                let gi = range.start + bi;
                if gi < row_start || gi >= row_end {
                    continue;
                }
                let li = (gi - row_start) / p;
                let oc_i = (gi - row_start) % p;
                for bj in 0..b {
                    let gj = range.start + bj;
                    if gj < col_start || gj >= col_end {
                        continue;
                    }
                    let oc_j = (gj - col_start) % p;
                    if oc_i != oc_j {
                        continue;
                    }
                    let lj = (gj - col_start) / p;
                    out[[bi, bj]] += blk.data[[li, lj]];
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.p;
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for blk in &self.blocks {
            let (m_i, m_j) = blk.data.dim();
            for li in 0..m_i {
                let gi_base = (blk.row_off + li) * p;
                for lj in 0..m_j {
                    let a_ij = blk.data[[li, lj]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    let gj_base = (blk.col_off + lj) * p;
                    for oc in 0..p {
                        out[[gi_base + oc, gj_base + oc]] += a_ij;
                    }
                }
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("sparse-block-kronecker-penalty-op-v1");
        hasher.write_usize(self.p);
        hasher.write_usize(self.dim_a);
        hasher.write_usize(self.k);
        hasher.write_usize(self.blocks.len());
        for blk in &self.blocks {
            hasher.write_usize(blk.row_off);
            hasher.write_usize(blk.col_off);
            hasher.write_f64_array2(&blk.data);
        }
    }
}

/// One co-occurring `(atom_i, atom_j)` block of the **frame-factored** data-fit
/// Gauss–Newton β-Hessian (issue #972 / #977 T1). Carries the basis-space Gram
/// `g` (`m_i × m_j`) AND the per-pair frame output factor `w = U_iᵀ U_j`
/// (`r_i × r_j`); the contributed Hessian sub-block is the Kronecker product
/// `g ⊗ w`.
#[derive(Debug, Clone)]
pub struct FactoredFrameGBlock {
    /// Atom index of the row factor (selects rank `r_i` and β offset).
    pub atom_i: usize,
    /// Atom index of the column factor (selects rank `r_j` and β offset).
    pub atom_j: usize,
    /// Basis-space coupling `G_{ij}` (`m_i × m_j`).
    pub g: Array2<f64>,
    /// Frame output factor `U_iᵀ U_j` (`r_i × r_j`). For `i == j` with an
    /// orthonormal frame this is `I_{r_i}` (the clean within-atom `g ⊗ I_r`
    /// collapse); across atoms it is the dense principal-angle cosine matrix
    /// between the two frames.
    pub w: Array2<f64>,
}

/// Frame-factored data-fit Gauss–Newton β-Hessian operator (#972 / #977 T1):
/// the `Σ_k M_k·r_k` reduced-border analogue of [`SparseBlockKroneckerPenaltyOp`].
///
/// When every atom's decoder `B_k = C_k U_kᵀ` is profiled onto a Grassmann
/// frame `U_k ∈ St(p, r_k)`, the border carries only the shape coefficients
/// `C_k` (`M_k · r_k` entries) instead of the full `B_k` (`M_k · p`). The data
/// Gram in this reduced space is, for the isotropic likelihood,
/// `H[(i,li,a),(j,lj,b)] = G_{ij}[li,lj] · (U_iᵀ U_j)[a,b]` — within an atom the
/// orthonormal frame gives `U_iᵀU_i = I_{r_i}` and the block is the clean
/// `G ⊗ I_r` collapse; across co-active atoms the frames do not share a basis
/// so the output factor is the dense `U_iᵀU_j`.
///
/// The β layout is `μ`-major / frame-minor with a **variable** per-atom width
/// `r_k`: the index of (atom `k`, basis `li`, frame coord `a`) is
/// `offset[k] + li·r_k + a`, where `offset` is the prefix sum of `M_k · r_k`.
/// With every `r_k = p` and `U_k = I_p` this reproduces
/// [`SparseBlockKroneckerPenaltyOp`] exactly (a unit test pins the reduction),
/// so it is a strict generalization, not a separate code path.
pub struct FactoredFrameKroneckerOp {
    /// Per-atom frame rank `r_k` (the factored output width).
    pub ranks: Vec<usize>,
    /// Per-atom basis size `M_k`.
    pub basis_sizes: Vec<usize>,
    /// Per-atom β offset (prefix sum of `M_k · r_k`); `offsets[k]` is the start
    /// of atom `k`'s `C_k` block, `offsets[n_atoms]` the total dim.
    pub offsets: Vec<usize>,
    /// Total reduced β dimension `Σ_k M_k · r_k`.
    pub dim: usize,
    /// Non-empty co-occurring `(atom_i, atom_j)` blocks.
    pub blocks: Vec<FactoredFrameGBlock>,
}

/// Frame output Gram `U_iᵀ U_j` (`r_i × r_j`) between two per-atom output
/// frames (each `p × r`). This is the dense principal-angle cosine matrix that
/// becomes the `w` factor of a [`FactoredFrameGBlock`]; for `i == j` with an
/// orthonormal frame it is `I_{r_i}`. Shared with `sae_manifold.rs`, which
/// builds the same factors when profiling decoders onto Grassmann frames.
pub fn frame_output_gram(u_i: ArrayView2<f64>, u_j: ArrayView2<f64>) -> Array2<f64> {
    let (p_i, r_i) = u_i.dim();
    let (p_j, r_j) = u_j.dim();
    assert_eq!(
        p_i, p_j,
        "frame_output_gram: frames live in different ambient dims ({p_i} vs {p_j})"
    );
    let mut w = Array2::<f64>::zeros((r_i, r_j));
    for a in 0..r_i {
        for b in 0..r_j {
            let mut acc = 0.0;
            for c in 0..p_i {
                acc += u_i[[c, a]] * u_j[[c, b]];
            }
            w[[a, b]] = acc;
        }
    }
    w
}

impl FactoredFrameKroneckerOp {
    /// Build from per-atom ranks + basis sizes and the co-occurring blocks.
    /// Computes the β offsets (prefix sum of `M_k·r_k`) and validates that each
    /// block's `g`/`w` shapes match the atoms' `(M, r)`.
    pub fn new(
        ranks: Vec<usize>,
        basis_sizes: Vec<usize>,
        blocks: Vec<FactoredFrameGBlock>,
    ) -> Result<Self, String> {
        if ranks.len() != basis_sizes.len() {
            return Err(format!(
                "FactoredFrameKroneckerOp: {} ranks but {} basis sizes",
                ranks.len(),
                basis_sizes.len()
            ));
        }
        let n_atoms = ranks.len();
        let mut offsets = Vec::with_capacity(n_atoms + 1);
        let mut acc = 0usize;
        for k in 0..n_atoms {
            offsets.push(acc);
            acc += basis_sizes[k] * ranks[k];
        }
        offsets.push(acc);
        let dim = acc;
        for blk in &blocks {
            if blk.atom_i >= n_atoms || blk.atom_j >= n_atoms {
                return Err(format!(
                    "FactoredFrameKroneckerOp: block atom indices ({}, {}) out of range (n_atoms = {n_atoms})",
                    blk.atom_i, blk.atom_j
                ));
            }
            if blk.g.dim() != (basis_sizes[blk.atom_i], basis_sizes[blk.atom_j]) {
                return Err(format!(
                    "FactoredFrameKroneckerOp: block ({}, {}) g has shape {:?} but expected ({}, {})",
                    blk.atom_i,
                    blk.atom_j,
                    blk.g.dim(),
                    basis_sizes[blk.atom_i],
                    basis_sizes[blk.atom_j]
                ));
            }
            if blk.w.dim() != (ranks[blk.atom_i], ranks[blk.atom_j]) {
                return Err(format!(
                    "FactoredFrameKroneckerOp: block ({}, {}) w has shape {:?} but expected ({}, {})",
                    blk.atom_i,
                    blk.atom_j,
                    blk.w.dim(),
                    ranks[blk.atom_i],
                    ranks[blk.atom_j]
                ));
            }
        }
        Ok(Self {
            ranks,
            basis_sizes,
            offsets,
            dim,
            blocks,
        })
    }

    /// Convenience constructor that builds the operator directly from per-atom
    /// output frames + the basis-space Gram block map, computing the per-pair
    /// frame factors `W_ij = U_iᵀ U_j` itself.
    ///
    /// `frames[k]` is either `Some(U_k)` — a `p × r_k` (`r_k ≤ p`) output frame
    /// (a Grassmann representative `St(p, r_k)` need not be orthonormal here; the
    /// `W` factor carries whatever frame is supplied) — or `None`, meaning atom
    /// `k` keeps the full ambient output (`U_k = I_p`, so `r_k = p`). For each
    /// non-empty Gram block `(atom_i, atom_j)` the factor `W` is
    /// `U_iᵀ U_j` (`r_i × r_j`), with the `None` frame standing in for `I_p`:
    /// a framed×unframed cross gives `W = U_iᵀ` (`r_i × p`) and an unframed
    /// diagonal gives `W = I_p` — exactly reproducing the `g ⊗ I_p` full-`B`
    /// block. The resulting blocks are handed to [`Self::new`], which validates
    /// the `(M, r)` shapes and computes the β offsets.
    pub fn from_frames_and_blocks(
        frames: &[Option<Array2<f64>>],
        basis_sizes: &[usize],
        p: usize,
        g_blocks: &std::collections::BTreeMap<(usize, usize), Array2<f64>>,
    ) -> Result<Self, String> {
        if frames.len() != basis_sizes.len() {
            return Err(format!(
                "FactoredFrameKroneckerOp::from_frames_and_blocks: {} frames but {} basis sizes",
                frames.len(),
                basis_sizes.len()
            ));
        }
        let n_atoms = frames.len();
        // Per-atom rank: ncols of a supplied frame, else the ambient dim p.
        let mut ranks = Vec::with_capacity(n_atoms);
        for (k, frame) in frames.iter().enumerate() {
            match frame {
                Some(u) => {
                    let (pr, r) = u.dim();
                    if pr != p {
                        return Err(format!(
                            "FactoredFrameKroneckerOp::from_frames_and_blocks: frame {k} has {pr} rows but ambient dim is {p}"
                        ));
                    }
                    if r > p {
                        return Err(format!(
                            "FactoredFrameKroneckerOp::from_frames_and_blocks: frame {k} has rank {r} > ambient dim {p}"
                        ));
                    }
                    ranks.push(r);
                }
                None => ranks.push(p),
            }
        }
        // Materialize each atom's frame as a `p × r_k` view source: the supplied
        // `U_k`, or `I_p` for the unframed atoms.
        let identity = Array2::<f64>::eye(p);
        let frame_or_ident = |k: usize| -> ArrayView2<f64> {
            match &frames[k] {
                Some(u) => u.view(),
                None => identity.view(),
            }
        };
        let mut blocks = Vec::with_capacity(g_blocks.len());
        for (&(atom_i, atom_j), g) in g_blocks {
            if atom_i >= n_atoms || atom_j >= n_atoms {
                return Err(format!(
                    "FactoredFrameKroneckerOp::from_frames_and_blocks: block atom indices ({atom_i}, {atom_j}) out of range (n_atoms = {n_atoms})"
                ));
            }
            let w = frame_output_gram(frame_or_ident(atom_i), frame_or_ident(atom_j));
            blocks.push(FactoredFrameGBlock {
                atom_i,
                atom_j,
                g: g.clone(),
                w,
            });
        }
        Self::new(ranks, basis_sizes.to_vec(), blocks)
    }
}

impl BetaPenaltyOp for FactoredFrameKroneckerOp {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for blk in &self.blocks {
            let r_i = self.ranks[blk.atom_i];
            let r_j = self.ranks[blk.atom_j];
            let off_i = self.offsets[blk.atom_i];
            let off_j = self.offsets[blk.atom_j];
            let (m_i, m_j) = blk.g.dim();
            for li in 0..m_i {
                let yi_base = off_i + li * r_i;
                for lj in 0..m_j {
                    let g = blk.g[[li, lj]];
                    if g == 0.0 {
                        continue;
                    }
                    let xj_base = off_j + lj * r_j;
                    // y_block[li, a] += g · Σ_b w[a, b] · x_block[lj, b]
                    for a in 0..r_i {
                        let mut acc = 0.0;
                        for b in 0..r_j {
                            acc += blk.w[[a, b]] * x[xj_base + b];
                        }
                        y[yi_base + a] += g * acc;
                    }
                }
            }
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.matvec(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        for blk in &self.blocks {
            // Only on-diagonal atom blocks carry diagonal mass; the entry at
            // (atom k, basis li, coord a) is g[li,li]·w[a,a].
            if blk.atom_i != blk.atom_j {
                continue;
            }
            let r = self.ranks[blk.atom_i];
            let off = self.offsets[blk.atom_i];
            let (m_i, m_j) = blk.g.dim();
            let m = m_i.min(m_j);
            for li in 0..m {
                let gii = blk.g[[li, li]];
                let base = off + li * r;
                for a in 0..r {
                    diag[base + a] += gii * blk.w[[a, a]];
                }
            }
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        // Dense sub-block over the β index range `offsets[id.0]`. Mirror the
        // global (i,a) ↔ (j,b) coupling, keeping only indices inside the range.
        let range = &offsets[id.0];
        let b_dim = range.end - range.start;
        for blk in &self.blocks {
            let r_i = self.ranks[blk.atom_i];
            let r_j = self.ranks[blk.atom_j];
            let off_i = self.offsets[blk.atom_i];
            let off_j = self.offsets[blk.atom_j];
            let (m_i, m_j) = blk.g.dim();
            for li in 0..m_i {
                for a in 0..r_i {
                    let gi = off_i + li * r_i + a;
                    if gi < range.start || gi >= range.end {
                        continue;
                    }
                    let bi = gi - range.start;
                    for lj in 0..m_j {
                        let g = blk.g[[li, lj]];
                        if g == 0.0 {
                            continue;
                        }
                        for b in 0..r_j {
                            let gj = off_j + lj * r_j + b;
                            if gj < range.start || gj >= range.end {
                                continue;
                            }
                            let bj = gj - range.start;
                            if bi < b_dim && bj < b_dim {
                                out[[bi, bj]] += g * blk.w[[a, b]];
                            }
                        }
                    }
                }
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        for blk in &self.blocks {
            let r_i = self.ranks[blk.atom_i];
            let r_j = self.ranks[blk.atom_j];
            let off_i = self.offsets[blk.atom_i];
            let off_j = self.offsets[blk.atom_j];
            let (m_i, m_j) = blk.g.dim();
            for li in 0..m_i {
                for lj in 0..m_j {
                    let g = blk.g[[li, lj]];
                    if g == 0.0 {
                        continue;
                    }
                    for a in 0..r_i {
                        let gi = off_i + li * r_i + a;
                        for b in 0..r_j {
                            let gj = off_j + lj * r_j + b;
                            out[[gi, gj]] += g * blk.w[[a, b]];
                        }
                    }
                }
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("factored-frame-kronecker-op-v1");
        hasher.write_usize(self.dim);
        for &r in &self.ranks {
            hasher.write_usize(r);
        }
        for &m in &self.basis_sizes {
            hasher.write_usize(m);
        }
        hasher.write_usize(self.blocks.len());
        for blk in &self.blocks {
            hasher.write_usize(blk.atom_i);
            hasher.write_usize(blk.atom_j);
            hasher.write_f64_array2(&blk.g);
            hasher.write_f64_array2(&blk.w);
        }
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

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        hasher.write_str("composite-penalty-op-v1");
        hasher.write_usize(self.k);
        hasher.write_usize(self.ops.len());
        for op in &self.ops {
            op.fingerprint(hasher);
        }
    }
}

/// Adapts a closure-based matrix-free `H_ββ` operator (from
/// [`ArrowSchurSystem::set_shared_beta_operator`]) to the `BetaPenaltyOp` trait.
///
/// `diagonal` holds the precomputed `diag(H_ββ)` supplied alongside the matvec;
/// `to_dense` falls back to probing all `K` canonical basis vectors.
pub struct MatvecDiagPenaltyOp {
    k: usize,
    matvec: SharedBetaMatvec,
    diagonal_vec: Array1<f64>,
}

impl MatvecDiagPenaltyOp {
    pub fn new(k: usize, matvec: SharedBetaMatvec, diagonal_vec: Array1<f64>) -> Self {
        assert_eq!(diagonal_vec.len(), k);
        Self {
            k,
            matvec,
            diagonal_vec,
        }
    }
}

impl BetaPenaltyOp for MatvecDiagPenaltyOp {
    fn dim(&self) -> usize {
        self.k
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let x_arr = Array1::from_iter(x.iter().copied());
        let mut out = Array1::<f64>::zeros(self.k);
        (self.matvec)(x_arr.view(), &mut out);
        for a in 0..self.k {
            y[a] += out[a];
        }
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        let beta_arr = Array1::from_iter(beta.iter().copied());
        let mut hb = Array1::<f64>::zeros(self.k);
        (self.matvec)(beta_arr.view(), &mut hb);
        for a in 0..self.k {
            out[a] += hb[a];
        }
    }

    fn diagonal(&self, diag: &mut [f64]) {
        for j in 0..self.k.min(diag.len()) {
            diag[j] += self.diagonal_vec[j];
        }
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        // Probe each basis vector in the block range to extract the sub-block.
        let range = &offsets[id.0];
        let b = range.end - range.start;
        let mut probe = Array1::<f64>::zeros(self.k);
        for bj in 0..b {
            probe.fill(0.0);
            probe[range.start + bj] = 1.0;
            let mut col = Array1::<f64>::zeros(self.k);
            (self.matvec)(probe.view(), &mut col);
            for bi in 0..b {
                out[[bi, bj]] += col[range.start + bi];
            }
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let k = self.k;
        let mut out = Array2::<f64>::zeros((k, k));
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let mut col = Array1::<f64>::zeros(k);
            (self.matvec)(probe.view(), &mut col);
            for i in 0..k {
                out[[i, j]] = col[i];
            }
        }
        out
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        // The matvec closure cannot be hashed by content; the precomputed
        // diagonal is the operator's stable defining proxy (it is recomputed
        // alongside the matvec each time the operator is installed).
        hasher.write_str("matvec-diag-penalty-op-v1");
        hasher.write_usize(self.k);
        for &value in self.diagonal_vec.iter() {
            hasher.write_f64(value);
        }
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
///   `run_pcg_with_preconditioner` are wired.
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
    /// Mixed-precision certificate outcome for this solve.
    pub mixed_precision_status: MixedPrecisionStatus,
}

/// Outcome of an opt-in mixed-precision arrow solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MixedPrecisionStatus {
    /// The caller did not request mixed precision or this solve mode cannot use it.
    #[default]
    Off,
    /// The f32 factor solve was refined until the f64 backward-error certificate held.
    Certified { refinement_steps: usize },
    /// The kappa gate or solve shape rejected mixed precision and the f64 path ran.
    F64Fallback,
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

/// Opt-in Carson--Higham mixed-precision refinement for dense arrow solves.
///
/// Default is [`MixedPrecisionPolicy::Off`]: exact f64 solves remain the default.
/// [`MixedPrecisionPolicy::Certified`] stores f32 copies of the per-row Cholesky
/// factors and dense Schur factor, solves corrections in f32, and recomputes the
/// residual in f64 against the original arrow blocks. The standard refinement
/// certificate is the normwise backward error
///
/// `||r||_inf / (||H||_inf ||x||_inf + ||b||_inf) <= residual_relative_tolerance`.
///
/// The kappa gate enforces `kappa_estimate * u_f32 < kappa_unit_roundoff_margin`;
/// when it fails, the solver emits a loud fallback message and uses the f64 path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixedPrecisionPolicy {
    Off,
    Certified {
        max_refinement_steps: usize,
        residual_relative_tolerance: f64,
        kappa_unit_roundoff_margin: f64,
    },
}

impl Default for MixedPrecisionPolicy {
    fn default() -> Self {
        Self::Off
    }
}

impl MixedPrecisionPolicy {
    pub fn certified() -> Self {
        Self::Certified {
            max_refinement_steps: DEFAULT_MIXED_PRECISION_MAX_REFINEMENTS,
            residual_relative_tolerance: DEFAULT_MIXED_PRECISION_CERTIFICATE_TOLERANCE,
            kappa_unit_roundoff_margin: DEFAULT_MIXED_PRECISION_KAPPA_MARGIN,
        }
    }

    fn is_enabled(self) -> bool {
        matches!(self, MixedPrecisionPolicy::Certified { .. })
    }
}

/// Complete BA Schur solve options.
///
/// Use [`ArrowSolveOptions::automatic`] for normal latent-coordinate fits;
/// use [`ArrowSolveOptions::sqrt_ba`] when the assembler has single-precision
/// row blocks or an ill-conditioned gauge; use [`ArrowSolveOptions::inexact_pcg`]
/// for SAE-manifold scale `K`.
#[derive(Clone)]
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
    /// When set, `run_pcg_with_preconditioner` delegates each `S·p` call to
    /// this closure instead of the CPU `schur_matvec`. Constructed by
    /// `crate::gpu::arrow_schur::gpu_schur_matvec_backend` when `cuda_selected()`
    /// and the system has dense per-row H_tβ slabs. `None` means CPU-only PCG.
    pub gpu_matvec: Option<GpuSchurMatvec>,
    /// Skip the ill-conditioning *rejection* (the κ-based
    /// [`ArrowSchurError::PerRowFactorIllConditioned`] per-row guard and the
    /// matching reduced-Schur κ guard) while still requiring genuine positive
    /// definiteness (a non-PD Cholesky pivot still errors).
    ///
    /// The κ guards exist to protect the accuracy of the Newton *step*: a
    /// barely-PD `H_tt^(i)` or an over-conditioned reduced Schur yields an
    /// inaccurate `Δβ`/`Δt`. Evidence-only callers
    /// (e.g. `SaeManifoldTerm::reml_criterion_with_cache`) do not consume the
    /// step — they need only the factor cache for the log-determinant
    /// (`½log|H|`, exact from `diag(L)` regardless of κ) and the selected-inverse
    /// traces. For those callers the κ rejection is a false abort when ρ sweeps
    /// to extreme values, so this flag lifts it and hands the
    /// "is this step trustworthy" decision back to the caller.
    ///
    /// Default `false`: ordinary solves keep the full guard.
    pub tolerate_ill_conditioning: bool,
    /// Opt-in certified mixed-precision direct solve. Default is off.
    pub mixed_precision: MixedPrecisionPolicy,
}

impl std::fmt::Debug for ArrowSolveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSolveOptions")
            .field("mode", &self.mode)
            .field("pcg", &self.pcg)
            .field("trust_region", &self.trust_region)
            .field("streaming_chunk_size", &self.streaming_chunk_size)
            .field("riemannian_trust_region", &self.riemannian_trust_region)
            .field("gpu_matvec", &self.gpu_matvec.is_some())
            .field("tolerate_ill_conditioning", &self.tolerate_ill_conditioning)
            .field("mixed_precision", &self.mixed_precision)
            .finish()
    }
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
    /// Relative objective resolution below which the proximal correction
    /// declares convergence instead of failing.
    ///
    /// Near a stationary point the largest decrease the damped Newton model can
    /// still achieve shrinks to the floating-point resolution of the objective
    /// itself: at proximal ridge `μ → μ_max` the accepted step length is
    /// `O(‖g‖ / μ)`, so the realised change in the objective falls below
    /// `rel_tol · (|f| + 1)`. At that scale the Armijo sufficient-decrease test
    /// compares two values that differ only by rounding noise, and no further
    /// productive decrease is achievable. Rather than raise
    /// `AdaptiveCorrectionFailed`, the loop then returns the incumbent state
    /// (a zero step) as converged. This does NOT mask genuine non-convergence:
    /// it triggers only when every attempted step either fails to decrease the
    /// objective by more than this resolution OR increases it by no more than
    /// this resolution (pure rounding). A step that genuinely reduces the
    /// objective is always taken first.
    pub convergence_objective_rel_tol: f64,
}

impl Default for ArrowProximalCorrectionOptions {
    fn default() -> Self {
        Self {
            initial_ridge: DEFAULT_PROXIMAL_INITIAL_RIDGE,
            ridge_growth: DEFAULT_PROXIMAL_RIDGE_GROWTH,
            max_attempts: DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            armijo_c1: DEFAULT_ARMIJO_C1,
            gradient_tolerance: DEFAULT_GRADIENT_TOLERANCE,
            convergence_objective_rel_tol: DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL,
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
            tolerate_ill_conditioning: false,
            mixed_precision: MixedPrecisionPolicy::Off,
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
            tolerate_ill_conditioning: false,
            mixed_precision: MixedPrecisionPolicy::Off,
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
            tolerate_ill_conditioning: false,
            mixed_precision: MixedPrecisionPolicy::Off,
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
            tolerate_ill_conditioning: false,
            mixed_precision: MixedPrecisionPolicy::Off,
        }
    }

    pub fn with_streaming_chunk_size(mut self, chunk_size: Option<usize>) -> Self {
        self.streaming_chunk_size = chunk_size.filter(|&chunk| chunk > 0);
        self
    }

    /// Lift the ill-conditioning *rejection* for evidence/log-det-only callers
    /// while still requiring genuine PD. See [`Self::tolerate_ill_conditioning`].
    ///
    /// Use this when the returned `(Δt, Δβ)` Newton step is discarded and only
    /// the factor cache is consumed (log-determinant + selected-inverse traces).
    /// The cache stays undamped at `ridge_t = 0`, so the log-determinant is
    /// exact regardless of κ.
    pub fn with_ill_conditioning_tolerated(mut self) -> Self {
        self.tolerate_ill_conditioning = true;
        self
    }

    pub fn with_mixed_precision_policy(mut self, policy: MixedPrecisionPolicy) -> Self {
        self.mixed_precision = policy;
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
    ///
    /// `tolerate_ill_conditioning` lifts the per-row κ rejection (still
    /// requiring genuine PD); see [`ArrowSolveOptions::tolerate_ill_conditioning`].
    fn factor_blocks(
        &self,
        rows: &[ArrowRowBlock],
        ridge_t: f64,
        d: usize,
        tolerate_ill_conditioning: bool,
    ) -> Result<ArrowFactorSlab, ArrowSchurError>;

    /// Solve one factored point block against a vector RHS.
    fn solve_block_vector(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView1<'_, f64>,
    ) -> Array1<f64>;

    /// Solve one factored point block against a dense matrix RHS.
    fn solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64>;

    /// Apply the Square-Root BA lower-triangular solve `L_i^-1 rhs`.
    fn sqrt_solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64>;

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
        tolerate_ill_conditioning: bool,
    ) -> Result<ArrowFactorSlab, ArrowSchurError> {
        // Multi-GPU fast path: the per-row blocks `H_tt^(i) + ridge_t·I` are
        // independent same-size SPD systems — exactly the batch
        // `crate::gpu::try_cholesky_batched_lower_inplace` spreads across ALL
        // usable devices (the batched POTRF tiles over the pool). It is only
        // valid when every row is the uniform `d×d` shape; heterogeneous row
        // dimensions keep the per-row CPU loop because the current cuSOLVER
        // batched POTRF wrapper accepts one `(d, d)` shape per launch. It only
        // succeeds when EVERY block is PD at
        // the base ridge; a non-PD block returns `None`, so we fall back to the
        // exact per-row CPU path that performs minimal per-block ridge
        // escalation. After a successful batched factorization we re-apply the
        // identical κ-conditioning rejection `factor_one_row` enforces, so the
        // result is bit-for-bit equivalent (modulo IEEE reduction order) to the
        // CPU loop: a barely-PD but ill-conditioned block forces the whole batch
        // back onto the per-row path so its ridge can lift, never silently using
        // a contaminated factor.
        if let Some(batched) =
            try_factor_blocks_batched(rows, ridge_t, d, tolerate_ill_conditioning)
        {
            return Ok(batched);
        }
        let mut out = Vec::with_capacity(rows.len());
        for (row_idx, row) in rows.iter().enumerate() {
            out.push(factor_one_row(
                row,
                ridge_t,
                d,
                row_idx,
                tolerate_ill_conditioning,
            )?);
        }
        Ok(ArrowFactorSlab::from_blocks(out))
    }

    fn solve_block_vector(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        match (factor.nrows(), factor.ncols(), rhs.len()) {
            (1, 1, 1) => cholesky_solve_vector_fixed::<1>(factor, rhs),
            (2, 2, 2) => cholesky_solve_vector_fixed::<2>(factor, rhs),
            (3, 3, 3) => cholesky_solve_vector_fixed::<3>(factor, rhs),
            (4, 4, 4) => cholesky_solve_vector_fixed::<4>(factor, rhs),
            _ => cholesky_solve_vector(factor, rhs),
        }
    }

    fn solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        cholesky_solve_matrix(factor, rhs)
    }

    fn sqrt_solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        forward_substitution_lower_matrix(factor, rhs)
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

/// Attempt the per-row block factorization as one device batch spread across
/// every usable GPU.
///
/// The `n` per-row blocks `H_tt^(i) + ridge_t·I` are independent SPD systems of
/// the uniform shape `d×d`; `crate::gpu::try_cholesky_batched_lower_inplace`
/// factors the whole batch with a batched POTRF that the shared device pool
/// tiles across all ordinals. Returns `Some(factors)` only when:
///   * every row really is the uniform `(d, d)` shape with a length-`d` `g_t`
///     (heterogeneous systems keep the per-row CPU loop), and
///   * a device is available and EVERY block is positive-definite at the base
///     ridge (a non-PD block makes the batched POTRF return `None`), and
///   * unless `tolerate_ill_conditioning`, every resulting factor passes the
///     same diagonal-ratio κ ceiling `factor_one_row` enforces.
///
/// Any of those failing returns `None`, so the caller runs the exact per-row
/// CPU path (which performs minimal per-block ridge escalation and the κ check).
/// The factor a device POTRF produces is the lower Cholesky of the identical
/// SPD matrix the CPU `cholesky_lower` would, with the strict upper triangle
/// zeroed — bit-for-bit equivalent modulo IEEE-754 reduction order.
fn try_factor_blocks_batched(
    rows: &[ArrowRowBlock],
    ridge_t: f64,
    d: usize,
    tolerate_ill_conditioning: bool,
) -> Option<ArrowFactorSlab> {
    if d == 0 || rows.is_empty() {
        return None;
    }
    // Uniform-shape gate: a heterogeneous row defeats the single-shape batched
    // POTRF and deliberately falls through to per-row CPU escalation.
    if rows
        .iter()
        .any(|row| row.htt.dim() != (d, d) || row.gt.len() != d)
    {
        return None;
    }
    // No device → let the CPU path own the work (it is the exact fallback).
    if !crate::gpu::runtime::GpuRuntime::is_available() {
        return None;
    }

    // Assemble the damped blocks `H_tt^(i) + ridge_t·I` for the batched POTRF.
    let mut blocks: Vec<Array2<f64>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut block = row.htt.clone();
        for a in 0..d {
            block[[a, a]] += ridge_t;
        }
        blocks.push(block);
    }

    // Batched lower Cholesky over ALL usable GPUs. `None` ⇒ either no device
    // accepted the workload or some block was not PD at the base ridge; either
    // way the per-row CPU path must own escalation.
    crate::gpu::try_cholesky_batched_lower_inplace(&mut blocks)?;

    // Re-apply the κ-conditioning rejection so a barely-PD block forces the
    // whole batch back to the per-row path (where its ridge lifts), matching
    // `factor_one_row` semantics exactly. Evidence/log-det-only callers
    // tolerate ill-conditioning and skip this, as on the CPU path.
    if !tolerate_ill_conditioning {
        for (row, factor) in rows.iter().zip(blocks.iter()) {
            let diag_scale = row_block_diag_scale(row, d);
            let kappa_est = cholesky_factor_kappa_estimate(factor);
            if !cholesky_factor_passes_safe_inversion(factor, d, diag_scale, kappa_est) {
                return None;
            }
        }
    }
    Some(ArrowFactorSlab::from_blocks(blocks))
}

fn row_block_diag_scale(row: &ArrowRowBlock, d: usize) -> f64 {
    (0..d)
        .map(|a| row.htt[[a, a]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0)
}

/// Diagonal-ratio condition-number proxy for an SPD matrix from its lower
/// Cholesky factor `L` (where `A = L Lᵀ`):
///     κ(A) ≈ (max_i L_ii / min_i L_ii)².
///
/// (Golub & Van Loan, "Matrix Computations" 4th ed., §4.2.4 — the ratio of
/// diagonal entries of the Cholesky factor bounds the 2-norm condition number
/// of the SPD matrix.) Returns `f64::INFINITY` when the factor has a
/// non-positive or non-finite diagonal pivot, which the callers treat as a
/// hard ill-conditioning signal.
fn cholesky_factor_kappa_estimate(factor: &Array2<f64>) -> f64 {
    let d = factor.nrows();
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
        ratio * ratio
    } else {
        f64::INFINITY
    }
}

/// Smallest Cholesky pivot estimate for `A = L Lᵀ`, using `L_ii²`.
///
/// The diagonal-ratio κ proxy is blind for scalar blocks: every positive
/// `1×1` factor has κ=1 even when the pivot is tiny. This pivot floor catches
/// absolute near-singularity relative to the row block scale.
fn cholesky_factor_min_pivot_estimate(factor: &Array2<f64>) -> f64 {
    let d = factor.nrows();
    if d == 0 {
        return 0.0;
    }
    let mut min_pivot = f64::INFINITY;
    for a in 0..d {
        let v = factor[[a, a]];
        if !(v > 0.0 && v.is_finite()) {
            return 0.0;
        }
        let pivot = v * v;
        if pivot < min_pivot {
            min_pivot = pivot;
        }
    }
    min_pivot
}

fn safe_spd_pivot_min(diag_scale: f64) -> f64 {
    f64::EPSILON.sqrt() * diag_scale.max(1.0)
}

fn cholesky_factor_passes_safe_inversion(
    factor: &Array2<f64>,
    dim: usize,
    diag_scale: f64,
    kappa_est: f64,
) -> bool {
    kappa_est.is_finite()
        && kappa_est <= safe_spd_kappa_max(dim)
        && cholesky_factor_min_pivot_estimate(factor) >= safe_spd_pivot_min(diag_scale)
}

/// Near-singularity condition-number ceiling for double precision at dimension
/// `dim`: κ_max = 1 / (sqrt(DBL_EPS) · max(dim, 1)).
///
/// Classic Higham rule (Higham, "Accuracy and Stability of Numerical
/// Algorithms" 2nd ed., §10.1): a system is treated as numerically
/// rank-deficient once κ · ε approaches 1/sqrt(ε), scaled by problem dimension.
fn safe_spd_kappa_max(dim: usize) -> f64 {
    let d_scale = (dim as f64).max(1.0);
    1.0 / (f64::EPSILON.sqrt() * d_scale)
}

fn factor_row_block_cholesky(
    row: &ArrowRowBlock,
    ridge_eff: f64,
    d: usize,
) -> Result<Array2<f64>, String> {
    match d {
        1 => factor_row_block_cholesky_fixed::<1>(row, ridge_eff),
        2 => factor_row_block_cholesky_fixed::<2>(row, ridge_eff),
        3 => factor_row_block_cholesky_fixed::<3>(row, ridge_eff),
        4 => factor_row_block_cholesky_fixed::<4>(row, ridge_eff),
        _ => factor_row_block_cholesky_dynamic(row, ridge_eff, d),
    }
}

fn factor_row_block_cholesky_dynamic(
    row: &ArrowRowBlock,
    ridge_eff: f64,
    d: usize,
) -> Result<Array2<f64>, String> {
    let mut block = row.htt.clone();
    for a in 0..d {
        block[[a, a]] += ridge_eff;
    }
    cholesky_lower(&block)
}

fn factor_row_block_cholesky_fixed<const D: usize>(
    row: &ArrowRowBlock,
    ridge_eff: f64,
) -> Result<Array2<f64>, String> {
    for i in 0..D {
        for j in 0..D {
            let value = if i == j {
                row.htt[[i, j]] + ridge_eff
            } else {
                row.htt[[i, j]]
            };
            if !value.is_finite() {
                let idx = i * D + j;
                return Err(format!(
                    "cholesky_lower: non-finite entry at linear index {idx}"
                ));
            }
        }
    }

    let mut l = [[0.0_f64; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut sum = if i == j {
                row.htt[[i, j]] + ridge_eff
            } else {
                row.htt[[i, j]]
            };
            for kk in 0..j {
                sum -= l[i][kk] * l[j][kk];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut out = Array2::<f64>::zeros((D, D));
    for i in 0..D {
        for j in 0..=i {
            out[[i, j]] = l[i][j];
        }
    }
    Ok(out)
}

fn cholesky_solve_vector_fixed<const D: usize>(
    l: ArrayView2<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut y = [0.0_f64; D];
    for i in 0..D {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }

    let mut x = [0.0_f64; D];
    for i in (0..D).rev() {
        let mut sum = y[i];
        for k in (i + 1)..D {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }

    let mut out = Array1::<f64>::zeros(D);
    for i in 0..D {
        out[i] = x[i];
    }
    out
}

fn factor_one_row(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    row_idx: usize,
    tolerate_ill_conditioning: bool,
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
    // Per-row adaptive Tikhonov ridge. A non-convex objective (e.g. softmax
    // assignment) can leave an individual token's latent Hessian H_tt^(i)
    // indefinite, so `H_tt + ridge_t·I` has a negative Cholesky pivot. Rather
    // than fail and force the OUTER LM loop to lift `ridge_t` for EVERY row
    // (over-damping the well-conditioned tokens), damp only this block by the
    // minimal amount it needs: escalate this row's ridge geometrically from the
    // caller's base `ridge_t` until the factor is positive-definite. A
    // positive-definite block factors at the base ridge with zero escalation,
    // so the common case is bit-for-bit unchanged. The escalation is capped
    // relative to the block's diagonal scale, so a genuinely broken block
    // (non-finite, or unboundedly indefinite) still surfaces as
    // `PerRowFactorFailed` for the outer loop to handle rather than looping.
    // Per-row ridge escalation policy. The escalation starts at the caller's
    // base ridge (or, if that is zero, a tiny seed scaled by the block's
    // diagonal magnitude), multiplies geometrically each rejection, and is
    // capped at a large multiple of the base scale so a genuinely broken block
    // surfaces as an error instead of looping forever.
    const RIDGE_GROWTH_FACTOR: f64 = 10.0;
    const RIDGE_SEED_DIAG_FRACTION: f64 = 1.0e-10;
    const RIDGE_CAP_DIAG_FRACTION: f64 = 1.0e-12;
    const RIDGE_CAP_SCALE: f64 = 1.0e12;
    let diag_scale = row_block_diag_scale(row, d);
    let ridge_cap = ridge_t.max(RIDGE_CAP_DIAG_FRACTION * diag_scale) * RIDGE_CAP_SCALE;
    let mut ridge_eff = ridge_t;
    // Escalate the per-row ridge until the block is BOTH positive-definite AND
    // well-conditioned. Previously the escalation only fired on a *failed*
    // Cholesky (indefinite block); a barely-PD but ill-conditioned block
    // (pivots ~ε·trace — e.g. a rank-deficient / over-parameterized decoder
    // atom) factored successfully and was then rejected outright as
    // `PerRowFactorIllConditioned`, so the ridge the SAE audit advertises
    // ("the Arrow-Schur ridge will regularise the deficient directions") never
    // got the chance to. Folding the κ proxy into the loop lets the ridge lift
    // just enough to regularise the deficient directions, as advertised,
    // instead of aborting the whole fit (gam#578). A genuinely PD,
    // well-conditioned block factors at the base ridge with zero escalation and
    // is bit-for-bit unchanged; only a block that cannot be conditioned even at
    // `ridge_cap` (1e12 × base) still surfaces an error for the outer loop.
    let factor = loop {
        match factor_row_block_cholesky(row, ridge_eff, d) {
            Ok(factor) => {
                // Evidence/log-det-only callers tolerate ill-conditioning: the
                // factor is genuinely PD, so its diagonal gives an exact log|S|
                // and an inaccurate Δβ would be discarded anyway.
                if tolerate_ill_conditioning {
                    break factor;
                }
                // Diagonal-ratio condition-number proxy κ(LLᵀ) ≈
                // (max L_ii / min L_ii)², vs the dimension-scaled Higham
                // near-singularity ceiling. A barely-PD inverse plugged into
                //   S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
                // contaminates S by spectral terms scaled by κ_i, so an
                // over-threshold block is regularised further rather than used.
                let kappa_est = cholesky_factor_kappa_estimate(&factor);
                if cholesky_factor_passes_safe_inversion(&factor, d, diag_scale, kappa_est) {
                    break factor;
                }
                let next = if ridge_eff > 0.0 {
                    ridge_eff * RIDGE_GROWTH_FACTOR
                } else {
                    RIDGE_SEED_DIAG_FRACTION * diag_scale
                };
                if !next.is_finite() || next > ridge_cap {
                    return Err(ArrowSchurError::PerRowFactorIllConditioned {
                        row: row_idx,
                        kappa_estimate: kappa_est,
                    });
                }
                ridge_eff = next;
            }
            Err(e) => {
                // Evidence/log-det callers (`tolerate_ill_conditioning = true`)
                // consume the returned factor's diagonal as the exact
                // log|H_tt + ridge_t·I|. Silently lifting ridge past the
                // caller's base would shift that determinant by Σ d·log(1+δ/λ)
                // while returning Ok, corrupting the reported evidence. A
                // genuinely non-PD block at the base ridge must surface as
                // an error here, not be quietly conditioned.
                if tolerate_ill_conditioning {
                    return Err(ArrowSchurError::PerRowFactorFailed {
                        row: row_idx,
                        reason: format!(
                            "row {row_idx} H_tt is non-PD at base ridge {ridge_t:e}; \
                             evidence mode preserves the genuine Cholesky of \
                             H_tt and does not condition non-PD blocks: {e}"
                        ),
                    });
                }
                let next = if ridge_eff > 0.0 {
                    ridge_eff * RIDGE_GROWTH_FACTOR
                } else {
                    RIDGE_SEED_DIAG_FRACTION * diag_scale
                };
                if !next.is_finite() || next > ridge_cap {
                    return Err(ArrowSchurError::PerRowFactorFailed {
                        row: row_idx,
                        reason: format!(
                            "row {row_idx} H_tt remained non-PD up to ridge {ridge_eff:e} \
                             (base ridge_t={ridge_t}); last cholesky error: {e}"
                        ),
                    });
                }
                ridge_eff = next;
            }
        }
    };
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
    // row.htbeta is usually a zero slab that does not capture the operator
    // state. Hash the Arc pointer address as a proxy: a new Arc is allocated
    // per assemble call, so the fingerprint is invalidated each time the
    // system is rebuilt with a fresh Kronecker operator. Analytic penalties may
    // opt into a dense supplemental slab; when active, hash it as well.
    // SAFETY: We cast the fat pointer to a thin *const () to extract the data
    // pointer address as a fingerprint proxy. No dereference occurs; the only
    // use is as a usize hash input, which is sound for any aligned pointer.
    let htbeta_op_addr: Option<usize> = sys
        .htbeta_matvec
        .as_ref()
        .map(|op| Arc::as_ptr(op) as *const () as usize);
    for row in sys.rows.iter() {
        hasher.write_f64_array2(&row.htt);
        match htbeta_op_addr {
            Some(addr) => {
                hasher.write_usize(addr);
                if sys.htbeta_dense_supplement {
                    hasher.write_f64_array2(&row.htbeta);
                }
            }
            None => hasher.write_f64_array2(&row.htbeta),
        }
    }
    // Hash the β-block operator's defining state. When a structured
    // `penalty_op` is installed (e.g. the SAE composite carrying the data-fit
    // Gauss-Newton block as `G ⊗ I_p`), hashing the operator captures the full
    // β-block content cheaply; the dense `sys.hbb` no longer holds it. When no
    // `penalty_op` is installed, fall back to hashing the dense accumulator.
    match sys.penalty_op.as_ref() {
        Some(op) => {
            hasher.write_bool(true);
            op.fingerprint(&mut hasher);
        }
        None => {
            hasher.write_bool(false);
            hasher.write_f64_array2(&sys.hbb);
        }
    }
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
                hasher.write_f64(crate::linalg::utils::stable_softplus(active_raw_beta));
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

/// Structural/value fingerprint for a cross-row (non-row-block-diagonal)
/// Psi-tier analytic penalty.
///
/// Unlike [`analytic_penalty_row_hessian_fingerprint`], which can read a
/// closed-form per-row diagonal, a cross-row penalty's curvature only surfaces
/// through its Hessian-vector product. We probe the penalty's PSD majorizer
/// against the *current latent vector itself* — a deterministic, penalty- and
/// state-dependent probe — and hash the resulting vector together with the
/// penalty name, target length, and local ρ. Any change to the operator that
/// matters for the Newton solve (different ρ, different smoothing geometry,
/// different latent linearization point) perturbs this probe, correctly
/// invalidating any factor cache keyed on the row-Hessian fingerprint.
fn cross_row_penalty_fingerprint(
    penalty: &AnalyticPenaltyKind,
    target_t: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-analytic-cross-row-hessian-v1");
    hasher.write_str(penalty.name());
    hasher.write_usize(target_t.len());
    hasher.write_usize(rho_local.len());
    for &rho in rho_local.iter() {
        hasher.write_f64(rho);
    }
    let probe = penalty.psd_majorizer_hvp(target_t, rho_local, target_t);
    hasher.write_usize(probe.len());
    for &value in probe.iter() {
        hasher.write_f64(value);
    }
    hasher.finish_u64()
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
        Self::new_with_htbeta_cols(d, k)
    }

    /// Allocate one BA row whose dense cross-block slab has `htbeta_cols`
    /// columns. This is used by matrix-free assemblers that keep the shared
    /// beta tier at one width while dense row supplements live in another
    /// coordinate system.
    pub fn new_with_htbeta_cols(d: usize, htbeta_cols: usize) -> Self {
        Self {
            htt: Array2::<f64>::zeros((d, d)),
            htbeta: Array2::<f64>::zeros((d, htbeta_cols)),
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
    /// Optional row-local matrix-free transpose multiply `out += H_βt^(i) · v`.
    ///
    /// The sparse adjoint of [`Self::htbeta_matvec`]. When present, the
    /// reduced-Schur matvec applies `H_βt^(i)` directly (sparse `scatter`)
    /// instead of probing the forward operator against `K` basis vectors. This
    /// is the per-row sparse apply that lifts the `O(K)` column-probe in the
    /// GPU PCG and streaming Schur paths to `O(m_i · p)` per row. Installed in
    /// lock-step with `htbeta_matvec` by [`Self::set_row_htbeta_operator`].
    pub htbeta_transpose_matvec: Option<RowHtbetaTransposeMatvec>,
    /// Whether `rows[*].htbeta` contains a dense contribution that must be added
    /// on top of the matrix-free row operator.
    pub htbeta_dense_supplement: bool,
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
    /// Registered Psi-tier analytic penalties whose Hessian couples *distinct*
    /// latent rows (non-row-block-diagonal), captured by
    /// [`Self::add_analytic_penalty_contributions`].
    ///
    /// These penalties (`TotalVariationPenalty`, `SheafConsistencyPenalty`,
    /// block-orthogonality, …) produce off-row Hessian blocks `∂²P/∂t_i∂t_j`
    /// (`i ≠ j`) that the arrow elimination — which assumes each `H_tt^(i)` is
    /// independent of every other row — cannot represent. Their *gradient* is
    /// still folded into `g_t` exactly like every other Psi penalty; only their
    /// curvature is held here, applied during the solve as a full-latent
    /// Hessian-vector product `P_cross · Δt` against the penalty's
    /// `psd_majorizer_hvp`. When this vector is non-empty,
    /// [`solve_arrow_newton_step_artifacts`] auto-selects the matrix-free
    /// full-system PCG path (arrow block-diagonal inverse as preconditioner)
    /// instead of the exact one-shot Schur elimination. When empty, the system
    /// is purely row-block-diagonal and the exact Schur path is unchanged.
    pub cross_row_penalties: Vec<CrossRowLatentPenalty>,
}

/// A captured cross-row Psi-tier analytic penalty: the penalty kind plus the
/// global-ρ slice (`rho_local`) it was registered with.
///
/// Holds an owned copy of the local ρ-axes so the penalty's
/// [`AnalyticPenaltyKind::psd_majorizer_hvp`] can be evaluated during the
/// matrix-free full-system solve without re-deriving the ρ layout. The penalty
/// itself is an `Arc`-backed clone (cheap), so capturing it does not copy the
/// penalty payload.
#[derive(Clone)]
pub struct CrossRowLatentPenalty {
    /// The non-row-block-diagonal Psi penalty (e.g. `TotalVariationPenalty`).
    pub penalty: AnalyticPenaltyKind,
    /// The penalty's local ρ-axes (its slice of the global ρ vector).
    pub rho_local: Array1<f64>,
    /// The flat latent vector (`N·d`, row-major) the penalty's curvature was
    /// linearized at — i.e. the `target_t` passed to
    /// [`ArrowSchurSystem::add_analytic_penalty_contributions`]. The Hessian of
    /// a nonlinear penalty (the smoothed-TV curvature weights `φ''(D t)`,
    /// etc.) depends on this point, so `psd_majorizer_hvp` must be evaluated
    /// against it for the Newton operator to be the true Hessian at the
    /// current iterate.
    pub target_t: Array1<f64>,
}

impl ArrowSchurSystem {
    /// Allocate an empty BA reduced-camera-system instance sized
    /// `(N point/latent rows × d, K shared decoder parameters)`.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        Self::new_with_hbb(n, d, k, Array2::<f64>::zeros((k, k)))
    }

    /// Allocate an arrow system with no dense shared `H_ββ` block.
    ///
    /// Callers must install a penalty operator before solving if the shared block
    /// has nonzero curvature. This keeps large structured systems from allocating
    /// a `k × k` dense placeholder when all β curvature is supplied by operators.
    pub fn new_with_empty_hbb(n: usize, d: usize, k: usize) -> Self {
        Self::new_with_empty_hbb_and_htbeta_cols(n, d, k, k)
    }

    /// Allocate an arrow system with no dense shared `H_ββ` block and with
    /// per-row dense `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_empty_hbb_and_htbeta_cols(
        n: usize,
        d: usize,
        k: usize,
        htbeta_cols: usize,
    ) -> Self {
        let rows = (0..n)
            .map(|_| ArrowRowBlock::new_with_htbeta_cols(d, htbeta_cols))
            .collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
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
            cross_row_penalties: Vec::new(),
        };
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    /// Allocate an arrow system using a caller-owned dense shared-block buffer.
    /// The buffer must already have shape `(k, k)` and is zeroed in place before
    /// use so callers can recycle it across assemblies without changing
    /// numerics.
    pub fn new_with_hbb(n: usize, d: usize, k: usize, hbb: Array2<f64>) -> Self {
        Self::new_with_hbb_and_htbeta_cols(n, d, k, hbb, k)
    }

    /// Allocate an arrow system with a caller-owned dense shared-block buffer and
    /// per-row dense `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_hbb_and_htbeta_cols(
        n: usize,
        d: usize,
        k: usize,
        mut hbb: Array2<f64>,
        htbeta_cols: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        hbb.fill(0.0);
        let rows = (0..n)
            .map(|_| ArrowRowBlock::new_with_htbeta_cols(d, htbeta_cols))
            .collect();
        let row_dims: Arc<[usize]> = (0..n).map(|_| d).collect::<Vec<_>>().into();
        let row_offsets: Arc<[usize]> = (0..=n).map(|i| i * d).collect::<Vec<_>>().into();
        let mut sys = Self {
            rows,
            hbb,
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
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
            cross_row_penalties: Vec::new(),
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
        let matvec_arc: SharedBetaMatvec = Arc::new(matvec);
        // Mirror the closure into a BetaPenaltyOp so all hot paths (#296)
        // route through the trait while preserving hbb_matvec + hbb_diag for
        // code that inspects them directly.
        let penalty_op: Option<Arc<dyn BetaPenaltyOp>> = Some(Arc::new(MatvecDiagPenaltyOp::new(
            k,
            Arc::clone(&matvec_arc),
            diag.clone(),
        )));
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: Some(matvec_arc),
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
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
            penalty_op,
            cross_row_penalties: Vec::new(),
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
        Self::new_with_per_row_dims_and_hbb(per_row_dims, k, Array2::<f64>::zeros((k, k)))
    }

    /// Allocate a heterogeneous-row arrow system with no dense shared `H_ββ`
    /// block. See [`Self::new_with_empty_hbb`].
    pub fn new_with_per_row_dims_empty_hbb(per_row_dims: Vec<usize>, k: usize) -> Self {
        Self::new_with_per_row_dims_empty_hbb_and_htbeta_cols(per_row_dims, k, k)
    }

    /// Allocate a heterogeneous-row arrow system with no dense shared `H_ββ`
    /// block and with row `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_per_row_dims_empty_hbb_and_htbeta_cols(
        per_row_dims: Vec<usize>,
        k: usize,
        htbeta_cols: usize,
    ) -> Self {
        let n = per_row_dims.len();
        let d = per_row_dims.iter().copied().max().unwrap_or(0);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut cursor = 0usize;
        offsets.push(cursor);
        for &dim in &per_row_dims {
            cursor += dim;
            offsets.push(cursor);
        }
        let rows = per_row_dims
            .iter()
            .map(|&dim| ArrowRowBlock::new_with_htbeta_cols(dim, htbeta_cols))
            .collect();
        let mut sys = Self {
            rows,
            hbb: Array2::<f64>::zeros((0, 0)),
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
            hbb_diag: None,
            gb: Array1::<f64>::zeros(k),
            d,
            row_dims: Arc::from(per_row_dims.into_boxed_slice()),
            row_offsets: Arc::from(offsets.into_boxed_slice()),
            k,
            manifold_mode_fingerprint: EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT,
            row_hessian_fingerprint: 0,
            analytic_row_hessian_fingerprint: 0,
            block_offsets: Arc::from([] as [Range<usize>; 0]),
            penalty_op: None,
            cross_row_penalties: Vec::new(),
        };
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    /// Allocate a heterogeneous-row system using a caller-owned dense
    /// shared-block buffer. See [`Self::new_with_hbb`] for the reuse contract.
    pub fn new_with_per_row_dims_and_hbb(
        per_row_dims: Vec<usize>,
        k: usize,
        hbb: Array2<f64>,
    ) -> Self {
        Self::new_with_per_row_dims_and_hbb_and_htbeta_cols(per_row_dims, k, hbb, k)
    }

    /// Allocate a heterogeneous-row system using a caller-owned dense shared
    /// block and row `H_tβ` slabs allocated at `htbeta_cols` columns.
    pub fn new_with_per_row_dims_and_hbb_and_htbeta_cols(
        per_row_dims: Vec<usize>,
        k: usize,
        mut hbb: Array2<f64>,
        htbeta_cols: usize,
    ) -> Self {
        assert_eq!(hbb.dim(), (k, k));
        hbb.fill(0.0);
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
            .map(|&di| ArrowRowBlock::new_with_htbeta_cols(di, htbeta_cols))
            .collect();
        let mut sys = Self {
            rows,
            hbb,
            hbb_matvec: None,
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            htbeta_dense_supplement: false,
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
            cross_row_penalties: Vec::new(),
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
        let matvec_arc: SharedBetaMatvec = Arc::new(matvec);
        // Mirror the closure into a BetaPenaltyOp so all hot paths (#296)
        // route through the trait, preserving the existing hbb_matvec +
        // hbb_diag fields for code that inspects them directly.
        self.penalty_op = Some(Arc::new(MatvecDiagPenaltyOp::new(
            self.k,
            Arc::clone(&matvec_arc),
            diag.clone(),
        )));
        self.hbb_matvec = Some(matvec_arc);
        self.hbb_diag = Some(diag);
        self.refresh_row_hessian_fingerprint();
    }

    /// Mark the dense per-row cross-block slabs as active supplements to the
    /// installed matrix-free row operator.
    pub fn activate_dense_htbeta_supplement(&mut self) {
        self.htbeta_dense_supplement = true;
        self.refresh_row_hessian_fingerprint();
    }

    /// Install a matrix-free per-row cross-block operator and its sparse
    /// adjoint.
    ///
    /// `forward` must write `out = H_tβ^(row) x` for `out.len() == d` and
    /// `x.len() == K`. `transpose` must **add** `H_βt^(row) v` into `out` for
    /// `out.len() == K` and `v.len() == d` (the sparse `scatter` adjoint).
    ///
    /// When installed, the forward operator is used during the Newton solve
    /// (inside `reduced_rhs_beta`, `schur_matvec`, back-substitution, and
    /// `JacobiPreconditioner` construction) and afterwards by IFT/evidence
    /// predictors.  Per-row `htbeta` slabs in `ArrowRowBlock` may be left
    /// zero-sized when this operator is installed — all inner-Schur paths route
    /// through the matvec instead of indexing the dense block. The transpose
    /// operator lets the reduced-Schur matvec apply `H_βt^(row)` directly
    /// (`O(m_i · p)`) instead of probing `forward` against `K` basis vectors.
    pub fn set_row_htbeta_operator<F, T>(&mut self, forward: F, transpose: T)
    where
        F: for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
        T: for<'a> Fn(usize, ArrayView1<'a, f64>, &mut Array1<f64>) + Send + Sync + 'static,
    {
        self.htbeta_matvec = Some(Arc::new(forward));
        self.htbeta_transpose_matvec = Some(Arc::new(transpose));
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
    /// their own block layout.
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
        // The row-Hessian fingerprint now reads the β-block content from the
        // installed operator; refresh it so the factorization / evidence cache
        // (`cache_matches_system`) invalidates when the β-block changes.
        self.refresh_row_hessian_fingerprint();
    }

    /// Return the effective penalty operator: the installed `penalty_op` if
    /// present, otherwise a `DensePenaltyOp` wrapping the current `hbb`.
    ///
    /// Note: when `penalty_op` is `None`, this clones `hbb` into a new
    /// `DensePenaltyOp`. Callers in hot loops should call this once and
    /// store the result, not call it per-iteration.
    pub fn effective_penalty_op(&self) -> Arc<dyn BetaPenaltyOp> {
        match self.penalty_op.as_ref() {
            Some(op) => Arc::clone(op),
            None => Arc::new(DensePenaltyOp(self.hbb.clone())),
        }
    }

    /// `y += P x` without allocating a new Arc; dispatches to `penalty_op`
    /// or falls back to `hbb` inline, avoiding the K×K clone hot-path cost.
    #[inline]
    fn penalty_matvec_add(&self, x: &[f64], y: &mut [f64]) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.matvec(x, y);
        } else {
            let k = self.hbb.nrows();
            for a in 0..k {
                let mut acc = 0.0_f64;
                for b in 0..k {
                    acc += self.hbb[[a, b]] * x[b];
                }
                y[a] += acc;
            }
        }
    }

    /// `diag += diag(P)` without allocating; dispatches to `penalty_op`
    /// or falls back to `hbb` diagonal / `hbb_diag` inline.
    #[inline]
    fn penalty_diagonal_add(&self, diag: &mut [f64]) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.diagonal(diag);
        } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
            let k = hbb_diag.len().min(diag.len());
            for j in 0..k {
                diag[j] += hbb_diag[j];
            }
        } else {
            let k = self.hbb.nrows().min(diag.len());
            for j in 0..k {
                diag[j] += self.hbb[[j, j]];
            }
        }
    }

    /// Add the `b×b` penalty sub-block for `id` to `out`, routing through
    /// `penalty_op` or falling back to `hbb` / `hbb_diag` inline.
    #[inline]
    fn penalty_block_add(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        if let Some(op) = self.penalty_op.as_ref() {
            op.block(id, offsets, out);
        } else {
            let range = &offsets[id.0];
            let b = range.end - range.start;
            if self.hbb.dim() == (self.k, self.k) {
                for bi in 0..b {
                    for bj in 0..b {
                        out[[bi, bj]] += self.hbb[[range.start + bi, range.start + bj]];
                    }
                }
            } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
                for bi in 0..b {
                    out[[bi, bi]] += hbb_diag[range.start + bi];
                }
            }
        }
    }

    /// Fill a `b×b` penalty sub-block for a set of arbitrary (possibly
    /// non-contiguous) global column indices `cols`, routing through
    /// `penalty_op` or falling back to `hbb` / `hbb_diag` inline.
    ///
    /// Used by the cluster-Jacobi preconditioner (#299) which groups columns
    /// by spectral adjacency rather than contiguous block ranges.
    #[inline]
    fn penalty_subblock_add(&self, cols: &[usize], out: &mut Array2<f64>) {
        let b = cols.len();
        if let Some(op) = self.penalty_op.as_ref() {
            // Probe each column basis vector and extract the sub-block entries.
            let mut probe = Array1::<f64>::zeros(self.k);
            let mut result = Array1::<f64>::zeros(self.k);
            for bj in 0..b {
                probe.fill(0.0);
                probe[cols[bj]] = 1.0;
                result.fill(0.0);
                {
                    let p_slice = probe.as_slice().expect("probe contiguous");
                    let r_slice = result.as_slice_mut().expect("result contiguous");
                    op.matvec(p_slice, r_slice);
                }
                for bi in 0..b {
                    out[[bi, bj]] += result[cols[bi]];
                }
            }
        } else if self.hbb.dim() == (self.k, self.k) {
            for bi in 0..b {
                for bj in 0..b {
                    out[[bi, bj]] += self.hbb[[cols[bi], cols[bj]]];
                }
            }
        } else if let Some(hbb_diag) = self.hbb_diag.as_ref() {
            for bi in 0..b {
                out[[bi, bi]] += hbb_diag[cols[bi]];
            }
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
    /// sparsity kernels) are injected directly. The row-block-only Psi-tier
    /// penalties are `ARDPenalty`, `SparsityPenalty`,
    /// `SoftmaxAssignmentSparsity`, `IBPAssignment`,
    /// `RowPrecisionPrior`, `ParametricRowPrecisionPrior`, and
    /// `ScadMcpPenalty`. Their `d × d` per-row Hessian folds into
    /// `rows[i].htt`, so the exact arrow Schur elimination (`N` independent
    /// `d × d` row solves) represents them exactly. Dense Beta-tier penalties
    /// still fall back to `hvp` probes against the canonical basis vectors for
    /// `β`.
    ///
    /// **Cross-row Psi penalties.** Penalties whose Hessian couples *distinct*
    /// latent rows — `TotalVariationPenalty`, `SheafConsistencyPenalty`,
    /// block-orthogonality, … — produce off-row blocks `∂²P/∂t_i∂t_j`
    /// (`i ≠ j`) that the arrow elimination cannot store, since it assumes each
    /// `H_tt^(i)` is independent of every other row. These are handled without
    /// any approximation: their **gradient** is folded into `g_t` exactly as
    /// for every other Psi penalty (`grad_target → g_t`), and their full
    /// **curvature** is captured into [`Self::cross_row_penalties`] as a
    /// matrix-free operator. At solve time, `K = K0 + P_cross` where `K0` is
    /// the block-diagonal arrow operator and `P_cross · Δt = Σ_p ρ_p ·
    /// psd_majorizer_hvp_p(t, Δt)` is the cross-row penalty Hessian applied to
    /// the full flat latent vector. The presence of any captured cross-row
    /// penalty auto-routes [`Self::solve`] through the matrix-free full-system
    /// PCG path (the exact arrow block-diagonal inverse `K0⁻¹` is the
    /// preconditioner `M⁻¹`); a purely row-block-diagonal system keeps the
    /// exact one-shot Schur path unchanged. No new flag is involved — the route
    /// is selected from the captured penalty set alone (magic by default).
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
        self.cross_row_penalties.clear();
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(ndarray::s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    if analytic_penalty_is_row_block_diagonal(penalty) {
                        // Row-block-diagonal: fold gradient + per-row d×d
                        // curvature into rows[i].htt, exactly representable by
                        // the arrow Schur elimination.
                        self.add_ext_coord_penalty(penalty, target_t, rho_local);
                        if let Some(fingerprint) =
                            analytic_penalty_row_hessian_fingerprint(penalty, target_t, rho_local)
                        {
                            penalty_fingerprints.push(fingerprint);
                        }
                    } else {
                        // Cross-row: fold the gradient into g_t (exact, like
                        // every Psi penalty), but DO NOT fold any curvature into
                        // the row blocks — its off-row coupling cannot be stored
                        // there. Capture the penalty so the solve applies its
                        // full Hessian-vector product P_cross·Δt over the flat
                        // latent vector. This auto-selects the matrix-free
                        // full-system PCG path.
                        self.add_ext_coord_penalty_gradient_only(penalty, target_t, rho_local);
                        self.cross_row_penalties.push(CrossRowLatentPenalty {
                            penalty: penalty.clone(),
                            rho_local: rho_local.to_owned(),
                            target_t: target_t.to_owned(),
                        });
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
        // Cross-row penalties contribute to the Newton Hessian operator, not
        // the stored row blocks, so they must still invalidate the row-Hessian
        // cache when their curvature changes. Probe each captured penalty's PSD
        // majorizer against the current latent vector (a deterministic, generic
        // probe) and fold the resulting signature in.
        for cross in &self.cross_row_penalties {
            penalty_fingerprints.push(cross_row_penalty_fingerprint(
                &cross.penalty,
                target_t,
                cross.rho_local.view(),
            ));
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
            row.gt = manifold.project_gradient_to_tangent(t_i, gt_e.view());
            row.htt = manifold.riemannian_hessian_matrix(t_i, gt_e.view(), htt_e.view());
            row.htbeta = manifold.project_matrix_columns_to_gradient_tangent(
                t_i,
                gt_e.view(),
                htbeta_e.view(),
            );
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

    /// Fold ONLY the latent gradient `grad_target → g_t` of an analytic
    /// penalty, leaving the row-block Hessian untouched.
    ///
    /// Used for cross-row Psi penalties: their gradient enters `g_t` exactly
    /// like every other Psi penalty, but their curvature must NOT be scattered
    /// into the per-row `H_tt^(i)` blocks (the diagonal piece would be
    /// double-counted and the off-row coupling cannot be stored there). The
    /// full curvature is instead applied as a matrix-free `P_cross · Δt`
    /// during the solve, via [`Self::cross_row_penalties`].
    fn add_ext_coord_penalty_gradient_only(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_t: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        assert_eq!(target_t.len(), n * d);
        let grad = penalty.grad_target(target_t, rho_local);
        for flat in 0..n * d {
            self.rows[flat / d].gt[flat % d] += grad[flat];
        }
    }

    /// Apply the aggregate cross-row penalty Hessian `P_cross · v` over the
    /// full flat latent vector `v` (length `Σ_i row_dims[i]`), accumulating
    /// into `out`.
    ///
    /// `P_cross = Σ_p psd_majorizer_hvp_p(target_t, ·; ρ_p)` summed over every
    /// captured cross-row penalty. Each penalty's `psd_majorizer_hvp` is its
    /// exact (PSD) Hessian-vector product over the `N·d` flat latent vector —
    /// for `TotalVariationPenalty` this is `Dᵀ diag(φ''(D t)) D · v`, the
    /// graph/forward-difference Laplacian-style coupling that links distinct
    /// rows. The ρ scaling is already baked into each penalty's resolved
    /// weight, so no extra factor is applied here.
    ///
    /// This is only valid for homogeneous systems (every row of dimension
    /// `d`), the only shape cross-row latent penalties are defined on; the
    /// flat-index convention `flat = i·d + j` matches every penalty's
    /// `latent_dim`/row-major contract.
    fn apply_cross_row_penalty_hessian(&self, v: ArrayView1<'_, f64>, out: &mut Array1<f64>) {
        for cross in &self.cross_row_penalties {
            assert_eq!(cross.target_t.len(), v.len());
            let hv =
                cross
                    .penalty
                    .psd_majorizer_hvp(cross.target_t.view(), cross.rho_local.view(), v);
            assert_eq!(hv.len(), out.len());
            for i in 0..out.len() {
                out[i] += hv[i];
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

    /// Schur-eliminate the per-row latent block and solve for `(Δt, Δβ, diag)`.
    ///
    /// This uses [`ArrowSolveOptions::automatic`]: BA dense RCS for
    /// `K <= 2000`, and Agarwal-style inexact Schur PCG above that size.
    /// Call [`ArrowSchurSystem::solve_with_options`] to force Square-Root BA
    /// or a specific inexact solve policy.
    ///
    /// Returns `(delta_t, delta_beta, PcgDiagnostics)` with `delta_t` flat
    /// row-major of length `N · d` and `delta_beta` of length `K`. The sign
    /// convention matches `solve_newton_direction_dense`: the returned
    /// increments satisfy the bordered system with RHS `[-g_t; -g_β]`, i.e.
    /// they are the *negated* solutions of the standard Newton-direction
    /// formulation. `PcgDiagnostics` is zero-valued for the Direct path and
    /// carries live counters (PCG iters, ridge escalations, residual) for
    /// InexactPCG.
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
    ) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
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
    ) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
        let options = ArrowSolveOptions::automatic(self.k);
        solve_with_lm_escalation_inner(self, ridge_t, ridge_beta, &options)
    }

    /// Solve with an explicit BA Schur mode, returning `(Δt, Δβ, PcgDiagnostics)`.
    ///
    /// [`ArrowSolverMode::Direct`] is the classic dense reduced-camera-system
    /// Cholesky path; [`ArrowSolverMode::SqrtBA`] forms the same dense system
    /// through Square-Root BA factors; [`ArrowSolverMode::InexactPCG`] runs
    /// inexact-step LM on the reduced system with Jacobi-preconditioned
    /// Steihaug-CG. `PcgDiagnostics` is zero-valued for Direct/SqrtBA and
    /// carries live counters for InexactPCG (iterations, matvec calls,
    /// preconditioner escalations, final relative residual, stopping reason).
    pub fn solve_with_options(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
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
    /// Procedural cross-block operator `H_tβ^(i) x`. When present, the dense
    /// per-row `H_tβ` slabs are never materialized: `accumulate_chunk` and
    /// `back_substitute` probe this operator column-by-column to apply the
    /// cross-block, matching the Kronecker / matrix-free assembly path. When
    /// `None` (legacy dense BA callers), the per-row `row.htbeta` slab is used.
    htbeta_matvec: Option<RowHtbetaMatvec>,
    /// Sparse adjoint of `htbeta_matvec`. When present, `row_htbeta` rebuilds
    /// the dense `(d_i × K)` cross-block by probing the transpose with `d_i`
    /// basis vectors — `O(d_i · m_i · p)` total, vs the `O(K · m_i · p)` cost
    /// of probing the forward operator with `K` basis vectors. Since
    /// `d_i ≪ K`, this is the per-row sparse apply that replaces the `O(K)`
    /// column-probe in the streaming reduced-Schur accumulation.
    htbeta_transpose_matvec: Option<RowHtbetaTransposeMatvec>,
    /// Lift the per-row κ rejection for evidence/log-det-only solves; see
    /// [`ArrowSolveOptions::tolerate_ill_conditioning`]. Set by [`Self::solve`]
    /// from the options; defaults to `false` so direct callers of
    /// [`Self::accumulate_chunk`] keep the full guard.
    tolerate_ill_conditioning: bool,
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
            htbeta_matvec: None,
            htbeta_transpose_matvec: None,
            tolerate_ill_conditioning: false,
        }
    }

    #[must_use]
    pub fn from_system(sys: &ArrowSchurSystem, chunk_size: usize) -> Self {
        // When a Kronecker / matrix-free htbeta_matvec is installed, the dense
        // row.htbeta slabs may be zero-sized.  Rather than materialize every
        // `(d × K)` slab (the very `(N·K)`-scale buffer the streaming path
        // exists to avoid), retain the procedural operator and probe it per row
        // inside `accumulate_chunk` / `back_substitute`.  The row builder then
        // only carries the small `H_tt` / `g_t` blocks.
        let htbeta_matvec = sys.htbeta_matvec.clone();
        let rows: Vec<ArrowRowBlock> = if htbeta_matvec.is_some() {
            sys.rows
                .iter()
                .map(|row| ArrowRowBlock {
                    htt: row.htt.clone(),
                    htbeta: Array2::<f64>::zeros((0, 0)),
                    gt: row.gt.clone(),
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
        // Materialize the dense β-block from the effective penalty operator so
        // the streaming accumulator stays correct when contributions live in a
        // structured `BetaPenaltyOp` (e.g. the SAE data-fit Gauss-Newton block,
        // represented as `G ⊗ I_p`) rather than the dense `hbb` accumulator.
        // When no `penalty_op` is installed this reduces to `hbb.clone()`.
        let hbb_dense = sys.effective_penalty_op().to_dense();
        let mut streaming = Self::new(
            sys.rows.len(),
            sys.d,
            Arc::clone(&sys.row_dims),
            Arc::clone(&sys.row_offsets),
            sys.k,
            hbb_dense,
            sys.gb.clone(),
            row_builder,
            chunk_size,
        );
        streaming.htbeta_matvec = htbeta_matvec;
        streaming.htbeta_transpose_matvec = sys.htbeta_transpose_matvec.clone();
        streaming
    }

    /// Build the `(di × k)` cross-block for `row_idx` on demand.
    ///
    /// When the sparse transpose adjoint is installed, probes it with `di`
    /// standard basis vectors — each yields a full `K`-row of `H_βt^(i)`
    /// (i.e. a row of the `(di × k)` block) via the sparse scatter, for
    /// `O(di · m_i · p)` total, far below the `O(K · m_i · p)` cost of probing
    /// the forward operator with `K` basis vectors when `di ≪ K`.
    ///
    /// When only the forward operator is installed (no adjoint), falls back to
    /// the `k`-column forward probe. Otherwise clones the dense `row.htbeta`
    /// slab.
    fn row_htbeta(&self, row_idx: usize, row: &ArrowRowBlock, di: usize) -> Array2<f64> {
        if let Some(op_t) = self.htbeta_transpose_matvec.as_ref() {
            // Probe the adjoint: for each latent index c, scatter e_c to obtain
            // row c of the (di × k) block.
            let mut mat = Array2::<f64>::zeros((di, self.k));
            let mut e_c = Array1::<f64>::zeros(di);
            let mut beta_row = Array1::<f64>::zeros(self.k);
            for c in 0..di {
                e_c.fill(0.0);
                e_c[c] = 1.0;
                beta_row.fill(0.0);
                op_t(row_idx, e_c.view(), &mut beta_row);
                for a in 0..self.k {
                    mat[[c, a]] = beta_row[a];
                }
            }
            return mat;
        }
        match self.htbeta_matvec.as_ref() {
            Some(op) => {
                let mut mat = Array2::<f64>::zeros((di, self.k));
                let mut e_a = Array1::<f64>::zeros(self.k);
                let mut col = Array1::<f64>::zeros(di);
                for a in 0..self.k {
                    e_a.fill(0.0);
                    e_a[a] = 1.0;
                    col.fill(0.0);
                    op(row_idx, e_a.view(), &mut col);
                    for c in 0..di {
                        mat[[c, a]] = col[c];
                    }
                }
                mat
            }
            None => row.htbeta.clone(),
        }
    }

    /// Move out the accumulated reduced Schur block `s_acc` and reduced RHS
    /// `rhs_acc`, leaving fresh zero buffers in their place.
    ///
    /// The reduced contribution is `s_acc = hbb − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`
    /// (the β-block `hbb` seeded by `reset_accumulator`, minus the per-row
    /// reduction summed by `accumulate_chunk`) and
    /// `rhs_acc = +Σ_i H_βt^(i)(H_tt^(i))⁻¹g_t^(i)`. Used by external online
    /// drivers (e.g. the SAE streaming joint fit) that accumulate the reduced
    /// system across re-materialized chunk systems.
    #[must_use]
    pub fn take_accumulators(&mut self) -> (Array2<f64>, Array1<f64>) {
        let s = std::mem::replace(&mut self.s_acc, Array2::<f64>::zeros((self.k, self.k)));
        let rhs = std::mem::replace(&mut self.rhs_acc, Array1::<f64>::zeros(self.k));
        (s, rhs)
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
            let htbeta = self.row_htbeta(row_idx, &row, di);
            let factor =
                factor_one_row(&row, ridge_t, di, row_idx, self.tolerate_ill_conditioning)?;
            let v = backend.solve_block_vector(factor.view(), row.gt.view());
            for c in 0..di {
                let vc = v[c];
                if vc == 0.0 {
                    continue;
                }
                for a in 0..self.k {
                    self.rhs_acc[a] += htbeta[[c, a]] * vc;
                }
            }
            match mode {
                // The streaming accumulator forms the dense reduced Schur
                // complement `S = H_ββ + ridge·I − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`
                // incrementally across chunks. `InexactPCG` differs from
                // `Direct` only in how the *reduced* system is solved, not in
                // how it is assembled — and `solve_dense_reduced_system` already
                // owns the reduced solve. So InexactPCG reduces, by construction,
                // to the same dense Schur subtraction here; the prior hard
                // rejection at this site is lifted because chunked assembly is
                // exactly the matrix-free reduction the PCG path wants.
                ArrowSolverMode::Direct | ArrowSolverMode::InexactPCG => {
                    let solved = backend.solve_block_matrix(factor.view(), htbeta.view());
                    backend.block_gemm_subtract(&mut self.s_acc, &htbeta, &solved);
                }
                ArrowSolverMode::SqrtBA => {
                    let whitened = backend.sqrt_solve_block_matrix(factor.view(), htbeta.view());
                    backend.block_gemm_subtract(&mut self.s_acc, &whitened, &whitened);
                }
            }
        }
        Ok(())
    }

    /// Compute the exact arrow Hessian log-determinant by accumulating the
    /// reduced Schur complement in row chunks, without retaining the full set
    /// of per-row Cholesky factors.
    ///
    /// This is the streaming analogue of [`ArrowFactorCache::arrow_log_det`]:
    ///
    /// ```text
    /// log|H| = Σ_i log|H_tt^(i)| + log|H_ββ - Σ_i H_βt^(i) H_tt^(i)⁻¹ H_tβ^(i)|.
    /// ```
    ///
    /// The same row builder and procedural `H_tβ` callbacks used by the
    /// streaming Newton solve are consumed here, so callers can score REML
    /// evidence without materialising the full `(N × q × K)` cross block or
    /// the full list of row factors.
    pub fn reduced_schur_and_log_det_tt(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(f64, Array2<f64>), ArrowSchurError> {
        self.tolerate_ill_conditioning = options.tolerate_ill_conditioning;
        self.reset_accumulator(ridge_beta)?;
        let backend = CpuBatchedBlockSolver;
        let mut log_det_tt = 0.0_f64;
        for start in (0..self.n_rows).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(self.n_rows);
            for row_idx in start..end {
                let row = (self.row_builder)(row_idx)?;
                let di = row.htt.nrows();
                self.validate_row(row_idx, &row)?;
                let htbeta = self.row_htbeta(row_idx, &row, di);
                let factor =
                    factor_one_row(&row, ridge_t, di, row_idx, self.tolerate_ill_conditioning)?;
                for axis in 0..di {
                    log_det_tt += 2.0 * factor[[axis, axis]].ln();
                }
                match options.mode {
                    ArrowSolverMode::Direct | ArrowSolverMode::InexactPCG => {
                        let solved = backend.solve_block_matrix(factor.view(), htbeta.view());
                        backend.block_gemm_subtract(&mut self.s_acc, &htbeta, &solved);
                    }
                    ArrowSolverMode::SqrtBA => {
                        let whitened =
                            backend.sqrt_solve_block_matrix(factor.view(), htbeta.view());
                        backend.block_gemm_subtract(&mut self.s_acc, &whitened, &whitened);
                    }
                }
            }
        }
        symmetrize_upper_from_lower(&mut self.s_acc);
        let schur = std::mem::replace(&mut self.s_acc, Array2::<f64>::zeros((self.k, self.k)));
        Ok((log_det_tt, schur))
    }

    pub fn reduced_schur_log_det(
        schur: &Array2<f64>,
        options: &ArrowSolveOptions,
    ) -> Result<f64, ArrowSchurError> {
        let rhs = Array1::<f64>::zeros(schur.nrows());
        let trust_metric_weights = None;
        let (delta, schur_factor, diag) =
            solve_dense_reduced_system(schur, &rhs, options, trust_metric_weights)?;
        if delta.len() != schur.nrows() || diag.iterations != 0 {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "streaming log-det reduced solve returned incoherent diagnostics"
                    .to_string(),
            });
        }
        let schur_factor = schur_factor.ok_or_else(|| ArrowSchurError::SchurFactorFailed {
            reason: "streaming log-det requires a dense reduced Schur factor".to_string(),
        })?;
        let mut log_det_schur = 0.0_f64;
        for axis in 0..schur_factor.nrows() {
            log_det_schur += 2.0 * schur_factor[[axis, axis]].ln();
        }
        Ok(log_det_schur)
    }

    pub fn exact_arrow_log_det(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<f64, ArrowSchurError> {
        let (log_det_tt, schur) =
            self.reduced_schur_and_log_det_tt(ridge_t, ridge_beta, options)?;
        Ok(log_det_tt + Self::reduced_schur_log_det(&schur, options)?)
    }

    pub fn solve(
        &mut self,
        ridge_t: f64,
        ridge_beta: f64,
        options: &ArrowSolveOptions,
    ) -> Result<(Array1<f64>, Array1<f64>, Option<Array2<f64>>), ArrowSchurError> {
        // Propagate the evidence/log-det ill-conditioning tolerance to the
        // per-row factor calls inside `accumulate_chunk` / `back_substitute`,
        // which take their stable public signatures. Direct callers of
        // `accumulate_chunk` keep the conservative default (`false`, full guard).
        self.tolerate_ill_conditioning = options.tolerate_ill_conditioning;
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
        let (delta_beta, schur_factor, _diag) =
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
                let factor =
                    factor_one_row(&row, ridge_t, di, row_idx, self.tolerate_ill_conditioning)?;
                // `H_tβ^(i) Δβ`: route through the procedural operator when
                // present (no dense slab), else through the dense slab.
                let mut htbeta_delta = Array1::<f64>::zeros(di);
                if let Some(op) = self.htbeta_matvec.as_ref() {
                    op(row_idx, delta_beta, &mut htbeta_delta);
                } else {
                    for c in 0..di {
                        let mut acc = 0.0_f64;
                        for a in 0..self.k {
                            acc += row.htbeta[[c, a]] * delta_beta[a];
                        }
                        htbeta_delta[c] = acc;
                    }
                }
                for c in 0..di {
                    rhs[c] = row.gt[c] + htbeta_delta[c];
                }
                let dt_i = backend.solve_block_vector(factor.view(), rhs.view());
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
        // The dense `H_tβ` slab is only validated when no procedural operator is
        // installed; with `htbeta_matvec` the slab is intentionally zero-sized
        // and the cross-block is probed in `row_htbeta`.
        if self.htbeta_matvec.is_none() && row.htbeta.dim() != (expected_di, self.k) {
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
                reason: format!("streaming row g_t length {} != {expected_di}", row.gt.len()),
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

    // The scattered curvature lands in the arrow-Schur `H_tt` / `H_ββ` blocks,
    // which are Cholesky-factored (with LM ridge escalation) as the Newton /
    // PIRLS curvature operator and must therefore stay PSD. Nonconvex
    // sparsifiers (log sparsity, JumpReLU) have an *indefinite* exact Hessian
    // that would destroy that positive-definiteness, so we scatter the PSD
    // majorizer here — never the exact `hessian_diag` / `hvp`. For convex
    // penalties the majorizer equals the exact Hessian (the trait default
    // delegates), so this is exact for them. Exact-derivative consumers (the
    // outer objective Hessian) use `hessian_diag` / `hvp` directly elsewhere.
    if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
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
        let hv = penalty.psd_majorizer_hvp(target, rho_local, probe.view());
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
pub struct ArrowFactorSlab {
    data: Arc<[f64]>,
    offsets: Arc<[usize]>,
    dims: Arc<[usize]>,
}

impl ArrowFactorSlab {
    pub fn from_blocks(blocks: Vec<Array2<f64>>) -> Self {
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        let mut dims = Vec::with_capacity(blocks.len());
        offsets.push(0);
        for block in blocks {
            let (rows, cols) = block.dim();
            assert_eq!(rows, cols, "ArrowFactorSlab stores square row factors");
            dims.push(rows);
            data.extend(block.iter().copied());
            offsets.push(data.len());
        }
        Self {
            data: data.into(),
            offsets: offsets.into(),
            dims: dims.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn factor(&self, row: usize) -> ArrayView2<'_, f64> {
        let dim = self.dims[row];
        let range = self.offsets[row]..self.offsets[row + 1];
        ArrayView2::from_shape((dim, dim), &self.data[range])
            .expect("ArrowFactorSlab row offset/dim invariant violated")
    }

    pub fn iter(&self) -> impl Iterator<Item = ArrayView2<'_, f64>> + '_ {
        (0..self.len()).map(|row| self.factor(row))
    }
}

impl std::fmt::Debug for ArrowFactorSlab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowFactorSlab")
            .field("rows", &self.len())
            .field("values", &self.data.len())
            .finish()
    }
}

#[derive(Clone)]
pub enum ArrowUndampedFactors {
    SameAsDamped,
    Owned(ArrowFactorSlab),
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
/// Sums the installed matrix-free operator, when present, and any correctly
/// shaped dense `row.htbeta` slab. This lets structured data-fit rows coexist
/// with dense analytic-penalty cross blocks on the same row.
fn sys_htbeta_apply_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    x: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    out.fill(0.0);
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        op(row_idx, x, out);
    }
    if (sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none())
        && row.htbeta.dim() == (out.len(), sys.k)
    {
        let di = row.htbeta.nrows();
        for c in 0..di {
            let mut acc = 0.0_f64;
            for a in 0..sys.k {
                acc += row.htbeta[[c, a]] * x[a];
            }
            out[c] += acc;
        }
    }
}

/// Accumulate `H_βt^(row) · v` into `out` (length `k`).
///
/// `out[a] += Σ_c H_tβ^(row)[c, a] · v[c]`
///
/// Sums the installed matrix-free operator, when present, and any correctly
/// shaped dense `row.htbeta` slab.
fn sys_htbeta_accumulate_transpose(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    v: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        htbeta_probe_transpose(row_idx, op, v, out, v.len(), sys.k);
    }
    if (sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none())
        && row.htbeta.dim() == (v.len(), sys.k)
    {
        let di = row.htbeta.nrows();
        for c in 0..di {
            let vc = v[c];
            if vc == 0.0 {
                continue;
            }
            for a in 0..sys.k {
                out[a] += row.htbeta[[c, a]] * vc;
            }
        }
    }
}

/// Materialize the dense `(di, k)` cross-block for one row.
///
/// Materializes the sum of the installed matrix-free operator and any correctly
/// shaped dense slab on the row.
fn sys_htbeta_materialize_row(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
) -> Array2<f64> {
    let di = sys.row_dims[row_idx];
    let k = sys.k;
    let use_dense = sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none();
    let mut mat = if use_dense && row.htbeta.dim() == (di, k) {
        row.htbeta.clone()
    } else {
        Array2::<f64>::zeros((di, k))
    };
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        let mut e_a = Array1::<f64>::zeros(k);
        let mut col = Array1::<f64>::zeros(di);
        for a in 0..k {
            e_a.fill(0.0);
            e_a[a] = 1.0;
            col.fill(0.0);
            op(row_idx, e_a.view(), &mut col);
            for c in 0..di {
                mat[[c, a]] += col[c];
            }
        }
    } else if use_dense && row.htbeta.dim() != (di, k) {
        // SAFETY: reaching here means the assembler installed neither a
        // correctly-shaped (di, k) dense H_tβ block nor an htbeta_matvec
        // operator — a construction-time invariant violation in the caller, not
        // recoverable runtime input. The cross-block is mandatory for the Schur
        // reduction and there is no meaningful fallback, so this is a hard bug.
        panic!(
            "row {row_idx}: htbeta shape {:?} != ({di}, {k}) and no htbeta_matvec installed",
            row.htbeta.dim()
        );
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
    pub htt_factors: ArrowFactorSlab,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrowFactorMinPivot {
    pub min_row_pivot: Option<f64>,
    pub min_schur_pivot: Option<f64>,
    pub min_pivot: Option<f64>,
}

impl ArrowFactorMinPivot {
    fn combine(row: Option<f64>, schur: Option<f64>) -> Self {
        let min_pivot = match (row, schur) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        Self {
            min_row_pivot: row,
            min_schur_pivot: schur,
            min_pivot,
        }
    }
}

fn lower_cholesky_min_pivot(factor: ArrayView2<'_, f64>) -> Option<f64> {
    let width = factor.nrows().min(factor.ncols());
    let mut out = None;
    for idx in 0..width {
        let pivot = factor[[idx, idx]] * factor[[idx, idx]];
        out = Some(match out {
            Some(current) => f64::min(current, pivot),
            None => pivot,
        });
    }
    out
}

fn lower_cholesky_max_pivot(factor: ArrayView2<'_, f64>) -> Option<f64> {
    let width = factor.nrows().min(factor.ncols());
    let mut out = None;
    for idx in 0..width {
        let pivot = factor[[idx, idx]] * factor[[idx, idx]];
        out = Some(match out {
            Some(current) => f64::max(current, pivot),
            None => pivot,
        });
    }
    out
}

/// Smallest cached Cholesky pivot for row blocks and the dense Schur factor.
///
/// Pivots are returned as squared lower-factor diagonals, matching the Hessian
/// scale rather than the Cholesky-factor scale. In inexact PCG mode the dense
/// Schur factor is absent, so `min_schur_pivot` is `None`.
pub fn arrow_factor_min_pivot(cache: &ArrowFactorCache) -> ArrowFactorMinPivot {
    let mut min_row_pivot = None;
    for factor in cache.htt_factors.iter() {
        if let Some(pivot) = lower_cholesky_min_pivot(factor) {
            min_row_pivot = Some(match min_row_pivot {
                Some(current) => f64::min(current, pivot),
                None => pivot,
            });
        }
    }
    let min_schur_pivot = cache
        .schur_factor
        .as_ref()
        .and_then(|factor| lower_cholesky_min_pivot(factor.view()));
    ArrowFactorMinPivot::combine(min_row_pivot, min_schur_pivot)
}

impl ArrowFactorCache {
    pub fn n_rows(&self) -> usize {
        self.htt_factors.len()
    }

    pub fn htbeta_available(&self) -> bool {
        self.htbeta.is_available()
    }

    pub fn undamped_factor(&self, row: usize) -> ArrayView2<'_, f64> {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => self.htt_factors.factor(row),
            ArrowUndampedFactors::Owned(factors) => factors.factor(row),
        }
    }

    pub fn undamped_factor_count(&self) -> usize {
        match &self.htt_factors_undamped {
            ArrowUndampedFactors::SameAsDamped => self.htt_factors.len(),
            ArrowUndampedFactors::Owned(factors) => factors.len(),
        }
    }

    pub fn undamped_factors_iter(&self) -> impl Iterator<Item = ArrayView2<'_, f64>> + '_ {
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

    /// Diagonal of the latent (`t`-block) of the *full* bordered-arrow
    /// inverse `(H⁻¹)_tt`, in `delta_t` layout (length [`Self::delta_t_len`]).
    ///
    /// For the bordered arrow Hessian
    /// `H = [[A, B], [Bᵀ, H_ββ]]` with `A = H_tt` (block-diagonal per row,
    /// `A_i = H_tt^(i)`) and `B = H_tβ`, the standard block-inverse identity
    /// gives the `t`-block
    /// `(H⁻¹)_tt = A⁻¹ + A⁻¹ B S⁻¹ Bᵀ A⁻¹`, where
    /// `S = H_ββ − Bᵀ A⁻¹ B` is the Schur complement on `β`. Because `A` is
    /// block-diagonal, the `(i, j)` diagonal entry of `(H⁻¹)_tt` is computed
    /// purely from row `i`'s factor and cross-block:
    ///
    /// ```text
    /// a    = A_i⁻¹ e_j                       (chol_solve on the per-row factor)
    /// [A_i⁻¹]_{jj} = a[j]
    /// w    = B_iᵀ a = H_βt^(i) a             (a K-vector)
    /// z    = S⁻¹ w                           (chol_solve on the Schur factor)
    /// diag = a[j] + w · z
    /// ```
    ///
    /// The UNDAMPED per-row factors ([`Self::undamped_factor`]) are used so
    /// the result is the inverse of the *true* `H_tt`, not the LM-damped
    /// `H_tt + ridge_t·I` — same rationale the IFT predictor docstring gives
    /// at the top of this struct.
    ///
    /// # Consuming the diagonal as a per-(atom, axis) trace
    ///
    /// `(H⁻¹)_tt` is the latent covariance block. The selected-inverse trace
    /// for a contiguous group of latent coordinates (e.g. one atom's rows, or
    /// one axis across rows) is simply the sum of the returned diagonal entries
    /// over those `row_offsets[i] + j` indices — no off-diagonal terms are
    /// needed for the trace `tr[(H⁻¹)_tt · D]` against a diagonal selector `D`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowSchurError::SchurFactorFailed`] when this cache has no
    /// dense Schur factor or no usable `H_βt` coupling — i.e. it was produced
    /// by an [`ArrowSolverMode::InexactPCG`] solve (no dense `K × K` factor) or
    /// by a `Disabled` `htbeta` cache. The selected-inverse block-trace is not
    /// yet supported for the matrix-free PCG mode; that branch needs a separate
    /// Lanczos/Hutchinson estimator.
    pub fn latent_block_inverse_diagonal(&self) -> Result<Array1<f64>, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "latent_block_inverse_diagonal requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if !self.htbeta_available() {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "latent_block_inverse_diagonal requires the H_tβ coupling, \
                         but this cache's htbeta is Disabled"
                    .to_string(),
            });
        }
        let n = self.undamped_factor_count();
        let total_len = self.delta_t_len();
        let mut out = Array1::<f64>::zeros(total_len);
        // Per-row scratch, sized to the max latent dim / K.
        let mut e_j = Array1::<f64>::zeros(self.d);
        let mut w = Array1::<f64>::zeros(self.k);
        for i in 0..n {
            let di = self.row_dims[i];
            let row_base = self.row_offsets[i];
            let factor = self.undamped_factor(i);
            for j in 0..di {
                // a = A_i⁻¹ e_j.
                for c in 0..di {
                    e_j[c] = 0.0;
                }
                e_j[j] = 1.0;
                let e_j_slice = e_j.slice(ndarray::s![..di]).to_owned();
                let a = cholesky_solve_vector(factor, &e_j_slice);
                // w = H_βt^(i) a (a K-vector); accumulator must start zeroed.
                w.fill(0.0);
                if !self.apply_htbeta_row_transpose(i, a.view(), &mut w, None) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "latent_block_inverse_diagonal: H_βt^({i}) apply failed \
                             (htbeta cache could not supply row {i})"
                        ),
                    });
                }
                // z = S⁻¹ w; correction = w · z.
                let z = cholesky_solve_vector(schur_factor, &w);
                let mut corr = 0.0_f64;
                for c in 0..self.k {
                    corr += w[c] * z[c];
                }
                out[row_base + j] = a[j] + corr;
            }
        }
        Ok(out)
    }

    /// Solve the full bordered-arrow system `H·u = w` on the cached factor
    /// (#1006): `w` arrives in arrow layout — `w_t` flat per
    /// [`Self::delta_t_len`] / `row_offsets`, `w_beta` of length `K` — and the
    /// solution comes back in the same layout. Standard block elimination on
    /// the SAME factors whose log-determinant the evidence reports:
    ///
    /// ```text
    ///   y_i      = H_tt^(i)⁻¹ · w_t^(i)
    ///   r_β      = w_β − Σ_i H_βt^(i) · y_i
    ///   u_β      = Schur⁻¹ · r_β
    ///   u_t^(i)  = y_i − H_tt^(i)⁻¹ · (H_tβ^(i) · u_β)
    /// ```
    ///
    /// This is the IFT / adjoint back-solve the analytic outer ρ-gradient
    /// consumes: `u_j = H⁻¹ (∂g/∂ρ_j)` per outer coordinate and the
    /// `H⁻¹`-side of the third-order correction `−½·Γᵀ·H⁻¹·(∂g/∂ρ_j)`.
    /// Contract: the cache must be the ridge-0 Direct evidence factor
    /// (undamped per-row factors + dense Schur), so the solve is against the
    /// criterion's own `H` — never a damped surrogate (that would desync the
    /// gradient from the reported evidence).
    pub fn solve_full(
        &self,
        w_t: ArrayView1<'_, f64>,
        w_beta: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let total_len = self.delta_t_len();
        if w_t.len() != total_len || w_beta.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "solve_full: rhs shapes (w_t={}, w_beta={}) != (delta_t_len={}, K={})",
                    w_t.len(),
                    w_beta.len(),
                    total_len,
                    self.k
                ),
            });
        }
        let n = self.undamped_factor_count();
        // Forward pass: y_i = H_tt^(i)⁻¹ w_t^(i), accumulating the border RHS.
        let mut y = Array1::<f64>::zeros(total_len);
        let mut r_beta = w_beta.to_owned();
        for i in 0..n {
            let di = self.row_dims[i];
            let base = self.row_offsets[i];
            let factor = self.undamped_factor(i);
            let w_row = w_t.slice(ndarray::s![base..base + di]).to_owned();
            let y_row = cholesky_solve_vector(factor, &w_row);
            if self.k > 0 {
                // r_β −= H_βt^(i) y_i: accumulate into a scratch then subtract,
                // because the helper ACCUMULATES (+=) into its output.
                let mut acc = Array1::<f64>::zeros(self.k);
                if !self.apply_htbeta_row_transpose(i, y_row.view(), &mut acc, None) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "solve_full: H_βt^({i}) apply failed (htbeta cache \
                             could not supply row {i})"
                        ),
                    });
                }
                for c in 0..self.k {
                    r_beta[c] -= acc[c];
                }
            }
            for j in 0..di {
                y[base + j] = y_row[j];
            }
        }
        // Border solve + back-substitution.
        let u_beta = if self.k > 0 {
            self.schur_inverse_apply(r_beta.view())?
        } else {
            Array1::<f64>::zeros(0)
        };
        let mut u_t = y;
        if self.k > 0 {
            let mut cross = Array1::<f64>::zeros(self.d);
            for i in 0..n {
                let di = self.row_dims[i];
                let base = self.row_offsets[i];
                let mut cross_row = cross.slice_mut(ndarray::s![..di]);
                cross_row.fill(0.0);
                let mut cross_owned = cross_row.to_owned();
                if !self.apply_htbeta_row(i, u_beta.view(), &mut cross_owned) {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "solve_full: H_tβ^({i}) apply failed (htbeta cache \
                             could not supply row {i})"
                        ),
                    });
                }
                let factor = self.undamped_factor(i);
                let corr = cholesky_solve_vector(factor, &cross_owned);
                for j in 0..di {
                    u_t[base + j] -= corr[j];
                }
            }
        }
        Ok((u_t, u_beta))
    }

    /// Apply the β-block of the full inverse, `(H⁻¹)_ββ · rhs = S_β⁻¹ · rhs`,
    /// where `S_β` is the Schur complement on β whose Cholesky factor this
    /// cache holds in [`Self::schur_factor`].
    ///
    /// For the bordered arrow Hessian `H = [[A, B], [Bᵀ, H_ββ]]`, the
    /// β-block of `H⁻¹` is exactly the inverse of the Schur complement
    /// `S_β = H_ββ − Bᵀ A⁻¹ B`. One Cholesky back-substitution per call,
    /// reusing the cached factor; `rhs` and the returned vector both have
    /// length `K`.
    ///
    /// This is the general single-solve primitive for the β border. Callers
    /// that need a Schur-inverse trace `tr(S_β⁻¹ M)` against a structured
    /// penalty `M` (e.g. the SAE λ_smooth Fellner-Schall step, where
    /// `M = blockdiag_k(λ_k S_k ⊗ I_p)`) build it as
    /// `Σ_col e_colᵀ S_β⁻¹ M e_col` — apply this to each column of `M`
    /// (exploiting whatever sparsity `M` has) and read off `result[col]`.
    /// Keeping `M`'s layout on the caller side avoids coupling this solver
    /// to penalty-op types.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowSchurError::SchurFactorFailed`] when this cache has no
    /// dense Schur factor (an [`ArrowSolverMode::InexactPCG`] solve) — the
    /// same not-yet-supported branch as [`Self::latent_block_inverse_diagonal`]
    /// — or when `rhs.len() != k`.
    pub fn schur_inverse_apply(
        &self,
        rhs: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_apply requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if rhs.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_apply: rhs length {} != K {}",
                    rhs.len(),
                    self.k
                ),
            });
        }
        let rhs_owned = rhs.to_owned();
        Ok(cholesky_solve_vector(schur_factor, &rhs_owned))
    }

    /// Diagonal of the β-block of the full inverse, `diag((H⁻¹)_ββ) = diag(S_β⁻¹)`,
    /// length `K`.
    ///
    /// Convenience built from `K` Cholesky back-substitutions against the
    /// unit vectors (`[S_β⁻¹]_{jj} = e_jᵀ S_β⁻¹ e_j`), reusing the cached
    /// factor. Useful for the per-β-coordinate effective dof. Same dense-Schur
    /// requirement / error contract as [`Self::schur_inverse_apply`].
    pub fn schur_inverse_diagonal(&self) -> Result<Array1<f64>, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_diagonal requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        let mut out = Array1::<f64>::zeros(self.k);
        let mut e_j = Array1::<f64>::zeros(self.k);
        for j in 0..self.k {
            for c in 0..self.k {
                e_j[c] = 0.0;
            }
            e_j[j] = 1.0;
            let col = cholesky_solve_vector(schur_factor, &e_j);
            out[j] = col[j];
        }
        Ok(out)
    }

    /// Dense principal sub-block of the β-block of the full inverse,
    /// `(H⁻¹)_ββ[block, block] = S_β⁻¹[block, block]`, shape `(W, W)` with
    /// `W = block.len()`.
    ///
    /// For the bordered arrow Hessian `H = [[A, B], [Bᵀ, H_ββ]]`, the β-block
    /// of `H⁻¹` is exactly `S_β⁻¹` (the inverse of the Schur complement whose
    /// Cholesky factor this cache holds). This returns the contiguous
    /// `block × block` sub-block — e.g. one SAE atom's decoder coefficients via
    /// [`crate::terms::sae_manifold::SaeManifoldTerm::beta_block_offsets`] — by
    /// solving `S_β x = e_j` for each `j ∈ block` (reusing the cached factor)
    /// and gathering the `block` rows of each solution column. `W`
    /// back-substitutions of size `K`; the result is symmetrized to clear
    /// back-substitution rounding asymmetry. Up to a dispersion scale `φ`, this
    /// block is the joint posterior covariance `Cov(β_block)` of those
    /// coefficients with the latent coordinates already marginalized out (that
    /// is precisely what Schur-eliminating the per-row `t`-blocks does).
    ///
    /// Same dense-Schur requirement / error contract as
    /// [`Self::schur_inverse_apply`]; additionally errors when `block` runs past
    /// `K`.
    pub fn schur_inverse_block(
        &self,
        block: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, ArrowSchurError> {
        let Some(schur_factor) = self.schur_factor.as_ref() else {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "schur_inverse_block requires a dense Schur factor; \
                         the InexactPCG mode does not form one"
                    .to_string(),
            });
        };
        if block.end > self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "schur_inverse_block: block end {} exceeds K {}",
                    block.end, self.k
                ),
            });
        }
        let w = block.len();
        let mut out = Array2::<f64>::zeros((w, w));
        let mut e_j = Array1::<f64>::zeros(self.k);
        for (jc, j) in block.clone().enumerate() {
            e_j.fill(0.0);
            e_j[j] = 1.0;
            let col = cholesky_solve_vector(schur_factor, &e_j);
            for (ic, i) in block.clone().enumerate() {
                out[[ic, jc]] = col[i];
            }
        }
        // S_β⁻¹ is symmetric; symmetrize to clear back-substitution rounding.
        for ic in 0..w {
            for jc in (ic + 1)..w {
                let avg = 0.5 * (out[[ic, jc]] + out[[jc, ic]]);
                out[[ic, jc]] = avg;
                out[[jc, ic]] = avg;
            }
        }
        Ok(out)
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
    let htt_factors = step.htt_factors;
    let htt_factors_undamped = if ridge_t == 0.0 {
        ArrowUndampedFactors::SameAsDamped
    } else {
        ArrowUndampedFactors::Owned(backend.factor_blocks(
            &sys.rows,
            0.0,
            sys.d,
            options.tolerate_ill_conditioning,
        )?)
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
/// returning `(Δt, Δβ, PcgDiagnostics)`.
///
/// The diagnostics are zero-valued (default) when the selected mode is
/// `Direct` or `SqrtBA` — use them to monitor `InexactPCG` iteration counts
/// and preconditioner escalation in production solves. Callers that do not
/// need diagnostics may pattern-match only the first two tuple elements.
pub fn solve_arrow_newton_step_core(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    if let Some(chunk_size) = options.streaming_chunk_size {
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        return streaming
            .solve(ridge_t, ridge_beta, options)
            .map(|(delta_t, delta_beta, _)| (delta_t, delta_beta, PcgDiagnostics::default()));
    }
    solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)
        .map(|step| (step.delta_t, step.delta_beta, step.pcg_diagnostics))
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
/// `solve`. Adaptive-correction exhaustion surfaces immediately because it is
/// not recoverable by shifting the diagonal.
///
/// A `PcgFailed` is likewise treated as recoverable: when the inexact-PCG
/// path stalls (all preconditioner tiers hit `MaxIter`, negative curvature on
/// an unbounded solve, or a non-PD preconditioned residual), shifting the
/// diagonal improves both conditioning and curvature, so a ridge bump is the
/// right response. Only `AdaptiveCorrectionFailed` surfaces immediately, since
/// it is an option-validation / line-search failure that a ridge shift cannot
/// repair.
///
/// Returns `(Δt, Δβ, PcgDiagnostics)` from `solve_arrow_newton_step_core`,
/// computed with the smallest escalated ridge that produced a successful factor.
/// `PcgDiagnostics::ridge_escalations` records how many ridge bumps were needed.
pub fn solve_with_lm_escalation_inner(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    let mut proximal_ridge = 0.0_f64;
    let mut escalations: usize = 0;
    let mut last_err: Option<ArrowSchurError> = None;
    for attempt in 0..=DEFAULT_PROXIMAL_MAX_ATTEMPTS {
        let damped_ridge_t = ridge_t + proximal_ridge;
        let damped_ridge_beta = ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_artifacts(sys, damped_ridge_t, damped_ridge_beta, options) {
            Ok(mut step) => {
                step.pcg_diagnostics.ridge_escalations = escalations;
                return Ok((step.delta_t, step.delta_beta, step.pcg_diagnostics));
            }
            Err(err) => {
                let recoverable = matches!(
                    err,
                    ArrowSchurError::PerRowFactorFailed { .. }
                        | ArrowSchurError::PerRowFactorIllConditioned { .. }
                        | ArrowSchurError::SchurFactorFailed { .. }
                        | ArrowSchurError::PcgFailed { .. }
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

    // Objective-scale resolution: the floating-point granularity of the
    // penalised objective at the incumbent value. Decreases smaller than this
    // are indistinguishable from rounding noise; increases smaller than this
    // are pure rounding and indicate the incumbent is already a (numerical)
    // stationary point.
    let objective_resolution =
        correction.convergence_objective_rel_tol.max(0.0) * (current_objective_value.abs() + 1.0);

    let mut proximal_ridge = correction.initial_ridge.max(0.0);
    let mut last_reason = String::from("no attempts were made");
    // Best strictly-decreasing trial seen across all ridge attempts. The Armijo
    // sufficient-decrease test can reject a step that nonetheless lowers the
    // objective; in the heavily-damped near-stationary regime, banking any
    // genuine decrease is a valid (relaxed) globalisation, so we retain the
    // best such candidate as a fallback to the strict Armijo accept.
    // Tuple: (delta_t, delta_beta, trial_value, g_dot_p, ridge_t, ridge_beta,
    // proximal_ridge) — the full step record for the best attempt so the
    // returned damping metadata matches the step actually banked.
    let mut best_decrease: Option<(Array1<f64>, Array1<f64>, f64, f64, f64, f64, f64)> = None;
    // Smallest objective INCREASE observed (over attempts that produced a
    // finite trial value but did not decrease). If even the best attempt only
    // raises the objective, but by no more than the objective resolution, the
    // incumbent is numerically stationary and we converge in place.
    let mut smallest_increase = f64::INFINITY;
    for attempt in 0..correction.max_attempts {
        let ridge_t = base_ridge_t + proximal_ridge;
        let ridge_beta = base_ridge_beta + proximal_ridge;
        match solve_arrow_newton_step_core(sys, ridge_t, ridge_beta, options) {
            Ok((delta_t, delta_beta, _diag)) => {
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
                    if trial_value.is_finite() {
                        let delta_obj = trial_value - current_objective_value;
                        if delta_obj < -objective_resolution {
                            // Genuine (Armijo-failing) decrease: keep the best.
                            let improves = best_decrease.as_ref().is_none_or(
                                |(_, _, best_value, _, _, _, _)| trial_value < *best_value,
                            );
                            if improves {
                                best_decrease = Some((
                                    delta_t.clone(),
                                    delta_beta.clone(),
                                    trial_value,
                                    g_dot_p,
                                    ridge_t,
                                    ridge_beta,
                                    proximal_ridge,
                                ));
                            }
                        } else if delta_obj < smallest_increase {
                            smallest_increase = delta_obj;
                        }
                    }
                    last_reason = {
                        let step_norm = (delta_t.iter().map(|v| v * v).sum::<f64>()
                            + delta_beta.iter().map(|v| v * v).sum::<f64>())
                        .sqrt();
                        format!(
                            "Armijo rejected trial objective {trial_value}; bound {armijo_bound}; \
                             |g|={grad_norm:.4e} g.p={g_dot_p:.4e} |step|={step_norm:.4e} ridge={proximal_ridge:.3e}"
                        )
                    };
                }
            }
            Err(err) => {
                last_reason = err.to_string();
            }
        }
        proximal_ridge = next_proximal_ridge(proximal_ridge, correction.ridge_growth);
    }

    // ── Fallback 1: bank the best genuine (Armijo-failing) decrease ──────────
    // Re-apply the best decreasing step so `self` (the caller's state, mutated
    // through the `trial_objective` closure) is left exactly at that step; the
    // returned deltas describe the move from the incumbent.
    if let Some((delta_t, delta_beta, trial_value, g_dot_p, ridge_t, ridge_beta, best_ridge)) =
        best_decrease
    {
        let reapplied = trial_objective(delta_t.view(), delta_beta.view());
        // The closure is deterministic (restore-then-apply), so `reapplied`
        // matches the recorded `trial_value` up to rounding; trust the live
        // value to keep the returned record consistent with `self`'s state.
        let final_value = if reapplied.is_finite() {
            reapplied
        } else {
            trial_value
        };
        return Ok(ArrowAcceptedProximalStep {
            delta_t,
            delta_beta,
            ridge_t,
            ridge_beta,
            proximal_ridge: best_ridge,
            objective_value: current_objective_value,
            trial_objective_value: final_value,
            gradient_dot_step: g_dot_p,
            attempts: correction.max_attempts,
        });
    }

    // ── Fallback 2: near-stationary convergence exit ─────────────────────────
    // No attempt decreased the objective, but the best attempt raised it by no
    // more than the objective's own resolution. The damped Newton model cannot
    // make distinguishable progress: the incumbent is a numerical stationary
    // point. Return a zero step at the incumbent state so the caller accepts it
    // as converged instead of failing. (`smallest_increase` is finite only if
    // at least one descent direction produced a finite trial value.)
    if smallest_increase.is_finite() && smallest_increase <= objective_resolution {
        return Ok(ArrowAcceptedProximalStep {
            delta_t: Array1::<f64>::zeros(sys.row_offsets[sys.rows.len()]),
            delta_beta: Array1::<f64>::zeros(sys.k),
            ridge_t: base_ridge_t,
            ridge_beta: base_ridge_beta,
            proximal_ridge: 0.0,
            objective_value: current_objective_value,
            trial_objective_value: current_objective_value,
            gradient_dot_step: 0.0,
            attempts: correction.max_attempts,
        });
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

    // Route H_ββ · Δβ through penalty_matvec_add (#296):
    // no Arc-clone; dispatches inline to penalty_op or hbb.
    let mut hbb_delta = Array1::<f64>::zeros(sys.k);
    {
        let x_slice = delta_beta
            .as_slice()
            .expect("delta_beta must be contiguous");
        let y_slice = hbb_delta
            .as_slice_mut()
            .expect("hbb_delta must be contiguous");
        sys.penalty_matvec_add(x_slice, y_slice);
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
    let damped =
        arrow_damped_quadratic_model_reduction(sys, delta_t, delta_beta, ridge_t, ridge_beta)?;
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
    htt_factors: ArrowFactorSlab,
    schur_factor: Option<Array2<f64>>,
    pcg_diagnostics: PcgDiagnostics,
}

enum MixedPrecisionAttempt {
    Certified {
        delta_t: Array1<f64>,
        delta_beta: Array1<f64>,
        schur_factor: Array2<f64>,
        refinement_steps: usize,
    },
    Fallback {
        reason: String,
    },
}

fn back_substitute_delta_t<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    delta_beta: ArrayView1<'_, f64>,
    backend: &B,
) -> Array1<f64> {
    let n = sys.rows.len();
    let total_dt_len = sys.row_offsets[n];
    let mut delta_t = Array1::<f64>::zeros(total_dt_len);
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
        sys_htbeta_apply_row(sys, i, &sys.rows[i], delta_beta, &mut htbeta_slice);
        {
            let mut rhs_i = rhs.slice_mut(ndarray::s![..di]);
            for c in 0..di {
                rhs_i[c] = sys.rows[i].gt[c] + htbeta_slice[c];
            }
        }
        let rhs_slice = rhs.slice(ndarray::s![..di]).to_owned();
        let dt_i = backend.solve_block_vector(htt_factors.factor(i), rhs_slice.view());
        for c in 0..di {
            delta_t[row_base + c] = -dt_i[c];
        }
    }
    delta_t
}

fn try_mixed_precision_arrow_solve(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    htt_factors: &ArrowFactorSlab,
    schur: &Array2<f64>,
    options: &ArrowSolveOptions,
) -> Result<Option<MixedPrecisionAttempt>, ArrowSchurError> {
    let MixedPrecisionPolicy::Certified {
        max_refinement_steps,
        residual_relative_tolerance,
        kappa_unit_roundoff_margin,
    } = options.mixed_precision
    else {
        return Ok(None);
    };

    if options.trust_region.radius.is_finite() {
        return Ok(Some(MixedPrecisionAttempt::Fallback {
            reason: "trust-region-truncated dense solves are not certified by the mixed-precision refinement path".to_string(),
        }));
    }

    let schur_factor =
        cholesky_lower(schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
    if !options.tolerate_ill_conditioning {
        let schur_kappa = cholesky_factor_kappa_estimate(&schur_factor);
        if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)^-1 contamination would yield an inaccurate delta_beta"
                ),
            });
        }
    }

    if let Some(reason) =
        mixed_precision_kappa_gate_failure(htt_factors, &schur_factor, kappa_unit_roundoff_margin)
    {
        return Ok(Some(MixedPrecisionAttempt::Fallback { reason }));
    }

    let row_factors_f32 = arrow_factor_slab_to_f32(htt_factors);
    let schur_factor_f32 = schur_factor.mapv(|v| v as f32);
    let (rhs_t, rhs_beta) = arrow_rhs(sys);
    let mut x = solve_arrow_system_f32(
        sys,
        &row_factors_f32,
        &schur_factor_f32,
        rhs_t.view(),
        rhs_beta.view(),
    );
    let certificate_tol = residual_relative_tolerance.max(64.0 * f64::EPSILON);
    for refinement_steps in 0..=max_refinement_steps {
        let (res_t, res_beta) = arrow_residual(
            sys,
            ridge_t,
            ridge_beta,
            x.0.view(),
            x.1.view(),
            rhs_t.view(),
            rhs_beta.view(),
        );
        let certificate = arrow_backward_error_certificate(
            sys,
            ridge_t,
            ridge_beta,
            x.0.view(),
            x.1.view(),
            rhs_t.view(),
            rhs_beta.view(),
            res_t.view(),
            res_beta.view(),
        );
        if certificate <= certificate_tol {
            return Ok(Some(MixedPrecisionAttempt::Certified {
                delta_t: x.0,
                delta_beta: x.1,
                schur_factor,
                refinement_steps,
            }));
        }
        if refinement_steps == max_refinement_steps {
            return Ok(Some(MixedPrecisionAttempt::Fallback {
                reason: format!(
                    "f64 residual certificate did not converge after {max_refinement_steps} refinement steps \
                     (backward_error={certificate:e}, tolerance={certificate_tol:e})"
                ),
            }));
        }
        let correction = solve_arrow_system_f32(
            sys,
            &row_factors_f32,
            &schur_factor_f32,
            res_t.view(),
            res_beta.view(),
        );
        if !correction
            .0
            .iter()
            .chain(correction.1.iter())
            .all(|v| v.is_finite())
        {
            return Ok(Some(MixedPrecisionAttempt::Fallback {
                reason: "f32 refinement correction produced a non-finite value".to_string(),
            }));
        }
        for i in 0..x.0.len() {
            x.0[i] += correction.0[i];
        }
        for i in 0..x.1.len() {
            x.1[i] += correction.1[i];
        }
    }

    unreachable!("mixed refinement loop returns on certification, fallback, or max-step exhaustion")
}

fn mixed_precision_kappa_gate_failure(
    htt_factors: &ArrowFactorSlab,
    schur_factor: &Array2<f64>,
    margin: f64,
) -> Option<String> {
    let mut max_kappa = cholesky_factor_kappa_estimate(schur_factor);
    let mut min_pivot = lower_cholesky_min_pivot(schur_factor.view());
    let mut max_pivot = lower_cholesky_max_pivot(schur_factor.view());
    for factor in htt_factors.iter() {
        let owned = factor.to_owned();
        max_kappa = max_kappa.max(cholesky_factor_kappa_estimate(&owned));
        if let Some(pivot) = lower_cholesky_min_pivot(owned.view()) {
            min_pivot = Some(match min_pivot {
                Some(current) => current.min(pivot),
                None => pivot,
            });
        }
        if let Some(pivot) = lower_cholesky_max_pivot(owned.view()) {
            max_pivot = Some(match max_pivot {
                Some(current) => current.max(pivot),
                None => pivot,
            });
        }
    }
    if let (Some(min_pivot), Some(max_pivot)) = (min_pivot, max_pivot) {
        if min_pivot > 0.0 && max_pivot.is_finite() {
            max_kappa = max_kappa.max(max_pivot / min_pivot);
        } else {
            max_kappa = f64::INFINITY;
        }
    }
    let kappa_u = max_kappa * F32_UNIT_ROUNDOFF;
    let threshold = margin.min(1.0).max(F32_UNIT_ROUNDOFF);
    if !(max_kappa.is_finite() && kappa_u < threshold) {
        Some(format!(
            "kappa gate refused f32 refinement: kappa_estimate={max_kappa:e}, \
             kappa*u_f32={kappa_u:e}, required < {threshold:e}"
        ))
    } else {
        None
    }
}

fn arrow_factor_slab_to_f32(htt_factors: &ArrowFactorSlab) -> Vec<Array2<f32>> {
    htt_factors
        .iter()
        .map(|factor| factor.mapv(|v| v as f32))
        .collect()
}

fn arrow_rhs(sys: &ArrowSchurSystem) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let mut rhs_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        for c in 0..di {
            rhs_t[base + c] = -sys.rows[i].gt[c];
        }
    }
    let mut rhs_beta = Array1::<f64>::zeros(sys.k);
    for c in 0..sys.k {
        rhs_beta[c] = -sys.gb[c];
    }
    (rhs_t, rhs_beta)
}

fn solve_arrow_system_f32(
    sys: &ArrowSchurSystem,
    row_factors: &[Array2<f32>],
    schur_factor: &Array2<f32>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let mut y_rows = Vec::<Array1<f32>>::with_capacity(n);
    let mut reduced_beta = rhs_beta.mapv(|v| v as f32);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let rhs_i = rhs_t.slice(ndarray::s![base..base + di]).mapv(|v| v as f32);
        let y_i = cholesky_solve_lower_f32(&row_factors[i], &rhs_i);
        let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i]).mapv(|v| v as f32);
        for beta_col in 0..sys.k {
            let mut acc = 0.0_f32;
            for row_axis in 0..di {
                acc += htbeta[[row_axis, beta_col]] * y_i[row_axis];
            }
            reduced_beta[beta_col] -= acc;
        }
        y_rows.push(y_i);
    }

    let x_beta_f32 = cholesky_solve_lower_f32(schur_factor, &reduced_beta);
    let mut x_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i]).mapv(|v| v as f32);
        let mut cross = Array1::<f32>::zeros(di);
        for row_axis in 0..di {
            let mut acc = 0.0_f32;
            for beta_col in 0..sys.k {
                acc += htbeta[[row_axis, beta_col]] * x_beta_f32[beta_col];
            }
            cross[row_axis] = acc;
        }
        let correction = cholesky_solve_lower_f32(&row_factors[i], &cross);
        for row_axis in 0..di {
            x_t[base + row_axis] = (y_rows[i][row_axis] - correction[row_axis]) as f64;
        }
    }
    let x_beta = x_beta_f32.mapv(|v| v as f64);
    (x_t, x_beta)
}

fn cholesky_solve_lower_f32(l: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = l.nrows();
    let mut y = Array1::<f32>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f32>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

fn arrow_residual(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let (ax_t, ax_beta) = arrow_operator_apply(sys, ridge_t, ridge_beta, x_t, x_beta);
    let mut res_t = rhs_t.to_owned();
    let mut res_beta = rhs_beta.to_owned();
    for i in 0..res_t.len() {
        res_t[i] -= ax_t[i];
    }
    for i in 0..res_beta.len() {
        res_beta[i] -= ax_beta[i];
    }
    (res_t, res_beta)
}

fn arrow_operator_apply(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let mut y_t = Array1::<f64>::zeros(sys.row_offsets[n]);
    let mut y_beta = Array1::<f64>::zeros(sys.k);
    {
        let x_slice = x_beta.as_slice().expect("x_beta contiguous");
        let y_slice = y_beta.as_slice_mut().expect("y_beta contiguous");
        sys.penalty_matvec_add(x_slice, y_slice);
    }
    for beta_col in 0..sys.k {
        y_beta[beta_col] += ridge_beta * x_beta[beta_col];
    }
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let row = &sys.rows[i];
        for a in 0..di {
            let mut acc = ridge_t * x_t[base + a];
            for b in 0..di {
                acc += row.htt[[a, b]] * x_t[base + b];
            }
            y_t[base + a] = acc;
        }
        let mut htbeta_xb = Array1::<f64>::zeros(di);
        sys_htbeta_apply_row(sys, i, row, x_beta, &mut htbeta_xb);
        for a in 0..di {
            y_t[base + a] += htbeta_xb[a];
        }
        let x_ti = x_t.slice(ndarray::s![base..base + di]).to_owned();
        sys_htbeta_accumulate_transpose(sys, i, row, x_ti.view(), &mut y_beta);
    }
    (y_t, y_beta)
}

fn arrow_backward_error_certificate(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
    res_t: ArrayView1<'_, f64>,
    res_beta: ArrayView1<'_, f64>,
) -> f64 {
    let residual_norm = infinity_norm_pair(res_t, res_beta);
    let operator_norm = arrow_operator_infinity_norm(sys, ridge_t, ridge_beta);
    let solution_norm = infinity_norm_pair(x_t, x_beta);
    let rhs_norm = infinity_norm_pair(rhs_t, rhs_beta);
    let denom = operator_norm * solution_norm + rhs_norm;
    if denom > 0.0 {
        residual_norm / denom
    } else {
        residual_norm
    }
}

fn infinity_norm_pair(lhs: ArrayView1<'_, f64>, rhs: ArrayView1<'_, f64>) -> f64 {
    let mut out = 0.0_f64;
    for &v in lhs.iter().chain(rhs.iter()) {
        out = out.max(v.abs());
    }
    out
}

fn arrow_operator_infinity_norm(sys: &ArrowSchurSystem, ridge_t: f64, ridge_beta: f64) -> f64 {
    let mut out = 0.0_f64;
    for i in 0..sys.rows.len() {
        let di = sys.row_dims[i];
        let row = &sys.rows[i];
        let htbeta = sys_htbeta_materialize_row(sys, i, row);
        for a in 0..di {
            let mut row_sum = 0.0_f64;
            for b in 0..di {
                row_sum += row.htt[[a, b]].abs();
            }
            row_sum += ridge_t;
            for beta_col in 0..sys.k {
                row_sum += htbeta[[a, beta_col]].abs();
            }
            out = out.max(row_sum);
        }
    }
    let hbb = sys.effective_penalty_op().to_dense();
    for beta_row in 0..sys.k {
        let mut row_sum = 0.0_f64;
        for beta_col in 0..sys.k {
            row_sum += hbb[[beta_row, beta_col]].abs();
        }
        row_sum += ridge_beta;
        for i in 0..sys.rows.len() {
            let di = sys.row_dims[i];
            let htbeta = sys_htbeta_materialize_row(sys, i, &sys.rows[i]);
            for a in 0..di {
                row_sum += htbeta[[a, beta_row]].abs();
            }
        }
        out = out.max(row_sum);
    }
    out
}

fn solve_arrow_newton_step_artifacts(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    // Auto-select the cross-row path: when any registered Psi penalty couples
    // distinct latent rows, the exact one-shot Schur elimination (which assumes
    // each H_tt^(i) is independent) cannot represent the off-row Hessian blocks.
    // Route the FULL (t, β) Newton system through matrix-free preconditioned CG
    // with the exact arrow block-diagonal inverse as the preconditioner. No
    // flag: the route is implied by the captured cross-row penalty set.
    if !sys.cross_row_penalties.is_empty() {
        return solve_arrow_newton_step_cross_row(sys, ridge_t, ridge_beta, options);
    }
    if let Some(chunk_size) = options.streaming_chunk_size {
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        let (delta_t, delta_beta, schur_factor) = streaming.solve(ridge_t, ridge_beta, options)?;
        return Ok(ArrowNewtonStepArtifacts {
            delta_t,
            delta_beta,
            htt_factors: ArrowFactorSlab::from_blocks(Vec::new()),
            schur_factor,
            pcg_diagnostics: PcgDiagnostics::default(),
        });
    }
    let backend = CpuBatchedBlockSolver;

    // 1. BA point elimination: per-row Cholesky factors of
    // (H_tt^(i) + ridge_t · I).  `factor_blocks` reads the actual row
    // dimension from `row.htt.nrows()` so heterogeneous systems work.
    let htt_factors =
        backend.factor_blocks(&sys.rows, ridge_t, sys.d, options.tolerate_ill_conditioning)?;

    // 2. Reduced RHS r_β = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i).
    let rhs_beta = reduced_rhs_beta(sys, &htt_factors, &backend);
    // The Schur solve is over the reduced β vector. Latent manifold metric
    // weights live on each d-dimensional t_i block, so the induced metric for
    // this β-only Steihaug problem is Euclidean.
    let trust_metric_weights = None;

    // 3. Solve reduced shared system using the selected BA mode.
    let mut mixed_precision_status = MixedPrecisionStatus::Off;
    let (delta_beta, schur_factor, mut pcg_diagnostics) = match options.mode {
        ArrowSolverMode::Direct => {
            let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, &backend)?;
            if let Some(attempt) = try_mixed_precision_arrow_solve(
                sys,
                ridge_t,
                ridge_beta,
                &htt_factors,
                &schur,
                options,
            )? {
                match attempt {
                    MixedPrecisionAttempt::Certified {
                        delta_t,
                        delta_beta,
                        schur_factor,
                        refinement_steps,
                    } => {
                        let mut pcg_diagnostics = PcgDiagnostics::default();
                        pcg_diagnostics.mixed_precision_status =
                            MixedPrecisionStatus::Certified { refinement_steps };
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t,
                            delta_beta,
                            htt_factors,
                            schur_factor: Some(schur_factor),
                            pcg_diagnostics,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        eprintln!("arrow-Schur mixed precision fallback to f64: {reason}");
                        mixed_precision_status = MixedPrecisionStatus::F64Fallback;
                    }
                }
            }
            let (db, sf, diag) =
                solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::SqrtBA => {
            let schur = build_dense_schur_sqrt_ba(sys, &htt_factors, ridge_beta, &backend)?;
            if let Some(attempt) = try_mixed_precision_arrow_solve(
                sys,
                ridge_t,
                ridge_beta,
                &htt_factors,
                &schur,
                options,
            )? {
                match attempt {
                    MixedPrecisionAttempt::Certified {
                        delta_t,
                        delta_beta,
                        schur_factor,
                        refinement_steps,
                    } => {
                        let mut pcg_diagnostics = PcgDiagnostics::default();
                        pcg_diagnostics.mixed_precision_status =
                            MixedPrecisionStatus::Certified { refinement_steps };
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t,
                            delta_beta,
                            htt_factors,
                            schur_factor: Some(schur_factor),
                            pcg_diagnostics,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        eprintln!("arrow-Schur mixed precision fallback to f64: {reason}");
                        mixed_precision_status = MixedPrecisionStatus::F64Fallback;
                    }
                }
            }
            let (db, sf, diag) =
                solve_dense_reduced_system(&schur, &rhs_beta, options, trust_metric_weights)?;
            (db, sf, diag)
        }
        ArrowSolverMode::InexactPCG => {
            if options.mixed_precision.is_enabled() {
                eprintln!(
                    "arrow-Schur mixed precision fallback to f64: InexactPCG does not expose a dense Schur factor for certified f32 refinement"
                );
                mixed_precision_status = MixedPrecisionStatus::F64Fallback;
            }
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
    if mixed_precision_status != MixedPrecisionStatus::Off {
        pcg_diagnostics.mixed_precision_status = mixed_precision_status;
    }

    // 4. Back-substitute Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
    let delta_t = back_substitute_delta_t(sys, &htt_factors, delta_beta.view(), &backend);

    Ok(ArrowNewtonStepArtifacts {
        delta_t,
        delta_beta,
        htt_factors,
        schur_factor,
        pcg_diagnostics,
    })
}

/// Exact inverse of the block-diagonal arrow operator `K0 + ridge`, used as
/// the preconditioner for the cross-row full-system CG.
///
/// Holds the per-row `H_tt^(i) + ridge_t·I` Cholesky factors and the dense
/// Schur-complement factor `S = (H_ββ + ridge_β·I) − Σ_i H_tβ^(i)ᵀ
/// (H_tt^(i))⁻¹ H_tβ^(i)`, so applying `M⁻¹` to an arbitrary RHS is a single
/// Schur back/forward substitution — exactly the algebra
/// [`solve_arrow_newton_step_artifacts`] performs, generalized to a free RHS.
struct ArrowBlockDiagInverse<'a, B: BatchedBlockSolver> {
    sys: &'a ArrowSchurSystem,
    backend: &'a B,
    htt_factors: ArrowFactorSlab,
    schur_factor: Array2<f64>,
}

impl<'a, B: BatchedBlockSolver> ArrowBlockDiagInverse<'a, B> {
    fn build(
        sys: &'a ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
        tolerate_ill_conditioning: bool,
        backend: &'a B,
    ) -> Result<Self, ArrowSchurError>
    where
        B: Sync,
    {
        let htt_factors =
            backend.factor_blocks(&sys.rows, ridge_t, sys.d, tolerate_ill_conditioning)?;
        let schur = build_dense_schur_direct(sys, &htt_factors, ridge_beta, backend)?;
        let schur_factor =
            cholesky_lower(&schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
        Ok(Self {
            sys,
            backend,
            htt_factors,
            schur_factor,
        })
    }

    /// Solve `(K0 + ridge) · [x_t; x_β] = [r_t; r_β]` exactly.
    ///
    /// `r_t` is flat row-major (`Σ_i row_dims[i]`); `r_β` is length `K`. The
    /// outputs `x_t` / `x_β` use the same layout.
    fn apply(
        &self,
        r_t: ArrayView1<'_, f64>,
        r_beta: ArrayView1<'_, f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let sys = self.sys;
        let n = sys.rows.len();
        let k = sys.k;
        // Reduced β RHS: r_β − Σ_i H_βt^(i) (H_tt^(i))⁻¹ r_t,i.
        let mut rhs_beta = r_beta.to_owned();
        for i in 0..n {
            let di = sys.row_dims[i];
            let base = sys.row_offsets[i];
            let r_ti = r_t.slice(ndarray::s![base..base + di]).to_owned();
            let u_i = self
                .backend
                .solve_block_vector(self.htt_factors.factor(i), r_ti.view());
            let mut acc = Array1::<f64>::zeros(k);
            sys_htbeta_accumulate_transpose(sys, i, &sys.rows[i], u_i.view(), &mut acc);
            for a in 0..k {
                rhs_beta[a] -= acc[a];
            }
        }
        // x_β = S⁻¹ rhs_β.
        let x_beta = cholesky_solve_lower(&self.schur_factor, &rhs_beta);
        // x_t,i = (H_tt^(i))⁻¹ (r_t,i − H_tβ^(i) x_β).
        let total_dt = sys.row_offsets[n];
        let mut x_t = Array1::<f64>::zeros(total_dt);
        let mut htbeta_xb = Array1::<f64>::zeros(sys.d);
        for i in 0..n {
            let di = sys.row_dims[i];
            let base = sys.row_offsets[i];
            for c in 0..di {
                htbeta_xb[c] = 0.0;
            }
            let mut slab = htbeta_xb.slice_mut(ndarray::s![..di]).to_owned();
            sys_htbeta_apply_row(sys, i, &sys.rows[i], x_beta.view(), &mut slab);
            let mut rhs_i = Array1::<f64>::zeros(di);
            for c in 0..di {
                rhs_i[c] = r_t[base + c] - slab[c];
            }
            let xi = self
                .backend
                .solve_block_vector(self.htt_factors.factor(i), rhs_i.view());
            for c in 0..di {
                x_t[base + c] = xi[c];
            }
        }
        (x_t, x_beta)
    }
}

/// Apply the full cross-row Newton operator `A = (K0 + ridge) + P_cross` to
/// `[x_t; x_β]`, writing `[y_t; y_β]`.
///
/// `(K0 + ridge)` is the block-diagonal arrow operator: per row
/// `y_t,i = (H_tt^(i) + ridge_t·I) x_t,i + H_tβ^(i) x_β`, and
/// `y_β = Σ_i H_βt^(i) x_t,i + (H_ββ + ridge_β·I) x_β`. `P_cross` adds the
/// captured cross-row penalty Hessian to the latent block only:
/// `y_t += P_cross · x_t`.
fn arrow_cross_row_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let k = sys.k;
    let total_dt = sys.row_offsets[n];
    let mut y_t = Array1::<f64>::zeros(total_dt);
    let mut y_beta = Array1::<f64>::zeros(k);
    let mut htbeta_xb = Array1::<f64>::zeros(sys.d);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let row = &sys.rows[i];
        // H_tt^(i) x_t,i + ridge_t x_t,i.
        for a in 0..di {
            let mut acc = ridge_t * x_t[base + a];
            for b in 0..di {
                acc += row.htt[[a, b]] * x_t[base + b];
            }
            y_t[base + a] = acc;
        }
        // + H_tβ^(i) x_β.
        for c in 0..di {
            htbeta_xb[c] = 0.0;
        }
        let mut slab = htbeta_xb.slice_mut(ndarray::s![..di]).to_owned();
        sys_htbeta_apply_row(sys, i, row, x_beta, &mut slab);
        for c in 0..di {
            y_t[base + c] += slab[c];
        }
        // y_β += H_βt^(i) x_t,i.
        let x_ti = x_t.slice(ndarray::s![base..base + di]).to_owned();
        sys_htbeta_accumulate_transpose(sys, i, row, x_ti.view(), &mut y_beta);
    }
    // y_β += (H_ββ + ridge_β·I) x_β.
    {
        let x_beta_slice = x_beta.as_slice().expect("x_beta contiguous");
        let y_beta_slice = y_beta.as_slice_mut().expect("y_beta contiguous");
        sys.penalty_matvec_add(x_beta_slice, y_beta_slice);
    }
    for a in 0..k {
        y_beta[a] += ridge_beta * x_beta[a];
    }
    // y_t += P_cross · x_t (cross-row penalty Hessian, latent block only).
    sys.apply_cross_row_penalty_hessian(x_t, &mut y_t);
    (y_t, y_beta)
}

/// Solve the full bordered Newton system when one or more registered Psi
/// penalties couple distinct latent rows.
///
/// The operator is `A = (K0 + ridge) + P_cross`, SPD whenever the arrow
/// block-diagonal `K0 + ridge` is PD (enforced by the per-row factor checks)
/// and every cross-row penalty contributes a PSD `psd_majorizer_hvp`. We solve
/// `A · [Δt; Δβ] = −[g_t; g_β]` by preconditioned conjugate gradients, using
/// the exact arrow block-diagonal inverse `M⁻¹ = (K0 + ridge)⁻¹` as the
/// preconditioner — the same Schur elimination the row-block-diagonal path
/// uses, here applied to the CG residual rather than the negated gradient.
/// Because `M⁻¹` inverts everything except the (small, structured) `P_cross`
/// coupling, the preconditioned operator `M⁻¹ A = I + M⁻¹ P_cross` has a
/// tightly clustered spectrum and CG converges in a handful of iterations.
fn solve_arrow_newton_step_cross_row(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowNewtonStepArtifacts, ArrowSchurError> {
    let backend = CpuBatchedBlockSolver;
    let precond = ArrowBlockDiagInverse::build(
        sys,
        ridge_t,
        ridge_beta,
        options.tolerate_ill_conditioning,
        &backend,
    )?;

    let n = sys.rows.len();
    let k = sys.k;
    let total_dt = sys.row_offsets[n];

    // RHS b = −g = [−g_t; −g_β].
    let mut b_t = Array1::<f64>::zeros(total_dt);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        for c in 0..di {
            b_t[base + c] = -sys.rows[i].gt[c];
        }
    }
    let mut b_beta = Array1::<f64>::zeros(k);
    for a in 0..k {
        b_beta[a] = -sys.gb[a];
    }

    // Preconditioned CG on the full (t, β) system.
    // x = 0; r = b − A·0 = b; z = M⁻¹ r; p = z.
    let mut x_t = Array1::<f64>::zeros(total_dt);
    let mut x_beta = Array1::<f64>::zeros(k);
    let mut r_t = b_t.clone();
    let mut r_beta = b_beta.clone();
    let (mut z_t, mut z_beta) = precond.apply(r_t.view(), r_beta.view());
    let mut p_t = z_t.clone();
    let mut p_beta = z_beta.clone();
    let mut rz = dot2(&r_t, &r_beta, &z_t, &z_beta);

    let b_norm = (dot2(&b_t, &b_beta, &b_t, &b_beta)).sqrt();
    // Solve the linear Newton system to tight relative accuracy. The cross-row
    // path is exact-CG (no trust region), so we drive the residual to machine-
    // scale relative tolerance; the spectrum I + M⁻¹P_cross makes this cheap.
    // Absolute floor guards b_norm → 0; relative term tracks the RHS scale.
    const CROSS_ROW_CG_ABS_TOL: f64 = 1e-12;
    const CROSS_ROW_CG_REL_TOL: f64 = 1e-13;
    // CG converges in at most (dim) iterations; allow a few passes over the
    // dimension to absorb round-off, with a small floor for tiny systems.
    const CROSS_ROW_CG_MIN_ITER_BUDGET: usize = 64;
    const CROSS_ROW_CG_ITER_MULTIPLE: usize = 4;
    let tol = CROSS_ROW_CG_ABS_TOL.max(CROSS_ROW_CG_REL_TOL * b_norm);
    let max_iter = (total_dt + k).max(CROSS_ROW_CG_MIN_ITER_BUDGET) * CROSS_ROW_CG_ITER_MULTIPLE;

    let mut iters = 0usize;
    let mut converged = b_norm == 0.0;
    while iters < max_iter && !converged {
        let (ap_t, ap_beta) =
            arrow_cross_row_matvec(sys, ridge_t, ridge_beta, p_t.view(), p_beta.view());
        let pap = dot2(&p_t, &p_beta, &ap_t, &ap_beta);
        if !(pap.is_finite() && pap > 0.0) {
            return Err(ArrowSchurError::PcgFailed {
                reason: format!(
                    "cross-row full-system CG hit non-positive curvature pᵀAp={pap:e}; \
                     the cross-row penalty Hessian or arrow block is not PD at this iterate"
                ),
            });
        }
        let alpha = rz / pap;
        for i in 0..total_dt {
            x_t[i] += alpha * p_t[i];
            r_t[i] -= alpha * ap_t[i];
        }
        for a in 0..k {
            x_beta[a] += alpha * p_beta[a];
            r_beta[a] -= alpha * ap_beta[a];
        }
        let r_norm = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
        iters += 1;
        if r_norm <= tol {
            converged = true;
            break;
        }
        let (nz_t, nz_beta) = precond.apply(r_t.view(), r_beta.view());
        z_t = nz_t;
        z_beta = nz_beta;
        let rz_new = dot2(&r_t, &r_beta, &z_t, &z_beta);
        let beta_cg = rz_new / rz;
        for i in 0..total_dt {
            p_t[i] = z_t[i] + beta_cg * p_t[i];
        }
        for a in 0..k {
            p_beta[a] = z_beta[a] + beta_cg * p_beta[a];
        }
        rz = rz_new;
    }

    if !converged {
        let r_norm = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "cross-row full-system CG did not converge in {iters} iters \
                 (‖r‖={r_norm:e}, tol={tol:e})"
            ),
        });
    }

    let final_residual = (dot2(&r_t, &r_beta, &r_t, &r_beta)).sqrt();
    let diag = PcgDiagnostics {
        iterations: iters,
        matvec_calls: iters,
        precond_apply_calls: iters + 1,
        ridge_escalations: 0,
        final_relative_residual: if b_norm > 0.0 {
            final_residual / b_norm
        } else {
            0.0
        },
        stopping_reason: PcgStopReason::Converged,
        mixed_precision_status: MixedPrecisionStatus::Off,
    };

    Ok(ArrowNewtonStepArtifacts {
        delta_t: x_t,
        delta_beta: x_beta,
        htt_factors: precond.htt_factors,
        schur_factor: Some(precond.schur_factor),
        pcg_diagnostics: diag,
    })
}

/// `⟨[a_t; a_β], [b_t; b_β]⟩` over the stacked latent/β vector.
fn dot2(a_t: &Array1<f64>, a_beta: &Array1<f64>, b_t: &Array1<f64>, b_beta: &Array1<f64>) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..a_t.len() {
        acc += a_t[i] * b_t[i];
    }
    for a in 0..a_beta.len() {
        acc += a_beta[a] * b_beta[a];
    }
    acc
}

/// Solve `L Lᵀ x = b` given the lower Cholesky factor `L`.
fn cholesky_solve_lower(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    // Forward solve L y = b.
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

fn reduced_rhs_beta<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
) -> Array1<f64> {
    // Numerical invariant: each per-row `H_tt^(i)` factor must be PD
    // (already enforced by the adaptive-ridge `factor_blocks`).
    let k = sys.k;
    let mut rhs_beta = Array1::<f64>::zeros(k);
    for (i, row) in sys.rows.iter().enumerate() {
        let v = backend.solve_block_vector(htt_factors.factor(i), row.gt.view());
        // H_βt^(i) · v accumulates into rhs_beta.  Routes through
        // sys.htbeta_matvec when the dense block is absent.
        sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut rhs_beta);
    }
    for j in 0..k {
        rhs_beta[j] -= sys.gb[j];
    }
    rhs_beta
}

/// Which Square-Root / direct factorization the per-row Schur contribution
/// uses. `Direct` forms `H_tβᵀ (H_tt)⁻¹ H_tβ` via a full block solve; `SqrtBa`
/// forms the equivalent `(L⁻¹ H_tβ)ᵀ (L⁻¹ H_tβ)` from the lower triangular
/// solve only. The reduction `Σ_i contribution_i` is identical in both axes.
#[derive(Clone, Copy)]
enum SchurReductionKind {
    Direct,
    SqrtBa,
}

/// Form one row block's `(left, right)` Schur contribution factors so that the
/// contribution is `leftᵀ · right` (`k×k`). `Direct` solves the full block,
/// `SqrtBa` uses only the lower-triangular whitening; both give the same
/// `H_tβᵀ (H_tt)⁻¹ H_tβ` because `H_tt = L Lᵀ`.
#[inline]
fn row_schur_contribution_factors<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    htt_factor: ArrayView2<'_, f64>,
    backend: &B,
    kind: SchurReductionKind,
) -> (Array2<f64>, Array2<f64>) {
    // Materialize the (d, k) cross-block, probing via the matvec when the
    // dense slab is absent.
    let htbeta = sys_htbeta_materialize_row(sys, row_idx, row);
    match kind {
        SchurReductionKind::Direct => {
            let solved = backend.solve_block_matrix(htt_factor, htbeta.view());
            (htbeta, solved)
        }
        SchurReductionKind::SqrtBa => {
            let whitened = backend.sqrt_solve_block_matrix(htt_factor, htbeta.view());
            (whitened.clone(), whitened)
        }
    }
}

/// Subtract one row block's Schur contribution from `schur` using the selected
/// reduction kind. Identical algebra to the inline loop bodies the dense
/// builders used; factored out so the serial and multi-GPU partition paths
/// share one definition.
#[inline]
fn subtract_row_schur_contribution<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    htt_factor: ArrayView2<'_, f64>,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
) {
    let (left, right) =
        row_schur_contribution_factors(sys, row_idx, row, htt_factor, backend, kind);
    backend.block_gemm_subtract(schur, &left, &right);
}

/// Reduce one contiguous device tile's rows into a private `-Σ leftᵀ·right`
/// partial (`k×k`).
///
/// The tile stacks its per-row `left_i` / `right_i` factors (each `d×k`) into
/// two `(Σ_i d_i × k)` matrices and tries a single per-ordinal `AᵀB` device
/// GEMM (`crate::gpu::try_fast_atb_on_ordinal`), which runs on the device this
/// worker thread already bound — one big GPU GEMM per tile rather than `n` small
/// CPU ones. When the device primitive declines (no GPU, shape below policy,
/// transient failure) the tile reduces with the exact CPU `block_gemm_subtract`
/// loop, so the result is unchanged. The partial is negated so the caller's
/// `schur += partial` reproduces the serial `schur -= Σ contribution`.
fn tile_schur_partial<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    ordinal: usize,
    range: Range<usize>,
) -> Array2<f64> {
    let k = sys.k;

    // Build the per-row contribution factors once; both the GPU stacked-GEMM
    // and the CPU fallback consume them.
    let mut factors: Vec<(Array2<f64>, Array2<f64>)> = Vec::with_capacity(range.len());
    let mut total_d = 0usize;
    for i in range.clone() {
        let (left, right) = row_schur_contribution_factors(
            sys,
            i,
            &sys.rows[i],
            htt_factors.factor(i),
            backend,
            kind,
        );
        total_d += left.nrows();
        factors.push((left, right));
    }

    // Stack into (total_d × k) left/right matrices for one device AᵀB GEMM on
    // this tile's bound ordinal. `try_fast_atb_on_ordinal` returns leftᵀ·right
    // (k×k); negate into the partial. At an SAE-shaped whole-fit tile with
    // n=2000 rows, k=2048 shared columns, M=12 local rows per observation, and
    // K=8 candidate/atom batches, the stacked GEMM is
    // 2*(n*M)*k^2 = 201_326_592_000 flops per batch, or
    // 1_610_612_736_000 flops across K=8, so the policy work gate is cleared
    // even though the observation count is far below the old row floor.
    if total_d > 0 && k > 0 {
        let mut left_stack = Array2::<f64>::zeros((total_d, k));
        let mut right_stack = Array2::<f64>::zeros((total_d, k));
        let mut base = 0usize;
        for (left, right) in &factors {
            let di = left.nrows();
            left_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(left);
            right_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(right);
            base += di;
        }
        if let Some(product) =
            crate::gpu::try_fast_atb_on_ordinal(ordinal, left_stack.view(), right_stack.view())
        {
            return product.mapv(|v| -v);
        }
    }

    // CPU fallback: exact per-row block_gemm_subtract into a zero-seeded partial.
    let mut partial = Array2::<f64>::zeros((k, k));
    for (left, right) in &factors {
        backend.block_gemm_subtract(&mut partial, left, right);
    }
    partial
}

/// Reduce the per-row Schur contributions `Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`
/// out of `schur` (seeded with `H_ββ + ρ_β·I`).
///
/// The per-row contributions are independent — exactly the "sum over independent
/// arrow-tip blocks" axis the device pool partitions. When more than one GPU is
/// usable, [`crate::gpu::pool::balanced_partition`] splits the `0..n` rows into
/// per-device contiguous tiles; each tile is reduced on its own scoped thread
/// (binding that ordinal's context so the per-row GEMM-subtract offloads to its
/// device) into a private `k×k` partial, and the partials are summed back into
/// `schur` in tile order. The tiles are contiguous, ordered to cover `0..n`, and
/// folded back in that same order, so within each tile the per-row accumulation
/// order is preserved and the only departure from the serial loop is the
/// inter-tile reassociation of the reduction sum — the established
/// reduction-order equivalence the device pool already operates under, well
/// inside the Newton solve's tolerance.
///
/// With a single device (or no GPU) the row loop runs serially in place, which
/// is bit-for-bit the original behaviour.
fn reduce_row_schur_contributions<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
) {
    let n = sys.rows.len();
    let k = sys.k;

    let tiles = crate::gpu::runtime::GpuRuntime::global()
        .map(|rt| crate::gpu::pool::balanced_partition(rt, n))
        .filter(|tiles| tiles.len() > 1);

    let Some(tiles) = tiles else {
        // Single-device / CPU: reduce serially in place (original order).
        for (i, row) in sys.rows.iter().enumerate() {
            subtract_row_schur_contribution(
                sys,
                i,
                row,
                htt_factors.factor(i),
                backend,
                kind,
                schur,
            );
        }
        return;
    };

    // Multi-GPU: one private `-Σ leftᵀ·right` partial per contiguous device
    // tile. Each tile runs on its own scoped worker thread that binds its
    // ordinal's context and issues a single stacked AᵀB GEMM on that device, so
    // the tiles' GEMMs overlap across the pool. Folding the partials back into
    // the H_ββ-seeded `schur` reproduces the serial reduction (up to inter-tile
    // reassociation).
    let partials: Vec<Array2<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = tiles
            .iter()
            .map(|(ordinal, range)| {
                let ordinal = *ordinal;
                let range = range.clone();
                scope.spawn(move || {
                    // Bind this ordinal's CUDA context on this worker thread so
                    // the per-row GPU GEMM shims issued from `tile_schur_partial`
                    // offload to that device. A missing context or bind failure
                    // is intentionally consumed without escalation — the shims
                    // no-op back to CPU and the math is unchanged. Off Linux
                    // `GpuRuntime::global()` is always `None`, so this branch
                    // is unreachable and the bind is omitted entirely.
                    #[cfg(target_os = "linux")]
                    {
                        if let Some(ctx) = crate::gpu::runtime::cuda_context_for(ordinal) {
                            if ctx.bind_to_thread().is_err() {
                                // Fall through: this tile reduces on the CPU.
                            }
                        }
                    }
                    tile_schur_partial(sys, htt_factors, backend, kind, ordinal, range)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("schur-reduction tile thread panicked"))
            .collect()
    });

    // Fold partials into `schur` in tile order (contiguous, covering 0..n) so
    // the per-tile and inter-tile accumulation order is the row order; each
    // partial holds `-Σ contribution` over its rows, so `schur += partial`
    // reproduces `schur -= Σ contribution`.
    for partial in &partials {
        for a in 0..k {
            for b in 0..k {
                schur[[a, b]] += partial[[a, b]];
            }
        }
    }
}

fn build_dense_schur_direct<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
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
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::Direct,
        &mut schur,
    );
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

fn build_dense_schur_sqrt_ba<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
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
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::SqrtBa,
        &mut schur,
    );
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
    // Ill-conditioned-but-PD Schur guard. The per-row factor checks reject
    // any single barely-PD H_tt^(i) block, but the reduced Schur complement
    //     S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
    // accumulates the (H_tt^(i))⁻¹ contributions of every row in finite
    // precision. With many weak-but-admissible rows those terms can sum to a
    // Schur matrix whose Cholesky succeeds yet whose condition number is far
    // past the safe inversion regime, so `cholesky_solve_vector` yields an
    // inaccurate Δβ that is silently propagated to the Newton step. Apply the
    // same diagonal-ratio κ proxy used per-row to the reduced factor and treat
    // an over-threshold estimate as a Schur-stability failure: `SchurFactorFailed`
    // is already recoverable in `solve_with_lm_escalation_inner`, so this lifts
    // `ridge_beta` and re-forms a better-conditioned Schur. This guard is
    // exclusive to the dense Direct / SqrtBA path (the only caller of this
    // function); the inexact-PCG path tolerates higher κ(S) and is unaffected.
    // Evidence/log-det-only callers (`tolerate_ill_conditioning`) skip this
    // rejection: the factor is genuinely PD (Cholesky above succeeded), so its
    // diagonal still yields an exact `log|S|`, and an inaccurate Δβ is harmless
    // because the step is discarded.
    if !options.tolerate_ill_conditioning {
        let schur_kappa = cholesky_factor_kappa_estimate(&factor);
        if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)⁻¹ contamination would yield an inaccurate Δβ"
                ),
            });
        }
    }
    let direct = cholesky_solve_vector(&factor, rhs_beta);
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

/// Solve an externally accumulated dense reduced β system
/// `S Δβ = rhs_β` with the same LM-style ridge escalation the full-batch
/// driver applies: on a `SchurFactorFailed` (non-PD or ill-conditioned `S`),
/// geometrically grow a proximal ridge on `S`'s diagonal and retry.
///
/// Used by the SAE streaming joint fit, which accumulates `S` and `rhs_β` over
/// re-materialized row chunks (via [`StreamingArrowSchur::take_accumulators`])
/// and must solve the single global reduced system without a per-row
/// `ArrowSchurSystem`. `S` is symmetrized from its lower triangle before each
/// factorization. `base_ridge_beta` is folded into the caller's `S` already;
/// this routine only adds the *escalation* ridge on top.
pub fn solve_streaming_reduced_beta(
    s_acc: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
) -> Result<Array1<f64>, ArrowSchurError> {
    let mut proximal_ridge = 0.0_f64;
    let mut last_err: Option<ArrowSchurError> = None;
    for attempt in 0..=DEFAULT_PROXIMAL_MAX_ATTEMPTS {
        let mut schur = s_acc.clone();
        symmetrize_upper_from_lower(&mut schur);
        if proximal_ridge > 0.0 {
            for j in 0..schur.nrows() {
                schur[[j, j]] += proximal_ridge;
            }
        }
        // Reduced K-system on device: Jacobi-preconditioned CG over the dense
        // symmetric `S`. The `O(K²)` `S·p` matvec runs device-side; only the
        // K-vectors cross the boundary per CG iteration. This is the dominant
        // cost of the streaming SAE joint fit at `K = 100K`. Any device-side
        // failure (`Unavailable`, non-PD Jacobi diagonal) falls through to the
        // CPU `solve_dense_reduced_system`, which then drives the same proximal
        // ridge escalation. A genuine device PD failure is non-recoverable for
        // this attempt's `schur`, so we let the CPU path re-confirm and escalate.
        if crate::gpu::runtime::GpuRuntime::is_available() {
            match crate::gpu::arrow_schur::solve_reduced_beta_pcg(
                &schur,
                rhs_beta,
                options.trust_region.max_iterations,
                options.trust_region.steihaug_relative_tolerance,
            ) {
                Ok(delta_beta) => return Ok(delta_beta),
                Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::Unavailable) => {}
                Err(_) => {
                    // Device declined this `schur` (e.g. non-PD Jacobi diag);
                    // let the CPU path confirm and escalate the proximal ridge.
                }
            }
        }
        match solve_dense_reduced_system(&schur, rhs_beta, options, None) {
            Ok((delta_beta, _factor, _diag)) => return Ok(delta_beta),
            Err(err) => {
                let recoverable = matches!(
                    err,
                    ArrowSchurError::SchurFactorFailed { .. } | ArrowSchurError::PcgFailed { .. }
                );
                last_err = Some(err);
                if !recoverable || attempt == DEFAULT_PROXIMAL_MAX_ATTEMPTS {
                    break;
                }
                proximal_ridge = if proximal_ridge == 0.0 {
                    DEFAULT_PROXIMAL_INITIAL_RIDGE
                } else {
                    proximal_ridge * DEFAULT_PROXIMAL_RIDGE_GROWTH
                };
            }
        }
    }
    Err(last_err.expect("escalation loop set last_err on failure"))
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
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    backend: &B,
) {
    // `steihaug_cg` reuses one output buffer across iterations and requires
    // `matvec` to ASSIGN every entry of `out` (the contract `dense_matvec`
    // upholds). This routine builds `S·x` purely by accumulation
    // (`penalty_matvec_add`, `out[a] += ridge·x`, `out[a] -= neg_contrib`), so it
    // MUST clear `out` first. Without this, iteration n>0 returns `S·x` plus the
    // previous call's `S·p`, the PCG solves a corrupted reduced system, and the
    // resulting Newton step is inconsistent with the assembled gradient
    // (g·δ ≈ 0 — a non-descent direction that defeats the line search).
    out.fill(0.0);
    let k = sys.k;
    // Route the penalty-side H_ββ x product through penalty_matvec_add (#296):
    // no Arc-clone hot-path cost when penalty_op is None (falls back to hbb inline).
    {
        let x_slice = x.as_slice().expect("x must be contiguous");
        let out_slice = out.as_slice_mut().expect("out must be contiguous");
        sys.penalty_matvec_add(x_slice, out_slice);
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
        let solved = backend.solve_block_vector(htt_factors.factor(i), local_i.view());
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
    Chol {
        factor: FaerLlt<f64>,
        range: Range<usize>,
    },
    /// Scalar fallback: per-element `1/s_aa` for each column in `range`.
    Scalar {
        inv: Array1<f64>,
        range: Range<usize>,
    },
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
/// their own block layout.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    blocks: Vec<BlockFactor>,
}

/// Maximum block size for which we attempt dense block-Jacobi factorization.
const BLOCK_JACOBI_MAX_BLOCK: usize = 256;

/// Positive-definiteness floor on a Schur-complement Jacobi diagonal entry.
/// A diagonal at or below this value (or non-finite) signals a non-PD reduced
/// system: the preconditioner cannot invert it, so the PCG solve fails loudly
/// and demands operator regularization rather than returning a garbage scale.
const JACOBI_DIAGONAL_PD_FLOOR: f64 = 1e-18;

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
        htt_factors: &ArrowFactorSlab,
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
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        // Extract diagonal of H_ββ via penalty_diagonal_add (#296):
        // no Arc-clone; falls back to hbb_diag or hbb[[a,a]] inline.
        let mut diag = Array1::<f64>::zeros(k);
        {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            sys.penalty_diagonal_add(diag_slice);
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
                let solved = backend.solve_block_vector(htt_factors.factor(i), col_i.view());
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
            if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
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
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;

        // Initialise every b×b Schur sub-block from H_ββ + ridge·I via
        // penalty_block_add (#296): routes to penalty_op or falls back to
        // hbb / hbb_diag inline without Arc-clone per loop iteration. These are
        // the block-diagonal restrictions of the reduced Schur complement; the
        // per-row cross-block contributions are accumulated in the row sweep
        // below.
        let mut schur_blocks: Vec<Array2<f64>> = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let mut schur_block = Array2::<f64>::zeros((b, b));
            sys.penalty_block_add(
                BetaBlockId(block_idx),
                block_offsets.as_ref(),
                &mut schur_block,
            );
            for bi in 0..b {
                schur_block[[bi, bi]] += ridge_beta;
            }
            schur_blocks.push(schur_block);
        }

        // Subtract Schur contributions:
        // S_kk -= H_βt_k^(i) (H_tt^(i))^{-1} H_tβ_k^(i)
        //
        // Materialize each row's (d_i × K) cross-block ONCE and scatter its
        // contribution into every block-diagonal sub-block — mirroring the
        // row-outer structure of `build_dense_schur_direct`. The previous
        // block-outer form re-materialized every row for each β-block
        // (O(n_blocks · n · K) probes); for the matrix-free softmax cross-block
        // each materialize is itself O(K²), so that nesting made the
        // preconditioner build quadratically more expensive than the direct
        // dense Schur it preconditions. sys_htbeta_materialize_row handles the
        // Kronecker / htbeta_matvec path transparently.
        for (i, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[i];
            let htbeta_full = sys_htbeta_materialize_row(sys, i, row);
            for (block_idx, range) in block_offsets.iter().enumerate() {
                let b = range.end - range.start;
                let mut solved_cols = Array2::<f64>::zeros((di, b));
                for bj in 0..b {
                    let gj = range.start + bj;
                    let rhs = htbeta_full.column(gj).to_owned();
                    let solved = backend.solve_block_vector(htt_factors.factor(i), rhs.view());
                    for c in 0..di {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                let schur_block = &mut schur_blocks[block_idx];
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
        }

        // Factor each accumulated block: LLT, with scalar-diagonal fallback for
        // a block that comes out non-PD at this ridge.
        let mut blocks = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let schur_block = &schur_blocks[block_idx];
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(schur_block);
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
                    if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
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
                    let rhs_mat =
                        unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
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
    Chol {
        cols: Vec<usize>,
        factor: FaerLlt<f64>,
    },
    Scalar {
        cols: Vec<usize>,
        inv: Vec<f64>,
    },
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
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            return Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &[cols]);
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>(),
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
        htt_factors: &ArrowFactorSlab,
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
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
                continue;
            }
            let mut s_block = Array2::<f64>::zeros((b, b));
            // Initialise from H_ββ via penalty_subblock_add (#296): routes
            // through penalty_op or falls back to hbb / hbb_diag inline.
            sys.penalty_subblock_add(cols, &mut s_block);
            for bi in 0..b {
                s_block[[bi, bi]] += ridge_beta;
            }
            let mut col_vec = Array1::<f64>::zeros(d);
            let mut solved_cols = Array2::<f64>::zeros((d, b));
            for (row_idx, row) in sys.rows.iter().enumerate() {
                for bj in 0..b {
                    let gj = cols[bj];
                    for c in 0..d {
                        col_vec[c] = row.htbeta[[c, gj]];
                    }
                    let solved =
                        backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
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
                clusters.push(ClusterFactor::Chol {
                    cols: cols.clone(),
                    factor: llt,
                });
            } else {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
            }
        }
        Ok(Self { clusters })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(cluster, r, &mut out, &ClusterApplyMode::Overwrite);
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
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            let inner = ClusterJacobiPreconditioner::build_from_column_groups(
                sys,
                htt_factors,
                ridge_beta,
                backend,
                &[cols],
            )?;
            return Ok(Self {
                clusters: inner.clusters,
                weights: vec![1.0f64; sys.k],
            });
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>(),
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
        let weights: Vec<f64> = counts
            .iter()
            .map(|&c| if c == 0 { 1.0 } else { 1.0 / c as f64 })
            .collect();
        let inner = ClusterJacobiPreconditioner::build_from_column_groups(
            sys,
            htt_factors,
            ridge_beta,
            backend,
            &col_groups,
        )?;
        Ok(Self {
            clusters: inner.clusters,
            weights,
        })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(
                cluster,
                r,
                &mut out,
                &ClusterApplyMode::Accumulate {
                    weights: &self.weights,
                },
            );
        }
        out
    }
}

/// How a cluster factor's contribution is written into the output vector.
///
/// `Overwrite` assigns `out[gi] = value` (non-overlapping clusters, each global
/// column touched by exactly one cluster). `Accumulate` adds the partition-of-unity
/// weighted contribution `out[gi] += weights[gi] * value` (overlapping Schwarz
/// clusters, where a column may belong to several clusters).
enum ClusterApplyMode<'w> {
    Overwrite,
    Accumulate { weights: &'w [f64] },
}

impl ClusterApplyMode<'_> {
    #[inline]
    fn write(&self, out: &mut Array1<f64>, gi: usize, value: f64) {
        match self {
            ClusterApplyMode::Overwrite => out[gi] = value,
            ClusterApplyMode::Accumulate { weights } => out[gi] += weights[gi] * value,
        }
    }
}

/// Apply a single cluster factor to the residual `r`, writing into `out`
/// according to `mode` (overwrite for non-overlapping clusters, weighted
/// accumulate for overlapping Schwarz clusters).
fn apply_cluster(
    cluster: &ClusterFactor,
    r: &Array1<f64>,
    out: &mut Array1<f64>,
    mode: &ClusterApplyMode<'_>,
) {
    match cluster {
        ClusterFactor::Scalar { cols, inv } => {
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, inv[local] * r[gi]);
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
            let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
            let solved = factor.solve(rhs_mat);
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, solved[(local, 0)]);
            }
        }
    }
}

/// Build scalar diagonal inverses for a set of global column indices.
///
/// Used when a cluster is non-PD or exceeds `CLUSTER_JACOBI_MAX_CLUSTER`.
fn build_schur_scalar_inv<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Result<Vec<f64>, ArrowSchurError> {
    let d = sys.d;
    let mut result = Vec::with_capacity(cols.len());
    let mut col_vec = Array1::<f64>::zeros(d);
    // Extract the penalty diagonal for all K columns once, then index per-column.
    let mut full_diag = Array1::<f64>::zeros(sys.k);
    {
        let fd_slice = full_diag.as_slice_mut().expect("full_diag contiguous");
        sys.penalty_diagonal_add(fd_slice);
    }
    for &gi in cols {
        let mut s = full_diag[gi] + ridge_beta;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            for c in 0..d {
                col_vec[c] = row.htbeta[[c, gi]];
            }
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
            let mut acc = 0.0;
            for c in 0..d {
                acc += col_vec[c] * solved[c];
            }
            s -= acc;
        }
        if !s.is_finite() || s <= JACOBI_DIAGONAL_PD_FLOOR {
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
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    let jacobi = JacobiPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend)?;
    let (x0, diag0) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| jacobi.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
    )?;
    if sys.k <= PRECOND_ESCALATE_K_THRESHOLD || diag0.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x0, diag0));
    }
    let cluster =
        ClusterJacobiPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend)?;
    let (x1, diag1) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| cluster.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
    )?;
    if diag1.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x1, diag1));
    }
    let schwarz =
        AdditiveSchwarzPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend, 1)?;
    let (x2, diag2) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| schwarz.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
    )?;
    // All three preconditioner tiers (Jacobi -> ClusterJacobi ->
    // AdditiveSchwarz) exhausted their iteration budget without driving the
    // residual below tolerance. Returning the truncated AdditiveSchwarz iterate
    // as `Ok` would feed an arbitrarily-large-residual step into the Newton
    // driver, where the PCG diagnostics are discarded. Surface a recoverable
    // failure instead so `solve_with_lm_escalation_inner` escalates the
    // proximal ridge: better conditioning is precisely what a stalled PCG on
    // an ill-conditioned reduced system needs.
    if diag2.stopping_reason == PcgStopReason::MaxIter {
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "Schur PCG exhausted all preconditioner tiers (Jacobi, ClusterJacobi, \
                 AdditiveSchwarz) at MaxIter; final relative residual = {:e}",
                diag2.final_relative_residual
            ),
        });
    }
    Ok((x2, diag2))
}

/// Run Steihaug-CG with a generic preconditioner closure.
/// Routes matvec through GPU when `gpu_matvec` is set.
fn run_pcg_with_preconditioner<ApplyPrec, B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
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
    let tol = pcg
        .relative_tolerance
        .max(trust.steihaug_relative_tolerance);
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
    let tol = (relative_tolerance.max(0.0) * rhs_norm).max(PCG_ABSOLUTE_TOLERANCE_FLOOR);
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
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
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
                diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
                diag.stopping_reason = PcgStopReason::TrustRegion;
                return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
            }
            return Err(ArrowSchurError::PcgFailed {
                reason: "negative curvature in unbounded Schur PCG".to_string(),
            });
        }
        let alpha = rz / pap;
        for i in 0..n {
            candidate[i] = x[i] + alpha * p[i];
        }
        if radius.is_finite() && metric_norm(candidate.view(), metric_weights) >= radius {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
        }
        x.assign(&candidate);
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        if metric_norm(r.view(), metric_weights) <= tol {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Converged;
            return Ok((x, diag));
        }
        z = apply_preconditioner(&r);
        diag.precond_apply_calls += 1;
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
    /// A per-row `H_tt^(i)` block factored, but the Cholesky factor failed
    /// the safe-inversion guard for the Schur reduction. This can be either
    /// an excessive diagonal-ratio condition-number estimate or a numerically
    /// tiny pivot relative to the row block scale. Cholesky technically
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
                "arrow-Schur: per-row H_tt^({row}) Cholesky succeeded but failed \
                 the safe-inversion guard (kappa_estimate={kappa_estimate:e}); \
                 Schur reduction would be numerically contaminated"
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

    let mut maybe_device = a.clone();
    if crate::gpu::try_cholesky_lower_inplace(&mut maybe_device).is_some() {
        return Ok(maybe_device);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `SparseBlockKroneckerPenaltyOp` must reproduce the dense
    /// `KroneckerPenaltyOp { factor_a: G, factor_b: I_p }` on every interface
    /// (matvec, gradient, diagonal, to_dense) when the sparse block set covers
    /// the same `(atom, atom')` couplings — this is the equivalence that makes
    /// the sparse op a drop-in replacement for the dense data Gram.
    #[test]
    fn sparse_block_kronecker_matches_dense_kronecker() {
        // Two atoms: atom 0 has m_0 = 2 basis cols (μ offset 0), atom 1 has
        // m_1 = 3 (μ offset 2). p = 2 output channels ⇒ dim_a = 5, k = 10.
        let p = 2usize;
        let dim_a = 5usize;
        let k = dim_a * p;
        // Dense G (5×5) with non-zero (0,0), (0,1), (1,0), (1,1) atom blocks.
        let g_dense = array![
            [3.0_f64, 0.5, 0.2, -0.1, 0.0],
            [0.5, 4.0, 0.0, 0.3, 0.1],
            [0.2, 0.0, 2.0, 0.4, -0.2],
            [-0.1, 0.3, 0.4, 5.0, 0.6],
            [0.0, 0.1, -0.2, 0.6, 1.5],
        ];
        let dense = KroneckerPenaltyOp {
            factor_a: g_dense.clone(),
            factor_b: Array2::<f64>::eye(p),
            global_offset: 0,
            k,
        };
        // Sparse: atom 0 block = G[0..2, 0..2], cross blocks G[0..2,2..5] and
        // its transpose, atom 1 block = G[2..5, 2..5].
        let block_00 = g_dense.slice(ndarray::s![0..2, 0..2]).to_owned();
        let block_01 = g_dense.slice(ndarray::s![0..2, 2..5]).to_owned();
        let block_10 = g_dense.slice(ndarray::s![2..5, 0..2]).to_owned();
        let block_11 = g_dense.slice(ndarray::s![2..5, 2..5]).to_owned();
        let sparse = SparseBlockKroneckerPenaltyOp {
            p,
            dim_a,
            k,
            blocks: vec![
                SparseGBlock {
                    row_off: 0,
                    col_off: 0,
                    data: block_00,
                },
                SparseGBlock {
                    row_off: 0,
                    col_off: 2,
                    data: block_01,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 0,
                    data: block_10,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 2,
                    data: block_11,
                },
            ],
        };

        // to_dense parity.
        let d_dense = dense.to_dense();
        let d_sparse = sparse.to_dense();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (d_dense[[i, j]] - d_sparse[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    d_dense[[i, j]],
                    d_sparse[[i, j]]
                );
            }
        }

        // matvec / gradient parity on an arbitrary vector.
        let x: Vec<f64> = (0..k).map(|i| 0.1 * (i as f64) - 0.3).collect();
        let mut y_dense = vec![0.0_f64; k];
        let mut y_sparse = vec![0.0_f64; k];
        dense.matvec(&x, &mut y_dense);
        sparse.matvec(&x, &mut y_sparse);
        for i in 0..k {
            assert!(
                (y_dense[i] - y_sparse[i]).abs() < 1e-12,
                "matvec mismatch at {i}: {} vs {}",
                y_dense[i],
                y_sparse[i]
            );
        }

        // diagonal parity.
        let mut diag_dense = vec![0.0_f64; k];
        let mut diag_sparse = vec![0.0_f64; k];
        dense.diagonal(&mut diag_dense);
        sparse.diagonal(&mut diag_sparse);
        for i in 0..k {
            assert!(
                (diag_dense[i] - diag_sparse[i]).abs() < 1e-12,
                "diagonal mismatch at {i}: {} vs {}",
                diag_dense[i],
                diag_sparse[i]
            );
        }

        // block parity: probe the per-atom β block ranges.
        let offsets = [0..(2 * p), (2 * p)..k];
        for id in 0..offsets.len() {
            let b = offsets[id].end - offsets[id].start;
            let mut blk_dense = Array2::<f64>::zeros((b, b));
            let mut blk_sparse = Array2::<f64>::zeros((b, b));
            dense.block(BetaBlockId(id), &offsets, &mut blk_dense);
            sparse.block(BetaBlockId(id), &offsets, &mut blk_sparse);
            for i in 0..b {
                for j in 0..b {
                    assert!(
                        (blk_dense[[i, j]] - blk_sparse[[i, j]]).abs() < 1e-12,
                        "block {id} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    /// Hand-built dense reference for the frame-factored Gram
    /// `H[(i,li,a),(j,lj,b)] = g_ij[li,lj]·(U_iᵀU_j)[a,b]`, with the variable
    /// per-atom width `r_k`.
    fn factored_reference_dense(
        ranks: &[usize],
        basis_sizes: &[usize],
        blocks: &[FactoredFrameGBlock],
    ) -> Array2<f64> {
        let n_atoms = ranks.len();
        let mut offsets = vec![0usize; n_atoms + 1];
        for k in 0..n_atoms {
            offsets[k + 1] = offsets[k] + basis_sizes[k] * ranks[k];
        }
        let dim = offsets[n_atoms];
        let mut h = Array2::<f64>::zeros((dim, dim));
        for blk in blocks {
            let (r_i, r_j) = (ranks[blk.atom_i], ranks[blk.atom_j]);
            let (off_i, off_j) = (offsets[blk.atom_i], offsets[blk.atom_j]);
            let (m_i, m_j) = blk.g.dim();
            for li in 0..m_i {
                for lj in 0..m_j {
                    for a in 0..r_i {
                        for b in 0..r_j {
                            h[[off_i + li * r_i + a, off_j + lj * r_j + b]] +=
                                blk.g[[li, lj]] * blk.w[[a, b]];
                        }
                    }
                }
            }
        }
        h
    }

    /// `FactoredFrameKroneckerOp` must equal its dense `g ⊗ (UᵀU)` reference on
    /// every interface, with VARIABLE per-atom rank (`r_0 = 2`, `r_1 = 3`) and a
    /// genuine cross-atom output factor `U_0ᵀU_1 ≠ 0`.
    #[test]
    fn factored_frame_kronecker_matches_dense_reference() {
        // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3. dim = 4 + 9 = 13.
        let ranks = vec![2usize, 3];
        let basis_sizes = vec![2usize, 3];
        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
        let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
        let g10 = g01.t().to_owned();
        // Within-atom frame factors are identity (orthonormal U); the cross
        // factor U_0ᵀU_1 (2×3) is a generic dense principal-angle matrix.
        let w00 = Array2::<f64>::eye(2);
        let w11 = Array2::<f64>::eye(3);
        let w01 = array![[0.8_f64, 0.1, -0.05], [0.0, 0.7, 0.2]];
        let w10 = w01.t().to_owned();
        let blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00.clone(),
                w: w00.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11.clone(),
                w: w11.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01.clone(),
                w: w01.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10.clone(),
                w: w10.clone(),
            },
        ];
        let op = FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), blocks.clone())
            .expect("op");
        assert_eq!(op.dim(), 13);
        let reference = factored_reference_dense(&ranks, &basis_sizes, &blocks);

        // to_dense.
        let dense = op.to_dense();
        for i in 0..13 {
            for j in 0..13 {
                assert!(
                    (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    dense[[i, j]],
                    reference[[i, j]]
                );
            }
        }
        // matvec == reference·x.
        let x: Vec<f64> = (0..13).map(|i| 0.13 * (i as f64) - 0.4).collect();
        let mut y = vec![0.0_f64; 13];
        op.matvec(&x, &mut y);
        for i in 0..13 {
            let mut expect = 0.0;
            for j in 0..13 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
        // diagonal.
        let mut diag = vec![0.0_f64; 13];
        op.diagonal(&mut diag);
        for i in 0..13 {
            assert!(
                (diag[i] - reference[[i, i]]).abs() < 1e-12,
                "diagonal mismatch at {i}"
            );
        }
        // block over each atom's β range.
        let offsets_ranges = [0..4usize, 4..13usize];
        for id in 0..2 {
            let b = offsets_ranges[id].end - offsets_ranges[id].start;
            let mut blk = Array2::<f64>::zeros((b, b));
            op.block(BetaBlockId(id), &offsets_ranges, &mut blk);
            for bi in 0..b {
                for bj in 0..b {
                    let gi = offsets_ranges[id].start + bi;
                    let gj = offsets_ranges[id].start + bj;
                    assert!(
                        (blk[[bi, bj]] - reference[[gi, gj]]).abs() < 1e-12,
                        "block {id} mismatch at ({bi},{bj})"
                    );
                }
            }
        }
    }

    /// Strict-generalization pin: with every `r_k = p` and `U_k = I_p` (so all
    /// frame factors are identity), `FactoredFrameKroneckerOp` reproduces
    /// `SparseBlockKroneckerPenaltyOp` (the `G ⊗ I_p` data Gram) bit-for-bit on
    /// matvec — i.e. the full-`B` border is the `r = p` special case of the
    /// factored op, not a separate path.
    #[test]
    fn factored_frame_kronecker_reduces_to_sparse_block_at_full_rank() {
        let p = 2usize;
        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
        let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
        let g10 = g01.t().to_owned();
        // Factored op with r_k = p, U = I_p (w = I_p everywhere).
        let ident = Array2::<f64>::eye(p);
        let factored = FactoredFrameKroneckerOp::new(
            vec![p, p],
            vec![2, 2],
            vec![
                FactoredFrameGBlock {
                    atom_i: 0,
                    atom_j: 0,
                    g: g00.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 1,
                    atom_j: 1,
                    g: g11.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 0,
                    atom_j: 1,
                    g: g01.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 1,
                    atom_j: 0,
                    g: g10.clone(),
                    w: ident.clone(),
                },
            ],
        )
        .expect("factored op");
        // Equivalent SparseBlockKroneckerPenaltyOp (μ-major / oc-minor, p=2).
        let sparse = SparseBlockKroneckerPenaltyOp {
            p,
            dim_a: 4,
            k: 8,
            blocks: vec![
                SparseGBlock {
                    row_off: 0,
                    col_off: 0,
                    data: g00,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 2,
                    data: g11,
                },
                SparseGBlock {
                    row_off: 0,
                    col_off: 2,
                    data: g01,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 0,
                    data: g10,
                },
            ],
        };
        assert_eq!(factored.dim(), sparse.dim());
        let x: Vec<f64> = (0..8).map(|i| 0.2 * (i as f64) - 0.5).collect();
        let mut yf = vec![0.0_f64; 8];
        let mut ys = vec![0.0_f64; 8];
        factored.matvec(&x, &mut yf);
        sparse.matvec(&x, &mut ys);
        for i in 0..8 {
            assert!(
                (yf[i] - ys[i]).abs() < 1e-12,
                "full-rank factored op must equal SparseBlockKronecker at {i}: {} vs {}",
                yf[i],
                ys[i]
            );
        }
    }

    /// Modified Gram–Schmidt orthonormalization of the columns of a `p × r`
    /// matrix (`r ≤ p`), used by the frame-constructor tests to build genuine
    /// `St(p, r)` representatives. Returns the orthonormal `Q` (`p × r`).
    fn mgs_orthonormalize(a: &Array2<f64>) -> Array2<f64> {
        let (p, r) = a.dim();
        let mut q = a.clone();
        for j in 0..r {
            // Subtract projections onto the already-orthonormalized columns.
            for i in 0..j {
                let mut dot = 0.0;
                for c in 0..p {
                    dot += q[[c, i]] * q[[c, j]];
                }
                for c in 0..p {
                    q[[c, j]] -= dot * q[[c, i]];
                }
            }
            let mut nrm = 0.0;
            for c in 0..p {
                nrm += q[[c, j]] * q[[c, j]];
            }
            let nrm = nrm.sqrt();
            assert!(nrm > 1e-9, "mgs column {j} degenerate");
            for c in 0..p {
                q[[c, j]] /= nrm;
            }
        }
        q
    }

    /// `frame_output_gram` of an orthonormal frame with itself is the identity.
    #[test]
    fn frame_output_gram_orthonormal_is_identity() {
        let p = 5usize;
        let r = 3usize;
        // A deterministic-but-generic p×r seed, then orthonormalize.
        let mut seed = Array2::<f64>::zeros((p, r));
        for c in 0..p {
            for a in 0..r {
                seed[[c, a]] = ((c as f64) * 0.37 + (a as f64) * 1.31).sin() + 0.1 * (a as f64);
            }
        }
        let u = mgs_orthonormalize(&seed);
        let g = frame_output_gram(u.view(), u.view());
        assert_eq!(g.dim(), (r, r));
        for a in 0..r {
            for b in 0..r {
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (g[[a, b]] - expect).abs() < 1e-12,
                    "UᵀU not identity at ({a},{b}): {}",
                    g[[a, b]]
                );
            }
        }
    }

    /// `from_frames_and_blocks` with two genuinely orthonormal frames must
    /// reproduce the hand-built dense `g ⊗ (UᵀU)` reference on every interface,
    /// computing the `W_ij` factors itself from the supplied frames.
    #[test]
    fn from_frames_and_blocks_matches_dense_reference() {
        let p = 4usize;
        // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3.
        let basis_sizes = vec![2usize, 3];
        // Build two generic seeds and orthonormalize into St(p, r) frames.
        let mut seed0 = Array2::<f64>::zeros((p, 2));
        let mut seed1 = Array2::<f64>::zeros((p, 3));
        for c in 0..p {
            for a in 0..2 {
                seed0[[c, a]] = ((c as f64) * 0.91 - (a as f64) * 0.5).cos() + 0.2 * (c as f64);
            }
            for a in 0..3 {
                seed1[[c, a]] = ((c as f64) * 0.23 + (a as f64) * 1.7).sin() - 0.3 * (a as f64);
            }
        }
        let u0 = mgs_orthonormalize(&seed0);
        let u1 = mgs_orthonormalize(&seed1);

        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
        let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
        let g10 = g01.t().to_owned();

        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        g_blocks.insert((0, 0), g00.clone());
        g_blocks.insert((1, 1), g11.clone());
        g_blocks.insert((0, 1), g01.clone());
        g_blocks.insert((1, 0), g10.clone());

        let frames = vec![Some(u0.clone()), Some(u1.clone())];
        let op =
            FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
                .expect("from_frames_and_blocks");
        // dim = M_0·r_0 + M_1·r_1 = 2·2 + 3·3 = 13.
        assert_eq!(op.dim(), 13);

        // Hand-built dense reference: W_ij = U_iᵀ U_j computed independently.
        let ranks = vec![2usize, 3];
        let w00 = frame_output_gram(u0.view(), u0.view());
        let w11 = frame_output_gram(u1.view(), u1.view());
        let w01 = frame_output_gram(u0.view(), u1.view());
        let w10 = frame_output_gram(u1.view(), u0.view());
        let ref_blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00,
                w: w00,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11,
                w: w11,
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01,
                w: w01,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10,
                w: w10,
            },
        ];
        let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

        let dense = op.to_dense();
        for i in 0..13 {
            for j in 0..13 {
                assert!(
                    (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    dense[[i, j]],
                    reference[[i, j]]
                );
            }
        }
        // matvec == reference·x.
        let x: Vec<f64> = (0..13).map(|i| 0.17 * (i as f64) - 0.6).collect();
        let mut y = vec![0.0_f64; 13];
        op.matvec(&x, &mut y);
        for i in 0..13 {
            let mut expect = 0.0;
            for j in 0..13 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
    }

    /// Mixed framed/unframed case: atom 0 framed (`r_0 = 2 < p = 4`), atom 1
    /// unframed (`None → r_1 = p = 4`). The constructor must stand `I_p` in for
    /// the missing frame, so the within-atom-1 block is exactly `g_11 ⊗ I_4`.
    #[test]
    fn from_frames_and_blocks_mixed_framed_unframed() {
        let p = 4usize;
        let basis_sizes = vec![2usize, 2]; // M_0 = 2, M_1 = 2.
        // Atom 0 gets a genuine orthonormal 4×2 frame; atom 1 stays full-B.
        let mut seed0 = Array2::<f64>::zeros((p, 2));
        for c in 0..p {
            for a in 0..2 {
                seed0[[c, a]] = ((c as f64) * 0.61 + (a as f64) * 0.9).cos() - 0.15 * (c as f64);
            }
        }
        let u0 = mgs_orthonormalize(&seed0);

        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
        let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
        let g10 = g01.t().to_owned();

        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        g_blocks.insert((0, 0), g00.clone());
        g_blocks.insert((1, 1), g11.clone());
        g_blocks.insert((0, 1), g01.clone());
        g_blocks.insert((1, 0), g10.clone());

        let frames = vec![Some(u0.clone()), None];
        let op =
            FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
                .expect("from_frames_and_blocks mixed");

        // dim = M_0·r_0 + M_1·r_1 = 2·2 + 2·4 = 12.
        assert_eq!(op.ranks, vec![2usize, 4]);
        assert_eq!(op.dim(), 12);

        // The within-unframed-atom block (atom 1) must be exactly g_11 ⊗ I_4.
        // Atom 1's β range starts at offset M_0·r_0 = 4 and spans M_1·r_1 = 8.
        let dense = op.to_dense();
        let off1 = 4usize;
        for li in 0..2 {
            for lj in 0..2 {
                for a in 0..4 {
                    for b in 0..4 {
                        let gi = off1 + li * 4 + a;
                        let gj = off1 + lj * 4 + b;
                        let expect = if a == b { g11[[li, lj]] } else { 0.0 };
                        assert!(
                            (dense[[gi, gj]] - expect).abs() < 1e-12,
                            "g_11 ⊗ I_4 mismatch at ({gi},{gj}): {} vs {expect}",
                            dense[[gi, gj]]
                        );
                    }
                }
            }
        }

        // Full dense reference: W computed with U_1 = I_p for the unframed atom.
        let ranks = vec![2usize, 4];
        let ident_p = Array2::<f64>::eye(p);
        let w00 = frame_output_gram(u0.view(), u0.view());
        let w11 = frame_output_gram(ident_p.view(), ident_p.view());
        let w01 = frame_output_gram(u0.view(), ident_p.view());
        let w10 = frame_output_gram(ident_p.view(), u0.view());
        let ref_blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00,
                w: w00,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11.clone(),
                w: w11,
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01,
                w: w01,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10,
                w: w10,
            },
        ];
        let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

        // matvec == reference·x.
        let x: Vec<f64> = (0..12).map(|i| 0.11 * (i as f64) - 0.4).collect();
        let mut y = vec![0.0_f64; 12];
        op.matvec(&x, &mut y);
        for i in 0..12 {
            let mut expect = 0.0;
            for j in 0..12 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "mixed matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
    }

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

        let (delta_t, delta_beta, _diag) = sys.solve(0.0, 0.0).expect("arrow-schur solve");
        let streaming_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(1));
        let (delta_t_stream, delta_beta_stream, _diag_stream) = sys
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
        let xref = cholesky_solve_vector(&lj, &neg_g);
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

    fn diagonal_arrow_fixture(row_min: f64, schur_min: f64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(2, 2, 2);
        sys.rows[0].htt = array![[row_min, 0.0], [0.0, row_min + 1.0]];
        sys.rows[1].htt = array![[row_min + 2.0, 0.0], [0.0, row_min + 3.0]];
        for row in sys.rows.iter_mut() {
            row.htbeta.fill(0.0);
            row.gt.fill(0.0);
        }
        sys.hbb = array![[schur_min, 0.0], [0.0, schur_min + 1.0]];
        sys.gb.fill(0.0);
        sys
    }

    fn diagonal_fixture_dense_lambda_min(sys: &ArrowSchurSystem) -> f64 {
        let mut out = f64::INFINITY;
        for row in &sys.rows {
            for axis in 0..row.htt.nrows() {
                out = out.min(row.htt[[axis, axis]]);
            }
        }
        for axis in 0..sys.hbb.nrows() {
            out = out.min(sys.hbb[[axis, axis]]);
        }
        out
    }

    #[test]
    fn arrow_factor_min_pivot_matches_dense_lambda_min_ordering() {
        let weak = diagonal_arrow_fixture(0.2, 0.8);
        let strong = diagonal_arrow_fixture(0.7, 1.2);
        let options = ArrowSolveOptions::direct();
        let (_dt_w, _db_w, weak_cache) =
            solve_arrow_newton_step_with_options(&weak, 0.0, 0.0, &options)
                .expect("weak diagonal fixture should factor");
        let (_dt_s, _db_s, strong_cache) =
            solve_arrow_newton_step_with_options(&strong, 0.0, 0.0, &options)
                .expect("strong diagonal fixture should factor");

        let weak_lambda = diagonal_fixture_dense_lambda_min(&weak);
        let strong_lambda = diagonal_fixture_dense_lambda_min(&strong);
        assert!(weak_lambda < strong_lambda);

        let weak_pivot = arrow_factor_min_pivot(&weak_cache)
            .min_pivot
            .expect("weak pivot");
        let strong_pivot = arrow_factor_min_pivot(&strong_cache)
            .min_pivot
            .expect("strong pivot");
        assert_abs_diff_eq!(weak_pivot, weak_lambda, epsilon = 1.0e-14);
        assert_abs_diff_eq!(strong_pivot, strong_lambda, epsilon = 1.0e-14);
        assert!(weak_pivot < strong_pivot);
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
            convergence_objective_rel_tol: DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL,
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

    /// Issue #195 / gam#578: a per-row block that is barely-PD (smallest
    /// pivot on the order of ε·trace — a rank-deficient / over-parameterized
    /// decoder atom) factors successfully but is unsafe to use raw in the
    /// Schur reduction. The κ proxy is folded INTO the per-row ridge
    /// escalation loop: rather than reject such a block outright (which made
    /// the advertised Arrow-Schur ridge never actually run and aborted the
    /// whole SAE fit, gam#578), `factor_one_row` lifts this row's ridge until
    /// the block is BOTH positive-definite and well-conditioned, then returns
    /// a genuinely conditioned factor safe to plug into
    /// `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`.
    /// Only a block that cannot be conditioned even at `ridge_cap` errors.
    #[test]
    fn factor_one_row_conditions_barely_pd_block_via_ridge() {
        let d = 2;
        let k = 2;
        let mut row = ArrowRowBlock::new(d, k);
        // Matrix from the issue body: PD by an exact ε along the second
        // direction. Cholesky succeeds at ridge 0, but κ ≈ 1e14 — far past
        // the safe inversion regime. This is exactly the rank-deficient
        // decoder-atom block gam#578 advertised the ridge would stabilize.
        row.htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
        row.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row.gt = array![0.0_f64, 0.0];

        // The fix: instead of rejecting, the escalation loop lifts this
        // row's ridge until the factor is well-conditioned. The returned
        // factor must satisfy the κ ceiling that a raw barely-PD block fails.
        let factor = factor_one_row(&row, 0.0, d, 0, false).expect(
            "barely-PD H_tt must be CONDITIONED by per-row ridge escalation, not rejected (gam#578)",
        );
        let kappa = cholesky_factor_kappa_estimate(&factor);
        assert!(
            kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
            "conditioned factor must be within the safe-inversion κ ceiling; got κ={kappa:e}"
        );
        // The factor is a genuine Cholesky of the ridge-lifted block
        // H_tt + ridge_eff·I (ridge_eff ≥ 0), so reconstructing L Lᵀ must
        // match H_tt up to a nonnegative diagonal shift (never below).
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += factor[[i, kk]] * factor[[j, kk]];
                }
                if i == j {
                    assert!(
                        acc >= row.htt[[i, j]] - 1e-12,
                        "diagonal of L Lᵀ must be H_tt + (nonneg ridge) at ({i},{j}): \
                         {acc} vs {}",
                        row.htt[[i, j]]
                    );
                } else {
                    assert!(
                        (acc - row.htt[[i, j]]).abs() < 1e-9,
                        "off-diagonal of L Lᵀ must equal H_tt at ({i},{j}): {acc} vs {}",
                        row.htt[[i, j]]
                    );
                }
            }
        }

        // Evidence/log-det mode (`tolerate_ill_conditioning = true`) must
        // accept the same barely-PD block and return its genuine Cholesky
        // factor — the diagonal gives an exact log-determinant.
        let factor = factor_one_row(&row, 0.0, d, 0, true)
            .expect("tolerate_ill_conditioning must accept a barely-PD-but-PD block");
        // L Lᵀ must reproduce the original block (the factor is real, not a
        // damped surrogate).
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += factor[[i, kk]] * factor[[j, kk]];
                }
                assert!(
                    (acc - row.htt[[i, j]]).abs() < 1e-12,
                    "tolerated factor must satisfy L Lᵀ = H_tt at ({i},{j})"
                );
            }
        }

        // A genuinely non-PD block must STILL error even under tolerance —
        // the flag lifts only the κ rejection, not the PD requirement.
        let mut row_npd = ArrowRowBlock::new(d, k);
        row_npd.htt = array![[1.0_f64, 2.0], [2.0, 1.0]]; // indefinite (eigvals 3, -1)
        row_npd.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_npd.gt = array![0.0_f64, 0.0];
        let npd = factor_one_row(&row_npd, 0.0, d, 0, true);
        assert!(
            matches!(npd, Err(ArrowSchurError::PerRowFactorFailed { .. })),
            "non-PD block must error even with tolerate_ill_conditioning; got {npd:?}"
        );

        // Sanity: a well-conditioned block at the same dimension still
        // factors successfully.
        let mut row_ok = ArrowRowBlock::new(d, k);
        row_ok.htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        row_ok.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_ok.gt = array![0.0_f64, 0.0];
        factor_one_row(&row_ok, 0.0, d, 0, false)
            .expect("well-conditioned block must still factor at ridge_t=0");

        // A block that cannot be conditioned at all — a non-finite entry —
        // is genuinely broken: no finite ridge shift repairs it, so the
        // escalation loop must still surface a typed `PerRowFactorFailed`
        // for the outer loop rather than loop forever or return garbage.
        let mut row_nan = ArrowRowBlock::new(d, k);
        row_nan.htt = array![[f64::NAN, 0.0], [0.0, 1.0]];
        row_nan.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_nan.gt = array![0.0_f64, 0.0];
        let nan = factor_one_row(&row_nan, 1.0e-6, d, 0, false);
        assert!(
            matches!(nan, Err(ArrowSchurError::PerRowFactorFailed { .. })),
            "non-finite block must surface PerRowFactorFailed, not loop or condition; got {nan:?}"
        );
    }

    #[test]
    fn factor_one_row_conditions_scalar_tiny_pivot_via_ridge() {
        let d = 1;
        let k = 1;
        let mut row = ArrowRowBlock::new(d, k);
        row.htt = array![[1.0e-20_f64]];
        row.htbeta = array![[1.0_f64]];
        row.gt = array![0.0_f64];

        let factor = factor_one_row(&row, 0.0, d, 0, false)
            .expect("tiny positive scalar pivot must be ridge-conditioned");
        let pivot = factor[[0, 0]] * factor[[0, 0]];
        assert!(
            pivot >= safe_spd_pivot_min(1.0),
            "scalar pivot must be lifted above the absolute safe floor; got {pivot:e}"
        );
        assert!(
            pivot > row.htt[[0, 0]],
            "scalar block must not be accepted at the raw tiny pivot"
        );

        let tolerated = factor_one_row(&row, 0.0, d, 0, true)
            .expect("tolerated log-det path must accept a positive scalar block");
        let raw_pivot = tolerated[[0, 0]] * tolerated[[0, 0]];
        assert!(
            (raw_pivot - row.htt[[0, 0]]).abs() < 1.0e-30,
            "tolerated factor must remain the raw scalar Cholesky"
        );
    }

    #[test]
    fn sys_htbeta_materialize_row_sums_operator_and_dense_slab() {
        let mut sys = ArrowSchurSystem::new(1, 1, 3);
        sys.rows[0].htbeta = array![[0.25_f64, 0.5, 0.75]];
        sys.activate_dense_htbeta_supplement();
        sys.set_row_htbeta_operator(
            |row_idx, x, out| {
                assert_eq!(row_idx, 0);
                out[0] += 2.0 * x[0] - x[1] + 0.5 * x[2];
            },
            |row_idx, v, out| {
                assert_eq!(row_idx, 0);
                out[0] += 2.0 * v[0];
                out[1] -= v[0];
                out[2] += 0.5 * v[0];
            },
        );

        let htbeta = sys_htbeta_materialize_row(&sys, 0, &sys.rows[0]);
        assert_eq!(htbeta, array![[2.25_f64, -0.5, 1.25]]);
    }

    /// Issue #195 / gam#578 / gam#845: when the per-row block is barely-PD at
    /// `ridge_t = 0` (a rank-deficient atom), the per-row factor must
    /// CONDITION it through the folded ridge escalation, and the full
    /// `solve_with_lm_escalation_inner` must produce a finite Newton step
    /// rather than aborting the whole fit.
    ///
    /// Note (gam#845): per-row κ-conditioning bounds each block's inverse
    /// spectrum, but it cannot on its own guarantee the *dense Schur
    /// complement* `S = H_ββ − Σ_i H_tβᵀ(H_tt+ridge)⁻¹H_tβ` stays PD: the
    /// per-row ceiling still admits a ~`1/κ_ceiling`-scale smallest pivot, so
    /// `(H_tt+ridge)⁻¹` retains a ~`κ_ceiling`-scale eigenvalue that, after the
    /// Schur subtraction, can drive `S` strongly indefinite when
    /// `‖H_tβ‖²·κ_ceiling ≫ ‖H_ββ‖`. Outer LM ridge escalation is the correct,
    /// principled recovery for that regime. The achievable invariant is
    /// therefore: a finite, well-conditioned Newton step is produced (via a
    /// bounded number of outer ridge escalations), NOT zero escalations.
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

        // Direct factor at ridge_t=0 CONDITIONS the barely-PD block via the
        // folded per-row ridge escalation (gam#578: the advertised ridge
        // genuinely stabilizes the deficient direction instead of rejecting
        // it) and returns a well-conditioned factor satisfying the κ ceiling.
        let factor = factor_one_row(&sys.rows[0], 0.0, d, 0, false)
            .expect("barely-PD row must be conditioned, not rejected (gam#578)");
        let kappa = cholesky_factor_kappa_estimate(&factor);
        assert!(
            kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
            "conditioned per-row factor must satisfy the κ ceiling; got κ={kappa:e}"
        );

        // The full LM-escalating wrapper produces a finite, well-conditioned
        // Newton step. Per-row conditioning alone cannot keep the dense Schur
        // complement PD here (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖), so the proximal
        // wrapper escalates the outer ridge a bounded number of times — this
        // is the correct recovery (gam#845), not a failure.
        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta, diag) = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
            .expect("LM escalation must recover from a barely-PD per-row block");
        for v in delta_t.iter().chain(delta_beta.iter()) {
            assert!(v.is_finite(), "recovered step must be finite: {v}");
        }
        assert!(
            diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            "recovery must use a bounded number of outer ridge escalations; got {}",
            diag.ridge_escalations
        );
    }

    /// `latent_block_inverse_diagonal` must reproduce the `t`-block diagonal of
    /// the dense bordered-arrow inverse `(H⁻¹)_tt` to machine precision.
    ///
    /// Build a small `(N=3, d=2, K=2)` arrow system, factor it through the
    /// real solve to obtain an [`ArrowFactorCache`], then assemble the full
    /// dense `(N·d + K) × (N·d + K)` Hessian from the same per-row blocks,
    /// invert it via dense Cholesky, and compare diagonals.
    #[test]
    fn latent_block_inverse_diagonal_matches_dense() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);

        // Distinct, well-conditioned per-row blocks and cross-blocks.
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        // SPD shared block; the full bordered H must stay PD.
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("direct arrow solve should factor this SPD system");

        // Assemble the dense bordered-arrow Hessian H (t-coords first, then β).
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            // H_tt^(i) block.
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
            }
            // H_tβ^(i) (d×K) and its transpose into the β border.
            for r in 0..d {
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        // H_ββ.
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }

        // Dense inverse via Cholesky against the identity.
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

        let diag = cache
            .latent_block_inverse_diagonal()
            .expect("dense Schur cache must support the selected-inverse diagonal");
        assert_eq!(diag.len(), n * d);
        for i in 0..n {
            for j in 0..d {
                let idx = i * d + j; // homogeneous system ⇒ row_offsets[i] == i*d.
                let expected = h_inv[[idx, idx]];
                let got = diag[idx];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "row {i} axis {j}: selected-inverse diag {got} vs dense {expected}"
                );
            }
        }

        // The per-(atom, axis) trace is a sum over the relevant indices; e.g.
        // tr[(H⁻¹)_tt] over all latent coords equals the dense t-block trace.
        let trace_selected: f64 = diag.iter().sum();
        let trace_dense: f64 = (0..n * d).map(|idx| h_inv[[idx, idx]]).sum();
        assert!(
            (trace_selected - trace_dense).abs() < 1e-9,
            "full latent trace {trace_selected} vs dense {trace_dense}"
        );
    }

    /// `solve_full` (#1006 IFT/adjoint back-solve) must reproduce the dense
    /// bordered-arrow inverse applied to an arbitrary arrow-layout RHS, and
    /// solving against the system's own gradient must reproduce the Newton
    /// step the solver itself returned (`Δ = H⁻¹g`) — both to near machine
    /// precision on the ridge-0 Direct factor.
    #[test]
    fn solve_full_matches_dense_inverse_and_newton_step() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[0].gt = array![0.4_f64, -0.7];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[1].gt = array![-0.2_f64, 0.9];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        sys.rows[2].gt = array![1.1_f64, 0.3];
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.5_f64, -0.8];

        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("direct arrow solve should factor this SPD system");

        // (a) The solver returns the DESCENT step Δ = −H⁻¹g; solve_full is the
        // bare inverse application H⁻¹g, so u must equal −Δ exactly.
        let mut g_t = Array1::<f64>::zeros(n * d);
        for i in 0..n {
            for j in 0..d {
                g_t[i * d + j] = sys.rows[i].gt[j];
            }
        }
        let (u_t, u_beta) = cache
            .solve_full(g_t.view(), sys.gb.view())
            .expect("solve_full on the ridge-0 Direct cache");
        for idx in 0..n * d {
            assert!(
                (u_t[idx] + delta_t[idx]).abs() < 1e-10,
                "t[{idx}]: solve_full {} vs −(Newton step) {}",
                u_t[idx],
                -delta_t[idx]
            );
        }
        for c in 0..k {
            assert!(
                (u_beta[c] + delta_beta[c]).abs() < 1e-10,
                "beta[{c}]: solve_full {} vs −(Newton step) {}",
                u_beta[c],
                -delta_beta[c]
            );
        }

        // (b) Arbitrary RHS vs the dense bordered inverse.
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let mut w_full = Array1::<f64>::zeros(dim);
        for (idx, v) in w_full.iter_mut().enumerate() {
            *v = 0.3 + 0.17 * (idx as f64) * (if idx % 2 == 0 { 1.0 } else { -1.0 });
        }
        let dense_u = cholesky_solve_vector(&l, &w_full);
        let (u_t2, u_beta2) = cache
            .solve_full(
                w_full.slice(ndarray::s![..n * d]),
                w_full.slice(ndarray::s![n * d..]),
            )
            .expect("solve_full on arbitrary RHS");
        for idx in 0..n * d {
            assert!(
                (u_t2[idx] - dense_u[idx]).abs() < 1e-10,
                "t[{idx}]: solve_full {} vs dense {}",
                u_t2[idx],
                dense_u[idx]
            );
        }
        for c in 0..k {
            assert!(
                (u_beta2[c] - dense_u[n * d + c]).abs() < 1e-10,
                "beta[{c}]: solve_full {} vs dense {}",
                u_beta2[c],
                dense_u[n * d + c]
            );
        }
    }

    /// `schur_inverse_apply` / `schur_inverse_diagonal` must reproduce the
    /// β-block of the dense bordered-arrow inverse `(H⁻¹)_ββ = S_β⁻¹`, and a
    /// caller-assembled `tr(S_β⁻¹ M)` must match the dense Kron-block trace —
    /// the β-side analogue used by the SAE λ_smooth Fellner-Schall step.
    #[test]
    fn schur_inverse_beta_block_matches_dense() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("direct arrow solve should factor this SPD system");

        // Dense bordered H and its inverse (same assembly as the t-block test).
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
            }
            for r in 0..d {
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

        // The β-block of H⁻¹ is the bottom-right K×K corner.
        let beta_off = n * d;

        // schur_inverse_diagonal vs dense β-block diagonal.
        let sdiag = cache
            .schur_inverse_diagonal()
            .expect("dense Schur cache must support schur_inverse_diagonal");
        assert_eq!(sdiag.len(), k);
        for j in 0..k {
            let expected = h_inv[[beta_off + j, beta_off + j]];
            assert!(
                (sdiag[j] - expected).abs() < 1e-9,
                "β diag {j}: {} vs dense {expected}",
                sdiag[j]
            );
        }

        // schur_inverse_apply against each unit column reproduces the full
        // β-block (every entry, not just the diagonal).
        for col in 0..k {
            let mut e = Array1::<f64>::zeros(k);
            e[col] = 1.0;
            let x = cache
                .schur_inverse_apply(e.view())
                .expect("dense Schur cache must support schur_inverse_apply");
            for r in 0..k {
                let expected = h_inv[[beta_off + r, beta_off + col]];
                assert!(
                    (x[r] - expected).abs() < 1e-9,
                    "S_β⁻¹[{r},{col}] {} vs dense {expected}",
                    x[r]
                );
            }
        }

        // Caller-assembled Kron trace tr(S_β⁻¹ M) for a single atom block
        // M = A_k ⊗ I_p with K = M_k · p. Here M_k = 1, p = 2 ⇒ K = 2, so
        // A_k is 1×1 = [a] and M = a·I_2. tr(S_β⁻¹ M) = a·tr(S_β⁻¹).
        let a_scalar = 0.75_f64;
        let mut trace = 0.0_f64;
        for col in 0..k {
            // (A_k ⊗ I_p) e_col = a_scalar · e_col for this M_k=1 block.
            let mut m_col = Array1::<f64>::zeros(k);
            m_col[col] = a_scalar;
            let z = cache
                .schur_inverse_apply(m_col.view())
                .expect("schur_inverse_apply");
            trace += z[col];
        }
        let trace_dense: f64 = a_scalar
            * (0..k)
                .map(|j| h_inv[[beta_off + j, beta_off + j]])
                .sum::<f64>();
        assert!(
            (trace - trace_dense).abs() < 1e-9,
            "Kron-block trace {trace} vs dense {trace_dense}"
        );

        // schur_inverse_block must reproduce a contiguous dense sub-block of
        // (H⁻¹)_ββ — both the full β-block and an interior single-coordinate
        // window — and be exactly symmetric.
        let full = cache
            .schur_inverse_block(0..k)
            .expect("dense Schur cache must support schur_inverse_block");
        assert_eq!(full.dim(), (k, k));
        for r in 0..k {
            for c in 0..k {
                let expected = h_inv[[beta_off + r, beta_off + c]];
                assert!(
                    (full[[r, c]] - expected).abs() < 1e-9,
                    "block[{r},{c}] {} vs dense {expected}",
                    full[[r, c]]
                );
                assert!(
                    (full[[r, c]] - full[[c, r]]).abs() < 1e-12,
                    "schur_inverse_block must be symmetric at [{r},{c}]"
                );
            }
        }
        let sub = cache
            .schur_inverse_block(1..k)
            .expect("interior block must be supported");
        assert_eq!(sub.dim(), (k - 1, k - 1));
        assert!(
            (sub[[0, 0]] - h_inv[[beta_off + 1, beta_off + 1]]).abs() < 1e-9,
            "interior block [1,1] {} vs dense {}",
            sub[[0, 0]],
            h_inv[[beta_off + 1, beta_off + 1]]
        );
        // Out-of-range block must error rather than panic.
        assert!(cache.schur_inverse_block(0..(k + 1)).is_err());
    }

    /// Evidence/log-det mode: a per-row `H_tt` that is PD but ill-conditioned
    /// (κ above the safe-Schur ceiling) is handled differently by the two
    /// solve paths. The default `direct()` path conditions each row to the
    /// safe-Schur κ ceiling; when that per-row conditioning is insufficient to
    /// keep the *dense Schur complement* PD (gam#845), the single-shot solve
    /// correctly reports a recoverable factorization error and the
    /// LM-escalating wrapper recovers it with a finite, well-conditioned step.
    ///
    /// `with_ill_conditioning_tolerated()` accepts the RAW (undamped) blocks.
    /// Its contract has two sides, pinned on two fixtures:
    ///   * row-PD but assembled-INDEFINITE H (strong coupling into near-null
    ///     t-directions) → honest refusal. Per-row PD does not imply bordered-
    ///     system PD, and an exact `log|H|` does not exist on the Cholesky
    ///     branch — fabricating one would corrupt the evidence.
    ///   * row κ ≈ 1e9 but assembled H genuinely PD (coupling subordinate to
    ///     the weak curvature) → a usable cache whose log-determinant equals
    ///     the exact dense `log|H|`, undistorted by any κ-ceiling ridge. This
    ///     is the SAE evidence path under a wide ARD α sweep.
    #[test]
    fn ill_conditioning_tolerated_returns_cache_with_exact_logdet() {
        let n = 2usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        // Barely-PD rows: second pivot ~1e-9 of the first ⇒ κ ≈ 1e9, above
        // the safe-Schur ceiling but genuinely PD (Cholesky succeeds).
        sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
        sys.rows[0].htbeta = array![[0.3_f64, 0.1], [0.05, 0.2]];
        sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
        sys.rows[1].htbeta = array![[0.2_f64, -0.1], [0.1, 0.15]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
        sys.gb = array![0.0_f64, 0.0];

        // factor_one_row conditions each barely-PD per-row block to the
        // safe-Schur κ ceiling (gam#578): the raw block fails the ceiling but
        // the ridge-lifted factor satisfies it. Verify the per-row contract
        // directly — this is what per-row conditioning genuinely guarantees.
        for i in 0..n {
            let factor = factor_one_row(&sys.rows[i], 0.0, d, i, false)
                .expect("barely-PD row must be conditioned, not rejected (gam#578)");
            let kappa = cholesky_factor_kappa_estimate(&factor);
            assert!(
                kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
                "conditioned per-row factor {i} must satisfy the safe-Schur κ ceiling; got κ={kappa:e}"
            );
        }

        // Per-row conditioning alone cannot keep the dense Schur complement PD
        // for these inputs (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖, gam#845), so the
        // single-shot strict solve reports a recoverable factorization error
        // rather than a finite step.
        let single_shot =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &ArrowSolveOptions::direct());
        assert!(
            matches!(
                single_shot,
                Err(ArrowSchurError::SchurFactorFailed { .. })
                    | Err(ArrowSchurError::PerRowFactorIllConditioned { .. })
                    | Err(ArrowSchurError::PcgFailed { .. })
            ),
            "single-shot strict direct() cannot keep the dense Schur PD with per-row \
             conditioning alone; expected a recoverable factorization error, got {single_shot:?}"
        );

        // The LM-escalating wrapper is the correct recovery: a bounded number
        // of outer ridge escalations yields a finite, well-conditioned step.
        let (strict_dt, strict_db, strict_diag) =
            solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &ArrowSolveOptions::direct())
                .expect("LM escalation must recover the ill-conditioned strict solve (gam#845)");
        for v in strict_dt.iter().chain(strict_db.iter()) {
            assert!(v.is_finite(), "recovered strict step must be finite: {v}");
        }
        assert!(
            strict_diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            "recovery must use a bounded number of outer ridge escalations; got {}",
            strict_diag.ridge_escalations
        );

        // Evidence mode accepts the RAW (undamped) blocks. For THIS system the
        // honest answer is refusal: each per-row `H_tt` is PD in isolation, but
        // the strong coupling into the near-null t-directions makes the
        // assembled bordered H indefinite (its true Schur complement has a
        // ≈ −7.5e6 leading pivot; the full spectrum has two negative
        // eigenvalues). An exact log|H| does not exist on the Cholesky branch,
        // and tolerating ill-CONDITIONING must never fabricate a determinant
        // for an in-DEFINITE system — the SchurFactorFailed refusal is the
        // contract, not a defect.
        let opts = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let tolerate_indefinite = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &opts);
        assert!(
            matches!(
                tolerate_indefinite,
                Err(ArrowSchurError::SchurFactorFailed { .. })
            ),
            "tolerate mode must refuse the indefinite assembled H rather than fabricate \
             a log-determinant; got {tolerate_indefinite:?}"
        );

        // The regime the tolerate flag exists for: per-row κ ≈ 1e9 (above the
        // safe-Schur ceiling, so the strict path would ridge-condition the row
        // and distort the determinant) yet the assembled H is genuinely PD
        // because the coupling into the near-null t-directions is subordinate
        // to their curvature (‖H_tβ row‖² ≲ λ_min(H_tt)·λ_min(H_ββ)). Evidence
        // mode must factor the RAW blocks and report the EXACT dense log|H|,
        // undistorted by any κ-ceiling ridge.
        let mut pd_sys = ArrowSchurSystem::new(n, d, k);
        pd_sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
        pd_sys.rows[0].htbeta = array![[0.3_f64, 0.1], [3e-6, 1e-6]];
        pd_sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
        pd_sys.rows[1].htbeta = array![[0.2_f64, -0.1], [2e-6, 4e-6]];
        for row in pd_sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        pd_sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
        pd_sys.gb = array![0.0_f64, 0.0];

        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&pd_sys, 0.0, 0.0, &opts)
            .expect("tolerate mode must factor the ill-conditioned-but-PD system");

        // Cache log-determinant (Σ log|H_tt^i| + log|S_β|) must equal the exact
        // dense log|H|, regardless of conditioning — the whole point.
        let (log_det_tt, log_det_schur) = cache.arrow_log_det();
        let log_det_cache = log_det_tt + log_det_schur.expect("dense Schur factor present");

        // Dense reference: assemble H and take log|H| = 2 Σ log L_ii.
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = pd_sys.rows[i].htt[[r, c]];
                }
            }
            for r in 0..d {
                for c in 0..k {
                    let v = pd_sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = pd_sys.hbb[[r, c]];
            }
        }
        let lh = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let log_det_dense: f64 = 2.0 * (0..dim).map(|i| lh[[i, i]].ln()).sum::<f64>();

        assert!(
            (log_det_cache - log_det_dense).abs() < 1e-6,
            "tolerated-cache log|H| {log_det_cache} vs dense {log_det_dense}"
        );

        // Selected-inverse traces must still be available from the cache.
        let tdiag = cache
            .latent_block_inverse_diagonal()
            .expect("tolerated cache must support latent_block_inverse_diagonal");
        assert_eq!(tdiag.len(), n * d);
        assert!(tdiag.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn arrow_factor_slab_accessor_matches_array_blocks_bitwise() {
        let blocks = vec![
            array![[1.0_f64]],
            array![[2.0_f64, 0.0], [0.25, 3.0]],
            array![[4.0_f64, 0.0, 0.0], [0.5, 5.0, 0.0], [-0.25, 0.75, 6.0]],
        ];
        let slab = ArrowFactorSlab::from_blocks(blocks.clone());
        assert_eq!(slab.len(), blocks.len());
        for row in 0..blocks.len() {
            let view = slab.factor(row);
            assert_eq!(view.dim(), blocks[row].dim());
            for r in 0..blocks[row].nrows() {
                for c in 0..blocks[row].ncols() {
                    assert_eq!(view[[r, c]].to_bits(), blocks[row][[r, c]].to_bits());
                }
            }
        }
    }

    fn fixed_row_kernel_fixture<const D: usize>() -> (ArrowRowBlock, Array1<f64>) {
        let mut row = ArrowRowBlock::new(D, 0);
        for r in 0..D {
            for c in 0..D {
                row.htt[[r, c]] = if r == c {
                    4.0 + r as f64
                } else {
                    0.03125 * ((r + c + 1) as f64)
                };
            }
        }
        let rhs = Array1::from_iter((0..D).map(|i| 0.5 + i as f64 * 0.25));
        (row, rhs)
    }

    fn assert_fixed_row_kernels_match_dynamic<const D: usize>() -> usize {
        let (row, rhs) = fixed_row_kernel_fixture::<D>();
        let ridge = 0.125_f64;
        let fixed = factor_row_block_cholesky_fixed::<D>(&row, ridge).expect("fixed factor");
        let dynamic = factor_row_block_cholesky_dynamic(&row, ridge, D).expect("dynamic factor");
        for r in 0..D {
            for c in 0..D {
                assert_eq!(
                    fixed[[r, c]].to_bits(),
                    dynamic[[r, c]].to_bits(),
                    "factor mismatch at D={D} ({r},{c})"
                );
            }
        }

        let fixed_solve = cholesky_solve_vector_fixed::<D>(fixed.view(), rhs.view());
        let dynamic_solve = cholesky_solve_vector(dynamic.view(), rhs.view());
        for i in 0..D {
            assert_eq!(
                fixed_solve[i].to_bits(),
                dynamic_solve[i].to_bits(),
                "solve mismatch at D={D} index {i}"
            );
        }
        D
    }

    #[test]
    fn fixed_row_kernels_match_dynamic_path_bitwise() {
        let checked = assert_fixed_row_kernels_match_dynamic::<1>()
            + assert_fixed_row_kernels_match_dynamic::<2>()
            + assert_fixed_row_kernels_match_dynamic::<3>()
            + assert_fixed_row_kernels_match_dynamic::<4>();
        assert_eq!(checked, 10);
    }
}
