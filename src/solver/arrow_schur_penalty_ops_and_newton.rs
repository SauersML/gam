use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use std::ops::Range;

use std::sync::Arc;


use crate::cache::Fingerprinter;

use crate::linalg::faer_ndarray::{FaerArrayView, FaerEigh, FaerLlt};
use faer::Side;

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

/// Backward-error certificate floor, expressed as a small multiple of f64 epsilon.
const MIXED_PRECISION_CERTIFICATE_EPSILON_MULTIPLIER: f64 = 64.0;

/// User-supplied kappa margins above this are no stricter than the unit gate.
const MIXED_PRECISION_KAPPA_MARGIN_CEILING: f64 = 1.0;


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


#[derive(Debug, Clone)]
pub struct DeviceSaeSmoothBlock {
    pub global_offset: usize,
    pub factor_a: Array2<f64>,
}


#[derive(Debug, Clone)]
pub struct DeviceSaePcgData {
    pub p: usize,
    pub beta_dim: usize,
    pub a_phi: Vec<Vec<(usize, f64)>>,
    pub local_jac: Vec<Vec<f64>>,
    pub smooth_blocks: Vec<DeviceSaeSmoothBlock>,
    pub sparse_g_blocks: Vec<SparseGBlock>,
}


impl DeviceSaePcgData {
    /// Snapshot the per-row active-atom support `a_phi` into a shared `Arc<[…]>`
    /// for the CPU residency operator ([`SaeResidentReducedSchur`]). Cloned once
    /// per CG-solve build (cost `O(Σ_i m_i)`, dwarfed by the per-row factor solves
    /// in the same build), so the resident matvec borrows the index lists without
    /// re-cloning them on every CG iteration.
    fn a_phi_shared(&self) -> Arc<[Vec<(usize, f64)>]> {
        Arc::from(self.a_phi.clone().into_boxed_slice())
    }
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
    /// True when this Direct-mode point solve was served by the fully
    /// device-resident batched Arrow-Schur sequence (#1017). Lets harnesses and
    /// parity tests observe that the production auto-selection routed to the
    /// device rather than the CPU dense Cholesky, without changing the numbers.
    pub used_device_arrow: bool,
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
    /// The declining reason is logged at `info` level when the fallback fires.
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
/// when it fails, the solve reports [`MixedPrecisionStatus::F64Fallback`] and
/// logs the reason before using the f64 path.
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

    /// Turn certified mixed precision ON for the streaming/residency reduced
    /// solve unless the caller already pinned an explicit policy (#1014).
    ///
    /// Only `Off` (the inherited default) is upgraded to `Certified`; a caller
    /// that deliberately set a policy keeps it. The reduced-Schur f64 factor and
    /// every evidence log-determinant are unaffected — see
    /// [`mixed_precision_reduced_beta`].
    #[must_use]
    pub fn with_streaming_mixed_precision_default(&self) -> Self {
        let mut out = self.clone();
        if matches!(out.mixed_precision, MixedPrecisionPolicy::Off) {
            out.mixed_precision = MixedPrecisionPolicy::certified();
        }
        out
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


#[derive(Debug, Clone)]
pub struct ArrowRowGaugeDeflation {
    pub directions: Arc<[Vec<Array1<f64>>]>,
}


impl ArrowRowGaugeDeflation {
    pub fn new(directions: Vec<Vec<Array1<f64>>>) -> Self {
        Self {
            directions: Arc::from(directions.into_boxed_slice()),
        }
    }

    fn row(&self, row: usize) -> &[Array1<f64>] {
        self.directions.get(row).map(Vec::as_slice).unwrap_or(&[])
    }
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


#[derive(Debug, Clone)]
struct ArrowRowFactorResult {
    factor: Array2<f64>,
    gauge_deflated_directions: usize,
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


fn row_gauge_curvature(row: &ArrowRowBlock, d: usize, gauge: &Array1<f64>) -> Option<f64> {
    if gauge.len() != d {
        return None;
    }
    let mut acc = 0.0_f64;
    for i in 0..d {
        let gi = gauge[i];
        for j in 0..d {
            acc += gi * row.htt[[i, j]] * gauge[j];
        }
    }
    if acc.is_finite() { Some(acc) } else { None }
}


fn factor_gauge_deflated_evidence_row(
    row: &ArrowRowBlock,
    d: usize,
    gauges: &[Array1<f64>],
) -> Option<ArrowRowFactorResult> {
    const GAUGE_RAYLEIGH_EPS: f64 = 1.0e-8;
    if gauges.is_empty() {
        return None;
    }
    let max_diag = row_block_diag_scale(row, d);
    if !(max_diag.is_finite() && max_diag > 0.0) {
        return None;
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for gauge in gauges {
        if gauge.len() != d {
            continue;
        }
        let norm_sq = gauge.iter().map(|&v| v * v).sum::<f64>();
        if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
            continue;
        }
        let curvature = row_gauge_curvature(row, d, gauge)?;
        // Two-sided gauge qualification: a true orbit direction has Rayleigh
        // quotient ~ 0 from EITHER side (the observed failures sit at ~ -1e-10).
        // Strongly NEGATIVE curvature is genuine indefiniteness (e.g. the raw
        // assignment-prior logit curvature off-optimum), not a gauge — deflating
        // it would mask a real non-PD block and bias the evidence. Only
        // |g^T H g| <= eps * scale * |g|^2 qualifies (the absolute value is what
        // makes this two-sided: a large-magnitude curvature of EITHER sign is
        // disqualified, so only a genuine near-null orbit is deflated).
        if curvature.abs() > GAUGE_RAYLEIGH_EPS * max_diag * norm_sq {
            continue;
        }
        let mut direction = gauge.clone();
        for existing in &basis {
            let coeff = direction.dot(existing);
            for idx in 0..d {
                direction[idx] -= coeff * existing[idx];
            }
        }
        let residual_norm_sq = direction.iter().map(|&v| v * v).sum::<f64>();
        if !(residual_norm_sq.is_finite() && residual_norm_sq > 1.0e-24) {
            continue;
        }
        let inv_norm = residual_norm_sq.sqrt().recip();
        for value in direction.iter_mut() {
            *value *= inv_norm;
        }
        basis.push(direction);
    }
    if basis.is_empty() {
        return None;
    }

    // Faddeev-Popov stiffening of the orbit, at UNIT stiffness kappa = 1.0
    // (NOT max_diag). The direction is already unit-normalized, so each deflated
    // direction contributes exactly +1 to that eigenvalue of `deflated`, hence
    // log(1) = 0 to log|H|. This is the codebase's quotient PSEUDO-DETERMINANT
    // convention (cf. `PenaltyPseudologdet`, which evaluates log|S| over the
    // non-degenerate subspace and drops the kernel): the gauge orbit is a
    // genuine null direction of the criterion, so it must contribute NOTHING to
    // the Laplace normalizer. A theta/rho-dependent kappa (e.g. max_diag) would
    // inject a spurious log(kappa(theta,rho)) into the evidence and bias the
    // REML rho-gradient whenever a deflated direction survives to the optimum —
    // holding the deflated COUNT fixed across the solve does NOT make the VALUE
    // theta/rho-constant. kappa = 1.0 is exactly zero-derivative by
    // construction. The quotient-complement solve is identical either way, so
    // evidence-mode exactness on the non-degenerate subspace is preserved.
    // `max_diag` stays ONLY in the qualification threshold above, where it is
    // the curvature unit the orbit's near-zero Rayleigh quotient is measured
    // against — never in the stiffness. (d <= 3 blocks; kappa = 1 against large
    // other-direction curvature is a condition number the Cholesky handles
    // trivially.)
    let mut deflated = row.htt.clone();
    for direction in &basis {
        for i in 0..d {
            for j in 0..d {
                deflated[[i, j]] += direction[i] * direction[j];
            }
        }
    }
    let factor = cholesky_lower(&deflated).ok()?;
    Some(ArrowRowFactorResult {
        factor,
        gauge_deflated_directions: basis.len(),
    })
}


/// Relative spectral floor (vs the block's largest-magnitude eigenvalue) below
/// which a per-row `H_tt` eigen-direction is treated as non-identified and
/// unit-stiffness deflated rather than ridge-damped. Matches the magnitude of
/// the gauge Rayleigh qualifier and the `SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF`
/// data-null detection so the three deflation paths agree on what "flat" means.
const SPECTRAL_DEFLATION_REL_FLOOR: f64 = 1.0e-8;


/// Hysteresis half-width (as a fraction of `SPECTRAL_DEFLATION_REL_FLOOR`)
/// applied to the spectral-deflation decision for *positive* near-floor
/// eigenvalues, to stop the per-row deflation COUNT from flickering as a small
/// positive curvature direction wanders across the cutoff over a ρ/θ-walk
/// (#1117). The quotient-dimension guard (`record_evidence_gauge_deflation_count`)
/// correctly refuses to compare Laplace normalizers across different deflated
/// dimensions, so a single eigenvalue oscillating around the bare floor would
/// otherwise toggle the count 6↔7 within one optimization and trip the guard
/// spuriously, forcing a slow seed/homotopy cascade.
///
/// The decision is split by the only physically meaningful distinction at the
/// inner optimum: a NON-POSITIVE (or non-finite) eigenvalue is a genuine null /
/// indefinite quotient direction and is ALWAYS deflated — that boundary sits at
/// exact zero, far from where live curvature lives, so it does not flicker (a
/// curvature direction genuinely crossing zero IS a structural event the guard
/// must still catch). Only a *positive* eigenvalue near `floor` is ambiguous,
/// and for it we use the LOWER band edge `floor·(1−ε)`: a positive eigenvalue
/// parked at the bare floor is `> floor·(1−ε)` and is therefore consistently
/// KEPT on both sides of the walk, so the count is stable by construction. A
/// direction that is genuinely numerically flat sits orders of magnitude below
/// `floor` (a true rank deficiency, `λ ≪ floor·(1−ε)`), so it is still deflated
/// exactly as before — the converged result is unchanged wherever the old path
/// already deflated a clearly-flat or indefinite direction.
const SPECTRAL_DEFLATION_HYSTERESIS_FRACTION: f64 = 1.0e-2;


/// Unit-stiffness **spectral** Faddeev-Popov conditioning of a per-row evidence
/// block `H_tt` that the undamped Cholesky refused because it is genuinely
/// indefinite or numerically flat off the closed-form gauge orbit.
///
/// This is the spectral sibling of [`factor_gauge_deflated_evidence_row`]: that
/// one deflates a SUPPLIED orbit direction (the circle rotation gauge, etc.);
/// this one DISCOVERS the offending directions from the block's own symmetric
/// eigendecomposition, for the case (#1117/#1118, K>1 IBP/softmax row-sharing)
/// where the logit×coordinate Gauss-Newton cross term drives an eigenvalue of
/// `H_tt` negative (or to a numerically-flat near-zero) at a direction that is
/// NOT a known gauge vector. `d ≤ 3` here so the eigendecomposition is trivial.
///
/// Each eigenvalue at or below `floor = SPECTRAL_DEFLATION_REL_FLOOR · max|λ|`
/// (this INCLUDES every negative eigenvalue) is replaced by exactly `+1` while
/// its eigenvector is preserved; the strictly-positive, well-separated
/// directions are reconstructed bit-for-bit (`Σ λ_i v_i v_iᵀ`). The result is
/// SPD by construction, so the Cholesky succeeds and the evidence log-det is
/// finite.
///
/// The stiffness is UNIT (`+1`), not `max|λ|` and not `ridge·I`: a deflated
/// direction therefore contributes exactly `log 1 = 0` to `log|H|`, with ZERO
/// θ/ρ dependence — the same quotient pseudo-determinant convention the gauge
/// deflation (κ=1) and the #1117 data-null projector use. This is what makes
/// the value and the analytic outer ρ-gradient consistent: a ridge fallback
/// (`+ridge·I`) injects a ρ-dependent bias `½·log|I + ridge·H_tt⁻¹|` into the
/// VALUE that the analytic gradient (built for the undamped Laplace log-det)
/// never sees, desyncing the outer line-search; unit-stiffness deflation has no
/// such bias because the deflated direction's contribution is the ρ-independent
/// constant `0`. Returns `None` only if the block is non-finite or the
/// eigendecomposition fails (the caller then surfaces the hard refusal).
fn factor_spectral_deflated_evidence_row(
    row: &ArrowRowBlock,
    d: usize,
) -> Option<ArrowRowFactorResult> {
    if d == 0 || row.htt.dim() != (d, d) {
        return None;
    }
    // Symmetrise defensively before the eigendecomposition (the assembled
    // block is symmetric up to reduction order; the eig routine assumes exact
    // symmetry).
    let mut sym = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let v = 0.5 * (row.htt[[i, j]] + row.htt[[j, i]]);
            if !v.is_finite() {
                return None;
            }
            sym[[i, j]] = v;
        }
    }
    let (evals, evecs) = sym.eigh(Side::Lower).ok()?;
    let max_abs = evals
        .iter()
        .fold(0.0_f64, |acc, &v| if v.is_finite() { acc.max(v.abs()) } else { acc });
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let floor = SPECTRAL_DEFLATION_REL_FLOOR * max_abs;
    // Hysteresis-banded deflation floor for *positive* near-cutoff eigenvalues.
    // The bare `floor` is a knife-edge: a small positive curvature direction
    // parked at ~`floor` toggles deflated/not-deflated as ρ/θ move, flipping the
    // per-row count and tripping the quotient-dimension guard spuriously
    // (#1117). We deflate a positive eigenvalue only once it drops below the
    // LOWER band edge `floor·(1−ε)`, so a value oscillating around the bare
    // floor stays consistently KEPT. Non-positive / non-finite eigenvalues
    // (genuine null / indefinite quotient directions) are still always deflated
    // at the exact-zero boundary, which the guard must continue to honour.
    let deflate_floor = floor * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
    // Reconstruct `Σ_i λ̃_i v_i v_iᵀ`, replacing every deflated eigenvalue
    // (every non-positive/non-finite one, plus any positive one that has
    // dropped below the hysteresis floor) with unit stiffness `+1` and keeping
    // the genuine positive spectrum untouched.
    //
    // PD GUARANTEE (#1118). This function is reached ONLY after the genuine
    // undamped Cholesky has already REFUSED the block (`factor_one_row_result`
    // Err arm), so we are committed to delivering a PD factor — declining here
    // surfaces the hard "non-PD per-row H_tt" refusal that kills the whole K>1
    // fit. The previous code declined in two silent ways that violated that
    // contract on a barely-(non)-PD knife-edge: (1) `deflated_count == 0`, when
    // the symmetric `eigh` rounds the offending direction to a tiny-POSITIVE
    // eigenvalue just above `deflate_floor` while the unrolled scalar Cholesky
    // underflowed its pivot to `≤ 0`; and (2) the reconstruction's own Cholesky
    // failing because a kept eigenvalue was positive but `≪ floor`, so the
    // assembled `Σ λ v vᵀ` was numerically indefinite. Both routed a genuinely
    // non-PD block to the hard refusal even though a valid quotient factor
    // exists. We instead FLOOR every reconstructed eigenvalue to a strictly
    // positive `floor`: a direction at or below the hysteresis edge is deflated
    // to unit stiffness `+1` (the ρ-independent `log 1 = 0` quotient convention),
    // and any other near-floor positive direction is clamped UP to `floor` so
    // the assembled block is PD by construction and the Cholesky cannot fail.
    // The genuine, well-separated positive spectrum (`λ ≫ floor`) is untouched,
    // so every block the old path already conditioned is bit-for-bit unchanged.
    let mut conditioned = Array2::<f64>::zeros((d, d));
    let mut deflated_count = 0usize;
    for eig_idx in 0..evals.len() {
        let lambda = evals[eig_idx];
        let lambda_tilde = if lambda.is_finite() && lambda > deflate_floor {
            // Genuine positive direction: keep it, but clamp UP to the positive
            // `floor` so a tiny-but-kept eigenvalue cannot make the reconstructed
            // block numerically non-PD (it never lowers a healthy `λ ≫ floor`).
            lambda.max(floor)
        } else {
            // Null / indefinite / numerically-flat quotient direction: unit
            // stiffness `+1`, contributing `log 1 = 0` to the evidence log-det.
            deflated_count += 1;
            1.0
        };
        for i in 0..d {
            let vi = evecs[[i, eig_idx]];
            for j in 0..d {
                conditioned[[i, j]] += lambda_tilde * vi * evecs[[j, eig_idx]];
            }
        }
    }
    if deflated_count == 0 {
        // The hysteresis band kept every direction (the offending eigenvalue
        // rounded just above `deflate_floor`), yet the genuine Cholesky still
        // refused the block — a barely-non-PD knife-edge. We are on the refused
        // path, so we must not decline: deflate the single smallest-eigenvalue
        // direction to unit stiffness, which removes the marginal pivot while
        // leaving the rest of the (now positive-floored) spectrum exact.
        let mut min_idx = 0usize;
        let mut min_lambda = f64::INFINITY;
        for eig_idx in 0..evals.len() {
            let lambda = evals[eig_idx];
            if lambda < min_lambda {
                min_lambda = lambda;
                min_idx = eig_idx;
            }
        }
        // Subtract the kept (floored) contribution of `min_idx` and add unit
        // stiffness in its place: `conditioned += (1 − λ̃_min) v_min v_minᵀ`.
        let kept = min_lambda.max(floor);
        let delta = 1.0 - kept;
        for i in 0..d {
            let vi = evecs[[i, min_idx]];
            for j in 0..d {
                conditioned[[i, j]] += delta * vi * evecs[[j, min_idx]];
            }
        }
        deflated_count = 1;
    }
    let factor = cholesky_lower(&conditioned).ok()?;
    Some(ArrowRowFactorResult {
        factor,
        gauge_deflated_directions: deflated_count,
    })
}


/// Unit-stiffness **spectral** Faddeev-Popov conditioning of the dense REDUCED
/// SCHUR complement `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`
/// for an evidence/log-det-only solve whose plain Cholesky refused `S` because
/// it is genuinely indefinite or numerically flat (#1118, the β-block analogue
/// of [`factor_spectral_deflated_evidence_row`]).
///
/// On a rank-deficient multi-atom dictionary (the OLMo K>1 capstone: several
/// circle atoms whose 5-column decoder design is rank 3/5 on near-degenerate
/// PCA geometry) the per-row `H_tt^(i)` blocks are unit-stiffness deflated to
/// stay PD, but the Schur subtraction `Σ H_tβᵀ H_tt⁻¹ H_tβ` then accumulates in
/// finite precision and can drive a pivot of the reduced β complement NEGATIVE
/// off the inner optimum — so the evidence Cholesky hits a non-PD pivot (the
/// reported `-0.064 at index 256`) even though every row block factored.
///
/// Like the per-row version this DISCOVERS the offending directions from `S`'s
/// own symmetric eigendecomposition and replaces every eigenvalue at or below
/// the relative floor (this INCLUDES every negative eigenvalue) by exactly `+1`
/// while preserving its eigenvector, then clamps every kept eigenvalue UP to the
/// positive floor so the reconstruction is PD by construction and its Cholesky
/// cannot fail. The unit stiffness contributes a ρ-INDEPENDENT `log 1 = 0` to
/// `log|S|` — the quotient pseudo-determinant convention shared with the gauge
/// (#1037), per-row spectral (#1118), and data-null (#1117) deflations — so the
/// evidence VALUE stays consistent with the analytic ρ-gradient and the outer
/// REML line-search does not desync (a `+ridge·I` fallback would inject a
/// ρ-dependent `½·log|I + ridge·S⁻¹|` bias). This is reached ONLY after the
/// genuine Cholesky already refused `S` and ONLY for evidence/log-det callers
/// (`tolerate_ill_conditioning`), so every PD Schur complement is bit-for-bit
/// unchanged. Returns `None` only when `S` is non-finite or the
/// eigendecomposition fails (the caller then surfaces the hard refusal).
fn factor_spectral_deflated_evidence_dense(schur: &Array2<f64>) -> Option<Array2<f64>> {
    let d = schur.nrows();
    if d == 0 || schur.ncols() != d {
        return None;
    }
    let mut sym = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let v = 0.5 * (schur[[i, j]] + schur[[j, i]]);
            if !v.is_finite() {
                return None;
            }
            sym[[i, j]] = v;
        }
    }
    let (evals, evecs) = sym.eigh(Side::Lower).ok()?;
    let max_abs = evals.iter().fold(0.0_f64, |acc, &v| {
        if v.is_finite() {
            acc.max(v.abs())
        } else {
            acc
        }
    });
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let floor = SPECTRAL_DEFLATION_REL_FLOOR * max_abs;
    let deflate_floor = floor * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
    // Reconstruct `Σ_i λ̃_i v_i v_iᵀ`: deflate non-positive / sub-floor
    // directions to unit stiffness `+1`, clamp every kept direction UP to the
    // positive `floor` so the assembled complement is PD by construction.
    let mut conditioned = Array2::<f64>::zeros((d, d));
    for eig_idx in 0..evals.len() {
        let lambda = evals[eig_idx];
        let lambda_tilde = if lambda.is_finite() && lambda > deflate_floor {
            lambda.max(floor)
        } else {
            1.0
        };
        for i in 0..d {
            let vi = evecs[[i, eig_idx]];
            for j in 0..d {
                conditioned[[i, j]] += lambda_tilde * vi * evecs[[j, eig_idx]];
            }
        }
    }
    cholesky_lower(&conditioned).ok()
}


fn cholesky_solve_vector_fixed<const D: usize>(
    l: ArrayView2<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> Array1<f64> {
    // Precondition: `l` is a Cholesky factor whose diagonals are strictly
    // positive and finite (every f64 factor in this module is produced by
    // `cholesky_lower`, which rejects `!is_finite() || sum <= 0.0` pivots). The
    // back/forward substitution below divides by `l[[i, i]]` with no per-row
    // guard; a future caller that hands an unvalidated factor here would emit a
    // silent `NaN` into the Schur reduction (#1038). Catch that loudly —
    // always, release included — rather than letting it flow into the
    // evidence/gradient. The check is O(D) over a small fixed-size factor, so
    // it is negligible next to the substitution it guards.
    assert!(
        (0..D).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f64::MIN_POSITIVE),
        "cholesky_solve_vector_fixed: factor diagonal must be finite and non-subnormal"
    );
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
    factor_one_row_result(row, ridge_t, d, row_idx, tolerate_ill_conditioning, &[])
        .map(|result| result.factor)
}


fn factor_one_row_result(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    row_idx: usize,
    tolerate_ill_conditioning: bool,
    row_gauges: &[Array1<f64>],
) -> Result<ArrowRowFactorResult, ArrowSchurError> {
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
                    break ArrowRowFactorResult {
                        factor,
                        gauge_deflated_directions: 0,
                    };
                }
                // Diagonal-ratio condition-number proxy κ(LLᵀ) ≈
                // (max L_ii / min L_ii)², vs the dimension-scaled Higham
                // near-singularity ceiling. A barely-PD inverse plugged into
                //   S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
                // contaminates S by spectral terms scaled by κ_i, so an
                // over-threshold block is regularised further rather than used.
                let kappa_est = cholesky_factor_kappa_estimate(&factor);
                if cholesky_factor_passes_safe_inversion(&factor, d, diag_scale, kappa_est) {
                    break ArrowRowFactorResult {
                        factor,
                        gauge_deflated_directions: 0,
                    };
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
                    if ridge_t == 0.0 {
                        if let Some(deflated) =
                            factor_gauge_deflated_evidence_row(row, d, row_gauges)
                        {
                            // Faddeev-Popov row-gauge deflation: only the
                            // closed-form orbit direction is stiffened, at UNIT
                            // stiffness kappa = 1.0, so each deflated direction
                            // contributes log(1) = 0 to log|H| — the quotient
                            // pseudo-determinant convention (the gauge orbit is a
                            // criterion null direction, contributing nothing to
                            // the Laplace normalizer). Zero theta/rho dependence,
                            // so criterion derivatives stay exact on the quotient.
                            return Ok(deflated);
                        }
                        // #1117/#1118 — the offending direction is NOT a supplied
                        // gauge vector: under K>1 IBP/softmax row-sharing the
                        // logit×coordinate Gauss-Newton cross term drives an
                        // eigenvalue of this row's H_tt negative (or numerically
                        // flat) at a direction the closed-form gauge orbit does
                        // not span. DISCOVER it from the block's own symmetric
                        // eigendecomposition and deflate it at the SAME unit
                        // stiffness (eigenvalue → +1), so its evidence
                        // contribution is the ρ-independent constant log 1 = 0.
                        // This replaces the previous ridge-damped evidence
                        // fallback, whose ½·log|I + ridge·H_tt⁻¹| bias was
                        // ρ-DEPENDENT and therefore desynced the outer REML value
                        // (which saw it) from the analytic ρ-gradient (built for
                        // the undamped Laplace log-det, which did not) — the
                        // multi-atom outer line-search non-convergence (#1117).
                        // The undamped exact Cholesky still owns every genuinely
                        // PD block (this arm is reached only on a refused factor),
                        // so K=1 and any PD K>1 row are bit-for-bit unchanged.
                        if let Some(deflated) = factor_spectral_deflated_evidence_row(row, d) {
                            return Ok(deflated);
                        }
                    }
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
    /// Device-uploadable SAE Kronecker data for CUDA-resident reduced PCG.
    ///
    /// The generic matrix-free closures remain the authoritative CPU path. This
    /// descriptor is installed only when SAE assembly has a matching CUDA sparse
    /// representation for both `H_tβ` and `H_ββ`.
    pub device_sae_pcg: Option<Arc<DeviceSaePcgData>>,
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
    /// Optional row-local gauge directions for evidence-only Faddeev-Popov
    /// deflation of an otherwise non-PD `H_tt` row block.
    ///
    /// These vectors live in each row's actual chart block, so compact SAE rows
    /// and dense rows share the same factorization path. Ordinary Newton solves
    /// ignore them; only undamped evidence factors with
    /// `tolerate_ill_conditioning` set may stiffen a gauge-explained row
    /// direction.
    pub row_gauge_deflation: Option<ArrowRowGaugeDeflation>,
    /// Optional exact cross-row IBP low-rank source (#1038). When set, the
    /// factorization downdates the per-row logit-slot self term and layers the
    /// exact rank-`R` Woodbury correction onto the evidence cache (value,
    /// log-determinant, and θ/ρ-adjoint together). `None` for all non-IBP
    /// systems — the row-block-diagonal arrow path is then unchanged.
    pub ibp_cross_row: Option<IbpCrossRowSource>,
}


impl Clone for ArrowSchurSystem {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows.clone(),
            hbb: self.hbb.clone(),
            hbb_matvec: self.hbb_matvec.clone(),
            htbeta_matvec: self.htbeta_matvec.clone(),
            htbeta_transpose_matvec: self.htbeta_transpose_matvec.clone(),
            htbeta_dense_supplement: self.htbeta_dense_supplement,
            hbb_diag: self.hbb_diag.clone(),
            gb: self.gb.clone(),
            d: self.d,
            row_dims: Arc::clone(&self.row_dims),
            row_offsets: Arc::clone(&self.row_offsets),
            k: self.k,
            manifold_mode_fingerprint: self.manifold_mode_fingerprint,
            row_hessian_fingerprint: self.row_hessian_fingerprint,
            analytic_row_hessian_fingerprint: self.analytic_row_hessian_fingerprint,
            block_offsets: Arc::clone(&self.block_offsets),
            penalty_op: self.penalty_op.clone(),
            device_sae_pcg: self.device_sae_pcg.clone(),
            cross_row_penalties: self.cross_row_penalties.clone(),
            row_gauge_deflation: self.row_gauge_deflation.clone(),
            ibp_cross_row: self.ibp_cross_row.clone(),
        }
    }
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


/// Exact cross-row low-rank IBP source (#1038): the per-column rank-one Hessian
/// terms `H_(i,k),(j,k) = d_k·z'_ik·z'_jk` (for ALL `i,j`, including the `i=j`
/// self term) that couple DISTINCT latent rows through a shared atom column `k`.
///
/// Stacking over rows, this is `H_full = H₀' + U D Uᵀ`, where:
/// * `U` is `delta_t_len × R` with `U[g, k] = z'_ik` at the global latent index
///   `g` of row `i`'s logit slot for atom `k` (zero elsewhere) — i.e. column `k`
///   is supported on the atom-`k` logit slot of every row;
/// * `D = diag(d_k)`, `d_k = w·s'_k` ([`crate::terms::analytic_penalties::IbpHessianDiagThirdChannels::cross_row_d`]);
/// * `H₀'` is the assembled latent block-diagonal `H₀` with the per-row self
///   term `d_k·z'_ik²` REMOVED from each logit-slot diagonal (the assembled
///   `H₀` already carries it, so the FULL rank-one outer product `U D Uᵀ` —
///   which re-adds the `i=j` diagonal — would double-count without this
///   downdate). The determinant lemma `log det(I_R + D UᵀH₀'⁻¹U)` is only the
///   exact rank-`R` correction against this no-self base.
///
/// The arrow elimination assumes each row's `H_tt^(i)` is independent of every
/// other row, so it structurally cannot hold this coupling block-locally. The
/// factorization owner (`solver::arrow_schur`) consumes this source to (a)
/// downdate the per-row logit diagonal before factoring, (b) build `U`/`D` onto
/// the resulting [`ArrowFactorCache`] as a [`CrossRowWoodbury`], and (c) apply
/// the exact Woodbury correction to the value/curvature solve, the evidence
/// log-determinant, and the θ/ρ-adjoint TOGETHER (they all describe the SAME
/// `H_full`).
#[derive(Clone, Debug, Default)]
pub struct IbpCrossRowSource {
    /// Number of atom columns `R` (the rank of the cross-row update).
    pub r: usize,
    /// `d_k = w·s'_k`, the scalar `D`-coefficient of column `k`. Length `R`.
    pub d: Array1<f64>,
    /// Per-row column entries `(global_t_index, atom_k, z'_ik)`: each tuple
    /// places `z'_ik` at `U[global_t_index, atom_k]`. The `global_t_index` is
    /// `row_offsets[i] + local_slot` for the row's logit slot of atom `k`. Only
    /// nonzero entries are listed (one per active (row, atom) pair).
    pub entries: Vec<(usize, usize, f64)>,
}


impl IbpCrossRowSource {
    /// Build the dense `delta_t_len × R` factor `U` (each column supported on
    /// its atom's per-row logit slots) from the sparse entry list.
    fn dense_u(&self, delta_t_len: usize) -> Array2<f64> {
        let mut u = Array2::<f64>::zeros((delta_t_len, self.r));
        for &(g, k, z) in &self.entries {
            u[[g, k]] += z;
        }
        u
    }

    /// Per-row-slot self-term downdate: returns, for each global latent index,
    /// the scalar `Σ_k d_k·z'_ik²` to subtract from that logit slot's diagonal
    /// so the factored base is `H₀'` (self term removed). Indexed by global
    /// `delta_t` position.
    fn self_term_downdate(&self, delta_t_len: usize) -> Array1<f64> {
        let mut down = Array1::<f64>::zeros(delta_t_len);
        for &(g, k, z) in &self.entries {
            down[g] += self.d[k] * z * z;
        }
        down
    }
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
        Self {
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
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            ibp_cross_row: None,
        }
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
        Self {
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
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            ibp_cross_row: None,
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
        Self {
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
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            ibp_cross_row: None,
        }
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
        Self {
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
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            ibp_cross_row: None,
        }
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
        Self {
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
            device_sae_pcg: None,
            cross_row_penalties: Vec::new(),
            row_gauge_deflation: None,
            ibp_cross_row: None,
        }
    }

    pub fn set_row_gauge_deflation(&mut self, deflation: ArrowRowGaugeDeflation) {
        self.row_gauge_deflation = Some(deflation);
    }

    /// Register the exact cross-row IBP low-rank source (#1038). The assembly
    /// passes the per-column `D`-coefficients (`cross_row_d`) and the `(global
    /// latent index, atom, z'_ik)` entries built from `z_jac`; the factorization
    /// then carries the exact rank-`R` Woodbury (value + log-determinant +
    /// θ/ρ-adjoint) on the evidence cache. An empty source (`r == 0` or no
    /// entries) is treated as absent so the row-block-diagonal path is unchanged.
    pub fn set_ibp_cross_row_source(&mut self, source: IbpCrossRowSource) {
        if source.r == 0 || source.entries.is_empty() {
            self.ibp_cross_row = None;
        } else {
            self.ibp_cross_row = Some(source);
        }
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
    ///
    /// This is intentionally explicit and expensive. Cache and evidence callers
    /// use [`Self::current_row_hessian_fingerprint`] at the point they need the
    /// value, after assembly has populated the system, instead of hashing each
    /// intermediate construction/mutation step.
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
    }

    /// Mark the dense per-row cross-block slabs as active supplements to the
    /// installed matrix-free row operator.
    pub fn activate_dense_htbeta_supplement(&mut self) {
        self.htbeta_dense_supplement = true;
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
    }

    pub fn set_device_sae_pcg_data(&mut self, data: DeviceSaePcgData) {
        assert_eq!(data.beta_dim, self.k);
        assert_eq!(data.a_phi.len(), self.rows.len());
        assert_eq!(data.local_jac.len(), self.rows.len());
        self.device_sae_pcg = Some(Arc::new(data));
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

    /// Reduced-Schur matvec prologue `y = (P + ridge·I) x` written fresh into a
    /// zeroed `y` (the caller clears `out` first; this is the first writer).
    ///
    /// At the SAE LLM border width (#1017) the dense `H_ββ` fallback is a `k×k`
    /// GEMV whose `O(k²)` cost (≈4M flops at k=2048) runs once per CG iteration
    /// and was the serial Amdahl ceiling on the per-row-parallel matvec: while
    /// the `n`-row point-elimination term fans across all cores, this prologue
    /// pinned one core and grows as `k²`. The dense GEMV is embarrassingly
    /// parallel over output rows `a` — each `y[a] = Σ_b hbb[a,b]·x[b] + ridge·x[a]`
    /// is independent and its inner sum order is identical whether one thread or
    /// many compute it, so the result is bit-identical run-to-run (the #1017
    /// determinism gate: the criterion ranking across topology candidates must
    /// not move). The `penalty_op` path stays serial — it is an opaque operator
    /// with its own structure (SAE uses the dense `hbb`), and small `k` stays
    /// serial to avoid rayon overhead on a trivial GEMV.
    ///
    /// `parallel` is the caller's top-level / not-nested-in-rayon decision (the
    /// same guard the row loop uses), so this never oversubscribes inside the
    /// topology race.
    fn penalty_ridge_prologue_into(&self, x: &[f64], ridge: f64, y: &mut [f64], parallel: bool) {
        let k = self.hbb.nrows();
        let dense_parallel = parallel
            && self.penalty_op.is_none()
            && self.hbb.dim() == (k, k)
            && k >= SCHUR_PROLOGUE_PARALLEL_K_MIN;
        if dense_parallel {
            use rayon::prelude::*;
            let hbb = &self.hbb;
            y.par_iter_mut().enumerate().for_each(|(a, ya)| {
                let mut acc = 0.0_f64;
                for b in 0..k {
                    acc += hbb[[a, b]] * x[b];
                }
                *ya = acc + ridge * x[a];
            });
        } else {
            self.penalty_matvec_add(x, y);
            for a in 0..k {
                y[a] += ridge * x[a];
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
    /// Set when the source system carried an exact cross-row IBP source
    /// ([`IbpCrossRowSource`], #1038). The streaming chunked accumulator cannot
    /// hold the rank-`R` Woodbury correction chunk-locally — `U`'s columns span
    /// ALL rows, so the capacitance `I_R + D Uᵀ H₀'⁻¹ U` needs the per-row
    /// factors retained for a global `H₀'⁻¹U` back-solve, which is exactly the
    /// `(N·K)`-scale residency the streaming path exists to avoid. Rather than
    /// silently DROP the cross-row term (an inexact logdet that would desync
    /// from the dense-resident gradient), the streaming log-determinant errors
    /// loudly when this is set, forcing IBP-active fits onto the dense resident
    /// [`ArrowFactorCache::arrow_log_det`] path (which carries the exact
    /// Woodbury). See the #1038 streaming note.
    ibp_cross_row_active: bool,
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
            ibp_cross_row_active: false,
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
        streaming.ibp_cross_row_active = sys.ibp_cross_row.is_some();
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
        if self.ibp_cross_row_active {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "streaming arrow log-det cannot carry the exact cross-row IBP \
                         Woodbury correction (#1038): U's columns span all rows, so the \
                         rank-R capacitance needs the per-row factors retained — the very \
                         (N·K) residency the streaming path avoids. Route IBP-active fits \
                         through the dense resident ArrowFactorCache::arrow_log_det instead."
                    .to_string(),
            });
        }
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
        if self.ibp_cross_row_active {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "streaming arrow solve cannot carry the exact cross-row IBP \
                         Woodbury correction (#1038); route IBP-active fits through the \
                         dense resident solve_arrow_newton_step_with_options instead."
                    .to_string(),
            });
        }
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
    /// Number of row-local gauge directions stiffened in an undamped evidence
    /// factorization.
    ///
    /// Each direction is stiffened at UNIT stiffness `kappa = 1.0`, so it
    /// contributes `log(1) = 0` to the row-block logdet through the returned
    /// Cholesky factor: the gauge orbit is a criterion null direction and adds
    /// nothing to the Laplace normalizer (the quotient pseudo-determinant
    /// convention, cf. `PenaltyPseudologdet`). Zero theta/rho dependence.
    pub gauge_deflated_directions: usize,
    /// Exact cross-row IBP rank-`R` Woodbury correction (#1038), present iff the
    /// source system carried an [`IbpCrossRowSource`]. When set, the per-row
    /// factors above are of the NO-SELF base `H₀'` (self term `d_k·z'_ik²`
    /// downdated from each logit diagonal), and this carrier supplies the exact
    /// rank-`R` correction so the value/curvature solve
    /// ([`Self::full_inverse_apply`]), the evidence log-determinant
    /// ([`Self::arrow_log_det`]), and the θ/ρ-adjoint all describe the same
    /// `H_full = H₀' + U D Uᵀ`.
    pub cross_row_woodbury: Option<CrossRowWoodbury>,
}


/// Materialized exact cross-row IBP Woodbury correction (#1038), built against
/// an [`ArrowFactorCache`] whose per-row factors are the NO-SELF base `H₀'`.
///
/// Holds `U` (the `delta_t_len × R` arrow-`t` factor, β-part implicitly zero),
/// `D = diag(d_k)`, the projected `M = UᵀH₀'⁻¹U`, the columns `H₀'⁻¹U`, and the
/// **LU factorization of the (generally non-symmetric, possibly indefinite)
/// capacitance** `C = I_R + D·M`. `d_k = w·s'_k` is not sign-definite, so the
/// capacitance is factored by a partial-pivot LU (exact for any sign); the same
/// factorization serves the log-determinant `log det C`, the inverse correction
/// `H_full⁻¹w = H₀'⁻¹w − H₀'⁻¹U·C⁻¹·(D Uᵀ H₀'⁻¹w)`, and the adjoint's
/// selected-inverse (`C⁻¹` and `M`). The full inverse, value/curvature solve,
/// log-determinant, and adjoint therefore all describe the SAME
/// `H_full = H₀' + U D Uᵀ`.
#[derive(Debug, Clone)]
pub struct CrossRowWoodbury {
    /// `U`: `delta_t_len × R`, column `k` supported on atom-`k` logit slots.
    pub u: Array2<f64>,
    /// `d_k`, length `R`.
    pub d: Array1<f64>,
    /// `H₀'⁻¹ U` (the `t`-block), `delta_t_len × R`.
    pub h0inv_u: Array2<f64>,
    /// `(H₀'⁻¹ U)` β-block, `K × R`. `U` has no β support, but the bordered
    /// solve couples the latent columns to `β` through the Schur complement, so
    /// this block is generally nonzero and the inverse correction must apply it
    /// to the `β` output too.
    pub h0inv_u_beta: Array2<f64>,
    /// `M = Uᵀ H₀'⁻¹ U`, `R × R` (symmetric). Retained for the θ/ρ-adjoint.
    pub m: Array2<f64>,
    /// Partial-pivot LU of the capacitance `C = I_R + D·M` (`lu` packs `L`/`U`,
    /// `piv` the row swaps), built by [`small_lu_factor`].
    pub capacitance_lu: SmallLu,
    /// The sparse `U` entries `(global_t_index, atom_k, z'_ik)` — retained so
    /// `Uᵀ·v` can be formed over the atom slots without re-deriving them.
    pub entries: Vec<(usize, usize, f64)>,
}


/// Dense partial-pivot LU of a small square matrix. Used for the cross-row IBP
/// capacitance `C = I_R + D·M`, which is generally non-symmetric and possibly
/// indefinite (`d_k = w·s'_k` is not sign-definite), so a Cholesky/LDLᵀ is
/// unavailable. `R` is the atom count, so this is a cheap dense factorization.
#[derive(Debug, Clone)]
pub struct SmallLu {
    /// Packed `L` (unit lower, below diagonal) and `U` (upper, on/above
    /// diagonal), `R × R`, in the row-permuted order encoded by `piv`.
    lu: Array2<f64>,
    /// Row permutation: `piv[i]` is the original row now in position `i`.
    piv: Vec<usize>,
    /// Sign of the permutation (`±1`), folded into the determinant.
    perm_sign: f64,
}


/// Partial-pivot LU factorization of a small dense square matrix `a` (`R × R`).
/// Returns `None` only when a pivot is exactly zero (singular `C`).
fn small_lu_factor(a: &Array2<f64>) -> Option<SmallLu> {
    let r = a.nrows();
    assert_eq!(a.ncols(), r, "small_lu_factor: non-square input");
    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..r).collect();
    let mut perm_sign = 1.0_f64;
    for col in 0..r {
        // Partial pivot: pick the largest-magnitude entry on/below the diagonal.
        let mut pivot_row = col;
        let mut pivot_mag = lu[[col, col]].abs();
        for row in (col + 1)..r {
            let mag = lu[[row, col]].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = row;
            }
        }
        // Reject not just an exactly-zero pivot, but any non-finite or
        // subnormal magnitude: dividing by a subnormal in the elimination /
        // back-solve produces `Inf`/`NaN` that would otherwise flow silently
        // into the Woodbury inverse and the evidence log-det (#1038). A
        // capacitance this degenerate is a desync the caller must surface
        // (→ `Ok(None)` cross-row-absent / `SchurFactorFailed`), not consume.
        if !pivot_mag.is_finite() || pivot_mag < f64::MIN_POSITIVE {
            return None;
        }
        if pivot_row != col {
            for c in 0..r {
                lu.swap((col, c), (pivot_row, c));
            }
            piv.swap(col, pivot_row);
            perm_sign = -perm_sign;
        }
        let pivot = lu[[col, col]];
        for row in (col + 1)..r {
            let factor = lu[[row, col]] / pivot;
            lu[[row, col]] = factor;
            for c in (col + 1)..r {
                let v = lu[[col, c]];
                lu[[row, c]] -= factor * v;
            }
        }
    }
    // Post-elimination invariant: every U diagonal is finite and not subnormal.
    // The per-column pivot guard above validates each diagonal as it is chosen,
    // but assert it explicitly so `SmallLu::solve` can divide by `lu[[i, i]]`
    // without a per-entry guard and so a `SmallLu` value can never carry a
    // factor that would silently emit `Inf`/`NaN` into the capacitance solve.
    for i in 0..r {
        let u = lu[[i, i]];
        if !u.is_finite() || u.abs() < f64::MIN_POSITIVE {
            return None;
        }
    }
    Some(SmallLu { lu, piv, perm_sign })
}


impl SmallLu {
    fn dim(&self) -> usize {
        self.lu.nrows()
    }

    /// `log|det|` and the determinant sign (`±1`).
    fn log_abs_det_and_sign(&self) -> (f64, f64) {
        let mut log_abs = 0.0_f64;
        let mut sign = self.perm_sign;
        for i in 0..self.dim() {
            let u = self.lu[[i, i]];
            log_abs += u.abs().ln();
            if u < 0.0 {
                sign = -sign;
            }
        }
        (log_abs, sign)
    }

    /// Solve `C x = b` reusing the factorization (in place into a fresh vector).
    ///
    /// Returns `None` when the solve cannot produce a finite result — either a
    /// `U` diagonal is non-finite/subnormal (defensive: `small_lu_factor`
    /// already rejects such factors, but a future construction path might not)
    /// or the back-substitution overflows to `Inf`/`NaN` for an extreme RHS on
    /// an ill-conditioned (yet validly factored) capacitance. Surfacing `None`
    /// lets the Woodbury / evidence consumers fail loudly (#1038) instead of
    /// flowing a silent `NaN` into the log-det and outer gradient.
    fn solve(&self, b: &Array1<f64>) -> Option<Array1<f64>> {
        let r = self.dim();
        // Apply the row permutation: y = P b.
        let mut y = Array1::<f64>::zeros(r);
        for i in 0..r {
            y[i] = b[self.piv[i]];
        }
        // Forward solve L y' = P b (L unit-lower).
        for i in 0..r {
            let mut sum = y[i];
            for j in 0..i {
                sum -= self.lu[[i, j]] * y[j];
            }
            y[i] = sum;
        }
        // Back solve U x = y' (U upper, explicit diagonal).
        let mut x = Array1::<f64>::zeros(r);
        for i in (0..r).rev() {
            let mut sum = y[i];
            for j in (i + 1)..r {
                sum -= self.lu[[i, j]] * x[j];
            }
            let pivot = self.lu[[i, i]];
            if !pivot.is_finite() || pivot.abs() < f64::MIN_POSITIVE {
                return None;
            }
            x[i] = sum / pivot;
        }
        if x.iter().all(|v| v.is_finite()) {
            Some(x)
        } else {
            None
        }
    }
}


impl CrossRowWoodbury {
    /// Build the exact rank-`R` cross-row Woodbury carrier from the IBP source
    /// and a cache whose per-row factors are the NO-SELF base `H₀'`.
    ///
    /// Computes `H₀'⁻¹U` (one [`ArrowFactorCache::full_inverse_apply`] back-solve
    /// per column, β-RHS zero — the `t`-block of the result is `H₀'⁻¹U`'s
    /// column), `M = UᵀH₀'⁻¹U`, and the LU of `C = I_R + D·M`. Returns `None`
    /// when the capacitance is exactly singular (the only un-representable case;
    /// the caller then proceeds with the bare `H₀'` cache and the cross-row term
    /// is absent — never silently inconsistent, since logdet/inverse/adjoint all
    /// key off the presence of this carrier).
    fn build(
        cache: &ArrowFactorCache,
        source: &IbpCrossRowSource,
    ) -> Result<Option<Self>, ArrowSchurError> {
        let r = source.r;
        let total_len = cache.delta_t_len();
        let u = source.dense_u(total_len);
        let d = source.d.clone();
        let zero_beta = Array1::<f64>::zeros(cache.k);
        // h0inv_u[:, k] = (H₀'⁻¹ U)_t for column k; h0inv_u_beta[:, k] its β-block.
        let mut h0inv_u = Array2::<f64>::zeros((total_len, r));
        let mut h0inv_u_beta = Array2::<f64>::zeros((cache.k, r));
        for k in 0..r {
            let col = u.column(k).to_owned();
            let (sol_t, sol_beta) = cache.full_inverse_apply(col.view(), zero_beta.view())?;
            for g in 0..total_len {
                h0inv_u[[g, k]] = sol_t[g];
            }
            for c in 0..cache.k {
                h0inv_u_beta[[c, k]] = sol_beta[c];
            }
        }
        // M = Uᵀ (H₀'⁻¹ U), symmetric R×R. U is sparse (atom-slot supported), so
        // contract over the listed entries.
        let mut m = Array2::<f64>::zeros((r, r));
        for a in 0..r {
            for b in 0..r {
                let mut acc = 0.0_f64;
                for &(g, k, z) in &source.entries {
                    if k == a {
                        acc += z * h0inv_u[[g, b]];
                    }
                }
                m[[a, b]] = acc;
            }
        }
        // Symmetrize M to clear back-substitution rounding asymmetry.
        for a in 0..r {
            for b in (a + 1)..r {
                let avg = 0.5 * (m[[a, b]] + m[[b, a]]);
                m[[a, b]] = avg;
                m[[b, a]] = avg;
            }
        }
        // Capacitance C = I_R + D·M (row k scaled by d_k).
        let mut c = Array2::<f64>::zeros((r, r));
        for a in 0..r {
            for b in 0..r {
                c[[a, b]] = d[a] * m[[a, b]];
            }
            c[[a, a]] += 1.0;
        }
        let Some(capacitance_lu) = small_lu_factor(&c) else {
            return Ok(None);
        };
        Ok(Some(Self {
            u,
            d,
            h0inv_u,
            h0inv_u_beta,
            m,
            capacitance_lu,
            entries: source.entries.clone(),
        }))
    }

    /// The sparse `U` entry list `(global_t_index, atom_k, z'_ik)`.
    fn source_entries(&self) -> &[(usize, usize, f64)] {
        &self.entries
    }

    /// `C⁻¹ D` as a dense `R × R` matrix (`R` capacitance solves; column `l` is
    /// `d_l · C⁻¹ e_l`). Shared by the inverse-diagonal correction and any
    /// adjoint trace that needs the selected inverse of the capacitance.
    ///
    /// Returns `None` when any capacitance solve fails to produce a finite
    /// result (#1038); the consumer must surface this as a loud failure rather
    /// than propagate a `NaN` into the evidence/gradient.
    pub fn capacitance_inv_times_d(&self) -> Option<Array2<f64>> {
        let r = self.d.len();
        let mut out = Array2::<f64>::zeros((r, r));
        let mut e_l = Array1::<f64>::zeros(r);
        for l in 0..r {
            e_l.fill(0.0);
            e_l[l] = 1.0;
            let col = self.capacitance_lu.solve(&e_l)?;
            for k in 0..r {
                out[[k, l]] = col[k] * self.d[l];
            }
        }
        Some(out)
    }

    /// Subtract the rank-`R` Woodbury term from the latent inverse diagonal:
    /// `diag ← diag − diag(H₀'⁻¹U C⁻¹ D Uᵀ H₀'⁻¹)`. With `G = h0inv_u` and
    /// `(C⁻¹D) = capacitance_inv_times_d()`, the entry at global index `g` is
    /// `Σ_{k,l} G[g,k] (C⁻¹D)[k,l] G[g,l]`.
    fn subtract_inverse_diagonal(
        &self,
        diag: &mut Array1<f64>,
    ) -> Result<(), ArrowSchurError> {
        let r = self.d.len();
        let cinv_d = self.capacitance_inv_times_d().ok_or_else(|| {
            ArrowSchurError::SchurFactorFailed {
                reason: "cross-row Woodbury capacitance solve produced a non-finite \
                         C⁻¹D for the inverse-diagonal correction (#1038): \
                         singular/ill-conditioned cross-row capacitance"
                    .to_string(),
            }
        })?;
        let total_len = self.h0inv_u.nrows();
        for g in 0..total_len {
            let mut acc = 0.0_f64;
            for k in 0..r {
                let gk = self.h0inv_u[[g, k]];
                if gk == 0.0 {
                    continue;
                }
                for l in 0..r {
                    acc += gk * cinv_d[[k, l]] * self.h0inv_u[[g, l]];
                }
            }
            diag[g] -= acc;
        }
        Ok(())
    }

    /// `log det(I_R + D·M)` (the matrix-determinant-lemma correction). Returns
    /// `None` when the capacitance LU has a negative determinant — i.e. the
    /// implied `H_full` is non-PD, which is a desync the evidence must reject
    /// loudly rather than return a complex/`NaN` log-det.
    pub fn log_det(&self) -> Option<f64> {
        let (log_abs, sign) = self.log_det_correction();
        if sign > 0.0 { Some(log_abs) } else { None }
    }

    /// `log det(I_R + D·M)`: the exact additive correction
    /// `log det H_full − log det H₀'` (matrix-determinant lemma). For a genuine
    /// PD `H_full` this is real; the LU sign is returned for the caller to
    /// surface a non-PD capacitance as an error rather than a silent `NaN`.
    fn log_det_correction(&self) -> (f64, f64) {
        self.capacitance_lu.log_abs_det_and_sign()
    }

    /// Apply the rank-`R` inverse correction in place on BOTH arrow blocks:
    /// `u ← u − (H₀'⁻¹U) · C⁻¹ · (D Uᵀ (H₀'⁻¹ rhs)_t)`, where `h0inv_rhs_t` is
    /// the `t`-block of `H₀'⁻¹ rhs` already computed by the base
    /// [`ArrowFactorCache::full_inverse_apply`]. Implements the Woodbury
    /// identity `H_full⁻¹ = H₀'⁻¹ − H₀'⁻¹U C⁻¹ D Uᵀ H₀'⁻¹`. `U` has no `β`
    /// support so `Uᵀ·v` reads only the `t`-block, but `H₀'⁻¹U` couples to `β`
    /// through the Schur complement, so the correction touches `u_beta` too.
    ///
    /// `entries` lets `Uᵀ·v` be formed over the sparse atom slots.
    fn apply_inverse_correction(
        &self,
        h0inv_rhs_t: ArrayView1<'_, f64>,
        entries: &[(usize, usize, f64)],
        u_t: &mut Array1<f64>,
        u_beta: &mut Array1<f64>,
    ) -> Result<(), ArrowSchurError> {
        let r = self.d.len();
        // p = D Uᵀ (H₀'⁻¹ rhs)_t.
        let mut p = Array1::<f64>::zeros(r);
        for &(g, k, z) in entries {
            p[k] += z * h0inv_rhs_t[g];
        }
        for k in 0..r {
            p[k] *= self.d[k];
        }
        // q = C⁻¹ p. A non-finite solve is a singular/ill-conditioned cross-row
        // capacitance (#1038): fail loudly rather than write `NaN` into the
        // Newton step / adjoint solve.
        let q = self.capacitance_lu.solve(&p).ok_or_else(|| {
            ArrowSchurError::SchurFactorFailed {
                reason: "cross-row Woodbury capacitance solve produced a non-finite \
                         C⁻¹p for the inverse correction (#1038): \
                         singular/ill-conditioned cross-row capacitance"
                    .to_string(),
            }
        })?;
        // u_t -= (H₀'⁻¹U)_t · q.
        for g in 0..u_t.len() {
            let mut acc = 0.0_f64;
            for k in 0..r {
                acc += self.h0inv_u[[g, k]] * q[k];
            }
            u_t[g] -= acc;
        }
        // u_beta -= (H₀'⁻¹U)_β · q.
        for c in 0..u_beta.len() {
            let mut acc = 0.0_f64;
            for k in 0..r {
                acc += self.h0inv_u_beta[[c, k]] * q[k];
            }
            u_beta[c] -= acc;
        }
        Ok(())
    }
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


/// Largest cached Cholesky pivot across the row blocks and the dense Schur
/// factor (Hessian scale, i.e. squared lower-factor diagonal). This is the
/// diagonal magnitude scale a safe-SPD pivot floor is measured against: the
/// curvature-homotopy tracker (#1007) compares the min pivot against
/// `√eps · max(this, 1)`, the same floor the inner solver's
/// [`safe_spd_pivot_min`] uses. `None` only for an empty cache.
pub fn arrow_factor_max_pivot(cache: &ArrowFactorCache) -> Option<f64> {
    let mut max_pivot: Option<f64> = None;
    for factor in cache.htt_factors.iter() {
        if let Some(pivot) = lower_cholesky_max_pivot(factor) {
            max_pivot = Some(match max_pivot {
                Some(current) => f64::max(current, pivot),
                None => pivot,
            });
        }
    }
    if let Some(factor) = cache.schur_factor.as_ref()
        && let Some(pivot) = lower_cholesky_max_pivot(factor.view())
    {
        max_pivot = Some(match max_pivot {
            Some(current) => f64::max(current, pivot),
            None => pivot,
        });
    }
    max_pivot
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
            2.0 * s + self.cross_row_woodbury_log_det()
        });
        (log_det_tt, log_det_schur)
    }

    /// The exact cross-row IBP correction `log det(I_R + D·M)` to add to the
    /// base `log det H₀'` (#1038). Zero when no [`CrossRowWoodbury`] is present,
    /// so non-IBP caches are unaffected. The determinant lemma gives
    /// `log det H_full = log det H₀' + log det(I_R + D Uᵀ H₀'⁻¹ U)`; this is the
    /// second term, the only piece beyond the bare arrow log-determinant.
    ///
    /// Panics-free: a negative capacitance determinant (non-PD `H_full`) yields
    /// `NaN` here so the evidence surfaces the desync rather than silently
    /// dropping the imaginary part. Callers that must reject it should check
    /// [`CrossRowWoodbury::log_det`] directly.
    pub fn cross_row_woodbury_log_det(&self) -> f64 {
        match self.cross_row_woodbury.as_ref() {
            Some(w) => w.log_det().unwrap_or(f64::NAN),
            None => 0.0,
        }
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
        if let Some(woodbury) = self.cross_row_woodbury.as_ref() {
            // #1038: the factors above are `H₀'`, so `out` is diag((H₀'⁻¹)_tt).
            // The full inverse diagonal subtracts the rank-`R` Woodbury term
            // diag(H₀'⁻¹U C⁻¹ D Uᵀ H₀'⁻¹). With `G = h0inv_u = (H₀'⁻¹U)_t` and
            // (by symmetry of `H₀'⁻¹`) `(Uᵀ H₀'⁻¹)_t = Gᵀ`, the diagonal entry at
            // global index `g` is `Σ_{k,l} G[g,k] (C⁻¹D)[k,l] G[g,l]`. Form the
            // `R×R` matrix `C⁻¹D` once (R solves), then contract per row index.
            woodbury.subtract_inverse_diagonal(&mut out)?;
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
    ///
    /// When the cache carries an exact cross-row IBP
    /// [`CrossRowWoodbury`] (#1038), the per-row factors are the NO-SELF base
    /// `H₀'` and this method layers the rank-`R` Woodbury correction so the
    /// returned solve is against the FULL `H_full = H₀' + U D Uᵀ` — the same
    /// operator whose log-determinant [`Self::arrow_log_det`] reports. The
    /// θ/ρ-adjoint that consumes this therefore sees the cross-row curvature.
    pub fn full_inverse_apply(
        &self,
        w_t: ArrayView1<'_, f64>,
        w_beta: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let (mut u_t, mut u_beta) = self.full_inverse_apply_base(w_t, w_beta)?;
        if let Some(woodbury) = self.cross_row_woodbury.as_ref() {
            // u ← u − H₀'⁻¹U C⁻¹ D Uᵀ u. `u_t` is the `t`-block of `H₀'⁻¹ w`.
            let h0inv_w_t = u_t.clone();
            woodbury.apply_inverse_correction(
                h0inv_w_t.view(),
                woodbury.source_entries(),
                &mut u_t,
                &mut u_beta,
            )?;
        }
        Ok((u_t, u_beta))
    }

    /// Bare bordered-arrow inverse solve against the cached per-row factors and
    /// Schur factor (the NO-SELF base `H₀'` when a cross-row Woodbury is
    /// present). [`Self::full_inverse_apply`] wraps this with the rank-`R`
    /// correction; [`CrossRowWoodbury::build`] calls this directly (before the
    /// carrier exists) to form `H₀'⁻¹U`.
    fn full_inverse_apply_base(
        &self,
        w_t: ArrayView1<'_, f64>,
        w_beta: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let total_len = self.delta_t_len();
        if w_t.len() != total_len || w_beta.len() != self.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "full_inverse_apply: rhs shapes (w_t={}, w_beta={}) != (delta_t_len={}, K={})",
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
                            "full_inverse_apply: H_βt^({i}) apply failed (htbeta cache \
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
                            "full_inverse_apply: H_tβ^({i}) apply failed (htbeta cache \
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
    ///
    /// Cross-row IBP (#1038) note: this is the β-block primitive of the
    /// factored base `S_β` (`H₀'` when a [`CrossRowWoodbury`] is present), used
    /// internally by [`Self::full_inverse_apply_base`]; it is deliberately NOT
    /// Woodbury-corrected so the base solve stays bare. The cross-row term has
    /// no `β` support, so `(H_full⁻¹)_ββ = S_β⁻¹` exactly on the directions any
    /// IBP ρ-trace contracts. A consumer needing the full `(H_full⁻¹)_ββ` for a
    /// β-supported direction should call [`Self::full_inverse_apply`] with a
    /// unit `β`-RHS (which applies the rank-`R` correction).
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
    // #1038 cross-row IBP: when the system carries the exact rank-`R` source, the
    // evidence base must be the NO-SELF `H₀'` (per-row logit-slot self term
    // `d_k·z'_ik²` downdated), so the full rank-one outer product `U D Uᵀ` — which
    // re-adds the `i=j` diagonal — does not double-count. We factor against `H₀'`,
    // then layer the exact Woodbury correction (value + logdet + adjoint) onto the
    // resulting cache. The Newton step is corrected to `H_full⁻¹(−g)` below so the
    // returned step and the reported curvature describe the SAME `H_full`.
    let downdated_owner;
    let (sys, ibp_source): (&ArrowSchurSystem, Option<&IbpCrossRowSource>) =
        match sys.ibp_cross_row.as_ref() {
            Some(source) => {
                let mut downdated = sys.clone();
                let total_len = downdated.row_offsets[downdated.rows.len()];
                let down = source.self_term_downdate(total_len);
                let offsets = Arc::clone(&downdated.row_offsets);
                for (i, row) in downdated.rows.iter_mut().enumerate() {
                    let base = offsets[i];
                    let di = row.htt.nrows();
                    for j in 0..di {
                        row.htt[[j, j]] -= down[base + j];
                    }
                }
                // The downdated rows carry a new curvature fingerprint.
                downdated.refresh_row_hessian_fingerprint();
                downdated_owner = downdated;
                (&downdated_owner, Some(source))
            }
            None => (sys, None),
        };
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
    let (htt_factors_undamped, gauge_deflated_directions) = if ridge_t == 0.0 {
        (
            ArrowUndampedFactors::SameAsDamped,
            step.gauge_deflated_directions,
        )
    } else {
        let undamped = factor_blocks_for_system(sys, 0.0, options, &backend)?;
        (
            ArrowUndampedFactors::Owned(undamped.factors),
            undamped.gauge_deflated_directions,
        )
    };
    let mut cache = ArrowFactorCache {
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
        gauge_deflated_directions,
        cross_row_woodbury: None,
    };
    let mut delta_t = step.delta_t;
    let mut delta_beta = step.delta_beta;
    if let Some(source) = ibp_source {
        // The cache's per-row factors are now `H₀'`; build the exact rank-`R`
        // Woodbury (one back-solve per atom column + the `R×R` capacitance LU)
        // and store it so the logdet/inverse/adjoint all read the same
        // `H_full = H₀' + U D Uᵀ`.
        if let Some(woodbury) = CrossRowWoodbury::build(&cache, source)? {
            // Correct the Newton step from `H₀'⁻¹(−g)` to `H_full⁻¹(−g)`. The base
            // `step.delta_t/β` solve `H₀' Δ₀ = −g`, so `delta_t` is the `t`-block
            // of `H₀'⁻¹(−g)`; the rank-`R` Woodbury inverse correction reads
            // `Uᵀ Δ₀ₜ` and writes both the `t` and `β` blocks of `H_full⁻¹(−g)`.
            let h0inv_neg_g_t = delta_t.clone();
            woodbury.apply_inverse_correction(
                h0inv_neg_g_t.view(),
                &source.entries,
                &mut delta_t,
                &mut delta_beta,
            )?;
            cache.cross_row_woodbury = Some(woodbury);
        }
    }
    Ok((delta_t, delta_beta, cache))
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
        // #1014: the streaming/residency path is the memory-bound assembly wall,
        // so its reduced dense Schur solve runs certified mixed precision by
        // default (κ-gated f32 factor + f64 residual refinement, automatic f64
        // fallback). The reduced-Schur f64 factor — and therefore every evidence
        // log-determinant — is unaffected: only the Δβ solve drops to f32. An
        // explicit caller policy is honored as-is.
        let streaming_options = options.with_streaming_mixed_precision_default();
        let mut streaming = StreamingArrowSchur::from_system(sys, chunk_size);
        return streaming
            .solve(ridge_t, ridge_beta, &streaming_options)
            .map(|(delta_t, delta_beta, _)| (delta_t, delta_beta, PcgDiagnostics::default()));
    }
    // #1017 phase-3 production seam: when a device is present and the dense
    // Schur work clears the work-based dispatch threshold (LLM/SAE shapes —
    // few rows, thousands of border columns), route the Direct-mode point solve
    // through the fully device-resident batched Arrow-Schur sequence. The host
    // never sees the factors here (this `_core` entry discards the IFT cache),
    // so the device's scalars-only `(Δt, Δβ)` readback is exactly the contract.
    // Magic-by-default: no flag — the predicate fires from the shape. Any
    // non-admission or device failure falls through to the bit-identical CPU
    // path below, so the numbers are unchanged when the device declines.
    if let Some(device_step) = try_device_arrow_direct(sys, ridge_t, ridge_beta, options) {
        return device_step;
    }
    // #1017 production seam for the matrix-free SAE path: the real SAE decoder
    // β-block is the Kronecker operator (`htbeta_matvec`), never a dense slab,
    // so the dense device-resident solve above declines and the mode is
    // `InexactPCG`. The reduced-Schur matvec `Σ_i Y_i^T(Y_i x)` is the PCG hot
    // loop and is exactly what `gpu_schur_matvec_backend` offloads (dense rows)
    // or the row-procedural Kronecker apply handles (matrix-free). When the
    // device admits and the caller did not already supply a matvec, build one
    // and inject it through a cloned options so the existing InexactPCG branch
    // consumes it. On any device decline the original (CPU) options are used
    // unchanged, so results are bit-identical.
    if let Some(device_options) = maybe_inject_gpu_schur_matvec(sys, ridge_t, ridge_beta, options) {
        return solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, &device_options).map(
            |step| {
                let mut diagnostics = step.pcg_diagnostics;
                diagnostics.used_device_arrow = true;
                (step.delta_t, step.delta_beta, diagnostics)
            },
        );
    }
    solve_arrow_newton_step_artifacts(sys, ridge_t, ridge_beta, options)
        .map(|step| (step.delta_t, step.delta_beta, step.pcg_diagnostics))
}


/// Build and inject the GPU reduced-Schur matvec backend for an admitted
/// `InexactPCG` solve, returning a cloned `ArrowSolveOptions` carrying it.
///
/// Returns `None` (caller keeps the original CPU options) when: the mode is not
/// `InexactPCG`; the caller already supplied a `gpu_matvec`; no device is
/// present; the work-based predicate declines the shape; or the backend build
/// fails for any reason. The PCG numerics are identical whether the matvec runs
/// on host or device (same reduced Schur operator, same f64 accumulation), so
/// injecting it changes only where the `Σ_i Y_i^T(Y_i x)` flops execute.
fn maybe_inject_gpu_schur_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Option<ArrowSolveOptions> {
    if options.mode != ArrowSolverMode::InexactPCG || options.gpu_matvec.is_some() {
        return None;
    }
    if !sys.cross_row_penalties.is_empty() || options.streaming_chunk_size.is_some() {
        return None;
    }
    let runtime = crate::gpu::runtime::GpuRuntime::global()?;
    // #1017 Phase-1 call-site re-key: the reduced-Schur matvec is `O(n · d · k)`
    // per apply and the PCG runs `cg_iters` applies over device-resident frames,
    // so the offload becomes profitable on the CG-AMORTISED batched work — the
    // exact `n × k × d` arithmetic the dense-Direct `(n, k)` floor misses (it
    // ignores the per-row frame depth `d` and the `1/cg_iters` staging
    // amortisation). The CG budget here is the same `max_iterations` the PCG loop
    // launches with (`pcg.max_iterations.min(trust_region.max_iterations)`).
    // `try_device_arrow_direct` deliberately keeps the dense gate — that path is
    // one large factorization, not the amortised matvec.
    let cg_iters = options
        .pcg
        .max_iterations
        .min(options.trust_region.max_iterations);
    if !runtime
        .policy()
        .reduced_schur_matvec_should_offload(sys.rows.len(), sys.k, sys.d, cg_iters)
    {
        return None;
    }
    let matvec =
        crate::gpu::arrow_schur::gpu_schur_matvec_backend(sys, ridge_t, ridge_beta).ok()?;
    let mut device_options = options.clone();
    device_options.gpu_matvec = Some(matvec);
    Some(device_options)
}


/// Admission + dispatch for the device-resident Direct Arrow-Schur point solve.
///
/// Returns `Some(Ok(..))` when the device path produced a step, `Some(Err(..))`
/// only for a genuine numerical failure the device surfaced that the caller
/// must see (a non-PD pivot the LM escalation should respond to), and `None`
/// when the device declined for any reason — shape below threshold, no CUDA,
/// matrix-free operators present, or a transient device-unavailable — so the
/// caller transparently falls back to the CPU path.
///
/// The predicate is the same work-based gate the device-resident PIRLS loop
/// uses (`dense_hessian_work_target_is_gpu`) keyed on `(n_rows, border_k)`:
/// the reduced Schur assembly is `O(n · d · k²)`, dominated by the `k²` border,
/// so `k` is the dense-Hessian width. Below `DEVICE_LOOP_MIN_P` border columns
/// or below the flop floor the launch/staging overhead loses to the CPU dense
/// Cholesky, so the device is not engaged.
fn try_device_arrow_direct(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Option<Result<(Array1<f64>, Array1<f64>, PcgDiagnostics), ArrowSchurError>> {
    // Only the dense Direct mode maps onto the device dense-Schur sequence.
    // SqrtBA / InexactPCG have distinct numerics (square-root factors,
    // truncated-CG trust region) and must stay on their CPU implementations so
    // results are unchanged.
    if options.mode != ArrowSolverMode::Direct {
        return None;
    }
    // Cross-row penalties, streaming, and matrix-free H_ββ / H_tβ operators are
    // all outside the dense device path; the GPU entry itself rejects the
    // matrix-free cases, but short-circuit here so we never pay a device probe
    // for a system that cannot route.
    if !sys.cross_row_penalties.is_empty()
        || options.streaming_chunk_size.is_some()
        || sys.hbb_matvec.is_some()
        || sys.htbeta_matvec.is_some()
    {
        return None;
    }
    let runtime = crate::gpu::runtime::GpuRuntime::global()?;
    let admitted = runtime
        .policy()
        .dense_hessian_work_target_is_gpu(sys.rows.len(), sys.k);
    if !admitted {
        return None;
    }
    match crate::gpu::arrow_schur::solve_arrow_newton_step(sys, ridge_t, ridge_beta) {
        Ok(solution) => {
            let diagnostics = PcgDiagnostics {
                used_device_arrow: true,
                ..PcgDiagnostics::default()
            };
            Some(Ok((solution.delta_t, solution.delta_beta, diagnostics)))
        }
        // A non-PD per-row block or Schur pivot is a real numerical condition
        // the LM escalation around this solve must respond to; surface it as the
        // matching CPU error variant so `solve_with_lm_escalation_inner` bumps
        // the ridge and retries (it re-enters here and may route to device again
        // at the larger ridge, or fall to CPU if the device keeps declining).
        Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::RidgeBumpRequired { row, bump }) => {
            Some(Err(ArrowSchurError::PerRowFactorFailed {
                row,
                reason: format!("device per-row block non-PD; suggested ridge bump {bump:e}"),
            }))
        }
        // A non-PD reduced Schur is a real numerical condition the LM escalation
        // must respond to (bump the β-ridge and retry); surface it as the
        // matching CPU error rather than re-running the same factorisation on
        // the CPU only to fail identically.
        Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::SchurFactorFailed { reason }) => {
            Some(Err(ArrowSchurError::SchurFactorFailed { reason }))
        }
        // Unavailable (transient / below device policy) and
        // GpuRequiresDenseSystem (matrix-free, already filtered above) both mean
        // "device declined" — fall back to CPU transparently.
        Err(_) => None,
    }
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
    gauge_deflated_directions: usize,
}


struct ArrowBlockFactorization {
    factors: ArrowFactorSlab,
    gauge_deflated_directions: usize,
}


fn factor_blocks_for_system<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    options: &ArrowSolveOptions,
    backend: &B,
) -> Result<ArrowBlockFactorization, ArrowSchurError> {
    let Some(deflation) = sys.row_gauge_deflation.as_ref() else {
        return Ok(ArrowBlockFactorization {
            factors: backend.factor_blocks(
                &sys.rows,
                ridge_t,
                sys.d,
                options.tolerate_ill_conditioning,
            )?,
            gauge_deflated_directions: 0,
        });
    };
    let mut blocks = Vec::with_capacity(sys.rows.len());
    let mut count = 0usize;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let result = factor_one_row_result(
            row,
            ridge_t,
            sys.row_dims[row_idx],
            row_idx,
            options.tolerate_ill_conditioning,
            deflation.row(row_idx),
        )?;
        count += result.gauge_deflated_directions;
        blocks.push(result.factor);
    }
    Ok(ArrowBlockFactorization {
        factors: ArrowFactorSlab::from_blocks(blocks),
        gauge_deflated_directions: count,
    })
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


fn back_substitute_delta_t<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    delta_beta: ArrayView1<'_, f64>,
    backend: &B,
) -> Array1<f64> {
    let n = sys.rows.len();
    let total_dt_len = sys.row_offsets[n];
    let mut delta_t = Array1::<f64>::zeros(total_dt_len);
    // `Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ)` is row-block-independent:
    // each row writes only its own contiguous `delta_t[row_offsets[i]..]`
    // segment. Fan out over the SAE LLM row count with the same nesting guard
    // (`rayon::current_thread_index()`) and row-min gate the `schur_matvec` hot
    // loop uses (#1017), so the topology race's outer candidate fan-out is not
    // oversubscribed. Disjoint writes ⇒ no reduction, no run-to-run drift.
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    let solve_row = |i: usize, out: &mut [f64]| {
        let di = sys.row_dims[i];
        assert!(
            sys.rows[i].gt.len() == di,
            "back_substitute_delta_t: row {i} gt len {} != row dim {di}",
            sys.rows[i].gt.len()
        );
        let mut htbeta_slice = Array1::<f64>::zeros(di);
        sys_htbeta_apply_row(sys, i, &sys.rows[i], delta_beta, &mut htbeta_slice);
        let mut rhs = Array1::<f64>::zeros(di);
        for c in 0..di {
            rhs[c] = sys.rows[i].gt[c] + htbeta_slice[c];
        }
        let dt_i = backend.solve_block_vector(htt_factors.factor(i), rhs.view());
        for c in 0..di {
            out[c] = -dt_i[c];
        }
    };
    if parallel {
        use rayon::prelude::*;
        const CHUNK: usize = 64;
        let row_offsets = &sys.row_offsets;
        // `par_chunks_mut` over uniform chunks does not align with variable row
        // dims, so partition by row chunk and hand each chunk its own contiguous
        // output segment via `split_at_mut` keyed on `row_offsets`.
        let dt_slice = delta_t.as_slice_mut().expect("delta_t contiguous");
        let n_chunks = n.div_ceil(CHUNK);
        let mut remaining = dt_slice;
        let mut segments: Vec<(usize, &mut [f64])> = Vec::with_capacity(n_chunks);
        let mut prev_end = 0usize;
        for chunk in 0..n_chunks {
            let start = chunk * CHUNK;
            let end = (start + CHUNK).min(n);
            let seg_len = row_offsets[end] - row_offsets[start];
            assert!(
                prev_end == row_offsets[start],
                "back_substitute_delta_t: non-contiguous row segment at chunk start {start} \
                 (prev_end={prev_end}, row_offset={})",
                row_offsets[start]
            );
            let (seg, rest) = remaining.split_at_mut(seg_len);
            remaining = rest;
            segments.push((start, seg));
            prev_end = row_offsets[end];
        }
        segments.into_par_iter().for_each(|(start, seg)| {
            let end = (start + CHUNK).min(n);
            let mut local = 0usize;
            for i in start..end {
                let di = sys.row_dims[i];
                solve_row(i, &mut seg[local..local + di]);
                local += di;
            }
        });
    } else {
        for i in 0..n {
            let row_base = sys.row_offsets[i];
            let di = sys.row_dims[i];
            solve_row(
                i,
                delta_t
                    .as_slice_mut()
                    .expect("delta_t contiguous")
                    .get_mut(row_base..row_base + di)
                    .expect("row segment in bounds"),
            );
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
    let certificate_tol = residual_relative_tolerance
        .max(MIXED_PRECISION_CERTIFICATE_EPSILON_MULTIPLIER * f64::EPSILON);
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

    Ok(Some(MixedPrecisionAttempt::Fallback {
        reason: "mixed refinement loop exhausted without certification".to_string(),
    }))
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
    let threshold = margin
        .min(MIXED_PRECISION_KAPPA_MARGIN_CEILING)
        .max(F32_UNIT_ROUNDOFF);
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
    // Precondition: positive, finite factor diagonals (see
    // `cholesky_solve_vector_fixed`). The certified mixed-precision streaming
    // path refines in f64 and falls back when this f32 solve is not usable, but
    // guard the precondition loudly — always, release included — so a future
    // factor source that skips that refinement cannot divide by a
    // zero/non-finite pivot silently.
    assert!(
        (0..n).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f32::MIN_POSITIVE),
        "cholesky_solve_lower_f32: factor diagonal must be finite and non-subnormal"
    );
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
            gauge_deflated_directions: 0,
        });
    }
    let backend = CpuBatchedBlockSolver;

    // 1. BA point elimination: per-row Cholesky factors of
    // (H_tt^(i) + ridge_t · I).  `factor_blocks` reads the actual row
    // dimension from `row.htt.nrows()` so heterogeneous systems work.
    let block_factorization = factor_blocks_for_system(sys, ridge_t, options, &backend)?;
    let htt_factors = block_factorization.factors;
    let gauge_deflated_directions = block_factorization.gauge_deflated_directions;

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
                            gauge_deflated_directions,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        log::info!("arrow-Schur mixed precision fallback to f64: {reason}");
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
                            gauge_deflated_directions,
                        });
                    }
                    MixedPrecisionAttempt::Fallback { reason } => {
                        log::info!("arrow-Schur mixed precision fallback to f64: {reason}");
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
                log::info!(
                    "arrow-Schur mixed precision fallback to f64: InexactPCG does not expose a dense Schur factor for certified f32 refinement"
                );
                mixed_precision_status = MixedPrecisionStatus::F64Fallback;
            }
            // Auto-select preconditioner level: starts with JacobiPreconditioner
            // (Diagonal / BetaBlockJacobi) and escalates to ClusterJacobi or
            // AdditiveSchwarz when K > 100 and PCG exhausts max_iterations.
            if options.trust_region.radius == f64::INFINITY {
                if let Some(device_data) = sys.device_sae_pcg.as_ref() {
                    let max_iterations = options
                        .pcg
                        .max_iterations
                        .min(options.trust_region.max_iterations);
                    let relative_tolerance = options
                        .pcg
                        .relative_tolerance
                        .max(options.trust_region.steihaug_relative_tolerance);
                    if let Ok((delta, mut diag)) =
                        crate::gpu::arrow_schur::solve_sae_matrix_free_pcg(
                            sys,
                            device_data.as_ref(),
                            ridge_t,
                            ridge_beta,
                            &rhs_beta,
                            max_iterations,
                            relative_tolerance,
                        )
                    {
                        diag.used_device_arrow = true;
                        return Ok(ArrowNewtonStepArtifacts {
                            delta_t: back_substitute_delta_t(
                                sys,
                                &htt_factors,
                                delta.view(),
                                &backend,
                            ),
                            delta_beta: delta,
                            htt_factors,
                            schur_factor: None,
                            pcg_diagnostics: diag,
                            gauge_deflated_directions,
                        });
                    }
                }
            }
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
        gauge_deflated_directions,
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
        used_device_arrow: false,
    };

    Ok(ArrowNewtonStepArtifacts {
        delta_t: x_t,
        delta_beta: x_beta,
        htt_factors: precond.htt_factors,
        schur_factor: Some(precond.schur_factor),
        pcg_diagnostics: diag,
        gauge_deflated_directions: 0,
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
    // Precondition: positive, finite factor diagonals (see
    // `cholesky_solve_vector_fixed`). Guard loudly — always, release included —
    // so a future caller supplying an unvalidated factor cannot divide by a
    // zero/non-finite pivot and leak a silent `NaN` into the Schur β-solve
    // (#1038).
    assert!(
        (0..n).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f64::MIN_POSITIVE),
        "cholesky_solve_lower: factor diagonal must be finite and non-subnormal"
    );
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


fn reduced_rhs_beta<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
) -> Array1<f64> {
    // Numerical invariant: each per-row `H_tt^(i)` factor must be PD
    // (already enforced by the adaptive-ridge `factor_blocks`).
    let k = sys.k;
    let n = sys.rows.len();
    let mut rhs_beta = Array1::<f64>::zeros(k);
    // The reduced RHS sum `Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i)` is the same
    // embarrassingly-parallel per-row reduction the `schur_matvec` hot loop
    // already fans out (#1017): each row contributes an independent length-`K`
    // vector. Reuse the identical deterministic chunk-fold so the f64 reduction
    // is bit-identical run-to-run (the topology-candidate ranking gate must not
    // move), and the identical nesting guard (`rayon::current_thread_index()`)
    // so the topology race's outer fan-out is not oversubscribed.
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    if parallel {
        use rayon::prelude::*;
        const CHUNK: usize = 64;
        let partials: Vec<Array1<f64>> = (0..n)
            .into_par_iter()
            .chunks(CHUNK)
            .map(|idxs| {
                let mut acc = Array1::<f64>::zeros(k);
                for i in idxs {
                    let row = &sys.rows[i];
                    let v = backend.solve_block_vector(htt_factors.factor(i), row.gt.view());
                    sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut acc);
                }
                acc
            })
            .collect();
        for acc in &partials {
            for j in 0..k {
                rhs_beta[j] += acc[j];
            }
        }
    } else {
        for (i, row) in sys.rows.iter().enumerate() {
            let v = backend.solve_block_vector(htt_factors.factor(i), row.gt.view());
            // H_βt^(i) · v accumulates into rhs_beta.  Routes through
            // sys.htbeta_matvec when the dense block is absent.
            sys_htbeta_accumulate_transpose(sys, i, row, v.view(), &mut rhs_beta);
        }
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
