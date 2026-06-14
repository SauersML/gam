//! Matrix-free penalty-side `H_ββ` operators: the [`BetaPenaltyOp`] trait and
//! every concrete operator (dense, block, Kronecker, factored-frame, composite,
//! matvec-diagonal) plus the β-coupling graph used for block-Jacobi clustering.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct BetaEdge {
    pub(crate) a: usize,
    pub(crate) b: usize,
}


#[derive(Debug, Clone)]
pub(crate) struct BetaCouplingGraph {
    pub(crate) num_blocks: usize,
    pub(crate) edges: Vec<BetaEdge>,
    pub(crate) adj_start: Vec<usize>,
    pub(crate) adj_targets: Vec<usize>,
}


impl BetaCouplingGraph {
    pub(crate) fn build(block_offsets: &[Range<usize>], htbeta_rows: &[Array2<f64>]) -> Self {
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

    pub(crate) fn neighbours(&self, node: usize) -> &[usize] {
        &self.adj_targets[self.adj_start[node]..self.adj_start[node + 1]]
    }

    pub(crate) fn component_partition(&self) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..self.num_blocks).collect();
        let mut rank = vec![0u8; self.num_blocks];

        pub(crate) fn find(parent: &mut [usize], mut x: usize) -> usize {
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

    pub(crate) fn expand_one_hop(&self, seed: &[usize]) -> Vec<usize> {
        let mut expanded = seed.to_vec();
        for &block in seed {
            expanded.extend_from_slice(self.neighbours(block));
        }
        expanded.sort_unstable();
        expanded.dedup();
        expanded
    }
}
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
    pub(crate) fn a_phi_shared(&self) -> Arc<[Vec<(usize, f64)>]> {
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
    pub(crate) k: usize,
    pub(crate) matvec: SharedBetaMatvec,
    pub(crate) diagonal_vec: Array1<f64>,
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
