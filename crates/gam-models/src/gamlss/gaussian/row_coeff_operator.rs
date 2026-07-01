// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

pub(crate) struct RowCoeffChannel {
    pub(crate) block: usize,
    pub(crate) design: Arc<Array2<f64>>,
}

/// Symmetric pair coefficients `c_{ab}` for `a ≤ b`. The operator adds
/// `X_a^T diag(c_{ab}) X_b` to block `block_a`'s output and the transpose
/// contribution `X_b^T diag(c_{ab}) X_a` to block `block_b` when `a != b`.
pub(crate) struct RowCoeffPair {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) coeff: Array1<f64>,
}

/// Pooled per-call scratch for `RowCoeffOperator::mul_vec`. Each call
/// pops a buffer set; if the pool is empty (parallel callers exhausted
/// it) we allocate fresh — the alloc is amortized as concurrent callers
/// recycle. The pool's `Mutex` is taken only for `pop`/`push` (constant
/// time), never during the matmul.
///
/// **Invariant**: every buffer in `pool[k].u[ch]` and `pool[k].r[ch]` has
/// length `nrows`. `mul_vec` overwrites `u` via `fast_av_into` and
/// zeroes-then-accumulates `r`, leaving both buffers in any state on
/// return — callers must not depend on residual content.
pub(crate) struct RowCoeffScratch {
    pub(crate) u: Vec<Array1<f64>>,
    pub(crate) r: Vec<Array1<f64>>,
}

/// Matrix-free operator for two-block-style joint-Hessian directional
/// derivatives that decompose as `H = sum_{a,b} X_a^T diag(c_{ab}) X_b`
/// with each `X_a` an `n × p_a` design and `c_{ab}` an `n` row coefficient
/// vector. `mul_vec` applies the operator in O(n · sum_a p_a) per call,
/// reusing pre-sized scratch buffers for `u`, `r` from a small lock-pool
/// so concurrent `mul_vec` callers do not serialize on the same scratch.
///
/// `block_offsets` gives the starting column of each output block; the
/// operator dimension is the sum of all block widths. Each channel's
/// `mul_vec` contribution is added into the slice for its output block.
pub(crate) struct RowCoeffOperator {
    pub(crate) channels: Vec<RowCoeffChannel>,
    pub(crate) block_offsets: Vec<usize>,
    pub(crate) block_widths: Vec<usize>,
    pub(crate) dim: usize,
    pub(crate) pair_coeffs: Vec<RowCoeffPair>,
    pub(crate) nrows: usize,
    pub(crate) scratch_pool: std::sync::Mutex<Vec<RowCoeffScratch>>,
}

impl RowCoeffOperator {
    /// One-line constructor for the standard (channels, pair-coeffs)
    /// recipe used by every GAMLSS LS workspace: pass the block widths,
    /// the channel list as `(block_id, design)` tuples, and the pair
    /// list as `(a, b, coeff)` tuples. Pre-allocates one scratch in the
    /// pool so the first warm `mul_vec` call skips allocation.
    pub(crate) fn from_directions(
        block_widths: Vec<usize>,
        channels: Vec<(usize, Arc<Array2<f64>>)>,
        pairs: Vec<(usize, usize, Array1<f64>)>,
        nrows: usize,
    ) -> Self {
        let channels: Vec<RowCoeffChannel> = channels
            .into_iter()
            .map(|(block, design)| RowCoeffChannel { block, design })
            .collect();
        let pair_coeffs: Vec<RowCoeffPair> = pairs
            .into_iter()
            .map(|(a, b, coeff)| RowCoeffPair { a, b, coeff })
            .collect();
        let mut block_offsets = Vec::with_capacity(block_widths.len());
        let mut acc = 0;
        for w in &block_widths {
            block_offsets.push(acc);
            acc += *w;
        }
        let n_ch = channels.len();
        let initial = RowCoeffScratch {
            u: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
            r: (0..n_ch).map(|_| Array1::<f64>::zeros(nrows)).collect(),
        };
        Self {
            channels,
            block_offsets,
            block_widths,
            dim: acc,
            pair_coeffs,
            nrows,
            scratch_pool: std::sync::Mutex::new(vec![initial]),
        }
    }

    pub(crate) fn acquire_scratch(&self) -> RowCoeffScratch {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .pop()
            .unwrap_or_else(|| {
                let n_ch = self.channels.len();
                RowCoeffScratch {
                    u: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                    r: (0..n_ch)
                        .map(|_| Array1::<f64>::zeros(self.nrows))
                        .collect(),
                }
            })
    }

    pub(crate) fn release_scratch(&self, scratch: RowCoeffScratch) {
        self.scratch_pool
            .lock()
            .expect("RowCoeffOperator scratch pool poisoned")
            .push(scratch);
    }

    pub(crate) fn projected_trace(&self, factor: &Array2<f64>) -> f64 {
        let grams = self.projected_pair_gram_table(factor);
        self.trace_from_pair_gram_table(grams.view())
    }

    pub(crate) fn projected_pair_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "RowCoeffOperator::projected_pair_gram_table".hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        self.block_widths.hash(&mut hasher);
        self.block_offsets.hash(&mut hasher);
        self.channels.len().hash(&mut hasher);
        self.pair_coeffs.len().hash(&mut hasher);
        for (idx, ch) in self.channels.iter().enumerate() {
            idx.hash(&mut hasher);
            (Arc::as_ptr(&ch.design) as usize).hash(&mut hasher);
            ch.block.hash(&mut hasher);
            ch.design.nrows().hash(&mut hasher);
            ch.design.ncols().hash(&mut hasher);
            self.block_widths[ch.block].hash(&mut hasher);
        }
        for (idx, pair) in self.pair_coeffs.iter().enumerate() {
            idx.hash(&mut hasher);
            pair.a.hash(&mut hasher);
            pair.b.hash(&mut hasher);
        }
        hasher.finish() as usize
    }

    pub(crate) fn projected_pair_gram_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "row-coefficient cached projected trace factor row mismatch: factor rows={} but dim={}",
            factor.nrows(),
            self.dim
        );
        let rank = factor.ncols();
        let pair_count = self.pair_coeffs.len();
        if self.nrows == 0 || rank == 0 || pair_count == 0 {
            return Array2::<f64>::zeros((self.nrows, pair_count));
        }
        let rows_per_chunk =
            gamlss_projected_trace_chunk_rows(rank, self.channels.len(), pair_count)
                .min(self.nrows.max(1));
        let mut grams = Array2::<f64>::zeros((self.nrows, pair_count));
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let mut projected: Vec<Array2<f64>> = Vec::with_capacity(self.channels.len());
            for ch in &self.channels {
                let block_start = self.block_offsets[ch.block];
                let width = self.block_widths[ch.block];
                let design_chunk = ch.design.slice(s![rows.clone(), ..]);
                let factor_block = factor.slice(s![block_start..block_start + width, ..]);
                projected.push(fast_ab(&design_chunk, &factor_block));
            }
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let u_a = &projected[pair.a];
                let u_b = &projected[pair.b];
                for local_i in 0..u_a.nrows() {
                    let mut value = 0.0;
                    for col in 0..rank {
                        value += u_a[[local_i, col]] * u_b[[local_i, col]];
                    }
                    out_chunk[[local_i, pair_idx]] = value;
                }
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    pub(crate) fn trace_from_pair_gram_table(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), self.pair_coeffs.len());
        let mut trace = 0.0;
        for i in 0..self.nrows {
            for (pair_idx, pair) in self.pair_coeffs.iter().enumerate() {
                let multiplier = if pair.a == pair.b { 1.0 } else { 2.0 };
                trace += multiplier * pair.coeff[i] * grams[[i, pair_idx]];
            }
        }
        trace
    }
}

impl gam_problem::HyperOperator for RowCoeffOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let mut scratch = self.acquire_scratch();
        let RowCoeffScratch { u, r } = &mut scratch;

        // 1) u_a = X_a · v[block_a slice]. `fast_av_into` writes directly
        //    into the pre-sized scratch buffer — no per-call n-sized
        //    allocation.
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            assert_eq!(ch.design.ncols(), width);
            let v_slice = v.slice(s![start..start + width]);
            gam_linalg::faer_ndarray::fast_av_into(ch.design.as_ref(), &v_slice, &mut u[k]);
        }

        // 2) r_a = sum_b c_{ab} ⊙ u_b. Zero-then-accumulate; pair coeffs
        //    contribute symmetrically when `a != b`.
        for slot in r.iter_mut() {
            slot.fill(0.0);
        }
        for pair in &self.pair_coeffs {
            let a = pair.a;
            let b = pair.b;
            let coeff = pair
                .coeff
                .as_slice()
                .expect("RowCoeffOperator pair coeff must be contiguous");
            // r[a] += coeff ⊙ u[b]; if a != b also r[b] += coeff ⊙ u[a].
            // Split the borrow so r[a] and r[b] (or u[a] and u[b]) can be
            // accessed simultaneously when a != b.
            // These per-row `r += c ⊙ u` updates are memory-bandwidth-bound
            // O(n) saxpys. Fanning each one out over Rayon is a net loss: the
            // real arithmetic is the design matvecs (`fast_av`/`fast_atv`,
            // already faer-parallel), and `to_dense` calls `mul_vec` `dim`
            // times, so a per-call Rayon fan-out here spends almost all its
            // wall time in join/latch thread-sync rather than the saxpy (#1720).
            // Run them serially.
            if a == b {
                let u_a = u[a]
                    .as_slice()
                    .expect("RowCoeffOperator u must be contiguous");
                let r_a = r[a]
                    .as_slice_mut()
                    .expect("RowCoeffOperator r must be contiguous");
                for ((r, c), u) in r_a.iter_mut().zip(coeff.iter()).zip(u_a.iter()) {
                    *r += c * u;
                }
            } else {
                let (r_a_slice, r_b_slice) = if a < b {
                    let (left, right) = r.split_at_mut(b);
                    (
                        left[a].as_slice_mut().expect("contiguous"),
                        right[0].as_slice_mut().expect("contiguous"),
                    )
                } else {
                    let (left, right) = r.split_at_mut(a);
                    (
                        right[0].as_slice_mut().expect("contiguous"),
                        left[b].as_slice_mut().expect("contiguous"),
                    )
                };
                let u_a = u[a].as_slice().expect("contiguous");
                let u_b = u[b].as_slice().expect("contiguous");
                for ((((ra, rb), c), ua), ub) in r_a_slice
                    .iter_mut()
                    .zip(r_b_slice.iter_mut())
                    .zip(coeff.iter())
                    .zip(u_a.iter())
                    .zip(u_b.iter())
                {
                    *ra += c * ub;
                    *rb += c * ua;
                }
            }
        }

        // 3) Output[block] += X_a^T r_a per channel. Single output alloc.
        let mut out = Array1::<f64>::zeros(self.dim);
        for (k, ch) in self.channels.iter().enumerate() {
            let start = self.block_offsets[ch.block];
            let width = self.block_widths[ch.block];
            let mut block = out.slice_mut(s![start..start + width]);
            // Atv into a temporary, then accumulate; `fast_atv` allocates
            // a `width`-sized array, which is bounded and small relative
            // to the n-sized u/r buffers we already reuse.
            let contrib = fast_atv(ch.design.as_ref(), &r[k]);
            block += &contrib;
        }
        self.release_scratch(scratch);
        out
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ndarray::ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        // Assemble the dense operator directly from its `Σ X_aᵀ diag(c_ab) X_b`
        // structure instead of probing `dim` basis columns through `mul_vec`
        // (#1720). Basis probing runs `dim` matvecs on every channel design —
        // `O(dim · n · Σ p_b)` — and re-forms the same block Grams `dim` times;
        // the direct form computes each `p_a × p_b` block once as a single
        // weighted `Xᵀ(diag(c) X)` gemm, cutting both the matmul FLOPs and the
        // per-`mul_vec` scratch/latch traffic. The result is identical column
        // for column: `mul_vec(e_j)` is exactly column `j` of this matrix.
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        for pair in &self.pair_coeffs {
            let ch_a = &self.channels[pair.a];
            let ch_b = &self.channels[pair.b];
            let oa = self.block_offsets[ch_a.block];
            let wa = self.block_widths[ch_a.block];
            let ob = self.block_offsets[ch_b.block];
            let wb = self.block_widths[ch_b.block];
            // block = X_aᵀ diag(coeff) X_b, formed by scaling each row i of X_b
            // by coeff[i] then a single AᵀB gemm.
            let coeff_col = pair.coeff.view().insert_axis(Axis(1));
            let weighted_b = &(*ch_b.design.as_ref()) * &coeff_col;
            let block = gam_linalg::faer_ndarray::fast_atb(ch_a.design.as_ref(), &weighted_b);
            let mut dst_ab = out.slice_mut(s![oa..oa + wa, ob..ob + wb]);
            dst_ab += &block;
            if pair.a != pair.b {
                // X_bᵀ diag(coeff) X_a is the transpose of the block above.
                let mut dst_ba = out.slice_mut(s![ob..ob + wb, oa..oa + wa]);
                dst_ba += &block.t();
            }
        }
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.projected_trace(factor)
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &gam_problem::ProjectedFactorCache,
    ) -> f64 {
        let key = gam_problem::ProjectedFactorKey::from_factor_view(
            self.projected_pair_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_pair_gram_table(factor));
        self.trace_from_pair_gram_table(grams.view())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

/// Two-block row-coefficient operator backed by `DesignMatrix`.
///
/// This is the operator-form counterpart to `DesignTwoBlockRowCoeffOperator`'s
/// old dense-array storage: it must keep the realized block designs lazy all
/// the way through `Xv` and `X^T r`. Do not cache `Array2` snapshots here;
/// `NoDensifyOperator` regression tests rely on this type to panic if a future
/// change materializes spec-backed designs.
pub(crate) struct DesignTwoBlockRowCoeffOperator {
    pub(crate) x_a: DesignMatrix,
    pub(crate) x_b: DesignMatrix,
    pub(crate) c_aa: Arc<Array1<f64>>,
    pub(crate) c_ab: Arc<Array1<f64>>,
    pub(crate) c_bb: Arc<Array1<f64>>,
    pub(crate) dim: usize,
    pub(crate) nrows: usize,
    pub(crate) pa: usize,
}

impl gam_problem::HyperOperator for DesignTwoBlockRowCoeffOperator {
    fn dim(&self) -> usize {
        self.dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.dim);
        let v_a = v.slice(s![0..self.pa]);
        let v_b = v.slice(s![self.pa..self.dim]);
        let u_a = self.x_a.matrixvectormultiply(&v_a.to_owned());
        let u_b = self.x_b.matrixvectormultiply(&v_b.to_owned());
        assert_eq!(u_a.len(), self.nrows);
        assert_eq!(u_b.len(), self.nrows);
        let r_a = self.c_aa.as_ref() * &u_a + self.c_ab.as_ref() * &u_b;
        let r_b = self.c_ab.as_ref() * &u_a + self.c_bb.as_ref() * &u_b;
        let out_a = self.x_a.transpose_vector_multiply(&r_a);
        let out_b = self.x_b.transpose_vector_multiply(&r_b);
        let mut out = Array1::<f64>::zeros(self.dim);
        out.slice_mut(s![0..self.pa]).assign(&out_a);
        out.slice_mut(s![self.pa..self.dim]).assign(&out_b);
        out
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ndarray::ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.dim);
        let mut basis = Array1::<f64>::zeros(self.dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(local_col).assign(&col);
            basis[global_col] = 0.0;
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim, self.dim));
        self.mul_basis_columns_into(0, out.view_mut());
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        // For the two-block row-coefficient operator
        //   B v = [X_a^T (c_aa·u_a + c_ab·u_b),  X_b^T (c_ab·u_a + c_bb·u_b)]
        // with u_a = X_a v_a, u_b = X_b v_b, the column-wise quadratic form is
        //   F[:,k]^T B F[:,k] = u_a^T r_a + u_b^T r_b
        //                    = Σ_i (c_aa[i] u_a[i]² + 2 c_ab[i] u_a[i] u_b[i]
        //                            + c_bb[i] u_b[i]²)
        // so the projected trace never needs the X^T r step that the default
        // mul_vec path computes, and the per-row coefficients fold the K
        // columns into a single weighted sum once U_a, U_b are formed.
        let grams = self.projected_row_gram_triples(factor);
        self.trace_from_row_gram_triples(grams.view())
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &gam_problem::ProjectedFactorCache,
    ) -> f64 {
        // Validate the factor row count up front. Without this, a caller that
        // hands in a factor whose row count does not equal the joint p slips
        // into the per-column `mul_vec` slicing where a `assert_eq!`
        // panics with the generic `left/right` message — that loses the
        // operator identity and the (pa, pb) split which is the only useful
        // diagnostic when the trace caller's own dimension bookkeeping is
        // off. Validate at the operator boundary so the panic localises the
        // caller, and so this contract is enforced in release builds too
        // (the inner `assert_eq!` is a debug-only safety net).
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let key = gam_problem::ProjectedFactorKey::from_factor_view(
            self.projected_row_gram_cache_id(),
            factor.view(),
        );
        let grams = cache.get_or_insert_with(key, || self.projected_row_gram_triples(factor));
        self.trace_from_row_gram_triples(grams.view())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

impl DesignTwoBlockRowCoeffOperator {
    pub(crate) fn design_cache_token(design: &DesignMatrix) -> usize {
        match design {
            DesignMatrix::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                Arc::as_ptr(matrix) as usize
            }
            DesignMatrix::Dense(DenseDesignMatrix::Lazy(op)) => {
                Arc::as_ptr(op) as *const () as usize
            }
            DesignMatrix::Sparse(sparse) => sparse as *const _ as usize,
        }
    }

    pub(crate) fn projected_row_gram_cache_id(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        "DesignTwoBlockRowCoeffOperator::projected_row_gram_triples".hash(&mut hasher);
        Self::design_cache_token(&self.x_a).hash(&mut hasher);
        Self::design_cache_token(&self.x_b).hash(&mut hasher);
        self.nrows.hash(&mut hasher);
        self.pa.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        hasher.finish() as usize
    }

    pub(crate) fn projected_row_gram_triples(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            factor.nrows(),
            self.dim,
            "two-block cached projected trace factor row mismatch: factor rows={} \
             but joint p={} (pa={}, pb={})",
            factor.nrows(),
            self.dim,
            self.pa,
            self.dim - self.pa,
        );
        let rank = factor.ncols();
        let mut grams = Array2::<f64>::zeros((self.nrows, 3));
        if self.nrows == 0 || rank == 0 {
            return grams;
        }
        let rows_per_chunk = gamlss_projected_trace_chunk_rows(rank, 2, 3).min(self.nrows.max(1));
        let f_a = factor.slice(s![0..self.pa, ..]);
        let f_b = factor.slice(s![self.pa..self.dim, ..]);
        let fill_chunk = |start: usize, mut out_chunk: ndarray::ArrayViewMut2<'_, f64>| {
            let end = (start + rows_per_chunk).min(self.nrows);
            let rows = start..end;
            let x_a_chunk = self
                .x_a
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_a row chunk materialization failed");
            let x_b_chunk = self
                .x_b
                .try_row_chunk(rows.clone())
                .expect("two-block projected trace x_b row chunk materialization failed");
            let u_a = fast_ab(&x_a_chunk, &f_a);
            let u_b = fast_ab(&x_b_chunk, &f_b);
            for local_i in 0..u_a.nrows() {
                let mut aa = 0.0;
                let mut ab = 0.0;
                let mut bb = 0.0;
                for col in 0..rank {
                    let a = u_a[[local_i, col]];
                    let b = u_b[[local_i, col]];
                    aa += a * a;
                    ab += a * b;
                    bb += b * b;
                }
                out_chunk[[local_i, 0]] = aa;
                out_chunk[[local_i, 1]] = ab;
                out_chunk[[local_i, 2]] = bb;
            }
        };
        if rayon::current_thread_index().is_none() && self.nrows > rows_per_chunk {
            grams
                .axis_chunks_iter_mut(Axis(0), rows_per_chunk)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    fill_chunk(chunk_idx * rows_per_chunk, out_chunk)
                });
        } else {
            for start in (0..self.nrows).step_by(rows_per_chunk) {
                let end = (start + rows_per_chunk).min(self.nrows);
                let out_chunk = grams.slice_mut(s![start..end, ..]);
                fill_chunk(start, out_chunk);
            }
        }
        grams
    }

    pub(crate) fn trace_from_row_gram_triples(&self, grams: ArrayView2<'_, f64>) -> f64 {
        assert_eq!(grams.nrows(), self.nrows);
        assert_eq!(grams.ncols(), 3);
        let c_aa = self
            .c_aa
            .as_slice()
            .expect("c_aa is constructed contiguous");
        let c_ab = self
            .c_ab
            .as_slice()
            .expect("c_ab is constructed contiguous");
        let c_bb = self
            .c_bb
            .as_slice()
            .expect("c_bb is constructed contiguous");
        let mut trace = 0.0;
        for i in 0..self.nrows {
            trace +=
                c_aa[i] * grams[[i, 0]] + 2.0 * c_ab[i] * grams[[i, 1]] + c_bb[i] * grams[[i, 2]];
        }
        trace
    }
}

/// Matrix-free joint-Hessian operator for the two-block Gaussian
/// location-scale family. The dense Hessian decomposes as
///
///   H = [[X_mu^T diag(w) X_mu,    X_mu^T diag(cross) X_ls],
///        [X_ls^T diag(cross) X_mu, X_ls^T diag(scale) X_ls]],
///
/// with `cross = 0` and `scale = 2κ²a` — the block-diagonal Gaussian Fisher
/// (expected) information (μ ⊥ σ, #684; residual-free (log σ, log σ) block,
/// #566). This MUST match the dense `exact_newton_joint_hessian_from_designs`
/// curvature object exactly: the observed cross term `2κm` (mean-zero noise)
/// over-smooths the scale and is its Fisher expectation 0. The matvec applies
/// each block by a single design-matrix multiply on each side, so the cost
/// is Θ(n (p_mu + p_ls)) per `Hv` rather than Θ(n (p_mu + p_ls)²) to form
/// the dense matrix.
pub(crate) struct GaussianLocationScaleHessianWorkspace {
    pub(crate) family: GaussianLocationScaleFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) xmu: Arc<Array2<f64>>,
    pub(crate) x_ls: Arc<Array2<f64>>,
    pub(crate) coeff_mm: Array1<f64>,
    pub(crate) coeff_ml: Array1<f64>,
    pub(crate) coeff_ll: Array1<f64>,
}

impl GaussianLocationScaleHessianWorkspace {
    pub(crate) fn new(
        family: GaussianLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        xmu: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let etamu = &block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = family.get_or_compute_row_scalars(etamu, eta_ls)?;
        // Single source of truth shared with the dense
        // `exact_newton_joint_hessian_from_designs`: μ ⊥ σ ⇒ cross = 0 (#684),
        // (ls,ls) = 2κ²a (#566). Reading the same coefficients as the dense path
        // makes the cross-block drift that caused #684 structurally impossible.
        let (coeff_mm, coeff_ml, coeff_ll) = gaussian_locscale_fisher_joint_row_coeffs(&rows);
        Ok(Self {
            family,
            block_states,
            xmu: Arc::new(xmu),
            x_ls: Arc::new(x_ls),
            coeff_mm,
            coeff_ml,
            coeff_ll,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) X`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian.
    pub(crate) fn apply_outer_subsample(
        &mut self,
        rows: &[crate::outer_subsample::WeightedOuterRow],
    ) {
        let n = self.coeff_mm.len();
        let mut mask_mm = Array1::<f64>::zeros(n);
        let mut mask_ml = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            mask_mm[i] = self.coeff_mm[i] * r.weight;
            mask_ml[i] = self.coeff_ml[i] * r.weight;
            mask_ll[i] = self.coeff_ll[i] * r.weight;
        }
        self.coeff_mm = mask_mm;
        self.coeff_ml = mask_ml;
        self.coeff_ll = mask_ll;
    }
}

impl ExactNewtonJointHessianWorkspace for GaussianLocationScaleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but built once via 3 GEMMs
        // (`Xᵀ diag(W) X` per block) instead of letting
        // `MatrixFreeSpdOperator::materialize_dense_operator` reconstruct the
        // dense Hessian via `total` canonical-basis HVPs. At large scale
        // (n≈320k, p_total≈82) the canonical-basis path takes ~568s per κ-iter
        // while the dense build via fast_xt_diag_x/y is ~1s.
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        let h_mm = xt_diag_x_dense(self.xmu.as_ref(), &self.coeff_mm)?;
        let h_ml = xt_diag_y_dense(self.xmu.as_ref(), &self.coeff_ml, self.x_ls.as_ref())?;
        let h_ll = xt_diag_x_dense(self.x_ls.as_ref(), &self.coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pmu, 0..pmu]).assign(&h_mm);
        h.slice_mut(s![0..pmu, pmu..total]).assign(&h_ml);
        h.slice_mut(s![pmu..total, pmu..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "GaussianLocationScale matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let u_mu = fast_av(self.xmu.as_ref(), &v.slice(s![0..pmu]));
        let u_ls = fast_av(self.x_ls.as_ref(), &v.slice(s![pmu..total]));
        let r_mu = &self.coeff_mm * &u_mu + &self.coeff_ml * &u_ls;
        let r_ls = &self.coeff_ml * &u_mu + &self.coeff_ll * &u_ls;
        let out_mu = fast_atv(self.xmu.as_ref(), &r_mu);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pmu]).assign(&out_mu);
        out.slice_mut(s![pmu..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pmu = self.xmu.ncols();
        let p_ls = self.x_ls.ncols();
        let total = pmu + p_ls;
        // Per-column reduction is independent; parallelize across columns.
        let diag_mu: Vec<f64> = (0..pmu)
            .into_par_iter()
            .map(|j| {
                let col = self.xmu.column(j);
                col.iter()
                    .zip(self.coeff_mm.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let diag_ls: Vec<f64> = (0..p_ls)
            .into_par_iter()
            .map(|j| {
                let col = self.x_ls.column(j);
                col.iter()
                    .zip(self.coeff_ll.iter())
                    .map(|(&v, &c)| c * v * v)
                    .sum()
            })
            .collect();
        let mut diag = Array1::<f64>::zeros(total);
        for (j, v) in diag_mu.into_iter().enumerate() {
            diag[j] = v;
        }
        for (j, v) in diag_ls.into_iter().enumerate() {
            diag[pmu + j] = v;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_flat,
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu = fast_av(self.xmu.as_ref(), &d_beta_flat.slice(s![0..pmu]));
        let xi_ls = fast_av(self.x_ls.as_ref(), &d_beta_flat.slice(s![pmu..total]));
        let directional = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684), so its directional derivative is
        // identically 0 — matching the dense
        // `exact_newton_joint_hessian_directional_derivative_from_designs`, which
        // likewise does not assemble `directional.1`.
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
                &self.block_states,
                &DenseOrOperator::Borrowed(self.xmu.as_ref()),
                &DenseOrOperator::Borrowed(self.x_ls.as_ref()),
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn gam_problem::HyperOperator>>, String> {
        let n = self.xmu.nrows();
        let pmu = self.xmu.ncols();
        let pls = self.x_ls.ncols();
        let total = pmu + pls;
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "GaussianLocationScale d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        let etamu = &self.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
        let eta_ls = &self.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let rows = self.family.get_or_compute_row_scalars(etamu, eta_ls)?;
        let ximu_u = fast_av(self.xmu.as_ref(), &d_beta_u.slice(s![0..pmu]));
        let xi_ls_u = fast_av(self.x_ls.as_ref(), &d_beta_u.slice(s![pmu..total]));
        let ximu_v = fast_av(self.xmu.as_ref(), &d_beta_v.slice(s![0..pmu]));
        let xi_ls_v = fast_av(self.x_ls.as_ref(), &d_beta_v.slice(s![pmu..total]));
        let directional =
            gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximu_v, &xi_ls_v);
        let c_mm = directional.0;
        let c_ll = directional.2;
        // Fisher cross block ≡ 0 (μ ⊥ σ; #684); its second directional
        // derivative is identically 0 too — match the dense path (which does not
        // assemble `directional.1`).
        let c_ml = Array1::<f64>::zeros(c_mm.len());
        Ok(Some(Arc::new(make_two_block_row_coeff_operator(
            self.xmu.clone(),
            self.x_ls.clone(),
            c_mm,
            c_ml,
            c_ll,
            n,
        ))))
    }
}

/// Build a `RowCoeffOperator` for the standard two-block GAMLSS structure
/// with one design per block and three pair coefficients (a,a), (a,b), (b,b).
/// The resulting matrix mirrors the dense
/// `X_a^T diag(c_aa) X_a + X_a^T diag(c_ab) X_b + X_b^T diag(c_ab) X_a + X_b^T diag(c_bb) X_b`
/// assembly emitted by `gaussian_joint_hessian_from_designs` (Gaussian path)
/// and the `xt_diag_*` block writers (binomial path).
pub(crate) fn make_two_block_row_coeff_operator(
    x_a: Arc<Array2<f64>>,
    x_b: Arc<Array2<f64>>,
    c_aa: Array1<f64>,
    c_ab: Array1<f64>,
    c_bb: Array1<f64>,
    nrows: usize,
) -> RowCoeffOperator {
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    RowCoeffOperator::from_directions(
        vec![pa, pb],
        vec![(0, x_a), (1, x_b)],
        vec![(0, 0, c_aa), (0, 1, c_ab), (1, 1, c_bb)],
        nrows,
    )
}

pub(crate) fn make_two_block_design_row_coeff_operator(
    x_a: DesignMatrix,
    x_b: DesignMatrix,
    c_aa: Arc<Array1<f64>>,
    c_ab: Arc<Array1<f64>>,
    c_bb: Arc<Array1<f64>>,
) -> Result<DesignTwoBlockRowCoeffOperator, String> {
    let nrows = x_a.nrows();
    if x_b.nrows() != nrows || c_aa.len() != nrows || c_ab.len() != nrows || c_bb.len() != nrows {
        return Err(GamlssError::DimensionMismatch { reason: format!(
            "two-block row coefficient operator dimension mismatch: rows a={}, b={}, coeffs={}/{}/{}",
            nrows,
            x_b.nrows(),
            c_aa.len(),
            c_ab.len(),
            c_bb.len()
        ) }.into());
    }
    let pa = x_a.ncols();
    let pb = x_b.ncols();
    Ok(DesignTwoBlockRowCoeffOperator {
        x_a,
        x_b,
        c_aa,
        c_ab,
        c_bb,
        dim: pa + pb,
        nrows,
        pa,
    })
}

#[cfg(test)]
mod to_dense_direct_1720_tests {
    use super::*;
    use gam_problem::HyperOperator;
    use std::time::Instant;

    /// Deterministic, allocation-free pseudo-random value in `[-1, 1)` keyed by
    /// `(i, j, salt)` — a splitmix64 finaliser over a mixed index. Gives every
    /// test run identical designs/coefficients without an RNG dependency.
    fn pseudo(i: usize, j: usize, salt: u64) -> f64 {
        let mut z = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9);
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    /// A two-block Gaussian-location-scale-shaped row-coefficient operator:
    /// designs `X_mu (n×pa)`, `X_ls (n×pb)`, and the three symmetric per-row
    /// pair coefficients `(0,0)`, `(0,1)`, `(1,1)` — the exact channel/pair
    /// layout `GaussianLocationScaleHessianWorkspace` builds for the outer
    /// Hessian correction.
    fn build_op(n: usize, pa: usize, pb: usize) -> RowCoeffOperator {
        let x_a = Arc::new(Array2::from_shape_fn((n, pa), |(i, j)| pseudo(i, j, 1)));
        let x_b = Arc::new(Array2::from_shape_fn((n, pb), |(i, j)| pseudo(i, j, 2)));
        let c_aa = Array1::from_shape_fn(n, |i| 0.5 + 0.25 * pseudo(i, 0, 3).abs());
        let c_ab = Array1::from_shape_fn(n, |i| pseudo(i, 0, 4));
        let c_bb = Array1::from_shape_fn(n, |i| 0.5 + 0.25 * pseudo(i, 0, 5).abs());
        RowCoeffOperator::from_directions(
            vec![pa, pb],
            vec![(0, x_a), (1, x_b)],
            vec![(0, 0, c_aa), (0, 1, c_ab), (1, 1, c_bb)],
            n,
        )
    }

    /// The direct block-Gram `to_dense` (#1720) must be identical, column for
    /// column, to the basis-vector-probing reference it replaced: column `j` of
    /// the dense operator is exactly `mul_vec(e_j)`. This is an independent code
    /// path (per-column matvec) from the direct `X_aᵀ diag(c) X_b` assembly, so
    /// agreement pins the fast path's correctness, not a value against itself.
    #[test]
    fn to_dense_matches_basis_probe_reference_1720() {
        let op = build_op(257, 5, 4);
        let direct = op.to_dense();
        let dim = op.dim();
        let mut reference = Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            let mut e = Array1::<f64>::zeros(dim);
            e[j] = 1.0;
            reference.column_mut(j).assign(&op.mul_vec(&e));
        }
        let max_abs = direct
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let scale = reference
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        assert!(
            max_abs <= 1e-9 * scale,
            "direct to_dense diverged from the basis-probe reference: max_abs={max_abs} scale={scale}"
        );
    }

    /// Best-of-`blocks` mean per-call `to_dense` wall time at `n` rows.
    fn per_call_secs(n: usize, pa: usize, pb: usize) -> f64 {
        let op = build_op(n, pa, pb);
        for _ in 0..3 {
            std::hint::black_box(op.to_dense());
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let reps = 20;
            let t0 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(op.to_dense());
            }
            best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
        }
        best
    }

    /// Root cause of #1720: the outer-Hessian correction densified the joint
    /// row-coefficient operator by probing `dim` basis vectors through
    /// `mul_vec`, so each `to_dense` cost `O(dim · n · Σ p_b)` and — because a
    /// plain Gaussian REML is ~flat in n — the location-scale fit's *relative*
    /// overhead grew super-linearly (3× at n=100 → 9× at n=2000). The direct
    /// block-Gram assembly is `O(Σ_pairs n · p_a p_b)`, i.e. linear in n at
    /// fixed width. Pin that scaling: quadrupling the rows must cost no worse
    /// than ~6× the wall time (linear 4× plus fixed-overhead / measurement
    /// slack), never the super-linear blow-up the probing regression showed.
    #[test]
    fn to_dense_scales_linearly_in_n_1720() {
        let (pa, pb) = (12, 12);
        let t_small = per_call_secs(1_000, pa, pb);
        let t_large = per_call_secs(4_000, pa, pb);
        let ratio = t_large / t_small.max(1e-9);
        assert!(
            ratio <= 6.0,
            "#1720 to_dense scales super-linearly in n: \
             t(1000)={t_small:.6}s t(4000)={t_large:.6}s ratio={ratio:.2} \
             (want <= 6.0 for a 4x row increase)"
        );
    }
}
