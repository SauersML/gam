//! Row-wise (Khatri-Rao) Kronecker and tensor-product design operators,
//! split out of `matrix/mod.rs` by concern (#1145). Re-exported from
//! `matrix` so the public paths `crate::matrix::{RowwiseKroneckerOperator,
//! TensorProductDesignOperator, dense_rowwise_kronecker}` stay stable.

use super::*;

/// Rowwise-Kronecker design operator: represents the (n, p_cov × p_time) matrix
/// whose row i is the Kronecker product cov[i,:] ⊗ time[i,:].
///
/// This avoids materializing the full tensor product design, which at large-scale
/// scale can be tens of GB.
///
///   X[i, j*p_time + t] = cov[i, j] * time[i, t]
///
/// All matvec and Gram operations are performed in factored form.
#[derive(Clone)]
pub struct RowwiseKroneckerOperator {
    /// Covariate factor: (n, p_cov).
    pub cov: DesignMatrix,
    /// Time basis factor: (n, p_time).  Dense because B-spline bases are
    /// always dense (though banded — only ~4 nonzeros per row for degree 3).
    pub time_basis: Arc<Array2<f64>>,
    /// Cached dimensions.
    pub n: usize,
    pub p_cov: usize,
    pub p_time: usize,
}

/// Generic rowwise Kronecker operator for dense marginal designs.
///
/// Decode a flat index into per-dimension indices for a row-major tensor
/// with the given dimension sizes.  Writes into the provided `out` slice
/// to avoid allocation.
///
///   decode_multi_index(flat, &[3, 4], &mut out) → out = [flat / 4, flat % 4]
fn decode_multi_index(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for d in (0..dims.len()).rev() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
}

pub fn upper_triangle_pair_from_index(pair_idx: usize, n: usize) -> (usize, usize) {
    let span = 2 * n + 1;
    let discriminant = span * span - 8 * pair_idx;
    let row = ((span as f64 - (discriminant as f64).sqrt()) * 0.5) as usize;
    let row_start = row * (2 * n - row + 1) / 2;
    (row, row + pair_idx - row_start)
}

fn lower_triangle_pair_from_index(pair_idx: usize) -> (usize, usize) {
    let row = (((8 * pair_idx + 1) as f64).sqrt() as usize - 1) / 2;
    let row_start = row * (row + 1) / 2;
    (row, pair_idx - row_start)
}

/// Each row is the Kronecker product of the corresponding marginal rows:
/// `X[i, :] = B_1[i, :] ⊗ ... ⊗ B_d[i, :]`.
///
/// This keeps tensor-product terms operator-backed in the main model path so
/// fitting no longer requires an eager `n x prod(q_j)` realization.
pub struct TensorProductDesignOperator {
    marginals: Vec<Arc<Array2<f64>>>,
    n: usize,
    total_cols: usize,
}

impl TensorProductDesignOperator {
    pub fn new(marginals: Vec<Arc<Array2<f64>>>) -> Result<Self, String> {
        if marginals.is_empty() {
            return Err("TensorProductDesignOperator requires at least one marginal".to_string());
        }
        let n = marginals[0].nrows();
        let total_cols = marginals.iter().try_fold(1usize, |acc, marginal| {
            if marginal.nrows() != n {
                return Err(format!(
                    "TensorProductDesignOperator row mismatch: expected {n}, got {}",
                    marginal.nrows()
                ));
            }
            acc.checked_mul(marginal.ncols()).ok_or_else(|| {
                "TensorProductDesignOperator total column count overflow".to_string()
            })
        })?;
        Ok(Self {
            marginals,
            n,
            total_cols,
        })
    }

    /// Compute Xβ via column-wise BLAS matvecs across all n observations.
    ///
    /// β is conceptually a (q₁, q₂, …, qₖ) tensor.  We iterate over all
    /// "tail columns" (indices into dimensions 2..k), and for each:
    ///
    ///   1. Extract β_slice = β[:, t₂, …, tₖ]          — q₁-vector
    ///   2. contrib = B₁ · β_slice                       — ONE BLAS matvec, O(n·q₁)
    ///   3. contrib ⊙= B₂[:,t₂] ⊙ … ⊙ Bₖ[:,tₖ]        — k-1 elementwise O(n) passes
    ///   4. result += contrib
    ///
    /// Total: ∏_{j>1}qⱼ BLAS matvecs.  Same asymptotic cost as per-row scalar
    /// contraction, but each operation is a vectorized n-length pass with BLAS
    /// cache optimization.  Zero per-row allocation.
    fn apply_vectorized(&self, vector: &Array1<f64>) -> Array1<f64> {
        let d = self.marginals.len();
        let n = self.n;
        if d == 0 {
            return Array1::zeros(n);
        }
        let b0 = &self.marginals[0];
        let q0 = b0.ncols();
        if d == 1 {
            return fast_av(b0.as_ref(), vector);
        }

        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let intermediate_bytes = n * tail_total * std::mem::size_of::<f64>();

        if intermediate_bytes <= TENSOR_GEMM_MAX_INTERMEDIATE_BYTES {
            // ── GEMM path: one BLAS3 call for the B₁ contraction ────────
            //
            // Reshape β to (q₁, tail_total), compute B₁ · β_mat → (n, tail_total)
            // via a single GEMM.  Then elementwise-multiply each column by the
            // corresponding tail marginal products and row-sum.
            //
            // Zero-copy reshape: β is contiguous and q₁·tail_total = total_cols.
            let beta_view =
                ndarray::ArrayView2::from_shape((q0, tail_total), vector.as_slice().unwrap())
                    .expect("β reshape for GEMM");
            let temp = fast_ab(b0.as_ref(), &beta_view); // (n, tail_total)

            let mut out = Array1::<f64>::zeros(n);
            let mut tail_indices = vec![0usize; tail_dims.len()];
            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for i in 0..n {
                    let mut val = temp[[i, t_flat]];
                    for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                        val *= self.marginals[dim_idx + 1][[i, ti]];
                    }
                    out[i] += val;
                }
            }
            out
        } else {
            // ── GEMV fallback: one BLAS2 call per tail column ───────────
            let mut tail_indices = vec![0usize; tail_dims.len()];
            let mut out = Array1::<f64>::zeros(n);
            let mut beta_slice = Array1::<f64>::zeros(q0);
            let mut contrib = Array1::<f64>::zeros(n);

            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for j1 in 0..q0 {
                    beta_slice[j1] = vector[j1 * tail_total + t_flat];
                }
                fast_av_into(b0.as_ref(), &beta_slice, &mut contrib);
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let m = &self.marginals[dim_idx + 1];
                    for i in 0..n {
                        contrib[i] *= m[[i, ti]];
                    }
                }
                out += &contrib;
            }
            out
        }
    }

    /// Compute X'v via column-wise BLAS transpose matvecs across all n observations.
    ///
    /// For each tail column t = (t₂, …, tₖ):
    ///   1. scaled_v = v ⊙ B₂[:,t₂] ⊙ … ⊙ Bₖ[:,tₖ]   — elementwise O(n)
    ///   2. out[:, t] = B₁' · scaled_v                   — ONE BLAS transpose matvec
    ///
    /// Total: ∏_{j>1}qⱼ BLAS transpose matvecs.
    fn apply_transpose_vectorized(&self, vector: &Array1<f64>) -> Array1<f64> {
        let d = self.marginals.len();
        let n = self.n;
        if d == 0 {
            return Array1::zeros(self.total_cols);
        }
        let b0 = &self.marginals[0];
        let q0 = b0.ncols();
        if d == 1 {
            return fast_atv(b0.as_ref(), vector);
        }

        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let intermediate_bytes = n * tail_total * std::mem::size_of::<f64>();

        if intermediate_bytes <= TENSOR_GEMM_MAX_INTERMEDIATE_BYTES {
            // ── GEMM path: build W matrix, one BLAS3 call ───────────────
            //
            // W[i, t_flat] = v[i] · ∏_{d>1} Bᵈ[i, tᵈ]
            // Then B₁' · W → (q₁, tail_total) via one GEMM.
            let mut w_mat = Array2::<f64>::zeros((n, tail_total));
            let mut tail_indices = vec![0usize; tail_dims.len()];
            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for i in 0..n {
                    let mut val = vector[i];
                    for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                        val *= self.marginals[dim_idx + 1][[i, ti]];
                    }
                    w_mat[[i, t_flat]] = val;
                }
            }
            let result_mat = fast_atb(b0.as_ref(), &w_mat); // (q₁, tail_total)

            // Scatter from (q₁, tail_total) matrix into flat output.
            let mut out = Array1::<f64>::zeros(self.total_cols);
            for j1 in 0..q0 {
                for t_flat in 0..tail_total {
                    out[j1 * tail_total + t_flat] = result_mat[[j1, t_flat]];
                }
            }
            out
        } else {
            // ── GEMV fallback ───────────────────────────────────────────
            let mut tail_indices = vec![0usize; tail_dims.len()];
            let mut out = Array1::<f64>::zeros(self.total_cols);
            let mut scaled_v = Array1::<f64>::zeros(n);
            let mut col_result = Array1::<f64>::zeros(q0);

            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                scaled_v.assign(vector);
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let m = &self.marginals[dim_idx + 1];
                    for i in 0..n {
                        scaled_v[i] *= m[[i, ti]];
                    }
                }
                fast_atv_into(b0.as_ref(), &scaled_v, &mut col_result);
                for j1 in 0..q0 {
                    out[j1 * tail_total + t_flat] = col_result[j1];
                }
            }
            out
        }
    }
}

impl LinearOperator for TensorProductDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.total_cols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_vectorized(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_transpose_vectorized(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.n {
            return Err(format!(
                "TensorProductDesignOperator::diag_xtw_x: weights length {} != n {}",
                weights.len(),
                self.n
            ));
        }
        let d = self.marginals.len();
        if d == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        let n = self.n;
        let q0 = self.marginals[0].ncols();

        // ── Factored Gram computation ──────────────────────────────────
        //
        // Generalizes RowwiseKroneckerOperator's gamma approach to k factors.
        //
        // X'WX[multi_a, multi_b] =
        //   Σ_i w[i] · B₁[i,a₁]·B₂[i,a₂]·…·Bₖ[i,aₖ] · B₁[i,b₁]·B₂[i,b₂]·…·Bₖ[i,bₖ]
        //
        // Factor out B₁:
        //   = Σ_i (w[i] · B₂[i,a₂]·B₂[i,b₂] · … · Bₖ[i,aₖ]·Bₖ[i,bₖ]) · B₁[i,a₁]·B₁[i,b₁]
        //
        // For each tuple (a₂,b₂,…,aₖ,bₖ), form γ[i] = w[i]·∏_{d>1} Bd[i,ad]·Bd[i,bd],
        // then the (a₁,b₁) block = B₁'·diag(γ)·B₁  which is a q₁×q₁ gram.
        //
        // This avoids per-row allocation and computes many small BLAS grams
        // instead of one huge (∏qⱼ)×(∏qⱼ) outer product.

        let mut xtwx = Array2::<f64>::zeros((self.total_cols, self.total_cols));
        let b0 = &self.marginals[0];

        // Collect tail marginal dimensions.
        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();

        // Iterate over all (a_tail, b_tail) index pairs in a deterministic
        // task order. Each rayon task owns its gamma vector and q0×q0 block,
        // and the collected blocks are scattered sequentially in pair order so
        // reductions do not depend on worker scheduling.
        let tail_d = tail_dims.len();
        let pair_count = tail_total * (tail_total + 1) / 2;
        let blocks: Vec<(usize, usize, Array2<f64>)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (a_flat, b_flat) = upper_triangle_pair_from_index(pair_idx, tail_total);
                let mut a_indices = vec![0usize; tail_d];
                let mut b_indices = vec![0usize; tail_d];
                decode_multi_index(a_flat, &tail_dims, &mut a_indices);
                decode_multi_index(b_flat, &tail_dims, &mut b_indices);

                let mut gamma = Array1::<f64>::zeros(n);
                for i in 0..n {
                    let mut prod = weights[i].max(0.0);
                    if prod != 0.0 {
                        for dim_idx in 0..tail_d {
                            let m = &self.marginals[dim_idx + 1];
                            prod *= m[[i, a_indices[dim_idx]]] * m[[i, b_indices[dim_idx]]];
                            if prod == 0.0 {
                                break;
                            }
                        }
                    }
                    gamma[i] = prod;
                }

                let mut block = Array2::<f64>::zeros((q0, q0));
                stream_weighted_crossprod_into(
                    b0.as_ref(),
                    &gamma,
                    &mut block,
                    CrossprodStructure::Full,
                    CrossprodAccum::Replace,
                    effective_global_parallelism(),
                );
                (a_flat, b_flat, block)
            })
            .collect();

        for (a_flat, b_flat, block) in blocks {
            // Scatter block into the full xtwx.
            // Global column for (j₁, tail_flat) = j₁ * tail_total + tail_flat.
            for a1 in 0..q0 {
                let ga = a1 * tail_total + a_flat;
                for b1 in 0..q0 {
                    let gb = b1 * tail_total + b_flat;
                    xtwx[[ga, gb]] += block[[a1, b1]];
                    if a_flat != b_flat {
                        let ga_mirror = a1 * tail_total + b_flat;
                        let gb_mirror = b1 * tail_total + a_flat;
                        xtwx[[ga_mirror, gb_mirror]] += block[[a1, b1]];
                    }
                }
            }
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.n {
            return Err(format!(
                "TensorProductDesignOperator::diag_gram: weights length {} != n {}",
                weights.len(),
                self.n
            ));
        }
        // diag(X'WX)[j] = Σ_i w[i] · x_{ij}²
        // For tensor product: x_{i, j₁·tail+j_tail} = B₁[i,j₁] · ∏_{d>1} Bᵈ[i,jᵈ]
        // So: diag[j] = Σ_i w[i] · B₁[i,j₁]² · ∏_{d>1} Bᵈ[i,jᵈ]²
        //
        // O(n · ∏qⱼ) instead of O(n · (∏qⱼ)²) from the full gram.
        let d = self.marginals.len();
        if d == 0 {
            return Ok(Array1::zeros(0));
        }
        let mut diag = vec![0.0_f64; self.total_cols];
        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let q0 = self.marginals[0].ncols();
        let mut tail_indices = vec![0usize; tail_dims.len()];

        for t_flat in 0..tail_total {
            decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
            for i in 0..self.n {
                let wi = weights[i].max(0.0);
                if wi == 0.0 {
                    continue;
                }
                let mut tail_prod_sq = wi;
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let val = self.marginals[dim_idx + 1][[i, ti]];
                    tail_prod_sq *= val * val;
                    if tail_prod_sq == 0.0 {
                        break;
                    }
                }
                if tail_prod_sq == 0.0 {
                    continue;
                }
                for j1 in 0..q0 {
                    let b1 = self.marginals[0][[i, j1]];
                    diag[j1 * tail_total + t_flat] += tail_prod_sq * b1 * b1;
                }
            }
        }
        Ok(Array1::from_vec(diag))
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for TensorProductDesignOperator {
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        if middle.nrows() != self.total_cols || middle.ncols() != self.total_cols {
            return Err(format!(
                "TensorProductDesignOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.total_cols,
                self.total_cols
            ));
        }
        let mut out = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.try_row_chunk(start..end).map_err(|e| e.to_string())?;
            let chunk_m = fast_ab(&chunk, middle);
            for local in 0..(end - start) {
                out[start + local] = chunk.row(local).dot(&chunk_m.row(local)).max(0.0);
            }
        }
        Ok(out)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.total_cols {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "TensorProductDesignOperator::row_chunk_into shape mismatch",
            });
        }
        // Reuse two scratch buffers across all rows in the chunk instead of
        // allocating a fresh `Vec` per row (and a fresh `next` per marginal
        // per row). The Khatri-Rao contraction is written straight into the
        // contiguous `out` row, bit-identically: the same
        // `prefix * marginal[[row, col]]` products in the same order as the
        // sequential per-row materialization it replaces.
        let mut cur = Vec::<f64>::with_capacity(self.total_cols);
        let mut next = Vec::<f64>::with_capacity(self.total_cols);
        for (local_row, global_row) in rows.enumerate() {
            cur.clear();
            cur.push(1.0);
            for marginal in &self.marginals {
                let q = marginal.ncols();
                next.clear();
                next.resize(cur.len() * q, 0.0);
                for (prefix_idx, &prefix) in cur.iter().enumerate() {
                    let off = prefix_idx * q;
                    for col in 0..q {
                        next[off + col] = prefix * marginal[[global_row, col]];
                    }
                }
                std::mem::swap(&mut cur, &mut next);
            }
            let mut out_view = out.row_mut(local_row);
            let out_row = out_view
                .as_slice_mut()
                .expect("design chunk row is contiguous in C-major Array2");
            out_row.copy_from_slice(&cur);
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        self.try_row_chunk(0..self.n)
            .expect("TensorProductDesignOperator row_chunk_into is total")
    }
}

/// Row-wise (Khatri–Rao) Kronecker product of two dense matrices sharing the
/// same number of rows: `out[i, j * pb + k] = a[i, j] * b[i, k]`. Parallel
/// across row chunks; short-circuits on zeros in `a` to skip the inner
/// `b`-column loop when the left factor is structurally sparse.
///
/// Canonical home for the dense path; the operator-backed streaming variant
/// lives on [`RowwiseKroneckerOperator`]. Survival families that need a
/// `DesignMatrix`-typed output still use the operator wrapper because the
/// product would otherwise materialize an `n × (p_cov · p_time)` dense block.
pub fn dense_rowwise_kronecker(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    assert_eq!(
        a.nrows(),
        b.nrows(),
        "dense_rowwise_kronecker requires matching row counts: a={}, b={}",
        a.nrows(),
        b.nrows()
    );
    let n = a.nrows();
    let pa = a.ncols();
    let pb = b.ncols();
    let mut out = Array2::<f64>::zeros((n, pa * pb));
    if n == 0 || pa == 0 || pb == 0 {
        return out;
    }
    const CHUNK: usize = 1024;
    out.axis_chunks_iter_mut(Axis(0), CHUNK)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let start = chunk_idx * CHUNK;
            let rows = out_chunk.nrows();
            for local in 0..rows {
                let i = start + local;
                for j in 0..pa {
                    let a_ij = a[[i, j]];
                    if a_ij == 0.0 {
                        continue;
                    }
                    let off = j * pb;
                    for k in 0..pb {
                        out_chunk[[local, off + k]] = a_ij * b[[i, k]];
                    }
                }
            }
        });
    out
}

impl RowwiseKroneckerOperator {
    pub fn new(cov: DesignMatrix, time_basis: Arc<Array2<f64>>) -> Result<Self, String> {
        let n = cov.nrows();
        if time_basis.nrows() != n {
            return Err(format!(
                "RowwiseKroneckerOperator: cov has {} rows but time_basis has {}",
                n,
                time_basis.nrows()
            ));
        }
        let p_cov = cov.ncols();
        let p_time = time_basis.ncols();
        Ok(Self {
            cov,
            time_basis,
            n,
            p_cov,
            p_time,
        })
    }
}

impl LinearOperator for RowwiseKroneckerOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p_cov * self.p_time
    }

    /// X β where β is reshaped as (p_cov, p_time):
    ///   result[i] = Σⱼ cov[i,j] * Σₜ time[i,t] * β[j*p_time + t]
    ///
    /// Computed via p_time calls to cov.apply() to stay sparse-native:
    ///   For each t: result += time[:,t] ⊙ cov · β[:,t]
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let n = self.n;
        let time = self.time_basis.as_ref();
        let mut out = Array1::<f64>::zeros(n);
        // For each time column t, extract β[:,t] = [β[0*pt+t], β[1*pt+t], ...],
        // compute cov · β[:,t], then weight by time[:,t].
        let mut beta_slice = Array1::<f64>::zeros(p_cov);
        for t in 0..p_time {
            for j in 0..p_cov {
                beta_slice[j] = vector[j * p_time + t];
            }
            let cov_beta_t = self.cov.matrixvectormultiply(&beta_slice);
            let time_col = time.column(t);
            ndarray::Zip::from(&mut out)
                .and(&cov_beta_t)
                .and(&time_col)
                .par_for_each(|o, &cb, &tt| *o += cb * tt);
        }
        out
    }

    /// X' v where the result is (p_cov * p_time):
    ///   result[j*p_time + t] = Σᵢ v[i] * cov[i,j] * time[i,t]
    ///
    /// Computed via p_time calls to cov.apply_transpose() to stay sparse-native:
    ///   For each t: result[:,t] = cov' · (v ⊙ time[:,t])
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let n = self.n;
        let time = self.time_basis.as_ref();
        let mut out = Array1::<f64>::zeros(p_cov * p_time);
        // For each time column t, form w_t = v ⊙ time[:,t], compute cov' · w_t.
        let mut w_t = Array1::<f64>::zeros(n);
        for t in 0..p_time {
            let time_col = time.column(t);
            ndarray::Zip::from(&mut w_t)
                .and(vector)
                .and(&time_col)
                .par_for_each(|o, &v, &tt| *o = v * tt);
            let col_t = self.cov.transpose_vector_multiply(&w_t);
            for j in 0..p_cov {
                out[j * p_time + t] = col_t[j];
            }
        }
        out
    }

    /// X'WX via factored Gram computation.
    ///
    /// (X'WX)[j1*pt+t1, j2*pt+t2]
    ///   = Σᵢ w[i] * cov[i,j1] * time[i,t1] * cov[i,j2] * time[i,t2]
    ///   = Σᵢ (w[i] * cov[i,j1] * cov[i,j2]) * (time[i,t1] * time[i,t2])
    ///
    /// For each (t1, t2) pair, we form the n-vector
    ///   γ_{t1,t2}[i] = w[i] * time[i,t1] * time[i,t2]
    /// and then the (p_cov, p_cov) block is cov' diag(γ_{t1,t2}) cov.
    ///
    /// Cost: O(n * p_time² * p_cov²) vs O(n * (p_cov*p_time)²) for the naive path.
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let p_total = p_cov * p_time;
        if weights.len() != n {
            return Err(format!(
                "RowwiseKroneckerOperator::diag_xtw_x: weights length {} != n {}",
                weights.len(),
                n
            ));
        }
        let mut xtwx = Array2::<f64>::zeros((p_total, p_total));
        let time = self.time_basis.as_ref();

        // For each time-basis pair (t1, t2), the (p_cov, p_cov) block is
        //   cov' diag(γ_{t1,t2}) cov
        // where γ[i] = w[i] * time[i,t1] * time[i,t2].  Blocks are computed
        // in rayon tasks with task-local gamma arrays, then scattered in
        // lexicographic pair order for deterministic reductions.
        let pair_count = p_time * (p_time + 1) / 2;
        let blocks: Result<Vec<(usize, usize, Array2<f64>)>, String> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (t1, t2) = lower_triangle_pair_from_index(pair_idx);
                let time_t1 = time.column(t1);
                let time_t2 = time.column(t2);
                let mut gamma = Array1::<f64>::zeros(n);
                ndarray::Zip::from(&mut gamma)
                    .and(weights)
                    .and(&time_t1)
                    .and(&time_t2)
                    .for_each(|g, &w, &a, &b| *g = w.max(0.0) * a * b);
                self.cov
                    .xt_diag_x_signed_op(SignedWeightsView::from_array(&gamma))
                    .map(|block| (t1, t2, block))
            })
            .collect();
        for (t1, t2, block) in blocks? {
            // Scatter block into xtwx for both (t1, t2) and (t2, t1).
            for j1 in 0..p_cov {
                for j2 in 0..p_cov {
                    xtwx[[j1 * p_time + t1, j2 * p_time + t2]] = block[[j1, j2]];
                    if t1 != t2 {
                        xtwx[[j1 * p_time + t2, j2 * p_time + t1]] = block[[j1, j2]];
                    }
                }
            }
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        if weights.len() != n {
            return Err(format!(
                "RowwiseKroneckerOperator::diag_gram: weights {} != n {}",
                weights.len(),
                n
            ));
        }
        let time = self.time_basis.as_ref();
        // diag(X'WX)[j*pt+t] = Σᵢ w[i] * cov[i,j]² * time[i,t]²
        // Use cov.diag_gram(w ⊙ time[:,t]²) which stays sparse-native.
        let mut out = Array1::<f64>::zeros(p_cov * p_time);
        let mut gamma = Array1::<f64>::zeros(n);
        for t in 0..p_time {
            let time_col = time.column(t);
            ndarray::Zip::from(&mut gamma)
                .and(weights)
                .and(&time_col)
                .par_for_each(|g, &w, &tt| *g = w.max(0.0) * tt * tt);
            let cov_diag = <DesignMatrix as LinearOperator>::diag_gram(&self.cov, &gamma)?;
            for j in 0..p_cov {
                out[j * p_time + t] = cov_diag[j];
            }
        }
        Ok(out)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for RowwiseKroneckerOperator {
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let p_total = self.p_cov * self.p_time;
        if middle.nrows() != p_total || middle.ncols() != p_total {
            return Err(format!(
                "RowwiseKroneckerOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                p_total,
                p_total
            ));
        }
        let mut out = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.try_row_chunk(start..end).map_err(|e| e.to_string())?;
            let chunk_m = fast_ab(&chunk, middle);
            for local in 0..(end - start) {
                out[start + local] = chunk.row(local).dot(&chunk_m.row(local)).max(0.0);
            }
        }
        Ok(out)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let chunk_rows = rows.end - rows.start;
        if out.nrows() != chunk_rows || out.ncols() != p_cov * p_time {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "RowwiseKroneckerOperator::row_chunk_into shape mismatch",
            });
        }
        out.fill(0.0);
        let cov_chunk = self.cov.try_row_chunk(rows.clone())?;
        let time = self.time_basis.as_ref();
        for local in 0..chunk_rows {
            let global = rows.start + local;
            for j in 0..p_cov {
                let cij = cov_chunk[[local, j]];
                if cij == 0.0 {
                    continue;
                }
                for t in 0..p_time {
                    out[[local, j * p_time + t]] = cij * time[[global, t]];
                }
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let bytes = n
            .saturating_mul(p_cov)
            .saturating_mul(p_time)
            .saturating_mul(std::mem::size_of::<f64>());
        // SAFETY: `RowwiseKroneckerOperator` is explicitly an operator-only
        // representation: the wrapping `LazyOperator::to_dense` contract
        // forbids dense materialization for large-scale n*p_cov*p_time
        // tensors. Any caller that reaches this point bypassed the
        // operator-aware dispatch and would otherwise silently allocate a
        // matrix sized to crash the process.
        // SAFETY: operator-only type; reaching to_dense means dispatch bypassed the operator-aware path.
        std::panic::panic_any(format!(
            "RowwiseKroneckerOperator must remain operator-backed; refused persistent n x p_covariate x p_time materialization (n={n}, p_covariate={p_cov}, p_time={p_time}, dense={:.1} MiB)",
            bytes as f64 / (1024.0 * 1024.0),
        ));
    }
}
