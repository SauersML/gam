//! Symmetric-matrix container and `XᵀWX` Gram assembly, split out of
//! `matrix/mod.rs` by concern (#1145). Re-exported from `matrix` so the
//! public paths `crate::matrix::{SymmetricMatrix, xt_diag_x_*, ...}` stay stable.

use super::*;

/// A unified representation of a symmetric matrix, typically an assembled Hessian.
#[derive(Clone, Debug)]
pub enum SymmetricMatrix {
    Dense(Array2<f64>),
    Sparse(faer::sparse::SparseColMat<usize, f64>),
}

impl SymmetricMatrix {
    pub fn as_dense(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(mat) => Some(mat),
            Self::Sparse(_) => None,
        }
    }

    pub fn as_sparse(&self) -> Option<&faer::sparse::SparseColMat<usize, f64>> {
        match self {
            Self::Sparse(mat) => Some(mat),
            Self::Dense(_) => None,
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(mat) => mat.clone(),
            Self::Sparse(mat) => {
                let mut out = Array2::<f64>::zeros((mat.nrows(), mat.ncols()));
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        out[[row, col]] += value;
                        if row != col {
                            out[[col, row]] += value;
                        }
                    }
                }
                out
            }
        }
    }

    /// Materialize this exact symmetric matrix as a dense `Array2` and validate
    /// that it is suitable for dense linear solves.
    ///
    /// This does not approximate or synthesize missing entries: sparse matrices
    /// are expanded from their stored exact upper-triangular representation, and
    /// dense matrices are cloned as-is. Callers that require explicit Hessians
    /// (for example ALO solve setup) should use this instead of a blind
    /// `to_dense()` so shape and derivative-validity failures are reported at
    /// the export boundary.
    pub fn try_to_dense_exact(&self, context: &str) -> Result<Array2<f64>, String> {
        if self.nrows() != self.ncols() {
            return Err(format!(
                "{context}: exact symmetric matrix must be square, got {}x{}",
                self.nrows(),
                self.ncols()
            ));
        }

        let dense = self.to_dense();
        if dense.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{context}: exact dense materialization contains non-finite entries"
            ));
        }
        Ok(dense)
    }

    pub fn factorize(&self) -> Result<Box<dyn FactorizedSystem>, String> {
        match self {
            Self::Dense(mat) => {
                let factor = crate::utils::StableSolver::new("unnamed")
                    .factorize(mat)
                    .map_err(|e| format!("Dense SymmetricMatrix factorization failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
            Self::Sparse(mat) => {
                let factor = crate::sparse_exact::factorize_sparse_spd(mat)
                    .map_err(|e| format!("Sparse SymmetricMatrix factorization failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
        }
    }

    pub fn add(&self, other: &SymmetricMatrix) -> Result<Self, String> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(format!(
                "SymmetricMatrix::add shape mismatch: lhs {}x{}, rhs {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            ));
        }
        match (self, other) {
            (Self::Dense(a), Self::Dense(b)) => Ok(Self::Dense(a + b)),
            (Self::Dense(a), Self::Sparse(_)) => {
                let b_dense = other.to_dense();
                Ok(Self::Dense(a + &b_dense))
            }
            (Self::Sparse(_), Self::Dense(b)) => {
                let a_dense = self.to_dense();
                Ok(Self::Dense(&a_dense + b))
            }
            (Self::Sparse(a), Self::Sparse(b)) => {
                Ok(Self::Sparse(add_sparse_symmetric_upper(a, b)?))
            }
        }
    }

    pub fn add_dense(&self, other: &Array2<f64>) -> Result<Self, String> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(format!(
                "SymmetricMatrix::add_dense shape mismatch: lhs {}x{}, rhs {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            ));
        }
        match self {
            Self::Dense(mat) => {
                let mut out = mat.clone();
                out += other;
                Ok(Self::Dense(out))
            }
            Self::Sparse(mat) => {
                let other_sparse = crate::sparse_exact::dense_to_sparse_symmetric_upper(other, 0.0)
                    .map_err(|e| format!("SymmetricMatrix::add_dense failed: {e}"))?;
                Ok(Self::Sparse(add_sparse_symmetric_upper(
                    mat,
                    &other_sparse,
                )?))
            }
        }
    }

    pub fn addridge(&self, ridge: f64) -> Result<Self, String> {
        if ridge == 0.0 {
            return Ok(self.clone());
        }
        match self {
            Self::Dense(mat) => {
                let mut out = mat.clone();
                for i in 0..out.nrows() {
                    out[[i, i]] += ridge;
                }
                Ok(Self::Dense(out))
            }
            Self::Sparse(mat) => {
                let n = mat.nrows();
                let mut trip = Vec::with_capacity(n);
                for i in 0..n {
                    trip.push(Triplet::new(i, i, ridge));
                }
                let diagonal = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip)
                    .map_err(|_| {
                        "SymmetricMatrix::addridge failed to assemble sparse diagonal".to_string()
                    })?;
                Ok(Self::Sparse(add_sparse_symmetric_upper(mat, &diagonal)?))
            }
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::Sparse(m) => m.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(m) => m.ncols(),
            Self::Sparse(m) => m.ncols(),
        }
    }

    pub fn dot(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => fast_av(mat, rhs),
            Self::Sparse(mat) => {
                let mut out = Array1::<f64>::zeros(mat.nrows());
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let rhs_j = rhs[col];
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        out[row] += value * rhs_j;
                        if row != col {
                            out[col] += value * rhs[row];
                        }
                    }
                }
                out
            }
        }
    }

    /// Maximum absolute value on the diagonal.
    pub fn max_abs_diag(&self) -> f64 {
        match self {
            Self::Dense(mat) => {
                let n = mat.nrows().min(mat.ncols());
                (0..n).map(|i| mat[[i, i]].abs()).fold(0.0_f64, f64::max)
            }
            Self::Sparse(mat) => {
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                let mut max_val = 0.0_f64;
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        if row_idx[idx] == col {
                            max_val = max_val.max(values[idx].abs());
                        }
                    }
                }
                max_val
            }
        }
    }

    /// Multiply on the right by a dense matrix: self * rhs.
    /// Returns a dense Array2.
    pub fn dot_matrix(&self, rhs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(mat) => fast_ab(mat, rhs),
            Self::Sparse(mat) => {
                let n = mat.nrows();
                let k = rhs.ncols();
                let mut out = Array2::<f64>::zeros((n, k));
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        for c in 0..k {
                            out[[row, c]] += value * rhs[[col, c]];
                            if row != col {
                                out[[col, c]] += value * rhs[[row, c]];
                            }
                        }
                    }
                }
                out
            }
        }
    }

    /// Left-multiply by a dense matrix: lhs * self.
    /// Returns a dense Array2.
    pub fn left_dot_matrix(&self, lhs: &Array2<f64>) -> Array2<f64> {
        // (lhs * S)^T = S^T * lhs^T = S * lhs^T  (S is symmetric)
        // So lhs * S = (S * lhs^T)^T
        let lhs_t = lhs.t().to_owned();
        let result_t = self.dot_matrix(&lhs_t);
        result_t.t().to_owned()
    }
}

/// Build `XᵀWX` from a design + signed weights, returning a symmetric matrix.
///
/// This is the observed-Hessian / non-canonical-link route: the input `diag`
/// may contain negative entries when the working curvature is not guaranteed
/// PSD (e.g. binomial + cloglog, Gamma + identity, any IRLS step that uses
/// the true Hessian rather than the Fisher information). All internal kernels
/// — `stream_weighted_crossprod_into`, `streaming_sparse_csc_xt_diag_x`, and
/// the sparse-row accumulator — preserve the sign of the weights; only the
/// PSD-precondition kernels in this module (`sparse_csr_weighted_xtwx_rows`,
/// `weighted_crossprod_dense_rows`, `dense_diag_gram_view`) clip / assert
/// nonneg, and none of them is reachable from this entry.
///
/// Callers in PIRLS should select `_signed` for observed-Hessian / Newton
/// curvature assembly and `_psd` for Fisher-scoring updates where the working
/// weights are guaranteed nonneg. The sign character is now encoded in the
/// argument types: `xt_diag_x_signed` takes a `SignedWeightsView<'_>` (free
/// construction), and `xt_diag_x_psd` takes a `PsdWeightsView<'_>` (one-time
/// `try_new` scan at the call site).
pub fn xt_diag_x_signed(
    design: &DesignMatrix,
    diag: SignedWeightsView<'_>,
) -> Result<SymmetricMatrix, String> {
    xt_diag_x_symmetric(design, &diag.view().to_owned())
}

/// In-place symmetrization of a square dense matrix: replace each
/// off-diagonal pair `(m[i,j], m[j,i])` with their average so the result is
/// exactly symmetric (the diagonal is untouched).
///
/// Canonical single source of truth for the "average the transpose" cleanup
/// that every Hessian/penalty assembly applies to kill the small asymmetry
/// left by floating-point accumulation order.
pub fn symmetrize_in_place(matrix: &mut Array2<f64>) {
    let p = matrix.nrows();
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = v;
            matrix[[j, i]] = v;
        }
    }
}

/// Allocating variant of [`symmetrize_in_place`]: return `0.5 * (M + Mᵀ)`.
///
/// Same canonical "average the transpose" cleanup, returning a fresh matrix
/// for callers that must keep the original intact.
pub fn symmetrize(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t()) * 0.5
}

/// PSD-precondition Gram: `XᵀWX` with `w ≥ 0`.
///
/// Use for Fisher-scoring / canonical-link IRLS, where the working weights are
/// guaranteed nonneg by construction. The `w ≥ 0` precondition is discharged
/// at the `PsdWeightsView::try_new` constructor; the kernel below performs no
/// further scan. Numeric path is identical to `xt_diag_x_signed`.
pub fn xt_diag_x_psd(
    design: &DesignMatrix,
    diag: PsdWeightsView<'_>,
) -> Result<SymmetricMatrix, String> {
    xt_diag_x_symmetric(design, &diag.view().to_owned())
}

pub fn xt_diag_x_symmetric(
    design: &DesignMatrix,
    diag: &Array1<f64>,
) -> Result<SymmetricMatrix, String> {
    if design.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_x_symmetric row mismatch: design has {} rows but diag has {} entries",
            design.nrows(),
            diag.len()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(SymmetricMatrix::Dense(x.diag_xtw_x(diag)?)),
        DesignMatrix::Sparse(xs) => {
            // The macOS sample profile of matern60 fingered this function as
            // 58% of main-thread cycles, all in
            // `SparseHessianAccumulator::from_multi_csr → BTreeSet::insert`:
            // every PIRLS Newton iteration was rebuilding the symbolic upper
            // pattern from scratch via O(nnz²·log) BTreeSet insertions, even
            // though the symbolic pattern depends only on X (not on the
            // weights). For a Matern radial design at n=10K it dominates over
            // the actual numeric assembly.
            //
            // Two regimes:
            //   (A) Numerically dense — Matern / Duchon: every column has a
            //       nonzero, so XᵀWX fills in completely. Use the BLAS path
            //       when policy permits materializing the dense design, or a
            //       bounded CSC-to-dense-row-chunk BLAS path when it does not.
            //       Both avoid the symbolic sparse-Hessian build and scalar
            //       accumulation.
            //   (B) Genuinely sparse — B-spline / banded: per-row work is
            //       O(nnz_row²) at small constant factor; the sparse
            //       row-parallel accumulator is the right tool.
            // Heuristic: avg_nnz_per_row · 4 ≥ p picks (A). The sparse-native
            // PIRLS path upstream already routes truly-sparse Hessians around
            // this function, so (A) is the dominant call site we have today.
            let n = xs.nrows();
            let p = xs.ncols();
            let nnz_x = xs.val().len();
            let avg_nnz_row = if n > 0 { nnz_x / n } else { p };
            let dense_regime = 4 * avg_nnz_row >= p;
            if dense_regime {
                let mut xtwx = Array2::<f64>::zeros((p, p));
                let dense_bytes =
                    checked_dense_nbytes(n, p, "xt_diag_x_symmetric dense sparse route")?;
                if dense_bytes <= MAX_SPARSE_TO_DENSE_BYTES {
                    let xd = xs.try_to_dense_arc("xt_diag_x_symmetric dense sparse route")?;
                    stream_weighted_crossprod_into(
                        xd.as_ref(),
                        diag,
                        &mut xtwx,
                        CrossprodStructure::Full,
                        CrossprodAccum::Replace,
                        effective_global_parallelism(),
                    );
                } else {
                    let (symbolic, values) = xs.parts();
                    streaming_sparse_csc_xt_diag_x(
                        symbolic.col_ptr(),
                        symbolic.row_idx(),
                        values,
                        n,
                        p,
                        diag.view(),
                        &mut xtwx,
                    );
                }
                return Ok(SymmetricMatrix::Dense(xtwx));
            }
            // Genuinely-sparse fallback: row-parallel accumulator that
            // shares the symbolic pattern via Arc, so the BTreeSet build
            // happens once and the values buffers are zero-init per chunk.
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let csr = xs
                .to_csr_arc()
                .ok_or_else(|| "xt_diag_x_symmetric: failed to obtain CSR view".to_string())?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            let acc_template = SparseHessianAccumulator::from_single_csr(&csr, p);
            let n_threads = rayon::current_num_threads().max(1);
            let target_chunks = (n_threads * 16).max(n_threads);
            let chunk_rows = (n / target_chunks).max(256).min(n.max(1));
            let chunk_starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
            let mut local_accs: Vec<SparseHessianAccumulator> = chunk_starts
                .into_par_iter()
                .map(|start| {
                    let end = (start + chunk_rows).min(n);
                    let mut local = acc_template.empty_clone();
                    for i in start..end {
                        let wi = diag[i];
                        if wi == 0.0 {
                            continue;
                        }
                        let r_start = row_ptr[i];
                        let r_end = row_ptr[i + 1];
                        for a_ptr in r_start..r_end {
                            let a = col_idx[a_ptr];
                            let wxa = wi * vals[a_ptr];
                            local.add_upper(a, a, wxa * vals[a_ptr]);
                            for b_ptr in (a_ptr + 1)..r_end {
                                let b = col_idx[b_ptr];
                                local.add_upper(a, b, wxa * vals[b_ptr]);
                            }
                        }
                    }
                    local
                })
                .collect();
            let mut acc = if let Some(first) = local_accs.pop() {
                first
            } else {
                acc_template.empty_clone()
            };
            for other in local_accs.into_iter() {
                acc.add_values(&other.values);
            }
            Ok(SymmetricMatrix::Sparse(acc.into_sparse_col_mat()))
        }
    }
}

fn add_sparse_symmetric_upper(
    lhs: &SparseColMat<usize, f64>,
    rhs: &SparseColMat<usize, f64>,
) -> Result<SparseColMat<usize, f64>, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(format!(
            "add_sparse_symmetric_upper shape mismatch: lhs {}x{}, rhs {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    let mut upper = BTreeMap::<(usize, usize), f64>::new();
    for matrix in [lhs, rhs] {
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                let row = row_idx[idx];
                let key = if row <= col { (row, col) } else { (col, row) };
                *upper.entry(key).or_insert(0.0) += values[idx];
            }
        }
    }
    let triplets: Vec<_> = upper
        .into_iter()
        .filter_map(|((row, col), value)| (value != 0.0).then_some(Triplet::new(row, col, value)))
        .collect();
    SparseColMat::try_new_from_triplets(lhs.nrows(), lhs.ncols(), &triplets)
        .map_err(|_| "add_sparse_symmetric_upper failed to assemble CSC".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn dense2x2() -> SymmetricMatrix {
        SymmetricMatrix::Dense(array![[1.0_f64, 2.0], [2.0, 4.0]])
    }

    // ── variant dispatch ──────────────────────────────────────────────────────

    #[test]
    fn as_dense_returns_some_for_dense_variant() {
        let m = dense2x2();
        assert!(m.as_dense().is_some());
        assert!(m.as_sparse().is_none());
    }

    // ── nrows / ncols ─────────────────────────────────────────────────────────

    #[test]
    fn nrows_and_ncols_match_inner_array() {
        let m = dense2x2();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    }

    // ── to_dense ──────────────────────────────────────────────────────────────

    #[test]
    fn to_dense_for_dense_variant_is_clone() {
        let m = dense2x2();
        let d = m.to_dense();
        assert_eq!(d[[0, 0]], 1.0);
        assert_eq!(d[[0, 1]], 2.0);
        assert_eq!(d[[1, 1]], 4.0);
    }

    // ── try_to_dense_exact ────────────────────────────────────────────────────

    #[test]
    fn try_to_dense_exact_ok_for_square_finite() {
        let m = dense2x2();
        assert!(m.try_to_dense_exact("ctx").is_ok());
    }

    #[test]
    fn try_to_dense_exact_err_for_nan_entry() {
        let m = SymmetricMatrix::Dense(array![[f64::NAN, 0.0], [0.0, 1.0]]);
        let err = m.try_to_dense_exact("nantest").unwrap_err();
        assert!(err.contains("nantest"), "error should mention context: {err}");
    }

    // ── add ───────────────────────────────────────────────────────────────────

    #[test]
    fn add_dense_dense_is_elementwise_sum() {
        let a = dense2x2();
        let b = SymmetricMatrix::Dense(array![[3.0_f64, 0.0], [0.0, 1.0]]);
        let c = a.add(&b).unwrap().to_dense();
        assert_eq!(c[[0, 0]], 4.0);
        assert_eq!(c[[1, 1]], 5.0);
    }

    #[test]
    fn add_shape_mismatch_is_error() {
        let a = dense2x2();
        let b = SymmetricMatrix::Dense(array![[1.0_f64]]);
        assert!(a.add(&b).is_err());
    }

    // ── addridge ──────────────────────────────────────────────────────────────

    #[test]
    fn addridge_zero_returns_clone() {
        let m = dense2x2();
        let r = m.addridge(0.0).unwrap().to_dense();
        assert_eq!(r[[0, 0]], 1.0);
        assert_eq!(r[[1, 1]], 4.0);
    }

    #[test]
    fn addridge_nonzero_adds_to_diagonal_only() {
        let m = dense2x2();
        let r = m.addridge(10.0).unwrap().to_dense();
        assert_eq!(r[[0, 0]], 11.0);
        assert_eq!(r[[0, 1]], 2.0); // off-diagonal unchanged
        assert_eq!(r[[1, 1]], 14.0);
    }

    // ── dot ───────────────────────────────────────────────────────────────────

    #[test]
    fn dot_identity_times_vector_is_vector() {
        let m = SymmetricMatrix::Dense(ndarray::Array2::eye(3));
        let x = array![1.0_f64, 2.0, 3.0];
        let y = m.dot(&x);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 3.0);
    }

    #[test]
    fn dot_known_2x2_result() {
        // A = [[1, 2], [2, 4]], x = [1, 1] → Ax = [3, 6]
        let m = dense2x2();
        let x = array![1.0_f64, 1.0];
        let y = m.dot(&x);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 6.0).abs() < 1e-14);
    }

    // ── max_abs_diag ──────────────────────────────────────────────────────────

    #[test]
    fn max_abs_diag_finds_largest_diagonal() {
        // A = [[1, 2], [2, 4]]: diag = {1, 4}, max_abs = 4
        let m = dense2x2();
        assert_eq!(m.max_abs_diag(), 4.0);
    }

    #[test]
    fn max_abs_diag_with_negative_diagonal_entry() {
        let m = SymmetricMatrix::Dense(array![[-5.0_f64, 0.0], [0.0, 3.0]]);
        assert_eq!(m.max_abs_diag(), 5.0);
    }
}
