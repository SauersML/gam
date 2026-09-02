//! Symmetric-matrix container and `XᵀWX` Gram assembly, split out of
//! `matrix/mod.rs` by concern (#1145). Re-exported from `matrix` so the
//! public paths `crate::matrix::{SymmetricMatrix, xt_diag_x_*, ...}` stay stable.

use super::*;
use crate::faer_ndarray::FaerCholesky;

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
                let factor = crate::utils::StableSolver::new()
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

    /// Strict factorization for covariance and other SPD-only estimands.
    ///
    /// No LDLT/LBLT route and no diagonal jitter is admitted: dense matrices
    /// must pass an unperturbed Cholesky factorization, while sparse matrices
    /// use the existing exact sparse-SPD factorization.
    pub fn factorize_spd(&self) -> Result<Box<dyn FactorizedSystem>, String> {
        match self {
            Self::Dense(matrix) => {
                crate::utils::validate_finite_symmetric_matrix(
                    matrix,
                    "Dense SymmetricMatrix strict SPD factorization",
                )
                .map_err(|error| error.to_string())?;
                matrix
                    .cholesky(faer::Side::Lower)
                    .map(|factor| Box::new(factor) as Box<dyn FactorizedSystem>)
                    .map_err(|error| {
                        format!("Dense SymmetricMatrix strict SPD factorization failed: {error}")
                    })
            }
            Self::Sparse(matrix) => crate::sparse_exact::factorize_sparse_spd_strict(matrix)
                .map(|factor| Box::new(factor) as Box<dyn FactorizedSystem>)
                .map_err(|error| {
                    format!("Sparse SymmetricMatrix strict SPD factorization failed: {error:?}")
                }),
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

    /// Exact maximum absolute stored matrix entry.
    ///
    /// Sparse matrices store one triangle of a symmetric matrix, so scanning
    /// their stored values is also the max-entry norm of the full matrix.
    pub fn max_abs_entry(&self) -> f64 {
        match self {
            Self::Dense(matrix) => matrix.iter().copied().map(f64::abs).fold(0.0_f64, f64::max),
            Self::Sparse(matrix) => {
                let (_, values) = matrix.parts();
                values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max)
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
/// argument types: `xt_diag_x_signed` takes a `FiniteSignedWeightsView<'_>` and
/// `xt_diag_x_psd` takes a `PsdWeightsView<'_>`; both perform a deterministic
/// one-time certificate at their construction site.
pub fn xt_diag_x_signed(
    design: &DesignMatrix,
    diag: FiniteSignedWeightsView<'_>,
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
///
/// The input must be square: the scan is driven by `nrows()` and indexes
/// `[[j, i]]` for `j < i`, so a wider-than-tall matrix would be read outside
/// its own column range. Every caller symmetrizes a Gram, a Hessian, a penalty
/// or a covariance, all square by construction.
///
/// This function was `pub` and carried the sentence above while having no
/// caller outside this crate; four crates (`gam-solve` twice, `gam-inference`,
/// `gam-terms`) had each re-derived the same loop as a private `fn`. The
/// bodies agreed, so nothing was numerically wrong -- but "single source of
/// truth" was a claim about a state that did not hold, and any change made
/// here would have reached none of them.
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

/// The 2-norm of the skew part `K = (M − Mᵀ)/2` that
/// [`symmetrize_in_place`] is about to throw away.
///
/// # Why this is worth returning rather than discarding (#2748)
///
/// A Hessian is symmetric — Clairaut, for any twice-continuously-differentiable
/// criterion. So for an assembled Hessian the skew part is **exactly zero in
/// exact arithmetic**, whatever the data, the family or the point: whatever
/// survives is assembly error and nothing else. And because `M[i,j]` and
/// `M[j,i]` are separate accumulations of the same mixed partial — different
/// summation orders, different intermediate cancellations, and in an implicit
/// -function assembly different contraction routes through the same inner
/// solve — their difference samples the whole assembly error, not just its
/// last-bit rounding.
///
/// By Weyl, an error `δM` moves every eigenvalue by at most `‖δM‖₂`; the skew
/// part IS the skew part of `δM` (a symmetric matrix has none), so `‖K‖₂` is a
/// **certified lower bound on `‖δM‖₂`** — a measured curvature resolution in
/// the sense of [`crate::curvature_resolution`], obtained for free at a site
/// that was already computing `(M + Mᵀ)/2`.
///
/// `‖K‖₂` is `√(λ_max(KᵀK))`; for a real skew-symmetric `K` that is the largest
/// singular value. Computed here through the symmetric `KᵀK` rather than an
/// SVD because these matrices are the ρ-Hessian's size (the smoothing-parameter
/// count), where the cube is free.
///
/// Returns `0.0` for a non-square or non-finite input: an absent measurement
/// must not become a large one.
pub fn symmetrization_defect_2norm(matrix: &Array2<f64>) -> f64 {
    use crate::faer_ndarray::FaerEigh;

    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n || matrix.iter().any(|value| !value.is_finite()) {
        return 0.0;
    }
    let mut skew = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            let half_difference = 0.5 * (matrix[[i, j]] - matrix[[j, i]]);
            skew[[i, j]] = half_difference;
            skew[[j, i]] = -half_difference;
        }
    }
    // `KᵀK = −K²` is symmetric PSD; its largest eigenvalue is `‖K‖₂²`.
    let gram = skew.t().dot(&skew);
    match gram.eigh(faer::Side::Lower) {
        Ok((eigenvalues, _)) => eigenvalues
            .iter()
            .fold(0.0_f64, |accumulated, value| accumulated.max(*value))
            .max(0.0)
            .sqrt(),
        Err(_) => 0.0,
    }
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
    FiniteSignedWeightsView::try_from_array(diag)
        .map_err(|reason| format!("xt_diag_x_symmetric: {reason}"))?;
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
                // Reserve-or-stream: the dense BLAS route runs only while its
                // full dense footprint is admitted by the process-wide
                // governor; a refusal picks the bounded streaming CSC path.
                if let Ok(xd) = xs.try_to_dense_governed("xt_diag_x_symmetric dense sparse route") {
                    stream_weighted_crossprod_into(
                        &**xd,
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
            // Built once per design, not once per assembly: the comment above
            // identified the repeated symbolic build as the dominant cost but
            // the fix only routed the DENSE regime around it, leaving the
            // genuinely-sparse arm here rebuilding the same pattern on every
            // Newton iteration. The design owns the memo, so the pattern is
            // shared by every later assembly against this same `X`.
            // The memo is built for `self.ncols()`, which IS `p` here, so the
            // template's dimension needs no separate check.
            let acc_template = xs
                .hessian_accumulator_template()
                .ok_or_else(|| "xt_diag_x_symmetric: failed to obtain CSR view".to_string())?;
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
        assert!(
            err.contains("nantest"),
            "error should mention context: {err}"
        );
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
    fn raw_symmetric_gram_rejects_smallest_nonfinite_row_before_sparse_assembly() {
        let sparse = SparseColMat::try_new_from_triplets(
            3,
            1,
            &[
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(2, 0, 3.0),
            ],
        )
        .unwrap();
        let design = DesignMatrix::Sparse(SparseDesignMatrix::new(sparse));
        let err = xt_diag_x_symmetric(&design, &array![1.0, f64::NAN, f64::INFINITY]).unwrap_err();
        assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
    }

    /// The two `XᵀWX` routes inside `xt_diag_x_symmetric` must agree, because
    /// which one runs is decided by a MEMORY RESERVATION, not by the problem.
    ///
    /// `xt_diag_x_symmetric`'s dense-regime arm picks between a densify-then-
    /// BLAS crossprod and a streaming CSC accumulation on `try_to_dense_governed`
    /// succeeding. That governor's budget derives from *available host memory*,
    /// so on a busy machine the same fit can take either branch. Nothing about
    /// the problem changed — only what else the machine was doing.
    ///
    /// That is only safe because the two branches agree, and nothing was
    /// asserting it. They do agree today (measured bit-identical when this was
    /// written), so this locks the property in rather than reporting a defect:
    /// a future blocking change to either kernel that broke it would make
    /// fitted answers depend on machine load, which is close to impossible to
    /// diagnose from the outside.
    #[test]
    fn both_xtwx_routes_agree_so_the_memory_governor_cannot_change_the_answer() {
        use crate::faer_ndarray::{
            CrossprodAccum, CrossprodStructure, stream_weighted_crossprod_into,
        };
        use faer::sparse::{SparseColMat, Triplet};

        // Dense-regime shape (`4·avg_nnz_row ≥ p`), and wide enough that the
        // two accumulation orders have room to disagree in the low bits.
        let (n, p, per_row) = (900_usize, 24_usize, 9_usize);
        let mut triplets = Vec::new();
        let mut dense = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for k in 0..per_row {
                let col = (i * 7 + k * 5) % p;
                // Mixed magnitudes: a summation-order difference is invisible
                // when every term is the same size.
                let value = ((i % 13) as f64 - 6.0) * 0.5 + (k as f64) * 1.7 + 0.25;
                triplets.push(Triplet::new(i, col, value));
                dense[[i, col]] += value;
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(n, p, &triplets).expect("design");
        // Signed weights, as the signed-weight route allows.
        let w = Array1::from_shape_fn(n, |i| ((i % 11) as f64 - 5.0) * 0.37 + 0.05);

        let mut via_dense = Array2::<f64>::zeros((p, p));
        stream_weighted_crossprod_into(
            &dense,
            &w,
            &mut via_dense,
            CrossprodStructure::Full,
            CrossprodAccum::Replace,
            effective_global_parallelism(),
        );

        let (symbolic, values) = sparse.parts();
        let mut via_sparse = Array2::<f64>::zeros((p, p));
        streaming_sparse_csc_xt_diag_x(
            symbolic.col_ptr(),
            symbolic.row_idx(),
            values,
            n,
            p,
            w.view(),
            &mut via_sparse,
        );

        let mut worst = 0.0_f64;
        let mut worst_at = (0, 0);
        for a in 0..p {
            for b in 0..p {
                let scale = via_dense[[a, b]]
                    .abs()
                    .max(via_sparse[[a, b]].abs())
                    .max(1.0);
                let rel = (via_dense[[a, b]] - via_sparse[[a, b]]).abs() / scale;
                if rel > worst {
                    worst = rel;
                    worst_at = (a, b);
                }
            }
        }
        // Measured bit-identical (worst == 0.0) on this fixture when written;
        // the bound is a tolerance rather than bit-equality so a legitimate
        // blocking change in either kernel does not fail the build. What must
        // never happen is a difference large enough to move a fit.
        assert!(
            worst <= 1e-13,
            "the two XtWX routes disagree by {worst:.3e} (relative) at [{}, {}]; \
             the memory governor picks between them, so the fitted answer would \
             depend on host memory pressure (#2486)",
            worst_at.0,
            worst_at.1
        );
    }

    /// Reusing one design's memoized Hessian pattern must not leak state
    /// between assemblies.
    ///
    /// The symbolic pattern is now built once per design and shared by every
    /// later `XᵀWX` against it. That is only sound if each assembly starts
    /// from a zeroed values buffer: if the buffer were shared rather than the
    /// pattern, the second call would return the SUM of both weightings and
    /// the third would drift further — a wrong Hessian on every Newton
    /// iteration after the first. Weights are varied and then repeated so a
    /// stale-values bug shows up as a changed answer for identical input.
    #[test]
    fn repeated_assemblies_against_one_sparse_design_do_not_accumulate() {
        use faer::sparse::{SparseColMat, Triplet};

        // Banded, 3 nonzeros per row over 40 columns, so `4·avg_nnz ≥ p` is
        // false and this takes the genuinely-sparse accumulator arm — the one
        // that consults the memo.
        let (n, p) = (200_usize, 40_usize);
        let mut triplets = Vec::new();
        let mut dense = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for k in 0..3 {
                let col = (i + k) % p;
                let value = 0.5 + ((i * 7 + k * 13) % 11) as f64 * 0.25;
                triplets.push(Triplet::new(i, col, value));
                dense[[i, col]] += value;
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(n, p, &triplets).expect("banded design");
        let design = DesignMatrix::Sparse(SparseDesignMatrix::new(sparse));

        let reference = |w: &Array1<f64>| -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    out[[a, b]] = (0..n).map(|i| w[i] * dense[[i, a]] * dense[[i, b]]).sum();
                }
            }
            out
        };
        let w1 = Array1::from_shape_fn(n, |i| 0.25 + (i % 5) as f64 * 0.5);
        let w2 = Array1::from_shape_fn(n, |i| 1.5 + (i % 3) as f64 * 0.75);

        // w1, then a DIFFERENT w2, then w1 again: the third result must equal
        // the first exactly, not merely be close to it.
        let mut first: Option<Array2<f64>> = None;
        for (label, w) in [("w1", &w1), ("w2", &w2), ("w1-again", &w1)] {
            let got = xt_diag_x_symmetric(&design, w)
                .unwrap_or_else(|e| panic!("{label}: {e}"))
                .to_dense();
            let want = reference(w);
            for a in 0..p {
                for b in 0..p {
                    assert!(
                        (got[[a, b]] - want[[a, b]]).abs() <= 1e-9 * want[[a, b]].abs().max(1.0),
                        "{label}: XtWX[{a},{b}] = {} but direct sum gives {}",
                        got[[a, b]],
                        want[[a, b]]
                    );
                }
            }
            match first.as_ref() {
                None => first = Some(got),
                Some(f) if label == "w1-again" => {
                    assert_eq!(*f, got, "repeat of w1 changed after w2");
                }
                Some(_) => assert_eq!(
                    label, "w2",
                    "only the w2 pass may differ from the stored w1 result"
                ),
            }
        }
    }

    // ── symmetrization_defect_2norm (#2748) ──────────────────────────────────

    /// An exactly symmetric matrix has no defect: the measurement must read
    /// zero rather than round-off, or every caller would inherit a floor it
    /// did not measure.
    #[test]
    fn an_exactly_symmetric_matrix_has_no_symmetrization_defect() {
        let m = array![[4.0_f64, 1.0, -0.5], [1.0, 3.0, 0.25], [-0.5, 0.25, 2.0]];
        assert_eq!(symmetrization_defect_2norm(&m), 0.0);
    }

    /// A single injected asymmetry `M[i,j] - M[j,i] = 2d` gives a skew part
    /// with the single nonzero pair `±d`, whose 2-norm is exactly `|d|`.
    #[test]
    fn one_injected_asymmetry_is_recovered_exactly() {
        let d = 3.5e-8_f64;
        let mut m = array![[4.0_f64, 1.0, -0.5], [1.0, 3.0, 0.25], [-0.5, 0.25, 2.0]];
        m[[0, 1]] += d;
        m[[1, 0]] -= d;
        let measured = symmetrization_defect_2norm(&m);
        assert!(
            (measured - d).abs() <= 8.0 * f64::EPSILON * d.max(1.0),
            "expected {d:.17e}, measured {measured:.17e}"
        );
    }

    /// It measures the SKEW part and nothing else: adding an arbitrary
    /// symmetric matrix, however large, cannot move it. That is what makes the
    /// quantity a certified lower bound on the assembly error rather than a
    /// norm of the matrix.
    #[test]
    fn a_symmetric_perturbation_however_large_does_not_move_the_defect() {
        let d = 1.25e-9_f64;
        let mut m = array![[4.0_f64, 1.0, -0.5], [1.0, 3.0, 0.25], [-0.5, 0.25, 2.0]];
        m[[0, 2]] += d;
        m[[2, 0]] -= d;
        let baseline = symmetrization_defect_2norm(&m);
        let symmetric = array![
            [1.0e6_f64, -2.0e5, 3.0e5],
            [-2.0e5, 7.0e5, 1.0e5],
            [3.0e5, 1.0e5, 5.0e5]
        ];
        let perturbed = &m + &symmetric;
        let measured = symmetrization_defect_2norm(&perturbed);
        assert!(
            (measured - baseline).abs() <= 8.0 * f64::EPSILON * 1.0e6,
            "a symmetric addition moved the skew measurement from {baseline:.17e} to \
             {measured:.17e}"
        );
    }

    /// The defect is exactly what `symmetrize_in_place` discards, stated as an
    /// identity between the two functions rather than asserted separately.
    #[test]
    fn the_defect_is_exactly_what_symmetrize_in_place_removes() {
        let m = array![
            [4.0_f64, 1.0 + 1.0e-7, -0.5],
            [1.0, 3.0, 0.25 - 4.0e-8],
            [-0.5 + 2.0e-8, 0.25, 2.0]
        ];
        let defect = symmetrization_defect_2norm(&m);
        let mut symmetrized = m.clone();
        symmetrize_in_place(&mut symmetrized);
        let discarded = &m - &symmetrized;
        // `discarded` IS the skew part `K` already, and for a skew `K` the
        // measurement returns `||(K - K')/2||_2 = ||K||_2` — so re-measuring the
        // discarded part must reproduce the reported defect exactly.
        let reported = symmetrization_defect_2norm(&discarded);
        assert!(
            (reported - defect).abs() <= 8.0 * f64::EPSILON * defect.max(1.0),
            "the discarded part measures {reported:.17e} against the reported {defect:.17e}"
        );
        assert!(defect > 0.0, "this fixture is asymmetric by construction");
    }

    /// A non-square or non-finite input yields no measurement, and "no
    /// measurement" must be zero rather than a large number a caller would
    /// then treat as a resolution.
    #[test]
    fn an_unmeasurable_input_yields_no_measurement() {
        let rectangular = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(symmetrization_defect_2norm(&rectangular), 0.0);
        let non_finite = array![[1.0_f64, f64::NAN], [0.0, 1.0]];
        assert_eq!(symmetrization_defect_2norm(&non_finite), 0.0);
    }
}
