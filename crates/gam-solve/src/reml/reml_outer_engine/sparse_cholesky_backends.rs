use super::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Cholesky HessianFactorization implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse Cholesky Hessian operator.
///
/// Wraps an existing `SparseExactFactor` and provides logdet, trace, and solve
/// from the same Cholesky factorization.
pub struct SparseCholeskyOperator {
    /// The sparse Cholesky factorization.
    pub(crate) factor: std::sync::Arc<gam_linalg::sparse_exact::SparseExactFactor>,
    /// Takahashi selected inverse (precomputed H^{-1} entries on the filled pattern of L).
    /// When available, trace computations use direct lookups instead of column solves.
    pub(crate) takahashi: Option<std::sync::Arc<gam_linalg::sparse_exact::TakahashiInverse>>,
    /// Precomputed log-determinant from the Cholesky diagonal.
    pub(crate) cached_logdet: f64,
    /// Dimension of H.
    pub(crate) n_dim: usize,
}

impl SparseCholeskyOperator {
    /// Create from an existing sparse factorization and its precomputed logdet.
    pub fn new(
        factor: std::sync::Arc<gam_linalg::sparse_exact::SparseExactFactor>,
        logdet_h: f64,
        dim: usize,
    ) -> Self {
        Self {
            factor,
            takahashi: None,
            cached_logdet: logdet_h,
            n_dim: dim,
        }
    }

    pub fn with_takahashi(
        mut self,
        taka: std::sync::Arc<gam_linalg::sparse_exact::TakahashiInverse>,
    ) -> Self {
        self.takahashi = Some(taka);
        self
    }

    pub(crate) const OPERATOR_SOLVE_CHUNK: usize = 64;

    pub(crate) fn takahashi_block_trace(
        taka: &gam_linalg::sparse_exact::TakahashiInverse,
        block: &Array2<f64>,
        start: usize,
    ) -> f64 {
        assert_eq!(block.nrows(), block.ncols());
        let mut trace = 0.0;
        for i in 0..block.nrows() {
            let diag = block[[i, i]];
            if diag.abs() > 1e-30 {
                trace += taka.get(start + i, start + i) * diag;
            }
            for j in (i + 1)..block.ncols() {
                let pair = block[[i, j]] + block[[j, i]];
                if pair.abs() > 1e-30 {
                    trace += taka.get(start + i, start + j) * pair;
                }
            }
        }
        trace
    }

    pub(crate) fn takahashi_left_multiply_block(
        taka: &gam_linalg::sparse_exact::TakahashiInverse,
        block: &Array2<f64>,
        start: usize,
    ) -> Array2<f64> {
        let dim = block.nrows();
        let mut out = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            let z_diag = taka.get(start + i, start + i);
            if z_diag.abs() > 1e-30 {
                for k in 0..dim {
                    out[[i, k]] += z_diag * block[[i, k]];
                }
            }
            for j in (i + 1)..dim {
                let z = taka.get(start + i, start + j);
                if z.abs() <= 1e-30 {
                    continue;
                }
                for k in 0..dim {
                    out[[i, k]] += z * block[[j, k]];
                    out[[j, k]] += z * block[[i, k]];
                }
            }
        }
        out
    }

    pub(crate) fn trace_hinv_operator_exact(&self, op: &dyn HyperOperator) -> f64 {
        let (range_start, range_end) = op
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut trace = 0.0_f64;
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let diagonal_sum = if cols == chunk {
                gam_linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_block,
                    start,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                gam_linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_view,
                    start,
                )
            };
            trace += diagonal_sum.unwrap_or_else(|e| {
                // SAFETY: `SparseCholeskyOperator` is constructed only with a
                // successfully-factorized SPD `self.factor`. The sparse SPD
                // multi-RHS solve only fails on factor corruption or RHS
                // shape mismatch; the RHS comes from `mul_basis_columns_into`
                // matching the factor's dimension, so failure here means
                // the cached factor was corrupted after construction —
                // a hard invariant violation.
                // SAFETY: self.factor is validated SPD; sparse-SPD solve only fails on factor corruption.
                reml_contract_panic(format!(
                    "SparseCholeskyOperator exact trace_hinv_operator solve failed: {e}"
                ))
            });
            start = end;
        }

        trace
    }

    pub(crate) fn solve_operator_column_range_rows_exact(
        &self,
        op: &dyn HyperOperator,
        col_start: usize,
        col_end: usize,
        row_start: usize,
        row_end: usize,
    ) -> Result<Array2<f64>, String> {
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let cols_total = col_end - col_start;
        let rows_total = row_end - row_start;
        let mut solved = Array2::<f64>::zeros((rows_total, cols_total));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = col_start;

        while start < col_end {
            let end = (start + chunk).min(col_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let solved_block = if cols == chunk {
                gam_linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_block,
                    row_start,
                    row_end,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                gam_linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_view,
                    row_start,
                    row_end,
                )
            }
            .map_err(|e| {
                format!(
                    "SparseCholeskyOperator::solve_operator_column_range_rows_exact multi-solve failed: {e}"
                )
            })?;
            solved
                .slice_mut(ndarray::s![.., start - col_start..end - col_start])
                .assign(&solved_block);
            start = end;
        }

        Ok(solved)
    }

    pub(crate) fn trace_hinv_matrix_operator_cross_exact(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        if let Some((_, range_start, range_end)) = op.block_local_data()
            && range_end - range_start < self.n_dim
        {
            return self.trace_hinv_matrix_block_operator_cross_exact(
                matrix,
                op,
                range_start,
                range_end,
            );
        }

        let solved_matrix = self.solve_multi(matrix);
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0_f64;
        let (range_start, range_end) = op
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, rhs_block.slice_mut(ndarray::s![.., ..cols]));

            let solved_op = if cols == chunk {
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_block)
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };

            let solved_op = solved_op.unwrap_or_else(|e| {
                // SAFETY: `self.factor` is the validated SPD Cholesky factor
                // (set only after successful factorization); the RHS shape
                // is `n_dim × cols` by construction. A sparse-SPD multi-RHS
                // failure here would mean factor corruption, which the
                // construction invariant forbids.
                // SAFETY: self.factor is validated SPD; matrix/operator multi-solve only fails on corruption.
                panic!("SparseCholeskyOperator exact matrix/operator cross solve failed: {e}")
            });

            for local_col in 0..cols {
                let matrix_row = start + local_col;
                for row in 0..self.n_dim {
                    trace += solved_matrix[[matrix_row, row]] * solved_op[[row, local_col]];
                }
            }
            start = end;
        }

        trace
    }

    pub(crate) fn trace_hinv_matrix_block_operator_cross_exact(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
        range_start: usize,
        range_end: usize,
    ) -> f64 {
        let t_start = std::time::Instant::now();
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(self.n_dim.max(1));
        let mut op_rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut eye_rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0_f64;
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk).min(range_end);
            let cols = end - start;
            op.mul_basis_columns_into(start, op_rhs_block.slice_mut(ndarray::s![.., ..cols]));

            eye_rhs_block.fill(0.0);
            for local_col in 0..cols {
                eye_rhs_block[[start + local_col, local_col]] = 1.0;
            }

            let solved_op = if cols == chunk {
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &op_rhs_block)
            } else {
                let rhs_view = op_rhs_block.slice(ndarray::s![.., ..cols]);
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };
            let solved_op = solved_op.unwrap_or_else(|e| {
                // SAFETY: same invariant — `self.factor` is the validated
                // SPD factor and `op_rhs_block` is allocated as
                // `n_dim × chunk`, so dimensions are compatible by
                // construction. Any failure indicates factor corruption.
                // SAFETY: self.factor is validated SPD; block-operator multi-solve only fails on corruption.
                panic!(
                    "SparseCholeskyOperator exact matrix/block-operator cross operator solve failed: {e}"
                )
            });

            let solved_eye = if cols == chunk {
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &eye_rhs_block)
            } else {
                let rhs_view = eye_rhs_block.slice(ndarray::s![.., ..cols]);
                gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
            };
            let solved_eye = solved_eye.unwrap_or_else(|e| {
                // SAFETY: same invariant — `self.factor` is validated SPD
                // and `eye_rhs_block` was just filled as an identity-block
                // RHS sized `n_dim × chunk`. Failure indicates factor
                // corruption, forbidden by the construction invariant.
                // SAFETY: self.factor is validated SPD; identity-RHS multi-solve only fails on corruption.
                panic!(
                    "SparseCholeskyOperator exact matrix/block-operator cross identity solve failed: {e}"
                )
            });

            let selected_rows_t = matrix.t().dot(&solved_eye);
            for local_col in 0..cols {
                for row in 0..self.n_dim {
                    trace += selected_rows_t[[row, local_col]] * solved_op[[row, local_col]];
                }
            }
            start = end;
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > REML_TRACE_SLOW_LOG_MS {
            log::info!(
                "[REML-trace] matrix_block_op_cross_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                range_end - range_start,
                elapsed_ms
            );
        }
        trace
    }

    pub(crate) fn trace_hinv_operator_cross_exact(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let (left_start, left_end) = left
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));
        let (right_start, right_end) = right
            .block_local_data()
            .map(|(_, start, end)| (start, end))
            .unwrap_or((0, self.n_dim));

        let solved_left = self
            .solve_operator_column_range_rows_exact(
                left,
                left_start,
                left_end,
                right_start,
                right_end,
            )
            .unwrap_or_else(|e| {
                // SAFETY: `solve_operator_column_range_rows_exact` only
                // forwards `solve_sparse_spdmulti` errors. `self.factor` is
                // the validated SPD Cholesky factor; column ranges come
                // from the operator's own `block_local_data` (or fall back
                // to `0..n_dim`), so failure indicates factor corruption.
                // SAFETY: self.factor is validated SPD; operator cross-left solve only fails on corruption.
                panic!("SparseCholeskyOperator exact operator cross left solve failed: {e}")
            });
        let same_operator =
            std::ptr::addr_eq(left, right) && left_start == right_start && left_end == right_end;
        let solved_right = if same_operator {
            None
        } else {
            Some(
                self.solve_operator_column_range_rows_exact(
                    right,
                    right_start,
                    right_end,
                    left_start,
                    left_end,
                )
                .unwrap_or_else(|e| {
                    // SAFETY: mirrors the left-solve invariant above —
                    // `self.factor` is validated SPD and the column range
                    // is taken from `right`'s own `block_local_data`,
                    // so failure indicates factor corruption.
                    // SAFETY: self.factor is validated SPD; operator cross-right solve only fails on corruption.
                    panic!("SparseCholeskyOperator exact operator cross right solve failed: {e}")
                }),
            )
        };

        let right_cols = right_end - right_start;
        let mut trace = 0.0;
        for left_col in 0..(left_end - left_start) {
            for right_col in 0..right_cols {
                let right_value = match solved_right.as_ref() {
                    Some(solved) => solved[[left_col, right_col]],
                    None => solved_left[[left_col, right_col]],
                };
                trace += solved_left[[right_col, left_col]] * right_value;
            }
        }
        trace
    }
}

impl HessianFactorization for SparseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        let h = gam_linalg::sparse_exact::assemble_sparse_factor_h_dense(&self.factor)
            .map_err(|e| e.to_string())?;
        if h.nrows() != self.n_dim || h.ncols() != self.n_dim {
            return Err(format!(
                "sparse Cholesky tangent projection dense H has shape {}x{}, expected {}x{}",
                h.nrows(),
                h.ncols(),
                self.n_dim,
                self.n_dim
            ));
        }
        Ok(h)
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // When Takahashi is available, use direct entry lookup for tr(H^{-1} A).
        // This is O(p^2) via dense A iteration but avoids p column solves.
        if let Some(ref taka) = self.takahashi {
            let mut trace = 0.0;
            for i in 0..a.nrows() {
                let a_ii = a[[i, i]];
                if a_ii.abs() > 1e-30 {
                    trace += taka.get(i, i) * a_ii;
                }
                for j in (i + 1)..a.ncols() {
                    let pair = a[[i, j]] + a[[j, i]];
                    if pair.abs() > 1e-30 {
                        trace += taka.get(i, j) * pair;
                    }
                }
            }
            return trace;
        }
        gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, a)
            .unwrap_or_else(|e| {
                // SAFETY: `self.factor` is the validated SPD Cholesky factor
                // (created by `SparseCholeskyOperator::new` only after a
                // successful factorization); a single-square multi-RHS solve
                // here can only fail on factor corruption, which the
                // construction invariant forbids.
                // SAFETY: self.factor is validated SPD; single-square multi-solve only fails on corruption.
                panic!("SparseCholeskyOperator exact trace_hinv_product solve failed: {e}")
            })
            .diag()
            .sum()
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if let Some(ref taka) = self.takahashi {
            if let Some((local, start, end)) = op.block_local_data() {
                assert_eq!(local.nrows(), end - start);
                return Self::takahashi_block_trace(taka, local, start);
            }
            // For other non-implicit operators: materialize and use Takahashi lookups
            if !op.is_implicit() {
                let dense = op.to_dense();
                return self.trace_hinv_product(&dense);
            }
        }
        self.trace_hinv_operator_exact(op)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.trace_hinv_operator(op)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // SAFETY: `self.factor` is the validated SPD Cholesky factor stored
        // at construction time; a triangular solve against an already-built
        // factor can only fail on factor corruption, which the
        // `SparseCholeskyOperator` construction invariant forbids.
        gam_linalg::sparse_exact::solve_sparse_spd(&self.factor, rhs)
            // SAFETY: self.factor is validated SPD; triangular solve only fails on corruption.
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact solve failed: {e}"))
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        // SAFETY: same SPD-factor invariant as `solve` above — `self.factor`
        // was created from a successful Cholesky factorization, so a
        // multi-RHS solve can only fail on factor corruption.
        gam_linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, rhs)
            // SAFETY: self.factor is validated SPD; multi-RHS solve only fails on corruption.
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact multi-solve failed: {e}"))
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        // For general dense matrices, column solves are better than materializing
        // full Z from Takahashi (O(p * nnz) vs O(p³)). Takahashi cross-traces
        // are only used for block-local operators via trace_hinv_operator_cross.
        let solved_a = self.solve_multi(a);
        if std::ptr::eq(a, b) {
            return dense::trace_product(&solved_a, &solved_a);
        }
        let solved_b = self.solve_multi(b);
        dense::trace_product(&solved_a, &solved_b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        // For mixed dense-matrix × block-local-operator, column solves are
        // still better than materializing full Z. Only use Takahashi when both
        // sides are block-local (handled in trace_hinv_operator_cross).
        self.trace_hinv_matrix_operator_cross_exact(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        // Takahashi fast path: when both operators are block-local to the same
        // block, compute tr(Z A Z B) using only the block of Z = H⁻¹.
        if let Some(ref taka) = self.takahashi
            && let (Some((a_local, a_start, a_end)), Some((b_local, b_start, b_end))) =
                (left.block_local_data(), right.block_local_data())
            && a_start == b_start
            && a_end == b_end
        {
            // Same block: tr(Z_block * A_local * Z_block * B_local)
            let za = Self::takahashi_left_multiply_block(taka, a_local, a_start);
            if std::ptr::addr_eq(left, right) {
                return dense::trace_product(&za, &za);
            }
            let zb = Self::takahashi_left_multiply_block(taka, b_local, b_start);
            // tr(ZA * ZB) = sum_ij (ZA)_ij * (ZB^T)_ij
            return (&za * &zb.t()).sum();
        }
        // Different blocks: column solves are better than materializing
        // full p×p Z. Fall through to exact path.
        self.trace_hinv_operator_cross_exact(left, right)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_matrix_operator_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_operator_cross(h_i, h_j)
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }
}

// BlockCoupledDerivativeProvider was removed — its functionality is now handled
// by the `deriv_provider` trait (HessianDerivativeProvider), with concrete
// implementations like JointModelDerivProvider and SurvivalDerivProvider
// capturing the full correction including Jacobian sensitivity, weight
// sensitivity, and basis sensitivity.

// ═══════════════════════════════════════════════════════════════════════════
//  Cholesky-backed exact positive-definite HessianFactorization
// ═══════════════════════════════════════════════════════════════════════════

/// Dense Cholesky-backed [`HessianFactorization`] for positive-definite Hessians.
///
/// A single LLT factor supplies the exact positive-definite log-determinant,
/// solves, first-order traces, and second-order cross traces. Consequently every
/// derivative lane prices the same scalar `log|H|`; no eigenspace threshold or
/// pseudo-spectral floor is involved.
///
/// LLT costs `O(p³/3)` flops versus
/// the `O(9·p³)` full eigendecomposition of [`DenseSpectralOperator`], giving
/// a multi-× speedup at the small and medium dense dimensions where exact outer
/// derivatives are required.
pub struct DenseCholeskyOperator {
    /// LLT Cholesky factor.
    pub(crate) chol: gam_linalg::faer_ndarray::FaerCholeskyFactor,
    /// `2 · Σ ln(diag L)` — cached at construction time.
    pub(crate) cached_logdet: f64,
    /// Full parameter dimension.
    pub(crate) n_dim: usize,
    /// Exact symmetric matrix represented by `chol`, retained for the uncommon
    /// active-constraint tangent-projection surface.
    pub(crate) matrix: Array2<f64>,
    /// Upper-triangular `F = L⁻ᵀ`, where `H = L Lᵀ`, so
    /// `H⁻¹ = F Fᵀ`. Operator traces use exact projected contractions through
    /// this factor instead of materializing each Hessian drift.
    pub(crate) inverse_root: Array2<f64>,
}

impl DenseCholeskyOperator {
    /// Replace the cached `2·Σ ln(diag L)` with a value computed at ROOT
    /// scale. The Cholesky factor itself is untouched — it is the operator's
    /// solve/trace kernel, and only the log-determinant scalar is
    /// `O(ε·κ(H))`-limited (#2644).
    pub(crate) fn install_root_scale_logdet(&mut self, value: f64) {
        self.cached_logdet = value;
    }

    /// Construct `L⁻ᵀ` by stable triangular substitution without forming
    /// `H⁻¹`. This is the exact projection factor needed by
    /// `tr(H⁻¹A) = tr(FᵀAF)` and
    /// `tr(H⁻¹AH⁻¹B) = tr((FᵀAF)(FᵀBF))`.
    fn inverse_transpose_root(lower: &Array2<f64>) -> Array2<f64> {
        let n = lower.nrows();
        let lower_values = lower
            .as_slice()
            .expect("Cholesky lower triangle is standard-layout");
        let mut lower_inverse = vec![0.0_f64; n * n];

        // Solve L R = I one column at a time. Only row >= column can be
        // nonzero because both L and R are lower triangular.
        for column in 0..n {
            for row in column..n {
                let mut value = if row == column { 1.0 } else { 0.0 };
                for inner in column..row {
                    value -= lower_values[row * n + inner] * lower_inverse[inner * n + column];
                }
                lower_inverse[row * n + column] = value / lower_values[row * n + row];
            }
        }

        // F = Rᵀ.
        let mut inverse_root = Array2::<f64>::zeros((n, n));
        let root_values = inverse_root
            .as_slice_mut()
            .expect("fresh inverse root is standard-layout");
        for row in 0..n {
            for column in row..n {
                root_values[row * n + column] = lower_inverse[column * n + row];
            }
        }
        inverse_root
    }

    #[inline]
    fn projected_dense(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let matrix_factor = gam_linalg::faer_ndarray::fast_ab(matrix, &self.inverse_root);
        gam_linalg::faer_ndarray::fast_atb(&self.inverse_root, &matrix_factor)
    }

    #[inline]
    fn projected_cross(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        dense::trace_product(left, right)
    }

    /// Factorize `h` via LLT and cache the exact positive-definite
    /// log-determinant.
    fn factorize_positive_definite(h: &Array2<f64>) -> Result<Self, String> {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(format!(
                "DenseCholeskyOperator: expected square matrix, got {}×{}",
                n,
                h.ncols()
            ));
        }
        if h.iter().any(|entry| !entry.is_finite()) {
            return Err("DenseCholeskyOperator: Hessian contains a non-finite entry".to_string());
        }

        // A Hessian represents a symmetric bilinear form. LLT libraries consume
        // one triangle, so accepting an asymmetric matrix would silently make
        // the represented operator depend on storage convention. Admit only
        // roundoff-scale skew, then factor and retain the same averaged matrix.
        let scale = h
            .iter()
            .fold(0.0_f64, |maximum, entry| maximum.max(entry.abs()))
            .max(1.0);
        let symmetry_tolerance = 64.0 * f64::EPSILON * n.max(1) as f64 * scale;
        let mut matrix = h.clone();
        for row in 0..n {
            for col in (row + 1)..n {
                let upper = h[[row, col]];
                let lower = h[[col, row]];
                let skew = (upper - lower).abs();
                if skew > symmetry_tolerance {
                    return Err(format!(
                        "DenseCholeskyOperator: Hessian is not symmetric at ({row}, {col}); \
                         |H_ij-H_ji|={skew:.3e} exceeds the roundoff envelope \
                         {symmetry_tolerance:.3e}"
                    ));
                }
                let symmetric = 0.5 * (upper + lower);
                matrix[[row, col]] = symmetric;
                matrix[[col, row]] = symmetric;
            }
        }

        let chol = matrix
            .cholesky(Side::Lower)
            .map_err(|e| format!("DenseCholeskyOperator LLT failed: {e}"))?;
        let diag = chol.diag();
        let cached_logdet = 2.0 * diag.iter().map(|&d| d.ln()).sum::<f64>();
        let inverse_root = Self::inverse_transpose_root(&chol.lower_triangular());
        Ok(Self {
            chol,
            cached_logdet,
            n_dim: n,
            matrix,
            inverse_root,
        })
    }

    /// Exact factorization for [`PseudoLogdetMode::PositiveDefinite`].
    ///
    /// Failure is the requested definiteness certificate: callers must refuse
    /// the candidate rather than floor a saddle or singular mode.
    pub fn from_positive_definite(h: &Array2<f64>) -> Result<Self, String> {
        Self::factorize_positive_definite(h)
    }

    /// Smooth-logdet value-lane shortcut, admitted only when its exact
    /// log-determinant is certified to agree with the smooth spectral scalar.
    ///
    /// Returns `Err` if `h` is not SPD or if its exact log-determinant would not
    /// agree with the smooth-floored one every derivative lane prices
    /// (gam#2457, below).
    /// On refusal, the caller routes the evaluation to
    /// [`DenseSpectralOperator`], which owns the floored convention.
    pub fn from_spd_with_smooth_logdet_agreement(h: &Array2<f64>) -> Result<Self, String> {
        let operator = Self::factorize_positive_definite(h)?;
        let n = operator.n_dim;
        let cached_logdet = operator.cached_logdet;

        // gam#2457 — THIS OPERATOR AND THE SPECTRAL ONE PRICE DIFFERENT SCALARS.
        //
        // The LLT returns the exact `Σ ln σ_j`.  Every derivative-bearing lane
        // reaches [`DenseSpectralOperator`] instead, whose smooth floor makes
        // its log-determinant `Σ ln r_ε(σ_j)` with
        // `r_ε(σ) = ½(σ + √(σ² + 4ε²))` and `ε = spectral_epsilon` — and the
        // analytic gradient `tr(G_ε Ḣ)` and its Hessian are the exact
        // derivatives of THAT floored object.  So the floored log-determinant
        // is the criterion, and this fast path is a legitimate shortcut only
        // where the two coincide.  Where they do not, the outer objective
        // returns one value to `OuterEvalOrder::Value` (line-search probes,
        // the terminal value certificate) and another to `ValueAndGradient`
        // (the trust-region model, the certificate's analytic sample) at the
        // SAME ρ — measured at 663× the value-agreement envelope on
        // `kappa_zero_fit_recovers_planted_flat_signal`, whose `H = XᵀWX + S_λ`
        // carries an eigenvalue at ≈7ε once λ = e^−8.9 stops regularizing it.
        //
        // The gap is bounded without ever forming the spectrum.  For SPD `H`,
        // `√(1 + 4t) ≤ 1 + 2t` gives `r_ε(σ)/σ ≤ 1 + ε²/σ²`, and `ln(1+t) ≤ t`,
        // so
        //
        //     0 ≤ Σ_j ln(r_ε(σ_j)/σ_j) ≤ ε² · Σ_j σ_j⁻² = ε² · tr(H⁻²)
        //
        // and `tr(H⁻²) = ‖H⁻¹‖_F²` comes straight out of the factorization
        // already in hand.  The bound is tight in the regime that matters (a
        // single near-floor eigenvalue dominates both sides), so gating on it
        // costs the speedup only where the floor genuinely bites.
        //
        // Admit the fast path exactly when that certified gap is inside the
        // same relative envelope the outer audit applies to the scalar this
        // log-determinant feeds — ONE predicate, named once, reused rather than
        // re-derived.  The decline is one-sided: it can cost an LLT speedup, it
        // can never admit a value the derivative lanes disagree with.  Both
        // call sites already handle `Err` by building the spectral operator.
        let epsilon = spectral_epsilon_for_dim(n);
        let h_inverse = operator.chol.solve_mat(&Array2::<f64>::eye(n));
        let floor_gap_bound =
            epsilon * epsilon * h_inverse.iter().map(|entry| entry * entry).sum::<f64>();
        let agreement_envelope =
            crate::rho_optimizer::outer_value_agreement_bound(cached_logdet, cached_logdet);
        if !(floor_gap_bound <= agreement_envelope) {
            return Err(format!(
                "DenseCholeskyOperator declines a {n}-dimensional Hessian: its exact \
                 log-determinant can differ from the smooth-floored log|H| the derivative lanes \
                 price by up to {floor_gap_bound:.3e}, above the {agreement_envelope:.3e} \
                 value-agreement envelope (spectral floor eps={epsilon:.3e})"
            ));
        }

        Ok(operator)
    }
}

impl HessianFactorization for DenseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        Ok(self.matrix.clone())
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        let a_factor = gam_linalg::faer_ndarray::fast_ab(a, &self.inverse_root);
        self.inverse_root
            .iter()
            .zip(a_factor.iter())
            .map(|(&factor, &a_factor)| factor * a_factor)
            .sum()
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        op.trace_projected_factor(&self.inverse_root)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.trace_hinv_operator(op)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.chol.solvevec(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.chol.solve_mat(rhs)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let projected_a = self.projected_dense(a);
        if std::ptr::eq(a, b) {
            return Self::projected_cross(&projected_a, &projected_a);
        }
        let projected_b = self.projected_dense(b);
        Self::projected_cross(&projected_a, &projected_b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let projected_matrix = self.projected_dense(matrix);
        let projected_operator = op.projected_matrix(&self.inverse_root);
        Self::projected_cross(&projected_matrix, &projected_operator)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let projected_left = left.projected_matrix(&self.inverse_root);
        if std::ptr::addr_eq(left, right) {
            return Self::projected_cross(&projected_left, &projected_left);
        }
        let projected_right = right.projected_matrix(&self.inverse_root);
        Self::projected_cross(&projected_left, &projected_right)
    }

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        assert_eq!(block.dim(), (end - start, end - start));
        let factor_block = self.inverse_root.slice(ndarray::s![start..end, ..]);
        let block_factor = gam_linalg::faer_ndarray::fast_ab(block, &factor_block);
        scale
            * factor_block
                .iter()
                .zip(block_factor.iter())
                .map(|(&factor, &block_factor)| factor * block_factor)
                .sum::<f64>()
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_matrix_operator_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        -self.trace_hinv_operator_cross(h_i, h_j)
    }

    fn active_rank(&self) -> usize {
        // LLT succeeded ⟹ all pivots are positive ⟹ full rank.
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Block-coupled HessianFactorization for joint multi-block models
// ═══════════════════════════════════════════════════════════════════════════

/// Block-coupled Hessian operator for joint multi-block models (GAMLSS, survival).
///
/// Retains block-structure metadata around one factorization of the full
/// assembled joint Hessian. Strictly positive-definite models use LLT; models
/// with genuine quotient or smooth pseudo-logdet semantics use one spectral
/// decomposition. Every [`HessianFactorization`] operation delegates to that
/// same inner factorization.
///
/// # Block structure
///
/// A joint model with B parameter blocks has a joint Hessian of dimension
/// `p_total = sum_b p_b`. Each block occupies rows/columns
/// # When to use
///
/// Use `BlockCoupledOperator` whenever building an [`InnerSolution`] for a joint
/// multi-block model. It replaces the pattern of constructing a raw
/// `DenseSpectralOperator` and manually tracking block ranges separately.
enum BlockCoupledFactorization {
    Spectral(DenseSpectralOperator),
    PositiveDefinite(DenseCholeskyOperator),
}

impl BlockCoupledFactorization {
    fn as_factorization(&self) -> &dyn HessianFactorization {
        match self {
            Self::Spectral(operator) => operator,
            Self::PositiveDefinite(operator) => operator,
        }
    }
}

pub struct BlockCoupledOperator {
    /// One exact factorization over the full joint Hessian. Positive-definite
    /// models use LLT; pseudo-logdet modes retain their spectral semantics.
    inner: BlockCoupledFactorization,
}

impl BlockCoupledOperator {
    /// Construct from an assembled joint Hessian using the supplied
    /// [`PseudoLogdetMode`]. Positive-definite mode uses an exact LLT;
    /// pseudo-logdet modes use a single eigendecomposition.
    pub fn from_joint_hessian_with_mode(
        joint_hessian: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        let inner = match mode {
            PseudoLogdetMode::PositiveDefinite => BlockCoupledFactorization::PositiveDefinite(
                DenseCholeskyOperator::from_positive_definite(joint_hessian)
                    .map_err(|e| format!("BlockCoupledOperator positive-definite factor: {e}"))?,
            ),
            PseudoLogdetMode::Smooth | PseudoLogdetMode::HardPseudo => {
                BlockCoupledFactorization::Spectral(
                    DenseSpectralOperator::from_symmetric_with_mode(joint_hessian, mode)
                        .map_err(|e| format!("BlockCoupledOperator eigendecomposition: {e}"))?,
                )
            }
        };

        Ok(Self { inner })
    }
}

impl HessianFactorization for BlockCoupledOperator {
    fn logdet(&self) -> f64 {
        self.inner.as_factorization().logdet()
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        match &self.inner {
            BlockCoupledFactorization::Spectral(operator) => Some(operator),
            BlockCoupledFactorization::PositiveDefinite(_) => None,
        }
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        self.inner
            .as_factorization()
            .assemble_h_dense_for_tangent_projection()
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.inner.as_factorization().trace_hinv_product(a)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.inner.as_factorization().trace_hinv_operator(op)
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.inner.as_factorization().trace_logdet_gradient(a)
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        self.inner.as_factorization().xt_logdet_kernel_x_diagonal(x)
    }

    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_logdet_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.inner.as_factorization().trace_logdet_operator(op)
    }

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_logdet_block_local(block, scale, start, end)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.inner
            .as_factorization()
            .trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_logdet_hessian_cross_matrix_operator(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_logdet_hessian_cross_operator(h_i, h_j)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.inner.as_factorization().solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.inner.as_factorization().solve_multi(rhs)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.inner.as_factorization().trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.inner
            .as_factorization()
            .trace_hinv_operator_cross(left, right)
    }

    fn active_rank(&self) -> usize {
        self.inner.as_factorization().active_rank()
    }

    fn dim(&self) -> usize {
        self.inner.as_factorization().dim()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        self.inner
            .as_factorization()
            .logdet_traces_match_hinv_kernel()
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        match &self.inner {
            BlockCoupledFactorization::Spectral(operator) => Some(operator),
            BlockCoupledFactorization::PositiveDefinite(_) => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix-free SPD HessianFactorization implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Operator-backed SPD Hessian with exact spectral REML algebra.
///
/// The operator closure is still useful for construction paths that naturally
/// expose HVPs, but REML cost/gradient/Hessian terms must all come from one
/// exact decomposition so `∂ log|H| = tr(H⁻¹ ∂H)` holds.  We therefore
/// materialize the coefficient Hessian by canonical-basis HVPs under an
/// explicit memory cap and delegate logdet, traces, and solves to
/// `DenseSpectralOperator`.
pub struct MatrixFreeSpdOperator {
    pub(crate) apply: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    // Optional single-pass dense assembly of the SAME penalized operator that
    // `apply` realizes matrix-free, i.e. `H_unpen + S_λ + scale·H_Φ`. When the
    // operator source can structurally build its full dense matrix in one
    // chunked BLAS-3 `XᵀWX` row pass (BMS's `hessian_dense_forced` +
    // construction-site penalty/Jeffreys assembly), `materialize_dense_operator`
    // calls THIS instead of `dim` canonical-basis matvecs — each of which is a
    // full n-row pass through the matrix-free operator. One n-pass replaces
    // `dim` n-passes for the LAML logdet factorization. The closure must return
    // a matrix numerically identical (up to symmetrization) to the matvec
    // reconstruction `H·I`; `None` means no direct build is available and the
    // matvec path is used (the result is bit-for-bit the prior behavior).
    pub(crate) dense_assemble: Option<Arc<dyn Fn() -> Option<Array2<f64>> + Send + Sync>>,
    pub(crate) cached_logdet: gam_runtime::resource::RayonSafeOnce<f64>,
    pub(crate) n_dim: usize,
    // `RayonSafeOnce`, not `OnceLock`: `materialize_dense_operator` invokes
    // `apply`, which for operator-source joint Hessians dispatches a nested
    // `into_par_iter` (e.g. `exact_newton_joint_hessian_matvec_from_cache`).
    // With a plain `OnceLock`, concurrent rayon workers entering
    // `solve`/`logdet` from inside an outer par_iter would park on the
    // OnceLock's OS condvar; the leader's nested par_iter would then starve
    // for workers. `RayonSafeOnce` keeps init lock-free — racers may
    // duplicate the dim²-matvec build, but the first to publish wins and
    // steady-state matches `OnceLock`.
    pub(crate) dense_spectral: gam_runtime::resource::RayonSafeOnce<Option<DenseSpectralOperator>>,
    // Pseudo-logdet convention threaded from the family. The dense outer path
    // already plumbs `PseudoLogdetMode` into `BlockCoupledOperator`; the
    // matrix-free path materializes a `DenseSpectralOperator` lazily and must
    // use the same convention so that `logdet`, `trace_hinv_product`, the
    // IFT response `H⁻¹ g`, and every cross-trace agree with the dense path.
    // Without this, families that declare `HardPseudo` (BMS, GAMLSS) silently
    // get Smooth full-spectrum semantics on the matrix-free path, and outer
    // gradients are inflated by `1/σ_j` over numerical null directions.
    pub(crate) mode: PseudoLogdetMode,
}

impl MatrixFreeSpdOperator {
    pub(crate) const EXACT_DENSE_SPECTRAL_MAX_BYTES: usize = 512 * 1024 * 1024;
    pub(crate) const EXACT_DENSE_SPECTRAL_ARRAYS: usize = 6;

    pub fn new_with_mode<F>(dim: usize, apply: F, mode: PseudoLogdetMode) -> Self
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        Self::new_with_mode_and_dense_assemble(dim, apply, mode, None)
    }

    /// Like `new_with_mode`, but additionally accepts an optional single-pass
    /// dense assembly of the same penalized operator. When present and it yields
    /// a matrix, `materialize_dense_operator` uses it instead of the `dim`
    /// canonical-basis matvecs. See the field doc on `dense_assemble`.
    pub fn new_with_mode_and_dense_assemble<F>(
        dim: usize,
        apply: F,
        mode: PseudoLogdetMode,
        dense_assemble: Option<Arc<dyn Fn() -> Option<Array2<f64>> + Send + Sync>>,
    ) -> Self
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        let apply = Arc::new(apply);

        Self {
            apply,
            dense_assemble,
            cached_logdet: gam_runtime::resource::RayonSafeOnce::new(),
            n_dim: dim,
            dense_spectral: gam_runtime::resource::RayonSafeOnce::new(),
            mode,
        }
    }

    pub(crate) fn exact_dense_spectral_bytes(&self) -> Option<usize> {
        self.n_dim
            .checked_mul(self.n_dim)?
            .checked_mul(std::mem::size_of::<f64>())?
            .checked_mul(Self::EXACT_DENSE_SPECTRAL_ARRAYS)
    }

    pub(crate) fn exact_dense_spectral_budget_ok(&self) -> bool {
        match self.exact_dense_spectral_bytes() {
            Some(bytes) if bytes <= Self::EXACT_DENSE_SPECTRAL_MAX_BYTES => true,
            Some(bytes) => {
                log::error!(
                    "MatrixFreeSpdOperator exact dense spectral materialization requires {:.2} GiB \
                     for dim={}, exceeding the {:.2} GiB cap",
                    bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    self.n_dim,
                    Self::EXACT_DENSE_SPECTRAL_MAX_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                false
            }
            None => {
                log::error!(
                    "MatrixFreeSpdOperator exact dense spectral byte count overflow for dim={}",
                    self.n_dim
                );
                false
            }
        }
    }

    pub(crate) fn materialize_dense_operator(&self) -> Option<DenseSpectralOperator> {
        if !self.exact_dense_spectral_budget_ok() {
            return None;
        }
        let materialize_start = std::time::Instant::now();
        // Fast path: structural single-pass dense assembly of the SAME penalized
        // operator (`H_unpen + S_λ + scale·H_Φ`). One chunked BLAS-3 `XᵀWX`
        // row pass replaces `n_dim` canonical-basis matvecs, each a full n-row
        // pass through the matrix-free operator. The matvec fallback below is the
        // exact same algebra column-for-column, so the spectrum/logdet match.
        let (matrix, matvec_count) =
            match self.dense_assemble.as_ref().and_then(|assemble| assemble()) {
                Some(mut direct)
                    if direct.nrows() == self.n_dim
                        && direct.ncols() == self.n_dim
                        && direct.iter().all(|v| v.is_finite()) =>
                {
                    // Symmetrize defensively; the direct build is structurally
                    // symmetric but reduction-order f.p. noise can desync mirror
                    // entries, exactly as the matvec path symmetrizes below.
                    for i in 0..self.n_dim {
                        for j in (i + 1)..self.n_dim {
                            let avg = 0.5 * (direct[[i, j]] + direct[[j, i]]);
                            direct[[i, j]] = avg;
                            direct[[j, i]] = avg;
                        }
                    }
                    (direct, 0usize)
                }
                _ => {
                    let mut matrix = Array2::<f64>::zeros((self.n_dim, self.n_dim));
                    let mut basis = Array1::<f64>::zeros(self.n_dim);
                    for j in 0..self.n_dim {
                        basis[j] = 1.0;
                        let col = (self.apply)(&basis);
                        basis[j] = 0.0;
                        if col.len() != self.n_dim || !col.iter().all(|v| v.is_finite()) {
                            return None;
                        }
                        matrix.column_mut(j).assign(&col);
                    }
                    for i in 0..self.n_dim {
                        for j in (i + 1)..self.n_dim {
                            let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
                            matrix[[i, j]] = avg;
                            matrix[[j, i]] = avg;
                        }
                    }
                    (matrix, self.n_dim)
                }
            };
        let result = match DenseSpectralOperator::from_symmetric_with_mode(&matrix, self.mode) {
            Ok(operator) => Some(operator),
            Err(err) => {
                // `None` here silently demotes the caller to a slower path.
                // Say why, or the demotion is indistinguishable from "not
                // requested" in a profile.
                log::warn!(
                    "[matrix_free_spd] dense spectral materialization declined at n_dim={}: {err}",
                    self.n_dim
                );
                None
            }
        };
        log::info!(
            "[STAGE] matrix_free_spd materialize n_dim={} matvec_count={} elapsed={:.3}s",
            self.n_dim,
            matvec_count,
            materialize_start.elapsed().as_secs_f64(),
        );
        result
    }

    pub(crate) fn dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.dense_spectral
            .get_or_compute(|| self.materialize_dense_operator())
            .as_ref()
    }

    pub(crate) fn exact_dense_spectral(&self) -> &DenseSpectralOperator {
        self.dense_spectral().expect(
            "MatrixFreeSpdOperator exact REML algebra requires dense spectral materialization within the configured budget",
        )
    }

    pub(crate) fn use_trace_cg(&self, rel_tol: f64) -> bool {
        rel_tol.is_finite()
            && rel_tol > 0.0
            && self.prefers_stochastic_trace_estimation()
            && self.has_matrix_free_trace_cg_operator()
    }

    pub(crate) fn cg_trace_solve(
        &self,
        rhs: &Array1<f64>,
        rel_tol: f64,
        probe_id: Option<u64>,
        trace_state: Option<&Arc<Mutex<StochasticTraceState>>>,
    ) -> Array1<f64> {
        let dim = rhs.len();
        if dim != self.n_dim {
            return self.solve(rhs);
        }

        let (initial, warm_start_used) = match (probe_id, trace_state) {
            (Some(id), Some(state)) => {
                let cached = match state.lock() {
                    Ok(guard) => guard.cg_warm_starts.get(&id).cloned(),
                    Err(poisoned) => poisoned.into_inner().cg_warm_starts.get(&id).cloned(),
                };
                match cached {
                    Some(x) if x.len() == dim => (x, true),
                    _ => (Array1::<f64>::zeros(dim), false),
                }
            }
            _ => (Array1::<f64>::zeros(dim), false),
        };

        let Some((solution, iters, residual_norm)) =
            conjugate_gradient_trace_solve(rhs, rel_tol, initial, |v| (self.apply)(v))
        else {
            return self.solve(rhs);
        };

        if let Some(state) = trace_state {
            let mut guard = match state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.last_linear_residual_norm = Some(
                guard
                    .last_linear_residual_norm
                    .unwrap_or(0.0)
                    .max(residual_norm),
            );
            if let Some(id) = probe_id {
                guard.cg_warm_starts.insert(id, solution.clone());
            }
        }

        let probe_label = probe_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "untracked".to_string());
        log::info!(
            "[CG-TRACE] probe_id={} iters={} rel_tol={} warm_start_used={}",
            probe_label,
            iters,
            rel_tol,
            warm_start_used
        );

        solution
    }
}

pub(crate) fn conjugate_gradient_trace_solve<F>(
    rhs: &Array1<f64>,
    rel_tol: f64,
    mut x: Array1<f64>,
    apply: F,
) -> Option<(Array1<f64>, usize, f64)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let dim = rhs.len();
    if x.len() != dim {
        return None;
    }

    let rhs_norm_sq = rhs.dot(rhs);
    if !rhs_norm_sq.is_finite() {
        return None;
    }
    if rhs_norm_sq <= f64::MIN_POSITIVE {
        return Some((Array1::<f64>::zeros(dim), 0, 0.0));
    }

    let target_sq = (rel_tol * rel_tol * rhs_norm_sq).max(f64::MIN_POSITIVE);
    let mut r = rhs.clone();
    if x.iter().any(|value| *value != 0.0) {
        let ax = apply(&x);
        if ax.len() != dim || !ax.iter().all(|value| value.is_finite()) {
            return None;
        }
        r.scaled_add(-1.0, &ax);
    }

    let mut rs_old = r.dot(&r);
    if !rs_old.is_finite() {
        return None;
    }
    if rs_old <= target_sq {
        return Some((x, 0, rs_old.max(0.0).sqrt()));
    }

    let mut p = r.clone();
    let mut iters = 0usize;
    let mut residual_norm = rs_old.max(0.0).sqrt();
    for k in 0..dim.max(1) {
        let ap = apply(&p);
        if ap.len() != dim || !ap.iter().all(|value| value.is_finite()) {
            return None;
        }
        let denom = p.dot(&ap);
        if !denom.is_finite() || denom <= 0.0 {
            log::warn!(
                "[CG-TRACE] non-positive curvature in trace CG at iter={} denom={}",
                k + 1,
                denom
            );
            break;
        }
        let alpha = rs_old / denom;
        if !alpha.is_finite() {
            return None;
        }
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return None;
        }
        iters = k + 1;
        residual_norm = rs_new.max(0.0).sqrt();
        if rs_new <= target_sq {
            break;
        }
        let beta = rs_new / rs_old;
        if !beta.is_finite() {
            return None;
        }
        p.mapv_inplace(|value| beta * value);
        p += &r;
        rs_old = rs_new;
    }

    Some((x, iters, residual_norm))
}

impl HessianFactorization for MatrixFreeSpdOperator {
    fn logdet(&self) -> f64 {
        *self
            .cached_logdet
            .get_or_compute(|| self.exact_dense_spectral().logdet())
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self.exact_dense_spectral())
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.exact_dense_spectral().trace_hinv_product(a)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.exact_dense_spectral().trace_hinv_operator(op)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.exact_dense_spectral().trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_hinv_operator_cross(left, right)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        let trace_start = std::time::Instant::now();
        let result = self.exact_dense_spectral().trace_logdet_operator(op);
        log::info!(
            "[STAGE] matrix_free_spd trace_logdet_operator implicit={} dim={} elapsed={:.3}s",
            op.is_implicit(),
            op.dim(),
            trace_start.elapsed().as_secs_f64(),
        );
        result
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.exact_dense_spectral().solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.exact_dense_spectral().solve_multi(rhs)
    }

    fn stochastic_trace_solve(&self, rhs: &Array1<f64>, rel_tol: f64) -> Array1<f64> {
        if self.use_trace_cg(rel_tol) {
            return self.cg_trace_solve(rhs, rel_tol, None, None);
        }
        self.solve(rhs)
    }

    fn stochastic_trace_solve_for_probe(
        &self,
        rhs: &Array1<f64>,
        rel_tol: f64,
        probe_id: u64,
        trace_state: Option<&Arc<Mutex<StochasticTraceState>>>,
    ) -> Array1<f64> {
        if self.use_trace_cg(rel_tol) {
            return self.cg_trace_solve(rhs, rel_tol, Some(probe_id), trace_state);
        }
        self.solve(rhs)
    }

    fn stochastic_trace_solve_multi(&self, rhs: &Array2<f64>, rel_tol: f64) -> Array2<f64> {
        if self.use_trace_cg(rel_tol) {
            let mut out = Array2::<f64>::zeros(rhs.raw_dim());
            for j in 0..rhs.ncols() {
                let solved = self.cg_trace_solve(&rhs.column(j).to_owned(), rel_tol, None, None);
                out.column_mut(j).assign(&solved);
            }
            return out;
        }
        self.solve_multi(rhs)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross_matrix_operator(h_i, h_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        self.exact_dense_spectral()
            .trace_logdet_hessian_cross_operator(h_i, h_j)
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    /// The operator delegates `logdet`, `trace_hinv_*`, `trace_logdet_*`,
    /// `solve`, and `solve_multi` to a lazily-built `DenseSpectralOperator`
    /// whenever the exact-dense materialization fits the configured byte cap
    /// (see `exact_dense_spectral_budget_ok` / `EXACT_DENSE_SPECTRAL_MAX_BYTES`).
    /// In that regime the algebra is exact spectral — there is no stochastic
    /// preference to advertise, and forcing the caller to take the Hutchinson
    /// path would replace an O(p²) exact reduction with O(k·apply) noisy probes.
    ///
    /// When the budget is exceeded the dense factor cannot be built and the
    /// CG trace-solve path added in 2bd6af68 is the only feasible route; the
    /// flag flips to `true` so `stochastic_trace_solve*` callers route through
    /// `cg_trace_solve` instead of crashing in `exact_dense_spectral().expect`.
    fn prefers_stochastic_trace_estimation(&self) -> bool {
        !self.exact_dense_spectral_budget_ok()
    }

    /// Mirror the `prefers_stochastic_trace_estimation` gate: when the dense
    /// factor is reachable the operator's logdet / trace_hinv reductions all
    /// resolve through `DenseSpectralOperator`, whose
    /// `logdet_traces_match_hinv_kernel` is `false` for the smooth-spectral
    /// regularization variants we run. Reporting `true` here would let the
    /// outer evaluator route logdet-gradient/Hessian traces through the
    /// Hutchinson `H⁻¹` kernel which does not satisfy
    /// `∂ log|H| = tr(H⁻¹ ∂H)` under smooth-spectral. The CG-only regime
    /// (budget exceeded) lacks a dense reference so falling back to the
    /// stochastic kernel is acceptable as a best-effort estimate.
    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        !self.exact_dense_spectral_budget_ok()
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.dense_spectral()
    }

    fn has_matrix_free_trace_cg_operator(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers for custom family → InnerSolution conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the square root of a symmetric positive semidefinite penalty matrix.
///
/// Returns R such that S = RᵀR, with R having `rank(S)` rows.
/// Uses eigendecomposition: S = U Λ U^T → R = Λ_+^{1/2} U_+^T.
pub fn penalty_matrix_root(s: &Array2<f64>) -> Result<Array2<f64>, String> {
    use faer::Side;
    let n = s.nrows();
    if n != s.ncols() {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "penalty_matrix_root: expected square matrix, got {}×{}",
                n,
                s.ncols()
            ),
        }
        .into());
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let (eigenvalues, eigenvectors) = s
        .eigh(Side::Lower)
        .map_err(|e| format!("penalty_matrix_root eigendecomposition failed: {e}"))?;

    let max_ev = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tol = (n.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);

    let active: Vec<usize> = eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > tol)
        .map(|(i, _)| i)
        .collect();
    let rank = active.len();

    let mut r = Array2::zeros((rank, n));
    for (out_row, &idx) in active.iter().enumerate() {
        let scale = eigenvalues[idx].sqrt();
        for col in 0..n {
            r[[out_row, col]] = scale * eigenvectors[[col, idx]];
        }
    }
    Ok(r)
}

/// Immutable, λ-independent geometry of one fixed collection of PSD penalty
/// components.
///
/// The structural range is built once from the sum of the component range
/// projectors, so it is invariant to each component's arbitrary physical
/// scale.  Every outer evaluation then factorizes the stacked, λ-scaled roots
/// in that fixed range with QR.  This retains the root-scale conditioning of
/// [`PenaltyPseudologdet`](super::super::penalty_logdet::PenaltyPseudologdet)'s
/// SVD construction without repeating component eigendecompositions or a
/// substantially more expensive SVD at every trial ρ.
#[derive(Debug)]
struct FixedPenaltyLogdetGeometry {
    /// Unit component roots expressed in the fixed structural range.
    reduced_roots: Vec<Array2<f64>>,
    /// Largest unit eigenvalue of each component. Used only to choose a common
    /// numerical scale before QR; it does not alter the represented penalty.
    component_scales: Vec<f64>,
    rank: usize,
    /// Exact unit pseudo-logdet when this geometry contains one component.
    singleton_unit_logdet: Option<f64>,
}

impl FixedPenaltyLogdetGeometry {
    fn new(components: &[Array2<f64>]) -> Result<Self, String> {
        use gam_linalg::faer_ndarray::{FaerEigh, fast_ab};

        if components.is_empty() {
            return Ok(Self {
                reduced_roots: Vec::new(),
                component_scales: Vec::new(),
                rank: 0,
                singleton_unit_logdet: None,
            });
        }

        let p = components[0].nrows();
        if components
            .iter()
            .any(|component| component.nrows() != p || component.ncols() != p)
        {
            return Err(
                "penalty-logdet geometry requires equally-sized square component matrices"
                    .to_string(),
            );
        }

        let mut ambient_roots = Vec::with_capacity(components.len());
        let mut component_scales = Vec::with_capacity(components.len());
        let mut component_unit_logdets = Vec::with_capacity(components.len());
        let mut structural_projector = Array2::<f64>::zeros((p, p));

        for (component_index, component) in components.iter().enumerate() {
            if !component.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "penalty-logdet component {component_index} contains a non-finite entry"
                ));
            }
            let (eigenvalues, eigenvectors) = component.eigh(faer::Side::Lower).map_err(|error| {
                format!(
                    "penalty-logdet component {component_index} eigendecomposition failed: {error}"
                )
            })?;
            let eigenvalue_slice = eigenvalues
                .as_slice()
                .expect("eigh returns contiguous eigenvalues");
            let threshold = positive_eigenvalue_threshold(eigenvalue_slice);
            if let Some(negative) = eigenvalues
                .iter()
                .copied()
                .find(|value| *value < -threshold)
            {
                return Err(format!(
                    "penalty-logdet component {component_index} is indefinite: eigenvalue {negative} is below the PSD noise band {}",
                    -threshold
                ));
            }

            let active: Vec<usize> = eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value > threshold).then_some(index))
                .collect();
            let mut root = Array2::<f64>::zeros((active.len(), p));
            let mut unit_logdet = 0.0;
            for (root_row, &eigen_index) in active.iter().enumerate() {
                let eigenvalue = eigenvalues[eigen_index];
                unit_logdet += eigenvalue.ln();
                let root_scale = eigenvalue.sqrt();
                for row in 0..p {
                    let basis_value = eigenvectors[[row, eigen_index]];
                    root[[root_row, row]] = root_scale * basis_value;
                    for col in 0..p {
                        structural_projector[[row, col]] +=
                            basis_value * eigenvectors[[col, eigen_index]];
                    }
                }
            }
            component_scales.push(
                active
                    .iter()
                    .map(|&index| eigenvalues[index])
                    .fold(0.0_f64, f64::max),
            );
            component_unit_logdets.push(unit_logdet);
            ambient_roots.push(root);
        }

        if p == 0 {
            return Ok(Self {
                reduced_roots: ambient_roots,
                component_scales,
                rank: 0,
                singleton_unit_logdet: (components.len() == 1).then(|| component_unit_logdets[0]),
            });
        }

        let (range_eigenvalues, range_eigenvectors) = structural_projector
            .eigh(faer::Side::Lower)
            .map_err(|error| {
                format!("penalty-logdet structural-range eigendecomposition failed: {error}")
            })?;
        let range_threshold = positive_eigenvalue_threshold(
            range_eigenvalues
                .as_slice()
                .expect("eigh returns contiguous eigenvalues"),
        );
        let range_indices: Vec<usize> = range_eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > range_threshold).then_some(index))
            .collect();
        let rank = range_indices.len();
        let mut range_basis = Array2::<f64>::zeros((p, rank));
        for (range_col, &eigen_index) in range_indices.iter().enumerate() {
            range_basis
                .column_mut(range_col)
                .assign(&range_eigenvectors.column(eigen_index));
        }

        let reduced_roots = ambient_roots
            .iter()
            .map(|root| {
                if root.nrows() == 0 || rank == 0 {
                    Array2::<f64>::zeros((root.nrows(), rank))
                } else {
                    fast_ab(root, &range_basis)
                }
            })
            .collect();

        Ok(Self {
            reduced_roots,
            component_scales,
            rank,
            singleton_unit_logdet: (components.len() == 1).then(|| component_unit_logdets[0]),
        })
    }

    fn evaluate(
        &self,
        lambdas: &[f64],
        ridge: f64,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        use gam_linalg::faer_ndarray::{FaerQr, fast_ab, fast_atb};

        if lambdas.len() != self.reduced_roots.len() {
            return Err(format!(
                "penalty-logdet geometry has {} components but received {} lambdas",
                self.reduced_roots.len(),
                lambdas.len()
            ));
        }
        if !(ridge.is_finite() && ridge >= 0.0) {
            return Err(format!(
                "penalty-logdet ridge must be finite and nonnegative, got {ridge}"
            ));
        }
        for (index, &lambda) in lambdas.iter().enumerate() {
            if !(lambda.is_finite() && lambda > 0.0) {
                return Err(format!(
                    "penalty-logdet lambda {index} must be finite and positive, got {lambda}"
                ));
            }
        }

        let component_count = lambdas.len();
        if self.rank == 0 {
            return Ok((
                0.0,
                Array1::zeros(component_count),
                Array2::zeros((component_count, component_count)),
            ));
        }

        // A singleton un-ridged factor is exactly affine in log λ. Besides
        // avoiding any factorization, this returns the algebraic zero Hessian
        // rather than a roundoff-sized approximation to zero.
        if component_count == 1
            && ridge == 0.0
            && let Some(unit_logdet) = self.singleton_unit_logdet
        {
            return Ok((
                unit_logdet + (self.rank as f64) * lambdas[0].ln(),
                Array1::from_elem(1, self.rank as f64),
                Array2::zeros((1, 1)),
            ));
        }

        // Scale the entire stacked root by one common physical precision.
        // The represented matrix is unchanged after adding rank·log(scale) to
        // the QR logdet, while every root entry stays near unit magnitude.
        let common_scale = lambdas
            .iter()
            .zip(&self.component_scales)
            .map(|(&lambda, &unit_scale)| lambda * unit_scale)
            .chain(std::iter::once(ridge))
            .fold(0.0_f64, f64::max);
        if !(common_scale.is_finite() && common_scale > 0.0) {
            return Err(format!(
                "penalty-logdet fixed structural range has nonpositive numerical scale {common_scale}"
            ));
        }

        let root_rows: usize = self.reduced_roots.iter().map(Array2::nrows).sum();
        let ridge_rows = usize::from(ridge > 0.0) * self.rank;
        let mut stacked = Array2::<f64>::zeros((root_rows + ridge_rows, self.rank));
        let mut row_offset = 0;
        for (&lambda, root) in lambdas.iter().zip(&self.reduced_roots) {
            let scale = (lambda / common_scale).sqrt();
            let end = row_offset + root.nrows();
            stacked
                .slice_mut(ndarray::s![row_offset..end, ..])
                .assign(&root.mapv(|value| scale * value));
            row_offset = end;
        }
        if ridge > 0.0 {
            let scale = (ridge / common_scale).sqrt();
            for index in 0..self.rank {
                stacked[[row_offset + index, index]] = scale;
            }
        }

        let (_, upper) = stacked
            .qr()
            .map_err(|error| format!("penalty-logdet root-scale QR failed: {error}"))?;
        if upper.dim() != (self.rank, self.rank) {
            return Err(format!(
                "penalty-logdet root-scale QR returned {}x{} R for structural rank {}",
                upper.nrows(),
                upper.ncols(),
                self.rank
            ));
        }

        let mut logdet = (self.rank as f64) * common_scale.ln();
        for index in 0..self.rank {
            let diagonal = upper[[index, index]].abs();
            if !(diagonal.is_finite() && diagonal > 0.0) {
                return Err(format!(
                    "penalty-logdet root-scale QR produced invalid diagonal {diagonal} at {index}"
                ));
            }
            logdet += 2.0 * diagonal.ln();
        }

        // RᵀR is the common-scale-normalized precision. R⁻¹R⁻ᵀ is therefore
        // its inverse, and all ρ derivatives can be evaluated with the
        // dimensionless weights λ/common_scale without ever squaring the
        // condition number in an assembled matrix.
        let mut inverse_upper = Array2::<f64>::zeros((self.rank, self.rank));
        for rhs_col in 0..self.rank {
            for reverse_row in 0..self.rank {
                let row = self.rank - 1 - reverse_row;
                let mut residual = f64::from(row == rhs_col);
                for col in (row + 1)..self.rank {
                    residual -= upper[[row, col]] * inverse_upper[[col, rhs_col]];
                }
                inverse_upper[[row, rhs_col]] = residual / upper[[row, row]];
            }
        }

        let projected_components: Vec<Array2<f64>> = self
            .reduced_roots
            .iter()
            .map(|root| {
                let transformed = fast_ab(root, &inverse_upper);
                fast_atb(&transformed, &transformed)
            })
            .collect();
        let scaled_lambdas: Vec<f64> = lambdas.iter().map(|lambda| lambda / common_scale).collect();
        let mut first = Array1::<f64>::zeros(component_count);
        for k in 0..component_count {
            first[k] = scaled_lambdas[k]
                * (0..self.rank)
                    .map(|index| projected_components[k][[index, index]])
                    .sum::<f64>();
        }
        let mut second = Array2::<f64>::zeros((component_count, component_count));
        for k in 0..component_count {
            for l in 0..component_count {
                let cross = super::super::penalty_logdet::PenaltyPseudologdet::trace_dense_product(
                    &projected_components[k],
                    &projected_components[l],
                );
                second[[k, l]] = if k == l { first[k] } else { 0.0 }
                    - scaled_lambdas[k] * scaled_lambdas[l] * cross;
            }
        }
        Ok((logdet, first, second))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DensePenaltyGeometryKey {
    rows: usize,
    cols: usize,
    values: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BlockPenaltyGeometryKey {
    blocks: Vec<Vec<DensePenaltyGeometryKey>>,
    prior_factor_masks: Vec<Vec<bool>>,
}

impl BlockPenaltyGeometryKey {
    fn new(per_block_penalties: &[&[Array2<f64>]], prior_factor_masks: &[Vec<bool>]) -> Self {
        Self {
            blocks: per_block_penalties
                .iter()
                .map(|block| {
                    block
                        .iter()
                        .map(|matrix| DensePenaltyGeometryKey {
                            rows: matrix.nrows(),
                            cols: matrix.ncols(),
                            values: matrix.iter().map(|value| value.to_bits()).collect(),
                        })
                        .collect()
                })
                .collect(),
            prior_factor_masks: prior_factor_masks.to_vec(),
        }
    }
}

#[derive(Debug)]
struct PenaltyLogdetGroupGeometry {
    coordinate_indices: Vec<usize>,
    geometry: FixedPenaltyLogdetGeometry,
}

#[derive(Debug)]
struct PenaltyLogdetBlockGeometry {
    coordinate_count: usize,
    groups: Vec<PenaltyLogdetGroupGeometry>,
}

#[derive(Debug)]
struct BlockPenaltyLogdetGeometry {
    blocks: Vec<PenaltyLogdetBlockGeometry>,
}

impl BlockPenaltyLogdetGeometry {
    fn new(
        per_block_penalties: &[&[Array2<f64>]],
        prior_factor_masks: &[Vec<bool>],
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(per_block_penalties.len());
        for (block_index, penalties) in per_block_penalties.iter().enumerate() {
            let mask = &prior_factor_masks[block_index];
            let coalesced_indices: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(index, &is_factor)| (!is_factor).then_some(index))
                .collect();
            let mut groups = Vec::new();
            if !coalesced_indices.is_empty() {
                let components: Vec<Array2<f64>> = coalesced_indices
                    .iter()
                    .map(|&index| penalties[index].clone())
                    .collect();
                groups.push(PenaltyLogdetGroupGeometry {
                    coordinate_indices: coalesced_indices,
                    geometry: FixedPenaltyLogdetGeometry::new(&components)
                        .map_err(|error| format!("penalty-logdet block {block_index}: {error}"))?,
                });
            }
            for (index, &is_factor) in mask.iter().enumerate() {
                if is_factor {
                    groups.push(PenaltyLogdetGroupGeometry {
                        coordinate_indices: vec![index],
                        geometry: FixedPenaltyLogdetGeometry::new(std::slice::from_ref(
                            &penalties[index],
                        ))
                        .map_err(|error| {
                            format!(
                                "penalty-logdet block {block_index} prior factor {index}: {error}"
                            )
                        })?,
                    });
                }
            }
            blocks.push(PenaltyLogdetBlockGeometry {
                coordinate_count: penalties.len(),
                groups,
            });
        }
        Ok(Self { blocks })
    }

    fn evaluate(
        &self,
        per_block_rho: &[Array1<f64>],
        ridge: f64,
    ) -> Result<PenaltyLogdetDerivs, String> {
        if per_block_rho.len() != self.blocks.len() {
            return Err(format!(
                "penalty-logdet geometry has {} blocks but received {} rho blocks",
                self.blocks.len(),
                per_block_rho.len()
            ));
        }

        struct BlockResult {
            offset: usize,
            value: f64,
            first: Array1<f64>,
            second: Array2<f64>,
        }

        let offsets: Vec<usize> = self
            .blocks
            .iter()
            .scan(0usize, |offset, block| {
                let current = *offset;
                *offset += block.coordinate_count;
                Some(current)
            })
            .collect();
        let evaluate_block = |(block_index, block): (usize, &PenaltyLogdetBlockGeometry)| {
            let rho = &per_block_rho[block_index];
            if rho.len() != block.coordinate_count {
                return Err(format!(
                    "penalty-logdet block {block_index} has {} components but received {} rho coordinates",
                    block.coordinate_count,
                    rho.len()
                ));
            }
            let lambdas = gam_problem::checked_exp_log_strengths(rho.iter().copied())
                .map_err(|error| format!("penalty-logdet block {block_index}: {error}"))?;
            let mut value = 0.0;
            let mut first = Array1::<f64>::zeros(block.coordinate_count);
            let mut second = Array2::<f64>::zeros((block.coordinate_count, block.coordinate_count));
            for group in &block.groups {
                let group_lambdas: Vec<f64> = group
                    .coordinate_indices
                    .iter()
                    .map(|&index| lambdas[index])
                    .collect();
                let (group_value, group_first, group_second) = group
                    .geometry
                    .evaluate(&group_lambdas, ridge)
                    .map_err(|error| format!("penalty-logdet block {block_index}: {error}"))?;
                value += group_value;
                for (local_k, &global_k) in group.coordinate_indices.iter().enumerate() {
                    first[global_k] = group_first[local_k];
                    for (local_l, &global_l) in group.coordinate_indices.iter().enumerate() {
                        second[[global_k, global_l]] = group_second[[local_k, local_l]];
                    }
                }
            }
            Ok(BlockResult {
                offset: offsets[block_index],
                value,
                first,
                second,
            })
        };

        let block_results: Vec<BlockResult> = if rayon::current_thread_index().is_some() {
            self.blocks
                .iter()
                .enumerate()
                .map(evaluate_block)
                .collect::<Result<Vec<_>, String>>()?
        } else {
            self.blocks
                .par_iter()
                .enumerate()
                .map(evaluate_block)
                .collect::<Result<Vec<_>, String>>()?
        };

        let total_coordinates: usize = self.blocks.iter().map(|block| block.coordinate_count).sum();
        let mut value = 0.0;
        let mut first = Array1::<f64>::zeros(total_coordinates);
        let mut second = Array2::<f64>::zeros((total_coordinates, total_coordinates));
        for block in block_results {
            value += block.value;
            for k in 0..block.first.len() {
                first[block.offset + k] = block.first[k];
                for l in 0..block.first.len() {
                    second[[block.offset + k, block.offset + l]] = block.second[[k, l]];
                }
            }
        }
        Ok(PenaltyLogdetDerivs {
            value,
            first,
            second: Some(second),
        })
    }
}

std::thread_local! {
    /// One exact penalty-layout geometry per calling thread. Outer iterations
    /// execute serially on one driver thread, so this retains the current fit's
    /// immutable geometry without a process-global, ever-growing model cache.
    static BLOCK_PENALTY_LOGDET_GEOMETRY:
        std::cell::RefCell<Option<(BlockPenaltyGeometryKey, std::sync::Arc<BlockPenaltyLogdetGeometry>)>>
        = const { std::cell::RefCell::new(None) };
}

/// [`compute_block_penalty_logdet_derivs`] with per-penalty prior-factor
/// structure.
///
/// `prior_factor_mask[b][k] == true` declares block `b`'s penalty `k` an
/// INDEPENDENT Gaussian prior factor rather than an additive piece of one
/// smooth prior. The evidence normalizer of one Gaussian with precision
/// `Σ_k λ_k S_k` is the coalesced `log|Σ_k λ_k S_k|₊` (the default, and the
/// correct convention for multi-penalty smooths), but a PRODUCT of
/// independent factors `∏_k N(0, (λ_k S_k)⁻¹)` contributes
///
/// ```text
/// Σ_k log|λ_k S_k|₊ = Σ_k ( rank(S_k)·ρ_k + log|S_k|₊ ),
/// ```
///
/// which differs from the coalesced form exactly when factors overlap: two
/// factors with precision λ on one scalar coefficient carry
/// `λ^{1/2}·λ^{1/2} = λ`, while coalescing their quadratics into `2λβ²` and
/// taking one normalizer yields `(2λ)^{1/2}` — losing `½ log λ` from the
/// outer ρ-posterior (hierarchical coefficient groups, audit finding 40).
/// Each masked penalty therefore becomes its own singleton pseudo-logdet
/// block; unmasked penalties within the block coalesce as before. `None`
/// masks (or an all-false mask) reproduce the coalesced behaviour exactly.
pub fn compute_block_penalty_logdet_derivs_with_prior_factors(
    per_block_rho: &[Array1<f64>],
    per_block_penalties: &[&[Array2<f64>]],
    prior_factor_mask: Option<&[Vec<bool>]>,
    ridge: f64,
) -> Result<PenaltyLogdetDerivs, String> {
    if per_block_rho.len() != per_block_penalties.len() {
        return Err(format!(
            "penalty-logdet received {} rho blocks and {} penalty blocks",
            per_block_rho.len(),
            per_block_penalties.len()
        ));
    }
    let masks = match prior_factor_mask {
        Some(masks) => {
            if masks.len() != per_block_penalties.len() {
                return Err(format!(
                    "penalty-logdet received {} prior-factor masks for {} penalty blocks",
                    masks.len(),
                    per_block_penalties.len()
                ));
            }
            for (block, (mask, penalties)) in masks.iter().zip(per_block_penalties).enumerate() {
                if mask.len() != penalties.len() {
                    return Err(format!(
                        "penalty-logdet block {block} has {} penalties but {} prior-factor flags",
                        penalties.len(),
                        mask.len()
                    ));
                }
            }
            masks.to_vec()
        }
        None => per_block_penalties
            .iter()
            .map(|penalties| vec![false; penalties.len()])
            .collect(),
    };
    let key = BlockPenaltyGeometryKey::new(per_block_penalties, &masks);
    let geometry = BLOCK_PENALTY_LOGDET_GEOMETRY.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|(cached_key, _)| cached_key == &key)
            .map(|(_, geometry)| std::sync::Arc::clone(geometry))
    });
    let geometry = match geometry {
        Some(geometry) => geometry,
        None => {
            let geometry = std::sync::Arc::new(BlockPenaltyLogdetGeometry::new(
                per_block_penalties,
                &masks,
            )?);
            BLOCK_PENALTY_LOGDET_GEOMETRY.with(|slot| {
                *slot.borrow_mut() = Some((key, std::sync::Arc::clone(&geometry)));
            });
            geometry
        }
    };
    geometry.evaluate(per_block_rho, ridge)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stochastic trace estimation via Rademacher probes
// ═══════════════════════════════════════════════════════════════════════════
//
// For large-scale models, computing tr(H⁻¹ A_k) exactly via the full p×p
// eigendecomposition or column-by-column sparse solves costs O(p²) per
// coordinate k.  Stochastic trace estimation gives an unbiased estimate
// using only matrix–vector products (solves), at cost O(M·p) where M is the
// number of random probe vectors (typically 10–200).
//
// The Girard–Hutchinson estimator:
//
//   tr(H⁻¹ A_k) ≈ (1/M) Σ_m  z_mᵀ H⁻¹ A_k z_m
//
// where z_m are i.i.d. random vectors with E[zzᵀ] = I.
//
// Rademacher probes (entries ±1 with equal probability) have strictly
// lower variance than Gaussian probes:
//   Var_Rad = 2(‖S‖²_F − Σ_i S²_{ii})
//   Var_Gau = 2‖S‖²_F
// where S = sym(H⁻¹ A_k).  The diagonal variance term is always removed.
//
// Key efficiency: ONE H⁻¹ solve per probe, shared across ALL k
// coordinates.  For each probe z we compute w = H⁻¹z once, then for each k
// we get q_k = zᵀ(A_k w) with a cheap matrix–vector multiply.
