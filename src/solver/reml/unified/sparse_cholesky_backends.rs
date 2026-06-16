use super::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Cholesky HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse Cholesky Hessian operator.
///
/// Wraps an existing `SparseExactFactor` and provides logdet, trace, and solve
/// from the same Cholesky factorization.
pub struct SparseCholeskyOperator {
    /// The sparse Cholesky factorization.
    pub(crate) factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
    /// Takahashi selected inverse (precomputed H^{-1} entries on the filled pattern of L).
    /// When available, trace computations use direct lookups instead of column solves.
    pub(crate) takahashi: Option<std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>>,
    /// Precomputed log-determinant from the Cholesky diagonal.
    pub(crate) cached_logdet: f64,
    /// Dimension of H.
    pub(crate) n_dim: usize,
}

impl SparseCholeskyOperator {
    /// Create from an existing sparse factorization and its precomputed logdet.
    pub fn new(
        factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
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
        taka: std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>,
    ) -> Self {
        self.takahashi = Some(taka);
        self
    }

    pub(crate) const OPERATOR_SOLVE_CHUNK: usize = 64;

    pub(crate) fn takahashi_block_trace(
        taka: &crate::linalg::sparse_exact::TakahashiInverse,
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
        taka: &crate::linalg::sparse_exact::TakahashiInverse,
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
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_block,
                    start,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
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
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_block,
                    row_start,
                    row_end,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
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

    pub(crate) fn fill_scaled_block_columns(
        block: &Array2<f64>,
        scale: f64,
        block_start: usize,
        local_col_start: usize,
        cols: usize,
        mut rhs_block: ndarray::ArrayViewMut2<'_, f64>,
    ) {
        let block_end = block_start + block.nrows();
        let source = block.slice(ndarray::s![.., local_col_start..local_col_start + cols]);
        let mut target = rhs_block.slice_mut(ndarray::s![block_start..block_end, ..cols]);
        if scale == 1.0 {
            target.assign(&source);
        } else {
            Zip::from(target)
                .and(source)
                .for_each(|dst, &value| *dst = scale * value);
        }
    }

    pub(crate) fn trace_hinv_block_local_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if scale == 0.0 {
            return 0.0;
        }
        assert_eq!(block.nrows(), end - start);
        let t_start = std::time::Instant::now();
        let block_size = end - start;
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(block_size.max(1));
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut trace = 0.0;
        let mut local_col_start = 0usize;

        while local_col_start < block_size {
            let cols = (block_size - local_col_start).min(chunk);
            Self::fill_scaled_block_columns(
                block,
                scale,
                start,
                local_col_start,
                cols,
                rhs_block.view_mut(),
            );
            let diagonal_sum = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_block,
                    start + local_col_start,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_diagonal_sum(
                    &self.factor,
                    &rhs_view,
                    start + local_col_start,
                )
            };
            trace += diagonal_sum.unwrap_or_else(|e| {
                // SAFETY: same invariant as `trace_hinv_operator_exact`
                // above — `self.factor` is the validated SPD factor;
                // sparse-SPD multi-RHS solves only fail on factor
                // corruption, which `SparseCholeskyOperator`'s
                // construction invariant forbids.
                // SAFETY: self.factor is validated SPD; block-local solve only fails on factor corruption.
                reml_contract_panic(format!(
                    "SparseCholeskyOperator exact block-local trace solve failed: {e}"
                ))
            });
            local_col_start += cols;
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > REML_TRACE_SLOW_LOG_MS {
            log::info!(
                "[REML-trace] block_local_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                block_size,
                elapsed_ms
            );
        }
        trace
    }

    pub(crate) fn solve_block_local_rows_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> Result<Array2<f64>, String> {
        assert_eq!(block.nrows(), end - start);
        let block_size = end - start;
        let chunk = Self::OPERATOR_SOLVE_CHUNK.min(block_size.max(1));
        let mut solved = Array2::<f64>::zeros((block_size, block_size));
        if scale == 0.0 {
            return Ok(solved);
        }
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut local_col_start = 0usize;

        while local_col_start < block_size {
            let cols = (block_size - local_col_start).min(chunk);
            Self::fill_scaled_block_columns(
                block,
                scale,
                start,
                local_col_start,
                cols,
                rhs_block.view_mut(),
            );
            let solved_block = if cols == chunk {
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_block,
                    start,
                    end,
                )
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti_rows(
                    &self.factor,
                    &rhs_view,
                    start,
                    end,
                )
            }
            .map_err(|e| {
                format!(
                    "SparseCholeskyOperator::solve_block_local_rows_exact multi-solve failed: {e}"
                )
            })?;
            solved
                .slice_mut(ndarray::s![.., local_col_start..local_col_start + cols])
                .assign(&solved_block);
            local_col_start += cols;
        }

        Ok(solved)
    }

    pub(crate) fn trace_hinv_block_local_cross_exact(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let t_start = std::time::Instant::now();
        let solved = self
            .solve_block_local_rows_exact(block, scale, start, end)
            // SAFETY: same SPD-factor invariant as the trace solves above
            // — `self.factor` is the validated SPD factorization;
            // failure here means factor corruption, forbidden by the
            // `SparseCholeskyOperator` construction invariant.
            .unwrap_or_else(|e| {
                // SAFETY: self.factor is validated SPD; cross solve only fails on factor corruption.
                panic!("SparseCholeskyOperator exact block-local cross solve failed: {e}")
            });
        let result = trace_matrix_product(&solved, &solved);
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        if elapsed_ms > REML_TRACE_SLOW_LOG_MS {
            log::info!(
                "[REML-trace] block_local_cross_exact | n_dim={} | block={} | {:.1}ms",
                self.n_dim,
                end - start,
                elapsed_ms
            );
        }
        result
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
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_block)
            } else {
                let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
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
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &op_rhs_block)
            } else {
                let rhs_view = op_rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
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
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &eye_rhs_block)
            } else {
                let rhs_view = eye_rhs_block.slice(ndarray::s![.., ..cols]);
                crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, &rhs_view)
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

impl HessianOperator for SparseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        let h = crate::linalg::sparse_exact::assemble_sparse_factor_h_dense(&self.factor)
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
        crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, a)
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

    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if let Some(ref taka) = self.takahashi {
            assert_eq!(block.nrows(), end - start);
            return scale * Self::takahashi_block_trace(taka, block, start);
        }
        self.trace_hinv_block_local_exact(block, scale, start, end)
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        if let Some(ref taka) = self.takahashi {
            assert_eq!(block.nrows(), end - start);
            let za = Self::takahashi_left_multiply_block(taka, block, start);
            return scale * scale * trace_matrix_product(&za, &za);
        }
        self.trace_hinv_block_local_cross_exact(block, scale, start, end)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // SAFETY: `self.factor` is the validated SPD Cholesky factor stored
        // at construction time; a triangular solve against an already-built
        // factor can only fail on factor corruption, which the
        // `SparseCholeskyOperator` construction invariant forbids.
        crate::linalg::sparse_exact::solve_sparse_spd(&self.factor, rhs)
            // SAFETY: self.factor is validated SPD; triangular solve only fails on corruption.
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact solve failed: {e}"))
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        // SAFETY: same SPD-factor invariant as `solve` above — `self.factor`
        // was created from a successful Cholesky factorization, so a
        // multi-RHS solve can only fail on factor corruption.
        crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, rhs)
            // SAFETY: self.factor is validated SPD; multi-RHS solve only fails on corruption.
            .unwrap_or_else(|e| panic!("SparseCholeskyOperator exact multi-solve failed: {e}"))
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        // For general dense matrices, column solves are better than materializing
        // full Z from Takahashi (O(p * nnz) vs O(p³)). Takahashi cross-traces
        // are only used for block-local operators via trace_hinv_operator_cross.
        let solved_a = self.solve_multi(a);
        if std::ptr::eq(a, b) {
            return trace_matrix_product(&solved_a, &solved_a);
        }
        let solved_b = self.solve_multi(b);
        trace_matrix_product(&solved_a, &solved_b)
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
                return trace_matrix_product(&za, &za);
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
//  Cholesky-backed value-only HessianOperator (logdet + solve, no traces)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense Cholesky-backed [`HessianOperator`] for `EvalMode::ValueOnly` paths.
///
/// When the penalized Hessian is known to be SPD (no Firth bias reduction, no
/// hard linear constraints, no `HardPseudo` mode), the REML/LAML cost needs
/// only two Hessian services:
///
/// - `logdet()` — used directly in the `½ log|H|` cost term.
/// - `solve(rhs)` / `solve_multi(rhs)` — used for the optional IFT
///   cost correction `−½ rᵀ H⁻¹ r`.
///
/// An LLT Cholesky factorization delivers both in `O(p³/3)` flops versus
/// the `O(9·p³)` full eigendecomposition of [`DenseSpectralOperator`], giving
/// a multi-× speedup per outer REML line-search probe.
///
/// Gradient traces (`trace_hinv_product`) are satisfied via column-by-column
/// forward/back solves so that the operator remains valid if the evaluator
/// ever reaches a gradient path unexpectedly. Under normal use
/// `EvalMode::ValueOnly` returns before any trace call.
pub struct DenseCholeskyValueOnlyOperator {
    /// LLT Cholesky factor.
    pub(crate) chol: crate::faer_ndarray::FaerCholeskyFactor,
    /// `2 · Σ ln(diag L)` — cached at construction time.
    pub(crate) cached_logdet: f64,
    /// Full parameter dimension.
    pub(crate) n_dim: usize,
}

impl DenseCholeskyValueOnlyOperator {
    /// Factorize `h` (assumed SPD) via LLT and cache the log-determinant.
    ///
    /// Returns `Err` if `h` is not square, not SPD, or contains non-finite
    /// entries. Callers should fall back to [`DenseSpectralOperator`] on
    /// failure (e.g. near-singular Hessians that need soft regularization).
    pub fn from_spd(h: &Array2<f64>) -> Result<Self, String> {
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(format!(
                "DenseCholeskyValueOnlyOperator: expected square matrix, got {}×{}",
                n,
                h.ncols()
            ));
        }
        let chol = h
            .cholesky(Side::Lower)
            .map_err(|e| format!("DenseCholeskyValueOnlyOperator LLT failed: {e}"))?;
        let diag = chol.diag();
        let cached_logdet = 2.0 * diag.iter().map(|&d| d.ln()).sum::<f64>();
        Ok(Self {
            chol,
            cached_logdet,
            n_dim: n,
        })
    }
}

impl HessianOperator for DenseCholeskyValueOnlyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H⁻¹ A) = Σ_j [H⁻¹ A]_jj.
        // Compute H⁻¹ A via multi-column solve and sum the diagonal.
        let hinv_a = self.chol.solve_mat(a);
        hinv_a.diag().iter().sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.chol.solvevec(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.chol.solve_mat(rhs)
    }

    fn active_rank(&self) -> usize {
        // LLT succeeded ⟹ all pivots are positive ⟹ full rank.
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Block-coupled HessianOperator for joint multi-block models
// ═══════════════════════════════════════════════════════════════════════════

/// Block-coupled Hessian operator for joint multi-block models (GAMLSS, survival).
///
/// Wraps a [`DenseSpectralOperator`] over the full assembled joint Hessian while
/// retaining block-structure metadata. All [`HessianOperator`] trait methods
/// delegate to the inner spectral decomposition, ensuring a single
/// eigendecomposition governs logdet, trace, and solve.
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
pub struct BlockCoupledOperator {
    /// Inner spectral operator over the full joint Hessian.
    pub(crate) inner: DenseSpectralOperator,
}

impl BlockCoupledOperator {
    /// Construct from an assembled joint Hessian using the supplied
    /// [`PseudoLogdetMode`].  Internally performs a single
    /// eigendecomposition of `joint_hessian`.
    pub fn from_joint_hessian_with_mode(
        joint_hessian: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        let inner = DenseSpectralOperator::from_symmetric_with_mode(joint_hessian, mode)
            .map_err(|e| format!("BlockCoupledOperator eigendecomposition: {e}"))?;

        Ok(Self { inner })
    }
}

impl HessianOperator for BlockCoupledOperator {
    fn logdet(&self) -> f64 {
        self.inner.logdet()
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.inner.as_exact_dense_spectral()
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        self.inner.assemble_h_dense_for_tangent_projection()
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product(a)
    }

    fn trace_hinv_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_hinv_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_gradient(a)
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        self.inner.xt_logdet_kernel_x_diagonal(x)
    }

    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_logdet_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.inner.trace_logdet_operator(op)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_hessian_cross(h_i, h_j)
    }

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        self.inner.trace_logdet_hessian_crosses(matrices)
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        self.inner
            .trace_hinv_block_local_cross(block, scale, start, end)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.inner.solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.inner.solve_multi(rhs)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product_cross(a, b)
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_matrix_operator_cross(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        self.inner.trace_hinv_operator_cross(left, right)
    }

    fn active_rank(&self) -> usize {
        self.inner.active_rank()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(&self.inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix-free SPD HessianOperator implementation
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
    pub(crate) cached_logdet: crate::solver::resource::RayonSafeOnce<f64>,
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
    pub(crate) dense_spectral: crate::solver::resource::RayonSafeOnce<Option<DenseSpectralOperator>>,
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

    /// Like [`new_with_mode`], but additionally accepts an optional single-pass
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
            cached_logdet: crate::solver::resource::RayonSafeOnce::new(),
            n_dim: dim,
            dense_spectral: crate::solver::resource::RayonSafeOnce::new(),
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
        let result = DenseSpectralOperator::from_symmetric_with_mode(&matrix, self.mode).ok();
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

impl HessianOperator for MatrixFreeSpdOperator {
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

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        self.exact_dense_spectral()
            .trace_logdet_hessian_crosses(matrices)
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

/// Compute the exact pseudo-logdet log|S|₊ and its ρ-derivatives for a
/// blockwise penalty structure.
///
/// For each block, eigendecomposes S_b = Σ λ_k S_k, identifies the positive
/// eigenspace (structural nullspace detected from the eigenspectrum), and
/// computes exact derivatives on that subspace:
///
/// - L(S) = Σ_{σ_i > ε} log σ_i
/// - ∂/∂ρₖ L = tr(S⁺ Aₖ)
/// - ∂²/(∂ρₖ∂ρₗ) L = δ_{kl} ∂_k L − tr(S⁺ Aₗ S⁺ Aₖ)
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
/// is structurally fixed (independent of ρ), so L is C∞ in ρ and these are
/// its exact derivatives.
///
/// `per_block_rho[b]` contains the log-lambdas for block b.
/// `per_block_penalties[b]` contains the penalty matrices for block b.
/// `ridge` is an additional ridge for logdet stability (0 if not applicable).
pub fn compute_block_penalty_logdet_derivs(
    per_block_rho: &[Array1<f64>],
    per_block_penalties: &[&[Array2<f64>]],
    ridge: f64,
) -> Result<PenaltyLogdetDerivs, String> {
    use super::super::penalty_logdet::PenaltyPseudologdet;

    let total_k: usize = per_block_rho.iter().map(|r| r.len()).sum();
    let block_offsets: Vec<usize> = per_block_rho
        .iter()
        .scan(0usize, |at, rho| {
            let current = *at;
            *at += rho.len();
            Some(current)
        })
        .collect();

    struct BlockPenaltyLogdetResult {
        pub(crate) offset: usize,
        pub(crate) value: f64,
        pub(crate) first: Array1<f64>,
        pub(crate) second: Array2<f64>,
    }

    let compute_block = |(b, block_rho): (usize, &Array1<f64>)| {
        let penalties = per_block_penalties[b];
        let kb = block_rho.len();
        if penalties.is_empty() || kb == 0 {
            return Ok(BlockPenaltyLogdetResult {
                offset: block_offsets[b],
                value: 0.0,
                first: Array1::zeros(kb),
                second: Array2::zeros((kb, kb)),
            });
        }
        let lambdas: Vec<f64> = block_rho.iter().map(|&r| r.exp()).collect();

        // Single eigendecomposition via canonical PenaltyPseudologdet.
        //
        // No metadata-based structural-nullity hint: the classifier derives
        // the positive eigenspace from the assembled spectrum alone (issues
        // #192/#318).
        let pld = PenaltyPseudologdet::from_components(penalties, &lambdas, ridge)
            .map_err(|e| format!("penalty logdet failed for block {b}: {e}"))?;

        let value = pld.value();
        let (first, second) = pld.rho_derivatives(penalties, &lambdas);
        Ok(BlockPenaltyLogdetResult {
            offset: block_offsets[b],
            value,
            first,
            second,
        })
    };

    let block_results: Vec<BlockPenaltyLogdetResult> = if rayon::current_thread_index().is_some() {
        per_block_rho
            .iter()
            .enumerate()
            .map(compute_block)
            .collect::<Result<Vec<_>, String>>()?
    } else {
        per_block_rho
            .par_iter()
            .enumerate()
            .map(compute_block)
            .collect::<Result<Vec<_>, String>>()?
    };

    let mut log_det_total = 0.0;
    let mut first = Array1::zeros(total_k);
    let mut second = Array2::zeros((total_k, total_k));
    for block in block_results {
        log_det_total += block.value;
        let kb = block.first.len();
        for k in 0..kb {
            first[block.offset + k] = block.first[k];
        }
        for k in 0..kb {
            for l in 0..kb {
                second[[block.offset + k, block.offset + l]] = block.second[[k, l]];
            }
        }
    }

    Ok(PenaltyLogdetDerivs {
        value: log_det_total,
        first,
        second: Some(second),
    })
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
