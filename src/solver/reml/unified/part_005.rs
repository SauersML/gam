
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

    const OPERATOR_SOLVE_CHUNK: usize = 64;

    fn takahashi_block_trace(
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

    fn takahashi_left_multiply_block(
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

    fn trace_hinv_operator_exact(&self, op: &dyn HyperOperator) -> f64 {
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

    fn solve_operator_column_range_rows_exact(
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

    fn fill_scaled_block_columns(
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

    fn trace_hinv_block_local_exact(
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

    fn solve_block_local_rows_exact(
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

    fn trace_hinv_block_local_cross_exact(
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

    fn trace_hinv_matrix_operator_cross_exact(
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

    fn trace_hinv_matrix_block_operator_cross_exact(
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

    fn trace_hinv_operator_cross_exact(
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
    chol: crate::faer_ndarray::FaerCholeskyFactor,
    /// `2 · Σ ln(diag L)` — cached at construction time.
    cached_logdet: f64,
    /// Full parameter dimension.
    n_dim: usize,
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
    inner: DenseSpectralOperator,
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
    apply: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    cached_logdet: crate::resource::RayonSafeOnce<f64>,
    n_dim: usize,
    // `RayonSafeOnce`, not `OnceLock`: `materialize_dense_operator` invokes
    // `apply`, which for operator-source joint Hessians dispatches a nested
    // `into_par_iter` (e.g. `exact_newton_joint_hessian_matvec_from_cache`).
    // With a plain `OnceLock`, concurrent rayon workers entering
    // `solve`/`logdet` from inside an outer par_iter would park on the
    // OnceLock's OS condvar; the leader's nested par_iter would then starve
    // for workers. `RayonSafeOnce` keeps init lock-free — racers may
    // duplicate the dim²-matvec build, but the first to publish wins and
    // steady-state matches `OnceLock`.
    dense_spectral: crate::resource::RayonSafeOnce<Option<DenseSpectralOperator>>,
    // Pseudo-logdet convention threaded from the family. The dense outer path
    // already plumbs `PseudoLogdetMode` into `BlockCoupledOperator`; the
    // matrix-free path materializes a `DenseSpectralOperator` lazily and must
    // use the same convention so that `logdet`, `trace_hinv_product`, the
    // IFT response `H⁻¹ g`, and every cross-trace agree with the dense path.
    // Without this, families that declare `HardPseudo` (BMS, GAMLSS) silently
    // get Smooth full-spectrum semantics on the matrix-free path, and outer
    // gradients are inflated by `1/σ_j` over numerical null directions.
    mode: PseudoLogdetMode,
}


impl MatrixFreeSpdOperator {
    const EXACT_DENSE_SPECTRAL_MAX_BYTES: usize = 512 * 1024 * 1024;
    const EXACT_DENSE_SPECTRAL_ARRAYS: usize = 6;

    pub fn new_with_mode<F>(dim: usize, apply: F, mode: PseudoLogdetMode) -> Self
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        let apply = Arc::new(apply);

        Self {
            apply,
            cached_logdet: crate::resource::RayonSafeOnce::new(),
            n_dim: dim,
            dense_spectral: crate::resource::RayonSafeOnce::new(),
            mode,
        }
    }

    fn exact_dense_spectral_bytes(&self) -> Option<usize> {
        self.n_dim
            .checked_mul(self.n_dim)?
            .checked_mul(std::mem::size_of::<f64>())?
            .checked_mul(Self::EXACT_DENSE_SPECTRAL_ARRAYS)
    }

    fn exact_dense_spectral_budget_ok(&self) -> bool {
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

    fn materialize_dense_operator(&self) -> Option<DenseSpectralOperator> {
        if !self.exact_dense_spectral_budget_ok() {
            return None;
        }
        let materialize_start = std::time::Instant::now();
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
        let result = DenseSpectralOperator::from_symmetric_with_mode(&matrix, self.mode).ok();
        log::info!(
            "[STAGE] matrix_free_spd materialize n_dim={} matvec_count={} elapsed={:.3}s",
            self.n_dim,
            self.n_dim,
            materialize_start.elapsed().as_secs_f64(),
        );
        result
    }

    fn dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        self.dense_spectral
            .get_or_compute(|| self.materialize_dense_operator())
            .as_ref()
    }

    fn exact_dense_spectral(&self) -> &DenseSpectralOperator {
        self.dense_spectral().expect(
            "MatrixFreeSpdOperator exact REML algebra requires dense spectral materialization within the configured budget",
        )
    }

    fn use_trace_cg(&self, rel_tol: f64) -> bool {
        rel_tol.is_finite()
            && rel_tol > 0.0
            && self.prefers_stochastic_trace_estimation()
            && self.has_matrix_free_trace_cg_operator()
    }

    fn cg_trace_solve(
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


fn conjugate_gradient_trace_solve<F>(
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
    use super::penalty_logdet::PenaltyPseudologdet;

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
        offset: usize,
        value: f64,
        first: Array1<f64>,
        second: Array2<f64>,
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

/// Configuration for stochastic trace estimation.
#[derive(Clone, Debug)]
pub struct StochasticTraceConfig {
    /// Minimum number of probe vectors (default: 10).
    pub n_probes_min: usize,
    /// Maximum number of probe vectors (default: 200).
    pub n_probes_max: usize,
    /// Target relative accuracy ε for the adaptive stopping criterion (default: 0.01).
    pub relative_tol: f64,
    /// Protection threshold τ_rel for near-zero traces (default: 1e-8).
    pub tau_rel: f64,
    /// Relative tolerance for iterative solves inside stochastic trace probes.
    pub solve_rel_tol: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Hutch++ low-rank sketch dimension. `None` = plain Hutchinson.
    /// `Some(m_s)` runs the Meyer–Musco Hutch++ split: m_s sketch matvecs
    /// build an orthonormal range basis Q via randomized range finder, the
    /// projected trace tr(QᵀM Q) is computed exactly (m_s additional
    /// matvecs), and the residual tr((I-QQᵀ)M(I-QQᵀ)) is estimated by
    /// Hutchinson with the remaining probe budget. Achieves O(1/ε)
    /// matvecs for ε relative error vs O(1/ε²) for plain Hutchinson;
    /// the gain is largest when M has rapidly decaying singular values.
    pub hutchpp_sketch_dim: Option<usize>,
}


impl Default for StochasticTraceConfig {
    fn default() -> Self {
        Self {
            n_probes_min: 10,
            n_probes_max: 200,
            relative_tol: 0.01,
            tau_rel: 1e-8,
            solve_rel_tol: 1e-8,
            seed: 0xCAFE_BABE,
            hutchpp_sketch_dim: None,
        }
    }
}


impl StochasticTraceConfig {
    /// Fast, scale-aware estimator for second-order outer-Hessian traces.
    ///
    /// These traces shape the ARC/Newton model; they are not the REML
    /// objective itself. The default 200-probe estimator is too strict for
    /// high-dimensional marginal-slope jobs because near-zero off-diagonal
    /// cross traces never satisfy a pure relative-error test. A bounded probe
    /// budget with a scale-relative zero floor preserves the large curvature
    /// entries and lets ARC's trust-region logic absorb residual noise.
    fn outer_hessian(dim: usize, n_coords: usize) -> Self {
        let large_problem = dim >= 512 || n_coords >= 4;
        Self {
            n_probes_min: if large_problem { 4 } else { 6 },
            n_probes_max: if large_problem { 8 } else { 24 },
            relative_tol: if large_problem { 0.12 } else { 0.05 },
            tau_rel: 1e-3,
            solve_rel_tol: if large_problem { 1e-4 } else { 1e-5 },
            seed: 0xC0A5_7ACE,
            hutchpp_sketch_dim: None,
        }
    }
}


/// Stochastic trace estimator using Rademacher probes with adaptive stopping.
///
/// Estimates `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously,
/// sharing a single `H⁻¹` solve per probe across all coordinates.
///
/// # Adaptive stopping
///
/// After each probe (once `n_probes_min` is reached), the estimator checks:
///
/// ```text
/// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
/// ```
///
/// where `s_{M,k}` is the sample standard deviation of the per-probe
/// estimates for coordinate k, and `q̄_{M,k}` is the running mean.
///
/// # Bias from approximate solves
///
/// If `H⁻¹` is computed approximately (e.g., via PCG with tolerance δ_PCG),
/// the bias satisfies `|bias| ≤ (δ_PCG · p / λ_min(H)) · ‖Ḣ_k‖₂`.
/// Set δ_PCG small enough that this is below the Monte Carlo tolerance.
pub struct StochasticTraceEstimator {
    config: StochasticTraceConfig,
    trace_state: Arc<Mutex<StochasticTraceState>>,
}


enum StochasticTraceTargets<'a> {
    Dense(&'a [&'a Array2<f64>]),
    Mixed {
        dense_matrices: &'a [&'a Array2<f64>],
        operators: &'a [&'a dyn HyperOperator],
    },
    Structural {
        dense_matrices: &'a [&'a Array2<f64>],
        implicit_ops: &'a [&'a ImplicitHyperOperator],
    },
}


impl StochasticTraceTargets<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(matrices) => matrices.len(),
            Self::Mixed {
                dense_matrices,
                operators,
            } => dense_matrices.len() + operators.len(),
            Self::Structural {
                dense_matrices,
                implicit_ops,
            } => dense_matrices.len() + implicit_ops.len(),
        }
    }
}


impl StochasticTraceEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StochasticTraceConfig) -> Self {
        Self {
            config,
            trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// Create a new estimator sharing fit-level stochastic trace state.
    fn with_shared_trace_state(
        mut config: StochasticTraceConfig,
        trace_state: Arc<Mutex<StochasticTraceState>>,
    ) -> Self {
        let override_tol = match trace_state.lock() {
            Ok(guard) => guard.solve_rel_tol_override,
            Err(poisoned) => poisoned.into_inner().solve_rel_tol_override,
        };
        if let Some(rel_tol) = override_tol.filter(|v| v.is_finite() && *v > 0.0) {
            config.solve_rel_tol = rel_tol;
        }
        Self {
            config,
            trace_state,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StochasticTraceConfig::default())
    }

    fn for_outer_hessian(dim: usize, n_coords: usize) -> Self {
        Self::new(StochasticTraceConfig::outer_hessian(dim, n_coords))
    }

    fn for_outer_hessian_with_trace_state(
        dim: usize,
        n_coords: usize,
        trace_state: Arc<Mutex<StochasticTraceState>>,
    ) -> Self {
        Self::with_shared_trace_state(
            StochasticTraceConfig::outer_hessian(dim, n_coords),
            trace_state,
        )
    }

    fn effective_probe_min(&self) -> usize {
        let floor = match self.trace_state.lock() {
            Ok(guard) => guard.monotone_probe_floor,
            Err(poisoned) => poisoned.into_inner().monotone_probe_floor,
        };
        self.config
            .n_probes_min
            .max(floor)
            .min(self.config.n_probes_max)
    }

    fn raise_probe_floor(&self, k_drawn: usize) {
        let mut state = match self.trace_state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if k_drawn > state.monotone_probe_floor {
            let old = state.monotone_probe_floor;
            state.monotone_probe_floor = k_drawn;
            log::info!("[CRN-PIN] probe_floor raised {old}->{k_drawn} (k_drawn={k_drawn})");
        }
    }

    fn estimate_from_probe_batch<F>(
        &self,
        hop: &dyn HessianOperator,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Vec<f64>
    where
        F: FnMut(&Array1<f64>, &Array1<f64>, &mut [f64]),
    {
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut probe_values = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;
        let effective_n_probes_min = self.effective_probe_min();

        let mut z = Array1::<f64>::zeros(p);
        let mut n_drawn = 0usize;
        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let probe_id = stochastic_trace_probe_id(self.config.seed, m);
            let w = hop.stochastic_trace_solve_for_probe(
                &z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            evaluate_probe(&z, &w, &mut probe_values);

            for k in 0..n_coords {
                let q_k = probe_values[k];
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;
            n_drawn = n_done;
            if n_done >= effective_n_probes_min
                && n_done % check_interval == 0
                && self.check_convergence(n_done, &means, &m2s)
            {
                break;
            }
        }

        self.record_probe_batch(Self::max_probe_variance(&m2s, n_drawn), n_drawn);
        self.raise_probe_floor(n_drawn);
        means
    }

    fn estimate_matrix_from_probe_batch<F>(
        &self,
        hop: &dyn HessianOperator,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Array2<f64>
    where
        F: FnMut(u64, &Array1<f64>, &mut Array2<f64>),
    {
        if n_coords == 0 {
            return Array2::zeros((0, 0));
        }
        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((n_coords, n_coords));
        }

        let mut means = Array2::<f64>::zeros((n_coords, n_coords));
        let mut m2s = Array2::<f64>::zeros((n_coords, n_coords));
        let mut probe_values = Array2::<f64>::zeros((n_coords, n_coords));
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;
        let effective_n_probes_min = self.effective_probe_min();
        let mut z = Array1::<f64>::zeros(p);
        let mut n_drawn = 0usize;

        for m in 0..self.config.n_probes_max {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let probe_id = stochastic_trace_probe_id(self.config.seed, m);
            probe_values.fill(0.0);
            evaluate_probe(probe_id, &z, &mut probe_values);

            let count = (m + 1) as f64;
            for d in 0..n_coords {
                for e in 0..n_coords {
                    let q = probe_values[[d, e]];
                    let delta = q - means[[d, e]];
                    means[[d, e]] += delta / count;
                    let delta2 = q - means[[d, e]];
                    m2s[[d, e]] += delta * delta2;
                }
            }

            let n_done = m + 1;
            n_drawn = n_done;
            if n_done >= effective_n_probes_min
                && n_done % check_interval == 0
                && self.check_matrix_convergence(n_done, &means, &m2s)
            {
                break;
            }
        }

        self.record_probe_batch(
            Self::max_probe_variance(m2s.as_slice().unwrap(), n_drawn),
            n_drawn,
        );
        self.raise_probe_floor(n_drawn);
        for d in 0..n_coords {
            for e in (d + 1)..n_coords {
                let avg = 0.5 * (means[[d, e]] + means[[e, d]]);
                means[[d, e]] = avg;
                means[[e, d]] = avg;
            }
        }
        means
    }

    fn max_probe_variance(m2s: &[f64], n_drawn: usize) -> f64 {
        if n_drawn <= 1 {
            return 0.0;
        }
        let denom = (n_drawn - 1) as f64;
        m2s.iter()
            .map(|m2| (*m2 / denom).max(0.0))
            .fold(0.0_f64, f64::max)
    }

    fn record_probe_batch(&self, sigma_sq: f64, n_drawn: usize) {
        let mut state = match self.trace_state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.last_probe_sigma_sq = Some(state.last_probe_sigma_sq.unwrap_or(0.0).max(sigma_sq));
        state.last_probe_count = state.last_probe_count.max(n_drawn);
    }

    fn estimate_hinv_traces(
        &self,
        hop: &dyn HessianOperator,
        targets: StochasticTraceTargets<'_>,
    ) -> Vec<f64> {
        let n_coords = targets.len();
        if n_coords == 0 {
            return Vec::new();
        }

        match targets {
            StochasticTraceTargets::Dense(matrices) => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..matrices.len() {
                        dense_matvec_into(matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }
                })
            }
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            } => {
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..dense_matrices.len() {
                        dense_matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in operators.iter().enumerate() {
                        let k = dense_count + oi;
                        if op.has_fast_bilinear_view() {
                            probe_values[k] = op.bilinear_view(w.view(), z.view());
                        } else {
                            op.mul_vec_into(w.view(), a_w.view_mut());
                            probe_values[k] = z.dot(&a_w);
                        }
                    }
                })
            }
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            } => {
                if implicit_ops.is_empty() {
                    let no_ops: [&dyn HyperOperator; 0] = [];
                    return self.estimate_hinv_traces(
                        hop,
                        StochasticTraceTargets::Mixed {
                            dense_matrices,
                            operators: &no_ops,
                        },
                    );
                }

                let x_design = implicit_ops[0].x_design.clone();
                let mut x_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut y_vec = Array1::<f64>::zeros(x_design.nrows());
                let mut a_w = Array1::<f64>::zeros(hop.dim());
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    design_matrix_apply_view_into(x_design.as_ref(), z.view(), x_vec.view_mut());
                    design_matrix_apply_view_into(x_design.as_ref(), w.view(), y_vec.view_mut());

                    for k in 0..dense_matrices.len() {
                        dense_matvec_into(dense_matrices[k], w.view(), a_w.view_mut());
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in implicit_ops.iter().enumerate() {
                        let k = dense_count + oi;
                        probe_values[k] = op.bilinear_with_shared_x(&x_vec, &y_vec, z, w);
                    }
                })
            }
        }
    }

    /// Estimate a single trace `tr(H⁻¹ A)` using the same batched Hutchinson
    /// core as the multi-coordinate path.
    pub fn estimate_single_trace(&self, hop: &dyn HessianOperator, matrix: &Array2<f64>) -> f64 {
        let matrices = [matrix];
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(&matrices))[0]
    }

    /// Estimate `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously.
    ///
    /// Uses Rademacher probes and adaptive stopping. Each probe requires
    /// exactly ONE `H⁻¹` solve (shared across all k), plus one `A_k`
    /// matrix–vector product per coordinate k.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `matrices`: the `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    ///
    /// # Returns
    /// A vector of estimated traces, one per input matrix.
    pub fn estimate_traces(
        &self,
        hop: &dyn HessianOperator,
        matrices: &[&Array2<f64>],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(matrices))
    }

    /// Estimate `tr(H⁻¹ A_k)` for a mix of dense matrices and implicit operators.
    ///
    /// This extends [`estimate_traces`] to support implicit `HyperOperator` trait
    /// objects alongside dense matrices. The dense matrices are passed first,
    /// followed by the operators. Each probe requires ONE `H⁻¹` solve (shared),
    /// plus one matvec per coordinate.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    /// - `operators`: implicit `HyperOperator` trait objects.
    ///
    /// # Returns
    /// A vector of estimated traces: first for dense matrices, then for operators.
    pub fn estimate_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            },
        )
    }

    /// Estimate first-order traces `tr(H⁻¹ A_d)` for implicit operators using the
    /// weighted-Gram structure, sharing one H⁻¹ solve and two X multiplies per probe.
    ///
    /// For each implicit operator d, the bilinear form `u^T A_d z` is computed using
    /// shared `x_vec = X z` and `y_vec = X u`, plus per-axis `forward_mul` calls.
    /// This avoids the X^T multiply per axis that the standard `mul_vec` requires.
    ///
    /// Dense matrices are handled alongside implicit operators in a single pass.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated traces: first for dense matrices, then for implicit operators.
    pub fn estimate_traces_structural(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Vec<f64> {
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            },
        )
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for implicit operators, using the CORRECT estimator.
    ///
    /// The correct Girard-Hutchinson estimator for `tr(H⁻¹ A_d H⁻¹ A_e)` is:
    ///
    /// ```text
    /// u = H⁻¹ z
    /// q_e = A_e z        for each axis e
    /// r_e = H⁻¹ q_e      for each axis e  (block solve, D RHS)
    /// estimate = u^T A_d r_e
    /// ```
    ///
    /// This gives tr(H⁻¹ A_d H⁻¹ A_e) correctly, NOT tr(A_d H⁻² A_e).
    ///
    /// Dense matrices are included alongside implicit operators. The output
    /// is a (total × total) matrix of cross-traces, symmetrized.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve` and `solve_multi`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated D×D matrix of `tr(H⁻¹ A_d H⁻¹ A_e)` values, symmetrized.
    pub fn estimate_second_order_traces(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = implicit_ops.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_implicit(hop, implicit_ops[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        // Get the shared X reference from the first implicit operator.
        let x_design = if n_ops > 0 {
            Some(implicit_ops[0].x_design.clone())
        } else {
            None
        };

        let mut q_columns = Array2::zeros((p, total));
        let mut dense_a_u: Vec<Array1<f64>> = (0..n_dense).map(|_| Array1::zeros(p)).collect();
        let n_obs = implicit_ops.first().map(|op| op.w_diag.len()).unwrap_or(0);
        let mut x_vec = Array1::<f64>::zeros(n_obs);
        let mut y_vec = Array1::<f64>::zeros(n_obs);
        let mut x_r: Vec<Array1<f64>> = (0..total).map(|_| Array1::zeros(n_obs)).collect();

        struct ImplicitSecondOrderScratch {
            w_dx_u: Array1<f64>,
            w_y: Array1<f64>,
            u_s: Array1<f64>,
        }

        self.estimate_matrix_from_probe_batch(hop, total, |probe_id, z, probe_values| {
            // Step 1: u = H⁻¹ z (shared solve)
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );

            if let Some(ref x) = x_design {
                design_matrix_apply_view_into(x.as_ref(), z.view(), x_vec.view_mut());
            }

            // Step 2: Form q_e = A_e z for all axes e. Each operator column is
            // independent, so fill the destination columns in parallel while
            // keeping only per-worker implicit matvec scratch.
            {
                use ndarray::Axis;
                use ndarray::parallel::prelude::*;

                q_columns
                    .axis_iter_mut(Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(e, q_col)| {
                        if e < n_dense {
                            dense_matvec_into(dense_matrices[e], z.view(), q_col);
                        } else {
                            let op = implicit_ops[e - n_dense];
                            let mut n_work = Array1::<f64>::zeros(n_obs);
                            let mut p_work = Array1::<f64>::zeros(p);
                            op.matvec_with_shared_xz_into(
                                x_vec.view(),
                                z.view(),
                                q_col,
                                n_work.view_mut(),
                                p_work.view_mut(),
                            );
                        }
                    });
            }

            // Step 3: R = H⁻¹ [q_1, ..., q_D] (block solve, total RHS)
            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            // Step 4: Compute T[d, e] = u^T A_d r_e for all (d, e) pairs.
            // For dense A_d: T[d, e] = (A_d^T u)^T r_e = (A_d u)^T r_e (A_d symmetric)
            // For implicit A_d: use shared X multiplies and bounded per-pair scratch.

            // Precompute X u and X r_e for implicit operators.
            if let Some(ref x) = x_design {
                design_matrix_apply_view_into(x.as_ref(), u.view(), y_vec.view_mut());
            }

            // For dense operators, precompute A_d u once.
            for d in 0..n_dense {
                dense_matvec_into(dense_matrices[d], u.view(), dense_a_u[d].view_mut());
            }

            // Precompute X r_e for all axes e (for implicit operators). These
            // columns are independent and reused by every implicit row.
            if let Some(ref x) = x_design {
                use rayon::prelude::*;
                x_r.par_iter_mut().enumerate().for_each(|(e, x_r_e)| {
                    design_matrix_apply_view_into(x.as_ref(), r.column(e), x_r_e.view_mut());
                });
            }

            // Precompute row-wise implicit quantities that are reused across all
            // columns. Deliberately do not materialize (∂X/∂ψ_d) r_e for every
            // d×e pair; those n_obs-sized vectors are built inside the pair task
            // below, which bounds scratch by the number of active rayon workers
            // rather than n_ops * total.
            let implicit_scratch: Vec<ImplicitSecondOrderScratch> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_ops)
                    .into_par_iter()
                    .map(|idx| {
                        let op = implicit_ops[idx];
                        let dx_u = op
                            .implicit_deriv
                            .forward_mul(op.axis, &u.view())
                            .expect(
                                "radial scalar evaluation failed during implicit derivative forward_mul",
                            );
                        let w = &*op.w_diag;
                        let mut w_dx_u = Array1::<f64>::zeros(n_obs);
                        let mut w_y = Array1::<f64>::zeros(n_obs);
                        for i in 0..w.len() {
                            w_dx_u[i] = w[i] * dx_u[i];
                            w_y[i] = w[i] * y_vec[i];
                        }
                        let mut u_s = Array1::<f64>::zeros(p);
                        dense_transpose_matvec_into(&op.s_psi, u.view(), u_s.view_mut());
                        ImplicitSecondOrderScratch { w_dx_u, w_y, u_s }
                    })
                    .collect()
            };

            let pair_count = total * total;
            let pair_values: Vec<(usize, usize, f64)> = {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..pair_count)
                    .into_par_iter()
                    .map(|pair_idx| {
                        let d = pair_idx / total;
                        let e = pair_idx % total;
                        let r_e = r.column(e);
                        let val = if d < n_dense {
                            // Dense A_d: u^T A_d r_e = (A_d u)^T r_e
                            dense_a_u[d].dot(&r_e)
                        } else {
                            // Implicit A_d: compute u^T A_d r_e using shared X multiplies.
                            // u^T A_d r_e = ((∂X/∂ψ_d)u)^T (W X r_e)
                            //             + (Xu)^T (W (∂X/∂ψ_d) r_e)
                            //             + u^T S_psi r_e
                            let oi = d - n_dense;
                            let op = implicit_ops[oi];
                            let scratch = &implicit_scratch[oi];
                            let x_re = &x_r[e];
                            let dx_re = op
                                .implicit_deriv
                                .forward_mul(op.axis, &r_e)
                                .expect(
                                    "radial scalar evaluation failed during implicit derivative forward_mul",
                                );

                            let mut design_val = 0.0f64;
                            for i in 0..scratch.w_dx_u.len() {
                                design_val += scratch.w_dx_u[i] * x_re[i];
                                design_val += scratch.w_y[i] * dx_re[i];
                            }

                            // Non-Gaussian fixed-β third-derivative correction:
                            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r_e
                            //   = Σ_i y_vec[i] · c_x_psi_beta_i · x_re[i]
                            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                                let c = c_x_psi_beta.as_ref();
                                for i in 0..scratch.w_dx_u.len() {
                                    design_val += y_vec[i] * c[i] * x_re[i];
                                }
                            }

                            // Penalty: u^T S_psi r_e = (S_psi^T u)^T r_e
                            let penalty_val = scratch.u_s.dot(&r_e);
                            design_val + penalty_val
                        };
                        (d, e, val)
                    })
                    .collect()
            };

            for (d, e, val) in pair_values {
                probe_values[[d, e]] = val;
            }
        })
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for a mix of dense matrices and generic hyperoperators.
    pub fn estimate_second_order_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = operators.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        if total == 1 {
            let value = if n_dense == 1 {
                self.estimate_second_order_single_dense(hop, dense_matrices[0])
            } else {
                self.estimate_second_order_single_operator(hop, operators[0])
            };
            return Array2::from_elem((1, 1), value);
        }

        let mut q_columns = Array2::zeros((p, total));
        let mut a_u_columns = Array2::zeros((p, total));

        self.estimate_matrix_from_probe_batch(hop, total, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );

            for e in 0..n_dense {
                dense_matvec_into(dense_matrices[e], z.view(), q_columns.column_mut(e));
                dense_matvec_into(dense_matrices[e], u.view(), a_u_columns.column_mut(e));
            }
            for (oi, op) in operators.iter().enumerate() {
                let e = n_dense + oi;
                op.mul_vec_into(z.view(), q_columns.column_mut(e));
                op.mul_vec_into(u.view(), a_u_columns.column_mut(e));
            }

            let r = hop.stochastic_trace_solve_multi(&q_columns, self.config.solve_rel_tol);

            for d in 0..total {
                let a_d_u = a_u_columns.column(d);
                for e in d..total {
                    let r_e = r.column(e);
                    let val = a_d_u.dot(&r_e);
                    probe_values[[d, e]] = val;
                    if d != e {
                        let r_d = r.column(d);
                        let val_sym = a_u_columns.column(e).dot(&r_d);
                        probe_values[[e, d]] = val_sym;
                    }
                }
            }
        })
    }

    fn estimate_second_order_single_dense(
        &self,
        hop: &dyn HessianOperator,
        matrix: &Array2<f64>,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        if self.config.hutchpp_sketch_dim.is_some() {
            let op = DenseMatrixHyperOperator {
                matrix: matrix.clone(),
            };
            return hutchpp_estimate_trace_hinv_op_squared(hop, &op, &self.config);
        }

        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            dense_matvec_into(matrix, z.view(), q.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = dense_bilinear(matrix, u.view(), r.view());
        })[[0, 0]]
    }

    fn estimate_second_order_single_implicit(
        &self,
        hop: &dyn HessianOperator,
        op: &ImplicitHyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        if self.config.hutchpp_sketch_dim.is_some() {
            return hutchpp_estimate_trace_hinv_op_squared(hop, op, &self.config);
        }

        let n_obs = op.w_diag.len();
        let mut x_z = Array1::<f64>::zeros(n_obs);
        let mut x_u = Array1::<f64>::zeros(n_obs);
        let mut x_r = Array1::<f64>::zeros(n_obs);
        let mut n_work = Array1::<f64>::zeros(n_obs);
        let mut p_work = Array1::<f64>::zeros(p);
        let mut q = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            design_matrix_apply_view_into(&op.x_design, z.view(), x_z.view_mut());
            op.matvec_with_shared_xz_into(
                x_z.view(),
                z.view(),
                q.view_mut(),
                n_work.view_mut(),
                p_work.view_mut(),
            );
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);

            design_matrix_apply_view_into(&op.x_design, u.view(), x_u.view_mut());
            design_matrix_apply_view_into(&op.x_design, r.view(), x_r.view_mut());
            let dx_u = op
                .implicit_deriv
                .forward_mul(op.axis, &u.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");
            let dx_r = op
                .implicit_deriv
                .forward_mul(op.axis, &r.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul");

            let w = &*op.w_diag;
            let mut value = 0.0;
            for i in 0..w.len() {
                let wi = w[i];
                value += dx_u[i] * wi * x_r[i];
                value += x_u[i] * wi * dx_r[i];
            }
            // Non-Gaussian fixed-β third-derivative correction:
            //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X r = Σ_i (X u)_i · c_x_psi_beta_i · (X r)_i
            if let Some(c_x_psi_beta) = op.c_x_psi_beta.as_ref() {
                let c = c_x_psi_beta.as_ref();
                for i in 0..w.len() {
                    value += x_u[i] * c[i] * x_r[i];
                }
            }
            value += dense_bilinear(&op.s_psi, r.view(), u.view());

            probe_values[[0, 0]] = value;
        })[[0, 0]]
    }

    fn estimate_second_order_single_operator(
        &self,
        hop: &dyn HessianOperator,
        op: &dyn HyperOperator,
    ) -> f64 {
        let p = hop.dim();
        if p == 0 {
            return 0.0;
        }

        let mut q = Array1::<f64>::zeros(p);
        let mut a_u = Array1::<f64>::zeros(p);
        self.estimate_matrix_from_probe_batch(hop, 1, |probe_id, z, probe_values| {
            let u = hop.stochastic_trace_solve_for_probe(
                z,
                self.config.solve_rel_tol,
                probe_id,
                Some(&self.trace_state),
            );
            op.mul_vec_into(z.view(), q.view_mut());
            op.mul_vec_into(u.view(), a_u.view_mut());
            let r = hop.stochastic_trace_solve(&q, self.config.solve_rel_tol);
            probe_values[[0, 0]] = a_u.dot(&r);
        })[[0, 0]]
    }

    /// Check the adaptive stopping criterion.
    ///
    /// Returns `true` if all coordinates have converged:
    /// ```text
    /// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
    /// ```
    fn check_convergence(&self, n: usize, means: &[f64], m2s: &[f64]) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;

        for k in 0..means.len() {
            let variance = m2s[k] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * means[k].abs().max(self.config.tau_rel);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }

    fn check_matrix_convergence(&self, n: usize, means: &Array2<f64>, m2s: &Array2<f64>) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;
        let scale_floor = means
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
            .max(1.0)
            * self.config.tau_rel;
        for ((d, e), &mean) in means.indexed_iter() {
            let variance = m2s[[d, e]] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * mean.abs().max(scale_floor);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }
}


fn stochastic_trace_hinv_products_with_floor(
    hop: &dyn HessianOperator,
    targets: StochasticTraceTargets<'_>,
    trace_state: Option<Arc<Mutex<StochasticTraceState>>>,
) -> Vec<f64> {
    let estimator = match trace_state {
        Some(state) => StochasticTraceEstimator::with_shared_trace_state(
            StochasticTraceConfig::default(),
            state,
        ),
        None => StochasticTraceEstimator::with_defaults(),
    };
    match targets {
        StochasticTraceTargets::Dense(matrices) if matrices.len() == 1 => {
            vec![estimator.estimate_single_trace(hop, matrices[0])]
        }
        StochasticTraceTargets::Dense(matrices) => estimator.estimate_traces(hop, matrices),
        StochasticTraceTargets::Mixed {
            dense_matrices,
            operators,
        } => estimator.estimate_traces_with_operators(hop, dense_matrices, operators),
        StochasticTraceTargets::Structural {
            dense_matrices,
            implicit_ops,
        } => estimator.estimate_traces_structural(hop, dense_matrices, implicit_ops),
    }
}


fn stochastic_trace_hinv_crosses<'a>(
    hop: &dyn HessianOperator,
    dense_matrices: &'a [Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
) -> Array2<f64> {
    // The `_with_floor` variant takes a slice of references; adapt the owned
    // slice without copying the matrices.
    let dense_refs: Vec<&'a Array2<f64>> = dense_matrices.iter().collect();
    stochastic_trace_hinv_crosses_with_floor(
        hop,
        &dense_refs,
        coord_has_operator,
        generic_ops,
        implicit_ops,
        None,
    )
}


fn stochastic_trace_hinv_crosses_with_floor<'a>(
    hop: &dyn HessianOperator,
    dense_matrices: &[&'a Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
    trace_state: Option<Arc<Mutex<StochasticTraceState>>>,
) -> Array2<f64> {
    let estimator = match trace_state {
        Some(state) => StochasticTraceEstimator::for_outer_hessian_with_trace_state(
            hop.dim(),
            coord_has_operator.len(),
            state,
        ),
        None => StochasticTraceEstimator::for_outer_hessian(hop.dim(), coord_has_operator.len()),
    };
    let raw_cross = if generic_ops.len() == implicit_ops.len() {
        estimator.estimate_second_order_traces(hop, dense_matrices, implicit_ops)
    } else {
        estimator.estimate_second_order_traces_with_operators(hop, dense_matrices, generic_ops)
    };

    let total_coords = coord_has_operator.len();
    let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
    let mut original_to_raw = Vec::with_capacity(total_coords);
    let mut dense_cursor = 0usize;
    let mut operator_cursor = n_dense_total;
    for &has_operator in coord_has_operator {
        if has_operator {
            original_to_raw.push(operator_cursor);
            operator_cursor += 1;
        } else {
            original_to_raw.push(dense_cursor);
            dense_cursor += 1;
        }
    }

    let mut mapped = Array2::zeros((total_coords, total_coords));
    for i in 0..total_coords {
        for j in 0..total_coords {
            mapped[[i, j]] = raw_cross[[original_to_raw[i], original_to_raw[j]]];
        }
    }
    mapped
}


// Lightweight xoshiro256ss RNG
//
// We use a self-contained xoshiro256ss implementation so that the stochastic
// trace estimator does not impose any new dependency requirements. The
// codebase already uses `rand` (0.10), but a minimal inline RNG avoids
// pulling in the full `rand` trait machinery for what is just a stream of
// random bits for ±1 generation.

/// Minimal xoshiro256** PRNG (period 2^256 − 1).
///
/// This is used exclusively for Rademacher probe generation. The state is
/// seeded deterministically from a u64 via splitmix64.
struct Xoshiro256SS {
    s: [u64; 4],
}


impl Xoshiro256SS {
    /// Seed from a single u64 via splitmix64 expansion.
    fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        // Guard against the all-zero state (astronomically unlikely but
        // formally required for xoshiro correctness).
        let s = if s0 | s1 | s2 | s3 == 0 {
            [1, 0, 0, 0]
        } else {
            [s0, s1, s2, s3]
        };
        Self { s }
    }

    /// Generate the next u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}


/// Splitmix64: deterministic expansion of a single u64 seed into a sequence.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    crate::linalg::utils::splitmix64(state)
}


#[inline]
fn stochastic_trace_probe_id(seed: u64, probe_index: usize) -> u64 {
    let mut state = seed ^ (probe_index as u64).wrapping_mul(0xD1B54A32D192ED03);
    splitmix64(&mut state)
}


fn rademacher_probe_into(mut z: ArrayViewMut1<'_, f64>, rng: &mut Xoshiro256SS) {
    let mut bits: u64 = 0;
    let mut remaining_bits = 0u32;

    for i in 0..z.len() {
        if remaining_bits == 0 {
            bits = rng.next_u64();
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { 1.0 } else { -1.0 };
        bits >>= 1;
        remaining_bits -= 1;
    }
}


/// Modified Gram–Schmidt orthonormalization of the columns of `y`,
/// writing the orthonormal basis into `q` and returning the retained
/// rank.
///
/// `y` and `q` must have the same shape `(p, m)`. Columns whose
/// reduction norm falls below `1e-12` of the largest input column
/// norm are dropped (numerical-rank cutoff). After this call,
/// `q.column(0..rank)` is column-orthonormal and approximates
/// `range(y)`; later columns of `q` are zeroed.
fn modified_gram_schmidt(y: &Array2<f64>, q: &mut Array2<f64>) -> usize {
    let p = y.nrows();
    let m = y.ncols();
    assert_eq!(q.dim(), (p, m));
    q.fill(0.0);
    if p == 0 || m == 0 {
        return 0;
    }
    let mut max_norm: f64 = 0.0;
    for j in 0..m {
        let n = y.column(j).dot(&y.column(j)).sqrt();
        if n > max_norm {
            max_norm = n;
        }
    }
    let drop_tol = (max_norm * 1.0e-12).max(f64::MIN_POSITIVE);
    let mut rank = 0usize;
    for j in 0..m {
        let mut v = y.column(j).to_owned();
        for k in 0..rank {
            let qk = q.column(k);
            let proj = qk.dot(&v);
            if proj != 0.0 {
                v.scaled_add(-proj, &qk);
            }
        }
        let norm = v.dot(&v).sqrt();
        if !norm.is_finite() || norm <= drop_tol {
            continue;
        }
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
        q.column_mut(rank).assign(&v);
        rank += 1;
    }
    rank
}


/// Shared Hutch++ stochastic-trace scaffold (Meyer–Musco 2021, SOSA).
///
/// Estimates `tr(B)` for a linear map `B: x ↦ apply(hop, x, &mut tmp)`,
/// where `apply` is the per-probe action of `B` on a vector (using `tmp`
/// as scratch and returning a fresh `Array1<f64>`). The three public
/// estimators below differ *only* in this closure:
///
/// * `tr(H⁻¹ M)`        — `apply` = `M`-apply then one solve;
/// * `tr((H⁻¹ A)²)`     — `apply` = apply/solve/apply/solve;
/// * `tr(H⁻¹ A_L H⁻¹ A_R)` — `apply` = `A_R`/solve/`A_L`/solve.
///
/// Everything else (sketch dim, RNG seeding, randomized range finder +
/// modified Gram–Schmidt, exact low-rank trace `tr(Qᵀ B Q)`, residual
/// Hutchinson on `(I - Q Qᵀ) B (I - Q Qᵀ)` with the Welford-style
/// adaptive relative-error stop) is identical, so it lives here once.
/// `B` need not be self-adjoint: on Rademacher probes `E[zᵀ B z] = tr(B)`
/// regardless, and the projected `tr(Qᵀ B Q)` is exact on `range(Q)`.
fn hutchpp_estimate_trace_with_apply<F>(p: usize, config: &StochasticTraceConfig, apply: F) -> f64
where
    F: Fn(ArrayView1<'_, f64>, &mut Array1<f64>) -> Array1<f64>,
{
    if p == 0 {
        return 0.0;
    }
    let sketch_dim = config.hutchpp_sketch_dim.unwrap_or(0).min(p);
    let mut rng_state = Xoshiro256SS::from_seed(config.seed);

    // Phase 1: build orthonormal Q ∈ R^{p × sketch_dim} approximating
    // range(B) via a randomized range finder.
    let mut q = Array2::<f64>::zeros((p, sketch_dim));
    let mut q_rank = 0usize;
    if sketch_dim > 0 {
        let mut y = Array2::<f64>::zeros((p, sketch_dim));
        let mut z = Array1::<f64>::zeros(p);
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..sketch_dim {
            rademacher_probe_into(z.view_mut(), &mut rng_state);
            let w = apply(z.view(), &mut tmp);
            y.column_mut(j).assign(&w);
        }
        q_rank = modified_gram_schmidt(&y, &mut q);
    }

    // Phase 2: T_low = tr(Qᵀ B Q), exact on range(Q).
    let mut t_low = 0.0;
    if q_rank > 0 {
        let mut tmp = Array1::<f64>::zeros(p);
        for j in 0..q_rank {
            let qcol = q.column(j).to_owned();
            let w = apply(qcol.view(), &mut tmp);
            t_low += qcol.dot(&w);
        }
    }

    // Phase 3: residual Hutchinson on (I - Q Qᵀ) B (I - Q Qᵀ).
    // Budget = remaining matvecs from n_probes_max minus the 2*q_rank
    // we already spent (sketch + Q-trace), but never below n_probes_min.
    let used = 2 * q_rank;
    let residual_budget_max = config.n_probes_max.saturating_sub(used);
    let residual_min = config.n_probes_min.min(residual_budget_max);
    let residual_budget = residual_budget_max.max(residual_min);
    if residual_budget == 0 {
        return t_low;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut z = Array1::<f64>::zeros(p);
    let mut z_tilde = Array1::<f64>::zeros(p);
    let mut tmp = Array1::<f64>::zeros(p);
    let check_interval = 4usize;
    for _ in 0..residual_budget {
        rademacher_probe_into(z.view_mut(), &mut rng_state);
        // z_tilde = (I - Q Qᵀ) z = z - Q (Qᵀ z)
        z_tilde.assign(&z);
        if q_rank > 0 {
            for j in 0..q_rank {
                let qcol = q.column(j);
                let proj = qcol.dot(&z);
                if proj != 0.0 {
                    z_tilde.scaled_add(-proj, &qcol);
                }
            }
        }
        let w = apply(z_tilde.view(), &mut tmp);
        let q_val = z_tilde.dot(&w);
        sum += q_val;
        sum_sq += q_val * q_val;
        count += 1;

        // Adaptive stopping: same Welford-style relative-error check
        // as `estimate_from_probe_batch`, applied to the residual mean.
        if count >= residual_min && count.is_multiple_of(check_interval) && count >= 2 {
            let n = count as f64;
            let mean = sum / n;
            let var = (sum_sq - n * mean * mean) / (n - 1.0).max(1.0);
            if var.is_finite() && var >= 0.0 {
                let stderr = (var / n).sqrt();
                let denom = (mean.abs()).max(config.tau_rel);
                if stderr / denom <= config.relative_tol {
                    break;
                }
            }
        }
    }
    let mean_residual = if count > 0 { sum / count as f64 } else { 0.0 };
    t_low + mean_residual
}


/// Hutch++ estimate of `tr(H⁻¹ M)` where `M` is accessed through its
/// matrix-vector product (operator-only, dim p).
///
/// Total cost: `2 m_s + m_h` H⁻¹ solves and `M·v` matvecs, where
/// `m_s = config.hutchpp_sketch_dim.unwrap_or(0)` and `m_h` is the
/// number of residual Hutchinson probes drawn (between
/// `config.n_probes_min` and `config.n_probes_max - 2 m_s`).
///
/// When `hutchpp_sketch_dim` is `None`, this falls back to plain
/// Hutchinson on the full probe budget — the result is deterministic
/// for a given seed because the probe RNG is seeded from
/// `config.seed`.
///
/// # Algorithm (Meyer–Musco 2021, SOSA)
///
/// 1. Sketch: draw `Z_s ∈ {±1}^{p × m_s}` Rademacher, compute
///    `Y = H⁻¹ M Z_s`, orthonormalize columns: `Y = Q R`.
/// 2. Low-rank trace: `T_low = tr(Qᵀ H⁻¹ M Q)` exactly via `m_s`
///    additional matvecs into `W = H⁻¹ M Q` and accumulating
///    `Σ_j Q[:,j] · W[:,j]`.
/// 3. Residual Hutchinson on the orthogonal complement: for each
///    residual probe `z`, set `z̃ = (I - Q Qᵀ) z`, compute
///    `w̃ = H⁻¹ M z̃`, and accumulate `z̃ · w̃` (which equals
///    `z̃ᵀ (H⁻¹ M) z̃` because `z̃` is in the complement).
/// 4. Output: `T_low + (1/m_h) Σ residual estimates`.
///
/// # When this wins over plain Hutchinson
///
/// Hutch++ converges in `O(1/ε)` matvecs vs `O(1/ε²)` for Hutchinson.
/// The gain is largest when `H⁻¹ M` has rapid singular-value decay —
/// the sketch captures the dominant subspace exactly and Hutchinson
/// only handles the small residual. For roughly-flat spectra both
/// methods perform similarly per-matvec.
pub(crate) fn hutchpp_estimate_trace_hinv_operator<H, O>(
    hop: &H,
    op: &O,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianOperator + ?Sized,
    O: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(op.dim(), p, "Hutch++: operator dim mismatch");
    // B x = H⁻¹ M x: apply M then a single solve.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        op.mul_vec_into(x, tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}


/// Hutch++ estimate of `tr((H⁻¹ A)²) = tr(H⁻¹ A H⁻¹ A)` for a symmetric
/// HVP-only operator `A`. Cost per applied "matvec" is 2 H⁻¹ solves and
/// 2 A applies; total cost is `2 m_s + m_h` such matvecs.
///
/// Although `B = H⁻¹ A` is not symmetric in the standard inner product,
/// `B²` is similar to `(H^{-1/2} A H^{-1/2})²` (PSD), so its trace is
/// nonnegative and Hutch++ on the linear map `x ↦ B² x` produces an
/// unbiased estimate of `tr(B²)` on standard probes.
pub(crate) fn hutchpp_estimate_trace_hinv_op_squared<H, O>(
    hop: &H,
    op: &O,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianOperator + ?Sized,
    O: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(op.dim(), p, "Hutch++ squared: operator dim mismatch");
    // B x = (H⁻¹ A)² x = H⁻¹ A H⁻¹ A x via two solve+apply legs.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        op.mul_vec_into(x, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        op.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}


/// Hutch++-style estimate of `tr(H⁻¹ A_left H⁻¹ A_right)` for two
/// (possibly distinct) symmetric HVP-only operators. Uses a shared
/// sketch built from `M = M_L M_R` where `M_L = H⁻¹ A_left` and
/// `M_R = H⁻¹ A_right`; per matvec is 2 H⁻¹ solves + 2 A applies.
///
/// On standard Rademacher probes `E[zᵀ M z] = tr(M)` regardless of
/// symmetry, so the residual Hutchinson average is unbiased even when
/// `M` is not self-adjoint in the standard inner product.
///
/// A leave-one-out XTrace estimator (Epperly & Tropp 2024, arxiv
/// 2301.07825) would reduce variance further by exchanging each probe
/// between sketch and residual roles, at O(m²) bookkeeping cost.
pub(crate) fn hutchpp_estimate_trace_hinv_operator_cross<H, L, R>(
    hop: &H,
    left: &L,
    right: &R,
    config: &StochasticTraceConfig,
) -> f64
where
    H: HessianOperator + ?Sized,
    L: HyperOperator + ?Sized,
    R: HyperOperator + ?Sized,
{
    let p = hop.dim();
    assert_eq!(left.dim(), p, "cross trace: left operator dim mismatch");
    assert_eq!(right.dim(), p, "cross trace: right operator dim mismatch");
    // M x = H⁻¹ A_L H⁻¹ A_R x.
    hutchpp_estimate_trace_with_apply(p, config, |x, tmp| {
        right.mul_vec_into(x, tmp.view_mut());
        let mid = hop.stochastic_trace_solve(tmp, config.solve_rel_tol);
        left.mul_vec_into(mid.view(), tmp.view_mut());
        hop.stochastic_trace_solve(tmp, config.solve_rel_tol)
    })
}
