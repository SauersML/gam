use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::row_kernel::*;
use super::*;
use crate::gpu::kernels::row_hessian_ops;

impl BernoulliMarginalSlopeFamily {
    pub(super) fn exact_newton_joint_gradient_evaluation_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<ExactNewtonJointGradientEvaluation, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS exact-gradient eval n={n} p={}",
            slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-gradient] eval start n={} p={} source=cache",
                n,
                slices.total
            );
        }
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(slices.marginal.len()),
                Array1::<f64>::zeros(slices.logslope.len()),
                slices
                    .h
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                slices
                    .w
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
            )
        };
        let row_chunk = bms_row_chunk_size(n);
        let (log_likelihood, grad_marginal, grad_logslope, grad_h, grad_w) = (0..n
            .div_ceil(row_chunk))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * row_chunk;
                let end = (start + row_chunk).min(n);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in start..end {
                    let row_ctx = Self::row_ctx(cache, row);
                    let row_moments = cache
                        .row_cell_moments
                        .as_ref()
                        .and_then(|bundle| bundle.row(row, 3));
                    let neglog = self.compute_row_analytic_flex_into_with_moments(
                        row,
                        block_states,
                        primary,
                        row_ctx,
                        row_moments,
                        cache.cell_family_forest.as_ref(),
                        false,
                        &mut scratch,
                    )?;
                    acc.0 -= neglog;
                    {
                        let mut marginal = acc.1.view_mut();
                        self.marginal_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[0],
                            ),
                            &mut marginal,
                        )?;
                    }
                    {
                        let mut logslope = acc.2.view_mut();
                        self.logslope_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[1],
                            ),
                            &mut logslope,
                        )?;
                    }
                    if let (Some(primary_h), Some(grad_h)) = (primary.h.as_ref(), acc.3.as_mut()) {
                        for idx in 0..primary_h.len() {
                            grad_h[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_h.start + idx],
                                );
                        }
                    }
                    if let (Some(primary_w), Some(grad_w)) = (primary.w.as_ref(), acc.4.as_mut()) {
                        for idx in 0..primary_w.len() {
                            grad_w[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_w.start + idx],
                                );
                        }
                    }
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                left.2 += &right.2;
                if let (Some(lhs), Some(rhs)) = (left.3.as_mut(), right.3.as_ref()) {
                    *lhs += rhs;
                }
                if let (Some(lhs), Some(rhs)) = (left.4.as_mut(), right.4.as_ref()) {
                    *lhs += rhs;
                }
                Ok(left)
            })?;

        let mut gradient = Array1::<f64>::zeros(slices.total);
        gradient
            .slice_mut(s![slices.marginal.clone()])
            .assign(&grad_marginal);
        gradient
            .slice_mut(s![slices.logslope.clone()])
            .assign(&grad_logslope);
        if let (Some(range), Some(grad_h)) = (slices.h.as_ref(), grad_h.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_h);
        }
        if let (Some(range), Some(grad_w)) = (slices.w.as_ref(), grad_w.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_w);
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-gradient] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        })
    }

    pub(super) fn exact_newton_joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(cache.slices.total);
        self.exact_newton_joint_hessian_matvec_from_cache_into(
            direction,
            block_states,
            cache,
            &mut out,
        )?;
        Ok(out)
    }

    /// Allocation-free HVP entry point.  Fills `out` (length
    /// `cache.slices.total`) with `H·direction`.  `out` is zeroed on entry
    /// and fully overwritten on success.
    pub(crate) fn exact_newton_joint_hessian_matvec_from_cache_into(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        out.fill(0.0);

        // ── Rigid closed-form: scalar kernel + design row ops ────────
        if !self.effective_flex_active(block_states)? {
            let row_chunk = bms_row_chunk_size(n);
            let partial = (0..n.div_ceil(row_chunk))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_out, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        // The β-direction sub-views for the two design blocks are
                        // constant across every row in the chunk; building the
                        // `s![..]` SliceInfo and re-slicing once per chunk instead
                        // of once per row removes the per-row `ndarray::slice`
                        // churn the HVP CG inner loop pays (the matvec is called
                        // many times per Newton step). Exact: the sliced views are
                        // identical to what the per-row slice produced.
                        let dir_marginal = direction.slice(s![slices.marginal.clone()]);
                        let dir_logslope = direction.slice(s![slices.logslope.clone()]);
                        for row in start..end {
                            let marginal_eta = block_states[0].eta[row];
                            let marginal = self.marginal_link_map(marginal_eta)?;
                            let g = block_states[1].eta[row];
                            let (_, _, h) = self.rigid_row_kernel_eval(row, marginal, g)?;
                            let v_q = self.marginal_design.dot_row_view(row, dir_marginal);
                            let v_g = self.logslope_design.dot_row_view(row, dir_logslope);
                            let a_q = h[0][0] * v_q + h[0][1] * v_g;
                            let a_g = h[1][0] * v_q + h[1][1] * v_g;
                            {
                                let mut m = chunk_out.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.axpy_row_into(row, a_q, &mut m)?;
                            }
                            {
                                let mut l = chunk_out.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.axpy_row_into(row, a_g, &mut l)?;
                            }
                        }
                        Ok(chunk_out)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            *out += &partial;
            return Ok(());
        }

        // Phase-3 device-resident shortcut: when the row Hessian + designs
        // are pinned on the GPU, dispatch the HVP partial+reduce kernels and
        // return the joint-β image without ever touching the host
        // `row_primary_hessians` array.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = cache.row_primary_hessians.device() {
                match crate::families::bms::gpu::row::launch_bms_flex_row_hvp(
                    device_state,
                    direction.as_slice().expect("direction is contiguous"),
                ) {
                    Ok(host) => {
                        if host.len() != out.len() {
                            return Err(format!(
                                "BMS GPU HVP length mismatch: got {}, expected {}",
                                host.len(),
                                out.len()
                            ));
                        }
                        out.iter_mut().zip(host.iter()).for_each(|(o, &v)| *o = v);
                        return Ok(());
                    }
                    Err(err) => {
                        log::info!(
                            "[BMS exact-newton HVP] gpu_hvp_failed: {err}; falling \
                             back to CPU row-loop (this should be rare under \
                             gpu=auto and is treated as a runtime degradation)"
                        );
                    }
                }
            }
        }

        // Host-pin shortcut: when the per-row Hessian is materialised on host
        // (the legacy path before Phase 3), build the joint-β image by
        // batching the per-row primary directions and dispatching the
        // per-row matvec helper from `gpu::kernels::row_hessian_ops`. On Linux this
        // can be GPU-accelerated by `launch_row_hessian_matvec`; on every
        // host the CPU oracle `cpu_row_hessian_matvec` is the in-process
        // fallback so the call sites stay consistent. The design pullback
        // (`pullback_primary_vector_add_into`) stays on host because the
        // designs are not necessarily resident on the device in this branch.
        if let Some(host_pin) = cache.row_primary_hessians.host_pin() {
            let r_pr = primary.total;
            let mut v_rows = vec![0.0_f64; n * r_pr];
            // The per-row primary direction is a pure projection of `direction`
            // through the two design rows (`dot_row_view`); each row writes its
            // own disjoint `r_pr`-length slot in `v_rows`, so the fill is
            // embarrassingly parallel. This runs once per HVP, and the HVP is
            // called many times per Newton step (CG inner loop), so a serial
            // `n`-row fill leaves every core idle while one thread walks the
            // designs row-by-row (process-monitor `active_threads=0`).
            //
            // Deadlock-safety: `dot_row_view` on a kernel/coefficient-transform
            // design lazily materialises its dense block via an internal
            // `par_chunks_mut` build under a `OnceLock`. Touching a single row
            // serially on the calling thread before the `par_chunks_mut` fan-out
            // forces that build exactly once (matching the warm-up discipline in
            // the directional-derivative passes), so the parallel chunks below
            // read already-materialised rows in O(r_pr) with no nested lazy
            // build / nested-rayon race. Fall back to a serial fill when already
            // inside a rayon worker (an outer par_iter holds the pool) or when
            // the pool is single-threaded.
            if n > 0 {
                let mut warm = Array1::<f64>::zeros(r_pr);
                self.row_primary_direction_from_flat_into(
                    0, slices, primary, direction, &mut warm,
                )?;
                v_rows[0..r_pr].copy_from_slice(warm.as_slice().expect("contiguous"));
            }
            let fill_serial =
                rayon::current_thread_index().is_some() || rayon::current_num_threads() <= 1;
            if fill_serial {
                let mut row_dir_scratch = Array1::<f64>::zeros(r_pr);
                for row in 1..n {
                    self.row_primary_direction_from_flat_into(
                        row,
                        slices,
                        primary,
                        direction,
                        &mut row_dir_scratch,
                    )?;
                    v_rows[row * r_pr..(row + 1) * r_pr]
                        .copy_from_slice(row_dir_scratch.as_slice().expect("contiguous"));
                }
            } else {
                use rayon::iter::{IndexedParallelIterator, ParallelIterator};
                use rayon::slice::ParallelSliceMut;
                v_rows
                    .par_chunks_mut(r_pr)
                    .enumerate()
                    .skip(1)
                    .try_for_each(|(row, slot)| -> Result<(), String> {
                        let mut row_dir_scratch = Array1::<f64>::zeros(r_pr);
                        self.row_primary_direction_from_flat_into(
                            row,
                            slices,
                            primary,
                            direction,
                            &mut row_dir_scratch,
                        )?;
                        slot.copy_from_slice(row_dir_scratch.as_slice().expect("contiguous"));
                        Ok(())
                    })?;
            }
            let h_rows_arr = host_pin.hess();
            let h_rows_slice = h_rows_arr
                .as_slice()
                .expect("row_primary_hessians.hess() is row-major contiguous");
            let inputs = row_hessian_ops::RowHessianMatvecInputs {
                n_rows: n,
                r: r_pr,
                h_rows: h_rows_slice,
                v_rows: &v_rows,
            };
            let y_rows = {
                #[cfg(target_os = "linux")]
                {
                    match row_hessian_ops::launch_row_hessian_matvec(
                        row_hessian_ops::RowHessianMatvecInputs {
                            n_rows: n,
                            r: r_pr,
                            h_rows: h_rows_slice,
                            v_rows: &v_rows,
                        },
                    ) {
                        Ok(result) => result.y_rows,
                        Err(err) => {
                            log::info!(
                                "[BMS exact-newton HVP] host-pin GPU matvec failed: {err}; \
                                 falling back to CPU oracle"
                            );
                            row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                        }
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                }
            };
            // Pull back every row's action `y_rows[row]` through its design
            // rows into the joint-β image. The pullback accumulates into the
            // shared output, so fan it across rayon row chunks with a private
            // per-chunk partial + a sum reduce — numerically identical to the
            // serial single-buffer accumulation, with each worker owning its
            // own action scratch.
            let row_chunk = bms_row_chunk_size(n);
            let partial = (0..n.div_ceil(row_chunk))
                .into_par_iter()
                .try_fold(
                    || {
                        (
                            Array1::<f64>::zeros(slices.total),
                            Array1::<f64>::zeros(r_pr),
                        )
                    },
                    |(mut chunk_out, mut action_scratch), chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        for row in start..end {
                            let action_slice = &y_rows[row * r_pr..(row + 1) * r_pr];
                            action_scratch
                                .iter_mut()
                                .zip(action_slice.iter())
                                .for_each(|(dst, &src)| *dst = src);
                            self.pullback_primary_vector_add_into(
                                row,
                                slices,
                                primary,
                                &action_scratch,
                                &mut chunk_out,
                            )?;
                        }
                        Ok((chunk_out, action_scratch))
                    },
                )
                .map(|res| res.map(|(chunk_out, _)| chunk_out))
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            *out += &partial;
            return Ok(());
        }

        if let Some(tiles) = cache.row_primary_hessians.tiles() {
            if tiles.r != primary.total || tiles.n_rows != n {
                return Err(format!(
                    "BMS tiled row-primary Hessian shape mismatch: tiles n={} r={}, expected n={} r={}",
                    tiles.n_rows, tiles.r, n, primary.total
                ));
            }
            if tiles.is_empty() {
                return Ok(());
            }
            if log_exact_work(n) {
                log::info!(
                    "[BMS exact-newton HVP] route=tiled-host rows={} r={} tiles={} bytes={}",
                    n,
                    tiles.r,
                    tiles.tiles.len(),
                    tiles.total_bytes()
                );
            }
            let r_pr = primary.total;
            // Fan the tiles across rayon: each tile owns an independent row
            // block, so a per-tile partial pullback into a private accumulator
            // followed by a reduce is numerically identical to the serial
            // single-buffer accumulation (the reduction order differs only by
            // tile, and each tile contributes a disjoint set of rows). This
            // mirrors the chunked `try_fold`/`try_reduce` used by the streaming
            // fallback below and gives every worker its own row-direction /
            // action scratch, so there is no per-call `v_rows` allocation on a
            // shared path and no serial bottleneck across the ~`n/tile_rows`
            // tiles.
            let partial = tiles
                .tiles
                .par_iter()
                .try_fold(
                    || (Array1::<f64>::zeros(slices.total), Array1::<f64>::zeros(r_pr)),
                    |(mut tile_out, mut row_dir_scratch), tile| -> Result<_, String> {
                        let tile_rows = tile.rows.hess().nrows();
                        let mut v_rows = vec![0.0_f64; tile_rows * r_pr];
                        for local in 0..tile_rows {
                            let row = tile.row_start + local;
                            self.row_primary_direction_from_flat_into(
                                row,
                                slices,
                                primary,
                                direction,
                                &mut row_dir_scratch,
                            )?;
                            v_rows[local * r_pr..(local + 1) * r_pr]
                                .copy_from_slice(row_dir_scratch.as_slice().expect("contiguous"));
                        }
                        let h_rows_slice = tile.rows.hess().as_slice().expect(
                            "tiled row_primary_hessians.hess() is row-major contiguous",
                        );
                        let inputs = row_hessian_ops::RowHessianMatvecInputs {
                            n_rows: tile_rows,
                            r: r_pr,
                            h_rows: h_rows_slice,
                            v_rows: &v_rows,
                        };
                        let y_rows = {
                            #[cfg(target_os = "linux")]
                            {
                                match row_hessian_ops::launch_row_hessian_matvec(
                                    row_hessian_ops::RowHessianMatvecInputs {
                                        n_rows: tile_rows,
                                        r: r_pr,
                                        h_rows: h_rows_slice,
                                        v_rows: &v_rows,
                                    },
                                ) {
                                    Ok(result) => result.y_rows,
                                    Err(err) => {
                                        log::info!(
                                            "[BMS exact-newton HVP] tiled GPU matvec failed: {err}; \
                                             falling back to CPU oracle"
                                        );
                                        row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                                    }
                                }
                            }
                            #[cfg(not(target_os = "linux"))]
                            {
                                row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                            }
                        };
                        for local in 0..tile_rows {
                            let row = tile.row_start + local;
                            let action_slice = &y_rows[local * r_pr..(local + 1) * r_pr];
                            row_dir_scratch
                                .iter_mut()
                                .zip(action_slice.iter())
                                .for_each(|(dst, &src)| *dst = src);
                            self.pullback_primary_vector_add_into(
                                row,
                                slices,
                                primary,
                                &row_dir_scratch,
                                &mut tile_out,
                            )?;
                        }
                        Ok((tile_out, row_dir_scratch))
                    },
                )
                .map(|res| res.map(|(tile_out, _)| tile_out))
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            *out += &partial;
            return Ok(());
        }

        let row_chunk = bms_row_chunk_size(n);
        let partial = (0..n.div_ceil(row_chunk))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_out, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * row_chunk;
                    let end = (start + row_chunk).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    // Per-thread scratch for row direction — allocated once per
                    // chunk thread rather than once per row.
                    let mut row_dir = Array1::<f64>::zeros(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        self.row_primary_direction_from_flat_into(
                            row,
                            slices,
                            primary,
                            direction,
                            &mut row_dir,
                        )?;
                        let row_action =
                            if let Some(row_hess) = Self::cached_row_primary_hessian(cache, row) {
                                row_hess.dot(&row_dir)
                            } else {
                                let row_moments = cache
                                    .row_cell_moments
                                    .as_ref()
                                    .and_then(|bundle| bundle.row(row, 9));
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    cache.cell_family_forest.as_ref(),
                                    true,
                                    &mut scratch,
                                )?;
                                scratch.hess.dot(&row_dir)
                            };
                        self.pullback_primary_vector_add_into(
                            row,
                            slices,
                            primary,
                            &row_action,
                            &mut chunk_out,
                        )?;
                    }
                    Ok(chunk_out)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        *out += &partial;
        Ok(())
    }

    /// Batched multi-RHS coefficient-Hessian apply: writes `H · V` into `out`,
    /// where `V` and `out` are `(total, n_rhs)` and each column is an
    /// independent direction. Column for column the result is **numerically
    /// identical** to calling
    /// [`Self::exact_newton_joint_hessian_matvec_from_cache_into`] on each
    /// column: the same per-row primary Hessian `Hᵢ`, the same projection and
    /// pullback algebra, only with the row tiles swept once for all columns.
    ///
    /// For the tiled row-primary Hessian cache the win is structural — each
    /// tile's resident `Hᵢ` block is materialised and consumed once per tile
    /// pass instead of once per (column × tile). Dense reconstruction of the
    /// matrix-free joint Hessian (`H = H · I`) is then a single tile sweep of
    /// width `total` rather than `total` separate sweeps. Every non-tiled cache
    /// state (rigid closed-form, host-pin, device-resident, matrix-free
    /// stream) routes column-by-column through the single-vector entry point,
    /// so those paths are unchanged.
    pub(crate) fn exact_newton_joint_hessian_matvec_mat_from_cache_into(
        &self,
        v_cols: &Array2<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        out: &mut Array2<f64>,
    ) -> Result<(), String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        let n = self.y.len();
        if v_cols.nrows() != total || out.nrows() != total {
            return Err(format!(
                "BMS batched HVP: row mismatch v_cols={}x{} out={}x{} expected rows={total}",
                v_cols.nrows(),
                v_cols.ncols(),
                out.nrows(),
                out.ncols()
            ));
        }
        if v_cols.ncols() != out.ncols() {
            return Err(format!(
                "BMS batched HVP: column mismatch v_cols has {} columns, out has {}",
                v_cols.ncols(),
                out.ncols()
            ));
        }
        let n_rhs = v_cols.ncols();
        out.fill(0.0);
        if n_rhs == 0 {
            return Ok(());
        }

        // Fast path: tiled row-primary Hessian. Sweep each tile once and apply
        // its `Hᵢ` to every RHS column. Tiles own disjoint row blocks, so a
        // per-tile private `(total, n_rhs)` partial followed by a reduce is
        // identical (up to f.p. reduction order, by tile) to the serial
        // single-buffer accumulation — exactly as the single-vector tiled HVP
        // does for one column.
        if let Some(tiles) = cache.row_primary_hessians.tiles() {
            if tiles.r != primary.total || tiles.n_rows != n {
                return Err(format!(
                    "BMS tiled row-primary Hessian batched-HVP shape mismatch: tiles n={} r={}, expected n={} r={}",
                    tiles.n_rows, tiles.r, n, primary.total
                ));
            }
            if tiles.is_empty() {
                return Ok(());
            }
            let r_pr = primary.total;
            let partial = tiles
                .tiles
                .par_iter()
                .try_fold(
                    || {
                        (
                            Array2::<f64>::zeros((total, n_rhs)),
                            Array1::<f64>::zeros(total),
                            Array1::<f64>::zeros(r_pr),
                        )
                    },
                    |(mut tile_out, mut col_scratch, mut row_dir_scratch), tile| -> Result<_, String> {
                        let tile_rows = tile.rows.hess().nrows();
                        let h_rows_slice = tile.rows.hess().as_slice().expect(
                            "tiled row_primary_hessians.hess() is row-major contiguous",
                        );
                        // One `v_rows` / `y_rows` buffer reused across all RHS
                        // columns within this tile so the per-tile working set
                        // stays one column wide regardless of `n_rhs`.
                        let mut v_rows = vec![0.0_f64; tile_rows * r_pr];
                        for col in 0..n_rhs {
                            col_scratch.assign(&v_cols.column(col));
                            for local in 0..tile_rows {
                                let row = tile.row_start + local;
                                self.row_primary_direction_from_flat_into(
                                    row,
                                    slices,
                                    primary,
                                    &col_scratch,
                                    &mut row_dir_scratch,
                                )?;
                                v_rows[local * r_pr..(local + 1) * r_pr].copy_from_slice(
                                    row_dir_scratch.as_slice().expect("contiguous"),
                                );
                            }
                            let inputs = row_hessian_ops::RowHessianMatvecInputs {
                                n_rows: tile_rows,
                                r: r_pr,
                                h_rows: h_rows_slice,
                                v_rows: &v_rows,
                            };
                            let y_rows = {
                                #[cfg(target_os = "linux")]
                                {
                                    match row_hessian_ops::launch_row_hessian_matvec(
                                        row_hessian_ops::RowHessianMatvecInputs {
                                            n_rows: tile_rows,
                                            r: r_pr,
                                            h_rows: h_rows_slice,
                                            v_rows: &v_rows,
                                        },
                                    ) {
                                        Ok(result) => result.y_rows,
                                        Err(err) => {
                                            log::info!(
                                                "[BMS exact-newton batched-HVP] tiled GPU matvec failed: {err}; \
                                                 falling back to CPU oracle"
                                            );
                                            row_hessian_ops::cpu_row_hessian_matvec(
                                                &inputs,
                                            )
                                        }
                                    }
                                }
                                #[cfg(not(target_os = "linux"))]
                                {
                                    row_hessian_ops::cpu_row_hessian_matvec(&inputs)
                                }
                            };
                            let mut tile_out_col = tile_out.column_mut(col);
                            for local in 0..tile_rows {
                                let row = tile.row_start + local;
                                let action_slice = &y_rows[local * r_pr..(local + 1) * r_pr];
                                row_dir_scratch
                                    .iter_mut()
                                    .zip(action_slice.iter())
                                    .for_each(|(dst, &src)| *dst = src);
                                self.pullback_primary_vector_add_into_view(
                                    row,
                                    slices,
                                    primary,
                                    &row_dir_scratch,
                                    &mut tile_out_col,
                                )?;
                            }
                        }
                        Ok((tile_out, col_scratch, row_dir_scratch))
                    },
                )
                .map(|res| res.map(|(tile_out, _, _)| tile_out))
                .try_reduce(
                    || Array2::<f64>::zeros((total, n_rhs)),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            *out += &partial;
            return Ok(());
        }

        // Every other cache state keeps the single-vector fast paths. Loop the
        // columns through the single-vector `_into`; the result is identical.
        let mut col_in = Array1::<f64>::zeros(total);
        let mut col_out = Array1::<f64>::zeros(total);
        for col in 0..n_rhs {
            col_in.assign(&v_cols.column(col));
            self.exact_newton_joint_hessian_matvec_from_cache_into(
                &col_in,
                block_states,
                cache,
                &mut col_out,
            )?;
            out.column_mut(col).assign(&col_out);
        }
        Ok(())
    }

    pub(super) fn exact_newton_joint_hessian_diagonal_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // ── Rigid closed-form: no jets, no row contexts ──────────────
        if !self.effective_flex_active(block_states)? {
            let row_chunk = bms_row_chunk_size(n);
            let diagonal = (0..n.div_ceil(row_chunk))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        for row in start..end {
                            let marginal_eta = block_states[0].eta[row];
                            let marginal = self.marginal_link_map(marginal_eta)?;
                            let g = block_states[1].eta[row];
                            let (_, _, h) = self.rigid_row_kernel_eval(row, marginal, g)?;
                            {
                                let mut m = chunk_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design
                                    .squared_axpy_row_into(row, h[0][0], &mut m)?;
                            }
                            {
                                let mut l = chunk_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design
                                    .squared_axpy_row_into(row, h[1][1], &mut l)?;
                            }
                        }
                        Ok(chunk_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        // Phase-3 device-resident shortcut: same idea as the HVP path.
        #[cfg(target_os = "linux")]
        {
            if let Some(device_state) = cache.row_primary_hessians.device() {
                match crate::families::bms::gpu::row::launch_bms_flex_row_diagonal(device_state) {
                    Ok(host) => {
                        return Ok(Array1::<f64>::from_vec(host));
                    }
                    Err(err) => {
                        log::info!(
                            "[BMS exact-newton diag] gpu_diag_failed: {err}; falling \
                             back to CPU row-loop"
                        );
                    }
                }
            }
        }

        // Host-pin shortcut: extract every row's primary diagonal via the
        // per-row diagonal helper from `gpu::kernels::row_hessian_ops`, then perform
        // the design² accumulation on host (matches the rayon-loop algebra
        // below without rebuilding `r²` blocks per row). On Linux this uses
        // the GPU `launch_row_hessian_diag` kernel; on every host the CPU
        // oracle `cpu_row_hessian_diag` is the in-process fallback so the
        // call sites stay consistent.
        if let Some(host_pin) = cache.row_primary_hessians.host_pin() {
            let r_pr = primary.total;
            let h_rows_arr = host_pin.hess();
            let h_rows_slice = h_rows_arr
                .as_slice()
                .expect("row_primary_hessians.hess() is row-major contiguous");
            let inputs = row_hessian_ops::RowHessianDiagInputs {
                n_rows: n,
                r: r_pr,
                h_rows: h_rows_slice,
            };
            let d_rows = {
                #[cfg(target_os = "linux")]
                {
                    match row_hessian_ops::launch_row_hessian_diag(
                        row_hessian_ops::RowHessianDiagInputs {
                            n_rows: n,
                            r: r_pr,
                            h_rows: h_rows_slice,
                        },
                    ) {
                        Ok(out) => out.d_rows,
                        Err(err) => {
                            log::info!(
                                "[BMS exact-newton diag] host-pin GPU diag failed: {err}; \
                                 falling back to CPU oracle"
                            );
                            row_hessian_ops::cpu_row_hessian_diag(&inputs)
                        }
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    row_hessian_ops::cpu_row_hessian_diag(&inputs)
                }
            };
            // The per-row diagonals `d_rows` are already materialised; the
            // remaining design² accumulation is a reduction over rows (every
            // row contributes to the same marginal/logslope columns). Fan it
            // across rayon row chunks with private diagonal partials + a sum
            // reduce, exactly as the streaming fallback below does — numerically
            // identical to the serial single-buffer accumulation up to f.p.
            // reduction order, and removing the serial walk over all `n` rows.
            let row_chunk = bms_row_chunk_size(n);
            let diagonal = (0..n.div_ceil(row_chunk))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        for row in start..end {
                            let d_row_base = row * r_pr;
                            let h00 = d_rows[d_row_base];
                            let h11 = d_rows[d_row_base + 1];
                            {
                                let mut marginal_diag =
                                    chunk_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.squared_axpy_row_into(
                                    row,
                                    h00,
                                    &mut marginal_diag,
                                )?;
                            }
                            {
                                let mut logslope_diag =
                                    chunk_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.squared_axpy_row_into(
                                    row,
                                    h11,
                                    &mut logslope_diag,
                                )?;
                            }
                            if let (Some(primary_h), Some(block_h)) =
                                (primary.h.as_ref(), slices.h.as_ref())
                            {
                                for (local_idx, global_idx) in block_h.clone().enumerate() {
                                    let ii = primary_h.start + local_idx;
                                    chunk_diag[global_idx] += d_rows[d_row_base + ii];
                                }
                            }
                            if let (Some(primary_w), Some(block_w)) =
                                (primary.w.as_ref(), slices.w.as_ref())
                            {
                                for (local_idx, global_idx) in block_w.clone().enumerate() {
                                    let ii = primary_w.start + local_idx;
                                    chunk_diag[global_idx] += d_rows[d_row_base + ii];
                                }
                            }
                        }
                        Ok(chunk_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        if let Some(tiles) = cache.row_primary_hessians.tiles() {
            if tiles.r != primary.total || tiles.n_rows != n {
                return Err(format!(
                    "BMS tiled row-primary Hessian diagonal shape mismatch: tiles n={} r={}, expected n={} r={}",
                    tiles.n_rows, tiles.r, n, primary.total
                ));
            }
            if log_exact_work(n) {
                log::info!(
                    "[BMS exact-newton diag] route=tiled-host rows={} r={} tiles={} bytes={}",
                    n,
                    tiles.r,
                    tiles.tiles.len(),
                    tiles.total_bytes()
                );
            }
            let r_pr = primary.total;
            // Fan tiles across rayon with a per-tile private diagonal partial
            // and a reduce, exactly as the streaming fallback below does over
            // row chunks. Each tile owns a disjoint row block, so the
            // accumulation is order-independent up to f.p. reduction; the
            // partials sum to the same diagonal the serial loop produced.
            let diagonal = tiles
                .tiles
                .par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut tile_diag, tile| -> Result<_, String> {
                        let tile_rows = tile.rows.hess().nrows();
                        let h_rows_slice =
                            tile.rows.hess().as_slice().expect(
                                "tiled row_primary_hessians.hess() is row-major contiguous",
                            );
                        let inputs = row_hessian_ops::RowHessianDiagInputs {
                            n_rows: tile_rows,
                            r: r_pr,
                            h_rows: h_rows_slice,
                        };
                        let d_rows = {
                            #[cfg(target_os = "linux")]
                            {
                                match row_hessian_ops::launch_row_hessian_diag(
                                    row_hessian_ops::RowHessianDiagInputs {
                                        n_rows: tile_rows,
                                        r: r_pr,
                                        h_rows: h_rows_slice,
                                    },
                                ) {
                                    Ok(out) => out.d_rows,
                                    Err(err) => {
                                        log::info!(
                                            "[BMS exact-newton diag] tiled GPU diag failed: {err}; \
                                             falling back to CPU oracle"
                                        );
                                        row_hessian_ops::cpu_row_hessian_diag(&inputs)
                                    }
                                }
                            }
                            #[cfg(not(target_os = "linux"))]
                            {
                                row_hessian_ops::cpu_row_hessian_diag(&inputs)
                            }
                        };
                        for local in 0..tile_rows {
                            let row = tile.row_start + local;
                            let d_row_base = local * r_pr;
                            let h00 = d_rows[d_row_base];
                            let h11 = d_rows[d_row_base + 1];
                            {
                                let mut marginal_diag =
                                    tile_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.squared_axpy_row_into(
                                    row,
                                    h00,
                                    &mut marginal_diag,
                                )?;
                            }
                            {
                                let mut logslope_diag =
                                    tile_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.squared_axpy_row_into(
                                    row,
                                    h11,
                                    &mut logslope_diag,
                                )?;
                            }
                            if let (Some(primary_h), Some(block_h)) =
                                (primary.h.as_ref(), slices.h.as_ref())
                            {
                                for (local_idx, global_idx) in block_h.clone().enumerate() {
                                    let ii = primary_h.start + local_idx;
                                    tile_diag[global_idx] += d_rows[d_row_base + ii];
                                }
                            }
                            if let (Some(primary_w), Some(block_w)) =
                                (primary.w.as_ref(), slices.w.as_ref())
                            {
                                for (local_idx, global_idx) in block_w.clone().enumerate() {
                                    let ii = primary_w.start + local_idx;
                                    tile_diag[global_idx] += d_rows[d_row_base + ii];
                                }
                            }
                        }
                        Ok(tile_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        let row_chunk = bms_row_chunk_size(n);
        let diagonal = (0..n.div_ceil(row_chunk))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_diag, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * row_chunk;
                    let end = (start + row_chunk).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        // When the per-row primary Hessian is materialized at
                        // workspace construction (`row_primary_hessians`), the
                        // entire `r×r` block lives in the cache and we can read
                        // every diagonal entry directly. Otherwise rebuild the
                        // scratch Hessian on the fly.
                        let cached_hess = Self::cached_row_primary_hessian(cache, row);
                        if cached_hess.is_none() {
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            self.compute_row_analytic_flex_into_with_moments(
                                row,
                                block_states,
                                primary,
                                row_ctx,
                                row_moments,
                                cache.cell_family_forest.as_ref(),
                                true,
                                &mut scratch,
                            )?;
                        }
                        let h00 = if let Some(row_hess) = cached_hess {
                            row_hess[[0, 0]]
                        } else {
                            scratch.hess[[0, 0]]
                        };
                        let h11 = if let Some(row_hess) = cached_hess {
                            row_hess[[1, 1]]
                        } else {
                            scratch.hess[[1, 1]]
                        };
                        {
                            let mut marginal_diag =
                                chunk_diag.slice_mut(s![slices.marginal.clone()]);
                            self.marginal_design.squared_axpy_row_into(
                                row,
                                h00,
                                &mut marginal_diag,
                            )?;
                        }
                        {
                            let mut logslope_diag =
                                chunk_diag.slice_mut(s![slices.logslope.clone()]);
                            self.logslope_design.squared_axpy_row_into(
                                row,
                                h11,
                                &mut logslope_diag,
                            )?;
                        }

                        if let (Some(primary_h), Some(block_h)) =
                            (primary.h.as_ref(), slices.h.as_ref())
                        {
                            for (local_idx, global_idx) in block_h.clone().enumerate() {
                                let ii = primary_h.start + local_idx;
                                chunk_diag[global_idx] += if let Some(row_hess) = cached_hess {
                                    row_hess[[ii, ii]]
                                } else {
                                    scratch.hess[[ii, ii]]
                                };
                            }
                        }
                        if let (Some(primary_w), Some(block_w)) =
                            (primary.w.as_ref(), slices.w.as_ref())
                        {
                            for (local_idx, global_idx) in block_w.clone().enumerate() {
                                let ii = primary_w.start + local_idx;
                                chunk_diag[global_idx] += if let Some(row_hess) = cached_hess {
                                    row_hess[[ii, ii]]
                                } else {
                                    scratch.hess[[ii, ii]]
                                };
                            }
                        }
                    }
                    Ok(chunk_diag)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        Ok(diagonal)
    }

    pub(super) fn exact_newton_joint_psi_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psi_terms_from_cache`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, score vector, Hessian operator blocks) is accumulated with the
    /// row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn exact_newton_joint_psi_terms_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let axis = self.resolve_psi_axis_spec(derivative_blocks, block_idx, local_idx)?;
        let mut results = self.run_psi_row_pass_for_axes(block_states, cache, options, &[axis])?;
        Ok(Some(results.remove(0)))
    }

    pub(super) fn resolve_psi_axis_spec(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        block_idx: usize,
        local_idx: usize,
    ) -> Result<PsiAxisSpec, String> {
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi terms only support marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;
        Ok(PsiAxisSpec {
            block_idx,
            idx_primary: if block_idx == 0 { 0 } else { 1 },
            psi_map,
        })
    }

    pub(super) fn run_psi_row_pass_for_axes(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
        axes: &[PsiAxisSpec],
    ) -> Result<Vec<ExactNewtonJointPsiTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let k = axes.len();

        // Eager-prime the per-row uncontracted third-derivative cache *before*
        // entering the per-axis row `par_iter` so the build's own `par_iter`
        // does not nest inside an active rayon job. Subsequent ψ-axis sweeps
        // hit the cache via O(1) lookups in `rigid_third_full_cached`. Skipped
        // on the FLEX path because that branch routes through the flex jet
        // machinery, which has its own row-cell-moments cache.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_third_full_cached(block_states, cache, 0)?;
            ensure_finite_third_full_cache_row(
                warmed,
                "run_psi_row_pass_for_axes rigid third-cache warm-up",
            )?;
        }
        // FLEX analogue: prewarm the degree-15 cell-moment bundle so the
        // per-row third-order recompute reuses prebuilt moments instead of
        // recomputing them per row on every operator application (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 15)?;

        // Block-local accumulator path: avoids O(n p^2) dense Hessian
        // materialization by keeping one accumulator per ψ axis in the
        // rayon fold.
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let make_acc = || -> Vec<(f64, Array1<f64>, BernoulliBlockHessianAccumulator)> {
            (0..k)
                .map(|_| {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                })
                .collect()
        };
        let folded = weighted_rows
            .par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_ctx = Self::row_ctx(cache, row);
                let (f_pi, f_pipi_base) = self.compute_row_primary_gradient_hessian_reusing_cache(
                    row,
                    block_states,
                    primary,
                    row_ctx,
                    cache,
                )?;
                for (axis_idx, axis) in axes.iter().enumerate() {
                    // Single psi-map row materialization shared by `dir` and
                    // `psi_row`; the prior code paths each issued an
                    // independent `psi_map.row_vector(row)` call for the
                    // same (row, axis) which doubled the per-row operator
                    // dispatch cost for joint-spatial Hessian builds.
                    let psi_local = axis
                        .psi_map
                        .row_vector(row)
                        .map_err(|e| format!("bernoulli psi map row {row}: {e}"))?;
                    let dir_idx = if axis.block_idx == 0 {
                        primary.q
                    } else {
                        primary.logslope
                    };
                    let mut dir = Array1::<f64>::zeros(primary.total);
                    dir[dir_idx] = psi_local.dot(&block_states[axis.block_idx].beta);
                    let mut f_pipi = f_pipi_base.clone();
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &dir,
                    )?;
                    let psi_row = BlockPsiRow {
                        block_idx: axis.block_idx,
                        range: if axis.block_idx == 0 {
                            slices.marginal.clone()
                        } else {
                            slices.logslope.clone()
                        },
                        local_vec: psi_local,
                    };
                    let mut f_pipi_dir = f_pipi.dot(&dir);
                    if w != 1.0 {
                        f_pipi.mapv_inplace(|v| v * w);
                        third.mapv_inplace(|v| v * w);
                        f_pipi_dir.mapv_inplace(|v| v * w);
                    }
                    let slot = &mut acc[axis_idx];
                    slot.0 += w * f_pi.dot(&dir);
                    slot.1
                        .slice_mut(s![psi_row.range.clone()])
                        .scaled_add(w * f_pi[axis.idx_primary], &psi_row.local_vec);
                    slot.1 += &self.pullback_primary_vector(row, slices, primary, &f_pipi_dir)?;

                    let right_primary = f_pipi.row(axis.idx_primary).to_owned();
                    slot.2.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        axis.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    slot.2.add_pullback(self, row, slices, primary, &third);
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                for (l, r) in left.iter_mut().zip(right.into_iter()) {
                    l.0 += r.0;
                    l.1 += &r.1;
                    l.2.add(&r.2);
                }
                Ok(left)
            })?;

        let mut out = Vec::with_capacity(k);
        for (objective_psi, score_psi, block_acc) in folded.into_iter() {
            out.push(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(block_acc.into_operator(slices))),
            });
        }
        Ok(out)
    }

    pub(super) fn exact_newton_joint_psisecond_order_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psisecond_order_terms_from_cache`.
    /// See `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psisecond_order_terms_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_i, local_i)) = psi_derivative_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = psi_derivative_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv_i = &derivative_blocks[block_i][local_i];
        let deriv_j = &derivative_blocks[block_j][local_j];
        let (p_psi_i, label_i) = match block_i {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        };
        let (p_psi_j, label_j) = match block_j {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_j}"
                ));
            }
        };

        // Build psi design maps once outside the row loop.
        let psi_map_i = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_i,
            n,
            p_psi_i,
            0..n,
            label_i,
            &self.policy,
        )?;
        let psi_map_j = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            n,
            p_psi_j,
            0..n,
            label_j,
            &self.policy,
        )?;
        let psi_map_ij = if block_i == block_j {
            Some(
                crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i,
                    deriv_j,
                    local_j,
                    n,
                    p_psi_i,
                    0..n,
                    label_i,
                    &self.policy,
                )?,
            )
        } else {
            None
        };

        // Prewarm the high-degree full-row bundle from serial setup code. Row
        // kernels only read existing bundles, so parallel workers never launch
        // duplicate full-`n` degree-21 builds on first touch.
        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;

        // Block-local accumulator path for second-order psi terms
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let (objective_psi_psi, score_psi_psi, block_acc) = weighted_rows
            .par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    {
                        // Materialize each psi-design row once and reuse for
                        // both the primary-space direction and the
                        // BlockPsiRow embedding (previously two separate
                        // psi_map.row_vector(row) calls per map per row).
                        let psi_local_i = psi_map_i
                            .row_vector(row)
                            .map_err(|e| format!("bernoulli psi map_i row {row}: {e}"))?;
                        let psi_local_j = psi_map_j
                            .row_vector(row)
                            .map_err(|e| format!("bernoulli psi map_j row {row}: {e}"))?;
                        let dir_idx_i = if block_i == 0 {
                            primary.q
                        } else {
                            primary.logslope
                        };
                        let dir_idx_j = if block_j == 0 {
                            primary.q
                        } else {
                            primary.logslope
                        };
                        let mut dir_i = Array1::<f64>::zeros(primary.total);
                        dir_i[dir_idx_i] = psi_local_i.dot(&block_states[block_i].beta);
                        let mut dir_j = Array1::<f64>::zeros(primary.total);
                        dir_j[dir_idx_j] = psi_local_j.dot(&block_states[block_j].beta);

                        // dir_ij and br_ij share psi_map_ij; materialize once.
                        let (dir_ij, psi_local_ij) = if let Some(ref pm_ij) = psi_map_ij {
                            if block_i != block_j {
                                (Array1::<f64>::zeros(primary.total), None)
                            } else {
                                let v = pm_ij
                                    .row_vector(row)
                                    .map_err(|e| format!("bernoulli psi map_ij row {row}: {e}"))?;
                                let mut d = Array1::<f64>::zeros(primary.total);
                                d[dir_idx_i] = v.dot(&block_states[block_i].beta);
                                (d, Some(v))
                            }
                        } else {
                            (Array1::<f64>::zeros(primary.total), None)
                        };
                        let row_ctx = Self::row_ctx(cache, row);
                        let (mut f_pi, mut f_pipi) = self
                            .compute_row_primary_gradient_hessian_reusing_cache(
                                row,
                                block_states,
                                primary,
                                row_ctx,
                                cache,
                            )?;
                        let mut third_i = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                        )?;
                        let mut third_j = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_j,
                        )?;
                        let mut fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                            &dir_j,
                        )?;
                        // Per-row HT weighting: scale every row contribution
                        // (gradient, Hessian, third, fourth) by w before
                        // accumulation. The post-sum scalar that the legacy
                        // code path applied is biased under variable weights.
                        if w != 1.0 {
                            f_pi.mapv_inplace(|v| v * w);
                            f_pipi.mapv_inplace(|v| v * w);
                            third_i.mapv_inplace(|v| v * w);
                            third_j.mapv_inplace(|v| v * w);
                            fourth.mapv_inplace(|v| v * w);
                        }
                        let br_i = BlockPsiRow {
                            block_idx: block_i,
                            range: if block_i == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: psi_local_i,
                        };
                        let br_j = BlockPsiRow {
                            block_idx: block_j,
                            range: if block_j == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: psi_local_j,
                        };
                        let br_ij = psi_local_ij.map(|v| BlockPsiRow {
                            block_idx: block_i,
                            range: if block_i == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            },
                            local_vec: v,
                        });

                        // --- scalar and score accumulation (unchanged) ---
                        acc.0 += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            acc.1
                                .slice_mut(s![bij.range.clone()])
                                .scaled_add(f_pi[idx_ij], &bij.local_vec);
                        }
                        acc.1
                            .slice_mut(s![br_i.range.clone()])
                            .scaled_add(f_pipi.row(idx_i).dot(&dir_j), &br_i.local_vec);
                        acc.1
                            .slice_mut(s![br_j.range.clone()])
                            .scaled_add(f_pipi.row(idx_j).dot(&dir_i), &br_j.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir_ij),
                        )?;
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &third_i.dot(&dir_j),
                        )?;

                        // --- Hessian: bij outer pullback(f_pipi[idx_ij,:]) + transpose ---
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            let right_primary_ij = f_pipi.row(idx_ij).to_owned();
                            acc.2.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                bij.block_idx,
                                &bij.local_vec,
                                &right_primary_ij,
                            );
                        }

                        // --- br_i outer br_j * f_pipi[[idx_i, idx_j]] + transpose ---
                        let scalar_ij = f_pipi[[idx_i, idx_j]];
                        acc.2.add_psi_psi_outer(
                            block_i,
                            &br_i.local_vec,
                            block_j,
                            &br_j.local_vec,
                            scalar_ij,
                        );

                        // --- br_i outer pullback(third_j[idx_i,:]) + transpose ---
                        let right_primary_i = third_j.row(idx_i).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_i,
                            &br_i.local_vec,
                            &right_primary_i,
                        );

                        // --- br_j outer pullback(third_i[idx_j,:]) + transpose ---
                        let right_primary_j = third_i.row(idx_j).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_j,
                            &br_j.local_vec,
                            &right_primary_j,
                        );

                        // --- fourth tensor pullback ---
                        acc.2.add_pullback(self, row, slices, primary, &fourth);

                        // --- third_ij tensor pullback ---
                        let mut third_ij = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_ij,
                        )?;
                        if w != 1.0 {
                            third_ij.mapv_inplace(|v| v * w);
                        }
                        acc.2.add_pullback(self, row, slices, primary, &third_ij);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        // Per-row HT weighting was applied inside the closure (every
        // gradient / Hessian / third / fourth tensor scaled by `w` before
        // accumulation), so the unbiased estimator is already in
        // `block_acc` and no post-sum rescale is required.
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
        }))
    }

    /// Direction-contracted second-order ψ terms (#740).
    ///
    /// Returns, for every non-σ ψ output row `i`, the `α`-contraction of the
    /// per-pair second-order terms against the combined ψ-direction
    /// `ψ(α) = Σ_j alpha_psi[j] · ψ_j`:
    ///
    /// ```text
    ///   objective[i] = Σ_j α_j V_{ψ_i ψ_j}
    ///   score[i,:]   = Σ_j α_j g_{ψ_i ψ_j}
    ///   hessian[i]   = Σ_j α_j D²_β H_L[ψ_i, ψ_j]  (as a block-Hessian operator)
    /// ```
    ///
    /// This is the single-pass generalization of
    /// [`Self::exact_newton_joint_psisecond_order_terms_from_cache_with_options`]:
    /// instead of `K²` per-pair calls (each tracing a distinct
    /// `D²_β H_L[ψ_i, ψ_j]` operator at O(n·r), see `compute_base_h2_traces`),
    /// it streams the data rows ONCE and, per row, contracts the j-leg quantities
    /// (`dir_j`, `psi_local_j`, `third_j`, the same-block cross design term
    /// `dir_ij`, and the `f_pipi`/`f_pi` projections onto the j-leg) into their
    /// `α`-combinations across all non-σ axes before accumulating each of the `K`
    /// output rows. The heavy per-row third/fourth jet is the same cached tensor
    /// the per-pair path reads (`rigid_third_full_cached`/
    /// `rigid_fourth_full_cached` and the FLEX axis-tensor caches), so the only
    /// change is which directions are contracted — the row math, weighting, and
    /// `BernoulliBlockHessianAccumulator` pullbacks are identical, term for term,
    /// to the per-pair fold above. Exactness is checked by
    /// `profiled_theta_hvp_outer_hessian_fd`.
    ///
    /// Returns `Ok(None)` when any participating axis cannot resolve a primary
    /// block (marginal/log-slope) location, so the caller keeps the exact
    /// per-pair fallback.
    pub(crate) fn exact_newton_joint_psisecond_order_terms_contracted_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        alpha_psi: &[f64],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderContracted>, String>
    {
        use crate::reml_contracts::DriftDerivResult;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let psi_dim: usize = derivative_blocks.iter().map(Vec::len).sum();
        if alpha_psi.len() != psi_dim {
            return Err(format!(
                "bernoulli marginal-slope contracted second-order: alpha_psi length {} != psi_dim {}",
                alpha_psi.len(),
                psi_dim
            ));
        }

        // Resolve every ψ axis to its (block, local) primary location and build
        // its single-axis design map once. A σ-aux or otherwise unresolvable axis
        // disables the contracted path (caller falls back to exact per-pair).
        struct AxisInfo {
            pub(crate) block: usize,
            pub(crate) dir_idx: usize,
            pub(crate) map: crate::families::custom_family::PsiDesignMap,
            pub(crate) deriv_block: usize,
            pub(crate) deriv_local: usize,
        }
        let mut axes: Vec<AxisInfo> = Vec::with_capacity(psi_dim);
        for psi_index in 0..psi_dim {
            let Some((block, local)) = psi_derivative_location(derivative_blocks, psi_index) else {
                return Ok(None);
            };
            let (p_psi, label) = match block {
                0 => (
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                ),
                1 => (
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                ),
                _ => return Ok(None),
            };
            let deriv = &derivative_blocks[block][local];
            let map = crate::families::custom_family::resolve_custom_family_x_psi_map(
                deriv,
                n,
                p_psi,
                0..n,
                label,
                &self.policy,
            )?;
            let dir_idx = if block == 0 {
                primary.q
            } else {
                primary.logslope
            };
            axes.push(AxisInfo {
                block,
                dir_idx,
                map,
                deriv_block: block,
                deriv_local: local,
            });
        }

        // Same-block second design maps ∂²X/∂ψ_i∂ψ_j, built once per (i, j)
        // same-block pair. These are the bilinear cross terms; the contracted
        // path needs Σ_j α_j (∂²X/∂ψ_i∂ψ_j · β) per output row i.
        let mut cross_maps: std::collections::HashMap<
            (usize, usize),
            crate::families::custom_family::PsiDesignMap,
        > = std::collections::HashMap::new();
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                if alpha_psi[j] == 0.0 {
                    continue;
                }
                if axes[i].block != axes[j].block {
                    continue;
                }
                let p_psi = if axes[i].block == 0 {
                    self.marginal_design.ncols()
                } else {
                    self.logslope_design.ncols()
                };
                let label = if axes[i].block == 0 {
                    "BernoulliMarginalSlopeFamily marginal"
                } else {
                    "BernoulliMarginalSlopeFamily log-slope"
                };
                let deriv_i = &derivative_blocks[axes[i].deriv_block][axes[i].deriv_local];
                let deriv_j = &derivative_blocks[axes[j].deriv_block][axes[j].deriv_local];
                let map = crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i,
                    deriv_j,
                    axes[j].deriv_local,
                    n,
                    p_psi,
                    0..n,
                    label,
                    &self.policy,
                )?;
                cross_maps.insert((i, j), map);
            }
        }

        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_psisecond_order_terms_contracted rigid fourth-cache warm-up",
            )?;
        }

        // One accumulator per output row i (the K D²_β H_L[ψ_i, ψ(α)] block
        // operators), plus the contracted objective scalar and score vector per
        // output row. The data rows are streamed ONCE; every output row reads the
        // same per-row primary grad/Hess and the same cached third/fourth jets.
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let per_row = weighted_rows
            .par_iter()
            .try_fold(
                || {
                    (
                        Array1::<f64>::zeros(psi_dim),
                        Array2::<f64>::zeros((psi_dim, slices.total)),
                        (0..psi_dim)
                            .map(|_| BernoulliBlockHessianAccumulator::new(slices))
                            .collect::<Vec<_>>(),
                    )
                },
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_ctx = Self::row_ctx(cache, row);
                    let (f_pi, f_pipi) = self.compute_row_primary_gradient_hessian_reusing_cache(
                        row,
                        block_states,
                        primary,
                        row_ctx,
                        cache,
                    )?;

                    // Per-axis row quantities (computed once per row, reused by
                    // every output row through their α-combinations).
                    let mut psi_local: Vec<Array1<f64>> = Vec::with_capacity(psi_dim);
                    let mut dir: Vec<Array1<f64>> = Vec::with_capacity(psi_dim);
                    for axis in &axes {
                        let pl = axis
                            .map
                            .row_vector(row)
                            .map_err(|e| format!("bernoulli psi contracted map row {row}: {e}"))?;
                        let mut d = Array1::<f64>::zeros(primary.total);
                        d[axis.dir_idx] = pl.dot(&block_states[axis.block].beta);
                        psi_local.push(pl);
                        dir.push(d);
                    }

                    // Combined second leg ψ(α) = Σ_j α_j ψ_j and its third
                    // contraction third(dir(α)) (linear in direction).
                    let mut dir_alpha = Array1::<f64>::zeros(primary.total);
                    for (j, d) in dir.iter().enumerate() {
                        if alpha_psi[j] != 0.0 {
                            dir_alpha.scaled_add(alpha_psi[j], d);
                        }
                    }
                    let third_alpha = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &dir_alpha,
                    )?;

                    for i in 0..psi_dim {
                        let block_i = axes[i].block;
                        let idx_i = if block_i == 0 { 0 } else { 1 };
                        let dir_i = &dir[i];
                        // third_i = third(dir_i); reused below.
                        let third_i = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            dir_i,
                        )?;
                        // fourth(dir_i, dir(α)) — bilinear, one cached-tensor
                        // contraction per (output row i, data row).
                        let mut fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            dir_i,
                            &dir_alpha,
                        )?;

                        // Combined cross design term Σ_j α_j dir_ij(i,j) and its
                        // primary block index (same as dir_i's, by construction).
                        let mut dir_ij_alpha = Array1::<f64>::zeros(primary.total);
                        let mut psi_local_ij_alpha = Array1::<f64>::zeros(psi_local[i].len());
                        let mut have_ij = false;
                        for j in 0..psi_dim {
                            if alpha_psi[j] == 0.0 {
                                continue;
                            }
                            if let Some(map_ij) = cross_maps.get(&(i, j)) {
                                let v = map_ij.row_vector(row).map_err(|e| {
                                    format!("bernoulli psi contracted map_ij row {row}: {e}")
                                })?;
                                let scaled = v.dot(&block_states[block_i].beta) * alpha_psi[j];
                                dir_ij_alpha[axes[i].dir_idx] += scaled;
                                psi_local_ij_alpha.scaled_add(alpha_psi[j], &v);
                                have_ij = true;
                            }
                        }

                        let br_i_range = if block_i == 0 {
                            slices.marginal.clone()
                        } else {
                            slices.logslope.clone()
                        };

                        // Per-row HT weighting, applied to every contribution
                        // before accumulation (matches the per-pair fold).
                        let mut f_pi_w = f_pi.clone();
                        let mut f_pipi_w = f_pipi.clone();
                        let mut third_i_w = third_i;
                        let mut third_alpha_w = third_alpha.clone();
                        if w != 1.0 {
                            f_pi_w.mapv_inplace(|v| v * w);
                            f_pipi_w.mapv_inplace(|v| v * w);
                            third_i_w.mapv_inplace(|v| v * w);
                            third_alpha_w.mapv_inplace(|v| v * w);
                            fourth.mapv_inplace(|v| v * w);
                        }

                        // --- scalar (objective) accumulation:
                        //   Σ_j α_j [ dir_i·(f_pipi·dir_j) + f_pi·dir_ij ]
                        //   = dir_i·(f_pipi·dir(α)) + f_pi·dir_ij(α).
                        acc.0[i] +=
                            dir_i.dot(&f_pipi_w.dot(&dir_alpha)) + f_pi_w.dot(&dir_ij_alpha);

                        // --- score accumulation (mirrors per-pair lines, j→α):
                        // (a) bij term: f_pi[idx_ij] · psi_local_ij(α)
                        if have_ij {
                            acc.1
                                .row_mut(i)
                                .slice_mut(s![br_i_range.clone()])
                                .scaled_add(f_pi_w[idx_i], &psi_local_ij_alpha);
                        }
                        // (b) br_i term: (f_pipi.row(idx_i)·dir(α)) · psi_local_i
                        acc.1
                            .row_mut(i)
                            .slice_mut(s![br_i_range.clone()])
                            .scaled_add(f_pipi_w.row(idx_i).dot(&dir_alpha), &psi_local[i]);
                        // (c) br_j term contracted: Σ_j α_j (f_pipi.row(idx_j)·dir_i) psi_local_j
                        for j in 0..psi_dim {
                            if alpha_psi[j] == 0.0 {
                                continue;
                            }
                            let idx_j = if axes[j].block == 0 { 0 } else { 1 };
                            let coeff = alpha_psi[j] * f_pipi_w.row(idx_j).dot(dir_i);
                            let range_j = if axes[j].block == 0 {
                                slices.marginal.clone()
                            } else {
                                slices.logslope.clone()
                            };
                            acc.1
                                .row_mut(i)
                                .slice_mut(s![range_j])
                                .scaled_add(coeff, &psi_local[j]);
                        }
                        // (d) primary pullback of f_pipi·dir_ij(α)
                        {
                            let pulled = self.pullback_primary_vector(
                                row,
                                slices,
                                primary,
                                &f_pipi_w.dot(&dir_ij_alpha),
                            )?;
                            let mut srow = acc.1.row_mut(i);
                            srow += &pulled;
                        }
                        // (e) primary pullback of third_i·dir(α)
                        {
                            let pulled = self.pullback_primary_vector(
                                row,
                                slices,
                                primary,
                                &third_i_w.dot(&dir_alpha),
                            )?;
                            let mut srow = acc.1.row_mut(i);
                            srow += &pulled;
                        }

                        // --- Hessian accumulation (mirrors per-pair, j→α): ---
                        let block_acc = &mut acc.2[i];
                        // bij outer pullback(f_pipi.row(idx_ij)) + transpose
                        if have_ij {
                            let right_primary_ij = f_pipi_w.row(idx_i).to_owned();
                            block_acc.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                block_i,
                                &psi_local_ij_alpha,
                                &right_primary_ij,
                            );
                        }
                        // br_i outer br_j(α) * f_pipi[[idx_i, idx_j]] (contracted over j)
                        for j in 0..psi_dim {
                            if alpha_psi[j] == 0.0 {
                                continue;
                            }
                            let idx_j = if axes[j].block == 0 { 0 } else { 1 };
                            let scalar_ij = alpha_psi[j] * f_pipi_w[[idx_i, idx_j]];
                            if scalar_ij != 0.0 {
                                block_acc.add_psi_psi_outer(
                                    block_i,
                                    &psi_local[i],
                                    axes[j].block,
                                    &psi_local[j],
                                    scalar_ij,
                                );
                            }
                        }
                        // br_i outer pullback(third(dir(α)).row(idx_i)) + transpose
                        {
                            let right_primary_i = third_alpha_w.row(idx_i).to_owned();
                            block_acc.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                block_i,
                                &psi_local[i],
                                &right_primary_i,
                            );
                        }
                        // br_j(α) outer pullback(third_i.row(idx_j)) + transpose
                        for j in 0..psi_dim {
                            if alpha_psi[j] == 0.0 {
                                continue;
                            }
                            let idx_j = if axes[j].block == 0 { 0 } else { 1 };
                            let mut right_primary_j = third_i_w.row(idx_j).to_owned();
                            right_primary_j.mapv_inplace(|v| v * alpha_psi[j]);
                            block_acc.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axes[j].block,
                                &psi_local[j],
                                &right_primary_j,
                            );
                        }
                        // fourth tensor pullback (fourth already α-weighted via dir(α))
                        block_acc.add_pullback(self, row, slices, primary, &fourth);
                        // third_ij(α) tensor pullback
                        if have_ij {
                            let mut third_ij = self.row_primary_third_contracted_recompute(
                                row,
                                block_states,
                                cache,
                                row_ctx,
                                &dir_ij_alpha,
                            )?;
                            if w != 1.0 {
                                third_ij.mapv_inplace(|v| v * w);
                            }
                            block_acc.add_pullback(self, row, slices, primary, &third_ij);
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        Array1::<f64>::zeros(psi_dim),
                        Array2::<f64>::zeros((psi_dim, slices.total)),
                        (0..psi_dim)
                            .map(|_| BernoulliBlockHessianAccumulator::new(slices))
                            .collect::<Vec<_>>(),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += &right.0;
                    left.1 += &right.1;
                    for (l, r) in left.2.iter_mut().zip(right.2.into_iter()) {
                        l.add(&r);
                    }
                    Ok(left)
                },
            )?;

        let (objective, score, accs) = per_row;
        let hessian: Vec<DriftDerivResult> = accs
            .into_iter()
            .map(|acc| DriftDerivResult::Operator(Arc::new(acc.into_operator(slices))))
            .collect();
        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderContracted {
                objective,
                score,
                hessian,
            },
        ))
    }

    pub(super) fn exact_newton_joint_psihessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_psihessian_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// are visited and the accumulated dense Hessian-action matrix uses
    /// per-row Horvitz-Thompson inverse-inclusion weights. See
    /// `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;

        // FLEX: prewarm the degree-21 cell-moment bundle so per-row third/
        // fourth recompute reuses prebuilt moments rather than recomputing
        // them per row on every operator application. The fourth-order
        // recompute reads degree-21 cells, and a degree-21 bundle also serves
        // the third-order (degree-15) lookups (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;

        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let block_acc = weighted_rows
            .par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let psi_dir = self.row_primary_psi_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        block_states,
                        primary,
                    )?;
                    let psi_action = self.row_primary_psi_action_on_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        slices,
                        d_beta_flat,
                        primary,
                    )?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third_beta = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    let mut fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                        &psi_dir,
                    )?;
                    if w != 1.0 {
                        third_beta.mapv_inplace(|v| v * w);
                        fourth.mapv_inplace(|v| v * w);
                    }
                    let psi_row = self.block_psi_row_from_map(row, block_idx, &psi_map, slices)?;
                    let right_primary = third_beta.row(idx_primary).to_owned();
                    acc.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        psi_row.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    acc.add_pullback(self, row, slices, primary, &fourth);
                    let mut third_action = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &psi_action,
                    )?;
                    if w != 1.0 {
                        third_action.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third_action);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the Hessian directional derivative
    /// from a cached eval. The default-options variant is unused (the
    /// workspace always threads its own `BlockwiseFitOptions`), so the legacy
    /// non-`_with_options` shim is omitted.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated block Hessian operator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the
    /// `HyperOperator`. See
    /// `exact_newton_joint_psi_terms_from_cache_with_options` for the
    /// row-iter / weighting contract.
    pub(crate) fn exact_newton_joint_psihessian_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian operator only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            n,
            p_psi,
            0..n,
            psi_label,
            &self.policy,
        )?;

        // FLEX: prewarm the degree-21 cell-moment bundle so per-row third/
        // fourth recompute reuses prebuilt moments rather than recomputing
        // them per row on every operator application. The fourth-order
        // recompute reads degree-21 cells, and a degree-21 bundle also serves
        // the third-order (degree-15) lookups (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;

        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let block_acc = weighted_rows
            .par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let psi_dir = self.row_primary_psi_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        block_states,
                        primary,
                    )?;
                    let psi_action = self.row_primary_psi_action_on_direction_from_map(
                        row,
                        block_idx,
                        &psi_map,
                        slices,
                        d_beta_flat,
                        primary,
                    )?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third_beta = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    let mut fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                        &psi_dir,
                    )?;
                    if w != 1.0 {
                        third_beta.mapv_inplace(|v| v * w);
                        fourth.mapv_inplace(|v| v * w);
                    }
                    let psi_row = self.block_psi_row_from_map(row, block_idx, &psi_map, slices)?;
                    let right_primary = third_beta.row(idx_primary).to_owned();
                    acc.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        psi_row.block_idx,
                        &psi_row.local_vec,
                        &right_primary,
                    );
                    acc.add_pullback(self, row, slices, primary, &fourth);
                    let mut third_action = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &psi_action,
                    )?;
                    if w != 1.0 {
                        third_action.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third_action);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(super) fn exact_newton_joint_hessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
            block_states,
            d_beta_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `exact_newton_joint_hessian_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated dense Hessian directional-derivative
    /// matrix uses per-row Horvitz-Thompson inverse-inclusion weights before
    /// densification.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);

        // ── Rigid closed-form: 3rd-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = self.rigid_row_third_contracted(row, marginal, g, dq, dg)?;
                        acc.add_pullback_rigid_2x2(self, row, &t, w);
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        // FLEX: prewarm the degree-15 cell-moment bundle so the per-row
        // third-order recompute reuses prebuilt moments rather than
        // recomputing them per row on every operator application (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 15)?;

        let block_acc = weighted_rows
            .par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the joint-Hessian directional
    /// derivative. The default-options shim is omitted because the
    /// `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace` always threads
    /// its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the operator.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);

        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let marginal_eta = block_states[0].eta[row];
                        let marginal = self.marginal_link_map(marginal_eta)?;
                        let g = block_states[1].eta[row];
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = self.rigid_row_third_contracted(row, marginal, g, dq, dg)?;
                        acc.add_pullback_rigid_2x2(self, row, &t, w);
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| -> Result<_, String> {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        // FLEX: prewarm the degree-15 cell-moment bundle so the per-row
        // third-order recompute reuses prebuilt moments rather than
        // recomputing them per row on every operator application (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 15)?;

        let block_acc = weighted_rows
            .par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dir =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let mut third = self.row_primary_third_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dir,
                    )?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &third);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operators_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flats: &[Array1<f64>],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        if d_beta_flats.is_empty() {
            return Ok(Vec::new());
        }
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let make_accs = || {
            (0..d_beta_flats.len())
                .map(|_| BernoulliBlockHessianAccumulator::new(slices))
                .collect::<Vec<_>>()
        };
        let started = std::time::Instant::now();

        let n_rows = weighted_rows.len();
        let n_dirs = d_beta_flats.len();
        let flex_active = self.effective_flex_active(block_states)?;
        let bundle_present = cache.row_cell_moments.is_some();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS batched dH n={n} rows={n_rows} p={} dirs={n_dirs} flex={flex_active} cell_moments_bundle={bundle_present}",
            slices.total
        ));
        log::info!(
            "[BMS batched dH start] n={} rows={} p={} dirs={} flex={} cell_moments_bundle={}",
            n,
            n_rows,
            slices.total,
            n_dirs,
            flex_active,
            bundle_present,
        );
        let progress = Arc::new(AtomicUsize::new(0));
        let progress_step = (n_rows / 8).max(1);
        let progress_started = started;
        let bump_progress = |progress: &AtomicUsize| {
            let now = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if now == n_rows || now.is_multiple_of(progress_step) {
                log::info!(
                    "[BMS batched dH progress] rows={}/{} dirs={} elapsed={:.3}s",
                    now,
                    n_rows,
                    n_dirs,
                    progress_started.elapsed().as_secs_f64(),
                );
            }
        };
        let dense_contiguous_rows = weighted_rows.len() == n
            && weighted_rows
                .iter()
                .enumerate()
                .all(|(row, wr)| wr.index == row && wr.weight == 1.0);
        // Pre-warm the per-row caches the row loop transitively reaches, so the
        // first row to enter the par_iter does not lazily run a nested
        // `into_par_iter()` inside a lazy cache initializer / `RayonSafeOnce`
        // race and starve the outer pool. Two distinct hazards:
        //   1. The rigid third-derivative tensor cache (`rigid_third_full_cached`).
        //      Used by the chunked / else branches when `!flex_active`; harmless
        //      to populate even when the `!flex_active` rank-1 branch will not
        //      consult it.
        //   2. Lazy dense materialization of any kernel / coefficient-transform
        //      design operator (`ChunkedKernelDesignOperator::materialized_combined`
        //      → `build_row_chunk_combined` runs `par_chunks_mut`). Multiple
        //      racing outer workers each spawn that nested parallel build, and
        //      with the rigid rank-1 row body (`dot_row_view`, `syr_row_into`)
        //      every row touches both designs — guaranteed contention. Touching
        //      a single row on the main thread before the par_iter forces the
        //      lazy build once, leaving the par_iter body to read
        //      already-materialized `Array2` rows in O(p).
        // Mirrors the same discipline applied in
        // `exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options`.
        if !flex_active && n > 0 {
            let warmed = self.rigid_third_full_cached(block_states, cache, 0)?;
            ensure_finite_third_full_cache_row(
                warmed,
                "compute_gradient_and_hessian_via_psi_axes rigid third-cache warm-up",
            )?;
        }
        // FLEX analogue: prewarm the degree-15 cell-moment bundle once, serially,
        // so the per-row third-order recompute reuses prebuilt moments instead
        // of recomputing them per row across all directions/chunks (gam#683).
        if flex_active && n > 0 {
            self.prewarm_flex_cell_bundle(block_states, cache, 15)?;
        }
        if n > 0 {
            let warm_marg = Array1::<f64>::zeros(slices.marginal.end - slices.marginal.start);
            let marginal_probe = self.marginal_design.dot_row_view(0, warm_marg.view());
            if !marginal_probe.is_finite() {
                return Err(
                    "compute_gradient_and_hessian_via_psi_axes marginal design warm-up produced a non-finite value"
                        .to_string(),
                );
            }
            let warm_log = Array1::<f64>::zeros(slices.logslope.end - slices.logslope.start);
            let logslope_probe = self.logslope_design.dot_row_view(0, warm_log.view());
            if !logslope_probe.is_finite() {
                return Err(
                    "compute_gradient_and_hessian_via_psi_axes logslope design warm-up produced a non-finite value"
                        .to_string(),
                );
            }
        }
        // Even with the warm-up above, fall back to a serial row loop when the
        // par_iter cannot pay for its own dispatch overhead, or when we are
        // already inside a rayon worker (so an outer par_iter is holding pool
        // slots and a nested `into_par_iter` here would risk pool starvation
        // on the LRU mutex inside `evaluate_cell_derivative_moments_lru`,
        // etc.). At large-scale n_rows the per-row body's design materialization
        // and pullback work dominates dispatch, so the par_iter is preserved.
        const ROW_PAR_MIN_ROWS: usize = 4_096;
        let run_rows_serial = rayon::current_thread_index().is_some()
            || rayon::current_num_threads() <= 1
            || n_rows < ROW_PAR_MIN_ROWS;
        let mut accs = if !flex_active && dense_contiguous_rows {
            // RIGID BLAS-3 chunked Gram path. The rigid directional Hessian
            // drift for direction `idx` is exactly
            //   `H_drift[idx] = Σ_row X_rᵀ · contract_third_full(T3_row, dq, dg) · X_r`,
            // a 2×2-weighted pullback through the STATIC primary designs (no
            // h/w blocks in the rigid accumulator: `dense_correction = None`).
            // The per-row `add_pullback_rigid_2x2` form issues, for every one of
            // the 326k+ rows × `n_dirs` directions, two `syr_row_into` rank-1
            // SYRs plus a rank-1 `row_outer_into_view` straight into the dense
            // p×p blocks — O(n·n_dirs·p²) of memory-bandwidth-bound BLAS-1 that
            // never reaches a BLAS-3 kernel and dominates the large-scale fit.
            //
            // Mirror the FLEX `dense_contiguous_rows` path: accumulate the 2×2
            // contraction weights `(w_mm, w_mg, w_gg)` per chunk row, then close
            // each chunk with ONE pair of `Xᵀ diag(w) X` / `Xᵀ diag(w) G` BLAS-3
            // Gram products (`add_weighted_design_grams_from_chunks` →
            // `fast_xt_diag_x` / `fast_xt_diag_y`). Identical arithmetic
            // (`w_mm = t[0][0]`, `w_mg = t[0][1]`, `w_gg = t[1][1]`, the same
            // entries `add_pullback_rigid_2x2` writes), just batched and
            // vectorized — bit-for-bit the same Hessian drift, computed in `k`
            // GEMMs per chunk instead of `n·k` rank-1 updates. `dense_contiguous_rows`
            // guarantees `weight ≡ 1.0` and `wr.index == row`, so the chunk
            // borrows the contiguous design rows directly.
            let marginal_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.marginal.clone());
            let logslope_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.logslope.clone());
            let (chunk_rows, _gpu_sized_chunks) =
                Self::batched_directional_derivative_chunk_rows(n, d_beta_flats.len());
            let chunks = (0..n)
                .step_by(chunk_rows)
                .map(|start| (start, (start + chunk_rows).min(n)))
                .collect::<Vec<_>>();
            log::info!(
                "[BMS batched dH chunks] mode=rigid-blas3 rows_per_chunk={} chunks={}",
                chunk_rows,
                chunks.len(),
            );
            let chunk_body =
                |(start, end): (usize, usize)| -> Result<Vec<BernoulliBlockHessianAccumulator>, String> {
                    let n_dirs = d_beta_flats.len();
                    let len = end - start;
                    let mut accs = make_accs();
                    let mut w_mm = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_mg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_gg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    // Zero-copy borrow of the contiguous chunk rows (mirrors the
                    // FLEX chunked path); falls back to a chunk copy only for a
                    // non-dense design representation.
                    let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.marginal_design.as_dense_ref() {
                            Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                            None => self
                                .marginal_design
                                .try_row_chunk(start..end)
                                .map_err(|e| format!("bernoulli marginal_design try_row_chunk: {e}"))?
                                .into(),
                        };
                    let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.logslope_design.as_dense_ref() {
                            Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                            None => self
                                .logslope_design
                                .try_row_chunk(start..end)
                                .map_err(|e| format!("bernoulli logslope_design try_row_chunk: {e}"))?
                                .into(),
                        };
                    // Per-direction projected primary perturbations `(dq, dg)` for
                    // every chunk row: `dq = X_chunk · marginal_dir`,
                    // `dg = G_chunk · logslope_dir` (BLAS-3 GEMM, all directions
                    // at once), matching the per-row `dot_row_view` the scalar
                    // path used.
                    let marginal_projected = crate::faer_ndarray::fast_ab(&x_chunk, &marginal_dirs);
                    let logslope_projected = crate::faer_ndarray::fast_ab(&g_chunk, &logslope_dirs);
                    for row in start..end {
                        let local = row - start;
                        let full = self.rigid_third_full_cached(block_states, cache, row)?;
                        for idx in 0..n_dirs {
                            let dq = marginal_projected[[local, idx]];
                            let dg = logslope_projected[[local, idx]];
                            let t = contract_third_full(full, dq, dg);
                            w_mm[idx][local] = t[0][0];
                            w_mg[idx][local] = t[0][1];
                            w_gg[idx][local] = t[1][1];
                        }
                        bump_progress(&progress);
                    }
                    for idx in 0..n_dirs {
                        accs[idx].add_weighted_design_grams_from_chunks(
                            &x_chunk,
                            &g_chunk,
                            &w_mm[idx],
                            &w_mg[idx],
                            &w_gg[idx],
                        );
                    }
                    Ok(accs)
                };
            if run_rows_serial {
                let mut accs = make_accs();
                for chunk in chunks {
                    let partial = chunk_body(chunk)?;
                    for (l, r) in accs.iter_mut().zip(partial.iter()) {
                        l.add(r);
                    }
                }
                accs
            } else {
                chunks
                    .into_par_iter()
                    // Pin faer's per-chunk GEMM parallelism to `Par::Seq` so the
                    // chunk fan-out (this `into_par_iter`) owns the global Rayon
                    // pool and the inner `fast_ab` / weighted-Gram GEMMs do not
                    // re-fan it and oversubscribe — same discipline as the FLEX
                    // chunked path.
                    .map(|chunk| crate::faer_ndarray::with_nested_parallel(|| chunk_body(chunk)))
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        } else if !flex_active {
            // Non-contiguous rigid rows (e.g. an outer-score subsample mask): the
            // chunked zero-copy Gram borrow above assumes contiguous unit-weight
            // rows, so fall back to the per-row pullback for the sampled subset.
            if run_rows_serial {
                let mut accs = make_accs();
                for wr in weighted_rows.iter() {
                    let row = wr.index;
                    let w = wr.weight;
                    // The per-row uncontracted third tensor is
                    // direction-independent; build it once per row (via the
                    // serially-prewarmed cache, populated above) and fold each
                    // direction in with the cheap `contract_third_full` bilinear
                    // instead of recomputing the heavy empirical/closed-form jet
                    // once per (row, direction) pair.
                    let full = self.rigid_third_full_cached(block_states, cache, row)?;
                    for (idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
                        let dq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                        let dg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                        let t = contract_third_full(full, dq, dg);
                        accs[idx].add_pullback_rigid_2x2(self, row, &t, w);
                    }
                    bump_progress(&progress);
                }
                accs
            } else {
                weighted_rows
                    .par_iter()
                    .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        // Direction-independent per-row third tensor: read the
                        // serially-prewarmed cache (built once before this
                        // par_iter, so no nested lazy build / nested par_iter
                        // races inside a Rayon worker) and contract each
                        // direction cheaply, instead of rebuilding the heavy jet
                        // `n_dirs` times per row.
                        let full = self.rigid_third_full_cached(block_states, cache, row)?;
                        for (idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = contract_third_full(full, dq, dg);
                            accs[idx].add_pullback_rigid_2x2(self, row, &t, w);
                        }
                        bump_progress(&progress);
                        Ok(accs)
                    })
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        } else if dense_contiguous_rows {
            let marginal_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.marginal.clone());
            let logslope_dirs =
                Self::stacked_direction_block(d_beta_flats, slices.logslope.clone());
            let (chunk_rows, gpu_sized_chunks) =
                Self::batched_directional_derivative_chunk_rows(n, d_beta_flats.len());
            let chunks = (0..n)
                .step_by(chunk_rows)
                .map(|start| (start, (start + chunk_rows).min(n)))
                .collect::<Vec<_>>();
            log::info!(
                "[BMS batched dH chunks] rows_per_chunk={} chunks={} gpu_sized={}",
                chunk_rows,
                chunks.len(),
                gpu_sized_chunks,
            );
            let chunk_body =
                |(start, end): (usize, usize)| -> Result<Vec<BernoulliBlockHessianAccumulator>, String> {
                    let n_dirs = d_beta_flats.len();
                    let len = end - start;
                    let mut accs = make_accs();
                    let mut w_mm = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_mg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    let mut w_gg = (0..n_dirs)
                        .map(|_| Array1::<f64>::zeros(len))
                        .collect::<Vec<_>>();
                    // Zero-copy fast path: borrow the chunk rows from the stored
                    // dense matrix as `ArrayView2` (wrapped in `CowArray`) when
                    // materialised, instead of `.to_owned()`-copying a fresh
                    // `Array2` per chunk per cycle. `fast_ab` and
                    // `add_weighted_design_grams_from_chunks` are generic over
                    // `Data<Elem = f64>`, so the view drives the identical BLAS-3
                    // kernels with identical arithmetic — exact, no copy.
                    let x_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.marginal_design.as_dense_ref() {
                            Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                            None => self
                                .marginal_design
                                .try_row_chunk(start..end)
                                .map_err(|e| {
                                    format!("bernoulli marginal_design try_row_chunk: {e}")
                                })?
                                .into(),
                        };
                    let g_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                        match self.logslope_design.as_dense_ref() {
                            Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                            None => self
                                .logslope_design
                                .try_row_chunk(start..end)
                                .map_err(|e| {
                                    format!("bernoulli logslope_design try_row_chunk: {e}")
                                })?
                                .into(),
                        };
                    let marginal_projected =
                        crate::faer_ndarray::fast_ab(&x_chunk, &marginal_dirs);
                    let logslope_projected =
                        crate::faer_ndarray::fast_ab(&g_chunk, &logslope_dirs);

                    for row in start..end {
                        let local = row - start;
                        let row_ctx = Self::row_ctx(cache, row);
                        let row_dirs = Self::row_primary_directions_from_projected(
                            local,
                            slices,
                            primary,
                            d_beta_flats,
                            &marginal_projected,
                            &logslope_projected,
                        );
                        let thirds = self.row_primary_third_contracted_many_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dirs,
                        )?;
                        for (idx, third) in thirds.iter().enumerate() {
                            w_mm[idx][local] = third[[0, 0]];
                            w_mg[idx][local] = third[[0, 1]];
                            w_gg[idx][local] = third[[1, 1]];
                            accs[idx].add_hw_pullback_only(self, row, slices, primary, third);
                        }
                        bump_progress(&progress);
                    }

                    for idx in 0..n_dirs {
                        accs[idx].add_weighted_design_grams_from_chunks(
                            &x_chunk,
                            &g_chunk,
                            &w_mm[idx],
                            &w_mg[idx],
                            &w_gg[idx],
                        );
                    }
                    Ok(accs)
                };
            if run_rows_serial || gpu_sized_chunks {
                let mut accs = make_accs();
                for chunk in chunks {
                    let partial = chunk_body(chunk)?;
                    for (l, r) in accs.iter_mut().zip(partial.iter()) {
                        l.add(r);
                    }
                }
                accs
            } else {
                chunks
                    .into_par_iter()
                    // Each chunk runs on a Rayon worker and issues `fast_ab` /
                    // weighted-Gram GEMMs; pin their faer parallelism to
                    // `Par::Seq` so they do not re-fan the global Rayon pool
                    // against this already-parallel chunk fan-out. The serial
                    // path above intentionally keeps top-level pool parallelism.
                    .map(|chunk| crate::faer_ndarray::with_nested_parallel(|| chunk_body(chunk)))
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        } else {
            let row_body = |wr: WeightedOuterRow,
                            accs: &mut Vec<BernoulliBlockHessianAccumulator>|
             -> Result<(), String> {
                let row = wr.index;
                let w = wr.weight;
                let row_ctx = Self::row_ctx(cache, row);
                let row_dirs = d_beta_flats
                    .iter()
                    .map(|d_beta_flat| {
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_flat)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mut thirds = self.row_primary_third_contracted_many_with_moments(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_dirs,
                )?;
                for (idx, third) in thirds.iter_mut().enumerate() {
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }
                    accs[idx].add_pullback(self, row, slices, primary, third);
                }
                bump_progress(&progress);
                Ok(())
            };
            if run_rows_serial {
                let mut accs = make_accs();
                for wr in weighted_rows.iter() {
                    row_body(*wr, &mut accs)?;
                }
                accs
            } else {
                weighted_rows
                    .par_iter()
                    .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                        row_body(*wr, &mut accs)?;
                        Ok(accs)
                    })
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        };

        let elapsed = started.elapsed().as_secs_f64();
        log::info!(
            "[BMS batched dH] n={} rows={} p={} dirs={} elapsed={:.3}s",
            n,
            n_rows,
            slices.total,
            n_dirs,
            elapsed
        );
        let operators = accs
            .drain(..)
            .map(|acc| Some(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>))
            .collect();
        drop(process_monitor_guard);
        Ok(operators)
    }

    pub(super) fn exact_newton_joint_hessiansecond_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of
    /// `exact_newton_joint_hessiansecond_directional_derivative_from_cache`.
    /// When `options.outer_score_subsample` is `Some`, only the masked rows
    /// are visited and the accumulated dense second-directional Hessian
    /// matrix uses per-row Horvitz-Thompson inverse-inclusion weights.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);

        // Eager-prime the per-row uncontracted fourth-derivative cache *before*
        // entering the per-row `par_iter` so the cache's nested-`par_iter`
        // build does not race with Rayon workers already inside the outer
        // loop — see `feedback_oncelock_rayon_deadlock` and the mirror
        // pre-warm for the third-derivative tensor at the top of
        // `compute_gradient_and_hessian_via_psi_axes`. Skipped on the FLEX
        // path because that branch routes through the flex jet machinery
        // instead of `rigid_fourth_full_cached`.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_hessiansecond_directional_derivative_from_cache rigid fourth-cache warm-up",
            )?;
        }
        // FLEX analogue: prewarm the degree-21 cell-moment bundle so the per-row
        // fourth-order recompute reuses prebuilt moments rather than recomputing
        // them per row on every operator application. The fourth-order recompute
        // reads degree-21 cells (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;

        // ── Rigid closed-form: 4th-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .par_iter()
                .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let uq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                    let ug = self
                        .logslope_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                    let vq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                    let vg = self
                        .logslope_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                    let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                    let f = contract_fourth_full(t, uq, ug, vq, vg);
                    let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                    if w != 1.0 {
                        f_arr.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &f_arr);
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = weighted_rows
            .par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_u =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                let row_v =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                let row_ctx = Self::row_ctx(cache, row);
                let mut fourth = self.row_primary_fourth_contracted_recompute(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_u,
                    &row_v,
                )?;
                if w != 1.0 {
                    fourth.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, slices, primary, &fourth);
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    /// Outer-aware operator builder for the joint-Hessian second directional
    /// derivative. The default-options shim is omitted because the
    /// `BernoulliMarginalSlopeExactNewtonJointHessianWorkspace` always threads
    /// its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the operator.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);

        // Eager-prime the per-row uncontracted fourth-derivative cache *before*
        // entering the per-row `par_iter` to avoid the lazy-cache-under-rayon
        // deadlock pattern — see `feedback_oncelock_rayon_deadlock`.
        if !self.effective_flex_active(block_states)? {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache rigid fourth-cache warm-up",
            )?;
        }
        // FLEX analogue: prewarm the degree-21 cell-moment bundle so the per-row
        // fourth-order recompute reuses prebuilt moments rather than recomputing
        // them per row on every operator application. The fourth-order recompute
        // reads degree-21 cells (gam#683).
        self.prewarm_flex_cell_bundle(block_states, cache, 21)?;

        if !self.effective_flex_active(block_states)? {
            let block_acc = weighted_rows
                .par_iter()
                .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let uq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                    let ug = self
                        .logslope_design
                        .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                    let vq = self
                        .marginal_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                    let vg = self
                        .logslope_design
                        .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                    let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                    let f = contract_fourth_full(t, uq, ug, vq, vg);
                    let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                    if w != 1.0 {
                        f_arr.mapv_inplace(|v| v * w);
                    }
                    acc.add_pullback(self, row, slices, primary, &f_arr);
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        let block_acc = weighted_rows
            .par_iter()
            .try_fold(make_acc, |mut acc, wr| -> Result<_, String> {
                let row = wr.index;
                let w = wr.weight;
                let row_u =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                let row_v =
                    self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                let row_ctx = Self::row_ctx(cache, row);
                let mut fourth = self.row_primary_fourth_contracted_recompute(
                    row,
                    block_states,
                    cache,
                    row_ctx,
                    &row_u,
                    &row_v,
                )?;
                if w != 1.0 {
                    fourth.mapv_inplace(|v| v * w);
                }
                acc.add_pullback(self, row, slices, primary, &fourth);
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_operators_from_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_pairs: &[(Array1<f64>, Array1<f64>)],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        if d_beta_pairs.is_empty() {
            return Ok(Vec::new());
        }
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let mut unique_dirs = Vec::<Array1<f64>>::new();
        let mut pair_indices = Vec::<(usize, usize)>::with_capacity(d_beta_pairs.len());
        for (u, v) in d_beta_pairs {
            let u_idx = Self::find_or_push_unique_direction(&mut unique_dirs, u);
            let v_idx = Self::find_or_push_unique_direction(&mut unique_dirs, v);
            pair_indices.push((u_idx, v_idx));
        }
        let make_accs = || {
            (0..d_beta_pairs.len())
                .map(|_| BernoulliBlockHessianAccumulator::new(slices))
                .collect::<Vec<_>>()
        };

        let started = std::time::Instant::now();
        let n_rows = weighted_rows.len();
        let n_pairs = d_beta_pairs.len();
        let n_unique_dirs = unique_dirs.len();
        let flex_active = self.effective_flex_active(block_states)?;
        let bundle_present = cache.row_cell_moments.is_some();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS batched d2H n={n} rows={n_rows} p={} pairs={n_pairs} unique_dirs={n_unique_dirs} flex={flex_active} cell_moments_bundle={bundle_present}",
            slices.total
        ));
        log::info!(
            "[BMS batched d2H start] n={} rows={} p={} pairs={} unique_dirs={} flex={} cell_moments_bundle={}",
            n,
            n_rows,
            slices.total,
            n_pairs,
            n_unique_dirs,
            flex_active,
            bundle_present,
        );
        let progress = Arc::new(AtomicUsize::new(0));
        let progress_step = (n_rows / 8).max(1);
        let bump_progress = |progress: &AtomicUsize| {
            let now = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if now == n_rows || now.is_multiple_of(progress_step) {
                log::info!(
                    "[BMS batched d2H progress] rows={}/{} pairs={} unique_dirs={} elapsed={:.3}s",
                    now,
                    n_rows,
                    n_pairs,
                    n_unique_dirs,
                    started.elapsed().as_secs_f64(),
                );
            }
        };

        if !flex_active && n > 0 {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "exact_newton_joint_hessiansecond_directional_derivative_operators_from_cache rigid fourth-cache warm-up",
            )?;
        }
        // FLEX analogue: prewarm the degree-21 cell-moment bundle once, serially,
        // so the per-row fourth-order recompute reuses prebuilt moments instead
        // of recomputing them per row across all direction pairs. The
        // fourth-order recompute reads degree-21 cells (gam#683).
        if flex_active && n > 0 {
            self.prewarm_flex_cell_bundle(block_states, cache, 21)?;
        }
        const ROW_PAR_MIN_ROWS: usize = 4_096;
        let run_rows_serial = rayon::current_thread_index().is_some()
            || rayon::current_num_threads() <= 1
            || n_rows < ROW_PAR_MIN_ROWS;

        let accs = if !flex_active {
            if run_rows_serial {
                let mut accs = make_accs();
                for wr in weighted_rows.iter() {
                    let row = wr.index;
                    let w = wr.weight;
                    let projections = unique_dirs
                        .iter()
                        .map(|direction| {
                            let q = self
                                .marginal_design
                                .dot_row_view(row, direction.slice(s![slices.marginal.clone()]));
                            let g = self
                                .logslope_design
                                .dot_row_view(row, direction.slice(s![slices.logslope.clone()]));
                            (q, g)
                        })
                        .collect::<Vec<_>>();
                    let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                    for (idx, (u_idx, v_idx)) in pair_indices.iter().copied().enumerate() {
                        let (uq, ug) = projections[u_idx];
                        let (vq, vg) = projections[v_idx];
                        let f = contract_fourth_full(t, uq, ug, vq, vg);
                        let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                        if w != 1.0 {
                            f_arr.mapv_inplace(|value| value * w);
                        }
                        accs[idx].add_pullback(self, row, slices, primary, &f_arr);
                    }
                    bump_progress(&progress);
                }
                accs
            } else {
                weighted_rows
                    .par_iter()
                    .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                        let row = wr.index;
                        let w = wr.weight;
                        let projections = unique_dirs
                            .iter()
                            .map(|direction| {
                                let q = self.marginal_design.dot_row_view(
                                    row,
                                    direction.slice(s![slices.marginal.clone()]),
                                );
                                let g = self.logslope_design.dot_row_view(
                                    row,
                                    direction.slice(s![slices.logslope.clone()]),
                                );
                                (q, g)
                            })
                            .collect::<Vec<_>>();
                        let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
                        for (idx, (u_idx, v_idx)) in pair_indices.iter().copied().enumerate() {
                            let (uq, ug) = projections[u_idx];
                            let (vq, vg) = projections[v_idx];
                            let f = contract_fourth_full(t, uq, ug, vq, vg);
                            let mut f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                            if w != 1.0 {
                                f_arr.mapv_inplace(|value| value * w);
                            }
                            accs[idx].add_pullback(self, row, slices, primary, &f_arr);
                        }
                        bump_progress(&progress);
                        Ok(accs)
                    })
                    .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            l.add(r);
                        }
                        Ok(left)
                    })?
            }
        } else if run_rows_serial {
            let mut accs = make_accs();
            for wr in weighted_rows.iter() {
                let row = wr.index;
                let w = wr.weight;
                let row_dirs = unique_dirs
                    .iter()
                    .map(|direction| {
                        self.row_primary_direction_from_flat(row, slices, primary, direction)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let row_ctx = Self::row_ctx(cache, row);
                for (idx, (u_idx, v_idx)) in pair_indices.iter().copied().enumerate() {
                    let mut fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_dirs[u_idx],
                        &row_dirs[v_idx],
                    )?;
                    if w != 1.0 {
                        fourth.mapv_inplace(|value| value * w);
                    }
                    accs[idx].add_pullback(self, row, slices, primary, &fourth);
                }
                bump_progress(&progress);
            }
            accs
        } else {
            weighted_rows
                .par_iter()
                .try_fold(make_accs, |mut accs, wr| -> Result<_, String> {
                    let row = wr.index;
                    let w = wr.weight;
                    let row_dirs = unique_dirs
                        .iter()
                        .map(|direction| {
                            self.row_primary_direction_from_flat(row, slices, primary, direction)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let row_ctx = Self::row_ctx(cache, row);
                    for (idx, (u_idx, v_idx)) in pair_indices.iter().copied().enumerate() {
                        let mut fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dirs[u_idx],
                            &row_dirs[v_idx],
                        )?;
                        if w != 1.0 {
                            fourth.mapv_inplace(|value| value * w);
                        }
                        accs[idx].add_pullback(self, row, slices, primary, &fourth);
                    }
                    bump_progress(&progress);
                    Ok(accs)
                })
                .try_reduce(make_accs, |mut left, right| -> Result<_, String> {
                    for (l, r) in left.iter_mut().zip(right.iter()) {
                        l.add(r);
                    }
                    Ok(left)
                })?
        };
        log::info!(
            "[BMS batched d2H done] n={} rows={} p={} pairs={} unique_dirs={} elapsed={:.3}s",
            n,
            n_rows,
            slices.total,
            n_pairs,
            n_unique_dirs,
            started.elapsed().as_secs_f64(),
        );
        drop(process_monitor_guard);
        Ok(accs
            .into_iter()
            .map(|acc| Some(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>))
            .collect())
    }

    pub(crate) fn find_or_push_unique_direction(
        unique_dirs: &mut Vec<Array1<f64>>,
        candidate: &Array1<f64>,
    ) -> usize {
        if let Some(idx) = unique_dirs.iter().position(|existing| {
            existing.len() == candidate.len()
                && existing
                    .iter()
                    .zip(candidate.iter())
                    .all(|(left, right)| left == right)
        }) {
            return idx;
        }
        let idx = unique_dirs.len();
        unique_dirs.push(candidate.clone());
        idx
    }

    pub(super) fn evaluate_flex_block_diagonals_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<FamilyEvaluation, String> {
        let primary = cache.primary.clone();
        let n = self.y.len();
        let row_chunk = bms_row_chunk_size(n);
        let n_chunks = n.div_ceil(row_chunk);
        // Pool of per-worker workspaces reused across chunks within this
        // evaluate. The previous implementation seeded a fresh accumulator
        // per try_fold chunk, paying p_marginal² + p_logslope² (+ optional
        // p_h², p_w²) dense Hessian allocations per chunk. The pool caps
        // total allocations at the number of distinct rayon workers that
        // ever grab a chunk; each chunk reuses the worker's existing
        // dense buffers via in-place += accumulation. Keep the row scratch in
        // the same pool: it owns primary_dim² arrays and is also chunk-local.
        let pool: Mutex<
            Vec<(
                BernoulliExactNewtonAccumulator,
                BernoulliMarginalSlopeFlexRowScratch,
            )>,
        > = Mutex::new(Vec::new());
        let result: Result<(), String> =
            (0..n_chunks)
                .into_par_iter()
                .try_for_each(|chunk_idx| -> Result<(), String> {
                    let (mut acc, mut scratch) = pool
                        .lock()
                        .expect("bernoulli exact newton accumulator pool poisoned")
                        .pop()
                        .unwrap_or_else(|| {
                            (
                                BernoulliExactNewtonAccumulator::new(slices),
                                BernoulliMarginalSlopeFlexRowScratch::new(primary.total),
                            )
                        });
                    let start = chunk_idx * row_chunk;
                    let end = (start + row_chunk).min(n);
                    let chunk_res: Result<(), String> = (|| {
                        for row in start..end {
                            let row_ctx = Self::row_ctx(cache, row);
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            let row_neglog = self.compute_row_analytic_flex_into_with_moments(
                                row,
                                block_states,
                                &primary,
                                row_ctx,
                                row_moments,
                                cache.cell_family_forest.as_ref(),
                                true,
                                &mut scratch,
                            )?;
                            acc.add_pullback_block_diagonals(
                                self, row, &primary, row_neglog, &scratch,
                            )?;
                        }
                        Ok(())
                    })();
                    pool.lock()
                        .expect("bernoulli exact newton accumulator pool poisoned")
                        .push((acc, scratch));
                    chunk_res
                });
        result?;
        let mut pooled = pool
            .into_inner()
            .expect("bernoulli exact newton accumulator pool poisoned");
        let reduced = match pooled.pop() {
            Some((mut first, _)) => {
                for (other, _) in &pooled {
                    first.add(other);
                }
                first
            }
            None => BernoulliExactNewtonAccumulator::new(slices),
        };

        let BernoulliExactNewtonAccumulator {
            ll,
            grad_marginal,
            grad_logslope,
            hess_marginal,
            hess_logslope,
            grad_h,
            grad_w,
            hess_h,
            hess_w,
        } = reduced;

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_marginal,
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_logslope,
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (grad_h, hess_h) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (grad_w, hess_w) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    pub(super) fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(self);
        let flex_active = self.effective_flex_active(block_states)?;

        // ── Block-diagonal direct path (rigid, p < 512) ─────────────────
        //
        // The RowKernel<2> is the single source of truth in objective space
        // (negative log-likelihood). The full joint Hessian's off-diagonal
        // marginal/logslope cross block is unused by the per-block working
        // sets the inner solver consumes, so we accumulate only the two
        // diagonal blocks via the family's sparse-aware syr / axpy.  This
        // avoids the Θ(n·(p_m+p_g)²) joint assembly + immediate slice that
        // the previous implementation paid.
        if !flex_active && slices.total < 512 {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern, &crate::families::row_kernel::RowSet::All)?;
            let ll = row_kernel_log_likelihood(&cache, &crate::families::row_kernel::RowSet::All);
            let joint_gradient = Self::exact_newton_score_from_objective_gradient(
                row_kernel_gradient(&kern, &cache, &crate::families::row_kernel::RowSet::All),
            );

            let n = cache.n;
            let p_marginal = slices.marginal.len();
            let p_logslope = slices.logslope.len();
            let make_pair = || {
                (
                    Array2::<f64>::zeros((p_marginal, p_marginal)),
                    Array2::<f64>::zeros((p_logslope, p_logslope)),
                )
            };
            let row_chunk = bms_row_chunk_size(n);
            let (hess_marginal, hess_logslope) = (0..n.div_ceil(row_chunk))
                .into_par_iter()
                .try_fold(
                    make_pair,
                    |(mut hm, mut hl), chunk_idx| -> Result<(Array2<f64>, Array2<f64>), String> {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        let rows = end - start;
                        // Zero-copy fast path: borrow the chunk rows from the
                        // stored dense matrix as `ArrayView2` (wrapped in
                        // `CowArray`) when materialised, avoiding the per-chunk
                        // `.to_owned()` copy on every pre-warm cycle.
                        // `add_weighted_chunk_gram` is generic over
                        // `Data<Elem = f64>`, so the view drives the identical
                        // Gram kernel with identical arithmetic.
                        let marginal_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.marginal_design.as_dense_ref() {
                                Some(x_full) => x_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .marginal_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli marginal_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let logslope_chunk: ndarray::CowArray<'_, f64, ndarray::Ix2> =
                            match self.logslope_design.as_dense_ref() {
                                Some(g_full) => g_full.slice(s![start..end, ..]).into(),
                                None => self
                                    .logslope_design
                                    .try_row_chunk(start..end)
                                    .map_err(|e| {
                                        format!("bernoulli logslope_design try_row_chunk: {e}")
                                    })?
                                    .into(),
                            };
                        let mut hm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let mut hl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let hm_w = &mut hm_w_buf[..rows];
                        let hl_w = &mut hl_w_buf[..rows];
                        for local_row in 0..rows {
                            let h = &cache.hessians[start + local_row];
                            hm_w[local_row] = h[0][0];
                            hl_w[local_row] = h[1][1];
                        }
                        add_weighted_chunk_gram(&marginal_chunk, hm_w, &mut hm);
                        add_weighted_chunk_gram(&logslope_chunk, hl_w, &mut hl);
                        Ok((hm, hl))
                    },
                )
                .try_reduce(
                    make_pair,
                    |(mut lhm, mut lhl),
                     (rhm, rhl)|
                     -> Result<(Array2<f64>, Array2<f64>), String> {
                        lhm += &rhm;
                        lhl += &rhl;
                        Ok((lhm, lhl))
                    },
                )?;

            let hess_marginal =
                Self::exact_newton_observed_information_from_objective_hessian(hess_marginal);
            let hess_logslope =
                Self::exact_newton_observed_information_from_objective_hessian(hess_logslope);

            let grad_marginal = joint_gradient.slice(s![slices.marginal.clone()]).to_owned();
            let grad_logslope = joint_gradient.slice(s![slices.logslope.clone()]).to_owned();

            let mut sets = vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Dense(hess_marginal),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Dense(hess_logslope),
                },
            ];
            if let Some(range) = slices.h.as_ref() {
                // Rigid mode does not exercise h/w; mirror the blockwise
                // fallback by exposing zero working sets.
                sets.push(BlockWorkingSet::ExactNewton {
                    gradient: Array1::zeros(range.len()),
                    hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                });
            }
            if let Some(range) = slices.w.as_ref() {
                sets.push(BlockWorkingSet::ExactNewton {
                    gradient: Array1::zeros(range.len()),
                    hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                });
            }
            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: sets,
            });
        }

        // ── Flex block-diagonal path ─────────────────────────────────
        // Flexible rows are independent once the intercept cache is built, so
        // `evaluate_flex_block_diagonals_from_cache` accumulates each row into
        // Rayon thread-local block buffers and reduces the buffers once. The
        // off-diagonal joint blocks are not consumed by the inner block solver.
        if flex_active {
            let cache = self.build_exact_eval_cache_with_order(block_states)?;
            return self.evaluate_flex_block_diagonals_from_cache(block_states, &slices, &cache);
        }

        // ── Blockwise fallback (p >= 512) ───────────────────────────────
        //
        // The joint dense Hessian is too large to materialise.  Block
        // Hessians are assembled independently via the same per-row
        // kernel, so the algebra is correct but not structurally guaranteed
        // identical to the joint object.  This path should only be reached
        // for very large models where memory is the binding constraint.
        let n = self.y.len();
        let p_marginal = slices.marginal.len();
        let p_logslope = slices.logslope.len();
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(p_marginal),
                Array1::<f64>::zeros(p_logslope),
                Array2::<f64>::zeros((p_marginal, p_marginal)),
                Array2::<f64>::zeros((p_logslope, p_logslope)),
            )
        };
        let row_chunk = bms_row_chunk_size(n);
        let (ll, grad_marginal, grad_logslope, hess_marginal, hess_logslope) = (0..n
            .div_ceil(row_chunk))
            .into_par_iter()
            .try_fold(
                make_acc,
                |(mut ll, mut gm, mut gl, mut hm, mut hl), chunk_idx| -> Result<_, String> {
                    // Per-cycle exact-Newton block-Hessian assembly: this chunk
                    // runs on a Rayon worker and issues `fast_xt_diag_x` /
                    // `fast_atv` GEMMs (via `add_weighted_chunk_gram/gradient`).
                    // Pin their faer parallelism to `Par::Seq` so they do not
                    // re-fan the global Rayon pool against this already-parallel
                    // chunk fold — the rayon×BLAS oversubscription behind the
                    // intermittent `hessian_qp` stalls. Bit-identical: faer
                    // partitions the matmul output, never the contracted axis.
                    crate::faer_ndarray::with_nested_parallel(|| {
                        let start = chunk_idx * row_chunk;
                        let end = (start + row_chunk).min(n);
                        let rows = end - start;
                        // Zero-copy chunk binding: a materialised dense design lets us
                        // borrow the chunk rows straight out of the stored matrix
                        // (`Owned(None)` arm holds the owned fallback for sparse /
                        // operator-backed designs). `try_row_chunk` would `.to_owned()`
                        // a fresh `(rows × p)` `Array2` every chunk every inner cycle —
                        // the `OwnedRepr<f64>` alloc/`drop_in_place` churn the cold
                        // marginal-slope fit pays in its repeated ρ-homotopy / inner
                        // Newton passes. The downstream `fast_atv` / `fast_xt_diag_x`
                        // kernels are generic over the storage, so a borrowed view runs
                        // identical BLAS-3 arithmetic — exact, just without the copy.
                        let marginal_owned = match self.marginal_design.as_dense_ref() {
                            Some(_) => None,
                            None => Some(self.marginal_design.try_row_chunk(start..end).map_err(
                                |e| format!("bernoulli marginal_design try_row_chunk: {e}"),
                            )?),
                        };
                        let logslope_owned = match self.logslope_design.as_dense_ref() {
                            Some(_) => None,
                            None => Some(self.logslope_design.try_row_chunk(start..end).map_err(
                                |e| format!("bernoulli logslope_design try_row_chunk: {e}"),
                            )?),
                        };
                        let mut gm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let mut gl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let mut hm_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let mut hl_w_buf = [0.0f64; ROW_CHUNK_SIZE];
                        let gm_w = &mut gm_w_buf[..rows];
                        let gl_w = &mut gl_w_buf[..rows];
                        let hm_w = &mut hm_w_buf[..rows];
                        let hl_w = &mut hl_w_buf[..rows];
                        for local_row in 0..rows {
                            let row = start + local_row;
                            let marginal_eta = block_states[0].eta[row];
                            let marginal = self.marginal_link_map(marginal_eta)?;
                            let g = block_states[1].eta[row];
                            let (neglog, grad, h) = self.rigid_row_kernel_eval(row, marginal, g)?;
                            ll -= neglog;
                            gm_w[local_row] =
                                Self::exact_newton_score_component_from_objective_gradient(grad[0]);
                            gl_w[local_row] =
                                Self::exact_newton_score_component_from_objective_gradient(grad[1]);
                            hm_w[local_row] = h[0][0];
                            hl_w[local_row] = h[1][1];
                        }
                        match (self.marginal_design.as_dense_ref(), &marginal_owned) {
                            (Some(dense), _) => {
                                let view = dense.slice(s![start..end, ..]);
                                add_weighted_chunk_gradient(&view, gm_w, &mut gm);
                                add_weighted_chunk_gram(&view, hm_w, &mut hm);
                            }
                            (None, Some(owned)) => {
                                add_weighted_chunk_gradient(owned, gm_w, &mut gm);
                                add_weighted_chunk_gram(owned, hm_w, &mut hm);
                            }
                            (None, None) => {
                                return Err(
                                    "bernoulli marginal chunk: owned fallback missing for \
                                 non-dense design"
                                        .to_string(),
                                );
                            }
                        }
                        match (self.logslope_design.as_dense_ref(), &logslope_owned) {
                            (Some(dense), _) => {
                                let view = dense.slice(s![start..end, ..]);
                                add_weighted_chunk_gradient(&view, gl_w, &mut gl);
                                add_weighted_chunk_gram(&view, hl_w, &mut hl);
                            }
                            (None, Some(owned)) => {
                                add_weighted_chunk_gradient(owned, gl_w, &mut gl);
                                add_weighted_chunk_gram(owned, hl_w, &mut hl);
                            }
                            (None, None) => {
                                return Err(
                                    "bernoulli logslope chunk: owned fallback missing for \
                                 non-dense design"
                                        .to_string(),
                                );
                            }
                        }
                        Ok((ll, gm, gl, hm, hl))
                    })
                },
            )
            .try_reduce(
                make_acc,
                |(lll, mut lgm, mut lgl, mut lhm, mut lhl),
                 (rll, rgm, rgl, rhm, rhl)|
                 -> Result<_, String> {
                    lgm += &rgm;
                    lgl += &rgl;
                    lhm += &rhm;
                    lhl += &rhl;
                    Ok((lll + rll, lgm, lgl, lhm, lhl))
                },
            )?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Dense(hess_marginal),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Dense(hess_logslope),
                    },
                ];
                if let Some(range) = slices.h.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.w.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }
}
