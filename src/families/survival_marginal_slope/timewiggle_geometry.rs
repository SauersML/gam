//! Time-wiggle dynamic-q geometry: the monotone wiggle basis geometry, the
//! first-/second-order dynamic-q row values and gradients, and the
//! marginal psi-row lift through the time-wiggle deformation.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn flex_timewiggle_active(&self) -> bool {
        self.time_wiggle_ncols > 0
    }

    pub(crate) fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.design_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        (p_total - p_w)..p_total
    }

    pub(crate) fn time_wiggle_first_order_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle basis/beta mismatch: B..B''={},{},{} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
        }))
    }

    pub(crate) fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleGeometry>, String> {
        let first = self.time_wiggle_first_order_geometry(h0, beta_w)?;
        let Some(first) = first else {
            return Ok(None);
        };
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        let basis_d4 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 4)?;
        let basis_d5 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 5)?;
        if basis_d3.ncols() != beta_w.len()
            || basis_d4.ncols() != beta_w.len()
            || basis_d5.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle high-order basis/beta mismatch: B'''..B'''''={},{},{} betaw={}",
                    basis_d3.ncols(),
                    basis_d4.ncols(),
                    basis_d5.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        let d4q_dq04 = fast_av(&basis_d4, &beta_w);
        let d5q_dq05 = fast_av(&basis_d5, &beta_w);
        Ok(Some(SurvivalTimeWiggleGeometry {
            basis: first.basis,
            basis_d1: first.basis_d1,
            basis_d2: first.basis_d2,
            basis_d3,
            basis_d4,
            dq_dq0: first.dq_dq0,
            d2q_dq02: first.d2q_dq02,
            d3q_dq03,
            d4q_dq04,
            d5q_dq05,
        }))
    }

    pub(crate) fn row_dynamic_q_values(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowValues, String> {
        let beta_time = &block_states[0].beta;
        if !self.flex_timewiggle_active() {
            return Ok(SurvivalMarginalSlopeDynamicRowValues {
                q0: self.design_entry.dot_row(row, beta_time)
                    + self.offset_entry[row]
                    + block_states[1].eta[row],
                q1: self.design_exit.dot_row(row, beta_time)
                    + self.offset_exit[row]
                    + block_states[1].eta[row],
                qd1: self.design_derivative_exit.dot_row(row, beta_time)
                    + self.derivative_offset_exit[row],
            });
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at exit"
                    .to_string()
            })?;

        Ok(SurvivalMarginalSlopeDynamicRowValues {
            q0: h0 + entry_geom.basis.row(0).dot(&beta_time_w),
            q1: h1 + exit_geom.basis.row(0).dot(&beta_time_w),
            qd1: exit_geom.dq_dq0[0] * d_raw,
        })
    }

    pub(crate) fn row_dynamic_q_gradient(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowGradient, String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        let mut out = SurvivalMarginalSlopeDynamicRowGradient {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(p_time),
            dq1_time: Array1::zeros(p_time),
            dqd1_time: Array1::zeros(p_time),
            dq0_marginal: Array1::zeros(p_marginal),
            dq1_marginal: Array1::zeros(p_marginal),
            dqd1_marginal: Array1::zeros(p_marginal),
        };

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(out);
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
            }
        }

        Ok(out)
    }

    pub(crate) fn row_dynamic_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRow, String> {
        let mut out = SurvivalMarginalSlopeDynamicRow::empty_workspace();
        self.row_dynamic_q_geometry_into(row, block_states, &mut out)?;
        Ok(out)
    }

    /// Pooled-buffer variant of [`row_dynamic_q_geometry`]. Resizes and
    /// zero-fills `out` in place, then writes the same row geometry into the
    /// caller-owned workspace. Used by hot rayon `try_fold` accumulators that
    /// thread one `SurvivalMarginalSlopeDynamicRow` per worker thread to
    /// eliminate the ~250 KB-per-call allocator traffic the fresh-allocation
    /// path would incur at large-scale `n`.
    pub(crate) fn row_dynamic_q_geometry_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        out: &mut SurvivalMarginalSlopeDynamicRow,
    ) -> Result<(), String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        out.reset(p_time, p_marginal);

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
            // Perf (#large-scale): assign the design rows directly from the chunk
            // views into the reused `out` buffers. The previous `.to_owned()`
            // round-trip allocated three fresh `Array1<f64>` per row only to
            // immediately copy them in — removing it drops O(n·p_time) heap
            // traffic with bit-identical values.
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(());
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
        // Perf (#large-scale): hold the base/marginal design rows as borrowed views
        // into the chunk storage rather than `.to_owned()` copies. Every use
        // below is either a `.dot(...)` or scalar indexing, both of which work
        // directly on `ArrayView1`, so this is bit-identical while removing the
        // per-row Array1 allocations.
        let entry_row_view = entry_chunk.row(0);
        let exit_row_view = exit_chunk.row(0);
        let deriv_row_view = deriv_chunk.row(0);
        let x_entry_base = entry_row_view.slice(s![..p_base]);
        let x_exit_base = exit_row_view.slice(s![..p_base]);
        let x_deriv_base = deriv_row_view.slice(s![..p_base]);
        let marginal_chunk = if p_marginal > 0 {
            Some(
                self.marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?,
            )
        } else {
            None
        };
        let marginal_row = marginal_chunk.as_ref().map(|chunk| chunk.row(0));

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
            for k in 0..p_base {
                out.d2q0_time_time[[j, k]] =
                    entry_geom.d2q_dq02[0] * x_entry_base[j] * x_entry_base[k];
                out.d2q1_time_time[[j, k]] =
                    exit_geom.d2q_dq02[0] * x_exit_base[j] * x_exit_base[k];
                out.d2qd1_time_time[[j, k]] =
                    exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j] * x_exit_base[k]
                        + exit_geom.d2q_dq02[0]
                            * (x_exit_base[j] * x_deriv_base[k] + x_deriv_base[j] * x_exit_base[k]);
            }
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
            for j in 0..p_base {
                let q0_tw = entry_geom.basis_d1[[0, local_idx]] * x_entry_base[j];
                let q1_tw = exit_geom.basis_d1[[0, local_idx]] * x_exit_base[j];
                out.d2q0_time_time[[j, coeff_idx]] = q0_tw;
                out.d2q0_time_time[[coeff_idx, j]] = q0_tw;
                out.d2q1_time_time[[j, coeff_idx]] = q1_tw;
                out.d2q1_time_time[[coeff_idx, j]] = q1_tw;
                let qd1_cross = exit_geom.basis_d2[[0, local_idx]] * d_raw * x_exit_base[j]
                    + exit_geom.basis_d1[[0, local_idx]] * x_deriv_base[j];
                out.d2qd1_time_time[[j, coeff_idx]] = qd1_cross;
                out.d2qd1_time_time[[coeff_idx, j]] = qd1_cross;
            }
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
                for k in 0..p_marginal {
                    out.d2q0_marginal_marginal[[j, k]] =
                        entry_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2q1_marginal_marginal[[j, k]] =
                        exit_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2qd1_marginal_marginal[[j, k]] =
                        exit_geom.d3q_dq03[0] * d_raw * marginal_row[j] * marginal_row[k];
                }
                for k in 0..p_base {
                    out.d2q0_time_marginal[[k, j]] =
                        entry_geom.d2q_dq02[0] * x_entry_base[k] * marginal_row[j];
                    out.d2q1_time_marginal[[k, j]] =
                        exit_geom.d2q_dq02[0] * x_exit_base[k] * marginal_row[j];
                    out.d2qd1_time_marginal[[k, j]] =
                        exit_geom.d3q_dq03[0] * d_raw * x_exit_base[k] * marginal_row[j]
                            + exit_geom.d2q_dq02[0] * x_deriv_base[k] * marginal_row[j];
                }
                for local_idx in 0..time_tail.len() {
                    let coeff_idx = time_tail.start + local_idx;
                    out.d2q0_time_marginal[[coeff_idx, j]] =
                        entry_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2q1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2qd1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d2[[0, local_idx]] * d_raw * marginal_row[j];
                }
            }
        }

        Ok(())
    }

    pub(crate) fn timewiggle_marginal_psi_row_lift(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_marginal: &Array1<f64>,
    ) -> Result<TimewiggleMarginalPsiRowLift, String> {
        let beta_time = &block_states[0].beta;
        let p_time = beta_time.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift marginal_design: {e}"))?;
        let marginal_row = marginal_chunk.row(0).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time.slice(s![..p_base]))
            + self.offset_entry[row]
            + base_marginal;
        let h1 =
            x_exit_base.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + base_marginal;
        let d_raw =
            x_deriv_base.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle geometry for marginal psi lift".to_string())?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle geometry for marginal psi lift".to_string())?;

        let mu = psi_row.dot(beta_marginal);
        let mut dir =
            Array1::<f64>::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        dir[q0_idx] = entry_geom.dq_dq0[0] * mu;
        dir[q1_idx] = exit_geom.dq_dq0[0] * mu;
        dir[qd1_idx] = exit_geom.d2q_dq02[0] * d_raw * mu;

        let mut u_q0_time = Array1::<f64>::zeros(p_time);
        let mut u_q1_time = Array1::<f64>::zeros(p_time);
        let mut u_qd1_time = Array1::<f64>::zeros(p_time);
        for j in 0..p_base {
            u_q0_time[j] = entry_geom.d2q_dq02[0] * mu * x_entry_base[j];
            u_q1_time[j] = exit_geom.d2q_dq02[0] * mu * x_exit_base[j];
            u_qd1_time[j] = mu
                * (exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j]
                    + exit_geom.d2q_dq02[0] * x_deriv_base[j]);
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            u_q0_time[coeff_idx] = entry_geom.basis_d1[[0, local_idx]] * mu;
            u_q1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * mu;
            u_qd1_time[coeff_idx] = exit_geom.basis_d2[[0, local_idx]] * d_raw * mu;
        }

        let u_q0_marginal =
            psi_row * entry_geom.dq_dq0[0] + &marginal_row * (entry_geom.d2q_dq02[0] * mu);
        let u_q1_marginal =
            psi_row * exit_geom.dq_dq0[0] + &marginal_row * (exit_geom.d2q_dq02[0] * mu);
        let u_qd1_marginal = psi_row * (exit_geom.d2q_dq02[0] * d_raw)
            + &marginal_row * (exit_geom.d3q_dq03[0] * d_raw * mu);

        Ok(TimewiggleMarginalPsiRowLift {
            dir,
            u_q0_time,
            u_q1_time,
            u_qd1_time,
            u_q0_marginal,
            u_q1_marginal,
            u_qd1_marginal,
            x_entry_base,
            x_exit_base,
            x_deriv_base,
            marginal_row,
            entry_basis_d1: entry_geom.basis_d1.row(0).to_owned(),
            entry_basis_d2: entry_geom.basis_d2.row(0).to_owned(),
            exit_basis_d1: exit_geom.basis_d1.row(0).to_owned(),
            exit_basis_d2: exit_geom.basis_d2.row(0).to_owned(),
            exit_basis_d3: exit_geom.basis_d3.row(0).to_owned(),
            entry_m2: entry_geom.d2q_dq02[0],
            entry_m3: entry_geom.d3q_dq03[0],
            exit_m2: exit_geom.d2q_dq02[0],
            exit_m3: exit_geom.d3q_dq03[0],
            exit_m4: exit_geom.d4q_dq04[0],
            d_raw,
            mu,
            psi_row: psi_row.clone(),
        })
    }

}

