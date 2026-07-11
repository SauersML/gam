//! Dynamic-q blockwise accumulation: assembling per-row gradient, block
//! Hessians and identity-primary cross terms into the joint blockwise
//! evaluation, including the dense-submatrix scatter helpers and the
//! blockwise exact-Newton dynamic-q driver.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn accumulate_dynamic_q_core_gradient(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    pub(crate) fn accumulate_dynamic_q_core_gradient_first_order(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    pub(crate) fn accumulate_dynamic_q_core_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Perf (#large-scale): scatter each core block-Hessian contribution
        // directly into `joint_hessian` as it is computed, instead of building
        // six fresh `Array2` per row in `dynamic_q_core_hessian_blocks` and
        // then copying them in. This removes the per-row heap traffic (6×p²
        // allocations + zero-fills + the logslope-row `to_owned`) and the
        // extra read/write pass over the temporaries. Each cell receives the
        // identical `value` it did before (the temporaries were written with
        // `=` then added here), so the accumulated result is bit-identical.
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let d2q_time_marginal = [
            &q_geom.d2q0_time_marginal,
            &q_geom.d2q1_time_marginal,
            &q_geom.d2qd1_time_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("accumulate_dynamic_q_core_hessian logslope: {e}"))?;
        let logslope_row = logslope_chunk.row(0);

        let t0 = slices.time.start;
        let m0 = slices.marginal.start;
        let g0 = slices.logslope.start;

        // time × time
        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                joint_hessian[[t0 + a, t0 + b]] += value;
            }
        }
        // marginal × marginal
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                joint_hessian[[m0 + a, m0 + b]] += value;
            }
        }
        // logslope × logslope (rank-1: h_gg · xxᵀ); zero cells skipped exactly
        // as the prior `Array2::zeros`-backed block left them zero.
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    joint_hessian[[g0 + a, g0 + b]] += row_scale * logslope_row[b];
                }
            }
        }
        // time × marginal (symmetric scatter)
        for a in 0..p_t {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_marginal[q_u][[a, b]];
                }
                joint_hessian[[t0 + a, m0 + b]] += value;
                joint_hessian[[m0 + b, t0 + a]] += value;
            }
        }
        // time × logslope (symmetric scatter)
        for a in 0..p_t {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_time[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    let value = weight * logslope_row[b];
                    joint_hessian[[t0 + a, g0 + b]] += value;
                    joint_hessian[[g0 + b, t0 + a]] += value;
                }
            }
        }
        // marginal × logslope (symmetric scatter)
        for a in 0..p_m {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_marginal[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    let value = weight * logslope_row[b];
                    joint_hessian[[m0 + a, g0 + b]] += value;
                    joint_hessian[[g0 + b, m0 + a]] += value;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn accumulate_dynamic_q_blockwise_gradient(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        grad_time: &mut Array1<f64>,
        grad_marginal: &mut Array1<f64>,
        grad_logslope: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_time[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_marginal[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut grad_logslope.view_mut(),
        )?;
        Ok(())
    }

    pub(crate) fn accumulate_dynamic_q_core_block_hessians(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        hess_time: &mut Array2<f64>,
        hess_marginal: &mut Array2<f64>,
        hess_logslope: &mut Array2<f64>,
    ) -> Result<(), String> {
        // Perf (#large-scale): accumulate the three diagonal block-Hessian
        // contributions directly into the caller's per-thread workspace
        // buffers with `+=`, rather than allocating three fresh
        // `Array2::zeros` per row (`dynamic_q_core_diagonal_hessian_blocks`)
        // and then folding them in with `*hess += &local`. This removes
        // O(n·p²) heap allocation + zero-fill + a redundant add pass. The
        // arithmetic per cell is identical: the old path wrote `value` into a
        // zeroed local then added it here, so accumulating `value` directly is
        // bit-identical. The logslope block skips zero cells exactly as before
        // (adding the implicit zeros was a no-op).
        let p_t = hess_time.nrows();
        let p_m = hess_marginal.nrows();
        let p_g = hess_logslope.nrows();

        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("accumulate_dynamic_q_core_block_hessians logslope: {e}"))?;
        let logslope_row = logslope_chunk.row(0);

        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                hess_time[[a, b]] += value;
            }
        }
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                hess_marginal[[a, b]] += value;
            }
        }
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    hess_logslope[[a, b]] += row_scale * logslope_row[b];
                }
            }
        }
        Ok(())
    }

    pub(crate) fn accumulate_dynamic_q_blockwise_row(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        acc: &mut DynamicQBlockwiseAccumulator,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_blockwise_gradient(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            &mut acc.grad_time,
            &mut acc.grad_marginal,
            &mut acc.grad_logslope,
        )?;
        self.accumulate_dynamic_q_core_block_hessians(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            &mut acc.hess_time,
            &mut acc.hess_marginal,
            &mut acc.hess_logslope,
        )?;
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.h.as_ref(),
            acc.grad_score_warp.as_mut(),
            acc.hess_score_warp.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.w.as_ref(),
            acc.grad_link_dev.as_mut(),
            acc.hess_link_dev.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        // Absorbed-influence diagonal block (#461). Unlike the identity flex
        // blocks above (whose basis IS the primary coordinate, so the block grad
        // is a slice copy), the absorber's `p₁` coefficients project from the
        // single `o_infl` primary scalar through `Z̃_infl[row,:]`:
        //   grad_i = -primary_gradient[infl] · Z̃[row,i]
        //   hess_ij += primary_hessian[[infl,infl]] · Z̃[row,i] · Z̃[row,j]
        if let (Some(infl_idx), Some(gradient), Some(hessian)) = (
            primary.infl,
            acc.grad_influence.as_mut(),
            acc.hess_influence.as_mut(),
        ) {
            let z_tilde = self.influence_absorber.as_ref().ok_or_else(|| {
                "accumulate_dynamic_q_blockwise_row: influence primary index present but no Z̃ design"
                    .to_string()
            })?;
            let z_row = z_tilde.row(row);
            let g_infl = primary_gradient[infl_idx];
            let h_infl = primary_hessian[[infl_idx, infl_idx]];
            for i in 0..z_row.len() {
                gradient[i] -= g_infl * z_row[i];
                if h_infl != 0.0 {
                    let hz = h_infl * z_row[i];
                    for j in 0..z_row.len() {
                        hessian[[i, j]] += hz * z_row[j];
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn accumulate_identity_primary_cross_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        core_hessian_column: ndarray::ArrayView1<'_, f64>,
        joint_block: &std::ops::Range<usize>,
        joint_local: usize,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let joint_idx = joint_block.start + joint_local;
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for coeff_idx in 0..slices.time.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_time[q_idx][coeff_idx];
            }
            joint_hessian[[slices.time.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.time.start + coeff_idx]] += value;
        }
        for coeff_idx in 0..slices.marginal.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_marginal[q_idx][coeff_idx];
            }
            joint_hessian[[slices.marginal.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.marginal.start + coeff_idx]] += value;
        }
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("accumulate_identity_primary_cross_hessian logslope try_row_chunk: {e}")
            })?;
        let logslope_row = logslope_chunk.row(0);
        let logslope_weight = core_hessian_column[3];
        if logslope_weight != 0.0 {
            for coeff_idx in 0..slices.logslope.len() {
                let value = logslope_weight * logslope_row[coeff_idx];
                joint_hessian[[slices.logslope.start + coeff_idx, joint_idx]] += value;
                joint_hessian[[joint_idx, slices.logslope.start + coeff_idx]] += value;
            }
        }
        Ok(())
    }

    /// Perf (#large-scale): pre-scaled variant of
    /// [`Self::accumulate_identity_primary_cross_hessian`]. The influence
    /// absorber needs the `o_infl` core-Hessian column scaled by the per-row
    /// per-coefficient factor `Z̃[row, i]`. The previous call site materialised
    /// `&core_col.to_owned() * z_i` — two fresh `Array1` allocations per (row,
    /// influence-coefficient) pair — purely to pass a scaled view in. Here the
    /// scale is folded `core_hessian_column[q] * scale` *before* multiplying by
    /// the Jacobian, exactly matching the original operand grouping
    /// `(core_col[q] * z_i) * dq`, so every accumulated cell is bit-identical
    /// while the per-row heap traffic in the influence-active path is removed.
    pub(crate) fn accumulate_identity_primary_cross_hessian_scaled(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        core_hessian_column: ndarray::ArrayView1<'_, f64>,
        scale: f64,
        joint_block: &std::ops::Range<usize>,
        joint_local: usize,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let joint_idx = joint_block.start + joint_local;
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        // Fold the scale into the three primary-q weights up front so the
        // per-coefficient inner product reads `(core[q] * scale) * dq[q]`,
        // matching the pre-scaled-column arithmetic exactly.
        let scaled_core = [
            core_hessian_column[0] * scale,
            core_hessian_column[1] * scale,
            core_hessian_column[2] * scale,
        ];

        for coeff_idx in 0..slices.time.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += scaled_core[q_idx] * dq_time[q_idx][coeff_idx];
            }
            joint_hessian[[slices.time.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.time.start + coeff_idx]] += value;
        }
        for coeff_idx in 0..slices.marginal.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += scaled_core[q_idx] * dq_marginal[q_idx][coeff_idx];
            }
            joint_hessian[[slices.marginal.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.marginal.start + coeff_idx]] += value;
        }
        let logslope_weight = core_hessian_column[3] * scale;
        if logslope_weight != 0.0 {
            let logslope_chunk = self
                .logslope_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| {
                    format!(
                        "accumulate_identity_primary_cross_hessian_scaled logslope try_row_chunk: {e}"
                    )
                })?;
            let logslope_row = logslope_chunk.row(0);
            for coeff_idx in 0..slices.logslope.len() {
                let value = logslope_weight * logslope_row[coeff_idx];
                joint_hessian[[slices.logslope.start + coeff_idx, joint_idx]] += value;
                joint_hessian[[joint_idx, slices.logslope.start + coeff_idx]] += value;
            }
        }
        Ok(())
    }

    pub(crate) fn add_dense_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        target_rows: &std::ops::Range<usize>,
        target_cols: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for row_local in 0..target_rows.len() {
            for col_local in 0..target_cols.len() {
                joint_hessian[[target_rows.start + row_local, target_cols.start + col_local]] +=
                    source[[row_local, col_local]];
            }
        }
    }

    pub(crate) fn add_dense_symmetric_cross_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        left_range: &std::ops::Range<usize>,
        right_range: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for left_local in 0..left_range.len() {
            for right_local in 0..right_range.len() {
                let value = source[[left_local, right_local]];
                joint_hessian[[
                    left_range.start + left_local,
                    right_range.start + right_local,
                ]] += value;
                joint_hessian[[
                    right_range.start + right_local,
                    left_range.start + left_local,
                ]] += value;
            }
        }
    }

    pub(crate) fn accumulate_dynamic_q_joint_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        joint_gradient: &mut Array1<f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_core_gradient(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            joint_gradient,
        )?;
        self.accumulate_dynamic_q_core_hessian(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            joint_hessian,
        )?;

        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                joint_gradient[joint_range.start + local] -=
                    primary_gradient[primary_range.start + local];
                self.accumulate_identity_primary_cross_hessian(
                    row,
                    slices,
                    q_geom,
                    primary_hessian.slice(s![0..N_PRIMARY, primary_range.start + local]),
                    joint_range,
                    local,
                    joint_hessian,
                )?;
            }
            self.add_dense_submatrix(
                joint_hessian,
                joint_range,
                joint_range,
                primary_hessian.slice(s![primary_range.clone(), primary_range.clone()]),
            );
        }

        for left_idx in 0..identity_blocks.len() {
            for right_idx in left_idx + 1..identity_blocks.len() {
                let (left_primary, left_joint) = &identity_blocks[left_idx];
                let (right_primary, right_joint) = &identity_blocks[right_idx];
                self.add_dense_symmetric_cross_submatrix(
                    joint_hessian,
                    left_joint,
                    right_joint,
                    primary_hessian.slice(s![left_primary.clone(), right_primary.clone()]),
                );
            }
        }

        // Absorbed Stage-1 influence block (#461). The absorber is a SINGLE
        // primary scalar `o_infl` at index `primary.infl` whose `p₁` joint
        // coefficients `γ` map to it through the residualized design row
        // `Z̃_infl[row,:]` (NOT an identity block — unlike score_warp/link_dev
        // whose bases are themselves primary coordinates). It therefore projects
        // like a non-identity single-scalar channel: each γ-coefficient `i` acts
        // as a copy of the `o_infl` primary direction scaled by `Z̃[row,i]`, so
        // its gradient and all cross-Hessians (vs core time/marginal/logslope and
        // vs the identity flex blocks) and its own diagonal are the `o_infl`
        // primary entries weighted by `Z̃[row,·]`.
        if let (Some(infl_primary), Some(infl_joint)) =
            (flex_primary_slices(self).infl, slices.influence.as_ref())
        {
            let z_tilde = self.influence_absorber.as_ref().ok_or_else(|| {
                "accumulate_dynamic_q_joint_row: influence primary index present but no Z̃ design"
                    .to_string()
            })?;
            let z_row = z_tilde.row(row);
            let core_col = primary_hessian.slice(s![0..N_PRIMARY, infl_primary]);
            // Per-coefficient gradient + cross with core blocks (time/marginal/
            // logslope), reusing the identity-channel core-cross helper with the
            // `o_infl` core Hessian column scaled by `Z̃[row, i]`.
            for i in 0..z_row.len() {
                let z_i = z_row[i];
                joint_gradient[infl_joint.start + i] -= primary_gradient[infl_primary] * z_i;
                if z_i != 0.0 {
                    // Perf (#large-scale): pass the unscaled `o_infl` core-Hessian
                    // column plus `z_i` to the pre-scaled cross-Hessian helper,
                    // which folds the scale in `(core[q] * z_i) * dq` form —
                    // bit-identical to the previous `&core_col.to_owned() * z_i`
                    // path but without the two per-(row,coeff) Array1 allocations.
                    self.accumulate_identity_primary_cross_hessian_scaled(
                        row,
                        slices,
                        q_geom,
                        core_col,
                        z_i,
                        infl_joint,
                        i,
                        joint_hessian,
                    )?;
                }
            }
            // Influence × influence diagonal: primary_hessian[[infl,infl]]·Z̃Z̃ᵀ.
            let ii_weight = primary_hessian[[infl_primary, infl_primary]];
            if ii_weight != 0.0 {
                for i in 0..z_row.len() {
                    for j in 0..z_row.len() {
                        joint_hessian[[infl_joint.start + i, infl_joint.start + j]] +=
                            ii_weight * z_row[i] * z_row[j];
                    }
                }
            }
            // Influence × identity-flex (score_warp/link_dev) cross-blocks: each
            // flex coefficient `f` (a primary coordinate) crossed with each
            // absorber coefficient `i` is `primary_hessian[[flex, infl]]·Z̃[row,i]`.
            for (flex_primary, flex_joint) in identity_blocks {
                for f in 0..flex_primary.len() {
                    let weight = primary_hessian[[flex_primary.start + f, infl_primary]];
                    if weight == 0.0 {
                        continue;
                    }
                    let fj = flex_joint.start + f;
                    for i in 0..z_row.len() {
                        let value = weight * z_row[i];
                        joint_hessian[[fj, infl_joint.start + i]] += value;
                        joint_hessian[[infl_joint.start + i, fj]] += value;
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn evaluate_blockwise_exact_newton_dynamic_q<RowTerms>(
        &self,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        row_terms: RowTerms,
    ) -> Result<FamilyEvaluation, String>
    where
        RowTerms: Fn(
                usize,
                &SurvivalMarginalSlopeDynamicRow,
            ) -> Result<(f64, Array1<f64>, Array2<f64>), String>
            + Sync,
    {
        let slices = block_slices(self, block_states);
        let make_acc = || DynamicQBlockwiseAccumulator::new(&slices);
        // See `evaluate_exact_newton_joint_dynamic_q_dense` for rationale.
        let make_acc_ws = || {
            (
                make_acc(),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let acc = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<_, String> {
                let mut acc = make_acc_ws();
                for row in range {
                    let (state, q_geom) = &mut acc;
                    self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                    let (row_nll, primary_gradient, primary_hessian) = row_terms(row, q_geom)?;
                    state.log_likelihood -= row_nll;
                    self.accumulate_dynamic_q_blockwise_row(
                        row,
                        q_geom,
                        primary,
                        primary_gradient.view(),
                        primary_hessian.view(),
                        state,
                    )?;
                }
                Ok(acc)
            },
            |mut left, right| -> Result<_, String> {
                left.0.add_assign(&right.0);
                Ok(left)
            },
        )?
        .unwrap_or_else(make_acc_ws)
        .0;

        Ok(acc.into_family_evaluation())
    }
}
