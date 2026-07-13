//! Primary-Hessian pullback into coefficient space: pulling the per-row
//! 4x4 primary Hessian back through the block Jacobians (full and
//! block-diagonal), and the psi-block routing metadata.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    ///
    /// Writes directly into `target` subslices via sparse-aware row primitives —
    /// no dense row buffers or temporary blocks are allocated.
    pub(crate) fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];

        // time-time block: Σ_{a,b} H[a,b] * time_a_row ⊗ time_b_row
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(
                        row,
                        time_designs[b],
                        alpha,
                        target.slice_mut(s![slices.time.clone(), slices.time.clone()]),
                    )
                    .expect("time block row_outer_into dimension mismatch");
            }
        }

        // marginal-marginal block: (H[0,0]+H[0,1]+H[1,0]+H[1,1]) * m_row ⊗ m_row
        self.marginal_design
            .syr_row_into_view(
                row,
                h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]],
                target.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()]),
            )
            .expect("marginal syr_row_into dimension mismatch");

        // logslope-logslope block: H[3,3] * g_row ⊗ g_row
        self.logslope_layout.coefficient_design()
            .syr_row_into_view(
                row,
                h[[3, 3]],
                target.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()]),
            )
            .expect("logslope syr_row_into dimension mismatch");

        // marginal-logslope block: (H[0,3]+H[1,3]) * m_row ⊗ g_row  (+ transpose)
        {
            let alpha_mg = h[[0, 3]] + h[[1, 3]];
            if alpha_mg != 0.0 {
                self.marginal_design
                    .row_outer_into_view(
                        row,
                        &self.logslope_layout.coefficient_design(),
                        alpha_mg,
                        target.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()]),
                    )
                    .expect("marginal-logslope row_outer_into dimension mismatch");
                self.logslope_layout.coefficient_design()
                    .row_outer_into_view(
                        row,
                        &self.marginal_design,
                        alpha_mg,
                        target.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()]),
                    )
                    .expect("logslope-marginal row_outer_into dimension mismatch");
            }
        }

        // time-logslope block: H[a,3] * time_a_row ⊗ g_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 3]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.logslope_layout.coefficient_design(),
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.logslope.clone()]),
                )
                .expect("time-logslope row_outer_into dimension mismatch");
            self.logslope_layout.coefficient_design()
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.logslope.clone(), slices.time.clone()]),
                )
                .expect("logslope-time row_outer_into dimension mismatch");
        }

        // time-marginal block: (H[a,0]+H[a,1]) * time_a_row ⊗ m_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 0]] + h[[a, 1]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.marginal_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.marginal.clone()]),
                )
                .expect("time-marginal row_outer_into dimension mismatch");
            self.marginal_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.marginal.clone(), slices.time.clone()]),
                )
                .expect("marginal-time row_outer_into dimension mismatch");
        }
    }

    /// Block-diagonal-only pullback: writes only the principal time-time,
    /// marginal-marginal, and logslope-logslope rowwise contributions into
    /// per-block targets. Used by `evaluate()` to populate per-block working
    /// sets without ever materializing the cross blocks.
    pub(crate) fn add_pullback_block_diagonals(
        &self,
        row: usize,
        primary_hessian: &Array2<f64>,
        time_target: &mut Array2<f64>,
        marginal_target: &mut Array2<f64>,
        logslope_target: &mut Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(row, time_designs[b], alpha, time_target.view_mut())
                    .expect("time block row_outer_into dimension mismatch");
            }
        }
        let alpha_mm = h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]];
        self.marginal_design
            .syr_row_into_view(row, alpha_mm, marginal_target.view_mut())
            .expect("marginal syr_row_into dimension mismatch");
        self.logslope_layout.coefficient_design()
            .syr_row_into_view(row, h[[3, 3]], logslope_target.view_mut())
            .expect("logslope syr_row_into dimension mismatch");
    }

    pub(crate) fn row_primary_direction_from_flat_dynamic(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        self.row_primary_direction_from_flat_dynamic_with_q_geometry(
            row,
            block_states,
            slices,
            &q_geom,
            d_beta_flat,
        )
    }

    pub(crate) fn row_primary_direction_from_flat_dynamic_with_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let flex_primary = self
            .effective_flex_active(block_states)?
            .then(|| flex_primary_slices(self));
        let mut out = Array1::<f64>::zeros(flex_primary.as_ref().map_or(N_PRIMARY, |p| p.total));
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);

        let q0_dir = q_geom.dq0_time.dot(&d_time) + q_geom.dq0_marginal.dot(&d_marginal);
        let q1_dir = q_geom.dq1_time.dot(&d_time) + q_geom.dq1_marginal.dot(&d_marginal);
        let qd1_dir = q_geom.dqd1_time.dot(&d_time) + q_geom.dqd1_marginal.dot(&d_marginal);
        let g_dir = self
            .logslope_layout.coefficient_design()
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));

        if let Some(primary) = flex_primary.as_ref() {
            out[primary.q0] = q0_dir;
            out[primary.q1] = q1_dir;
            out[primary.qd1] = qd1_dir;
            out[primary.g] = g_dir;
            for (primary_range, block_range) in flex_identity_block_pairs(primary, slices) {
                out.slice_mut(s![primary_range])
                    .assign(&d_beta_flat.slice(s![block_range]));
            }
        } else {
            out[0] = q0_dir;
            out[1] = q1_dir;
            out[2] = qd1_dir;
            out[3] = g_dir;
        }
        Ok(out)
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    // ── Psi terms (first and second order) ────────────────────────────
    //
    // All three psi methods (first-order, second-order, directional derivative)
    // use block-local accumulation via BlockHessianAccumulator. Per-row work is
    // O(max(p_block²)) instead of O(p²), eliminating the dense p×p bottleneck
    // that breaks multi-axis Duchon / per-axis length scaling.

    /// Resolve psi block info: (block_idx, local_idx, p_block, label).
    pub(crate) fn psi_block_info(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, usize, usize, &'static str)>, String> {
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        match block_idx {
            1 => Ok(Some((
                block_idx,
                local_idx,
                self.marginal_design.ncols(),
                "SurvivalMarginalSlope marginal",
            ))),
            2 => Ok(Some((
                block_idx,
                local_idx,
                self.logslope_layout.coefficient_design().ncols(),
                "SurvivalMarginalSlope logslope",
            ))),
            _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival marginal-slope psi: only baseline/slope spatial blocks are supported, got block {block_idx}"
                ),
            }
            .into()),
        }
    }
}
