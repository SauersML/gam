//! Denested survival calibration (externally consumed): partitioning the
//! observation window into integration cells and the closed-form denested
//! intercept calibration evaluated over them.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        if self.score_dim() == 1 {
            return shared_denested_partition_cells(
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }
        let score_breaks = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| self.score_warp_local_cubic_at(beta_h, z),
            |u| {
                if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
                    runtime.local_cubic_at(beta_w, u)
                } else {
                    Ok(Self::zero_score_warp_span())
                }
            },
        )?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    pub(crate) fn evaluate_denested_survival_calibration(
        &self,
        a: f64,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -crate::probability::normal_cdf(-q);
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let pos_cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: pos_cell.left,
                right: pos_cell.right,
                c0: -pos_cell.c0,
                c1: -pos_cell.c1,
                c2: -pos_cell.c2,
                c3: -pos_cell.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(neg_cell, 9)?;
            f += state.value;
            let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (dc_daa_pos, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_pos, -scale);
            let dc_daa = scale_coeff4(dc_daa_pos, -scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact_kernel::cell_second_derivative_from_moments(
                neg_cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
        }
        Ok((f, f_a, f_aa))
    }
}
