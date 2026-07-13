//! Per-cell denested partials: the observed eta/chi cell partials, their
//! fixed first/second partials, the survival denominator, and the flex-row
//! negative-log-likelihood value assembled from cached cell parts.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn observed_denested_eta_chi(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let z_obs = self.observed_score_projection(row);
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta = eval_coeff4_at(&obs.coeff, z_obs);
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        Ok((eta, chi))
    }

    pub(crate) fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        let z_obs = self.observed_score_projection(row);
        if self.score_dim() == 1 {
            return shared_observed_denested_cell_partials(
                z_obs,
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }

        // Observed vector-z contribution is the row-wise direct-sum value
        //     h_i = sum_k W_k(z_ik) beta_k.
        // In the denested additive transport, h_i enters as b*h_i at the
        // observed row.  Holding z_i fixed makes this a constant coefficient
        // in the observed polynomial while preserving the standard link
        // partials in a and b.
        let h_obs = self.score_warp_observed_value(row, beta_h)?;
        let u_obs = a + b * z_obs;
        let link_span = if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
            runtime.local_cubic_at(beta_w.view(), u_obs)?
        } else {
            Self::zero_score_warp_span()
        };
        let (d0, d1, d2, d3) = exact_kernel::transformed_link_cubic(link_span, a, b);
        let coeff_raw = [a + b * h_obs + d0, b + d1, d2, d3];
        let shift = a - link_span.left;
        let alpha1 = link_span.c1;
        let alpha2 = link_span.c2;
        let alpha3 = link_span.c3;
        let dc_da_raw = [
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
            0.0,
        ];
        let dc_db_raw = [
            h_obs,
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
        ];
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::link_basis_cell_second_partials(link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::link_basis_cell_third_partials(link_span);
        let scale = self.probit_frailty_scale();
        Ok(ObservedDenestedCellPartials {
            coeff: scale_coeff4(coeff_raw, scale),
            dc_da: scale_coeff4(dc_da_raw, scale),
            dc_db: scale_coeff4(dc_db_raw, scale),
            dc_daa: scale_coeff4(dc_daa_raw, scale),
            dc_dab: scale_coeff4(dc_dab_raw, scale),
            dc_dbb: scale_coeff4(dc_dbb_raw, scale),
            dc_daaa: scale_coeff4(dc_daaa_raw, scale),
            dc_daab: scale_coeff4(dc_daab_raw, scale),
            dc_dabb: scale_coeff4(dc_dabb_raw, scale),
            dc_dbbb: scale_coeff4(dc_dbbb_raw, scale),
        })
    }

    pub(crate) fn denested_cell_primary_fixed_partials(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        score_span: exact_kernel::LocalSpanCubic,
        link_span: exact_kernel::LocalSpanCubic,
        z_basis: f64,
        u_basis: f64,
    ) -> Result<DenestedCellPrimaryFixedPartials, String> {
        let scale = self.probit_frailty_scale();
        let r = primary.total;
        let mut coeff_u = vec![[0.0; 4]; r];
        let mut coeff_au = vec![[0.0; 4]; r];
        let mut coeff_bu = vec![[0.0; 4]; r];
        let mut coeff_aau = vec![[0.0; 4]; r];
        let mut coeff_abu = vec![[0.0; 4]; r];
        let mut coeff_bbu = vec![[0.0; 4]; r];
        let mut coeff_aaau = vec![[0.0; 4]; r];
        let mut coeff_aabu = vec![[0.0; 4]; r];
        let mut coeff_abbu = vec![[0.0; 4]; r];
        let mut coeff_bbbu = vec![[0.0; 4]; r];

        let (dc_da_raw, dc_db_raw) =
            exact_kernel::denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::denested_cell_third_partials(link_span);
        let dc_da = scale_coeff4(dc_da_raw, scale);
        let dc_db = scale_coeff4(dc_db_raw, scale);
        let dc_daa = scale_coeff4(dc_daa_raw, scale);
        let dc_dab = scale_coeff4(dc_dab_raw, scale);
        let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
        let dc_daaa = scale_coeff4(dc_daaa_raw, scale);
        let dc_daab = scale_coeff4(dc_daab_raw, scale);
        let dc_dabb = scale_coeff4(dc_dabb_raw, scale);
        let dc_dbbb = scale_coeff4(dc_dbbb_raw, scale);

        coeff_u[primary.g] = dc_db;
        coeff_au[primary.g] = dc_dab;
        coeff_bu[primary.g] = dc_dbb;
        coeff_aau[primary.g] = dc_daab;
        coeff_abu[primary.g] = dc_dabb;
        coeff_bbu[primary.g] = dc_dbbb;
        coeff_aaau[primary.g] = [0.0; 4];
        coeff_aabu[primary.g] = [0.0; 4];
        coeff_abbu[primary.g] = [0.0; 4];
        coeff_bbbu[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, b)?,
                    scale,
                );
                coeff_bu[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, 1.0)?,
                    scale,
                );
            }
        }

        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_basis)?;
                let idx = w_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                let (dc_aw_raw, dc_bw_raw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let link_second = exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                    (link_second.0, link_second.1, link_second.2);
                let (dc_aaaw_raw, dc_aabw_raw, dc_abbw_raw, dc_bbbw_raw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                coeff_aaau[idx] = scale_coeff4(dc_aaaw_raw, scale);
                coeff_aabu[idx] = scale_coeff4(dc_aabw_raw, scale);
                coeff_abbu[idx] = scale_coeff4(dc_abbw_raw, scale);
                coeff_bbbu[idx] = scale_coeff4(dc_bbbw_raw, scale);
            }
        }

        Ok(DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa,
            dc_daaa,
            coeff_u,
            coeff_au,
            coeff_bu,
            coeff_aau,
            coeff_abu,
            coeff_bbu,
            coeff_aaau,
            coeff_aabu,
            coeff_abbu,
            coeff_bbbu,
        })
    }

    pub(crate) fn evaluate_survival_denom_d(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        // Density normalization is |F'(a)| for the same calibration equation
        // solved by `solve_row_survival_intercept`. Reusing that exact
        // derivative convention avoids sign drift between the solver path and
        // the direct-check path.
        let (_, f_a, _) = self.evaluate_denested_survival_calibration(a, 0.0, b, beta_h, beta_w)?;
        let d = f_a.abs();
        if !d.is_finite() || d <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope produced non-positive calibration derivative |F'(a)|={d:.3e}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    pub(crate) fn row_neglog_flex_value(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        // Absorbed Stage-1 influence offset (#461): a per-row additive shift of
        // the de-nested observed index η₁ (un-`c(g)`-scaled), `0.0` when no
        // absorber is installed.
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.row_neglog_flex_value_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl,
        )
    }

    pub(crate) fn row_neglog_flex_value_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<f64, String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }
        let (a0, _) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        if !d1.is_finite() || d1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D1={d1:.3e} (calibration derivative {:.3e})",
                    d1
                ),
            }
            .into());
        }
        // The absorbed-influence offset shifts the observed index η₁ additively
        // at both the entry (eta0) and exit (eta1) calibration roots — `o_infl`
        // is independent of the calibration intercept `a`, so the de-nesting
        // derivative `chi1 = ∂η₁/∂a` is unchanged.
        let (eta0_raw, _) = self.observed_denested_eta_chi(row, a0, g, beta_h, beta_w)?;
        let (eta1_raw, chi1) = self.observed_denested_eta_chi(row, a1, g, beta_h, beta_w)?;
        let eta0 = eta0_raw + o_infl;
        let eta1 = eta1_raw + o_infl;
        if !chi1.is_finite() || chi1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={chi1:.3e}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        Ok(wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * chi1.ln()
                - di * log_phi_q1
                + di * d1.ln()
                - di * qd1.ln()))
    }
}
