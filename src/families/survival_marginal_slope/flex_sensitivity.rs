//! Flex primary sensitivities: the per-row flex primary gradient and
//! gradient+Hessian, computed exactly and from cached cell parts.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn compute_row_flex_primary_gradient_hessian_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry(
            "compute_row_flex_primary_gradient_hessian_exact",
        )?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.compute_row_flex_primary_gradient_hessian_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl, primary,
        )
    }

    pub(crate) fn compute_row_flex_primary_gradient_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry("compute_row_flex_primary_gradient_exact")?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;
        self.compute_row_flex_primary_gradient_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, o_infl, primary,
        )
    }

    pub(crate) fn compute_row_flex_primary_gradient_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
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
        let entry = self.compute_survival_timepoint_first_order_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl,
        )?;
        let exit = self.compute_survival_timepoint_first_order_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, _, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, _, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;
        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let p = primary.total;
        let mut grad = Array1::<f64>::zeros(p);
        let entry_u1 = -entry_k1;
        let exit_surv_u1 = -exit_k1;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        Ok((row_nll, grad))
    }

    pub(crate) fn compute_row_flex_primary_gradient_hessian_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
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
        let entry = self.compute_survival_timepoint_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, o_infl, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, o_infl, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, entry_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, exit_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;

        // ── Step-6 dispatcher: try GPU Step-5 assembly first ──────────────
        //
        // The per-row primary G/H assembly is pure scalar/vector algebra over
        // the jets; the prep dispatchers (`try_device_partition_cells` +
        // `try_device_cell_primary_fixed_partials`) already produced these
        // jets via `compute_survival_timepoint_exact` →
        // `build_cached_partition_with_moment_order`, so the only remaining
        // family-side hop is the Step-5 G/H pullback in
        // [`crate::families::survival_marginal_slope_gpu::try_device_step5_primary_assembly`].
        //
        // The GPU entry takes flat `&[f64]` views; both `Array1` and `Array2`
        // returned by the timepoint-exact pass live in standard contiguous
        // layout (built via `Array1::zeros` / `Array2::zeros` above), so the
        // `as_slice()` extractions below are infallible in practice; we
        // route through the CPU fallback when any of them happens not to be
        // contiguous, preserving the legacy code path as the source of
        // truth.
        let p = primary.total;
        let entry_eta_u = entry.eta_u.as_slice();
        let entry_eta_uv = entry.eta_uv.as_slice();
        let entry_chi_u = entry.chi_u.as_slice();
        let entry_chi_uv = entry.chi_uv.as_slice();
        let entry_d_u = entry.d_u.as_slice();
        let entry_d_uv = entry.d_uv.as_slice();
        let exit_eta_u = exit.eta_u.as_slice();
        let exit_eta_uv = exit.eta_uv.as_slice();
        let exit_chi_u = exit.chi_u.as_slice();
        let exit_chi_uv = exit.chi_uv.as_slice();
        let exit_d_u = exit.d_u.as_slice();
        let exit_d_uv = exit.d_uv.as_slice();
        let all_contiguous = entry_eta_u.is_some()
            && entry_eta_uv.is_some()
            && entry_chi_u.is_some()
            && entry_chi_uv.is_some()
            && entry_d_u.is_some()
            && entry_d_uv.is_some()
            && exit_eta_u.is_some()
            && exit_eta_uv.is_some()
            && exit_chi_u.is_some()
            && exit_chi_uv.is_some()
            && exit_d_u.is_some()
            && exit_d_uv.is_some();
        if all_contiguous
            && exit.chi.is_finite()
            && exit.chi > 0.0
            && exit.d.is_finite()
            && exit.d > 0.0
        {
            let row_inputs = [
                crate::families::survival_marginal_slope_gpu::SurvivalFlexStep5RowInputs {
                    entry: crate::families::survival_marginal_slope_gpu::SurvivalFlexTimepointJet {
                        eta: entry.eta,
                        chi: entry.chi,
                        d: entry.d,
                        eta_u: entry_eta_u.unwrap(),
                        eta_uv: entry_eta_uv.unwrap(),
                        chi_u: entry_chi_u.unwrap(),
                        chi_uv: entry_chi_uv.unwrap(),
                        d_u: entry_d_u.unwrap(),
                        d_uv: entry_d_uv.unwrap(),
                    },
                    exit: crate::families::survival_marginal_slope_gpu::SurvivalFlexTimepointJet {
                        eta: exit.eta,
                        chi: exit.chi,
                        d: exit.d,
                        eta_u: exit_eta_u.unwrap(),
                        eta_uv: exit_eta_uv.unwrap(),
                        chi_u: exit_chi_u.unwrap(),
                        chi_uv: exit_chi_uv.unwrap(),
                        d_u: exit_d_u.unwrap(),
                        d_uv: exit_d_uv.unwrap(),
                    },
                    wi,
                    di,
                    q1,
                    qd1,
                    q1_index: primary.q1,
                    qd1_index: primary.qd1,
                    entry_k1,
                    entry_k2,
                    exit_k1,
                    exit_k2,
                    log_surv0,
                    log_surv1,
                },
            ];
            // `try_device_step5_primary_assembly` is the device-shape Step-5
            // pullback (`survival_marginal_slope_gpu`). Its current body
            // is CPU-resident scalar algebra producing the same `(row_nll,
            // grad, hess)` the inline CPU loop below builds; when the NVRTC
            // kernel lands, this call site becomes the device-dispatch seam
            // without further family-side rework.  We fall back to the
            // inline CPU loop on `Err` so any device-side validation failure
            // surfaces as a row-level CPU re-evaluation rather than a hard
            // panic.
            if let Ok(mut out) =
                crate::families::survival_marginal_slope_gpu::try_device_step5_primary_assembly(
                    &row_inputs,
                )
                && out.len() == 1
            {
                let row = out.remove(0);
                if row.grad.len() == p && row.hess.len() == p * p {
                    let grad = Array1::from_vec(row.grad);
                    let hess =
                        Array2::from_shape_vec((p, p), row.hess).map_err(|e| e.to_string())?;
                    return Ok((row.row_nll, grad, hess));
                }
            }
        }

        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let mut grad = Array1::<f64>::zeros(p);
        let mut hess = Array2::<f64>::zeros((p, p));
        let entry_u1 = -entry_k1;
        let entry_u2 = entry_k2;
        let exit_surv_u1 = -exit_k1;
        let exit_surv_u2 = exit_k2;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        for u in 0..p {
            for v in u..p {
                let mut value = 0.0;
                value +=
                    entry_u2 * entry.eta_u[u] * entry.eta_u[v] + entry_u1 * entry.eta_uv[[u, v]];
                value += exit_surv_u2 * exit.eta_u[u] * exit.eta_u[v]
                    + exit_surv_u1 * exit.eta_uv[[u, v]];
                value += wi * di * (exit.eta_u[u] * exit.eta_u[v] + exit.eta * exit.eta_uv[[u, v]]);
                value -= wi
                    * di
                    * (exit.chi_uv[[u, v]] / exit.chi
                        - (exit.chi_u[u] * exit.chi_u[v]) / (exit.chi * exit.chi));
                if u == primary.q1 && v == primary.q1 {
                    value += wi * di;
                }
                value += wi
                    * di
                    * (exit.d_uv[[u, v]] / exit.d
                        - (exit.d_u[u] * exit.d_u[v]) / (exit.d * exit.d));
                if u == primary.qd1 && v == primary.qd1 {
                    value += wi * di / (qd1 * qd1);
                }
                hess[[u, v]] = value;
                hess[[v, u]] = value;
            }
        }

        Ok((row_nll, grad, hess))
    }

}

