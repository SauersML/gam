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

        // #932 single-source: the flex row NLL value + gradient fall out of the
        // ONE generic row-NLL expression (`flex_row_nll`) instantiated at the
        // value/grad jet — no separate hand-assembled probit chain / quotient
        // rules. Grad-only: the first-order packs carry no `*_uv`, so the Hessian
        // views are `None` (the value/gradient channels never read the Hessian).
        let (row_nll, grad, _) = self.flex_row_nll_value_grad_hess(
            row,
            primary,
            q1,
            qd1,
            entry.eta,
            entry.eta_u.view(),
            None,
            exit.eta,
            exit.eta_u.view(),
            None,
            exit.chi,
            exit.chi_u.view(),
            None,
            exit.d,
            exit.d_u.view(),
            None,
        )?;

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

        // #932 single-source: value, gradient AND Hessian fall out of the ONE
        // generic flex row-NLL expression (`flex_row_nll`) instantiated at the
        // value/grad/Hessian jet (`Jet2`). This replaces both the hand-assembled
        // probit-chain + quotient-rule loops AND the bespoke Step-5 device-shape
        // pullback: there is exactly one definition of the flex row likelihood,
        // and the (v, g, H) channels are the order-≤2 part of the same expression
        // whose order-3/4 directional contractions
        // `flex_row_nll_{third,fourth}_contracted` evaluate.
        let (row_nll, grad, hess) = self.flex_row_nll_value_grad_hess(
            row,
            primary,
            q1,
            qd1,
            entry.eta,
            entry.eta_u.view(),
            Some(entry.eta_uv.view()),
            exit.eta,
            exit.eta_u.view(),
            Some(exit.eta_uv.view()),
            exit.chi,
            exit.chi_u.view(),
            Some(exit.chi_uv.view()),
            exit.d,
            exit.d_u.view(),
            Some(exit.d_uv.view()),
        )?;

        Ok((row_nll, grad, hess))
    }
}
