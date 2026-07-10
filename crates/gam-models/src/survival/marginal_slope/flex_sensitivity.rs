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

        // The intercept solve's density-normalization check `d0`/`d1` is recomputed
        // inside the jet timepoint builder (`evaluate_survival_denom_d`), so only the
        // solved intercepts `a0`/`a1` are needed here (mirrors the grad+Hessian path).
        let (a0, _d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, _d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        // #932 grad-only single-source: the exact first-order timepoint value/grad
        // come from the single-source `flex_timepoint_inputs_generic` jet builder at
        // Jet1 (`compute_survival_timepoint_first_order_exact`), the order-≤1 twin of
        // the grad+Hessian `compute_survival_timepoint_exact_jet` (Jet2) — no hand
        // first-order cell-moment / IFT / moving-flux assembly.
        let entry = self.compute_survival_timepoint_first_order_exact(
            row, primary, q0, primary.q0, a0, g, beta_h, beta_w, o_infl,
        )?;
        let exit = self.compute_survival_timepoint_first_order_exact(
            row, primary, q1, primary.q1, a1, g, beta_h, beta_w, o_infl,
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
            crate::survival::marginal_slope::timepoint_exact::flex_jet::FlexRowJet2Channels {
                eta0_v: entry.eta,
                eta0_g: entry.eta_u.view(),
                eta0_h: None,
                eta1_v: exit.eta,
                eta1_g: exit.eta_u.view(),
                eta1_h: None,
                chi1_v: exit.chi,
                chi1_g: exit.chi_u.view(),
                chi1_h: None,
                d1_v: exit.d,
                d1_g: exit.d_u.view(),
                d1_h: None,
            },
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

        // The intercept solve's density-normalization check `d0`/`d1` is recomputed
        // inside the jet timepoint builder (`evaluate_survival_denom_d`), so only the
        // solved intercepts `a0`/`a1` are needed here.
        let (a0, _d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, _d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        // #932-2 PRODUCTION cutover: the exact timepoint value/grad/Hessian come from
        // the single-source `flex_timepoint_inputs_generic` jet builder (Jet2), not
        // the hand `compute_survival_timepoint_exact` probit-chain assembly.
        let entry = self.compute_survival_timepoint_exact_jet(
            row, primary, q0, primary.q0, a0, g, beta_h, beta_w, o_infl,
        )?;
        let exit = self.compute_survival_timepoint_exact_jet(
            row, primary, q1, primary.q1, a1, g, beta_h, beta_w, o_infl,
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
            crate::survival::marginal_slope::timepoint_exact::flex_jet::FlexRowJet2Channels {
                eta0_v: entry.eta,
                eta0_g: entry.eta_u.view(),
                eta0_h: Some(entry.eta_uv.view()),
                eta1_v: exit.eta,
                eta1_g: exit.eta_u.view(),
                eta1_h: Some(exit.eta_uv.view()),
                chi1_v: exit.chi,
                chi1_g: exit.chi_u.view(),
                chi1_h: Some(exit.chi_uv.view()),
                d1_v: exit.d,
                d1_g: exit.d_u.view(),
                d1_h: Some(exit.d_uv.view()),
            },
        )?;

        Ok((row_nll, grad, hess))
    }
}
