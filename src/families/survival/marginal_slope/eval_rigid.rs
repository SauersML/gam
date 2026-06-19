//! Rigid (no-flex) per-row evaluation: the rigid closed-form primary kernel
//! wrapper, scalar-flex score-geometry guard, the rigid vector NLL value,
//! and installation of the automatic outer-subsampling options.

use super::*;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn row_primary_closed_form_rigid(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
        let logslope_eta = &block_states[2].eta;
        let k = self.score_dim();
        if k == 1 {
            return row_primary_closed_form(
                q0,
                q1,
                qd1,
                logslope_eta[row],
                self.z[[row, 0]],
                self.weights[row],
                self.event[row],
                self.derivative_guard,
                probit_scale,
            )
            .map_err(|err| with_row_context(err, row));
        }
        if logslope_eta.len() != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope exact rigid row calculus for K={k} requires one shared log-slope surface (eta len n={}); got eta len {}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    logslope_eta.len(),
                    3 + k
                ),
            }
            .into());
        }
        let (z_sum, covariance_ones) =
            self.exact_shared_score_summary(row, block_states, "row_primary_closed_form_rigid")?;
        row_primary_closed_form_shared_score(
            q0,
            q1,
            qd1,
            logslope_eta[row],
            z_sum,
            covariance_ones,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
        .map_err(|err| with_row_context(err, row))
    }

    pub(crate) fn ensure_scalar_flex_exact_score_geometry(
        &self,
        context: &str,
    ) -> Result<(), String> {
        if self.score_dim() == 1 {
            return Ok(());
        }
        Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "{context}: survival marginal-slope exact flexible row calculus is scalar-z only; K={} must use the rigid shared-slope vector kernel or a widened per-z primary kernel",
                self.score_dim()
            ),
        }
        .into())
    }

    pub(crate) fn row_neglog_rigid_vector_value(
        &self,
        row: usize,
        q_geom: SurvivalMarginalSlopeDynamicRowValues,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<f64, String> {
        let slopes = if self.per_z_logslope_active() {
            self.logslope_surface_values_for_row(row, &block_states[2].beta)?
        } else {
            self.logslope_vector_for_row(row, &block_states[2].eta)?
        };
        let z = self.z.row(row).to_vec();
        survival_marginal_slope_vector_neglog(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            &slopes,
            &z,
            &self.score_covariance,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
    }

    /// Two-phase auto-subsample entry: when `options.auto_outer_subsample` is
    /// enabled, an outer-derivative-scoped `OuterEvalContext` is present, and
    /// Phase 1 still has budget, this returns a cloned `BlockwiseFitOptions`
    /// carrying a freshly built stratified Horvitz-Thompson mask. Otherwise
    /// returns `None` and the caller uses the original options unchanged.
    ///
    /// Keying is on the outer ρ published by the smoothing optimizer through
    /// `options.outer_eval_context` — never on the inner β. During inner
    /// trust-region / joint-Newton trial steps β changes between calls at the
    /// same outer ρ, so β-keying would re-fire phase prints and rebuild the
    /// row mask inside one outer eval, which makes the trust-region ratio
    /// compare objectives evaluated on different row measures (invalid).
    /// Inner-scope contexts (set by `coefficient_line_search_options`) make
    /// this entry return `None` immediately.
    ///
    /// CONTRACT: MUST NOT be called from inner-coefficient paths (line-search,
    /// trust-region globalization). The InnerCoefficient scope guard below is
    /// the enforcement mechanism; this comment makes the contract obvious.
    /// Cf. `src/solver/row_measure.rs` and the TR row-measure invariant in
    /// `inner_blockwise_fit`.
    pub(crate) fn install_auto_outer_subsample_options(
        &self,
        options: &BlockwiseFitOptions,
    ) -> Option<BlockwiseFitOptions> {
        let ctx = options.outer_eval_context.as_ref()?;
        if !matches!(ctx.scope, crate::custom_family::EvalScope::OuterDerivative) {
            return None;
        }
        let event_secondary: Vec<u8> = self
            .event
            .iter()
            .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
            .collect();
        let z_key = self.z_subsample_key();
        let small_fixture_auto = self.n >= 500 && self.n < 30_000;
        crate::families::marginal_slope_shared::maybe_install_auto_outer_subsample(
            options,
            z_key.as_slice().expect("z key must be contiguous"),
            Some(&event_secondary),
            ctx.rho.as_slice().expect("outer rho must be contiguous"),
            &self.auto_subsample_phase_counter,
            &self.auto_subsample_last_rho,
            SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET,
            "survival-mgs",
            // Per-K work-unit cost for the survival marginal-slope outer
            // gradient kernel. Calibrated from the large-scale repro
            // (n=195_780, K=19_661, predicted_gradient_work ≈ 4.33×10⁹):
            //   per_K-unit cost ≈ 4.33e9 / 19_661 ≈ 220_000 units.
            // With `AUTO_OUTER_WORK_BUDGET = 5×10⁸`, this caps
            //   K_work ≈ 5e8 / 250_000 ≈ 2_000,
            // bounding outer gradient work below ~5×10⁸ units even
            // when the noise-only rule would request K ≈ 0.1n. Without
            // this cap the rigid pilot and outer line search spend
            // ~57 minutes per evaluation on large-scale joint designs
            // before the identifiability gate even gets a chance to
            // veto rank-deficient configurations.
            250_000,
            if small_fixture_auto { 500 } else { 30_000 },
            if small_fixture_auto { 200 } else { 10_000 },
            if small_fixture_auto { 200 } else { 1_000 },
        )
    }
}
