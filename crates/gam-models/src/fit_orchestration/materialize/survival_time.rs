use super::*;

pub struct PreparedSurvivalTimeStack {
    pub eta_offset_entry: Array1<f64>,
    pub eta_offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub time_design_entry: gam_linalg::matrix::DesignMatrix,
    pub time_design_exit: gam_linalg::matrix::DesignMatrix,
    pub time_design_derivative_exit: gam_linalg::matrix::DesignMatrix,
    pub time_penalties: Vec<Array2<f64>>,
    pub time_nullspace_dims: Vec<usize>,
    pub timewiggle_build: Option<crate::survival::construction::SurvivalTimeWiggleBuild>,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
}

pub fn prepare_survival_time_stack(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &crate::survival::construction::SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
    inverse_link: Option<&InverseLink>,
    time_anchor: f64,
    derivative_guard: f64,
    time_build: &crate::survival::construction::SurvivalTimeBuildOutput,
    effective_timewiggle: Option<&LinkWiggleFormulaSpec>,
    latent_loading: Option<crate::survival::lognormal_kernel::HazardLoading>,
) -> Result<PreparedSurvivalTimeStack, String> {
    let (
        mut eta_offset_entry,
        mut eta_offset_exit,
        mut derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    ) = if let Some(loading) = latent_loading {
        let offsets =
            build_latent_survival_baseline_offsets(age_entry, age_exit, baseline_cfg, loading)?;
        (
            offsets.loaded_eta_entry,
            offsets.loaded_eta_exit,
            offsets.loaded_derivative_exit,
            offsets.unloaded_mass_entry,
            offsets.unloaded_mass_exit,
            offsets.unloaded_hazard_exit,
        )
    } else {
        // Baseline-hazard barrier conditioning for the marginal-slope likelihood
        // (gam#797). That likelihood carries `-d·log(qd1)`, a log-barrier on the
        // baseline-hazard time derivative `qd1 = X_d·β_time + derivative_offset`.
        // The default `baseline-target=linear` is DEGENERATE for this barrier:
        // `evaluate_survival_baseline` returns `(0, 0)` for Linear, so the offset
        // collapses to `derivative_guard` (1e-6) and the I-spline time seed starts
        // at `qd1 ≈ 1e-6` — exactly ON the barrier boundary, where the
        // self-concordant Newton step is `∝ qd1` (intrinsically ~1e-4), the
        // barrier gradient/Hessian are ~1e6 / ~1e12, and the inner joint-Newton
        // crawls and never reaches the data-scale baseline within the cycle
        // budget — every outer seed is rejected and the fit hard-fails.
        //
        // Condition the COLD START by building the baseline OFFSET from a fixed,
        // data-seeded Weibull (scale = mean positive exit time, shape = 1) instead
        // of the zero-derivative Linear baseline, but ONLY for the offset: the
        // outer `baseline_cfg.target` stays `Linear`, so the
        // `baseline_cfg.target != Linear` optimize gate
        // (the gradient baseline optimizers) never fires and no baseline-shape
        // search is introduced. With shape = 1 the Weibull baseline-hazard
        // derivative is `1/age_exit` (the natural data hazard scale), so the seed
        // starts with `qd1` at O(1/T) interior — barrier gradient O(10-10²),
        // comparable to the marginal/logslope blocks — and `β_time ≈ 0`. This
        // changes only the STARTING point / offset split: the I-spline still learns
        // the data-driven deviation from this parametric baseline (the converged
        // fitted hazard is the same flexible family), so the fix is a pure
        // preconditioning of the cold start. Gated to MarginalSlope with a Linear
        // target so every other Linear-baseline survival path is byte-unchanged.
        let conditioning_cfg;
        let offset_cfg = if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope
            && baseline_cfg.target == SurvivalBaselineTarget::Linear
        {
            let scale = crate::survival::construction::positive_survival_time_seed(age_exit);
            conditioning_cfg = crate::survival::construction::SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(scale),
                shape: Some(1.0),
                rate: None,
                makeham: None,
            };
            &conditioning_cfg
        } else {
            baseline_cfg
        };
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                age_entry,
                age_exit,
                offset_cfg,
                likelihood_mode,
                inverse_link,
            )?;
        let n = age_entry.len();
        (
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::zeros(n),
        )
    };
    add_survival_time_derivative_guard_offset(
        age_entry,
        age_exit,
        time_anchor,
        derivative_guard,
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative_exit = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(wiggle) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative_exit,
            wiggle.ncols,
        );
        for (idx, penalty) in wiggle.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + wiggle.ncols, p_base + wiggle.ncols));
            embedded
                .slice_mut(s![
                    p_base..p_base + wiggle.ncols,
                    p_base..p_base + wiggle.ncols
                ])
                .assign(penalty);
            time_penalties.push(embedded);
            time_nullspace_dims.push(wiggle.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: wiggle.knots.clone(),
            degree: wiggle.degree,
            ncols: wiggle.ncols,
        });
    }
    Ok(PreparedSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative_exit,
        time_penalties,
        time_nullspace_dims,
        timewiggle_build,
        timewiggle_block,
    })
}
