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
    /// Each time penalty's natural `log λ` REML seed: the log ratio of the mean Gram
    /// diagonal of the columns it penalizes (the exit time basis, or for a time
    /// wiggle the warp's Jacobian at the baseline predictor) to the penalty's mean
    /// diagonal. `None` when the time basis carries no penalty.
    pub time_initial_log_lambdas: Option<Array1<f64>>,
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
        // A Linear target has no parametric offset for any likelihood: the time
        // I-spline IS the baseline. Marginal slope used to offset a Linear target
        // by a data-seeded Weibull (scale = mean positive exit, shape 1; gam#797)
        // to start the `-d·log(qd1)` barrier interior. An offset is part of the
        // model, not a starting point: the time coefficients are held `≥ 0` and
        // both time penalties shrink them toward zero, so the fitted index was
        // that Weibull plus a level, and REML railed both time penalties to reach
        // it. On a landmarked cohort with a 3 %/y hazard it predicted a third of
        // the half-year risk (gnomon#2336). The interior start now comes from the
        // coefficients — the #2627 pilot warm start — so the represented class
        // (every nondecreasing function of log time, with linear tails) and its
        // shrinkage target do not depend on a seed.
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                age_entry,
                age_exit,
                baseline_cfg,
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
    // Each penalty is seeded against the columns it acts on. The time design's
    // wiggle tail is a zero placeholder, because the family evaluates the warp
    // dynamically, so a wiggle penalty is seeded against the warp's Jacobian at
    // the baseline predictor, B(h₀(t_exit)). The marginal-slope family seeds its
    // time block through the same two functions (#3061).
    let time_initial_log_lambdas = if time_penalties.is_empty() {
        None
    } else {
        let acting_exit = crate::survival::marginal_slope::time_block_acting_exit_design(
            &time_design_exit,
            eta_offset_exit.view(),
            timewiggle_block.as_ref(),
        )?;
        Some(Array1::from_vec(
            crate::survival::marginal_slope::time_block_log_lambda_seeds(
                &acting_exit,
                &time_penalties,
                timewiggle_block.as_ref().map_or(0, |wiggle| wiggle.ncols),
            )?,
        ))
    };
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
        time_initial_log_lambdas,
    })
}
