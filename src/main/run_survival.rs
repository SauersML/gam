use super::*;

pub(crate) fn survival_working_reml_score(state: &gam::pirls::WorkingState) -> f64 {
    0.5 * (state.deviance + state.penalty_term)
}

pub(crate) fn survival_time_initial_log_lambdas(
    time_build: &SurvivalTimeBuildOutput,
    penalties: &[Array2<f64>],
) -> Option<Array1<f64>> {
    if penalties.is_empty() {
        None
    } else {
        let lambda0 = time_build.smooth_lambda.unwrap_or(1e-2).max(1e-12).ln();
        Some(Array1::from_elem(penalties.len(), lambda0))
    }
}

pub(crate) fn build_survival_time_initial_beta(
    likelihood_mode: SurvivalLikelihoodMode,
    exact_derivative_guard: f64,
    prepared: &PreparedSurvivalTimeStack,
) -> Array1<f64> {
    let p = prepared.time_design_exit.ncols();
    // The marginal-slope time block runs on `TimeBlockMonotonicity::StructuralISpline`:
    // q(t) is monotone iff γ ≥ 0, and `add_survival_time_derivative_guard_offset`
    // has already absorbed `guard·t` into the offsets — so γ = 0 is the
    // canonical structurally-feasible seed (q reduces to the guard-linear
    // baseline, q'(t) = guard exactly). Projecting onto the row-wise
    // `D γ + o ≥ guard` constraint set would produce a γ that violates
    // γ ≥ 0 because that projection has no nonnegativity awareness; the
    // safe seed is the zero vector.
    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        return Array1::zeros(p);
    }
    let time_initial_constraints = if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        gam::pirls::LinearInequalityConstraints::new(
            prepared.time_design_derivative_exit.to_dense(),
            prepared
                .derivative_offset_exit
                .mapv(|offset| exact_derivative_guard - offset),
        )
        .ok()
    } else {
        None
    };
    time_initial_constraints.as_ref().map_or_else(
        || Array1::zeros(p),
        |constraints| {
            // `beta0` is `None`, so the projection starts at the length-`p`
            // origin and cannot hit a beta0/dim mismatch; the constraints come
            // from `LinearInequalityConstraints::new` with matching A/b shapes.
            // On any unexpected error fall back to the zero seed described
            // above rather than panic out of this seed builder.
            project_onto_linear_constraints(p, constraints, None)
                .unwrap_or_else(|_| Array1::zeros(p))
        },
    )
}

/// Recover the fitted Weibull `(scale, shape)` baseline from the anchor-CENTERED
/// linear `[1, log t]` time-basis coefficients.
///
/// The fit centers the time basis at the survival time anchor
/// (`center_survival_time_designs_at_anchor`), which zeroes the constant column,
/// so the constant-column coefficient `beta[0]` is UNIDENTIFIED (left at its
/// stale seed). The identified baseline the model actually carries is
/// `eta(t) = beta[1] * (log t - log anchor)`, exactly the Weibull form
/// `eta(t) = shape * (log t - log scale)` with `shape = beta[1]` and
/// `scale = anchor`. Reconstructing `scale` from `beta[0]` (the old
/// `exp(-beta[0]/shape)`) reads the stale constant column and produces a wrong
/// scale, so any consumer that rebuilds `H0(t) = (t/scale)^shape` from the saved
/// scale (e.g. competing-risks CIF) is misled. Recover `scale` from the
/// identified anchor instead (issue #899).
pub(crate) fn fitted_weibull_baseline_from_linear_time_beta(
    beta: &Array1<f64>,
    anchor: f64,
) -> Option<(f64, f64)> {
    if beta.len() < 2 {
        return None;
    }
    let shape = beta[1];
    if !shape.is_finite() || shape <= 0.0 {
        return None;
    }
    if !anchor.is_finite() || anchor <= 0.0 {
        return None;
    }
    let scale = anchor;
    Some((scale, shape))
}

pub(crate) fn baseline_timewiggle_is_present(model: &SavedModel) -> bool {
    model.has_baseline_time_wiggle()
}

/// Inner-PIRLS options shared by both survival-baseline fit sites (the
/// per-candidate trial fit and the final baseline fit). Centralised so the two
/// call sites cannot drift in their convergence policy: a generous 400-iter /
/// 40-halving budget with a 1e-6 coefficient-change tolerance and a 1e-12
/// step-size floor, matching the survival baseline's BFGS envelope solver.
pub(crate) fn survival_baseline_pirls_options() -> gam::pirls::WorkingModelPirlsOptions {
    gam::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        adaptive_kkt_tolerance: None,
        max_step_halving: 40,
        min_step_size: 1e-12,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
        geodesic_acceleration: false,
        arrow_schur: None,
    }
}

pub(crate) fn run_survival(args: SurvivalArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let survival_total_steps = if args.out.is_some() { 5 } else { 4 };
    progress.start_workflow("Survival Fit", survival_total_steps);
    let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
    let formula = format!("{response_expr} ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    progress.set_stage("fit", "loading survival data");
    let requested_columns = required_columns_for_survival(&args, &parsed)?;
    let ds = load_dataset_projected(&args.data, &requested_columns)?;
    progress.advance_workflow(1);
    let col_map = ds.column_map();

    // `entry_col == None` is the right-censored shorthand `Surv(time, event)`:
    // entry times are synthesized as zero, no column lookup required.
    let entry_col: Option<usize> = args
        .entry
        .as_deref()
        .map(|name| resolve_role_col(&col_map, name, "entry"))
        .transpose()?;
    let exit_col = resolve_role_col(&col_map, &args.exit, "exit")?;
    let event_col = resolve_role_col(&col_map, &args.event, "event")?;

    let n = ds.values.nrows();
    if n == 0 {
        return Err("survival dataset has no rows".to_string());
    }
    let formula_surv = parsed.survivalspec.clone();
    let formula_link = parsed.linkspec.clone();
    let formula_linkwiggle = parsed.linkwiggle.clone();
    let formula_timewiggle = parsed.timewiggle.clone();
    let effectivespec = formula_surv
        .as_ref()
        .and_then(|s| s.spec.clone())
        .unwrap_or_else(|| "net".to_string());
    let effective_survival_distribution = formula_surv
        .as_ref()
        .and_then(|s| s.survival_distribution.clone())
        .unwrap_or_else(|| "gaussian".to_string());
    let mut effective_args = args.clone();
    if let Some(ls) = formula_link.as_ref() {
        effective_args.link = Some(ls.link.clone());
        effective_args.mixture_rho = ls.mixture_rho.clone();
        effective_args.sas_init = ls.sas_init.clone();
        effective_args.beta_logistic_init = ls.beta_logistic_init.clone();
    }
    effective_args.survival_distribution = effective_survival_distribution;
    let effective_config = fit_config_from_survival_args(&effective_args)?;
    let predict_noise_formula = effective_config
        .noise_formula
        .as_deref()
        .map(|raw| parse_matching_auxiliary_formula(raw, &response_expr, "--predict-noise"))
        .transpose()?;
    if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        validate_auxiliary_formula_controls(parsed_noise, "--predict-noise")?;
    }

    let survival_link_choice = match effective_config.link.as_deref() {
        Some(raw)
            if matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "loglog" | "cauchit"
            ) =>
        {
            None
        }
        raw => parse_link_choice(raw, false)?,
    };
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), survival_link_choice.as_ref());
    let effective_timewiggle = formula_timewiggle.clone();
    let learn_timewiggle = effective_timewiggle.is_some();

    let survivalspec = match effectivespec.to_ascii_lowercase().as_str() {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err(
                "survival spec 'crude' is not supported by the one-hazard fitter; use survmodel(spec=net) and compute crude risk from separate cause-specific hazards"
                    .to_string(),
            );
        }
        other => {
            return Err(format!(
                "unsupported survmodel(spec='{other}'); only spec=net is accepted by the one-hazard fitter"
            ));
        }
    };
    let requested_likelihood_mode =
        parse_survival_likelihood_mode(&effective_config.survival_likelihood)?;
    let likelihood_mode = if predict_noise_formula.is_some() {
        match requested_likelihood_mode {
            SurvivalLikelihoodMode::Weibull => {
                return Err(
                    "--predict-noise with Surv(...) requires survival location-scale; remove --survival-likelihood weibull"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::MarginalSlope => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood marginal-slope"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::Latent => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood latent"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::LatentBinary => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood latent-binary"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::LocationScale => {
                SurvivalLikelihoodMode::LocationScale
            }
        }
    } else {
        requested_likelihood_mode
    };
    // linkwiggle(...) is a nonparametric anchored correction to the inverse
    // link g^{-1}: eta -> mu. It is defined only for modes that expose such a
    // map. LocationScale uses a standard inverse link for the residual
    // distribution (Gaussian/SAS/BetaLogistic/Mixture) that linkwiggle can
    // correct; MarginalSlope routes it into its anchored internal link-
    // deviation/score-warp blocks (handled below). The remaining survival
    // modes — Transformation, Weibull, Latent, LatentBinary — parameterize
    // eta = log H(t|x) directly (Royston-Parmar) and therefore have no
    // separate eta -> mu inverse link to wiggle. Reject rather than silently
    // drop, so the user's published feature is not quietly ignored.
    if effective_linkwiggle.is_some()
        && !matches!(
            likelihood_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(format!(
            "linkwiggle(...) is not defined for --survival-likelihood={}; it corrects the inverse link eta -> mu, but Royston-Parmar parameterizes eta = log H(t|x) directly with no such map. Use --survival-likelihood=location-scale for a linkwiggle-corrected residual distribution, or --survival-likelihood=marginal-slope to route linkwiggle(...) into the anchored internal link-deviation block",
            survival_likelihood_modename(likelihood_mode),
        ));
    }
    if matches!(
        survival_link_choice.as_ref().map(|choice| &choice.mode),
        Some(LinkMode::Flexible)
    ) && likelihood_mode != SurvivalLikelihoodMode::LocationScale
    {
        return Err(
            "survival flexible(...) links are supported only with --survival-likelihood=location-scale"
                .to_string(),
        );
    }
    parse_survival_distribution(&effective_config.survival_distribution)?;
    let survival_inverse_link = crate::config_resolve::parse_survival_inverse_link(
        crate::config_resolve::SurvivalInverseLinkInput {
            link: effective_config.link.as_deref(),
            mixture_rho: effective_args.mixture_rho.as_deref(),
            sas_init: effective_args.sas_init.as_deref(),
            beta_logistic_init: effective_args.beta_logistic_init.as_deref(),
            survival_distribution: &effective_config.survival_distribution,
        },
    )?;
    if effective_linkwiggle.is_some() && likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        require_inverse_link_supports_joint_wiggle(&survival_inverse_link, "linkwiggle(...)")?;
    }
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
        if !matches!(
            effective_config
                .baseline_target
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "linear" | "weibull"
        ) {
            return Err(
                "--survival-likelihood weibull supports only --baseline-target=linear or --baseline-target=weibull without --learn-timewiggle"
                    .to_string(),
            );
        }
        if effective_config.baseline_rate.is_some() || effective_config.baseline_makeham.is_some() {
            return Err(
                "--survival-likelihood weibull does not use --baseline-rate or --baseline-makeham"
                    .to_string(),
            );
        }
    }
    let baseline_target_raw = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => effective_config.baseline_target.clone(),
        SurvivalLikelihoodMode::Weibull if learn_timewiggle => "weibull".to_string(),
        SurvivalLikelihoodMode::Weibull => "linear".to_string(),
    };
    let time_basis_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => {
            if learn_timewiggle {
                // Parametric baseline + timewiggle owns the full time structure.
                SurvivalTimeBasisConfig::None
            } else {
                parse_survival_time_basis_config(
                    &effective_config.time_basis,
                    effective_config.time_degree,
                    effective_config.time_num_internal_knots,
                    effective_config.time_smooth_lambda,
                )?
            }
        }
        SurvivalLikelihoodMode::Weibull => {
            if learn_timewiggle {
                SurvivalTimeBasisConfig::None
            } else {
                SurvivalTimeBasisConfig::Linear
            }
        }
    };
    let mut inference_notes = Vec::new();
    progress.set_stage("fit", "building survival design matrices");
    // Survival marginal-slope formulas may reference the literal placeholder
    // `z` to bind to the auxiliary score supplied via --z-column. Alias `z`
    // to the actual `z_column` index in a local copy of `col_map` so
    // build_termspec resolves it without the user renaming their data column.
    let col_map_local = if matches!(likelihood_mode, SurvivalLikelihoodMode::MarginalSlope) {
        effective_config
            .z_column
            .as_deref()
            .map(|z_name| column_map_with_alias(&col_map, "z", z_name))
            .unwrap_or_else(|| col_map.clone())
    } else {
        col_map.clone()
    };
    let col_map_for_termspec: &HashMap<String, usize> = &col_map_local;
    let mut termspec = build_termspec(
        &parsed.terms,
        &ds,
        col_map_for_termspec,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    if effective_config.scale_dimensions {
        enable_scale_dimensions(&mut termspec);
    }
    let log_sigmaspec = if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        let mut spec = build_termspec(
            &parsed_noise.terms,
            &ds,
            col_map_for_termspec,
            &mut inference_notes,
            &gam::ResourcePolicy::default_library(),
        )?;
        if effective_config.scale_dimensions {
            enable_scale_dimensions(&mut spec);
        }
        spec
    } else {
        // No `--predict-noise` ⇒ default to an empty log-σ spec (constant
        // log-σ baseline owned by the family adapter). Cloning the mean
        // `termspec` here duplicated every threshold term onto log-σ; for a
        // smooth `s(x)` on the mean the canonical-gauge identifiability
        // audit then dropped every aliased log-σ column (time > threshold >
        // log_sigma priorities, #366) so the solver's per-block spec had
        // width 0 while the family kept x_log_sigma at the smooth width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then failed "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    let cov_design = build_term_collection_design(ds.values.view(), &termspec)
        .map_err(|e| format!("failed to build survival term collection design: {e}"))?;
    let frozen_termspec =
        freeze_term_collection_from_design(&termspec, &cov_design).map_err(|e| e.to_string())?;

    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = resolve_weight_column(&ds, &col_map, effective_config.weight_column.as_deref())?;
    let threshold_offset =
        resolve_offset_column(&ds, &col_map, effective_config.offset_column.as_deref())?;
    let log_sigma_offset = resolve_offset_column(
        &ds,
        &col_map,
        effective_config.noise_offset_column.as_deref(),
    )?;

    for i in 0..n {
        let entry_val = entry_col.map_or(0.0, |idx| ds.values[[i, idx]]);
        let (t0, t1) = normalize_survival_time_pair(entry_val, ds.values[[i, exit_col]], i)?;
        let ev = ds.values[[i, event_col]];
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = survival_event_code_from_value(ev, i)?;
    }
    let cause_count =
        gam::survival::cause_count_from_event_codes(event_target.view()).into_cli_result()?;
    if cause_count > 1
        && !matches!(
            likelihood_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(format!(
            "cause-specific competing risks with {cause_count} causes are currently supported for --survival-likelihood transformation and weibull"
        ));
    }
    // All-censored (zero-event) fittability gate. The survival likelihood has
    // no event score when no row marks a target event, so the inner/outer
    // solve cannot identify the hazard. The single-hazard engine's structural
    // checks (in `WorkingModelSurvival::validate_common_inputs`) intentionally
    // permit construction on censored fixtures so the engine's update_state
    // / monotonicity-collocation contracts can be unit-tested in isolation;
    // production fit dispatchers own the fittability gate.
    if !event_target.iter().any(|&code| code > 0) {
        return Err(
            "survival fit requires at least one target event; all rows are censored, so the likelihood has no event score and cannot identify the hazard"
                .to_string(),
        );
    }
    let mut baseline_cfg = initial_survival_baseline_config_for_fit(
        &baseline_target_raw,
        effective_config.baseline_scale,
        effective_config.baseline_shape,
        effective_config.baseline_rate,
        effective_config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent survival/binary likelihoods require a non-linear scalar baseline target; use --baseline-target weibull, gompertz, or gompertz-makeham"
                .to_string(),
        );
    }
    let weibull_builtin_beta_seed =
        if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
            let scale = effective_config
                .baseline_scale
                .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
            let shape = effective_config.baseline_shape.unwrap_or(1.0);
            Some(array![-shape * scale.ln(), shape])
        } else {
            None
        };
    if learn_timewiggle && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle(...) requires a non-linear scalar survival baseline target; use --baseline-target weibull|gompertz|gompertz-makeham, or combine it with --survival-likelihood weibull"
                .to_string(),
        );
    }
    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && learn_timewiggle
    {
        return Err(
            "timewiggle(...) is not implemented for latent survival/binary likelihoods; use the learned time basis and scalar baseline target directly"
                .to_string(),
        );
    }
    // Marginal-slope centers the baseline-hazard I-spline at a robust interior
    // exit-scale time (median exit) instead of the earliest entry age; under
    // left truncation the earliest entry is a positive left-tail point and
    // centering there inflates the unpenalized linear-trend column, blowing up
    // the time-block seed score so REML rejects every seed (issue #751). The
    // location-scale path keeps the earliest-entry anchor. An explicit
    // `--survival-time-anchor` is honored by both.
    let time_anchor = if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        resolve_survival_marginal_slope_time_anchor_value(
            &age_entry,
            &age_exit,
            args.survival_time_anchor,
        )?
    } else {
        resolve_survival_time_anchor_value(&age_entry, args.survival_time_anchor)?
    };
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(likelihood_mode);
    if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        inference_notes.push(format!(
            "survival time block enforces structural monotonicity with derivative floor {:.3e}; boundary solutions may clamp at that floor",
            exact_derivative_guard
        ));
    }
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_basis_cfg,
        Some((
            effective_config.time_num_internal_knots,
            effective_config.ridge_lambda,
        )),
    )?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    if likelihood_mode != SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
        require_structural_survival_time_basis(&time_build.basisname, "survival fitting")?;
    }
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    progress.advance_workflow(2);
    print_inference_summary(&inference_notes);

    if likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let threshold_template = if let Some(tk) = effective_config.threshold_time_k {
            cli_err!(
                "[survival location-scale] building time-varying threshold: k={tk}, degree={}",
                effective_config.threshold_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                tk,
                effective_config.threshold_time_degree,
                "threshold",
            )?
        } else {
            SurvivalCovariateTermBlockTemplate::Static
        };

        let log_sigma_template = if let Some(sk) = effective_config.sigma_time_k {
            cli_err!(
                "[survival location-scale] building time-varying sigma: k={sk}, degree={}",
                effective_config.sigma_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                sk,
                effective_config.sigma_time_degree,
                "sigma",
            )?
        } else {
            SurvivalCovariateTermBlockTemplate::Static
        };

        let kappa_options = {
            let mut opts = SpatialLengthScaleOptimizationOptions::default();
            opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
            opts
        };
        let optimize_inverse_link = match &survival_inverse_link {
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
            InverseLink::Mixture(state) => !state.rho.is_empty(),
            InverseLink::LatentCLogLog(_) | InverseLink::Standard(_) => false,
        };
        let buildtermspec = |prepared: &PreparedSurvivalTimeStack,
                             inverse_link: InverseLink|
         -> SurvivalLocationScaleTermSpec {
            let time_initial_beta =
                build_survival_time_initial_beta(likelihood_mode, exact_derivative_guard, prepared);
            SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.mapv(f64::from),
                weights: weights.clone(),
                inverse_link,
                derivative_guard: exact_derivative_guard,
                max_iter: 400,
                tol: 1e-6,
                time_block: TimeBlockInput {
                    design_entry: prepared.time_design_entry.clone(),
                    design_exit: prepared.time_design_exit.clone(),
                    design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                    offset_entry: prepared.eta_offset_entry.clone(),
                    offset_exit: prepared.eta_offset_exit.clone(),
                    derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                    time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                    penalties: prepared.time_penalties.clone(),
                    nullspace_dims: prepared.time_nullspace_dims.clone(),
                    initial_log_lambdas: survival_time_initial_log_lambdas(
                        &time_build,
                        &prepared.time_penalties,
                    ),
                    initial_beta: Some(time_initial_beta.clone()),
                },
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block.clone(),
                linkwiggle_block: None,
                initial_threshold_log_lambdas: None,
                initial_log_sigma_log_lambdas: None,
                cache_session: None,
                cache_mirror_sessions: Vec::new(),
            }
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            // BFGS on the analytic θ-gradient from
            // `SurvivalLocationScaleTermFitResult::baseline_offset_residuals`
            // contracted against `baseline_offset_theta_partials` (η-channel)
            // or `marginal_slope_baseline_offset_theta_partials` (probit
            // channel), depending on which baseline parametrization the
            // location-scale family is consuming for this inverse link. The
            // envelope-theorem argument that justifies this contraction is
            // documented at `baseline_chain_rule_gradient` and at the
            // analogous marginal-slope dispatch.
            let probit_channel =
                location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
            baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
                &baseline_cfg,
                "survival location-scale baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        SurvivalLikelihoodMode::LocationScale,
                        Some(&survival_inverse_link),
                        time_anchor,
                        exact_derivative_guard,
                        &time_build,
                        effective_timewiggle.as_ref(),
                        None,
                    )?;
                    let fit = match fit_model(FitRequest::SurvivalLocationScale(
                        SurvivalLocationScaleFitRequest {
                            data: ds.values.view(),
                            spec: buildtermspec(&prepared, survival_inverse_link.clone()),
                            wiggle: effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
                                degree: cfg.degree,
                                num_internal_knots: cfg.num_internal_knots,
                                penalty_orders: cfg.penalty_orders,
                                double_penalty: cfg.double_penalty,
                            }),
                            kappa_options: kappa_options.clone(),
                            optimize_inverse_link,
                            cache_session: None,
                        },
                    )) {
                        Ok(FitResult::SurvivalLocationScale(result)) => result,
                        Ok(_) => {
                            return Err(
                                "internal survival location-scale workflow returned the wrong result variant"
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            return Err(format!("survival location-scale fit failed: {e}"));
                        }
                    };
                    let residuals = &fit.fit.baseline_offset_residuals;
                    let gradient = if probit_channel {
                        marginal_slope_baseline_chain_rule_gradient(
                            age_entry.view(),
                            age_exit.view(),
                            candidate,
                            residuals,
                        )?
                    } else {
                        baseline_chain_rule_gradient(
                            age_entry.view(),
                            age_exit.view(),
                            // No interval channel on this path; `residuals.right`
                            // is all-zero so `age_exit` is an unconsulted placeholder.
                            age_exit.view(),
                            candidate,
                            residuals,
                        )?
                    }
                    .ok_or_else(|| {
                        "survival location-scale baseline unexpectedly has no theta gradient"
                            .to_string()
                    })?;
                    // The envelope residual contraction gives the θ-gradient
                    // of the profile penalized NLL −ℓ + ½βᵀSβ at converged
                    // (β̂, ρ̂). REML/LAML log-determinant corrections have
                    // additional θ-dependence through H(β̂, θ), so optimizing
                    // `reml_score` against this gradient would mismatch the
                    // cost. Use the matching profile-NLL cost.
                    let profile_cost =
                        -fit.fit.fit.log_likelihood + 0.5 * fit.fit.fit.stable_penalty_term;
                    if !profile_cost.is_finite() {
                        return Err(format!(
                            "survival location-scale baseline: non-finite profile cost \
                             (log_likelihood={}, stable_penalty_term={}, cost={})",
                            fit.fit.fit.log_likelihood,
                            fit.fit.fit.stable_penalty_term,
                            profile_cost
                        ));
                    }
                    Ok((profile_cost, gradient))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            SurvivalLikelihoodMode::LocationScale,
            Some(&survival_inverse_link),
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let time_design_exit = prepared.time_design_exit.clone();
        progress.set_stage("fit", "running survival location-scale optimization");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] survival-location-scale fit start n={}",
            ds.values.nrows()
        );
        let fit = match fit_model(FitRequest::SurvivalLocationScale(
            SurvivalLocationScaleFitRequest {
                data: ds.values.view(),
                spec: buildtermspec(&prepared, survival_inverse_link.clone()),
                wiggle: effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
                    degree: cfg.degree,
                    num_internal_knots: cfg.num_internal_knots,
                    penalty_orders: cfg.penalty_orders,
                    double_penalty: cfg.double_penalty,
                }),
                kappa_options: kappa_options.clone(),
                optimize_inverse_link,
                cache_session: None,
            },
        )) {
            Ok(FitResult::SurvivalLocationScale(result)) => {
                log::info!(
                    "[PHASE] survival-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                return Err(
                    "internal survival location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                return Err(format!("survival location-scale fit failed: {e}"));
            }
        };
        let fitted_inverse_link = fit.inverse_link.clone();
        cli_out!(
            "survival location-scale fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            fit.fit.fit.pirls_status.label(),
            fit.fit.fit.outer_iterations,
            fit.fit.fit.log_likelihood,
            fit.fit.fit.reml_score
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival model");
            let mut fit_result = compact_saved_survival_location_scale_fit_result(
                &fit.fit.fit,
                &fitted_inverse_link,
            )?;
            fit_result.artifacts.survival_link_wiggle_knots = fit.wiggle_knots.clone();
            fit_result.artifacts.survival_link_wiggle_degree = fit.wiggle_degree;
            // Source-specific work: extract the baseline-timewiggle block (from
            // the first block-state beta), re-encode the survival noise
            // scale-deviation transform, and freeze the threshold / log-sigma
            // term specs. The shared core then assembles the canonical payload
            // exactly as the FFI does.
            let baseline_timewiggle = prepared.timewiggle_build.as_ref().map(|w| {
                let p_base = time_build.x_exit_time.ncols();
                let beta = fit
                    .fit
                    .fit
                    .block_states
                    .first()
                    .map(|state| state.beta.slice(s![p_base..]).to_vec())
                    .unwrap_or_default();
                SurvivalTimewiggle {
                    degree: w.degree,
                    knots: w.knots.to_vec(),
                    penalty_orders: effective_timewiggle
                        .as_ref()
                        .map(|cfg| cfg.penalty_orders.clone()),
                    double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                    beta: SurvivalTimewiggleBeta::Single(beta),
                }
            });
            let survival_primary_design = DesignMatrix::hstack(vec![
                time_design_exit.clone(),
                fit.fit.threshold_design.design.clone(),
            ])?;
            let survival_noise_transform = build_scale_deviation_transform_design(
                &survival_primary_design,
                &fit.fit.log_sigma_design.design,
                &weights,
                infer_non_intercept_start_design(&fit.fit.log_sigma_design.design, &weights)?,
            )
            .map_err(|e| format!("failed to encode survival noise transform: {e}"))?;
            let resolved_thresholdspec = freeze_term_collection_from_design(
                &fit.fit.resolved_thresholdspec,
                &fit.fit.threshold_design,
            )
            .map_err(|e| e.to_string())?;
            let resolved_log_sigmaspec = freeze_term_collection_from_design(
                &fit.fit.resolved_log_sigmaspec,
                &fit.fit.log_sigma_design,
            )
            .map_err(|e| e.to_string())?;
            let payload = assemble_survival_location_scale_payload(
                SurvivalLocationScaleInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result,
                    fitted_inverse_link: fitted_inverse_link.clone(),
                    linkwiggle_degree: fit.wiggle_degree,
                    linkwiggle_knots: fit.wiggle_knots.as_ref().map(|k| k.to_vec()),
                    beta_link_wiggle: fit.fit.fit.beta_link_wiggle().as_ref().map(|b| b.to_vec()),
                    baseline_timewiggle,
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    formula_noise: predict_noise_formula
                        .as_ref()
                        .map(|(noise_formula, _)| noise_formula.clone()),
                    survival_beta_time: fit.fit.fit.beta_time().to_vec(),
                    survival_beta_threshold: fit.fit.fit.beta_threshold().to_vec(),
                    survival_beta_log_sigma: fit.fit.fit.beta_log_sigma().to_vec(),
                    noise_transform: &survival_noise_transform,
                    resolved_thresholdspec,
                    resolved_log_sigmaspec,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival fit complete");
        return Ok(());
    }

    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let survival_marginal_slope_base_link = resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "survival marginal-slope",
        )?;
        let logslope_formula_raw =
            effective_config
                .logslope_formula
                .as_deref()
                .ok_or_else(|| {
                    "--logslope-formula is required with --survival-likelihood marginal-slope"
                        .to_string()
                })?;
        let z_column_name = effective_config.z_column.as_ref().ok_or_else(|| {
            "--z-column is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
        let (logslope_formula, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula_raw,
            &response_expr,
            "--logslope-formula",
        )?;
        if parsed_logslope.linkspec.is_some() {
            return Err(
                "link(...) is not supported in --logslope-formula for the survival marginal-slope family"
                    .to_string(),
            );
        }
        validate_marginal_slope_z_column_exclusion(
            &parsed,
            &parsed_logslope,
            z_column_name,
            "survival marginal-slope",
            "--logslope-formula",
        )?;
        let mut logslopespec = build_termspec(
            &parsed_logslope.terms,
            &ds,
            col_map_for_termspec,
            &mut inference_notes,
            &gam::ResourcePolicy::default_library(),
        )?;
        if effective_config.scale_dimensions {
            enable_scale_dimensions(&mut logslopespec);
        }

        let z_col = resolve_role_col(&col_map, z_column_name, "z")?;
        let z = ds.values.column(z_col).to_owned();

        let routed_deviations = route_marginal_slope_deviation_blocks(
            parsed.linkwiggle.as_ref(),
            parsed_logslope.linkwiggle.as_ref(),
        )?;
        let routed_link_dev = routed_deviations.link_dev;
        let routed_score_warp = routed_deviations.score_warp;
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes main-formula linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if parsed_logslope.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes --logslope-formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if routed_link_dev.is_none() && routed_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }

        let frailty = fixed_gaussian_shift_frailty_from_spec(
            &fit_frailty_spec_from_survival_args(&args, "survival marginal-slope")?,
            "survival marginal-slope",
        )?;
        let kappa_options = {
            let mut opts = SpatialLengthScaleOptimizationOptions::default();
            opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
            opts
        };
        let mut options = gam::families::custom_family::BlockwiseFitOptions::default();
        options.compute_covariance = true;
        let buildspec = |prepared: &PreparedSurvivalTimeStack| {
            SurvivalMarginalSlopeTermSpec {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: event_target.mapv(f64::from),
            weights: weights.clone(),
            z: z.clone().insert_axis(Axis(1)),
            base_link: survival_marginal_slope_base_link.clone(),
            marginalspec: termspec.clone(),
            marginal_offset: threshold_offset.clone(),
            frailty: frailty.clone(),
            derivative_guard: exact_derivative_guard,
            time_block: TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                // The marginal-slope time block runs on `SurvivalTimeBasisConfig::ISpline`
                // (the survival CLI default and the only basis `parse_survival_time_basis_config`
                // accepts under `require_structural_survival_time_basis`), so `q(t)` is
                // structurally monotone whenever `γ ≥ 0`. Declaring `StructuralISpline`
                // tells the family to skip the row-wise `D β + o ≥ guard` constraint
                // generator (vacuous on this basis) and rely on the γ-cone coordinate
                // bound instead. See `src/families/ispline_base_time.rs` for the why.
                time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::StructuralISpline,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: survival_time_initial_log_lambdas(
                    &time_build,
                    &prepared.time_penalties,
                ),
                initial_beta: Some(build_survival_time_initial_beta(
                    likelihood_mode,
                    exact_derivative_guard,
                    prepared,
                )),
            },
            timewiggle_block: prepared.timewiggle_block.clone(),
            logslopespec: logslopespec.clone(),
            logslopespecs: None,
            logslope_offset: log_sigma_offset.clone(),
            score_warp: routed_score_warp.clone(),
            link_dev: routed_link_dev.clone(),
            latent_z_policy: LatentZPolicy::default(),
            // CLI survival marginal-slope fits directly from a raw `--z-column`
            // with no in-process CTN Stage-1 chain to cross-fit, so the
            // score-influence projection is inactive (#461 §5).
            score_influence_jacobian: None,
        }
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            baseline_cfg = optimize_survival_baseline_config_with_gradient(
                &baseline_cfg,
                "survival marginal-slope baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        SurvivalLikelihoodMode::MarginalSlope,
                        None,
                        time_anchor,
                        exact_derivative_guard,
                        &time_build,
                        effective_timewiggle.as_ref(),
                        None,
                    )?;
                    // Disable kappa optimization during baseline search so each
                    // candidate evaluation is cheap (inner solve only, no spatial
                    // length-scale outer loop).
                    let mut baseline_kappa = SpatialLengthScaleOptimizationOptions::default();
                    baseline_kappa.enabled = false;
                    let mut baseline_options = options.clone();
                    baseline_options.compute_covariance = false;
                    let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
                        SurvivalMarginalSlopeFitRequest {
                            data: ds.values.view(),
                            spec: buildspec(&prepared),
                            options: baseline_options,
                            kappa_options: baseline_kappa,
                        },
                    )) {
                        Ok(FitResult::SurvivalMarginalSlope(result)) => result,
                        Ok(_) => {
                            return Err(
                                "internal survival marginal-slope workflow returned the wrong result variant"
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            return Err(format!("survival marginal-slope fit failed: {e}"));
                        }
                    };
                    let gradient = marginal_slope_baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        &fit.baseline_offset_residuals,
                    )?
                    .ok_or_else(|| {
                        "survival marginal-slope baseline unexpectedly has no theta gradient"
                            .to_string()
                    })?;
                    let hessian = marginal_slope_baseline_chain_rule_hessian(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        &fit.baseline_offset_residuals,
                        &fit.baseline_offset_curvatures,
                    )?
                    .ok_or_else(|| {
                        "survival marginal-slope baseline unexpectedly has no theta Hessian"
                            .to_string()
                    })?;
                    Ok((fit.fit.reml_score, gradient, hessian))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            SurvivalLikelihoodMode::MarginalSlope,
            None,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        progress.set_stage("fit", "running survival marginal-slope optimization");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] survival-margslope fit start n={}",
            ds.values.nrows()
        );
        let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
            SurvivalMarginalSlopeFitRequest {
                data: ds.values.view(),
                spec: buildspec(&prepared),
                options: options.clone(),
                kappa_options,
            },
        )) {
            Ok(FitResult::SurvivalMarginalSlope(result)) => {
                log::info!(
                    "[PHASE] survival-margslope fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                return Err(
                    "internal survival marginal-slope workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                return Err(format!("survival marginal-slope fit failed: {e}"));
            }
        };
        cli_out!(
            "survival marginal-slope fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e} | baseline_slope={:.4}",
            fit.fit.pirls_status.label(),
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score,
            fit.baseline_slope,
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival marginal-slope model");
            let save_frailty = match (&frailty, fit.gaussian_frailty_sd) {
                (
                    gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                        sigma_fixed: None,
                    },
                    Some(learned),
                ) => gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                    sigma_fixed: Some(learned),
                },
                _ => frailty,
            };
            // Source-specific work: freeze the term collections from their
            // designs and snapshot the time basis. The semantic payload is
            // assembled by the same shared core the FFI uses.
            let resolved_marginalspec = freeze_term_collection_from_design(
                &fit.marginalspec_resolved,
                &fit.marginal_design,
            )
            .map_err(|e| e.to_string())?;
            let resolved_logslopespec = freeze_term_collection_from_design(
                &fit.logslopespec_resolved,
                &fit.logslope_design,
            )
            .map_err(|e| e.to_string())?;
            let payload = assemble_survival_marginal_slope_payload(
                SurvivalMarginalSlopeInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.fit.clone(),
                    frailty: save_frailty,
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    resolved_marginalspec,
                    resolved_logslopespec,
                    logslope_formula,
                    z_column: z_column_name.clone(),
                    latent_z_normalization: SavedLatentZNormalization {
                        mean: fit.z_normalization.mean,
                        sd: fit.z_normalization.sd,
                    },
                    baseline_logslope: fit.baseline_slope,
                    score_warp_runtime: fit.score_warp_runtime.as_ref(),
                    link_dev_runtime: fit.link_dev_runtime.as_ref(),
                    influence_absorber_width: fit.influence_absorber_width,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival marginal-slope fit complete");
        return Ok(());
    }

    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        if parsed.linkspec.is_some() {
            return Err(
                "link(...) is not implemented for latent survival/binary likelihoods".to_string(),
            );
        }
        let latent_context = if likelihood_mode == SurvivalLikelihoodMode::Latent {
            "latent survival"
        } else {
            "latent binary"
        };
        let frailty = fit_frailty_spec_from_survival_args(&args, latent_context)?;
        let latent_loading = latent_hazard_loading(&frailty, latent_context)?;
        let latent_derivative_guard = survival_derivative_guard_for_likelihood(likelihood_mode);
        let options = gam::families::custom_family::BlockwiseFitOptions {
            compute_covariance: false,
            ..Default::default()
        };
        let build_time_block = |prepared: &PreparedSurvivalTimeStack| {
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas =
                survival_time_initial_log_lambdas(&time_build, &prepared.time_penalties);
            TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            }
        };
        let build_survival_request =
            |prepared: PreparedSurvivalTimeStack| LatentSurvivalFitRequest {
                data: ds.values.view(),
                spec: gam::families::latent_survival::LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_target.clone(),
                    weights: weights.clone(),
                    derivative_guard: latent_derivative_guard,
                    time_block: build_time_block(&prepared),
                    time_design_right: None,
                    time_offset_right: None,
                    unloaded_mass_entry: prepared.unloaded_mass_entry.clone(),
                    unloaded_mass_exit: prepared.unloaded_mass_exit.clone(),
                    unloaded_mass_right: Array1::zeros(0),
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit.clone(),
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: frailty.clone(),
                options: options.clone(),
            };
        let build_binary_request = |prepared: PreparedSurvivalTimeStack| LatentBinaryFitRequest {
            data: ds.values.view(),
            spec: gam::families::latent_survival::LatentBinaryTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.clone(),
                weights: weights.clone(),
                derivative_guard: latent_derivative_guard,
                time_block: build_time_block(&prepared),
                unloaded_mass_entry: prepared.unloaded_mass_entry.clone(),
                unloaded_mass_exit: prepared.unloaded_mass_exit.clone(),
                meanspec: termspec.clone(),
                mean_offset: threshold_offset.clone(),
            },
            frailty: frailty.clone(),
            options: options.clone(),
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            // Analytic-gradient BFGS over the latent baseline shape params
            // (weibull scale/shape; gompertz rate/shape; gompertz-makeham
            // rate/shape/makeham). The baseline θ enters the inner latent fit
            // only through the three additive time-block offsets (entry η,
            // exit η, exit ∂η/∂t), exactly as the transformation path does.
            //
            // We optimize the *profile penalized NLL*
            //   V(θ) = −ℓ(β̂(θ)) + ½·β̂ᵀS β̂,
            // NOT the LAML `reml_score` (which adds ½log|H+S_λ| with its own
            // θ-dependence through H(β̂,θ)). At the converged (constrained) β̂
            // the envelope theorem gives the exact gradient
            //   dV/dθ_k = Σ_i Σ_ch r^ch_i ∂o^ch_i/∂θ_k,
            // with r^ch = LatentSurvivalFamily::offset_channel_residuals(β̂)
            // (carried on the fit result as `baseline_offset_residuals`) and the
            // η-channel offset partials supplied by `baseline_offset_theta_partials`
            // (contracted by `baseline_chain_rule_gradient`). λ is re-optimized
            // inside each inner fit, so its channel drops by the envelope; the
            // downstream final refit re-picks ρ on the full REML surface at the
            // converged baseline θ. BFGS on this exact gradient converges in ≲10
            // outer evaluations on the small 2–3 dim θ-surface.
            baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
                &baseline_cfg,
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "latent survival baseline"
                } else {
                    "latent binary baseline"
                },
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        likelihood_mode,
                        None,
                        time_anchor,
                        latent_derivative_guard,
                        &time_build,
                        None,
                        Some(latent_loading),
                    )?;
                    let (log_likelihood, stable_penalty_term, residuals) = match likelihood_mode {
                        SurvivalLikelihoodMode::Latent => match fit_model(
                            FitRequest::LatentSurvival(build_survival_request(prepared)),
                        ) {
                            Ok(FitResult::LatentSurvival(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err(
                                    "internal latent survival workflow returned the wrong result variant"
                                        .to_string(),
                                );
                            }
                            Err(e) => return Err(format!("latent survival fit failed: {e}")),
                        },
                        SurvivalLikelihoodMode::LatentBinary => match fit_model(
                            FitRequest::LatentBinary(build_binary_request(prepared)),
                        ) {
                            Ok(FitResult::LatentBinary(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err(
                                    "internal latent binary workflow returned the wrong result variant"
                                        .to_string(),
                                );
                            }
                            Err(e) => return Err(format!("latent binary fit failed: {e}")),
                        },
                        // Enclosing block gates this to Latent | LatentBinary —
                        // defensively error out for any other discriminant.
                        SurvivalLikelihoodMode::Transformation
                        | SurvivalLikelihoodMode::Weibull
                        | SurvivalLikelihoodMode::LocationScale
                        | SurvivalLikelihoodMode::MarginalSlope => {
                            return Err(format!(
                                "internal: latent baseline closure reached for non-latent mode {:?}",
                                likelihood_mode
                            ));
                        }
                    };
                    let profile_cost = -log_likelihood + 0.5 * stable_penalty_term;
                    if !profile_cost.is_finite() {
                        return Err(format!(
                            "latent baseline: non-finite profile cost \
                             (log_likelihood={log_likelihood}, \
                             stable_penalty_term={stable_penalty_term}, cost={profile_cost})"
                        ));
                    }
                    let gradient = baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        // CLI latent path does not materialize interval brackets;
                        // `residuals.right` is all-zero so `age_exit` is an
                        // unconsulted placeholder for `age_right`.
                        age_exit.view(),
                        candidate,
                        &residuals,
                    )?
                    .ok_or_else(|| {
                        "latent baseline unexpectedly has no theta gradient".to_string()
                    })?;
                    Ok((profile_cost, gradient))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            likelihood_mode,
            None,
            time_anchor,
            latent_derivative_guard,
            &time_build,
            None,
            Some(latent_loading),
        )?;
        progress.set_stage(
            "fit",
            if likelihood_mode == SurvivalLikelihoodMode::Latent {
                "running latent survival optimization"
            } else {
                "running latent binary optimization"
            },
        );
        let (fit, learned_latent_sd) = match likelihood_mode {
            SurvivalLikelihoodMode::Latent => {
                match fit_model(FitRequest::LatentSurvival(build_survival_request(prepared))) {
                    Ok(FitResult::LatentSurvival(result)) => (result.fit, Some(result.latent_sd)),
                    Ok(_) => {
                        return Err(
                            "internal latent survival workflow returned the wrong result variant"
                                .to_string(),
                        );
                    }
                    Err(e) => return Err(format!("latent survival fit failed: {e}")),
                }
            }
            SurvivalLikelihoodMode::LatentBinary => {
                match fit_model(FitRequest::LatentBinary(build_binary_request(prepared))) {
                    Ok(FitResult::LatentBinary(result)) => (result.fit, None),
                    Ok(_) => {
                        return Err(
                            "internal latent binary workflow returned the wrong result variant"
                                .to_string(),
                        );
                    }
                    Err(e) => return Err(format!("latent binary fit failed: {e}")),
                }
            }
            // Outer block guards `likelihood_mode` to Latent or LatentBinary;
            // defensively error out for any other discriminant.
            SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull
            | SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope => {
                return Err(format!(
                    "internal: latent fit dispatch reached for non-latent mode {:?}",
                    likelihood_mode
                ));
            }
        };
        cli_out!(
            "{} fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            if likelihood_mode == SurvivalLikelihoodMode::Latent {
                "latent survival"
            } else {
                "latent binary"
            },
            fit.pirls_status.label(),
            fit.outer_iterations,
            fit.log_likelihood,
            fit.reml_score,
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage(
                "fit",
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "writing latent survival model"
                } else {
                    "writing latent binary model"
                },
            );
            let is_latent_survival = likelihood_mode == SurvivalLikelihoodMode::Latent;
            // Source-specific work: resolve the latent family (splicing the
            // learned latent SD into the survival frailty) and its labels. The
            // shared core then assembles the canonical payload as the FFI does.
            let family = match likelihood_mode {
                SurvivalLikelihoodMode::Latent => FittedFamily::LatentSurvival {
                    frailty: match &frailty {
                        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                            sigma_fixed: None,
                            loading,
                        } => gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                            sigma_fixed: learned_latent_sd,
                            loading: *loading,
                        },
                        _ => frailty.clone(),
                    },
                },
                SurvivalLikelihoodMode::LatentBinary => FittedFamily::LatentBinary {
                    frailty: frailty.clone(),
                },
                // Same outer gate — `likelihood_mode` is restricted to Latent /
                // LatentBinary on this path; defensively error out.
                SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::LocationScale
                | SurvivalLikelihoodMode::MarginalSlope => {
                    return Err(format!(
                        "internal: model payload constructor reached for non-latent mode {:?}",
                        likelihood_mode
                    ));
                }
            };
            let resolved_termspec = freeze_term_collection_from_design(&termspec, &cov_design)
                .map_err(|e| e.to_string())?;
            let payload = assemble_latent_window_payload(
                LatentWindowInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.clone(),
                    family,
                    model_class_label: if is_latent_survival {
                        "latent-survival".to_string()
                    } else {
                        "latent-binary".to_string()
                    },
                    likelihood_label: if is_latent_survival {
                        "latent".to_string()
                    } else {
                        "latent-binary".to_string()
                    },
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    beta_time: fit.beta_time().to_vec(),
                    resolved_termspec,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress(if likelihood_mode == SurvivalLikelihoodMode::Latent {
            "latent survival fit complete"
        } else {
            "latent binary fit complete"
        });
        return Ok(());
    }

    if effective_config.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                .to_string(),
        );
    }
    let covariate_offset =
        resolve_offset_column(&ds, &col_map, effective_config.offset_column.as_deref())?;
    let dense_cov_design = cov_design.design.to_dense();
    if cause_count > 1 {
        let weibull_seed = if likelihood_mode == SurvivalLikelihoodMode::Weibull
            && !learn_timewiggle
        {
            let scale = effective_config
                .baseline_scale
                .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
            let shape = effective_config.baseline_shape.unwrap_or(1.0);
            if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                return Err(
                    "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                        .to_string(),
                );
            }
            Some((scale, shape))
        } else {
            None
        };
        progress.set_stage("fit", "running cause-specific survival optimization");
        let fit = match fit_model(FitRequest::SurvivalTransformation(
            SurvivalTransformationFitRequest {
                data: ds.values.view(),
                spec: SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_target.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: covariate_offset.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    likelihood_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    ridge_lambda: effective_config.ridge_lambda,
                    // Gamma precision hyperpriors on penalty blocks are only reachable via the
                    // Python FFI fit config. The CLI exposes no flag,
                    // config file, or formula-DSL syntax for them, and the magic-by-default
                    // policy forbids inventing one here, so an empty prior list is correct.
                    penalty_block_gamma_priors: Vec::new(),
                },
                cache_session: None,
            },
        )) {
            Ok(FitResult::SurvivalTransformation(result)) => result,
            Ok(_) => {
                return Err(
                    "internal cause-specific survival workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => return Err(format!("cause-specific survival fit failed: {e}")),
        };
        cli_out!();
        cli_out!(
            "cause-specific survival fit | causes={} | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            cause_count,
            fit.fit.pirls_status.label(),
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing cause-specific survival model");
            // Source-specific work: extract the cause-specific baseline-timewiggle
            // coefficients from the first fitted block (this CLI path persists a
            // single shared timewiggle block; cause_count > 1 guarantees the
            // block exists). The shared core then assembles the canonical
            // payload exactly as the FFI does.
            let timewiggle = fit
                .baseline_timewiggle
                .as_ref()
                .zip(fit.fit.blocks.first())
                .map(|(timewiggle, block)| {
                    let start = fit.time_base_ncols;
                    let end = start + timewiggle.ncols;
                    SurvivalTimewiggle {
                        degree: timewiggle.degree,
                        knots: timewiggle.knots.to_vec(),
                        penalty_orders: effective_timewiggle
                            .as_ref()
                            .map(|cfg| cfg.penalty_orders.clone()),
                        double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                        beta: SurvivalTimewiggleBeta::Single(
                            block.beta.slice(s![start..end]).to_vec(),
                        ),
                    }
                });
            let payload = assemble_survival_transformation_payload(
                SurvivalTransformationInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.fit.clone(),
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec,
                    cause_count: Some(cause_count),
                    baseline_cfg: fit.baseline_cfg.clone(),
                    time_basis: fit.time_basis.clone(),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    resolved_termspec: fit.resolvedspec.clone(),
                    survival_beta_time: Some(fit.fit.beta.to_vec()),
                    timewiggle,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: None,
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("cause-specific survival fit complete");
        return Ok(());
    }
    let build_working_model = |candidate: &SurvivalBaselineConfig| {
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            candidate,
            likelihood_mode,
            None,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let mut eta_offset_entry = prepared.eta_offset_entry.clone();
        let mut eta_offset_exit = prepared.eta_offset_exit.clone();
        eta_offset_entry += &covariate_offset;
        eta_offset_exit += &covariate_offset;
        let p_time_total = prepared.time_design_exit.ncols();
        let p = p_time_total + p_cov;
        let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
        for (idx, s) in prepared.time_penalties.iter().enumerate() {
            if s.nrows() == p_time_total && s.ncols() == p_time_total {
                penalty_blocks.push(PenaltyBlock {
                    matrix: s.clone(),
                    lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                    range: 0..p_time_total,
                    nullspace_dim: prepared.time_nullspace_dims.get(idx).copied().unwrap_or(0),
                });
            }
        }
        let ridge_range_start = if time_build.basisname == "linear" && !learn_timewiggle {
            1
        } else {
            0
        };
        if effective_config.ridge_lambda > 0.0 && p > ridge_range_start {
            let dim = p - ridge_range_start;
            let mut ridge = Array2::<f64>::zeros((dim, dim));
            for d in 0..dim {
                ridge[[d, d]] = 1.0;
            }
            penalty_blocks.push(PenaltyBlock {
                matrix: ridge,
                lambda: effective_config.ridge_lambda,
                range: ridge_range_start..p,
                nullspace_dim: 0,
            });
        }
        let penalties = PenaltyBlocks::new(penalty_blocks.clone());
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 0.0 };
        let dense_time_entry = prepared.time_design_entry.to_dense();
        let dense_time_exit = prepared.time_design_exit.to_dense();
        let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
        let mut model = gam::families::royston_parmar::working_model_from_time_covariateshared(
            penalties,
            monotonicity,
            survivalspec,
            gam::families::royston_parmar::RoystonParmarSharedTimeCovariateInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                weights: weights.view(),
                time_entry: dense_time_entry.view(),
                time_exit: dense_time_exit.view(),
                time_derivative: dense_time_derivative.view(),
                covariates: dense_cov_design.view(),
                monotonicity_constraint_rows: None,
                monotonicity_constraint_offsets: None,
                eta_offset_entry: Some(eta_offset_entry.view()),
                eta_offset_exit: Some(eta_offset_exit.view()),
                derivative_offset_exit: Some(prepared.derivative_offset_exit.view()),
            },
        )
        .map_err(|e| format!("failed to construct survival model: {e}"))?;
        if likelihood_mode != SurvivalLikelihoodMode::Weibull {
            model
                .set_structural_monotonicity(true, p_time_total)
                .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
        }
        let mut beta0 = Array1::<f64>::zeros(p);
        if let Some(seed) = weibull_builtin_beta_seed.as_ref() {
            if p_time_total < seed.len() {
                return Err(format!(
                    "weibull built-in time basis has {} columns but needs at least {} to seed scale/shape",
                    p_time_total,
                    seed.len()
                ));
            }
            beta0.slice_mut(s![..seed.len()]).assign(seed);
        }
        let structural_lower_bounds =
            if likelihood_mode != SurvivalLikelihoodMode::Weibull && p_time_total > 0 {
                let mut lb = Array1::from_elem(p, f64::NEG_INFINITY);
                for j in 0..p_time_total {
                    lb[j] = 0.0;
                    beta0[j] = 1e-4;
                }
                Some(lb)
            } else {
                None
            };
        Ok((
            prepared,
            penalty_blocks,
            p_time_total,
            beta0,
            structural_lower_bounds,
            model,
        ))
    };
    if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        // Analytic-gradient BFGS over the baseline shape params (weibull
        // scale/shape; gompertz rate/shape; gompertz-makeham rate/shape/makeham).
        //
        // The optimized cost is the *profile penalized NLL*
        //   V(θ) = 0.5·deviance(β̂(θ); o(θ)) + 0.5·β̂ᵀSβ̂   (= survival_working_reml_score).
        // The baseline θ enters this working model only through the three additive
        // time-block offsets (entry η, exit η, exit ∂η/∂t). At the (constrained)
        // PIRLS optimum β̂ the envelope theorem gives
        //   dV/dθ_k = ∂V/∂θ_k|_{β=β̂}
        //           = Σ_i r^X_i ∂o_X_i/∂θ_k + r^E_i ∂o_E_i/∂θ_k + r^D_i ∂o_D_i/∂θ_k,
        // the residual×offset-partial contraction of
        // WorkingModelSurvival::offset_channel_residuals(β̂) against the η-channel
        // offset partials (baseline_chain_rule_gradient → baseline_offset_theta_partials).
        // The β_j ≥ 0 active-set constraints carry no θ-dependence, so the
        // constrained envelope identity is exact. See baseline_chain_rule_gradient
        // for the full derivation. BFGS on this exact gradient converges in ≲10
        // outer evaluations on the 2–3 dim θ-surface.
        baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "survival baseline",
            |candidate| {
                let (_, _, _, beta0, structural_lower_bounds, mut model) =
                    build_working_model(candidate)?;
                let pirls_opts = survival_baseline_pirls_options();
                let beta = if likelihood_mode == SurvivalLikelihoodMode::Weibull {
                    let summary = gam::pirls::runworking_model_pirls(
                        &mut model,
                        gam::types::Coefficients::new(beta0.clone()),
                        &pirls_opts,
                        |_| {},
                    )
                    .map_err(|e| format!("survival PIRLS failed: {e}"))?;
                    summary.beta.as_ref().to_owned()
                } else {
                    let constrained_opts = gam::pirls::WorkingModelPirlsOptions {
                        coefficient_lower_bounds: structural_lower_bounds,
                        ..pirls_opts
                    };
                    let summary = gam::pirls::runworking_model_pirls(
                        &mut model,
                        gam::types::Coefficients::new(beta0.clone()),
                        &constrained_opts,
                        |_| {},
                    )
                    .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
                    summary.beta.as_ref().to_owned()
                };
                let state = model.update_state(&beta).map_err(|e| {
                    format!("failed to evaluate survival optimum in coefficient coordinates: {e}")
                })?;
                let cost = survival_working_reml_score(&state);
                let residuals = model.offset_channel_residuals(&beta).map_err(|e| {
                    format!("failed to form survival baseline offset residuals: {e}")
                })?;
                let gradient = baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    // RP transformation path has no interval channel; `residuals.right`
                    // is all-zero so `age_exit` is an unconsulted placeholder.
                    age_exit.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "survival baseline unexpectedly has no theta gradient".to_string()
                })?;
                Ok((cost, gradient))
            },
        )?;
    }
    let (prepared, penalty_blocks, p_time_total, beta0, structural_lower_bounds, model) =
        build_working_model(&baseline_cfg)?;
    let beta0_norm = beta0.dot(&beta0).sqrt();
    progress.set_stage("fit", "running survival pirls");
    let pirls_opts = survival_baseline_pirls_options();
    let pirls_start = std::time::Instant::now();
    let pirls_callback = |info: &gam::pirls::WorkingModelIterationInfo| {
        let elapsed = pirls_start.elapsed().as_secs_f64();
        log::debug!(
            "[PIRLS] iter {:>3} | deviance {:.6e} | |grad| {:.3e} | step {:.3e} | halving {} | {:.1}s",
            info.iteration,
            info.deviance,
            info.gradient_norm,
            info.step_size,
            info.step_halving,
            elapsed,
        );
    };
    let (summary, beta, state, constraint_mode, surv_model) =
        if likelihood_mode == SurvivalLikelihoodMode::Weibull {
            let mut plain_model = model;
            let summary = gam::pirls::runworking_model_pirls(
                &mut plain_model,
                gam::types::Coefficients::new(beta0.clone()),
                &pirls_opts,
                pirls_callback,
            )
            .map_err(|e| format!("survival PIRLS failed: {e}"))?;
            let beta = summary.beta.as_ref().to_owned();
            let state = plain_model.update_state(&beta).map_err(|e| {
                format!("failed to evaluate survival optimum in coefficient coordinates: {e}")
            })?;
            (
                summary,
                beta,
                state,
                "baseline-timewiggle".to_string(),
                plain_model,
            )
        } else {
            let mut constrained_model = model;
            let constrained_opts = gam::pirls::WorkingModelPirlsOptions {
                coefficient_lower_bounds: structural_lower_bounds,
                ..pirls_opts
            };
            let summary = gam::pirls::runworking_model_pirls(
                &mut constrained_model,
                gam::types::Coefficients::new(beta0.clone()),
                &constrained_opts,
                pirls_callback,
            )
            .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
            let beta = summary.beta.as_ref().to_owned();
            let state = constrained_model.update_state(&beta).map_err(|e| {
                format!("failed to evaluate structural survival optimum in spline coordinates: {e}")
            })?;
            (
                summary,
                beta,
                state,
                "constrained-structural-time".to_string(),
                constrained_model,
            )
        };
    log::debug!(
        "[PIRLS] finished: {:?} after {} iterations, deviance={:.6e}, {:.1}s total",
        summary.status,
        summary.iterations,
        state.deviance,
        pirls_start.elapsed().as_secs_f64(),
    );
    // Evaluate LAML objective via unified evaluator for diagnostic logging.
    // Move surv_model into block so it is dropped at block end.
    {
        let surv_model = surv_model;
        let rho = ndarray::Array1::from_iter(
            penalty_blocks
                .iter()
                .filter(|b| b.lambda > 0.0)
                .map(|b| b.lambda.ln()),
        );
        if !rho.is_empty() {
            match surv_model.unified_lamlobjective_and_rhogradient(&beta, &state, &rho) {
                Ok((laml_obj, laml_grad)) => {
                    log::debug!(
                        "[LAML] unified objective={:.6e}, |grad|={:.3e}",
                        laml_obj,
                        laml_grad.dot(&laml_grad).sqrt(),
                    );
                }
                Err(e) => {
                    log::debug!("[LAML] unified evaluation skipped: {e}");
                }
            }
        }
    }
    match summary.status {
        gam::pirls::PirlsStatus::Converged | gam::pirls::PirlsStatus::StalledAtValidMinimum => {}
        other => {
            let event_count = event_target.iter().filter(|&&ev| ev > 0).count();
            let event_rate = if n > 0 {
                event_count as f64 / n as f64
            } else {
                0.0
            };
            let min_entry = age_entry.iter().copied().fold(f64::INFINITY, f64::min);
            let max_exit = age_exit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let beta_norm = beta.dot(&beta).sqrt();
            return Err(format!(
                "survival constrained PIRLS did not converge: status={other:?}, grad_norm={:.3e}, iterations={}, deviance={:.6e}, last_deviance_change={:.3e}, last_step_size={:.3e}, last_step_halving={}, max_abs_eta={:.3e}, beta0_norm={:.3e}, beta_norm={:.3e}; run[likelihood={}, spec={}, baseline_target={}, time_basis={}, constraint_mode={}, n={}, events={}, event_rate={:.4}, time_range=[{:.3e}, {:.3e}], p_time={}, p_cov={}, formula=\"{}\"]",
                summary.lastgradient_norm,
                summary.iterations,
                state.deviance,
                summary.last_deviance_change,
                summary.last_step_size,
                summary.last_step_halving,
                summary.max_abs_eta,
                beta0_norm,
                beta_norm,
                survival_likelihood_modename(likelihood_mode),
                effectivespec,
                if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
                    survival_baseline_targetname(SurvivalBaselineTarget::Weibull)
                } else {
                    survival_baseline_targetname(baseline_cfg.target)
                },
                time_build.basisname,
                constraint_mode,
                n,
                event_count,
                event_rate,
                min_entry,
                max_exit,
                p_time_total,
                p_cov,
                formula
            ));
        }
    }

    let fitted_baseline_cfg = if likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !learn_timewiggle
    {
        let time_beta = beta.slice(s![..p_time_total]).to_owned();
        let (scale, shape) = fitted_weibull_baseline_from_linear_time_beta(&time_beta, time_anchor)
            .ok_or_else(|| {
                "failed to recover fitted Weibull scale/shape from the linear time coefficients"
                    .to_string()
            })?;
        SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(scale),
            shape: Some(shape),
            rate: None,
            makeham: None,
        }
    } else {
        baseline_cfg.clone()
    };

    cli_out!();
    cli_out!(
        "survival config | likelihood={} | time_basis={} | baseline_target={}",
        survival_likelihood_modename(likelihood_mode),
        time_build.basisname,
        survival_baseline_targetname(fitted_baseline_cfg.target)
    );

    progress.advance_workflow(3);
    if let Some(out) = args.out {
        progress.set_stage("fit", "writing survival model");
        let hessian = state.hessian.to_dense();
        let cov = match invert_symmetric_matrix(&hessian) {
            Ok(c) => Some(c),
            Err(e) => {
                cli_err!(
                    "warning: failed to invert survival Hessian for covariance ({}); saving model without covariance",
                    e
                );
                None
            }
        };
        let fit_result = core_saved_fit_result(
            beta.clone(),
            Array1::from_iter(penalty_blocks.iter().map(|b| b.lambda)),
            1.0,
            cov.clone(),
            cov.clone(),
            SavedFitSummary::from_survivalworking_summary(&summary, &state)?,
        );
        // Source-specific work: snapshot the time basis and, when present,
        // extract the single-block baseline-timewiggle coefficients. The shared
        // core (the same path the FFI uses) assembles the canonical payload —
        // and routes the time-basis write through `apply_survival_time_basis`,
        // which is what makes the historic `survival_time_anchor` omission
        // impossible.
        let timewiggle = prepared.timewiggle_build.as_ref().map(|w| {
            let start = time_build.x_exit_time.ncols();
            let end = start + w.ncols;
            SurvivalTimewiggle {
                degree: w.degree,
                knots: w.knots.to_vec(),
                penalty_orders: effective_timewiggle
                    .as_ref()
                    .map(|cfg| cfg.penalty_orders.clone()),
                double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                beta: SurvivalTimewiggleBeta::Single(beta.slice(s![start..end]).to_vec()),
            }
        });
        let payload = assemble_survival_transformation_payload(
            SurvivalTransformationInputs {
                formula,
                data_schema: ds.schema.clone(),
                fit_result,
                survival_entry: args.entry,
                survival_exit: args.exit,
                survival_event: args.event,
                survivalspec: effectivespec,
                cause_count: None,
                baseline_cfg: fitted_baseline_cfg.clone(),
                time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                ridge_lambda: effective_config.ridge_lambda,
                survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                    .to_string(),
                resolved_termspec: frozen_termspec,
                survival_beta_time: None,
                timewiggle,
            },
            SavedModelSourceMetadata {
                training_headers: ds.headers.clone(),
                training_feature_ranges: Some(ds.feature_ranges()),
                offset_column: effective_config.offset_column.clone(),
                noise_offset_column: effective_config.noise_offset_column.clone(),
            },
        );
        write_payload_json(&out, payload)?;
        progress.advance_workflow(survival_total_steps);
    }
    progress.finish_progress("survival fit complete");
    Ok(())
}
