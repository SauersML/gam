use super::*;

fn resolve_survival_marginal_slope_base_link(
    linkspec: Option<&crate::inference::formula_dsl::LinkFormulaSpec>,
) -> Result<InverseLink, String> {
    let Some(linkspec) = linkspec else {
        return Ok(InverseLink::Standard(StandardLink::Probit));
    };
    let choice = parse_link_choice(Some(&linkspec.link), false)?
        .ok_or_else(|| "invalid survival marginal-slope link".to_string())?;
    if choice.mixture_components.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survival marginal-slope currently supports only link(type=probit)".to_string(),
        }
        .into());
    }
    match choice.link {
        LinkFunction::Probit => Ok(InverseLink::Standard(StandardLink::Probit)),
        other => Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival marginal-slope currently supports only link(type=probit), got {other:?}"
            ),
        }
        .into()),
    }
}

pub(crate) fn materialize_survival<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
    entry_col: Option<&str>,
    exit_col: &str,
    event_col: &str,
    interval_right_col: Option<&str>,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let mut inference_notes = Vec::new();

    // Extract columns. `entry_col == None` is the right-censored shorthand
    // `Surv(time, event)`: every subject enters at time zero, so we
    // synthesize a constant-zero entry vector instead of resolving a column.
    let entry_idx = entry_col
        .map(|name| resolve_role_col(col_map, name, "entry"))
        .transpose()?;
    let exit_idx = resolve_role_col(col_map, exit_col, "exit")?;
    let event_idx = resolve_role_col(col_map, event_col, "event")?;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = data.values.nrows();
    let event = data.values.column(event_idx).to_owned();
    let event_codes = Array1::from_iter(
        event
            .iter()
            .copied()
            .enumerate()
            .map(|(i, value)| crate::families::survival::survival_event_code_from_value(value, i))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let entry_val = entry_idx.map_or(0.0, |idx| data.values[[i, idx]]);
            normalize_survival_time_pair(entry_val, data.values[[i, exit_idx]], i)
        })
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (e, x)) in pairs.into_iter().enumerate() {
        age_entry[i] = e;
        age_exit[i] = x;
    }

    // Interval-censored `SurvInterval(L, R, event)`: `exit_col` carried the
    // LEFT boundary `L` (resolved into `age_exit` above), and `interval_right_col`
    // carries the RIGHT boundary `R`. The kernel's interval contribution
    // `log[S(L) − S(R)]` requires a finite `R ≥ L` per row (`event >= 0.5`); a
    // row with `event < 0.5` is right-censored at `L` (its `R` is ignored). We
    // resolve `age_right` here so the downstream latent time stack can evaluate
    // the baseline at `R`.
    let age_right = if let Some(right_col) = interval_right_col {
        let right_idx = resolve_role_col(col_map, right_col, "interval right")?;
        let mut right = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = data.values[[i, right_idx]];
            let is_bracketed = data.values[[i, event_idx]] >= 0.5;
            if is_bracketed {
                if !(r.is_finite()) || r < age_exit[i] {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "SurvInterval(L, R, event) requires a finite R >= L on bracketed rows (event >= 1); row {} has L={}, R={r}",
                            i + 1,
                            age_exit[i]
                        ),
                    });
                }
                right[i] = r;
            } else {
                // Right-censored row: R is unused by the likelihood. Pin it to L
                // so the (ignored) right channel stays well-defined and the
                // `age_exit <= age_right` time-basis invariant holds.
                right[i] = age_exit[i];
            }
        }
        Some(right)
    } else {
        None
    };

    let survival_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    if age_right.is_some() && survival_mode != SurvivalLikelihoodMode::Latent {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "interval-censored SurvInterval(L, R, event) is only defined for the latent \
                 hazard-window survival likelihood (its kernel carries the log[S(L) − S(R)] \
                 interval contribution); got survival_likelihood='{}'",
                config.survival_likelihood
            ),
        });
    }
    // Fail fast on all-censored (zero-event) survival data for every survival
    // likelihood (#789B / construction-time fittability split). With no row
    // marking a target event, the survival likelihood has no event score: the
    // hazard direction is unidentified and the inner/outer solve either spins
    // on a flat landscape (marginal-slope) or returns a numerically degenerate
    // fit (other modes). This is the single chokepoint every survival fit
    // dispatcher routes through (Surv(...) responses + all FitConfig survival
    // modes), so catching it here keeps every downstream constructor —
    // `WorkingModelSurvival`, the Royston-Parmar wrapper, the marginal-slope
    // builders — free to materialize models on censored fixtures (which the
    // engine's structural unit tests rely on) without losing the user-facing
    // safety on real fits.
    if !event_codes.iter().any(|&code| code > 0) {
        let mode_label = match survival_mode {
            SurvivalLikelihoodMode::MarginalSlope => "survival marginal-slope",
            _ => "survival fit",
        };
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "{mode_label} requires at least one target event; all rows are censored, so the likelihood has no event score and cannot identify the hazard"
            ),
        });
    }
    let cause_count = crate::families::survival::cause_count_from_event_codes(event_codes.view())
        .into_workflow_result()?;
    if cause_count > 1
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "cause-specific competing risks with {cause_count} causes are currently supported for survival_likelihood='transformation' and 'weibull'; got '{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkwiggle.is_some()
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linkwiggle(...) is not defined for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkspec.is_some()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "link(...) is not implemented for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    // Hoist the survival marginal-slope z-column exclusion check above the
    // time-basis / termspec construction below.  Those downstream steps fail
    // fast on small or tightly-spaced time data (e.g. an I-spline of degree 3
    // cannot be supported by a 2-row fixture), which would otherwise swallow
    // the z-column misuse error and surface a knot-count error instead.
    // Checking here keeps the user-visible error tied to the actual config
    // problem the caller can fix (rename `z` or remove the alias) rather than
    // to an unrelated basis-shape failure further downstream.
    if matches!(survival_mode, SurvivalLikelihoodMode::MarginalSlope)
        && let Some(z_column) = config.z_column.as_deref()
    {
        let logslope_parsed_for_check = match config.logslope_formula.as_deref() {
            Some(ls_formula) => Some(
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?
                    .1,
            ),
            None => None,
        };
        let logslope_ref = logslope_parsed_for_check.as_ref().unwrap_or(parsed);
        validate_marginal_slope_z_column_exclusion(
            parsed,
            logslope_ref,
            z_column,
            "survival marginal-slope",
            "logslope_formula",
        )?;
    }
    let effective_timewiggle = parsed.timewiggle.clone();
    let baseline_target_raw = match survival_mode {
        SurvivalLikelihoodMode::Weibull if effective_timewiggle.is_some() => "weibull",
        SurvivalLikelihoodMode::Weibull => "linear",
        _ => &config.baseline_target,
    };
    let baseline_cfg = initial_survival_baseline_config_for_fit(
        baseline_target_raw,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent hazard-window families require a non-linear scalar baseline target; use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }
    let time_cfg = if effective_timewiggle.is_some() {
        // Match the CLI path: the parametric baseline plus timewiggle supplies
        // the time structure, so the base time basis is disabled.
        SurvivalTimeBasisConfig::None
    } else if survival_mode == SurvivalLikelihoodMode::Weibull {
        SurvivalTimeBasisConfig::Linear
    } else {
        parse_survival_time_basis_config(
            &config.time_basis,
            config.time_degree,
            config.time_num_internal_knots,
            config.time_smooth_lambda,
        )?
    };
    // Marginal-slope centers the baseline-hazard I-spline at a robust interior
    // exit-scale time (median exit) rather than the earliest entry age: under
    // left truncation the earliest entry is a positive left-tail point and
    // centering there inflates the unpenalized linear-trend column, blowing up
    // the time-block seed score so REML rejects every seed (issue #751).
    // Location-scale keeps the earliest-entry anchor.
    let time_anchor = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, None)?
    } else {
        resolve_survival_time_anchor_value(&age_entry, None)?
    };
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(survival_mode);

    // Build time basis
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg.clone(),
        Some((config.time_num_internal_knots, config.time_smooth_lambda)),
    )?;
    if survival_mode != SurvivalLikelihoodMode::Weibull && effective_timewiggle.is_none() {
        require_structural_survival_time_basis(&time_build.basisname, "workflow survival fitting")?;
    }
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    // Interval-censored data needs the SAME monotone time basis evaluated at the
    // RIGHT boundary `R` (so `q_right = X_time(R)·β_time + offset_right`). Rebuild
    // it from the FROZEN knots (`resolved_time_cfg`, carrying the knot vector the
    // exit basis just inferred) at `age_right` in the exit slot — no knot drift —
    // and anchor-center its exit design identically. The resulting `x_exit_time`
    // row is exactly the design at `R`. `time_build_right.x_entry_time` /
    // `x_derivative_time` are unused by the interval-right channel.
    let time_build_right = if let Some(age_right) = age_right.as_ref() {
        let mut build_right = build_survival_time_basis(
            &age_entry,
            age_right,
            resolved_time_cfg.clone(),
            Some((config.time_num_internal_knots, config.time_smooth_lambda)),
        )?;
        center_survival_time_designs_at_anchor(
            &mut build_right.x_entry_time,
            &mut build_right.x_exit_time,
            &time_anchor_row,
        )?;
        Some(build_right)
    } else {
        None
    };
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }

    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            // Survival marginal-slope shares the operator-only invariant with
            // the Bernoulli path; flag it as such so strict mode is selected
            // even at small n.
            marginal_slope_large_scale_active: survival_mode
                == SurvivalLikelihoodMode::MarginalSlope,
        },
    );
    // Alias `z` to the dose column for the marginal termspec only when a raw
    // z_column is supplied. With a CTN Stage-1 recipe there is no dose column
    // (z is produced out-of-fold by cross-fitting) and the marginal formula
    // references only the x covariates, so no alias is needed.
    let marginal_slope_aliased_col_map = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(z_column) => Some(column_map_with_alias(col_map, "z", z_column)),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let termspec_col_map = marginal_slope_aliased_col_map.as_ref().unwrap_or(col_map);
    let mut termspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        termspec_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        prune_unidentified_linear_terms_for_marginal_slope(
            &mut termspec,
            data,
            "survival marginal-slope marginal formula",
            &mut inference_notes,
        )?;
    }

    let residual_dist = parse_survival_distribution(&config.survival_distribution)?;
    let survival_inverse_link = residual_distribution_inverse_link(residual_dist);
    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());
    let effective_linkwiggle_cfg = effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let threshold_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let threshold_template = if let Some(k) = config.threshold_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.threshold_time_degree,
            "threshold",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigma_template = if let Some(k) = config.sigma_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.sigma_time_degree,
            "sigma",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigmaspec = if let Some(noise) = config.noise_formula.as_deref() {
        let mut noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());
        // Use the same aliased col_map as the main termspec — survival
        // marginal-slope reserves `z` as a placeholder for `--z-column`,
        // and the logslope/noise formula may reference it too.
        build_termspec_with_geometry_and_overrides(
            &noise_parsed.terms,
            data,
            termspec_col_map,
            &mut inference_notes,
            config.scale_dimensions,
            &policy,
            config.smooth_overrides.as_ref(),
        )?
    } else {
        // No `noise_formula` ⇒ default to an empty log-σ spec for every
        // survival likelihood (constant log-σ baseline owned by the family
        // adapter). The previous `LocationScale`-only branch cloned the
        // mean `termspec` here, which duplicated every threshold term onto
        // the log-σ block. For a smooth `s(x)` on the mean that was
        // structurally fatal: the canonical-gauge identifiability audit
        // saw the log-σ block as exact-aliased to threshold and (per the
        // descending priorities time=200 > threshold=150 > log_sigma=120,
        // issue #366) attributed/dropped every log-σ column, leaving the
        // solver's `ParameterBlockSpec` design at width 0 while the
        // family kept the un-audited `x_log_sigma` at the smooth's width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then errored "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        // The empty default routes through the same
        // `infer_non_intercept_start_design`/`design_column_tail`
        // contract every other mode uses (yielding a 0-column
        // `x_log_sigma` that matches the spec), so the family and spec
        // agree by construction.
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    // `z_column` is OPTIONAL for the survival marginal-slope when a CTN Stage-1
    // recipe is present: the calibrated chain produces the single `z` surface
    // out-of-fold from the cross-fitted CTN, so there is no raw dose column to
    // read (no throwaway pre-fit column — the no-slop cutover, #461). Without a
    // recipe, the primitive standalone survival marginal-slope still requires a
    // raw `z_column` dose.
    let marginal_z_column_name = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(name) => Some(name),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let (
        marginal_z,
        marginal_logslopespec,
        marginal_logslopespecs,
        marginal_slope_deviation_routing,
        marginal_slope_base_link,
    ) = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        let base_link = resolve_survival_marginal_slope_base_link(parsed.linkspec.as_ref())?;
        if marginal_z_column_name.is_none() {
            // Calibrated chain: the CTN Stage-1 recipe produces a SINGLE z surface
            // out-of-fold, so no dose column is read. Stand in an n×1 placeholder
            // surface (the cross-fit below overrides column 0) and build the
            // logslope surface from the formula (or the marginal termspec). The
            // single-surface invariant matches the cross-fit guard further down.
            let placeholder_z = Array2::<f64>::zeros((data.values.nrows(), 1));
            let (logslopespec, routing) = if let Some(ls_formula) =
                config.logslope_formula.as_deref()
            {
                let (_, ls_parsed) = parse_matching_auxiliary_formula(
                    ls_formula,
                    &parsed.response,
                    "logslope_formula",
                )?;
                if ls_parsed.linkspec.is_some() {
                    return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.timewiggle.is_some() {
                    return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.survivalspec.is_some() {
                    return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &ls_parsed.terms,
                    data,
                    col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                let routing = route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?;
                (spec, routing)
            } else {
                (
                    termspec.clone(),
                    route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                )
            };
            (
                Some(placeholder_z),
                Some(logslopespec.clone()),
                Some(vec![logslopespec]),
                routing,
                Some(base_link),
            )
        } else if let Some(ls_formula) = config.logslope_formula.as_deref() {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            validate_marginal_slope_z_column_exclusion(
                parsed,
                &ls_parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let surfaces = marginal_slope_logslope_surfaces(&ls_parsed, default_z_column)?;
            let mut z = Array2::<f64>::zeros((data.values.nrows(), surfaces.len()));
            let mut specs = Vec::with_capacity(surfaces.len());
            for (surface_idx, surface) in surfaces.iter().enumerate() {
                let z_idx = resolve_role_col(col_map, &surface.z_column, "z")?;
                z.column_mut(surface_idx).assign(&data.values.column(z_idx));
                let aliased_col_map = column_map_with_alias(col_map, "z", &surface.z_column);
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &surface.terms,
                    data,
                    &aliased_col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                specs.push(spec);
            }
            (
                Some(z),
                specs.first().cloned(),
                Some(specs),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?,
                Some(base_link),
            )
        } else {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            validate_marginal_slope_z_column_exclusion(
                parsed,
                parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let z_idx = resolve_role_col(col_map, default_z_column, "z")?;
            let z = data.values.column(z_idx).to_owned().insert_axis(Axis(1));
            (
                Some(z),
                Some(termspec.clone()),
                Some(vec![termspec.clone()]),
                route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                Some(base_link),
            )
        }
    } else {
        (
            None,
            None,
            None,
            MarginalSlopeDeviationRouting {
                score_warp: None,
                link_dev: None,
            },
            None,
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when the
    // survival marginal-slope `z` was generated by a CTN Stage-1 fit (design
    // §5). Computed once (it refits the CTN K times) — outside the per-baseline
    // request closure below. When active it replaces the (single) CTN-generated
    // z surface with its out-of-fold value and captures the score-influence
    // Jacobian `J` for Stage-2's leakage-projection block. With no CTN Stage-1
    // recipe, the raw z surfaces stand and `score_warp` is the fallback basis.
    let crossfit_calibration = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
    } else {
        None
    };
    let (marginal_z, marginal_slope_jac_oof) = match (marginal_z, crossfit_calibration) {
        (Some(mut z_surfaces), Some(calibration)) => {
            // A CTN Stage-1 chain produces exactly one latent score surface; the
            // OOF projection is defined against that single column.
            if z_surfaces.ncols() != 1 {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "cross-fitted score calibration applies to a single CTN-generated z \
                         surface, but the survival marginal-slope model has {} z surfaces; \
                         multi-surface logslope is incompatible with the CTN Stage-1 chain",
                        z_surfaces.ncols()
                    ),
                });
            }
            z_surfaces.column_mut(0).assign(&calibration.z_oof);
            (Some(z_surfaces), Some(calibration.jac_oof))
        }
        (z, _) => (z, None),
    };

    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes formula-level linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_score_warp.is_some() {
            inference_notes.push(
                "survival marginal-slope routes logslope_formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_link_dev.is_none() && marginal_slope_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }
    }
    let marginal_slope_frailty = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        Some(fixed_gaussian_shift_frailty_from_spec(
            config.frailty.as_ref().unwrap_or(&FrailtySpec::None),
            "survival marginal-slope",
        )?)
    } else {
        None
    };
    match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
            if config.frailty.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "frailty is not supported for transformation/weibull survival models"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::LocationScale if config.frailty.is_some() => {
            return Err(WorkflowError::InvalidConfig {
                reason: "config.frailty is not implemented for survival-likelihood=location-scale"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
            if effective_timewiggle.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "timewiggle is not implemented for latent survival/binary likelihoods"
                    .to_string(),
            }
            .into());
        }
        _ => {}
    }
    let latent_loading = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let frailty = config.frailty.as_ref().unwrap_or(&FrailtySpec::None);
        Some(latent_hazard_loading(
            frailty,
            "workflow latent survival/binary",
        )?)
    } else {
        None
    };

    let build_time_block =
        |candidate: &crate::families::survival::construction::SurvivalBaselineConfig| {
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                (survival_mode == SurvivalLikelihoodMode::LocationScale)
                    .then_some(&survival_inverse_link),
                time_anchor,
                exact_derivative_guard,
                &time_build,
                effective_timewiggle.as_ref(),
                None,
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let initial_beta = if survival_mode == SurvivalLikelihoodMode::LocationScale {
                None
            } else {
                Some(Array1::from_elem(time_p, 1e-4))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta,
            };
            Ok::<_, String>((prepared, time_block))
        };

    // Warm-start cache for the outer baseline-config optimization: each probe
    // runs a complete inner BFGS over ρ (log-smoothing) starting from zeros if cold; by
    // capturing the previous probe's converged ρ (threshold + log_sigma blocks) and
    // injecting it here, the next inner BFGS typically converges in 1-3 iterations
    // instead of ~10, cutting per-probe cost roughly 5-10× across the probes per fit.
    let location_scale_smoothing_warm_start: RefCell<Option<(Array1<f64>, Array1<f64>)>> =
        RefCell::new(None);
    let build_location_scale_request =
        |candidate: &crate::families::survival::construction::SurvivalBaselineConfig,
         allow_inverse_link_optimization: bool| {
            let (prepared, time_block) = build_time_block(candidate)?;
            let (initial_threshold_log_lambdas, initial_log_sigma_log_lambdas) =
                match location_scale_smoothing_warm_start.borrow().as_ref() {
                    Some((thr, lsg)) => (Some(thr.clone()), Some(lsg.clone())),
                    None => (None, None),
                };
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                inverse_link: survival_inverse_link.clone(),
                derivative_guard: exact_derivative_guard,
                max_iter: 200,
                tol: 1e-7,
                time_block,
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block,
                linkwiggle_block: None,
                initial_threshold_log_lambdas,
                initial_log_sigma_log_lambdas,
                cache_session: None,
                cache_mirror_sessions: Vec::new(),
            };
            // During baseline-θ BFGS probes we hold the inverse-link state
            // fixed: otherwise every probe would trigger a nested
            // optimization over the SAS / BetaLogistic / Mixture link
            // parameters, defeating the BFGS speedup entirely. The final
            // fit (after baseline has converged) flips this back on, so
            // joint baseline + link optimization still happens — just
            // alternating instead of nested.
            let optimize_inverse_link = allow_inverse_link_optimization
                && survival_inverse_link_has_free_parameters(&spec.inverse_link);
            Ok::<_, String>(SurvivalLocationScaleFitRequest {
                data: data.values.view(),
                spec,
                wiggle: effective_linkwiggle_cfg.clone(),
                kappa_options: SpatialLengthScaleOptimizationOptions {
                    outer_wall_clock_budget_secs: config.outer_wall_clock_budget_secs,
                    ..SpatialLengthScaleOptimizationOptions::default()
                },
                optimize_inverse_link,
                cache_session: None,
            })
        };

    let build_marginal_slope_request =
        |candidate: &crate::families::survival::construction::SurvivalBaselineConfig| {
            let (prepared, time_block) = build_time_block(candidate)?;
            Ok::<_, String>(SurvivalMarginalSlopeFitRequest {
                data: data.values.view(),
                spec: SurvivalMarginalSlopeTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.clone(),
                    weights: weights.clone(),
                    z: marginal_z.clone().ok_or_else(|| {
                        "marginal-slope survival requires z_column in FitConfig".to_string()
                    })?,
                    base_link: marginal_slope_base_link.clone().ok_or_else(|| {
                        "internal error: marginal-slope base link validation missing".to_string()
                    })?,
                    marginalspec: termspec.clone(),
                    marginal_offset: threshold_offset.clone(),
                    frailty: marginal_slope_frailty.clone().ok_or_else(|| {
                        "internal error: marginal-slope frailty validation missing".to_string()
                    })?,
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    timewiggle_block: prepared.timewiggle_block,
                    logslopespec: marginal_logslopespec.clone().ok_or_else(|| {
                        "marginal-slope survival is missing logslope spec".to_string()
                    })?,
                    logslopespecs: marginal_logslopespecs.clone(),
                    logslope_offset: log_sigma_offset.clone(),
                    score_warp: marginal_slope_score_warp.clone(),
                    link_dev: marginal_slope_link_dev.clone(),
                    latent_z_policy: Default::default(),
                    score_influence_jacobian: marginal_slope_jac_oof.clone(),
                },
                options: BlockwiseFitOptions {
                    compute_covariance: false,
                    // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                    // default for survival marginal-slope — no flag to thread.
                    ..Default::default()
                },
                kappa_options: SpatialLengthScaleOptimizationOptions {
                    outer_wall_clock_budget_secs: config.outer_wall_clock_budget_secs,
                    ..SpatialLengthScaleOptimizationOptions::default()
                },
            })
        };

    let build_latent_survival_request =
        |candidate: &crate::families::survival::construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent survival loading missing after frailty validation"
                    .to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            // Interval-censored: build the matching time stack at the RIGHT
            // boundary `R` (the exit slot holds `age_right`, evaluated through the
            // frozen-knot `time_build_right`). Its exit channel is exactly the
            // `R`-evaluated design / offset / unloaded mass, which feed the
            // dedicated `_right` spec fields the kernel consumes for
            // `log[S(L) − S(R)]`. The `event_target` then marks bracketed rows
            // (`event >= 1`) with the `LATENT_SURVIVAL_EVENT_INTERVAL` sentinel
            // and leaves `event < 1` rows right-censored at `L`.
            let (time_design_right, time_offset_right, unloaded_mass_right, event_target) =
                if let (Some(age_right), Some(time_build_right)) =
                    (age_right.as_ref(), time_build_right.as_ref())
                {
                    let prepared_right = prepare_survival_time_stack(
                        &age_entry,
                        age_right,
                        candidate,
                        survival_mode,
                        None,
                        time_anchor,
                        exact_derivative_guard,
                        time_build_right,
                        None,
                        Some(loading),
                    )?;
                    if prepared_right.time_design_exit.ncols() != prepared.time_design_exit.ncols()
                    {
                        return Err(format!(
                            "interval-censored right time design has {} columns but the left/exit design has {}; the right boundary basis must share the exit basis columns",
                            prepared_right.time_design_exit.ncols(),
                            prepared.time_design_exit.ncols()
                        ));
                    }
                    let event_target = event.mapv(|v| {
                        if v >= 0.5 {
                            crate::families::survival::latent::LATENT_SURVIVAL_EVENT_INTERVAL
                        } else {
                            0
                        }
                    });
                    (
                        Some(prepared_right.time_design_exit.clone()),
                        Some(prepared_right.eta_offset_exit.clone()),
                        prepared_right.unloaded_mass_exit.clone(),
                        event_target,
                    )
                } else {
                    (
                        None,
                        None,
                        Array1::zeros(0),
                        event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    )
                };
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentSurvivalFitRequest {
                data: data.values.view(),
                spec: LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target,
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    time_design_right,
                    time_offset_right,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    unloaded_mass_right,
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let build_latent_binary_request =
        |candidate: &crate::families::survival::construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent binary loading missing after frailty validation".to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentBinaryFitRequest {
                data: data.values.view(),
                spec: LatentBinaryTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let baseline_cfg = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        baseline_cfg
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::MarginalSlope
    {
        optimize_survival_baseline_config_with_gradient(
            &baseline_cfg,
            "workflow survival marginal-slope baseline",
            |candidate| {
                let fit =
                    fit_survival_marginal_slope_model(build_marginal_slope_request(candidate)?)
                        .map_err(|e| format!("survival marginal-slope fit failed: {e}"))?;
                let gradient = marginal_slope_baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &fit.baseline_offset_residuals,
                )?
                .ok_or_else(|| {
                    "workflow survival marginal-slope baseline unexpectedly has no theta gradient"
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
                    "workflow survival marginal-slope baseline unexpectedly has no theta Hessian"
                        .to_string()
                })?;
                Ok((fit.fit.reml_score, gradient, hessian))
            },
        )?
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::LocationScale
    {
        // Analytic θ-gradient path. The baseline configuration enters the
        // location-scale fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t); at the converged β the
        // envelope theorem gives
        //
        //   d(NLL)/dθ_k = Σ_i r^(E)_i ∂o_E_i/∂θ_k
        //               + r^(X)_i ∂o_X_i/∂θ_k
        //               + r^(D)_i ∂o_D_i/∂θ_k
        //
        // where r^(*) are populated by
        // `SurvivalLocationScaleFamily::offset_channel_geometry` and the
        // partials by `baseline_offset_theta_partials`. When the inverse
        // link is probit/SAS/Mixture/etc., the location-scale family uses
        // the probit-channel baseline q(t) instead, so we contract against
        // `marginal_slope_baseline_offset_theta_partials` exactly as the
        // marginal-slope path does. BFGS w/ this analytic gradient
        // typically converges in ≲10 outer evaluations.
        let probit_channel =
            location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow survival location-scale baseline",
            |candidate| {
                let fit_result = fit_survival_location_scale_model(build_location_scale_request(
                    candidate, false,
                )?)
                .map_err(|e| format!("survival location-scale fit failed: {e}"))?;
                // Warm-start the next probe's threshold / log-σ smoothing parameters
                // at the converged values for this probe.
                let threshold_rho = fit_result.fit.fit.lambdas_threshold().mapv(f64::ln);
                let log_sigma_rho = fit_result.fit.fit.lambdas_log_sigma().mapv(f64::ln);
                *location_scale_smoothing_warm_start.borrow_mut() =
                    Some((threshold_rho, log_sigma_rho));
                let residuals = &fit_result.fit.baseline_offset_residuals;
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
                        // Location-scale has no interval channel; `residuals.right`
                        // is all-zero so `age_exit` is an unconsulted placeholder.
                        age_exit.view(),
                        candidate,
                        residuals,
                    )?
                }
                .ok_or_else(|| {
                    "workflow survival location-scale baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                // The envelope-theorem residual contraction is the exact
                // θ-gradient of the *profile penalized NLL* −ℓ + ½βᵀSβ at
                // converged (β̂, ρ̂). Optimizing `reml_score` (which includes
                // ½ log|S_λ| − ½ log|H| LAML corrections) against this
                // gradient would mismatch the cost surface, because the
                // log-determinant terms have their own θ-dependence through
                // H(β̂, θ). Use the matching profile-NLL cost here; the final
                // model refit downstream still picks ρ via the full REML
                // surface at the converged baseline θ.
                let profile_cost = -fit_result.fit.fit.log_likelihood
                    + 0.5 * fit_result.fit.fit.stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(format!(
                        "workflow survival location-scale baseline: non-finite profile cost \
                         (log_likelihood={}, stable_penalty_term={}, cost={})",
                        fit_result.fit.fit.log_likelihood,
                        fit_result.fit.fit.stable_penalty_term,
                        profile_cost
                    ));
                }
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(e) => return Err(e.into()),
        }
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        // Latent / LatentBinary baseline-θ. The baseline configuration enters
        // the inner latent fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t), so the envelope theorem at the
        // converged β̂ gives the exact θ-gradient of the *profile penalized NLL*
        //   V(θ) = −ℓ(β̂(θ)) + ½·β̂ᵀS β̂,
        //     dV/dθ_k = Σ_i Σ_ch r^ch_i ∂o^ch_i/∂θ_k,
        // with r^ch = LatentSurvivalFamily::offset_channel_residuals(β̂)
        // (`baseline_offset_residuals` on the fit result) contracted against
        // `baseline_offset_theta_partials` by `baseline_chain_rule_gradient`.
        // We optimize the profile-NLL — not the LAML `reml_score` whose
        // ½log|H+S_λ| term carries its own θ-dependence through H(β̂,θ) — and
        // the downstream final refit re-picks ρ on the full REML surface at the
        // converged baseline θ. BFGS converges in ≲10 outer evaluations.
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow latent survival baseline",
            |candidate| {
                let (log_likelihood, stable_penalty_term, residuals) = match survival_mode {
                    SurvivalLikelihoodMode::Latent => {
                        let request = build_latent_survival_request(candidate)?;
                        match fit_model(FitRequest::LatentSurvival(request)) {
                            Ok(FitResult::LatentSurvival(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err("internal latent survival workflow returned the wrong result variant".to_string());
                            }
                            Err(e) => return Err(format!("latent survival fit failed: {e}")),
                        }
                    }
                    SurvivalLikelihoodMode::LatentBinary => {
                        let request = build_latent_binary_request(candidate)?;
                        match fit_model(FitRequest::LatentBinary(request)) {
                            Ok(FitResult::LatentBinary(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err("internal latent binary workflow returned the wrong result variant".to_string());
                            }
                            Err(e) => return Err(format!("latent binary fit failed: {e}")),
                        }
                    }
                    SurvivalLikelihoodMode::Transformation
                    | SurvivalLikelihoodMode::Weibull
                    | SurvivalLikelihoodMode::LocationScale
                    | SurvivalLikelihoodMode::MarginalSlope => {
                        return Err(format!(
                            "internal: workflow latent baseline closure reached for non-latent mode {survival_mode:?}"
                        ));
                    }
                };
                let profile_cost = -log_likelihood + 0.5 * stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(format!(
                        "workflow latent baseline: non-finite profile cost \
                         (log_likelihood={log_likelihood}, \
                         stable_penalty_term={stable_penalty_term}, cost={profile_cost})"
                    ));
                }
                // Interval upper-bound boundary ages for the `right` channel.
                // When the formula carries no `SurvInterval(L, R, event)` column
                // `age_right` is None and every row's `residuals.right` is 0, so
                // `age_exit` is an unconsulted placeholder; when present it is the
                // per-row `R` the interval likelihood `log[S(L) − S(R)]` brackets
                // against, and its baseline-θ η-offset sensitivity is the missing
                // gradient channel the latent interval fit needs.
                let age_right_view = age_right.as_ref().unwrap_or(&age_exit);
                let gradient = baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    age_right_view.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "workflow latent baseline unexpectedly has no theta gradient".to_string()
                })?;
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(e) => return Err(WorkflowError::InvalidConfig { reason: e }.into()),
        }
    } else {
        baseline_cfg
    };

    let request = match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
            if config.noise_offset_column.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason:
                        "noise_offset_column is supported only for survival location-scale or marginal-slope"
                            .to_string(),
                }
                .into());
            }
            let weibull_seed = if survival_mode == SurvivalLikelihoodMode::Weibull
                && effective_timewiggle.is_none()
            {
                let scale = config
                    .baseline_scale
                    .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
                let shape = config.baseline_shape.unwrap_or(1.0);
                if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                    return Err(WorkflowError::InvalidConfig {
                        reason:
                            "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                                .to_string(),
                    }
                    .into());
                }
                Some((scale, shape))
            } else {
                None
            };
            FitRequest::SurvivalTransformation(SurvivalTransformationFitRequest {
                data: data.values.view(),
                spec: SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_codes.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: threshold_offset.clone(),
                    baseline_cfg,
                    likelihood_mode: survival_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    ridge_lambda: config.ridge_lambda,
                    penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
                },
                cache_session: None,
            })
        }
        SurvivalLikelihoodMode::LocationScale => {
            FitRequest::SurvivalLocationScale(build_location_scale_request(&baseline_cfg, true)?)
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            FitRequest::SurvivalMarginalSlope(build_marginal_slope_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::Latent => {
            FitRequest::LatentSurvival(build_latent_survival_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::LatentBinary => {
            FitRequest::LatentBinary(build_latent_binary_request(&baseline_cfg)?)
        }
    };

    Ok(MaterializedModel {
        request,
        inference_notes,
    })
}
