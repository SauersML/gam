use super::*;

pub(crate) fn blockwise_options_from_fit_args()
-> Result<gam::families::custom_family::BlockwiseFitOptions, String> {
    let options = gam::families::custom_family::BlockwiseFitOptions::default();
    Ok(options)
}

pub(crate) fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    // GUARD (#2030): the persisted geometry carrier's row-sized working
    // vectors (`working_weights` / `working_response`) MUST survive
    // compaction. `gam diagnose` (and post-fit `--alo` diagnostics) takes the
    // geometry ALO path in `run_diagnose.rs` whenever `unified.geometry` is
    // `Some`, handing `geom.working_weights` / `geom.working_response` to
    // `AloInput::from_geometry`. Zeroing them to length 0 makes ALO fail its
    // length-N validation ("ALO diagnostics require hessian_weights length N;
    // got 0") on EVERY standard fit, because the field is present-but-empty so
    // diagnose never falls through to its refit fallback. A prior fix carried
    // this warning; commit 57bdc8011 deleted it and reintroduced the defect.
    // The two carriers must also stay length-synchronized: `validate` bails
    // with "UnifiedFitResult geometry working_weights must match
    // inference.working_weights" if one is emptied while the other is not, so
    // we cannot drop just one copy. We therefore keep BOTH working vectors and
    // reclaim memory only from `reparam_qs` and the heavier `artifacts`.
    if let Some(inf) = fit.inference.as_mut() {
        inf.reparam_qs = None;
    }
    fit.artifacts = gam::estimate::FitArtifacts {
        pirls: None,
        ..Default::default()
    };
}

fn read_fit_request_json_file(path: &Path, label: &str) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {label} '{}': {error}", path.display()))
}

fn fit_request_document_from_fit_args(
    args: &FitArgs,
) -> Result<crate::config_resolve::FitRequestDocument, String> {
    let formula = args
        .formula_positional
        .as_deref()
        .ok_or_else(|| "fit requires FORMULA when --request is not provided".to_string())?;

    let ctn_stage1 = args
        .ctn_stage1
        .as_ref()
        .map(|path| {
            let raw = read_fit_request_json_file(path, "--ctn-stage1 JSON")?;
            serde_json::from_str::<crate::config_resolve::CtnStage1Document>(&raw)
                .map_err(|error| format!("invalid --ctn-stage1 JSON: {error}"))
        })
        .transpose()?;
    let precision_hyperpriors = args
        .precision_hyperpriors
        .as_ref()
        .map(|path| {
            let raw = read_fit_request_json_file(path, "--precision-hyperpriors JSON")?;
            serde_json::from_str(&raw)
                .map_err(|error| format!("invalid --precision-hyperpriors JSON: {error}"))
        })
        .transpose()?;
    let latent_coordinates = args
        .latent_coordinates
        .as_ref()
        .map(|path| {
            let raw = read_fit_request_json_file(path, "--latent-coordinates JSON")?;
            serde_json::from_str(&raw)
                .map_err(|error| format!("invalid --latent-coordinates JSON: {error}"))
        })
        .transpose()?;
    let analytic_penalties = args
        .analytic_penalties
        .as_ref()
        .map(|path| {
            let raw = read_fit_request_json_file(path, "--analytic-penalties JSON")?;
            serde_json::from_str(&raw)
                .map_err(|error| format!("invalid --analytic-penalties JSON: {error}"))
        })
        .transpose()?;
    let smooth_descriptors = args
        .smooth_descriptors
        .as_ref()
        .map(|path| {
            let raw = read_fit_request_json_file(path, "--smooth-descriptors JSON")?;
            serde_json::from_str(&raw)
                .map_err(|error| format!("invalid --smooth-descriptors JSON: {error}"))
        })
        .transpose()?;

    let frailty_kind = args.frailty_kind.map(|kind| match kind {
        FrailtyKindArg::GaussianShift => "gaussian-shift".to_string(),
        FrailtyKindArg::HazardMultiplier => "hazard-multiplier".to_string(),
    });
    let hazard_loading = args.hazard_loading.map(|loading| match loading {
        HazardLoadingArg::Full => "full".to_string(),
        HazardLoadingArg::LoadedVsUnloaded => "loaded-vs-unloaded".to_string(),
    });
    let config = crate::config_resolve::FitRequestConfigDocument {
        adaptive_regularization: Some(args.adaptive_regularization),
        baseline_makeham: args.baseline_makeham,
        baseline_rate: args.baseline_rate,
        baseline_scale: args.baseline_scale,
        baseline_shape: args.baseline_shape,
        baseline_target: Some(args.baseline_target.clone()),
        ctn_stage1,
        expectile_tau: args.expectile_tau,
        family: family_arg_canonical_name(args.family).map(str::to_string),
        firth: args.firth.then_some(true),
        frailty_kind,
        frailty_sd: args.frailty_sd,
        hazard_loading,
        latent_coordinates,
        logslope_formula: args.logslope_formula.clone(),
        negative_binomial_theta: args.negative_binomial_theta,
        noise_formula: args.predict_noise.clone(),
        noise_offset: args.noise_offset_column.clone(),
        offset: args.offset_column.clone(),
        analytic_penalties,
        pilot_subsample_threshold: Some(args.pilot_subsample_threshold),
        precision_hyperpriors,
        ridge_lambda: Some(args.ridge_lambda),
        scale_dimensions: args.scale_dimensions.then_some(true),
        sigma_time_degree: Some(args.sigma_time_degree),
        sigma_time_k: args.sigma_time_k,
        smooth_descriptors,
        survival_likelihood: Some(args.survival_likelihood.clone()),
        threshold_time_degree: Some(args.threshold_time_degree),
        threshold_time_k: args.threshold_time_k,
        time_basis: Some(args.time_basis.clone()),
        time_degree: Some(args.time_degree),
        time_num_internal_knots: Some(args.time_num_internal_knots),
        time_smooth_lambda: Some(args.time_smooth_lambda),
        transformation_normal: args.transformation_normal.then_some(true),
        weights: args.weights_column.clone(),
        z_column: args.z_column.clone(),
        ..crate::config_resolve::FitRequestConfigDocument::default()
    };
    crate::config_resolve::FitRequestDocument::new(formula, config)
}

pub(crate) fn resolve_fit_invocation(
    args: &FitArgs,
) -> Result<crate::config_resolve::ResolvedFitRequest, String> {
    if let Some(path) = args.request.as_ref() {
        let raw = read_fit_request_json_file(path, "--request document")?;
        crate::config_resolve::parse_fit_request_json(&raw)
    } else {
        crate::config_resolve::resolve_fit_request_document(fit_request_document_from_fit_args(
            args,
        )?)
    }
}

#[cfg(test)]
pub fn fit_config_from_fit_args(args: &FitArgs) -> Result<FitConfig, String> {
    resolve_fit_invocation(args).map(|resolved| resolved.fit_config)
}

pub(crate) fn fit_config_from_survival_args(args: &SurvivalArgs) -> Result<FitConfig, String> {
    let frailty = crate::config_resolve::resolve_cli_frailty_spec(
        cli_frailty_kind(args.frailty_kind),
        args.frailty_sd,
        cli_hazard_loading(args.hazard_loading),
        "survival fit",
    )?;
    FitConfig {
        link: args.link.clone(),
        offset_column: args.offset_column.clone(),
        weight_column: args.weights_column.clone(),
        noise_offset_column: args.noise_offset_column.clone(),
        baseline_target: args.baseline_target.clone(),
        baseline_scale: args.baseline_scale,
        baseline_shape: args.baseline_shape,
        baseline_rate: args.baseline_rate,
        baseline_makeham: args.baseline_makeham,
        time_basis: args.time_basis.clone(),
        time_degree: args.time_degree,
        time_num_internal_knots: args.time_num_internal_knots,
        time_smooth_lambda: args.time_smooth_lambda,
        survival_likelihood: args.survival_likelihood.clone(),
        survival_distribution: args.survival_distribution.clone(),
        threshold_time_k: args.threshold_time_k,
        threshold_time_degree: args.threshold_time_degree,
        sigma_time_k: args.sigma_time_k,
        sigma_time_degree: args.sigma_time_degree,
        noise_formula: args.predict_noise.clone(),
        logslope_formula: args.logslope_formula.clone(),
        z_column: args.z_column.clone(),
        scale_dimensions: args.scale_dimensions,
        spatial_optimization: SpatialLengthScaleOptimizationOptions {
            pilot_subsample_threshold: args.pilot_subsample_threshold,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
        ridge_lambda: args.ridge_lambda,
        frailty,
        ..FitConfig::default()
    }
    .resolve()
}

fn required_columns_for_resolved_fit(
    args: &FitArgs,
    parsed: &ParsedFormula,
    fit_config: &FitConfig,
) -> Result<Vec<String>, String> {
    let mut required = required_columns_for_fit(args, parsed)?
        .into_iter()
        .collect::<BTreeSet<_>>();

    if let Some(noise_formula) = fit_config.noise_formula.as_deref() {
        let (_, parsed_noise) =
            parse_matching_auxiliary_formula(noise_formula, &parsed.response, "noise_formula")?;
        required.extend(required_columns_for_formula(&parsed_noise)?);
    }
    if let Some(logslope_formula) = fit_config.logslope_formula.as_deref() {
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula,
            &parsed.response,
            "logslope_formula",
        )?;
        required.extend(required_columns_for_formula(&parsed_logslope)?);
    }
    required.extend(fit_config.z_column.iter().cloned());
    required.extend(fit_config.weight_column.iter().cloned());
    required.extend(fit_config.offset_column.iter().cloned());
    required.extend(fit_config.noise_offset_column.iter().cloned());

    if let Some(stage1) = fit_config.ctn_stage1.as_ref() {
        let stage1_formula = format!(
            "{} ~ {}",
            stage1.response_column, stage1.covariate_formula_rhs
        );
        let parsed_stage1 = parse_formula(&stage1_formula)?;
        required.extend(required_columns_for_formula(&parsed_stage1)?);
        required.extend(stage1.weight_column.iter().cloned());
        required.extend(stage1.offset_column.iter().cloned());
    }

    if let Some(descriptors) = fit_config
        .smooth_overrides
        .as_ref()
        .and_then(serde_json::Value::as_object)
    {
        for descriptor in descriptors
            .values()
            .filter_map(serde_json::Value::as_object)
        {
            if let Some(vars) = descriptor.get("vars").and_then(serde_json::Value::as_array) {
                required.extend(
                    vars.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string),
                );
            }
            if let Some(by) = descriptor.get("by").and_then(serde_json::Value::as_str) {
                required.insert(by.to_string());
            }
        }
    }

    Ok(required.into_iter().collect())
}

pub(crate) fn run_fit(args: FitArgs) -> Result<(), String> {
    let resolved_invocation = resolve_fit_invocation(&args)?;
    let formula_text = resolved_invocation.formula;
    let fit_config = resolved_invocation.fit_config;
    let parsed = parse_formula(&formula_text)?;
    validate_fit_args_preflight(&args, &parsed, &fit_config)?;
    let formula_link = parsed.linkspec.clone();
    let effective_link_arg = formula_link.as_ref().map(|s| s.link.clone());
    let effective_mixture_rho = formula_link.as_ref().and_then(|s| s.mixture_rho.clone());
    let effective_sas_init = formula_link.as_ref().and_then(|s| s.sas_init.clone());
    let effective_beta_logistic_init = formula_link
        .as_ref()
        .and_then(|s| s.beta_logistic_init.clone());
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        validate_cli_firth_configuration(CliFirthValidation {
            enabled: args.firth,
            family: LikelihoodSpec::royston_parmar(),
            predict_noise: args.predict_noise.is_some(),
            is_survival: true,
            link_choice: None,
        })?;
        let rhs = formula_rhs_text(&formula_text)?;
        let formula_surv = parsed.survivalspec.clone();
        let surv_args = SurvivalArgs {
            data: args.data.clone(),
            entry,
            exit,
            event,
            // `entry == None` = right-censored shorthand `Surv(time, event)`;
            // entry times are synthesized as zero at materialization time.
            formula: rhs,
            predict_noise: fit_config.noise_formula.clone(),
            survival_likelihood: fit_config.survival_likelihood.clone(),
            survival_distribution: formula_surv
                .as_ref()
                .and_then(|s| s.survival_distribution.clone())
                .unwrap_or_else(|| "gaussian".to_string()),
            link: effective_link_arg.clone(),
            mixture_rho: effective_mixture_rho.clone(),
            sas_init: effective_sas_init.clone(),
            beta_logistic_init: effective_beta_logistic_init.clone(),
            survival_time_anchor: args.survival_time_anchor,
            baseline_target: fit_config.baseline_target.clone(),
            baseline_scale: fit_config.baseline_scale,
            baseline_shape: fit_config.baseline_shape,
            baseline_rate: fit_config.baseline_rate,
            baseline_makeham: fit_config.baseline_makeham,
            time_basis: fit_config.time_basis.clone(),
            time_degree: fit_config.time_degree,
            time_num_internal_knots: fit_config.time_num_internal_knots,
            time_smooth_lambda: fit_config.time_smooth_lambda,
            ridge_lambda: fit_config.ridge_lambda,
            threshold_time_k: fit_config.threshold_time_k,
            threshold_time_degree: fit_config.threshold_time_degree,
            sigma_time_k: fit_config.sigma_time_k,
            sigma_time_degree: fit_config.sigma_time_degree,
            scale_dimensions: fit_config.scale_dimensions,
            pilot_subsample_threshold: args.pilot_subsample_threshold,
            out: args.out.clone(),
            logslope_formula: fit_config.logslope_formula.clone(),
            z_column: fit_config.z_column.clone(),
            weights_column: fit_config.weight_column.clone(),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
            frailty_kind: args.frailty_kind,
            frailty_sd: args.frailty_sd,
            hazard_loading: args.hazard_loading,
        };
        return run_survival(surv_args);
    }
    // Multinomial is a softmax multi-output family (categorical response, K-1
    // active-class linear predictors): it owns its own dataset load (with the
    // response forced to a factor) and persistence envelope, so dispatch it
    // before the scalar-response standard path. The stale note below about "the
    // CLI has no multinomial family" no longer holds for this early return.
    if fit_config.family.as_deref() == Some("multinomial") {
        return run_fit_multinomial(&args, &parsed, &formula_text, &fit_config);
    }
    let requested_columns = required_columns_for_resolved_fit(&args, &parsed, &fit_config)?;
    // Force `group(g)` / `factor(g)` / `re(g)` grouping columns to a factor
    // encoding even when their labels are numeric. An untyped CSV cannot carry
    // the typed-frame categorical sentinel the Python path uses, so without this
    // a numeric-coded grouping column would be demoted to a single continuous
    // ramp — a strictly lower-capacity design than `gamfit.fit` builds for the
    // same data. The CLI has no multinomial family, so the response is never a
    // forced factor here; only the random-effect roles are. Bare `+ x` and
    // `s(x)` stay value-inferred, so a continuous integer covariate is untouched.
    let ds = load_fit_dataset_with_roles(&args.data, &requested_columns, &parsed, false)?;
    require_dataset_rows("fit", &args.data, ds.values.nrows())?;

    let col_map = ds.column_map();

    let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;
    let y = ds.values.column(y_col);
    // Reject a constant response upfront with a clear message rather than
    // letting REML fail with the cryptic
    //   "no candidate seeds passed outer startup validation (standard REML)"
    // which gave the user no idea what was wrong with their data.
    {
        let mut seen_finite: Option<f64> = None;
        let mut all_one_value = true;
        for &v in y.iter() {
            if !v.is_finite() {
                continue;
            }
            match seen_finite {
                None => seen_finite = Some(v),
                Some(s) if (s - v).abs() < 1e-12 => {}
                _ => {
                    all_one_value = false;
                    break;
                }
            }
        }
        if all_one_value && seen_finite.is_some() {
            let value = seen_finite.unwrap();
            return Err(format!(
                "response column '{}' is constant (every finite value equals {value}) — \
                 there is nothing to fit. Check the data: this is usually a column-mapping mistake \
                 or a degenerate subset.",
                parsed.response
            ));
        }
    }
    let mut inference_notes: Vec<String> = Vec::new();

    if fit_config.transformation_normal {
        if fit_config.noise_offset_column.is_some() {
            return Err(
                "--noise-offset-column is not supported with --transformation-normal".to_string(),
            );
        }
        let y = y.to_owned();
        return run_fit_transformation_normal(
            &args,
            &ds,
            &col_map,
            &parsed,
            &formula_text,
            &y,
            &mut inference_notes,
        );
    }

    if fit_config.logslope_formula.is_some() || fit_config.z_column.is_some() {
        if fit_config.logslope_formula.is_none() || fit_config.z_column.is_none() {
            return Err("--logslope-formula and --z-column must be provided together".to_string());
        }
        let y = y.to_owned();
        return run_fit_bernoulli_marginal_slope(
            &args,
            &ds,
            &col_map,
            &parsed,
            &formula_text,
            &y,
            &mut inference_notes,
        );
    }

    // `--expectile-tau` only has meaning under `--family expectile`; reject the
    // combination upfront rather than silently ignoring the asymmetry.
    if fit_config.expectile_tau.is_some() && fit_config.family.as_deref() != Some("expectile") {
        return Err(
            "--expectile-tau requires --family expectile (the asymmetry is only used by the \
             expectile estimator)"
                .to_string(),
        );
    }
    // Expectile (Newey–Powell LAWS) family (#1777): an OUTER estimator that wraps
    // the standard Gaussian-identity GAM with iterative asymmetric reweighting.
    // It is dispatched here — before `resolve_family`/the standard fit, neither of
    // which knows the expectile family — through the same shared seam the Python
    // FFI and the in-process `fit_from_formula` use, so the CLI reaches the same
    // estimator instead of failing with `unknown family 'expectile'`.
    if fit_config.family.as_deref() == Some("expectile") {
        return run_canonical_standard_fit(&args, &ds, &parsed, &formula_text, &fit_config);
    }

    // Every single-parameter formula fit is owned end-to-end by gam-models.
    // The remainder of this function is only the still-specialized
    // location-scale presentation/persistence adapter.
    if fit_config.noise_formula.is_none() {
        return run_canonical_standard_fit(&args, &ds, &parsed, &formula_text, &fit_config);
    }

    // Specialized location-scale fitting owns the response beyond this point.
    // Canonical standard/expectile fits returned above without making this
    // redundant copy; their materializer creates the one request backing.
    let y = y.to_owned();

    let link_choice = parse_link_choice(effective_link_arg.as_deref(), false)?;
    let mixture_linkspec = if let Some(choice) = link_choice.as_ref() {
        if let Some(components) = choice.mixture_components.as_ref() {
            let expected = components.len().saturating_sub(1);
            let initial_rho = if let Some(raw) = effective_mixture_rho.as_deref() {
                let vals = crate::config_resolve::parse_comma_f64(raw, "link(rho=...)")?;
                if vals.len() != expected {
                    return Err(format!(
                        "link(rho=...) length mismatch: expected {expected}, got {}",
                        vals.len()
                    ));
                }
                Array1::from_vec(vals)
            } else {
                Array1::zeros(expected)
            };
            Some(MixtureLinkSpec {
                components: components.clone(),
                initial_rho,
            })
        } else {
            if effective_mixture_rho.is_some() {
                return Err(
                    "link(rho=...) requires link(type=blended(...)/mixture(...))".to_string(),
                );
            }
            None
        }
    } else {
        if effective_mixture_rho.is_some() {
            return Err("link(rho=...) requires link(type=blended(...)/mixture(...))".to_string());
        }
        None
    };
    if let Some(choice) = link_choice.as_ref() {
        if choice.mixture_components.is_none() && choice.link == LinkFunction::Sas {
            if effective_beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            if let Some(raw) = effective_sas_init.as_deref() {
                let vals = crate::config_resolve::parse_comma_f64(raw, "link(sas_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(sas_init=...) expects two values: epsilon,log_delta (got {})",
                        vals.len()
                    ));
                }
                Some(SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                })
            } else {
                Some(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
            }
        } else if choice.mixture_components.is_none() && choice.link == LinkFunction::BetaLogistic {
            if effective_sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            if let Some(raw) = effective_beta_logistic_init.as_deref() {
                let vals =
                    crate::config_resolve::parse_comma_f64(raw, "link(beta_logistic_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(beta_logistic_init=...) expects two values: epsilon,delta (got {})",
                        vals.len()
                    ));
                }
                Some(SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                })
            } else {
                Some(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
            }
        } else {
            if effective_sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            if effective_beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            None
        }
    } else {
        if effective_sas_init.is_some() {
            return Err("link(sas_init=...) requires link(type=sas)".to_string());
        }
        if effective_beta_logistic_init.is_some() {
            return Err(
                "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
            );
        }
        None
    };

    let y_kind = response_column_kind(&ds, y_col);
    let family = gam::families::fit_orchestration::resolve_family(
        fit_config.family.as_deref(),
        fit_config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation (Gamma `y > 0`, Poisson /
    // NegBin / Tweedie `y ≥ 0`, Beta `y ∈ (0,1)`). Owned by `ResponseFamily`
    // so the CLI, the formula API, and the external-design GLM path all
    // produce identical messages.
    if let Err(violation) = family.response.validate_response_support(y.view()) {
        return Err(violation.message_for(&parsed.response));
    }
    if link_choice.is_none() && fit_config.family.is_none() {
        if is_binary_response(y.view()) {
            inference_notes.push(format!(
                "Inferred binomial-logit family for response '{}' because all values are binary {{0,1}}. Override with link(type=...).",
                parsed.response
            ));
        } else {
            inference_notes.push(format!(
                "Inferred gaussian-identity family for response '{}' because values are not strictly binary. Override with link(type=...).",
                parsed.response
            ));
        }
    }
    let formula_linkwiggle = parsed.linkwiggle.clone();
    if parsed.timewiggle.is_some() {
        return Err("timewiggle(...) is only supported for survival models".to_string());
    }
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), link_choice.as_ref());
    let learn_linkwiggle = effective_linkwiggle.is_some();
    if learn_linkwiggle {
        require_likelihood_spec_supports_joint_wiggle(&family, "linkwiggle(...)")?;
        if let Some(choice) = link_choice.as_ref() {
            require_linkchoice_supports_joint_wiggle(choice, "linkwiggle(...)")?;
        }
    }
    let mean_only_flexible_linkwiggle = link_choice
        .as_ref()
        .is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible));
    let mean_only_binomial_linkwiggle = fit_config.noise_formula.is_none()
        && binomial_mean_linkwiggle_supports_family(&family, link_choice.as_ref());
    if learn_linkwiggle
        && fit_config.noise_formula.is_none()
        && !mean_only_flexible_linkwiggle
        && !mean_only_binomial_linkwiggle
    {
        return Err(
            "link wiggle without --predict-noise currently supports binomial mean fitting with non-flexible links and binomial flexible(...) mean fitting"
                .to_string(),
        );
    }
    let noise_formula_raw = fit_config
        .noise_formula
        .as_deref()
        .expect("the standard path returned before location-scale dispatch");
    run_fitwith_predict_noise(
        &args,
        &ds,
        &col_map,
        &parsed,
        &y,
        family,
        link_choice.as_ref(),
        mixture_linkspec.as_ref(),
        effective_linkwiggle.as_ref(),
        &mut inference_notes,
        noise_formula_raw,
        &formula_text,
    )
}

fn standard_fast_path_feature_columns(parsed: &ParsedFormula) -> Vec<String> {
    parsed
        .terms
        .iter()
        .find_map(|term| match term {
            ParsedTerm::Smooth { vars, .. } => Some(vars.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn canonical_standard_fit_error(error: WorkflowError) -> String {
    let detail = error.to_string();
    if detail.contains("Parameter constraint violation") && detail.contains("no candidate seeds") {
        format!(
            "standard term fit failed: every candidate fit violates the requested coefficient \
             constraint. Remove the constraint, change its direction/bounds, or check the data. \
             Underlying error: {error}"
        )
    } else {
        format!("standard formula fit failed: {error}")
    }
}

fn run_canonical_standard_fit(
    args: &FitArgs,
    dataset: &Dataset,
    parsed: &ParsedFormula,
    formula: &str,
    fit_config: &FitConfig,
) -> Result<(), String> {
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] canonical formula fit start n={}",
        dataset.values.nrows()
    );
    let outcome = fit_from_formula_with_notes(formula, dataset, fit_config)
        .map_err(canonical_standard_fit_error)?;
    log::info!(
        "[PHASE] canonical formula fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );
    print_inference_summary(&outcome.inference_notes);

    match outcome.result {
        FitResult::Standard(mut result) => {
            let family = result
                .fit
                .likelihood_family
                .clone()
                .unwrap_or_else(LikelihoodSpec::gaussian_identity);
            let model_label = if fit_config.family.as_deref() == Some("expectile") {
                "expectile"
            } else {
                "standard"
            };
            let spatial_warnings =
                collect_smooth_structure_warnings(&result.resolvedspec, &dataset.headers, "model");
            print_spatial_aniso_scales(&result.resolvedspec);
            cli_out!(
                "{} fit | family={} | status={} | iterations={} | terms={} | edf={:.3} | loglik={:.6e} | objective={:.6e}",
                model_label,
                family.name(),
                result.fit.convergence_evidence().inner_status().label(),
                result.fit.outer_iterations,
                result.resolvedspec.smooth_terms.len() + result.resolvedspec.linear_terms.len(),
                result.fit.edf_total().unwrap_or(f64::NAN),
                result.fit.log_likelihood,
                result.fit.reml_score,
            );
            if let Some(out) = args.out.as_ref() {
                compact_fit_result_for_batch(&mut result.fit);
                let mut payload = assemble_standard_payload(StandardPayloadInputs {
                    formula: formula.to_string(),
                    dataset,
                    fit_config,
                    result,
                })?;
                payload.group_metadata = fit_config.group_metadata.clone();
                payload.inference_notes = outcome.inference_notes;
                write_payload_json(out, payload)?;
            }
            emit_smooth_structure_warnings("fit-end", &spatial_warnings);
            Ok(())
        }
        FitResult::SplineScan(scan) => {
            let feature_column = standard_fast_path_feature_columns(parsed)
                .into_iter()
                .next()
                .ok_or_else(|| {
                    "canonical spline-scan result has no smooth feature in its formula".to_string()
                })?;
            cli_out!(
                "spline-scan fit | knots={} | edf={:.3} | sigma2={:.6e} | log_lambda={:.4} | reml={:.6e}",
                scan.knots.len(),
                scan.edf(),
                scan.sigma2,
                scan.log_lambda,
                scan.restricted_loglik,
            );
            if let Some(out) = args.out.as_ref() {
                let mut payload = assemble_spline_scan_payload(
                    formula.to_string(),
                    feature_column,
                    &scan,
                    dataset.schema.clone(),
                    dataset.headers.clone(),
                    dataset.feature_ranges(),
                );
                payload.group_metadata = fit_config.group_metadata.clone();
                payload.inference_notes = outcome.inference_notes;
                write_payload_json(out, payload)?;
            }
            Ok(())
        }
        FitResult::ResidualCascade(cascade) => {
            let feature_columns = standard_fast_path_feature_columns(parsed);
            if feature_columns.is_empty() {
                return Err(
                    "canonical residual-cascade result has no smooth features in its formula"
                        .to_string(),
                );
            }
            cli_out!(
                "residual-cascade fit | levels={} | centers={} | sigma2={:.6e} | \
                 log_lambda={:.4} | reml={:.6e} | rel_resid={:.2e}",
                cascade.num_levels(),
                cascade.num_centers(),
                cascade.sigma2,
                cascade.log_lambda,
                cascade.restricted_loglik,
                cascade.certificate.solve_rel_residual,
            );
            if let Some(out) = args.out.as_ref() {
                let mut payload = assemble_residual_cascade_payload(
                    formula.to_string(),
                    feature_columns,
                    &cascade,
                    dataset.schema.clone(),
                    dataset.headers.clone(),
                    dataset.feature_ranges(),
                )?;
                payload.group_metadata = fit_config.group_metadata.clone();
                payload.inference_notes = outcome.inference_notes;
                write_payload_json(out, payload)?;
            }
            Ok(())
        }
        _ => Err(
            "canonical standard fit returned a non-standard model; specialized formula families \
             must be dispatched before the standard service"
                .to_string(),
        ),
    }
}
pub(crate) fn run_fit_bernoulli_marginal_slope(
    args: &FitArgs,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    formula_text: &str,
    y: &Array1<f64>,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    if !is_binary_response(y.view()) {
        return Err(
            "bernoulli marginal-slope fitting requires a binary {0,1} response".to_string(),
        );
    }
    if args.firth {
        inference_notes.push(
            "--firth is redundant for bernoulli marginal-slope: the robust Jeffreys/Firth stabilizer is installed by policy"
                .to_string(),
        );
    }
    if args.predict_noise.is_some() {
        return Err(
            "--predict-noise cannot be combined with --logslope-formula/--z-column".to_string(),
        );
    }
    let logslope_formula_raw = args
        .logslope_formula
        .as_deref()
        .ok_or_else(|| "missing --logslope-formula".to_string())?;
    let z_column = args
        .z_column
        .as_ref()
        .ok_or_else(|| "missing --z-column".to_string())?;
    let base_link = resolve_bernoulli_marginal_slope_base_link(
        parsed.linkspec.as_ref(),
        "bernoulli marginal-slope",
    )?;
    let (logslope_formula, parsed_logslope) = parse_matching_auxiliary_formula(
        logslope_formula_raw,
        &parsed.response,
        "--logslope-formula",
    )?;
    if parsed_logslope.linkspec.is_some() {
        return Err(
            "link(...) is not supported in --logslope-formula for the bernoulli marginal-slope family"
                .to_string(),
        );
    }
    validate_marginal_slope_z_column_exclusion(
        parsed,
        &parsed_logslope,
        z_column,
        "bernoulli marginal-slope",
        "--logslope-formula",
    )?;

    // Marginal-slope formulas may reference the literal placeholder `z` to
    // bind to the auxiliary score supplied via --z-column. Alias `z` in the
    // column map to the actual `z_column` index so build_termspec can resolve
    // it without the user having to rename their data column.
    let col_map_with_z_alias = column_map_with_alias(col_map, "z", z_column);
    let col_map_for_termspec: &HashMap<String, usize> = &col_map_with_z_alias;
    let mut marginalspec = build_termspec(
        &parsed.terms,
        ds,
        col_map_for_termspec,
        inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    let mut logslopespec = build_termspec(
        &parsed_logslope.terms,
        ds,
        col_map_for_termspec,
        inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut marginalspec);
        enable_scale_dimensions(&mut logslopespec);
    }
    let mut spatial_usagewarnings =
        collect_smooth_structure_warnings(&marginalspec, &ds.headers, "marginal model");
    spatial_usagewarnings.extend(collect_smooth_structure_warnings(
        &logslopespec,
        &ds.headers,
        "logslope model",
    ));
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let z_col = resolve_role_col(col_map, z_column, "z")?;
    let z = ds.values.column(z_col).to_owned();
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let marginal_offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let logslope_offset = resolve_offset_column(ds, col_map, args.noise_offset_column.as_deref())?;
    let frailty = fit_frailty_spec_from_args(args, "bernoulli marginal-slope")?
        .resolve_fixed_gaussian_shift("bernoulli marginal-slope")?;
    let routed_deviations = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
    )?;
    let routed_link_dev = routed_deviations.link_dev;
    let routed_score_warp = routed_deviations.score_warp;
    let requested_flex = routed_link_dev.is_some() || routed_score_warp.is_some();
    inference_notes.push(
        "bernoulli marginal-slope auto-detects the latent score law: standard-normal calibration is used only when z passes diagnostics; otherwise the fitted empirical latent measure is carried through the marginal calibration"
            .to_string(),
    );
    if parsed.linkwiggle.is_some() {
        inference_notes.push(
            "bernoulli marginal-slope routes main-formula linkwiggle(...) into its anchored internal link-deviation block"
                .to_string(),
        );
    }
    if parsed_logslope.linkwiggle.is_some() {
        inference_notes.push(
            "bernoulli marginal-slope routes --logslope-formula linkwiggle(...) into its anchored internal score-warp block"
                .to_string(),
        );
    }
    inference_notes.push(
        "bernoulli marginal-slope uses link(type=probit) for the calibrated marginal target"
            .to_string(),
    );
    if !requested_flex {
        inference_notes.push(
            "bernoulli marginal-slope rigid probit mode is exact under the active latent measure"
                .to_string(),
        );
    } else {
        inference_notes.push(
            "bernoulli marginal-slope flexible score/link mode uses a calibrated de-nested cubic transport kernel: closed-form affine cells plus transported quartic/sextic non-affine cells with analytic gradients and Hessians"
                .to_string(),
        );
    }
    let mut options = blockwise_options_from_fit_args()?;
    options.compute_covariance = true;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] bernoulli-margslope fit start n={}",
        ds.values.nrows()
    );
    let solved = match fit_model(FitRequest::BernoulliMarginalSlope(
        BernoulliMarginalSlopeFitRequest {
            data: ds.values.view(),
            spec: BernoulliMarginalSlopeTermSpec {
                y: y.clone(),
                weights,
                z,
                base_link: base_link.clone(),
                marginalspec: marginalspec.clone(),
                logslopespec: logslopespec.clone(),
                marginal_offset,
                logslope_offset,
                frailty: frailty.clone(),
                score_warp: routed_score_warp,
                link_dev: routed_link_dev,
                latent_z_policy: LatentZPolicy::default(),
                // This CLI path fits the marginal-slope model directly from a raw
                // `--z-column`; there is no in-process CTN Stage-1 chain to
                // cross-fit, so the score-influence projection is inactive and
                // the free-warp `score_warp` is the fallback basis (#461 §5).
                score_influence_jacobian: None,
            },
            options,
            kappa_options: kappa_options.clone(),
            policy: gam::ResourcePolicy::default_library(),
        },
    )) {
        Ok(FitResult::BernoulliMarginalSlope(result)) => {
            log::info!(
                "[PHASE] bernoulli-margslope fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            for w in &result.cross_block_warnings {
                cli_out!(
                    "WARNING: cross-block identifiability dropped flex block '{}' \
                     (anchors: {}). {}",
                    w.candidate_label,
                    w.anchor_summary,
                    w.reason
                );
            }
            result
        }
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal bernoulli marginal-slope workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(format!("bernoulli marginal-slope fit failed: {e}"));
        }
    };

    let frozen_marginal =
        freeze_term_collection_from_design(&solved.marginalspec_resolved, &solved.marginal_design)
            .map_err(|e| e.to_string())?;
    let frozen_logslope =
        freeze_term_collection_from_design(&solved.logslopespec_resolved, &solved.logslope_design)
            .map_err(|e| e.to_string())?;
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_BERNOULLI_MARGINAL_SLOPE,
        solved.fit.outer_iterations,
        solved.fit.convergence_evidence().inner_status().label()
    );
    print_spatial_aniso_scales(&solved.marginalspec_resolved);
    print_spatial_aniso_scales(&solved.logslopespec_resolved);

    if let Some(out) = args.out.as_ref() {
        let save_frailty = match (&frailty, solved.gaussian_frailty_sd) {
            (
                gam::families::survival::lognormal_kernel::FrailtySpec::GaussianShift {
                    sigma_fixed: None,
                },
                Some(learned),
            ) => gam::families::survival::lognormal_kernel::FrailtySpec::GaussianShift {
                sigma_fixed: Some(learned),
            },
            _ => frailty,
        };
        let mut model = build_bernoulli_marginal_slope_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            logslope_formula,
            z_column.clone(),
            ds.headers.clone(),
            ds.feature_ranges(),
            frozen_marginal,
            frozen_logslope,
            solved.fit,
            solved.marginal_design.design.ncols(),
            solved.baseline_marginal,
            solved.baseline_logslope,
            SavedLatentZNormalization {
                mean: solved.z_normalization.mean,
                sd: solved.z_normalization.sd,
            },
            solved.latent_measure.clone(),
            solved.latent_z_rank_int_calibration.clone(),
            solved.latent_z_conditional_calibration.clone(),
            solved.score_warp_runtime.as_ref(),
            solved.link_dev_runtime.as_ref(),
            base_link,
            save_frailty,
        )?;
        model.offset_column = args.offset_column.clone();
        model.noise_offset_column = args.noise_offset_column.clone();
        write_model_json(out, &model)?;
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    Ok(())
}

pub(crate) fn run_fit_transformation_normal(
    args: &FitArgs,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    formula_text: &str,
    y: &Array1<f64>,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    if args.firth {
        return Err("--firth is not supported for the transformation-normal family".to_string());
    }
    if parsed.linkspec.is_some() {
        return Err("link(...) is not supported for the transformation-normal family".to_string());
    }
    if parsed.linkwiggle.is_some() {
        return Err(
            "linkwiggle(...) is not supported for the transformation-normal family".to_string(),
        );
    }
    if args.predict_noise.is_some() {
        return Err("--predict-noise cannot be combined with --transformation-normal".to_string());
    }

    let mut covariate_spec = build_termspec(
        &parsed.terms,
        ds,
        col_map,
        inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut covariate_spec);
    }

    let spatial_usagewarnings =
        collect_smooth_structure_warnings(&covariate_spec, &ds.headers, "transformation-normal");
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let options = blockwise_options_from_fit_args()?;
    let config = TransformationNormalConfig::default();
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };

    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] CTN(transformation-normal) fit start n={} cov_terms={}",
        ds.values.nrows(),
        covariate_spec.linear_terms.len()
            + covariate_spec.smooth_terms.len()
            + covariate_spec.random_effect_terms.len()
    );
    let solved = match fit_model(FitRequest::TransformationNormal(
        TransformationNormalFitRequest {
            data: ds.values.view(),
            response: y.clone(),
            weights,
            offset,
            covariate_spec: covariate_spec.clone(),
            config,
            options,
            kappa_options: kappa_options.clone(),
            warm_start: None,
        },
    )) {
        Ok(FitResult::TransformationNormal(result)) => result,
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal transformation-normal workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(format!("transformation-normal fit failed: {e}"));
        }
    };
    log::info!(
        "[PHASE] CTN(transformation-normal) fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );

    let frozen_covariate = solved.covariate_spec_resolved.clone();
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_TRANSFORMATION_NORMAL,
        solved.fit.outer_iterations,
        solved.fit.convergence_evidence().inner_status().label()
    );
    print_spatial_aniso_scales(&solved.covariate_spec_resolved);

    if let Some(out) = args.out.as_ref() {
        let mut model = build_transformation_normal_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            ds.headers.clone(),
            ds.feature_ranges(),
            frozen_covariate,
            solved.fit,
            &solved.family,
            solved.score_calibration,
        );
        model.offset_column = args.offset_column.clone();
        model.noise_offset_column = args.noise_offset_column.clone();
        write_model_json(out, &model)?;
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    Ok(())
}

pub(crate) fn run_fitwith_predict_noise(
    args: &FitArgs,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    y: &Array1<f64>,
    family: LikelihoodSpec,
    link_choice: Option<&LinkChoice>,
    mixture_linkspec: Option<&MixtureLinkSpec>,
    formula_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    inference_notes: &mut Vec<String>,
    noise_formula_raw: &str,
    formula_text: &str,
) -> Result<(), String> {
    let (noise_formula, parsed_noise) =
        parse_matching_auxiliary_formula(noise_formula_raw, &parsed.response, "--predict-noise")?;
    validate_auxiliary_formula_controls(&parsed_noise, "--predict-noise")?;
    let mut noisespec = build_termspec(
        &parsed_noise.terms,
        ds,
        col_map,
        inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    let mut meanspec = build_termspec(
        &parsed.terms,
        ds,
        col_map,
        inference_notes,
        &gam::ResourcePolicy::default_library(),
    )?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut meanspec);
        enable_scale_dimensions(&mut noisespec);
    }
    let mut spatial_usagewarnings =
        collect_smooth_structure_warnings(&meanspec, &ds.headers, "mean model");
    spatial_usagewarnings.extend(collect_smooth_structure_warnings(
        &noisespec,
        &ds.headers,
        "noise model",
    ));
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let mean_offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(ds, col_map, args.noise_offset_column.as_deref())?;
    if family == LikelihoodSpec::gaussian_identity() {
        // Response standardization (and the inverse remap back to raw units) now
        // lives in the single Gaussian location-scale model entry point
        // (`fit_gaussian_location_scale_model`), so the CLI hands it the RAW
        // response and receives coefficients/covariance/summary already in raw
        // response units — there is no CLI-side prefit or post-fit rescaling.
        let options = blockwise_options_from_fit_args()?;
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] gaussian-location-scale fit start n={}",
            ds.values.nrows()
        );
        let solved = match fit_model(FitRequest::GaussianLocationScale(
            GaussianLocationScaleFitRequest {
                data: ds.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y: y.clone(),
                    weights: weights.clone(),
                    meanspec: meanspec.clone(),
                    log_sigmaspec: noisespec.clone(),
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: formula_linkwiggle.cloned().map(|cfg| LinkWiggleConfig {
                    degree: cfg.degree,
                    num_internal_knots: cfg.num_internal_knots,
                    penalty_orders: cfg.penalty_orders,
                    double_penalty: cfg.double_penalty,
                }),
                options,
                kappa_options: kappa_options.clone(),
            },
        )) {
            Ok(FitResult::GaussianLocationScale(result)) => {
                log::info!(
                    "[PHASE] gaussian-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal gaussian location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(format!("gaussian location-scale fit failed: {e}"));
            }
        };
        let wiggle_meta = match (
            solved.wiggle_knots,
            solved.wiggle_degree,
            solved.beta_link_wiggle,
        ) {
            (Some(knots), Some(degree), Some(beta)) => Some((knots, degree, beta)),
            _ => None,
        };
        // Capture the response standardization factor before moving `solved.fit`
        // out below; the Gaussian σ floor is persisted at
        // `response_scale·LOGB_SIGMA_FLOOR` so prediction stays
        // response-scale-equivariant (#884).
        let gaussian_response_scale = solved.response_scale;
        let BlockwiseTermFitResult {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        } = solved.fit;
        let frozen_meanspec = freeze_term_collection_from_design(&meanspec_resolved, &mean_design)
            .map_err(|e| e.to_string())?;
        let frozen_noisespec =
            freeze_term_collection_from_design(&noisespec_resolved, &noise_design)
                .map_err(|e| e.to_string())?;
        cli_out!(
            "model fit complete | family={} | outer_iter={} | status={}",
            FAMILY_GAUSSIAN_LOCATION_SCALE,
            fit.outer_iterations,
            fit.convergence_evidence().inner_status().label()
        );
        print_spatial_aniso_scales(&meanspec_resolved);
        print_spatial_aniso_scales(&noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            // `fit` already carries raw-unit coefficients, covariance, and a
            // raw-unit residual-scale summary (the standardization and its
            // inverse remap live in `fit_gaussian_location_scale_model`), so the
            // save path persists them verbatim and records the actual
            // `gaussian_response_scale` — predict reconstructs raw σ as
            // `response_scale·0.01 + exp(Xβ)`, scaling the σ floor with the
            // response so predictive σ is response-scale-equivariant (#884). The
            // unrelated `compact_saved_multiblock_fit_result` scalar below is the
            // fit's dispersion summary (1.0 for Gaussian), not the response scale.
            let fit_result = compact_saved_multiblock_fit_result(
                fit.blocks.clone(),
                fit.lambdas.clone(),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_corrected.clone(),
                fit.geometry.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?,
            );
            let resolved_base_link = link_choice
                .map(|choice| {
                    crate::config_resolve::effective_link_to_standard(
                        choice.link,
                        "gaussian location-scale base link",
                    )
                    .map(InverseLink::Standard)
                })
                .transpose()?;
            // Knots/coefficients are already in raw response units.
            let wiggle = wiggle_meta.map(|(knots, degree, beta_link_wiggle)| LocationScaleWiggle {
                knots: knots.to_vec(),
                degree,
                beta_link_wiggle,
            });
            let payload = assemble_location_scale_payload(
                LocationScaleInputs {
                    formula: formula_text.to_string(),
                    data_schema: ds.schema.clone(),
                    noise_formula: noise_formula.clone(),
                    resolved_termspec: frozen_meanspec,
                    resolved_termspec_noise: frozen_noisespec,
                    fit_result,
                    beta_noise: fit
                        .block_by_role(BlockRole::Scale)
                        .map(|block| block.beta.to_vec()),
                    wiggle,
                },
                LocationScaleResponse::Gaussian {
                    response_scale: gaussian_response_scale,
                    base_link: resolved_base_link,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: args.offset_column.clone(),
                    noise_offset_column: args.noise_offset_column.clone(),
                },
            )?;
            write_payload_json(out, payload)?;
        }
        emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
        return Ok(());
    }

    // Genuine-dispersion mean families (NegativeBinomial / Gamma / Beta /
    // Tweedie): `noise_formula` models the overdispersion channel (#913).
    if let Some(kind) = dispersion_location_scale_kind_for_cli(&family.response) {
        if formula_linkwiggle.is_some() {
            return Err(format!(
                "link-wiggle is not supported for {} location-scale models",
                kind.family_tag()
            ));
        }
        let options = blockwise_options_from_fit_args()?;
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] dispersion-location-scale ({}) fit start n={}",
            kind.family_tag(),
            ds.values.nrows()
        );
        let solved = match fit_model(FitRequest::DispersionLocationScale(
            DispersionLocationScaleFitRequest {
                data: ds.values.view(),
                spec: gam::gamlss::DispersionGlmLocationScaleTermSpec {
                    kind,
                    y: y.clone(),
                    weights: weights.clone(),
                    meanspec: meanspec.clone(),
                    log_dispspec: noisespec.clone(),
                    mean_offset,
                    log_disp_offset: noise_offset,
                },
                options,
                kappa_options: kappa_options.clone(),
            },
        )) {
            Ok(FitResult::DispersionLocationScale(result)) => {
                log::info!(
                    "[PHASE] dispersion-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal dispersion location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(format!("dispersion location-scale fit failed: {e}"));
            }
        };
        let fit = solved.fit.fit;
        let frozen_meanspec = freeze_term_collection_from_design(
            &solved.fit.meanspec_resolved,
            &solved.fit.mean_design,
        )
        .map_err(|e| e.to_string())?;
        let frozen_noisespec = freeze_term_collection_from_design(
            &solved.fit.noisespec_resolved,
            &solved.fit.noise_design,
        )
        .map_err(|e| e.to_string())?;
        cli_out!(
            "model fit complete | family={} | outer_iter={} | status={}",
            kind.family_tag(),
            fit.outer_iterations,
            fit.convergence_evidence().inner_status().label()
        );
        print_spatial_aniso_scales(&solved.fit.meanspec_resolved);
        print_spatial_aniso_scales(&solved.fit.noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            let fit_result = compact_saved_multiblock_fit_result(
                fit.blocks.clone(),
                fit.lambdas.clone(),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_corrected.clone(),
                fit.geometry.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?,
            );
            let payload = assemble_location_scale_payload(
                LocationScaleInputs {
                    formula: formula_text.to_string(),
                    data_schema: ds.schema.clone(),
                    noise_formula: noise_formula.clone(),
                    resolved_termspec: frozen_meanspec,
                    resolved_termspec_noise: frozen_noisespec,
                    fit_result,
                    beta_noise: fit
                        .block_by_role(BlockRole::Scale)
                        .map(|block| block.beta.to_vec()),
                    wiggle: None,
                },
                LocationScaleResponse::Dispersion {
                    likelihood: kind.likelihood_spec(),
                    base_link: kind.base_link(),
                    family_tag: kind.family_tag(),
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: args.offset_column.clone(),
                    noise_offset_column: args.noise_offset_column.clone(),
                },
            )?;
            write_payload_json(out, payload)?;
        }
        emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
        return Ok(());
    }

    if !family.is_binomial() {
        return Err(
            "--predict-noise currently supports Gaussian, dispersion (negbin/gamma/beta/tweedie), \
             and binomial families"
                .to_string(),
        );
    }
    // family is already gated as binomial by is_binomial() above, so we
    // only need to discriminate on the link.
    let location_scale_link_kind = match &family.link {
        InverseLink::Standard(StandardLink::Logit) => match mixture_linkspec {
            // An explicit `link(type=blended(...))` upgrades the default logit
            // base link to a blended inverse-link mixture. Absent a blend spec,
            // the default logit base link fits directly — no blend is required
            // for the ordinary binomial-logit location-scale model.
            Some(spec) => {
                let state = state_fromspec(spec)
                    .map_err(|e| format!("invalid blended link configuration: {e}"))?;
                InverseLink::Mixture(state)
            }
            None => InverseLink::Standard(StandardLink::Logit),
        },
        // `resolve_family` already upgrades `LinkFunction::Sas` /
        // `LinkFunction::BetaLogistic` to their state-bearing variants,
        // so the family arrives here fully typed.
        InverseLink::Sas(state) => InverseLink::Sas(*state),
        InverseLink::BetaLogistic(state) => InverseLink::BetaLogistic(*state),
        InverseLink::Mixture(state) => InverseLink::Mixture(state.clone()),
        InverseLink::LatentCLogLog(state) => InverseLink::LatentCLogLog(*state),
        InverseLink::Standard(link) => InverseLink::Standard(*link),
    };
    if formula_linkwiggle.is_some() {
        require_inverse_link_supports_joint_wiggle(&location_scale_link_kind, "linkwiggle(...)")?;
    }

    let options = blockwise_options_from_fit_args()?;
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] binomial-location-scale fit start n={}",
        ds.values.nrows()
    );
    let solved = match fit_model(FitRequest::BinomialLocationScale(
        BinomialLocationScaleFitRequest {
            data: ds.values.view(),
            spec: BinomialLocationScaleTermSpec {
                y: y.clone(),
                weights: weights.clone(),
                link_kind: location_scale_link_kind.clone(),
                thresholdspec: meanspec.clone(),
                log_sigmaspec: noisespec.clone(),
                threshold_offset: mean_offset,
                log_sigma_offset: noise_offset,
            },
            wiggle: formula_linkwiggle.cloned().map(|cfg| LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders,
                double_penalty: cfg.double_penalty,
            }),
            options,
            kappa_options: kappa_options.clone(),
        },
    )) {
        Ok(FitResult::BinomialLocationScale(result)) => {
            log::info!(
                "[PHASE] binomial-location-scale fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            result
        }
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal binomial location-scale workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(e.to_string());
        }
    };
    if let (Some(knots), Some(degree)) = (solved.wiggle_knots.as_ref(), solved.wiggle_degree) {
        let final_q0 = compute_probit_q0_from_fit(&solved.fit.fit)?;
        let domain = summarizewiggle_domain(final_q0.view(), knots.view(), degree)?;
        if domain.outside_count > 0 {
            cli_err!(
                "warning: {} of {} link-wiggle q values ({:.1}%) fell outside the knot domain [{:.3}, {:.3}] after fitting",
                domain.outside_count,
                final_q0.len(),
                100.0 * domain.outside_fraction,
                domain.domain_min,
                domain.domain_max
            );
        }
    }
    let wiggle_meta = match (
        solved.wiggle_knots,
        solved.wiggle_degree,
        solved.beta_link_wiggle,
    ) {
        (Some(knots), Some(degree), Some(beta_link_wiggle)) => {
            Some((knots, degree, beta_link_wiggle))
        }
        _ => None,
    };
    // The binomial location-scale path links through a probit/threshold scale,
    // not a standardized response, so there is no `response_scale` to persist
    // (unlike the Gaussian path's #884 σ-floor factor). The σ contribution rides
    // entirely on the persisted noise transform below.
    let fit = solved.fit.fit;
    let frozen_meanspec =
        freeze_term_collection_from_design(&solved.fit.meanspec_resolved, &solved.fit.mean_design)
            .map_err(|e| e.to_string())?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &solved.fit.noisespec_resolved,
        &solved.fit.noise_design,
    )
    .map_err(|e| e.to_string())?;
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_BINOMIAL_LOCATION_SCALE,
        fit.outer_iterations,
        fit.convergence_evidence().inner_status().label()
    );
    print_spatial_aniso_scales(&solved.fit.meanspec_resolved);
    print_spatial_aniso_scales(&solved.fit.noisespec_resolved);
    if let Some(out) = args.out.as_ref() {
        let fit_result = compact_saved_multiblock_fit_result(
            fit.blocks.clone(),
            fit.lambdas.clone(),
            1.0,
            fit.covariance_conditional.clone(),
            fit.covariance_corrected.clone(),
            fit.geometry.clone(),
            SavedFitSummary::from_blockwise_fit(&fit)?,
        );
        let binomial_noise_transform = build_scale_deviation_transform_design(
            &solved.fit.mean_design.design,
            &solved.fit.noise_design.design,
            &weights,
            solved
                .fit
                .noise_design
                .intercept_range
                .end
                .min(solved.fit.noise_design.design.ncols()),
        )
        .map_err(|e| format!("failed to encode binomial noise transform: {e}"))?;
        let wiggle = wiggle_meta.map(|(knots, degree, beta_link_wiggle)| LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
            beta_link_wiggle,
        });
        let payload = assemble_location_scale_payload(
            LocationScaleInputs {
                formula: formula_text.to_string(),
                data_schema: ds.schema.clone(),
                noise_formula,
                resolved_termspec: frozen_meanspec,
                resolved_termspec_noise: frozen_noisespec,
                fit_result,
                beta_noise: fit
                    .block_by_role(BlockRole::Scale)
                    .map(|block| block.beta.to_vec()),
                wiggle,
            },
            LocationScaleResponse::Binomial {
                link: location_scale_link_kind.clone(),
                noise_transform: &binomial_noise_transform,
            },
            SavedModelSourceMetadata {
                training_headers: ds.headers.clone(),
                training_feature_ranges: Some(ds.feature_ranges()),
                offset_column: args.offset_column.clone(),
                noise_offset_column: args.noise_offset_column.clone(),
            },
        )?;
        write_payload_json(out, payload)?;
    }
    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    Ok(())
}

/// Map a [`ResponseFamily`] to the dispersion-GAM kind whose log-precision
/// channel can carry a `noise_formula` in the CLI `--predict-noise` path
/// (#913). Mirrors `fit_orchestration::dispersion_location_scale_kind`.
pub(crate) fn dispersion_location_scale_kind_for_cli(
    response: &ResponseFamily,
) -> Option<gam::gamlss::DispersionFamilyKind> {
    use gam::gamlss::DispersionFamilyKind;
    match response {
        ResponseFamily::NegativeBinomial { .. } => Some(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Some(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Some(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Some(DispersionFamilyKind::Tweedie { p: *p }),
        _ => None,
    }
}

pub(crate) fn block_role_label(role: &gam::estimate::BlockRole) -> &'static str {
    match role {
        gam::estimate::BlockRole::Mean => "mean",
        gam::estimate::BlockRole::Location => "location",
        gam::estimate::BlockRole::Scale => "scale",
        gam::estimate::BlockRole::Time => "time",
        gam::estimate::BlockRole::Threshold => "threshold",
        gam::estimate::BlockRole::LinkWiggle => "link-wiggle",
    }
}

pub(crate) fn validate_fit_args_preflight(
    args: &FitArgs,
    parsed: &ParsedFormula,
    fit_config: &FitConfig,
) -> Result<(), String> {
    if let (Some(logslope_formula), Some(z_column)) =
        (args.logslope_formula.as_deref(), args.z_column.as_deref())
    {
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula,
            &parsed.response,
            "--logslope-formula",
        )?;
        validate_marginal_slope_z_column_exclusion(
            parsed,
            &parsed_logslope,
            z_column,
            "bernoulli marginal-slope",
            "--logslope-formula",
        )?;
    }
    if args.out.is_none() {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
    }
    if args.request.is_some() {
        let is_survival = parse_surv_response(&parsed.response)?.is_some();
        if is_survival {
            let likelihood = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
            gam::families::fit_orchestration::validate_survival_baseline_config(
                likelihood,
                &fit_config.baseline_target,
                fit_config.baseline_scale,
                fit_config.baseline_shape,
                fit_config.baseline_rate,
                fit_config.baseline_makeham,
            )?;
            validate_time_margin_args(
                "request.config.threshold_time_k",
                fit_config.threshold_time_k,
                fit_config.threshold_time_degree,
            )?;
            validate_time_margin_args(
                "request.config.sigma_time_k",
                fit_config.sigma_time_k,
                fit_config.sigma_time_degree,
            )?;
            if fit_config.time_basis.trim().eq_ignore_ascii_case("ispline") {
                parse_survival_time_basis_config(
                    &fit_config.time_basis,
                    fit_config.time_degree,
                    fit_config.time_num_internal_knots,
                    fit_config.time_smooth_lambda,
                )?;
            }
        }
        return Ok(());
    }
    if args.family == FamilyArg::TransformationNormal && !args.transformation_normal {
        return Err(
            "--family transformation-normal does not select the transformation-normal fitter; use --transformation-normal"
                .to_string(),
        );
    }
    if args.transformation_normal
        && !matches!(
            args.family,
            FamilyArg::Auto | FamilyArg::TransformationNormal
        )
    {
        return Err(format!(
            "--transformation-normal conflicts with --family {}",
            family_arg_name(args.family)
        ));
    }
    if args.transformation_normal {
        if args.predict_noise.is_some() {
            return Err("--transformation-normal conflicts with --predict-noise".to_string());
        }
        if args.noise_offset_column.is_some() {
            return Err("--transformation-normal conflicts with --noise-offset-column".to_string());
        }
        if args.logslope_formula.is_some() || args.z_column.is_some() {
            return Err(
                "--transformation-normal conflicts with marginal-slope --logslope-formula/--z-column"
                    .to_string(),
            );
        }
        if args.firth {
            return Err("--transformation-normal conflicts with --firth".to_string());
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
        if args.frailty_kind.is_some() || args.frailty_sd.is_some() || args.hazard_loading.is_some()
        {
            return Err("--transformation-normal conflicts with frailty flags".to_string());
        }
    }
    if args.logslope_formula.is_some() != args.z_column.is_some() {
        return Err("--logslope-formula and --z-column must be provided together".to_string());
    }
    if args.logslope_formula.is_some() {
        if args.predict_noise.is_some() {
            return Err(
                "--predict-noise cannot be combined with --logslope-formula/--z-column".to_string(),
            );
        }
        if args.firth {
            log::info!(
                "--firth is redundant for marginal-slope fitting: the robust Jeffreys/Firth stabilizer is installed by policy"
            );
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
        if args.family != FamilyArg::Auto {
            return Err(
                "--family is ignored by marginal-slope fitting; select its link in the formula"
                    .to_string(),
            );
        }
    }
    if args.predict_noise.is_some() && args.adaptive_regularization {
        return Err(
            "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
        );
    }
    if args.negative_binomial_theta.is_some() && args.family != FamilyArg::NegativeBinomial {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    let is_survival = parse_surv_response(&parsed.response)?.is_some();
    let survival_likelihood = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
    let survival_likelihood_raw = fit_config.survival_likelihood.trim().to_ascii_lowercase();
    let baseline_target_raw = fit_config.baseline_target.trim().to_ascii_lowercase();
    let time_basis_raw = fit_config.time_basis.trim().to_ascii_lowercase();
    if is_survival {
        if !matches!(args.family, FamilyArg::Auto | FamilyArg::RoystonParmar) {
            return Err(
                "--family is ignored by Surv(...) fitting; use survival formula/link options"
                    .to_string(),
            );
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
    }
    if !is_survival {
        if args.family == FamilyArg::RoystonParmar {
            return Err(
                "--family royston-parmar requires a Surv(entry, exit, event) response".to_string(),
            );
        }
        if args.survival_time_anchor.is_some()
            || fit_config.baseline_scale.is_some()
            || fit_config.baseline_shape.is_some()
            || fit_config.baseline_rate.is_some()
            || fit_config.baseline_makeham.is_some()
            || args.threshold_time_k.is_some()
            || args.sigma_time_k.is_some()
            || survival_likelihood_raw != "transformation"
            || baseline_target_raw != "linear"
            || time_basis_raw != "ispline"
        {
            return Err(
                "survival-only options require a Surv(entry, exit, event) response".to_string(),
            );
        }
        if args.noise_offset_column.is_some() && args.predict_noise.is_none() {
            return Err("--noise-offset-column requires --predict-noise".to_string());
        }
    }
    gam::families::fit_orchestration::validate_survival_baseline_config(
        survival_likelihood,
        &baseline_target_raw,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    validate_time_margin_args(
        "--threshold-time-k",
        args.threshold_time_k,
        args.threshold_time_degree,
    )?;
    validate_time_margin_args("--sigma-time-k", args.sigma_time_k, args.sigma_time_degree)?;
    if time_basis_raw == "ispline" {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )?;
    }
    Ok(())
}

pub(crate) fn family_arg_name(arg: FamilyArg) -> &'static str {
    match arg {
        FamilyArg::Auto => "auto",
        FamilyArg::Gaussian => "gaussian",
        FamilyArg::BinomialLogit => "binomial-logit",
        FamilyArg::BinomialProbit => "binomial-probit",
        FamilyArg::BinomialCloglog => "binomial-cloglog",
        FamilyArg::LatentCloglogBinomial => "latent-cloglog-binomial",
        FamilyArg::PoissonLog => "poisson-log",
        FamilyArg::NegativeBinomial => "negative-binomial",
        FamilyArg::GammaLog => "gamma-log",
        FamilyArg::Tweedie => "tweedie",
        FamilyArg::Beta => "beta",
        FamilyArg::RoystonParmar => "royston-parmar",
        FamilyArg::TransformationNormal => "transformation-normal",
        FamilyArg::Expectile => "expectile",
        FamilyArg::Multinomial => "multinomial",
    }
}

pub(crate) fn validate_time_margin_args(
    flag: &str,
    k: Option<usize>,
    degree: usize,
) -> Result<(), String> {
    if let Some(k) = k {
        let min_k = degree + 1;
        if k < min_k {
            return Err(format!("{flag} must be >= degree + 1 = {min_k}, got {k}"));
        }
    }
    Ok(())
}

pub(crate) fn validate_positive_optional_usize(
    flag: &str,
    value: Option<usize>,
) -> Result<(), String> {
    if matches!(value, Some(0)) {
        return Err(format!("{flag} must be > 0"));
    }
    Ok::<(), _>(())
}

pub(crate) fn smooth_term_primary_column(term: &SmoothTermSpec) -> Option<usize> {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_term_primary_column(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_term_primary_column(&SmoothTermSpec {
            name: term.name.clone(),
            basis: (**smooth).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        }),
        SmoothBasisSpec::FactorSmooth { spec } => {
            if spec.continuous_cols.len() == 1 {
                Some(spec.continuous_cols[0])
            } else {
                None
            }
        }
        SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(*feature_col),
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            if feature_cols.len() == 1 {
                Some(feature_cols[0])
            } else {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WiggleDomainDiagnostics {
    pub(crate) domain_min: f64,
    pub(crate) domain_max: f64,
    pub(crate) outside_count: usize,
    pub(crate) outside_fraction: f64,
}

pub(crate) fn compute_probit_q0_from_eta(
    eta_t: ArrayView1<'_, f64>,
    eta_ls: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    if eta_t.len() != eta_ls.len() {
        return Err(format!(
            "probit q0 eta length mismatch: threshold={} log_sigma={}",
            eta_t.len(),
            eta_ls.len()
        ));
    }
    let mut q0 = Array1::<f64>::zeros(eta_t.len());
    for i in 0..q0.len() {
        q0[i] = gam::families::sigma_link::survival_q0_from_eta(eta_t[i], eta_ls[i]);
    }
    Ok(q0)
}

pub(crate) fn compute_probit_q0_from_fit(
    fit: &gam::estimate::UnifiedFitResult,
) -> Result<Array1<f64>, String> {
    let eta_t = fit
        .block_states
        .first()
        .ok_or_else(|| "pilot fit is missing threshold block".to_string())?
        .eta
        .view();
    let eta_ls = fit
        .block_states
        .get(1)
        .ok_or_else(|| "pilot fit is missing log-sigma block".to_string())?
        .eta
        .view();
    compute_probit_q0_from_eta(eta_t, eta_ls)
}

pub(crate) fn summarizewiggle_domain(
    q0: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<WiggleDomainDiagnostics, String> {
    if knots.len() < degree + 2 {
        return Err(format!(
            "wiggle knot vector too short for degree {}: {}",
            degree,
            knots.len()
        ));
    }
    let domain_min = knots[degree];
    let domain_max = knots[knots.len() - degree - 1];
    let outside_count = q0
        .iter()
        .filter(|&&v| v < domain_min || v > domain_max)
        .count();
    let outside_fraction = outside_count as f64 / q0.len().max(1) as f64;
    Ok(WiggleDomainDiagnostics {
        domain_min,
        domain_max,
        outside_count,
        outside_fraction,
    })
}
