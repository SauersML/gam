use super::*;

pub(crate) fn blockwise_options_from_fit_args()
-> Result<gam::families::custom_family::BlockwiseFitOptions, String> {
    let options = gam::families::custom_family::BlockwiseFitOptions::default();
    Ok(options)
}

pub(crate) fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    if let Some(inf) = fit.inference.as_mut() {
        // Keep working_weights/response on inference too — `diagnose --alo`
        // and other post-fit diagnostics consume them; clearing here zeroed
        // out the ALO geometry path entirely (failing with
        // "ALO diagnostics require hessian_weights length N; got 0").
        // reparam_qs is genuinely large (p × p) and not needed at predict
        // time, so still drop it.
        inf.reparam_qs = None;
    }
    fit.artifacts = gam::estimate::FitArtifacts {
        pirls: None,
        ..Default::default()
    };
}

pub(crate) fn fit_config_from_fit_args(args: &FitArgs) -> Result<FitConfig, String> {
    crate::config_resolve::resolve_cli_fit_config(crate::config_resolve::CliFitConfigInput {
        family: family_arg_canonical_name(args.family).map(str::to_string),
        expectile_tau: args.expectile_tau,
        negative_binomial_theta: args.negative_binomial_theta,
        link: None,
        flexible_link: false,
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
        survival_distribution: "gaussian".to_string(),
        threshold_time_k: args.threshold_time_k,
        threshold_time_degree: args.threshold_time_degree,
        sigma_time_k: args.sigma_time_k,
        sigma_time_degree: args.sigma_time_degree,
        noise_formula: args.predict_noise.clone(),
        logslope_formula: args.logslope_formula.clone(),
        z_column: args.z_column.clone(),
        scale_dimensions: args.scale_dimensions,
        adaptive_regularization: Some(args.adaptive_regularization),
        ridge_lambda: args.ridge_lambda,
        transformation_normal: args.transformation_normal,
        firth: args.firth,
        outer_max_iter: None,
        gpu: None,
        frailty_kind: cli_frailty_kind(args.frailty_kind),
        frailty_sd: args.frailty_sd,
        hazard_loading: cli_hazard_loading(args.hazard_loading),
    })
}

pub(crate) fn fit_config_from_survival_args(args: &SurvivalArgs) -> Result<FitConfig, String> {
    crate::config_resolve::resolve_cli_fit_config(crate::config_resolve::CliFitConfigInput {
        family: None,
        expectile_tau: None,
        negative_binomial_theta: None,
        link: args.link.clone(),
        flexible_link: false,
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
        adaptive_regularization: None,
        ridge_lambda: args.ridge_lambda,
        transformation_normal: false,
        firth: false,
        outer_max_iter: None,
        gpu: None,
        frailty_kind: cli_frailty_kind(args.frailty_kind),
        frailty_sd: args.frailty_sd,
        hazard_loading: cli_hazard_loading(args.hazard_loading),
    })
}

pub(crate) fn run_fit(args: FitArgs) -> Result<(), String> {
    let formula_text = choose_formula(&args)?;
    let parsed = parse_formula(&formula_text)?;
    validate_fit_args_preflight(&args, &parsed)?;
    let fit_config = fit_config_from_fit_args(&args)?;
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
    let requested_columns = required_columns_for_fit(&args, &parsed)?;
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
    let y = ds.values.column(y_col).to_owned();
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
    if args.expectile_tau.is_some() && !matches!(args.family, FamilyArg::Expectile) {
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
    if matches!(args.family, FamilyArg::Expectile) {
        return run_fit_expectile(&args, &ds, &formula_text, &fit_config);
    }

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
    let sas_linkspec = if let Some(choice) = link_choice.as_ref() {
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

    let y_kind = response_column_kind_for_dataset(&ds, y_col);
    let family = resolve_family(
        args.family,
        args.negative_binomial_theta,
        link_choice.clone(),
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
    if link_choice.is_none() {
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
    let effective_link = link_choice
        .as_ref()
        .map(|c| c.link)
        .unwrap_or_else(|| family.link_function());

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
    if let Some(noise_formula_raw) = &fit_config.noise_formula {
        return run_fitwith_predict_noise(
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
        );
    }
    if fit_config.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column requires --predict-noise or survival location-scale".to_string(),
        );
    }

    // Shape-derived resource policy: at large-scale n we auto-select strict
    // (analytic-operator-required) so any silent dense fallback in the
    // term-construction layer fails fast.
    let bare_fit_policy =
        gam::ResourcePolicy::for_problem(ds.values.nrows(), 0, gam::ProblemHints::default());
    let mut spec = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &bare_fit_policy,
    )?;
    if fit_config.scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    let route_flexible_through_standard = link_choice.as_ref().is_some_and(|choice| {
        matches!(choice.mode, LinkMode::Flexible) && choice.mixture_components.is_none()
    });
    let spatial_usagewarnings = collect_smooth_structure_warnings(&spec, &ds.headers, "model");
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(&inference_notes);
    let has_bounded_terms = termspec_has_bounded_terms(&spec);
    validate_cli_firth_configuration(CliFirthValidation {
        enabled: fit_config.firth,
        family: family.clone(),
        predict_noise: fit_config.noise_formula.is_some(),
        is_survival: false,
        link_choice: link_choice.as_ref(),
    })?;
    // `--firth` with `bounded()` is *redundant*, not unsupported. Firth
    // bias-reduction is exactly penalized maximum likelihood with Jeffreys'
    // prior `½ log|I(β)|`, and that prior is reparameterization-INVARIANT: its
    // MAP is equivariant under any smooth change of coordinates. Bounded terms
    // fit through the custom-family blockwise solver
    // (`fit_bounded_term_collection_with_design` -> `fit_custom_family`), whose
    // inner/outer joint Newton ALWAYS carries the full-span Jeffreys curvature
    // `H_Φ` and score `∇Φ` (its `joint_jeffreys_term_required()` is the trait
    // default `true`; `BoundedLinearFamily` does not opt out). That term is the
    // Jeffreys prior on the bounded LATENT coordinates `θ`, whose log-det
    // already threads the interval reparameterization's log-Jacobian
    // (`½ log|I_θ| = ½ log|I_β| + log|det J|`), so the latent MAP maps back
    // through the interval transform to the exact user-scale Firth estimate.
    // The explicit `--firth` branch below instead fits through
    // `optimize_external_design` on the raw unconstrained design and would
    // silently DROP the bounds — wrong for a bounded model. We therefore keep
    // bounded models on the standard branch (which is already Firth-equivalent)
    // and record the redundancy, rather than refusing the combination.
    let firth_redundant_for_bounded = fit_config.firth && has_bounded_terms;
    if firth_redundant_for_bounded {
        inference_notes.push(
            "--firth is redundant for bounded() coefficients: the bounded custom-family solver \
             already installs the reparameterization-invariant Jeffreys/Firth bias-reduction in \
             the bounded latent coordinates, which is the exact Firth estimate on the user scale."
                .to_string(),
        );
        print_inference_summary(std::slice::from_ref(
            inference_notes.last().expect("note just pushed is present"),
        ));
    }
    let fit_max_iter = 200usize;
    let fit_tol = 1e-6f64;
    let weights = resolve_weight_column(&ds, &col_map, fit_config.weight_column.as_deref())?;
    let offset = resolve_offset_column(&ds, &col_map, fit_config.offset_column.as_deref())?;
    let frailty = fit_frailty_spec_from_args(&args, "fit")?;
    if let Some(choice) = link_choice.as_ref()
        && matches!(choice.mode, LinkMode::Flexible)
    {
        if choice.mixture_components.is_some() {
            return Err(
                    "flexible(blended(...)/mixture(...)) is currently supported only with --predict-noise binomial location-scale fitting or --survival-likelihood=location-scale"
                        .to_string(),
                );
        }
        if has_bounded_terms {
            return Err(
                "flexible(...) links are not yet supported with bounded() coefficients".to_string(),
            );
        }
        if !family.is_binomial() {
            return Err("flexible(...) links currently require a binomial family/link".to_string());
        }
    }
    let adaptive_opts = if fit_config.adaptive_regularization.unwrap_or(false) {
        Some(AdaptiveRegularizationOptions {
            enabled: true,
            ..AdaptiveRegularizationOptions::default()
        })
    } else {
        None
    };
    let latent_cloglog_state = if family.is_latent_cloglog() {
        Some(latent_cloglog_state_from_frailty_spec(
            &frailty,
            "latent-cloglog-binomial",
        )?)
    } else {
        if !matches!(
            frailty,
            gam::families::survival::lognormal_kernel::FrailtySpec::None
        ) {
            return Err(
                "frailty is only supported here for --family latent-cloglog-binomial; use the frailty-aware marginal-slope or survival paths instead"
                    .to_string(),
            );
        }
        None
    };
    // Standard-fit `FitOptions` are built through the single shared policy
    // source (#1196). The CLI passes only the request-specific link/Firth/
    // adaptive inputs; the outer-REML optimization policy
    // (`compute_inference`, `skip_rho_posterior_inference`, `tol`, the
    // `max_iter` default, the penalty shrinkage floor) is filled in by
    // `canonical_standard_fit_options`, identical to the formula/Python path,
    // so the same model can no longer fit differently across entry points.
    // (Previously this hand-built block used `tol: 1e-6` /
    // `skip_rho_posterior_inference: false`, diverging from the formula path's
    // `1e-10`/`true` — the structural defect behind #1191/#1196.)
    // `fit_max_iter`/`fit_tol` remain the inputs to the separate forced-Firth
    // external-design branch below.
    let base_fit_options = gam::families::fit_orchestration::canonical_standard_fit_options(
        &fit_config,
        gam::families::fit_orchestration::StandardFitOptionsInputs {
            latent_cloglog: latent_cloglog_state,
            mixture_link: mixture_linkspec.clone(),
            optimize_mixture: true,
            sas_link: sas_linkspec,
            optimize_sas: sas_linkspec.is_some()
                && matches!(
                    effective_link,
                    LinkFunction::Sas | LinkFunction::BetaLogistic
                ),
            firth_bias_reduction: false,
            adaptive_regularization: adaptive_opts,
            ..Default::default()
        },
    );
    let standard_wiggle = if learn_linkwiggle
        && fit_config.noise_formula.is_none()
        && (!mean_only_flexible_linkwiggle || route_flexible_through_standard)
    {
        let wiggle_cfg = effective_linkwiggle
            .as_ref()
            .expect("learn_linkwiggle guarantees wiggle config");
        let link_kind = resolve_binomial_inverse_link_for_fit(
            family.clone(),
            effective_link,
            mixture_linkspec.as_ref(),
            "binomial mean-only link wiggle",
        )?;
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_orders: wiggle_cfg.penalty_orders.clone(),
                double_penalty: wiggle_cfg.double_penalty,
            },
            // CLI path: keep `blockwise_options_from_fit_args()` as the
            // option source (it currently returns defaults but is the hook
            // for future fit-arg overrides). Bound together with the pilot
            // config inside `StandardBinomialWiggleConfig` so the two can
            // never disagree (#320).
            refit_options: blockwise_options_from_fit_args()?,
        })
    } else {
        None
    };

    let (
        fit,
        design,
        resolvedspec,
        adaptive_regularization_diagnostics,
        standard_saved_link_state,
        standard_wiggle_meta,
    ): (
        UnifiedFitResult,
        gam::smooth::TermCollectionDesign,
        TermCollectionSpec,
        Option<gam::smooth::AdaptiveRegularizationDiagnostics>,
        FittedLinkState,
        Option<(Vec<f64>, usize)>,
    ) = if fit_config.firth && !firth_redundant_for_bounded {
        let design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        let ext = optimize_external_design(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            design.penalties.clone(),
            &ExternalOptimOptions {
                family: family.clone(),
                latent_cloglog: None,
                mixture_link: None,
                optimize_mixture: true,
                sas_link: None,
                optimize_sas: false,
                // Always compute inference so `predict --uncertainty` works
                // for Gaussian fits too (see comment near the other compute_inference site).
                compute_inference: true,
                skip_rho_posterior_inference: false,
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
                penalty_shrinkage_floor: Some(1e-6),
                rho_prior: Default::default(),
                kronecker_penalty_system: None,
                kronecker_factored: None,
                persist_warm_start_disk: false,
            },
        )
        .map_err(|e| format!("fit_gam (forced Firth) failed: {e}"))?;
        (
            fit_result_from_external(ext),
            design,
            spec.clone(),
            None,
            FittedLinkState::Standard(None),
            None,
        )
    } else {
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] standard-GAM fit start n={} family={:?}",
            ds.values.nrows(),
            family
        );
        let standard_request = StandardFitRequest {
            data: ds.values.to_owned(),
            y: y.clone(),
            weights: weights.clone(),
            offset: offset.clone(),
            spec: spec.clone(),
            family: family.clone(),
            options: base_fit_options,
            kappa_options: kappa_options.clone(),
            wiggle: standard_wiggle,
            // Request fields that are derived from the resolved `FitConfig` are
            // sourced from `fit_config` here exactly as `materialize_standard`
            // sources them (#1196), instead of being hardcoded to empty. The CLI
            // arg→config resolver (`resolve_cli_fit_config`) currently leaves
            // `coefficient_groups` / `penalty_block_gamma_priors` at their empty
            // defaults because no CLI flag sets them, so this is value-identical
            // today; routing them through `fit_config` makes the CLI request a
            // function of the same resolved config the Python/PyO3 path consumes,
            // so a future config knob can never be silently dropped on the CLI
            // side while the FFI honors it — the divergence class behind #1191.
            coefficient_groups: fit_config.coefficient_groups.clone(),
            penalty_block_gamma_priors: fit_config.penalty_block_gamma_priors.clone(),
            // `latent_coord` is resolved (its `term_index` bound to a smooth in
            // the spec) only inside `materialize_standard` from a formula latent
            // term; the CLI standard-fit path parses no latent coordinate, so
            // `None` is the materialized value, not a dropped config field.
            latent_coord: None,
            _marker: std::marker::PhantomData,
        };
        // Exact O(n) spline-scan fast path (#1030/#1034): a single 1-D
        // Gaussian cubic smooth routes through the state-space scan — the
        // same penalized posterior at O(n) per λ-trial instead of the dense
        // design/Gram route — and persists the smoother state directly.
        if let Some(inputs) = spline_scan_fast_path(&standard_request) {
            let scan = gam::solver::spline_scan::fit_spline_scan(
                &inputs.x,
                &inputs.y,
                &inputs.w,
                inputs.order,
            )
            .map_err(|e| format!("spline-scan fit failed: {e}"))?;
            log::info!(
                "[PHASE] spline-scan fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            let feature_col = match &spec.smooth_terms[0].basis {
                gam::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => *feature_col,
                other => {
                    return Err(format!(
                        "internal error: spline-scan detection accepted a non-1D basis {other:?}"
                    ));
                }
            };
            let feature_column = ds.headers.get(feature_col).cloned().ok_or_else(|| {
                format!("internal error: spline-scan feature column {feature_col} has no header")
            })?;
            cli_out!(
                "spline-scan fit | knots={} | edf={:.3} | sigma2={:.6e} | log_lambda={:.4} | reml={:.6e}",
                scan.knots.len(),
                scan.edf(),
                scan.sigma2,
                scan.log_lambda,
                scan.restricted_loglik,
            );
            if let Some(out) = args.out {
                let payload = assemble_spline_scan_payload(
                    formula_text,
                    feature_column,
                    &scan,
                    ds.schema.clone(),
                    ds.headers.clone(),
                    ds.feature_ranges(),
                );
                write_payload_json(&out, payload)?;
            }
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Ok(());
        }
        // O(n log n) multiresolution residual-cascade fast path (#1032): a
        // single scattered 2–3D Gaussian Duchon/Matérn smooth past the
        // dense-kernel cliff routes through the Wendland multilevel-frame fit.
        // Unlike the 1-D scan this is a DIFFERENT posterior, so the seam only
        // fires on the exact structural signature; rejected metric or ineligible
        // shape fall through to the dense `fit_model` path.
        if let Some(inputs) = residual_cascade_fast_path(&standard_request) {
            let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
            if let Ok(cascade_fit) = gam::solver::residual_cascade::fit_residual_cascade(
                &coord_refs,
                &inputs.y,
                &inputs.w,
                &inputs.metric,
                inputs.sobolev_s,
            ) {
                log::info!(
                    "[PHASE] residual-cascade fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                // Resolve the d feature column names from the single smooth term.
                let feature_columns: Vec<String> = {
                    let feature_cols = match &spec.smooth_terms[0].basis {
                        gam::smooth::SmoothBasisSpec::Duchon { feature_cols, .. } => {
                            feature_cols.clone()
                        }
                        gam::smooth::SmoothBasisSpec::Matern { feature_cols, .. } => {
                            feature_cols.clone()
                        }
                        other => {
                            return Err(format!(
                                "internal error: cascade detection accepted non-radial basis \
                                 {other:?}"
                            ));
                        }
                    };
                    feature_cols
                        .iter()
                        .map(|&c| {
                            ds.headers.get(c).cloned().ok_or_else(|| {
                                format!("internal error: cascade feature column {c} has no header")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let cert = &cascade_fit.certificate;
                cli_out!(
                    "residual-cascade fit | levels={} | centers={} | sigma2={:.6e} | \
                     log_lambda={:.4} | reml={:.6e} | rel_resid={:.2e}",
                    cascade_fit.num_levels(),
                    cascade_fit.num_centers(),
                    cascade_fit.sigma2,
                    cascade_fit.log_lambda,
                    cascade_fit.restricted_loglik,
                    cert.solve_rel_residual,
                );
                if let Some(out) = args.out {
                    let payload = assemble_residual_cascade_payload(
                        formula_text,
                        feature_columns,
                        &cascade_fit,
                        ds.schema.clone(),
                        ds.headers.clone(),
                        ds.feature_ranges(),
                    )?;
                    write_payload_json(&out, payload)?;
                }
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Ok(());
            }
            // Quasi-uniformity guard (caveat 2) or degenerate design: fall
            // through to the dense kernel path.
        }
        let fitted = match fit_model(FitRequest::Standard(standard_request)) {
            Ok(FitResult::Standard(result)) => {
                log::info!(
                    "[PHASE] standard-GAM fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal standard workflow returned the wrong result variant".to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                // Recognize the common "user's sign / box constraint fights
                // the data" failure mode and surface a focused hint above
                // the technical REML / KKT breakdown. Without this the user
                // sees only:
                //   "no candidate seeds passed outer startup validation
                //    (standard REML); ... reasons: [seed 0 (validation):
                //    Parameter constraint violation: KKT residuals exceed
                //    tolerance: primal=0.81 ..."
                // which is incomprehensible jargon for the case where they
                // wrote `nonpositive(x)` on data where the sign of the
                // covariate-response correlation is actually positive.
                let estr = e.to_string();
                if estr.contains("Parameter constraint violation")
                    && estr.contains("no candidate seeds")
                {
                    return Err(format!(
                        "standard term fit failed: every candidate fit violates the \
                         parameter constraint you set (nonpositive() / nonnegative() / \
                         constrain() / bounded()). The constraint and the data appear to \
                         disagree about the sign or magnitude of the effect. \
                         Either remove the constraint, flip its direction, or check the \
                         data. Underlying error: {e}"
                    ));
                }
                return Err(format!("standard term fit failed: {e}"));
            }
        };
        (
            fitted.fit,
            fitted.design,
            fitted.resolvedspec,
            fitted.adaptive_diagnostics,
            fitted.saved_link_state,
            match (fitted.wiggle_knots, fitted.wiggle_degree) {
                (Some(knots), Some(degree)) => Some((knots.to_vec(), degree)),
                _ => None,
            },
        )
    };
    print_spatial_aniso_scales(&resolvedspec);

    let frozenspec =
        freeze_term_collection_from_design(&resolvedspec, &design).map_err(|e| e.to_string())?;
    let mut saved_fit = fit.clone();
    saved_fit.fitted_link = standard_saved_link_state.clone();
    let saved_termspec = frozenspec.clone();
    if let Some((wiggle_knots, wiggle_degree)) = standard_wiggle_meta.as_ref() {
        let beta_eta = fit
            .block_by_role(BlockRole::Mean)
            .ok_or_else(|| "standard wiggle fit is missing eta block".to_string())?
            .beta
            .clone();
        let q0_final = design.design.dot(&beta_eta);
        let domain = summarizewiggle_domain(
            q0_final.view(),
            ArrayView1::from(wiggle_knots),
            *wiggle_degree,
        )?;
        if domain.outside_count > 0 {
            cli_err!(
                "warning: {} of {} link-wiggle eta values ({:.1}%) fell outside the knot domain [{:.3}, {:.3}] after fitting",
                domain.outside_count,
                q0_final.len(),
                100.0 * domain.outside_fraction,
                domain.domain_min,
                domain.domain_max
            );
        }
    }
    compact_fit_result_for_batch(&mut saved_fit);

    if let Some(out) = args.out {
        let latent_cloglog_state = if family.is_latent_cloglog() {
            Some(saved_latent_cloglog_state_from_fit(&saved_fit).expect(
                "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
            ))
        } else {
            saved_latent_cloglog_state_from_fit(&saved_fit)
        };
        let mut payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            formula_text,
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family.clone(),
                link: StandardLink::try_from(effective_link).ok(),
                latent_cloglog_state,
                mixture_state: saved_mixture_state_from_fit(&saved_fit),
                sas_state: saved_sas_state_from_fit(&saved_fit),
            },
            family.name().to_string(),
        );
        payload.unified = Some(saved_fit.clone());
        payload.fit_result = Some(saved_fit.clone());
        payload.data_schema = Some(ds.schema.clone());
        payload.link = inverse_link_from_fitted_link_state(&saved_fit.fitted_link);
        if let Some((wiggle_knots, wiggle_degree)) = standard_wiggle_meta {
            payload.linkwiggle_knots = Some(wiggle_knots);
            payload.linkwiggle_degree = Some(wiggle_degree);
        }
        match &saved_fit.fitted_link {
            FittedLinkState::Mixture { covariance, .. } => {
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nestedvec);
            }
            FittedLinkState::Sas { covariance, .. }
            | FittedLinkState::BetaLogistic { covariance, .. } => {
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            FittedLinkState::LatentCLogLog { .. } => {}
            FittedLinkState::Standard(_) => {}
        }
        set_training_feature_metadata_from_dataset(&mut payload, &ds);
        payload.resolved_termspec = Some(saved_termspec);
        payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
        // Populate the exact Gaussian jackknife+ substrate (#1098) when the fit
        // is a standard Gaussian-identity model with unit prior weights and the
        // converged penalized Hessian M = X'X + S(λ) is available from the
        // FitGeometry.  The exchangeability proof requires unit weights — a
        // non-unit weight makes the test row non-exchangeable with training rows.
        // When all conditions hold the substrate is factored once here; predict
        // calls GaussianJackknifePlusStats::interval per test point in O(p)
        // from the precomputed LOO quantities.
        if family.is_gaussian_identity() {
            if let Some(geo) = fit.geometry.as_ref() {
                let m = &geo.penalized_hessian.0;
                let x_dense = design.design.to_dense();
                match gam::inference::full_conformal::GaussianJackknifePlusStats::from_design_unit_weight_normal_matrix(
                    &x_dense,
                    &y,
                    &weights,
                    m,
                ) {
                    Ok(stats) => {
                        payload.gaussian_jackknife_plus = Some(stats);
                    }
                    Err(_) => {
                        // Non-unit weights or other precondition failure: silently skip.
                        // predict falls back to the posterior band as documented.
                    }
                }
                // Exact full-conformal substrate (#1098): same eligibility, persists
                // X + y + frozen Sλ so the EXACT distribution-free set replays per
                // test point. Sλ = M − XᵀX is recovered inside the substrate ctor.
                match gam::inference::full_conformal::ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
                    &x_dense,
                    &y,
                    &weights,
                    m,
                ) {
                    Ok(sub) => {
                        payload.full_conformal = Some(sub);
                    }
                    Err(_) => {
                        // Precondition failure: skip; predict errors clearly or
                        // falls back to jackknife+/posterior band.
                    }
                }
            }
        }
        set_saved_offset_columns(
            &mut payload,
            fit_config.offset_column.clone(),
            fit_config.noise_offset_column.clone(),
        );
        write_payload_json(&out, payload)?;
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    Ok(())
}

/// Expectile (Newey–Powell LAWS) standard-GAM fit for the CLI.
///
/// Mirrors the dedicated transformation-normal / marginal-slope CLI sub-fits:
/// it routes through the single shared dispatch seam
/// [`fit_expectile_if_requested`], which runs the iteratively-reweighted
/// Gaussian-identity GAM and returns an ordinary [`StandardFitResult`] whose
/// coefficients ARE the τ-expectile. The fit is reported and persisted as the
/// Gaussian-identity standard model it is — predict, bands, and persistence work
/// unchanged. The exact Gaussian jackknife+/full-conformal substrates the
/// symmetric path attaches are deliberately omitted: their finite-sample
/// coverage proof needs exchangeable unit-weight rows, which the asymmetric LAWS
/// weighting breaks, so attaching them would advertise a guarantee the expectile
/// fit does not hold.
pub(crate) fn run_fit_expectile(
    args: &FitArgs,
    ds: &Dataset,
    formula_text: &str,
    fit_config: &FitConfig,
) -> Result<(), String> {
    let phase_start = std::time::Instant::now();
    let tau = args.expectile_tau.unwrap_or(0.5);
    log::info!(
        "[PHASE] expectile-LAWS fit start n={} tau={tau}",
        ds.values.nrows()
    );
    let result =
        gam::families::fit_orchestration::fit_expectile_if_requested(formula_text, ds, fit_config)
            .map_err(|e| format!("expectile fit failed: {e}"))?
            .ok_or_else(|| {
                "internal error: run_fit_expectile entered for a non-expectile family".to_string()
            })?;
    log::info!(
        "[PHASE] expectile-LAWS fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );

    let StandardFitResult {
        fit,
        design,
        resolvedspec,
        adaptive_diagnostics,
        saved_link_state,
        ..
    } = result;
    // The inner family is Gaussian-identity; the LAWS coefficients are reported
    // and persisted as that standard model.
    let family = fit
        .likelihood_family
        .clone()
        .unwrap_or_else(LikelihoodSpec::gaussian_identity);

    cli_out!(
        "expectile fit | tau={:.4} | terms={} | edf={:.3} | scale={:.6e}",
        tau,
        resolvedspec.smooth_terms.len() + resolvedspec.linear_terms.len(),
        fit.edf_total().unwrap_or(f64::NAN),
        fit.dispersion_phi(),
    );

    if let Some(out) = args.out.as_ref() {
        let frozenspec = freeze_term_collection_from_design(&resolvedspec, &design)
            .map_err(|e| e.to_string())?;
        let mut saved_fit = fit.clone();
        saved_fit.fitted_link = saved_link_state.clone();
        compact_fit_result_for_batch(&mut saved_fit);
        let mut payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            formula_text.to_string(),
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family.clone(),
                link: StandardLink::try_from(family.link_function()).ok(),
                latent_cloglog_state: None,
                mixture_state: saved_mixture_state_from_fit(&saved_fit),
                sas_state: saved_sas_state_from_fit(&saved_fit),
            },
            family.name().to_string(),
        );
        payload.unified = Some(saved_fit.clone());
        payload.fit_result = Some(saved_fit.clone());
        payload.data_schema = Some(ds.schema.clone());
        payload.link = inverse_link_from_fitted_link_state(&saved_fit.fitted_link);
        set_training_feature_metadata_from_dataset(&mut payload, ds);
        payload.resolved_termspec = Some(frozenspec);
        payload.adaptive_regularization_diagnostics = adaptive_diagnostics;
        set_saved_offset_columns(
            &mut payload,
            fit_config.offset_column.clone(),
            fit_config.noise_offset_column.clone(),
        );
        write_payload_json(out, payload)?;
    }
    Ok(())
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
    let frailty = fixed_gaussian_shift_frailty_from_spec(
        &fit_frailty_spec_from_args(args, "bernoulli marginal-slope")?,
        "bernoulli marginal-slope",
    )?;
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
        solved.fit.pirls_status.label()
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
        solved.fit.pirls_status.label()
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
            fit.pirls_status.label()
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
            fit.pirls_status.label()
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
        InverseLink::Standard(StandardLink::Logit) => {
            let spec = mixture_linkspec
                .ok_or_else(|| {
                    "binomial blended-inverse-link location-scale fitting requires link(type=blended(...))"
                        .to_string()
                })?
                .clone();
            let state = state_fromspec(&spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            InverseLink::Mixture(state)
        }
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
        fit.pirls_status.label()
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
) -> Result<(), String> {
    if args.out.is_none() {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
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
    let fit_config = fit_config_from_fit_args(args)?;
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
    crate::config_resolve::validate_survival_baseline_args(
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

pub(crate) fn choose_formula(args: &FitArgs) -> Result<String, CliError> {
    let v = args.formula_positional.trim();
    if v.is_empty() {
        return Err(CliError::ArgumentInvalid {
            reason: "FORMULA cannot be empty".to_string(),
        });
    }
    Ok(v.to_string())
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
        q0[i] = -eta_t[i] * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(eta_ls[i]);
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
