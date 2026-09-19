use super::*;

pub(crate) fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    // GUARD (#2030): the geometry carrier's optional owned row evidence MUST
    // survive compaction. Saved ALO explicitly requires `geometry.working`;
    // `None` correctly means unavailable, while truncating a present vector
    // would corrupt a valid single-diagonal fit. `FitInference` deliberately
    // has no duplicate copy, so only this one source of truth is retained.
    if let Some(inf) = fit.inference.as_mut() {
        inf.reparam_qs = None;
    }
    // Only the PIRLS diagnostic payload is heavy; every other artifact field
    // is a handful of scalars. Resetting the whole struct to `Default`
    // previously also wiped `criterion_certificate`, which the very next
    // save-time `validate_numeric_finiteness` call requires whenever
    // `outer_iterations > 0` (#934) — every standard `gam fit --out ...`
    // then failed with "outer iterations ran without an analytic
    // stationarity certificate" despite having genuinely converged.
    fit.artifacts.pirls = None;
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

    let frailty_kind = args.frailty_kind.map(|kind| match kind {
        FrailtyKindArg::GaussianShift => "gaussian-shift".to_string(),
        FrailtyKindArg::HazardMultiplier => "hazard-multiplier".to_string(),
    });
    let hazard_loading = args.hazard_loading.map(|loading| match loading {
        HazardLoadingArg::Full => "full".to_string(),
        HazardLoadingArg::LoadedVsUnloaded => "loaded-vs-unloaded".to_string(),
    });
    let config = crate::config_resolve::FitRequestConfigDocument {
        baseline_makeham: args.baseline_makeham,
        baseline_rate: args.baseline_rate,
        baseline_scale: args.baseline_scale,
        baseline_shape: args.baseline_shape,
        baseline_target: Some(args.baseline_target.clone()),
        expectile_tau: args.expectile_tau,
        family: family_arg_canonical_name(args.family).map(str::to_string),
        firth: args.firth.then_some(true),
        frailty_kind,
        frailty_sd: args.frailty_sd,
        hazard_loading,
        slope_formula: args.slope_formula.clone(),
        negative_binomial_theta: args.negative_binomial_theta,
        noise_formula: args.predict_noise.clone(),
        noise_offset: args.noise_offset_column.clone(),
        offset: args.offset_column.clone(),
        precompute_conformal: Some(args.precompute_conformal),
        persistent_warm_start_root: args.persistent_warm_start_root.clone(),
        scale_dimensions: args.scale_dimensions.then_some(true),
        sigma_time_k: args.sigma_time_k,
        slope_time_k: args.slope_time_k,
        // `None` (flag unset) flows through so the Surv() seam resolves the one
        // canonical default; `Some(mode)` is the explicit request (#2301).
        survival_likelihood: args.survival_likelihood.clone(),
        threshold_time_k: args.threshold_time_k,
        time_basis: Some(args.time_basis.clone()),
        transformation_normal: args.transformation_normal.then_some(true),
        weights: args.weights_column.clone(),
        z_column: args.z_column.clone(),
        residual_columns: (!args.residual_columns.is_empty())
            .then(|| args.residual_columns.clone()),
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

pub(crate) fn run_fit(args: FitArgs) -> Result<(), String> {
    let resolved_invocation = resolve_fit_invocation(&args)?;
    let formula_text = resolved_invocation.formula;
    let fit_config = resolved_invocation.fit_config;
    let parsed = parse_formula(&formula_text)?;
    if fit_config.ctn_stage1.is_some() || fit_config.frozen_ctn.is_some() {
        let out = args.out.as_ref().ok_or("CTN fitting requires --out")?;
        let required = gam::inference::ctn::required_fit_columns(&formula_text, &fit_config)?;
        let dataset = load_fit_dataset_with_roles(&args.data, &required.into_iter().collect::<Vec<_>>(), &parsed, false)?;
        let payload = gam::inference::model_payload_builders::fit_formula_to_payload(formula_text, &dataset, &fit_config)
            .map_err(|error| error.to_string())?;
        let model = SavedModel::from_payload(payload);
        return write_model_json(out, &model);
    }
    validate_fit_args_preflight(&args, &parsed, &fit_config)?;
    if parse_surv_response(&parsed.response)?.is_some() {
        validate_cli_firth_configuration(CliFirthValidation {
            enabled: fit_config.firth,
            family: LikelihoodSpec::royston_parmar(),
            predict_noise: fit_config.noise_formula.is_some(),
            is_survival: true,
            link_choice: None,
        })?;
        return run_library_formula_fit(&args, &parsed, formula_text, &fit_config);
    }
    // Multinomial is a softmax multi-output family (categorical response, K-1
    // active-class linear predictors): it owns its own dataset load (with the
    // response forced to a factor) and persistence envelope, so dispatch it
    // before the scalar-response standard path. The stale note below about "the
    // CLI has no multinomial family" no longer holds for this early return.
    if fit_config.family.as_deref() == Some("multinomial") {
        return run_fit_multinomial(&args, &parsed, &formula_text, &fit_config);
    }
    // Transformation-normal fits go through the library materializer, which refuses
    // link(...), linkwiggle(...), frailty, a noise formula and marginal-slope
    // settings. It reads neither Firth nor another family, so those are refused here.
    let family_names_transformation_normal = fit_config
        .family
        .as_deref()
        .is_some_and(|name| name.eq_ignore_ascii_case("transformation-normal"));
    if fit_config.transformation_normal || family_names_transformation_normal {
        if fit_config.firth {
            return Err("--firth is not supported for the transformation-normal family".to_string());
        }
        if !family_names_transformation_normal {
            if let Some(family) = fit_config.family.as_deref() {
                return Err(format!("--transformation-normal conflicts with --family {family}"));
            }
        }
        return run_library_formula_fit(&args, &parsed, formula_text, &fit_config);
    }
    // Bernoulli marginal-slope fits go through the library materializer, which
    // resolves the probit base link and refuses the settings this family cannot use.
    // It reads no other family, so a family other than this one is refused here.
    if fit_config.slope_formula.is_some() || fit_config.z_column.is_some() {
        if let Some(family) = fit_config.family.as_deref() {
            let canonical = family.to_ascii_lowercase().replace('_', "-");
            if canonical != "bernoulli-marginal-slope" && canonical != "binary-marginal-slope" {
                return Err(format!(
                    "--family {family} is ignored by marginal-slope fitting; select its link in the formula"
                ));
            }
        }
        return run_library_formula_fit(&args, &parsed, formula_text, &fit_config);
    }
    // Location-scale fits go through the library materializer, like the other
    // specialized families. It reads no Firth setting, so that is refused here.
    if fit_config.noise_formula.is_some() {
        if fit_config.firth {
            return Err(
                "--firth is not supported with --predict-noise location-scale fitting".to_string(),
            );
        }
        return run_library_formula_fit(&args, &parsed, formula_text, &fit_config);
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
    let requested_columns = fit_required_columns(&parsed, &fit_config)
        .map_err(|error| error.to_string())?
        .into_iter()
        .collect::<Vec<_>>();
    // Force `group(g)` / `factor(g)` / `re(g)` grouping columns to a factor
    // encoding even when their labels are numeric. An untyped CSV cannot carry
    // the typed-frame categorical sentinel the Python path uses, so without this
    // a numeric-coded grouping column would be demoted to a single continuous
    // ramp — a strictly lower-capacity design than `gamfit.fit` builds for the
    // same data. Bare `+ x` and `s(x)` stay value-inferred, so a continuous
    // integer covariate is untouched.
    let ds = load_fit_dataset_with_roles(&args.data, &requested_columns, &parsed, false)?;
    require_dataset_rows("fit", &args.data, ds.values.nrows())?;
    // Every single-parameter formula fit, the expectile estimator included, is
    // owned end-to-end by gam-models; this route adds the CLI's summary lines and
    // its compact saved fit.
    run_canonical_standard_fit(&args, &ds, &parsed, &formula_text, &fit_config)
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
            print_spatial_aniso_scales(&result.resolvedspec);
            let status = result.fit.convergence_evidence().inner_status().label();
            let iterations = result.fit.outer_iterations;
            let term_count =
                result.resolvedspec.smooth_terms.len() + result.resolvedspec.linear_terms.len();
            let edf = result.fit.edf_total().unwrap_or(f64::NAN);
            let log_likelihood = result.fit.log_likelihood;
            // The comparable criterion reads the null-space metadata that payload
            // assembly derives from the realized penalty topology, so the criterion
            // is printed from the assembled payload, as the library route prints it.
            // Printed before assembly, the fit carried no metadata and the line
            // published the raw criterion under reml_score.
            compact_fit_result_for_batch(&mut result.fit);
            let mut payload = assemble_standard_payload(StandardPayloadInputs {
                formula: formula.to_string(),
                dataset,
                fit_config,
                result,
            })?;
            let fit = payload
                .fit_result
                .as_ref()
                .ok_or("standard payload assembly returned no fit result")?;
            cli_out!(
                "{} fit | family={} | status={} | iterations={} | terms={} | edf={:.3} | loglik={:.6e} | reml_score={} | raw_reml_score={}",
                model_label,
                family.name(),
                status,
                iterations,
                term_count,
                edf,
                log_likelihood,
                // An exactly-interpolating Gaussian fit has no criterion at
                // all, and a fit without null-space metadata has no comparable
                // one; each absence prints its own words (#2595, #2627).
                gam::report::criterion_row(
                    fit.comparable_reml_score()
                        .map_err(|err| format!("failed to compute comparable REML score: {err}"))?,
                    fit.reml_score(),
                    |value| format!("{value:.6e}"),
                ),
                gam::report::criterion_display(fit.reml_score()),
            );
            if let Some(out) = args.out.as_ref() {
                apply_request_metadata(&mut payload, fit_config, outcome.inference_notes);
                write_payload_json(out, payload)?;
            }
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
                scan.log_lambda(),
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
                payload.weight_column = fit_config.weight_column.clone();
                apply_request_metadata(&mut payload, fit_config, outcome.inference_notes);
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
                cascade.log_lambda(),
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
                apply_request_metadata(&mut payload, fit_config, outcome.inference_notes);
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

/// Fit a formula through the library's formula-to-payload service, the one the
/// Python bindings use, and save the model. The CLI owns only loading the data
/// and writing the file, so a `--request` document reaches the fit whole.
fn run_library_formula_fit(
    args: &FitArgs,
    parsed: &ParsedFormula,
    formula: String,
    fit_config: &FitConfig,
) -> Result<(), String> {
    let out = args
        .out
        .as_ref()
        .ok_or("fit requires --out; refusing to run a training job that writes no model")?;
    let requested_columns = fit_required_columns(parsed, fit_config)
        .map_err(|error| error.to_string())?
        .into_iter()
        .collect::<Vec<_>>();
    let dataset = load_fit_dataset_with_roles(&args.data, &requested_columns, parsed, false)?;
    require_dataset_rows("fit", &args.data, dataset.values.nrows())?;
    let phase_start = std::time::Instant::now();
    log::info!("[PHASE] formula fit start n={}", dataset.values.nrows());
    let payload = gam::inference::model_payload_builders::fit_formula_to_payload(
        formula,
        &dataset,
        fit_config,
    )
    .map_err(|error| format!("formula fit failed: {error}"))?;
    log::info!(
        "[PHASE] formula fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );
    print_inference_summary(&payload.inference_notes);
    if let Some(fit) = payload.fit_result.as_ref() {
        cli_out!(
            "{} fit | status={} | iterations={} | loglik={:.6e} | reml_score={} | raw_reml_score={}",
            payload.family,
            fit.convergence_evidence().inner_status().label(),
            fit.outer_iterations,
            fit.log_likelihood,
            gam::report::criterion_row(
                fit.comparable_reml_score()
                    .map_err(|err| format!("failed to compute comparable REML score: {err}"))?,
                fit.reml_score(),
                |value| format!("{value:.6e}"),
            ),
            gam::report::criterion_display(fit.reml_score()),
        );
    }
    write_payload_json(out, payload)
}

/// Refuse survival-only settings on a response that is not `Surv(...)`. Only the
/// survival fit path reads them, so on any other response they would be dropped
/// without a word. Both entry points check the resolved configuration, so a
/// `--request` document meets the same refusal as the flags.
fn refuse_survival_only_settings_without_surv(fit_config: &FitConfig) -> Result<(), String> {
    let survival_only = fit_config.baseline_scale.is_some()
        || fit_config.baseline_shape.is_some()
        || fit_config.baseline_rate.is_some()
        || fit_config.baseline_makeham.is_some()
        || fit_config.threshold_time_k.is_some()
        || fit_config.sigma_time_k.is_some()
        || fit_config.slope_time_k.is_some()
        || fit_config.survival_time_anchor.is_some()
        || !fit_config
            .resolved_survival_likelihood()
            .eq_ignore_ascii_case("transformation")
        || !fit_config.baseline_target.trim().eq_ignore_ascii_case("linear")
        || !fit_config.time_basis.trim().eq_ignore_ascii_case("ispline");
    if survival_only {
        return Err("survival-only options require a Surv(entry, exit, event) response".to_string());
    }
    if fit_config.noise_offset_column.is_some() && fit_config.noise_formula.is_none() {
        return Err("--noise-offset-column requires --predict-noise".to_string());
    }
    Ok(())
}

/// Refuse a family that conflicts with the response. A `Surv(...)` response selects
/// the survival fit path, which reads no family, so any family except royston-parmar
/// would be dropped without a word; royston-parmar names that path and needs a
/// `Surv(...)` response. Both entry points check the resolved configuration.
fn refuse_family_mismatched_with_the_response(
    fit_config: &FitConfig,
    is_survival: bool,
) -> Result<(), String> {
    let royston_parmar = fit_config
        .family
        .as_deref()
        .map(|name| name.eq_ignore_ascii_case("royston-parmar"));
    if is_survival && royston_parmar == Some(false) {
        return Err(
            "--family is ignored by Surv(...) fitting; use survival formula/link options"
                .to_string(),
        );
    }
    if !is_survival && royston_parmar == Some(true) {
        return Err(
            "--family royston-parmar requires a Surv(entry, exit, event) response".to_string(),
        );
    }
    Ok(())
}

pub(crate) fn validate_fit_args_preflight(
    args: &FitArgs,
    parsed: &ParsedFormula,
    fit_config: &FitConfig,
) -> Result<(), String> {
    if args.out.is_none() {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
    }
    // The flags resolve into the same document a --request supplies, so one set of
    // refusals reads the resolved configuration for both entry points. The family
    // routes (survival, transformation-normal, marginal-slope, location-scale) refuse
    // their own conflicting settings in the library materializers.
    let is_survival = parse_surv_response(&parsed.response)?.is_some();
    refuse_family_mismatched_with_the_response(fit_config, is_survival)?;
    if !is_survival {
        return refuse_survival_only_settings_without_surv(fit_config);
    }
    let likelihood = parse_survival_likelihood_mode(fit_config.resolved_survival_likelihood())?;
    gam::families::fit_orchestration::validate_survival_baseline_config(
        likelihood,
        &fit_config.baseline_target.trim().to_ascii_lowercase(),
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    validate_time_margin_args(
        "threshold_time_k (--threshold-time-k)",
        fit_config.threshold_time_k,
        fit_config.threshold_time_degree,
    )?;
    validate_time_margin_args(
        "sigma_time_k (--sigma-time-k)",
        fit_config.sigma_time_k,
        fit_config.sigma_time_degree,
    )?;
    validate_time_margin_args(
        "slope_time_k (--slope-time-k)",
        fit_config.slope_time_k,
        fit_config.slope_time_degree,
    )?;
    if fit_config.time_basis.trim().eq_ignore_ascii_case("ispline") {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
        )?;
    }
    Ok(())
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
                frozen_parametric_residualization: None,
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_term_primary_column(&SmoothTermSpec {
            frozen_parametric_residualization: None,
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

