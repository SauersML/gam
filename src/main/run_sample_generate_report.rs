use super::*;

pub(crate) fn run_sample(args: SampleArgs) -> Result<(), String> {
    validate_positive_optional_usize("--chains", args.chains)?;
    validate_positive_optional_usize("--samples", args.samples)?;
    validate_positive_optional_usize("--warmup", args.warmup)?;
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.set_stage("sample", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    progress.set_stage("sample", "loading sampling data");
    let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("sample", &args.data, ds.values.nrows())?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    let n_base_params = model
        .fit_result
        .as_ref()
        .map(|fr| fr.beta.len())
        .unwrap_or(0);
    let adaptive = NutsConfig::for_dimension(n_base_params);
    let cfg = NutsConfig {
        n_samples: args.samples.unwrap_or(adaptive.n_samples),
        nwarmup: args.warmup.unwrap_or(adaptive.nwarmup),
        n_chains: args.chains.unwrap_or(adaptive.n_chains),
        seed: args.seed.unwrap_or(adaptive.seed),
        ..adaptive
    };

    progress.set_stage("sample", "running posterior sampling");
    progress.teardown();
    // Unified dispatch over saved model class; the inference::sample module
    // routes Survival/Standard to their NUTS paths and every other class to
    // the Laplace-Gaussian fallback.
    let nuts = gam::sample::sample_saved_model(
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &cfg,
    )?;

    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".posterior.csv"));
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.advance_workflow(4);
    progress.set_stage("sample", "writing posterior draws");

    let n_coeffs = nuts.samples.ncols();
    let coeff_name = |j: usize| -> String { format!("beta_{j}") };

    // Write raw posterior samples CSV with appropriate column headers.
    {
        let headers: Vec<String> = (0..n_coeffs).map(&coeff_name).collect();
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&out)
            .map_err(|e| format!("failed to create output csv '{}': {e}", out.display()))?;
        wtr.write_record(&headers)
            .map_err(|e| format!("failed to write csv header: {e}"))?;
        for i in 0..nuts.samples.nrows() {
            let row: Vec<String> = (0..n_coeffs)
                .map(|j| format!("{:.12}", nuts.samples[[i, j]]))
                .collect();
            wtr.write_record(&row)
                .map_err(|e| format!("failed to write csv row {i}: {e}"))?;
        }
        wtr.flush()
            .map_err(|e| format!("failed to flush posterior samples csv: {e}"))?;
    }
    progress.advance_workflow(5);
    progress.finish_progress("sampling complete");
    cli_out!(
        "wrote posterior samples: {} (rows={}, cols={})",
        out.display(),
        nuts.samples.nrows(),
        nuts.samples.ncols()
    );

    // Print posterior coefficient summary with 95% credible intervals.
    cli_out!();
    cli_out!(
        "  {:<10} {:>12} {:>12} {:>12} {:>12}",
        "coeff",
        "post_mean",
        "post_std",
        "ci_2.5%",
        "ci_97.5%"
    );
    cli_out!("  {}", "-".repeat(62));
    for j in 0..n_coeffs {
        // Use posterior_mean_of to compute per-coefficient posterior mean from
        // the MCMC draws (functional API over the sample matrix).
        let pm = nuts.posterior_mean_of(|row| row[j]);
        let (lo, hi) = nuts.posterior_interval_of(|row| row[j], 2.5, 97.5);
        cli_out!(
            "  {:<10} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            coeff_name(j),
            pm,
            nuts.posterior_std[j],
            lo,
            hi,
        );
    }
    cli_out!();
    cli_out!(
        "  convergence: rhat={:.4}  ess={:.1}  converged={}",
        nuts.rhat,
        nuts.ess,
        nuts.converged
    );

    // Write per-coefficient posterior summary (mean, std, 95% CI) to CSV.
    let summary_path = out.with_extension("summary.csv");
    {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&summary_path)
            .map_err(|e| {
                format!(
                    "failed to create summary csv '{}': {e}",
                    summary_path.display()
                )
            })?;
        wtr.write_record([
            "coeff",
            "posterior_mean",
            "posterior_std",
            "ci_2.5",
            "ci_97.5",
        ])
        .map_err(|e| format!("failed to write summary csv header: {e}"))?;
        for j in 0..n_coeffs {
            let pm = nuts.posterior_mean_of(|row| row[j]);
            let (lo, hi) = nuts.posterior_interval_of(|row| row[j], 2.5, 97.5);
            wtr.write_record(&[
                coeff_name(j),
                format!("{pm:.8}"),
                format!("{:.8}", nuts.posterior_std[j]),
                format!("{lo:.8}"),
                format!("{hi:.8}"),
            ])
            .map_err(|e| format!("failed to write summary row: {e}"))?;
        }
        wtr.flush()
            .map_err(|e| format!("failed to flush summary csv: {e}"))?;
    }
    cli_out!("wrote posterior summary: {}", summary_path.display());

    Ok(())
}


pub(crate) fn run_generate(args: GenerateArgs) -> Result<(), String> {
    if args.n_draws == 0 {
        return Err("--n-draws must be > 0".to_string());
    }
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Generate", 5);
    progress.set_stage("generate", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);

    if model.predict_model_class() == PredictModelClass::Survival {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    progress.set_stage("generate", "loading conditioning data");
    let ds = load_datasetwith_model_schema(&args.data, &model)?;
    require_dataset_rows("generate", &args.data, ds.values.nrows())?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(&model);
    let (generate_offset, generate_noise_offset) = resolve_predict_offsets(
        &model,
        &ds,
        &col_map,
        saved_offset_column,
        saved_noise_offset_column,
    )?;
    progress.set_stage("generate", "building predictive state");
    let spec = run_generate_unified(
        &mut progress,
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &generate_offset,
        &generate_noise_offset,
        saved_noise_offset_column.is_some(),
    )?;
    progress.advance_workflow(3);

    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(42));
    progress.set_stage("generate", "sampling synthetic observations");
    let draws = sampleobservation_replicates(&spec, args.n_draws, &mut rng)
        .map_err(|e| format!("failed to sample synthetic observations: {e}"))?;
    progress.advance_workflow(4);

    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".generated.csv"));
    progress.set_stage("generate", "writing synthetic draws");
    // `sampleobservation_replicates` returns shape (n_draws, nobs): each
    // row is one synthetic observation vector. The natural CSV layout for
    // users is: one row per input row, one column per draw — so column
    // headers `draw_0..draw_{n_draws-1}` actually correspond to draws.
    // Without this transpose the headers were misleading: the file had
    // n_draws rows and nobs columns labeled "draw_*" even though each
    // column was really an observation index.
    let draws_per_row = draws.t().to_owned();
    write_matrix_csv(&out, &draws_per_row, "draw")?;
    progress.advance_workflow(5);
    progress.finish_progress("generation complete");
    cli_out!(
        "wrote synthetic draws: {} (input_rows={}, draws={})",
        out.display(),
        draws_per_row.nrows(),
        draws_per_row.ncols()
    );
    Ok(())
}


pub(crate) fn saved_likelihood_spec_for_generate(model: &SavedModel) -> Result<LikelihoodSpec, String> {
    match &model.payload().family_state {
        FittedFamily::Standard { likelihood, .. }
        | FittedFamily::LocationScale { likelihood, .. }
        | FittedFamily::MarginalSlope { likelihood, .. }
        | FittedFamily::Survival { likelihood, .. }
        | FittedFamily::TransformationNormal { likelihood } => Ok(likelihood.clone()),
        FittedFamily::LatentSurvival { .. } | FittedFamily::LatentBinary { .. } => Err(
            "generate is not available for latent survival/binary model family states".to_string(),
        ),
    }
}


/// Unified generate path: uses `PredictableModel` to produce a
/// `GenerativeSpec` for every non-survival model class.
///
/// For Gaussian LS the sigma vector is extracted via `predict_noise_scale`;
/// all other families derive their observation model from
/// `generativespec_from_predict`.
pub(crate) fn run_generate_unified(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam::generative::GenerativeSpec, String> {
    progress.set_stage("generate", "building unified generation design");

    let pred_input = build_predict_input_for_model(
        model,
        data,
        col_map,
        training_headers,
        offset,
        offset_noise,
        noise_offset_supplied,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "failed to build predictor for generate".to_string())?;

    let model_class = model.predict_model_class();
    let family = model.likelihood();
    let likelihood = saved_likelihood_spec_for_generate(model)?;

    if model_class == PredictModelClass::GaussianLocationScale {
        // Gaussian LS needs the per-observation sigma for its GenerativeSpec.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let sigma = predictor
            .predict_noise_scale(&pred_input)
            .map_err(|e| format!("predict_noise_scale failed: {e}"))?
            .ok_or_else(|| {
                "gaussian location-scale predictor did not produce sigma via predict_noise_scale"
                    .to_string()
            })?;
        Ok(gam::generative::GenerativeSpec {
            mean: pred.mean,
            noise: gam::generative::NoiseModel::Gaussian { sigma },
        })
    } else if model_class == PredictModelClass::DispersionLocationScale {
        // Dispersion location-scale (#913/#1125): the fit learned a per-row
        // precision surface `exp(eta_d(x))`. Thread it into the generative noise
        // model so synthetic data reproduces the non-constant dispersion,
        // exactly as the Gaussian-LS branch above threads per-row sigma. Without
        // this the family fell into the scalar `else` branch and generated
        // homoscedastic data at the seed dispersion.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        match predictor
            .predict_dispersion_scale(&pred_input)
            .map_err(|e| format!("predict_dispersion_scale failed: {e}"))?
        {
            Some(dispersion) => {
                let noise = gam::generative::NoiseModel::from_likelihood_with_per_row_dispersion(
                    &likelihood,
                    dispersion,
                )
                .map_err(|e| format!("failed to build per-row dispersion noise: {e}"))?;
                Ok(gam::generative::GenerativeSpec {
                    mean: pred.mean,
                    noise,
                })
            }
            None => {
                // No usable per-row precision channel (e.g. a dispersion family
                // fitted without a noise formula): fall back to the scalar
                // estimated dispersion.
                let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
                generativespec_from_predict(
                    pred,
                    likelihood,
                    family_noise_parameter(&fit_saved, family),
                )
                .map_err(|e| format!("failed to build generative spec: {e}"))
            }
        }
    } else {
        // Non-Gaussian models produce their response-scale plug-in mean
        // directly here.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        generativespec_from_predict(pred, likelihood, family_noise_parameter(&fit_saved, family))
            .map_err(|e| format!("failed to build generative spec: {e}"))
    }
}


/// Render the report for a spline-scan model (#1046) from its reconstructed
/// scalar quantities and the single smooth's EDF block, reusing the standard
/// `report::write_report` renderer. A scan model retains no dense design/Gram,
/// so there is no coefficient table or data-dependent diagnostics surface here
/// — the headline EDF / λ / REML / deviance are recovered exactly from the
/// saved `SplineScanFit`, matching the FFI `summary()` path.
pub(crate) fn run_report_spline_scan(
    mut progress: gam::visualizer::VisualizerSession,
    args: &ReportArgs,
    model: &SavedModel,
    feature_column: &str,
    scan: &gam::solver::spline_scan::SplineScanFit,
) -> Result<(), String> {
    progress.advance_workflow(1);
    progress.set_stage("report", "generating html");
    let mut notes = vec![format!(
        "Exact O(n) state-space spline scan for s({feature_column}): \
         λ={:.4e}, EDF={:.3}, knots={}. The smoother retains the per-knot \
         posterior, not a dense design/Gram, so no coefficient table is shown.",
        scan.lambda(),
        scan.edf(),
        scan.knots.len(),
    )];
    if args.data.is_some() {
        notes.push(
            "Data provided, but held-out diagnostics for spline-scan models are \
             served through the Python diagnose() / predict() path; the CLI \
             report shows the fitted scalar quantities only."
                .to_string(),
        );
    }
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs: Some(scan.n_obs()),
        deviance: scan.deviance(),
        reml_score: -scan.restricted_loglik,
        iterations: 0,
        convergence_status: "exact (state-space spline scan)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        edf_total: scan.edf(),
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: vec![report::EdfBlockRow {
            index: 0,
            edf: scan.edf(),
            role: Some("smooth".to_string()),
        }],
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;
    progress.finish_progress("report complete");
    cli_out!("wrote report: {}", out.display());
    Ok(())
}


pub(crate) fn run_report(args: ReportArgs) -> Result<(), String> {
    use gam::probability::standard_normal_quantile;

    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let report_total_steps = if args.data.is_some() { 5 } else { 3 };
    progress.start_workflow("Report", report_total_steps);
    progress.set_stage("report", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    // Spline-scan model (#1030/#1034/#1046): no dense fit_result exists — the
    // exact O(n) state-space smoother keeps only the per-knot posterior. Render
    // the report from the reconstructed scalar quantities (the same EDF / λ /
    // REML the fit log prints) and return, instead of demanding a dense fit.
    if let Some((feature_column, scan)) = model
        .saved_spline_scan()
        .map_err(|e| e.to_string())?
        .map(|(c, f)| (c.to_string(), f))
    {
        return run_report_spline_scan(progress, &args, &model, &feature_column, &scan);
    }
    let family = model.likelihood();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    progress.advance_workflow(1);

    let beta_se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());

    let coefficients: Vec<report::CoefficientRow> = fit
        .beta
        .iter()
        .copied()
        .enumerate()
        .map(|(i, b)| report::CoefficientRow {
            index: i,
            estimate: b,
            std_error: beta_se.and_then(|s| s.get(i).copied()),
        })
        .collect();

    let edf_blocks: Vec<report::EdfBlockRow> = if let Some(unified) = model.unified() {
        unified
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| report::EdfBlockRow {
                index: i,
                edf: block.edf,
                role: Some(block_role_label(&block.role).to_string()),
            })
            .collect()
    } else {
        fit.edf_by_block()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, edf)| report::EdfBlockRow {
                index: i,
                edf,
                role: None,
            })
            .collect()
    };

    let mut notes = Vec::new();
    if let Some(unified) = model.unified() {
        if unified.blocks.len() > 1 {
            let role_labels: Vec<&str> = unified
                .blocks
                .iter()
                .map(|b| block_role_label(&b.role))
                .collect();
            notes.push(format!("Block roles: {}", role_labels.join(", ")));
        }
        notes.push(format!(
            "Outer iterations: {} (status: {})",
            unified.outer_iterations,
            unified.pirls_status.label()
        ));
        notes.push(format!(
            "Log-likelihood: {:.4}, penalized objective: {:.4}",
            unified.log_likelihood, unified.penalized_objective
        ));
    }
    let mut diagnostics = None;
    let mut smooth_plots = Vec::new();
    let mut continuous_order = Vec::new();
    let mut measure_jet_spectra = Vec::new();
    let mut alo_data = None;
    let mut n_obs = None;
    let mut r_squared = None;

    if let Some(data_path) = args.data.as_ref() {
        progress.set_stage("report", "loading report dataset");
        let ds = load_datasetwith_model_schema_for_diagnostics(data_path, &model)?;
        require_dataset_rows("report", data_path, ds.values.nrows())?;
        progress.advance_workflow(2);

        let col_map = ds.column_map();
        let training_headers = model.training_headers.as_ref();
        let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(&model);
        let parsed = parse_formula(&model.formula)?;

        if let Some(y_col) = col_map.get(&parsed.response).copied() {
            if model.predict_model_class() == PredictModelClass::BernoulliMarginalSlope {
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());
                if let Some(predictor) = model.predictor() {
                    let (report_offset, report_noise_offset) = resolve_predict_offsets(
                        &model,
                        &ds,
                        &col_map,
                        saved_offset_column,
                        saved_noise_offset_column,
                    )?;
                    let pred_input = build_predict_input_for_model(
                        &model,
                        ds.values.view(),
                        &col_map,
                        training_headers,
                        &report_offset,
                        &report_noise_offset,
                        saved_noise_offset_column.is_some(),
                    )?;
                    progress.set_stage("report", "building report diagnostics design");
                    progress.advance_workflow(3);
                    let pred = predictor
                        .predict_plugin_response(&pred_input)
                        .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;

                    let residuals: Vec<f64> =
                        y.iter().zip(pred.mean.iter()).map(|(o, p)| o - p).collect();
                    let mut residuals_sorted = residuals.clone();
                    residuals_sorted
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = residuals_sorted.len().max(1);
                    let theoretical_quantiles = (0..n)
                        .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut bin_pred = [0.0f64; 10];
                    let mut bin_obs = [0.0f64; 10];
                    let mut counts = [0usize; 10];
                    for i in 0..y.len() {
                        let p = pred.mean[i].clamp(0.0, 1.0);
                        let b = ((p * 10.0).floor() as usize).min(9);
                        bin_pred[b] += p;
                        bin_obs[b] += y[i];
                        counts[b] += 1;
                    }
                    let mut mp = Vec::new();
                    let mut or = Vec::new();
                    for b in 0..10 {
                        if counts[b] > 0 {
                            mp.push(bin_pred[b] / counts[b] as f64);
                            or.push((bin_obs[b] / counts[b] as f64).clamp(0.0, 1.0));
                        }
                    }
                    diagnostics = Some(report::DiagnosticsInput {
                        residuals_sorted,
                        theoretical_quantiles,
                        y_observed: y.to_vec(),
                        y_predicted: pred.mean.to_vec(),
                        calibration: Some(report::CalibrationData {
                            mean_predicted: mp,
                            observed_rate: or,
                        }),
                    });
                }
            } else if matches!(
                model.predict_model_class(),
                PredictModelClass::Standard | PredictModelClass::BinomialLocationScale
            ) {
                let spec = resolve_termspec_for_prediction(
                    &model.resolved_termspec,
                    training_headers,
                    &col_map,
                    "resolved_termspec",
                )?;
                progress.set_stage("report", "building report diagnostics design");
                let design = build_term_collection_design(ds.values.view(), &spec)
                    .map_err(|e| format!("failed to build design for report diagnostics: {e}"))?;
                progress.advance_workflow(3);

                let (offset, _report_noise_offset) = report_offset_for(&model, &ds, &col_map)?;
                let pred = predict_gam(
                    design.design.clone(),
                    fit.beta.view(),
                    offset.view(),
                    family.clone(),
                )
                .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());

                // R-squared for Gaussian
                if family.is_gaussian_identity() {
                    let y_mean = y.mean().unwrap_or(0.0);
                    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
                    let ss_res: f64 = y
                        .iter()
                        .zip(pred.mean.iter())
                        .map(|(&yi, &pi)| (yi - pi).powi(2))
                        .sum();
                    if ss_tot > 1e-15 {
                        r_squared = Some(1.0 - ss_res / ss_tot);
                    }
                }

                // Continuous smoothness order
                let reportweights = Array1::<f64>::ones(ds.values.nrows());
                let summary = build_model_summary(
                    &design,
                    &spec,
                    &fit,
                    family.clone(),
                    y.view(),
                    reportweights.view(),
                );
                for st in &summary.smooth_terms {
                    if let Some(ord) = st.continuous_order.as_ref() {
                        let status = match ord.status {
                            ContinuousSmoothnessOrderStatus::Ok => "Ok",
                            ContinuousSmoothnessOrderStatus::NonMaternRegime => "Non-Matern",
                            ContinuousSmoothnessOrderStatus::FirstOrderLimit => "1st-Order Limit",
                            ContinuousSmoothnessOrderStatus::IntrinsicLimit => "Intrinsic Limit",
                            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "Undef",
                        };
                        let fin = |v: Option<f64>| v.filter(|x| x.is_finite());
                        continuous_order.push(report::ContinuousOrderRow {
                            name: st.name.clone(),
                            lambda0: ord.lambda0,
                            lambda1: ord.lambda1,
                            lambda2: ord.lambda2,
                            r_ratio: fin(ord.r_ratio),
                            nu: fin(ord.nu),
                            kappa2: fin(ord.kappa2),
                            status: status.to_string(),
                        });
                    }
                }

                // Measure-jet scale spectrum: realized band per term, plus
                // the per-scale fitted λ̂_ℓ and implied order when the term
                // carries one non-ridge λ per band scale (per-scale-candidate
                // mode); a single fused jet-energy penalty reports only the
                // band and the spec's order. The implied-order diagnostic uses
                // λ_raw = λ̃ / ||S_raw,ℓ||_F, before the arbitrary Mellin
                // ε_ℓ^(-2s0)·log_step gauge is folded into the fit-time forms.
                {
                    let mut penalty_cursor = design.random_effect_ranges.len();
                    for term in &design.smooth.terms {
                        let k = term.penalties_local.len();
                        let term_penalty_start = penalty_cursor;
                        penalty_cursor += k;
                        let gam::basis::BasisMetadata::MeasureJet {
                            eps_band,
                            length_scale,
                            order_s,
                            raw_penalty_normalization_scales,
                            ..
                        } = &term.metadata
                        else {
                            continue;
                        };
                        let (Some(&eps_min), Some(&eps_max)) = (eps_band.first(), eps_band.last())
                        else {
                            continue;
                        };
                        let mut scale_lambdas = vec![None; eps_band.len()];
                        for idx in term_penalty_start..term_penalty_start + k {
                            let (Some(info), Some(&lambda_tilde)) =
                                (design.penaltyinfo.get(idx), fit.lambdas.get(idx))
                            else {
                                break;
                            };
                            let gam::basis::PenaltySource::Other(label) = &info.penalty.source
                            else {
                                continue;
                            };
                            let Some(level_txt) = label.strip_prefix("measure_jet_scale_") else {
                                continue;
                            };
                            let Ok(level) = level_txt.parse::<usize>() else {
                                continue;
                            };
                            let Some(&c_raw) = raw_penalty_normalization_scales.get(level) else {
                                continue;
                            };
                            if level < scale_lambdas.len() && c_raw.is_finite() && c_raw > 0.0 {
                                scale_lambdas[level] = Some(lambda_tilde / c_raw);
                            }
                        }
                        // Per-scale-candidate mode ⇔ exactly one non-ridge λ
                        // per band scale, and at least two scales (one point
                        // has no slope to regress).
                        let per_scale: Vec<(f64, f64)> =
                            if scale_lambdas.iter().all(Option::is_some) && eps_band.len() >= 2 {
                                eps_band
                                    .iter()
                                    .copied()
                                    .zip(scale_lambdas.into_iter().flatten())
                                    .collect()
                            } else {
                                Vec::new()
                            };
                        let implied_order = measure_jet_implied_order(&per_scale);
                        measure_jet_spectra.push(report::MeasureJetSpectrumRow {
                            term_name: term.name.clone(),
                            eps_min,
                            eps_max,
                            n_scales: eps_band.len(),
                            length_scale: *length_scale,
                            spec_order_s: *order_s,
                            per_scale,
                            implied_order,
                        });
                    }
                }

                // Residual QQ data
                let residuals: Vec<f64> =
                    y.iter().zip(pred.mean.iter()).map(|(o, p)| o - p).collect();
                let mut residuals_sorted = residuals.clone();
                residuals_sorted
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = residuals_sorted.len().max(1);
                let theoretical_quantiles = (0..n)
                    .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                    .collect::<Result<Vec<_>, _>>()?;

                // Calibration for binary responses
                let calibration = if is_binary_response(y.view()) {
                    let mut bin_pred = [0.0f64; 10];
                    let mut bin_obs = [0.0f64; 10];
                    let mut counts = [0usize; 10];
                    for i in 0..y.len() {
                        let p = pred.mean[i].clamp(0.0, 1.0);
                        let b = ((p * 10.0).floor() as usize).min(9);
                        bin_pred[b] += p;
                        bin_obs[b] += y[i];
                        counts[b] += 1;
                    }
                    let mut mp = Vec::new();
                    let mut or = Vec::new();
                    for b in 0..10 {
                        if counts[b] > 0 {
                            mp.push(bin_pred[b] / counts[b] as f64);
                            or.push((bin_obs[b] / counts[b] as f64).clamp(0.0, 1.0));
                        }
                    }
                    Some(report::CalibrationData {
                        mean_predicted: mp,
                        observed_rate: or,
                    })
                } else {
                    None
                };

                diagnostics = Some(report::DiagnosticsInput {
                    residuals_sorted,
                    theoretical_quantiles,
                    y_observed: y.to_vec(),
                    y_predicted: pred.mean.to_vec(),
                    calibration,
                });

                // ALO diagnostics: try geometry-based path from unified
                // result first, fall back to PIRLS-based path.
                if let Some(link) = model
                    .resolved_inverse_link()
                    .ok()
                    .and_then(|r| r.map(|lk| lk.link_function()))
                {
                    let alo_result = if let Some(unified) = model.unified() {
                        let (report_offset, _report_noise_offset) =
                            report_offset_for(&model, &ds, &col_map)?;
                        let eta = &design.design.dot(&fit.beta) + &report_offset;
                        let dense_alo_design = design.design.to_dense();
                        // φ must match the PIRLS-backed refit fallback: Gaussian
                        // (Identity) uses σ̂², not a hard-coded 1.0, or the
                        // reported ALO SEs are off by √φ̂ (#881-class).
                        let phi = geometry_alo_phi(unified, link);
                        gam::alo::compute_alo_diagnostics_from_unified(
                            unified,
                            &dense_alo_design,
                            &eta,
                            &report_offset,
                            link,
                            phi,
                        )
                    } else {
                        compute_alo_diagnostics_from_fit(&fit, y.view(), link)
                    };
                    match alo_result {
                        Ok(alo) => {
                            alo_data = Some(report::AloData {
                                rows: (0..alo.leverage.len())
                                    .map(|i| report::AloRow {
                                        index: i,
                                        leverage: alo.leverage[i],
                                        eta_tilde: alo.eta_tilde[i],
                                        se_sandwich: alo.se_sandwich[i],
                                    })
                                    .collect(),
                            });
                        }
                        Err(e) => notes.push(format!("ALO diagnostics unavailable: {e}")),
                    }
                }

                // Smooth term partial-effect plots
                for st in &spec.smooth_terms {
                    if let Some(col) = smooth_term_primary_column(st)
                        && col < ds.values.ncols()
                        && let Some(dt) = design.smooth.terms.iter().find(|t| t.name == st.name)
                    {
                        let x_col = ds.values.column(col);
                        let dense_for_smooth = design.design.to_dense();
                        let contrib = dense_for_smooth
                            .slice(s![.., dt.coeff_range.clone()])
                            .dot(&fit.beta.slice(s![dt.coeff_range.clone()]));
                        let mut pairs: Vec<(f64, f64)> =
                            x_col.iter().copied().zip(contrib.iter().copied()).collect();
                        pairs.sort_by(|a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        smooth_plots.push(report::SmoothPlotData {
                            name: st.name.clone(),
                            x: pairs.iter().map(|p| p.0).collect(),
                            y: pairs.iter().map(|p| p.1).collect(),
                        });
                    }
                }
            }
        }
    } else {
        notes.push(
            "No data provided \u{2014} diagnostics are omitted. \
             Pass training data as the second positional argument."
                .to_string(),
        );
        progress.advance_workflow(2);
    }

    // The realized band is frozen onto the saved spec, so the measure-jet
    // spectrum line still prints when the report runs without a dataset (or
    // for model classes that skip the design rebuild). Per-scale λ̂_ℓ need the
    // rebuilt penalty layout, so only band + spec order are available here.
    if measure_jet_spectra.is_empty() {
        measure_jet_spectra = measure_jet_spectrum_rows_from_spec(model.resolved_termspec.as_ref());
    }

    progress.set_stage("report", "generating html");
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: family.pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        convergence_status: fit.pirls_status.label().to_string(),
        converged: fit.pirls_status.is_converged(),
        outer_gradient_norm: fit.outer_gradient_norm,
        criterion_certificate: fit.artifacts.criterion_certificate.as_ref().map(|cert| {
            report::CriterionCertificateRow {
                analytic_directional: cert.analytic_directional,
                fd_directional: cert.fd_directional,
                fd_error: cert.fd_error,
                agreement_z: cert.agreement_z,
                grad_norm: cert.grad_norm,
                hessian_pd: cert.hessian_pd,
                lambdas_railed: cert.lambdas_railed.clone(),
                consistent: cert.first_order_consistent(),
                clean: cert.is_clean(),
            }
        }),
        edf_total: model
            .unified()
            .and_then(|u| u.edf_total())
            .unwrap_or_else(|| fit.edf_total().unwrap_or(0.0)),
        r_squared,
        coefficients,
        edf_blocks,
        continuous_order,
        anisotropic_scales: build_anisotropic_scales_rows(model.resolved_termspec.as_ref()),
        measure_jet_spectra,
        diagnostics,
        smooth_plots,
        alo: alo_data,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;

    progress.advance_workflow(report_total_steps);
    progress.finish_progress("report complete");
    cli_out!("wrote report: {}", out.display());

    // Terminal quick-look: a unicode sparkline of each smooth term's fitted
    // partial effect, straight from the values we already computed for the
    // HTML. This is purely a rendering of `input.smooth_plots` — it reads the
    // fitted contributions and touches no fit/REML/prediction value.
    if !input.smooth_plots.is_empty() {
        cli_out!("smooth terms:");
        for sp in &input.smooth_plots {
            cli_out!(
                "{}",
                gam::sparkline::render_smooth_line(&sp.name, &sp.x, &sp.y)
            );
        }
    }
    Ok(())
}
