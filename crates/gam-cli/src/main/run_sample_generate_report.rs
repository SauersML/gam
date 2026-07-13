use super::*;

pub(crate) fn run_sample(args: SampleArgs) -> Result<(), String> {
    validate_positive_optional_usize("--chains", args.chains)?;
    validate_positive_optional_usize("--samples", args.samples)?;
    validate_positive_optional_usize("--warmup", args.warmup)?;
    reject_multinomial_model(&args.model, "sample")?;
    let model = SavedModel::load_from_path(&args.model)?;
    let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("sample", &args.data, ds.values.nrows())?;
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
    reject_multinomial_model(&args.model, "generate")?;
    let model = SavedModel::load_from_path(&args.model)?;

    if model.predict_model_class() == PredictModelClass::Survival {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    let ds = load_datasetwith_model_schema(&args.data, &model)?;
    require_dataset_rows("generate", &args.data, ds.values.nrows())?;
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
    let spec = run_generate_unified(
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &generate_offset,
        &generate_noise_offset,
        saved_noise_offset_column.is_some(),
    )?;

    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(42));
    let draws = sampleobservation_replicates(&spec, args.n_draws, &mut rng)
        .map_err(|e| format!("failed to sample synthetic observations: {e}"))?;

    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".generated.csv"));
    // `sampleobservation_replicates` returns shape (n_draws, nobs): each
    // row is one synthetic observation vector. The natural CSV layout for
    // users is: one row per input row, one column per draw — so column
    // headers `draw_0..draw_{n_draws-1}` actually correspond to draws.
    // Without this transpose the headers were misleading: the file had
    // n_draws rows and nobs columns labeled "draw_*" even though each
    // column was really an observation index.
    let draws_per_row = draws.t().to_owned();
    write_matrix_csv(&out, &draws_per_row, "draw")?;
    cli_out!(
        "wrote synthetic draws: {} (input_rows={}, draws={})",
        out.display(),
        draws_per_row.nrows(),
        draws_per_row.ncols()
    );
    Ok(())
}

/// Thin CLI adapter over the canonical saved-model generative capability.
///
/// A weighted saved model must recover its requested-row weight values from
/// the persisted column name. `None` is passed only for a genuinely unweighted
/// fit; a missing named column is an error, never a unit-weight substitution.
pub(crate) fn run_generate_unified(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam::generative::GenerativeSpec, String> {
    let prior_weights = match model.payload().weight_column.as_deref() {
        Some(column) => {
            let index = *col_map.get(column).ok_or_else(|| {
                format!(
                    "generate requires saved row-weight column {column:?}; unit-weight \
                     substitution would change the fitted observation law"
                )
            })?;
            if index >= data.ncols() {
                return Err(format!(
                    "generate row-weight column {column:?} resolves to {index}, outside the \
                     {}-column input",
                    data.ncols(),
                ));
            }
            Some(data.column(index).to_owned())
        }
        None => None,
    };
    gam_predict::generative_spec_for_saved_model(
        model,
        gam_predict::SavedGenerativeInput {
            data,
            col_map,
            training_headers,
            offset,
            offset_noise,
            noise_offset_supplied,
            prior_weights: prior_weights.as_ref(),
        },
    )
    .map_err(|error| error.to_string())
}

fn smoothing_forensics_rows(
    fit: &UnifiedFitResult,
    edf_blocks: &[report::EdfBlockRow],
) -> Vec<report::SmoothingForensicsRow> {
    let sigma2 = (fit.standard_deviation * fit.standard_deviation).max(0.0);
    let assembly_edfs = fit.edf_by_block();
    fit.blocks
        .iter()
        .enumerate()
        .map(|(block_idx, block)| {
            let lambda_path = block.lambdas.iter().copied().collect::<Vec<_>>();
            let edf_criterion = edf_blocks
                .iter()
                .find(|row| row.index == block_idx)
                .map(|row| row.edf)
                .or(Some(block.edf));
            let edf_assembly = assembly_edfs.get(block_idx).copied().or(Some(block.edf));
            let role = block_role_label(&block.role);
            report::SmoothingForensicsRow {
                term: format!("block {block_idx} ({role})"),
                lambda_path,
                sigma2_path: vec![sigma2],
                edf_criterion,
                edf_assembly,
                double_penalty_range: None,
                double_penalty_null_space: fit.artifacts.null_space_dim.and_then(|dim| {
                    if dim > 0 && block_idx == 0 {
                        Some(dim as f64)
                    } else {
                        None
                    }
                }),
                seed_screening: Vec::new(),
            }
        })
        .collect()
}

/// Render the report for a spline-scan model (#1046) from its reconstructed
/// scalar quantities and the single smooth's EDF block, reusing the standard
/// `report::write_report` renderer. A scan model retains no dense design/Gram,
/// so there is no coefficient table or data-dependent diagnostics surface here
/// — the headline EDF / λ / REML / deviance are recovered exactly from the
/// saved `SplineScanFit`, matching the FFI `summary()` path.
pub(crate) fn run_report_spline_scan(
    args: &ReportArgs,
    model: &SavedModel,
    feature_column: &str,
    scan: &gam::solver::spline_scan::SplineScanFit,
) -> Result<(), String> {
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
        smoothing_forensics: Vec::new(),
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
    cli_out!("wrote report: {}", out.display());
    Ok(())
}

pub(crate) fn run_report_residual_cascade(
    args: &ReportArgs,
    model: &SavedModel,
    feature_columns: &[String],
    fit: &gam::solver::residual_cascade::ResidualCascadeFit,
) -> Result<(), String> {
    let lambda = fit.lambda();
    let mut notes = vec![format!(
        "Exact O(n log n) multiresolution residual cascade for s({features}): \
         λ={lambda:.4e}, σ²={sigma2:.4e}, levels={levels}, centers={centers}, \
         coeffs={coeffs}. CG backward error={cg_resid:.2e} over {cg_iters} \
         iterations. The cascade retains the multilevel Wendland posterior, not \
         a dense design/Gram, so no coefficient table is shown.",
        features = feature_columns.join(", "),
        sigma2 = fit.sigma2,
        levels = fit.num_levels(),
        centers = fit.num_centers(),
        coeffs = fit.num_coeffs(),
        cg_resid = fit.certificate.solve_rel_residual,
        cg_iters = fit.certificate.solve_iters,
    )];
    if let Some(refinement) = fit.refinement.as_ref() {
        notes.push(format!(
            "Refinement loop ran past the initial levels; the level-(L+1) \
             discretization certificate bounds the remaining penalized-objective \
             gain at {:.2e}.",
            refinement.next_level_gain_bound,
        ));
    }
    if args.data.is_some() {
        notes.push(
            "Data provided, but held-out diagnostics for residual-cascade models \
             are served through the predict() path; the CLI report shows the \
             fitted scalar quantities only."
                .to_string(),
        );
    }
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs: None,
        // Gaussian-identity deviance ≡ the penalized residual quadratic
        // `y'Wy − ĉ'X'Wy` the fit profiles σ² from.
        deviance: fit.rss_pen,
        reml_score: -fit.restricted_loglik,
        iterations: 0,
        convergence_status: "exact (multiresolution residual cascade)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        smoothing_forensics: Vec::new(),
        edf_total: 0.0,
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: Vec::new(),
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;
    cli_out!("wrote report: {}", out.display());
    Ok(())
}

pub(crate) fn run_report(args: ReportArgs) -> Result<(), String> {
    reject_multinomial_model(&args.model, "report")?;
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
        return run_report_spline_scan(&args, &model, &feature_column, &scan);
    }
    // Residual-cascade model (#1032): the multi-resolution analogue of the
    // spline scan. The exact O(n log n) multilevel Wendland smoother keeps only
    // the nested ε-net posterior (no dense fit_result), so render the report
    // from its reconstructed scalar quantities and return.
    if let Some((feature_columns, fit)) = model
        .saved_residual_cascade()
        .map_err(|e| e.to_string())?
        .map(|(c, f)| (c.to_vec(), f))
    {
        return run_report_residual_cascade(&args, &model, &feature_columns, &fit);
    }
    let family = model.likelihood();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    // Total EDF: shown on the summary card and used as the residual degrees
    // of freedom in the dispersion estimates behind the report residuals.
    let edf_total = model
        .unified()
        .and_then(|u| u.edf_total())
        .unwrap_or_else(|| fit.edf_total().unwrap_or(0.0));

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
            unified.convergence_evidence().inner_status().label()
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
        let ds = load_datasetwith_model_schema_for_diagnostics(data_path, &model)?;
        require_dataset_rows("report", data_path, ds.values.nrows())?;

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
                    let pred = predictor
                        .predict_plugin_response(&pred_input)
                        .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;

                    // Bernoulli response: randomized-quantile residuals (the
                    // raw y − p residual is two-valued and can never track a
                    // normal Q-Q reference), plus equal-count calibration
                    // deciles.
                    let y_vec = y.to_vec();
                    let p_vec = pred.mean.to_vec();
                    let residuals = report_residual_diagnostics(
                        &ResponseFamily::Binomial,
                        &y_vec,
                        &p_vec,
                        None,
                        edf_total,
                        &mut notes,
                    )?;
                    let calibration = binary_calibration_deciles(&y_vec, &p_vec);
                    diagnostics = Some(report::DiagnosticsInput {
                        residuals,
                        y_observed: y_vec,
                        y_predicted: p_vec,
                        calibration,
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
                let design = build_term_collection_design(ds.values.view(), &spec)
                    .map_err(|e| format!("failed to build design for report diagnostics: {e}"))?;

                let (offset, _report_noise_offset) = report_offset_for(&model, &ds, &col_map)?;
                let effective_offset = design
                    .compose_offset(offset.view(), "report saved-model design")
                    .map_err(|error| error.to_string())?;
                let pred = predict_gam(
                    design.design.clone(),
                    fit.beta.view(),
                    effective_offset.view(),
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
                )?;
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
                    let mut penalty_cursor = design.leading_penalty_blocks_before_smooth();
                    for term in &design.smooth.terms {
                        let k = term.active_penalties.len();
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
                        let effective_report_offset = design
                            .compose_offset(report_offset.view(), "report ALO design")
                            .map_err(|error| error.to_string())?;
                        let eta = &design.design.dot(&fit.beta) + &effective_report_offset;
                        let dense_alo_design = design.design.to_dense();
                        // φ must match the PIRLS-backed refit fallback: Gaussian
                        // (Identity) uses σ̂², not a hard-coded 1.0, or the
                        // reported ALO SEs are off by √φ̂ (#881-class).
                        let phi = geometry_alo_phi(unified, link);
                        gam::alo::compute_alo_diagnostics_from_unified(
                            unified,
                            &dense_alo_design,
                            &eta,
                            &effective_report_offset,
                            phi,
                        )
                    } else {
                        compute_alo_diagnostics_from_fit(&fit, y.view())
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

                // Residual diagnostics: family-appropriate residuals with a
                // standard-normal null (randomized-quantile / deviance),
                // leverage-aware where ALO hat values are available, plus
                // equal-count calibration deciles for binary responses. Raw
                // y − μ residuals fail a normal Q-Q by construction for
                // non-Gaussian families (Bernoulli residuals are two-valued;
                // Poisson residual variance grows with μ).
                let y_vec = y.to_vec();
                let mu_vec = pred.mean.to_vec();
                let leverage: Option<Vec<f64>> = alo_data
                    .as_ref()
                    .map(|a| a.rows.iter().map(|r| r.leverage).collect());
                let residuals = report_residual_diagnostics(
                    &family.response,
                    &y_vec,
                    &mu_vec,
                    leverage.as_deref(),
                    edf_total,
                    &mut notes,
                )?;
                let calibration = if is_binary_response(y.view()) {
                    binary_calibration_deciles(&y_vec, &mu_vec)
                } else {
                    None
                };
                diagnostics = Some(report::DiagnosticsInput {
                    residuals,
                    y_observed: y_vec,
                    y_predicted: mu_vec,
                    calibration,
                });

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
    }

    // The realized band is frozen onto the saved spec, so the measure-jet
    // spectrum line still prints when the report runs without a dataset (or
    // for model classes that skip the design rebuild). Per-scale λ̂_ℓ need the
    // rebuilt penalty layout, so only band + spec order are available here.
    if measure_jet_spectra.is_empty() {
        measure_jet_spectra = measure_jet_spectrum_rows_from_spec(model.resolved_termspec.as_ref());
    }

    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: family.pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        convergence_status: fit
            .convergence_evidence()
            .inner_status()
            .label()
            .to_string(),
        converged: true,
        outer_gradient_norm: fit.outer_gradient_norm,
        criterion_certificate: fit.artifacts.criterion_certificate.as_ref().map(|cert| {
            let stationarity = match &cert.stationarity {
                gam::model_types::OuterStationarityCertificate::AnalyticGradient {
                    grad_norm,
                    projected_grad_norm,
                    bound,
                } => report::CriterionStationarityRow::AnalyticGradient {
                    grad_norm: *grad_norm,
                    projected_grad_norm: *projected_grad_norm,
                    bound: *bound,
                },
                gam::model_types::OuterStationarityCertificate::FixedPoint {
                    residual_inf_norm,
                    projected_residual_inf_norm,
                    bound,
                    covered_coordinates,
                } => report::CriterionStationarityRow::FixedPoint {
                    residual_inf_norm: *residual_inf_norm,
                    projected_residual_inf_norm: *projected_residual_inf_norm,
                    bound: *bound,
                    covered_coordinates: *covered_coordinates,
                },
            };
            report::CriterionCertificateRow {
                stationarity,
                hessian_psd: cert.hessian_psd,
                lambdas_railed: cert.lambdas_railed.clone(),
                stationary: cert.is_stationary(),
                clean: cert.is_clean(),
            }
        }),
        smoothing_forensics: smoothing_forensics_rows(&fit, &edf_blocks),
        edf_total,
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
                gam::report::sparkline::render_smooth_line(&sp.name, &sp.x, &sp.y)
            );
        }
    }
    Ok(())
}

/// Deterministic seed for the randomization component of Dunn–Smyth
/// quantile residuals: reports must be identical run to run.
const REPORT_RESIDUAL_SEED: u64 = 0x0D5E_ED11;

/// Observation-aligned residuals whose null distribution is standard normal
/// under a correct model — exactly so for the randomized-quantile families,
/// asymptotically for Tweedie deviance residuals.
struct FamilyResiduals {
    values: Vec<f64>,
    label: &'static str,
}

/// Build the report's residual diagnostics for one fitted family, or `None`
/// (with an explanatory note) when no residual definition with a
/// standard-normal null is available — the residual plots are omitted rather
/// than drawn against a false normal reference.
fn report_residual_diagnostics(
    response: &ResponseFamily,
    y: &[f64],
    mu: &[f64],
    leverage: Option<&[f64]>,
    edf_total: f64,
    notes: &mut Vec<String>,
) -> Result<Option<report::ResidualDiagnostics>, String> {
    match report_family_residuals(response, y, mu, leverage, edf_total) {
        Ok(res) => {
            let mut sorted = res.values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len().max(1);
            let theoretical_quantiles = (0..n)
                .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(report::ResidualDiagnostics {
                values: res.values,
                sorted,
                theoretical_quantiles,
                label: res.label.to_string(),
            }))
        }
        Err(reason) => {
            notes.push(format!("Residual diagnostics omitted: {reason}"));
            Ok(None)
        }
    }
}

/// Family-appropriate residuals on the N(0,1) scale (Dunn & Smyth 1996).
///
/// Continuous families map the response through its own fitted CDF and then
/// Φ⁻¹; discrete families draw u ~ U(F(y−1), F(y)) so u is exactly U(0,1)
/// under a correct model. Gaussian uses the internally-studentized residual
/// (equivalent to its quantile residual) with the √(1 − h_ii) leverage factor
/// when hat values are available. Dispersions that the saved model does not
/// carry (Gamma φ, Tweedie φ) use the Pearson estimate with `n − edf` degrees
/// of freedom.
fn report_family_residuals(
    response: &ResponseFamily,
    y: &[f64],
    mu: &[f64],
    leverage: Option<&[f64]>,
    edf_total: f64,
) -> Result<FamilyResiduals, String> {
    use rand::RngExt;
    use statrs::distribution::{Beta, Discrete, DiscreteCDF, Gamma, NegativeBinomial, Poisson};

    let n = y.len().min(mu.len());
    if n == 0 {
        return Err("no observations".to_string());
    }
    // Residual degrees of freedom for the Pearson dispersion estimates.
    let residual_dof = (n as f64 - edf_total).max(1.0);
    let mut rng = StdRng::seed_from_u64(REPORT_RESIDUAL_SEED);
    // Predictive CDF value → normal scale, clamped away from the exact
    // endpoints so every plotted point stays finite.
    let to_normal = |u: f64| standard_normal_quantile(u.clamp(1e-12, 1.0 - 1e-12));

    match response {
        ResponseFamily::Gaussian => {
            let ssr: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
            let sigma = (ssr / residual_dof).sqrt();
            if !(sigma.is_finite() && sigma > 0.0) {
                return Err("Gaussian residual scale is zero or non-finite".to_string());
            }
            let values = (0..n)
                .map(|i| {
                    // Var(y_i − μ̂_i) = σ²(1 − h_ii): without the leverage
                    // factor even a correct Gaussian fit under-disperses at
                    // high-leverage rows.
                    let h = leverage
                        .and_then(|l| l.get(i).copied())
                        .filter(|h| h.is_finite())
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0 - 1e-8);
                    (y[i] - mu[i]) / (sigma * (1.0 - h).sqrt())
                })
                .collect();
            Ok(FamilyResiduals {
                values,
                label: "Standardized Residual",
            })
        }
        ResponseFamily::Binomial => {
            // Bernoulli: u ~ U(F(y−), F(y)) with F(0) = 1 − p, F(1) = 1.
            let values = (0..n)
                .map(|i| {
                    let p = mu[i].clamp(0.0, 1.0);
                    let v: f64 = rng.random();
                    let u = if y[i] < 0.5 {
                        v * (1.0 - p)
                    } else {
                        (1.0 - p) + v * p
                    };
                    to_normal(u)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Randomized Quantile Residual",
            })
        }
        ResponseFamily::Poisson => {
            let values = (0..n)
                .map(|i| {
                    let k = discrete_count_response(y[i], "Poisson")?;
                    let dist = Poisson::new(mu[i].max(f64::MIN_POSITIVE))
                        .map_err(|e| format!("Poisson residual at μ={}: {e}", mu[i]))?;
                    let lower = if k == 0 { 0.0 } else { dist.cdf(k - 1) };
                    let u = lower + rng.random::<f64>() * dist.pmf(k);
                    to_normal(u)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Randomized Quantile Residual",
            })
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            if !(theta.is_finite() && theta > 0.0) {
                return Err(format!("negative-binomial θ={theta} is not positive"));
            }
            let values = (0..n)
                .map(|i| {
                    let k = discrete_count_response(y[i], "negative-binomial")?;
                    // Failures-before-r-th-success parameterization: r = θ and
                    // p = θ/(θ+μ) give mean μ and variance μ + μ²/θ.
                    let p = theta / (theta + mu[i].max(f64::MIN_POSITIVE));
                    let dist = NegativeBinomial::new(theta, p)
                        .map_err(|e| format!("negative-binomial residual at μ={}: {e}", mu[i]))?;
                    let lower = if k == 0 { 0.0 } else { dist.cdf(k - 1) };
                    let u = lower + rng.random::<f64>() * dist.pmf(k);
                    to_normal(u)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Randomized Quantile Residual",
            })
        }
        ResponseFamily::Gamma => {
            // Pearson dispersion under V(μ) = μ²: φ̂ = Σ((y−μ)/μ)²/(n − edf).
            let phi = (0..n)
                .map(|i| ((y[i] - mu[i]) / mu[i]).powi(2))
                .sum::<f64>()
                / residual_dof;
            if !(phi.is_finite() && phi > 0.0) {
                return Err("Gamma dispersion estimate is not positive".to_string());
            }
            let shape = 1.0 / phi;
            let values = (0..n)
                .map(|i| {
                    if !(y[i] > 0.0) {
                        return Err(format!("Gamma response must be positive, got {}", y[i]));
                    }
                    // shape/rate with mean μ: rate = shape/μ.
                    let dist = Gamma::new(shape, shape / mu[i])
                        .map_err(|e| format!("Gamma residual at μ={}: {e}", mu[i]))?;
                    to_normal(dist.cdf(y[i]))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Quantile Residual",
            })
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !(phi.is_finite() && phi > 0.0) {
                return Err(format!("Beta precision φ={phi} is not positive"));
            }
            let values = (0..n)
                .map(|i| {
                    if !(y[i] > 0.0 && y[i] < 1.0) {
                        return Err(format!("Beta response must lie in (0,1), got {}", y[i]));
                    }
                    let m = mu[i].clamp(1e-12, 1.0 - 1e-12);
                    let dist = Beta::new(m * phi, (1.0 - m) * phi)
                        .map_err(|e| format!("Beta residual at μ={m}: {e}"))?;
                    to_normal(dist.cdf(y[i]))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Quantile Residual",
            })
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            // No practical closed-form Tweedie CDF, so use deviance residuals
            // r = sign(y−μ)·√(d(y,μ)/φ̂) — asymptotically N(0,1) under the
            // fitted model — with the Pearson φ̂ under V(μ) = μ^p.
            if !(p > 1.0 && p < 2.0) {
                return Err(format!(
                    "Tweedie deviance residuals are implemented for power p ∈ (1,2), got {p}"
                ));
            }
            let phi = (0..n)
                .map(|i| (y[i] - mu[i]).powi(2) / mu[i].powf(p))
                .sum::<f64>()
                / residual_dof;
            if !(phi.is_finite() && phi > 0.0) {
                return Err("Tweedie dispersion estimate is not positive".to_string());
            }
            let values = (0..n)
                .map(|i| {
                    if y[i] < 0.0 {
                        return Err(format!(
                            "Tweedie response must be nonnegative, got {}",
                            y[i]
                        ));
                    }
                    let dev = tweedie_unit_deviance(y[i], mu[i], p);
                    Ok((y[i] - mu[i]).signum() * (dev.max(0.0) / phi).sqrt())
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FamilyResiduals {
                values,
                label: "Deviance Residual",
            })
        }
        ResponseFamily::RoystonParmar => Err(
            "no standard-normal residual definition is implemented for the Royston–Parmar family"
                .to_string(),
        ),
    }
}

/// Validate a discrete count response and convert it to `u64`.
fn discrete_count_response(y: f64, family: &str) -> Result<u64, String> {
    let k = y.round();
    if !(y.is_finite() && k >= 0.0 && (y - k).abs() <= 1e-8) {
        return Err(format!(
            "{family} response must be a nonnegative integer count, got {y}"
        ));
    }
    Ok(k as u64)
}

/// Tweedie unit deviance for 1 < p < 2 (the compound Poisson–Gamma range):
/// d(y,μ) = 2·[ y^{2−p}/((1−p)(2−p)) − y·μ^{1−p}/(1−p) + μ^{2−p}/(2−p) ],
/// with the y = 0 limit d = 2·μ^{2−p}/(2−p). Zero at y = μ.
fn tweedie_unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
    if y == 0.0 {
        return 2.0 * mu.powf(2.0 - p) / (2.0 - p);
    }
    2.0 * (y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - y * mu.powf(1.0 - p) / (1.0 - p)
        + mu.powf(2.0 - p) / (2.0 - p))
}

/// Equal-count calibration bins for a binary response: observations are
/// sorted by predicted probability and split into (up to) ten groups of
/// near-equal size — true deciles. Equal-width probability bins read as
/// "deciles" but put 90% of a skewed score distribution into one bin, so a
/// point can summarise 9 observations or 900.
fn binary_calibration_deciles(y: &[f64], p: &[f64]) -> Option<report::CalibrationData> {
    let n = y.len().min(p.len());
    if n == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let bins = 10usize.min(n);
    let mut mean_predicted = Vec::with_capacity(bins);
    let mut observed_rate = Vec::with_capacity(bins);
    for b in 0..bins {
        let lo = b * n / bins;
        let hi = (b + 1) * n / bins;
        if hi == lo {
            continue;
        }
        let m = (hi - lo) as f64;
        let sum_p: f64 = order[lo..hi].iter().map(|&i| p[i].clamp(0.0, 1.0)).sum();
        let sum_y: f64 = order[lo..hi].iter().map(|&i| y[i]).sum();
        mean_predicted.push(sum_p / m);
        observed_rate.push((sum_y / m).clamp(0.0, 1.0));
    }
    Some(report::CalibrationData {
        mean_predicted,
        observed_rate,
    })
}
