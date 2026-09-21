use super::*;
use gam::families::inference::saved_summary::saved_model_report_input;


fn saved_alo_report_data(
    alo: gam_predict::SavedModelAloDiagnostics,
) -> Result<report::AloData, String> {
    if alo.coordinate_names.is_empty() {
        return Err(format!(
            "saved {} ALO result has no affine coordinate names",
            alo.model_class.name()
        ));
    }
    let diagnostics = alo.diagnostics;
    let n = diagnostics.leverage.len();
    if diagnostics.eta_tilde.len() != n
        || diagnostics.alo_variance.len() != n
        || diagnostics.cook_distance.len() != n
    {
        return Err(format!(
            "saved {} ALO result has inconsistent row counts: leverage={n}, eta={}, variance={}, cook={}",
            alo.model_class.name(),
            diagnostics.eta_tilde.len(),
            diagnostics.alo_variance.len(),
            diagnostics.cook_distance.len(),
        ));
    }
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let coordinates = diagnostics.eta_tilde[row].to_vec();
        let variances = &diagnostics.alo_variance[row];
        let leverage = diagnostics.leverage[row];
        let cook_distance = diagnostics.cook_distance[row];
        if !leverage.is_finite()
            || !cook_distance.is_finite()
            || cook_distance < 0.0
            || coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "saved {} ALO row {row} has non-representable diagnostics",
                alo.model_class.name()
            ));
        }
        if coordinates.len() != alo.coordinate_names.len()
            || variances.len() != alo.coordinate_names.len()
        {
            return Err(format!(
                "saved {} ALO row {row} has eta/variance widths {}/{}; expected {}",
                alo.model_class.name(),
                coordinates.len(),
                variances.len(),
                alo.coordinate_names.len(),
            ));
        }
        let mut standard_errors = Vec::with_capacity(variances.len());
        for (coordinate, &variance) in variances.iter().enumerate() {
            if !variance.is_finite() || variance < 0.0 {
                return Err(format!(
                    "saved {} ALO row {row} coordinate {coordinate} has invalid variance {variance}",
                    alo.model_class.name()
                ));
            }
            standard_errors.push(variance.sqrt());
        }
        rows.push(report::AloRow {
            index: row,
            leverage,
            eta_tilde: coordinates,
            standard_errors,
            cook_distance,
        });
    }
    Ok(report::AloData {
        coordinate_names: alo.coordinate_names,
        rows,
    })
}

pub(crate) fn run_sample(args: SampleArgs) -> Result<(), String> {
    validate_positive_optional_usize("--samples", args.samples)?;
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
        "  convergence: rhat={:.4}  ess={:.1}  converged={}  warmup={} transitions per chain",
        nuts.rhat,
        nuts.ess,
        nuts.converged,
        nuts.warmup_transitions
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

    let seed = args.seed.unwrap_or(42);
    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".generated.csv"));
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(&out)
        .map_err(|error| {
            format!(
                "failed to create generated-data csv '{}': {error}",
                out.display()
            )
        })?;
    writer
        .write_record(["draw", "row", "value"])
        .map_err(|error| format!("failed to write generated-data header: {error}"))?;
    let chunk_draws = gam_runtime::resource::rows_for_target_bytes(
        gam::ResourcePolicy::default_library().row_chunk_target_bytes,
        spec.nobs(),
    )
    .min(args.n_draws)
    .max(1);
    gam::generative::sampleobservation_seeded_replicate_chunks(
        &spec,
        0,
        args.n_draws,
        chunk_draws,
        seed,
        |draw_start, chunk| {
            for local_draw in 0..chunk.nrows() {
                let draw = draw_start + local_draw;
                for row in 0..chunk.ncols() {
                    writer
                        .write_record([
                            draw.to_string(),
                            row.to_string(),
                            format!("{:.17}", chunk[[local_draw, row]]),
                        ])
                        .map_err(|error| {
                            gam::estimate::EstimationError::InvalidInput(format!(
                                "failed to write generated-data row draw={draw}, row={row}: {error}"
                            ))
                        })?;
                }
            }
            Ok(())
        },
    )
    .map_err(|error| format!("failed to sample synthetic observations: {error}"))?;
    writer.flush().map_err(|error| {
        format!(
            "failed to flush generated-data csv '{}': {error}",
            out.display()
        )
    })?;
    cli_out!(
        "wrote synthetic draws: {} (long rows={}, input_rows={}, draws={})",
        out.display(),
        spec.nobs().saturating_mul(args.n_draws),
        spec.nobs(),
        args.n_draws
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

pub(crate) fn run_report(args: ReportArgs) -> Result<(), String> {
    reject_multinomial_model(&args.model, "report")?;
    let model = SavedModel::load_from_path(&args.model)?;
    // The report card of the saved model has one owner, which gamfit's
    // `Model.report()` renders too. This command adds only what the data it is
    // given can show.
    let mut input = saved_model_report_input(&model, args.model.display().to_string())?;
    // Spline-scan (#1030/#1034/#1046) and residual-cascade (#1032) models keep
    // only their exact smoother's posterior, not a dense fit_result, so the card
    // shows the reconstructed scalar quantities and nothing is computed from data.
    if model.spline_scan.is_some() || model.residual_cascade.is_some() {
        if args.data.is_some() {
            input.notes.push(
                "Data provided, but held-out diagnostics for exact state-space smoother \
                 models are served through the predict() path; the CLI report shows the \
                 fitted scalar quantities only."
                    .to_string(),
            );
        }
        let out = report::write_report(&input, args.out.as_deref(), &args.model)?;
        cli_out!("wrote report: {}", out.display());
        return Ok(());
    }
    let family = model.likelihood();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    // The residual degrees of freedom behind the report residuals' dispersion.
    let edf_total = input.edf_total;
    let mut notes = Vec::new();
    let mut diagnostics = None;
    let mut smooth_plots = Vec::new();
    let mut continuous_order = Vec::new();
    let mut measure_jet_spectra = Vec::new();
    let mut alo_data = None;
    let mut r_squared = None;

    if let Some(data_path) = args.data.as_ref() {
        let ds = load_datasetwith_model_schema_for_diagnostics(data_path, &model)?;
        require_dataset_rows("report", data_path, ds.values.nrows())?;

        let col_map = ds.column_map();
        let training_headers = model.training_headers.as_ref();
        let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(&model);
        let parsed = parse_formula(&model.formula)?;

        let saved_alo_response_col = match resolve_saved_alo_response_col(&model, &parsed, &col_map)
        {
            Ok(column) => Some(column),
            Err(error) => {
                notes.push(format!("ALO diagnostics unavailable: {error}"));
                None
            }
        };
        if let Some(y_col) = saved_alo_response_col {
            let alo_response = ds.values.column(y_col).to_owned();
            let alo_weights =
                resolve_weight_column(&ds, &col_map, model.payload().weight_column.as_deref())
                    .map_err(|error| {
                        format!("failed to resolve saved report ALO weights: {error}")
                    })?;
            let (alo_offset, alo_noise_offset) = report_offset_for(&model, &ds, &col_map)?;
            let alo_result = build_saved_alo_predict_input(
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
                &alo_offset,
                &alo_noise_offset,
                saved_noise_offset_column.is_some(),
            )
            .and_then(|alo_input| {
                gam_predict::compute_saved_model_alo(
                    &model,
                    &alo_input,
                    gam_predict::SavedAloObservations {
                        response: &alo_response,
                        prior_weights: &alo_weights,
                    },
                )
                .map_err(|error| error.to_string())
            })
            .and_then(saved_alo_report_data);
            match alo_result {
                Ok(alo) => alo_data = Some(alo),
                Err(error) => notes.push(format!("ALO diagnostics unavailable: {error}")),
            }

            if model.predict_model_class() == PredictModelClass::BernoulliMarginalSlope {
                let y = ds.values.column(y_col).to_owned();
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
                    let leverage = alo_data
                        .as_ref()
                        .map(|alo| alo.rows.iter().map(|row| row.leverage).collect::<Vec<_>>());
                    let residuals = report_residual_diagnostics(
                        &ResponseFamily::Binomial,
                        &y_vec,
                        &p_vec,
                        leverage.as_deref(),
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

                // R-squared for Gaussian
                if family.is_gaussian_identity() {
                    let y_mean = y.mean().unwrap_or(0.0);
                    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
                    let ss_res: f64 = y
                        .iter()
                        .zip(pred.mean.iter())
                        .map(|(&yi, &pi)| (yi - pi).powi(2))
                        .sum();
                    // A mean over `n` rows is resolved to `γ_{n+1}·max|y|`, so a total sum of
                    // squares inside `γ_{n+1}²·Σy²` is the rounding residue of a constant
                    // response, which has no variance to explain.
                    let energy: f64 = y.iter().map(|&yi| yi * yi).sum();
                    let band = gam::linalg::roundoff::accumulation_growth(y.len() + 1).powi(2) * energy;
                    if ss_tot > band {
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
                            // MeasureJet is the family that stores its range
                            // STANDARDIZED, so this row reports a standardized
                            // length beside a standardized `eps_band` — the
                            // section is internally consistent but the rendered
                            // label does not say so (see #2636 discussion).
                            length_scale: length_scale.standardized_value(),
                            spec_order_s: *order_s,
                            per_scale,
                            implied_order,
                        });
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

    // The card carries the realized measure-jet band from the frozen spec. A
    // design rebuilt from the data also has the penalty layout the per-scale
    // λ̂_ℓ come from, and those rows replace the spec-only ones.
    if !measure_jet_spectra.is_empty() {
        input.measure_jet_spectra = measure_jet_spectra;
    }
    input.r_squared = r_squared;
    input.continuous_order = continuous_order;
    input.diagnostics = diagnostics;
    input.smooth_plots = smooth_plots;
    input.alo = alo_data;
    input.notes.extend(notes);
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
    // Residual degrees of freedom for the Pearson dispersion estimates. With none
    // left there is no residual scale to estimate, and the diagnostics are omitted
    // rather than divided by a dof of one that the fit does not have.
    let residual_dof = n as f64 - edf_total;
    if !(residual_dof > 0.0) {
        return Err(format!(
            "no residual degrees of freedom to estimate a scale (n = {n}, edf = {edf_total})"
        ));
    }
    let mut rng = StdRng::seed_from_u64(REPORT_RESIDUAL_SEED);
    // Predictive CDF value → normal scale. Only the exact endpoints have no
    // finite quantile, so u is held inside the representable open interval:
    // the smallest positive double and the largest double below one.
    let to_normal = |u: f64| {
        standard_normal_quantile(u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON / 2.0))
    };

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
                    // high-leverage rows. A row with leverage one is fitted
                    // exactly, so its residual has no variance to standardize.
                    let h = leverage
                        .and_then(|l| l.get(i).copied())
                        .filter(|h| h.is_finite())
                        .unwrap_or(0.0)
                        .max(0.0);
                    if !(h < 1.0) {
                        return Err(format!(
                            "observation {i} has leverage {h}: it is fitted exactly, so its \
                             residual has no variance to standardize"
                        ));
                    }
                    Ok((y[i] - mu[i]) / (sigma * (1.0 - h).sqrt()))
                })
                .collect::<Result<Vec<_>, _>>()?;
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
                    // A zero mean is the point mass at 0: F(−1) = 0, P(0) = 1, and
                    // every positive count has F(k−1) = 1 and P(k) = 0.
                    let (lower, mass) = if mu[i] == 0.0 {
                        if k == 0 { (0.0, 1.0) } else { (1.0, 0.0) }
                    } else {
                        let dist = Poisson::new(mu[i])
                            .map_err(|e| format!("Poisson residual at μ={}: {e}", mu[i]))?;
                        (if k == 0 { 0.0 } else { dist.cdf(k - 1) }, dist.pmf(k))
                    };
                    let u = lower + rng.random::<f64>() * mass;
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
                    let p = theta / (theta + mu[i]);
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
                    let m = mu[i];
                    if !(m >= 0.0 && m <= 1.0) {
                        return Err(format!("Beta mean must lie in [0,1], got {m}"));
                    }
                    // As a shape parameter reaches zero the Beta law becomes the
                    // point mass at that end, so a response strictly inside
                    // (0,1) has F(y) = 1 when α = μφ is zero and F(y) = 0 when
                    // β = (1−μ)φ is zero. That includes shapes that underflow.
                    let (alpha, beta) = (m * phi, (1.0 - m) * phi);
                    let u = if alpha == 0.0 {
                        1.0
                    } else if beta == 0.0 {
                        0.0
                    } else {
                        Beta::new(alpha, beta)
                            .map_err(|e| format!("Beta residual at μ={m}: {e}"))?
                            .cdf(y[i])
                    };
                    to_normal(u)
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
    if !(y.is_finite() && k >= 0.0 && y == k) {
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
