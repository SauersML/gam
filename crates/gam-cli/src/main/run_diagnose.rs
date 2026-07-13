use super::*;

fn format_alo_coordinates(values: &Array1<f64>) -> String {
    values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_multicoordinate_alo(alo: &gam_predict::SavedModelAloDiagnostics) {
    let diagnostics = &alo.diagnostics;
    let mut rows = (0..diagnostics.leverage.len()).collect::<Vec<_>>();
    rows.sort_by(|&left, &right| {
        diagnostics.leverage[right]
            .partial_cmp(&diagnostics.leverage[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["row", "leverage", "alo_coordinates", "alo_se", "cook"]);
    for row in rows.into_iter().take(12) {
        let standard_errors = diagnostics.alo_variance[row].mapv(|variance| variance.sqrt());
        table.add_row(Row::from(vec![
            Cell::new(row),
            Cell::new(format!("{:.4}", diagnostics.leverage[row])),
            Cell::new(format_alo_coordinates(&diagnostics.eta_tilde[row])),
            Cell::new(format_alo_coordinates(&standard_errors)),
            Cell::new(format!("{:.6}", diagnostics.cook_distance[row])),
        ]));
    }
    cli_out!(
        "ALO coordinates ({}): {}",
        alo.model_class.name(),
        alo.coordinate_names.join(", ")
    );
    cli_out!("ALO diagnostics (top leverage rows):");
    cli_out!("{table}");
}

pub(crate) fn run_diagnose(args: DiagnoseArgs) -> Result<(), String> {
    // `diagnose` currently has exactly one implemented diagnostic: ALO. Rather
    // than erroring with "only --alo is currently implemented for diagnose"
    // when the user runs the bare subcommand, just run ALO. This is the
    // useful default and matches user expectation that `gam diagnose` does
    // SOMETHING (a smoke-test for the most common workflow). If/when more
    // diagnostics land, this path can route based on explicit flags.
    // (`args.alo` is intentionally ignored until other diagnostics land.)

    reject_multinomial_model(&args.model, "diagnose")?;
    let model = SavedModel::load_from_path(&args.model)?;
    let parsed = parse_formula(&model.formula)?;
    if matches!(
        model.predict_model_class(),
        PredictModelClass::GaussianLocationScale
            | PredictModelClass::BinomialLocationScale
            | PredictModelClass::DispersionLocationScale
            | PredictModelClass::BernoulliMarginalSlope
            | PredictModelClass::TransformationNormal
    ) {
        let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
        require_dataset_rows("diagnose", &args.data, ds.values.nrows())?;
        let col_map = ds.column_map();
        let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;
        let response = ds.values.column(y_col).to_owned();
        let prior_weights =
            resolve_weight_column(&ds, &col_map, model.payload().weight_column.as_deref())
                .map_err(|error| format!("failed to resolve saved diagnose weights: {error}"))?;
        let (offset, noise_offset) = report_offset_for(&model, &ds, &col_map)?;
        let observations = gam_predict::SavedAloObservations {
            response: &response,
            prior_weights: &prior_weights,
        };
        let alo = match model.predict_model_class() {
            PredictModelClass::TransformationNormal => {
                let spec = resolve_termspec_for_prediction(
                    &model.resolved_termspec,
                    model.training_headers.as_ref(),
                    &col_map,
                    "resolved_termspec",
                )?;
                let design =
                    build_term_collection_design(ds.values.view(), &spec).map_err(|error| {
                        format!("failed to build saved transformation-normal ALO design: {error}")
                    })?;
                let effective_offset = design
                    .compose_offset(offset.view(), "saved transformation-normal ALO design")
                    .map_err(|error| error.to_string())?;
                gam_predict::compute_saved_transformation_normal_alo(
                    &model,
                    &design.design,
                    &effective_offset,
                    observations,
                )
            }
            class => {
                let input = build_predict_input_for_model(
                    &model,
                    ds.values.view(),
                    &col_map,
                    model.training_headers.as_ref(),
                    &offset,
                    &noise_offset,
                    model.payload().noise_offset_column.is_some(),
                )?;
                match class {
                    PredictModelClass::BernoulliMarginalSlope => {
                        gam_predict::compute_saved_bernoulli_marginal_slope_alo(
                            &model,
                            &input,
                            observations,
                        )
                    }
                    _ => {
                        gam_predict::compute_saved_location_scale_alo(&model, &input, observations)
                    }
                }
            }
        }
        .map_err(|error| format!("saved-model ALO failed: {error}"))?;
        print_multicoordinate_alo(&alo);
        return Ok(());
    }

    // Survival responses need risk-set/event replay, while marginal-slope and
    // transformation models have their own row likelihood coordinates. Keep
    // the class factual here until its exact dispatcher arm is installed.
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(format!(
            "saved {} ALO row-likelihood replay is unavailable in this binary",
            model.predict_model_class().name()
        ));
    }
    // A spline-scan model (a Standard fit routed through the exact O(n)
    // smoother) keeps no dense design/Gram, and ALO leverage is defined off
    // exactly that dense leave-one-out hat matrix — so it cannot be computed
    // from the per-knot posterior. Surface a precise error rather than the
    // cryptic missing-resolved_termspec one (#1046).
    if model.spline_scan.is_some() {
        return Err(
            "diagnose --alo cannot replay this spline-scan model because its \
             saved state has no coefficient-space penalized Hessian"
                .to_string(),
        );
    }
    // A residual-cascade model (#1032) is the multi-resolution analogue: the
    // scattered low-d smooth is routed through the multilevel Wendland
    // posterior past the dense-kernel cliff, so it likewise retains no dense
    // design/Gram (only the nested ε-net geometry + factored precision). ALO
    // leverage is undefined off that — surface the precise error rather than
    // the downstream missing-resolved_termspec one.
    if model.residual_cascade.is_some() {
        return Err(
            "diagnose --alo cannot replay this residual-cascade model because \
             its saved state has no coefficient-space penalized Hessian"
                .to_string(),
        );
    }
    let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("diagnose", &args.data, ds.values.nrows())?;
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    let family = model.likelihood();
    let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;

    let y = ds.values.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;

    let link = family.link_function();
    // Prior (case) weights the model was fit with, reloaded by the saved weight
    // column name — the same column the fit persisted. Hard-coding `weights = 1`
    // silently dropped the case weights from every diagnostic on a
    // `--weights-column` fit: the geometry ALO path's working weights
    // `w_i = prior_i · Fisher_i` need the true prior weights, and the replay
    // below reproduces the fit's stored working
    // weights only when seeded with them. `resolve_weight_column` returns all-ones
    // when the model carries no weight column (the unweighted default).
    let weights = resolve_weight_column(&ds, &col_map, model.weight_column.as_deref())
        .map_err(|e| format!("failed to resolve saved weight column for diagnose: {e}"))?;
    // Re-apply the offset the model was fit with, resolved by the saved offset
    // column name exactly as the predict path does. Diagnose is Standard-only
    // (non-standard classes are rejected above), so the noise-offset slot is
    // always zero here. Hard-coding `offset = 0` made every ALO diagnostic
    // (eta_tilde / leverage / alo_se) wrong by the entire offset for any
    // `--offset-column` fit (#881): the saved working response is offset-
    // inclusive, so a zero offset broke the `eta − offset` centering in
    // `alo_eta_update`. `report_offset_for` reads the saved offset column and
    // returns a zero noise-offset for standard models.
    let (offset, _noise_offset) = report_offset_for(&model, &ds, &col_map)?;
    let effective_offset = design
        .compose_offset(offset.view(), "diagnose saved-model design")
        .map_err(|error| error.to_string())?;

    let unified = model
        .unified()
        .ok_or_else(|| "saved standard ALO requires a unified fit result".to_string())?;
    let geom = unified.geometry.as_ref().ok_or_else(|| {
        "saved standard ALO requires the exact unscaled penalized Hessian".to_string()
    })?;
    let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
    let eta = &design.design.dot(&fit_saved.beta) + &effective_offset;
    let alo_design_dense = design.design.to_dense();
    let phi = geometry_alo_phi(unified, link);
    let recomputed =
        geometry_alo_working_state(&family, unified, &eta, y.view(), weights.view())
            .map_err(|error| format!("failed to replay saved ALO working state: {error}"))?;
    let input = gam::alo::AloInput::from_geometry_with_working_state(
        geom,
        &alo_design_dense,
        &eta,
        &effective_offset,
        phi,
        &recomputed.working_weights,
        &recomputed.working_response,
    );
    let alo = gam::alo::compute_alo_from_input(&input)
        .map_err(|error| format!("saved standard ALO solve failed: {error}"))?;

    let mut rows: Vec<(usize, f64, f64, f64)> = (0..alo.leverage.len())
        .map(|i| (i, alo.leverage[i], alo.eta_tilde[i], alo.se_sandwich[i]))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["row", "leverage", "eta_tilde", "alo_se"]);
    for (row, lev, eta, se) in rows.into_iter().take(12) {
        table.add_row(Row::from(vec![
            Cell::new(row),
            Cell::new(format!("{lev:.4}")),
            Cell::new(format!("{eta:.6}")),
            Cell::new(format!("{se:.6}")),
        ]));
    }

    cli_out!("ALO diagnostics (top leverage rows):");
    cli_out!("{table}");

    // Model-comparison corroboration channels (#946): exact smoothing-corrected
    // conditional AIC and zero-refit PSIS-LOO, computed from the fit-retained
    // exact pieces (smoothing-parameter covariance Σ_ρ, ALO leave-one-out
    // predictions) and reported alongside the diagnostics. The ALO solves
    // already reused the fit's factored Hessian, so the LOO channel is free here.
    if let Some(unified) = model.unified() {
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        let eta_hat = &design.design.dot(&fit_saved.beta) + &effective_offset;
        let comparison = gam::model_comparison::model_comparison_from_unified(
            unified,
            y.view(),
            eta_hat.view(),
            weights.view(),
            Some(&alo),
        )
        .map_err(|err| format!("cannot resolve model-comparison dispersion: {err}"))?;
        let mut summary = Table::new();
        summary
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["criterion", "value"]);
        summary.add_row(Row::from(vec![
            Cell::new("edf (conditional)"),
            Cell::new(format!("{:.4}", comparison.edf.conditional)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("edf (corrected, WPS)"),
            Cell::new(
                comparison
                    .edf
                    .corrected
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            ),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("rho-uncertainty df"),
            Cell::new(
                comparison
                    .edf
                    .rho_uncertainty_df()
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            ),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (conditional)"),
            Cell::new(format!("{:.4}", comparison.aic_conditional)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (corrected)"),
            Cell::new(
                comparison
                    .aic_corrected
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            ),
        ]));
        if let Some(loo) = comparison.loo.as_ref() {
            let se = loo
                .se
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "n/a".to_string());
            summary.add_row(Row::from(vec![
                Cell::new("PSIS-LOO elpd"),
                Cell::new(format!("{:.4} (se {se})", loo.elpd)),
            ]));
            let k_hat = loo
                .k_hat_max
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_string());
            summary.add_row(Row::from(vec![
                Cell::new("PSIS k_hat (max)"),
                Cell::new(format!("{k_hat} ({} unreliable)", loo.n_k_bad)),
            ]));
        }
        cli_out!("Model comparison (corrected AIC + PSIS-LOO):");
        cli_out!("{summary}");
    }

    Ok(())
}

/// Row-sized IRLS working vectors reconstructed for the geometry ALO path.
struct GeometryAloWorkingState {
    /// Score-side Fisher working weights `w_i = prior_i · h'(η̂_i)²/(φ V(μ̂_i))`.
    working_weights: Array1<f64>,
    /// IRLS working response `z_i = η̂_i + (y_i − μ̂_i)/h'(η̂_i)` (offset-inclusive).
    working_response: Array1<f64>,
}

/// Reconstruct the fit's converged IRLS working weights/response from `η̂`.
///
/// Saved models are size-compacted (`compact_fit_result_for_batch` empties the
/// n-sized `FitGeometry` working vectors so persisted models stay n-independent),
/// so the geometry ALO path cannot read them off `geom`. They are *derived*
/// quantities, though: at convergence they are deterministic functions of the
/// linear predictor `η̂ = Xβ̂`, the response `y`, the prior weights, and the
/// family. Replaying the exact PIRLS working-state update the fit used
/// (`update_glmvectors_by_family`) reproduces the fit's stored working vectors
/// bit-for-bit — including scale-carrying families, since `likelihood_scale` is
/// threaded from the saved fit — while keeping saved models free of any n-sized
/// carrier. `eta` must be the full, offset-inclusive linear predictor (PIRLS
/// works on `Xβ + offset`), matching the offset-inclusive working response.
fn geometry_alo_working_state(
    family: &LikelihoodSpec,
    unified: &UnifiedFitResult,
    eta: &Array1<f64>,
    y: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
) -> Result<GeometryAloWorkingState, String> {
    let likelihood = gam::types::GlmLikelihoodSpec {
        spec: family.clone(),
        scale: unified.likelihood_scale.clone(),
    };
    let n = eta.len();
    let mut mu = Array1::<f64>::zeros(n);
    let mut working_weights = Array1::<f64>::zeros(n);
    let mut working_response = Array1::<f64>::zeros(n);
    gam::pirls::update_glmvectors_by_family(
        y,
        eta,
        &likelihood,
        prior_weights,
        &mut mu,
        &mut working_weights,
        &mut working_response,
    )
    .map_err(|e| e.to_string())?;
    Ok(GeometryAloWorkingState {
        working_weights,
        working_response,
    })
}
