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
    let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;
    let y = ds.values.column(y_col).to_owned();
    let weights = resolve_weight_column(&ds, &col_map, model.weight_column.as_deref())
        .map_err(|error| format!("failed to resolve saved diagnose weights: {error}"))?;
    let (offset, noise_offset) = report_offset_for(&model, &ds, &col_map)?;
    let input = build_saved_alo_predict_input(
        &model,
        ds.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &noise_offset,
        model.payload().noise_offset_column.is_some(),
    )?;
    let alo = gam_predict::compute_saved_model_alo(
        &model,
        &input,
        gam_predict::SavedAloObservations {
            response: &y,
            prior_weights: &weights,
        },
    )
    .map_err(|error| format!("saved-model ALO failed: {error}"))?;
    print_multicoordinate_alo(&alo);

    // Model-comparison corroboration channels (#946): exact smoothing-corrected
    // conditional AIC and zero-refit PSIS-LOO, computed from the fit-retained
    // exact pieces (smoothing-parameter covariance Σ_ρ, ALO leave-one-out
    // predictions) and reported alongside the diagnostics. The ALO solves
    // already reused the fit's factored Hessian, so the LOO channel is free here.
    if model.predict_model_class() == PredictModelClass::Standard
        && let Some(unified) = model.unified()
    {
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        let beta = fit_saved.blocks.first().ok_or_else(|| {
            "saved standard model comparison requires its affine coefficient block".to_string()
        })?;
        let eta_hat = input.design.dot(&beta.beta) + &input.offset;
        let eta_loo = Array1::from_iter(
            alo.diagnostics
                .eta_tilde
                .iter()
                .map(|coordinates| coordinates[0]),
        );
        let comparison = gam::model_comparison::model_comparison_from_unified(
            unified,
            y.view(),
            eta_hat.view(),
            weights.view(),
            Some(eta_loo.view()),
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
