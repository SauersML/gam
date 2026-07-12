use super::*;

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
    // Survival / location-scale / marginal-slope models don't have a single
    // bare-column response, so the lookup below would fail with the cryptic
    // "response column 'Surv(...)' not found in data" message. Reject up
    // front with a clear message naming the model class.
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(format!(
            "diagnose --alo is not yet supported for {model_class:?} models; \
             only standard GAM fits are covered. \
             (You can still inspect the model with `gam report <model>`.)",
            model_class = model.predict_model_class()
        ));
    }
    // A spline-scan model (a Standard fit routed through the exact O(n)
    // smoother) keeps no dense design/Gram, and ALO leverage is defined off
    // exactly that dense leave-one-out hat matrix — so it cannot be computed
    // from the per-knot posterior. Surface a precise error rather than the
    // cryptic missing-resolved_termspec one (#1046).
    if model.spline_scan.is_some() {
        return Err(
            "diagnose --alo needs the dense leave-one-out leverage, which a \
             spline-scan model does not retain (it stores only the per-knot \
             posterior of the exact O(n) smoother). Use `gam report <model>` \
             for its fitted quantities, or refit with double_penalty=true to \
             obtain the dense fit ALO requires."
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
            "diagnose --alo needs the dense leave-one-out leverage, which a \
             residual-cascade model does not retain (it stores only the \
             multilevel Wendland posterior of the exact O(n log n) cascade). \
             Use `gam report <model>` for its fitted quantities."
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
    // `w_i = prior_i · Fisher_i` and the refit fallback both need the true prior
    // weights, and the recomputation below reproduces the fit's stored working
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

    // Try geometry-based ALO from the unified result first (avoids refit).
    let alo = if let Some((unified, geom)) = model
        .unified()
        .and_then(|u| u.geometry.as_ref().map(|g| (u, g)))
        // Treat a present-but-emptied geometry carrier as "geometry
        // unavailable" and fall through to the refit branch. Batch-compacted
        // models (see `compact_fit_result_for_batch`) used to zero these
        // row-sized working vectors; `AloInput::from_geometry` then failed its
        // length-N validation instead of ever reaching the refit fallback
        // (#2030). This guard keeps diagnose working even for such saved
        // models produced before the compaction fix.
        .filter(|(_, geom)| !geom.working_weights.is_empty() && !geom.working_response.is_empty())
    {
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        // ALO's geometry constructor expects the *full* linear predictor (offset
        // included); it re-centres internally via the separate `offset` arg to
        // match the offset-inclusive saved working response. The refit branch
        // below already adds `offset` here — the geometry path must too (#881).
        let eta = &design.design.dot(&fit_saved.beta) + &offset;
        // ALO needs a dense X — materialize from row chunks when the design
        // is an operator-backed (lazy) one. `as_dense_cow` panicked on lazy
        // designs ("called on operator-backed design; use row chunks or
        // matrix-vector products"), which broke `diagnose --alo` for every
        // matern/duchon/sphere fit since those default to lazy storage.
        let alo_design_dense = design.design.to_dense();
        // φ must match the PIRLS-backed refit fallback: Gaussian (Identity) uses
        // the model's estimated dispersion σ̂², not a hard-coded 1.0 (#881-class
        // SE-scale bug). `geometry_alo_phi` reads the saved σ̂.
        let phi = geometry_alo_phi(unified, link);
        // Recompute the row-sized IRLS working vectors rather than reading them
        // off `geom`. Saved models are size-compacted for n-independence
        // (`compact_fit_result_for_batch` empties `geom.working_weights` /
        // `geom.working_response`), so reading them off `geom` fed empty vectors
        // to the ALO solve and every `gam diagnose` failed with
        // "ALO diagnostics require hessian_weights length N; got 0". These
        // vectors are *derived* — at convergence they are deterministic
        // functions of η̂ = Xβ̂, y, and the family — so replaying the same PIRLS
        // working-state update the fit used reproduces the fit's exact working
        // weights/response, keeping saved models n-independent while serving the
        // exact geometry ALO path (saved Hessian, no refit). `likelihood_scale`
        // is threaded from the fit so any scale-carrying family (fixed-φ
        // Gaussian, Tweedie, Gamma, Beta) reproduces bit-for-bit.
        let recomputed =
            geometry_alo_working_state(&family, unified, &eta, y.view(), weights.view())
                .map_err(|e| format!("failed to recompute working state for geometry ALO: {e}"))?;
        let input = gam::alo::AloInput::from_geometry_with_working_state(
            geom,
            &alo_design_dense,
            &eta,
            &offset,
            phi,
            &recomputed.working_weights,
            &recomputed.working_response,
        );
        gam::alo::compute_alo_from_input(&input)
            .map_err(|e| format!("compute_alo_from_input (geometry path) failed: {e}"))?
    } else {
        let fit_options = FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };
        let alo_result = match alo_refit_route_for_termspec(&spec) {
            AloRefitRoute::UnifiedTermCollection => {
                let fitted = fit_term_collection_forspec(
                    ds.values.view(),
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &spec,
                    family,
                    &fit_options,
                )
                .map_err(|e| {
                    format!("fit_term_collection_forspec failed during diagnose refit: {e}")
                })?;
                let eta = &fitted.design.design.dot(&fitted.fit.beta) + &offset;
                let dense_alo_design = fitted.design.design.to_dense();
                // φ for Gaussian (Identity) is the estimated dispersion σ̂², not
                // 1.0 — same SE-scale bug as the geometry path. Mirrors the
                // StandardGam sibling route, which computes φ inside
                // compute_alo_diagnostics_from_fit.
                let phi = geometry_alo_phi(&fitted.fit, link);
                gam::alo::compute_alo_diagnostics_from_unified(
                    &fitted.fit,
                    &dense_alo_design,
                    &eta,
                    &offset,
                    phi,
                )
                .map_err(|e| {
                    format!(
                        "compute_alo_diagnostics_from_unified failed during diagnose refit: {e}"
                    )
                })
            }
            AloRefitRoute::StandardGam => {
                let fit = fit_gam(
                    design.design.clone(),
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &design.penalties,
                    family,
                    &fit_options,
                )
                .map_err(|e| format!("fit_gam failed during diagnose refit: {e}"))?;
                compute_alo_diagnostics_from_fit(&fit, y.view())
                    .map_err(|e| format!("compute_alo_diagnostics_from_fit failed: {e}"))
            }
        };

        alo_result?
    };

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
        let eta_hat = &design.design.dot(&fit_saved.beta) + &offset;
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
            Cell::new(format!("{:.4}", comparison.edf.corrected)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("rho-uncertainty df"),
            Cell::new(format!("{:.4}", comparison.edf.rho_uncertainty_df())),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (conditional)"),
            Cell::new(format!("{:.4}", comparison.aic_conditional)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (corrected)"),
            Cell::new(format!("{:.4}", comparison.aic_corrected)),
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
