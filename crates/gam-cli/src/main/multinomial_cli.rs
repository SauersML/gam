use super::*;

use gam::families::multinomial::{
    MULTINOMIAL_MODEL_CLASS, MultinomialFitRequest, MultinomialModelEnvelope,
    MultinomialSavedModel, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_with_se,
};

/// Peek a model file's JSON discriminator to detect a persisted multinomial
/// envelope before committing to a full `SavedModel` deserialize. Returns
/// `false` for a standard `SavedModel` file (no matching discriminator) and for
/// an unreadable / non-JSON file — in the latter case the caller's own
/// `SavedModel::load_from_path` surfaces the real error.
pub(crate) fn is_multinomial_model_file(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value.get("model_class").and_then(|v| v.as_str()) == Some(MULTINOMIAL_MODEL_CLASS)
}

/// Reject a multinomial model at the entry of a post-fit command that has no
/// multinomial path (`diagnose` / `sample` / `generate` / `report`). The Python
/// surface exposes only `fit`/`predict`/`summary` for the multinomial family, so
/// this is a clean parity boundary rather than a deferral — it turns the
/// otherwise-cryptic `SavedModel` JSON parse failure into a directed message.
pub(crate) fn reject_multinomial_model(path: &Path, command: &str) -> Result<(), String> {
    if is_multinomial_model_file(path) {
        return Err(format!(
            "`gam {command}` does not support multinomial models; multinomial supports `fit` and \
             `predict` (per-class softmax probabilities)"
        ));
    }
    Ok(())
}

fn load_multinomial_model(path: &Path) -> Result<MultinomialSavedModel, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read multinomial model '{}': {e}", path.display()))?;
    let envelope = MultinomialModelEnvelope::from_json_bytes(&bytes).map_err(|e| {
        format!(
            "failed to parse multinomial model '{}': {e}",
            path.display()
        )
    })?;
    Ok(envelope.saved)
}

fn write_multinomial_model(path: &Path, saved: MultinomialSavedModel) -> Result<(), String> {
    let bytes = MultinomialModelEnvelope::new(saved)
        .map_err(|e| e.to_string())?
        .to_json_bytes()
        .map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| {
        format!(
            "failed to write multinomial model '{}': {e}",
            path.display()
        )
    })?;
    cli_out!("saved model: {}", path.display());
    Ok(())
}

fn print_multinomial_fit_summary(saved: &MultinomialSavedModel) {
    let reference = saved.class_levels.last().map(String::as_str).unwrap_or("?");
    cli_out!(
        "multinomial fit | classes={} | reference={} | p_per_class={} | iterations={} | \
         deviance={:.6e}",
        saved.class_levels.len(),
        reference,
        saved.p_per_class,
        saved.iterations,
        saved.deviance,
    );
    if let Some(edf) = saved.edf_per_class.as_ref() {
        let per_class = saved
            .class_levels
            .iter()
            .zip(edf.iter())
            .map(|(level, e)| format!("{level}={e:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        cli_out!("multinomial edf (per active class) | {}", per_class);
    }
}

/// Fit a penalized multinomial-logit GAM from a Wilkinson formula through the
/// same `fit_penalized_multinomial_formula` driver the Python surface uses, then
/// persist the resulting model. Dispatched from `run_fit` before the standard
/// service because the multinomial artifact is a softmax multi-output model, not
/// a scalar `SavedModel`.
pub(crate) fn run_fit_multinomial(
    args: &FitArgs,
    parsed: &ParsedFormula,
    formula_text: &str,
    fit_config: &FitConfig,
) -> Result<(), String> {
    // Surface-level flag rejections. The softmax link is fixed and the fit runs
    // a single joint softmax likelihood, so the location-scale / marginal-slope
    // / link-deviation controls have no meaning here; reject rather than
    // silently ignore.
    if args.predict_noise.is_some() {
        return Err("--predict-noise is not supported for --family multinomial".to_string());
    }
    if args.logslope_formula.is_some() || args.z_column.is_some() {
        return Err(
            "--logslope-formula/--z-column is not supported for --family multinomial".to_string(),
        );
    }
    if fit_config.transformation_normal {
        return Err("--transformation-normal conflicts with --family multinomial".to_string());
    }
    if parsed.linkspec.is_some() {
        return Err(
            "link(...) is not supported for --family multinomial; the softmax link is fixed"
                .to_string(),
        );
    }
    if parsed.linkwiggle.is_some() {
        return Err("linkwiggle(...) is not supported for --family multinomial".to_string());
    }
    if args.firth {
        return Err(
            "--firth is not accepted for --family multinomial: the Firth/Jeffreys separation \
             stabilizer is armed automatically when the fit detects complete separation"
                .to_string(),
        );
    }
    if args.frailty_kind.is_some() || args.frailty_sd.is_some() || args.hazard_loading.is_some() {
        return Err("frailty options are not supported for --family multinomial".to_string());
    }
    // Case weights (`--weights-column` → `fit_config.weight_column`) are
    // honored by the shared driver; offsets and the other config fields the
    // softmax family cannot consume are rejected with a typed error inside
    // `fit_penalized_multinomial_formula`, shared with the Python surface.
    if args.expectile_tau.is_some() {
        return Err("--expectile-tau requires --family expectile".to_string());
    }
    if args.adaptive_regularization {
        return Err(
            "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
        );
    }
    let Some(out) = args.out.as_ref() else {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
    };

    let mut requested_columns = required_columns_for_formula(parsed)?;
    // The weight column is consumed by the fit, not the formula; it must ride
    // along in the projected dataset for the driver to resolve it by name.
    requested_columns.extend(fit_config.weight_column.iter().cloned());
    // Force the categorical response to a factor encoding. An untyped CSV cannot
    // carry the typed-frame categorical sentinel the Python path uses, so this
    // is exactly the `response_is_categorical` role the dataset loader already
    // plumbs (`load_fit_dataset_with_roles`).
    let ds = load_fit_dataset_with_roles(&args.data, &requested_columns, parsed, true)?;
    require_dataset_rows("fit", &args.data, ds.values.nrows())?;

    let phase_start = std::time::Instant::now();
    log::info!("[PHASE] multinomial fit start n={}", ds.values.nrows());
    let saved = fit_penalized_multinomial_formula(&MultinomialFitRequest::new(
        &ds,
        formula_text,
        fit_config,
    ))
    .map_err(|e| format!("multinomial fit failed: {e}"))?;
    log::info!(
        "[PHASE] multinomial fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );

    print_multinomial_fit_summary(&saved);
    write_multinomial_model(out, saved)
}

/// Predict per-class softmax probabilities for a persisted multinomial model.
/// Dispatched from `run_predict` on the multinomial file discriminator. Emits a
/// CSV with one `prob_<class>` column per training class (columns aligned to the
/// saved `class_levels` order); with `--uncertainty`, appends per-class
/// delta-method standard-error columns `prob_se_<class>` when the saved model
/// carries the joint coefficient covariance.
pub(crate) fn run_predict_multinomial(args: &PredictArgs) -> Result<(), String> {
    let saved = load_multinomial_model(&args.model)?;
    let parsed = parse_formula(&saved.formula)?;

    // Prediction is for label-free new data: request the formula's feature
    // columns but not the response (which the predictor never references), and
    // force the same grouping-factor roles the fit used so by-factor encodings
    // line up with the frozen training basis.
    let mut requested_columns = required_columns_for_formula(&parsed)?;
    requested_columns.retain(|c| c != &parsed.response);
    let ds = load_fit_dataset_with_roles(&args.new_data, &requested_columns, &parsed, false)?;
    require_dataset_rows("predict", &args.new_data, ds.values.nrows())?;

    let id_values = args
        .id_column
        .as_ref()
        .map(|id_column| {
            load_prediction_id_values(&args.new_data, id_column, ds.values.nrows())
                .map(|values| (id_column.clone(), values))
        })
        .transpose()?;

    let (probs, prob_se) = if args.uncertainty {
        // #2296: multinomial fits persist only the conditional joint-Laplace
        // coefficient covariance. A smoothing-corrected request (the global
        // default) must refuse rather than silently deliver the narrower
        // conditional band under a corrected label.
        if args.covariance_mode == CovarianceModeArg::Corrected {
            return Err(
                "multinomial uncertainty carries only the conditional-on-\u{3bb}\u{302} \
                 joint-Laplace covariance; a smoothing-corrected (Vp) band is not \
                 persisted for multinomial fits (#2296). Pass --covariance-mode \
                 conditional to accept conditional standard errors."
                    .to_string(),
            );
        }
        let (probs, prob_se) = predict_multinomial_formula_with_se(&saved, &ds)
            .map_err(|e| format!("multinomial predict failed: {e}"))?;
        (probs, Some(prob_se))
    } else {
        let probs = predict_multinomial_formula(&saved, &ds)
            .map_err(|e| format!("multinomial predict failed: {e}"))?;
        (probs, None)
    };

    write_multinomial_prediction_csv(&args.out, &saved.class_levels, &probs, prob_se.as_ref())?;
    if let Some((id_column, values)) = id_values.as_ref() {
        prepend_id_column_to_prediction_csv(&args.out, id_column, values)?;
    }
    cli_out!(
        "wrote predictions: {} (rows={}, classes={}){}",
        args.out.display(),
        probs.nrows(),
        saved.class_levels.len(),
        covariance_provenance_note(
            None,
            args.uncertainty
                .then_some(InferenceCovarianceMode::Conditional),
        )
    );
    Ok(())
}

fn write_multinomial_prediction_csv(
    path: &Path,
    class_levels: &[String],
    probs: &Array2<f64>,
    prob_se: Option<&Array2<f64>>,
) -> Result<(), String> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;
    let mut headers: Vec<String> = class_levels
        .iter()
        .map(|level| format!("prob_{level}"))
        .collect();
    if prob_se.is_some() {
        headers.extend(class_levels.iter().map(|level| format!("prob_se_{level}")));
    }
    wtr.write_record(&headers)
        .map_err(|e| format!("failed to write csv header: {e}"))?;
    for i in 0..probs.nrows() {
        let mut row: Vec<String> = (0..probs.ncols())
            .map(|j| format!("{:.12}", probs[[i, j]]))
            .collect();
        if let Some(se) = prob_se {
            row.extend((0..se.ncols()).map(|j| format!("{:.12}", se[[i, j]])));
        }
        wtr.write_record(&row)
            .map_err(|e| format!("failed to write csv row {i}: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}
