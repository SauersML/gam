use super::*;

use gam::families::multinomial::{
    MULTINOMIAL_MODEL_CLASS, MultinomialFitRequest, MultinomialModelEnvelope,
    MultinomialSavedModel, MultinomialSpreadDecline, fit_penalized_multinomial_formula,
    predict_multinomial_formula, predict_multinomial_formula_with_se,
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
    // #2612: the two branches of the conditional Firth/Jeffreys engagement
    // publish different estimands, and `--firth` is rejected on this family with
    // the message "the separation stabilizer is armed automatically". A user told
    // the decision is automatic is owed the decision.
    match saved.separation_evidence.as_deref() {
        Some(evidence) => cli_out!(
            "multinomial separation | Jeffreys/Firth proper prior ARMED (coefficients carry the \
             Firth bias correction) | evidence: {evidence}"
        ),
        None => cli_out!(
            "multinomial separation | none detected; Jeffreys/Firth prior disarmed (unbiased \
             penalized-REML mode)"
        ),
    }
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
    // Refusals of settings the softmax family cannot use. They read the resolved
    // configuration, so a `--request` document meets the same refusals as the flags.
    // The softmax link is fixed and the fit runs
    // a single joint softmax likelihood, so the location-scale / marginal-slope
    // / link-deviation controls have no meaning here; reject rather than
    // silently ignore.
    if fit_config.noise_formula.is_some() {
        return Err("--predict-noise is not supported for --family multinomial".to_string());
    }
    if fit_config.slope_formula.is_some() || fit_config.z_column.is_some() {
        return Err(
            "--slope-formula/--z-column is not supported for --family multinomial".to_string(),
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
    if fit_config.firth {
        return Err(
            "--firth is not accepted for --family multinomial: the Firth/Jeffreys separation \
             stabilizer is armed automatically when the fit detects complete separation"
                .to_string(),
        );
    }
    if fit_config.frailty != gam::families::survival::lognormal_kernel::FrailtySpec::None {
        return Err("frailty options are not supported for --family multinomial".to_string());
    }
    // Case weights (`--weights-column` → `fit_config.weight_column`) are
    // honored by the shared driver; offsets and the other config fields the
    // softmax family cannot consume are rejected with a typed error inside
    // `fit_penalized_multinomial_formula`, shared with the Python surface.
    let Some(out) = args.out.as_ref() else {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
    };

    let mut requested_columns = formula_columns(parsed)
        .map_err(|error| error.to_string())?
        .into_iter()
        .collect::<Vec<_>>();
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
    log::debug!("[PHASE] multinomial fit start n={}", ds.values.nrows());
    let saved = fit_penalized_multinomial_formula(&MultinomialFitRequest::new(
        &ds,
        formula_text,
        fit_config,
    ))
    .map_err(|e| format!("multinomial fit failed: {e}"))?;
    log::debug!(
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
/// carries the joint coefficient covariance, and a `prob_se_decline` column. A
/// row with no publishable standard error leaves its `prob_se_<class>` cells
/// empty and names why in `prob_se_decline`; every other row publishes (#1082).
pub(crate) fn run_predict_multinomial(args: &PredictArgs) -> Result<(), String> {
    let saved = load_multinomial_model(&args.model)?;
    let parsed = parse_formula(&saved.formula)?;

    // Prediction is for label-free new data: request the formula's feature
    // columns but not the response (which the predictor never references), and
    // force the same grouping-factor roles the fit used so by-factor encodings
    // line up with the frozen training basis.
    let mut requested_columns = formula_columns(&parsed)
        .map_err(|error| error.to_string())?
        .into_iter()
        .collect::<Vec<_>>();
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
        if args.covariance_mode == Some(InferenceCovarianceMode::SmoothingCorrected) {
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
        // The published probability is the posterior MEAN `E[softmax(η)]` over
        // the Laplace posterior, the one point estimand every prediction
        // surface reports (SPEC: never MAP). The plug-in softmax at the
        // posterior mode stays a library function for its callers, but it is
        // not a CLI estimand.
        let probs = predict_multinomial_formula(&saved, &ds)
            .map_err(|e| format!("multinomial predict failed: {e}"))?;
        (probs, None)
    };

    write_multinomial_prediction_csv(&args.out, &saved.class_levels, &probs, prob_se.as_deref())?;
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
    prob_se: Option<&[Result<Array1<f64>, MultinomialSpreadDecline>]>,
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
        headers.push("prob_se_decline".to_string());
    }
    wtr.write_record(&headers)
        .map_err(|e| format!("failed to write csv header: {e}"))?;
    for i in 0..probs.nrows() {
        let mut row: Vec<String> = (0..probs.ncols())
            .map(|j| format!("{:.12}", probs[[i, j]]))
            .collect();
        if let Some(se) = prob_se {
            match &se[i] {
                Ok(values) => {
                    row.extend(values.iter().map(|value| format!("{value:.12}")));
                    row.push(String::new());
                }
                Err(decline) => {
                    row.extend(std::iter::repeat_n(String::new(), probs.ncols()));
                    row.push(decline.to_string());
                }
            }
        }
        wtr.write_record(&row)
            .map_err(|e| format!("failed to write csv row {i}: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #1082: one row with no publishable standard error leaves the other rows'
    /// standard errors in place, blanks only its own cells, and names why.
    #[test]
    fn a_declined_row_blanks_only_its_own_standard_errors_1082() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("predictions.csv");
        let class_levels = vec!["a".to_string(), "b".to_string()];
        let probs = ndarray::array![[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]];
        let prob_se = vec![
            Ok(ndarray::array![0.1, 0.1]),
            Err(MultinomialSpreadDecline {
                class: 0,
                variance: -0.02,
                envelope: 2.5e-4,
            }),
            Ok(ndarray::array![0.02, 0.02]),
        ];
        write_multinomial_prediction_csv(&path, &class_levels, &probs, Some(prob_se.as_slice()))
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert_eq!(
            lines[0],
            "prob_a,prob_b,prob_se_a,prob_se_b,prob_se_decline"
        );
        assert_eq!(
            lines[1],
            "0.300000000000,0.700000000000,0.100000000000,0.100000000000,"
        );
        let declined: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(declined.len(), 5, "{}", lines[2]);
        assert_eq!(
            &declined[..4],
            &["0.500000000000", "0.500000000000", "", ""]
        );
        assert!(
            declined[4].contains("negative probability variance"),
            "{}",
            lines[2]
        );
        assert_eq!(
            lines[3],
            "0.900000000000,0.100000000000,0.020000000000,0.020000000000,"
        );
    }
}
