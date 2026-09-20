//! The conformal prediction routes of a saved standard GAM: the exact
//! full-conformal set at the fitted (frozen) smoothing parameters, and the
//! split-conformal band calibrated on a held-out labeled fold. Every front end
//! builds its conformal columns here and only marshals them.

use crate::interval_policy::{PredictionRequest, resolve_prediction_request};
use crate::{
    ConformalCalibrationFold, FittedModelPredictExt, InferenceCovarianceMode, MeanIntervalMethod,
    PredictUncertaintyOptions, predict_full_uncertainty_conformal,
};
use gam_models::inference::model::{FittedModel, PredictModelClass};
use gam_models::inference::predict_input::build_predict_input_for_model;
use gam_models::survival::predict::{
    fit_result_from_saved_model_for_prediction, resolve_termspec_for_prediction,
};
use gam_terms::inference::formula_dsl::formula_response_column;
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, ArrayView2};
use std::collections::{BTreeMap, HashMap};

/// Rows a split-conformal route predicts at or calibrates on: the data in the
/// model schema, its column map, and the offsets the caller resolved from it.
pub struct ConformalRows<'a> {
    pub data: ArrayView2<'a, f64>,
    pub col_map: &'a HashMap<String, usize>,
    pub offset: &'a Array1<f64>,
    pub noise_offset: &'a Array1<f64>,
}

/// Rows the exact full-conformal set predicts at, or is built on (then they must
/// also carry the response column): the data in the model schema and its column
/// map.
pub struct DesignRows<'a> {
    pub data: ArrayView2<'a, f64>,
    pub col_map: &'a HashMap<String, usize>,
}

/// Full-conformal prediction columns for a Gaussian-identity fit.
///
/// Reads the frozen penalty `Sλ` and its smoothing-parameter count persisted at
/// fit time (only for Gaussian-identity, unit-weight, offset-free models without
/// a link wiggle), rebuilds the design of the `labeled` rows and of the `test`
/// rows from the saved `resolved_termspec`, and calls
/// `substrate.interval(x_*, alpha)` per test row. The saved model persists no
/// training rows, so the caller supplies the labeled rows the set is built on.
///
/// Each row's set is that of the fitting map which re-selects the smoothing
/// strength by REML on the augmented rows, so the finite-sample coverage
/// theorem holds for it. The `conformal_certificate` column says what each row
/// carries: `0` exact_frozen (no strength to re-select), `1` honest_refit, and a
/// negative code for a typed refusal (`-1` multi_penalty, `-2`
/// unknown_penalty_structure, `-3` augmented_gram_singular, `-4` reml_undefined,
/// `-5` refit_outside_tube, `-6` refit_failed), where the row gets the frozen-ρ
/// set with no finite-sample guarantee. The set is a union of intervals;
/// `posterior_mean_lower` / `posterior_mean_upper` are its outer envelope (a
/// superset).
///
/// `alpha = 1 − conformal_level`: the full-conformal set `C_α` has marginal
/// coverage `≥ 1 − α`, with no factor of two.
pub fn full_conformal_prediction_columns(
    model: &FittedModel,
    test: &DesignRows<'_>,
    labeled: &DesignRows<'_>,
    conformal_level: f64,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    if !(conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    if model
        .saved_spline_scan()
        .map_err(|err| err.to_string())?
        .is_some()
    {
        return Err(
            "exact full-conformal intervals require a penalised-spline (B-spline) model; \
             this model was fit by the exact O(n) state-space scan. Refit with \
             double_penalty=true to obtain the standard model that carries the frozen penalty."
                .to_string(),
        );
    }
    let penalty = model.full_conformal.as_ref().ok_or_else(|| {
        "exact full-conformal intervals require a Gaussian-identity GLM trained without \
         prior weights, offsets, or a link wiggle. This model carries no frozen \
         full-conformal penalty (non-Gaussian family, weighted data, offset, or link \
         wiggle). Calibrate split-conformal intervals on a held-out labeled fold for \
         other families."
            .to_string()
    })?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(
            "exact full-conformal prediction supports only standard GAM models".to_string(),
        );
    }
    let dense_design = |rows: &DesignRows<'_>, what: &str| {
        let spec = resolve_termspec_for_prediction(
            &model.resolved_termspec,
            model.training_headers.as_ref(),
            rows.col_map,
            "resolved_termspec",
        )?;
        let design = build_term_collection_design(rows.data, &spec)
            .map_err(|err| format!("full conformal: failed to build {what} design: {err}"))?;
        design
            .design
            .try_to_dense_by_chunks(&format!("full conformal {what} design"))
    };
    let response_name = formula_response_column(&model.payload().formula).ok_or_else(|| {
        "full conformal: could not resolve the response column from the saved formula"
            .to_string()
    })?;
    let response_col = *labeled.col_map.get(&response_name).ok_or_else(|| {
        format!(
            "exact full-conformal training data must contain the response column \
             '{response_name}' (the set is built on labeled rows)"
        )
    })?;
    let y_labeled = labeled.data.column(response_col).to_owned();
    let x_labeled = dense_design(labeled, "training")?;
    let substrate = penalty.with_labeled_rows(x_labeled, y_labeled)?;
    let x_test = dense_design(test, "test")?;
    let n_test = x_test.nrows();
    if x_test.ncols() != penalty.p() {
        return Err(format!(
            "full conformal: test design has {} columns but the stored penalty has p={}; \
             the model may need to be refit",
            x_test.ncols(),
            penalty.p()
        ));
    }
    let alpha = 1.0 - conformal_level;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    if fit.beta.len() != x_test.ncols() {
        return Err(format!(
            "full conformal: fit has {} coefficients but test design has {} columns",
            fit.beta.len(),
            x_test.ncols()
        ));
    }
    let mut mean_vec = Vec::with_capacity(n_test);
    let mut lower_vec = Vec::with_capacity(n_test);
    let mut upper_vec = Vec::with_capacity(n_test);
    let mut certificate_vec = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let x_star = x_test.row(i).to_owned();
        let iv = substrate
            .interval(&x_star, alpha)
            .map_err(|e| format!("full conformal at row {i}: {e}"))?;
        // The conformal set changes only the interval. The point is the fitted
        // Gaussian-identity posterior mean, which equals the plug-in X beta;
        // using the envelope centre made the point depend on interval shape.
        mean_vec.push(x_star.dot(&fit.beta));
        lower_vec.push(iv.lo);
        upper_vec.push(iv.hi);
        certificate_vec.push(f64::from(iv.certificate.code()));
    }
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("linear_predictor_plugin".to_string(), mean_vec.clone());
    columns.insert("mean_plugin".to_string(), mean_vec.clone());
    columns.insert("posterior_mean".to_string(), mean_vec);
    columns.insert("posterior_mean_lower".to_string(), lower_vec);
    columns.insert("posterior_mean_upper".to_string(), upper_vec);
    columns.insert("conformal_certificate".to_string(), certificate_vec);
    Ok(columns)
}

/// Split-conformal prediction columns.
///
/// Runs the model-based full-uncertainty predictor on `test` (honouring
/// `covariance_mode` and `observation_interval`), then replaces the
/// response-scale `posterior_mean_lower` / `posterior_mean_upper` with the
/// split-conformal interval `μ̂(x) ± q̂·s(x)` calibrated at `conformal_level`
/// from the held-out labeled `calibration` fold. The fold carries its own design,
/// may be of any size independent of the training set, and must contain the
/// response column resolved from the saved formula.
pub fn split_conformal_prediction_columns(
    model: &FittedModel,
    test: &ConformalRows<'_>,
    calibration: &ConformalRows<'_>,
    conformal_level: f64,
    covariance_mode: Option<InferenceCovarianceMode>,
    observation_interval: bool,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    if !(conformal_level.is_finite() && conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    // Split-conformal calibration is built on the dense predictor + fit_result,
    // neither of which a scan-routed model carries. Point and posterior-interval
    // prediction (the scan-aware predict path) work for these models, so direct
    // users there rather than failing with the cryptic missing-resolved_termspec
    // error (#1046).
    if let Some((feature_column, _)) = model.saved_spline_scan().map_err(|err| err.to_string())? {
        return Err(format!(
            "s({feature_column}) is fit by the exact O(n) state-space spline scan, which \
             does not carry the dense predictor split-conformal calibration needs. Use a \
             posterior interval (scan-aware), or refit with double_penalty=true for \
             conformal intervals."
        ));
    }
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err("conformal prediction currently supports only standard GAM models".to_string());
    }
    let predict_input = build_predict_input_for_model(
        model,
        test.data,
        test.col_map,
        model.training_headers.as_ref(),
        test.offset,
        test.noise_offset,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let family = model.likelihood();

    let covariance_mode = covariance_mode.unwrap_or_else(|| fit.published_covariance_mode());
    let uncertainty_options = PredictUncertaintyOptions {
        confidence_level: conformal_level,
        covariance_mode,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: observation_interval,
        conformal_level: Some(conformal_level),
        ..PredictUncertaintyOptions::default()
    };

    // Build the calibration-fold predict input the same way the test batch is
    // built, so the predict engine yields μ̂(x_cal) and s(x_cal) from exactly
    // the same source used at test time.
    let response_name = formula_response_column(&model.payload().formula).ok_or_else(|| {
        "conformal calibration: could not resolve the response column from the saved formula"
            .to_string()
    })?;
    let response_col = *calibration.col_map.get(&response_name).ok_or_else(|| {
        format!(
            "conformal calibration data must contain the response column '{response_name}' \
             (calibration is held-out labeled data)"
        )
    })?;
    let cal_y = calibration.data.column(response_col).to_owned();
    let cal_input = build_predict_input_for_model(
        model,
        calibration.data,
        calibration.col_map,
        model.training_headers.as_ref(),
        calibration.offset,
        calibration.noise_offset,
        false,
    )?;
    if cal_input.design.ncols() != fit.beta.len() {
        return Err(format!(
            "conformal calibration design has {} columns but the fit has {} coefficients",
            cal_input.design.ncols(),
            fit.beta.len()
        ));
    }
    let calibration_fold = ConformalCalibrationFold {
        input: cal_input,
        y: cal_y.view(),
    };
    let prediction = predict_full_uncertainty_conformal(
        predictor.as_ref(),
        &predict_input,
        &fit,
        &family,
        &uncertainty_options,
        &calibration_fold,
    )
    .map_err(|err| format!("conformal prediction failed: {err}"))?;

    // The conformal interval changes only the band. Its point remains the same
    // posterior response mean as ordinary prediction (#398, SPEC), while the
    // complete plug-in pair is exposed alongside it under explicit names.
    let point = resolve_prediction_request(
        predictor.as_ref(),
        &predict_input,
        &fit,
        model.prediction_uses_posterior_mean(),
        &PredictionRequest {
            interval: None,
            covariance_mode,
            observation_interval: false,
            observation_prior_weights: None,
            extrapolation_variance: None,
        },
    )
    .map_err(|err| format!("conformal point prediction failed: {err}"))?;
    let posterior_mean = point.posterior_mean.ok_or_else(|| {
        "conformal prediction did not produce the required posterior mean".to_string()
    })?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert(
        "linear_predictor_plugin".to_string(),
        point.linear_predictor_plugin.to_vec(),
    );
    columns.insert("mean_plugin".to_string(), point.mean_plugin.to_vec());
    columns.insert("posterior_mean".to_string(), posterior_mean.to_vec());
    // Response-scale SE beside the response-scale mean/band (#1536): emit
    // `mean_standard_error`, not the link-scale `eta_standard_error`.
    columns.insert(
        "posterior_mean_standard_error".to_string(),
        prediction.mean_standard_error.to_vec(),
    );
    columns.insert(
        "posterior_mean_lower".to_string(),
        prediction.mean_lower.to_vec(),
    );
    columns.insert(
        "posterior_mean_upper".to_string(),
        prediction.mean_upper.to_vec(),
    );
    if let (Some(obs_lower), Some(obs_upper)) =
        (prediction.observation_lower, prediction.observation_upper)
    {
        columns.insert("observation_lower".to_string(), obs_lower.to_vec());
        columns.insert("observation_upper".to_string(), obs_upper.to_vec());
    }
    Ok(columns)
}
