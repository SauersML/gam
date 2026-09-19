//! Per-row residuals of a saved standard GAM on caller-supplied rows.
//!
//! A saved model carries no per-row training data, so residuals are a
//! function of the model AND the rows: the caller passes the rows back (the
//! training rows for in-sample residuals, any labeled rows otherwise) and the
//! residual is evaluated at the saved coefficients, `η = Xβ̂ + offset`, with
//! the family's own row kernel ([`gam_solve::pirls::glm_residuals`]). Every
//! front door — `gamfit`'s `Model.residuals(data, type=...)`, `gam residuals`
//! and this Rust entry — goes through [`saved_model_residuals`].

use crate::fit_orchestration::{
    resolve_continuous_column, resolve_offset_column, resolve_weight_column,
};
use crate::inference::model::{FittedModel, PredictModelClass};
use crate::inference::predict_input::build_predict_input_for_model;
use gam_data::EncodedDataset;
use gam_solve::pirls::{ResidualKind, glm_residuals};
use gam_spec::GlmLikelihoodSpec;
use gam_terms::inference::formula_dsl::formula_response_column;
use ndarray::Array1;

/// Residuals of type `kind` of a saved standard (single-predictor GLM/GAM)
/// model on `data`, which must carry the response and — when the model was
/// fit with them — the weight and offset columns.
///
/// Prior weights enter exactly as at fit time, so on the training rows the
/// squared deviance residuals sum to the fit's reported deviance and the
/// squared Pearson residuals to its Pearson statistic.
pub fn saved_model_residuals(
    model: &FittedModel,
    data: &EncodedDataset,
    kind: ResidualKind,
) -> Result<Array1<f64>, String> {
    if model.spline_scan.is_some() || model.residual_cascade.is_some() {
        return Err(
            "residuals are defined for coefficient-basis fits; this model was fit by the O(n) \
             spline scan or the residual cascade, which carry no coefficient vector to \
             evaluate the linear predictor at"
                .to_string(),
        );
    }
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(format!(
            "residuals are defined for standard single-predictor models; this is a {:?} model",
            model.predict_model_class()
        ));
    }
    let fit = model
        .payload()
        .fit_result
        .as_ref()
        .ok_or_else(|| "saved model has no fitted coefficient state".to_string())?;
    if fit.blocks.len() != 1 {
        return Err(format!(
            "residuals require exactly one coefficient block; this model has {}",
            fit.blocks.len()
        ));
    }
    let beta = &fit.blocks[0].beta;
    let response = formula_response_column(&model.payload().formula).ok_or_else(|| {
        format!(
            "could not resolve a response column from the saved formula '{}'",
            model.payload().formula
        )
    })?;
    let col_map = data.column_map();
    let y = resolve_continuous_column(data, &col_map, &response, "response")?;
    let prior_weights = resolve_weight_column(data, &col_map, model.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, &col_map, model.offset_column.as_deref())?;
    let input = build_predict_input_for_model(
        model,
        data.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &Array1::zeros(offset.len()),
        false,
    )?;
    if input.design.ncols() != beta.len() {
        return Err(format!(
            "residual design has {} columns; the fitted coefficient block has {}",
            input.design.ncols(),
            beta.len()
        ));
    }
    let eta = input.design.dot(beta) + &input.offset;
    let likelihood = GlmLikelihoodSpec::try_new(model.likelihood(), fit.likelihood_scale.clone())
        .map_err(|err| format!("saved likelihood scale: {err}"))?;
    let inverse_link = likelihood.spec.link.clone();
    glm_residuals(
        y.view(),
        eta.view(),
        prior_weights.view(),
        &likelihood,
        &inverse_link,
        kind,
    )
    .map_err(|err| format!("failed to evaluate {kind} residuals: {err}"))
}
