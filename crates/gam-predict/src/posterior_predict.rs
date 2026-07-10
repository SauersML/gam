//! Posterior-draw prediction for saved, non-survival models.
//!
//! This module is the coefficient-draw analogue of the point-prediction
//! dispatch in this crate.  It deliberately delegates every row evaluation to
//! the same model-specific kernel used by [`PredictableModel`]: posterior
//! prediction changes the coefficient vector, not the definition of a model's
//! linear predictor or response transform.

use std::collections::HashMap;
use std::ops::Range;

use gam_models::inference::model::{FittedModel, PredictModelClass, SavedLinkWiggleRuntime};
use gam_problem::{BlockRole, EstimationError};
use gam_solve::model_types::UnifiedFitResult;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::binomial_location_scale::BinomialLocationScalePredictor;
use crate::dispersion_location_scale::DispersionLocationScalePredictor;
use crate::gaussian_location_scale::GaussianLocationScalePredictor;
use crate::input::{build_predict_input_for_model, build_transformation_normal_quantile_grid};
use crate::standard::StandardPredictor;
use crate::{FittedModelPredictExt, PredictInput, PredictResult, PredictableModel};

/// The two posterior-draw surfaces shared by every non-survival model class.
///
/// Rows index posterior draws and columns index prediction rows.  `eta` is the
/// model class's canonical primary predictor (for example the standardized
/// threshold-scale index for a binomial location-scale model), while `mean` is
/// its response-scale point prediction for the same draw.
#[derive(Clone, Debug)]
pub struct PosteriorDrawMatrices {
    pub eta: Array2<f64>,
    pub mean: Array2<f64>,
}

impl PosteriorDrawMatrices {
    #[inline]
    pub fn n_draws(&self) -> usize {
        self.eta.nrows()
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.eta.ncols()
    }
}

/// Typed posterior-draw prediction for every non-survival model class.
///
/// Keeping the model class in the type prevents a transformation-normal draw
/// surface, whose `eta` is its response-scale conditional mean, from being
/// accidentally interpreted as a standard GLM link-scale matrix.  Callers
/// that only need the common matrices can use [`Self::matrices`].
#[derive(Clone, Debug)]
pub enum PosteriorDrawPrediction {
    Standard(PosteriorDrawMatrices),
    GaussianLocationScale(PosteriorDrawMatrices),
    BinomialLocationScale(PosteriorDrawMatrices),
    DispersionLocationScale(PosteriorDrawMatrices),
    BernoulliMarginalSlope(PosteriorDrawMatrices),
    TransformationNormal(PosteriorDrawMatrices),
}

impl PosteriorDrawPrediction {
    #[inline]
    pub const fn model_class(&self) -> PredictModelClass {
        match self {
            Self::Standard(_) => PredictModelClass::Standard,
            Self::GaussianLocationScale(_) => PredictModelClass::GaussianLocationScale,
            Self::BinomialLocationScale(_) => PredictModelClass::BinomialLocationScale,
            Self::DispersionLocationScale(_) => PredictModelClass::DispersionLocationScale,
            Self::BernoulliMarginalSlope(_) => PredictModelClass::BernoulliMarginalSlope,
            Self::TransformationNormal(_) => PredictModelClass::TransformationNormal,
        }
    }

    #[inline]
    pub fn matrices(&self) -> &PosteriorDrawMatrices {
        match self {
            Self::Standard(matrices)
            | Self::GaussianLocationScale(matrices)
            | Self::BinomialLocationScale(matrices)
            | Self::DispersionLocationScale(matrices)
            | Self::BernoulliMarginalSlope(matrices)
            | Self::TransformationNormal(matrices) => matrices,
        }
    }

    #[inline]
    pub fn into_matrices(self) -> PosteriorDrawMatrices {
        match self {
            Self::Standard(matrices)
            | Self::GaussianLocationScale(matrices)
            | Self::BinomialLocationScale(matrices)
            | Self::DispersionLocationScale(matrices)
            | Self::BernoulliMarginalSlope(matrices)
            | Self::TransformationNormal(matrices) => matrices,
        }
    }
}

/// Typed failures from [`predict_posterior_draws`].
#[derive(Debug)]
pub enum PosteriorPredictError {
    EmptyDraws,
    SurvivalNotImplemented,
    MissingColumn {
        channel: &'static str,
        column: String,
    },
    ColumnOutOfBounds {
        channel: &'static str,
        column: String,
        index: usize,
        ncols: usize,
    },
    NonFiniteInput {
        channel: &'static str,
        row: usize,
        value: f64,
    },
    NonFiniteDraw {
        draw: usize,
        coefficient: usize,
        value: f64,
    },
    CoefficientCount {
        model_class: PredictModelClass,
        expected: usize,
        actual: usize,
    },
    MissingModelState {
        model_class: PredictModelClass,
        reason: String,
    },
    InconsistentModelState {
        model_class: PredictModelClass,
        reason: String,
    },
    InputAssembly {
        model_class: PredictModelClass,
        reason: String,
    },
    DrawEvaluation {
        model_class: PredictModelClass,
        draw: usize,
        source: EstimationError,
    },
}

impl std::fmt::Display for PosteriorPredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDraws => f.write_str("posterior prediction requires at least one draw"),
            Self::SurvivalNotImplemented => f.write_str(
                "survival posterior prediction is handled by the survival prediction module",
            ),
            Self::MissingColumn { channel, column } => {
                write!(
                    f,
                    "posterior prediction {channel} column {column:?} is missing"
                )
            }
            Self::ColumnOutOfBounds {
                channel,
                column,
                index,
                ncols,
            } => write!(
                f,
                "posterior prediction {channel} column {column:?} resolves to index {index}, \
                 outside the {ncols}-column numeric input",
            ),
            Self::NonFiniteInput {
                channel,
                row,
                value,
            } => write!(
                f,
                "posterior prediction {channel} is non-finite at row {row}: {value}",
            ),
            Self::NonFiniteDraw {
                draw,
                coefficient,
                value,
            } => write!(
                f,
                "posterior draw {draw} coefficient {coefficient} is non-finite: {value}",
            ),
            Self::CoefficientCount {
                model_class,
                expected,
                actual,
            } => write!(
                f,
                "posterior prediction coefficient count mismatch for {}: expected {expected}, \
                 got {actual}; draws must come from this fitted model",
                model_class.name(),
            ),
            Self::MissingModelState {
                model_class,
                reason,
            } => write!(
                f,
                "{} posterior prediction is missing fitted state: {reason}",
                model_class.name(),
            ),
            Self::InconsistentModelState {
                model_class,
                reason,
            } => write!(
                f,
                "{} posterior prediction found inconsistent fitted state: {reason}",
                model_class.name(),
            ),
            Self::InputAssembly {
                model_class,
                reason,
            } => write!(
                f,
                "failed to assemble {} posterior-prediction input: {reason}",
                model_class.name(),
            ),
            Self::DrawEvaluation {
                model_class,
                draw,
                source,
            } => write!(
                f,
                "{} posterior prediction failed for draw {draw}: {source}",
                model_class.name(),
            ),
        }
    }
}

impl std::error::Error for PosteriorPredictError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DrawEvaluation { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn fitted_state_error(
    model_class: PredictModelClass,
    reason: impl Into<String>,
) -> PosteriorPredictError {
    PosteriorPredictError::MissingModelState {
        model_class,
        reason: reason.into(),
    }
}

fn inconsistent_state_error(
    model_class: PredictModelClass,
    reason: impl Into<String>,
) -> PosteriorPredictError {
    PosteriorPredictError::InconsistentModelState {
        model_class,
        reason: reason.into(),
    }
}

fn validate_draw_values(draws: ArrayView2<'_, f64>) -> Result<(), PosteriorPredictError> {
    if draws.nrows() == 0 {
        return Err(PosteriorPredictError::EmptyDraws);
    }
    for ((draw, coefficient), &value) in draws.indexed_iter() {
        if !value.is_finite() {
            return Err(PosteriorPredictError::NonFiniteDraw {
                draw,
                coefficient,
                value,
            });
        }
    }
    Ok(())
}

fn validate_coefficient_count(
    model_class: PredictModelClass,
    expected: usize,
    draws: ArrayView2<'_, f64>,
) -> Result<(), PosteriorPredictError> {
    if draws.ncols() != expected {
        return Err(PosteriorPredictError::CoefficientCount {
            model_class,
            expected,
            actual: draws.ncols(),
        });
    }
    Ok(())
}

fn resolve_offset(
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    column: Option<&str>,
    channel: &'static str,
) -> Result<Array1<f64>, PosteriorPredictError> {
    let Some(column) = column else {
        return Ok(Array1::zeros(data.nrows()));
    };
    let index =
        col_map
            .get(column)
            .copied()
            .ok_or_else(|| PosteriorPredictError::MissingColumn {
                channel,
                column: column.to_string(),
            })?;
    if index >= data.ncols() {
        return Err(PosteriorPredictError::ColumnOutOfBounds {
            channel,
            column: column.to_string(),
            index,
            ncols: data.ncols(),
        });
    }
    let offset = data.column(index).to_owned();
    if let Some((row, &value)) = offset
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(PosteriorPredictError::NonFiniteInput {
            channel,
            row,
            value,
        });
    }
    Ok(offset)
}

fn build_non_survival_input(
    model: &FittedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<PredictInput, PosteriorPredictError> {
    let model_class = model.predict_model_class();
    let offset = resolve_offset(data, col_map, model.offset_column.as_deref(), "offset")?;
    let offset_noise = resolve_offset(
        data,
        col_map,
        model.noise_offset_column.as_deref(),
        "noise offset",
    )?;
    build_predict_input_for_model(
        model,
        data,
        col_map,
        training_headers,
        &offset,
        &offset_noise,
        model.noise_offset_column.is_some(),
    )
    .map_err(|reason| PosteriorPredictError::InputAssembly {
        model_class,
        reason,
    })
}

fn validate_flattened_fit(
    fit: &UnifiedFitResult,
    model_class: PredictModelClass,
) -> Result<(), PosteriorPredictError> {
    let block_len: usize = fit.blocks.iter().map(|block| block.beta.len()).sum();
    if block_len != fit.beta.len() {
        return Err(inconsistent_state_error(
            model_class,
            format!(
                "concatenated coefficient vector has length {}, but coefficient blocks total {block_len}",
                fit.beta.len(),
            ),
        ));
    }
    let mut cursor = 0usize;
    for (index, block) in fit.blocks.iter().enumerate() {
        let end = cursor + block.beta.len();
        if fit.beta.slice(s![cursor..end]) != block.beta.view() {
            return Err(inconsistent_state_error(
                model_class,
                format!(
                    "coefficient block {index} ({:?}) does not match the same range in the joint coefficient vector",
                    block.role,
                ),
            ));
        }
        cursor = end;
    }
    Ok(())
}

fn block_range(fit: &UnifiedFitResult, role: BlockRole) -> Option<Range<usize>> {
    let mut start = 0usize;
    for block in &fit.blocks {
        let end = start + block.beta.len();
        if block.role == role {
            return Some(start..end);
        }
        start = end;
    }
    None
}

fn preferred_block_range(
    fit: &UnifiedFitResult,
    roles: &[BlockRole],
    model_class: PredictModelClass,
    label: &'static str,
) -> Result<Range<usize>, PosteriorPredictError> {
    roles
        .iter()
        .find_map(|&role| block_range(fit, role))
        .ok_or_else(|| {
            fitted_state_error(
                model_class,
                format!("missing {label} coefficient block (accepted roles: {roles:?})"),
            )
        })
}

fn required_block_range(
    fit: &UnifiedFitResult,
    role: BlockRole,
    model_class: PredictModelClass,
    label: &'static str,
) -> Result<Range<usize>, PosteriorPredictError> {
    block_range(fit, role).ok_or_else(|| {
        fitted_state_error(
            model_class,
            format!("missing {label} coefficient block with role {role:?}"),
        )
    })
}

fn validate_runtime_width(
    runtime: Option<&SavedLinkWiggleRuntime>,
    range: Option<&Range<usize>>,
    model_class: PredictModelClass,
) -> Result<(), PosteriorPredictError> {
    match (runtime, range) {
        (None, None) => Ok(()),
        (Some(runtime), Some(range)) if runtime.beta.len() == range.len() => Ok(()),
        (Some(runtime), Some(range)) => Err(inconsistent_state_error(
            model_class,
            format!(
                "saved link-wiggle runtime has {} coefficients, but the fitted LinkWiggle block has {}",
                runtime.beta.len(),
                range.len(),
            ),
        )),
        (Some(_), None) => Err(fitted_state_error(
            model_class,
            "saved link-wiggle runtime has no fitted LinkWiggle coefficient block",
        )),
        (None, Some(_)) => Err(inconsistent_state_error(
            model_class,
            "fitted LinkWiggle coefficient block has no saved prediction runtime",
        )),
    }
}

fn assign_runtime_beta(
    runtime: &mut SavedLinkWiggleRuntime,
    draw: ArrayView1<'_, f64>,
    range: &Range<usize>,
) {
    for (coefficient, &value) in runtime
        .beta
        .iter_mut()
        .zip(draw.slice(s![range.start..range.end]).iter())
    {
        *coefficient = value;
    }
}

fn evaluate_draw_matrices<F>(
    draws: ArrayView2<'_, f64>,
    n_rows: usize,
    model_class: PredictModelClass,
    mut evaluate: F,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError>
where
    F: FnMut(ArrayView1<'_, f64>) -> Result<PredictResult, EstimationError>,
{
    let mut eta = Array2::<f64>::zeros((draws.nrows(), n_rows));
    let mut mean = Array2::<f64>::zeros((draws.nrows(), n_rows));
    for (draw_index, draw) in draws.rows().into_iter().enumerate() {
        let prediction =
            evaluate(draw).map_err(|source| PosteriorPredictError::DrawEvaluation {
                model_class,
                draw: draw_index,
                source,
            })?;
        if prediction.eta.len() != n_rows || prediction.mean.len() != n_rows {
            return Err(inconsistent_state_error(
                model_class,
                format!(
                    "draw {draw_index} produced eta/mean lengths {}/{}, expected {n_rows}",
                    prediction.eta.len(),
                    prediction.mean.len(),
                ),
            ));
        }
        eta.row_mut(draw_index).assign(&prediction.eta);
        mean.row_mut(draw_index).assign(&prediction.mean);
    }
    Ok(PosteriorDrawMatrices { eta, mean })
}

fn standard_draws(
    model: &FittedModel,
    input: &PredictInput,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::Standard;
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| fitted_state_error(model_class, "missing canonical fit_result"))?;
    validate_flattened_fit(fit, model_class)?;
    validate_coefficient_count(model_class, fit.beta.len(), draws)?;
    let mut link_wiggle = model
        .saved_prediction_runtime()
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?
        .link_wiggle;
    let (main_range, wiggle_range) = match link_wiggle.as_ref() {
        Some(_) => {
            let main = required_block_range(fit, BlockRole::Mean, model_class, "mean")?;
            let wiggle =
                required_block_range(fit, BlockRole::LinkWiggle, model_class, "link-wiggle")?;
            validate_runtime_width(link_wiggle.as_ref(), Some(&wiggle), model_class)?;
            (main, Some(wiggle))
        }
        None => (0..fit.beta.len(), None),
    };
    let mut predictor = StandardPredictor {
        beta: fit
            .beta
            .slice(s![main_range.start..main_range.end])
            .to_owned(),
        family: model.likelihood(),
        link_kind: model
            .resolved_inverse_link()
            .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?,
        covariance: None,
        link_wiggle: link_wiggle.take(),
    };
    evaluate_draw_matrices(draws, input.design.nrows(), model_class, |draw| {
        predictor
            .beta
            .assign(&draw.slice(s![main_range.start..main_range.end]));
        if let (Some(runtime), Some(range)) =
            (predictor.link_wiggle.as_mut(), wiggle_range.as_ref())
        {
            assign_runtime_beta(runtime, draw, range);
        }
        predictor.predict_plugin_response(input)
    })
}

fn gaussian_location_scale_draws(
    model: &FittedModel,
    input: &PredictInput,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::GaussianLocationScale;
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| fitted_state_error(model_class, "missing canonical fit_result"))?;
    validate_flattened_fit(fit, model_class)?;
    validate_coefficient_count(model_class, fit.beta.len(), draws)?;
    let mean_range = preferred_block_range(
        fit,
        &[BlockRole::Location, BlockRole::Mean],
        model_class,
        "location/mean",
    )?;
    let scale_range = required_block_range(fit, BlockRole::Scale, model_class, "scale")?;
    let mut link_wiggle = model
        .saved_prediction_runtime()
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?
        .link_wiggle;
    let wiggle_range = block_range(fit, BlockRole::LinkWiggle);
    validate_runtime_width(link_wiggle.as_ref(), wiggle_range.as_ref(), model_class)?;
    let mut predictor = GaussianLocationScalePredictor {
        beta_mu: fit
            .beta
            .slice(s![mean_range.start..mean_range.end])
            .to_owned(),
        beta_noise: fit
            .beta
            .slice(s![scale_range.start..scale_range.end])
            .to_owned(),
        sigma_floor: gam_model_kernels::sigma_link::LOGB_SIGMA_FLOOR,
        response_scale: model.gaussian_response_scale.unwrap_or(1.0),
        covariance: None,
        link_wiggle: link_wiggle.take(),
    };
    evaluate_draw_matrices(draws, input.design.nrows(), model_class, |draw| {
        predictor
            .beta_mu
            .assign(&draw.slice(s![mean_range.start..mean_range.end]));
        predictor
            .beta_noise
            .assign(&draw.slice(s![scale_range.start..scale_range.end]));
        if let (Some(runtime), Some(range)) =
            (predictor.link_wiggle.as_mut(), wiggle_range.as_ref())
        {
            assign_runtime_beta(runtime, draw, range);
        }
        predictor.predict_plugin_response(input)
    })
}

fn binomial_location_scale_draws(
    model: &FittedModel,
    input: &PredictInput,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::BinomialLocationScale;
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| fitted_state_error(model_class, "missing canonical fit_result"))?;
    validate_flattened_fit(fit, model_class)?;
    validate_coefficient_count(model_class, fit.beta.len(), draws)?;
    let threshold_range = preferred_block_range(
        fit,
        &[BlockRole::Threshold, BlockRole::Location, BlockRole::Mean],
        model_class,
        "threshold/location",
    )?;
    let scale_range = required_block_range(fit, BlockRole::Scale, model_class, "scale")?;
    let mut link_wiggle = model
        .saved_prediction_runtime()
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?
        .link_wiggle;
    let wiggle_range = block_range(fit, BlockRole::LinkWiggle);
    validate_runtime_width(link_wiggle.as_ref(), wiggle_range.as_ref(), model_class)?;
    let inverse_link = model
        .resolved_inverse_link()
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?
        .ok_or_else(|| fitted_state_error(model_class, "missing resolved inverse link"))?;
    let mut predictor = BinomialLocationScalePredictor {
        beta_threshold: fit
            .beta
            .slice(s![threshold_range.start..threshold_range.end])
            .to_owned(),
        beta_noise: fit
            .beta
            .slice(s![scale_range.start..scale_range.end])
            .to_owned(),
        covariance: None,
        inverse_link,
        link_wiggle: link_wiggle.take(),
    };
    evaluate_draw_matrices(draws, input.design.nrows(), model_class, |draw| {
        predictor
            .beta_threshold
            .assign(&draw.slice(s![threshold_range.start..threshold_range.end]));
        predictor
            .beta_noise
            .assign(&draw.slice(s![scale_range.start..scale_range.end]));
        if let (Some(runtime), Some(range)) =
            (predictor.link_wiggle.as_mut(), wiggle_range.as_ref())
        {
            assign_runtime_beta(runtime, draw, range);
        }
        predictor.predict_plugin_response(input)
    })
}

fn dispersion_location_scale_draws(
    model: &FittedModel,
    input: &PredictInput,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::DispersionLocationScale;
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| fitted_state_error(model_class, "missing canonical fit_result"))?;
    validate_flattened_fit(fit, model_class)?;
    validate_coefficient_count(model_class, fit.beta.len(), draws)?;
    let mean_range = preferred_block_range(
        fit,
        &[BlockRole::Location, BlockRole::Mean],
        model_class,
        "location/mean",
    )?;
    let scale_range = required_block_range(fit, BlockRole::Scale, model_class, "scale")?;
    let mut predictor = DispersionLocationScalePredictor {
        beta_mu: fit
            .beta
            .slice(s![mean_range.start..mean_range.end])
            .to_owned(),
        beta_noise: fit
            .beta
            .slice(s![scale_range.start..scale_range.end])
            .to_owned(),
        likelihood: model.likelihood(),
        inverse_link: model
            .resolved_inverse_link()
            .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?,
        covariance: None,
    };
    evaluate_draw_matrices(draws, input.design.nrows(), model_class, |draw| {
        predictor
            .beta_mu
            .assign(&draw.slice(s![mean_range.start..mean_range.end]));
        predictor
            .beta_noise
            .assign(&draw.slice(s![scale_range.start..scale_range.end]));
        predictor.predict_plugin_response(input)
    })
}

fn bernoulli_marginal_slope_draws(
    model: &FittedModel,
    input: &PredictInput,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::BernoulliMarginalSlope;
    let predictor = model
        .bernoulli_marginal_slope_predictor()
        .map_err(|reason| fitted_state_error(model_class, reason))?;
    validate_coefficient_count(model_class, predictor.theta_len(), draws)?;
    let mut theta = Array1::<f64>::zeros(predictor.theta_len());
    evaluate_draw_matrices(draws, input.design.nrows(), model_class, |draw| {
        theta.assign(&draw);
        let eta = predictor.final_eta_from_theta(input, &theta)?;
        let mean = predictor.mean_from_eta(&eta)?;
        Ok(PredictResult { eta, mean })
    })
}

fn assign_fit_theta(
    fit: &mut UnifiedFitResult,
    draw: ArrayView1<'_, f64>,
    model_class: PredictModelClass,
) -> Result<(), PosteriorPredictError> {
    if fit.beta.len() != draw.len() {
        return Err(inconsistent_state_error(
            model_class,
            format!(
                "redundant fitted representation has {} coefficients, expected {}",
                fit.beta.len(),
                draw.len(),
            ),
        ));
    }
    fit.beta.assign(&draw);
    let mut cursor = 0usize;
    for block in &mut fit.blocks {
        let end = cursor + block.beta.len();
        if end > draw.len() {
            return Err(inconsistent_state_error(
                model_class,
                "coefficient blocks extend past the joint coefficient vector",
            ));
        }
        block.beta.assign(&draw.slice(s![cursor..end]));
        cursor = end;
    }
    if cursor != draw.len() {
        return Err(inconsistent_state_error(
            model_class,
            format!(
                "coefficient blocks consume {cursor} entries, but the joint draw has {}",
                draw.len(),
            ),
        ));
    }
    Ok(())
}

fn transformation_normal_draws(
    model: &FittedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawMatrices, PosteriorPredictError> {
    let model_class = PredictModelClass::TransformationNormal;
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| fitted_state_error(model_class, "missing canonical fit_result"))?;
    validate_flattened_fit(fit, model_class)?;
    validate_coefficient_count(model_class, fit.beta.len(), draws)?;
    let unified = model
        .unified()
        .ok_or_else(|| fitted_state_error(model_class, "missing unified fit"))?;
    validate_flattened_fit(unified, model_class)?;
    if unified.beta.len() != fit.beta.len() {
        return Err(inconsistent_state_error(
            model_class,
            format!(
                "canonical fit_result has {} coefficients but unified fit has {}",
                fit.beta.len(),
                unified.beta.len(),
            ),
        ));
    }
    let offset = resolve_offset(data, col_map, model.offset_column.as_deref(), "offset")?;
    let mut draw_model = model.clone();
    let mut eta = Array2::<f64>::zeros((draws.nrows(), data.nrows()));
    let mut mean = Array2::<f64>::zeros((draws.nrows(), data.nrows()));
    for (draw_index, draw) in draws.rows().into_iter().enumerate() {
        let draw_fit = draw_model.fit_result.as_mut().ok_or_else(|| {
            fitted_state_error(model_class, "mutable model clone lost canonical fit_result")
        })?;
        assign_fit_theta(draw_fit, draw, model_class)?;
        let draw_unified = draw_model.unified.as_mut().ok_or_else(|| {
            fitted_state_error(model_class, "mutable model clone lost unified fit")
        })?;
        assign_fit_theta(draw_unified, draw, model_class)?;
        let grid = build_transformation_normal_quantile_grid(
            &draw_model,
            data,
            col_map,
            training_headers,
            &offset,
        )
        .map_err(|reason| PosteriorPredictError::DrawEvaluation {
            model_class,
            draw: draw_index,
            source: EstimationError::InvalidInput(reason),
        })?;
        if grid.conditional_mean.len() != data.nrows() {
            return Err(inconsistent_state_error(
                model_class,
                format!(
                    "draw {draw_index} produced {} conditional means, expected {}",
                    grid.conditional_mean.len(),
                    data.nrows(),
                ),
            ));
        }
        eta.row_mut(draw_index).assign(&grid.conditional_mean);
        mean.row_mut(draw_index).assign(&grid.conditional_mean);
    }
    Ok(PosteriorDrawMatrices { eta, mean })
}

/// Evaluate posterior coefficient draws on new numeric data.
///
/// `draws` must use the fitted model's canonical concatenated coefficient
/// order (the same order returned by the core posterior sampler).  The result
/// has shape `(n_draws, data.nrows())`.  Offsets are resolved from the saved
/// model's named offset columns in `col_map`, exactly as in point prediction.
///
/// Survival is intentionally outside this API: its prediction target is a
/// typed curve/surface rather than one eta/mean pair per input row and is
/// implemented in the survival prediction module.
pub fn predict_posterior_draws(
    model: &FittedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    draws: ArrayView2<'_, f64>,
) -> Result<PosteriorDrawPrediction, PosteriorPredictError> {
    validate_draw_values(draws)?;
    match model.predict_model_class() {
        PredictModelClass::Standard => {
            let input = build_non_survival_input(model, data, col_map, training_headers)?;
            standard_draws(model, &input, draws).map(PosteriorDrawPrediction::Standard)
        }
        PredictModelClass::GaussianLocationScale => {
            let input = build_non_survival_input(model, data, col_map, training_headers)?;
            gaussian_location_scale_draws(model, &input, draws)
                .map(PosteriorDrawPrediction::GaussianLocationScale)
        }
        PredictModelClass::BinomialLocationScale => {
            let input = build_non_survival_input(model, data, col_map, training_headers)?;
            binomial_location_scale_draws(model, &input, draws)
                .map(PosteriorDrawPrediction::BinomialLocationScale)
        }
        PredictModelClass::DispersionLocationScale => {
            let input = build_non_survival_input(model, data, col_map, training_headers)?;
            dispersion_location_scale_draws(model, &input, draws)
                .map(PosteriorDrawPrediction::DispersionLocationScale)
        }
        PredictModelClass::BernoulliMarginalSlope => {
            let input = build_non_survival_input(model, data, col_map, training_headers)?;
            bernoulli_marginal_slope_draws(model, &input, draws)
                .map(PosteriorDrawPrediction::BernoulliMarginalSlope)
        }
        PredictModelClass::TransformationNormal => {
            transformation_normal_draws(model, data, col_map, training_headers, draws)
                .map(PosteriorDrawPrediction::TransformationNormal)
        }
        PredictModelClass::Survival => Err(PosteriorPredictError::SurvivalNotImplemented),
    }
}

#[cfg(test)]
mod tests {
    use gam_linalg::matrix::DesignMatrix;
    use gam_spec::LikelihoodSpec;
    use ndarray::array;

    use super::*;

    #[test]
    fn one_standard_mode_draw_equals_canonical_point_prediction() {
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0, -0.5], [1.0, 2.0]]),
            offset: array![0.25, -0.75],
            design_noise: None,
            offset_noise: None,
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };
        let mode = array![0.4, -0.2];
        let point_predictor = StandardPredictor {
            beta: mode.clone(),
            family: LikelihoodSpec::poisson_log(),
            link_kind: None,
            covariance: None,
            link_wiggle: None,
        };
        let expected = point_predictor
            .predict_plugin_response(&input)
            .expect("canonical point prediction");
        let mut draw_predictor = StandardPredictor {
            beta: mode.clone(),
            family: LikelihoodSpec::poisson_log(),
            link_kind: None,
            covariance: None,
            link_wiggle: None,
        };
        let draws = mode.insert_axis(ndarray::Axis(0));
        let actual = evaluate_draw_matrices(
            draws.view(),
            input.design.nrows(),
            PredictModelClass::Standard,
            |draw| {
                draw_predictor.beta.assign(&draw);
                draw_predictor.predict_plugin_response(&input)
            },
        )
        .expect("posterior draw prediction");
        assert_eq!(actual.eta.row(0), expected.eta.view());
        assert_eq!(actual.mean.row(0), expected.mean.view());
    }

    #[test]
    fn named_offsets_are_resolved_without_copying_other_columns() {
        let data = array![[3.0, 10.0], [-2.0, 20.0]];
        let columns = HashMap::from([("exposure".to_string(), 0usize)]);
        let offset = resolve_offset(data.view(), &columns, Some("exposure"), "offset")
            .expect("named offset");
        assert_eq!(offset, array![3.0, -2.0]);
    }

    #[test]
    fn non_finite_draw_is_a_typed_error() {
        let draws = array![[0.0, f64::NAN]];
        assert!(matches!(
            validate_draw_values(draws.view()),
            Err(PosteriorPredictError::NonFiniteDraw {
                draw: 0,
                coefficient: 1,
                ..
            })
        ));
    }
}
