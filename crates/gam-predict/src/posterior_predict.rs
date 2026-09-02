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
use gam_solve::model_types::{FitGeometry, UnifiedFitResult};
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
    DrawOutsideCoefficientGauge {
        draw: usize,
        coefficient: usize,
        residual: f64,
        tolerance: f64,
    },
    InfeasibleDraw {
        draw: usize,
        constraint: usize,
        violation: f64,
        tolerance: f64,
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
            Self::DrawOutsideCoefficientGauge {
                draw,
                coefficient,
                residual,
                tolerance,
            } => write!(
                f,
                "posterior draw {draw} is outside the fitted coefficient gauge at raw coefficient \
                 {coefficient}: reconstruction residual {residual:.3e} exceeds the \
                 floating-point certificate {tolerance:.3e}",
            ),
            Self::InfeasibleDraw {
                draw,
                constraint,
                violation,
                tolerance,
            } => write!(
                f,
                "posterior draw {draw} violates fitted inequality row {constraint}: scaled \
                 violation {violation:.3e} exceeds {tolerance:.3e}",
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

fn validate_saved_constraint_feasibility(
    model: &FittedModel,
    draws: ArrayView2<'_, f64>,
) -> Result<(), PosteriorPredictError> {
    let model_class = model.predict_model_class();
    let Some(fit) = model.fit_result.as_ref() else {
        return Ok(());
    };
    // #2536: the geometry check below keys on `constrained_posterior`, so a fit
    // that certified a cone without persisting one passed VACUOUSLY. That state
    // is reachable and is not a malformed artifact: `gam-custom-family`'s
    // covariance assembly returns `constrained_posterior: None` for a genuinely
    // CONSTRAINED fit whose ambient precision is not positive definite (#2442's
    // typed decline), and records that a consumer cannot tell that apart from an
    // unconstrained fit. Draws for such a fit reach here and were admitted
    // without any cone being applied.
    //
    // A monotone link-wiggle block does not need the persisted identity to be
    // checked. Its cone is `A = I`, `b = 0`
    // (`monotone_wiggle_nonnegative_constraints`) for every class that carries
    // one, fixed by the block's role rather than read off the fit, so it is
    // enforced from the block layout directly. This is the DECLARATION-keyed
    // shape `laplace_fallback_route` uses on the sampling side, applied to the
    // consumer that takes draws it did not produce.
    if let Some(range) = block_range(fit, BlockRole::LinkWiggle) {
        validate_link_wiggle_cone(model_class, range, draws)?;
    }
    let Some(geometry) = fit.geometry.as_ref() else {
        return Ok(());
    };
    validate_constraint_geometry(model_class, geometry, draws)
}

/// Enforce the monotone link-wiggle cone `β_w ≥ 0` on supplied draws (#2536).
///
/// The wiggle block's coefficients multiply I-spline bases whose non-negative
/// combinations are exactly the monotone warps; a negative coefficient is a
/// non-monotone warp the fitted model cannot produce, so predicting from it
/// reports a curve outside the model. Both the law and the band are the fit's
/// own: the same `Aβ ≥ b` residual and the same
/// `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` that [`validate_constraint_geometry`]
/// applies to a persisted cone, reported through the same `InfeasibleDraw`
/// error, so the two routes cannot diverge on what "feasible" means.
///
/// `range` is the caller's `BlockRole::LinkWiggle` block range — the same range
/// every draw-evaluation arm in this module slices to fill the wiggle runtime.
fn validate_link_wiggle_cone(
    model_class: PredictModelClass,
    range: Range<usize>,
    draws: ArrayView2<'_, f64>,
) -> Result<(), PosteriorPredictError> {
    if range.end > draws.ncols() {
        return Err(inconsistent_state_error(
            model_class,
            format!(
                "the fitted monotone link-wiggle coefficient block spans {}..{}, but the supplied \
                 draws carry only {} coefficients, so the cone `β_w ≥ 0` the fit certified cannot \
                 be checked against them",
                range.start,
                range.end,
                draws.ncols(),
            ),
        ));
    }
    let tolerance = gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
    for (draw, row) in draws.rows().into_iter().enumerate() {
        for coefficient in range.clone() {
            let violation = (-row[coefficient]).max(0.0);
            if violation > tolerance {
                return Err(PosteriorPredictError::InfeasibleDraw {
                    draw,
                    constraint: coefficient - range.start,
                    violation,
                    tolerance,
                });
            }
        }
    }
    Ok(())
}

fn validate_constraint_geometry(
    model_class: PredictModelClass,
    geometry: &FitGeometry,
    draws: ArrayView2<'_, f64>,
) -> Result<(), PosteriorPredictError> {
    let Some(posterior) = geometry.constrained_posterior.as_ref() else {
        return Ok(());
    };
    geometry
        .validate_numeric_finiteness()
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?;
    let gauge = &geometry.coefficient_gauge;
    validate_coefficient_count(model_class, gauge.raw_total(), draws)?;

    // User-facing draws are in the saved/raw frame β = Tθ + a, while the
    // persisted cone lives in the active θ frame. Recover θ through the unique
    // least-squares inverse of the injective gauge. A strict SPD factorization
    // of TᵀT refuses a malformed/rank-deficient gauge; no pseudoinverse, ridge,
    // or guessed identity is allowed.
    let active = if gauge.is_identity() {
        draws.to_owned()
    } else {
        let mut centered = draws.to_owned();
        for mut draw in centered.rows_mut() {
            draw -= &gauge.affine_shift;
        }
        let gram = gauge.t_full.t().dot(&gauge.t_full);
        let factor = gam_linalg::utils::certified_spd_factorize(
            &gram,
            "posterior prediction coefficient-gauge inverse",
        )
        .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?;
        let rhs = gauge.t_full.t().dot(&centered.t());
        let (active_by_draw, _) = factor
            .solve_matrix(&rhs)
            .map_err(|error| inconsistent_state_error(model_class, error.to_string()))?;
        let active = active_by_draw.reversed_axes();

        // Solving the normal equations is only a coordinate candidate. Certify
        // that every supplied raw draw actually belongs to the affine section
        // by lifting it back and comparing in raw space. The band is an
        // operation-count-scaled f64 roundoff envelope, not model slack.
        let mut reconstructed = active.dot(&gauge.t_full.t());
        for mut draw in reconstructed.rows_mut() {
            draw += &gauge.affine_shift;
        }
        let operation_count =
            64.0 * (gauge.raw_total() + gauge.reduced_total() + 1) as f64 * f64::EPSILON;
        for ((draw, coefficient), &actual) in draws.indexed_iter() {
            let rebuilt = reconstructed[[draw, coefficient]];
            let residual = (rebuilt - actual).abs();
            let tolerance = operation_count
                * actual
                    .abs()
                    .max(rebuilt.abs())
                    .max(gauge.affine_shift[coefficient].abs())
                    .max(1.0);
            if residual > tolerance {
                return Err(PosteriorPredictError::DrawOutsideCoefficientGauge {
                    draw,
                    coefficient,
                    residual,
                    tolerance,
                });
            }
        }
        active
    };

    let constraints = posterior
        .constraints
        .canonicalized()
        .map_err(|reason| inconsistent_state_error(model_class, reason))?;
    let tolerance = gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
    for (draw, theta) in active.rows().into_iter().enumerate() {
        for constraint in 0..constraints.a.nrows() {
            let slack = constraints.a.row(constraint).dot(&theta) - constraints.b[constraint];
            let violation = (-slack).max(0.0);
            if violation > tolerance {
                return Err(PosteriorPredictError::InfeasibleDraw {
                    draw,
                    constraint,
                    violation,
                    tolerance,
                });
            }
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
    validate_saved_constraint_feasibility(model, draws)?;
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
    use gam_problem::LinearInequalityConstraints;
    use gam_solve::constrained_posterior::ConstrainedPosteriorGeometry;
    use gam_spec::LikelihoodSpec;
    use ndarray::array;

    use super::*;

    fn constrained_test_geometry() -> FitGeometry {
        FitGeometry {
            coefficient_gauge: gam_problem::Gauge::from_block_transform_with_shift(
                array![[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
                array![0.5, -1.0, 3.0],
            ),
            penalized_hessian: Array2::<f64>::eye(2).into(),
            constrained_posterior: Some(ConstrainedPosteriorGeometry::with_moments(
                LinearInequalityConstraints {
                    a: Array2::<f64>::eye(2),
                    b: Array1::<f64>::zeros(2),
                },
                array![0.0, 0.0],
                array![-1.0, -1.0],
                None,
            )),
            working: None,
        }
    }

    #[test]
    fn posterior_prediction_rejects_draws_outside_the_saved_cone_or_gauge() {
        let geometry = constrained_test_geometry();
        validate_constraint_geometry(
            PredictModelClass::Standard,
            &geometry,
            array![[1.5, 3.0, 2.0]].view(),
        )
        .expect("theta=[1,2] is in the affine gauge and nonnegative cone");

        let infeasible = validate_constraint_geometry(
            PredictModelClass::Standard,
            &geometry,
            array![[0.25, 3.0, 0.75]].view(),
        )
        .expect_err("theta=[-0.25,2] violates the first cone row");
        assert!(matches!(
            infeasible,
            PosteriorPredictError::InfeasibleDraw {
                draw: 0,
                constraint: 0,
                ..
            }
        ));

        let outside = validate_constraint_geometry(
            PredictModelClass::Standard,
            &geometry,
            array![[1.5, 3.0, 2.01]].view(),
        )
        .expect_err("the perturbed raw point is outside beta=T theta+a");
        assert!(matches!(
            outside,
            PosteriorPredictError::DrawOutsideCoefficientGauge { draw: 0, .. }
        ));
    }

    /// #2536: a monotone link-wiggle block's cone `β_w ≥ 0` is structural, so it
    /// is enforced on supplied draws whether or not the fit persisted a
    /// `constrained_posterior`. Before this the whole feasibility gate returned
    /// early on `constrained_posterior: None` — which #2442's typed decline
    /// produces for a genuinely constrained fit — and admitted negative wiggle
    /// coefficients, i.e. non-monotone warps, as feasible.
    #[test]
    fn the_monotone_wiggle_cone_is_enforced_without_a_persisted_posterior() {
        // Coefficients 1..3 are the wiggle block; coefficient 0 is a mean
        // coefficient and is unconstrained, which is what keeps the check from
        // being a blanket non-negativity rule on the whole draw.
        let feasible = array![[-5.0, 0.0, 2.0], [-1.0, 0.75, 0.0]];
        validate_link_wiggle_cone(
            PredictModelClass::GaussianLocationScale,
            1..3,
            feasible.view(),
        )
        .expect("non-negative wiggle coefficients are inside the cone the fit certified");

        let infeasible = array![[-5.0, 0.0, 2.0], [-1.0, 0.75, -0.25]];
        let error = validate_link_wiggle_cone(
            PredictModelClass::GaussianLocationScale,
            1..3,
            infeasible.view(),
        )
        .expect_err("a negative wiggle coefficient is a non-monotone warp");
        match error {
            PosteriorPredictError::InfeasibleDraw {
                draw,
                constraint,
                violation,
                ..
            } => {
                assert_eq!(draw, 1, "the refusal must name the offending draw");
                assert_eq!(
                    constraint, 1,
                    "the constraint index is the coefficient's position WITHIN the wiggle \
                     block, matching the `A = I` cone row it violates"
                );
                assert!((violation - 0.25).abs() < 1.0e-12, "violation {violation}");
            }
            other => panic!("expected an InfeasibleDraw refusal, got {other:?}"),
        }

        // The band is the fit's own primal-feasibility tolerance, not a fresh
        // constant: a draw inside it is a converged active coordinate, not a
        // violation.
        let rounding = 0.5 * gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
        let at_the_wall = array![[0.0, -rounding, 1.0]];
        validate_link_wiggle_cone(
            PredictModelClass::GaussianLocationScale,
            1..3,
            at_the_wall.view(),
        )
        .expect("a coordinate at its wall within the active-set band is feasible");

        // A draw narrower than the block cannot be checked against the cone, and
        // silently skipping it is the vacuous pass this gate exists to remove.
        let narrow = array![[0.0, 1.0]];
        let short = validate_link_wiggle_cone(
            PredictModelClass::GaussianLocationScale,
            1..3,
            narrow.view(),
        )
        .expect_err("draws that do not span the wiggle block cannot be certified");
        assert!(matches!(
            short,
            PosteriorPredictError::InconsistentModelState { .. }
        ));
    }

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
