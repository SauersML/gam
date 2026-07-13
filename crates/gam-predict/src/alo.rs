use std::ops::Range;

use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_models::bms::{
    BernoulliMarginalSlopeAloRowInput, bernoulli_marginal_slope_alo_row_geometry,
};
use gam_models::gamlss::{
    BinomialLocationScaleAloRowInput, DispersionFamilyKind, GaussianLocationScaleAloRowInput,
    binomial_location_scale_alo_row_geometry, dispersion_alo_row_geometry,
    gaussian_location_scale_alo_row_geometry,
};
use gam_models::inference::model::{
    FittedModel, PredictModelClass, binomial_location_scale_threshold_beta,
    gaussian_location_scale_mean_beta, location_scale_noise_beta,
};
use gam_models::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalAloRowInput,
    transformation_normal_alo_row_geometry,
};
use gam_problem::{EstimationError, ResponseFamily};
use gam_solve::inference::alo::{
    MultiBlockAloDiagnostics, MultiBlockAloInput, compute_multiblock_alo,
};
use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};
use ndarray::{Array1, Array2, s};

use crate::{FittedModelPredictExt, PredictInput};

/// Observed row data needed to replay a saved likelihood for ALO.
pub struct SavedAloObservations<'a> {
    pub response: &'a Array1<f64>,
    pub prior_weights: &'a Array1<f64>,
}

/// Class-neutral diagnostics returned by saved-model multi-coordinate ALO.
#[derive(Debug, Clone)]
pub struct SavedModelAloDiagnostics {
    pub model_class: PredictModelClass,
    /// Names of the affine local coordinates represented by each entry of
    /// `diagnostics.eta_tilde` / `diagnostics.alo_variance`.
    pub coordinate_names: Vec<String>,
    pub diagnostics: MultiBlockAloDiagnostics,
}

fn invalid(reason: impl Into<String>) -> EstimationError {
    EstimationError::InvalidInput(reason.into())
}

fn require_location_scale_inputs<'a>(
    class: PredictModelClass,
    input: &'a PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<&'a DesignMatrix, EstimationError> {
    let n = observations.response.len();
    if n == 0 {
        return Err(invalid(format!(
            "saved {} ALO requires at least one observation",
            class.name()
        )));
    }
    if observations.prior_weights.len() != n || input.design.nrows() != n || input.offset.len() != n
    {
        return Err(invalid(format!(
            "saved {} ALO row mismatch: response={n}, weights={}, primary_design={}, primary_offset={}",
            class.name(),
            observations.prior_weights.len(),
            input.design.nrows(),
            input.offset.len(),
        )));
    }
    if let Some((row, weight)) = observations
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid(format!(
            "saved {} ALO prior weight[{row}] must be finite and non-negative, got {weight}",
            class.name()
        )));
    }
    let noise_design = input.design_noise.as_ref().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires the persisted secondary design",
            class.name()
        ))
    })?;
    if noise_design.nrows() != n {
        return Err(invalid(format!(
            "saved {} ALO secondary design has {} rows; expected {n}",
            class.name(),
            noise_design.nrows()
        )));
    }
    if let Some(offset) = input.offset_noise.as_ref()
        && offset.len() != n
    {
        return Err(invalid(format!(
            "saved {} ALO secondary offset has {} rows; expected {n}",
            class.name(),
            offset.len()
        )));
    }
    Ok(noise_design)
}

fn secondary_eta(input: &PredictInput, design: &DesignMatrix, beta: &Array1<f64>) -> Array1<f64> {
    let mut eta = design.dot(beta);
    if let Some(offset) = input.offset_noise.as_ref() {
        eta += offset;
    }
    eta
}

fn score_outer_product(score: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((score.len(), score.len()), |(row, column)| {
        score[row] * score[column]
    })
}

fn constant_scalar_design(n: usize) -> DesignMatrix {
    DesignMatrix::Dense(DenseDesignMatrix::from(Array2::ones((n, 1))))
}

fn location_scale_coordinate_layout(
    primary_design: &DesignMatrix,
    secondary_design: &DesignMatrix,
    primary_dimension: usize,
    secondary_dimension: usize,
    wiggle_dimension: usize,
    primary_name: &str,
    secondary_name: &str,
) -> (Vec<DesignMatrix>, Vec<Range<usize>>, Vec<String>) {
    let n = primary_design.nrows();
    let mut designs = Vec::with_capacity(2 + wiggle_dimension);
    let mut ranges = Vec::with_capacity(2 + wiggle_dimension);
    let mut names = Vec::with_capacity(2 + wiggle_dimension);
    designs.push(primary_design.clone());
    ranges.push(0..primary_dimension);
    names.push(primary_name.to_string());
    designs.push(secondary_design.clone());
    ranges.push(primary_dimension..primary_dimension + secondary_dimension);
    names.push(secondary_name.to_string());
    for coordinate in 0..wiggle_dimension {
        let coefficient = primary_dimension + secondary_dimension + coordinate;
        designs.push(constant_scalar_design(n));
        ranges.push(coefficient..coefficient + 1);
        names.push(format!("link-wiggle[{coordinate}]"));
    }
    (designs, ranges, names)
}

fn require_saved_hessian<'a>(
    model: &'a FittedModel,
    class: PredictModelClass,
    parameter_dimension: usize,
) -> Result<&'a Array2<f64>, EstimationError> {
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires a canonical fitted coefficient state",
            class.name()
        ))
    })?;
    let hessian = fit.penalized_hessian().ok_or_else(|| {
        invalid(format!(
            "saved {} ALO requires the exact unscaled penalized Hessian",
            class.name()
        ))
    })?;
    if hessian.dim() != (parameter_dimension, parameter_dimension) {
        return Err(invalid(format!(
            "saved {} ALO precision is {}x{}; parameter layout requires {parameter_dimension}x{parameter_dimension}",
            class.name(),
            hessian.nrows(),
            hessian.ncols(),
        )));
    }
    Ok(hessian)
}

fn compute_saved_multicoordinate_core(
    class: PredictModelClass,
    coordinate_names: Vec<String>,
    coordinate_designs: Vec<DesignMatrix>,
    coordinate_ranges: Vec<Range<usize>>,
    penalized_hessian: &Array2<f64>,
    observed_hessians: Vec<Array2<f64>>,
    scores: Vec<Array1<f64>>,
    coordinate_values: Vec<Array1<f64>>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let score_covariances = scores.iter().map(score_outer_product).collect::<Vec<_>>();
    let diagnostics = compute_multiblock_alo(&MultiBlockAloInput {
        n_obs: scores.len(),
        n_coordinates: coordinate_names.len(),
        coordinate_designs: &coordinate_designs,
        coordinate_coefficient_ranges: &coordinate_ranges,
        penalized_hessian,
        observed_hessians: &observed_hessians,
        score_covariances: &score_covariances,
        scores: &scores,
        coordinate_values: &coordinate_values,
    })?;
    Ok(SavedModelAloDiagnostics {
        model_class: class,
        coordinate_names,
        diagnostics,
    })
}

fn compute_gaussian_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::GaussianLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved Gaussian location-scale ALO is missing the mean block"))?;
    let beta_scale = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO is missing the log-scale block")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved Gaussian location-scale ALO runtime: {error}"
        ))
    })?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle
        .as_ref()
        .map_or(&[][..], |runtime| runtime.beta.as_slice());
    let parameter_dimension = beta_mean.len() + beta_scale.len() + wiggle_beta.len();
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let base_mean = input.design.dot(&beta_mean) + &input.offset;
    let eta_scale = secondary_eta(input, secondary_design, &beta_scale);
    let n = observations.response.len();
    let (basis, basis_d1, basis_d2) = match wiggle.as_ref() {
        Some(runtime) => {
            runtime.derivative_q0(&base_mean).map_err(|error| {
                invalid(format!("saved Gaussian location-scale ALO warp: {error}"))
            })?;
            (
                runtime.constrained_basis(&base_mean, BasisOptions::value()),
                runtime.constrained_basis(&base_mean, BasisOptions::first_derivative()),
                runtime.constrained_basis(&base_mean, BasisOptions::second_derivative()),
            )
        }
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let basis =
        basis.map_err(|error| invalid(format!("saved Gaussian ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved Gaussian ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved Gaussian ALO warp d2: {error}")))?;
    let response_scale = model.payload().gaussian_response_scale.ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO is missing its response standardization scale")
    })?;

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let basis_row = basis.row(row);
        let basis_d1_row = basis_d1.row(row);
        let basis_d2_row = basis_d2.row(row);
        let geometry = gaussian_location_scale_alo_row_geometry(GaussianLocationScaleAloRowInput {
            row,
            y: observations.response[row],
            base_mean: base_mean[row],
            eta_log_sigma: eta_scale[row],
            prior_weight: observations.prior_weights[row],
            response_scale,
            wiggle_basis: basis_row.as_slice().expect("basis row contiguous"),
            wiggle_basis_d1: basis_d1_row
                .as_slice()
                .expect("basis derivative row contiguous"),
            wiggle_basis_d2: basis_d2_row
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        })
        .map_err(|reason| invalid(format!("saved Gaussian ALO row {row}: {reason}")))?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        let mut values = Array1::<f64>::zeros(2 + wiggle_beta.len());
        values[0] = base_mean[row];
        values[1] = eta_scale[row];
        for (coordinate, &value) in wiggle_beta.iter().enumerate() {
            values[2 + coordinate] = value;
        }
        coordinate_values.push(values);
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_mean.len(),
        beta_scale.len(),
        wiggle_beta.len(),
        "mean-base",
        "log-sigma",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        hessian,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn compute_binomial_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::BinomialLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved binomial location-scale ALO requires a canonical fit result")
    })?;
    let beta_threshold = binomial_location_scale_threshold_beta(fit).ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing the threshold block")
    })?;
    let beta_scale = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing the log-scale block")
    })?;
    let runtime = model.saved_prediction_runtime().map_err(|error| {
        invalid(format!(
            "saved binomial location-scale ALO runtime: {error}"
        ))
    })?;
    let inverse_link = runtime.inverse_link.ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing its resolved inverse link")
    })?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle
        .as_ref()
        .map_or(&[][..], |runtime| runtime.beta.as_slice());
    let parameter_dimension = beta_threshold.len() + beta_scale.len() + wiggle_beta.len();
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let threshold_eta = input.design.dot(&beta_threshold) + &input.offset;
    let eta_scale = secondary_eta(input, secondary_design, &beta_scale);
    // Replay the fitted binomial map exactly. This deliberately does not use
    // the survival prediction saturation helper: a non-representable q0 is a
    // factual saved-row geometry failure here, not a value to replace before
    // evaluating the fitted warp basis.
    let q0 = Array1::from_shape_fn(threshold_eta.len(), |row| {
        -threshold_eta[row] * (-eta_scale[row]).exp()
    });
    let n = observations.response.len();
    let (basis, basis_d1, basis_d2) = match wiggle.as_ref() {
        Some(runtime) => {
            runtime.derivative_q0(&q0).map_err(|error| {
                invalid(format!("saved binomial location-scale ALO warp: {error}"))
            })?;
            (
                runtime.constrained_basis(&q0, BasisOptions::value()),
                runtime.constrained_basis(&q0, BasisOptions::first_derivative()),
                runtime.constrained_basis(&q0, BasisOptions::second_derivative()),
            )
        }
        None => (
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
            Ok(Array2::zeros((n, 0))),
        ),
    };
    let basis =
        basis.map_err(|error| invalid(format!("saved binomial ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved binomial ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved binomial ALO warp d2: {error}")))?;

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let basis_row = basis.row(row);
        let basis_d1_row = basis_d1.row(row);
        let basis_d2_row = basis_d2.row(row);
        let geometry = binomial_location_scale_alo_row_geometry(BinomialLocationScaleAloRowInput {
            y: observations.response[row],
            threshold_eta: threshold_eta[row],
            eta_log_sigma: eta_scale[row],
            prior_weight: observations.prior_weights[row],
            inverse_link: &inverse_link,
            wiggle_basis: basis_row.as_slice().expect("basis row contiguous"),
            wiggle_basis_d1: basis_d1_row
                .as_slice()
                .expect("basis derivative row contiguous"),
            wiggle_basis_d2: basis_d2_row
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        })
        .map_err(|reason| invalid(format!("saved binomial ALO row {row}: {reason}")))?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        let mut values = Array1::<f64>::zeros(2 + wiggle_beta.len());
        values[0] = threshold_eta[row];
        values[1] = eta_scale[row];
        for (coordinate, &value) in wiggle_beta.iter().enumerate() {
            values[2 + coordinate] = value;
        }
        coordinate_values.push(values);
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_threshold.len(),
        beta_scale.len(),
        wiggle_beta.len(),
        "threshold",
        "log-sigma",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        hessian,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

fn dispersion_kind(response: &ResponseFamily) -> Result<DispersionFamilyKind, EstimationError> {
    match response {
        ResponseFamily::NegativeBinomial { .. } => Ok(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Ok(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Ok(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Ok(DispersionFamilyKind::Tweedie { p: *p }),
        response => Err(invalid(format!(
            "saved dispersion location-scale ALO cannot replay response family {response:?}"
        ))),
    }
}

fn compute_dispersion_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: &SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::DispersionLocationScale;
    let secondary_design = require_location_scale_inputs(class, input, observations)?;
    let fit = model.payload().fit_result.as_ref().ok_or_else(|| {
        invalid("saved dispersion location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved dispersion location-scale ALO is missing the mean block"))?;
    let beta_dispersion = location_scale_noise_beta(fit).ok_or_else(|| {
        invalid("saved dispersion location-scale ALO is missing the precision block")
    })?;
    let parameter_dimension = beta_mean.len() + beta_dispersion.len();
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let eta_mean = input.design.dot(&beta_mean) + &input.offset;
    let eta_dispersion = secondary_eta(input, secondary_design, &beta_dispersion);
    let kind = dispersion_kind(&model.payload().family_state.likelihood().response)?;
    let n = observations.response.len();
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let geometry = dispersion_alo_row_geometry(
            kind,
            row,
            observations.response[row],
            eta_mean[row],
            eta_dispersion[row],
            observations.prior_weights[row],
        )
        .map_err(|reason| invalid(format!("saved dispersion ALO row {row}: {reason}")))?;
        let score = Array1::from_vec(geometry.nll_score.to_vec());
        let hessian = Array2::from_shape_vec(
            (2, 2),
            geometry.observed_hessian.into_iter().flatten().collect(),
        )
        .expect("fixed 2x2 dispersion geometry");
        observed_hessians.push(hessian);
        scores.push(score);
        coordinate_values.push(Array1::from_vec(vec![eta_mean[row], eta_dispersion[row]]));
    }
    let (designs, ranges, names) = location_scale_coordinate_layout(
        &input.design,
        secondary_design,
        beta_mean.len(),
        beta_dispersion.len(),
        0,
        "mean",
        "log-precision",
    );
    compute_saved_multicoordinate_core(
        class,
        names,
        designs,
        ranges,
        hessian,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

/// Dispatch exact saved-H ALO for every location-scale model class.
///
/// This path never refits and never substitutes another model. The saved
/// penalized Hessian, row likelihood and coefficient topology must all be
/// present and dimensionally aligned or the request fails factually.
pub fn compute_saved_location_scale_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    match model.predict_model_class() {
        PredictModelClass::GaussianLocationScale => {
            compute_gaussian_location_scale_alo(model, input, &observations)
        }
        PredictModelClass::BinomialLocationScale => {
            compute_binomial_location_scale_alo(model, input, &observations)
        }
        PredictModelClass::DispersionLocationScale => {
            compute_dispersion_location_scale_alo(model, input, &observations)
        }
        class => Err(invalid(format!(
            "saved location-scale ALO dispatcher received model class {}",
            class.name()
        ))),
    }
}

/// Replay exact saved-H ALO for a rigid Bernoulli marginal-slope fit.
pub fn compute_saved_bernoulli_marginal_slope_alo(
    model: &FittedModel,
    input: &PredictInput,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::BernoulliMarginalSlope;
    if model.predict_model_class() != class {
        return Err(invalid(format!(
            "saved marginal-slope ALO dispatcher received model class {}",
            model.predict_model_class().name()
        )));
    }
    let n = observations.response.len();
    if n == 0
        || observations.prior_weights.len() != n
        || input.design.nrows() != n
        || input
            .design_noise
            .as_ref()
            .is_none_or(|design| design.nrows() != n)
    {
        return Err(invalid(format!(
            "saved marginal-slope ALO row mismatch: response={n}, weights={}, marginal_design={}, slope_design={}",
            observations.prior_weights.len(),
            input.design.nrows(),
            input.design_noise.as_ref().map_or(0, DesignMatrix::nrows),
        )));
    }
    if let Some((row, weight)) = observations
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid(format!(
            "saved marginal-slope ALO prior weight[{row}] must be finite and non-negative, got {weight}"
        )));
    }
    let predictor = model
        .bernoulli_marginal_slope_predictor()
        .map_err(|reason| invalid(format!("saved marginal-slope ALO predictor: {reason}")))?;
    let state = predictor.rigid_saved_alo_state(input)?;
    let parameter_dimension = predictor.beta_marginal.len() + predictor.beta_logslope.len();
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let geometry =
            bernoulli_marginal_slope_alo_row_geometry(BernoulliMarginalSlopeAloRowInput {
                base_link: &predictor.base_link,
                marginal_eta: state.marginal_eta[row],
                slope: state.slope[row],
                latent_z: state.latent_z[row],
                response: observations.response[row],
                prior_weight: observations.prior_weights[row],
                probit_frailty_scale: state.probit_frailty_scale,
            })
            .map_err(|reason| invalid(format!("saved marginal-slope ALO row {row}: {reason}")))?;
        scores.push(Array1::from_vec(geometry.nll_score.to_vec()));
        observed_hessians.push(
            Array2::from_shape_vec(
                (2, 2),
                geometry.observed_hessian.into_iter().flatten().collect(),
            )
            .expect("fixed 2x2 marginal-slope geometry"),
        );
        coordinate_values.push(Array1::from_vec(vec![
            state.marginal_eta[row],
            state.slope[row],
        ]));
    }
    let secondary_design = input
        .design_noise
        .as_ref()
        .expect("validated marginal-slope design");
    let coordinate_designs = vec![input.design.clone(), secondary_design.clone()];
    let coordinate_ranges = vec![
        0..predictor.beta_marginal.len(),
        predictor.beta_marginal.len()..parameter_dimension,
    ];
    compute_saved_multicoordinate_core(
        class,
        vec!["marginal-eta".to_string(), "slope".to_string()],
        coordinate_designs,
        coordinate_ranges,
        hessian,
        observed_hessians,
        scores,
        coordinate_values,
    )
}

/// Replay exact saved-H ALO for a finite-support transformation-normal fit.
pub fn compute_saved_transformation_normal_alo(
    model: &FittedModel,
    covariate_design: &DesignMatrix,
    additive_offset: &Array1<f64>,
    observations: SavedAloObservations<'_>,
) -> Result<SavedModelAloDiagnostics, EstimationError> {
    let class = PredictModelClass::TransformationNormal;
    if model.predict_model_class() != class {
        return Err(invalid(format!(
            "saved transformation-normal ALO dispatcher received model class {}",
            model.predict_model_class().name()
        )));
    }
    let n = observations.response.len();
    if n == 0
        || observations.prior_weights.len() != n
        || covariate_design.nrows() != n
        || additive_offset.len() != n
    {
        return Err(invalid(format!(
            "saved transformation-normal ALO row mismatch: response={n}, weights={}, design={}, offset={}",
            observations.prior_weights.len(),
            covariate_design.nrows(),
            additive_offset.len(),
        )));
    }
    let payload = model.payload();
    let knots = payload
        .transformation_response_knots
        .as_ref()
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing response knots"))?;
    let transform_rows = payload
        .transformation_response_transform
        .as_ref()
        .ok_or_else(|| {
            invalid("saved transformation-normal ALO is missing its response transform")
        })?;
    let degree = payload
        .transformation_response_degree
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing its response degree"))?;
    let response_median = payload
        .transformation_response_median
        .ok_or_else(|| invalid("saved transformation-normal ALO is missing its response median"))?;
    if knots.is_empty() || transform_rows.is_empty() {
        return Err(invalid(
            "saved transformation-normal ALO response basis metadata is empty",
        ));
    }
    let transform_columns = transform_rows[0].len();
    if transform_columns == 0
        || transform_rows
            .iter()
            .any(|row| row.len() != transform_columns)
    {
        return Err(invalid(
            "saved transformation-normal ALO response transform is empty or ragged",
        ));
    }
    let mut transform = Array2::<f64>::zeros((transform_rows.len(), transform_columns));
    for (row_index, row) in transform_rows.iter().enumerate() {
        for (column_index, &value) in row.iter().enumerate() {
            transform[[row_index, column_index]] = value;
        }
    }
    let knots = Array1::from_vec(knots.clone());
    let (raw_value_basis, _) = create_basis::<Dense>(
        observations.response.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| {
        invalid(format!(
            "saved transformation-normal ALO value basis: {error}"
        ))
    })?;
    let raw_value_basis = raw_value_basis.as_ref();
    let raw_derivative_basis =
        create_ispline_derivative_dense(observations.response.view(), &knots, degree, 1).map_err(
            |error| {
                invalid(format!(
                    "saved transformation-normal ALO derivative basis: {error}"
                ))
            },
        )?;
    if raw_value_basis.ncols() != transform.nrows()
        || raw_derivative_basis.dim() != raw_value_basis.dim()
    {
        return Err(invalid(format!(
            "saved transformation-normal ALO raw basis/transform mismatch: value={}x{}, derivative={}x{}, transform={}x{}",
            raw_value_basis.nrows(),
            raw_value_basis.ncols(),
            raw_derivative_basis.nrows(),
            raw_derivative_basis.ncols(),
            transform.nrows(),
            transform.ncols(),
        )));
    }
    let shape_value = raw_value_basis.dot(&transform);
    let shape_derivative = raw_derivative_basis.dot(&transform);
    let response_dimension = transform_columns + 1;
    let mut response_value_basis = Array2::<f64>::zeros((n, response_dimension));
    response_value_basis.column_mut(0).fill(1.0);
    response_value_basis
        .slice_mut(s![.., 1..])
        .assign(&shape_value);
    let mut response_derivative_basis = Array2::<f64>::zeros((n, response_dimension));
    response_derivative_basis
        .slice_mut(s![.., 1..])
        .assign(&shape_derivative);

    let lower_response = knots[0];
    let upper_response = knots[knots.len() - 1];
    if !(upper_response > lower_response) {
        return Err(invalid(format!(
            "saved transformation-normal ALO support is degenerate: lower={lower_response}, upper={upper_response}"
        )));
    }
    let endpoints = Array1::from_vec(vec![lower_response, upper_response]);
    let (raw_endpoint_basis, _) = create_basis::<Dense>(
        endpoints.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .map_err(|error| {
        invalid(format!(
            "saved transformation-normal ALO endpoints: {error}"
        ))
    })?;
    if raw_endpoint_basis.ncols() != transform.nrows() {
        return Err(invalid(format!(
            "saved transformation-normal ALO endpoint basis has {} columns; transform requires {}",
            raw_endpoint_basis.ncols(),
            transform.nrows(),
        )));
    }
    let endpoint_shape = raw_endpoint_basis.as_ref().dot(&transform);
    let mut lower_basis = Array1::<f64>::zeros(response_dimension);
    let mut upper_basis = Array1::<f64>::zeros(response_dimension);
    lower_basis[0] = 1.0;
    upper_basis[0] = 1.0;
    lower_basis
        .slice_mut(s![1..])
        .assign(&endpoint_shape.row(0));
    upper_basis
        .slice_mut(s![1..])
        .assign(&endpoint_shape.row(1));

    let fit = payload.fit_result.as_ref().ok_or_else(|| {
        invalid("saved transformation-normal ALO requires a canonical fit result")
    })?;
    if fit.blocks.len() != 1 {
        return Err(invalid(format!(
            "saved transformation-normal ALO requires one coefficient block, got {}",
            fit.blocks.len()
        )));
    }
    let beta = &fit.blocks[0].beta;
    let covariate_dimension = covariate_design.ncols();
    let parameter_dimension = response_dimension * covariate_dimension;
    if beta.len() != parameter_dimension {
        return Err(invalid(format!(
            "saved transformation-normal ALO beta has {} entries; response/covariate layout requires {parameter_dimension}",
            beta.len()
        )));
    }
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let mut gamma = Array2::<f64>::zeros((n, response_dimension));
    for component in 0..response_dimension {
        let start = component * covariate_dimension;
        let beta_component = beta
            .slice(s![start..start + covariate_dimension])
            .to_owned();
        gamma
            .column_mut(component)
            .assign(&covariate_design.dot(&beta_component));
    }

    let lower_floor = TRANSFORMATION_MONOTONICITY_EPS * (lower_response - response_median);
    let upper_floor = TRANSFORMATION_MONOTONICITY_EPS * (upper_response - response_median);
    let lower_basis_slice = lower_basis
        .as_slice()
        .expect("owned lower basis contiguous");
    let upper_basis_slice = upper_basis
        .as_slice()
        .expect("owned upper basis contiguous");
    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let response_value_row = response_value_basis.row(row);
        let response_derivative_row = response_derivative_basis.row(row);
        let gamma_row = gamma.row(row);
        let geometry = transformation_normal_alo_row_geometry(TransformationNormalAloRowInput {
            response_value_basis: response_value_row
                .as_slice()
                .expect("response value row contiguous"),
            response_derivative_basis: response_derivative_row
                .as_slice()
                .expect("response derivative row contiguous"),
            response_lower_basis: lower_basis_slice,
            response_upper_basis: upper_basis_slice,
            gamma: gamma_row.as_slice().expect("gamma row contiguous"),
            additive_offset: additive_offset[row],
            response_floor_offset: TRANSFORMATION_MONOTONICITY_EPS
                * (observations.response[row] - response_median),
            response_lower_floor_offset: lower_floor,
            response_upper_floor_offset: upper_floor,
            prior_weight: observations.prior_weights[row],
        })
        .map_err(|reason| {
            invalid(format!(
                "saved transformation-normal ALO row {row}: {reason}"
            ))
        })?;
        observed_hessians.push(geometry.observed_hessian);
        scores.push(geometry.nll_score);
        coordinate_values.push(gamma_row.to_owned());
    }
    let coordinate_designs = (0..response_dimension)
        .map(|_| covariate_design.clone())
        .collect::<Vec<_>>();
    let coordinate_ranges = (0..response_dimension)
        .map(|component| {
            let start = component * covariate_dimension;
            start..start + covariate_dimension
        })
        .collect::<Vec<_>>();
    let coordinate_names = (0..response_dimension)
        .map(|component| {
            if component == 0 {
                "location-gamma".to_string()
            } else {
                format!("shape-gamma[{component}]")
            }
        })
        .collect::<Vec<_>>();
    compute_saved_multicoordinate_core(
        class,
        coordinate_names,
        coordinate_designs,
        coordinate_ranges,
        hessian,
        observed_hessians,
        scores,
        coordinate_values,
    )
}
