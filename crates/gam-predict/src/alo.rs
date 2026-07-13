use std::ops::Range;

use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_model_kernels::sigma_link::survival_q0_from_eta;
use gam_models::gamlss::{
    DispersionFamilyKind, binomial_location_scale_alo_row_geometry,
    dispersion_alo_row_geometry, gaussian_location_scale_alo_row_geometry,
};
use gam_models::inference::model::{
    FittedModel, PredictModelClass, binomial_location_scale_threshold_beta,
    gaussian_location_scale_mean_beta, location_scale_noise_beta,
};
use gam_problem::{EstimationError, ResponseFamily};
use gam_solve::inference::alo::{
    MultiBlockAloDiagnostics, MultiBlockAloInput, compute_multiblock_alo,
};
use gam_terms::basis::BasisOptions;
use ndarray::{Array1, Array2};

use crate::PredictInput;

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
    if observations.prior_weights.len() != n
        || input.design.nrows() != n
        || input.offset.len() != n
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

fn secondary_eta(
    input: &PredictInput,
    design: &DesignMatrix,
    beta: &Array1<f64>,
) -> Array1<f64> {
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
    let fit = model
        .payload()
        .fit_result
        .as_ref()
        .or_else(|| model.unified())
        .ok_or_else(|| {
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

fn compute_location_scale_core(
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
    let fit = model.payload().fit_result.as_ref().or_else(|| model.unified()).ok_or_else(|| {
        invalid("saved Gaussian location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved Gaussian location-scale ALO is missing the mean block"))?;
    let beta_scale = location_scale_noise_beta(fit).or_else(|| {
        model
            .payload()
            .beta_noise
            .clone()
            .map(Array1::from_vec)
    }).ok_or_else(|| invalid("saved Gaussian location-scale ALO is missing the log-scale block"))?;
    let runtime = model
        .saved_prediction_runtime()
        .map_err(|error| invalid(format!("saved Gaussian location-scale ALO runtime: {error}")))?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle.as_ref().map_or(&[][..], |runtime| runtime.beta.as_slice());
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
    let basis = basis.map_err(|error| invalid(format!("saved Gaussian ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved Gaussian ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved Gaussian ALO warp d2: {error}")))?;
    let response_scale = model.payload().gaussian_response_scale.unwrap_or(1.0);

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let geometry = gaussian_location_scale_alo_row_geometry(
            row,
            observations.response[row],
            base_mean[row],
            eta_scale[row],
            observations.prior_weights[row],
            response_scale,
            basis.row(row).as_slice().expect("basis row contiguous"),
            basis_d1
                .row(row)
                .as_slice()
                .expect("basis derivative row contiguous"),
            basis_d2
                .row(row)
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        )
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
    compute_location_scale_core(
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
    let fit = model.payload().fit_result.as_ref().or_else(|| model.unified()).ok_or_else(|| {
        invalid("saved binomial location-scale ALO requires a canonical fit result")
    })?;
    let beta_threshold = binomial_location_scale_threshold_beta(fit)
        .ok_or_else(|| invalid("saved binomial location-scale ALO is missing the threshold block"))?;
    let beta_scale = location_scale_noise_beta(fit).or_else(|| {
        model
            .payload()
            .beta_noise
            .clone()
            .map(Array1::from_vec)
    }).ok_or_else(|| invalid("saved binomial location-scale ALO is missing the log-scale block"))?;
    let runtime = model
        .saved_prediction_runtime()
        .map_err(|error| invalid(format!("saved binomial location-scale ALO runtime: {error}")))?;
    let inverse_link = runtime.inverse_link.ok_or_else(|| {
        invalid("saved binomial location-scale ALO is missing its resolved inverse link")
    })?;
    let wiggle = runtime.link_wiggle;
    let wiggle_beta = wiggle.as_ref().map_or(&[][..], |runtime| runtime.beta.as_slice());
    let parameter_dimension = beta_threshold.len() + beta_scale.len() + wiggle_beta.len();
    let hessian = require_saved_hessian(model, class, parameter_dimension)?;
    let threshold_eta = input.design.dot(&beta_threshold) + &input.offset;
    let eta_scale = secondary_eta(input, secondary_design, &beta_scale);
    let q0 = Array1::from_shape_fn(threshold_eta.len(), |row| {
        survival_q0_from_eta(threshold_eta[row], eta_scale[row])
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
    let basis = basis.map_err(|error| invalid(format!("saved binomial ALO warp basis: {error}")))?;
    let basis_d1 =
        basis_d1.map_err(|error| invalid(format!("saved binomial ALO warp d1: {error}")))?;
    let basis_d2 =
        basis_d2.map_err(|error| invalid(format!("saved binomial ALO warp d2: {error}")))?;

    let mut observed_hessians = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut coordinate_values = Vec::with_capacity(n);
    for row in 0..n {
        let geometry = binomial_location_scale_alo_row_geometry(
            observations.response[row],
            threshold_eta[row],
            eta_scale[row],
            observations.prior_weights[row],
            &inverse_link,
            basis.row(row).as_slice().expect("basis row contiguous"),
            basis_d1
                .row(row)
                .as_slice()
                .expect("basis derivative row contiguous"),
            basis_d2
                .row(row)
                .as_slice()
                .expect("basis second derivative row contiguous"),
            wiggle_beta,
        )
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
    compute_location_scale_core(
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
    let fit = model.payload().fit_result.as_ref().or_else(|| model.unified()).ok_or_else(|| {
        invalid("saved dispersion location-scale ALO requires a canonical fit result")
    })?;
    let beta_mean = gaussian_location_scale_mean_beta(fit)
        .ok_or_else(|| invalid("saved dispersion location-scale ALO is missing the mean block"))?;
    let beta_dispersion = location_scale_noise_beta(fit).or_else(|| {
        model
            .payload()
            .beta_noise
            .clone()
            .map(Array1::from_vec)
    }).ok_or_else(|| invalid("saved dispersion location-scale ALO is missing the precision block"))?;
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
    compute_location_scale_core(
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
