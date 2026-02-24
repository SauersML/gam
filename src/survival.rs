use crate::basis::{
    BasisError, BasisOptions, Dense, KnotSource, SplineScratch, baseline_lambda_seed, create_basis,
    create_difference_penalty_matrix, evaluate_bspline_basis_scalar, null_range_whiten,
};
use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerSvd, ldlt_rook};
use crate::pirls::{WorkingModel as PirlsWorkingModel, WorkingState};
use crate::types::{Coefficients, LinearPredictor};
use log::warn;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Ix1, Zip, concatenate};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

const DEFAULT_DERIVATIVE_GUARD: f64 = 1e-8;
pub const DEFAULT_RISK_EPSILON: f64 = 1e-12;
const COMPANION_HORIZON_TOLERANCE: f64 = 1e-8;
// Hard monotonicity constraint: cumulative hazard must be non-decreasing.
// Any negative slope is ontologically invalid (survival probability cannot increase).
const MONOTONICITY_TOLERANCE: f64 = 0.0;
const DERIVATIVE_GUARD_WARNING_CEILING: f64 = 0.05;

/// Errors surfaced while validating survival data structures or evaluating the model.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("age vectors must have at least one element")]
    EmptyAgeVector,
    #[error("age values must be finite")]
    NonFiniteAge,
    #[error("age transform guard delta must be positive")]
    NonPositiveGuard,
    #[error("age {age} is outside the guarded log-age domain (a_min={minimum}, delta={delta})")]
    GuardDomainViolation { age: f64, minimum: f64, delta: f64 },
    #[error("age_entry must be strictly less than age_exit for every subject")]
    InvalidAgeOrder,
    #[error("event indicators must be 0 or 1")]
    InvalidEventFlag,
    #[error("event_target and event_competing indicators must be mutually exclusive")]
    ConflictingEvents,
    #[error("sample weights must be finite and non-negative")]
    InvalidSampleWeight,
    #[error("covariate arrays must have consistent dimensions")]
    CovariateDimensionMismatch,
    #[error("persisted static covariate ranges are missing for {expected} columns")]
    MissingCovariateRanges { expected: usize },
    #[error("persisted static covariate ranges size mismatch (expected {expected}, got {actual})")]
    CovariateRangeLengthMismatch { expected: usize, actual: usize },
    #[error(
        "covariate `{column}` (index {index}) has invalid persisted range: min {min}, max {max}"
    )]
    InvalidCovariateRange {
        column: String,
        index: usize,
        min: f64,
        max: f64,
    },
    #[error(
        "covariate `{column}` (index {index}) = {value} is below the persisted minimum {minimum}"
    )]
    CovariateBelowRange {
        column: String,
        index: usize,
        value: f64,
        minimum: f64,
    },
    #[error(
        "covariate `{column}` (index {index}) = {value} exceeds the persisted maximum {maximum}"
    )]
    CovariateAboveRange {
        column: String,
        index: usize,
        value: f64,
        maximum: f64,
    },
    #[error("coefficient length mismatch (expected {expected}, got {actual})")]
    CoefficientDimensionMismatch { expected: usize, actual: usize },
    #[error("covariate values must be finite")]
    NonFiniteCovariate,
    #[error("linear predictor became non-finite during evaluation")]
    NonFiniteLinearPredictor,
    #[error("design matrix columns do not match coefficient length")]
    DesignDimensionMismatch,
    #[error("stored Hessian factor dimensions do not match the design matrix")]
    HessianDimensionMismatch,
    #[error("stored Hessian factor is singular")]
    HessianSingular,
    #[error("competing-risk CIF must be supplied directly or through a companion model")]
    MissingCompanionCifData,
    #[error("competing-risk CIF value must be finite and lie in [0, 1], received {value}")]
    InvalidCompetingCif { value: f64 },
    #[error("companion model handle '{reference}' is not registered with the survival artifacts")]
    UnknownCompanionModelHandle { reference: String },
    #[error("companion model '{reference}' is unavailable during prediction")]
    CompanionModelUnavailable { reference: String },
    #[error("companion model '{reference}' does not expose CIF horizon {horizon}")]
    CompanionModelMissingHorizon { reference: String, horizon: f64 },
    #[error("time-varying tensor-product basis descriptor missing from artifacts")]
    MissingTimeVaryingBasis,
    #[error("interaction metadata for time-varying effect missing or inconsistent")]
    MissingInteractionMetadata,
    #[error("static covariate layout is missing the PGS column required for time-varying effects")]
    MissingPgsCovariate,
    #[error("time-varying interaction layout is inconsistent with stored coefficients")]
    InvalidTimeVaryingLayout,
    #[error("basis evaluation failed: {0}")]
    Basis(#[from] BasisError),
    #[error("calibrator inference failed: {0}")]
    Calibrator(String),
    #[error(
        "monotonicity violation: derivative {derivative} at age {age} fell below zero on the cached grid"
    )]
    MonotonicityViolation { age: f64, derivative: f64 },
}

pub use crate::pirls::WorkingModel;

/// Engine-facing flattened survival inputs that avoid domain-specific training bundles.
pub struct SurvivalEngineInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
}

/// Guarded log-age transformation used across training and scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub minimum_age: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn from_training(age_entry: &Array1<f64>, delta: f64) -> Result<Self, SurvivalError> {
        if delta <= 0.0 {
            return Err(SurvivalError::NonPositiveGuard);
        }
        if age_entry.is_empty() {
            return Err(SurvivalError::EmptyAgeVector);
        }
        let mut min_age = f64::INFINITY;
        for &value in age_entry.iter() {
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if value < min_age {
                min_age = value;
            }
        }
        Ok(Self {
            minimum_age: min_age,
            delta,
        })
    }

    fn guard_shift(&self, age: f64) -> Result<f64, SurvivalError> {
        if !age.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        let shifted = age - self.minimum_age + self.delta;
        if !shifted.is_finite() || shifted <= 0.0 {
            return Err(SurvivalError::GuardDomainViolation {
                age,
                minimum: self.minimum_age,
                delta: self.delta,
            });
        }
        Ok(shifted)
    }

    #[inline]
    pub fn transform(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = self.guard_shift(age)?;
        Ok(shifted.ln())
    }

    #[inline]
    pub fn derivative_factor(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = self.guard_shift(age)?;
        Ok(1.0 / shifted)
    }

    pub fn transform_array(&self, ages: &Array1<f64>) -> Result<Array1<f64>, SurvivalError> {
        let mut result = Array1::<f64>::zeros(ages.len());
        for (idx, &age) in ages.iter().enumerate() {
            result[idx] = self.transform(age)?;
        }
        Ok(result)
    }
}

/// Linear transform that removes the baseline spline's null direction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceConstraint {
    pub transform: Array2<f64>,
    pub reference_log_age: f64,
}

impl ReferenceConstraint {
    pub fn apply(&self, basis: &Array2<f64>) -> Array2<f64> {
        basis.dot(&self.transform)
    }

    // Apply constraint to a single row vector (e.g., evaluated at one point)
    // Uses general matrix multiplication to avoid allocation if possible, but ndarray dot does allocate.
    pub fn apply_row(&self, basis_row: &Array1<f64>) -> Array1<f64> {
        basis_row.dot(&self.transform)
    }
}

/// Describes a spline basis that can be reconstructed during scoring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasisDescriptor {
    pub knot_vector: Array1<f64>,
    pub degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnRange {
    pub start: usize,
    pub end: usize,
}

impl ColumnRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValueRange {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CenteringTransform {
    pub offsets: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionDescriptor {
    #[serde(default)]
    pub label: Option<String>,
    pub column_range: ColumnRange,
    #[serde(default)]
    pub value_ranges: Vec<ValueRange>,
    #[serde(default)]
    pub centering: Option<CenteringTransform>,
}

/// Stored smoothing metadata for reproduction at prediction time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyDescriptor {
    pub order: usize,
    pub lambda: f64,
    pub matrix: Array2<f64>,
    pub column_range: ColumnRange,
}

/// Configuration for the optional tensor-product time-varying effect.
#[derive(Debug, Clone)]
pub struct TensorProductConfig {
    pub label: Option<String>,
    pub pgs_basis: BasisDescriptor,
    pub pgs_penalty_order: usize,
    pub lambda_age: f64,
    pub lambda_pgs: f64,
    pub lambda_null: f64,
}

/// Column descriptions for static covariates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
    #[serde(default)]
    pub ranges: Vec<ValueRange>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
    pub range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }

            let view = beta.slice(s![block.range.clone()]);
            let contrib = block.matrix.dot(&view.to_owned());
            let mut grad_slice = grad.slice_mut(s![block.range.clone()]);
            grad_slice += &(2.0 * block.lambda * contrib);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut hessian = Array2::zeros((dim, dim));
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let rows = block.range.clone();
            for (local_i, row_idx) in rows.clone().enumerate() {
                for (local_j, col_idx) in rows.clone().enumerate() {
                    hessian[[row_idx, col_idx]] +=
                        2.0 * block.lambda * block.matrix[[local_i, local_j]];
                }
            }
        }
        hessian
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let view = beta.slice(s![block.range.clone()]);
            let quad = view.dot(&block.matrix.dot(&view.to_owned()));
            value += block.lambda * quad;
        }
        value
    }
}

/// Bundle returned by [`build_survival_layout`] containing cached designs and metadata
/// required for serialization.
#[derive(Debug, Clone)]
pub struct SurvivalLayoutBundle {
    pub layout: SurvivalLayout,
    pub monotonicity: MonotonicityPenalty,
    pub penalty_descriptors: Vec<PenaltyDescriptor>,
    pub interaction_metadata: Vec<InteractionDescriptor>,
    pub time_varying_basis: Option<BasisDescriptor>,
}

/// Training-time cached design matrices.
#[derive(Debug, Clone)]
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Array2<f64>,
    pub extra_static_covariates: Array2<f64>,
    pub static_covariate_names: Vec<String>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
    pub combined_entry: Array2<f64>,
    pub combined_exit: Array2<f64>,
    pub combined_derivative_exit: Array2<f64>,
    pub monotonicity: MonotonicityPenalty,
}

/// Frequency-weighted survival training data bundle.
#[derive(Debug, Clone)]
pub struct SurvivalLayoutInputs {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub static_covariates: Array2<f64>,
    pub static_covariate_names: Vec<String>,
    pub time_varying_covariate: Option<Array1<f64>>,
}

impl SurvivalLayoutInputs {
    pub fn validate(&self) -> Result<(), SurvivalError> {
        let n = self.age_entry.len();
        if n == 0 {
            return Err(SurvivalError::EmptyAgeVector);
        }
        let dimension_mismatch = self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
            || self.static_covariates.nrows() != n;
        if dimension_mismatch {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        if self.static_covariate_names.len() != self.static_covariates.ncols() {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        if let Some(ref tv) = self.time_varying_covariate {
            if tv.len() != n {
                return Err(SurvivalError::CovariateDimensionMismatch);
            }
        }

        for i in 0..n {
            let entry = self.age_entry[i];
            let exit = self.age_exit[i];
            if !entry.is_finite() || !exit.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if !(entry < exit) {
                return Err(SurvivalError::InvalidAgeOrder);
            }

            let target = self.event_target[i];
            let competing = self.event_competing[i];
            if target > 1 || competing > 1 {
                return Err(SurvivalError::InvalidEventFlag);
            }
            if target == 1 && competing == 1 {
                return Err(SurvivalError::ConflictingEvents);
            }

            let weight = self.sample_weight[i];
            if !weight.is_finite() || weight < 0.0 {
                return Err(SurvivalError::InvalidSampleWeight);
            }

            for j in 0..self.static_covariates.ncols() {
                if !self.static_covariates[[i, j]].is_finite() {
                    return Err(SurvivalError::NonFiniteCovariate);
                }
            }
            if let Some(ref tv) = self.time_varying_covariate {
                if !tv[i].is_finite() {
                    return Err(SurvivalError::NonFiniteCovariate);
                }
            }
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_survival_inputs(
    age_entry: ArrayView1<f64>,
    age_exit: ArrayView1<f64>,
    event_target: ArrayView1<u8>,
    event_competing: ArrayView1<u8>,
    sample_weight: ArrayView1<f64>,
    static_covariates: ArrayView2<f64>,
    time_varying_covariate: Option<ArrayView1<f64>>,
) -> Result<(), SurvivalError> {
    let n = age_entry.len();
    if n == 0 {
        return Err(SurvivalError::EmptyAgeVector);
    }
    let dimension_mismatch = age_exit.len() != n
        || event_target.len() != n
        || event_competing.len() != n
        || sample_weight.len() != n
        || static_covariates.nrows() != n;
    if dimension_mismatch {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }
    if let Some(tv) = time_varying_covariate {
        if tv.len() != n {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
    }

    for i in 0..n {
        let entry = age_entry[i];
        let exit = age_exit[i];
        if !entry.is_finite() || !exit.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        if !(entry < exit) {
            return Err(SurvivalError::InvalidAgeOrder);
        }

        let target = event_target[i];
        let competing = event_competing[i];
        if target > 1 || competing > 1 {
            return Err(SurvivalError::InvalidEventFlag);
        }
        if target == 1 && competing == 1 {
            return Err(SurvivalError::ConflictingEvents);
        }

        let weight = sample_weight[i];
        if !weight.is_finite() || weight < 0.0 {
            return Err(SurvivalError::InvalidSampleWeight);
        }

        for j in 0..static_covariates.ncols() {
            if !static_covariates[[i, j]].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
        if let Some(tv) = time_varying_covariate {
            if !tv[i].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
    }

    Ok(())
}

/// Guard that constrains the baseline spline at the chosen reference point.
fn make_reference_constraint(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    reference_u: f64,
) -> Result<ReferenceConstraint, SurvivalError> {
    let data = array![reference_u];
    let (basis_arc, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Provided(knot_vector),
        degree,
        BasisOptions::value(),
    )?;
    let basis = (*basis_arc).clone();
    let row = basis.row(0).to_owned();
    let transform = nullspace_transform(&row)?;
    Ok(ReferenceConstraint {
        transform,
        reference_log_age: reference_u,
    })
}

/// Build a nullspace transform for a single-row constraint.
fn nullspace_transform(constraint_row: &Array1<f64>) -> Result<Array2<f64>, SurvivalError> {
    let k = constraint_row.len();
    let mut row_mat = Array2::<f64>::zeros((k, 1));
    row_mat.column_mut(0).assign(constraint_row);
    let (u_opt, ..) = row_mat
        .svd(true, false)
        .map_err(|err| SurvivalError::Basis(BasisError::from(err)))?;
    let u = u_opt.ok_or(SurvivalError::Basis(
        BasisError::ConstraintNullspaceNotFound,
    ))?;
    Ok(u.slice(s![.., 1..]).to_owned())
}

/// Evaluate a basis and its derivative with respect to the guarded log-age.
fn evaluate_basis_and_derivative(
    log_ages: ArrayView1<f64>,
    descriptor: &BasisDescriptor,
) -> Result<(Array2<f64>, Array2<f64>), SurvivalError> {
    let (basis_arc, _) = create_basis::<Dense>(
        log_ages,
        KnotSource::Provided(descriptor.knot_vector.view()),
        descriptor.degree,
        BasisOptions::value(),
    )?;
    let basis = (*basis_arc).clone();
    let (derivative_arc, _) = create_basis::<Dense>(
        log_ages,
        KnotSource::Provided(descriptor.knot_vector.view()),
        descriptor.degree,
        BasisOptions::first_derivative(),
    )?;
    let derivative = (*derivative_arc).clone();

    Ok((basis, derivative))
}

/// Structure to hold scratch buffers for survival calculations to minimize allocations.
pub struct SurvivalScratch {
    // Basis scratch for internal spline evaluation
    pub spline_scratch: SplineScratch,

    // Buffers for basis evaluation
    pub basis_raw: Vec<f64>,
    pub basis_plus: Vec<f64>,
    pub basis_minus: Vec<f64>,

    // Buffers for derivative calculation
    pub derivative_raw: Vec<f64>,

    // Buffers for constrained basis (after nullspace transform)
    // Size: k-1
    pub constrained_basis: Vec<f64>,
    pub constrained_derivative: Vec<f64>,

    // Buffers for tensor product intermediate results if needed
    // Size: (k-1) * m where m is PGS basis size (usually small, e.g. 4)
    pub tensor_prod_buff: Vec<f64>,

    // Buffer for PGS basis evaluation
    pub pgs_basis_buff: Vec<f64>,
}

impl SurvivalScratch {
    pub fn new(age_degree: usize, max_basis_size: usize, max_pgs_basis_size: usize) -> Self {
        Self {
            spline_scratch: SplineScratch::new(age_degree),
            basis_raw: vec![0.0; max_basis_size],
            basis_plus: vec![0.0; max_basis_size],
            basis_minus: vec![0.0; max_basis_size],
            derivative_raw: vec![0.0; max_basis_size],
            constrained_basis: vec![0.0; max_basis_size],
            constrained_derivative: vec![0.0; max_basis_size],
            tensor_prod_buff: vec![0.0; max_basis_size * max_pgs_basis_size],
            pgs_basis_buff: vec![0.0; max_pgs_basis_size],
        }
    }
}

fn vec_mat_mul_into(v: &[f64], m: &Array2<f64>, out: &mut [f64]) {
    // v is 1xK, m is KxN, out is 1xN
    // out[j] = sum_i v[i] * m[i, j]
    let k = v.len();
    let n = m.ncols();
    assert_eq!(m.nrows(), k);
    assert!(out.len() >= n);

    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..k {
            sum += v[i] * m[[i, j]];
        }
        out[j] = sum;
    }
}

fn tensor_product_into(a: &[f64], b: &[f64], out: &mut [f64]) {
    // Flattened outer product
    let na = a.len();
    let nb = b.len();
    assert!(out.len() >= na * nb);

    let mut idx = 0;
    for &val_a in a {
        for &val_b in b {
            out[idx] = val_a * val_b;
            idx += 1;
        }
    }
}

/// Non-allocating version of evaluate_basis_and_derivative for a scalar point.
fn evaluate_basis_and_derivative_scalar_into(
    log_age: f64,
    descriptor: &BasisDescriptor,
    out_basis: &mut [f64],
    out_deriv: &mut [f64],
    spline_scratch: &mut SplineScratch,
    plus_buff: &mut [f64],
    minus_buff: &mut [f64],
) -> Result<(), SurvivalError> {
    let num_basis = descriptor.knot_vector.len() - descriptor.degree - 1;

    evaluate_bspline_basis_scalar(
        log_age,
        descriptor.knot_vector.view(),
        descriptor.degree,
        out_basis,
        spline_scratch,
    )?;

    let eps = 1e-6;
    evaluate_bspline_basis_scalar(
        log_age + eps,
        descriptor.knot_vector.view(),
        descriptor.degree,
        plus_buff,
        spline_scratch,
    )?;

    evaluate_bspline_basis_scalar(
        log_age - eps,
        descriptor.knot_vector.view(),
        descriptor.degree,
        minus_buff,
        spline_scratch,
    )?;

    for i in 0..num_basis {
        out_deriv[i] = (plus_buff[i] - minus_buff[i]) / (2.0 * eps);
    }

    Ok(())
}

fn accumulate_weighted_vector<S>(target: &mut Array1<f64>, scale: f64, values: &ArrayBase<S, Ix1>)
where
    S: Data<Elem = f64>,
{
    if scale == 0.0 {
        return;
    }
    Zip::from(target)
        .and(values)
        .for_each(|t, &v| *t += scale * v);
}

fn accumulate_symmetric_outer<S>(target: &mut Array2<f64>, scale: f64, values: &ArrayBase<S, Ix1>)
where
    S: Data<Elem = f64>,
{
    if scale == 0.0 {
        return;
    }
    let len = values.len();
    for j in 0..len {
        let vj = values[j];
        for k in j..len {
            let contribution = scale * vj * values[k];
            target[[j, k]] += contribution;
            if j != k {
                target[[k, j]] += contribution;
            }
        }
    }
}

pub fn build_survival_layout(
    data: &SurvivalLayoutInputs,
    age_basis: &BasisDescriptor,
    delta: f64,
    baseline_penalty_order: usize,
    monotonic_grid_size: usize,
    time_varying: Option<&TensorProductConfig>,
) -> Result<SurvivalLayoutBundle, SurvivalError> {
    data.validate()?;
    let n = data.age_entry.len();
    let age_transform = AgeTransform::from_training(&data.age_entry, delta)?;
    let log_entry = age_transform.transform_array(&data.age_entry)?;
    let log_exit = age_transform.transform_array(&data.age_exit)?;

    let reference_u = log_exit.mean().unwrap_or(0.0);
    let reference_constraint =
        make_reference_constraint(age_basis.knot_vector.view(), age_basis.degree, reference_u)?;

    let (baseline_entry_raw, _) = evaluate_basis_and_derivative(log_entry.view(), age_basis)?;
    let (baseline_exit_raw, baseline_exit_deriv_u) =
        evaluate_basis_and_derivative(log_exit.view(), age_basis)?;

    let constrained_entry = reference_constraint.apply(&baseline_entry_raw);
    let constrained_exit = reference_constraint.apply(&baseline_exit_raw);
    let constrained_derivative_exit_u = reference_constraint.apply(&baseline_exit_deriv_u);

    let mut baseline_derivative_exit = constrained_derivative_exit_u;
    for (mut row, age) in baseline_derivative_exit
        .rows_mut()
        .into_iter()
        .zip(data.age_exit.iter().copied())
    {
        let factor = age_transform.derivative_factor(age)?;
        row.mapv_inplace(|v| v * factor);
    }

    let static_covariates = assemble_static_covariates(data);
    let extra_static_covariates = Array2::<f64>::zeros((n, 0));
    let static_covariate_names = assemble_static_covariate_names(data);

    let baseline_cols = constrained_exit.ncols();
    let penalty_matrix =
        create_difference_penalty_matrix(baseline_cols, baseline_penalty_order, None)?;
    let baseline_lambda = baseline_lambda_seed(
        &age_basis.knot_vector,
        age_basis.degree,
        baseline_penalty_order,
    );
    let mut penalty_blocks = vec![PenaltyBlock {
        matrix: penalty_matrix.clone(),
        lambda: baseline_lambda,
        range: 0..baseline_cols,
    }];
    let mut penalty_descriptors = vec![PenaltyDescriptor {
        order: baseline_penalty_order,
        lambda: baseline_lambda,
        matrix: penalty_matrix.clone(),
        column_range: ColumnRange::new(0, baseline_cols),
    }];

    let mut time_varying_entry: Option<Array2<f64>> = None;
    let mut time_varying_exit: Option<Array2<f64>> = None;
    let mut time_varying_derivative_exit: Option<Array2<f64>> = None;
    let mut interaction_metadata: Vec<InteractionDescriptor> = Vec::new();
    let mut time_varying_basis_descriptor: Option<BasisDescriptor> = None;

    if let Some(config) = time_varying {
        let lambda_age = if config.lambda_age == 0.0 {
            baseline_lambda
        } else {
            config.lambda_age
        };
        let lambda_pgs = if config.lambda_pgs == 0.0 {
            baseline_lambda
        } else {
            config.lambda_pgs
        };
        let lambda_null = if config.lambda_null == 0.0 {
            baseline_lambda
        } else {
            config.lambda_null
        };

        let Some(tv_covariate) = data.time_varying_covariate.as_ref() else {
            warn!(
                "time-varying config provided but no time_varying_covariate supplied; skipping interaction"
            );
            let combined_entry = concatenate_design(
                &constrained_entry,
                time_varying_entry.as_ref(),
                &static_covariates,
                &extra_static_covariates,
            );
            let combined_exit = concatenate_design(
                &constrained_exit,
                time_varying_exit.as_ref(),
                &static_covariates,
                &extra_static_covariates,
            );
            let zero_static = Array2::<f64>::zeros((n, static_covariates.ncols()));
            let zero_extra = Array2::<f64>::zeros((n, extra_static_covariates.ncols()));
            let combined_derivative_exit = concatenate_design(
                &baseline_derivative_exit,
                time_varying_derivative_exit.as_ref(),
                &zero_static,
                &zero_extra,
            );
            let mut layout = SurvivalLayout {
                baseline_entry: constrained_entry,
                baseline_exit: constrained_exit,
                baseline_derivative_exit,
                time_varying_entry,
                time_varying_exit,
                time_varying_derivative_exit,
                static_covariates,
                extra_static_covariates,
                static_covariate_names,
                age_transform,
                reference_constraint,
                penalties: PenaltyBlocks::new(penalty_blocks),
                combined_entry,
                combined_exit,
                combined_derivative_exit,
                monotonicity: empty_monotonicity_constraint(),
            };
            let monotonicity = build_monotonicity_penalty(
                &layout,
                age_basis,
                &data.age_entry,
                &data.age_exit,
                monotonic_grid_size,
            )?;
            layout.monotonicity = monotonicity.clone();

            return Ok(SurvivalLayoutBundle {
                layout,
                monotonicity,
                penalty_descriptors,
                interaction_metadata,
                time_varying_basis: time_varying_basis_descriptor,
            });
        };

        let (pgs_basis_full_arc, _) = create_basis::<Dense>(
            tv_covariate.view(),
            KnotSource::Provided(config.pgs_basis.knot_vector.view()),
            config.pgs_basis.degree,
            BasisOptions::value(),
        )?;
        let pgs_basis_full = (*pgs_basis_full_arc).clone();
        if pgs_basis_full.ncols() <= 1 {
            warn!("PGS basis returned no range columns; skipping time-varying interaction");
        } else {
            let mut pgs_basis = pgs_basis_full.slice(s![.., 1..]).to_owned();
            let offsets = compute_weighted_column_means(&pgs_basis, &data.sample_weight);
            if offsets.len() == pgs_basis.ncols() {
                for (mut column, &offset) in pgs_basis.axis_iter_mut(Axis(1)).zip(offsets.iter()) {
                    column.mapv_inplace(|value: f64| value - offset);
                }
            }

            let time_entry = row_wise_tensor_product(&constrained_entry, &pgs_basis);
            let time_exit = row_wise_tensor_product(&constrained_exit, &pgs_basis);
            let time_derivative_exit =
                row_wise_tensor_product(&baseline_derivative_exit, &pgs_basis);

            let pgs_cols = pgs_basis.ncols();
            let time_cols = baseline_cols * pgs_cols;

            if time_cols > 0 {
                let age_penalty_1d = penalty_matrix.clone();
                let pgs_penalty_1d =
                    create_difference_penalty_matrix(pgs_cols, config.pgs_penalty_order, None)?;

                let identity_age = Array2::<f64>::eye(baseline_cols);
                let identity_pgs = Array2::<f64>::eye(pgs_cols);

                let kron_age = kronecker_product(&age_penalty_1d, &identity_pgs);
                let kron_pgs = kronecker_product(&identity_age, &pgs_penalty_1d);

                let norm_age = frobenius_norm(&kron_age).max(1e-12);
                let norm_pgs = frobenius_norm(&kron_pgs).max(1e-12);
                let kron_age_normed = kron_age.mapv(|v| v / norm_age);
                let kron_pgs_normed = kron_pgs.mapv(|v| v / norm_pgs);

                let time_range = baseline_cols..(baseline_cols + time_cols);

                penalty_blocks.push(PenaltyBlock {
                    matrix: kron_age_normed.clone(),
                    lambda: lambda_age,
                    range: time_range.clone(),
                });
                penalty_descriptors.push(PenaltyDescriptor {
                    order: baseline_penalty_order,
                    lambda: lambda_age,
                    matrix: kron_age_normed.clone(),
                    column_range: ColumnRange::new(time_range.start, time_range.end),
                });

                penalty_blocks.push(PenaltyBlock {
                    matrix: kron_pgs_normed.clone(),
                    lambda: lambda_pgs,
                    range: time_range.clone(),
                });
                penalty_descriptors.push(PenaltyDescriptor {
                    order: config.pgs_penalty_order,
                    lambda: lambda_pgs,
                    matrix: kron_pgs_normed.clone(),
                    column_range: ColumnRange::new(time_range.start, time_range.end),
                });

                if let (Ok((age_null, _)), Ok((pgs_null, _))) = (
                    null_range_whiten(&age_penalty_1d),
                    null_range_whiten(&pgs_penalty_1d),
                ) && age_null.ncols() > 0
                    && pgs_null.ncols() > 0
                {
                    let age_projector = age_null.dot(&age_null.t());
                    let pgs_projector = pgs_null.dot(&pgs_null.t());
                    let kron_null = kronecker_product(&age_projector, &pgs_projector);
                    let norm_null = frobenius_norm(&kron_null).max(1e-12);
                    let kron_null_normed = kron_null.mapv(|v| v / norm_null);
                    penalty_blocks.push(PenaltyBlock {
                        matrix: kron_null_normed.clone(),
                        lambda: lambda_null,
                        range: time_range.clone(),
                    });
                    penalty_descriptors.push(PenaltyDescriptor {
                        order: 0,
                        lambda: lambda_null,
                        matrix: kron_null_normed,
                        column_range: ColumnRange::new(time_range.start, time_range.end),
                    });
                }

                time_varying_entry = Some(time_entry);
                time_varying_exit = Some(time_exit);
                time_varying_derivative_exit = Some(time_derivative_exit);
                time_varying_basis_descriptor = Some(config.pgs_basis.clone());

                let mut min_pgs = f64::INFINITY;
                let mut max_pgs = f64::NEG_INFINITY;
                for &value in tv_covariate.iter() {
                    if value < min_pgs {
                        min_pgs = value;
                    }
                    if value > max_pgs {
                        max_pgs = value;
                    }
                }
                interaction_metadata.push(InteractionDescriptor {
                    label: config.label.clone(),
                    column_range: ColumnRange::new(time_range.start, time_range.end),
                    value_ranges: vec![ValueRange {
                        min: min_pgs,
                        max: max_pgs,
                    }],
                    centering: Some(CenteringTransform { offsets }),
                });
            }
        }
    }

    let combined_entry = concatenate_design(
        &constrained_entry,
        time_varying_entry.as_ref(),
        &static_covariates,
        &extra_static_covariates,
    );
    let combined_exit = concatenate_design(
        &constrained_exit,
        time_varying_exit.as_ref(),
        &static_covariates,
        &extra_static_covariates,
    );
    let zero_static = Array2::<f64>::zeros((n, static_covariates.ncols()));
    let zero_extra = Array2::<f64>::zeros((n, extra_static_covariates.ncols()));
    let combined_derivative_exit = concatenate_design(
        &baseline_derivative_exit,
        time_varying_derivative_exit.as_ref(),
        &zero_static,
        &zero_extra,
    );

    let mut layout = SurvivalLayout {
        baseline_entry: constrained_entry,
        baseline_exit: constrained_exit,
        baseline_derivative_exit,
        time_varying_entry,
        time_varying_exit,
        time_varying_derivative_exit,
        static_covariates,
        extra_static_covariates,
        static_covariate_names,
        age_transform,
        reference_constraint,
        penalties: PenaltyBlocks::new(penalty_blocks),
        combined_entry,
        combined_exit,
        combined_derivative_exit,
        monotonicity: empty_monotonicity_constraint(),
    };

    let monotonicity = build_monotonicity_penalty(
        &layout,
        age_basis,
        &data.age_entry,
        &data.age_exit,
        monotonic_grid_size,
    )?;
    layout.monotonicity = monotonicity.clone();

    Ok(SurvivalLayoutBundle {
        layout,
        monotonicity,
        penalty_descriptors,
        interaction_metadata,
        time_varying_basis: time_varying_basis_descriptor,
    })
}

fn assemble_static_covariates(data: &SurvivalLayoutInputs) -> Array2<f64> {
    data.static_covariates.clone()
}

fn assemble_static_covariate_names(data: &SurvivalLayoutInputs) -> Vec<String> {
    data.static_covariate_names.clone()
}

fn concatenate_design(
    baseline: &Array2<f64>,
    time_varying: Option<&Array2<f64>>,
    static_covariates: &Array2<f64>,
    extra_static_covariates: &Array2<f64>,
) -> Array2<f64> {
    let mut parts: Vec<ArrayView2<f64>> = Vec::new();
    parts.push(baseline.view());
    if let Some(tv) = time_varying {
        parts.push(tv.view());
    }
    if static_covariates.ncols() > 0 {
        parts.push(static_covariates.view());
    }
    if extra_static_covariates.ncols() > 0 {
        parts.push(extra_static_covariates.view());
    }
    concatenate(Axis(1), &parts).expect("design concatenation")
}

fn compute_weighted_column_means(matrix: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
    let cols = matrix.ncols();
    let rows = matrix.nrows();
    if cols == 0 || rows == 0 {
        return Array1::<f64>::zeros(cols);
    }
    let mut means = Array1::<f64>::zeros(cols);
    let mut total_weight = 0.0;
    for (row_idx, row) in matrix.rows().into_iter().enumerate() {
        let w = weights[row_idx];
        if w == 0.0 {
            continue;
        }
        total_weight += w;
        for (col_idx, value) in row.iter().enumerate() {
            means[col_idx] += w * value;
        }
    }
    if total_weight > 0.0 {
        means.mapv_inplace(|value| value / total_weight);
    }
    means
}

fn row_wise_tensor_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    assert_eq!(a.nrows(), b.nrows());
    let n = a.nrows();
    let a_cols = a.ncols();
    let b_cols = b.ncols();
    let mut result = Array2::<f64>::zeros((n, a_cols * b_cols));
    for i in 0..n {
        let mut idx = 0;
        for j in 0..a_cols {
            let a_val = a[[i, j]];
            for k in 0..b_cols {
                result[[i, idx]] = a_val * b[[i, k]];
                idx += 1;
            }
        }
    }
    result
}

fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let mut result = Array2::<f64>::zeros((a_rows * b_rows, a_cols * b_cols));
    for i in 0..a_rows {
        for j in 0..a_cols {
            let a_val = a[[i, j]];
            if a_val == 0.0 {
                continue;
            }
            for k in 0..b_rows {
                for l in 0..b_cols {
                    result[[i * b_rows + k, j * b_cols + l]] = a_val * b[[k, l]];
                }
            }
        }
    }
    result
}

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Cached derivative designs used to enforce monotonicity during optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonotonicityPenalty {
    pub derivative_design: Array2<f64>,
    pub quadrature_design: Array2<f64>,
    pub grid_ages: Array1<f64>,
    pub quadrature_left: Array1<f64>,
    pub quadrature_right: Array1<f64>,
}

fn empty_monotonicity_constraint() -> MonotonicityPenalty {
    MonotonicityPenalty {
        derivative_design: Array2::<f64>::zeros((0, 0)),
        quadrature_design: Array2::<f64>::zeros((0, 0)),
        grid_ages: Array1::<f64>::zeros(0),
        quadrature_left: Array1::<f64>::zeros(0),
        quadrature_right: Array1::<f64>::zeros(0),
    }
}

/// Configuration controlling guard behaviour and Hessian construction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SurvivalSpec {
    pub derivative_guard: f64,
    pub use_expected_information: bool,
}

impl Default for SurvivalSpec {
    fn default() -> Self {
        Self {
            derivative_guard: DEFAULT_DERIVATIVE_GUARD,
            use_expected_information: false,
        }
    }
}

fn build_monotonicity_penalty(
    layout: &SurvivalLayout,
    age_basis: &BasisDescriptor,
    ages_entry: &Array1<f64>,
    ages_exit: &Array1<f64>,
    grid_size: usize,
) -> Result<MonotonicityPenalty, SurvivalError> {
    if grid_size == 0 {
        let cols = layout.combined_exit.ncols();
        return Ok(MonotonicityPenalty {
            derivative_design: Array2::<f64>::zeros((0, cols)),
            quadrature_design: Array2::<f64>::zeros((0, cols)),
            grid_ages: Array1::<f64>::zeros(0),
            quadrature_left: Array1::<f64>::zeros(0),
            quadrature_right: Array1::<f64>::zeros(0),
        });
    }

    let mut min_age = f64::INFINITY;
    let mut max_age = f64::NEG_INFINITY;
    for &age in ages_entry.iter().chain(ages_exit.iter()) {
        if age < min_age {
            min_age = age;
        }
        if age > max_age {
            max_age = age;
        }
    }
    if !min_age.is_finite() || !max_age.is_finite() || min_age >= max_age {
        let cols = layout.combined_exit.ncols();
        return Ok(MonotonicityPenalty {
            derivative_design: Array2::<f64>::zeros((0, cols)),
            quadrature_design: Array2::<f64>::zeros((0, cols)),
            grid_ages: Array1::<f64>::zeros(0),
            quadrature_left: Array1::<f64>::zeros(0),
            quadrature_right: Array1::<f64>::zeros(0),
        });
    }

    let mut grid = Array1::<f64>::zeros(grid_size);
    if grid_size == 1 {
        grid[0] = min_age;
    } else {
        let span = max_age - min_age;
        for (idx, value) in grid.iter_mut().enumerate() {
            let frac = idx as f64 / (grid_size as f64 - 1.0);
            *value = min_age + frac * span;
        }
    }

    let mut log_grid = Array1::<f64>::zeros(grid_size);
    for (idx, &age) in grid.iter().enumerate() {
        log_grid[idx] = layout.age_transform.transform(age)?;
    }
    let (basis_grid, derivative_u) = evaluate_basis_and_derivative(log_grid.view(), age_basis)?;
    let constrained_basis_grid = layout.reference_constraint.apply(&basis_grid);
    let constrained_derivative_u = layout.reference_constraint.apply(&derivative_u);
    let mut derivative_age = constrained_derivative_u;
    for (mut row, &age) in derivative_age.rows_mut().into_iter().zip(grid.iter()) {
        let factor = layout.age_transform.derivative_factor(age)?;
        row *= factor;
    }

    let cols = layout.combined_exit.ncols();
    let mut combined = Array2::<f64>::zeros((grid_size, cols));
    let mut quadrature_design = Array2::<f64>::zeros((grid_size, cols));
    let baseline_cols = layout.baseline_exit.ncols();
    combined
        .slice_mut(s![.., ..baseline_cols])
        .assign(&derivative_age);
    quadrature_design
        .slice_mut(s![.., ..baseline_cols])
        .assign(&constrained_basis_grid);

    let mut quadrature_left = Array1::<f64>::zeros(grid_size);
    let mut quadrature_right = Array1::<f64>::zeros(grid_size);
    for idx in 0..grid_size {
        let left_bound = if idx == 0 {
            min_age
        } else {
            0.5 * (grid[idx - 1] + grid[idx])
        };
        let right_bound = if idx == grid_size - 1 {
            max_age
        } else {
            0.5 * (grid[idx] + grid[idx + 1])
        };
        quadrature_left[idx] = left_bound;
        quadrature_right[idx] = right_bound;
    }

    Ok(MonotonicityPenalty {
        derivative_design: combined,
        quadrature_design,
        grid_ages: grid,
        quadrature_left,
        quadrature_right,
    })
}

/// Royston–Parmar working model implementation.
pub struct WorkingModelSurvival {
    pub layout: Arc<SurvivalLayout>,
    pub sample_weight: Arc<Array1<f64>>,
    pub event_target: Arc<Array1<u8>>,
    pub age_entry: Arc<Array1<f64>>,
    pub age_exit: Arc<Array1<f64>>,
    pub monotonicity: Arc<MonotonicityPenalty>,
    pub spec: SurvivalSpec,
    /// Time-varying basis descriptor for computing tensor products at arbitrary ages.
    /// Required for expected Hessian computation when time-varying effects are present.
    pub time_varying_basis: Option<BasisDescriptor>,
}

impl WorkingModelSurvival {
    #[cfg(test)]
    pub(crate) fn new(
        layout: SurvivalLayout,
        data: &SurvivalLayoutInputs,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        data.validate()?;
        Ok(Self {
            layout: Arc::new(layout),
            sample_weight: Arc::new(data.sample_weight.clone()),
            event_target: Arc::new(data.event_target.clone()),
            age_entry: Arc::new(data.age_entry.clone()),
            age_exit: Arc::new(data.age_exit.clone()),
            monotonicity: Arc::new(monotonicity),
            spec,
            time_varying_basis: None,
        })
    }

    /// Set the time-varying basis descriptor for expected Hessian computation.
    pub fn with_time_varying_basis(mut self, basis: Option<BasisDescriptor>) -> Self {
        self.time_varying_basis = basis;
        self
    }

    pub fn from_observations(
        layout: SurvivalLayout,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
        age_entry: ArrayView1<'_, f64>,
        age_exit: ArrayView1<'_, f64>,
        event_target: ArrayView1<'_, u8>,
        sample_weight: ArrayView1<'_, f64>,
    ) -> Result<Self, SurvivalError> {
        if age_entry.len() == 0
            || age_entry.len() != age_exit.len()
            || age_entry.len() != event_target.len()
            || age_entry.len() != sample_weight.len()
        {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        for i in 0..age_entry.len() {
            let a0 = age_entry[i];
            let a1 = age_exit[i];
            if !a0.is_finite() || !a1.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if !(a0 < a1) {
                return Err(SurvivalError::InvalidAgeOrder);
            }
            let evt = event_target[i];
            if evt > 1 {
                return Err(SurvivalError::InvalidEventFlag);
            }
            let w = sample_weight[i];
            if !w.is_finite() || w < 0.0 {
                return Err(SurvivalError::InvalidSampleWeight);
            }
        }
        Ok(Self {
            layout: Arc::new(layout),
            sample_weight: Arc::new(sample_weight.to_owned()),
            event_target: Arc::new(event_target.to_owned()),
            age_entry: Arc::new(age_entry.to_owned()),
            age_exit: Arc::new(age_exit.to_owned()),
            monotonicity: Arc::new(monotonicity),
            spec,
            time_varying_basis: None,
        })
    }

    fn build_expected_information_hessian(
        &self,
        beta: &Array1<f64>,
        penalty_hessian: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, SurvivalError> {
        let grid_size = self.monotonicity.grid_ages.len();
        if grid_size <= 1 {
            return Ok(None);
        }

        let p = beta.len();
        let mut expected = Array2::<f64>::zeros((p, p));
        let baseline_cols = self.layout.baseline_exit.ncols();
        let time_cols = self
            .layout
            .time_varying_exit
            .as_ref()
            .map(|arr| arr.ncols())
            .unwrap_or(0);
        let static_cols = self.layout.static_covariates.ncols();
        let extra_cols = self.layout.extra_static_covariates.ncols();
        let static_offset = baseline_cols + time_cols;
        let extra_offset = static_offset + static_cols;
        let guard_threshold = self.spec.derivative_guard.max(f64::EPSILON);
        let left_bounds = &self.monotonicity.quadrature_left;
        let right_bounds = &self.monotonicity.quadrature_right;
        let derivative_raw = self.layout.combined_derivative_exit.dot(beta);

        for i in 0..self.age_entry.len() {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !(exit_age > entry_age) {
                continue;
            }

            let entry_design = self.layout.combined_entry.row(i);
            let eta_entry = entry_design.dot(beta);
            if !eta_entry.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let hazard_entry = eta_entry.exp();
            if !hazard_entry.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }

            let mut design = Array1::<f64>::zeros(p);
            let mut derivative_design = Array1::<f64>::zeros(p);

            // Pre-compute PGS basis for this subject if time-varying effects are present
            let pgs_basis_reduced: Option<Vec<f64>> = if time_cols > 0 {
                if let Some(ref tv_basis) = self.time_varying_basis {
                    // PGS is typically the first static covariate
                    let pgs_value = self.layout.static_covariates[[i, 0]];
                    let pgs_dim = tv_basis
                        .knot_vector
                        .len()
                        .saturating_sub(tv_basis.degree + 1);
                    if pgs_dim > 1 {
                        let mut pgs_buff = vec![0.0; pgs_dim];
                        let mut scratch = SplineScratch::new(tv_basis.degree);
                        if evaluate_bspline_basis_scalar(
                            pgs_value,
                            tv_basis.knot_vector.view(),
                            tv_basis.degree,
                            &mut pgs_buff,
                            &mut scratch,
                        )
                        .is_ok()
                        {
                            // Skip first column (same as build_survival_layout logic)
                            Some(pgs_buff[1..].to_vec())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            for j in 0..grid_size {
                if left_bounds[j] >= exit_age {
                    break;
                }
                if right_bounds[j] <= entry_age {
                    continue;
                }
                let left = left_bounds[j].max(entry_age);
                let right = right_bounds[j].min(exit_age);
                if right <= left {
                    continue;
                }

                // Assign baseline columns from quadrature design
                design.assign(&self.monotonicity.quadrature_design.row(j));
                derivative_design.assign(&self.monotonicity.derivative_design.row(j));

                // Compute time-varying columns: tensor product of baseline_basis(grid_age) ⊗ pgs_basis
                if time_cols > 0 {
                    if let Some(ref pgs_reduced) = pgs_basis_reduced {
                        // baseline basis at this grid point is in quadrature_design columns 0..baseline_cols
                        let baseline_at_grid: Vec<f64> = self
                            .monotonicity
                            .quadrature_design
                            .row(j)
                            .slice(s![..baseline_cols])
                            .iter()
                            .copied()
                            .collect();
                        let baseline_derivative_at_grid: Vec<f64> = self
                            .monotonicity
                            .derivative_design
                            .row(j)
                            .slice(s![..baseline_cols])
                            .iter()
                            .copied()
                            .collect();

                        // Tensor product: out[k*pgs_len + l] = baseline[k] * pgs[l]
                        let pgs_len = pgs_reduced.len();
                        let expected_time_cols = baseline_cols * pgs_len;
                        if expected_time_cols == time_cols {
                            let mut idx = baseline_cols;
                            for &base_val in &baseline_at_grid {
                                for &pgs_val in pgs_reduced {
                                    design[idx] = base_val * pgs_val;
                                    idx += 1;
                                }
                            }
                            let mut idx = baseline_cols;
                            for &base_val in &baseline_derivative_at_grid {
                                for &pgs_val in pgs_reduced {
                                    derivative_design[idx] = base_val * pgs_val;
                                    idx += 1;
                                }
                            }
                        }
                    }
                }

                if static_cols > 0 {
                    design
                        .slice_mut(s![static_offset..extra_offset])
                        .assign(&self.layout.static_covariates.row(i));
                }
                if extra_cols > 0 {
                    design
                        .slice_mut(s![extra_offset..extra_offset + extra_cols])
                        .assign(&self.layout.extra_static_covariates.row(i));
                }
                let eta = design.dot(beta);
                if !eta.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                let hazard = eta.exp();
                if !hazard.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                let derivative_eta = derivative_design.dot(beta);
                if !derivative_eta.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                if derivative_eta <= guard_threshold {
                    continue;
                }
                let hazard_rate = hazard * derivative_eta;
                let hazard_delta = (hazard - hazard_entry).max(0.0);
                let survival_prob = (-hazard_delta).exp();
                let scale = weight * (right - left) * hazard_rate * survival_prob;
                accumulate_symmetric_outer(&mut expected, scale, &design);
            }

            let exit_design = self.layout.combined_exit.row(i);
            let eta_exit = exit_design.dot(beta);
            if !eta_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let derivative_exit = derivative_raw[i];
            if !derivative_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            if derivative_exit > guard_threshold {
                let scale = 1.0 / derivative_exit;
                let event_scale = weight * eta_exit.exp() * scale * scale;
                if !event_scale.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                accumulate_symmetric_outer(
                    &mut expected,
                    event_scale,
                    &self.layout.combined_derivative_exit.row(i),
                );
            }
        }

        expected.mapv_inplace(|value| value * -2.0);
        expected += penalty_hessian;
        let mut neg_expected = expected.clone();
        neg_expected.mapv_inplace(|value| -value);
        let mut shift = 0.0;
        let mut attempts = 0usize;
        let max_attempts = 16usize;
        let n = neg_expected.nrows();
        loop {
            let mut shifted = neg_expected.clone();
            if shift > 0.0 {
                for idx in 0..n {
                    shifted[(idx, idx)] += shift;
                }
            }
            if let Ok((_, _, _, _, _, inertia)) = ldlt_rook(&shifted)
                && inertia.1 == 0
                && inertia.2 == 0
            {
                expected = -shifted;
                break;
            }
            attempts += 1;
            if attempts >= max_attempts {
                expected = -shifted;
                break;
            }
            shift = if shift == 0.0 { 1e-8 } else { shift * 10.0 };
        }

        Ok(Some(expected))
    }
}

impl WorkingModelSurvival {
    pub fn update_state(&self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        let expected_dim = self.layout.combined_exit.ncols();
        if beta.len() != expected_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }

        let eta_exit = self.layout.combined_exit.dot(beta);
        let eta_entry = self.layout.combined_entry.dot(beta);
        let derivative_raw = self.layout.combined_derivative_exit.dot(beta);

        let n = eta_exit.len();
        let p = beta.len();
        let mut hessian = Array2::<f64>::zeros((p, p));
        let guard_threshold = self.spec.derivative_guard.max(f64::EPSILON);
        if self.monotonicity.derivative_design.nrows() > 0 {
            if self.monotonicity.derivative_design.ncols() != p
                || self.monotonicity.quadrature_design.ncols() != p
            {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            if self.monotonicity.quadrature_design.nrows()
                != self.monotonicity.derivative_design.nrows()
            {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            if self.monotonicity.grid_ages.len() != self.monotonicity.derivative_design.nrows() {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            let derivative_grid = self.monotonicity.derivative_design.dot(beta);
            for (idx, slope) in derivative_grid.iter().enumerate() {
                if !slope.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                if *slope < MONOTONICITY_TOLERANCE {
                    let age = self.monotonicity.grid_ages[idx];
                    return Err(SurvivalError::MonotonicityViolation {
                        age,
                        derivative: *slope,
                    });
                }
            }
        }
        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);
        let mut log_derivative = Array1::<f64>::zeros(n);
        let mut derivative_scale = Array1::<f64>::zeros(n);
        let mut guarded_derivative_count = 0usize;
        for i in 0..n {
            let value = derivative_raw[i];
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            if value <= guard_threshold {
                log_derivative[i] = guard_threshold.ln();
                derivative_scale[i] = 0.0;
                guarded_derivative_count += 1;
            } else {
                log_derivative[i] = value.ln();
                derivative_scale[i] = 1.0 / value;
            }
        }

        if guarded_derivative_count > 0 {
            let total_grid = self.monotonicity.derivative_design.nrows();
            if total_grid > 0 {
                let ratio = guarded_derivative_count as f64 / total_grid as f64;
                if ratio > DERIVATIVE_GUARD_WARNING_CEILING {
                    warn!(
                        "derivative guard activated for {ratio:.2}% of exit derivatives (threshold {guard_threshold:.3e}); consider adding more knots or stronger smoothing",
                        ratio = ratio * 100.0,
                        guard_threshold = guard_threshold,
                    );
                }
            }
        }

        // Vectorized Hessian accumulation using weighted design matrices
        // H_exit = -X_exit^T * diag(w * h_e) * X_exit
        // H_entry = X_entry^T * diag(w * h_s) * X_entry
        // This replaces O(N) scalar outer products with O(1) BLAS-3 operations

        // Compute weight vectors for vectorized accumulation
        let mut w_exit = Array1::<f64>::zeros(n);
        let mut w_entry = Array1::<f64>::zeros(n);
        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let h_e = h_exit[i];
            let h_s = h_entry[i];
            if !h_e.is_finite() || !h_s.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            // sqrt of absolute weight for sqrt(W)*X formulation
            w_exit[i] = (weight * h_e).sqrt();
            w_entry[i] = (weight * h_s).sqrt();
        }

        let d_vec = self.event_target.mapv(f64::from);
        let w_event = self.sample_weight.as_ref() * &d_vec;
        let w_event_scaled = &w_event * &derivative_scale;
        let w_event_hess = w_event.mapv(f64::sqrt) * &derivative_scale;

        let term1 = &d_vec * (&eta_exit + &log_derivative);
        let term2 = &h_exit - &h_entry;
        let log_likelihood = self.sample_weight.dot(&(term1 - term2));

        let w_h_e = self.sample_weight.as_ref() * &h_exit;
        let w_h_s = self.sample_weight.as_ref() * &h_entry;
        let mut gradient = -self.layout.combined_exit.t().dot(&w_h_e);
        gradient += &self.layout.combined_entry.t().dot(&w_h_s);
        gradient += &self.layout.combined_exit.t().dot(&w_event);
        gradient += &self
            .layout
            .combined_derivative_exit
            .t()
            .dot(&w_event_scaled);

        // Vectorized Hessian: H += -sqrt(w*h_e)*X_exit)^T * (sqrt(w*h_e)*X_exit)
        // Using BLAS GEMM (via ndarray general_mat_mul) for efficiency
        {
            use ndarray::linalg::general_mat_mul;

            // Create weighted design matrices: sqrt(w) * X
            let n_rows = self.layout.combined_exit.nrows();
            let n_cols = self.layout.combined_exit.ncols();

            // Exit term: H -= (sqrt(w*h_e)*X)^T * (sqrt(w*h_e)*X)
            let mut wx_exit = Array2::<f64>::zeros((n_rows, n_cols));
            for i in 0..n_rows {
                let sqrt_w = w_exit[i];
                for j in 0..n_cols {
                    wx_exit[[i, j]] = sqrt_w * self.layout.combined_exit[[i, j]];
                }
            }
            // H -= wx_exit^T * wx_exit via BLAS GEMM
            let mut h_exit_contrib = Array2::<f64>::zeros((p, p));
            general_mat_mul(1.0, &wx_exit.t(), &wx_exit, 0.0, &mut h_exit_contrib);
            hessian -= &h_exit_contrib;

            // Entry term: H += (sqrt(w*h_s)*X)^T * (sqrt(w*h_s)*X)
            let mut wx_entry = Array2::<f64>::zeros((n_rows, n_cols));
            for i in 0..n_rows {
                let sqrt_w = w_entry[i];
                for j in 0..n_cols {
                    wx_entry[[i, j]] = sqrt_w * self.layout.combined_entry[[i, j]];
                }
            }
            // H += wx_entry^T * wx_entry via BLAS GEMM
            general_mat_mul(1.0, &wx_entry.t(), &wx_entry, 1.0, &mut hessian);

            // Event-specific Hessian term: H -= (sqrt(w*d)*scale*D_exit)^T * (sqrt(w*d)*scale*D_exit)
            let mut wx_event = Array2::<f64>::zeros((n_rows, n_cols));
            for i in 0..n_rows {
                let sqrt_w = w_event_hess[i];
                for j in 0..n_cols {
                    wx_event[[i, j]] = sqrt_w * self.layout.combined_derivative_exit[[i, j]];
                }
            }
            general_mat_mul(-1.0, &wx_event.t(), &wx_event, 1.0, &mut hessian);
        }

        gradient.mapv_inplace(|value| value * -2.0);
        hessian.mapv_inplace(|value| value * -2.0);
        let mut deviance = -2.0 * log_likelihood;

        let penalty_gradient = self.layout.penalties.gradient(beta);
        gradient += &penalty_gradient;
        let penalty_hessian = self.layout.penalties.hessian(beta.len());
        hessian += &penalty_hessian;
        let penalty_term = self.layout.penalties.deviance(beta);
        deviance += penalty_term;

        if self.spec.use_expected_information
            && let Some(expected_hessian) =
                self.build_expected_information_hessian(beta, &penalty_hessian)?
        {
            hessian = expected_hessian;
        }

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient,
            hessian,
            deviance,
            penalty_term,
            firth_log_det: None,
            firth_hat_diag: None,
            ridge_used: 0.0,
        })
    }

    pub fn new_flattened(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        Self::from_engine_inputs(inputs, penalties, monotonicity, spec)
    }

    /// Build a survival working model from fully flattened numeric inputs and penalties.
    ///
    /// This constructor is the engine boundary for survival/HMC entrypoints: it accepts
    /// only numeric arrays/matrices and avoids domain training bundle/layout inputs.
    pub fn from_engine_inputs(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        let _ = inputs.event_competing;
        let n = inputs.age_entry.len();
        if n == 0
            || inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.sample_weight.len() != n
            || inputs.x_entry.nrows() != n
            || inputs.x_exit.nrows() != n
            || inputs.x_derivative.nrows() != n
        {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }

        let p = inputs.x_exit.ncols();
        if p == 0 || inputs.x_entry.ncols() != p || inputs.x_derivative.ncols() != p {
            return Err(SurvivalError::DesignDimensionMismatch);
        }

        let mut min_age = f64::INFINITY;
        for i in 0..n {
            let a0 = inputs.age_entry[i];
            let a1 = inputs.age_exit[i];
            if !a0.is_finite() || !a1.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if !(a0 < a1) {
                return Err(SurvivalError::InvalidAgeOrder);
            }
            min_age = min_age.min(a0);
            let evt = inputs.event_target[i];
            if evt > 1 {
                return Err(SurvivalError::InvalidEventFlag);
            }
            let w = inputs.sample_weight[i];
            if !w.is_finite() || w < 0.0 {
                return Err(SurvivalError::InvalidSampleWeight);
            }
        }

        let mut transform = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            transform[[i, i]] = 1.0;
        }
        let layout = SurvivalLayout {
            baseline_entry: inputs.x_entry.to_owned(),
            baseline_exit: inputs.x_exit.to_owned(),
            baseline_derivative_exit: inputs.x_derivative.to_owned(),
            time_varying_entry: None,
            time_varying_exit: None,
            time_varying_derivative_exit: None,
            static_covariates: Array2::<f64>::zeros((n, 0)),
            extra_static_covariates: Array2::<f64>::zeros((n, 0)),
            static_covariate_names: Vec::new(),
            age_transform: AgeTransform {
                minimum_age: min_age,
                delta: 1.0,
            },
            reference_constraint: ReferenceConstraint {
                transform,
                reference_log_age: 0.0,
            },
            penalties,
            combined_entry: inputs.x_entry.to_owned(),
            combined_exit: inputs.x_exit.to_owned(),
            combined_derivative_exit: inputs.x_derivative.to_owned(),
            monotonicity: monotonicity.clone(),
        };

        Self::from_observations(
            layout,
            monotonicity,
            spec,
            inputs.age_entry,
            inputs.age_exit,
            inputs.event_target,
            inputs.sample_weight,
        )
    }
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_state(beta)
            .map_err(|err| EstimationError::InvalidSpecification(err.to_string()))
    }
}

/// Serialized representation of an LDLᵀ factor with Bunch–Kaufman pivoting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LdltFactor {
    pub lower: Array2<f64>,
    pub diag: Array1<f64>,
    pub subdiag: Array1<f64>,
}

/// Serialized permutation metadata captured during factorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermutationDescriptor {
    pub forward: Vec<usize>,
    pub inverse: Vec<usize>,
}

/// Stored factorization metadata for downstream diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HessianFactor {
    Observed {
        factor: LdltFactor,
        permutation: PermutationDescriptor,
        inertia: (usize, usize, usize),
    },
    Expected {
        factor: CholeskyFactor,
    },
}

/// Serialized Cholesky factor for SPD approximations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CholeskyFactor {
    pub lower: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionModelHandle {
    pub reference: String,
    #[serde(default)]
    pub cif_horizons: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: Vec<PenaltyDescriptor>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    #[serde(default = "empty_monotonicity_constraint")]
    pub monotonicity: MonotonicityPenalty,
    #[serde(default)]
    pub interaction_metadata: Vec<InteractionDescriptor>,
    #[serde(default)]
    pub companion_models: Vec<CompanionModelHandle>,
    /// Factorized Hessian used for computing standard errors via delta method.
    ///
    /// **Important:** This is the *penalized* Hessian from P-IRLS fitting. The resulting
    /// standard errors are Bayesian credible intervals (posterior width under the implicit
    /// smoothing prior), NOT frequentist confidence intervals. For heavily penalized terms,
    /// these intervals may be narrower than frequentist expectations.
    pub hessian_factor: Option<HessianFactor>,
    #[serde(default)]
    pub calibrator: Option<()>,
    /// Optional MCMC posterior samples for coefficient uncertainty quantification.
    /// Shape: (n_samples, n_coefficients). Used by crude risk calculations to properly
    /// propagate uncertainty through competing risk models.
    #[serde(default)]
    pub mcmc_samples: Option<Array2<f64>>,
    /// Optional cross-covariance from a companion model to the primary model coefficients.
    /// Stored on the companion artifacts so Crude Risk SE can include cross terms.
    #[serde(default)]
    pub cross_covariance_to_primary: Option<Array2<f64>>,
}

#[derive(Clone)]
pub struct CovariateViews<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView2<'a, f64>,
    pub static_covariates: ArrayView2<'a, f64>,
}

/// Prediction inputs referencing existing arrays.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}

/// Resolve a companion model declared in the survival artifacts.
pub fn resolve_companion_model<'a, 'b>(
    artifacts: &'a SurvivalModelArtifacts,
    reference: &str,
    registry: &'b HashMap<String, SurvivalModelArtifacts>,
) -> Result<(&'a CompanionModelHandle, &'b SurvivalModelArtifacts), SurvivalError> {
    let handle = artifacts
        .companion_models
        .iter()
        .find(|handle| handle.reference == reference)
        .ok_or_else(|| SurvivalError::UnknownCompanionModelHandle {
            reference: reference.to_string(),
        })?;

    let companion =
        registry
            .get(reference)
            .ok_or_else(|| SurvivalError::CompanionModelUnavailable {
                reference: reference.to_string(),
            })?;

    Ok((handle, companion))
}

/// Determine the competing-risk CIF at a given horizon.
pub fn competing_cif_value(
    horizon: f64,
    covariates: &Array1<f64>,
    explicit: Option<f64>,
    companion: Option<(&CompanionModelHandle, &SurvivalModelArtifacts)>,
) -> Result<f64, SurvivalError> {
    if let Some(value) = explicit {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(SurvivalError::InvalidCompetingCif { value });
        }
        return Ok(value);
    }

    if let Some((handle, model)) = companion {
        if !handle.cif_horizons.is_empty()
            && handle
                .cif_horizons
                .iter()
                .all(|&candidate| (candidate - horizon).abs() > COMPANION_HORIZON_TOLERANCE)
        {
            return Err(SurvivalError::CompanionModelMissingHorizon {
                reference: handle.reference.clone(),
                horizon,
            });
        }
        return cumulative_incidence(horizon, covariates, model);
    }

    Err(SurvivalError::MissingCompanionCifData)
}

fn covariate_label(layout: &CovariateLayout, index: usize) -> String {
    layout
        .column_names
        .get(index)
        .cloned()
        .unwrap_or_else(|| format!("column_{index}"))
}

/// Clamp covariates to training ranges, logging warnings for extrapolation.
/// Returns the clamped values. This is production-safe: out-of-range values
/// trigger linear extrapolation via splines rather than hard failures.
fn clamp_covariates_to_ranges(
    covariates: ArrayView1<f64>,
    layout: &CovariateLayout,
) -> Result<Array1<f64>, SurvivalError> {
    let expected = layout.column_names.len();
    if layout.ranges.is_empty() {
        return Err(SurvivalError::MissingCovariateRanges { expected });
    }
    if layout.ranges.len() != expected {
        return Err(SurvivalError::CovariateRangeLengthMismatch {
            expected,
            actual: layout.ranges.len(),
        });
    }
    if covariates.len() != expected {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }

    let mut clamped = covariates.to_owned();

    for (idx, value) in clamped.iter_mut().enumerate() {
        if !value.is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }
        let range = &layout.ranges[idx];
        if !range.min.is_finite() && !range.max.is_finite() {
            // Both bounds are infinite, nothing to clamp.
            continue;
        }
        if range.min.is_nan()
            || range.max.is_nan()
            || (range.min.is_finite() && range.max.is_finite() && range.min > range.max)
        {
            return Err(SurvivalError::InvalidCovariateRange {
                column: covariate_label(layout, idx),
                index: idx,
                min: range.min,
                max: range.max,
            });
        }

        let original = *value;
        if range.min.is_finite() && *value < range.min {
            *value = range.min;
            log::debug!(
                "Covariate '{}' clamped from {:.4} to minimum {:.4} (extrapolation)",
                covariate_label(layout, idx),
                original,
                range.min
            );
        }
        if range.max.is_finite() && *value > range.max {
            *value = range.max;
            log::debug!(
                "Covariate '{}' clamped from {:.4} to maximum {:.4} (extrapolation)",
                covariate_label(layout, idx),
                original,
                range.max
            );
        }
    }
    Ok(clamped)
}

/// Reconstruct the design row at a given age for prediction.
/// NOTE: This allocates a new SurvivalScratch per call. For batch prediction,
/// use `design_row_at_age_scratch` with a pre-allocated scratch buffer.
pub fn design_row_at_age(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<Array1<f64>, SurvivalError> {
    let (design, _) = design_and_derivative_at_age(age, covariates, artifacts)?;
    Ok(design)
}

/// Non-allocating version of design_row_at_age for batch prediction.
/// Reuses the provided scratch buffer and writes into the output array.
pub fn design_row_at_age_scratch(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
    out: &mut Array1<f64>,
    scratch: &mut SurvivalScratch,
) -> Result<(), SurvivalError> {
    let mut deriv_dummy = Array1::zeros(out.len());
    design_and_derivative_at_age_scratch(age, covariates, artifacts, out, &mut deriv_dummy, scratch)
}

/// Compute both design row and its time derivative at a given age.
fn design_and_derivative_at_age(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<(Array1<f64>, Array1<f64>), SurvivalError> {
    // This function allocates! Use design_and_derivative_at_age_scratch for performance.
    let total_cols = artifacts.coefficients.len();

    let max_basis = artifacts.age_basis.knot_vector.len().max(100);
    let max_pgs = 10;
    let mut scratch = SurvivalScratch::new(artifacts.age_basis.degree, max_basis, max_pgs);

    let mut design = Array1::zeros(total_cols);
    let mut design_deriv = Array1::zeros(total_cols);

    design_and_derivative_at_age_scratch(
        age,
        covariates,
        artifacts,
        &mut design,
        &mut design_deriv,
        &mut scratch,
    )?;

    Ok((design, design_deriv))
}

/// Non-allocating version of design_and_derivative_at_age.
fn design_and_derivative_at_age_scratch(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
    out_design: &mut Array1<f64>,
    out_deriv: &mut Array1<f64>,
    scratch: &mut SurvivalScratch,
) -> Result<(), SurvivalError> {
    let expected_covs = artifacts.static_covariate_layout.column_names.len();
    if covariates.len() != expected_covs {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }
    // Clamp covariates to training ranges (production-safe)
    let clamped_covs = clamp_covariates_to_ranges(covariates, &artifacts.static_covariate_layout)?;

    let log_age = artifacts.age_transform.transform(age)?;
    let age_deriv_factor = artifacts.age_transform.derivative_factor(age)?;

    // 1. Eval unconstrained basis and derivative
    let k = artifacts.reference_constraint.transform.nrows(); // This is num basis functions
    let k_minus_1 = artifacts.reference_constraint.transform.ncols();

    // Resize scratch if needed (unlikely given initialization)
    if scratch.basis_raw.len() < k {
        return Err(SurvivalError::Basis(
            BasisError::ConstraintMatrixRowMismatch {
                basis_rows: k,
                constraint_rows: scratch.basis_raw.len(),
            },
        ));
    }

    let basis_raw_buff = &mut scratch.basis_raw[..k];
    let deriv_raw_buff = &mut scratch.derivative_raw[..k];
    let plus_buff = &mut scratch.basis_plus[..k];
    let minus_buff = &mut scratch.basis_minus[..k];

    evaluate_basis_and_derivative_scalar_into(
        log_age,
        &artifacts.age_basis,
        basis_raw_buff,
        deriv_raw_buff,
        &mut scratch.spline_scratch,
        plus_buff,
        minus_buff,
    )?;

    // 2. Constrain
    // constrained = basis . transform (1 x k . k x k-1 -> 1 x k-1)
    let constrained_basis_buff = &mut scratch.constrained_basis[..k_minus_1];
    vec_mat_mul_into(
        basis_raw_buff,
        &artifacts.reference_constraint.transform,
        constrained_basis_buff,
    );

    let constrained_deriv_buff = &mut scratch.constrained_derivative[..k_minus_1];
    vec_mat_mul_into(
        deriv_raw_buff,
        &artifacts.reference_constraint.transform,
        constrained_deriv_buff,
    );

    let baseline_cols = k_minus_1;
    let static_cols = expected_covs;
    let total_cols = artifacts.coefficients.len();

    if baseline_cols + static_cols > total_cols {
        return Err(SurvivalError::InvalidTimeVaryingLayout);
    }

    // Write to output arrays
    // 1. Baseline
    let design_base = out_design.as_slice_mut().expect("contiguous array");
    let deriv_base = out_deriv.as_slice_mut().expect("contiguous array");

    // Initialize with zeros
    design_base.fill(0.0);
    deriv_base.fill(0.0);

    design_base[..baseline_cols].copy_from_slice(constrained_basis_buff);

    // Apply chain rule to derivative: d/dt = d/d(log t) * factor
    for i in 0..baseline_cols {
        deriv_base[i] = constrained_deriv_buff[i] * age_deriv_factor;
    }

    // 2. Time-Varying
    let time_cols = total_cols - baseline_cols - static_cols;
    if time_cols > 0 {
        let time_basis = artifacts
            .time_varying_basis
            .as_ref()
            .ok_or(SurvivalError::MissingTimeVaryingBasis)?;

        // Find descriptor
        let descriptor = artifacts
            .interaction_metadata
            .iter()
            .find(|meta| {
                meta.column_range.start == baseline_cols
                    && meta.column_range.end == baseline_cols + time_cols
            })
            .ok_or(SurvivalError::MissingInteractionMetadata)?;

        let pgs_idx = artifacts
            .static_covariate_layout
            .column_names
            .iter()
            .position(|name| name == "pgs")
            .ok_or(SurvivalError::MissingPgsCovariate)?;
        let mut pgs_value = clamped_covs[pgs_idx];

        // Clamp PGS to time-varying range instead of erroring
        if let Some(range) = descriptor.value_ranges.first() {
            if range.min.is_finite() && pgs_value < range.min {
                log::debug!(
                    "PGS clamped from {:.4} to time-varying minimum {:.4}",
                    pgs_value,
                    range.min
                );
                pgs_value = range.min;
            }
            if range.max.is_finite() && pgs_value > range.max {
                log::debug!(
                    "PGS clamped from {:.4} to time-varying maximum {:.4}",
                    pgs_value,
                    range.max
                );
                pgs_value = range.max;
            }
        }

        // Eval PGS basis
        let pgs_basis_dim = time_basis.knot_vector.len() - time_basis.degree - 1;
        if scratch.pgs_basis_buff.len() < pgs_basis_dim {
            // Should not happen if sized correctly
            return Err(SurvivalError::InvalidTimeVaryingLayout);
        }
        let pgs_buff = &mut scratch.pgs_basis_buff[..pgs_basis_dim];

        evaluate_bspline_basis_scalar(
            pgs_value,
            time_basis.knot_vector.view(),
            time_basis.degree,
            pgs_buff,
            &mut scratch.spline_scratch,
        )
        .map_err(|_| SurvivalError::InvalidTimeVaryingLayout)?; // Map basis error

        // The first column of PGS basis is usually dropped/centered, need to replicate logic.
        // Logic in `build_survival_layout`: `pgs_basis_full.slice(s![.., 1..])`.
        // So we skip index 0.
        if pgs_basis_dim <= 1 {
            return Err(SurvivalError::InvalidTimeVaryingLayout);
        }
        let pgs_reduced_len = pgs_basis_dim - 1;
        let pgs_reduced = &mut pgs_buff[1..];

        // Centering
        if let Some(centering) = &descriptor.centering {
            if centering.offsets.len() != pgs_reduced_len {
                return Err(SurvivalError::InvalidTimeVaryingLayout);
            }
            for i in 0..pgs_reduced_len {
                pgs_reduced[i] -= centering.offsets[i];
            }
        }

        // Tensor Product
        // baseline (1 x baseline_cols) x pgs (1 x pgs_reduced_len) -> (1 x baseline_cols*pgs_len)
        // Flattened: out[i*pgs_len + j] = base[i] * pgs[j]
        let tensor_len = baseline_cols * pgs_reduced_len;
        if tensor_len != time_cols {
            return Err(SurvivalError::InvalidTimeVaryingLayout);
        }

        let dest_design = &mut design_base[baseline_cols..baseline_cols + time_cols];
        tensor_product_into(constrained_basis_buff, pgs_reduced, dest_design);

        let dest_deriv = &mut deriv_base[baseline_cols..baseline_cols + time_cols];
        tensor_product_into(constrained_deriv_buff, pgs_reduced, dest_deriv);

        // Apply derivative factor
        for x in dest_deriv.iter_mut() {
            *x *= age_deriv_factor;
        }
    }

    // 3. Static - use clamped values
    let dest_static = &mut design_base[baseline_cols + time_cols..];
    for (i, &val) in clamped_covs.iter().enumerate() {
        dest_static[i] = val;
    }
    // Derivatives for static cols are 0.0 (already set)

    Ok(())
}

struct SurvivalDesignCache {
    baseline_cols: usize,
    time_cols: usize,
    static_covs: Vec<f64>,
    pgs_reduced: Vec<f64>,
}

impl SurvivalDesignCache {
    fn new(
        covariates: ArrayView1<'_, f64>,
        artifacts: &SurvivalModelArtifacts,
        scratch: &mut SurvivalScratch,
    ) -> Result<Self, SurvivalError> {
        let expected_covs = artifacts.static_covariate_layout.column_names.len();
        if covariates.len() != expected_covs {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        // Clamp covariates to training ranges (production-safe)
        let clamped_covs =
            clamp_covariates_to_ranges(covariates, &artifacts.static_covariate_layout)?;

        let baseline_cols = artifacts.reference_constraint.transform.ncols();
        let total_cols = artifacts.coefficients.len();
        if baseline_cols + expected_covs > total_cols {
            return Err(SurvivalError::InvalidTimeVaryingLayout);
        }
        let time_cols = total_cols - baseline_cols - expected_covs;

        let mut pgs_reduced = Vec::new();
        if time_cols > 0 {
            let time_basis = artifacts
                .time_varying_basis
                .as_ref()
                .ok_or(SurvivalError::MissingTimeVaryingBasis)?;

            let descriptor = artifacts
                .interaction_metadata
                .iter()
                .find(|meta| {
                    meta.column_range.start == baseline_cols
                        && meta.column_range.end == baseline_cols + time_cols
                })
                .ok_or(SurvivalError::MissingInteractionMetadata)?;

            let pgs_idx = artifacts
                .static_covariate_layout
                .column_names
                .iter()
                .position(|name| name == "pgs")
                .ok_or(SurvivalError::MissingPgsCovariate)?;
            let mut pgs_value = clamped_covs[pgs_idx];

            // Clamp PGS to time-varying range instead of erroring
            if let Some(range) = descriptor.value_ranges.first() {
                if range.min.is_finite() && pgs_value < range.min {
                    log::debug!(
                        "PGS clamped from {:.4} to time-varying minimum {:.4}",
                        pgs_value,
                        range.min
                    );
                    pgs_value = range.min;
                }
                if range.max.is_finite() && pgs_value > range.max {
                    log::debug!(
                        "PGS clamped from {:.4} to time-varying maximum {:.4}",
                        pgs_value,
                        range.max
                    );
                    pgs_value = range.max;
                }
            }

            let pgs_basis_dim = time_basis.knot_vector.len() - time_basis.degree - 1;
            if scratch.pgs_basis_buff.len() < pgs_basis_dim {
                return Err(SurvivalError::InvalidTimeVaryingLayout);
            }
            let pgs_buff = &mut scratch.pgs_basis_buff[..pgs_basis_dim];
            evaluate_bspline_basis_scalar(
                pgs_value,
                time_basis.knot_vector.view(),
                time_basis.degree,
                pgs_buff,
                &mut scratch.spline_scratch,
            )
            .map_err(|_| SurvivalError::InvalidTimeVaryingLayout)?;

            if pgs_basis_dim <= 1 {
                return Err(SurvivalError::InvalidTimeVaryingLayout);
            }
            pgs_reduced.extend_from_slice(&pgs_buff[1..]);

            if let Some(centering) = &descriptor.centering {
                if centering.offsets.len() != pgs_reduced.len() {
                    return Err(SurvivalError::InvalidTimeVaryingLayout);
                }
                for (dst, &offset) in pgs_reduced.iter_mut().zip(centering.offsets.iter()) {
                    *dst -= offset;
                }
            }

            let tensor_len = baseline_cols * pgs_reduced.len();
            if tensor_len != time_cols {
                return Err(SurvivalError::InvalidTimeVaryingLayout);
            }
        }

        Ok(SurvivalDesignCache {
            baseline_cols,
            time_cols,
            static_covs: clamped_covs.to_vec(),
            pgs_reduced,
        })
    }
}

fn design_and_derivative_at_age_cached(
    age: f64,
    artifacts: &SurvivalModelArtifacts,
    cache: &SurvivalDesignCache,
    out_design: &mut Array1<f64>,
    out_deriv: &mut Array1<f64>,
    scratch: &mut SurvivalScratch,
) -> Result<(), SurvivalError> {
    let expected_covs = artifacts.static_covariate_layout.column_names.len();
    if cache.static_covs.len() != expected_covs {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }

    let log_age = artifacts.age_transform.transform(age)?;
    let age_deriv_factor = artifacts.age_transform.derivative_factor(age)?;

    let k = artifacts.reference_constraint.transform.nrows();
    let k_minus_1 = artifacts.reference_constraint.transform.ncols();
    if scratch.basis_raw.len() < k {
        return Err(SurvivalError::Basis(
            BasisError::ConstraintMatrixRowMismatch {
                basis_rows: k,
                constraint_rows: scratch.basis_raw.len(),
            },
        ));
    }

    let basis_raw_buff = &mut scratch.basis_raw[..k];
    let deriv_raw_buff = &mut scratch.derivative_raw[..k];
    let plus_buff = &mut scratch.basis_plus[..k];
    let minus_buff = &mut scratch.basis_minus[..k];

    evaluate_basis_and_derivative_scalar_into(
        log_age,
        &artifacts.age_basis,
        basis_raw_buff,
        deriv_raw_buff,
        &mut scratch.spline_scratch,
        plus_buff,
        minus_buff,
    )?;

    let constrained_basis_buff = &mut scratch.constrained_basis[..k_minus_1];
    vec_mat_mul_into(
        basis_raw_buff,
        &artifacts.reference_constraint.transform,
        constrained_basis_buff,
    );

    let constrained_deriv_buff = &mut scratch.constrained_derivative[..k_minus_1];
    vec_mat_mul_into(
        deriv_raw_buff,
        &artifacts.reference_constraint.transform,
        constrained_deriv_buff,
    );

    let baseline_cols = cache.baseline_cols;
    let static_cols = expected_covs;
    let total_cols = artifacts.coefficients.len();
    if baseline_cols + static_cols > total_cols {
        return Err(SurvivalError::InvalidTimeVaryingLayout);
    }
    let time_cols = cache.time_cols;

    let design_base = out_design.as_slice_mut().expect("contiguous array");
    let deriv_base = out_deriv.as_slice_mut().expect("contiguous array");
    design_base.fill(0.0);
    deriv_base.fill(0.0);

    design_base[..baseline_cols].copy_from_slice(constrained_basis_buff);
    for i in 0..baseline_cols {
        deriv_base[i] = constrained_deriv_buff[i] * age_deriv_factor;
    }

    if time_cols > 0 {
        let dest_design = &mut design_base[baseline_cols..baseline_cols + time_cols];
        tensor_product_into(constrained_basis_buff, &cache.pgs_reduced, dest_design);

        let dest_deriv = &mut deriv_base[baseline_cols..baseline_cols + time_cols];
        tensor_product_into(constrained_deriv_buff, &cache.pgs_reduced, dest_deriv);
        for x in dest_deriv.iter_mut() {
            *x *= age_deriv_factor;
        }
    }

    let dest_static = &mut design_base[baseline_cols + time_cols..];
    for (dst, &val) in dest_static.iter_mut().zip(cache.static_covs.iter()) {
        *dst = val;
    }

    Ok(())
}

/// Evaluate the cumulative hazard at a given age.
pub fn cumulative_hazard(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let design = design_row_at_age(age, covariates.view(), artifacts)?;
    let eta = design.dot(&artifacts.coefficients);
    Ok(eta.exp())
}

pub fn cumulative_hazard_with_coeffs(
    age: f64,
    covariates: ArrayView1<'_, f64>,
    artifacts: &SurvivalModelArtifacts,
    coeffs: ArrayView1<'_, f64>,
) -> Result<f64, SurvivalError> {
    let design = design_row_at_age(age, covariates, artifacts)?;
    let eta = design.dot(&coeffs);
    Ok(eta.exp())
}

pub fn cumulative_incidence(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let h = cumulative_hazard(age, covariates, artifacts)?;
    Ok(-(-h).exp_m1())
}

/// Compute Net Risk (Hypothetical risk of disease assuming no competing events).
/// P(Event in (t0, t1] | Survival to t0, No competing risks)
/// Formula: 1 - exp( - (H_dis(t1) - H_dis(t0)) )
pub fn conditional_absolute_risk(
    t0: f64,
    t1: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let h0 = cumulative_hazard(t0, covariates, artifacts)?;
    let h1 = cumulative_hazard(t1, covariates, artifacts)?;
    // Ensure monotonicity numerically
    let delta_h = (h1 - h0).max(0.0);
    Ok(-(-delta_h).exp_m1())
}

/// Compute Gauss-Legendre quadrature nodes and weights for N points on interval [-1, 1].
fn compute_gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    let mut nodes_weights = Vec::with_capacity(n);
    let m = n.div_ceil(2);
    let pi = std::f64::consts::PI;

    for i in 0..m {
        // Initial guess for the root using asymptotic approximation
        let mut z = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;

        // Newton's method
        for _ in 0..100 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            // p1 is P_n(z)
            // pp is P'_n(z) calculated using derivative relation
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() < 1e-14 {
                break;
            }
        }

        let x = z;
        let w = 2.0 / ((1.0 - z * z) * pp * pp);

        if !n.is_multiple_of(2) && i == m - 1 {
            nodes_weights.push((0.0, w));
        } else {
            nodes_weights.push((-x, w));
            nodes_weights.push((x, w));
        }
    }
    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    nodes_weights
}

/// Returns cached Gauss-Legendre quadrature nodes and weights for N=40 on interval [-1, 1].
/// High precision is required for accurate integration of spline products.
/// Previously hardcoded as N=30 but values corresponded to N=40.
fn gauss_legendre_quadrature() -> &'static [(f64, f64)] {
    static CACHE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    CACHE.get_or_init(|| compute_gauss_legendre_nodes(40))
}

/// Compute Crude Risk (Real-world risk) via numerical integration.
/// Result of crude risk calculation with gradients for uncertainty quantification.
#[derive(Debug, Clone)]
pub struct CrudeRiskResult {
    /// The crude risk value (probability of event in time window)
    pub risk: f64,
    /// Gradient of risk with respect to disease model coefficients
    pub disease_gradient: Array1<f64>,
    /// Gradient of risk with respect to mortality model coefficients
    pub mortality_gradient: Array1<f64>,
}

/// Compute the Cumulative Incidence Function (Crude Risk) with competing mortality.
/// P(Event in (t0, t1] | Survival to t0, Accounting for Competing Mortality)
/// Formula: Integral_{t0}^{t1} h_dis(u) * S_total(u|t0) du
///
/// Returns gradients for BOTH disease and mortality models to enable proper
/// uncertainty propagation via the delta method.
pub fn calculate_crude_risk_quadrature<'a>(
    t0: f64,
    t1: f64,
    covariates: ArrayView1<'a, f64>,
    disease_model: &'a SurvivalModelArtifacts,
    mortality_model: &'a SurvivalModelArtifacts,
    disease_coeffs: Option<ArrayView1<'a, f64>>,
    mortality_coeffs: Option<ArrayView1<'a, f64>>,
) -> Result<CrudeRiskResult, SurvivalError> {
    let coeff_len_d = disease_model.coefficients.len();
    let coeff_len_m = mortality_model.coefficients.len();

    if t1 <= t0 {
        return Ok(CrudeRiskResult {
            risk: 0.0,
            disease_gradient: Array1::zeros(coeff_len_d),
            mortality_gradient: Array1::zeros(coeff_len_m),
        });
    }

    // 1. Collect all knots from both models within range [t0, t1]
    let mut breakpoints = Vec::new();
    breakpoints.push(t0);
    breakpoints.push(t1);

    // Helper to add valid knots
    let mut add_knots = |model: &SurvivalModelArtifacts| {
        let age_trans = &model.age_transform;
        for &k_log in model.age_basis.knot_vector.iter() {
            // Convert log-knot back to age: k_log = ln(age - min + delta)
            // age = exp(k_log) + min - delta
            let age_k = k_log.exp() + age_trans.minimum_age - age_trans.delta;
            if age_k > t0 && age_k < t1 {
                breakpoints.push(age_k);
            }
        }
    };
    add_knots(disease_model);
    add_knots(mortality_model);

    // Sort and deduplicate
    breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    breakpoints.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    // 2. Pre-calculate baselines at t0 to normalize conditional survival
    // H(u|t0) = H(u) - H(t0)
    let coeff_d = match disease_coeffs {
        Some(c) => {
            if c.len() != coeff_len_d {
                return Err(SurvivalError::CoefficientDimensionMismatch {
                    expected: coeff_len_d,
                    actual: c.len(),
                });
            }
            c
        }
        None => disease_model.coefficients.view(),
    };
    let coeff_m = match mortality_coeffs {
        Some(c) => {
            if c.len() != coeff_len_m {
                return Err(SurvivalError::CoefficientDimensionMismatch {
                    expected: coeff_len_m,
                    actual: c.len(),
                });
            }
            c
        }
        None => mortality_model.coefficients.view(),
    };

    let h_dis_t0 = cumulative_hazard_with_coeffs(t0, covariates, disease_model, coeff_d)?;
    let h_mor_t0 = cumulative_hazard_with_coeffs(t0, covariates, mortality_model, coeff_m)?;

    // Pre-allocate scratch for hot loop
    // Determine max sizes
    let max_basis = disease_model
        .age_basis
        .knot_vector
        .len()
        .max(mortality_model.age_basis.knot_vector.len());
    let max_pgs = 20; // Sufficient for most splines

    let mut scratch_d = SurvivalScratch::new(disease_model.age_basis.degree, max_basis, max_pgs);

    let mut scratch_m = SurvivalScratch::new(mortality_model.age_basis.degree, max_basis, max_pgs);

    let mut design_d = Array1::zeros(coeff_len_d);
    let mut deriv_d = Array1::zeros(coeff_len_d);

    let mut design_m = Array1::zeros(coeff_len_m);
    let mut deriv_m = Array1::zeros(coeff_len_m);

    let disease_cache = SurvivalDesignCache::new(covariates, disease_model, &mut scratch_d)?;
    let mortality_cache = SurvivalDesignCache::new(covariates, mortality_model, &mut scratch_m)?;

    // 3. Integrate segment by segment
    let mut total_risk = 0.0;
    let mut disease_gradient = Array1::zeros(coeff_len_d);
    let mut mortality_gradient = Array1::zeros(coeff_len_m);
    let nodes_weights = gauss_legendre_quadrature();

    // Design row at entry age for gradient term involving H_D(t0) X(t0)
    design_and_derivative_at_age_cached(
        t0,
        disease_model,
        &disease_cache,
        &mut design_d,
        &mut deriv_d,
        &mut scratch_d,
    )?;
    let design_d_t0 = design_d.clone();

    // Design row at entry age for mortality model (for H_M(t0) gradient)
    design_and_derivative_at_age_cached(
        t0,
        mortality_model,
        &mortality_cache,
        &mut design_m,
        &mut deriv_m,
        &mut scratch_m,
    )?;
    let design_m_t0 = design_m.clone();

    for i in 0..breakpoints.len() - 1 {
        let a = breakpoints[i];
        let b = breakpoints[i + 1];
        let center = 0.5 * (b + a);
        let half_width = 0.5 * (b - a);

        for &(x, w) in nodes_weights {
            let u = center + half_width * x;

            // Eval Disease: h(u), H(u)
            design_and_derivative_at_age_cached(
                u,
                disease_model,
                &disease_cache,
                &mut design_d,
                &mut deriv_d,
                &mut scratch_d,
            )?;

            let eta_d = design_d.dot(&coeff_d);
            let slope_d = deriv_d.dot(&coeff_d);
            let hazard_d = eta_d.exp();
            let inst_hazard_d = (hazard_d * slope_d).max(0.0);

            // Eval Mortality: H(u)
            design_and_derivative_at_age_cached(
                u,
                mortality_model,
                &mortality_cache,
                &mut design_m,
                &mut deriv_m,
                &mut scratch_m,
            )?;
            let eta_m = design_m.dot(&coeff_m);
            let hazard_m = eta_m.exp();

            // Conditional Cumulative Hazards
            // Ensure non-negative via max(0.0) to handle potential spline wiggle near t0
            let h_dis_cond = (hazard_d - h_dis_t0).max(0.0);
            let h_mor_cond = (hazard_m - h_mor_t0).max(0.0);

            // Survival Function S_total(u|t0)
            let s_total = (-(h_dis_cond + h_mor_cond)).exp();

            // Accumulate risk
            // Integral += h_dis(u) * S_total(u) * du
            // weight scaled by half_width due to change of variables
            total_risk += w * inst_hazard_d * s_total * half_width;

            // ========== Disease Gradient ==========
            // ∂Risk/∂β_d = ∫ (∂h_d/∂β_d * S - h_d * S * ∂H_d/∂β_d) du
            if inst_hazard_d > 0.0 {
                let weight = w * s_total * half_width;
                let mut grad_contrib = design_d.mapv(|x| inst_hazard_d * (1.0 - hazard_d) * x);
                grad_contrib.zip_mut_with(&deriv_d, |g, &d| {
                    *g += hazard_d * d;
                });
                grad_contrib.zip_mut_with(&design_d_t0, |g, &x0| {
                    *g += inst_hazard_d * h_dis_t0 * x0;
                });
                grad_contrib.mapv_inplace(|v| v * weight);
                disease_gradient += &grad_contrib;
            }

            // ========== Mortality Gradient ==========
            // ∂Risk/∂β_m = -∫ h_d(u) * S_total(u) * ∂H_m(u|t0)/∂β_m du
            // where ∂H_m(u|t0)/∂β_m = H_m(u) * X_m(u) - H_m(t0) * X_m(t0)
            // Since H_m = exp(η_m), ∂H_m/∂β_m = H_m * X_m
            if inst_hazard_d > 0.0 && hazard_m > 0.0 {
                let weight = w * inst_hazard_d * s_total * half_width;
                // ∂H_m(u|t0)/∂β_m = H_m(u) * X_m(u) - H_m(t0) * X_m(t0)
                let mut mort_grad_contrib = design_m.mapv(|x| -weight * hazard_m * x);
                mort_grad_contrib.zip_mut_with(&design_m_t0, |g, &x0| {
                    *g += weight * h_mor_t0 * x0;
                });
                mortality_gradient += &mort_grad_contrib;
            }
        }
    }

    Ok(CrudeRiskResult {
        risk: total_risk,
        disease_gradient,
        mortality_gradient,
    })
}

pub fn survival_calibrator_features(
    risks: &Array1<f64>,
    logit_design: &Array2<f64>,
    hessian_factor: Option<&HessianFactor>,
) -> Result<Array2<f64>, SurvivalError> {
    let n = risks.len();
    let mut features = Array2::zeros((n, 3)); // risk, se, 0.0

    // Column 0: logit(risk)
    for i in 0..n {
        let p = risks[i].max(1e-12).min(1.0 - 1e-12);
        features[[i, 0]] = (p / (1.0 - p)).ln();
    }

    // Column 1: SE of logit risk
    if let Some(factor) = hessian_factor {
        let se = delta_method_standard_errors(factor, logit_design)?;
        features.column_mut(1).assign(&se);
    }

    Ok(features)
}

pub fn delta_method_standard_errors(
    factor: &HessianFactor,
    jacobian: &Array2<f64>,
) -> Result<Array1<f64>, SurvivalError> {
    let n = jacobian.nrows();
    let mut se = Array1::zeros(n);

    match factor {
        HessianFactor::Expected { factor: chol } => {
            let l = &chol.lower;
            let dim = l.nrows();
            let mut y = Array1::zeros(dim);

            for i in 0..n {
                let row = jacobian.row(i);
                // Forward solve L y = row^T
                for r in 0..dim {
                    let mut sum = 0.0;
                    for c in 0..r {
                        sum += l[[r, c]] * y[c];
                    }
                    let diag = l[[r, r]];
                    if diag == 0.0 {
                        return Err(SurvivalError::HessianSingular);
                    }
                    y[r] = (row[r] - sum) / diag;
                }
                se[i] = y.dot(&y).sqrt();
            }
        }
        HessianFactor::Observed {
            factor,
            permutation,
            ..
        } => {
            let dim = factor.lower.nrows();
            let mut y = Array1::<f64>::zeros(dim);
            let mut g_perm = Array1::<f64>::zeros(dim);

            // Precompute D^-1 structure into diagonal and off-diagonal arrays for efficiency
            // inv_diag stores (D^-1)_{ii}
            // inv_offdiag stores (D^-1)_{i, i+1}
            let mut inv_diag = Array1::<f64>::zeros(dim);
            let mut inv_offdiag = Array1::<f64>::zeros(dim);

            let mut k = 0;
            while k < dim {
                if k + 1 < dim && factor.subdiag[k].abs() > 1e-12 {
                    // 2x2 block
                    let d11 = factor.diag[k];
                    let d22 = factor.diag[k + 1];
                    let d12 = factor.subdiag[k];
                    let det = d11 * d22 - d12 * d12;
                    if det == 0.0 {
                        return Err(SurvivalError::HessianSingular);
                    }
                    // Inverse of [[d11, d12], [d12, d22]] is [[d22, -d12], [-d12, d11]] / det
                    inv_diag[k] = d22 / det;
                    inv_diag[k + 1] = d11 / det;
                    inv_offdiag[k] = -d12 / det;
                    k += 2;
                } else {
                    // 1x1 block
                    let d = factor.diag[k];
                    if d == 0.0 {
                        return Err(SurvivalError::HessianSingular);
                    }
                    inv_diag[k] = 1.0 / d;
                    k += 1;
                }
            }

            for i in 0..n {
                let row = jacobian.row(i);

                // 1. Permute the Jacobian row (gradient)
                for k in 0..dim {
                    g_perm[k] = row[permutation.forward[k]];
                }

                // 2. Forward Solve L y = g_perm
                // L is unit lower triangular. y_r = g_perm_r - sum_{c<r} L_{rc} y_c
                // Optimization: Use dot product for inner loop
                for r in 0..dim {
                    // Safe slicing: factor.lower is dim x dim, y is dim.
                    // We need dot product of row r of L (up to r) and y (up to r).
                    let sum = factor.lower.row(r).slice(s![..r]).dot(&y.slice(s![..r]));
                    y[r] = g_perm[r] - sum;
                }

                // 3. Compute quadratic form val = y^T D^-1 y
                // Since D^-1 is block tridiagonal (max block size 2), this is:
                // sum_i (D^-1)_ii y_i^2 + 2 * sum_i (D^-1)_{i,i+1} y_i y_{i+1}
                let mut val = 0.0;
                for k in 0..dim {
                    val += y[k] * y[k] * inv_diag[k];
                }
                for k in 0..dim - 1 {
                    if inv_offdiag[k] != 0.0 {
                        val += 2.0 * y[k] * y[k + 1] * inv_offdiag[k];
                    }
                }

                // 4. Standard Error
                se[i] = if val > 0.0 { val.sqrt() } else { 0.0 };
            }
        }
    }
    Ok(se)
}

pub fn solve_hessian_matrix(
    factor: &HessianFactor,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, SurvivalError> {
    let n = rhs.ncols();
    let mut out = Array2::<f64>::zeros(rhs.dim());
    for j in 0..n {
        let col = rhs.column(j).to_owned();
        let solved = solve_hessian_vector(factor, &col)?;
        out.column_mut(j).assign(&solved);
    }
    Ok(out)
}

fn solve_hessian_vector(
    factor: &HessianFactor,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SurvivalError> {
    match factor {
        HessianFactor::Expected { factor: chol } => {
            let l = &chol.lower;
            let dim = l.nrows();
            let mut y = Array1::<f64>::zeros(dim);
            for r in 0..dim {
                let mut sum = 0.0;
                for c in 0..r {
                    sum += l[[r, c]] * y[c];
                }
                let diag = l[[r, r]];
                if diag == 0.0 {
                    return Err(SurvivalError::HessianSingular);
                }
                y[r] = (rhs[r] - sum) / diag;
            }
            let mut x = Array1::<f64>::zeros(dim);
            for r_rev in 0..dim {
                let r = dim - 1 - r_rev;
                let mut sum = 0.0;
                for c in (r + 1)..dim {
                    sum += l[[c, r]] * x[c];
                }
                let diag = l[[r, r]];
                if diag == 0.0 {
                    return Err(SurvivalError::HessianSingular);
                }
                x[r] = (y[r] - sum) / diag;
            }
            Ok(x)
        }
        HessianFactor::Observed {
            factor,
            permutation,
            ..
        } => {
            let dim = factor.lower.nrows();
            let mut rhs_perm = Array1::<f64>::zeros(dim);
            for r in 0..dim {
                rhs_perm[r] = rhs[permutation.forward[r]];
            }
            let mut y = Array1::<f64>::zeros(dim);
            for r in 0..dim {
                let mut sum = 0.0;
                for c in 0..r {
                    sum += factor.lower[[r, c]] * y[c];
                }
                y[r] = rhs_perm[r] - sum;
            }

            let mut inv_diag = Array1::<f64>::zeros(dim);
            let mut inv_offdiag = Array1::<f64>::zeros(dim.saturating_sub(1));
            let mut k = 0;
            while k < dim {
                if k + 1 < dim && factor.subdiag[k].abs() > 1e-12 {
                    let d11 = factor.diag[k];
                    let d22 = factor.diag[k + 1];
                    let d21 = factor.subdiag[k];
                    let det = d11 * d22 - d21 * d21;
                    if det.abs() < 1e-12 {
                        return Err(SurvivalError::HessianSingular);
                    }
                    inv_diag[k] = d22 / det;
                    inv_diag[k + 1] = d11 / det;
                    inv_offdiag[k] = -d21 / det;
                    k += 2;
                } else {
                    let d11 = factor.diag[k];
                    if d11.abs() < 1e-12 {
                        return Err(SurvivalError::HessianSingular);
                    }
                    inv_diag[k] = 1.0 / d11;
                    if k < dim - 1 {
                        inv_offdiag[k] = 0.0;
                    }
                    k += 1;
                }
            }

            let mut z = Array1::<f64>::zeros(dim);
            k = 0;
            while k < dim {
                if k + 1 < dim && factor.subdiag[k].abs() > 1e-12 {
                    let y1 = y[k];
                    let y2 = y[k + 1];
                    z[k] = inv_diag[k] * y1 + inv_offdiag[k] * y2;
                    z[k + 1] = inv_offdiag[k] * y1 + inv_diag[k + 1] * y2;
                    k += 2;
                } else {
                    z[k] = inv_diag[k] * y[k];
                    k += 1;
                }
            }

            let mut x_perm = Array1::<f64>::zeros(dim);
            for r_rev in 0..dim {
                let r = dim - 1 - r_rev;
                let mut sum = 0.0;
                for c in (r + 1)..dim {
                    sum += factor.lower[[c, r]] * x_perm[c];
                }
                x_perm[r] = z[r] - sum;
            }

            let mut x = Array1::<f64>::zeros(dim);
            for r in 0..dim {
                x[permutation.inverse[r]] = x_perm[r];
            }
            Ok(x)
        }
    }
}

pub fn survival_score_matrix(
    layout: &SurvivalLayout,
    data: &SurvivalLayoutInputs,
    spec: SurvivalSpec,
    monotonicity: &MonotonicityPenalty,
    beta: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalError> {
    let expected_dim = layout.combined_exit.ncols();
    if beta.len() != expected_dim {
        return Err(SurvivalError::DesignDimensionMismatch);
    }

    let eta_exit = layout.combined_exit.dot(beta);
    let eta_entry = layout.combined_entry.dot(beta);
    let derivative_raw = layout.combined_derivative_exit.dot(beta);

    let n = eta_exit.len();
    let p = beta.len();
    let mut score = Array2::<f64>::zeros((n, p));
    let guard_threshold = spec.derivative_guard.max(f64::EPSILON);

    if monotonicity.derivative_design.nrows() > 0 {
        let derivative_grid = monotonicity.derivative_design.dot(beta);
        for (idx, slope) in derivative_grid.iter().enumerate() {
            if !slope.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            if *slope < MONOTONICITY_TOLERANCE {
                let age = monotonicity.grid_ages[idx];
                return Err(SurvivalError::MonotonicityViolation {
                    age,
                    derivative: *slope,
                });
            }
        }
    }

    for i in 0..n {
        let weight = data.sample_weight[i];
        if weight == 0.0 {
            continue;
        }
        let d = f64::from(data.event_target[i]);
        let eta_e = eta_exit[i];
        let h_e = eta_e.exp();
        let h_s = eta_entry[i].exp();
        if !eta_e.is_finite() || !h_e.is_finite() || !h_s.is_finite() {
            return Err(SurvivalError::NonFiniteLinearPredictor);
        }
        let derivative_value = derivative_raw[i];
        let (log_guard, scale) = if derivative_value <= guard_threshold {
            (guard_threshold.ln(), 0.0)
        } else {
            (derivative_value.ln(), 1.0 / derivative_value)
        };
        if !log_guard.is_finite() {
            return Err(SurvivalError::NonFiniteLinearPredictor);
        }
        let x_exit = layout.combined_exit.row(i);
        let x_entry = layout.combined_entry.row(i);
        let d_exit = layout.combined_derivative_exit.row(i);
        let mut grad_row = Array1::<f64>::zeros(p);
        accumulate_weighted_vector(&mut grad_row, -weight * h_e, &x_exit);
        accumulate_weighted_vector(&mut grad_row, weight * h_s, &x_entry);

        let mut x_tilde = x_exit.to_owned();
        Zip::from(&mut x_tilde)
            .and(&d_exit)
            .for_each(|value, &deriv| *value += deriv * scale);
        if d > 0.0 {
            accumulate_weighted_vector(&mut grad_row, weight * d, &x_tilde);
        }

        grad_row.mapv_inplace(|value| value * -2.0);
        score.row_mut(i).assign(&grad_row);
    }

    Ok(score)
}

impl SurvivalModelArtifacts {
    /// Apply the logit-risk calibrator to convert conditional risk to calibrated probabilities.
    ///
    /// # Arguments
    /// * `conditional_risk` - Raw risk predictions
    /// * `logit_risk_design` - Design matrix for delta-method SE computation
    /// * `dist` - Optional signed distance to ancestry hull (negative = inside hull).
    ///            If None, zeros are used (no distance adjustment).
    pub fn apply_logit_risk_calibrator(
        &self,
        conditional_risk: &Array1<f64>,
        logit_risk_se: Option<&Array1<f64>>,
        logit_risk_design: &Array2<f64>,
        dist: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>, SurvivalError> {
        let _ = (conditional_risk, logit_risk_se, logit_risk_design, dist);
        Err(SurvivalError::Calibrator(
            "logit-risk calibrator pipeline is adapter-owned in gnomon".to_string(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use log::Level;
    use logtest::Logger;
    use ndarray::array;
    use serde_json;

    fn manual_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
        array![
            [matrix[[1, 1]] / det, -matrix[[0, 1]] / det],
            [-matrix[[1, 0]] / det, matrix[[0, 0]] / det]
        ]
    }

    fn finite_difference_derivatives(
        model: &mut WorkingModelSurvival,
        beta: &Array1<f64>,
        epsilon: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), SurvivalError> {
        let p = beta.len();
        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));

        for column in 0..p {
            let mut beta_plus = beta.to_owned();
            beta_plus[column] += epsilon;
            let plus_state = model.update_state(&beta_plus)?;

            let mut beta_minus = beta.to_owned();
            beta_minus[column] -= epsilon;
            let minus_state = model.update_state(&beta_minus)?;

            gradient[column] = (plus_state.deviance - minus_state.deviance) / (2.0 * epsilon);

            let diff = (&plus_state.gradient - &minus_state.gradient) * (0.5 / epsilon);
            hessian.column_mut(column).assign(&diff);
        }

        Ok((gradient, hessian))
    }

    #[test]
    fn delta_method_expected_factor_matches_manual_inverse() {
        let hessian = array![[4.0, 1.0], [1.0, 3.0]];
        let chol = CholeskyFactor {
            lower: array![[2.0, 0.0], [0.5, (2.75_f64).sqrt()]],
        };
        let factor = HessianFactor::Expected { factor: chol };
        let design = array![[1.0, 0.0], [0.0, 1.0], [0.3, -0.2]];

        let se = delta_method_standard_errors(&factor, &design).unwrap();
        let inv = manual_inverse(&hessian);

        for (idx, row) in design.rows().into_iter().enumerate() {
            let row_vec = row.to_owned();
            let tmp = inv.dot(&row_vec);
            let expected = row_vec.dot(&tmp).max(0.0).sqrt();
            assert_abs_diff_eq!(se[idx], expected, epsilon = 1e-10);
        }
    }

    // survival_calibrator_features_clamps_leverage_and_uses_delta_se SKIPPED because
    // survival_calibrator_features does not support leverage input yet.

    fn toy_training_data() -> SurvivalLayoutInputs {
        let static_covariates = array![
            [0.1, 0.0, 0.01, -0.02],
            [-0.2, 1.0, 0.02, 0.03],
            [0.3, 0.0, -0.04, 0.05]
        ];
        SurvivalLayoutInputs {
            age_entry: array![50.0, 55.0, 60.0],
            age_exit: array![55.0, 60.0, 65.0],
            event_target: array![1, 0, 1],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            static_covariates,
            static_covariate_names: vec![
                "pgs".to_string(),
                "sex".to_string(),
                "pc1".to_string(),
                "pc2".to_string(),
            ],
            time_varying_covariate: Some(array![0.1, -0.2, 0.3]),
        }
    }

    #[test]
    fn update_state_warns_when_derivative_guard_is_common() {
        let mut logger = Logger::start();
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 10, None).unwrap();
        let model = WorkingModelSurvival::new(
            layout_bundle.layout.clone(),
            &data,
            layout_bundle.monotonicity.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        let beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());

        model.update_state(&beta).unwrap();

        let mut matches = 0usize;
        while let Some(record) = logger.pop() {
            if record.level() == Level::Warn
                && record.args().contains("derivative guard activated for")
            {
                matches += 1;
            }
        }
        assert!(
            matches >= 1,
            "expected at least one derivative guard warning, got {}",
            matches
        );
    }

    fn repeat_rows(matrix: &Array2<f64>, pattern: &[usize]) -> Array2<f64> {
        let cols = matrix.ncols();
        let mut result = Array2::<f64>::zeros((pattern.len(), cols));
        for (row_idx, &source_idx) in pattern.iter().enumerate() {
            assert!(source_idx < matrix.nrows());
            result.row_mut(row_idx).assign(&matrix.row(source_idx));
        }
        result
    }

    fn repeat_optional(matrix: &Option<Array2<f64>>, pattern: &[usize]) -> Option<Array2<f64>> {
        matrix.as_ref().map(|array| repeat_rows(array, pattern))
    }

    fn combined_static_row(layout: &SurvivalLayout, idx: usize) -> Array1<f64> {
        let base = layout.static_covariates.row(idx);
        let extra = layout.extra_static_covariates.row(idx);
        let total = base.len() + extra.len();
        let mut result = Array1::<f64>::zeros(total);
        if base.len() > 0 {
            result.slice_mut(s![..base.len()]).assign(&base);
        }
        if extra.len() > 0 {
            result
                .slice_mut(s![base.len()..base.len() + extra.len()])
                .assign(&extra);
        }
        result
    }

    fn compute_value_ranges(matrix: &Array2<f64>) -> Vec<ValueRange> {
        (0..matrix.ncols())
            .map(|col_idx| {
                if matrix.nrows() == 0 {
                    return ValueRange { min: 0.0, max: 0.0 };
                }
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for &value in matrix.column(col_idx).iter() {
                    if value < min_val {
                        min_val = value;
                    }
                    if value > max_val {
                        max_val = value;
                    }
                }
                ValueRange {
                    min: min_val,
                    max: max_val,
                }
            })
            .collect()
    }

    fn make_covariate_layout(layout: &SurvivalLayout) -> CovariateLayout {
        let mut ranges = compute_value_ranges(&layout.static_covariates);
        ranges.extend(compute_value_ranges(&layout.extra_static_covariates));
        CovariateLayout {
            column_names: layout.static_covariate_names.clone(),
            ranges,
        }
    }

    fn baseline_penalty_descriptor(
        layout: &SurvivalLayout,
        order: usize,
        lambda: f64,
    ) -> PenaltyDescriptor {
        let baseline_cols = layout.baseline_exit.ncols();
        let matrix =
            create_difference_penalty_matrix(baseline_cols, order, None).expect("baseline penalty");
        PenaltyDescriptor {
            order,
            lambda,
            matrix,
            column_range: ColumnRange::new(0, baseline_cols),
        }
    }

    fn assert_array1_close(left: &Array1<f64>, right: &Array1<f64>, tol: f64) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            assert!((l - r).abs() <= tol, "array1 mismatch: {l} vs {r}");
        }
    }

    fn assert_array2_close(left: &Array2<f64>, right: &Array2<f64>, tol: f64) {
        assert_eq!(left.dim(), right.dim());
        for (l, r) in left.iter().zip(right.iter()) {
            assert!((l - r).abs() <= tol, "array2 mismatch: {l} vs {r}");
        }
    }

    fn assert_artifacts_close(left: &SurvivalModelArtifacts, right: &SurvivalModelArtifacts) {
        assert_array1_close(&left.coefficients, &right.coefficients, 1e-12);
        assert_array1_close(
            &left.age_basis.knot_vector,
            &right.age_basis.knot_vector,
            1e-12,
        );
        assert_eq!(left.age_basis.degree, right.age_basis.degree);
        assert_eq!(left.time_varying_basis, right.time_varying_basis);
        assert_eq!(
            left.static_covariate_layout.column_names,
            right.static_covariate_layout.column_names
        );
        for (l_range, r_range) in left
            .static_covariate_layout
            .ranges
            .iter()
            .zip(&right.static_covariate_layout.ranges)
        {
            assert!((l_range.min - r_range.min).abs() <= 1e-12);
            assert!((l_range.max - r_range.max).abs() <= 1e-12);
        }
        assert_eq!(left.penalties.len(), right.penalties.len());
        for (l_penalty, r_penalty) in left.penalties.iter().zip(&right.penalties) {
            assert_eq!(l_penalty.order, r_penalty.order);
            assert!((l_penalty.lambda - r_penalty.lambda).abs() <= 1e-12);
            assert_array2_close(&l_penalty.matrix, &r_penalty.matrix, 1e-12);
            assert_eq!(l_penalty.column_range, r_penalty.column_range);
        }
        assert_array2_close(
            &left.reference_constraint.transform,
            &right.reference_constraint.transform,
            1e-12,
        );
        assert!(
            (left.reference_constraint.reference_log_age
                - right.reference_constraint.reference_log_age)
                .abs()
                <= 1e-12
        );
        assert!((left.age_transform.minimum_age - right.age_transform.minimum_age).abs() <= 1e-12);
        assert!((left.age_transform.delta - right.age_transform.delta).abs() <= 1e-12);
        assert_eq!(
            left.interaction_metadata.len(),
            right.interaction_metadata.len()
        );
        for (l_meta, r_meta) in left
            .interaction_metadata
            .iter()
            .zip(&right.interaction_metadata)
        {
            assert_eq!(l_meta.label, r_meta.label);
            assert_eq!(l_meta.column_range, r_meta.column_range);
            assert_eq!(l_meta.value_ranges.len(), r_meta.value_ranges.len());
            for (l_range, r_range) in l_meta.value_ranges.iter().zip(&r_meta.value_ranges) {
                assert!((l_range.min - r_range.min).abs() <= 1e-12);
                assert!((l_range.max - r_range.max).abs() <= 1e-12);
            }
            match (&l_meta.centering, &r_meta.centering) {
                (Some(l), Some(r)) => {
                    assert_array1_close(&l.offsets, &r.offsets, 1e-12);
                }
                (None, None) => {}
                _ => panic!("centering mismatch"),
            }
        }
        assert_eq!(left.companion_models, right.companion_models);
        match (&left.hessian_factor, &right.hessian_factor) {
            (
                Some(HessianFactor::Observed {
                    factor: l_ldlt,
                    permutation: l_perm,
                    inertia: l_inertia,
                }),
                Some(HessianFactor::Observed {
                    factor: r_ldlt,
                    permutation: r_perm,
                    inertia: r_inertia,
                }),
            ) => {
                assert_array2_close(&l_ldlt.lower, &r_ldlt.lower, 1e-12);
                assert_array1_close(&l_ldlt.diag, &r_ldlt.diag, 1e-12);
                assert_array1_close(&l_ldlt.subdiag, &r_ldlt.subdiag, 1e-12);
                assert_eq!(l_perm, r_perm);
                assert_eq!(l_inertia, r_inertia);
            }
            (
                Some(HessianFactor::Expected { factor: l_chol }),
                Some(HessianFactor::Expected { factor: r_chol }),
            ) => {
                assert_array2_close(&l_chol.lower, &r_chol.lower, 1e-12);
            }
            (None, None) => {}
            _ => panic!("hessian factor mismatch"),
        }
        assert_eq!(left.calibrator.is_some(), right.calibrator.is_some());
        if let (Some(l), Some(r)) = (&left.calibrator, &right.calibrator) {
            let left_json = serde_json::to_string(l).unwrap();
            let right_json = serde_json::to_string(r).unwrap();
            assert_eq!(left_json, right_json);
        }
    }

    trait LogitExt {
        fn logit(self) -> f64;
    }
    impl LogitExt for f64 {
        fn logit(self) -> f64 {
            (self / (1.0 - self)).ln()
        }
    }

    #[test]
    fn logit_extension_behaves() {
        assert!(0.5f64.logit().abs() < 1e-12);
        assert!(f64::is_finite(0.01f64.logit()));
    }

    #[test]
    fn age_transform_rejects_non_positive_guard() {
        let ages = array![50.0, 55.0];
        let err = AgeTransform::from_training(&ages, 0.0).unwrap_err();
        assert!(matches!(err, SurvivalError::NonPositiveGuard));
    }

    #[test]
    fn monotonic_constraint_clamps_negative_derivatives() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let SurvivalLayoutBundle {
            layout,
            monotonicity,
            ..
        } = build_survival_layout(&data, &basis, 0.1, 2, 10, None).unwrap();
        let spec = SurvivalSpec::default();
        let beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        let model = WorkingModelSurvival::new(layout.clone(), &data, monotonicity, spec).unwrap();

        let state = model.update_state(&beta).unwrap();
        assert!(state.deviance.is_finite());

        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);
        let derivative_raw = layout.combined_derivative_exit.dot(&beta);
        let guard = model.spec.derivative_guard.max(f64::EPSILON);

        let mut log_likelihood = 0.0;
        for i in 0..data.age_entry.len() {
            let weight = data.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let d = f64::from(data.event_target[i]);
            let h_exit = eta_exit[i].exp();
            let h_entry = eta_entry[i].exp();
            let log_guard = if derivative_raw[i] <= guard {
                guard.ln()
            } else {
                derivative_raw[i].ln()
            };
            log_likelihood += weight * (d * (eta_exit[i] + log_guard) - (h_exit - h_entry));
        }

        let penalty = layout.penalties.deviance(&beta);
        let manual_deviance = -2.0 * log_likelihood + penalty;
        assert_abs_diff_eq!(state.deviance, manual_deviance, epsilon = 1e-10);
    }

    #[test]
    fn monotonic_constraint_rejects_negative_slopes() {
        let mut data = toy_training_data();
        data.sample_weight.fill(0.0);
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let SurvivalLayoutBundle {
            layout,
            monotonicity,
            ..
        } = build_survival_layout(&data, &basis, 0.1, 2, 12, None).unwrap();
        assert!(monotonicity.derivative_design.nrows() > 0);

        let mut beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        let mut found = false;
        for row in monotonicity.derivative_design.rows() {
            if row.iter().any(|value| value.abs() > 1e-12) {
                beta.assign(&row);
                // Scale the row so the induced slopes violate MONOTONICITY_TOLERANCE.
                beta.mapv_inplace(|value| -0.1 * value);
                found = true;
                break;
            }
        }
        assert!(found, "monotonicity grid returned only zero rows");

        let slopes = monotonicity.derivative_design.dot(&beta);
        assert!(slopes.iter().any(|value| *value < MONOTONICITY_TOLERANCE));

        let model = WorkingModelSurvival::new(layout, &data, monotonicity, SurvivalSpec::default())
            .unwrap();
        let err = model.update_state(&beta).unwrap_err();
        assert!(matches!(err, SurvivalError::MonotonicityViolation { .. }));
    }

    #[test]
    fn conditional_risk_monotone() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let SurvivalLayoutBundle {
            layout,
            monotonicity,
            penalty_descriptors,
            interaction_metadata,
            time_varying_basis,
        } = build_survival_layout(&data, &basis, 0.1, 2, 10, None).unwrap();
        let layout = layout;
        let model = WorkingModelSurvival::new(
            layout.clone(),
            &data,
            monotonicity.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(model.layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: penalty_descriptors,
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata,
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };
        let cov_cols =
            model.layout.static_covariates.ncols() + model.layout.extra_static_covariates.ncols();
        let covs = Array1::<f64>::zeros(cov_cols);
        let cif0 = cumulative_incidence(55.0, &covs, &artifacts).unwrap();
        let cif1 = cumulative_incidence(60.0, &covs, &artifacts).unwrap();
        assert!(cif1 >= cif0 - 1e-9);
        let risk = conditional_absolute_risk(55.0, 60.0, &covs, &artifacts).unwrap();
        assert!(risk >= -1e-9);
    }

    #[test]
    fn competing_cif_helpers_require_available_sources() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 6, None).unwrap();
        let layout = layout_bundle.layout;
        let make_artifacts = |companion_models: Vec<CompanionModelHandle>| SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models,
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let companion_artifacts = make_artifacts(Vec::new());
        let mut registry = HashMap::new();
        registry.insert("companion".to_string(), companion_artifacts);

        let base_artifacts = make_artifacts(vec![CompanionModelHandle {
            reference: "companion".to_string(),
            cif_horizons: vec![55.0],
        }]);
        let covs = Array1::<f64>::zeros(layout.static_covariates.ncols());

        let err = competing_cif_value(55.0, &covs, Some(f64::NAN), None).unwrap_err();
        assert!(matches!(err, SurvivalError::InvalidCompetingCif { .. }));

        let err = competing_cif_value(55.0, &covs, None, None).unwrap_err();
        assert!(matches!(err, SurvivalError::MissingCompanionCifData));

        {
            let (handle, resolved) =
                resolve_companion_model(&base_artifacts, "companion", &registry).unwrap();
            let explicit = 0.25;
            let value =
                competing_cif_value(55.0, &covs, Some(explicit), Some((handle, resolved))).unwrap();
            assert_abs_diff_eq!(value, explicit, epsilon = 1e-12);
        }

        let err = resolve_companion_model(&base_artifacts, "missing", &registry).unwrap_err();
        assert!(matches!(
            err,
            SurvivalError::UnknownCompanionModelHandle { .. }
        ));

        registry.clear();
        let err = resolve_companion_model(&base_artifacts, "companion", &registry).unwrap_err();
        assert!(matches!(
            err,
            SurvivalError::CompanionModelUnavailable { .. }
        ));
    }

    #[test]
    fn conditional_risk_uses_resolved_companion_model() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 6, None).unwrap();
        let layout = layout_bundle.layout;
        let make_artifacts = |companion_models: Vec<CompanionModelHandle>| SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout_bundle.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models,
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let companion_artifacts = make_artifacts(Vec::new());
        let mut registry = HashMap::new();
        registry.insert("companion".to_string(), companion_artifacts.clone());
        let base_artifacts = make_artifacts(vec![CompanionModelHandle {
            reference: "companion".to_string(),
            cif_horizons: vec![55.0, 60.0],
        }]);
        let covs = Array1::<f64>::zeros(layout.static_covariates.ncols());

        let via_companion = conditional_absolute_risk(55.0, 60.0, &covs, &base_artifacts).unwrap();
        let direct = conditional_absolute_risk(55.0, 60.0, &covs, &registry["companion"]).unwrap();
        assert_abs_diff_eq!(via_companion, direct, epsilon = 1e-12);
    }

    #[test]
    fn competing_cif_helpers_validate_horizons() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 6, None).unwrap();
        let layout = layout_bundle.layout;
        let make_artifacts = |companion_models: Vec<CompanionModelHandle>| SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout_bundle.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models,
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let companion_artifacts = make_artifacts(Vec::new());
        let mut registry = HashMap::new();
        registry.insert("companion".to_string(), companion_artifacts);
        let covs = Array1::<f64>::zeros(layout.static_covariates.ncols());

        let missing_horizon = make_artifacts(vec![CompanionModelHandle {
            reference: "companion".to_string(),
            cif_horizons: vec![60.0],
        }]);

        {
            let (handle, resolved) =
                resolve_companion_model(&missing_horizon, "companion", &registry).unwrap();
            let err = competing_cif_value(55.0, &covs, None, Some((handle, resolved))).unwrap_err();
            assert!(matches!(
                err,
                SurvivalError::CompanionModelMissingHorizon { .. }
            ));
        }

        let matching_horizon = make_artifacts(vec![CompanionModelHandle {
            reference: "companion".to_string(),
            cif_horizons: vec![55.0, 65.0],
        }]);

        {
            let (handle, resolved) =
                resolve_companion_model(&matching_horizon, "companion", &registry).unwrap();
            let value = competing_cif_value(55.0, &covs, None, Some((handle, resolved))).unwrap();
            let expected =
                cumulative_incidence(55.0, &covs, registry.get("companion").unwrap()).unwrap();
            assert_abs_diff_eq!(value, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn likelihood_matches_manual_computation() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let mut bundle = build_survival_layout(&data, &basis, 0.1, 2, 0, None).unwrap();
        bundle
            .layout
            .penalties
            .blocks
            .iter_mut()
            .for_each(|block| block.lambda = 0.0);
        bundle
            .penalty_descriptors
            .iter_mut()
            .for_each(|descriptor| descriptor.lambda = 0.0);
        let layout = bundle.layout.clone();
        let monotonicity = bundle.monotonicity.clone();
        let mut spec = SurvivalSpec::default();
        spec.derivative_guard = 1e-12;
        let model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let mut beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        for (idx, value) in beta.iter_mut().enumerate() {
            *value = 0.01 * (idx as f64 + 1.0);
        }
        if beta.len() > 0 {
            beta[0] = -0.15;
        }

        let state = model.update_state(&beta).unwrap();
        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);
        let derivative_exit = layout.combined_derivative_exit.dot(&beta);
        let guard = spec.derivative_guard.max(f64::EPSILON);
        assert!(derivative_exit.iter().any(|value| *value <= guard));

        let mut manual = 0.0;
        for i in 0..data.age_entry.len() {
            let d = f64::from(data.event_target[i]);
            let weight = data.sample_weight[i];
            let guarded = derivative_exit[i].max(guard);
            let h_exit = eta_exit[i].exp();
            let h_entry = eta_entry[i].exp();
            manual += weight * (d * (eta_exit[i] + guarded.ln()) - (h_exit - h_entry));
        }

        assert_abs_diff_eq!(state.deviance, -2.0 * manual, epsilon = 1e-10);
    }

    #[test]
    fn left_truncation_matches_scoring_difference() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.45, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let mut bundle = build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        bundle
            .layout
            .penalties
            .blocks
            .iter_mut()
            .for_each(|block| block.lambda = 0.0);
        bundle
            .penalty_descriptors
            .iter_mut()
            .for_each(|descriptor| descriptor.lambda = 0.0);
        let layout = bundle.layout.clone();
        let monotonicity = bundle.monotonicity.clone();
        let penalty_descriptors = bundle.penalty_descriptors.clone();
        let interaction_metadata = bundle.interaction_metadata.clone();
        let time_varying_basis = bundle.time_varying_basis.clone();
        let mut spec = SurvivalSpec::default();
        spec.derivative_guard = 1e-12;
        let model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        // Use zeros for baseline to ensure monotonic hazard (flat cumulative hazard)
        let beta = Array1::<f64>::zeros(p);

        let state = model.update_state(&beta).unwrap();
        assert!(state.deviance.is_finite());

        let artifacts = SurvivalModelArtifacts {
            coefficients: beta.clone(),
            age_basis: basis.clone(),
            time_varying_basis,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: penalty_descriptors,
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata,
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);

        for i in 0..data.age_entry.len() {
            let covariates = combined_static_row(&layout, i);
            let hazard_exit = cumulative_hazard(data.age_exit[i], &covariates, &artifacts).unwrap();
            let hazard_entry =
                cumulative_hazard(data.age_entry[i], &covariates, &artifacts).unwrap();
            let delta_scoring = hazard_exit - hazard_entry;
            let delta_training = eta_exit[i].exp() - eta_entry[i].exp();
            assert_abs_diff_eq!(delta_scoring, delta_training, epsilon = 1e-10);
        }
    }

    #[test]
    fn cumulative_hazard_matches_update_state_with_delayed_entry() {
        let data = SurvivalLayoutInputs {
            age_entry: array![45.0, 60.0, 58.0],
            age_exit: array![50.0, 66.0, 65.0],
            event_target: array![1, 0, 1],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            static_covariates: array![
                [0.05, 0.0, 0.01, -0.02],
                [-0.1, 1.0, 0.03, 0.04],
                [0.2, 0.0, -0.05, 0.06]
            ],
            static_covariate_names: vec![
                "pgs".to_string(),
                "sex".to_string(),
                "pc1".to_string(),
                "pc2".to_string(),
            ],
            time_varying_covariate: Some(array![0.05, -0.1, 0.2]),
        };

        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let mut bundle = build_survival_layout(&data, &basis, 0.1, 2, 6, None).unwrap();
        bundle
            .layout
            .penalties
            .blocks
            .iter_mut()
            .for_each(|block| block.lambda = 0.0);
        bundle
            .penalty_descriptors
            .iter_mut()
            .for_each(|descriptor| descriptor.lambda = 0.0);

        let layout = bundle.layout.clone();
        let monotonicity = bundle.monotonicity.clone();
        let penalty_descriptors = bundle.penalty_descriptors.clone();
        let interaction_metadata = bundle.interaction_metadata.clone();
        let time_varying_basis = bundle.time_varying_basis.clone();

        let mut spec = SurvivalSpec::default();
        spec.derivative_guard = 1e-12;
        let model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        // Use zeros for baseline to ensure monotonic hazard (flat cumulative hazard)
        let beta = Array1::<f64>::zeros(p);

        let state = model.update_state(&beta).unwrap();
        assert!(state.deviance.is_finite());

        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);
        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);
        let delta_training = &h_exit - &h_entry;

        let artifacts = SurvivalModelArtifacts {
            coefficients: beta.clone(),
            age_basis: basis.clone(),
            time_varying_basis,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: penalty_descriptors,
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata,
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        for i in 0..data.age_entry.len() {
            let covariates = combined_static_row(&layout, i);
            let hazard_exit = cumulative_hazard(data.age_exit[i], &covariates, &artifacts).unwrap();
            let hazard_entry =
                cumulative_hazard(data.age_entry[i], &covariates, &artifacts).unwrap();
            let delta_scoring = hazard_exit - hazard_entry;
            assert_abs_diff_eq!(delta_scoring, delta_training[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn gradient_and_hessian_match_numeric() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let mut bundle = build_survival_layout(&data, &basis, 0.1, 2, 0, None).unwrap();
        bundle
            .layout
            .penalties
            .blocks
            .iter_mut()
            .for_each(|block| block.lambda = 0.0);
        bundle
            .penalty_descriptors
            .iter_mut()
            .for_each(|descriptor| descriptor.lambda = 0.0);
        let layout = bundle.layout.clone();
        let monotonicity = bundle.monotonicity.clone();
        let mut spec = SurvivalSpec::default();
        spec.derivative_guard = 1e-12;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        // Choose beta so derivative_exit stays strictly above the guard, keeping the
        // finite-difference checks inside a smooth region of the objective.
        let d_exit = &layout.combined_derivative_exit;
        let guard = spec.derivative_guard.max(f64::EPSILON);
        let mut beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        let mut target_scale = 1.0;
        let mut derivative_exit = Array1::<f64>::zeros(d_exit.nrows());
        for _ in 0..8 {
            let target = Array1::from_elem(d_exit.nrows(), target_scale);
            let mut normal = d_exit.t().dot(d_exit);
            for i in 0..normal.nrows() {
                normal[(i, i)] += 1e-6;
            }
            let rhs = d_exit.t().dot(&target);
            let factor =
                crate::faer_ndarray::FaerCholesky::cholesky(&normal, faer::Side::Lower)
                    .unwrap();
            beta = factor.solve_vec(&rhs);
            derivative_exit = d_exit.dot(&beta);

            let mut min_abs = f64::INFINITY;
            for (row, value) in d_exit.rows().into_iter().zip(derivative_exit.iter()) {
                let row_norm: f64 = row.iter().map(|v| v.abs()).sum();
                if row_norm <= 1e-12 {
                    continue;
                }
                min_abs = min_abs.min(value.abs());
            }
            if min_abs > 10.0 * guard {
                break;
            }
            target_scale *= 10.0;
        }

        let mut min_abs = f64::INFINITY;
        for (row, value) in d_exit.rows().into_iter().zip(derivative_exit.iter()) {
            let row_norm: f64 = row.iter().map(|v| v.abs()).sum();
            if row_norm <= 1e-12 {
                continue;
            }
            min_abs = min_abs.min(value.abs());
        }
        if min_abs.is_finite() {
            assert!(min_abs > guard);
        }

        let epsilon = 1e-6;
        let (numeric_gradient, numeric_hessian) =
            finite_difference_derivatives(&mut model, &beta, epsilon).unwrap();

        let analytic_state = model.update_state(&beta).unwrap();

        for (numeric, analytic) in numeric_gradient.iter().zip(analytic_state.gradient.iter()) {
            assert_abs_diff_eq!(numeric, analytic, epsilon = 1e-7);
        }

        for (numeric_row, analytic_row) in numeric_hessian
            .rows()
            .into_iter()
            .zip(analytic_state.hessian.rows())
        {
            for (numeric, analytic) in numeric_row.iter().zip(analytic_row.iter()) {
                assert_abs_diff_eq!(numeric, analytic, epsilon = 2e-6);
            }
        }
    }

    #[test]
    fn frequency_weights_match_replication() {
        let weighted_data = SurvivalLayoutInputs {
            age_entry: array![50.0, 55.0],
            age_exit: array![55.0, 60.0],
            event_target: array![1, 0],
            event_competing: array![0, 0],
            sample_weight: array![1.0, 2.0],
            static_covariates: array![[0.1, 0.0, 0.01, -0.02], [-0.3, 1.0, 0.02, 0.03]],
            static_covariate_names: vec![
                "pgs".to_string(),
                "sex".to_string(),
                "pc1".to_string(),
                "pc2".to_string(),
            ],
            time_varying_covariate: Some(array![0.1, -0.3]),
        };

        let expanded_data = SurvivalLayoutInputs {
            age_entry: array![50.0, 55.0, 55.0],
            age_exit: array![55.0, 60.0, 60.0],
            event_target: array![1, 0, 0],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            static_covariates: array![
                [0.1, 0.0, 0.01, -0.02],
                [-0.3, 1.0, 0.02, 0.03],
                [-0.3, 1.0, 0.02, 0.03]
            ],
            static_covariate_names: vec![
                "pgs".to_string(),
                "sex".to_string(),
                "pc1".to_string(),
                "pc2".to_string(),
            ],
            time_varying_covariate: Some(array![0.1, -0.3, -0.3]),
        };

        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0],
            degree: 2,
        };

        let mut bundle = build_survival_layout(&weighted_data, &basis, 0.1, 2, 0, None).unwrap();
        bundle
            .layout
            .penalties
            .blocks
            .iter_mut()
            .for_each(|block| block.lambda = 0.0);
        bundle
            .penalty_descriptors
            .iter_mut()
            .for_each(|descriptor| descriptor.lambda = 0.0);
        let layout_weighted = bundle.layout.clone();
        let monotonic_weighted = bundle.monotonicity.clone();
        let replicate_pattern = [0usize, 1, 1];
        let layout_expanded = SurvivalLayout {
            baseline_entry: repeat_rows(&layout_weighted.baseline_entry, &replicate_pattern),
            baseline_exit: repeat_rows(&layout_weighted.baseline_exit, &replicate_pattern),
            baseline_derivative_exit: repeat_rows(
                &layout_weighted.baseline_derivative_exit,
                &replicate_pattern,
            ),
            time_varying_entry: repeat_optional(
                &layout_weighted.time_varying_entry,
                &replicate_pattern,
            ),
            time_varying_exit: repeat_optional(
                &layout_weighted.time_varying_exit,
                &replicate_pattern,
            ),
            time_varying_derivative_exit: repeat_optional(
                &layout_weighted.time_varying_derivative_exit,
                &replicate_pattern,
            ),
            static_covariates: repeat_rows(&layout_weighted.static_covariates, &replicate_pattern),
            extra_static_covariates: repeat_rows(
                &layout_weighted.extra_static_covariates,
                &replicate_pattern,
            ),
            static_covariate_names: layout_weighted.static_covariate_names.clone(),
            age_transform: layout_weighted.age_transform,
            reference_constraint: layout_weighted.reference_constraint.clone(),
            penalties: layout_weighted.penalties.clone(),
            combined_entry: repeat_rows(&layout_weighted.combined_entry, &replicate_pattern),
            combined_exit: repeat_rows(&layout_weighted.combined_exit, &replicate_pattern),
            combined_derivative_exit: repeat_rows(
                &layout_weighted.combined_derivative_exit,
                &replicate_pattern,
            ),
            monotonicity: monotonic_weighted.clone(),
        };

        let mut spec = SurvivalSpec::default();
        spec.derivative_guard = 1e-12;

        let weighted_model = WorkingModelSurvival::new(
            layout_weighted.clone(),
            &weighted_data,
            monotonic_weighted.clone(),
            spec,
        )
        .unwrap();
        let expanded_model = WorkingModelSurvival::new(
            layout_expanded.clone(),
            &expanded_data,
            monotonic_weighted,
            spec,
        )
        .unwrap();

        let p = layout_weighted.combined_exit.ncols();
        assert_eq!(p, layout_expanded.combined_exit.ncols());
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.03 * (idx as f64 + 1.0);
        }

        let state_weighted = weighted_model.update_state(&beta).unwrap();
        let state_expanded = expanded_model.update_state(&beta).unwrap();

        assert_abs_diff_eq!(
            state_weighted.deviance,
            state_expanded.deviance,
            epsilon = 1e-4
        );
        for (g_weighted, g_expanded) in state_weighted
            .gradient
            .iter()
            .zip(state_expanded.gradient.iter())
        {
            assert_abs_diff_eq!(*g_weighted, *g_expanded, epsilon = 1e-4);
        }
        for (h_weighted, h_expanded) in state_weighted
            .hessian
            .iter()
            .zip(state_expanded.hessian.iter())
        {
            assert_abs_diff_eq!(*h_weighted, *h_expanded, epsilon = 1e-4);
        }
    }

    #[test]
    fn time_varying_tensor_product_contributes_to_hazard() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let pgs_basis = BasisDescriptor {
            knot_vector: array![-0.6, -0.6, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.6, 0.6,],
            degree: 2,
        };
        let tensor_config = TensorProductConfig {
            label: Some("pgs_by_age".to_string()),
            pgs_basis: pgs_basis.clone(),
            pgs_penalty_order: 2,
            lambda_age: 0.15,
            lambda_pgs: 0.2,
            lambda_null: 0.05,
        };

        let SurvivalLayoutBundle {
            layout,
            monotonicity,
            penalty_descriptors,
            interaction_metadata,
            time_varying_basis,
        } = build_survival_layout(&data, &basis, 0.1, 2, 4, Some(&tensor_config)).unwrap();

        assert!(layout.time_varying_exit.is_some());
        assert_eq!(time_varying_basis, Some(pgs_basis.clone()));
        assert!(penalty_descriptors.len() >= 4);
        let metadata = interaction_metadata
            .first()
            .expect("time-varying interaction metadata");
        let time_exit = layout.time_varying_exit.as_ref().unwrap();
        assert_eq!(
            metadata.column_range.end - metadata.column_range.start,
            time_exit.ncols()
        );
        let offsets = metadata
            .centering
            .as_ref()
            .expect("centering metadata for tensor product")
            .offsets
            .clone();
        let (pgs_basis_full_arc, _) = create_basis::<Dense>(
            data.time_varying_covariate
                .as_ref()
                .expect("time-varying covariate")
                .view(),
            KnotSource::Provided(pgs_basis.knot_vector.view()),
            pgs_basis.degree,
            BasisOptions::value(),
        )
        .unwrap();
        let pgs_basis_full = (*pgs_basis_full_arc).clone();
        let mut pgs_basis_matrix = pgs_basis_full.slice(s![.., 1..]).to_owned();
        let raw_means = compute_weighted_column_means(&pgs_basis_matrix, &data.sample_weight);
        assert_eq!(raw_means.len(), offsets.len());
        for (raw, offset) in raw_means.iter().zip(offsets.iter()) {
            assert_abs_diff_eq!(raw, offset, epsilon = 1e-10);
        }
        for (mut column, &offset) in pgs_basis_matrix.axis_iter_mut(Axis(1)).zip(offsets.iter()) {
            column.mapv_inplace(|value: f64| value - offset);
        }
        let centered_means = compute_weighted_column_means(&pgs_basis_matrix, &data.sample_weight);
        for mean in centered_means.iter() {
            assert!(mean.abs() < 5e-10);
        }

        let baseline_cols = layout.baseline_exit.ncols();
        let time_cols = time_exit.ncols();
        let static_cols = layout.static_covariates.ncols();
        let mut beta = Array1::<f64>::zeros(baseline_cols + time_cols + static_cols);
        for idx in baseline_cols..baseline_cols + time_cols {
            beta[idx] = 0.05 * ((idx - baseline_cols + 1) as f64);
        }

        let eta_exit = layout.combined_exit.dot(&beta);
        let covariates = layout.static_covariates.row(0).to_owned();
        let artifacts = SurvivalModelArtifacts {
            coefficients: beta,
            age_basis: basis.clone(),
            time_varying_basis: time_varying_basis.clone(),
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: penalty_descriptors,
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: interaction_metadata.clone(),
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let hazard = cumulative_hazard(data.age_exit[0], &covariates, &artifacts).unwrap();
        assert_abs_diff_eq!(hazard, eta_exit[0].exp(), epsilon = 1e-10);

        // Test that out-of-range PGS values are clamped (not errored) for robustness
        let pgs_idx = artifacts
            .static_covariate_layout
            .column_names
            .iter()
            .position(|name| name == "pgs")
            .expect("pgs column present");

        // Values outside range should now be clamped silently, not error
        let mut covariates_high = covariates.clone();
        covariates_high[pgs_idx] = metadata.value_ranges[0].max + 0.5;
        let result_high = design_row_at_age(data.age_exit[0], covariates_high.view(), &artifacts);
        assert!(
            result_high.is_ok(),
            "out-of-range PGS should be clamped, not error"
        );

        let mut covariates_low = covariates;
        covariates_low[pgs_idx] = metadata.value_ranges[0].min - 0.5;
        let result_low = design_row_at_age(data.age_exit[0], covariates_low.view(), &artifacts);
        assert!(
            result_low.is_ok(),
            "out-of-range PGS should be clamped, not error"
        );

        let model = WorkingModelSurvival::new(
            layout.clone(),
            &data,
            monotonicity.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        let state = model.update_state(&artifacts.coefficients).unwrap();
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn cumulative_hazard_rejects_covariate_mismatch() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let SurvivalLayoutBundle {
            layout,
            penalty_descriptors,
            interaction_metadata,
            time_varying_basis,
            ..
        } = build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: penalty_descriptors,
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata,
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };
        let mismatched_covs = Array1::<f64>::zeros(layout.static_covariates.ncols() + 1);
        let err = cumulative_hazard(60.0, &mismatched_covs, &artifacts).unwrap_err();
        assert!(matches!(err, SurvivalError::CovariateDimensionMismatch));
    }

    #[test]
    fn cumulative_hazard_rejects_covariates_out_of_persisted_range() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        let layout = layout_bundle.layout;
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: None,
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };
        let mismatched_covs = Array1::<f64>::zeros(
            layout.static_covariates.ncols() + layout.extra_static_covariates.ncols() + 1,
        );
        let err = cumulative_hazard(60.0, &mismatched_covs, &artifacts).unwrap_err();
        assert!(matches!(err, SurvivalError::CovariateDimensionMismatch));
    }

    #[test]
    fn survival_artifacts_round_trip_serialization() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let SurvivalLayoutBundle { layout, .. } =
            build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        let penalty = baseline_penalty_descriptor(&layout, 2, 0.5);
        let interaction = InteractionDescriptor {
            label: Some("pgs_by_age".to_string()),
            column_range: ColumnRange::new(1, 3),
            value_ranges: vec![ValueRange {
                min: -0.5,
                max: 0.5,
            }],
            centering: Some(CenteringTransform {
                offsets: array![0.1, -0.1],
            }),
        };
        let companion = CompanionModelHandle {
            reference: "competing-risk-model".to_string(),
            cif_horizons: vec![55.0],
        };
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![penalty],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: vec![interaction],
            companion_models: vec![companion],
            hessian_factor: Some(HessianFactor::Expected {
                factor: CholeskyFactor {
                    lower: Array2::<f64>::eye(layout.combined_exit.ncols()),
                },
            }),
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let serialized = serde_json::to_string(&artifacts).unwrap();
        let round_trip: SurvivalModelArtifacts = serde_json::from_str(&serialized).unwrap();
        assert_artifacts_close(&artifacts, &round_trip);
    }

    #[test]
    fn delta_method_observed_factor_matches_expected() {
        let hessian = array![[4.0, 1.0], [1.0, 3.0]];
        let (lower, diag, subdiag, perm_fwd, perm_inv, inertia) = ldlt_rook(&hessian).unwrap();
        let factor = HessianFactor::Observed {
            factor: LdltFactor {
                lower: lower.clone(),
                diag: diag.clone(),
                subdiag: subdiag.clone(),
            },
            permutation: PermutationDescriptor {
                forward: perm_fwd.clone(),
                inverse: perm_inv.clone(),
            },
            inertia,
        };

        let design = array![[1.0, 0.0], [0.0, 1.0], [0.3, -0.2]];
        let expected_factor = HessianFactor::Expected {
            factor: CholeskyFactor {
                lower: array![[2.0, 0.0], [0.5, (2.75_f64).sqrt()]],
            },
        };

        let se_observed = delta_method_standard_errors(&factor, &design).unwrap();
        let se_expected = delta_method_standard_errors(&expected_factor, &design).unwrap();

        for i in 0..se_observed.len() {
            assert_abs_diff_eq!(se_observed[i], se_expected[i], epsilon = 1e-10);
        }
    }
}
