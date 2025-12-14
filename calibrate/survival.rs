use crate::calibrate::basis::{
    BasisError, SplineScratch, create_bspline_basis_with_knots, create_difference_penalty_matrix,
    evaluate_bspline_basis_scalar, null_range_whiten,
};
use crate::calibrate::calibrator::CalibratorModel;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::faer_ndarray::{FaerSvd, ldlt_rook};
use crate::calibrate::pirls::{WorkingModel as PirlsWorkingModel, WorkingState};
use log::warn;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Ix1, Zip, concatenate};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_DERIVATIVE_GUARD: f64 = 1e-8;
pub const DEFAULT_RISK_EPSILON: f64 = 1e-12;
const COMPANION_HORIZON_TOLERANCE: f64 = 1e-8;
const MONOTONICITY_TOLERANCE: f64 = -5e-2;
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

pub use crate::calibrate::pirls::WorkingModel;

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
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub extra_static_covariates: Array2<f64>,
    pub extra_static_names: Vec<String>,
}

impl SurvivalTrainingData {
    pub fn validate(&self) -> Result<(), SurvivalError> {
        let n = self.age_entry.len();
        if n == 0 {
            return Err(SurvivalError::EmptyAgeVector);
        }
        let dimension_mismatch = self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
            || self.pgs.len() != n
            || self.sex.len() != n
            || self.pcs.nrows() != n
            || self.extra_static_covariates.nrows() != n;
        if dimension_mismatch {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        if self.extra_static_names.len() != self.extra_static_covariates.ncols() {
            return Err(SurvivalError::CovariateDimensionMismatch);
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

            let pgs = self.pgs[i];
            let sex = self.sex[i];
            if !pgs.is_finite() || !sex.is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
            for j in 0..self.pcs.ncols() {
                if !self.pcs[[i, j]].is_finite() {
                    return Err(SurvivalError::NonFiniteCovariate);
                }
            }
            for j in 0..self.extra_static_covariates.ncols() {
                if !self.extra_static_covariates[[i, j]].is_finite() {
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
    pgs: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
    extra_static: ArrayView2<f64>,
) -> Result<(), SurvivalError> {
    let n = age_entry.len();
    if n == 0 {
        return Err(SurvivalError::EmptyAgeVector);
    }
    let dimension_mismatch = age_exit.len() != n
        || event_target.len() != n
        || event_competing.len() != n
        || sample_weight.len() != n
        || pgs.len() != n
        || sex.len() != n
        || pcs.nrows() != n
        || extra_static.nrows() != n;
    if dimension_mismatch {
        return Err(SurvivalError::CovariateDimensionMismatch);
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

        let pgs_val = pgs[i];
        let sex_val = sex[i];
        if !pgs_val.is_finite() || !sex_val.is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }
        for j in 0..pcs.ncols() {
            if !pcs[[i, j]].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
        for j in 0..extra_static.ncols() {
            if !extra_static[[i, j]].is_finite() {
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
    let (basis_arc, _) = create_bspline_basis_with_knots(data.view(), knot_vector, degree)?;
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
    let u = u_opt.ok_or_else(|| SurvivalError::Basis(BasisError::ConstraintNullspaceNotFound))?;
    Ok(u.slice(s![.., 1..]).to_owned())
}

/// Evaluate a basis and its derivative with respect to the guarded log-age.
fn evaluate_basis_and_derivative(
    log_ages: ArrayView1<f64>,
    descriptor: &BasisDescriptor,
) -> Result<(Array2<f64>, Array2<f64>), SurvivalError> {
    let (basis_arc, _) = create_bspline_basis_with_knots(
        log_ages,
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let basis = (*basis_arc).clone();

    let eps = 1e-6;
    let mut perturbed_plus = log_ages.to_owned();
    let mut perturbed_minus = log_ages.to_owned();
    perturbed_plus.mapv_inplace(|v| v + eps);
    perturbed_minus.mapv_inplace(|v| v - eps);
    let (basis_plus_arc, _) = create_bspline_basis_with_knots(
        perturbed_plus.view(),
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let (basis_minus_arc, _) = create_bspline_basis_with_knots(
        perturbed_minus.view(),
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let basis_plus = (*basis_plus_arc).clone();
    let basis_minus = (*basis_minus_arc).clone();
    let mut derivative = basis_plus;
    derivative -= &basis_minus;
    derivative.mapv_inplace(|v| v / (2.0 * eps));

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

/// Compute the initial smoothing weight for the survival baseline spline.
pub fn baseline_lambda_seed(age_basis: &BasisDescriptor, penalty_order: usize) -> f64 {
    let mut min_knot = f64::INFINITY;
    let mut max_knot = f64::NEG_INFINITY;
    for &value in age_basis.knot_vector.iter() {
        if !value.is_finite() {
            continue;
        }
        if value < min_knot {
            min_knot = value;
        }
        if value > max_knot {
            max_knot = value;
        }
    }

    let span = if min_knot.is_finite() && max_knot.is_finite() && max_knot > min_knot {
        max_knot - min_knot
    } else {
        1.0
    };
    let order = penalty_order.max(1) as f64;
    let degree = age_basis.degree.max(1) as f64;
    let normalized_span = (span / (span + 1.0)).max(1e-3);
    let lambda = 0.5 * (order / (degree + 1.0)) / normalized_span;
    lambda.clamp(1e-6, 1e3)
}

pub fn build_survival_layout(
    data: &SurvivalTrainingData,
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
    let extra_static_covariates = data.extra_static_covariates.clone();
    let static_covariate_names = assemble_static_covariate_names(data);

    let baseline_cols = constrained_exit.ncols();
    let penalty_matrix = create_difference_penalty_matrix(baseline_cols, baseline_penalty_order)?;
    let baseline_lambda = baseline_lambda_seed(age_basis, baseline_penalty_order);
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

        let (pgs_basis_full, _) = create_bspline_basis_with_knots(
            data.pgs.view(),
            config.pgs_basis.knot_vector.view(),
            config.pgs_basis.degree,
        )?;
        if pgs_basis_full.ncols() <= 1 {
            warn!("PGS basis returned no range columns; skipping time-varying interaction");
        } else {
            let mut pgs_basis = pgs_basis_full.slice(s![.., 1..]).to_owned();
            let offsets = compute_weighted_column_means(&pgs_basis, &data.sample_weight);
            if offsets.len() == pgs_basis.ncols() {
                for (mut column, &offset) in pgs_basis.axis_iter_mut(Axis(1)).zip(offsets.iter()) {
                    column.mapv_inplace(|value| value - offset);
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
                    create_difference_penalty_matrix(pgs_cols, config.pgs_penalty_order)?;

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
                ) {
                    if age_null.ncols() > 0 && pgs_null.ncols() > 0 {
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
                }

                time_varying_entry = Some(time_entry);
                time_varying_exit = Some(time_exit);
                time_varying_derivative_exit = Some(time_derivative_exit);
                time_varying_basis_descriptor = Some(config.pgs_basis.clone());

                let mut min_pgs = f64::INFINITY;
                let mut max_pgs = f64::NEG_INFINITY;
                for &value in data.pgs.iter() {
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

fn assemble_static_covariates(data: &SurvivalTrainingData) -> Array2<f64> {
    let n = data.age_entry.len();
    let num_pcs = data.pcs.ncols();
    let mut matrix = Array2::<f64>::zeros((n, 2 + num_pcs));
    for i in 0..n {
        matrix[[i, 0]] = data.pgs[i];
        matrix[[i, 1]] = data.sex[i];
        for j in 0..num_pcs {
            matrix[[i, 2 + j]] = data.pcs[[i, j]];
        }
    }
    matrix
}

fn assemble_static_covariate_names(data: &SurvivalTrainingData) -> Vec<String> {
    let mut names = Vec::with_capacity(2 + data.pcs.ncols() + data.extra_static_names.len());
    names.push("pgs".to_string());
    names.push("sex".to_string());
    for idx in 0..data.pcs.ncols() {
        names.push(format!("pc{}", idx + 1));
    }
    names.extend(data.extra_static_names.iter().cloned());
    names
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
    pub layout: SurvivalLayout,
    pub sample_weight: Array1<f64>,
    pub event_target: Array1<u8>,
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub monotonicity: MonotonicityPenalty,
    pub spec: SurvivalSpec,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        data: &SurvivalTrainingData,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        data.validate()?;
        Ok(Self {
            layout,
            sample_weight: data.sample_weight.clone(),
            event_target: data.event_target.clone(),
            age_entry: data.age_entry.clone(),
            age_exit: data.age_exit.clone(),
            monotonicity,
            spec,
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

            let mut design = Array1::<f64>::zeros(p);
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
                design.assign(&self.monotonicity.quadrature_design.row(j));
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
                let scale = weight * (right - left) * hazard;
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
            match ldlt_rook(&shifted) {
                Ok((_, _, _, _, _, inertia)) => {
                    if inertia.1 == 0 && inertia.2 == 0 {
                        expected = -shifted;
                        break;
                    }
                }
                Err(_) => {}
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
    pub fn update_state(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        let expected_dim = self.layout.combined_exit.ncols();
        if beta.len() != expected_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }

        let eta_exit = self.layout.combined_exit.dot(beta);
        let eta_entry = self.layout.combined_entry.dot(beta);
        let derivative_raw = self.layout.combined_derivative_exit.dot(beta);

        let n = eta_exit.len();
        let p = beta.len();
        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));
        let mut log_likelihood = 0.0;
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

        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let d = f64::from(self.event_target[i]);
            let eta_e = eta_exit[i];
            let h_e = h_exit[i];
            let h_s = h_entry[i];
            if !eta_e.is_finite() || !h_e.is_finite() || !h_s.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let scale = derivative_scale[i];
            let log_guard = log_derivative[i];
            let delta = h_e - h_s;
            log_likelihood += weight * (d * (eta_e + log_guard) - delta);

            let x_exit = self.layout.combined_exit.row(i);
            let x_entry = self.layout.combined_entry.row(i);
            let d_exit = self.layout.combined_derivative_exit.row(i);
            accumulate_weighted_vector(&mut gradient, -weight * h_e, &x_exit);
            accumulate_weighted_vector(&mut gradient, weight * h_s, &x_entry);

            let mut x_tilde = x_exit.to_owned();
            Zip::from(&mut x_tilde)
                .and(&d_exit)
                .for_each(|value, &deriv| *value += deriv * scale);

            if d > 0.0 {
                accumulate_weighted_vector(&mut gradient, weight * d, &x_tilde);
            }

            accumulate_symmetric_outer(&mut hessian, -weight * h_e, &x_exit);
            accumulate_symmetric_outer(&mut hessian, weight * h_s, &x_entry);

            let event_scale = weight * d;
            if event_scale != 0.0 && scale != 0.0 {
                accumulate_symmetric_outer(
                    &mut hessian,
                    -event_scale * scale * scale,
                    &self.layout.combined_derivative_exit.row(i),
                );
            }
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

        if self.spec.use_expected_information {
            if let Some(expected_hessian) =
                self.build_expected_information_hessian(beta, &penalty_hessian)?
            {
                hessian = expected_hessian;
            }
        }

        Ok(WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
            penalty_term,
        })
    }
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
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
    pub hessian_factor: Option<HessianFactor>,
    #[serde(default)]
    pub calibrator: Option<CalibratorModel>,
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
pub fn competing_cif_value<'a, 'b>(
    horizon: f64,
    covariates: &Array1<f64>,
    explicit: Option<f64>,
    companion: Option<(&'a CompanionModelHandle, &'b SurvivalModelArtifacts)>,
) -> Result<f64, SurvivalError> {
    if let Some(value) = explicit {
        if !value.is_finite() || value < 0.0 || value > 1.0 {
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

fn enforce_covariate_ranges(
    covariates: ArrayView1<f64>,
    layout: &CovariateLayout,
) -> Result<(), SurvivalError> {
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

    for (idx, value) in covariates.iter().enumerate() {
        if !value.is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }
        let range = &layout.ranges[idx];
        if !range.min.is_finite() && !range.max.is_finite() {
            // Both bounds are infinite, nothing to enforce.
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
        if range.min.is_finite() && *value < range.min {
            return Err(SurvivalError::CovariateBelowRange {
                column: covariate_label(layout, idx),
                index: idx,
                value: *value,
                minimum: range.min,
            });
        }
        if range.max.is_finite() && *value > range.max {
            return Err(SurvivalError::CovariateAboveRange {
                column: covariate_label(layout, idx),
                index: idx,
                value: *value,
                maximum: range.max,
            });
        }
    }
    Ok(())
}

/// Reconstruct the design row at a given age for prediction.
pub fn design_row_at_age(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<Array1<f64>, SurvivalError> {
    let (design, _) = design_and_derivative_at_age(age, covariates, artifacts)?;
    Ok(design)
}

/// Compute both design row and its time derivative at a given age.
fn design_and_derivative_at_age(
    age: f64,
    covariates: ArrayView1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<(Array1<f64>, Array1<f64>), SurvivalError> {
    // This function now allocates! We should use design_and_derivative_at_age_scratch for performance.
    // We keep this for compatibility and existing tests that don't care about allocs.

    let total_cols = artifacts.coefficients.len();

    // Just allocate buffers and delegate
    let max_basis = artifacts.age_basis.knot_vector.len().max(100); // safe guess
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
    // Range enforcement is cheap
    enforce_covariate_ranges(covariates, &artifacts.static_covariate_layout)?;

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
        let pgs_value = covariates[pgs_idx];

        // Check range
        if let Some(range) = descriptor.value_ranges.first() {
            if range.min.is_finite() && pgs_value < range.min {
                return Err(SurvivalError::CovariateBelowRange {
                    column: "pgs".into(),
                    index: pgs_idx,
                    value: pgs_value,
                    minimum: range.min,
                });
            }
            if range.max.is_finite() && pgs_value > range.max {
                return Err(SurvivalError::CovariateAboveRange {
                    column: "pgs".into(),
                    index: pgs_idx,
                    value: pgs_value,
                    maximum: range.max,
                });
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

    // 3. Static
    let dest_static = &mut design_base[baseline_cols + time_cols..];
    for (i, &val) in covariates.iter().enumerate() {
        dest_static[i] = val;
    }
    // Derivatives for static cols are 0.0 (already set)

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

pub fn cumulative_incidence(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let h = cumulative_hazard(age, covariates, artifacts)?;
    Ok(1.0 - (-h).exp())
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
    Ok(1.0 - (-delta_h).exp())
}

/// Compute Gauss-Legendre quadrature nodes and weights for N points on interval [-1, 1].
fn compute_gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    let mut nodes_weights = Vec::with_capacity(n);
    let m = (n + 1) / 2;
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

        if n % 2 != 0 && i == m - 1 {
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
/// P(Event in (t0, t1] | Survival to t0, Accounting for Competing Mortality)
/// Formula: Integral_{t0}^{t1} h_dis(u) * S_total(u|t0) du
pub fn calculate_crude_risk_quadrature(
    t0: f64,
    t1: f64,
    covariates: &Array1<f64>,
    disease_model: &SurvivalModelArtifacts,
    mortality_model: &SurvivalModelArtifacts,
) -> Result<(f64, Array1<f64>), SurvivalError> {
    if t1 <= t0 {
        return Ok((0.0, Array1::zeros(disease_model.coefficients.len())));
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
    let h_dis_t0 = cumulative_hazard(t0, covariates, disease_model)?;
    let h_mor_t0 = cumulative_hazard(t0, covariates, mortality_model)?;

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

    let coeff_len_d = disease_model.coefficients.len();
    let coeff_len_m = mortality_model.coefficients.len();

    let mut design_d = Array1::zeros(coeff_len_d);
    let mut deriv_d = Array1::zeros(coeff_len_d);

    let mut design_m = Array1::zeros(coeff_len_m);
    let mut deriv_m = Array1::zeros(coeff_len_m); // unused but needed for sig

    // 3. Integrate segment by segment
    let mut total_risk = 0.0;
    let mut total_gradient = Array1::zeros(coeff_len_d);
    let nodes_weights = gauss_legendre_quadrature();

    // Design row at entry age for gradient term involving H_D(t0) X(t0)
    design_and_derivative_at_age_scratch(
        t0,
        covariates.view(),
        disease_model,
        &mut design_d,
        &mut deriv_d,
        &mut scratch_d,
    )?;
    let design_d_t0 = design_d.clone();

    for i in 0..breakpoints.len() - 1 {
        let a = breakpoints[i];
        let b = breakpoints[i + 1];
        let center = 0.5 * (b + a);
        let half_width = 0.5 * (b - a);

        for &(x, w) in nodes_weights {
            let u = center + half_width * x;

            // Eval Disease: h(u), H(u)
            design_and_derivative_at_age_scratch(
                u,
                covariates.view(),
                disease_model,
                &mut design_d,
                &mut deriv_d,
                &mut scratch_d,
            )?;

            let eta_d = design_d.dot(&disease_model.coefficients);
            let slope_d = deriv_d.dot(&disease_model.coefficients);
            let hazard_d = eta_d.exp();
            let inst_hazard_d = (hazard_d * slope_d).max(0.0);

            // Eval Mortality: H(u) (hazard not needed for integrand)
            design_and_derivative_at_age_scratch(
                u,
                covariates.view(),
                mortality_model,
                &mut design_m,
                &mut deriv_m,
                &mut scratch_m,
            )?;
            let eta_m = design_m.dot(&mortality_model.coefficients);
            let hazard_m = eta_m.exp();

            // Conditional Cumulative Hazards
            // Ensure non-negative via max(0.0) to handle potential spline wiggle near t0
            let h_dis_cond = (hazard_d - h_dis_t0).max(0.0);
            let h_mor_cond = (hazard_m - h_mor_t0).max(0.0);

            // Survival Function S_total(u|t0)
            let s_total = (-(h_dis_cond + h_mor_cond)).exp();

            // Accumulate
            // Integral += h_dis(u) * S_total(u) * du
            // weight scaled by half_width due to change of variables
            total_risk += w * inst_hazard_d * s_total * half_width;

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
                total_gradient += &grad_contrib;
            }
        }
    }

    Ok((total_risk, total_gradient))
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
                    let sum = factor
                        .lower
                        .row(r)
                        .slice(s![..r])
                        .dot(&y.slice(s![..r]));
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

impl SurvivalModelArtifacts {
    pub fn apply_logit_risk_calibrator(
        &self,
        conditional_risk: &Array1<f64>,
        logit_risk_design: &Array2<f64>,
    ) -> Result<Array1<f64>, SurvivalError> {
        let cal = self
            .calibrator
            .as_ref()
            .ok_or_else(|| SurvivalError::Calibrator("Missing calibrator".to_string()))?;

        let n = conditional_risk.len();
        let mut logit_risk = Array1::zeros(n);
        for i in 0..n {
            let p = conditional_risk[i].max(1e-12).min(1.0 - 1e-12);
            logit_risk[i] = (p / (1.0 - p)).ln();
        }

        // Compute SE
        let se = if let Some(factor) = &self.hessian_factor {
            delta_method_standard_errors(factor, logit_risk_design)?
        } else {
            Array1::zeros(n)
        };

        // Distance (0 for now)
        let dist = Array1::zeros(n);

        // Predict
        let preds = crate::calibrate::calibrator::predict_calibrator(
            cal,
            logit_risk.view(),
            se.view(),
            dist.view(),
        )
        .map_err(|e| SurvivalError::Calibrator(e.to_string()))?;

        Ok(preds)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::calibrator::{CalibratorModel, CalibratorSpec};
    use crate::calibrate::model::{
        BasisConfig, LinkFunction, reset_calibrator_flag, set_calibrator_enabled,
    };
    use approx::assert_abs_diff_eq;
    use log::Level;
    use logtest::Logger;
    use ndarray::array;
    use serde_json;
    use serial_test::serial;

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

    fn toy_training_data() -> SurvivalTrainingData {
        SurvivalTrainingData {
            age_entry: array![50.0, 55.0, 60.0],
            age_exit: array![55.0, 60.0, 65.0],
            event_target: array![1, 0, 1],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            pgs: array![0.1, -0.2, 0.3],
            sex: array![0.0, 1.0, 0.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03], [-0.04, 0.05]],
            extra_static_covariates: Array2::<f64>::zeros((3, 0)),
            extra_static_names: Vec::new(),
        }
    }

    #[test]
    #[serial]
    fn apply_logit_risk_calibrator_respects_toggle() {
        reset_calibrator_flag();
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        let layout = layout_bundle.layout;
        let p = layout.combined_exit.ncols();

        let identity_calibrator = CalibratorModel {
            spec: CalibratorSpec {
                link: LinkFunction::Logit,
                pred_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                se_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                dist_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                penalty_order_pred: 2,
                penalty_order_se: 2,
                penalty_order_dist: 2,
                distance_enabled: false,
                distance_hinge: false,
                prior_weights: None,
                firth: None,
            },
            knots_pred: Array1::zeros(0),
            knots_se: Array1::zeros(0),
            knots_dist: Array1::zeros(0),
            pred_constraint_transform: Array2::zeros((0, 0)),
            stz_se: Array2::zeros((0, 0)),
            stz_dist: Array2::zeros((0, 0)),
            penalty_nullspace_dims: (0, 0, 0, 0),
            standardize_pred: (0.0, 1.0),
            standardize_se: (0.0, 1.0),
            standardize_dist: (0.0, 1.0),
            interaction_center_pred: None,
            se_wiggle_only_drop: false,
            dist_wiggle_only_drop: false,
            lambda_pred: 1.0,
            lambda_pred_param: 1.0,
            lambda_se: 1.0,
            lambda_dist: 1.0,
            coefficients: Array1::zeros(0),
            column_spans: (0..0, 0..0, 0..0),
            pred_param_range: 0..0,
            scale: None,
            assumes_frequency_weights: true,
        };

        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(p),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: Some(HessianFactor::Expected {
                factor: CholeskyFactor {
                    lower: Array2::<f64>::eye(p),
                },
            }),
            calibrator: Some(identity_calibrator),
        };

        let risks = array![0.2, 0.8];
        let mut design = Array2::<f64>::zeros((2, p));
        if p > 0 {
            design[[0, 0]] = 1.0;
        }
        if p > 1 {
            design[[1, 1]] = 1.0;
        }

        // We assume enabled by default or we enable it
        set_calibrator_enabled(true);
        let calibrated = artifacts
            .apply_logit_risk_calibrator(&risks, &design)
            .unwrap();
        assert_eq!(calibrated.len(), risks.len());
        // Since coeffs are zero, output should be transformed but likely different if standardization or basis shifts.
        // We just check that toggling it off yields original values.

        set_calibrator_enabled(false);
        let bypass = artifacts
            .apply_logit_risk_calibrator(&risks, &design)
            .unwrap();
        for (cal, base) in bypass.iter().zip(risks.iter()) {
            assert_abs_diff_eq!(*cal, *base, epsilon = 1e-12);
        }
        reset_calibrator_flag();
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
        let mut model = WorkingModelSurvival::new(
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
            create_difference_penalty_matrix(baseline_cols, order).expect("baseline penalty");
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

    trait LogitExt { fn logit(self) -> f64; }
    impl LogitExt for f64 { fn logit(self) -> f64 { (self / (1.0 - self)).ln() } }

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
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity, spec).unwrap();

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

        let mut model =
            WorkingModelSurvival::new(layout, &data, monotonicity, SurvivalSpec::default())
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
        let direct =
            conditional_absolute_risk(55.0, 60.0, &covs, &registry["companion"]).unwrap();
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
        let mut model =
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
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let baseline_cols = layout.baseline_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..baseline_cols {
            beta[idx] = 0.05 * (idx as f64 + 1.0);
        }

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
        let data = SurvivalTrainingData {
            age_entry: array![45.0, 60.0, 58.0],
            age_exit: array![50.0, 66.0, 65.0],
            event_target: array![1, 0, 1],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            pgs: array![0.05, -0.1, 0.2],
            sex: array![0.0, 1.0, 0.0],
            pcs: array![[0.01, -0.02], [0.03, 0.04], [-0.05, 0.06]],
            extra_static_covariates: Array2::<f64>::zeros((3, 0)),
            extra_static_names: Vec::new(),
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
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.02 * (idx as f64 + 1.0);
        }

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

        let baseline_cols = layout.baseline_exit.ncols();
        let time_cols = layout
            .time_varying_exit
            .as_ref()
            .map(|matrix| matrix.ncols())
            .unwrap_or(0);
        let static_cols = layout.static_covariates.ncols();
        let extra_cols = layout.extra_static_covariates.ncols();

        let mut beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        let weights = Array1::from_elem(layout.baseline_derivative_exit.nrows(), 0.5);
        let baseline_beta = layout.baseline_derivative_exit.t().dot(&weights);
        beta.slice_mut(s![..baseline_cols]).assign(&baseline_beta);

        let static_offset = baseline_cols + time_cols;
        if static_cols > 0 {
            for (idx, value) in beta
                .slice_mut(s![static_offset..static_offset + static_cols])
                .iter_mut()
                .enumerate()
            {
                *value = -0.1 + 0.05 * (idx as f64);
            }
        }
        if extra_cols > 0 {
            for (idx, value) in beta
                .slice_mut(s![
                    static_offset + static_cols..static_offset + static_cols + extra_cols
                ])
                .iter_mut()
                .enumerate()
            {
                *value = 0.02 * (idx as f64 + 1.0);
            }
        }

        let derivative_exit = layout.combined_derivative_exit.dot(&beta);
        assert!(
            derivative_exit
                .iter()
                .all(|value| *value > spec.derivative_guard)
        );

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
                assert_abs_diff_eq!(numeric, analytic, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn frequency_weights_match_replication() {
        let weighted_data = SurvivalTrainingData {
            age_entry: array![50.0, 55.0],
            age_exit: array![55.0, 60.0],
            event_target: array![1, 0],
            event_competing: array![0, 0],
            sample_weight: array![1.0, 2.0],
            pgs: array![0.1, -0.3],
            sex: array![0.0, 1.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03]],
            extra_static_covariates: Array2::<f64>::zeros((2, 0)),
            extra_static_names: Vec::new(),
        };

        let expanded_data = SurvivalTrainingData {
            age_entry: array![50.0, 55.0, 55.0],
            age_exit: array![55.0, 60.0, 60.0],
            event_target: array![1, 0, 0],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            pgs: array![0.1, -0.3, -0.3],
            sex: array![0.0, 1.0, 1.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03], [0.02, 0.03]],
            extra_static_covariates: Array2::<f64>::zeros((3, 0)),
            extra_static_names: Vec::new(),
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

        let mut weighted_model = WorkingModelSurvival::new(
            layout_weighted.clone(),
            &weighted_data,
            monotonic_weighted.clone(),
            spec,
        )
        .unwrap();
        let mut expanded_model = WorkingModelSurvival::new(
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
        let (pgs_basis_full, _) = create_bspline_basis_with_knots(
            data.pgs.view(),
            pgs_basis.knot_vector.view(),
            pgs_basis.degree,
        )
        .unwrap();
        let mut pgs_basis_matrix = pgs_basis_full.slice(s![.., 1..]).to_owned();
        let raw_means = compute_weighted_column_means(&pgs_basis_matrix, &data.sample_weight);
        assert_eq!(raw_means.len(), offsets.len());
        for (raw, offset) in raw_means.iter().zip(offsets.iter()) {
            assert_abs_diff_eq!(raw, offset, epsilon = 1e-10);
        }
        for (mut column, &offset) in pgs_basis_matrix.axis_iter_mut(Axis(1)).zip(offsets.iter()) {
            column.mapv_inplace(|value| value - offset);
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
        };

        let hazard = cumulative_hazard(data.age_exit[0], &covariates, &artifacts).unwrap();
        assert_abs_diff_eq!(hazard, eta_exit[0].exp(), epsilon = 1e-10);

        let pgs_idx = artifacts
            .static_covariate_layout
            .column_names
            .iter()
            .position(|name| name == "pgs")
            .expect("pgs column present");
        let mut covariates_high = covariates.clone();
        covariates_high[pgs_idx] = metadata.value_ranges[0].max + 0.5;
        let err_high = design_row_at_age(data.age_exit[0], covariates_high.view(), &artifacts)
            .expect_err("pgs above range should error");
        assert!(matches!(
            err_high,
            SurvivalError::CovariateAboveRange { .. }
        ));

        let mut covariates_low = covariates;
        covariates_low[pgs_idx] = metadata.value_ranges[0].min - 0.5;
        let err_low = design_row_at_age(data.age_exit[0], covariates_low.view(), &artifacts)
            .expect_err("pgs below range should error");
        assert!(matches!(err_low, SurvivalError::CovariateBelowRange { .. }));

        let mut model = WorkingModelSurvival::new(
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
