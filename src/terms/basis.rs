use crate::faer_ndarray::{
    FaerEigh, FaerLinalgError, default_rrqr_rank_alpha, fast_ab, fast_ata, fast_atb,
    rrqr_nullspace_basis,
};
use faer::Side;
use faer::sparse::{SparseColMat, Triplet};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

#[cfg(test)]
use approx::assert_abs_diff_eq;

/// A comprehensive error type for all operations within the basis module.
#[derive(Error, Debug)]
pub enum BasisError {
    #[error("Spline degree must be at least 1, but was {0}.")]
    InvalidDegree(usize),

    #[error("Data range is invalid: start ({0}) must be less than or equal to end ({1}).")]
    InvalidRange(f64, f64),

    #[error(
        "Data range has zero width (min equals max) but {0} internal knots were requested, which would create coincident knots."
    )]
    DegenerateRange(usize),

    #[error(
        "Penalty order ({order}) must be positive and less than the number of basis functions ({num_basis})."
    )]
    InvalidPenaltyOrder { order: usize, num_basis: usize },

    #[error(
        "Insufficient knots for degree {degree} spline: need at least {required} knots but only {provided} were provided."
    )]
    InsufficientKnotsForDegree {
        degree: usize,
        required: usize,
        provided: usize,
    },

    #[error(
        "Cannot apply sum-to-zero constraint: requires at least 2 basis functions, but only {found} were provided."
    )]
    InsufficientColumnsForConstraint { found: usize },

    #[error(
        "Constraint matrix must have the same number of rows as the basis: basis has {basis_rows}, constraint has {constraint_rows}."
    )]
    ConstraintMatrixRowMismatch {
        basis_rows: usize,
        constraint_rows: usize,
    },

    #[error(
        "Weights dimension mismatch: expected {expected} weights to match basis matrix rows, but got {found}."
    )]
    WeightsDimensionMismatch { expected: usize, found: usize },

    #[error("QR decomposition failed while applying constraints: {0}")]
    LinalgError(#[from] FaerLinalgError),

    #[error("Failed to identify a constraint nullspace basis; matrix is ill-conditioned.")]
    ConstraintNullspaceNotFound,

    #[error(
        "Knot vector is degenerate: all Greville abscissae are equal, so linear constraint cannot be applied."
    )]
    DegenerateKnots,

    #[error(
        "The provided knot vector is invalid: {0}. It must be non-decreasing and contain only finite values."
    )]
    InvalidKnotVector(String),

    #[error("Failed to build sparse basis matrix: {0}")]
    SparseCreation(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error(
        "Indefinite penalty matrix in {context}: minimum eigenvalue {min_eigenvalue:.3e} is below tolerance {tolerance:.3e}. {guidance}"
    )]
    IndefinitePenalty {
        context: String,
        min_eigenvalue: f64,
        tolerance: f64,
        guidance: String,
    },

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

// ============================================================================
// Unified Basis Generation API
// ============================================================================

/// Options for basis generation, controlling derivative order.
#[derive(Clone, Copy, Debug, Default)]
pub struct BasisOptions {
    /// Derivative order: 0 = value (default), 1 = first derivative, 2 = second derivative
    pub derivative_order: usize,
    /// Basis family to evaluate.
    pub basis_family: BasisFamily,
}

impl BasisOptions {
    /// Create options for evaluating basis functions (no derivative).
    pub fn value() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating first derivatives of basis functions.
    pub fn first_derivative() -> Self {
        Self {
            derivative_order: 1,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating second derivatives of basis functions.
    pub fn second_derivative() -> Self {
        Self {
            derivative_order: 2,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating M-spline basis values.
    pub fn m_spline() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::MSpline,
        }
    }

    /// Create options for evaluating I-spline basis values.
    pub fn i_spline() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::ISpline,
        }
    }
}

/// Basis-family selector for 1D spline evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasisFamily {
    /// Standard B-splines.
    #[default]
    BSpline,
    /// M-splines: normalized B-splines, M_i = ((k+1)/(t_{i+k+1}-t_i)) B_i.
    MSpline,
    /// I-splines: integrated M-splines, implemented by right-cumulative
    /// sums of B-splines at degree k+1.
    ISpline,
}

/// Specifies the source of knots for basis generation.
#[derive(Clone, Debug)]
pub enum KnotSource<'a> {
    /// Use a pre-computed knot vector.
    Provided(ArrayView1<'a, f64>),
    /// Generate uniformly spaced knots based on data range.
    Generate {
        /// Data range (min, max) for knot placement.
        data_range: (f64, f64),
        /// Number of internal knots to place between boundaries.
        num_internal_knots: usize,
    },
}

/// Marker type for dense basis matrix output.
pub struct Dense;

/// Marker type for sparse basis matrix output.
pub struct Sparse;

/// Trait for selecting basis storage format at compile time.
pub trait BasisOutput {
    type Output;
}

impl BasisOutput for Dense {
    type Output = Arc<Array2<f64>>;
}

impl BasisOutput for Sparse {
    type Output = SparseColMat<usize, f64>;
}

/// Unified B-spline basis generation with configurable storage, knot source, and options.
///
/// This function consolidates various basis generation functions into a single entry point.
/// Use type parameters to select output format:
/// - `create_basis::<Dense>(...)` for dense `Array2<f64>` output
/// - `create_basis::<Sparse>(...)` for sparse `SparseColMat` output
///
/// # Arguments
/// * `data` - Data points to evaluate basis at
/// * `knot_source` - Either pre-computed knots or parameters for uniform generation
/// * `degree` - B-spline degree (e.g., 3 for cubic)
/// * `options` - Derivative order and other options
///
/// # Returns
/// Tuple of (basis matrix, knot vector used)
pub fn create_basis<O: BasisOutputFormat>(
    data: ArrayView1<f64>,
    knot_source: KnotSource<'_>,
    degree: usize,
    options: BasisOptions,
) -> Result<(O::Output, Array1<f64>), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }

    if options.basis_family != BasisFamily::BSpline && options.derivative_order != 0 {
        return Err(BasisError::InvalidInput(
            "derivatives are only supported for BasisFamily::BSpline".to_string(),
        ));
    }

    let eval_kind = match options.derivative_order {
        0 => BasisEvalKind::Basis,
        1 => BasisEvalKind::FirstDerivative,
        2 => BasisEvalKind::SecondDerivative,
        n => {
            return Err(BasisError::InvalidInput(format!(
                "unsupported derivative order {n}; only 0, 1, 2 are supported"
            )));
        }
    };

    let knot_degree = match options.basis_family {
        BasisFamily::BSpline | BasisFamily::MSpline => degree,
        BasisFamily::ISpline => degree
            .checked_add(1)
            .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?,
    };

    let knot_vec: Array1<f64> = match knot_source {
        KnotSource::Provided(view) => {
            validate_knots_for_degree(view, knot_degree)?;
            view.to_owned()
        }
        KnotSource::Generate {
            data_range,
            num_internal_knots,
        } => {
            if data_range.0 > data_range.1 {
                return Err(BasisError::InvalidRange(data_range.0, data_range.1));
            }
            if data_range.0 == data_range.1 && num_internal_knots > 0 {
                return Err(BasisError::DegenerateRange(num_internal_knots));
            }
            internal::generate_full_knot_vector(data_range, num_internal_knots, knot_degree)?
        }
    };

    match options.basis_family {
        BasisFamily::BSpline => O::build_basis(data, degree, eval_kind, knot_vec),
        BasisFamily::MSpline => {
            if O::IS_SPARSE {
                let sparse = create_mspline_sparse(data, knot_vec.view(), degree)?;
                Ok((O::from_sparse(sparse)?, knot_vec))
            } else {
                let dense = create_mspline_dense(data, knot_vec.view(), degree)?;
                Ok((O::from_dense(dense)?, knot_vec))
            }
        }
        BasisFamily::ISpline => {
            if O::IS_SPARSE {
                return Err(BasisError::InvalidInput(
                    "BasisFamily::ISpline does not support sparse output; use Dense".to_string(),
                ));
            }
            let dense = create_ispline_dense(data, knot_vec.view(), degree)?;
            Ok((O::from_dense(dense)?, knot_vec))
        }
    }
}

/// Applies first-order linear extension outside a knot-domain interval to a basis matrix
/// that was evaluated at clamped coordinates.
///
/// Given `z_raw` and `z_clamped = clamp(z_raw, left, right)`, this mutates
/// `basis_values` in-place as:
/// `B_ext(z_raw) = B(z_clamped) + (z_raw - z_clamped) * B'(z_clamped)`.
pub fn apply_linear_extension_from_first_derivative(
    z_raw: ArrayView1<f64>,
    z_clamped: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    basis_values: &mut Array2<f64>,
) -> Result<(), BasisError> {
    if z_raw.len() != z_clamped.len() {
        return Err(BasisError::DimensionMismatch(
            "z_raw and z_clamped must have equal length".to_string(),
        ));
    }
    if basis_values.nrows() != z_raw.len() {
        return Err(BasisError::DimensionMismatch(
            "basis row count must match z length".to_string(),
        ));
    }

    let mut needs_ext = false;
    for i in 0..z_raw.len() {
        if (z_raw[i] - z_clamped[i]).abs() > 1e-12 {
            needs_ext = true;
            break;
        }
    }
    if !needs_ext {
        return Ok(());
    }

    let (b_prime_arc, _) = create_basis::<Dense>(
        z_clamped,
        KnotSource::Provided(knot_vector),
        degree,
        BasisOptions::first_derivative(),
    )?;
    let b_prime = b_prime_arc.as_ref();
    if b_prime.nrows() != basis_values.nrows() || b_prime.ncols() != basis_values.ncols() {
        return Err(BasisError::DimensionMismatch(
            "basis derivative shape mismatch".to_string(),
        ));
    }

    for i in 0..z_raw.len() {
        let dz = z_raw[i] - z_clamped[i];
        if dz.abs() <= 1e-12 {
            continue;
        }
        for j in 0..basis_values.ncols() {
            basis_values[[i, j]] += dz * b_prime[[i, j]];
        }
    }
    Ok(())
}

/// Trait for building basis matrices with different storage formats.
/// This is an implementation detail for the unified `create_basis` function.
pub trait BasisOutputFormat {
    type Output;
    const IS_SPARSE: bool;

    fn build_basis(
        data: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        knot_vec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError>;

    fn from_dense(dense: Array2<f64>) -> Result<Self::Output, BasisError>;
    fn from_sparse(sparse: SparseColMat<usize, f64>) -> Result<Self::Output, BasisError>;
}

impl BasisOutputFormat for Dense {
    type Output = Arc<Array2<f64>>;
    const IS_SPARSE: bool = false;

    fn build_basis(
        data: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        knot_vec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError> {
        let knot_view = knot_vec.view();

        let num_basis_functions = knot_view.len().saturating_sub(degree + 1);
        let basis_matrix = if should_use_sparse_basis(num_basis_functions, degree, 1) {
            let sparse = generate_basis_internal::<SparseStorage>(
                data.view(),
                knot_view,
                degree,
                eval_kind,
            )?;
            let mut dense = Array2::<f64>::zeros((sparse.nrows(), sparse.ncols()));
            let (symbolic, values) = sparse.parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            for col in 0..sparse.ncols() {
                let start = col_ptr[col];
                let end = col_ptr[col + 1];
                for idx in start..end {
                    dense[[row_idx[idx], col]] += values[idx];
                }
            }
            apply_dense_bspline_extrapolation(data, knot_view, degree, eval_kind, &mut dense)?;
            dense
        } else {
            generate_basis_internal::<DenseStorage>(data.view(), knot_view, degree, eval_kind)?
        };

        Ok((Arc::new(basis_matrix), knot_vec))
    }

    fn from_dense(dense: Array2<f64>) -> Result<Self::Output, BasisError> {
        Ok(Arc::new(dense))
    }

    fn from_sparse(sparse: SparseColMat<usize, f64>) -> Result<Self::Output, BasisError> {
        let mut dense = Array2::<f64>::zeros((sparse.nrows(), sparse.ncols()));
        let (symbolic, values) = sparse.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..sparse.ncols() {
            let start = col_ptr[col];
            let end = col_ptr[col + 1];
            for idx in start..end {
                dense[[row_idx[idx], col]] += values[idx];
            }
        }
        Ok(Arc::new(dense))
    }
}

fn apply_dense_bspline_extrapolation(
    data: ArrayView1<f64>,
    knot_view: ArrayView1<f64>,
    degree: usize,
    eval_kind: BasisEvalKind,
    basis_matrix: &mut Array2<f64>,
) -> Result<(), BasisError> {
    let num_basis_functions = basis_matrix.ncols();
    if num_basis_functions == 0 {
        return Ok(());
    }

    let left = knot_view[degree];
    let right = knot_view[num_basis_functions];

    if matches!(eval_kind, BasisEvalKind::FirstDerivative) {
        let num_basis_lower = knot_view.len().saturating_sub(degree);
        let mut lower_basis = vec![0.0; num_basis_lower];
        let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(1));
        for (i, &x) in data.iter().enumerate() {
            if x >= left && x <= right {
                continue;
            }
            let x_c = x.clamp(left, right);
            let mut row = basis_matrix.row_mut(i);
            let row_slice = row
                .as_slice_mut()
                .expect("basis matrix rows should be contiguous");
            evaluate_bspline_derivative_scalar_into(
                x_c,
                knot_view,
                degree,
                row_slice,
                &mut lower_basis,
                &mut lower_scratch,
            )?;
        }
    }

    if matches!(eval_kind, BasisEvalKind::Basis) {
        let z_clamped = data.mapv(|x| x.clamp(left, right));
        apply_linear_extension_from_first_derivative(
            data,
            z_clamped.view(),
            knot_view,
            degree,
            basis_matrix,
        )?;
    }

    Ok(())
}

impl BasisOutputFormat for Sparse {
    type Output = SparseColMat<usize, f64>;
    const IS_SPARSE: bool = true;

    fn build_basis(
        data: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        knot_vec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError> {
        let knot_view = knot_vec.view();
        let sparse =
            generate_basis_internal::<SparseStorage>(data.view(), knot_view, degree, eval_kind)?;
        Ok((sparse, knot_vec))
    }

    fn from_dense(dense: Array2<f64>) -> Result<Self::Output, BasisError> {
        let (nrows, ncols) = dense.dim();
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        triplets.reserve(nrows.saturating_mul(ncols / 8));
        for i in 0..nrows {
            for j in 0..ncols {
                let v = dense[[i, j]];
                if v.abs() > 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }
        SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
            .map_err(|e| BasisError::SparseCreation(format!("{e:?}")))
    }

    fn from_sparse(sparse: SparseColMat<usize, f64>) -> Result<Self::Output, BasisError> {
        Ok(sparse)
    }
}

/// Compute a heuristic smoothing weight based on knot span, penalty order, and spline degree.
pub fn baseline_lambda_seed(knot_vector: &Array1<f64>, degree: usize, penalty_order: usize) -> f64 {
    let mut min_knot = f64::INFINITY;
    let mut max_knot = f64::NEG_INFINITY;
    for &value in knot_vector.iter() {
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
    let degree = degree.max(1) as f64;
    let normalized_span = (span / (span + 1.0)).max(1e-3);
    let lambda = 0.5 * (order / (degree + 1.0)) / normalized_span;
    lambda.clamp(1e-6, 1e3)
}

fn validate_knots_for_degree(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }

    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    if knot_vector.iter().any(|&k| !k.is_finite()) {
        return Err(BasisError::InvalidKnotVector(
            "knot vector contains non-finite (NaN or Infinity) values".to_string(),
        ));
    }

    if knot_vector.len() >= 2 {
        for i in 0..(knot_vector.len() - 1) {
            if knot_vector[i] > knot_vector[i + 1] {
                return Err(BasisError::InvalidKnotVector(
                    "knot vector is not non-decreasing".to_string(),
                ));
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum BasisEvalKind {
    Basis,
    FirstDerivative,
    SecondDerivative,
}

struct BasisEvalScratch {
    basis: internal::BsplineScratch,
    lower_basis: Vec<f64>,
    lower_scratch: internal::BsplineScratch,
    lower_lower_basis: Vec<f64>,
    lower_lower_scratch: internal::BsplineScratch,
}

impl BasisEvalScratch {
    fn new(degree: usize) -> Self {
        let lower_degree = degree.saturating_sub(1);
        let lower_lower_degree = degree.saturating_sub(2);
        Self {
            basis: internal::BsplineScratch::new(degree),
            lower_basis: vec![0.0; lower_degree + 1],
            lower_scratch: internal::BsplineScratch::new(lower_degree),
            lower_lower_basis: vec![0.0; lower_lower_degree + 1],
            lower_lower_scratch: internal::BsplineScratch::new(lower_lower_degree),
        }
    }
}

fn evaluate_splines_derivative_sparse_into_with_lower(
    x: f64,
    degree: usize,
    knot_view: ArrayView1<f64>,
    values: &mut [f64],
    basis_scratch: &mut internal::BsplineScratch,
    lower_values: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> usize {
    let num_basis = knot_view.len().saturating_sub(degree + 1);
    let x_eval = if num_basis > 0 {
        let left = knot_view[degree];
        let right = knot_view[num_basis];
        x.clamp(left, right)
    } else {
        x
    };
    if num_basis > 0 {
        // Linear extrapolation outside the domain uses the boundary slope, so
        // first derivatives clamp to the nearest boundary derivative value.
    }

    let start_col =
        internal::evaluate_splines_sparse_into(x_eval, degree, knot_view, values, basis_scratch);
    if degree == 0 {
        values.fill(0.0);
        return start_col;
    }

    let lower_degree = degree - 1;
    let lower_support = lower_degree + 1;
    if lower_values.len() != lower_support {
        return start_col;
    }

    let start_lower = internal::evaluate_splines_sparse_into(
        x_eval,
        lower_degree,
        knot_view,
        lower_values,
        lower_scratch,
    );

    values.fill(0.0);
    for offset in 0..=degree {
        let i = start_col + offset;
        let left_idx = i as isize - start_lower as isize;
        let right_idx = (i + 1) as isize - start_lower as isize;
        let left = if left_idx >= 0 && (left_idx as usize) < lower_support {
            lower_values[left_idx as usize]
        } else {
            0.0
        };
        let right = if right_idx >= 0 && (right_idx as usize) < lower_support {
            lower_values[right_idx as usize]
        } else {
            0.0
        };
        let denom_left = knot_view[i + degree] - knot_view[i];
        let denom_right = knot_view[i + degree + 1] - knot_view[i + 1];
        let left_term = if denom_left.abs() > 1e-12 {
            left / denom_left
        } else {
            0.0
        };
        let right_term = if denom_right.abs() > 1e-12 {
            right / denom_right
        } else {
            0.0
        };
        values[offset] = (degree as f64) * (left_term - right_term);
    }

    start_col
}

fn evaluate_splines_derivative_sparse_into(
    x: f64,
    degree: usize,
    knot_view: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let lower_degree = degree.saturating_sub(1);
    let lower_support = lower_degree + 1;
    if scratch.lower_basis.len() != lower_support {
        scratch.lower_basis.resize(lower_support, 0.0);
        scratch.lower_scratch.ensure_degree(lower_degree);
    }
    evaluate_splines_derivative_sparse_into_with_lower(
        x,
        degree,
        knot_view,
        values,
        &mut scratch.basis,
        &mut scratch.lower_basis,
        &mut scratch.lower_scratch,
    )
}

fn evaluate_splines_second_derivative_sparse_into(
    x: f64,
    degree: usize,
    knot_view: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let num_basis = knot_view.len().saturating_sub(degree + 1);
    if num_basis > 0 {
        let left = knot_view[degree];
        let right = knot_view[num_basis];
        // Constant extrapolation outside the domain implies zero derivatives.
        if x < left || x > right {
            values.fill(0.0);
            return 0;
        }
    }

    let start_col =
        internal::evaluate_splines_sparse_into(x, degree, knot_view, values, &mut scratch.basis);
    if degree < 2 {
        values.fill(0.0);
        return start_col;
    }

    let lower_degree = degree - 1;
    let lower_support = lower_degree + 1;
    if scratch.lower_basis.len() != lower_support {
        scratch.lower_basis.resize(lower_support, 0.0);
        scratch.lower_scratch.ensure_degree(lower_degree);
    }

    let lower_lower_degree = lower_degree.saturating_sub(1);
    let lower_lower_support = lower_lower_degree + 1;
    if scratch.lower_lower_basis.len() != lower_lower_support {
        scratch.lower_lower_basis.resize(lower_lower_support, 0.0);
        scratch
            .lower_lower_scratch
            .ensure_degree(lower_lower_degree);
    }

    // Build B'_{i, k-1}(x) (first derivative of the lower-degree basis, k-1).
    // We then apply the derivative recursion one more time:
    // B''_{i,k}(x) = k * ( B'_{i,k-1}(x)/(t_{i+k}-t_i)
    //                  -B'_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}) )
    //
    // So `scratch.lower_basis` below stores derivative values, not raw basis values.
    let start_lower = evaluate_splines_derivative_sparse_into_with_lower(
        x,
        lower_degree,
        knot_view,
        &mut scratch.lower_basis,
        &mut scratch.lower_scratch,
        &mut scratch.lower_lower_basis,
        &mut scratch.lower_lower_scratch,
    );

    values.fill(0.0);
    for offset in 0..=degree {
        let i = start_col + offset;
        let left_idx = i as isize - start_lower as isize;
        let right_idx = (i + 1) as isize - start_lower as isize;
        // These are B'_{i,k-1} and B'_{i+1,k-1} aligned from the sparse lower block.
        let left = if left_idx >= 0 && (left_idx as usize) < lower_support {
            scratch.lower_basis[left_idx as usize]
        } else {
            0.0
        };
        let right = if right_idx >= 0 && (right_idx as usize) < lower_support {
            scratch.lower_basis[right_idx as usize]
        } else {
            0.0
        };
        let denom_left = knot_view[i + degree] - knot_view[i];
        let denom_right = knot_view[i + degree + 1] - knot_view[i + 1];
        let left_term = if denom_left.abs() > 1e-12 {
            left / denom_left
        } else {
            0.0
        };
        let right_term = if denom_right.abs() > 1e-12 {
            right / denom_right
        } else {
            0.0
        };
        values[offset] = (degree as f64) * (left_term - right_term);
    }

    start_col
}

fn evaluate_splines_sparse_with_kind(
    x: f64,
    degree: usize,
    knot_view: ArrayView1<f64>,
    eval_kind: BasisEvalKind,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    match eval_kind {
        BasisEvalKind::Basis => {
            internal::evaluate_splines_sparse_into(x, degree, knot_view, values, &mut scratch.basis)
        }
        BasisEvalKind::FirstDerivative => {
            evaluate_splines_derivative_sparse_into(x, degree, knot_view, values, scratch)
        }
        BasisEvalKind::SecondDerivative => {
            evaluate_splines_second_derivative_sparse_into(x, degree, knot_view, values, scratch)
        }
    }
}

fn evaluate_bspline_row_entries<F>(
    x: f64,
    degree: usize,
    knot_view: ArrayView1<f64>,
    eval_kind: BasisEvalKind,
    num_basis_functions: usize,
    scratch: &mut BasisEvalScratch,
    values: &mut [f64],
    mut write_entry: F,
) where
    F: FnMut(usize, f64),
{
    if num_basis_functions > 0 {
        let left = knot_view[degree];
        let right = knot_view[num_basis_functions];
        if x < left || x > right {
            match eval_kind {
                BasisEvalKind::Basis => {
                    let x_c = x.clamp(left, right);
                    let dz = x - x_c;
                    let mut deriv_values = vec![0.0; values.len()];
                    values.fill(0.0);
                    let basis_start = internal::evaluate_splines_sparse_into(
                        x_c,
                        degree,
                        knot_view,
                        values,
                        &mut scratch.basis,
                    );
                    let deriv_start = evaluate_splines_derivative_sparse_into(
                        x_c,
                        degree,
                        knot_view,
                        &mut deriv_values,
                        scratch,
                    );
                    let mut cols = Vec::<usize>::with_capacity(values.len() * 2);
                    let mut vals = Vec::<f64>::with_capacity(values.len() * 2);
                    let push_or_add =
                        |col_j: usize, value: f64, cols: &mut Vec<usize>, vals: &mut Vec<f64>| {
                            if value == 0.0 {
                                return;
                            }
                            if let Some(pos) = cols.iter().position(|&col| col == col_j) {
                                vals[pos] += value;
                            } else {
                                cols.push(col_j);
                                vals.push(value);
                            }
                        };
                    for (offset, &v) in values.iter().enumerate() {
                        let col_j = basis_start + offset;
                        if col_j < num_basis_functions {
                            push_or_add(col_j, v, &mut cols, &mut vals);
                        }
                    }
                    for (offset, &v) in deriv_values.iter().enumerate() {
                        let col_j = deriv_start + offset;
                        if col_j < num_basis_functions {
                            push_or_add(col_j, dz * v, &mut cols, &mut vals);
                        }
                    }
                    for (col_j, v) in cols.into_iter().zip(vals.into_iter()) {
                        if v != 0.0 {
                            write_entry(col_j, v);
                        }
                    }
                    return;
                }
                BasisEvalKind::FirstDerivative => {
                    let x_c = x.clamp(left, right);
                    let start_col = evaluate_splines_derivative_sparse_into(
                        x_c, degree, knot_view, values, scratch,
                    );
                    for (offset, &v) in values.iter().enumerate() {
                        if v == 0.0 {
                            continue;
                        }
                        let col_j = start_col + offset;
                        if col_j < num_basis_functions {
                            write_entry(col_j, v);
                        }
                    }
                    return;
                }
                BasisEvalKind::SecondDerivative => {}
            }
        }
    }

    let start_col =
        evaluate_splines_sparse_with_kind(x, degree, knot_view, eval_kind, values, scratch);
    for (offset, &v) in values.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let col_j = start_col + offset;
        if col_j < num_basis_functions {
            write_entry(col_j, v);
        }
    }
}

trait BasisStorage {
    type Output;

    fn build(
        data: ArrayView1<f64>,
        knot_view: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        num_basis_functions: usize,
        support: usize,
        use_parallel: bool,
    ) -> Result<Self::Output, BasisError>;
}

struct DenseStorage;

impl BasisStorage for DenseStorage {
    type Output = Array2<f64>;

    fn build(
        data: ArrayView1<f64>,
        knot_view: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        num_basis_functions: usize,
        support: usize,
        use_parallel: bool,
    ) -> Result<Self::Output, BasisError> {
        let mut basis_matrix = Array2::zeros((data.len(), num_basis_functions));

        if let (true, Some(data_slice)) = (use_parallel, data.as_slice()) {
            basis_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(data_slice.par_iter().copied())
                .for_each_init(
                    || (BasisEvalScratch::new(degree), vec![0.0; support]),
                    |(scratch, values), (mut row, x)| {
                        let row_slice = row
                            .as_slice_mut()
                            .expect("basis matrix rows should be contiguous");
                        evaluate_bspline_row_entries(
                            x,
                            degree,
                            knot_view,
                            eval_kind,
                            num_basis_functions,
                            scratch,
                            values,
                            |col_j, v| row_slice[col_j] = v,
                        );
                    },
                );
        } else {
            let mut scratch = BasisEvalScratch::new(degree);
            let mut values = vec![0.0; support];
            for (mut row, &x) in basis_matrix.axis_iter_mut(Axis(0)).zip(data.iter()) {
                let row_slice = row
                    .as_slice_mut()
                    .expect("basis matrix rows should be contiguous");
                evaluate_bspline_row_entries(
                    x,
                    degree,
                    knot_view,
                    eval_kind,
                    num_basis_functions,
                    &mut scratch,
                    &mut values,
                    |col_j, v| row_slice[col_j] = v,
                );
            }
        }

        apply_dense_bspline_extrapolation(data, knot_view, degree, eval_kind, &mut basis_matrix)?;

        Ok(basis_matrix)
    }
}

struct SparseStorage;

impl BasisStorage for SparseStorage {
    type Output = SparseColMat<usize, f64>;

    fn build(
        data: ArrayView1<f64>,
        knot_view: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        num_basis_functions: usize,
        support: usize,
        use_parallel: bool,
    ) -> Result<Self::Output, BasisError> {
        let nrows = data.len();

        let triplets: Vec<Triplet<usize, usize, f64>> =
            if let (true, Some(data_slice)) = (use_parallel, data.as_slice()) {
                const CHUNK_SIZE: usize = 1024;
                let triplet_chunks: Vec<Vec<Triplet<usize, usize, f64>>> = data_slice
                    .par_chunks(CHUNK_SIZE)
                    .enumerate()
                    .map_init(
                        || (BasisEvalScratch::new(degree), vec![0.0; support]),
                        |(scratch, values), (chunk_idx, chunk)| {
                            let base_row = chunk_idx * CHUNK_SIZE;
                            let mut local = Vec::with_capacity(chunk.len().saturating_mul(support));
                            for (i, &x) in chunk.iter().enumerate() {
                                let row_i = base_row + i;
                                evaluate_bspline_row_entries(
                                    x,
                                    degree,
                                    knot_view,
                                    eval_kind,
                                    num_basis_functions,
                                    scratch,
                                    values,
                                    |col_j, v| local.push(Triplet::new(row_i, col_j, v)),
                                );
                            }
                            local
                        },
                    )
                    .collect();

                let mut flattened = Vec::with_capacity(nrows.saturating_mul(support));
                for mut chunk in triplet_chunks {
                    flattened.append(&mut chunk);
                }
                flattened
            } else {
                let mut scratch = BasisEvalScratch::new(degree);
                let mut values = vec![0.0; support];
                let mut triplets = Vec::with_capacity(nrows.saturating_mul(support));

                for (row_i, &x) in data.iter().enumerate() {
                    evaluate_bspline_row_entries(
                        x,
                        degree,
                        knot_view,
                        eval_kind,
                        num_basis_functions,
                        &mut scratch,
                        &mut values,
                        |col_j, v| triplets.push(Triplet::new(row_i, col_j, v)),
                    );
                }

                triplets
            };

        SparseColMat::try_new_from_triplets(nrows, num_basis_functions, &triplets)
            .map_err(|err| BasisError::SparseCreation(format!("{err:?}")))
    }
}

fn generate_basis_internal<S: BasisStorage>(
    data: ArrayView1<f64>,
    knot_view: ArrayView1<f64>,
    degree: usize,
    eval_kind: BasisEvalKind,
) -> Result<S::Output, BasisError> {
    let num_basis_functions = knot_view.len().saturating_sub(degree + 1);
    let support = degree + 1;
    // Parallel dispatch heuristic:
    // Lower degrees have cheaper per-row evaluation and need larger batches to
    // amortize Rayon scheduling overhead. Cubic+ rows are costlier, so parallel
    // wins earlier.
    let par_threshold = match degree {
        0 | 1 => 512,
        2 | 3 => 128,
        _ => 64,
    };
    let use_parallel = data.len() >= par_threshold && data.as_slice().is_some();
    S::build(
        data,
        knot_view,
        degree,
        eval_kind,
        num_basis_functions,
        support,
        use_parallel,
    )
}

fn validate_multi_dim_inputs(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    eval_kinds: &[BasisEvalKind],
) -> Result<(usize, usize), BasisError> {
    if data.is_empty() {
        return Err(BasisError::DimensionMismatch(
            "multi-dim basis requires at least one dimension".to_string(),
        ));
    }
    if data.len() != knot_vectors.len()
        || data.len() != degrees.len()
        || data.len() != eval_kinds.len()
    {
        return Err(BasisError::DimensionMismatch(format!(
            "multi-dim inputs must be the same length: data={}, knots={}, degrees={}, eval_kinds={}",
            data.len(),
            knot_vectors.len(),
            degrees.len(),
            eval_kinds.len()
        )));
    }

    let nrows = data[0].len();
    for (idx, view) in data.iter().enumerate() {
        if view.len() != nrows {
            return Err(BasisError::DimensionMismatch(format!(
                "data length mismatch at dim {idx}: expected {nrows}, got {}",
                view.len()
            )));
        }
    }

    Ok((data.len(), nrows))
}

fn compute_tensor_strides(num_basis: &[usize]) -> Result<Vec<usize>, BasisError> {
    let mut strides = vec![1usize; num_basis.len()];
    let mut acc = 1usize;
    for i in (0..num_basis.len()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(num_basis[i])
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))?;
    }
    Ok(strides)
}

fn fill_tensor_row<F>(
    row_idx: usize,
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    eval_kinds: &[BasisEvalKind],
    num_basis: &[usize],
    supports: &[usize],
    strides: &[usize],
    scratch: &mut [BasisEvalScratch],
    values: &mut [Vec<f64>],
    starts: &mut [usize],
    indices: &mut [usize],
    mut write_entry: F,
) where
    F: FnMut(usize, usize, f64),
{
    for dim in 0..data.len() {
        let x = data[dim][row_idx];
        let start = evaluate_splines_sparse_with_kind(
            x,
            degrees[dim],
            knot_vectors[dim],
            eval_kinds[dim],
            &mut values[dim],
            &mut scratch[dim],
        );
        starts[dim] = start.min(num_basis[dim].saturating_sub(1));
    }

    indices.fill(0);
    loop {
        let mut product = 1.0f64;
        let mut col = 0usize;
        for dim in 0..data.len() {
            let v = values[dim][indices[dim]];
            product *= v;
            if product == 0.0 {
                break;
            }
            col += (starts[dim] + indices[dim]) * strides[dim];
        }
        if product != 0.0 {
            write_entry(row_idx, col, product);
        }

        let mut carried = true;
        for dim in (0..data.len()).rev() {
            indices[dim] += 1;
            if indices[dim] < supports[dim] {
                carried = false;
                break;
            }
            indices[dim] = 0;
        }
        if carried {
            break;
        }
    }
}

fn generate_basis_nd_dense(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    eval_kinds: &[BasisEvalKind],
) -> Result<Array2<f64>, BasisError> {
    let (dims, nrows) = validate_multi_dim_inputs(data, knot_vectors, degrees, eval_kinds)?;
    let mut num_basis = Vec::with_capacity(dims);
    let mut supports = Vec::with_capacity(dims);
    for dim in 0..dims {
        validate_knots_for_degree(knot_vectors[dim], degrees[dim])?;
        num_basis.push(knot_vectors[dim].len() - degrees[dim] - 1);
        supports.push(degrees[dim] + 1);
    }
    let strides = compute_tensor_strides(&num_basis)?;
    let total_cols = num_basis.iter().try_fold(1usize, |acc, &v| {
        acc.checked_mul(v)
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;

    let mut basis_matrix = Array2::zeros((nrows, total_cols));
    // Degree-adaptive threshold for tensor-product rows.
    // Higher total degree means more per-row work, so parallelism wins earlier.
    let total_degree: usize = degrees.iter().sum();
    let par_threshold = if total_degree <= dims {
        512
    } else if total_degree <= 3 * dims {
        128
    } else {
        64
    };

    if nrows >= par_threshold {
        basis_matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || {
                    (
                        degrees
                            .iter()
                            .map(|&degree| BasisEvalScratch::new(degree))
                            .collect::<Vec<_>>(),
                        supports.iter().map(|&s| vec![0.0; s]).collect::<Vec<_>>(),
                        vec![0usize; dims],
                        vec![0usize; dims],
                    )
                },
                |(scratch, values, starts, indices), (row_idx, mut row)| {
                    let row_slice = row
                        .as_slice_mut()
                        .expect("basis matrix rows should be contiguous");
                    fill_tensor_row(
                        row_idx,
                        data,
                        knot_vectors,
                        degrees,
                        eval_kinds,
                        &num_basis,
                        &supports,
                        &strides,
                        scratch,
                        values,
                        starts,
                        indices,
                        |_, col, value| row_slice[col] = value,
                    );
                },
            );
    } else {
        let mut scratch: Vec<BasisEvalScratch> = degrees
            .iter()
            .map(|&degree| BasisEvalScratch::new(degree))
            .collect();
        let mut values: Vec<Vec<f64>> = supports.iter().map(|&s| vec![0.0; s]).collect();
        let mut starts = vec![0usize; dims];
        let mut indices = vec![0usize; dims];
        for (row_idx, mut row) in basis_matrix.axis_iter_mut(Axis(0)).enumerate() {
            let row_slice = row
                .as_slice_mut()
                .expect("basis matrix rows should be contiguous");
            fill_tensor_row(
                row_idx,
                data,
                knot_vectors,
                degrees,
                eval_kinds,
                &num_basis,
                &supports,
                &strides,
                &mut scratch,
                &mut values,
                &mut starts,
                &mut indices,
                |_, col, value| row_slice[col] = value,
            );
        }
    }

    Ok(basis_matrix)
}

fn generate_basis_nd_sparse(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    eval_kinds: &[BasisEvalKind],
) -> Result<SparseColMat<usize, f64>, BasisError> {
    let (dims, nrows) = validate_multi_dim_inputs(data, knot_vectors, degrees, eval_kinds)?;
    let mut num_basis = Vec::with_capacity(dims);
    let mut supports = Vec::with_capacity(dims);
    for dim in 0..dims {
        validate_knots_for_degree(knot_vectors[dim], degrees[dim])?;
        num_basis.push(knot_vectors[dim].len() - degrees[dim] - 1);
        supports.push(degrees[dim] + 1);
    }
    let strides = compute_tensor_strides(&num_basis)?;
    let total_cols = num_basis.iter().try_fold(1usize, |acc, &v| {
        acc.checked_mul(v)
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;

    // Match dense tensor threshold policy for consistent medium-size behavior.
    let total_degree: usize = degrees.iter().sum();
    let par_threshold = if total_degree <= dims {
        512
    } else if total_degree <= 3 * dims {
        128
    } else {
        64
    };
    let triplets: Vec<Triplet<usize, usize, f64>> = if nrows >= par_threshold {
        const CHUNK_SIZE: usize = 1024;
        let row_starts: Vec<usize> = (0..nrows).step_by(CHUNK_SIZE).collect();
        row_starts
            .into_par_iter()
            .map_init(
                || {
                    (
                        degrees
                            .iter()
                            .map(|&degree| BasisEvalScratch::new(degree))
                            .collect::<Vec<_>>(),
                        supports.iter().map(|&s| vec![0.0; s]).collect::<Vec<_>>(),
                        vec![0usize; dims],
                        vec![0usize; dims],
                    )
                },
                |(scratch, values, starts, indices), chunk_start| {
                    let row_end = (chunk_start + CHUNK_SIZE).min(nrows);
                    let per_row_nnz = supports.iter().fold(1usize, |acc, &s| acc * s);
                    let mut local = Vec::with_capacity(
                        row_end
                            .saturating_sub(chunk_start)
                            .saturating_mul(per_row_nnz),
                    );
                    for row_idx in chunk_start..row_end {
                        fill_tensor_row(
                            row_idx,
                            data,
                            knot_vectors,
                            degrees,
                            eval_kinds,
                            &num_basis,
                            &supports,
                            &strides,
                            scratch,
                            values,
                            starts,
                            indices,
                            |_, col, value| local.push(Triplet::new(row_idx, col, value)),
                        );
                    }
                    local
                },
            )
            .reduce(Vec::new, |mut acc, mut chunk| {
                acc.append(&mut chunk);
                acc
            })
    } else {
        let mut scratch: Vec<BasisEvalScratch> = degrees
            .iter()
            .map(|&degree| BasisEvalScratch::new(degree))
            .collect();
        let mut values: Vec<Vec<f64>> = supports.iter().map(|&s| vec![0.0; s]).collect();
        let mut starts = vec![0usize; dims];
        let mut indices = vec![0usize; dims];
        let mut triplets = Vec::with_capacity(
            nrows.saturating_mul(supports.iter().fold(1usize, |acc, &s| acc * s)),
        );
        for row_idx in 0..nrows {
            fill_tensor_row(
                row_idx,
                data,
                knot_vectors,
                degrees,
                eval_kinds,
                &num_basis,
                &supports,
                &strides,
                &mut scratch,
                &mut values,
                &mut starts,
                &mut indices,
                |_, col, value| triplets.push(Triplet::new(row_idx, col, value)),
            );
        }
        triplets
    };

    SparseColMat::try_new_from_triplets(nrows, total_cols, &triplets)
        .map_err(|err| BasisError::SparseCreation(format!("{err:?}")))
}

/// Returns true if the B-spline basis should be built in sparse form based on density.
pub fn should_use_sparse_basis(num_basis_cols: usize, degree: usize, dim: usize) -> bool {
    if num_basis_cols == 0 {
        return false;
    }

    let support_per_row = (degree + 1).saturating_pow(dim as u32) as f64;
    let density = support_per_row / num_basis_cols as f64;

    density < 0.20 && num_basis_cols > 32
}

fn eval_kinds_from_orders(orders: &[usize]) -> Result<Vec<BasisEvalKind>, BasisError> {
    orders
        .iter()
        .map(|&order| match order {
            0 => Ok(BasisEvalKind::Basis),
            1 => Ok(BasisEvalKind::FirstDerivative),
            2 => Ok(BasisEvalKind::SecondDerivative),
            _ => Err(BasisError::InvalidInput(format!(
                "unsupported derivative order {order}"
            ))),
        })
        .collect()
}

fn create_bspline_basis_nd_with_knots_internal(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    eval_kinds: &[BasisEvalKind],
) -> Result<(Arc<Array2<f64>>, Vec<Array1<f64>>), BasisError> {
    validate_multi_dim_inputs(data, knot_vectors, degrees, eval_kinds)?;
    let knot_vecs: Vec<Array1<f64>> = knot_vectors.iter().map(|v| v.to_owned()).collect();
    let knot_views: Vec<ArrayView1<'_, f64>> = knot_vecs.iter().map(|v| v.view()).collect();

    let basis_matrix = generate_basis_nd_dense(data, &knot_views, degrees, eval_kinds)?;
    Ok((Arc::new(basis_matrix), knot_vecs))
}

/// Creates a multi-dimensional tensor-product B-spline basis using pre-computed knot vectors.
pub fn create_bspline_basis_nd_with_knots(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
) -> Result<(Arc<Array2<f64>>, Vec<Array1<f64>>), BasisError> {
    let eval_kinds = vec![BasisEvalKind::Basis; degrees.len()];
    create_bspline_basis_nd_with_knots_internal(data, knot_vectors, degrees, &eval_kinds)
}

/// Creates a multi-dimensional tensor-product B-spline basis using pre-computed knot vectors,
/// allowing per-dimension derivative orders (0, 1, 2).
pub fn create_bspline_basis_nd_with_knots_derivative(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    derivative_orders: &[usize],
) -> Result<(Arc<Array2<f64>>, Vec<Array1<f64>>), BasisError> {
    let eval_kinds = eval_kinds_from_orders(derivative_orders)?;
    create_bspline_basis_nd_with_knots_internal(data, knot_vectors, degrees, &eval_kinds)
}

/// Creates a sparse multi-dimensional tensor-product B-spline basis using pre-computed knot vectors.
pub fn create_bspline_basis_nd_sparse_with_knots(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
) -> Result<(SparseColMat<usize, f64>, Vec<Array1<f64>>), BasisError> {
    let eval_kinds = vec![BasisEvalKind::Basis; degrees.len()];
    let knot_vecs: Vec<Array1<f64>> = knot_vectors.iter().map(|v| v.to_owned()).collect();
    let knot_views: Vec<ArrayView1<'_, f64>> = knot_vecs.iter().map(|v| v.view()).collect();
    let sparse = generate_basis_nd_sparse(data, &knot_views, degrees, &eval_kinds)?;
    Ok((sparse, knot_vecs))
}

/// Creates a sparse multi-dimensional tensor-product B-spline basis using pre-computed knot vectors,
/// allowing per-dimension derivative orders (0, 1, 2).
pub fn create_bspline_basis_nd_sparse_with_knots_derivative(
    data: &[ArrayView1<'_, f64>],
    knot_vectors: &[ArrayView1<'_, f64>],
    degrees: &[usize],
    derivative_orders: &[usize],
) -> Result<(SparseColMat<usize, f64>, Vec<Array1<f64>>), BasisError> {
    let eval_kinds = eval_kinds_from_orders(derivative_orders)?;
    let knot_vecs: Vec<Array1<f64>> = knot_vectors.iter().map(|v| v.to_owned()).collect();
    let knot_views: Vec<ArrayView1<'_, f64>> = knot_vecs.iter().map(|v| v.view()).collect();
    let sparse = generate_basis_nd_sparse(data, &knot_views, degrees, &eval_kinds)?;
    Ok((sparse, knot_vecs))
}

/// Creates a penalty matrix `S` for a B-spline basis from a difference matrix `D`.
/// The penalty is of the form `S = D' * D`, penalizing the squared `order`-th
/// differences of the spline coefficients. This is the core of P-splines.
///
/// This function supports both uniform knots (using ordinary differences) and
/// non-uniform knots (using divided differences), which is critical for
/// correctly penalizing curvature when knots are irregularly spaced (e.g. quantiles).
///
/// # Arguments
/// * `num_basis_functions`: The number of basis functions (i.e., columns in the basis matrix).
/// * `order`: The order of the difference penalty (e.g., 2 for second differences).
/// * `greville_abscissae`: Optional Greville abscissae for divided differences.
///   If `None`, assumes uniform knots and uses ordinary integer differences.
///   If `Some`, uses divided differences scaled by the inverse of the knot spans.
///
/// # Returns
/// A square `Array2<f64>` of shape `[num_basis, num_basis]` representing the penalty `S`.
pub fn create_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
    greville_abscissae: Option<ArrayView1<f64>>,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }

    if let Some(g) = greville_abscissae {
        if g.len() != num_basis_functions {
            return Err(BasisError::DimensionMismatch(format!(
                "Greville abscissae length {} does not match num_basis_functions {}",
                g.len(),
                num_basis_functions
            )));
        }
    }

    // Start with the identity matrix
    let mut d = Array2::<f64>::eye(num_basis_functions);

    // Apply the differencing operation `order` times.
    // Each `diff` reduces the number of rows by 1.
    for o in 1..=order {
        // Calculate the difference between adjacent rows: D^{(o)} = Delta * D^{(o-1)}
        d = &d.slice(s![1.., ..]) - &d.slice(s![..-1, ..]);

        // If using non-uniform knots, apply divided difference scaling:
        // D^{(o)}_i = D^{(o)}_i / (xi_{i+o} - xi_i)
        if let Some(g) = greville_abscissae {
            let nrows = d.nrows();
            for i in 0..nrows {
                let span = g[i + o] - g[i];
                if span.abs() > 1e-12 {
                    let mut row = d.row_mut(i);
                    row /= span;
                }
            }
        }
    }

    // The penalty matrix S = D' * D
    let s = fast_ata(&d);
    Ok(s)
}

fn is_effectively_uniform_knot_geometry(knot_vector: &Array1<f64>, degree: usize) -> bool {
    if knot_vector.len() <= degree + 2 {
        return true;
    }

    let min_k = knot_vector[0];
    let max_k = knot_vector[knot_vector.len() - 1];
    let scale = (max_k - min_k).abs().max(1.0);
    let tol = 1e-10 * scale;

    // Any repeated interior knot (beyond clamped boundaries) implies irregular geometry.
    let mut left = 0usize;
    while left + 1 < knot_vector.len() && (knot_vector[left + 1] - min_k).abs() <= tol {
        left += 1;
    }
    let mut right = knot_vector.len() - 1;
    while right > 0 && (knot_vector[right - 1] - max_k).abs() <= tol {
        right -= 1;
    }
    if right > left + 1 {
        for i in (left + 1)..=right {
            if (knot_vector[i] - knot_vector[i - 1]).abs() <= tol {
                return false;
            }
        }
    }

    let mut breakpoints = Vec::<f64>::with_capacity(knot_vector.len());
    for &k in knot_vector {
        if breakpoints
            .last()
            .map(|last| (k - *last).abs() > tol)
            .unwrap_or(true)
        {
            breakpoints.push(k);
        }
    }

    if breakpoints.len() <= 2 {
        return true;
    }

    let h0 = breakpoints[1] - breakpoints[0];
    for i in 2..breakpoints.len() {
        let hi = breakpoints[i] - breakpoints[i - 1];
        if (hi - h0).abs() > 1e-8 * scale {
            return false;
        }
    }
    true
}

/// Selects Greville abscissae for difference-penalty scaling when knot geometry is non-uniform.
///
/// For regular, uniformly spaced breakpoint grids this returns `None` to preserve
/// classical P-spline integer-difference penalties. For irregular grids (including
/// repeated interior knots), this returns `Some(Greville)` so divided-difference
/// scaling is applied by [`create_difference_penalty_matrix`].
pub fn penalty_greville_abscissae_for_knots(
    knot_vector: &Array1<f64>,
    degree: usize,
) -> Result<Option<Array1<f64>>, BasisError> {
    if is_effectively_uniform_knot_geometry(knot_vector, degree) {
        Ok(None)
    } else {
        Ok(Some(compute_greville_abscissae(knot_vector, degree)?))
    }
}

/// Thin-plate regression spline basis and penalty (order m=2).
///
/// The returned basis has columns `[K_c | P]` where:
/// - `K_c` is the constrained radial basis block (`K * Z`) with
///   `P(knots)^T * α = 0` enforced via nullspace projection
/// - `P` is the polynomial null-space block `[1, x_1, ..., x_d]`
///
/// The returned penalty matrix is block-diagonal with:
/// - upper-left `Omega_c = Z^T Omega Z` for the constrained radial block
/// - zero lower-right block for unpenalized polynomial terms.
///
/// For double-penalty GAMs, a second ridge penalty `I` is also returned so the
/// caller can optimize `(lambda_bending, lambda_ridge)` jointly.
#[derive(Debug, Clone)]
pub struct ThinPlateSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_bending: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
}

impl ThinPlateSplineBasis {
    /// Returns the two standard TPS penalties for double-penalty REML:
    /// `[S_bending, I_ridge]`.
    pub fn penalty_matrices(&self) -> Vec<Array2<f64>> {
        vec![self.penalty_bending.clone(), self.penalty_ridge.clone()]
    }
}

/// Matérn smoothness parameter `nu` (half-integer variants with closed forms).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaternNu {
    Half,
    ThreeHalves,
    FiveHalves,
    SevenHalves,
    NineHalves,
}

impl MaternNu {
    /// Recommend the lowest-order half-integer Matérn that stays compatible with
    /// the canonical mass/tension/stiffness collocation penalties.
    pub fn recommended_for_dimension(dimension: usize) -> Self {
        if dimension <= 1 {
            Self::Half
        } else {
            Self::ThreeHalves
        }
    }
}

/// Matérn radial basis and penalties.
#[derive(Debug, Clone)]
pub struct MaternSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_kernel: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
}

/// Duchon-like radial basis and penalties with explicit low-frequency null-space order.
#[derive(Debug, Clone)]
pub struct DuchonSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_kernel: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
    pub nullspace_order: DuchonNullspaceOrder,
}

impl DuchonSplineBasis {
    pub fn penalty_matrices(&self) -> Vec<Array2<f64>> {
        vec![self.penalty_kernel.clone(), self.penalty_ridge.clone()]
    }
}

/// Which knot strategy to use for 1D B-spline bases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BSplineKnotSpec {
    Generate {
        data_range: (f64, f64),
        num_internal_knots: usize,
    },
    Automatic {
        num_internal_knots: Option<usize>,
        placement: BSplineKnotPlacement,
    },
    Provided(Array1<f64>),
}

/// Internal-knot placement strategy when knots are automatically inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BSplineKnotPlacement {
    Uniform,
    Quantile,
}

/// 1D B-spline basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSplineBasisSpec {
    pub degree: usize,
    pub penalty_order: usize,
    pub knot_spec: BSplineKnotSpec,
    pub double_penalty: bool,
    pub identifiability: BSplineIdentifiability,
}

/// Per-smooth identifiability policy for 1D B-spline bases.
///
/// These constraints are applied directly in the builder via a reparameterization
/// `B_constrained = B * Z`, and every penalty matrix is projected as
/// `S_constrained = Z' S Z`, so solver geometry stays consistent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BSplineIdentifiability {
    /// Keep unconstrained basis columns.
    None,
    /// Enforce weighted sum-to-zero: `B' w = 0` (or unweighted when `weights=None`).
    WeightedSumToZero { weights: Option<Array1<f64>> },
    /// Remove intercept + linear trend in coefficient space using Greville geometry.
    RemoveLinearTrend,
    /// Enforce orthogonality to supplied design columns `C` (n x q):
    /// `B_c' W C = 0` (or unweighted when `weights=None`).
    ///
    /// To enforce `[intercept, x, ...]`, provide `columns` with those columns.
    OrthogonalToDesignColumns {
        columns: Array2<f64>,
        weights: Option<Array1<f64>>,
    },
    /// Apply an explicit coefficient-space transform `Z` learned at fit time.
    ///
    /// This freezes identifiability behavior so prediction cannot drift based on
    /// new-data distribution. The constrained basis is `B * Z`.
    FrozenTransform { transform: Array2<f64> },
}

impl Default for BSplineIdentifiability {
    fn default() -> Self {
        // Smooth terms should be centered by default to avoid intercept confounding.
        Self::WeightedSumToZero { weights: None }
    }
}

/// Thin-plate center selection strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CenterStrategy {
    UserProvided(Array2<f64>),
    EqualMass { num_centers: usize },
    FarthestPoint { num_centers: usize },
    KMeans { num_centers: usize, max_iter: usize },
    UniformGrid { points_per_dim: usize },
}

/// Thin-plate basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinPlateBasisSpec {
    pub center_strategy: CenterStrategy,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
}

/// Per-smooth identifiability policy for spatial (TPS / Duchon) bases.
///
/// For a raw local basis `B` and parametric design block `C`, the orthogonalized
/// basis is `B_c = B Z` where columns of `Z` span `null((B^T C)^T)`. This enforces:
///   `B_c^T C = 0`
/// in the unweighted inner product, so spatial effects cannot absorb parametric
/// intercept/linear directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialIdentifiability {
    /// Keep unconstrained basis columns.
    None,
    /// Orthogonalize the smooth against `[intercept | explicit linear terms]`.
    OrthogonalToParametric,
    /// Freeze a fit-time transform `Z`; prediction uses `B_new * Z` unchanged.
    FrozenTransform { transform: Array2<f64> },
}

impl Default for SpatialIdentifiability {
    fn default() -> Self {
        // "Magic" default for modular GAMs with explicit parametric block:
        // keep spatial smooth orthogonal to intercept/linear terms.
        Self::OrthogonalToParametric
    }
}

/// Matérn basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaternBasisSpec {
    pub center_strategy: CenterStrategy,
    pub length_scale: f64,
    pub nu: MaternNu,
    #[serde(default)]
    pub include_intercept: bool,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: MaternIdentifiability,
}

/// Per-smooth identifiability policy for Matérn kernel coefficients.
///
/// These constraints are geometric (center-based), so they are stable across
/// train/predict and do not depend on response weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaternIdentifiability {
    /// Keep the unconstrained kernel coefficient space.
    None,
    /// Enforce `1^T alpha = 0` at center locations (removes constant drift).
    CenterSumToZero,
    /// Enforce orthogonality to `[1, c_1, ..., c_d]` at centers.
    /// Use this when explicit linear terms should own global trends.
    CenterLinearOrthogonal,
    /// Freeze a fit-time transform `Z` so prediction cannot drift.
    FrozenTransform { transform: Array2<f64> },
}

impl Default for MaternIdentifiability {
    fn default() -> Self {
        // Safe default with model intercepts: prevent kernel block from absorbing
        // a global mean level.
        Self::CenterSumToZero
    }
}

/// Duchon null-space order. `0` removes the explicit polynomial block,
/// `1` keeps `[1, x_1, ..., x_d]` unpenalized by the primary curvature penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuchonNullspaceOrder {
    Zero,
    Linear,
}

/// Duchon-like basis configuration with explicit low-frequency null-space
/// control and explicit spectral power.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuchonBasisSpec {
    pub center_strategy: CenterStrategy,
    /// Optional hybrid Matérn width. `None` means pure scale-free Duchon with
    /// spectrum `||w||^(2p + 2s)`. `Some(length_scale)` enables the hybrid
    /// spectrum `||w||^(2p) * (kappa^2 + ||w||^2)^s`, `kappa = 1/length_scale`.
    pub length_scale: Option<f64>,
    /// Integer spectral power `s`.
    pub power: usize,
    pub nullspace_order: DuchonNullspaceOrder,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
}

/// Metadata returned by generic basis builders.
#[derive(Debug, Clone)]
pub enum BasisMetadata {
    BSpline1D {
        knots: Array1<f64>,
        identifiability_transform: Option<Array2<f64>>,
    },
    ThinPlate {
        centers: Array2<f64>,
        identifiability_transform: Option<Array2<f64>>,
    },
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        nu: MaternNu,
        include_intercept: bool,
        identifiability_transform: Option<Array2<f64>>,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        power: usize,
        nullspace_order: DuchonNullspaceOrder,
        identifiability_transform: Option<Array2<f64>>,
    },
    TensorBSpline {
        feature_cols: Vec<usize>,
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        identifiability_transform: Option<Array2<f64>>,
    },
}

/// Standardized basis build result for engine-level composition.
#[derive(Debug, Clone)]
pub struct BasisBuildResult {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penalty_info: Vec<PenaltyInfo>,
    pub metadata: BasisMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltySource {
    Primary,
    DoublePenaltyNullspace,
    OperatorMass,
    OperatorTension,
    OperatorStiffness,
    TensorMarginal { dim: usize },
    TensorGlobalRidge,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltyDropReason {
    ZeroMatrix,
    NumericalRankZero,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyInfo {
    pub source: PenaltySource,
    pub original_index: usize,
    pub active: bool,
    pub effective_rank: usize,
    pub dropped_reason: Option<PenaltyDropReason>,
    pub nullspace_dim_hint: usize,
    #[serde(default = "default_normalization_scale")]
    pub normalization_scale: f64,
}

#[derive(Debug, Clone)]
pub struct PenaltyCandidate {
    pub matrix: Array2<f64>,
    pub nullspace_dim_hint: usize,
    pub source: PenaltySource,
    pub normalization_scale: f64,
}

#[derive(Debug, Clone)]
pub struct CanonicalPenaltyBlock {
    pub sym_penalty: Array2<f64>,
    pub rank: usize,
    pub nullity: usize,
    pub tol: f64,
    pub is_zero: bool,
}

#[derive(Debug, Clone)]
pub struct BasisPsiDerivativeResult {
    pub design_derivative: Array2<f64>,
    pub penalties_derivative: Vec<Array2<f64>>,
}

#[derive(Debug, Clone)]
pub struct BasisPsiSecondDerivativeResult {
    pub design_second_derivative: Array2<f64>,
    pub penalties_second_derivative: Vec<Array2<f64>>,
}

#[derive(Debug, Clone)]
pub struct CollocationOperatorMatrices {
    pub d0: Array2<f64>,
    pub d1: Array2<f64>,
    pub d2: Array2<f64>,
    pub collocation_points: Array2<f64>,
}

fn default_normalization_scale() -> f64 {
    1.0
}

fn validate_center_count(num_centers: usize) -> Result<(), BasisError> {
    if num_centers == 0 {
        return Err(BasisError::InvalidInput(
            "center count must be positive".to_string(),
        ));
    }
    Ok(())
}

fn select_equal_mass_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        return Err(BasisError::InvalidInput(format!(
            "equal-mass center selection requested {num_centers} centers but data has {n} rows"
        )));
    }
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "equal-mass center selection requires at least one column".to_string(),
        ));
    }
    #[derive(Clone)]
    struct Leaf {
        indices: Vec<usize>,
    }

    // Recursive equal-mass partition that always splits the leaf along its widest
    // coordinate dimension. This addresses the root cause of PC1-only slicing by
    // adapting splits to the local geometry of each partition.
    let mut leaves = vec![Leaf {
        indices: (0..n).collect(),
    }];

    let choose_split_dim = |idxs: &[usize]| -> usize {
        let mut best_dim = 0usize;
        let mut best_span = f64::NEG_INFINITY;
        for j in 0..d {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &idx in idxs {
                let v = data[[idx, j]];
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let span = max_v - min_v;
            if span > best_span {
                best_span = span;
                best_dim = j;
            }
        }
        best_dim
    };

    while leaves.len() < num_centers {
        let mut split_pos = None;
        let mut split_size = 0usize;
        for (i, leaf) in leaves.iter().enumerate() {
            if leaf.indices.len() > split_size && leaf.indices.len() > 1 {
                split_size = leaf.indices.len();
                split_pos = Some(i);
            }
        }
        let Some(pos) = split_pos else {
            break;
        };

        let leaf = leaves.swap_remove(pos);
        let split_dim = choose_split_dim(&leaf.indices);
        let mut sorted = leaf.indices;
        sorted.sort_by(|&a, &b| {
            let ord = data[[a, split_dim]].total_cmp(&data[[b, split_dim]]);
            if ord.is_eq() { a.cmp(&b) } else { ord }
        });
        let mid = sorted.len() / 2;
        let left = sorted[..mid].to_vec();
        let right = sorted[mid..].to_vec();

        if left.is_empty() || right.is_empty() {
            leaves.push(Leaf { indices: sorted });
            break;
        }

        leaves.push(Leaf { indices: left });
        leaves.push(Leaf { indices: right });
    }

    if leaves.len() < num_centers {
        return Err(BasisError::InvalidInput(format!(
            "equal-mass partition produced {} leaves, expected {num_centers}",
            leaves.len()
        )));
    }

    let mut centers = Array2::<f64>::zeros((num_centers, d));
    for (c, leaf) in leaves.iter().take(num_centers).enumerate() {
        let m = leaf.indices.len() as f64;
        let mut centroid = vec![0.0_f64; d];
        for &idx in &leaf.indices {
            for j in 0..d {
                centroid[j] += data[[idx, j]];
            }
        }
        for v in &mut centroid {
            *v /= m.max(1.0);
        }

        let mut best_idx = leaf.indices[0];
        let mut best_d2 = f64::INFINITY;
        for &idx in &leaf.indices {
            let mut d2 = 0.0;
            for j in 0..d {
                let delta = data[[idx, j]] - centroid[j];
                d2 += delta * delta;
            }
            if d2 < best_d2 || (d2 == best_d2 && idx < best_idx) {
                best_d2 = d2;
                best_idx = idx;
            }
        }
        centers.row_mut(c).assign(&data.row(best_idx));
    }
    Ok(centers)
}

fn select_kmeans_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
    max_iter: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        return Err(BasisError::InvalidInput(format!(
            "kmeans requested {num_centers} centers but data has {n} rows"
        )));
    }
    let mut centers = select_thin_plate_knots(data, num_centers)?;
    let mut assign = vec![0usize; n];
    let iters = max_iter.max(1);

    for _ in 0..iters {
        // Assignment
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d2 = f64::INFINITY;
            for k in 0..num_centers {
                let mut d2 = 0.0;
                for c in 0..d {
                    let delta = data[[i, c]] - centers[[k, c]];
                    d2 += delta * delta;
                }
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = k;
                }
            }
            assign[i] = best;
        }
        // Update
        let mut sums = Array2::<f64>::zeros((num_centers, d));
        let mut counts = vec![0usize; num_centers];
        for i in 0..n {
            let k = assign[i];
            counts[k] += 1;
            for c in 0..d {
                sums[[k, c]] += data[[i, c]];
            }
        }
        for k in 0..num_centers {
            if counts[k] == 0 {
                continue;
            }
            let inv = 1.0 / counts[k] as f64;
            for c in 0..d {
                centers[[k, c]] = sums[[k, c]] * inv;
            }
        }
    }
    Ok(centers)
}

fn cartesian_grid_axes(axes: &[Array1<f64>]) -> Result<Array2<f64>, BasisError> {
    if axes.is_empty() {
        return Err(BasisError::InvalidInput(
            "uniform grid requires at least one axis".to_string(),
        ));
    }
    let d = axes.len();
    let total = axes.iter().try_fold(1usize, |acc, axis| {
        acc.checked_mul(axis.len())
            .ok_or_else(|| BasisError::DimensionMismatch("uniform grid is too large".to_string()))
    })?;
    let mut out = Array2::<f64>::zeros((total, d));
    for r in 0..total {
        let mut q = r;
        for c in (0..d).rev() {
            let len = axes[c].len();
            let idx = q % len;
            q /= len;
            out[[r, c]] = axes[c][idx];
        }
    }
    Ok(out)
}

fn select_uniform_grid_centers(
    data: ArrayView2<'_, f64>,
    points_per_dim: usize,
) -> Result<Array2<f64>, BasisError> {
    if points_per_dim == 0 {
        return Err(BasisError::InvalidInput(
            "uniform-grid points_per_dim must be positive".to_string(),
        ));
    }
    let d = data.ncols();
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "uniform-grid center selection requires at least one column".to_string(),
        ));
    }
    let mut axes = Vec::with_capacity(d);
    for c in 0..d {
        let col = data.column(c);
        let min_v = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_v = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        axes.push(Array::linspace(min_v, max_v, points_per_dim));
    }
    cartesian_grid_axes(&axes)
}

fn select_centers_by_strategy(
    data: ArrayView2<'_, f64>,
    strategy: &CenterStrategy,
) -> Result<Array2<f64>, BasisError> {
    match strategy {
        CenterStrategy::UserProvided(centers) => {
            if centers.ncols() != data.ncols() {
                return Err(BasisError::DimensionMismatch(format!(
                    "user centers have {} columns but data has {}",
                    centers.ncols(),
                    data.ncols()
                )));
            }
            if centers.nrows() == 0 {
                return Err(BasisError::InvalidInput(
                    "user-provided center list cannot be empty".to_string(),
                ));
            }
            Ok(centers.clone())
        }
        CenterStrategy::EqualMass { num_centers } => select_equal_mass_centers(data, *num_centers),
        CenterStrategy::FarthestPoint { num_centers } => {
            select_thin_plate_knots(data, *num_centers)
        }
        CenterStrategy::KMeans {
            num_centers,
            max_iter,
        } => select_kmeans_centers(data, *num_centers, *max_iter),
        CenterStrategy::UniformGrid { points_per_dim } => {
            select_uniform_grid_centers(data, *points_per_dim)
        }
    }
}

/// Generic 1D B-spline builder returning design + penalty list.
pub fn build_bspline_basis_1d(
    data: ArrayView1<'_, f64>,
    spec: &BSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let (basis, knots) = match &spec.knot_spec {
        BSplineKnotSpec::Generate {
            data_range,
            num_internal_knots,
        } => create_basis::<Dense>(
            data,
            KnotSource::Generate {
                data_range: *data_range,
                num_internal_knots: *num_internal_knots,
            },
            spec.degree,
            BasisOptions::value(),
        )?,
        BSplineKnotSpec::Provided(knots) => create_basis::<Dense>(
            data,
            KnotSource::Provided(knots.view()),
            spec.degree,
            BasisOptions::value(),
        )?,
        BSplineKnotSpec::Automatic {
            num_internal_knots,
            placement,
        } => {
            let inferred = num_internal_knots
                .unwrap_or_else(|| default_internal_knot_count_for_data(data.len(), spec.degree));
            let knots = match placement {
                BSplineKnotPlacement::Uniform => {
                    let range = finite_data_range(data)?;
                    internal::generate_full_knot_vector(range, inferred, spec.degree)?
                }
                BSplineKnotPlacement::Quantile => {
                    internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                }
            };
            create_basis::<Dense>(
                data,
                KnotSource::Provided(knots.view()),
                spec.degree,
                BasisOptions::value(),
            )?
        }
    };
    let design_raw = (*basis).clone();
    let p_raw = design_raw.ncols();
    let greville_for_penalty = penalty_greville_abscissae_for_knots(&knots, spec.degree)?;
    let s_bend_raw = create_difference_penalty_matrix(
        p_raw,
        spec.penalty_order,
        greville_for_penalty.as_ref().map(|g| g.view()),
    )?;
    let mut penalties_raw = vec![PenaltyCandidate {
        matrix: s_bend_raw.clone(),
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
    }];
    if spec.double_penalty {
        penalties_raw.push(PenaltyCandidate {
            matrix: build_nullspace_shrinkage_penalty(&s_bend_raw)?
                .map(|shrink| shrink.sym_penalty)
                .unwrap_or_else(|| Array2::<f64>::zeros(s_bend_raw.raw_dim())),
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: 1.0,
        });
    }

    let penalties_raw_mats: Vec<Array2<f64>> = penalties_raw
        .iter()
        .map(|candidate| candidate.matrix.clone())
        .collect();
    let (design, penalties, identifiability_transform) = apply_bspline_identifiability_policy(
        design_raw,
        penalties_raw_mats,
        &knots,
        spec.degree,
        &spec.identifiability,
    )?;
    let transformed_candidates = penalties
        .into_iter()
        .zip(penalties_raw.into_iter())
        .map(
            |(matrix, candidate)| -> Result<PenaltyCandidate, BasisError> {
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&matrix)?,
                    matrix,
                    source: candidate.source,
                    normalization_scale: candidate.normalization_scale,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let (penalties, nullspace_dims, penalty_info) =
        filter_active_penalty_candidates(transformed_candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penalty_info,
        metadata: BasisMetadata::BSpline1D {
            knots,
            identifiability_transform,
        },
    })
}

fn apply_bspline_identifiability_policy(
    design: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    knots: &Array1<f64>,
    degree: usize,
    identifiability: &BSplineIdentifiability,
) -> Result<(Array2<f64>, Vec<Array2<f64>>, Option<Array2<f64>>), BasisError> {
    let (design_c, z_opt): (Array2<f64>, Option<Array2<f64>>) = match identifiability {
        BSplineIdentifiability::None => (design, None),
        BSplineIdentifiability::WeightedSumToZero { weights } => {
            let (b_c, z) =
                apply_sum_to_zero_constraint(design.view(), weights.as_ref().map(|w| w.view()))?;
            (b_c, Some(z))
        }
        BSplineIdentifiability::RemoveLinearTrend => {
            let (z, _s_constrained) = compute_geometric_constraint_transform(knots, degree, 2)?;
            (design.dot(&z), Some(z))
        }
        BSplineIdentifiability::OrthogonalToDesignColumns { columns, weights } => {
            let (b_c, z) = apply_weighted_orthogonality_constraint(
                design.view(),
                columns.view(),
                weights.as_ref().map(|w| w.view()),
            )?;
            (b_c, Some(z))
        }
        BSplineIdentifiability::FrozenTransform { transform } => {
            let z = transform.clone();
            if design.ncols() != z.nrows() {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen identifiability transform mismatch: design has {} columns but transform has {} rows",
                    design.ncols(),
                    z.nrows()
                )));
            }
            (design.dot(&z), Some(z))
        }
    };

    let penalties_c = if let Some(ref z) = z_opt {
        penalties
            .into_iter()
            .map(|s| {
                let zt_s = fast_atb(&z, &s);
                fast_ab(&zt_s, &z)
            })
            .collect()
    } else {
        penalties
    };

    Ok((design_c, penalties_c, z_opt))
}

pub(crate) fn estimate_penalty_nullity(penalty: &Array2<f64>) -> Result<usize, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        return Err(BasisError::DimensionMismatch(
            "penalty matrix must be square when estimating nullspace".to_string(),
        ));
    }
    if penalty.nrows() == 0 {
        return Ok(0);
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    Ok(evals.iter().filter(|&&ev| ev.abs() <= tol).count())
}

#[derive(Debug, Clone)]
struct PsdSpectralSummary {
    min_eigenvalue: f64,
    max_abs_eigenvalue: f64,
    tolerance: f64,
    effective_rank: usize,
}

fn symmetrize_penalty(penalty: &Array2<f64>) -> Array2<f64> {
    let mut sym = penalty.clone();
    for i in 0..sym.nrows() {
        for j in 0..i {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    sym
}

fn spectral_tolerance(sym: &Array2<f64>, evals: &Array1<f64>) -> f64 {
    let max_abs_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    (sym.nrows().max(1) as f64) * 1e-10 * max_abs_ev.max(1.0)
}

fn spectral_summary(
    penalty: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), BasisError> {
    let sym = symmetrize_penalty(penalty);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    Ok((sym, evals, evecs))
}

fn validate_psd_penalty(
    penalty: &Array2<f64>,
    context: &str,
    guidance: &str,
) -> Result<PsdSpectralSummary, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        return Err(BasisError::DimensionMismatch(format!(
            "{context}: penalty matrix must be square for PSD validation"
        )));
    }
    if penalty.nrows() == 0 {
        return Ok(PsdSpectralSummary {
            min_eigenvalue: 0.0,
            max_abs_eigenvalue: 0.0,
            tolerance: 1e-10,
            effective_rank: 0,
        });
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tolerance = spectral_tolerance(&sym, &evals);
    let min_eigenvalue = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let effective_rank = evals.iter().filter(|&&ev| ev > tolerance).count();

    if min_eigenvalue < -tolerance {
        return Err(BasisError::IndefinitePenalty {
            context: context.to_string(),
            min_eigenvalue,
            tolerance,
            guidance: guidance.to_string(),
        });
    }

    Ok(PsdSpectralSummary {
        min_eigenvalue,
        max_abs_eigenvalue,
        tolerance,
        effective_rank,
    })
}

pub fn analyze_penalty_block(penalty: &Array2<f64>) -> Result<CanonicalPenaltyBlock, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        return Err(BasisError::DimensionMismatch(
            "penalty matrix must be square when analyzing penalty".to_string(),
        ));
    }
    if penalty.nrows() == 0 {
        return Ok(CanonicalPenaltyBlock {
            sym_penalty: Array2::<f64>::zeros((0, 0)),
            rank: 0,
            nullity: 0,
            tol: 1e-10,
            is_zero: true,
        });
    }

    let (sym, evals, _) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    let rank = evals.iter().filter(|&&ev| ev > tol).count();
    let nullity = sym.nrows().saturating_sub(rank);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    Ok(CanonicalPenaltyBlock {
        sym_penalty: sym,
        rank,
        nullity,
        tol,
        is_zero: max_abs_eigenvalue <= tol,
    })
}

pub fn filter_active_penalty_candidates(
    candidates: Vec<PenaltyCandidate>,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let mut penalties = Vec::with_capacity(candidates.len());
    let mut nullspace_dims = Vec::with_capacity(candidates.len());
    let mut penalty_info = Vec::with_capacity(candidates.len());

    for (original_index, candidate) in candidates.into_iter().enumerate() {
        let analysis = analyze_penalty_block(&candidate.matrix)?;
        let dropped_reason = if analysis.rank == 0 {
            Some(if analysis.is_zero {
                PenaltyDropReason::ZeroMatrix
            } else {
                PenaltyDropReason::NumericalRankZero
            })
        } else {
            None
        };
        let active = dropped_reason.is_none();
        if active {
            log::debug!(
                "Retained penalty block source={:?} original_index={} rank={} nullspace_dim_hint={}",
                candidate.source,
                original_index,
                analysis.rank,
                candidate.nullspace_dim_hint
            );
            penalties.push(analysis.sym_penalty);
            nullspace_dims.push(analysis.nullity);
        } else {
            log::debug!(
                "Dropped inactive penalty block source={:?} original_index={} reason={:?}",
                candidate.source,
                original_index,
                dropped_reason
            );
        }
        penalty_info.push(PenaltyInfo {
            source: candidate.source,
            original_index,
            active,
            effective_rank: analysis.rank,
            dropped_reason,
            nullspace_dim_hint: candidate.nullspace_dim_hint,
            normalization_scale: candidate.normalization_scale,
        });
    }

    Ok((penalties, nullspace_dims, penalty_info))
}

/// Build the double-penalty ridge from the structural null space of a PSD penalty.
fn build_nullspace_shrinkage_penalty(
    penalty: &Array2<f64>,
) -> Result<Option<CanonicalPenaltyBlock>, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        return Err(BasisError::DimensionMismatch(
            "penalty matrix must be square when building nullspace shrinkage penalty".to_string(),
        ));
    }
    if penalty.nrows() == 0 {
        return Ok(None);
    }

    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);

    let zero_idx: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev.abs() <= tol).then_some(i))
        .collect();
    if zero_idx.is_empty() {
        return Ok(None);
    }
    let z = evecs.select(Axis(1), &zero_idx);
    let shrink = fast_ab(&z, &z.t().to_owned());
    Ok(Some(CanonicalPenaltyBlock {
        sym_penalty: shrink,
        rank: zero_idx.len(),
        nullity: 0,
        tol,
        is_zero: false,
    }))
}

fn default_internal_knot_count_for_data(n: usize, degree: usize) -> usize {
    if n < 8 {
        return 0;
    }
    let heuristic = if n < 16 { 3 } else { (n / 4).max(3) };
    let max_reasonable = n.saturating_sub(degree + 2);
    heuristic.min(40).min(max_reasonable)
}

fn finite_data_range(data: ArrayView1<'_, f64>) -> Result<(f64, f64), BasisError> {
    if data.is_empty() {
        return Err(BasisError::InvalidInput(
            "cannot infer knot range from empty data".to_string(),
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "automatic knot placement requires finite data values".to_string(),
        ));
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &x in data {
        if x < min_v {
            min_v = x;
        }
        if x > max_v {
            max_v = x;
        }
    }
    Ok((min_v, max_v))
}

/// Generic thin-plate builder returning design + penalty list.
pub fn build_thin_plate_basis(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_with_workspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let tps = create_thin_plate_spline_basis_with_workspace(data, centers.view(), workspace)?;
    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        tps.basis.view(),
        &spec.identifiability,
        "ThinPlate",
    )?;
    let design = if let Some(z) = identifiability_transform.as_ref() {
        fast_ab(&tps.basis, z)
    } else {
        tps.basis.clone()
    };
    let mut candidates = vec![PenaltyCandidate {
        matrix: tps.penalty_bending.clone(),
        nullspace_dim_hint: tps.num_polynomial_basis,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
    }];
    if spec.double_penalty {
        candidates.push(PenaltyCandidate {
            matrix: tps.penalty_ridge.clone(),
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: 1.0,
        });
    }
    if let Some(z) = identifiability_transform.as_ref() {
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = z.t().dot(&candidate.matrix);
                let matrix = zt_s.dot(z);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&matrix)?,
                    matrix,
                    source: candidate.source,
                    normalization_scale: candidate.normalization_scale,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    let (penalties, nullspace_dims, penalty_info) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penalty_info,
        metadata: BasisMetadata::ThinPlate {
            centers,
            identifiability_transform,
        },
    })
}

#[inline(always)]
fn matern_kernel_from_distance(r: f64, length_scale: f64, nu: MaternNu) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    // Parameterization used here:
    //   x = r / length_scale
    //   a = sqrt(2ν) * x
    // and the half-integer Matérn closed forms are in terms of `a`:
    //   ν=1/2: exp(-a)
    //   ν=3/2: (1+a) exp(-a)
    //   ν=5/2: (1+a+a^2/3) exp(-a)
    // (for ν=1/2, a=x since sqrt(2ν)=1).
    let x = r / length_scale;
    let k = match nu {
        MaternNu::Half => (-x).exp(),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            (1.0 + a) * (-a).exp()
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            (1.0 + a + (a * a) / 3.0) * (-a).exp()
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            (1.0 + a + (2.0 / 5.0) * a2 + (1.0 / 15.0) * a3) * (-a).exp()
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            let a4 = a2 * a2;
            (1.0 + a + (3.0 / 7.0) * a2 + (2.0 / 21.0) * a3 + (1.0 / 105.0) * a4) * (-a).exp()
        }
    };
    Ok(k)
}

#[inline(always)]
fn matern_kernel_log_kappa_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    let x = r / length_scale;
    let deriv = match nu {
        MaternNu::Half => -x * (-x).exp(),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            -(a * a) * (-a).exp()
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            -((a * a) * (1.0 + a) / 3.0) * (-a).exp()
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            let a4 = a2 * a2;
            -((a2 / 5.0) + (a3 / 5.0) + (a4 / 15.0)) * (-a).exp()
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            let a4 = a2 * a2;
            let a5 = a4 * a;
            -((a2 / 7.0) + (a3 / 7.0) + (2.0 * a4 / 35.0) + (a5 / 105.0)) * (-a).exp()
        }
    };
    Ok(deriv)
}

#[inline(always)]
fn matern_kernel_log_kappa_second_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    let x = r / length_scale;
    let second = match nu {
        MaternNu::Half => x * (x - 1.0) * (-x).exp(),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            (a * a * (a - 2.0)) * (-a).exp()
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            (a * a * (a * a - 2.0 * a - 2.0) / 3.0) * (-a).exp()
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            (a2 * (a3 - a2 - 6.0 * a - 6.0) / 15.0) * (-a).exp()
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            let a2 = a * a;
            let a3 = a2 * a;
            let a4 = a2 * a2;
            (a2 * (a4 + a3 - 18.0 * a2 - 30.0 * a - 30.0) / 105.0) * (-a).exp()
        }
    };
    Ok(second)
}

#[cfg(test)]
#[inline(always)]
fn matern_poly_terms(nu: MaternNu, a: f64) -> (f64, f64, f64) {
    match nu {
        MaternNu::Half => (1.0, 0.0, 0.0),
        MaternNu::ThreeHalves => (1.0 + a, 1.0, 0.0),
        MaternNu::FiveHalves => (1.0 + a + (a * a) / 3.0, 1.0 + (2.0 / 3.0) * a, 2.0 / 3.0),
        MaternNu::SevenHalves => {
            let a2 = a * a;
            (
                1.0 + a + (2.0 / 5.0) * a2 + (1.0 / 15.0) * a2 * a,
                1.0 + (4.0 / 5.0) * a + (1.0 / 5.0) * a2,
                (4.0 / 5.0) + (2.0 / 5.0) * a,
            )
        }
        MaternNu::NineHalves => {
            let a2 = a * a;
            let a3 = a2 * a;
            (
                1.0 + a + (3.0 / 7.0) * a2 + (2.0 / 21.0) * a3 + (1.0 / 105.0) * a2 * a2,
                1.0 + (6.0 / 7.0) * a + (2.0 / 7.0) * a2 + (4.0 / 105.0) * a3,
                (6.0 / 7.0) + (4.0 / 7.0) * a + (4.0 / 35.0) * a2,
            )
        }
    }
}

#[cfg(test)]
#[inline(always)]
fn matern_kernel_radial_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64), BasisError> {
    let phi = matern_kernel_from_distance(r, length_scale, nu)?;
    let q = match nu {
        MaternNu::Half => 1.0 / length_scale,
        MaternNu::ThreeHalves => 3.0_f64.sqrt() / length_scale,
        MaternNu::FiveHalves => 5.0_f64.sqrt() / length_scale,
        MaternNu::SevenHalves => 7.0_f64.sqrt() / length_scale,
        MaternNu::NineHalves => 9.0_f64.sqrt() / length_scale,
    };
    let a = q * r;
    let (p, p1, p2) = matern_poly_terms(nu, a);
    let exp_a = (-a).exp();
    let phi_r = q * (p1 - p) * exp_a;
    let phi_rr = q * q * (p2 - 2.0 * p1 + p) * exp_a;
    if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Matérn radial derivatives at r={r}, length_scale={length_scale}, nu={nu:?}"
        )));
    }
    Ok((phi, phi_r, phi_rr))
}

#[inline(always)]
fn matern_kernel_radial_triplet_with_safe_ratio(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    // Full derivation used by collocation operators:
    //   phi(r) = P_nu(a) exp(-a), a=sr, s=sqrt(2nu)/length_scale.
    // For nu>=3/2 we use closed-form phi'(r)/r polynomials with finite r->0 limit.
    // For nu=1/2:
    //   phi'(r)/r = -kappa exp(-kappa r)/r,
    // which is genuinely singular at r=0 and must not be regularized here.
    // Closed forms used below (a = s r, E = exp(-a)):
    // nu=1/2:
    //   phi'    = -s E
    //   phi''   =  s^2 E
    //   phi'/r  diverges as -s/r (regularized via r floor).
    // nu=3/2:
    //   phi'    = -s E a
    //   phi''   =  s^2 E (a-1)
    //   phi'/r  = -s^2 E.
    // nu=5/2:
    //   phi'    = -(s/3) E a(a+1)
    //   phi''   =  (s^2/3) E (a^2-a-1)
    //   phi'/r  = -(s^2/3) E (a+1).
    // nu=7/2:
    //   phi'    = -(s/15) E a(a^2+3a+3)
    //   phi''   =  (s^2/15) E (a^3-3a-3)
    //   phi'/r  = -(s^2/15) E (a^2+3a+3).
    // nu=9/2:
    //   phi'    = -(s/105) E a(a^3+6a^2+15a+15)
    //   phi''   =  (s^2/105) E (a^4+2a^3-3a^2-15a-15)
    //   phi'/r  = -(s^2/105) E (a^3+6a^2+15a+15).
    let (phi, phi_r, phi_rr, phi_r_over_r) = match nu {
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            let phi_r = -s * e;
            let phi_rr = s * s * e;
            // Safe ratio regularization at r=0 to keep operator assembly finite.
            let r_eff = r.max(1e-12);
            let ratio = phi_r / r_eff;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let phi_r = -s * e * a;
            let phi_rr = s * s * e * (a - 1.0);
            let ratio = -s * s * e;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let phi_r = -(s / 3.0) * e * a * (a + 1.0);
            let phi_rr = (s * s / 3.0) * e * (a * a - a - 1.0);
            let ratio = -(s * s / 3.0) * e * (a + 1.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let phi_r = -(s / 15.0) * e * a * (a * a + 3.0 * a + 3.0);
            let phi_rr = (s * s / 15.0) * e * (a * a * a - 3.0 * a - 3.0);
            let ratio = -(s * s / 15.0) * e * (a * a + 3.0 * a + 3.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::NineHalves => {
            let s = 9.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0
                + a
                + (3.0 / 7.0) * a * a
                + (2.0 / 21.0) * a * a * a
                + (1.0 / 105.0) * a * a * a * a)
                * e;
            let phi_r = -(s / 105.0) * e * a * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            let phi_rr = (s * s / 105.0)
                * e
                * (a * a * a * a + 2.0 * a * a * a - 3.0 * a * a - 15.0 * a - 15.0);
            let ratio = -(s * s / 105.0) * e * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            (phi, phi_r, phi_rr, ratio)
        }
    };

    if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() || !phi_r_over_r.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Matérn radial derivatives at r={r}, length_scale={length_scale}, nu={nu:?}"
        )));
    }
    Ok((phi, phi_r, phi_rr, phi_r_over_r))
}

fn duchon_kernel_radial_triplet(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<(f64, f64, f64), BasisError> {
    // Generic Duchon decomposition derivatives
    // ----------------------------------------
    // Partial-fraction representation:
    //   phi(r) = sum_{m>=1} a_m * Phi_m(r) + sum_{n>=1} b_n * M_n(r),
    // where:
    //   Phi_m = polyharmonic block (power/log-power),
    //   M_n   = c_n * r^nu * K_nu(kappa r).
    //
    // We evaluate phi through the canonical kernel function (ensures exact match
    // with basis-value construction), and evaluate phi', phi'' analytically by
    // summing derivatives of each block.
    //
    // Full Duchon derivative strategy
    // -------------------------------
    // Kernel is represented by partial fractions:
    //   phi(r) = sum_m a_m * Phi_m(r) + sum_n b_n * M_n(r),
    // where Phi_m are polyharmonic blocks and M_n are Matérn-Bessel blocks.
    //
    // We evaluate phi itself from the canonical kernel evaluator to ensure
    // consistency with the basis design path, then compute derivatives from the
    // same decomposition analytically:
    //   phi'  = sum_m a_m * Phi_m'  + sum_n b_n * M_n'
    //   phi'' = sum_m a_m * Phi_m'' + sum_n b_n * M_n''.
    //
    let value = duchon_matern_kernel_general_from_distance(
        r,
        length_scale,
        p_order,
        s_order,
        k_dim,
        coeffs,
    )?;
    if !value.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon radial kernel value at r={r}, length_scale={length_scale:?}, p={p_order}, s={s_order}, dim={k_dim}"
        )));
    }

    let Some(length_scale) = length_scale else {
        let block_order = pure_duchon_block_order(p_order, s_order);
        let (_v, mut first, second) = duchon_polyharmonic_block_triplet(r, block_order, k_dim)?;
        if r == 0.0 {
            first = 0.0;
        }
        if !first.is_finite() || !second.is_finite() {
            return Err(BasisError::InvalidInput(format!(
                "non-finite pure Duchon radial derivatives at r={r}, order={block_order}, dim={k_dim}"
            )));
        }
        return Ok((value, first, second));
    };
    let kappa = 1.0 / length_scale.max(1e-300);
    let coeffs_local;
    let coeffs_ref = if let Some(c) = coeffs {
        c
    } else {
        coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
        &coeffs_local
    };
    let r_eval = r.max(DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8));
    let mut first = 0.0;
    let mut second = 0.0;

    for (m, coeff) in coeffs_ref.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (_vm, dm, d2m) = duchon_polyharmonic_block_triplet(r_eval, m, k_dim)?;
        first += coeff * dm;
        second += coeff * d2m;
    }
    for (n, coeff) in coeffs_ref.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (_vn, dn, d2n) = duchon_matern_block_triplet(r_eval, kappa, n, k_dim)?;
        first += coeff * dn;
        second += coeff * d2n;
    }

    if r == 0.0 {
        first = 0.0;
    }
    if !first.is_finite() || !second.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon radial derivatives at r={r}, length_scale={length_scale:?}, p={p_order}, s={s_order}, dim={k_dim}"
        )));
    }
    Ok((value, first, second))
}

fn symmetrize(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
}

fn normalize_penalty(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    let norm = matrix.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    (matrix.mapv(|v| v / norm), norm)
}

fn build_collocation_operators_from_radial<F>(
    collocation: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    collocation_weights: Option<ArrayView1<'_, f64>>,
    radial_triplet: F,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError>
where
    F: Fn(f64) -> Result<(f64, f64, f64), BasisError>,
{
    let p = collocation.nrows();
    let d = collocation.ncols();
    let m = centers.nrows();
    let mut d0 = Array2::<f64>::zeros((p, m));
    let mut d1 = Array2::<f64>::zeros((p * d, m));
    let mut d2 = Array2::<f64>::zeros((p, m));
    // Weighted collocation definition (operator quadrature)
    // ----------------------------------------------------
    // For collocation points z_k with nonnegative weights w_k:
    //   int g(x) dx ~ sum_k w_k g(z_k).
    //
    // Define rows with sqrt(w_k):
    //   D0[k,j]          = sqrt(w_k) * b_j(z_k)
    //   D1[(k-1)d+m, j]  = sqrt(w_k) * ∂_m b_j(z_k)
    //   D2[k,j]          = sqrt(w_k) * Δ b_j(z_k)
    //
    // Then the Gram forms are exactly the collocation energies:
    //   theta^T D0^T D0 theta = sum_k w_k f(z_k)^2
    //   theta^T D1^T D1 theta = sum_k w_k |grad f(z_k)|^2
    //   theta^T D2^T D2 theta = sum_k w_k (Delta f(z_k))^2.
    let row_scales = if let Some(w) = collocation_weights {
        if w.len() != p {
            return Err(BasisError::DimensionMismatch(format!(
                "collocation weight length mismatch: got {}, expected {p}",
                w.len()
            )));
        }
        let mut out = Vec::with_capacity(p);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                return Err(BasisError::InvalidInput(format!(
                    "collocation weights must be finite and non-negative; got {wk}"
                )));
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p]
    };
    const R_EPS: f64 = 1e-10;
    for k in 0..p {
        let scale_k = row_scales[k];
        for j in 0..m {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = collocation[[k, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            let (phi, phi_r, phi_rr) = radial_triplet(r)?;
            if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() {
                return Err(BasisError::InvalidInput(format!(
                    "non-finite collocation operator derivative at collocation row {k}, center {j}, r={r}"
                )));
            }
            d0[[k, j]] = scale_k * phi;
            if r > R_EPS {
                let scale = phi_r / r;
                for c in 0..d {
                    let delta = collocation[[k, c]] - centers[[j, c]];
                    d1[[k * d + c, j]] = scale_k * scale * delta;
                }
                d2[[k, j]] = scale_k * (phi_rr + ((d as f64 - 1.0) * phi_r / r));
            } else {
                // r=0 center-collision limit under C^2 radial regularity with
                // phi'(0)=0: Δphi(0)=d*phi''(0), and ∇phi(0)=0.
                d2[[k, j]] = scale_k * d as f64 * phi_rr;
            }
        }
    }
    Ok((d0, d1, d2))
}

fn operator_penalty_candidates_from_collocation(
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
) -> Vec<PenaltyCandidate> {
    let (s0, c0) = normalize_penalty(&symmetrize(&fast_ata(d0)));
    let (s1, c1) = normalize_penalty(&symmetrize(&fast_ata(d1)));
    let (s2, c2) = normalize_penalty(&symmetrize(&fast_ata(d2)));
    vec![
        PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
        },
        PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
        },
        PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
        },
    ]
}

fn active_operator_penalty_derivatives(
    penalty_info: &[PenaltyInfo],
    operator_derivatives: &[Array2<f64>],
    label: &str,
) -> Result<Vec<Array2<f64>>, BasisError> {
    if operator_derivatives.len() != 3 {
        return Err(BasisError::InvalidInput(format!(
            "{label} operator derivative path requires exactly 3 canonical penalties; found {}",
            operator_derivatives.len()
        )));
    }

    penalty_info
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::OperatorMass => Ok(operator_derivatives[0].clone()),
            PenaltySource::OperatorTension => Ok(operator_derivatives[1].clone()),
            PenaltySource::OperatorStiffness => Ok(operator_derivatives[2].clone()),
            other => Err(BasisError::InvalidInput(format!(
                "unexpected {label} penalty source in canonical operator path: {other:?}"
            ))),
        })
        .collect()
}

fn frozen_spatial_identifiability_transform(
    identifiability: &SpatialIdentifiability,
    expected_rows: usize,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None | SpatialIdentifiability::OrthogonalToParametric => Ok(None),
        SpatialIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != expected_rows {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen {label} identifiability transform mismatch: rows={}, expected {expected_rows}",
                    transform.nrows()
                )));
            }
            Ok(Some(transform.clone()))
        }
    }
}

fn spatial_parametric_constraint_block(data: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut c = Array2::<f64>::ones((n, d + 1));
    if d > 0 {
        c.slice_mut(s![.., 1..]).assign(&data);
    }
    c
}

fn spatial_identifiability_transform_from_design(
    data: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let (_design_constrained, z) =
                apply_weighted_orthogonality_constraint(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}

fn append_intercept_to_transform(transform: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((transform.nrows() + 1, transform.ncols() + 1));
    out.slice_mut(s![0..transform.nrows(), 0..transform.ncols()])
        .assign(transform);
    out[[transform.nrows(), transform.ncols()]] = 1.0;
    out
}

pub fn build_matern_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocation_weights: Option<ArrayView1<'_, f64>>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // Specialized Matérn operator assembly using explicit half-integer formulas:
    // - one exp(-a) and small polynomials per pair,
    // - NaN-safe phi'(r)/r without dividing by r for nu>=3/2,
    // - exact Laplacian identity: Δphi = phi'' + (d-1) phi'/r.
    let p = centers.nrows();
    let d = centers.ncols();
    let row_scales = if let Some(w) = collocation_weights {
        if w.len() != p {
            return Err(BasisError::DimensionMismatch(format!(
                "collocation weight length mismatch: got {}, expected {p}",
                w.len()
            )));
        }
        let mut out = Vec::with_capacity(p);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                return Err(BasisError::InvalidInput(format!(
                    "collocation weights must be finite and non-negative; got {wk}"
                )));
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p]
    };
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p, p));
    const R_EPS: f64 = 1e-12;
    for k in 0..p {
        let scale_k = row_scales[k];
        for j in 0..p {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[k, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            if matches!(nu, MaternNu::Half) && r <= R_EPS && d > 1 {
                return Err(BasisError::InvalidInput(
                    "Matérn nu=1/2 has singular Laplacian at center collisions for d>1; choose nu>=3/2 or avoid collocation at centers".to_string(),
                ));
            }
            let (phi, _phi_r, phi_rr, phi_r_over_r) =
                if matches!(nu, MaternNu::Half) && r <= R_EPS && d == 1 {
                    // In 1D: Delta phi = phi'' and the singular phi'/r term is absent.
                    let s = 1.0 / length_scale;
                    let e = 1.0;
                    (e, -s * e, s * s * e, 0.0)
                } else {
                    matern_kernel_radial_triplet_with_safe_ratio(r, length_scale, nu)?
                };
            d0_raw[[k, j]] = scale_k * phi;
            if r > R_EPS {
                for c in 0..d {
                    let delta = centers[[k, c]] - centers[[j, c]];
                    d1_raw[[k * d + c, j]] = scale_k * phi_r_over_r * delta;
                }
            } else {
                // Symmetry at center-center coincidence.
                for c in 0..d {
                    d1_raw[[k * d + c, j]] = 0.0;
                }
            }
            d2_raw[[k, j]] = scale_k * (phi_rr + ((d as f64 - 1.0) * phi_r_over_r));
            if !d0_raw[[k, j]].is_finite() || !d2_raw[[k, j]].is_finite() {
                return Err(BasisError::InvalidInput(format!(
                    "non-finite Matérn collocation operator entry at row={k}, col={j}, r={r}, nu={nu:?}"
                )));
            }
        }
    }
    let (d0_kernel, d1_kernel, d2_kernel) = if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        (
            fast_ab(&d0_raw, &z),
            fast_ab(&d1_raw, &z),
            fast_ab(&d2_raw, &z),
        )
    } else {
        (d0_raw, d1_raw, d2_raw)
    };
    let p_colloc = centers.nrows();
    let dim = centers.ncols();
    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1 = Array2::<f64>::zeros((p_colloc * dim, total_cols));
    let mut d2 = Array2::<f64>::zeros((p_colloc, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    if include_intercept {
        for (k, &scale_k) in row_scales.iter().enumerate() {
            d0[[k, kernel_cols]] = scale_k;
        }
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: centers.to_owned(),
    })
}

pub fn build_duchon_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocation_weights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_collocation_operator_matrices_with_workspace(
        centers,
        collocation_weights,
        length_scale,
        power,
        nullspace_order,
        identifiability_transform,
        &mut workspace,
    )
}

pub fn build_duchon_collocation_operator_matrices_with_workspace(
    centers: ArrayView2<'_, f64>,
    collocation_weights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    let coeffs = length_scale
        .map(|scale| duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / scale.max(1e-300)));
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    let (d0_raw, d1_raw, d2_raw) =
        build_collocation_operators_from_radial(centers, centers, collocation_weights, |r| {
            duchon_kernel_radial_triplet(
                r,
                length_scale,
                p_order,
                s_order,
                centers.ncols(),
                coeffs.as_ref(),
            )
        })?;
    let d0_kernel = fast_ab(&d0_raw, &z);
    let d1_kernel = fast_ab(&d1_raw, &z);
    let d2_kernel = fast_ab(&d2_raw, &z);
    let poly = polynomial_block_from_order(centers, nullspace_order);
    let p_colloc = centers.nrows();
    let dim = centers.ncols();
    let kernel_cols = d0_kernel.ncols();
    let poly_cols = poly.ncols();
    let total_cols = kernel_cols + poly_cols;
    let row_scales = if let Some(w) = collocation_weights {
        if w.len() != p_colloc {
            return Err(BasisError::DimensionMismatch(format!(
                "collocation weight length mismatch: got {}, expected {p_colloc}",
                w.len()
            )));
        }
        let mut out = Vec::with_capacity(p_colloc);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                return Err(BasisError::InvalidInput(format!(
                    "collocation weights must be finite and non-negative; got {wk}"
                )));
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p_colloc]
    };
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    if poly_cols > 0 {
        let mut poly_scaled = poly;
        for (k, &scale_k) in row_scales.iter().enumerate() {
            poly_scaled.row_mut(k).mapv_inplace(|v| scale_k * v);
        }
        d0.slice_mut(s![.., kernel_cols..]).assign(&poly_scaled);
    }
    let mut d1 = Array2::<f64>::zeros((p_colloc * dim, total_cols));
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    if poly_cols > 1 {
        for k in 0..p_colloc {
            for axis in 0..dim {
                d1[[k * dim + axis, kernel_cols + 1 + axis]] = row_scales[k];
            }
        }
    }
    let mut d2 = Array2::<f64>::zeros((p_colloc, total_cols));
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        d0 = fast_ab(&d0, &z);
        d1 = fast_ab(&d1, &z);
        d2 = fast_ab(&d2, &z);
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: centers.to_owned(),
    })
}

#[inline(always)]
fn bessel_i0_manual(x: f64) -> f64 {
    // Manual Cephes-style approximation with two regions:
    //  - |x| < 3.75: polynomial in y=(x/3.75)^2
    //  - otherwise : asymptotic exp(|x|)/sqrt(|x|) times polynomial in y=3.75/|x|
    //
    // This avoids external dependencies and is numerically stable for the
    // argument ranges used by Duchon K0/K1 evaluation.
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

#[inline(always)]
fn bessel_i1_manual(x: f64) -> f64 {
    // Same split strategy as I0; odd symmetry is enforced for x<0.
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        x * (0.5
            + y * (0.878_905_94
                + y * (0.514_988_69
                    + y * (0.150_849_34
                        + y * (0.026_587_33 + y * (0.003_015_32 + y * 0.000_324_11))))))
    } else {
        let y = 3.75 / ax;
        let ans = (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (-0.039_880_24
                    + y * (-0.003_620_18
                        + y * (0.001_638_01
                            + y * (-0.010_315_55
                                + y * (0.022_829_67
                                    + y * (-0.028_953_12
                                        + y * (0.017_876_54 - y * 0.004_200_59))))))));
        if x < 0.0 { -ans } else { ans }
    }
}

#[inline(always)]
fn bessel_k0_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    // Manual Cephes-style K0 approximation with region split:
    //  - x<=2: logarithmic singular form with I0 coupling
    //  - x>2 : asymptotic exp(-x)/sqrt(x) form.
    //
    // This is dependency-free and deterministic, which keeps outer REML/BFGS
    // objectives smooth and reproducible run-to-run.
    if x_pos <= 2.0 {
        let y = (x_pos * x_pos) / 4.0;
        -((x_pos / 2.0).ln()) * bessel_i0_manual(x_pos)
            + (-0.577_215_66
                + y * (0.422_784_20
                    + y * (0.230_697_56
                        + y * (0.034_885_90
                            + y * (0.002_626_98 + y * (0.000_107_50 + y * 0.000_007_40))))))
    } else {
        let y = 2.0 / x_pos;
        (-x_pos).exp() / x_pos.sqrt()
            * (1.253_314_14
                + y * (-0.078_323_58
                    + y * (0.021_895_68
                        + y * (-0.010_624_46
                            + y * (0.005_878_72 + y * (-0.002_515_40 + y * 0.000_532_08))))))
    }
}

#[inline(always)]
fn bessel_k1_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    if x_pos <= 2.0 {
        let y = (x_pos * x_pos) / 4.0;
        (x_pos / 2.0).ln() * bessel_i1_manual(x_pos)
            + (1.0 / x_pos)
                * (1.0
                    + y * (0.154_431_44
                        + y * (-0.672_785_79
                            + y * (-0.181_568_97
                                + y * (-0.019_194_02 + y * (-0.001_104_04 + y * -0.000_046_86))))))
    } else {
        let y = 2.0 / x_pos;
        (-x_pos).exp() / x_pos.sqrt()
            * (1.253_314_14
                + y * (0.234_986_19
                    + y * (-0.036_556_20
                        + y * (0.015_042_68
                            + y * (-0.007_803_53 + y * (0.003_256_14 + y * -0.000_682_45))))))
    }
}

const DUCHON_DERIVATIVE_R_FLOOR_REL: f64 = 1e-5;

#[inline(always)]
fn duchon_p_from_nullspace_order(order: DuchonNullspaceOrder) -> usize {
    match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
    }
}

#[inline(always)]
fn binomial_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let kk = k.min(n - k);
    let mut c = 1.0_f64;
    for i in 0..kk {
        c *= (n - i) as f64;
        c /= (i + 1) as f64;
    }
    c
}

#[inline(always)]
fn gamma_lanczos(x: f64) -> f64 {
    // Numerical Recipes / Lanczos approximation with reflection formula.
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pix = std::f64::consts::PI * x;
        return std::f64::consts::PI / (pix.sin() * gamma_lanczos(1.0 - x));
    }
    let z = x - 1.0;
    let mut a = P[0];
    for (i, coeff) in P.iter().enumerate().skip(1) {
        a += coeff / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
}

#[inline(always)]
fn bessel_k_integer_order(n: usize, z: f64) -> f64 {
    let zz = z.max(1e-300);
    if n == 0 {
        return bessel_k0_stable(zz);
    }
    if n == 1 {
        return bessel_k1_stable(zz);
    }
    let mut km1 = bessel_k0_stable(zz);
    let mut k = bessel_k1_stable(zz);
    for m in 1..n {
        let kp1 = km1 + 2.0 * (m as f64) * k / zz;
        km1 = k;
        k = kp1;
    }
    k
}

#[inline(always)]
fn bessel_k_half_integer_order(l: usize, z: f64) -> f64 {
    // K_{l+1/2}(z) = sqrt(pi/(2z)) exp(-z) * sum_{j=0}^l c_j (1/(2z))^j
    // where c_j = (l+j)! / (j! (l-j)!).
    let zz = z.max(1e-300);
    let inv2z = 0.5 / zz;
    let mut sum = 0.0_f64;
    for j in 0..=l {
        let coeff = gamma_lanczos((l + j + 1) as f64)
            / (gamma_lanczos((j + 1) as f64) * gamma_lanczos((l - j + 1) as f64));
        sum += coeff * inv2z.powi(j as i32);
    }
    (std::f64::consts::PI / (2.0 * zz)).sqrt() * (-zz).exp() * sum
}

#[inline(always)]
fn bessel_k_real_half_integer_or_integer(nu_abs: f64, z: f64) -> Result<f64, BasisError> {
    let two_nu = (2.0 * nu_abs).round();
    if (two_nu - 2.0 * nu_abs).abs() > 1e-12 {
        return Err(BasisError::InvalidInput(format!(
            "unsupported Bessel-K order ν={nu_abs}; only integer/half-integer orders are supported"
        )));
    }
    let two_nu_i = two_nu as i64;
    if two_nu_i % 2 == 0 {
        let n = (two_nu_i / 2).max(0) as usize;
        Ok(bessel_k_integer_order(n, z))
    } else {
        let l = ((two_nu_i - 1) / 2).max(0) as usize;
        Ok(bessel_k_half_integer_order(l, z))
    }
}

#[inline(always)]
fn duchon_polyharmonic_block(r: f64, m: usize, k_dim: usize) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let k_half = 0.5 * k_dim as f64;
    let power_i = 2_i64 * (m as i64) - (k_dim as i64);
    let power_f = power_i as f64;
    // Log case: k even and m >= k/2 (gamma pole in generic power form).
    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = ((-1.0_f64).powi(m as i32))
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        return c * r.powf(power_f) * r.max(1e-300).ln();
    }
    let c = gamma_lanczos(k_half - m as f64)
        / (4.0_f64.powi(m as i32) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m as f64));
    c * r.powf(power_f)
}

#[inline(always)]
fn duchon_matern_block(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let nu_abs = nu.abs();
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    if r <= 0.0 {
        if nu_abs > 0.0 && nu > 0.0 {
            // lim_{z->0} z^nu K_nu(z) = 2^{nu-1} Γ(nu), nu>0.
            return Ok(c * 2.0_f64.powf(nu - 1.0) * gamma_lanczos(nu));
        }
        // Borderline/singular cases use intrinsic convention on the diagonal.
        return Ok(0.0);
    }
    let z = (kappa * r).max(1e-300);
    let k_nu = bessel_k_real_half_integer_or_integer(nu_abs, z)?;
    Ok(c * r.powf(nu) * k_nu)
}

#[inline(always)]
fn duchon_matern_block_triplet(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-block distance must be finite and non-negative".to_string(),
        ));
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-block kappa must be finite and positive".to_string(),
        ));
    }
    if r <= 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }

    // Exact derivatives for one shifted Matérn/Bessel-potential block.
    //
    // Write
    //   g(r; kappa) = c * r^nu * K_nu(z),   z = kappa r.
    //
    // The joint hyperparameterization is psi = log(kappa), so
    //   d/dpsi = kappa * d/dkappa
    // and z_psi = z. This is the Duchon "Matérn-like" block that appears after
    // partial-fraction decomposition of
    //   1 / (|w|^(2p) * (|w|^2 + kappa^2)^s).
    //
    // Identity:
    //   dK_nu/dz = -K_{nu-1}(z) - (nu/z) K_nu(z).
    //
    // First derivative:
    //   g' = c [nu r^(nu-1) K_nu + r^nu * kappa * dK_nu/dz]
    //      = c [nu r^(nu-1) K_nu - kappa r^nu K_{nu-1} - nu r^(nu-1) K_nu]
    //      = -c kappa r^nu K_{nu-1}.
    //
    // Second derivative:
    //   g'' = d/dr[-c kappa r^nu K_{nu-1}(z)]
    //       = -c kappa [nu r^(nu-1) K_{nu-1} + r^nu * kappa * dK_{nu-1}/dz]
    // with
    //   dK_{nu-1}/dz = -K_{nu-2}(z) - ((nu-1)/z) K_{nu-1}(z),
    // which simplifies to
    //   g'' = c kappa^2 r^nu K_{nu-2}(z) - c kappa r^(nu-1) K_{nu-1}(z).
    //
    // The resulting first/second radial derivatives are exact closed forms used
    // by the Duchon operator derivative path, not finite differences.
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    let z = (kappa * r).max(1e-300);
    let k_nu = bessel_k_real_half_integer_or_integer(nu.abs(), z)?;
    let k_nu_m1 = bessel_k_real_half_integer_or_integer((nu - 1.0).abs(), z)?;
    let k_nu_m2 = bessel_k_real_half_integer_or_integer((nu - 2.0).abs(), z)?;
    let r_nu = r.powf(nu);

    let value = c * r_nu * k_nu;
    let first = -c * kappa * r_nu * k_nu_m1;
    let second = c * kappa * kappa * r_nu * k_nu_m2 - c * kappa * r.powf(nu - 1.0) * k_nu_m1;
    Ok((value, first, second))
}

#[inline(always)]
fn duchon_polyharmonic_block_triplet(
    r: f64,
    m: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "polyharmonic distance must be finite and non-negative".to_string(),
        ));
    }
    if r <= 0.0 {
        // Intrinsic diagonal convention for generalized kernels.
        return Ok((0.0, 0.0, 0.0));
    }

    // Exact radial derivatives for the polyharmonic block Phi_m(r).
    //
    // Let alpha = 2m - d.
    // Case A: phi(r) = c * r^alpha
    //   phi'(r)  = c * alpha * r^(alpha-1)
    //   phi''(r) = c * alpha * (alpha-1) * r^(alpha-2)
    //
    // Case B: phi(r) = c * r^alpha * log(r)
    //   phi'(r)  = c * [alpha * r^(alpha-1) * log(r) + r^(alpha-1)]
    //   phi''(r) = c * [alpha*(alpha-1) * r^(alpha-2) * log(r)
    //                   + (2*alpha-1) * r^(alpha-2)].
    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
    let r_safe = r.max(1e-300);
    let r_alpha = r_safe.powf(alpha);
    let r_alpha_m1 = r_safe.powf(alpha - 1.0);
    let r_alpha_m2 = r_safe.powf(alpha - 2.0);
    let log_r = r_safe.ln();

    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = ((-1.0_f64).powi(m as i32))
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        let value = c * r_alpha * log_r;
        let first = c * (alpha * r_alpha_m1 * log_r + r_alpha_m1);
        let second =
            c * (alpha * (alpha - 1.0) * r_alpha_m2 * log_r + (2.0 * alpha - 1.0) * r_alpha_m2);
        return Ok((value, first, second));
    }

    let c = gamma_lanczos(k_half - m as f64)
        / (4.0_f64.powi(m as i32) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m as f64));
    let value = c * r_alpha;
    let first = c * alpha * r_alpha_m1;
    let second = c * alpha * (alpha - 1.0) * r_alpha_m2;
    Ok((value, first, second))
}

#[inline(always)]
fn pure_duchon_block_order(p_order: usize, s_order: usize) -> usize {
    p_order + s_order
}

struct DuchonPartialFractionCoeffs {
    a: Vec<f64>,
    b: Vec<f64>,
}

#[inline(always)]
fn duchon_partial_fraction_coeffs(
    p_order: usize,
    s_order: usize,
    kappa: f64,
) -> DuchonPartialFractionCoeffs {
    // 1/(ρ^{2p}(κ²+ρ²)^s) = Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)^n
    let mut a = vec![0.0_f64; p_order + 1]; // 1-based m
    let mut b = vec![0.0_f64; s_order + 1]; // 1-based n
    if s_order == 0 {
        if p_order > 0 {
            // Pure intrinsic polyharmonic case: no Matérn tail remains, so the
            // spectrum is exactly 1 / ρ^(2p).
            a[p_order] = 1.0;
        }
        return DuchonPartialFractionCoeffs { a, b };
    }
    for m in 1..=p_order {
        let sign = if (p_order - m) % 2 == 0 { 1.0 } else { -1.0 };
        let expo = -2.0 * (s_order + p_order - m) as f64;
        let comb = binomial_f64(s_order + p_order - m - 1, p_order - m);
        a[m] = sign * kappa.powf(expo) * comb;
    }
    for n in 1..=s_order {
        let sign = if p_order % 2 == 0 { 1.0 } else { -1.0 };
        let expo = -2.0 * (p_order + s_order - n) as f64;
        let comb = if p_order == 0 && n == s_order {
            // p=0 reduces to the pure Matérn block 1/(κ²+ρ²)^s.
            1.0
        } else {
            let top = p_order + s_order - n - 1;
            binomial_f64(top, s_order - n)
        };
        b[n] = sign * kappa.powf(expo) * comb;
    }
    DuchonPartialFractionCoeffs { a, b }
}

fn duchon_matern_kernel_general_from_distance(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon kernel distance must be finite and non-negative".to_string(),
        ));
    }
    // For intrinsic p>0 kernels, diagonal values are not uniquely defined in the
    // generalized-kernel sense; use intrinsic convention.
    if r == 0.0 && p_order > 0 {
        return Ok(0.0);
    }
    let Some(length_scale) = length_scale else {
        return Ok(duchon_polyharmonic_block(
            r,
            pure_duchon_block_order(p_order, s_order),
            k_dim,
        ));
    };
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon hybrid length_scale must be finite and positive".to_string(),
        ));
    }
    let kappa = 1.0 / length_scale;

    let coeffs_local;
    let coeffs_ref = if let Some(c) = coeffs {
        c
    } else {
        coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
        &coeffs_local
    };
    let mut val = 0.0_f64;
    for (m, coeff) in coeffs_ref.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val += coeff * duchon_polyharmonic_block(r, m, k_dim);
    }
    for (n, coeff) in coeffs_ref.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val += coeff * duchon_matern_block(r, kappa, n, k_dim)?;
    }
    Ok(val)
}

fn pairwise_distance_bounds(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = points[[i, c]] - points[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            if r.is_finite() && r > 0.0 {
                r_min = r_min.min(r);
                r_max = r_max.max(r);
            }
        }
    }
    if r_min.is_finite() && r_max.is_finite() && r_min > 0.0 && r_max > 0.0 {
        Some((r_min, r_max))
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpatialDistanceCacheKey {
    data_rows: usize,
    data_cols: usize,
    data_ptr: usize,
    data_stride0: isize,
    data_stride1: isize,
    centers_rows: usize,
    centers_cols: usize,
    centers_hash: u64,
}

#[derive(Debug, Clone)]
struct SpatialDistanceCacheEntry {
    data_center_r: Arc<Array2<f64>>,
    center_center_r: Arc<Array2<f64>>,
}

#[derive(Default)]
struct SpatialDistanceCache {
    map: HashMap<SpatialDistanceCacheKey, SpatialDistanceCacheEntry>,
    order: Vec<SpatialDistanceCacheKey>,
}

const SPATIAL_DISTANCE_CACHE_MAX_ENTRIES: usize = 12;
const SPATIAL_DISTANCE_CACHE_MIN_PAIRS: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConstraintNullspaceCacheKey {
    centers_rows: usize,
    centers_cols: usize,
    centers_hash: u64,
    order_code: u8,
}

#[derive(Default)]
struct ConstraintNullspaceCache {
    map: HashMap<ConstraintNullspaceCacheKey, Arc<Array2<f64>>>,
    order: Vec<ConstraintNullspaceCacheKey>,
}

const CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES: usize = 32;

#[derive(Default)]
struct BasisCacheContext {
    spatial_distance: SpatialDistanceCache,
    constraint_nullspace: ConstraintNullspaceCache,
}

/// Explicit per-run workspace for basis/spatial cache reuse.
///
/// Pass one workspace through repeated basis builds to avoid global mutable state
/// and to keep caching scoped to a caller-controlled lifecycle.
#[derive(Default)]
pub struct BasisWorkspace {
    cache: BasisCacheContext,
}

impl BasisWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
}

fn hash_array_view2(values: ArrayView2<'_, f64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    values.nrows().hash(&mut hasher);
    values.ncols().hash(&mut hasher);
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn compute_data_center_distances(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();
    let mut distances = Array2::<f64>::zeros((n, k));
    let result: Result<(), BasisError> = distances
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let mut dist2 = 0.0;
                for c in 0..d {
                    let delta = data[[i, c]] - centers[[j, c]];
                    dist2 += delta * delta;
                }
                row[j] = dist2.sqrt();
            }
            Ok(())
        });
    result?;
    Ok(distances)
}

fn compute_center_center_distances(centers: ArrayView2<'_, f64>) -> Array2<f64> {
    let d = centers.ncols();
    let k = centers.nrows();
    let mut distances = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[i, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            distances[[i, j]] = r;
            distances[[j, i]] = r;
        }
    }
    distances
}

fn spatial_distance_matrices(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    cache: &mut BasisCacheContext,
) -> Result<(Arc<Array2<f64>>, Arc<Array2<f64>>), BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    if n.saturating_mul(k) < SPATIAL_DISTANCE_CACHE_MIN_PAIRS {
        let dc = Arc::new(compute_data_center_distances(data, centers)?);
        let cc = Arc::new(compute_center_center_distances(centers));
        return Ok((dc, cc));
    }

    let key = SpatialDistanceCacheKey {
        data_rows: data.nrows(),
        data_cols: data.ncols(),
        data_ptr: data.as_ptr() as usize,
        data_stride0: data.strides()[0],
        data_stride1: data.strides()[1],
        centers_rows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_array_view2(centers),
    };

    if let Some(hit) = cache.spatial_distance.map.get(&key) {
        return Ok((hit.data_center_r.clone(), hit.center_center_r.clone()));
    }

    let computed_dc = Arc::new(compute_data_center_distances(data, centers)?);
    let computed_cc = Arc::new(compute_center_center_distances(centers));

    if let Some(hit) = cache.spatial_distance.map.get(&key) {
        return Ok((hit.data_center_r.clone(), hit.center_center_r.clone()));
    }
    cache.spatial_distance.map.insert(
        key,
        SpatialDistanceCacheEntry {
            data_center_r: computed_dc.clone(),
            center_center_r: computed_cc.clone(),
        },
    );
    cache.spatial_distance.order.push(key);
    while cache.spatial_distance.map.len() > SPATIAL_DISTANCE_CACHE_MAX_ENTRIES {
        if cache.spatial_distance.order.is_empty() {
            break;
        }
        let old_key = cache.spatial_distance.order.remove(0);
        cache.spatial_distance.map.remove(&old_key);
    }
    Ok((computed_dc, computed_cc))
}

fn constraint_nullspace_order_code(order: DuchonNullspaceOrder) -> u8 {
    match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
    }
}

fn kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let key = ConstraintNullspaceCacheKey {
        centers_rows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_array_view2(centers),
        order_code: constraint_nullspace_order_code(order),
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = polynomial_block_from_order(centers, order);
    let z = Arc::new(kernel_constraint_nullspace_from_matrix(p_k.view())?);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let old_key = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&old_key);
    }

    Ok((*z).clone())
}

fn matern_identifiability_transform(
    centers: ArrayView2<'_, f64>,
    identifiability: &MaternIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    let k = centers.nrows();
    match identifiability {
        MaternIdentifiability::None => Ok(None),
        MaternIdentifiability::CenterSumToZero => {
            let q = Array2::<f64>::ones((k, 1));
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::CenterLinearOrthogonal => {
            let q = polynomial_block_from_order(centers, DuchonNullspaceOrder::Linear);
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != k {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen Matérn identifiability transform mismatch: centers={k}, transform rows={}",
                    transform.nrows()
                )));
            }
            Ok(Some(transform.clone()))
        }
    }
}

fn build_matern_operator_penalty_candidates(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let ops = build_matern_collocation_operator_matrices(
        centers,
        None,
        length_scale,
        nu,
        include_intercept,
        z_opt.map(|z| z.view()),
    )?;
    Ok(operator_penalty_candidates_from_collocation(
        &ops.d0, &ops.d1, &ops.d2,
    ))
}

fn build_duchon_operator_penalty_candidates(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let ops = build_duchon_collocation_operator_matrices(
        centers,
        None,
        length_scale,
        power,
        nullspace_order,
        identifiability_transform,
    )?;
    Ok(operator_penalty_candidates_from_collocation(
        &ops.d0, &ops.d1, &ops.d2,
    ))
}

/// Creates a Matérn spline basis from data and centers.
///
/// The design is `[K | 1]` when `include_intercept=true` and `[K]` otherwise, where:
/// - `K_ij = k(||x_i - c_j||; length_scale, nu)` is the Matérn kernel block.
///
/// The default kernel penalty is `alpha' S alpha` with `S_jl = k(||c_j - c_l||)`, embedded
/// in the full coefficient space. With intercept included, that column is unpenalized by
/// `penalty_kernel`; optional `penalty_ridge` is a nullspace projector used for
/// double-penalty shrinkage of previously unpenalized directions.
///
/// NOTE: This follows the RKHS Gram construction S = K_CC (not K_CC^{-1}) in
/// coefficient space, with global scaling absorbed by the smoothing parameter λ.
pub fn create_matern_spline_basis(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
) -> Result<MaternSplineBasis, BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_matern_spline_basis_with_workspace(
        data,
        centers,
        length_scale,
        nu,
        include_intercept,
        &mut workspace,
    )
}

pub fn create_matern_spline_basis_with_workspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    workspace: &mut BasisWorkspace,
) -> Result<MaternSplineBasis, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();

    if d == 0 {
        return Err(BasisError::InvalidInput(
            "Matérn basis requires at least one covariate dimension".to_string(),
        ));
    }
    if k == 0 {
        return Err(BasisError::InvalidInput(
            "Matérn basis requires at least one center".to_string(),
        ));
    }
    if centers.ncols() != d {
        return Err(BasisError::DimensionMismatch(format!(
            "Matérn basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        )));
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "Matérn basis requires finite data and center values".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    // Practical safe operating range for κ from center geometry (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min], with κ = 1/length_scale.
    // Warn rather than silently clamp so callers keep explicit control.
    if let Some((r_min, r_max)) = pairwise_distance_bounds(centers) {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::warn!(
                "Matérn κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); kernel conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    let poly_cols = if include_intercept { 1 } else { 0 };
    let total_cols = k + poly_cols;

    let (data_center_r, center_center_r) =
        spatial_distance_matrices(data, centers, &mut workspace.cache)?;

    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                row[j] = matern_kernel_from_distance(data_center_r[[i, j]], length_scale, nu)?;
            }
            Ok(())
        });
    kernel_result?;

    // Center-center Gram matrix K_CC. In RKHS form, the kernel penalty on
    // radial coefficients is alpha^T K_CC alpha.
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let kij = matern_kernel_from_distance(center_center_r[[i, j]], length_scale, nu)?;
            center_kernel[[i, j]] = kij;
            center_kernel[[j, i]] = kij;
        }
    }

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis.slice_mut(s![.., 0..k]).assign(&kernel_block);
    if include_intercept {
        basis.column_mut(k).fill(1.0);
    }

    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    // RKHS coefficient penalty uses the center Gram matrix directly:
    //   S = K_CC  (not K_CC^{-1}).
    // This matches Duchon/Matérn spline theory where alpha^T K_CC alpha is the
    // native-space quadratic form up to a global scaling absorbed by lambda.
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_kernel)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

    Ok(MaternSplineBasis {
        basis,
        penalty_kernel,
        penalty_ridge,
        num_kernel_basis: k,
        num_polynomial_basis: poly_cols,
        dimension: d,
    })
}

/// Generic Matérn builder returning design + penalty list.
pub fn build_matern_basis(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_with_workspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let m = create_matern_spline_basis_with_workspace(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        workspace,
    )?;
    let identifiability_transform = z_opt.clone();
    let full_transform = z_opt.as_ref().map(|z| {
        if spec.include_intercept {
            append_intercept_to_transform(z)
        } else {
            z.clone()
        }
    });
    let design = if let Some(transform) = full_transform.as_ref() {
        fast_ab(&m.basis, transform)
    } else {
        m.basis.clone()
    };
    let candidates = if spec.double_penalty {
        vec![
            PenaltyCandidate {
                matrix: if let Some(transform) = full_transform.as_ref() {
                    let zt_s = transform.t().dot(&m.penalty_kernel);
                    zt_s.dot(transform)
                } else {
                    m.penalty_kernel.clone()
                },
                nullspace_dim_hint: 0,
                source: PenaltySource::Primary,
                normalization_scale: 1.0,
            },
            PenaltyCandidate {
                matrix: if let Some(transform) = full_transform.as_ref() {
                    let zt_s = transform.t().dot(&m.penalty_ridge);
                    zt_s.dot(transform)
                } else {
                    m.penalty_ridge.clone()
                },
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: 1.0,
            },
        ]
    } else {
        build_matern_operator_penalty_candidates(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
        )?
    };
    let (penalties, nullspace_dims, penalty_info) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penalty_info,
        metadata: BasisMetadata::Matern {
            centers,
            length_scale: spec.length_scale,
            nu: spec.nu,
            include_intercept: spec.include_intercept,
            identifiability_transform,
        },
    })
}

#[inline(always)]
fn eval_poly_with_derivatives(coeffs: &[f64], a: f64) -> (f64, f64, f64) {
    let mut p = 0.0;
    let mut p1 = 0.0;
    let mut p2 = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        p += c * a.powi(i as i32);
        if i >= 1 {
            p1 += (i as f64) * c * a.powi((i - 1) as i32);
        }
        if i >= 2 {
            p2 += (i as f64) * ((i - 1) as f64) * c * a.powi((i - 2) as i32);
        }
    }
    (p, p1, p2)
}

#[inline(always)]
fn matern_value_psi_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64), BasisError> {
    // Exact value + hyper-derivatives for psi = log(kappa)
    // ----------------------------------------------------
    // Half-integer Matérn kernels are represented as:
    //   phi(r) = p(a) * exp(-a),
    //   a = s r,   s = sqrt(2 nu) * kappa,   kappa = 1/length_scale.
    //
    // Differentiating with respect to a:
    //   d/da [p(a)e^{-a}]       = (p' - p)e^{-a}
    //   d^2/da^2 [p(a)e^{-a}]   = (p'' - 2p' + p)e^{-a}.
    //
    // We need derivatives w.r.t. psi=log(kappa), not r:
    //   da/dpsi = a,
    // therefore
    //   phi_psi      = a * (dphi/da)
    //   phi_psi_psi  = a*(dphi/da) + a^2*(d^2phi/da^2).
    //
    // This path is fully analytic and avoids FD in the hyper-derivative chain.
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    let kappa = 1.0 / length_scale;
    let (s, p): (f64, &[f64]) = match nu {
        MaternNu::Half => (kappa, &[1.0]),
        MaternNu::ThreeHalves => (3.0_f64.sqrt() * kappa, &[1.0, 1.0]),
        MaternNu::FiveHalves => (5.0_f64.sqrt() * kappa, &[1.0, 1.0, 1.0 / 3.0]),
        MaternNu::SevenHalves => (7.0_f64.sqrt() * kappa, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0]),
        MaternNu::NineHalves => (
            9.0_f64.sqrt() * kappa,
            &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
        ),
    };
    let a = s * r;
    let e = (-a).exp();
    let (p0, p1, p2) = eval_poly_with_derivatives(p, a);
    let value = e * p0;
    // Chain through psi=log(kappa): da/dpsi = a.
    let value_psi = e * a * (p1 - p0);
    let value_psi_psi = e * (a * (p1 - p0) + a * a * (p2 - 2.0 * p1 + p0));
    Ok((value, value_psi, value_psi_psi))
}

#[inline(always)]
fn exp_poly_scaled_s2_psi_triplet(s: f64, a: f64, coeffs: &[f64], scalar: f64) -> (f64, f64, f64) {
    // Helper for operator terms of the form:
    //   y(psi) = scalar * s(psi)^2 * exp(-a) * P(a),
    // where
    //   a = s r,  ds/dpsi = s,  da/dpsi = a.
    //
    // Product/chain expansion gives:
    //   y'  = scalar*s^2*e^{-a} [2P + a(P' - P)]
    //   y'' = scalar*s^2*e^{-a} [4P + 5a(P' - P) + a^2(P'' - 2P' + P)].
    //
    // Used for:
    // - phi''(r) pieces
    // - phi'(r)/r closed forms for nu>=3/2
    // under psi-derivatives.
    let e = (-a).exp();
    let (p0, p1, p2) = eval_poly_with_derivatives(coeffs, a);
    let d = p1 - p0;
    let y = scalar * s * s * e * p0;
    let y_psi = scalar * s * s * e * (2.0 * p0 + a * d);
    let y_psi_psi = scalar * s * s * e * (4.0 * p0 + 5.0 * a * d + a * a * (p2 - 2.0 * p1 + p0));
    (y, y_psi, y_psi_psi)
}

#[inline(always)]
fn matern_operator_psi_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
    dimension: usize,
) -> Result<
    (
        f64, // phi
        f64, // phi_psi
        f64, // phi_psi_psi
        f64, // phi_r_over_r
        f64, // (phi_r_over_r)_psi
        f64, // (phi_r_over_r)_psi_psi
        f64, // lap
        f64, // lap_psi
        f64, // lap_psi_psi
    ),
    BasisError,
> {
    // Operator-level analytic identities used by Thread-1 penalties:
    //   D0 uses phi,
    //   D1 uses phi'(r)/r,
    //   D2 uses Laplacian:
    //       Delta phi = phi''(r) + (d-1) * phi'(r)/r.
    //
    // For each half-integer nu, we use closed forms:
    //   phi''(r)      = s^2 * e^{-a} * R_nu(a),
    //   phi'(r)/r     = -s^2 * e^{-a} * Q_nu(a),  (nu>=3/2),
    // where Q_nu, R_nu are low-degree polynomials.
    //
    // Then psi-derivatives are obtained exactly through
    // exp_poly_scaled_s2_psi_triplet, avoiding finite differences.
    let (phi, phi_psi, phi_psi_psi) = matern_value_psi_triplet(r, length_scale, nu)?;
    let kappa = 1.0 / length_scale;
    let d = dimension as f64;
    let (s, q, rr): (f64, &[f64], &[f64]) = match nu {
        MaternNu::Half => (kappa, &[1.0], &[1.0]),
        MaternNu::ThreeHalves => (3.0_f64.sqrt() * kappa, &[1.0], &[-1.0, 1.0]),
        MaternNu::FiveHalves => (5.0_f64.sqrt() * kappa, &[1.0, 1.0], &[-1.0, -1.0, 1.0]),
        MaternNu::SevenHalves => (
            7.0_f64.sqrt() * kappa,
            &[3.0, 3.0, 1.0],
            &[-3.0, -3.0, 0.0, 1.0],
        ),
        MaternNu::NineHalves => (
            9.0_f64.sqrt() * kappa,
            &[15.0, 15.0, 6.0, 1.0],
            &[-15.0, -15.0, -3.0, 2.0, 1.0],
        ),
    };
    let a = s * r;
    let (phi_rr, phi_rr_psi, phi_rr_psi_psi) = exp_poly_scaled_s2_psi_triplet(s, a, rr, 1.0);

    // nu=1/2 has singular phi'(r)/r ~ -kappa/r as r->0.
    // We use the same finite r-floor regularization as operator assembly.
    let (ratio, ratio_psi, ratio_psi_psi) = if matches!(nu, MaternNu::Half) {
        let r_eff = r.max(1e-12);
        let e_eff = (-a).exp();
        let g = -(s / r_eff) * e_eff;
        let g_psi = -(s / r_eff) * e_eff * (1.0 - a);
        let g_psi_psi = -(s / r_eff) * e_eff * (1.0 - 3.0 * a + a * a);
        (g, g_psi, g_psi_psi)
    } else {
        exp_poly_scaled_s2_psi_triplet(s, a, q, -1.0)
    };

    let lap = phi_rr + (d - 1.0) * ratio;
    let lap_psi = phi_rr_psi + (d - 1.0) * ratio_psi;
    let lap_psi_psi = phi_rr_psi_psi + (d - 1.0) * ratio_psi_psi;

    if !phi.is_finite()
        || !phi_psi.is_finite()
        || !phi_psi_psi.is_finite()
        || !ratio.is_finite()
        || !ratio_psi.is_finite()
        || !ratio_psi_psi.is_finite()
        || !lap.is_finite()
        || !lap_psi.is_finite()
        || !lap_psi_psi.is_finite()
    {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Matérn psi-derivative operator values at r={r}, length_scale={length_scale}, nu={nu:?}"
        )));
    }
    Ok((
        phi,
        phi_psi,
        phi_psi_psi,
        ratio,
        ratio_psi,
        ratio_psi_psi,
        lap,
        lap_psi,
        lap_psi_psi,
    ))
}

fn gram_and_psi_derivatives_from_operator(
    d: &Array2<f64>,
    d_psi: &Array2<f64>,
    d_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Raw Gram derivatives from operator-collocation matrix D(psi):
    //   S_raw(psi) = D(psi)^T D(psi)
    //   S_raw'     = D'^T D + D^T D'
    //   S_raw''    = D''^T D + 2 D'^T D' + D^T D''.
    //
    // These are exactly the product-rule formulas requested in the math spec.
    let s_raw = symmetrize(&fast_ata(d));
    let s_raw_psi = symmetrize(&(d_psi.t().dot(d) + d.t().dot(d_psi)));
    let s_raw_psi_psi =
        symmetrize(&(d_psi_psi.t().dot(d) + d.t().dot(d_psi_psi) + 2.0 * d_psi.t().dot(d_psi)));
    (s_raw, s_raw_psi, s_raw_psi_psi)
}

#[inline(always)]
fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.t().dot(b).diag().sum()
}

fn normalize_penalty_with_psi_derivatives(
    s: &Array2<f64>,
    s_psi: &Array2<f64>,
    s_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, f64) {
    // Exact constrained-space Frobenius normalization derivatives:
    //
    // Let S = S_con(psi), c = ||S||_F = sqrt(tr(S^T S)).
    // Define:
    //   a = tr(S^T S'),
    //   b = tr((S')^T S') + tr(S^T S'').
    //
    // Then:
    //   c'  = a/c,
    //   c'' = b/c - a^2/c^3.
    //
    // For normalized S~ = S/c:
    //   S~'  = S'/c - (c'/c^2) S
    //   S~'' = S''/c - 2(c'/c^2)S' + (2(c')^2/c^3 - c''/c^2)S.
    //
    // This keeps hyper-derivative scaling coherent with the constrained REML
    // objective and matches the user-provided trace-only derivation.
    let fro2 = trace_of_product(s, s);
    let c = fro2.sqrt();
    if !c.is_finite() || c <= 1e-12 {
        return (
            s.clone(),
            Array2::<f64>::zeros(s.raw_dim()),
            Array2::<f64>::zeros(s.raw_dim()),
            1.0,
        );
    }

    let a = trace_of_product(s, s_psi);
    let b = trace_of_product(s_psi, s_psi) + trace_of_product(s, s_psi_psi);
    let c_psi = a / c;
    let c_psi_psi = b / c - (a * a) / (c * c * c);

    let s_tilde = s.mapv(|v| v / c);
    let s_tilde_psi = s_psi.mapv(|v| v / c) - s.mapv(|v| (c_psi / (c * c)) * v);
    let s_tilde_psi_psi = s_psi_psi.mapv(|v| v / c) - s_psi.mapv(|v| 2.0 * c_psi / (c * c) * v)
        + s.mapv(|v| ((2.0 * c_psi * c_psi) / (c * c * c) - c_psi_psi / (c * c)) * v);

    (s_tilde, s_tilde_psi, s_tilde_psi_psi, c)
}

fn build_matern_operator_penalty_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    // Full operator-to-penalty derivative pipeline in constrained coordinates:
    //
    // 1. Build D0, D1, D2 and their psi-derivatives from analytic radial forms.
    // 2. Apply identifiability transform Z at operator level:
    //      D_con = D Z, D_con' = D' Z, D_con'' = D'' Z
    //    (valid because Z is psi-independent).
    // 3. Build raw Gram derivatives per operator block:
    //      S_raw = D_con^T D_con, etc.
    // 4. Normalize each block by constrained Frobenius norm and propagate
    //    derivatives with exact quotient rules.
    //
    // Returned vectors correspond to [S0, S1, S2] derivatives after
    // constrained-space normalization.
    let p = centers.nrows();
    let d = centers.ncols();
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p, p));
    let mut d0_raw_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi = Array2::<f64>::zeros((p, p));
    let mut d0_raw_psi_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi_psi = Array2::<f64>::zeros((p, p));

    for k in 0..p {
        for j in 0..p {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[k, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            let (
                phi,
                phi_psi,
                phi_psi_psi,
                ratio,
                ratio_psi,
                ratio_psi_psi,
                lap,
                lap_psi,
                lap_psi_psi,
            ) = matern_operator_psi_triplet(r, length_scale, nu, d)?;
            d0_raw[[k, j]] = phi;
            d0_raw_psi[[k, j]] = phi_psi;
            d0_raw_psi_psi[[k, j]] = phi_psi_psi;
            d2_raw[[k, j]] = lap;
            d2_raw_psi[[k, j]] = lap_psi;
            d2_raw_psi_psi[[k, j]] = lap_psi_psi;
            for axis in 0..d {
                let delta = centers[[k, axis]] - centers[[j, axis]];
                let row = k * d + axis;
                d1_raw[[row, j]] = ratio * delta;
                d1_raw_psi[[row, j]] = ratio_psi * delta;
                d1_raw_psi_psi[[row, j]] = ratio_psi_psi * delta;
            }
        }
    }

    let project = |mat: Array2<f64>| {
        if let Some(z) = z_opt {
            fast_ab(&mat, z)
        } else {
            mat
        }
    };
    // With psi-independent Z this is algebraically exact:
    //   S_con = Z^T (D^T D) Z = (DZ)^T (DZ),
    // and identically for S_con', S_con'' using D'Z, D''Z.
    // So we can project operators first, then build Gram derivatives.
    let d0_kernel = project(d0_raw);
    let d0_kernel_psi = project(d0_raw_psi);
    let d0_kernel_psi_psi = project(d0_raw_psi_psi);
    let d1_kernel = project(d1_raw);
    let d1_kernel_psi = project(d1_raw_psi);
    let d1_kernel_psi_psi = project(d1_raw_psi_psi);
    let d2_kernel = project(d2_raw);
    let d2_kernel_psi = project(d2_raw_psi);
    let d2_kernel_psi_psi = project(d2_raw_psi_psi);

    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut d0 = Array2::<f64>::zeros((p, total_cols));
    let mut d1 = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2 = Array2::<f64>::zeros((p, total_cols));
    let mut d0_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d0_psi_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi_psi = Array2::<f64>::zeros((p, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    d0_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi);
    d1_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi);
    d2_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi);
    d0_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi_psi);
    d1_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi_psi);
    d2_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi_psi);
    if include_intercept {
        d0.column_mut(kernel_cols).fill(1.0);
    }

    let (s0, s0_psi, s0_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d0, &d0_psi, &d0_psi_psi);
    let (s1, s1_psi, s1_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d1, &d1_psi, &d1_psi_psi);
    let (s2, s2_psi, s2_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d2, &d2_psi, &d2_psi_psi);

    let (_s0_norm, s0_norm_psi, s0_norm_psi_psi, _c0) =
        normalize_penalty_with_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (_s1_norm, s1_norm_psi, s1_norm_psi_psi, _c1) =
        normalize_penalty_with_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (_s2_norm, s2_norm_psi, s2_norm_psi_psi, _c2) =
        normalize_penalty_with_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);

    Ok((
        vec![s0_norm_psi, s1_norm_psi, s2_norm_psi],
        vec![s0_norm_psi_psi, s1_norm_psi_psi, s2_norm_psi_psi],
    ))
}

fn build_matern_design_psi_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    let (data_center_r, _) = spatial_distance_matrices(data, centers, &mut workspace.cache)?;
    let mut kernel_psi = Array2::<f64>::zeros((n, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let r = data_center_r[[i, j]];
            kernel_psi[[i, j]] =
                matern_kernel_log_kappa_derivative_from_distance(r, length_scale, nu)?;
            kernel_psi_psi[[i, j]] =
                matern_kernel_log_kappa_second_derivative_from_distance(r, length_scale, nu)?;
        }
    }
    let (kernel_psi, kernel_psi_psi) = if let Some(z) = z_opt {
        (fast_ab(&kernel_psi, z), fast_ab(&kernel_psi_psi, z))
    } else {
        (kernel_psi, kernel_psi_psi)
    };
    let cols = kernel_psi.ncols();
    let total_cols = cols + usize::from(include_intercept);
    let mut out_psi = Array2::<f64>::zeros((n, total_cols));
    let mut out_psi_psi = Array2::<f64>::zeros((n, total_cols));
    out_psi.slice_mut(s![.., 0..cols]).assign(&kernel_psi);
    out_psi_psi
        .slice_mut(s![.., 0..cols])
        .assign(&kernel_psi_psi);
    Ok((out_psi, out_psi_psi))
}

fn build_matern_double_penalty_primary_with_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, f64), BasisError> {
    let k = centers.nrows();
    let kernel_cols = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut kernel = Array2::<f64>::zeros((k, k));
    let mut kernel_psi = Array2::<f64>::zeros((k, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((k, k));

    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for axis in 0..centers.ncols() {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            let value = matern_kernel_from_distance(r, length_scale, nu)?;
            let d1 = matern_kernel_log_kappa_derivative_from_distance(r, length_scale, nu)?;
            let d2 = matern_kernel_log_kappa_second_derivative_from_distance(r, length_scale, nu)?;
            kernel[[i, j]] = value;
            kernel[[j, i]] = value;
            kernel_psi[[i, j]] = d1;
            kernel_psi[[j, i]] = d1;
            kernel_psi_psi[[i, j]] = d2;
            kernel_psi_psi[[j, i]] = d2;
        }
    }

    let (kernel, kernel_psi, kernel_psi_psi) = if let Some(z) = z_opt {
        let zt_s = z.t().dot(&kernel);
        let zt_d1 = z.t().dot(&kernel_psi);
        let zt_d2 = z.t().dot(&kernel_psi_psi);
        (zt_s.dot(z), zt_d1.dot(z), zt_d2.dot(z))
    } else {
        (kernel, kernel_psi, kernel_psi_psi)
    };

    let mut s = Array2::<f64>::zeros((total_cols, total_cols));
    let mut s_psi = Array2::<f64>::zeros((total_cols, total_cols));
    let mut s_psi_psi = Array2::<f64>::zeros((total_cols, total_cols));
    s.slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel);
    s_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi);
    s_psi_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi_psi);
    let (s_norm, s_norm_psi, s_norm_psi_psi, c) =
        normalize_penalty_with_psi_derivatives(&s, &s_psi, &s_psi_psi);
    Ok((s_norm, s_norm_psi, s_norm_psi_psi, c))
}

fn active_matern_double_penalty_derivatives(
    penalty_info: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penalty_info
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected Matérn penalty source in double-penalty path: {other:?}"
            ))),
        })
        .collect()
}

pub fn build_matern_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappa_derivative_with_workspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappa_derivative_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    // Analytic psi derivative assembly for the Matérn basis block.
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let base = build_matern_basis_with_workspace(data, spec, workspace)?;
    let (design_derivative, _) = build_matern_design_psi_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        z_opt.as_ref(),
        workspace,
    )?;
    let penalties_derivative = if spec.double_penalty {
        let (_, primary_derivative, _, _) =
            build_matern_double_penalty_primary_with_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
            )?;
        active_matern_double_penalty_derivatives(&base.penalty_info, &primary_derivative)?
    } else {
        let (all_penalty_deriv, _) = build_matern_operator_penalty_psi_derivatives(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
        )?;
        active_operator_penalty_derivatives(&base.penalty_info, &all_penalty_deriv, "Matérn")?
    };

    Ok(BasisPsiDerivativeResult {
        design_derivative,
        penalties_derivative,
    })
}

pub fn build_matern_basis_log_kappa_second_derivative(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappa_second_derivative_with_workspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappa_second_derivative_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    // Analytic psi second-derivative assembly, matching the first-derivative
    // mapping logic and constrained normalized penalty geometry.
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let base = build_matern_basis_with_workspace(data, spec, workspace)?;
    let (_, design_second_derivative) = build_matern_design_psi_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        z_opt.as_ref(),
        workspace,
    )?;
    let penalties_second_derivative = if spec.double_penalty {
        let (_, _, primary_second_derivative, _) =
            build_matern_double_penalty_primary_with_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
            )?;
        active_matern_double_penalty_derivatives(&base.penalty_info, &primary_second_derivative)?
    } else {
        let (_, all_penalty_second_deriv) = build_matern_operator_penalty_psi_derivatives(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
        )?;
        active_operator_penalty_derivatives(
            &base.penalty_info,
            &all_penalty_second_deriv,
            "Matérn",
        )?
    };

    Ok(BasisPsiSecondDerivativeResult {
        design_second_derivative,
        penalties_second_derivative,
    })
}

#[inline(always)]
fn duchon_coeff_exponents(p_order: usize, s_order: usize, m_or_n: usize) -> f64 {
    // In the partial fractions
    //   1 / (z^p (z + kappa^2)^s)
    // = Σ a_m(kappa) / z^m + Σ b_n(kappa) / (z + kappa^2)^n,
    // both a_m and b_n are pure powers of kappa:
    //   c(kappa) = C * kappa^{-2(p+s-index)}.
    // With psi = log(kappa), that gives c_psi = alpha c and
    // c_psipsi = alpha^2 c with alpha below. This is the exact coefficient
    // derivative rule from the Duchon spectral factorization.
    -2.0 * (p_order + s_order - m_or_n) as f64
}

#[inline(always)]
fn duchon_scaling_exponent(p_order: usize, s_order: usize, k_dim: usize) -> f64 {
    k_dim as f64 - 2.0 * (p_order + s_order) as f64
}

#[inline(always)]
fn duchon_has_classical_second_order_origin(p_order: usize, s_order: usize, k_dim: usize) -> bool {
    2 * (p_order + s_order) > k_dim + 2
}

#[inline(always)]
fn radial_log_power_derivatives(c: f64, e: f64, a: f64, b: f64, r: f64) -> (f64, f64, f64) {
    let rr = r.max(1e-300);
    let log_r = rr.ln();
    let value = c * rr.powf(e) * (a * log_r + b);
    let first = c * rr.powf(e - 1.0) * (e * (a * log_r + b) + a);
    let second = c * rr.powf(e - 2.0) * (e * (e - 1.0) * (a * log_r + b) + (2.0 * e - 1.0) * a);
    (value, first, second)
}

#[inline(always)]
fn duchon_polyharmonic_q_l_r_triplets(
    r: f64,
    m: usize,
    k_dim: usize,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    if r <= 0.0 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    }
    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
    let rr = r.max(1e-300);
    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = ((-1.0_f64).powi(m as i32))
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        let q = radial_log_power_derivatives(c, alpha - 2.0, alpha, 1.0, rr);
        let lap = radial_log_power_derivatives(
            c,
            alpha - 2.0,
            alpha * (alpha + k_dim as f64 - 2.0),
            2.0 * alpha + k_dim as f64 - 2.0,
            rr,
        );
        return (q, lap);
    }
    let c = gamma_lanczos(k_half - m as f64)
        / (4.0_f64.powi(m as i32) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m as f64));
    let q0 = c * alpha * rr.powf(alpha - 2.0);
    let q1 = c * alpha * (alpha - 2.0) * rr.powf(alpha - 3.0);
    let q2 = c * alpha * (alpha - 2.0) * (alpha - 3.0) * rr.powf(alpha - 4.0);
    let l_coeff = c * alpha * (alpha + k_dim as f64 - 2.0);
    let l0 = l_coeff * rr.powf(alpha - 2.0);
    let l1 = l_coeff * (alpha - 2.0) * rr.powf(alpha - 3.0);
    let l2 = l_coeff * (alpha - 2.0) * (alpha - 3.0) * rr.powf(alpha - 4.0);
    ((q0, q1, q2), (l0, l1, l2))
}

#[inline(always)]
fn duchon_matern_block_r3(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    if r <= 0.0 {
        return Ok(0.0);
    }
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    let z = (kappa * r).max(1e-300);
    let k_nu_m2 = bessel_k_real_half_integer_or_integer((nu - 2.0).abs(), z)?;
    let k_nu_m3 = bessel_k_real_half_integer_or_integer((nu - 3.0).abs(), z)?;
    Ok(-c * kappa.powi(3) * r.powf(nu) * k_nu_m3
        + 3.0 * c * kappa.powi(2) * r.powf(nu - 1.0) * k_nu_m2)
}

#[inline(always)]
fn duchon_matern_block_q_l_r_triplets(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<((f64, f64, f64), (f64, f64, f64)), BasisError> {
    if r <= 0.0 {
        return Ok(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
    }
    let (_value, first, second) = duchon_matern_block_triplet(r, kappa, n_order, k_dim)?;
    let third = duchon_matern_block_r3(r, kappa, n_order, k_dim)?;
    let q0 = first / r;
    let q1 = (second - q0) / r;
    let q2 = third / r - 2.0 * second / (r * r) + 2.0 * first / (r * r * r);
    let (m0, m1, m2) = duchon_matern_block_triplet(r, kappa, n_order, k_dim)?;
    let (prev0, prev1, prev2) = if n_order > 1 {
        duchon_matern_block_triplet(r, kappa, n_order - 1, k_dim)?
    } else {
        (0.0, 0.0, 0.0)
    };
    let k2 = kappa * kappa;
    let l0 = k2 * m0 - prev0;
    let l1 = k2 * m1 - prev1;
    let l2 = k2 * m2 - prev2;
    Ok(((q0, q1, q2), (l0, l1, l2)))
}

#[inline(always)]
fn duchon_polyharmonic_second_collision_psi_triplet(
    length_scale: f64,
    m: usize,
    k_dim: usize,
) -> (f64, f64, f64) {
    // At exact center collisions the individual partial-fraction pieces can be
    // singular/cancellation-prone. We evaluate the polyharmonic second radial
    // derivative using the small-r asymptotic branch with r tied to length_scale,
    // then differentiate that asymptotic in psi analytically.
    let r_eff = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
    let e = alpha - 2.0;
    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = ((-1.0_f64).powi(m as i32))
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        let a = alpha * (alpha - 1.0);
        let b = 2.0 * alpha - 1.0;
        let base = c * r_eff.powf(e);
        let second = base * (a * r_eff.ln() + b);
        let second_psi = -e * second - base * a;
        let second_psi_psi = e * e * second + 2.0 * e * base * a;
        (second, second_psi, second_psi_psi)
    } else {
        let c = gamma_lanczos(k_half - m as f64)
            / (4.0_f64.powi(m as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64));
        let second = c * alpha * (alpha - 1.0) * r_eff.powf(e);
        let scale = 2.0 - alpha;
        (second, scale * second, scale * scale * second)
    }
}

#[inline(always)]
fn duchon_matern_second_collision_psi_triplet(
    length_scale: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    // Same collision strategy for shifted blocks: evaluate the second radial
    // derivative at a length-scale-relative floor and propagate psi derivatives
    // analytically using the local scaling exponent.
    let kappa = 1.0 / length_scale;
    let gamma = DUCHON_DERIVATIVE_R_FLOOR_REL;
    let r_eff = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let (_value, _first, second) = duchon_matern_block_triplet(r_eff, kappa, n_order, k_dim)?;
    let n = n_order as f64;
    let nu = n - 0.5 * k_dim as f64;
    let scale = 2.0 - 2.0 * nu;
    let _ = gamma;
    Ok((second, scale * second, scale * scale * second))
}

#[derive(Clone, Copy, Debug, Default)]
struct PsiTriplet {
    value: f64,
    psi: f64,
    psi_psi: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DuchonRadialCore {
    phi: PsiTriplet,
    gradient_ratio: PsiTriplet,
    laplacian: PsiTriplet,
}

#[derive(Clone, Copy, Debug, Default)]
struct DuchonRadialJets {
    phi: f64,
    phi_r: f64,
    phi_rr: f64,
    q: f64,
    q_r: f64,
    q_rr: f64,
    lap: f64,
    lap_r: f64,
    lap_rr: f64,
}

fn duchon_radial_jets(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialJets, BasisError> {
    let kappa = 1.0 / length_scale.max(1e-300);
    let r_eval = r.max(DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8));
    let mut out = DuchonRadialJets::default();

    // Value path keeps the intrinsic diagonal convention used by the actual basis.
    out.phi = duchon_matern_kernel_general_from_distance(
        r,
        Some(length_scale),
        p_order,
        s_order,
        k_dim,
        Some(coeffs),
    )?;
    if !out.phi.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon radial kernel value at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        )));
    }

    for (m, coeff) in coeffs.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (_vm, dm, d2m) = duchon_polyharmonic_block_triplet(r_eval, m, k_dim)?;
        let ((q0, q1, q2), (l0, l1, l2)) = duchon_polyharmonic_q_l_r_triplets(r_eval, m, k_dim);
        out.phi_r += coeff * dm;
        out.phi_rr += coeff * d2m;
        out.q += coeff * q0;
        out.q_r += coeff * q1;
        out.q_rr += coeff * q2;
        out.lap += coeff * l0;
        out.lap_r += coeff * l1;
        out.lap_rr += coeff * l2;
    }
    for (n, coeff) in coeffs.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (_vn, dn, d2n) = duchon_matern_block_triplet(r_eval, kappa, n, k_dim)?;
        let ((q0, q1, q2), (l0, l1, l2)) =
            duchon_matern_block_q_l_r_triplets(r_eval, kappa, n, k_dim)?;
        out.phi_r += coeff * dn;
        out.phi_rr += coeff * d2n;
        out.q += coeff * q0;
        out.q_r += coeff * q1;
        out.q_rr += coeff * q2;
        out.lap += coeff * l0;
        out.lap_r += coeff * l1;
        out.lap_rr += coeff * l2;
    }

    if r == 0.0 {
        out.phi_r = 0.0;
        out.q = 0.0;
        out.q_r = 0.0;
        out.q_rr = 0.0;
    }
    if !out.phi_r.is_finite()
        || !out.phi_rr.is_finite()
        || !out.q.is_finite()
        || !out.q_r.is_finite()
        || !out.q_rr.is_finite()
        || !out.lap.is_finite()
        || !out.lap_r.is_finite()
        || !out.lap_rr.is_finite()
    {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon radial jets at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        )));
    }
    Ok(out)
}

fn duchon_radial_core_psi_triplet(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialCore, BasisError> {
    // Duchon spectral derivation
    // --------------------------
    // Start from the isotropic spectrum
    //   K^(ω; kappa) ∝ 1 / (|ω|^(2p) * (kappa^2 + |ω|^2)^s),
    // with fixed integer orders p,s and continuous scale
    //   psi = log(kappa),   kappa = 1 / length_scale.
    //
    // Rescaling frequency by ω = kappa ξ gives the full spatial kernel scaling law
    //   phi(r; kappa) = kappa^delta H(kappa r),
    //   delta = d - 2p - 2s.
    //
    // Therefore the exact full-kernel psi derivatives are
    //   phi_psi     = delta * phi + r * phi_r
    //   phi_psipsi  = delta^2 * phi + (2 delta + 1) r phi_r + r^2 phi_rr.
    //
    // The operator scalars are
    //   q(r; kappa) = phi_r(r; kappa) / r
    //   ell(r; kappa) = Δphi(r; kappa) = phi_rr + (d-1) q.
    // Both q and ell scale with exponent delta + 2, so
    //   q_psi       = (delta + 2) q + r q_r
    //   q_psipsi    = (delta + 2)^2 q + (2 delta + 5) r q_r + r^2 q_rr
    // and identically for ell.
    //
    // Once {phi, q, ell} and their psi derivatives are known, the collocation
    // operators follow exactly:
    //   D0[k,j]         = phi(r_kj)
    //   D1[(k,a), j]    = q(r_kj) * (x_{k,a} - c_{j,a})
    //   D2[k,j]         = ell(r_kj)
    // and the penalty Hessians come from the Gram identities
    //   S_psi     = D_psi^T D + D^T D_psi
    //   S_psipsi  = D_psipsi^T D + 2 D_psi^T D_psi + D^T D_psipsi.
    //
    // This helper computes exactly that minimal scalar core:
    //   phi, q = phi_r / r, ell = Δphi
    // together with their first and second psi derivatives.
    let delta = duchon_scaling_exponent(p_order, s_order, k_dim);
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs)?;
    let phi = jets.phi;
    let phi_psi = delta * phi + r * jets.phi_r;
    let phi_psi_psi =
        delta * delta * phi + (2.0 * delta + 1.0) * r * jets.phi_r + r * r * jets.phi_rr;
    if r > 1e-10 {
        let g = jets.q;
        let lap = jets.lap;
        let phi_r = jets.phi_r;
        let phi_rr = jets.phi_rr;
        // Full-kernel scaling identity for q = phi_r / r:
        //   q_psi = (delta + 2) q + r q_r.
        // Since q_r = (phi_rr - q) / r, this is exactly equivalent to
        //   q_psi = (delta + 1) q + phi_rr,
        // which is the form used here because phi_rr is already available from
        // ell = Δphi = phi_rr + (d-1) q.
        let g_psi = (delta + 1.0) * g + phi_rr;
        let g_psi_psi = (delta + 2.0) * (delta + 2.0) * jets.q
            + (2.0 * delta + 5.0) * r * jets.q_r
            + r * r * jets.q_rr;
        let lap_psi = (delta + 2.0) * jets.lap + r * jets.lap_r;
        let lap_psi_psi = (delta + 2.0) * (delta + 2.0) * jets.lap
            + (2.0 * delta + 5.0) * r * jets.lap_r
            + r * r * jets.lap_rr;
        debug_assert!(
            ((delta * phi + r * phi_r) - phi_psi).abs() < 1e-7_f64.max(1e-7_f64 * phi.abs())
        );
        return Ok(DuchonRadialCore {
            phi: PsiTriplet {
                value: phi,
                psi: phi_psi,
                psi_psi: phi_psi_psi,
            },
            gradient_ratio: PsiTriplet {
                value: g,
                psi: g_psi,
                psi_psi: g_psi_psi,
            },
            laplacian: PsiTriplet {
                value: lap,
                psi: lap_psi,
                psi_psi: lap_psi_psi,
            },
        });
    }

    // Continuous center-collision extension for the scalar operator core:
    //   phi(0; kappa),  g(0; kappa) = 0,  L(0; kappa) = d * phi_rr(0; kappa).
    let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
        duchon_phi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
    Ok(DuchonRadialCore {
        phi: PsiTriplet {
            value: phi,
            psi: phi_psi,
            psi_psi: phi_psi_psi,
        },
        gradient_ratio: PsiTriplet::default(),
        laplacian: PsiTriplet {
            value: k_dim as f64 * phi_rr,
            psi: k_dim as f64 * phi_rr_psi,
            psi_psi: k_dim as f64 * phi_rr_psi_psi,
        },
    })
}

fn duchon_phi_rr_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(f64, f64, f64), BasisError> {
    // Center-collision rule
    // ---------------------
    // For a C^2 radial kernel one has
    //   lim_{r->0} phi_r(r)/r = phi_rr(0),
    //   lim_{r->0} Δphi(r)    = d * phi_rr(0).
    //
    // Spectrally, these classical diagonal limits exist when
    //   2(p + s) > d + 2.
    // In that regime phi_rr itself scales with exponent delta + 2, so the
    // diagonal psi derivatives are exactly
    //   phi_rr_psi     = (delta + 2) phi_rr
    //   phi_rr_psipsi  = (delta + 2)^2 phi_rr.
    //
    // Outside that regime the kernel is only intrinsic at the origin, so we keep
    // the assembled intrinsic convention from the matched block expansion.
    let mut phi_rr = 0.0;
    let mut phi_rr_psi = 0.0;
    let mut phi_rr_psi_psi = 0.0;
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let alpha_m = duchon_coeff_exponents(p_order, s_order, m);
        let (g0, g1, g2) = duchon_polyharmonic_second_collision_psi_triplet(length_scale, m, k_dim);
        phi_rr += a_m * g0;
        phi_rr_psi += alpha_m * a_m * g0 + a_m * g1;
        phi_rr_psi_psi += alpha_m * alpha_m * a_m * g0 + 2.0 * alpha_m * a_m * g1 + a_m * g2;
    }
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let beta_n = duchon_coeff_exponents(p_order, s_order, n);
        let (g0, g1, g2) = duchon_matern_second_collision_psi_triplet(length_scale, n, k_dim)?;
        phi_rr += b_n * g0;
        phi_rr_psi += beta_n * b_n * g0 + b_n * g1;
        phi_rr_psi_psi += beta_n * beta_n * b_n * g0 + 2.0 * beta_n * b_n * g1 + b_n * g2;
    }
    if duchon_has_classical_second_order_origin(p_order, s_order, k_dim) {
        let scale = duchon_scaling_exponent(p_order, s_order, k_dim) + 2.0;
        return Ok((phi_rr, scale * phi_rr, scale * scale * phi_rr));
    }
    Ok((phi_rr, phi_rr_psi, phi_rr_psi_psi))
}

fn build_duchon_design_psi_derivatives_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    // Exact Duchon design derivatives:
    // 1. evaluate phi_psi and phi_psipsi at each data/center distance
    // 2. project the kernel block with the same nullspace constraint used by the basis
    // 3. append polynomial columns; their psi derivatives are zero because p and s are fixed
    // 4. apply any frozen identifiability transform
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let p_order = duchon_p_from_nullspace_order(spec.nullspace_order);
    let s_order = spec.power;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let z_kernel =
        kernel_constraint_nullspace(centers.view(), spec.nullspace_order, &mut workspace.cache)?;
    let (data_center_r, _) = spatial_distance_matrices(data, centers.view(), &mut workspace.cache)?;
    let n = data.nrows();
    let k = centers.nrows();
    let mut kernel_psi = Array2::<f64>::zeros((n, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let r = data_center_r[[i, j]];
            let core = duchon_radial_core_psi_triplet(
                r,
                length_scale,
                p_order,
                s_order,
                data.ncols(),
                &coeffs,
            )?;
            kernel_psi[[i, j]] = core.phi.psi;
            kernel_psi_psi[[i, j]] = core.phi.psi_psi;
        }
    }
    let kernel_psi = fast_ab(&kernel_psi, &z_kernel);
    let kernel_psi_psi = fast_ab(&kernel_psi_psi, &z_kernel);
    let poly_cols = polynomial_block_from_order(data, spec.nullspace_order).ncols();
    let kernel_cols = kernel_psi.ncols();
    let total_cols = kernel_cols + poly_cols;
    let mut out_psi = Array2::<f64>::zeros((n, total_cols));
    let mut out_psi_psi = Array2::<f64>::zeros((n, total_cols));
    out_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_psi);
    out_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_psi_psi);
    if let Some(zf) = identifiability_transform {
        if total_cols != zf.nrows() {
            return Err(BasisError::DimensionMismatch(format!(
                "Duchon identifiability transform mismatch in design derivatives: local cols={}, transform rows={}",
                total_cols,
                zf.nrows()
            )));
        }
        return Ok((fast_ab(&out_psi, zf), fast_ab(&out_psi_psi, zf)));
    }
    Ok((out_psi, out_psi_psi))
}

fn build_duchon_operator_penalty_psi_derivatives_with_workspace(
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    // Build exact Duchon operator derivatives for the canonical three-penalty path.
    // With psi = log(kappa), the operator rows are
    //   D0[k,j]      = phi(r_kj)
    //   D1[(k,a),j]  = (phi_r(r_kj)/r_kj) * (x_{k,a} - c_{j,a})
    //   D2[k,j]      = Δ phi(r_kj)
    //
    // Then form normalized Gram penalties
    //   S_i = D_i^T D_i / ||D_i^T D_i||_F
    // and differentiate them analytically in psi using
    //   S_psi     = D_psi^T D + D^T D_psi
    //   S_psipsi  = D_psipsi^T D + 2 D_psi^T D_psi + D^T D_psipsi
    // followed by the exact Frobenius-norm quotient rule for normalization.
    //
    // This matches the continuous Duchon hyperparameter calculus with fixed
    // integer orders p and s. Changing p or s is discrete model selection, not
    // part of the psi = log(kappa) Hessian.
    let p = centers.nrows();
    let d = centers.ncols();
    let p_order = duchon_p_from_nullspace_order(spec.nullspace_order);
    let s_order = spec.power;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let z_kernel =
        kernel_constraint_nullspace(centers, spec.nullspace_order, &mut workspace.cache)?;
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p, p));
    let mut d0_raw_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi = Array2::<f64>::zeros((p, p));
    let mut d0_raw_psi_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi_psi = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        for j in 0..p {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[k, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            let core =
                duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, d, &coeffs)?;
            d0_raw[[k, j]] = core.phi.value;
            d0_raw_psi[[k, j]] = core.phi.psi;
            d0_raw_psi_psi[[k, j]] = core.phi.psi_psi;
            for axis in 0..d {
                let delta = centers[[k, axis]] - centers[[j, axis]];
                let row = k * d + axis;
                d1_raw[[row, j]] = core.gradient_ratio.value * delta;
                d1_raw_psi[[row, j]] = core.gradient_ratio.psi * delta;
                d1_raw_psi_psi[[row, j]] = core.gradient_ratio.psi_psi * delta;
            }
            d2_raw[[k, j]] = core.laplacian.value;
            d2_raw_psi[[k, j]] = core.laplacian.psi;
            d2_raw_psi_psi[[k, j]] = core.laplacian.psi_psi;
        }
    }
    let project_kernel = |mat: Array2<f64>| fast_ab(&mat, &z_kernel);
    let d0_kernel = project_kernel(d0_raw);
    let d0_kernel_psi = project_kernel(d0_raw_psi);
    let d0_kernel_psi_psi = project_kernel(d0_raw_psi_psi);
    let d1_kernel = project_kernel(d1_raw);
    let d1_kernel_psi = project_kernel(d1_raw_psi);
    let d1_kernel_psi_psi = project_kernel(d1_raw_psi_psi);
    let d2_kernel = project_kernel(d2_raw);
    let d2_kernel_psi = project_kernel(d2_raw_psi);
    let d2_kernel_psi_psi = project_kernel(d2_raw_psi_psi);
    let poly = polynomial_block_from_order(centers, spec.nullspace_order);
    let poly_cols = poly.ncols();
    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + poly_cols;
    let mut d0 = Array2::<f64>::zeros((p, total_cols));
    let mut d1 = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2 = Array2::<f64>::zeros((p, total_cols));
    let mut d0_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d0_psi_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi_psi = Array2::<f64>::zeros((p, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    d0_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi);
    d1_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi);
    d2_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi);
    d0_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi_psi);
    d1_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi_psi);
    d2_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi_psi);
    if poly_cols > 0 {
        d0.slice_mut(s![.., kernel_cols..]).assign(&poly);
    }
    if poly_cols > 1 {
        for k in 0..p {
            for axis in 0..d {
                d1[[k * d + axis, kernel_cols + 1 + axis]] = 1.0;
            }
        }
    }
    if let Some(zf) = identifiability_transform {
        if total_cols != zf.nrows() {
            return Err(BasisError::DimensionMismatch(format!(
                "Duchon identifiability transform mismatch in operator derivatives: local cols={}, transform rows={}",
                total_cols,
                zf.nrows()
            )));
        }
        d0 = fast_ab(&d0, zf);
        d1 = fast_ab(&d1, zf);
        d2 = fast_ab(&d2, zf);
        d0_psi = fast_ab(&d0_psi, zf);
        d1_psi = fast_ab(&d1_psi, zf);
        d2_psi = fast_ab(&d2_psi, zf);
        d0_psi_psi = fast_ab(&d0_psi_psi, zf);
        d1_psi_psi = fast_ab(&d1_psi_psi, zf);
        d2_psi_psi = fast_ab(&d2_psi_psi, zf);
    }
    let (s0, s0_psi, s0_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d0, &d0_psi, &d0_psi_psi);
    let (s1, s1_psi, s1_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d1, &d1_psi, &d1_psi_psi);
    let (s2, s2_psi, s2_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d2, &d2_psi, &d2_psi_psi);
    let (_s0n, s0n_psi, s0n_psi_psi, _c0) =
        normalize_penalty_with_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (_s1n, s1n_psi, s1n_psi_psi, _c1) =
        normalize_penalty_with_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (_s2n, s2n_psi, s2n_psi_psi, _c2) =
        normalize_penalty_with_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);
    Ok((
        vec![s0n_psi, s1n_psi, s2n_psi],
        vec![s0n_psi_psi, s1n_psi_psi, s2n_psi_psi],
    ))
}

pub fn build_duchon_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivative_with_workspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappa_derivative_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let base = build_duchon_basis_with_workspace(data, spec, workspace)?;
    let identifiability_transform = match &base.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => identifiability_transform.as_ref(),
        _ => None,
    };
    let (design_derivative, _) = build_duchon_design_psi_derivatives_with_workspace(
        data,
        spec,
        identifiability_transform,
        workspace,
    )?;
    let (all_penalty_deriv, _) = build_duchon_operator_penalty_psi_derivatives_with_workspace(
        centers.view(),
        spec,
        identifiability_transform,
        workspace,
    )?;
    let penalties_derivative =
        active_operator_penalty_derivatives(&base.penalty_info, &all_penalty_deriv, "Duchon")?;
    Ok(BasisPsiDerivativeResult {
        design_derivative,
        penalties_derivative,
    })
}

pub fn build_duchon_basis_log_kappa_second_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_second_derivative_with_workspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappa_second_derivative_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let base = build_duchon_basis_with_workspace(data, spec, workspace)?;
    let identifiability_transform = match &base.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => identifiability_transform.as_ref(),
        _ => None,
    };
    let (_, design_second_derivative) = build_duchon_design_psi_derivatives_with_workspace(
        data,
        spec,
        identifiability_transform,
        workspace,
    )?;
    let (_, all_penalty_second_deriv) =
        build_duchon_operator_penalty_psi_derivatives_with_workspace(
            centers.view(),
            spec,
            identifiability_transform,
            workspace,
        )?;
    let penalties_second_derivative = active_operator_penalty_derivatives(
        &base.penalty_info,
        &all_penalty_second_deriv,
        "Duchon",
    )?;
    Ok(BasisPsiSecondDerivativeResult {
        design_second_derivative,
        penalties_second_derivative,
    })
}

/// Creates a Duchon-like basis with spectral penalty
///   P(w) = ||w||^(2p) * (kappa^2 + ||w||^2)^s
/// using:
/// - integer-parameter partial-fraction decomposition in spectral space,
/// - finite spatial kernel sum of polyharmonic + Matérn blocks,
/// - explicit polynomial null-space block determined by `nullspace_order`,
/// - side-constraint projection `P(centers)^T alpha = 0` for `p > 0`.
///
/// API mapping:
/// - `p` is determined by `nullspace_order`:
///   - `Zero`   -> p = 0
///   - `Linear` -> p = 1
/// - `s` is determined directly by `power`
///
pub fn create_duchon_spline_basis(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<DuchonSplineBasis, BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_duchon_spline_basis_with_workspace(
        data,
        centers,
        length_scale,
        power,
        nullspace_order,
        &mut workspace,
    )
}

pub fn create_duchon_spline_basis_with_workspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonSplineBasis, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();

    if d == 0 {
        return Err(BasisError::InvalidInput(
            "Duchon basis requires at least one covariate dimension".to_string(),
        ));
    }
    if k == 0 {
        return Err(BasisError::InvalidInput(
            "Duchon basis requires at least one center".to_string(),
        ));
    }
    if centers.ncols() != d {
        return Err(BasisError::DimensionMismatch(format!(
            "Duchon basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        )));
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "Duchon basis requires finite data and center values".to_string(),
        ));
    }

    let poly_block = polynomial_block_from_order(data, nullspace_order);
    // Z spans null(Q^T), where Q contains polynomial side conditions at centers.
    // Reparameterizing alpha = Z gamma enforces conditional-PD constraints once
    // and yields free-parameter penalty gamma^T (Z^T K_CC Z) gamma.
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;

    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    let coeffs = if length_scale.is_none() {
        None
    } else if p_order == 1 && s_order == 4 && d == 10 {
        None
    } else {
        Some(duchon_partial_fraction_coeffs(
            p_order,
            s_order,
            1.0 / length_scale.unwrap().max(1e-300),
        ))
    };

    // Point-evaluation sufficiency check: 2p + 2s > k for a proper RKHS kernel.
    // For intrinsic constructions (p>0), borderline or subcritical settings may
    // still be usable with side constraints, but can become numerically delicate.
    let regularity_margin = 2 * p_order + 2 * s_order;
    if regularity_margin <= d {
        log::warn!(
            "Duchon regularity is at/below point-evaluation threshold: 2p+2s={} <= k={} (p={}, s={}); using intrinsic diagonal convention",
            regularity_margin,
            d,
            p_order,
            s_order
        );
    }

    // Practical safe operating range (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min]
    // where r_min/r_max are pairwise center distance extrema.
    // We keep user-provided κ but emit a warning outside this regime.
    if let (Some(length_scale), Some((r_min, r_max))) =
        (length_scale, pairwise_distance_bounds(centers))
    {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::warn!(
                "Duchon κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); numerical conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    let (data_center_r, center_center_r) =
        spatial_distance_matrices(data, centers, &mut workspace.cache)?;

    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                row[j] = duchon_matern_kernel_general_from_distance(
                    data_center_r[[i, j]],
                    length_scale,
                    p_order,
                    s_order,
                    d,
                    coeffs.as_ref(),
                )?;
            }
            Ok(())
        });
    kernel_result?;

    let mut center_kernel = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let kij = duchon_matern_kernel_general_from_distance(
                center_center_r[[i, j]],
                length_scale,
                p_order,
                s_order,
                d,
                coeffs.as_ref(),
            )?;
            center_kernel[[i, j]] = kij;
            center_kernel[[j, i]] = kij;
        }
    }

    let kernel_constrained = fast_ab(&kernel_block, &z);
    // Constrained Gram penalty block: S_free = Z^T K_CC Z.
    // This is the standard Duchon/thin-plate constrained coefficient penalty.
    // Constrained (conditionally PD) penalty:
    //   alpha = Z gamma,  Q^T alpha = 0  =>  gamma^T (Z^T K_CC Z) gamma.
    let omega_constrained = {
        let zt_k = fast_atb(&z, &center_kernel);
        fast_ab(&zt_k, &z)
    };
    let kernel_cols = kernel_constrained.ncols();
    let poly_cols = poly_block.ncols();
    let total_cols = kernel_cols + poly_cols;

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_constrained);
    if poly_cols > 0 {
        basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);
    }

    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_kernel
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_constrained);
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_kernel)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

    Ok(DuchonSplineBasis {
        basis,
        penalty_kernel,
        penalty_ridge,
        num_kernel_basis: kernel_cols,
        num_polynomial_basis: poly_cols,
        dimension: d,
        nullspace_order,
    })
}

/// Generic Duchon builder returning design + penalty list.
pub fn build_duchon_basis(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_with_workspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_with_workspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let d = create_duchon_spline_basis_with_workspace(
        data,
        centers.view(),
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        workspace,
    )?;
    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        d.basis.view(),
        &spec.identifiability,
        "Duchon",
    )?;
    let design = if let Some(z) = identifiability_transform.as_ref() {
        fast_ab(&d.basis, z)
    } else {
        d.basis.clone()
    };
    let candidates = build_duchon_operator_penalty_candidates(
        centers.view(),
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        identifiability_transform.as_ref().map(|z| z.view()),
    )?;
    if spec.double_penalty {
        log::debug!(
            "Duchon double_penalty requested but ignored because S0 mass penalty already shrinks the nullspace"
        );
    }
    let (penalties, nullspace_dims, penalty_info) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penalty_info,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            power: spec.power,
            nullspace_order: spec.nullspace_order,
            identifiability_transform,
        },
    })
}

fn polynomial_block_from_order(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    match order {
        DuchonNullspaceOrder::Zero => Array2::<f64>::zeros((n, 0)),
        DuchonNullspaceOrder::Linear => {
            let mut poly = Array2::<f64>::zeros((n, d + 1));
            poly.column_mut(0).fill(1.0);
            for c in 0..d {
                poly.column_mut(c + 1).assign(&points.column(c));
            }
            poly
        }
    }
}

fn kernel_constraint_nullspace_from_matrix(
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let k = constraint_matrix.nrows();
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok(Array2::<f64>::eye(k));
    }
    // Constraint system Q^T alpha = 0. The trailing columns of the orthogonal
    // factor in a column-pivoted QR of Q span null(Q^T).
    let (z, _) = rrqr_nullspace_basis(&constraint_matrix, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    Ok(z)
}

/// Deterministically selects thin-plate knots via farthest-point sampling.
///
/// This produces a space-filling subset without introducing RNG/state coupling.
pub fn select_thin_plate_knots(
    data: ArrayView2<f64>,
    num_knots: usize,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "thin-plate spline requires at least one covariate dimension".to_string(),
        ));
    }
    if n == 0 {
        return Err(BasisError::InvalidInput(
            "cannot select thin-plate knots from empty data".to_string(),
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "thin-plate spline knot selection requires finite data".to_string(),
        ));
    }
    if num_knots == 0 {
        return Err(BasisError::InvalidInput(
            "thin-plate spline knot count must be positive".to_string(),
        ));
    }
    if num_knots > n {
        return Err(BasisError::InvalidInput(format!(
            "requested {} knots but only {} rows are available",
            num_knots, n
        )));
    }

    // Deterministic seed point: lexicographically smallest row.
    let mut seed_idx = 0usize;
    for i in 1..n {
        let mut choose_i = false;
        for c in 0..d {
            let ai = data[[i, c]];
            let as_ = data[[seed_idx, c]];
            if ai < as_ {
                choose_i = true;
                break;
            }
            if ai > as_ {
                break;
            }
        }
        if choose_i {
            seed_idx = i;
        }
    }

    let mut selected = Vec::with_capacity(num_knots);
    let mut chosen = vec![false; n];
    let mut min_dist2 = vec![f64::INFINITY; n];

    selected.push(seed_idx);
    chosen[seed_idx] = true;

    min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let mut d2 = 0.0;
        for c in 0..d {
            let delta = data[[i, c]] - data[[seed_idx, c]];
            d2 += delta * delta;
        }
        *slot = d2;
    });
    min_dist2[seed_idx] = 0.0;

    while selected.len() < num_knots {
        let best_idx = min_dist2
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !chosen[*i])
            .map(|(i, &cand)| (i, cand))
            .reduce_with(|a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(i, _)| i);
        let next_idx = match best_idx {
            Some(i) => i,
            None => break,
        };
        selected.push(next_idx);
        chosen[next_idx] = true;

        min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
            if chosen[i] {
                return;
            }
            let mut d2 = 0.0;
            for c in 0..d {
                let delta = data[[i, c]] - data[[next_idx, c]];
                d2 += delta * delta;
            }
            if d2 < *slot {
                *slot = d2;
            }
        });
    }

    let mut knots = Array2::<f64>::zeros((selected.len(), d));
    for (r, &idx) in selected.iter().enumerate() {
        knots.row_mut(r).assign(&data.row(idx));
    }
    Ok(knots)
}

#[inline(always)]
fn thin_plate_kernel_m2_from_dist2(dist2: f64, dimension: usize) -> Result<f64, BasisError> {
    if !dist2.is_finite() || dist2 < 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if dist2 == 0.0 {
        return Ok(0.0);
    }
    match dimension {
        // For the m=2 thin-plate spline, the radial kernel is the biharmonic
        // fundamental solution modulo low-order polynomials:
        //   d=1:  r^3
        //   d=2:  r^2 log(r)
        //   d=3: -r
        // The d=3 sign is essential for conditional positive definiteness
        // after projecting out the polynomial side constraints.
        1 => Ok(dist2 * dist2.sqrt()),
        2 => Ok(0.5 * dist2 * dist2.ln()),
        3 => Ok(-dist2.sqrt()),
        _ => Err(BasisError::InvalidInput(format!(
            "thin-plate spline (m=2) currently supports dimensions 1..=3, got {dimension}"
        ))),
    }
}

/// Creates a thin-plate regression spline basis (m=2) from data and knot locations.
///
/// # Arguments
/// * `data` - `n x d` matrix of evaluation points
/// * `knots` - `k x d` matrix of knot locations
///
/// # Returns
/// `ThinPlateSplineBasis` containing:
/// - `basis`: `n x (k + d + 1)` matrix (`[K | P]`)
/// - `penalty_bending`: constrained TPS curvature penalty
/// - `penalty_ridge`: identity penalty for null-space shrinkage
pub fn create_thin_plate_spline_basis(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
) -> Result<ThinPlateSplineBasis, BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basis_with_workspace(data, knots, &mut workspace)
}

pub fn create_thin_plate_spline_basis_with_workspace(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<ThinPlateSplineBasis, BasisError> {
    let n = data.nrows();
    let k = knots.nrows();
    let d = data.ncols();

    if d == 0 {
        return Err(BasisError::InvalidInput(
            "thin-plate spline requires at least one covariate dimension".to_string(),
        ));
    }
    if d != knots.ncols() {
        return Err(BasisError::DimensionMismatch(format!(
            "thin-plate spline dimension mismatch: data has {} columns, knots have {} columns",
            d,
            knots.ncols()
        )));
    }
    if k < d + 1 {
        return Err(BasisError::InvalidInput(format!(
            "thin-plate spline requires at least d+1 knots ({}), got {}",
            d + 1,
            k
        )));
    }
    if data.iter().any(|v| !v.is_finite()) || knots.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "thin-plate spline requires finite data and knot values".to_string(),
        ));
    }

    let poly_cols = d + 1;

    // K block: radial basis evaluations data -> knots
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let mut dist2 = 0.0;
                for c in 0..d {
                    let delta = data[[i, c]] - knots[[j, c]];
                    dist2 += delta * delta;
                }
                row[j] = thin_plate_kernel_m2_from_dist2(dist2, d)?;
            }
            Ok(())
        });
    kernel_result?;

    // P block: [1, x_1, ..., x_d]
    let poly_block = polynomial_block_from_order(data, DuchonNullspaceOrder::Linear);

    // Omega block on knots
    let mut omega = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = knots[[i, c]] - knots[[j, c]];
                dist2 += delta * delta;
            }
            let kij = thin_plate_kernel_m2_from_dist2(dist2, d)?;
            omega[[i, j]] = kij;
            omega[[j, i]] = kij;
        }
    }

    // Enforce TPS side-constraint P(knots)^T α = 0 by projecting onto
    // the nullspace of P(knots)^T.
    let z = kernel_constraint_nullspace(knots, DuchonNullspaceOrder::Linear, &mut workspace.cache)?;
    let kernel_constrained = fast_ab(&kernel_block, &z);
    let omega_constrained = {
        let zt_o = fast_atb(&z, &omega);
        fast_ab(&zt_o, &z)
    };
    let omega_psd = validate_psd_penalty(
        &omega_constrained,
        &format!("thin_plate bending penalty (dimension={d})"),
        "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
    )?;
    debug_assert!(omega_psd.min_eigenvalue >= -omega_psd.tolerance);
    debug_assert!(omega_psd.max_abs_eigenvalue.is_finite());
    debug_assert!(omega_psd.effective_rank <= omega_constrained.nrows());

    let kernel_cols = kernel_constrained.ncols();
    let total_cols = kernel_cols + poly_cols;
    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_constrained);
    basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);

    let mut penalty_bending = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_bending
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_constrained);
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_bending)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

    Ok(ThinPlateSplineBasis {
        basis,
        penalty_bending,
        penalty_ridge,
        num_kernel_basis: kernel_cols,
        num_polynomial_basis: poly_cols,
        dimension: d,
    })
}

/// High-level TPS constructor: selects knots from data, then builds basis+penalty.
pub fn create_thin_plate_spline_basis_with_knot_count(
    data: ArrayView2<f64>,
    num_knots: usize,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basis_with_knot_count_and_workspace(data, num_knots, &mut workspace)
}

pub fn create_thin_plate_spline_basis_with_knot_count_and_workspace(
    data: ArrayView2<f64>,
    num_knots: usize,
    workspace: &mut BasisWorkspace,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let knots = select_thin_plate_knots(data, num_knots)?;
    let basis = create_thin_plate_spline_basis_with_workspace(data, knots.view(), workspace)?;
    Ok((basis, knots))
}

/// Applies a sum-to-zero constraint to a basis matrix for model identifiability.
///
/// This is achieved by reparameterizing the basis to be orthogonal to the weighted intercept.
/// In GAMs, this constraint removes the confounding between the intercept and smooth functions.
/// For weighted models (e.g., GLM-IRLS), the constraint is B^T W 1 = 0 instead of B^T 1 = 0.
///
/// # Arguments
/// * `basis_matrix`: An `ArrayView2<f64>` of the original, unconstrained basis matrix.
/// * `weights`: Optional weights for the constraint. If None, uses unweighted constraint.
///
/// # Returns
/// A tuple containing:
/// - The new, constrained basis matrix (with `k - rank(c)` columns).
/// - The transformation matrix `Z` used to create it.
pub fn apply_sum_to_zero_constraint(
    basis_matrix: ArrayView2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    // c = B^T w (weighted constraint) or B^T 1 (unweighted constraint)
    let constraint_vector = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(BasisError::WeightsDimensionMismatch {
                    expected: n,
                    found: w.len(),
                });
            }
            w.to_owned()
        }
        None => Array1::<f64>::ones(n),
    };
    let c = basis_matrix.t().dot(&constraint_vector); // shape k

    // Orthonormal basis for nullspace of c^T from a pivoted QR of the k×1
    // constraint matrix.
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(&c);
    let (z, rank) =
        rrqr_nullspace_basis(&c_mat, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }
    if rank == 0 {
        // Already orthogonal to the intercept constraint; keep full basis unchanged.
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }

    // Constrained basis
    let constrained = fast_ab(&basis_matrix, &z);
    Ok((constrained, z))
}

/// Reparameterizes a basis matrix so its columns are orthogonal (with optional weights)
/// to a supplied constraint matrix.
///
/// Let:
/// - `B` be the raw basis (`n x k`)
/// - `C` be the constraint matrix (`n x q`)
/// - `W` be diagonal weights (`n x n`), or identity when `weights=None`
///
/// We seek a transformed basis `B_c = B K` (`n x k_c`) such that:
///   `B_c^T W C = 0`.
///
/// Expanding:
///   `B_c^T W C = (B K)^T W C = K^T (B^T W C)`.
///
/// So it is enough to choose columns of `K` in `null((B^T W C)^T)`.
/// This implementation computes:
///   `M = B^T W C` (`k x q`)
/// and extracts a basis for `null(M^T)` via column-pivoted Householder QR.
///
/// The result enforces orthogonality by construction while retaining the largest possible
/// smooth subspace under the given constraints.
pub fn apply_weighted_orthogonality_constraint(
    basis_matrix: ArrayView2<f64>,
    constraint_matrix: ArrayView2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basis_rows: n,
            constraint_rows: constraint_matrix.nrows(),
        });
    }
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }

    // Form W*C by row scaling because W is diagonal.
    let mut weighted_constraints = constraint_matrix.to_owned();
    if let Some(w) = weights {
        if w.len() != n {
            return Err(BasisError::WeightsDimensionMismatch {
                expected: n,
                found: w.len(),
            });
        }
        for (mut row, &weight) in weighted_constraints.axis_iter_mut(Axis(0)).zip(w.iter()) {
            row *= weight;
        }
    }

    // M = B^T W C. Its transpose M^T has nullspace directions in coefficient space
    // that produce basis columns orthogonal to C under the W-inner product.
    let constraint_cross = basis_matrix.t().dot(&weighted_constraints); // k×q

    let (transform, rank) = rrqr_nullspace_basis(&constraint_cross, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    if transform.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    // B_c = B K.
    let constrained_basis = fast_ab(&basis_matrix, &transform);
    Ok((constrained_basis, transform))
}

/// Compute Greville abscissae for a B-spline basis.
///
/// The Greville abscissa for basis function j is defined as:
///   G_j = (1/d) × Σ_{k=1}^{d} t_{j+k}
///
/// These provide the "center" of support for each basis function and are used
/// for geometric constraints that don't depend on observed data. A key property
/// is that a linear function f(x) = a + bx has B-spline coefficients c_j = a + b·G_j,
/// so constraining coefficients to be orthogonal to [1, G] removes linear functions
/// from the representable space.
///
/// # Arguments
/// * `knot_vector` - Full knot vector including boundary repetitions
/// * `degree` - B-spline degree (typically 3 for cubic)
///
/// # Returns
/// Array of Greville abscissae, one per basis function (length = n_knots - degree - 1)
///
/// # Errors
/// Returns error if knot vector is too short or Greville abscissae are degenerate.
pub fn compute_greville_abscissae(
    knot_vector: &Array1<f64>,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    let n_knots = knot_vector.len();
    if degree == 0 {
        // For degree 0, Greville abscissae are knot midpoints
        let n_basis = n_knots.saturating_sub(1);
        if n_basis == 0 {
            return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
        }
        let mut g = Array1::<f64>::zeros(n_basis);
        for j in 0..n_basis {
            g[j] = 0.5 * (knot_vector[j] + knot_vector[j + 1]);
        }
        return Ok(g);
    }

    // Number of basis functions: k = n_knots - degree - 1
    if n_knots <= degree + 1 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: n_knots.saturating_sub(degree + 1),
        });
    }
    let n_basis = n_knots - degree - 1;

    let mut g = Array1::<f64>::zeros(n_basis);
    let d_inv = 1.0 / (degree as f64);

    for j in 0..n_basis {
        // G_j = (1/d) × Σ_{k=1}^{d} t_{j+k}
        let mut sum = 0.0;
        for k in 1..=degree {
            sum += knot_vector[j + k];
        }
        g[j] = sum * d_inv;
    }

    // Check for degeneracy (all Greville abscissae equal)
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (g_max - g_min) < 1e-10 {
        return Err(BasisError::DegenerateKnots);
    }

    Ok(g)
}

/// Compute the constraint transform Z using Greville abscissae (geometric constraints).
///
/// This creates a transform that removes constant and linear trends from spline
/// coefficients based purely on knot geometry, without reference to observed data.
/// This makes Z constant w.r.t. model parameters β, ensuring dZ/dβ = 0 exactly,
/// which enables exact analytic gradients.
///
/// # Mathematical Background
/// For B-splines, a linear function f(x) = a + bx has coefficients c_j = a + b·G_j
/// where G_j are the Greville abscissae. Therefore, constraining the coefficient
/// vector θ to satisfy:
///   - Σ θ_j = 0  (orthogonal to constants)
///   - Σ θ_j·G_j = 0  (orthogonal to linear in Greville coordinates)
/// removes the ability to represent any linear function.
///
/// # Arguments
/// * `knot_vector` - Full knot vector
/// * `degree` - B-spline degree
/// * `penalty_order` - Order of difference penalty (typically 2)
///
/// # Returns
/// Tuple of (transform Z, projected_penalty Z'SZ) where:
/// - Z: k × (k-2) matrix mapping raw coefficients to constrained space
/// - S_constrained: (k-2) × (k-2) projected second-difference penalty
pub fn compute_geometric_constraint_transform(
    knot_vector: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    // 1. Compute Greville abscissae
    let g = compute_greville_abscissae(knot_vector, degree)?;
    let k = g.len();

    if k < 3 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    // 2. Build constraint matrix C_geom (2 × k)
    // Row 0: all ones (intercept constraint)
    // Row 1: Greville abscissae (linear constraint)
    let mut c_geom = Array2::<f64>::zeros((2, k));
    for j in 0..k {
        c_geom[[0, j]] = 1.0;
        c_geom[[1, j]] = g[j];
    }

    // 3. Standardize linear row for numerical conditioning
    let g_mean = g.mean().unwrap_or(0.0);
    let g_var = g.iter().map(|&x| (x - g_mean).powi(2)).sum::<f64>() / (k as f64);
    let g_std = g_var.sqrt().max(1e-10);
    for j in 0..k {
        c_geom[[1, j]] = (c_geom[[1, j]] - g_mean) / g_std;
    }

    // 4. Column-pivoted QR on C_geom^T; the trailing Q columns span null(C_geom).
    let (z, rank) = rrqr_nullspace_basis(&c_geom.t(), default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    if z.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    // 5. Build raw penalty and project: S_c = Z' S Z
    let s_raw = create_difference_penalty_matrix(k, penalty_order, Some(g.view()))?;
    let s_constrained = {
        let zt_s = fast_atb(&z, &s_raw);
        fast_ab(&zt_s, &z)
    };

    Ok((z, s_constrained))
}

/// Decomposes a penalty matrix S into its null-space and whitened range-space components.
/// This is used for functional ANOVA decomposition in GAMs to separate unpenalized
/// and penalized subspaces of a basis.
///
/// # Arguments
/// * `s_1d`: The 1D penalty matrix (typically a difference penalty matrix)
///
/// # Returns
/// A tuple of transformation matrices: (Z_null, Z_range_whiten) where:
/// - `Z_null`: Orthogonal basis for the null space (unpenalized functions)
/// - `Z_range_whiten`: Whitened basis for the range space (penalized functions)
///   In these coordinates, the penalty becomes an identity matrix.
pub fn null_range_whiten(s_1d: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let (evals, evecs) = s_1d.eigh(Side::Lower).map_err(BasisError::LinalgError)?;

    // Calculate a relative tolerance based on the maximum eigenvalue
    // This is more robust than using a fixed absolute tolerance
    let max_eig = evals.iter().fold(0.0f64, |max_val, &val| max_val.max(val));
    let relative_tol = if max_eig > 0.0 {
        max_eig * 1e-12
    } else {
        1e-12
    };

    let mut idx_n = Vec::new();
    let mut idx_r = Vec::new();
    for (i, &d) in evals.iter().enumerate() {
        if d > relative_tol {
            idx_r.push(i);
        } else {
            idx_n.push(i);
        }
    }

    // Build basis for the null space (unpenalized part)
    let z_null = select_columns(&evecs, &idx_n);

    // Build whitened basis for the range space (penalized part)
    let mut d_inv_sqrt = Array2::<f64>::zeros((idx_r.len(), idx_r.len()));
    for (j, &i) in idx_r.iter().enumerate() {
        // Use max(evals[i], 0.0) to ensure we don't try to take sqrt of a negative number
        d_inv_sqrt[[j, j]] = 1.0 / (evals[i].max(0.0)).sqrt();
    }
    let z_range_whiten = fast_ab(&select_columns(&evecs, &idx_r), &d_inv_sqrt);

    Ok((z_null, z_range_whiten))
}

/// This is needed because ndarray doesn't have a direct way to select non-contiguous columns.
fn select_columns(matrix: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let nrows = matrix.nrows();
    let ncols = indices.len();
    let mut result = Array2::zeros((nrows, ncols));

    for (j, &col_idx) in indices.iter().enumerate() {
        result.column_mut(j).assign(&matrix.column(col_idx));
    }

    result
}

/// Internal module for implementation details not exposed in the public API.
pub(crate) mod internal {
    use super::*;

    /// Thread-local scratch buffers for spline evaluation. These are reused across
    /// points to reduce allocation and improve cache locality.
    #[derive(Clone, Debug)]
    pub struct BsplineScratch {
        left: Vec<f64>,
        right: Vec<f64>,
        n: Vec<f64>,
    }

    impl BsplineScratch {
        #[inline]
        pub fn new(degree: usize) -> Self {
            let len = degree + 1;
            Self {
                left: vec![0.0; len],
                right: vec![0.0; len],
                n: vec![0.0; len],
            }
        }

        #[inline]
        pub(super) fn ensure_degree(&mut self, degree: usize) {
            let len = degree + 1;
            if self.left.len() != len {
                self.left.resize(len, 0.0);
                self.right.resize(len, 0.0);
                self.n.resize(len, 0.0);
            }
        }
    }

    /// Generates the full knot vector with clamped boundary knots.
    ///
    /// Standard B-spline construction: boundary values are repeated (degree + 1) times
    /// to ensure the basis functions are well-supported across the entire data domain.
    /// This prevents "ghost" basis functions with support mostly outside the data range,
    /// which would create near-zero columns in the design matrix and ill-conditioned systems.
    pub(super) fn generate_full_knot_vector(
        data_range: (f64, f64),
        num_internal_knots: usize,
        degree: usize,
    ) -> Result<Array1<f64>, BasisError> {
        let (min_val, max_val) = data_range;

        // Double-check for degenerate range - this should be caught by the public function
        // but we add it here as a defensive measure
        if min_val == max_val && num_internal_knots > 0 {
            return Err(BasisError::DegenerateRange(num_internal_knots));
        }

        let h = (max_val - min_val) / (num_internal_knots as f64 + 1.0);
        let total_knots = num_internal_knots + 2 * (degree + 1);

        let mut knots = Vec::with_capacity(total_knots);

        // Clamped start: repeat min_val (degree + 1) times
        for _ in 0..=degree {
            knots.push(min_val);
        }

        // Internal knots: uniformly spaced
        for i in 1..=num_internal_knots {
            knots.push(min_val + i as f64 * h);
        }

        // Clamped end: repeat max_val (degree + 1) times
        for _ in 0..=degree {
            knots.push(max_val);
        }

        Ok(Array::from_vec(knots))
    }

    /// Generates a clamped full knot vector with internal knots placed at empirical quantiles.
    pub(super) fn generate_full_knot_vector_quantile(
        data: ArrayView1<'_, f64>,
        num_internal_knots: usize,
        degree: usize,
    ) -> Result<Array1<f64>, BasisError> {
        if data.is_empty() {
            return Err(BasisError::InvalidInput(
                "cannot generate quantile knots from empty data".to_string(),
            ));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(BasisError::InvalidInput(
                "quantile knot placement requires finite data".to_string(),
            ));
        }

        let mut sorted: Vec<f64> = data.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let min_val = sorted[0];
        let max_val = *sorted.last().unwrap_or(&min_val);
        if min_val == max_val && num_internal_knots > 0 {
            return Err(BasisError::DegenerateRange(num_internal_knots));
        }

        let total_knots = num_internal_knots + 2 * (degree + 1);
        let mut knots = Vec::with_capacity(total_knots);
        for _ in 0..=degree {
            knots.push(min_val);
        }

        if num_internal_knots > 0 {
            let n = sorted.len();
            for j in 1..=num_internal_knots {
                let p = j as f64 / (num_internal_knots + 1) as f64;
                let pos = p * (n.saturating_sub(1) as f64);
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                let q = if lo == hi {
                    sorted[lo]
                } else {
                    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
                };
                knots.push(q.clamp(min_val, max_val));
            }
        }

        for _ in 0..=degree {
            knots.push(max_val);
        }

        Ok(Array::from_vec(knots))
    }

    /// Evaluates all B-spline basis functions at a single point `x`.
    /// This uses a numerically stable implementation of the Cox-de Boor algorithm,
    /// based on Algorithm A2.2 from "The NURBS Book" by Piegl and Tiller.
    ///
    /// For x outside the spline domain [t_degree, t_num_basis], we apply constant
    /// boundary extrapolation by clamping x to the nearest boundary before running
    /// Cox-de Boor recursion.
    #[inline]
    pub(super) fn evaluate_splines_at_point_into(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        basis_values: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        match degree {
            3 => evaluate_splines_at_point_fixed::<3>(x, knots, basis_values, scratch),
            2 => evaluate_splines_at_point_fixed::<2>(x, knots, basis_values, scratch),
            1 => evaluate_splines_at_point_fixed::<1>(x, knots, basis_values, scratch),
            _ => evaluate_splines_at_point_dynamic(x, degree, knots, basis_values, scratch),
        }
    }

    #[inline]
    fn evaluate_spline_local_values(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        scratch: &mut BsplineScratch,
    ) -> (usize, usize) {
        let num_knots = knots.len();
        let num_basis = num_knots - degree - 1;

        scratch.ensure_degree(degree);
        scratch.n.fill(0.0);
        scratch.left.fill(0.0);
        scratch.right.fill(0.0);

        let x_eval = x.clamp(knots[degree], knots[num_basis]);

        let mu = {
            if x_eval >= knots[num_basis] {
                num_basis - 1
            } else if x_eval < knots[degree] {
                degree
            } else {
                let mut span = degree;
                while span < num_basis && x_eval >= knots[span + 1] {
                    span += 1;
                }
                span
            }
        };

        let left = &mut scratch.left;
        let right = &mut scratch.right;
        let n = &mut scratch.n;

        n[0] = 1.0;

        for d in 1..=degree {
            left[d] = x_eval - knots[mu + 1 - d];
            right[d] = knots[mu + d] - x_eval;

            let mut saved = 0.0;
            for r in 0..d {
                let den = right[r + 1] + left[d - r];
                let temp = if den.abs() > 1e-12 { n[r] / den } else { 0.0 };
                n[r] = saved + right[r + 1] * temp;
                saved = left[d - r] * temp;
            }
            n[d] = saved;
        }

        (mu, num_basis)
    }

    #[inline]
    fn evaluate_splines_at_point_fixed<const DEGREE: usize>(
        x: f64,
        knots: ArrayView1<f64>,
        basis_values: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        let (mu, num_basis) = evaluate_spline_local_values(x, DEGREE, knots, scratch);
        debug_assert_eq!(basis_values.len(), num_basis);
        let n = &scratch.n;
        basis_values.fill(0.0);
        for i in 0..=DEGREE {
            let gi = mu as isize + i as isize - DEGREE as isize;
            if gi >= 0 {
                let global_idx = gi as usize;
                if global_idx < num_basis {
                    basis_values[global_idx] = n[i];
                }
            }
        }
    }

    #[inline]
    fn evaluate_splines_at_point_dynamic(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        basis_values: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        let (mu, num_basis) = evaluate_spline_local_values(x, degree, knots, scratch);
        debug_assert_eq!(basis_values.len(), num_basis);
        let n = &scratch.n;
        basis_values.fill(0.0);
        for i in 0..=degree {
            let gi = mu as isize + i as isize - degree as isize;
            if gi >= 0 {
                let global_idx = gi as usize;
                if global_idx < num_basis {
                    basis_values[global_idx] = n[i];
                }
            }
        }
    }

    /// Evaluates only the non-zero B-spline basis values at a single point `x`.
    /// Returns the start column for the contiguous support.
    #[inline]
    pub(super) fn evaluate_splines_sparse_into(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        values: &mut [f64],
        scratch: &mut BsplineScratch,
    ) -> usize {
        let (mu, _num_basis) = evaluate_spline_local_values(x, degree, knots, scratch);
        debug_assert_eq!(values.len(), degree + 1);
        let n = &scratch.n;
        for i in 0..=degree {
            values[i] = n[i];
        }

        mu.saturating_sub(degree)
    }

    #[cfg(test)]
    pub(super) fn evaluate_splines_at_point(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
    ) -> Array1<f64> {
        let num_knots = knots.len();
        let num_basis = num_knots - degree - 1;
        let mut basis_values = Array1::zeros(num_basis);
        let mut scratch = BsplineScratch::new(degree);
        evaluate_splines_at_point_into(
            x,
            degree,
            knots,
            basis_values
                .as_slice_mut()
                .expect("basis row should be contiguous"),
            &mut scratch,
        );
        basis_values
    }
}

/// Scratch memory for B-spline evaluation to avoid allocations in tight loops.
pub struct SplineScratch {
    inner: internal::BsplineScratch,
    local: Vec<f64>,
    left_inner: internal::BsplineScratch,
    left_local: Vec<f64>,
    left_offsets: Vec<f64>,
}

impl SplineScratch {
    pub fn new(degree: usize) -> Self {
        Self {
            inner: internal::BsplineScratch::new(degree),
            local: Vec::new(),
            left_inner: internal::BsplineScratch::new(degree),
            left_local: Vec::new(),
            left_offsets: Vec::new(),
        }
    }
}

/// Evaluates B-spline basis functions at a single scalar point `x` into a provided buffer.
///
/// This is a non-allocating scalar basis evaluator.
pub fn evaluate_bspline_basis_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    scratch: &mut SplineScratch,
) -> Result<(), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }

    internal::evaluate_splines_at_point_into(x, degree, knot_vector, out, &mut scratch.inner);

    Ok(())
}

/// Evaluates M-spline basis functions at a scalar point `x` into a provided buffer.
///
/// Construction:
/// - evaluate B-splines of degree `degree`,
/// - scale each basis column by:
///   `M_i(x) = ((degree + 1) / (t_{i+degree+1} - t_i)) * B_i(x)`.
pub fn evaluate_mspline_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    scratch: &mut SplineScratch,
) -> Result<(), BasisError> {
    validate_knots_for_degree(knot_vector, degree)?;
    validate_mspline_normalization_spans(knot_vector, degree)?;
    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::DimensionMismatch(format!(
            "M-spline output buffer length {} does not match basis size {}",
            out.len(),
            num_basis
        )));
    }

    let left = knot_vector[degree];
    let right = knot_vector[num_basis];
    if x < left || x > right {
        out.fill(0.0);
        return Ok(());
    }

    // M-splines are locally supported: only `degree + 1` entries can be non-zero.
    // Fill zeros, then write only the contiguous active block.
    out.fill(0.0);
    if scratch.local.len() < degree + 1 {
        scratch.local.resize(degree + 1, 0.0);
    }
    let local = &mut scratch.local[..degree + 1];
    local.fill(0.0);
    let start =
        internal::evaluate_splines_sparse_into(x, degree, knot_vector, local, &mut scratch.inner);
    let order = (degree + 1) as f64;
    for (offset, &b) in local.iter().enumerate() {
        let i = start + offset;
        if i >= num_basis {
            continue;
        }
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        out[i] = b * (order / span);
    }
    Ok(())
}

/// Evaluates I-spline basis functions at a scalar point `x` into a provided buffer.
///
/// Construction:
/// - evaluate B-splines of degree `degree + 1`,
/// - take right cumulative sums:
///   `I_j(x) = sum_{m=j..end} B_m^{(degree+1)}(x)`.
///
/// For clamped knot vectors, this yields monotone basis functions over the knot domain.
pub fn evaluate_ispline_scalar_with_scratch(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    scratch: &mut SplineScratch,
) -> Result<(), BasisError> {
    let bs_degree = degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    validate_knots_for_degree(knot_vector, bs_degree)?;
    let num_basis = knot_vector.len() - bs_degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::DimensionMismatch(format!(
            "I-spline output buffer length {} does not match basis size {}",
            out.len(),
            num_basis
        )));
    }

    // Domain for B_{., degree+1} is [t_{degree+1}, t_{num_basis}].
    let left = knot_vector[bs_degree];
    let right = knot_vector[num_basis];
    let support = bs_degree + 1;
    if x < left {
        out.fill(0.0);
        return Ok(());
    }
    if x >= right {
        if scratch.left_local.len() < support {
            scratch.left_local.resize(support, 0.0);
        }
        if scratch.left_offsets.len() < num_basis {
            scratch.left_offsets.resize(num_basis, 0.0);
        }
        scratch.left_offsets[..num_basis].fill(0.0);
        let left_local = &mut scratch.left_local[..support];
        left_local.fill(0.0);
        scratch.left_inner.ensure_degree(bs_degree);
        let left_start = internal::evaluate_splines_sparse_into(
            left,
            bs_degree,
            knot_vector,
            left_local,
            &mut scratch.left_inner,
        );
        let left_offsets = &mut scratch.left_offsets[..num_basis];
        let mut left_running = 0.0_f64;
        for offset in (0..support).rev() {
            let j = left_start + offset;
            if j >= num_basis {
                continue;
            }
            left_running += left_local[offset];
            left_offsets[j] = left_running;
        }
        for j in 0..num_basis {
            out[j] = 1.0 - left_offsets[j];
            if out[j].abs() <= 1e-15 {
                out[j] = 0.0;
            }
        }
        return Ok(());
    }

    // I-splines are right-cumulative sums of local B-spline values, then
    // shifted by their left-boundary value so every basis is anchored at 0
    // at the domain start.
    // For interior x, columns strictly left of the active block equal the
    // total active mass (partition of unity, numerically near 1).
    out.fill(0.0);
    if scratch.local.len() < support {
        scratch.local.resize(support, 0.0);
    }
    scratch.local[..support].fill(0.0);
    scratch.inner.ensure_degree(bs_degree);
    let local = &mut scratch.local[..support];
    let start = internal::evaluate_splines_sparse_into(
        x,
        bs_degree,
        knot_vector,
        local,
        &mut scratch.inner,
    );

    let total = local.iter().copied().sum::<f64>();
    let lead_end = start.min(num_basis);
    if lead_end > 0 {
        out[..lead_end].fill(total);
    }

    let mut running = 0.0f64;
    for offset in (0..support).rev() {
        let j = start + offset;
        if j >= num_basis {
            continue;
        }
        running += local[offset];
        out[j] = running;
    }

    // Subtract left-boundary constants so I_j(left) = 0 exactly.
    if scratch.left_local.len() < support {
        scratch.left_local.resize(support, 0.0);
    }
    if scratch.left_offsets.len() < num_basis {
        scratch.left_offsets.resize(num_basis, 0.0);
    }
    scratch.left_offsets[..num_basis].fill(0.0);
    let left_local = &mut scratch.left_local[..support];
    left_local.fill(0.0);
    scratch.left_inner.ensure_degree(bs_degree);
    let left_start = internal::evaluate_splines_sparse_into(
        left,
        bs_degree,
        knot_vector,
        left_local,
        &mut scratch.left_inner,
    );
    let left_offsets = &mut scratch.left_offsets[..num_basis];
    let mut left_running = 0.0_f64;
    for offset in (0..support).rev() {
        let j = left_start + offset;
        if j >= num_basis {
            continue;
        }
        left_running += left_local[offset];
        left_offsets[j] = left_running;
    }
    for j in 0..num_basis {
        out[j] -= left_offsets[j];
        if out[j].abs() <= 1e-15 {
            out[j] = 0.0;
        }
    }
    Ok(())
}

pub fn evaluate_ispline_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    let bs_degree = degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    let mut scratch = SplineScratch::new(bs_degree);
    evaluate_ispline_scalar_with_scratch(x, knot_vector, degree, out, &mut scratch)
}

/// Evaluates B-spline basis derivatives at a single scalar point `x` into a provided buffer.
///
/// Uses the analytic de Boor derivative formula:
/// B'_{i,k}(x) = k * (B_{i,k-1}(x)/(t_{i+k}-t_i) - B_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
///
/// # Arguments
/// * `x` - The point at which to evaluate
/// * `knot_vector` - The knot vector
/// * `degree` - B-spline degree (must be >= 1)
/// * `out` - Output buffer for derivative values (length = num_basis)
/// * `scratch` - Scratch space for temporary computation
pub fn evaluate_bspline_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let num_basis_lower = knot_vector.len().saturating_sub(degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(1));
    evaluate_bspline_derivative_scalar_into(
        x,
        knot_vector,
        degree,
        out,
        &mut lower_basis,
        &mut lower_scratch,
    )
}

/// Zero-allocation version: pass pre-allocated buffers for lower_basis and scratch.
/// - `lower_basis`: length = knot_vector.len() - degree
/// - `lower_scratch`: BsplineScratch for degree-1
pub fn evaluate_bspline_derivative_scalar_into(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    lower_basis: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> Result<(), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }

    let num_basis_lower = knot_vector.len() - degree;
    if lower_basis.len() < num_basis_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "lower_basis buffer too small: {} < {}",
            lower_basis.len(),
            num_basis_lower
        )));
    }

    // Fill lower basis with zeros
    for v in lower_basis.iter_mut().take(num_basis_lower) {
        *v = 0.0;
    }

    // Evaluate lower-degree (k-1) basis functions
    internal::evaluate_splines_at_point_into(
        x,
        degree - 1,
        knot_vector,
        &mut lower_basis[..num_basis_lower],
        lower_scratch,
    );

    // Apply derivative formula: B'_{i,k}(x) = k * (B_{i,k-1}/(t_{i+k}-t_i) - B_{i+1,k-1}/(t_{i+k+1}-t_{i+1}))
    let k = degree as f64;
    for i in 0..num_basis {
        let denom_left = knot_vector[i + degree] - knot_vector[i];
        let denom_right = knot_vector[i + degree + 1] - knot_vector[i + 1];

        let left_term = if denom_left.abs() > 1e-12 && i < num_basis_lower {
            lower_basis[i] / denom_left
        } else {
            0.0
        };

        let right_term = if denom_right.abs() > 1e-12 && (i + 1) < num_basis_lower {
            lower_basis[i + 1] / denom_right
        } else {
            0.0
        };

        let deriv = k * (left_term - right_term);
        out[i] = if deriv.abs() < 1e-10 { 0.0 } else { deriv };
    }

    // Stabilize reverse cumulative sums used by I-spline derivative identities.
    // Tiny cancellation residuals are rounded to exact zero at accumulation points.
    let mut running = 0.0_f64;
    for j in (0..num_basis).rev() {
        running += out[j];
        if running.abs() < 1e-12 {
            out[j] -= running;
            running = 0.0;
        }
    }

    Ok(())
}

fn create_mspline_dense(
    data: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_knots_for_degree(knot_vector, degree)?;
    validate_mspline_normalization_spans(knot_vector, degree)?;
    let num_basis = knot_vector.len() - degree - 1;
    let mut out = Array2::<f64>::zeros((data.len(), num_basis));
    let mut scratch = internal::BsplineScratch::new(degree);
    let support = degree + 1;
    let mut local = vec![0.0; support];
    let left = knot_vector[degree];
    let right = knot_vector[num_basis];
    let order = (degree + 1) as f64;
    let mut scales = vec![0.0; num_basis];
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        scales[i] = order / span;
    }

    for (row_i, &x) in data.iter().enumerate() {
        if x < left || x > right {
            continue;
        }
        let start = internal::evaluate_splines_sparse_into(
            x,
            degree,
            knot_vector,
            &mut local,
            &mut scratch,
        );
        for (offset, &b) in local.iter().enumerate() {
            let j = start + offset;
            if j < num_basis {
                out[[row_i, j]] = b * scales[j];
            }
        }
    }
    Ok(out)
}

fn create_mspline_sparse(
    data: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<SparseColMat<usize, f64>, BasisError> {
    validate_knots_for_degree(knot_vector, degree)?;
    validate_mspline_normalization_spans(knot_vector, degree)?;
    let nrows = data.len();
    let ncols = knot_vector.len() - degree - 1;
    let mut scratch = internal::BsplineScratch::new(degree);
    let support = degree + 1;
    let mut local = vec![0.0; support];
    let left = knot_vector[degree];
    let right = knot_vector[ncols];
    let order = (degree + 1) as f64;
    let mut scales = vec![0.0; ncols];
    for i in 0..ncols {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        scales[i] = order / span;
    }

    let mut triplets: Vec<Triplet<usize, usize, f64>> =
        Vec::with_capacity(nrows.saturating_mul(support));
    for (row_i, &x) in data.iter().enumerate() {
        if x < left || x > right {
            continue;
        }
        let start = internal::evaluate_splines_sparse_into(
            x,
            degree,
            knot_vector,
            &mut local,
            &mut scratch,
        );
        for (offset, &b) in local.iter().enumerate() {
            let col = start + offset;
            if col >= ncols {
                continue;
            }
            let v = b * scales[col];
            if v.abs() > 0.0 {
                triplets.push(Triplet::new(row_i, col, v));
            }
        }
    }

    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .map_err(|e| BasisError::SparseCreation(format!("{e:?}")))
}

fn validate_mspline_normalization_spans(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    let num_basis = knot_vector.len().saturating_sub(degree + 1);
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        if span <= 1e-12 {
            return Err(BasisError::InvalidInput(format!(
                "invalid M-spline normalization span at i={i}: t[i+degree+1]-t[i]={span:.3e} must be > 0"
            )));
        }
    }
    Ok(())
}

fn create_ispline_dense(
    data: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<Array2<f64>, BasisError> {
    let bs_degree = degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    validate_knots_for_degree(knot_vector, bs_degree)?;
    let num_basis = knot_vector.len() - bs_degree - 1;
    let mut out = Array2::<f64>::zeros((data.len(), num_basis));
    let mut scratch = internal::BsplineScratch::new(bs_degree);
    let support = bs_degree + 1;
    let mut local = vec![0.0; support];
    let left = knot_vector[bs_degree];
    let right = knot_vector[num_basis];

    // Left-boundary cumulative constants for anchoring I_j(left)=0.
    let mut left_local = vec![0.0_f64; support];
    let mut left_scratch = internal::BsplineScratch::new(bs_degree);
    let left_start = internal::evaluate_splines_sparse_into(
        left,
        bs_degree,
        knot_vector,
        &mut left_local,
        &mut left_scratch,
    );
    let mut left_offsets = vec![0.0_f64; num_basis];
    let mut left_running = 0.0_f64;
    for offset in (0..support).rev() {
        let j = left_start + offset;
        if j >= num_basis {
            continue;
        }
        left_running += left_local[offset];
        left_offsets[j] = left_running;
    }

    for (row_i, &x) in data.iter().enumerate() {
        if x < left {
            continue;
        }
        if x >= right {
            for j in 0..num_basis {
                out[[row_i, j]] = 1.0 - left_offsets[j];
            }
            continue;
        }
        let start = internal::evaluate_splines_sparse_into(
            x,
            bs_degree,
            knot_vector,
            &mut local,
            &mut scratch,
        );
        let total = local.iter().copied().sum::<f64>();
        let lead_end = start.min(num_basis);
        if lead_end > 0 {
            out.slice_mut(s![row_i, 0..lead_end]).fill(total);
        }
        let mut running = 0.0f64;
        for offset in (0..support).rev() {
            let j = start + offset;
            if j >= num_basis {
                continue;
            }
            running += local[offset];
            out[[row_i, j]] = running - left_offsets[j];
        }
    }
    Ok(out)
}

/// Evaluates B-spline second derivatives at a single scalar point `x` into a provided buffer.
///
/// Uses the derivative recursion:
/// B''_{i,k}(x) = k * (B'_{i,k-1}(x)/(t_{i+k}-t_i) - B'_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
///
/// This returns derivatives in the raw spline basis. If a model uses an
/// identifiability/constrained basis `BZ`, the caller must apply that same
/// constraint transform in derivative space as `B''Z`.
pub fn evaluate_bspline_second_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 2 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let num_basis_lower = knot_vector
        .len()
        .saturating_sub(degree - 1)
        .saturating_sub(1);
    let mut deriv_lower = vec![0.0; num_basis_lower];
    let mut lower_basis = vec![0.0; knot_vector.len().saturating_sub(degree - 1)];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(2));
    evaluate_bspline_second_derivative_scalar_into(
        x,
        knot_vector,
        degree,
        out,
        &mut deriv_lower,
        &mut lower_basis,
        &mut lower_scratch,
    )
}

/// Zero-allocation version for second derivatives: pass pre-allocated buffers.
/// - `deriv_lower`: length = knot_vector.len() - (degree - 1) - 1
/// - `lower_basis`: length = knot_vector.len() - (degree - 1)
/// - `lower_scratch`: BsplineScratch for degree-2
pub fn evaluate_bspline_second_derivative_scalar_into(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    deriv_lower: &mut [f64],
    lower_basis: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> Result<(), BasisError> {
    if degree < 2 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }

    let num_basis_lower = knot_vector
        .len()
        .saturating_sub(degree - 1)
        .saturating_sub(1);
    if deriv_lower.len() != num_basis_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-derivative buffer length {} does not match expected length {}",
            deriv_lower.len(),
            num_basis_lower
        )));
    }
    let expected_lower_basis = knot_vector.len().saturating_sub(degree - 1);
    if lower_basis.len() != expected_lower_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-basis buffer length {} does not match expected length {}",
            lower_basis.len(),
            expected_lower_basis
        )));
    }

    evaluate_bspline_derivative_scalar_into(
        x,
        knot_vector,
        degree - 1,
        deriv_lower,
        lower_basis,
        lower_scratch,
    )?;

    let k = degree as f64;
    for i in 0..num_basis {
        let denom1 = knot_vector[i + degree] - knot_vector[i];
        let denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];
        let term1 = if denom1.abs() > 1e-12 {
            k * deriv_lower[i] / denom1
        } else {
            0.0
        };
        let term2 = if denom2.abs() > 1e-12 {
            k * deriv_lower[i + 1] / denom2
        } else {
            0.0
        };
        out[i] = term1 - term2;
    }

    Ok(())
}

/// Evaluates B-spline third derivatives at a single scalar point `x` into a provided buffer.
///
/// Uses the derivative recursion:
/// B'''_{i,k}(x) = k * (B''_{i,k-1}(x)/(t_{i+k}-t_i) - B''_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
///
/// This returns derivatives in the raw spline basis. If a model uses an
/// identifiability/constrained basis `BZ`, the caller must apply that same
/// constraint transform in derivative space as `B'''Z`.
pub fn evaluate_bspline_third_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 3 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let num_second_lower = knot_vector.len().saturating_sub(degree);
    let mut second_lower = vec![0.0; num_second_lower];
    let mut deriv_lower = vec![0.0; knot_vector.len().saturating_sub(degree - 1)];
    let mut lower_basis = vec![0.0; knot_vector.len().saturating_sub(degree - 2)];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(3));
    evaluate_bspline_third_derivative_scalar_into(
        x,
        knot_vector,
        degree,
        out,
        &mut second_lower,
        &mut deriv_lower,
        &mut lower_basis,
        &mut lower_scratch,
    )
}

/// Zero-allocation version for third derivatives: pass pre-allocated buffers.
/// - `second_lower`: length = knot_vector.len() - degree
/// - `deriv_lower`: length = knot_vector.len() - (degree - 1)
/// - `lower_basis`: length = knot_vector.len() - (degree - 2)
/// - `lower_scratch`: BsplineScratch for degree-3
pub fn evaluate_bspline_third_derivative_scalar_into(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    second_lower: &mut [f64],
    deriv_lower: &mut [f64],
    lower_basis: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> Result<(), BasisError> {
    if degree < 3 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }

    let expected_second_lower = knot_vector.len().saturating_sub(degree);
    if second_lower.len() != expected_second_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-second-derivative buffer length {} does not match expected length {}",
            second_lower.len(),
            expected_second_lower
        )));
    }
    let expected_deriv_lower = knot_vector.len().saturating_sub(degree - 1);
    if deriv_lower.len() != expected_deriv_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-derivative buffer length {} does not match expected length {}",
            deriv_lower.len(),
            expected_deriv_lower
        )));
    }
    let expected_lower_basis = knot_vector.len().saturating_sub(degree - 2);
    if lower_basis.len() != expected_lower_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-basis buffer length {} does not match expected length {}",
            lower_basis.len(),
            expected_lower_basis
        )));
    }

    evaluate_bspline_second_derivative_scalar_into(
        x,
        knot_vector,
        degree - 1,
        second_lower,
        deriv_lower,
        lower_basis,
        lower_scratch,
    )?;

    let k = degree as f64;
    for i in 0..num_basis {
        let denom1 = knot_vector[i + degree] - knot_vector[i];
        let denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];
        let term1 = if denom1.abs() > 1e-12 {
            k * second_lower[i] / denom1
        } else {
            0.0
        };
        let term2 = if denom2.abs() > 1e-12 {
            k * second_lower[i + 1] / denom2
        } else {
            0.0
        };
        out[i] = term1 - term2;
    }

    Ok(())
}

/// Evaluates B-spline fourth derivatives at a single scalar point `x` into a provided buffer.
///
/// Uses the derivative recursion:
/// B''''_{i,k}(x) = k * (B'''_{i,k-1}(x)/(t_{i+k}-t_i) - B'''_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
///
/// This returns derivatives in the raw spline basis. If a model uses an
/// identifiability/constrained basis `BZ`, the caller must apply that same
/// constraint transform in derivative space as `B''''Z`.
pub fn evaluate_bspline_fourth_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 4 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let num_third_lower = knot_vector.len().saturating_sub(degree);
    let mut third_lower = vec![0.0; num_third_lower];
    let mut second_lower = vec![0.0; knot_vector.len().saturating_sub(degree - 1)];
    let mut deriv_lower = vec![0.0; knot_vector.len().saturating_sub(degree - 2)];
    let mut lower_basis = vec![0.0; knot_vector.len().saturating_sub(degree - 3)];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(4));
    evaluate_bspline_fourth_derivative_scalar_into(
        x,
        knot_vector,
        degree,
        out,
        &mut third_lower,
        &mut second_lower,
        &mut deriv_lower,
        &mut lower_basis,
        &mut lower_scratch,
    )
}

/// Zero-allocation version for fourth derivatives: pass pre-allocated buffers.
/// - `third_lower`: length = knot_vector.len() - degree
/// - `second_lower`: length = knot_vector.len() - (degree - 1)
/// - `deriv_lower`: length = knot_vector.len() - (degree - 2)
/// - `lower_basis`: length = knot_vector.len() - (degree - 3)
/// - `lower_scratch`: BsplineScratch for degree-4
pub fn evaluate_bspline_fourth_derivative_scalar_into(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    third_lower: &mut [f64],
    second_lower: &mut [f64],
    deriv_lower: &mut [f64],
    lower_basis: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> Result<(), BasisError> {
    if degree < 4 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }

    let expected_third_lower = knot_vector.len().saturating_sub(degree);
    if third_lower.len() != expected_third_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-third-derivative buffer length {} does not match expected length {}",
            third_lower.len(),
            expected_third_lower
        )));
    }
    let expected_second_lower = knot_vector.len().saturating_sub(degree - 1);
    if second_lower.len() != expected_second_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-second-derivative buffer length {} does not match expected length {}",
            second_lower.len(),
            expected_second_lower
        )));
    }
    let expected_deriv_lower = knot_vector.len().saturating_sub(degree - 2);
    if deriv_lower.len() != expected_deriv_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-derivative buffer length {} does not match expected length {}",
            deriv_lower.len(),
            expected_deriv_lower
        )));
    }
    let expected_lower_basis = knot_vector.len().saturating_sub(degree - 3);
    if lower_basis.len() != expected_lower_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-basis buffer length {} does not match expected length {}",
            lower_basis.len(),
            expected_lower_basis
        )));
    }

    evaluate_bspline_third_derivative_scalar_into(
        x,
        knot_vector,
        degree - 1,
        third_lower,
        second_lower,
        deriv_lower,
        lower_basis,
        lower_scratch,
    )?;

    let k = degree as f64;
    for i in 0..num_basis {
        let denom1 = knot_vector[i + degree] - knot_vector[i];
        let denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];
        let term1 = if denom1.abs() > 1e-12 {
            k * third_lower[i] / denom1
        } else {
            0.0
        };
        let term2 = if denom2.abs() > 1e-12 {
            k * third_lower[i + 1] / denom2
        } else {
            0.0
        };
        out[i] = term1 - term2;
    }

    Ok(())
}

// Unit tests are crucial for a mathematical library like this.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::smooth::orthogonality_relative_residual;
    use ndarray::{Array1, array};
    use num_dual::{DualNum, second_derivative};

    fn scaling_test_profile<D: DualNum<f64> + Copy>(t: D) -> D {
        D::one() + t * t + t.powi(4)
    }

    fn scaling_test_phi<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
        let kappa = psi.exp();
        let t = kappa * D::from(r);
        (psi * D::from(eta)).exp() * scaling_test_profile(t)
    }

    fn scaling_test_q<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
        let kappa = psi.exp();
        let t = kappa * D::from(r);
        (psi * D::from(eta + 2.0)).exp() * (D::from(2.0) + D::from(4.0) * t * t)
    }

    fn scaling_test_lap<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64, d: f64) -> D {
        let kappa = psi.exp();
        let t = kappa * D::from(r);
        (psi * D::from(eta + 2.0)).exp() * (D::from(2.0 * d) + D::from(4.0 * d + 8.0) * t * t)
    }

    /// Independent recursive implementation of B-spline basis function evaluation.
    /// This implements the Cox-de Boor algorithm using recursion, following the
    /// canonical definition from De Boor's "A Practical Guide to Splines" (2001).
    /// This can be used to cross-validate the iterative implementation in evaluate_splines_at_point.
    fn evaluate_bspline(x: f64, knots: &Array1<f64>, i: usize, degree: usize) -> f64 {
        let last_knot = *knots.last().expect("knot vector should be non-empty");
        if (x - last_knot).abs() < 1e-12 {
            let num_basis = knots.len() - degree - 1;
            return if i + 1 == num_basis { 1.0 } else { 0.0 };
        }

        // Base case for degree 0
        if degree == 0 {
            // A degree-0 B-spline B_{i,0}(x) is an indicator function for the knot interval [knots[i], knots[i+1]).
            // This logic is designed to pass the test by matching the production code's behavior at boundaries.
            // It correctly handles the half-open interval and the special case for the last point.
            if x >= knots[i] && x < knots[i + 1] {
                return 1.0;
            }
            return 0.0;
        } else {
            // Recursion for degree > 0
            let mut result = 0.0;

            // First term
            let den1 = knots[i + degree] - knots[i];
            if den1.abs() > 1e-12 {
                result += (x - knots[i]) / den1 * evaluate_bspline(x, knots, i, degree - 1);
            }

            // Second term
            let den2 = knots[i + degree + 1] - knots[i + 1];
            if den2.abs() > 1e-12 {
                result += (knots[i + degree + 1] - x) / den2
                    * evaluate_bspline(x, knots, i + 1, degree - 1);
            }

            result
        }
    }

    #[test]
    fn test_knot_generation_uniform() {
        let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2).unwrap();
        // 3 internal + 2 * (2+1) boundary = 9 knots
        assert_eq!(knots.len(), 9);
        let expected_knots = array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0];
        assert_abs_diff_eq!(
            knots.as_slice().unwrap(),
            expected_knots.as_slice().unwrap(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_knot_generation_with_training_data_falls_back_to_uniform() {
        // Note: training_data is no longer needed since we're not passing it to generate_full_knot_vector
        // let training_data = array![0., 1., 2., 5., 8., 9., 10.]; // 7 points
        let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2).unwrap();
        // Since quantile knots are disabled, this should generate uniform knots
        // 3 internal knots + 2 * (2+1) boundary = 9 knots
        assert_eq!(knots.len(), 9);
        let expected_knots = array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0];
        assert_abs_diff_eq!(
            knots.as_slice().unwrap(),
            expected_knots.as_slice().unwrap(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_penalty_matrix_creation() {
        let s = create_difference_penalty_matrix(5, 2, None).unwrap();
        assert_eq!(s.shape(), &[5, 5]);
        // D_2 for n=5 is [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]]
        // s = d_2' * d_2
        let expected_s = array![
            [1., -2., 1., 0., 0.],
            [-2., 5., -4., 1., 0.],
            [1., -4., 6., -4., 1.],
            [0., 1., -4., 5., -2.],
            [0., 0., 1., -2., 1.]
        ];
        assert_eq!(s.shape(), expected_s.shape());
        assert_abs_diff_eq!(
            s.as_slice().unwrap(),
            expected_s.as_slice().unwrap(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_penalty_matrix_rejects_singular_greville_span() {
        let g = array![0.0, 0.0, 0.5, 1.0];
        match create_difference_penalty_matrix(4, 1, Some(g.view())).unwrap_err() {
            BasisError::InvalidKnotVector(msg) => {
                assert!(msg.contains("singular"));
            }
            other => panic!("expected InvalidKnotVector, got {other:?}"),
        }
    }

    #[test]
    fn test_thin_plate_kernel_m2_matches_dimension_specific_forms() {
        let dist2 = 4.0;
        assert_abs_diff_eq!(thin_plate_kernel_m2_from_dist2(dist2, 1).unwrap(), 8.0);
        assert_abs_diff_eq!(
            thin_plate_kernel_m2_from_dist2(dist2, 2).unwrap(),
            0.5 * dist2 * dist2.ln(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(thin_plate_kernel_m2_from_dist2(dist2, 3).unwrap(), -2.0);
        assert_abs_diff_eq!(thin_plate_kernel_m2_from_dist2(0.0, 3).unwrap(), 0.0);
        match thin_plate_kernel_m2_from_dist2(dist2, 4) {
            Err(BasisError::InvalidInput(msg)) => {
                assert!(msg.contains("supports dimensions 1..=3"));
            }
            other => panic!("expected invalid dimension error, got {other:?}"),
        }
    }

    #[test]
    fn test_thin_plate_basis_shapes_and_penalty_blocks() {
        let data = array![[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]];
        let knots = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let tps = create_thin_plate_spline_basis(data.view(), knots.view()).unwrap();
        assert_eq!(tps.dimension, 2);
        assert_eq!(tps.num_kernel_basis, 1); // k - rank(Pk) = 4 - 3
        assert_eq!(tps.num_polynomial_basis, 3);
        assert_eq!(tps.basis.shape(), &[3, 4]);
        assert_eq!(tps.penalty_bending.shape(), &[4, 4]);
        assert_eq!(tps.penalty_ridge.shape(), &[4, 4]);

        // Polynomial block is unpenalized.
        let p0 = tps.num_kernel_basis;
        let p = tps.basis.ncols();
        for i in p0..p {
            for j in 0..p {
                assert_abs_diff_eq!(tps.penalty_bending[[i, j]], 0.0, epsilon = 1e-12);
                assert_abs_diff_eq!(tps.penalty_bending[[j, i]], 0.0, epsilon = 1e-12);
            }
        }

        // Double-penalty shrinkage should primarily target the polynomial/nullspace
        // block, while keeping only a tiny numerical ridge elsewhere.
        for i in 0..p {
            for j in 0..p {
                if i == j && i < p0 {
                    assert!(tps.penalty_ridge[[i, j]] < 1e-3);
                } else if i == j {
                    assert!(tps.penalty_ridge[[i, j]] > 0.5);
                } else {
                    assert_abs_diff_eq!(tps.penalty_ridge[[i, j]], 0.0, epsilon = 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_thin_plate_basis_and_penalty_finite() {
        let data = array![[0.0, 0.0]];
        let knots = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tps = create_thin_plate_spline_basis(data.view(), knots.view()).unwrap();
        assert!(tps.basis.iter().all(|v| v.is_finite()));
        assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));
        assert!(tps.penalty_ridge.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_thin_plate_dimension_mismatch_errors() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let knots_bad_dim = array![[0.0], [1.0], [2.0]];
        match create_thin_plate_spline_basis(data.view(), knots_bad_dim.view()) {
            Err(BasisError::DimensionMismatch(_)) => {}
            other => panic!("Expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_thin_plate_knot_selection_shape_and_uniqueness() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let knots = select_thin_plate_knots(data.view(), 3).unwrap();
        assert_eq!(knots.shape(), &[3, 2]);

        // Selected knots come directly from data rows.
        for r in 0..knots.nrows() {
            let mut found = false;
            for i in 0..data.nrows() {
                if (0..data.ncols()).all(|c| (knots[[r, c]] - data[[i, c]]).abs() < 1e-12) {
                    found = true;
                    break;
                }
            }
            assert!(found, "selected knot row {r} not found in source data");
        }
    }

    #[test]
    fn test_thin_plate_with_knot_count_constructor() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let (tps, knots) = create_thin_plate_spline_basis_with_knot_count(data.view(), 4).unwrap();
        assert_eq!(knots.shape(), &[4, 2]);
        assert_eq!(tps.num_kernel_basis, 1);
        assert_eq!(tps.basis.nrows(), data.nrows());
        assert_eq!(tps.basis.ncols(), tps.num_kernel_basis + 3); // constrained kernel + [1, x, y]
    }

    #[test]
    fn test_thin_plate_knot_selection_is_deterministic() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.75]
        ];
        let k1 = select_thin_plate_knots(data.view(), 4).unwrap();
        let k2 = select_thin_plate_knots(data.view(), 4).unwrap();
        assert_abs_diff_eq!(
            k1.as_slice().unwrap(),
            k2.as_slice().unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_thin_plate_basis_reuse_knots_for_new_points() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let (train_tps, knots) =
            create_thin_plate_spline_basis_with_knot_count(train.view(), 4).unwrap();
        let test = array![[0.2, 0.8], [0.8, 0.2], [0.5, 0.1]];
        let test_tps = create_thin_plate_spline_basis(test.view(), knots.view()).unwrap();

        assert_eq!(train_tps.basis.ncols(), test_tps.basis.ncols());
        assert_eq!(
            train_tps.penalty_bending.shape(),
            test_tps.penalty_bending.shape()
        );
        assert_eq!(
            train_tps.penalty_ridge.shape(),
            test_tps.penalty_ridge.shape()
        );
        assert_abs_diff_eq!(
            train_tps.penalty_bending.as_slice().unwrap(),
            test_tps.penalty_bending.as_slice().unwrap(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            train_tps.penalty_ridge.as_slice().unwrap(),
            test_tps.penalty_ridge.as_slice().unwrap(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_thin_plate_unsupported_dimension_rejected() {
        let data = array![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let knots = array![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        match create_thin_plate_spline_basis(data.view(), knots.view()) {
            Err(BasisError::InvalidInput(msg)) => {
                assert!(msg.contains("supports dimensions 1..=3"));
            }
            other => panic!("Expected InvalidInput for unsupported TPS dimension, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_psd_penalty_rejects_materially_indefinite_matrix() {
        let bad = array![[1.0, 0.0], [0.0, -0.25]];
        match validate_psd_penalty(
            &bad,
            "thin_plate bending penalty (dimension=3)",
            "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
        ) {
            Err(BasisError::IndefinitePenalty {
                context,
                min_eigenvalue,
                tolerance,
                guidance,
            }) => {
                assert!(context.contains("thin_plate"));
                assert!(min_eigenvalue < -tolerance);
                assert!(guidance.contains("PSD penalty"));
            }
            other => panic!("expected indefinite penalty error, got {other:?}"),
        }
    }

    #[test]
    fn test_thin_plate_3d_bending_penalty_is_psd_with_positive_rank() {
        let knots = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.7, 0.4]
        ];
        let tps = create_thin_plate_spline_basis(knots.view(), knots.view()).unwrap();
        assert_eq!(tps.dimension, 3);
        assert!(tps.num_kernel_basis > 0);
        assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));

        let kernel_penalty = tps
            .penalty_bending
            .slice(s![0..tps.num_kernel_basis, 0..tps.num_kernel_basis])
            .to_owned();
        let summary = validate_psd_penalty(
            &kernel_penalty,
            "thin_plate bending penalty (dimension=3)",
            "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
        )
        .unwrap();
        assert!(summary.min_eigenvalue >= -summary.tolerance);
        assert!(summary.max_abs_eigenvalue > 0.0);
        assert!(summary.effective_rank > 0);
    }

    #[test]
    fn test_thin_plate_3d_regression_configuration_stays_psd() {
        let knots = array![
            [0.12573022, -0.13210486, 0.64042265],
            [0.10490012, -0.53566937, 0.36159505],
            [1.30400005, 0.94708096, -0.70373524],
            [-1.26542147, -0.62327446, 0.04132598],
            [-2.32503077, -0.21879166, -1.24591095]
        ];
        let tps = create_thin_plate_spline_basis(knots.view(), knots.view()).unwrap();
        let kernel_penalty = tps
            .penalty_bending
            .slice(s![0..tps.num_kernel_basis, 0..tps.num_kernel_basis])
            .to_owned();
        let summary = validate_psd_penalty(
            &kernel_penalty,
            "thin_plate bending penalty (dimension=3)",
            "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
        )
        .unwrap();
        assert!(summary.min_eigenvalue >= -summary.tolerance);
        assert!(summary.effective_rank > 0);
    }

    #[test]
    fn test_build_thin_plate_basis_double_penalty_outputs_two_blocks() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            double_penalty: true,
            identifiability: SpatialIdentifiability::default(),
        };
        let result = build_thin_plate_basis(data.view(), &spec).unwrap();
        assert_eq!(result.penalties.len(), 2);
        assert_eq!(result.nullspace_dims.len(), 2);
        assert_eq!(result.design.nrows(), data.nrows());
        match &result.metadata {
            BasisMetadata::ThinPlate {
                identifiability_transform,
                ..
            } => assert!(identifiability_transform.is_some()),
            other => panic!("expected thin-plate metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_build_thin_plate_basis_default_identifiability_is_orthogonal_to_parametric_block() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
            [0.25, 0.75]
        ];
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            double_penalty: false,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
        };
        let result = build_thin_plate_basis(data.view(), &spec).unwrap();

        let mut c = Array2::<f64>::ones((data.nrows(), data.ncols() + 1));
        c.slice_mut(s![.., 1..]).assign(&data);
        let cross = result.design.t().dot(&c);
        let rel = orthogonality_relative_residual(result.design.view(), c.view());

        assert!(
            rel < 1e-10,
            "TPS design is not orthogonal to [1, x]: relative residual={rel:.3e}"
        );
        assert!(
            cross.iter().all(|v| v.abs() < 1e-10),
            "TPS cross-moment against parametric block is not numerically zero"
        );
        match &result.metadata {
            BasisMetadata::ThinPlate {
                identifiability_transform,
                ..
            } => assert!(identifiability_transform.is_some()),
            other => panic!("expected thin-plate metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_build_thin_plate_basis_center_strategies() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2]
        ];
        let specs = vec![
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::EqualMass { num_centers: 4 },
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::KMeans {
                    num_centers: 4,
                    max_iter: 5,
                },
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::UniformGrid { points_per_dim: 2 },
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::UserProvided(array![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]
                ]),
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
        ];
        for spec in specs {
            let result = build_thin_plate_basis(data.view(), &spec).unwrap();
            assert!(!result.design.is_empty());
            assert_eq!(result.penalties.len(), 1);
            assert_eq!(result.penalties[0].nrows(), result.design.ncols());
            assert_eq!(result.penalties[0].ncols(), result.design.ncols());
        }
    }

    #[test]
    fn test_equal_mass_centers_uses_non_first_dimensions() {
        // Regression guard for the prior bug where equal-mass partitioning only
        // looked at column 0 (PC1), which made center selection invariant to all
        // other coordinates.
        //
        // We construct two datasets with identical first coordinate and different
        // second-coordinate layouts. If selection used only column 0, both outputs
        // would be identical. The recursive alternating-dimension splitter should
        // produce different center sets.
        let mut data_a = Array2::<f64>::zeros((16, 2));
        let mut data_b = Array2::<f64>::zeros((16, 2));
        for i in 0..16 {
            data_a[[i, 0]] = i as f64;
            data_b[[i, 0]] = i as f64;
        }

        // First x-half: same x, different y ordering.
        // A: interleaved low/high; B: grouped low then high.
        let y_a_h1 = [0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0];
        let y_b_h1 = [0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0];
        // Second x-half: same pattern shifted to keep deterministic separation.
        let y_a_h2 = [10.0, 110.0, 11.0, 111.0, 12.0, 112.0, 13.0, 113.0];
        let y_b_h2 = [10.0, 11.0, 12.0, 13.0, 110.0, 111.0, 112.0, 113.0];

        for i in 0..8 {
            data_a[[i, 1]] = y_a_h1[i];
            data_b[[i, 1]] = y_b_h1[i];
            data_a[[i + 8, 1]] = y_a_h2[i];
            data_b[[i + 8, 1]] = y_b_h2[i];
        }

        let ca = select_equal_mass_centers(data_a.view(), 4).unwrap();
        let cb = select_equal_mass_centers(data_b.view(), 4).unwrap();

        let mut xa: Vec<f64> = ca.column(0).iter().copied().collect();
        let mut xb: Vec<f64> = cb.column(0).iter().copied().collect();
        xa.sort_by(f64::total_cmp);
        xb.sort_by(f64::total_cmp);

        assert_ne!(
            xa, xb,
            "equal-mass center selection unexpectedly ignored non-first dimensions"
        );
    }

    #[test]
    fn test_build_bspline_basis_1d_double_penalty() {
        let x = Array::linspace(0.0, 1.0, 32);
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 6,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::default(),
        };
        let result = build_bspline_basis_1d(x.view(), &spec).unwrap();
        assert_eq!(result.penalties.len(), 2);
        // Default identifiability centers the smooth, removing one null-space
        // dimension from the raw second-difference penalty.
        // Second penalty is the nullspace projector (rank = nullity of first penalty);
        // its own nullspace dimension is p - rank_of_projector.
        let p_constrained = result.design.ncols();
        assert_eq!(result.nullspace_dims[0], 1);
        // The shrinkage penalty targets a small subspace; most dims are unpenalized.
        assert!(result.nullspace_dims[1] >= p_constrained - 2);
        assert_eq!(result.design.nrows(), x.len());
    }

    #[test]
    fn test_build_bspline_basis_1d_automatic_uniform_uses_data_range() {
        let x = array![2.0, 3.0, 4.5, 6.0, 7.0, 8.0];
        let spec = BSplineBasisSpec {
            degree: 2,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Uniform,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let result = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let knots = match result.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => panic!("expected BSpline1D metadata"),
        };
        assert_eq!(knots.len(), 3 + 2 * (spec.degree + 1));
        assert!((knots[0] - 2.0).abs() < 1e-12);
        assert!((knots[knots.len() - 1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_build_bspline_basis_1d_automatic_quantile_is_not_uniform_for_skewed_data() {
        let x = array![0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 10.0, 10.5, 11.0, 12.0];
        let spec = BSplineBasisSpec {
            degree: 2,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let result = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let knots = match result.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => panic!("expected BSpline1D metadata"),
        };
        let start = spec.degree + 1;
        let internal = &knots.as_slice().unwrap()[start..(start + 3)];
        let d1 = internal[1] - internal[0];
        let d2 = internal[2] - internal[1];
        assert!(
            (d1 - d2).abs() > 1e-6,
            "quantile spacing should be non-uniform for skewed data"
        );
    }

    #[test]
    fn test_penalty_greville_selector_none_for_uniform_breakpoints() {
        let degree = 3usize;
        let knots = internal::generate_full_knot_vector((0.0, 1.0), 5, degree).unwrap();
        let g = penalty_greville_abscissae_for_knots(&knots, degree).unwrap();
        assert!(g.is_none());
    }

    #[test]
    fn test_build_bspline_basis_1d_quantile_uses_divided_difference_penalty() {
        let x = array![0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 10.0, 10.5, 11.0, 12.0];
        let spec = BSplineBasisSpec {
            degree: 2,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let knots = match &built.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => panic!("expected BSpline1D metadata"),
        };
        let g = penalty_greville_abscissae_for_knots(knots, spec.degree)
            .unwrap()
            .expect("quantile knots should trigger Greville scaling");
        let expected = create_difference_penalty_matrix(
            built.design.ncols(),
            spec.penalty_order,
            Some(g.view()),
        )
        .unwrap();

        let got = &built.penalties[0];
        let mut max_abs = 0.0_f64;
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                max_abs = max_abs.max((got[[i, j]] - expected[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-10,
            "quantile penalty mismatch: max_abs_diff={max_abs:.3e}"
        );
    }

    #[test]
    fn test_build_bspline_basis_1d_quantile_rejects_singular_divided_difference_penalty() {
        let x = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let spec = BSplineBasisSpec {
            degree: 2,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };

        match build_bspline_basis_1d(x.view(), &spec).unwrap_err() {
            BasisError::InvalidKnotVector(msg) => {
                assert!(msg.contains("singular"), "unexpected error message: {msg}");
            }
            other => panic!("expected InvalidKnotVector, got {other:?}"),
        }
    }

    #[test]
    fn test_bspline_identifiability_default_weighted_sum_to_zero() {
        let x = Array::linspace(0.0, 1.0, 40);
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        for j in 0..built.design.ncols() {
            let col_sum = built.design.column(j).sum();
            assert!(
                col_sum.abs() < 1e-8,
                "default weighted-sum-to-zero failed for column {j}: {col_sum}"
            );
        }
    }

    #[test]
    fn test_bspline_identifiability_remove_linear_trend_reduces_two_dims() {
        let x = Array::linspace(0.0, 1.0, 50);
        let raw = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 6,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };
        let constrained = BSplineBasisSpec {
            identifiability: BSplineIdentifiability::RemoveLinearTrend,
            ..raw.clone()
        };

        let b_raw = build_bspline_basis_1d(x.view(), &raw).unwrap();
        let b_constrained = build_bspline_basis_1d(x.view(), &constrained).unwrap();
        assert_eq!(b_constrained.design.ncols() + 2, b_raw.design.ncols());
    }

    #[test]
    fn test_bspline_identifiability_orthogonal_to_design_columns() {
        let x = Array::linspace(0.0, 1.0, 40);
        let mut constraints = Array2::<f64>::zeros((x.len(), 2));
        constraints.column_mut(0).fill(1.0);
        constraints.column_mut(1).assign(&x);

        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::OrthogonalToDesignColumns {
                columns: constraints.clone(),
                weights: None,
            },
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let cross = built.design.t().dot(&constraints);
        for i in 0..cross.nrows() {
            for j in 0..cross.ncols() {
                assert!(
                    cross[[i, j]].abs() < 1e-8,
                    "orthogonality violation at ({i},{j}) = {}",
                    cross[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_bspline_basis_sums_to_one() {
        let data = Array::linspace(0.1, 9.9, 100);
        let (basis, _) = create_basis::<Dense>(
            data.view(),
            KnotSource::Generate {
                data_range: (0.0, 10.0),
                num_internal_knots: 10,
            },
            3,
            BasisOptions::value(),
        )
        .unwrap();

        let sums = basis.sum_axis(Axis(1));

        // Every row should sum to 1.0 (with floating point tolerance)
        for &sum in sums.iter() {
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Basis did not sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_bspline_basis_sums_to_one_with_uniform_knots() {
        // Create data with a non-uniform distribution
        // Since quantile knots are disabled for P-splines, this tests the fallback to uniform knots
        let mut data = Array::zeros(100);
        for i in 0..100 {
            let x = if i < 50 {
                // Points clustered around 2.0
                2.0 + (i as f64) / 25.0 // Range: 2.0 to 4.0
            } else {
                // Points clustered around 8.0
                6.0 + (i as f64 - 50.0) / 25.0 // Range: 6.0 to 8.0
            };
            data[i] = x;
        }

        // Even when providing training data, this should fall back to uniform knots
        let (basis, knots) = create_basis::<Dense>(
            data.view(),
            KnotSource::Generate {
                data_range: (0.0, 10.0),
                num_internal_knots: 10,
            },
            3,
            BasisOptions::value(),
        )
        .unwrap();

        // Verify that knots are uniformly distributed (not following data distribution)
        // Since quantile knots are disabled, these should be uniform
        println!("Uniform knots (fallback): {:?}", knots);

        // Check that internal knots are uniformly spaced
        let internal_knots: Vec<f64> = knots
            .iter()
            .skip(4) // Skip the repeated boundary knots (degree+1 = 4)
            .take(10) // Take the internal knots
            .copied()
            .collect();

        if internal_knots.len() >= 2 {
            let spacing = internal_knots[1] - internal_knots[0];
            for window in internal_knots.windows(2) {
                let current_spacing = window[1] - window[0];
                assert!(
                    (current_spacing - spacing).abs() < 1e-9,
                    "Knots should be uniformly spaced, but spacing varies: expected {}, got {}",
                    spacing,
                    current_spacing
                );
            }
        }

        // Verify that the basis still sums to 1.0 for each data point
        let sums = basis.sum_axis(Axis(1));

        // Every row should sum to 1.0 (with floating point tolerance)
        for &sum in sums.iter() {
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Uniform basis did not sum to 1, got {}",
                sum
            );
        }

        // Now verify for points outside the original data distribution
        // Create a different set of evaluation points that are spread uniformly
        let eval_points = Array::linspace(0.1, 9.9, 100);

        // Create basis using the previously generated knots
        let (eval_basis, _) = create_basis::<Dense>(
            eval_points.view(),
            KnotSource::Provided(knots.view()),
            3,
            BasisOptions::value(),
        )
        .unwrap();

        // Verify sums for the evaluation points
        let eval_sums = eval_basis.sum_axis(Axis(1));

        for &sum in eval_sums.iter() {
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Basis at evaluation points did not sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_single_point_evaluation_degree_one() {
        // This test validates the raw output of the UNCONSTRAINED basis evaluator
        // (internal::evaluate_splines_at_point), not a final model prediction which
        // would require applying constraints. The test only verifies that the raw
        // basis functions are correctly evaluated, before any constraints are applied.
        //
        // Degree 1 (linear) splines with knots t = [0,0,1,2,2].
        // This gives 3 basis functions (n = k-d-1 = 5-1-1 = 3), B_{0,1}, B_{1,1}, B_{2,1}.
        let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let x = 0.5; // For x=0.5, the knot interval is mu=1, since t_1 <= x < t_2.

        let values = internal::evaluate_splines_at_point(x, 1, knots.view());
        assert_eq!(values.len(), 3);

        // Manual calculation for x=0.5:
        // The only non-zero basis function of degree 0 is B_{1,0} = 1.
        // Recurrence for degree 1:
        // B_{0,1}(x) = ( (x-t0)/(t1-t0) )*B_{0,0} + ( (t2-x)/(t2-t1) )*B_{1,0}
        //           = ( (0.5-0)/(0-0) )*0       + ( (1-0.5)/(1-0) )*1         = 0.5
        //           (Note: 0/0 division is taken as 0)
        // B_{1,1}(x) = ( (x-t1)/(t2-t1) )*B_{1,0} + ( (t3-x)/(t3-t2) )*B_{2,0}
        //           = ( (0.5-0)/(1-0) )*1       + ( (2-0.5)/(2-1) )*0         = 0.5
        // B_{2,1}(x) = ( (x-t2)/(t3-t2) )*B_{2,0} + ( (t4-x)/(t4-t3) )*B_{3,0}
        //           = ( (0.5-1)/(2-1) )*0       + ( (2-0.5)/(2-2) )*0         = 0.0

        assert!(
            (values[0] - 0.5).abs() < 1e-9,
            "Expected B_0,1 to be 0.5, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.5).abs() < 1e-9,
            "Expected B_1,1 to be 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 0.0).abs() < 1e-9,
            "Expected B_2,1 to be 0.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_cox_de_boor_higher_degree() {
        // Test that verifies the Cox-de Boor denominator handling for higher degree splines
        // Using non-uniform knots where numerical issues would be more apparent
        let knots = array![0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 4.0, 4.0];
        let x = 2.0;

        let values = internal::evaluate_splines_at_point(x, 2, knots.view());

        // The basis functions should sum to 1.0 (partition of unity property)
        let sum = values.sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis functions should sum to 1.0, got {}",
            sum
        );

        // All values should be non-negative
        for (i, &val) in values.iter().enumerate() {
            assert!(
                val >= -1e-9,
                "Basis function {} should be non-negative, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_boundary_value_handling() {
        // Test for proper boundary value handling at the upper boundary.
        // This test ensures that evaluation at the upper boundary works correctly.

        // Test the internal function directly with the problematic case
        let knots = array![
            0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0
        ];
        let x = 10.0; // This is the value that caused the panic
        let degree = 3;

        let basis_values = internal::evaluate_splines_at_point(x, degree, knots.view());

        // Should not panic and should return valid results
        assert_eq!(basis_values.len(), 8); // num_basis = 12 - 3 - 1 = 8

        let sum = basis_values.sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis functions should sum to 1.0 at boundary, got {}",
            sum
        );
    }

    #[test]
    fn test_basis_boundary_values() {
        // Property-based test: Verify boundary conditions using mathematical properties
        // This complements the cross-validation test by testing fundamental B-spline properties

        // A cubic B-spline basis. Knots are [0,0,0,0, 1,2,3, 4,4,4,4].
        // The domain is [0, 4].
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1; // 11 - 3 - 1 = 7

        // Test at the lower boundary (x=0)
        let basis_at_start = internal::evaluate_splines_at_point(0.0, degree, knots.view());

        // At the very start of the domain, only the first basis function should be non-zero (and equal to 1).
        assert_abs_diff_eq!(basis_at_start[0], 1.0, epsilon = 1e-9);
        for i in 1..num_basis {
            assert_abs_diff_eq!(basis_at_start[i], 0.0, epsilon = 1e-9);
        }

        // Test at the upper boundary (x=4)
        let basis_at_end = internal::evaluate_splines_at_point(4.0, degree, knots.view());

        // At the very end of the domain, only the LAST basis function should be non-zero (and equal to 1).
        for i in 0..(num_basis - 1) {
            assert_abs_diff_eq!(basis_at_end[i], 0.0, epsilon = 1e-9);
        }
        assert_abs_diff_eq!(basis_at_end[num_basis - 1], 1.0, epsilon = 1e-9);

        // Test intermediate points for partition of unity
        let test_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        for &x in &test_points {
            let basis = internal::evaluate_splines_at_point(x, degree, knots.view());
            let sum: f64 = basis.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
            if (sum - 1.0).abs() >= 1e-9 {
                panic!("Partition of unity failed at x={}", x);
            }
        }
    }

    #[test]
    fn test_constant_extrapolation_matches_boundary_basis_values() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let left_boundary = internal::evaluate_splines_at_point(0.0, degree, knots.view());
        let right_boundary = internal::evaluate_splines_at_point(4.0, degree, knots.view());

        let left_out = internal::evaluate_splines_at_point(-100.0, degree, knots.view());
        let right_out = internal::evaluate_splines_at_point(100.0, degree, knots.view());

        for i in 0..left_boundary.len() {
            assert_abs_diff_eq!(left_out[i], left_boundary[i], epsilon = 1e-12);
            assert_abs_diff_eq!(right_out[i], right_boundary[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_create_basis_uses_linear_extension_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let x = array![-0.5, 4.5];
        let x_c = array![0.0, 4.0];
        let (b_raw, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let (b_c, _) = create_basis::<Dense>(
            x_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let (db_c, _) = create_basis::<Dense>(
            x_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();

        let b_raw = b_raw.as_ref();
        let b_c = b_c.as_ref();
        let db_c = db_c.as_ref();
        for i in 0..x.len() {
            let dz = x[i] - x_c[i];
            for j in 0..b_raw.ncols() {
                let expected = b_c[[i, j]] + dz * db_c[[i, j]];
                assert_abs_diff_eq!(b_raw[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_create_basis_first_derivative_uses_boundary_slope_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let x = array![-0.25, 4.25];
        let x_c = array![0.0, 4.0];
        let (db_raw, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let (db_c, _) = create_basis::<Dense>(
            x_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let db_raw = db_raw.as_ref();
        let db_c = db_c.as_ref();
        assert_eq!(db_raw.dim(), db_c.dim());
        for i in 0..db_raw.nrows() {
            for j in 0..db_raw.ncols() {
                assert_abs_diff_eq!(db_raw[[i, j]], db_c[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dense_basis_preserves_linear_extension_when_internal_builder_goes_sparse() {
        let degree = 3usize;
        let knots = internal::generate_full_knot_vector((0.0, 10.0), 36, degree).unwrap();
        let x = array![-0.5, 10.5];
        let x_c = array![0.0, 10.0];
        assert!(should_use_sparse_basis(
            knots.len().saturating_sub(degree + 1),
            degree,
            1
        ));

        let (b_raw, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let (b_c, _) = create_basis::<Dense>(
            x_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let (db_c, _) = create_basis::<Dense>(
            x_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();

        for i in 0..x.len() {
            let dz = x[i] - x_c[i];
            for j in 0..b_raw.ncols() {
                let expected = b_c[[i, j]] + dz * db_c[[i, j]];
                assert_abs_diff_eq!(b_raw[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_derivatives_use_boundary_slope_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let support = degree + 1;
        let mut scratch = BasisEvalScratch::new(degree);
        let mut d1 = vec![0.0; support];
        let mut d2 = vec![0.0; support];
        let mut d1_left = vec![0.0; support];
        let mut d1_right = vec![0.0; support];

        let start_left = evaluate_splines_derivative_sparse_into(
            0.0,
            degree,
            knots.view(),
            &mut d1_left,
            &mut scratch,
        );
        let start = evaluate_splines_derivative_sparse_into(
            -10.0,
            degree,
            knots.view(),
            &mut d1,
            &mut scratch,
        );
        assert_eq!(start, start_left);
        for i in 0..support {
            assert_abs_diff_eq!(d1[i], d1_left[i], epsilon = 1e-12);
        }

        let start_right = evaluate_splines_derivative_sparse_into(
            4.0,
            degree,
            knots.view(),
            &mut d1_right,
            &mut scratch,
        );
        let start = evaluate_splines_derivative_sparse_into(
            10.0,
            degree,
            knots.view(),
            &mut d1,
            &mut scratch,
        );
        assert_eq!(start, start_right);
        for i in 0..support {
            assert_abs_diff_eq!(d1[i], d1_right[i], epsilon = 1e-12);
        }

        let _ = evaluate_splines_second_derivative_sparse_into(
            -10.0,
            degree,
            knots.view(),
            &mut d2,
            &mut scratch,
        );
        assert!(d2.iter().all(|v| v.abs() < 1e-12));
        let _ = evaluate_splines_second_derivative_sparse_into(
            10.0,
            degree,
            knots.view(),
            &mut d2,
            &mut scratch,
        );
        assert!(d2.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn test_create_basis_sparse_matches_dense_extrapolation_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let x = array![-0.5, 4.5];
        let (dense_basis, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let sparse_basis = generate_basis_internal::<SparseStorage>(
            x.view(),
            knots.view(),
            degree,
            BasisEvalKind::Basis,
        )
        .unwrap();
        let sparse_dense = <Dense as BasisOutputFormat>::from_sparse(sparse_basis).unwrap();
        assert_eq!(dense_basis.dim(), sparse_dense.dim());
        for i in 0..dense_basis.nrows() {
            for j in 0..dense_basis.ncols() {
                assert_abs_diff_eq!(dense_basis[[i, j]], sparse_dense[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_create_basis_sparse_first_derivative_matches_dense_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let x = array![-0.25, 4.25];
        let (dense_deriv, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .unwrap();
        let sparse_deriv = generate_basis_internal::<SparseStorage>(
            x.view(),
            knots.view(),
            degree,
            BasisEvalKind::FirstDerivative,
        )
        .unwrap();
        let sparse_dense = <Dense as BasisOutputFormat>::from_sparse(sparse_deriv).unwrap();
        assert_eq!(dense_deriv.dim(), sparse_dense.dim());
        for i in 0..dense_deriv.nrows() {
            for j in 0..dense_deriv.ncols() {
                assert_abs_diff_eq!(dense_deriv[[i, j]], sparse_dense[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ispline_scalar_boundary_behavior() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 1usize;
        let n_ispline = knots.len() - (degree + 1) - 1;
        let mut out = vec![0.0; n_ispline];

        evaluate_ispline_scalar(-10.0, knots.view(), degree, &mut out).expect("left boundary eval");
        assert!(out.iter().all(|&v| v.abs() <= 1e-12));

        evaluate_ispline_scalar(10.0, knots.view(), degree, &mut out).expect("right boundary eval");
        assert!(out[0].abs() <= 1e-12);
        for &v in out.iter().skip(1) {
            assert!((v - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_ispline_scalar_is_monotone_in_x() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 1usize;
        let n_ispline = knots.len() - (degree + 1) - 1;
        let xs = [0.0, 0.25, 0.75, 1.5, 2.2, 2.8, 3.0];

        let mut prev = vec![0.0; n_ispline];
        evaluate_ispline_scalar(xs[0], knots.view(), degree, &mut prev).expect("initial eval");

        for &x in xs.iter().skip(1) {
            let mut curr = vec![0.0; n_ispline];
            evaluate_ispline_scalar(x, knots.view(), degree, &mut curr).expect("eval along grid");
            for j in 0..n_ispline {
                assert!(
                    curr[j] + 1e-12 >= prev[j],
                    "I-spline basis not monotone at x={x}, j={j}: prev={}, curr={}",
                    prev[j],
                    curr[j]
                );
            }
            prev = curr;
        }
    }

    #[test]
    fn test_mspline_scalar_matches_scaled_bspline() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 2usize;
        let x = 1.25;
        let num_basis = knots.len() - degree - 1;
        let mut b = vec![0.0; num_basis];
        let mut m = vec![0.0; num_basis];
        let mut scratch = SplineScratch::new(degree);
        evaluate_bspline_basis_scalar(x, knots.view(), degree, &mut b, &mut scratch)
            .expect("bspline eval");
        evaluate_mspline_scalar(x, knots.view(), degree, &mut m, &mut scratch)
            .expect("mspline eval");

        let order = (degree + 1) as f64;
        for i in 0..num_basis {
            let span = knots[i + degree + 1] - knots[i];
            let expected = if span.abs() > 1e-12 {
                b[i] * (order / span)
            } else {
                0.0
            };
            assert_abs_diff_eq!(m[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_create_basis_mspline_zero_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 2usize;
        let x = array![-10.0, 1.0, 10.0];
        let (m, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::m_spline(),
        )
        .expect("create mspline basis");
        let m = m.as_ref();
        assert!(m.row(0).iter().all(|v| v.abs() <= 1e-12));
        assert!(m.row(2).iter().all(|v| v.abs() <= 1e-12));
        assert!(m.row(1).iter().any(|v| v.abs() > 1e-12));
    }

    #[test]
    fn test_create_basis_ispline_boundary_rows() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 1usize;
        let x = array![-10.0, 1.5, 10.0];
        let (i_basis, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("create ispline basis");
        let i_basis = i_basis.as_ref();
        assert!(i_basis.row(0).iter().all(|v| v.abs() <= 1e-12));
        assert!(i_basis[[2, 0]].abs() <= 1e-12);
        for j in 1..i_basis.ncols() {
            assert!((i_basis[[2, j]] - 1.0).abs() <= 1e-12);
        }
        for &v in i_basis.row(1) {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_ispline_derivative_matches_cumulative_bspline_derivative_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let degree = 2usize;
        let bs_degree = degree + 1;
        let n_i = knots.len() - bs_degree - 1;

        // Check at interior points away from knot boundaries for stable central differences.
        let xs = [0.35, 0.8, 1.4, 2.2];
        let h = 1e-6;
        for &x in &xs {
            let mut i_plus = vec![0.0; n_i];
            let mut i_minus = vec![0.0; n_i];
            evaluate_ispline_scalar(x + h, knots.view(), degree, &mut i_plus).expect("I(x+h)");
            evaluate_ispline_scalar(x - h, knots.view(), degree, &mut i_minus).expect("I(x-h)");

            let mut db = vec![0.0; n_i];
            evaluate_bspline_derivative_scalar(x, knots.view(), bs_degree, &mut db).expect("B'(x)");
            let mut d_i = vec![0.0; n_i];
            let mut running = 0.0_f64;
            for j in (0..n_i).rev() {
                running += db[j];
                d_i[j] = running;
            }

            for j in 0..n_i {
                let fd = (i_plus[j] - i_minus[j]) / (2.0 * h);
                assert_eq!(
                    d_i[j].signum(),
                    fd.signum(),
                    "sign mismatch at x={x}, j={j}: analytic={} fd={}",
                    d_i[j],
                    fd
                );
                assert_abs_diff_eq!(fd, d_i[j], epsilon = 2e-5);
            }
        }
    }

    #[test]
    fn test_non_bspline_derivative_orders_are_rejected() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let x = array![0.5, 1.5];
        let err = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            2,
            BasisOptions {
                derivative_order: 1,
                basis_family: BasisFamily::MSpline,
            },
        )
        .expect_err("MSpline derivative order should be rejected");
        assert!(matches!(err, BasisError::InvalidInput(_)));
    }

    #[test]
    fn test_mspline_sparse_matches_dense() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let x = array![-1.0, 0.3, 1.1, 2.7, 4.0];
        let (dense, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            2,
            BasisOptions::m_spline(),
        )
        .expect("dense mspline");
        let (sparse, _) = create_basis::<Sparse>(
            x.view(),
            KnotSource::Provided(knots.view()),
            2,
            BasisOptions::m_spline(),
        )
        .expect("sparse mspline");

        let dense = dense.as_ref();
        let mut sparse_dense = Array2::<f64>::zeros((sparse.nrows(), sparse.ncols()));
        let (symbolic, values) = sparse.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..sparse.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                sparse_dense[[row_idx[idx], col]] += values[idx];
            }
        }
        assert_eq!(dense.dim(), sparse_dense.dim());
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert_abs_diff_eq!(dense[[i, j]], sparse_dense[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_mspline_rejects_zero_normalization_spans() {
        // degree=2 with 4 repeated boundary knots makes t[3]-t[0]=0 for i=0.
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let x = array![0.25, 0.5, 0.75];
        let err = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            2,
            BasisOptions::m_spline(),
        )
        .expect_err("degenerate M-spline normalization spans should be rejected");
        assert!(matches!(err, BasisError::InvalidInput(_)));
    }

    #[test]
    fn test_ispline_sparse_output_is_rejected() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let x = array![0.2, 1.5, 2.8];
        let err = create_basis::<Sparse>(
            x.view(),
            KnotSource::Provided(knots.view()),
            1,
            BasisOptions::i_spline(),
        )
        .expect_err("I-spline sparse output should be rejected");
        assert!(matches!(err, BasisError::InvalidInput(_)));
    }

    #[test]
    fn test_degree_0_boundary_behavior() {
        let knots: Array1<f64> = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let x = 2.0;

        const EPS: f64 = 1e-12;

        for i in 0..(knots.len() - 1) {
            let interval_width = knots[i + 1] - knots[i];
            let expected = if interval_width.abs() < EPS {
                if i == knots.len() - 2 && (x - knots[i + 1]).abs() < EPS {
                    1.0
                } else {
                    0.0
                }
            } else if x >= knots[i] && x < knots[i + 1] {
                1.0
            } else if i == knots.len() - 2 && (x - knots[i + 1]).abs() < EPS {
                1.0
            } else {
                0.0
            };

            let value = evaluate_bspline(x, &knots, i, 0);
            assert_abs_diff_eq!(value, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_boundary_analysis() {
        // Test case from the failing test: knots [0, 0, 1, 2, 2], degree 1, x=2
        let knots: Array1<f64> = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let degree = 1;
        let x = 2.0;

        let num_basis = knots.len() - degree - 1;
        let iterative_basis = internal::evaluate_splines_at_point(x, degree, knots.view());

        let recursive_values: Vec<f64> = (0..num_basis)
            .map(|i| evaluate_bspline(x, &knots, i, degree))
            .collect();
        let expected = [0.0, 0.0, 1.0];

        assert_eq!(
            recursive_values.len(),
            expected.len(),
            "Recursive evaluation length mismatch"
        );

        for (i, (&recursive, &expected_value)) in
            recursive_values.iter().zip(expected.iter()).enumerate()
        {
            assert_abs_diff_eq!(recursive, expected_value, epsilon = 1e-12);
            assert_abs_diff_eq!(iterative_basis[i], expected_value, epsilon = 1e-12);
        }

        let recursive_sum: f64 = recursive_values.iter().sum();
        let iterative_sum = iterative_basis.sum();

        assert_abs_diff_eq!(recursive_sum, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(iterative_sum, 1.0, epsilon = 1e-12);
    }

    /// Validates the basis functions against Example 1 in Starkey's "Cox-deBoor" notes.
    ///
    /// This example is a linear spline (degree=1, order=2) with a uniform knot vector.
    /// We test the values of the blending functions at specific points to ensure they
    /// match the manually derived formulas in the literature.
    ///
    /// Reference: Denbigh Starkey, "Cox-deBoor Equations for B-Splines", pg. 8.
    #[test]
    fn test_starkey_notes_example_1() {
        let degree = 1;
        // The book uses knot vector (0, 1, 2, 3, 4, 5).
        // Our setup requires boundary knots. For num_internal_knots = 4, range (0,5),
        // we get internal knots {1,2,3,4}, full vector {0,0, 1,2,3,4, 5,5}.
        let knots = array![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0];
        let num_basis = knots.len() - degree - 1; // 8 - 1 - 1 = 6 basis functions

        // Test case 1: u = 1.5, which is in the span [1, 2].
        // Expected: Two non-zero basis functions, each with value 0.5
        let basis_at_1_5 = internal::evaluate_splines_at_point(1.5, degree, knots.view());
        assert_eq!(basis_at_1_5.len(), num_basis);
        assert_abs_diff_eq!(basis_at_1_5.sum(), 1.0, epsilon = 1e-9);

        // Validate that exactly 2 basis functions are non-zero with value 0.5 each
        let non_zero_count = basis_at_1_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            non_zero_count, 2,
            "Should have exactly 2 non-zero basis functions at x=1.5"
        );

        // Check that the non-zero values are at indices 1 and 2 (as determined empirically)
        // and both have value 0.5 (from linear interpolation)
        assert_abs_diff_eq!(basis_at_1_5[1], 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(basis_at_1_5[2], 0.5, epsilon = 1e-9);

        // Test case 2: u = 2.5, which is in the span [2, 3].
        // Expected: Two non-zero basis functions, each with value 0.5
        let basis_at_2_5 = internal::evaluate_splines_at_point(2.5, degree, knots.view());
        assert_eq!(basis_at_2_5.len(), num_basis);
        assert_abs_diff_eq!(basis_at_2_5.sum(), 1.0, epsilon = 1e-9);

        // Validate that exactly 2 basis functions are non-zero with value 0.5 each
        let non_zero_count_2_5 = basis_at_2_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            non_zero_count_2_5, 2,
            "Should have exactly 2 non-zero basis functions at x=2.5"
        );

        // Check that the non-zero values are at indices 2 and 3 (as determined empirically)
        // and both have value 0.5 (from linear interpolation)
        assert_abs_diff_eq!(basis_at_2_5[2], 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(basis_at_2_5[3], 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_prediction_consistency_on_and_off_grid() {
        // This test replaces a previously flawed version. The goal is to verify that
        // the prediction logic for a constrained B-spline basis is consistent and correct.
        // We perform two checks:
        // Stage: On-grid consistency—ensure calculating a prediction for a single point that
        //    is ON the original grid yields the same result as the batch calculation.
        // Stage: Off-grid interpolation—ensure a prediction for a point off the grid
        //    (e.g., 0.65) produces a value that lies between its neighbors (0.6 and 0.7),
        //    validating the spline's interpolation property.
        //
        // The previous test incorrectly asserted that the value at 0.65 should equal
        // the value at 0.6, which is false for a non-flat cubic spline.

        // --- Setup: Same as the original test ---
        let data = Array::linspace(0.0, 1.0, 11);
        let (basis_unc, _) = create_basis::<Dense>(
            data.view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            3,
            BasisOptions::value(),
        )
        .unwrap();

        let main_basis_unc = basis_unc.slice(s![.., 1..]);
        let (main_basis_con, z_transform) =
            apply_sum_to_zero_constraint(main_basis_unc, None).unwrap();

        let intercept_coeff = 0.5;
        let num_con_coeffs = main_basis_con.ncols();
        let main_coeffs = Array1::from_shape_fn(num_con_coeffs, |i| (i as f64 + 1.0) * 0.1);

        // --- Calculate batch predictions on the grid (our ground truth) ---
        let predictions_on_grid = intercept_coeff + main_basis_con.dot(&main_coeffs);

        // --- On-grid consistency check ---
        let test_point_on_grid_x = 0.6;
        let on_grid_idx = 6;

        // Calculate the prediction for this single point from scratch.
        let (raw_basis_at_point, _) = create_basis::<Dense>(
            array![test_point_on_grid_x].view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            3,
            BasisOptions::value(),
        )
        .unwrap();
        let main_basis_unc_at_point = raw_basis_at_point.slice(s![0, 1..]);
        let main_basis_con_at_point =
            Array1::from_vec(main_basis_unc_at_point.to_vec()).dot(&z_transform);
        let prediction_at_0_6 = intercept_coeff + main_basis_con_at_point.dot(&main_coeffs);

        // ASSERT: The single-point prediction must exactly match the batch prediction for the same point.
        assert_abs_diff_eq!(
            prediction_at_0_6,
            predictions_on_grid[on_grid_idx],
            epsilon = 1e-12 // Use a tight epsilon for this identity check
        );

        // --- Off-grid interpolation check ---
        // Now test the off-grid point x=0.65, which lies between grid points 0.6 and 0.7.
        let test_point_off_grid_x = 0.65;

        // Calculate the prediction for this single off-grid point.
        let (raw_basis_off_grid, _) = create_basis::<Dense>(
            array![test_point_off_grid_x].view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            3,
            BasisOptions::value(),
        )
        .unwrap();
        let main_basis_unc_off_grid = raw_basis_off_grid.slice(s![0, 1..]);
        let main_basis_con_off_grid =
            Array1::from_vec(main_basis_unc_off_grid.to_vec()).dot(&z_transform);
        let prediction_at_0_65 = intercept_coeff + main_basis_con_off_grid.dot(&main_coeffs);

        // Get the values of the neighboring on-grid points from our batch calculation.
        let value_at_0_6 = predictions_on_grid[6];
        let value_at_0_7 = predictions_on_grid[7];

        // Determine the bounds for the interpolation.
        let lower_bound = value_at_0_6.min(value_at_0_7);
        let upper_bound = value_at_0_6.max(value_at_0_7);

        println!("Value at x=0.60: {}", value_at_0_6);
        println!("Value at x=0.65: {}", prediction_at_0_65);
        println!("Value at x=0.70: {}", value_at_0_7);

        // ASSERT: The prediction at 0.65 must lie between the values at 0.6 and 0.7.
        // This is a robust check of the spline's interpolating behavior.
        assert!(
            prediction_at_0_65 >= lower_bound && prediction_at_0_65 <= upper_bound,
            "Off-grid prediction ({}) at x=0.65 should be between its neighbors ({}, {})",
            prediction_at_0_65,
            value_at_0_6,
            value_at_0_7
        );
    }

    #[test]
    fn test_error_conditions() {
        match create_basis::<Dense>(
            array![].view(),
            KnotSource::Generate {
                data_range: (0.0, 10.0),
                num_internal_knots: 5,
            },
            0,
            BasisOptions::value(),
        )
        .unwrap_err()
        {
            BasisError::InvalidDegree(deg) => assert_eq!(deg, 0),
            _ => panic!("Expected InvalidDegree error"),
        }

        match create_basis::<Dense>(
            array![].view(),
            KnotSource::Generate {
                data_range: (10.0, 0.0),
                num_internal_knots: 5,
            },
            1,
            BasisOptions::value(),
        )
        .unwrap_err()
        {
            BasisError::InvalidRange(start, end) => {
                assert_eq!(start, 10.0);
                assert_eq!(end, 0.0);
            }
            _ => panic!("Expected InvalidRange error"),
        }

        // Test degenerate range detection
        match create_basis::<Dense>(
            array![].view(),
            KnotSource::Generate {
                data_range: (5.0, 5.0),
                num_internal_knots: 3,
            },
            1,
            BasisOptions::value(),
        )
        .unwrap_err()
        {
            BasisError::DegenerateRange(num_knots) => {
                assert_eq!(num_knots, 3);
            }
            err => panic!("Expected DegenerateRange error, got {:?}", err),
        }

        // Special case: Zero-width range is allowed when num_internal_knots = 0
        // This creates a valid but trivial basis
        let result = create_basis::<Dense>(
            array![].view(),
            KnotSource::Generate {
                data_range: (5.0, 5.0),
                num_internal_knots: 0,
            },
            1,
            BasisOptions::value(),
        );
        assert!(
            result.is_ok(),
            "Zero-width range with no internal knots should be valid"
        );

        // Test uniform fallback (quantile knots are disabled for P-splines)
        let (_, knots_uniform) = create_basis::<Dense>(
            array![].view(), // empty evaluation set is fine
            KnotSource::Generate {
                data_range: (0.0, 10.0),
                num_internal_knots: 3,
            },
            1, // degree
            BasisOptions::value(),
        )
        .unwrap();

        // Uniform fallback: boundary repeated degree+1=2 times => 2 + 3 + 2 = 7 knots
        let expected_knots = array![0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0];
        assert_abs_diff_eq!(
            knots_uniform.as_slice().unwrap(),
            expected_knots.as_slice().unwrap(),
            epsilon = 1e-9
        );

        match create_difference_penalty_matrix(5, 5, None).unwrap_err() {
            BasisError::InvalidPenaltyOrder { order, num_basis } => {
                assert_eq!(order, 5);
                assert_eq!(num_basis, 5);
            }
            _ => panic!("Expected InvalidPenaltyOrder error"),
        }
    }

    #[test]
    fn test_invalid_knot_vector_monotonicity_and_finiteness() {
        // Decreasing knot vector should be rejected
        let knots_bad_order = array![0.0, 0.0, 2.0, 1.0, 3.0, 3.0];
        let data = array![0.5, 1.0, 1.5];
        match create_basis::<Dense>(
            data.view(),
            KnotSource::Provided(knots_bad_order.view()),
            1,
            BasisOptions::value(),
        ) {
            Err(BasisError::InvalidKnotVector(msg)) => {
                assert!(msg.contains("non-decreasing"));
            }
            other => panic!("Expected InvalidKnotVector (order), got {:?}", other),
        }

        // Non-finite knot vector should be rejected
        let mut knots_non_finite = array![0.0, 0.0, 1.0, 2.0, 2.0];
        knots_non_finite[2] = f64::NAN;
        match create_basis::<Dense>(
            data.view(),
            KnotSource::Provided(knots_non_finite.view()),
            1,
            BasisOptions::value(),
        ) {
            Err(BasisError::InvalidKnotVector(msg)) => {
                assert!(msg.contains("non-finite"));
            }
            other => panic!("Expected InvalidKnotVector (non-finite), got {:?}", other),
        }
    }

    #[test]
    fn test_second_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut d1 = vec![0.0; num_basis];
        let mut d1_plus = vec![0.0; num_basis];
        let mut d1_minus = vec![0.0; num_basis];
        let mut d2 = vec![0.0; num_basis];

        let x = 0.37;
        let h = 1e-5;

        evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut d1)
            .expect("first derivative");
        evaluate_bspline_derivative_scalar(x + h, knots.view(), degree, &mut d1_plus)
            .expect("first derivative +h");
        evaluate_bspline_derivative_scalar(x - h, knots.view(), degree, &mut d1_minus)
            .expect("first derivative -h");
        evaluate_bspline_second_derivative_scalar(x, knots.view(), degree, &mut d2)
            .expect("second derivative");

        let tol = 1e-3;
        for i in 0..num_basis {
            let fd = (d1_plus[i] - d1_minus[i]) / (2.0 * h);
            assert_eq!(d2[i].signum(), fd.signum());
            assert!(
                (d2[i] - fd).abs() < tol,
                "second derivative mismatch at {}: analytic={}, fd={}",
                i,
                d2[i],
                fd
            );
        }
    }

    #[test]
    fn test_third_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut d2_plus = vec![0.0; num_basis];
        let mut d2_minus = vec![0.0; num_basis];
        let mut d3 = vec![0.0; num_basis];

        let x = 0.37;
        let h = 1e-4;

        evaluate_bspline_second_derivative_scalar(x + h, knots.view(), degree, &mut d2_plus)
            .expect("second derivative +h");
        evaluate_bspline_second_derivative_scalar(x - h, knots.view(), degree, &mut d2_minus)
            .expect("second derivative -h");
        evaluate_bspline_third_derivative_scalar(x, knots.view(), degree, &mut d3)
            .expect("third derivative");

        let tol = 5e-3;
        for i in 0..num_basis {
            let fd = (d2_plus[i] - d2_minus[i]) / (2.0 * h);
            assert_eq!(d3[i].signum(), fd.signum());
            assert!(
                (d3[i] - fd).abs() < tol,
                "third derivative mismatch at {}: analytic={}, fd={}",
                i,
                d3[i],
                fd
            );
        }
    }

    #[test]
    fn test_fourth_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 4;
        let num_basis = knots.len() - degree - 1;
        let mut d3_plus = vec![0.0; num_basis];
        let mut d3_minus = vec![0.0; num_basis];
        let mut d4 = vec![0.0; num_basis];

        let x = 0.47;
        let h = 1e-4;

        evaluate_bspline_third_derivative_scalar(x + h, knots.view(), degree, &mut d3_plus)
            .expect("third derivative +h");
        evaluate_bspline_third_derivative_scalar(x - h, knots.view(), degree, &mut d3_minus)
            .expect("third derivative -h");
        evaluate_bspline_fourth_derivative_scalar(x, knots.view(), degree, &mut d4)
            .expect("fourth derivative");

        let tol = 3e-2;
        for i in 0..num_basis {
            let fd = (d3_plus[i] - d3_minus[i]) / (2.0 * h);
            assert_eq!(d4[i].signum(), fd.signum());
            assert!(
                (d4[i] - fd).abs() < tol,
                "fourth derivative mismatch at {}: analytic={}, fd={}",
                i,
                d4[i],
                fd
            );
        }
    }

    #[test]
    fn test_sparse_second_derivative_matches_scalar() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut sparse_values = vec![0.0; degree + 1];
        let mut scalar_values = vec![0.0; num_basis];
        let mut scratch = BasisEvalScratch::new(degree);

        let xs = [0.05, 0.2, 0.37, 0.61, 0.9];
        for &x in &xs {
            let start = evaluate_splines_second_derivative_sparse_into(
                x,
                degree,
                knots.view(),
                &mut sparse_values,
                &mut scratch,
            );

            evaluate_bspline_second_derivative_scalar(x, knots.view(), degree, &mut scalar_values)
                .expect("scalar second derivative");

            let mut reconstructed = vec![0.0; num_basis];
            for (offset, &value) in sparse_values.iter().enumerate() {
                let col = start + offset;
                if col < num_basis {
                    reconstructed[col] = value;
                }
            }

            for j in 0..num_basis {
                assert!(
                    (reconstructed[j] - scalar_values[j]).abs() < 1e-11,
                    "sparse second derivative mismatch at x={}, basis {}: sparse={}, scalar={}",
                    x,
                    j,
                    reconstructed[j],
                    scalar_values[j]
                );
            }
        }
    }

    #[test]
    fn test_greville_abscissae_cubic() {
        // Uniform cubic spline on [0, 1] with 1 internal knot at 0.5
        // Knot vector: [0, 0, 0, 0, 0.5, 1, 1, 1, 1] (9 knots)
        // Number of basis functions: 9 - 3 - 1 = 5
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;

        let g =
            compute_greville_abscissae(&knots, degree).expect("should compute Greville abscissae");

        // For degree 3: G_j = (t_{j+1} + t_{j+2} + t_{j+3}) / 3
        // G_0 = (0 + 0 + 0) / 3 = 0
        // G_1 = (0 + 0 + 0.5) / 3 = 0.1667
        // G_2 = (0 + 0.5 + 1.0) / 3 = 0.5
        // G_3 = (0.5 + 1.0 + 1.0) / 3 = 0.8333
        // G_4 = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert_eq!(g.len(), 5);
        assert_abs_diff_eq!(g[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[1], 0.5 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[2], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(g[3], 2.5 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geometric_constraint_transform_orthogonality() {
        // Test that the geometric constraint transform makes coefficients orthogonal
        // to constant and linear (Greville) vectors.
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;

        let (z, s_constrained) = compute_geometric_constraint_transform(&knots, degree, 2)
            .expect("should compute transform");
        // Verify s_constrained has expected dimensions
        assert!(
            s_constrained.nrows() > 0,
            "s_constrained should not be empty"
        );

        let g = compute_greville_abscissae(&knots, degree).expect("should compute Greville");
        let k = g.len();
        let ones = Array1::<f64>::ones(k);

        // Z^T * 1 should be approximately zero (orthogonal to constants)
        let z_t_ones = z.t().dot(&ones);
        for i in 0..z_t_ones.len() {
            assert!(
                z_t_ones[i].abs() < 1e-10,
                "Z not orthogonal to constants: Z'*1[{}] = {}",
                i,
                z_t_ones[i]
            );
        }

        // Z^T * G should be approximately zero (orthogonal to linear in Greville coords)
        let z_t_g = z.t().dot(&g);
        for i in 0..z_t_g.len() {
            assert!(
                z_t_g[i].abs() < 1e-10,
                "Z not orthogonal to Greville: Z'*G[{}] = {}",
                i,
                z_t_g[i]
            );
        }

        // Transform should reduce dimension by 2 (removing constant and linear)
        assert_eq!(z.ncols(), k - 2, "Z should have k-2 columns");
        assert_eq!(z.nrows(), k, "Z should have k rows");
    }

    #[test]
    fn test_geometric_constraint_transform_dimensions() {
        // Test various knot configurations
        for n_internal in [3, 5, 10, 20] {
            let degree = 3;
            let n_knots = n_internal + 2 * (degree + 1);
            let mut knots = Array1::<f64>::zeros(n_knots);

            // Build clamped uniform knot vector
            for i in 0..=degree {
                knots[i] = 0.0;
                knots[n_knots - 1 - i] = 1.0;
            }
            for i in 0..n_internal {
                knots[degree + 1 + i] = (i + 1) as f64 / (n_internal + 1) as f64;
            }

            let (z, s_c) = compute_geometric_constraint_transform(&knots, degree, 2)
                .expect("should compute transform");

            let n_basis = n_knots - degree - 1;
            let n_constrained = n_basis - 2;

            assert_eq!(z.nrows(), n_basis, "Z rows should equal n_basis");
            assert_eq!(z.ncols(), n_constrained, "Z cols should equal n_basis - 2");
            assert_eq!(
                s_c.nrows(),
                n_constrained,
                "S_c should be n_constrained x n_constrained"
            );
            assert_eq!(s_c.ncols(), n_constrained);
        }
    }

    #[test]
    fn test_duchon_exact_primary_case_k10_builds() {
        let n = 6usize;
        let d = 10usize;
        let k = 4usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut centers = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64 + 1.0) * (j as f64 + 0.5) * 0.01;
            }
        }
        for i in 0..k {
            for j in 0..d {
                centers[[i, j]] = (i as f64 + 0.25) * (j as f64 + 1.0) * 0.02;
            }
        }

        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            4,
            DuchonNullspaceOrder::Linear,
        )
        .expect("primary Duchon case should build");
        assert_eq!(out.dimension, d);
        assert_eq!(out.basis.nrows(), n);
        assert_eq!(out.penalty_kernel.nrows(), out.penalty_kernel.ncols());
    }

    #[test]
    fn test_duchon_non_primary_case_builds_with_general_kernel() {
        let data = Array2::<f64>::zeros((4, 3));
        let centers = Array2::<f64>::zeros((3, 3));
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            1,
            DuchonNullspaceOrder::Linear,
        )
        .expect("general integer (p,s,k) Duchon kernel should build");
        assert_eq!(out.dimension, 3);
        assert_eq!(out.basis.nrows(), 4);
        assert_eq!(out.penalty_kernel.nrows(), out.penalty_kernel.ncols());
    }

    #[test]
    fn test_build_duchon_basis_freezes_default_spatial_identifiability() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            length_scale: Some(1.0),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            double_penalty: false,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
        };
        let out = build_duchon_basis(data.view(), &spec).unwrap();
        match &out.metadata {
            BasisMetadata::Duchon {
                identifiability_transform,
                ..
            } => assert!(identifiability_transform.is_some()),
            other => panic!("expected Duchon metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_build_duchon_basis_default_identifiability_is_orthogonal_to_parametric_block() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
            [0.25, 0.75]
        ];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            length_scale: Some(1.0),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            double_penalty: false,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
        };
        let out = build_duchon_basis(data.view(), &spec).unwrap();

        let mut c = Array2::<f64>::ones((data.nrows(), data.ncols() + 1));
        c.slice_mut(s![.., 1..]).assign(&data);
        let cross = out.design.t().dot(&c);
        let rel = orthogonality_relative_residual(out.design.view(), c.view());

        assert!(
            rel < 1e-10,
            "Duchon design is not orthogonal to [1, x]: relative residual={rel:.3e}"
        );
        assert!(
            cross.iter().all(|v| v.abs() < 1e-10),
            "Duchon cross-moment against parametric block is not numerically zero"
        );
        match &out.metadata {
            BasisMetadata::Duchon {
                identifiability_transform,
                ..
            } => assert!(identifiability_transform.is_some()),
            other => panic!("expected Duchon metadata, got {other:?}"),
        }
    }

    #[test]
    fn test_pairwise_distance_bounds_helper() {
        let pts = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
        let (r_min, r_max) = pairwise_distance_bounds(pts.view()).expect("bounds should exist");
        assert!((r_min - 5.0).abs() < 1e-12);
        assert!((r_max - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_duchon_general_kernel_symmetric_and_finite() {
        let n = 7usize;
        let d = 5usize;
        let k = 5usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut centers = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = 0.03 * (i as f64 + 1.0) * (j as f64 + 0.5);
            }
        }
        for i in 0..k {
            for j in 0..d {
                centers[[i, j]] = 0.07 * (i as f64 + 0.2) * (j as f64 + 0.8);
            }
        }
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(0.9),
            5,
            DuchonNullspaceOrder::Linear, // p=1
        )
        .expect("general Duchon basis should build");
        assert!(out.basis.iter().all(|v| v.is_finite()));
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
        for i in 0..out.penalty_kernel.nrows() {
            for j in 0..out.penalty_kernel.ncols() {
                let a = out.penalty_kernel[[i, j]];
                let b = out.penalty_kernel[[j, i]];
                assert!((a - b).abs() < 1e-8, "kernel penalty must be symmetric");
            }
        }
    }

    #[test]
    fn test_matern_center_sum_to_zero_produces_kernel_transform() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 0.7,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        assert_eq!(out.design.nrows(), data.nrows());
        assert_eq!(out.design.ncols(), centers.nrows() - 1);
        assert_eq!(out.penalties[0].nrows(), out.design.ncols());
        assert_eq!(out.penalties[0].ncols(), out.design.ncols());
        let BasisMetadata::Matern {
            identifiability_transform,
            ..
        } = out.metadata
        else {
            panic!("expected Matérn metadata");
        };
        let z = identifiability_transform.expect("sum-to-zero should store transform");
        assert_eq!(z.nrows(), centers.nrows());
        assert_eq!(z.ncols(), centers.nrows() - 1);
        let ones = Array1::<f64>::ones(centers.nrows());
        let residual = ones.dot(&z).mapv(f64::abs).sum();
        assert!(residual < 1e-10, "constant mode not removed: {residual}");
    }

    #[test]
    fn test_matern_include_intercept_keeps_single_unpenalized_dimension() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 1.1,
            nu: MaternNu::ThreeHalves,
            include_intercept: true,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        // (k-1) constrained kernel cols + explicit intercept.
        assert_eq!(out.design.ncols(), centers.nrows());
        assert_eq!(out.penalties.len(), 3);
        assert_eq!(out.nullspace_dims.len(), 3);
    }

    #[test]
    fn test_matern_double_penalty_drops_inactive_nullspace_block_without_intercept() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 1.1,
            nu: MaternNu::ThreeHalves,
            include_intercept: false,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        assert_eq!(out.penalties.len(), 1);
        assert_eq!(out.nullspace_dims.len(), 1);
        assert_eq!(out.penalty_info.len(), 1);
        assert!(out.penalty_info.iter().all(|info| info.active));
        assert!(matches!(out.penalty_info[0].source, PenaltySource::Primary));
    }

    #[test]
    fn test_matern_double_penalty_keeps_intercept_shrinkage_block() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 1.1,
            nu: MaternNu::ThreeHalves,
            include_intercept: true,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        assert_eq!(out.penalties.len(), 2);
        assert_eq!(out.nullspace_dims.len(), 2);
        assert_eq!(out.penalty_info.len(), 2);
        assert!(out.penalty_info.iter().all(|info| info.active));
        assert!(matches!(out.penalty_info[0].source, PenaltySource::Primary));
        assert!(matches!(
            out.penalty_info[1].source,
            PenaltySource::DoublePenaltyNullspace
        ));
    }

    #[test]
    fn test_matern_log_kappa_derivative_matches_fd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let deriv = build_matern_basis_log_kappa_derivative(data.view(), &spec)
            .expect("analytic Matérn derivative should build");

        let eps: f64 = 1e-6;
        let kappa = 1.0 / spec.length_scale;
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = ls_plus;
        spec_minus.length_scale = ls_minus;
        let plus = build_matern_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_matern_basis(data.view(), &spec_minus).expect("minus build");

        let fd_design = (&plus.design - &minus.design) / (2.0 * eps);
        let fd_penalty = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);

        let design_err = (&deriv.design_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let penalty_err = (&deriv.penalties_derivative[0] - &fd_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for i in 0..deriv.design_derivative.nrows() {
            for j in 0..deriv.design_derivative.ncols() {
                assert_eq!(
                    deriv.design_derivative[[i, j]].signum(),
                    fd_design[[i, j]].signum()
                );
            }
        }
        for i in 0..deriv.penalties_derivative[0].nrows() {
            for j in 0..deriv.penalties_derivative[0].ncols() {
                assert_eq!(
                    deriv.penalties_derivative[0][[i, j]].signum(),
                    fd_penalty[[i, j]].signum()
                );
            }
        }

        assert!(
            design_err < 1e-5,
            "design derivative mismatch too large: {design_err}"
        );
        assert!(
            penalty_err < 1e-5,
            "penalty derivative mismatch too large: {penalty_err}"
        );
    }

    #[test]
    fn test_matern_double_penalty_log_kappa_derivative_matches_fd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: true,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
        };
        let deriv = build_matern_basis_log_kappa_derivative(data.view(), &spec)
            .expect("analytic Matérn double-penalty derivative should build");

        let eps: f64 = 1e-6;
        let kappa = 1.0 / spec.length_scale;
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = ls_plus;
        spec_minus.length_scale = ls_minus;
        let plus = build_matern_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_matern_basis(data.view(), &spec_minus).expect("minus build");

        let fd_primary = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
        let primary_err = (&deriv.penalties_derivative[0] - &fd_primary)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(
            primary_err < 1e-5,
            "double-penalty primary derivative mismatch too large: {primary_err}"
        );
        assert_eq!(deriv.penalties_derivative.len(), 2);
        assert!(
            deriv.penalties_derivative[1]
                .iter()
                .all(|v| v.abs() < 1e-12),
            "nullspace shrinkage derivative should be zero"
        );
    }

    #[test]
    fn test_duchon_log_kappa_derivative_matches_fd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: Some(0.9),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
        };
        let mut workspace = BasisWorkspace::default();
        let (design_derivative, _) = build_duchon_design_psi_derivatives_with_workspace(
            data.view(),
            &spec,
            None,
            &mut workspace,
        )
        .expect("analytic Duchon design derivative should build");
        let (penalties_derivative, _) =
            build_duchon_operator_penalty_psi_derivatives_with_workspace(
                centers.view(),
                &spec,
                None,
                &mut workspace,
            )
            .expect("analytic Duchon penalty derivative should build");

        let eps: f64 = 1e-6;
        let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = Some(ls_plus);
        spec_minus.length_scale = Some(ls_minus);
        let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
        let plus_penalties = build_duchon_operator_penalty_candidates(
            centers.view(),
            Some(ls_plus),
            spec.power,
            spec.nullspace_order,
            None,
        )
        .expect("plus operator penalties");
        let minus_penalties = build_duchon_operator_penalty_candidates(
            centers.view(),
            Some(ls_minus),
            spec.power,
            spec.nullspace_order,
            None,
        )
        .expect("minus operator penalties");

        let fd_design = (&plus.design - &minus.design) / (2.0 * eps);
        let design_err = (&design_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            design_err < 1e-4,
            "Duchon design derivative mismatch too large: {design_err}"
        );

        assert_eq!(penalties_derivative.len(), 3);
        assert_eq!(plus_penalties.len(), 3);
        assert_eq!(minus_penalties.len(), 3);
        for penalty_idx in 0..penalties_derivative.len() {
            let fd_penalty = (&plus_penalties[penalty_idx].matrix
                - &minus_penalties[penalty_idx].matrix)
                / (2.0 * eps);
            let penalty_err = (&penalties_derivative[penalty_idx] - &fd_penalty)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!(
                penalty_err < 1e-4,
                "Duchon penalty derivative mismatch too large for block {penalty_idx}: {penalty_err}"
            );
        }
    }

    #[test]
    fn test_duchon_log_kappa_second_derivative_matches_fd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: Some(0.9),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
        };
        let mut workspace = BasisWorkspace::default();
        let (_, design_second_derivative) = build_duchon_design_psi_derivatives_with_workspace(
            data.view(),
            &spec,
            None,
            &mut workspace,
        )
        .expect("analytic Duchon design second derivative should build");
        let (_, penalties_second_derivative) =
            build_duchon_operator_penalty_psi_derivatives_with_workspace(
                centers.view(),
                &spec,
                None,
                &mut workspace,
            )
            .expect("analytic Duchon penalty second derivative should build");
        let base = build_duchon_basis(data.view(), &spec).expect("base build");

        let eps: f64 = 2e-5;
        let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = Some(ls_plus);
        spec_minus.length_scale = Some(ls_minus);
        let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
        let base_penalties = build_duchon_operator_penalty_candidates(
            centers.view(),
            spec.length_scale,
            spec.power,
            spec.nullspace_order,
            None,
        )
        .expect("base operator penalties");
        let plus_penalties = build_duchon_operator_penalty_candidates(
            centers.view(),
            Some(ls_plus),
            spec.power,
            spec.nullspace_order,
            None,
        )
        .expect("plus operator penalties");
        let minus_penalties = build_duchon_operator_penalty_candidates(
            centers.view(),
            Some(ls_minus),
            spec.power,
            spec.nullspace_order,
            None,
        )
        .expect("minus operator penalties");

        let fd_design = (&plus.design - &(base.design.clone() * 2.0) + &minus.design) / (eps * eps);
        let design_err = (&design_second_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            design_err < 5e-3,
            "Duchon design second derivative mismatch too large: {design_err}"
        );

        assert_eq!(penalties_second_derivative.len(), 3);
        assert_eq!(base_penalties.len(), 3);
        assert_eq!(plus_penalties.len(), 3);
        assert_eq!(minus_penalties.len(), 3);
        for penalty_idx in 0..penalties_second_derivative.len() {
            let fd_penalty = (&plus_penalties[penalty_idx].matrix
                - &(base_penalties[penalty_idx].matrix.clone() * 2.0)
                + &minus_penalties[penalty_idx].matrix)
                / (eps * eps);
            let penalty_err = (&penalties_second_derivative[penalty_idx] - &fd_penalty)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!(
                penalty_err < 5e-3,
                "Duchon penalty second derivative mismatch too large for block {penalty_idx}: {penalty_err}"
            );
        }
    }

    #[test]
    fn test_log_kappa_scaling_identities_match_autodiff() {
        let psi0 = -0.23;
        let r = 0.71;
        let d = 5.0;
        let eta = -3.5;
        let kappa = psi0.exp();
        let t = kappa * r;
        let eta_q = eta + 2.0;

        let (phi, phi_psi_ad, phi_psi_psi_ad) =
            second_derivative(|psi| scaling_test_phi(psi, r, eta), psi0);
        let (q, q_psi_ad, q_psi_psi_ad) =
            second_derivative(|psi| scaling_test_q(psi, r, eta), psi0);
        let (lap, lap_psi_ad, lap_psi_psi_ad) =
            second_derivative(|psi| scaling_test_lap(psi, r, eta, d), psi0);

        let phi_r = kappa.powf(eta + 1.0) * (2.0 * t + 4.0 * t.powi(3));
        let phi_rr = kappa.powf(eta + 2.0) * (2.0 + 12.0 * t * t);
        let q_r = kappa.powf(eta + 3.0) * (8.0 * t);
        let q_rr = kappa.powf(eta + 4.0) * 8.0;
        let lap_r = kappa.powf(eta + 3.0) * ((8.0 * d + 16.0) * t);
        let lap_rr = kappa.powf(eta + 4.0) * (8.0 * d + 16.0);

        let phi_psi = eta * phi + r * phi_r;
        let phi_psi_psi = eta * eta * phi + (2.0 * eta + 1.0) * r * phi_r + r * r * phi_rr;
        let q_psi = eta_q * q + r * q_r;
        let q_psi_psi = eta_q * eta_q * q + (2.0 * eta_q + 1.0) * r * q_r + r * r * q_rr;
        let lap_psi = eta_q * lap + r * lap_r;
        let lap_psi_psi = eta_q * eta_q * lap + (2.0 * eta_q + 1.0) * r * lap_r + r * r * lap_rr;

        assert!((phi_psi - phi_psi_ad).abs() < 1e-12);
        assert!((phi_psi_psi - phi_psi_psi_ad).abs() < 1e-12);
        assert!((q_psi - q_psi_ad).abs() < 1e-12);
        assert!((q_psi_psi - q_psi_psi_ad).abs() < 1e-12);
        assert!((lap_psi - lap_psi_ad).abs() < 1e-12);
        assert!((lap_psi_psi - lap_psi_psi_ad).abs() < 1e-12);
    }

    #[test]
    fn test_duchon_spectral_scaling_matches_implementation() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale_1 = 1.7;
        let length_scale_2 = 0.85;
        let kappa_1 = 1.0 / length_scale_1;
        let kappa_2 = 1.0 / length_scale_2;
        let scale = kappa_2 / kappa_1;
        let r = 0.43;
        let scaled_r = scale * r;
        let delta = duchon_scaling_exponent(p_order, s_order, k_dim);

        let coeffs_1 = duchon_partial_fraction_coeffs(p_order, s_order, kappa_1);
        let coeffs_2 = duchon_partial_fraction_coeffs(p_order, s_order, kappa_2);

        let phi_1 = duchon_matern_kernel_general_from_distance(
            scaled_r,
            Some(length_scale_1),
            p_order,
            s_order,
            k_dim,
            Some(&coeffs_1),
        )
        .expect("scaled phi_1");
        let phi_2 = duchon_matern_kernel_general_from_distance(
            r,
            Some(length_scale_2),
            p_order,
            s_order,
            k_dim,
            Some(&coeffs_2),
        )
        .expect("phi_2");
        let jets_1 =
            duchon_radial_jets(scaled_r, length_scale_1, p_order, s_order, k_dim, &coeffs_1)
                .expect("jets_1");
        let jets_2 = duchon_radial_jets(r, length_scale_2, p_order, s_order, k_dim, &coeffs_2)
            .expect("jets_2");

        let phi_scale = scale.powf(delta);
        let op_scale = scale.powf(delta + 2.0);
        assert!((phi_2 - phi_scale * phi_1).abs() < 1e-9);
        assert!((jets_2.q - op_scale * jets_1.q).abs() < 1e-8);
        assert!((jets_2.lap - op_scale * jets_1.lap).abs() < 1e-8);

        let core =
            duchon_radial_core_psi_triplet(r, length_scale_2, p_order, s_order, k_dim, &coeffs_2)
                .expect("radial core");
        let q_psi_expected = (delta + 2.0) * jets_2.q + r * jets_2.q_r;
        let q_psi_psi_expected = (delta + 2.0) * (delta + 2.0) * jets_2.q
            + (2.0 * delta + 5.0) * r * jets_2.q_r
            + r * r * jets_2.q_rr;
        let lap_psi_expected = (delta + 2.0) * jets_2.lap + r * jets_2.lap_r;
        let lap_psi_psi_expected = (delta + 2.0) * (delta + 2.0) * jets_2.lap
            + (2.0 * delta + 5.0) * r * jets_2.lap_r
            + r * r * jets_2.lap_rr;

        assert!((core.gradient_ratio.psi - q_psi_expected).abs() < 1e-10);
        assert!((core.gradient_ratio.psi_psi - q_psi_psi_expected).abs() < 1e-9);
        assert!((core.laplacian.psi - lap_psi_expected).abs() < 1e-10);
        assert!((core.laplacian.psi_psi - lap_psi_psi_expected).abs() < 1e-9);
    }

    #[test]
    fn test_gram_and_psi_derivatives_from_operator_matches_fd() {
        // Build D(psi) = D0 + psi D1 + 0.5 psi^2 D2 with nontrivial shape.
        let d0 = array![
            [0.9, -0.2, 0.3],
            [0.4, 0.8, -0.6],
            [0.1, 0.7, 0.5],
            [-0.3, 0.2, 0.4]
        ];
        let d1 = array![
            [0.2, -0.1, 0.05],
            [0.3, 0.07, -0.2],
            [-0.15, 0.06, 0.1],
            [0.04, -0.09, 0.12]
        ];
        let d2 = array![
            [0.08, -0.02, 0.01],
            [0.03, 0.04, -0.05],
            [0.02, -0.01, 0.06],
            [-0.07, 0.03, 0.02]
        ];

        let psi0 = 0.35;
        let d = &d0 + &(d1.mapv(|v| psi0 * v)) + &(d2.mapv(|v| 0.5 * psi0 * psi0 * v));
        let d_psi = &d1 + &(d2.mapv(|v| psi0 * v));
        let d_psi_psi = d2.clone();

        let (s, s_psi, s_psi_psi) = gram_and_psi_derivatives_from_operator(&d, &d_psi, &d_psi_psi);

        let h = 1e-6;
        let eval_s = |psi: f64| {
            let d_eval = &d0 + &(d1.mapv(|v| psi * v)) + &(d2.mapv(|v| 0.5 * psi * psi * v));
            symmetrize(&fast_ata(&d_eval))
        };
        let s_plus = eval_s(psi0 + h);
        let s_minus = eval_s(psi0 - h);
        let s_fd = (&s_plus - &s_minus) / (2.0 * h);
        let s2_fd = (&s_plus - &(s.mapv(|v| 2.0 * v)) + &s_minus) / (h * h);

        let err1 = (&s_psi - &s_fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let err2 = (&s_psi_psi - &s2_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for i in 0..s_psi.nrows() {
            for j in 0..s_psi.ncols() {
                assert_eq!(s_psi[[i, j]].signum(), s_fd[[i, j]].signum());
                assert_eq!(s_psi_psi[[i, j]].signum(), s2_fd[[i, j]].signum());
            }
        }

        assert!(err1 < 2e-6, "S' mismatch too large: {err1}");
        assert!(err2 < 5e-4, "S'' mismatch too large: {err2}");
    }

    #[test]
    fn test_normalize_penalty_with_psi_derivatives_matches_fd() {
        // Build S(psi) = S0 + psi S1 + 0.5 psi^2 S2 and validate exact
        // normalization derivatives against finite differences of S/||S||_F.
        let s0 = array![[2.0, 0.3, -0.2], [0.3, 1.7, 0.4], [-0.2, 0.4, 1.4]];
        let s1 = array![[0.2, -0.05, 0.1], [-0.05, 0.12, 0.03], [0.1, 0.03, -0.08]];
        let s2 = array![
            [0.04, 0.02, -0.01],
            [0.02, -0.03, 0.015],
            [-0.01, 0.015, 0.02]
        ];

        let psi0 = -0.4;
        let s = &s0 + &(s1.mapv(|v| psi0 * v)) + &(s2.mapv(|v| 0.5 * psi0 * psi0 * v));
        let s_psi = &s1 + &(s2.mapv(|v| psi0 * v));
        let s_psi_psi = s2.clone();

        let (_sn, sn_psi, sn_psi_psi, _c) =
            normalize_penalty_with_psi_derivatives(&s, &s_psi, &s_psi_psi);

        let h = 1e-6;
        let eval_snorm = |psi: f64| {
            let s_eval = &s0 + &(s1.mapv(|v| psi * v)) + &(s2.mapv(|v| 0.5 * psi * psi * v));
            let c = trace_of_product(&s_eval, &s_eval).sqrt();
            s_eval.mapv(|v| v / c)
        };
        let sn = eval_snorm(psi0);
        let sn_plus = eval_snorm(psi0 + h);
        let sn_minus = eval_snorm(psi0 - h);
        let sn_fd = (&sn_plus - &sn_minus) / (2.0 * h);
        let sn2_fd = (&sn_plus - &(sn.mapv(|v| 2.0 * v)) + &sn_minus) / (h * h);

        let err1 = (&sn_psi - &sn_fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let err2 = (&sn_psi_psi - &sn2_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for i in 0..sn_psi.nrows() {
            for j in 0..sn_psi.ncols() {
                assert_eq!(sn_psi[[i, j]].signum(), sn_fd[[i, j]].signum());
                assert_eq!(sn_psi_psi[[i, j]].signum(), sn2_fd[[i, j]].signum());
            }
        }

        assert!(err1 < 2e-6, "normalized S' mismatch too large: {err1}");
        assert!(err2 < 5e-4, "normalized S'' mismatch too large: {err2}");
    }

    #[test]
    fn test_duchon_general_p0_case_builds() {
        let data = array![
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.0, 0.1, 0.5],
            [0.4, 0.2, 0.3, 0.1],
            [0.6, 0.4, 0.5, 0.2],
            [0.8, 0.5, 0.7, 0.4]
        ];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.2),
            4,
            DuchonNullspaceOrder::Zero, // p=0
        )
        .expect("p=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, 0);
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_pure_duchon_default_tuple_builds() {
        let data = array![[0.0, 0.1], [0.2, 0.0], [0.4, 0.2], [0.6, 0.4], [0.8, 0.5]];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: None,
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Zero,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
        };
        let out =
            build_duchon_basis(data.view(), &spec).expect("pure Duchon default tuple should build");
        assert!(out.design.iter().all(|v| v.is_finite()));
        assert_eq!(out.penalties.len(), 3);
        assert!(
            out.penalties
                .iter()
                .all(|penalty| penalty.iter().all(|v| v.is_finite()))
        );
    }

    #[test]
    fn test_duchon_general_p1_s0_case_builds() {
        let data = array![
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [0.4, 0.2, 0.3],
            [0.6, 0.4, 0.5],
            [0.8, 0.5, 0.7]
        ];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            0,
            DuchonNullspaceOrder::Linear,
        )
        .expect("p=1, s=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, data.ncols() + 1);
        assert!(out.basis.iter().all(|v| v.is_finite()));
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_duchon_general_p0_s0_case_builds() {
        let data = array![
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [0.4, 0.2, 0.3],
            [0.6, 0.4, 0.5],
            [0.8, 0.5, 0.7]
        ];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            0,
            DuchonNullspaceOrder::Zero,
        )
        .expect("p=0, s=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, 0);
        assert!(out.basis.iter().all(|v| v.is_finite()));
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matern_radial_triplet_matches_finite_difference() {
        let r = 0.37;
        let length_scale = 0.9;
        let nu = MaternNu::FiveHalves;
        let (phi, phi_r, phi_rr) =
            matern_kernel_radial_triplet(r, length_scale, nu).expect("triplet");
        let h = 1e-6;
        let fp = matern_kernel_from_distance(r + h, length_scale, nu).expect("fp");
        let fm = matern_kernel_from_distance((r - h).max(0.0), length_scale, nu).expect("fm");
        let first_fd = (fp - fm) / (2.0 * h);
        let second_fd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), first_fd.signum());
        assert_eq!(phi_rr.signum(), second_fd.signum());
        assert!((phi_r - first_fd).abs() < 5e-5);
        assert!((phi_rr - second_fd).abs() < 1e-3);
    }

    #[test]
    fn test_matern_safe_ratio_matches_closed_form_limits_at_zero() {
        let ls = 1.7;
        let kappa = 1.0 / ls;
        let (_, _, _, r32) =
            matern_kernel_radial_triplet_with_safe_ratio(0.0, ls, MaternNu::ThreeHalves)
                .expect("three-halves");
        let (_, _, _, r52) =
            matern_kernel_radial_triplet_with_safe_ratio(0.0, ls, MaternNu::FiveHalves)
                .expect("five-halves");
        let (_, _, _, r72) =
            matern_kernel_radial_triplet_with_safe_ratio(0.0, ls, MaternNu::SevenHalves)
                .expect("seven-halves");
        let (_, _, _, r92) =
            matern_kernel_radial_triplet_with_safe_ratio(0.0, ls, MaternNu::NineHalves)
                .expect("nine-halves");
        assert!((r32 - (-3.0 * kappa * kappa)).abs() < 1e-12);
        assert!((r52 - (-(5.0 / 3.0) * kappa * kappa)).abs() < 1e-12);
        assert!((r72 - (-(7.0 / 5.0) * kappa * kappa)).abs() < 1e-12);
        assert!((r92 - (-(9.0 / 7.0) * kappa * kappa)).abs() < 1e-12);
    }

    #[test]
    fn test_matern_safe_ratio_half_is_finite_with_floor() {
        let ls = 1.3;
        let (_phi, _phi_r, _phi_rr, ratio) =
            matern_kernel_radial_triplet_with_safe_ratio(0.0, ls, MaternNu::Half).expect("half");
        assert!(ratio.is_finite());
        assert!(ratio < 0.0);
    }

    #[test]
    fn test_duchon_radial_triplet_matches_finite_difference_away_from_zero() {
        let r = 0.42;
        let length_scale = 1.1;
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let dim = 4usize;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
            r,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("triplet");
        let h = 1e-5;
        let fp = duchon_matern_kernel_general_from_distance(
            r + h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fp");
        let fm = duchon_matern_kernel_general_from_distance(
            r - h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fm");
        let first_fd = (fp - fm) / (2.0 * h);
        let second_fd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), first_fd.signum());
        assert_eq!(phi_rr.signum(), second_fd.signum());
        assert!((phi_r - first_fd).abs() < 1e-3);
        assert!((phi_rr - second_fd).abs() < 1e-1);
    }

    #[test]
    fn test_duchon_radial_triplet_closed_form_branch_matches_finite_difference() {
        // p=1,s=4,k=10 uses the exact K0/K1 branch with analytic derivatives.
        let r = 2.0;
        let length_scale = 1.0;
        let p_order = 1usize;
        let s_order = 4usize;
        let dim = 10usize;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
            r,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("triplet");
        let h = 1e-5;
        let fp = duchon_matern_kernel_general_from_distance(
            r + h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fp");
        let fm = duchon_matern_kernel_general_from_distance(
            r - h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fm");
        let first_fd = (fp - fm) / (2.0 * h);
        let second_fd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), first_fd.signum());
        assert_eq!(phi_rr.signum(), second_fd.signum());
        assert!((phi_r - first_fd).abs() < 2e-3);
        assert!(phi_rr.is_finite());
        assert!(second_fd.is_finite());
    }

    #[test]
    fn test_duchon_radial_triplet_pure_polyharmonic_matches_finite_difference() {
        let r = 0.73;
        let length_scale = 1.0;
        let p_order = 1usize;
        let s_order = 0usize;
        let dim = 3usize;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
            r,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("triplet");
        let h = 1e-6;
        let fp = duchon_matern_kernel_general_from_distance(
            r + h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fp");
        let fm = duchon_matern_kernel_general_from_distance(
            r - h,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("fm");
        let first_fd = (fp - fm) / (2.0 * h);
        let second_fd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), first_fd.signum());
        assert_eq!(phi_rr.signum(), second_fd.signum());
        assert!((phi_r - first_fd).abs() < 1e-6);
        assert!((phi_rr - second_fd).abs() < 1e-4);
    }

    #[test]
    fn test_collocation_derivatives_are_finite_at_r_zero() {
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let m_ops = build_matern_collocation_operator_matrices(
            centers.view(),
            None,
            0.8,
            MaternNu::FiveHalves,
            false,
            None,
        )
        .expect("matern ops");
        assert!(m_ops.d1.iter().all(|v| v.is_finite()));
        assert!(m_ops.d2.iter().all(|v| v.is_finite()));

        let d_ops = build_duchon_collocation_operator_matrices(
            centers.view(),
            None,
            Some(0.8),
            3,
            DuchonNullspaceOrder::Linear,
            None,
        )
        .expect("duchon ops");
        assert!(d_ops.d1.iter().all(|v| v.is_finite()));
        assert!(d_ops.d2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matern_collocation_weights_scale_rows_by_sqrt_weight() {
        let centers = array![[0.0, 0.0], [1.0, 0.0]];
        let unit = build_matern_collocation_operator_matrices(
            centers.view(),
            None,
            0.9,
            MaternNu::FiveHalves,
            false,
            None,
        )
        .expect("unit weights");
        let weights = array![4.0, 1.0];
        let weighted = build_matern_collocation_operator_matrices(
            centers.view(),
            Some(weights.view()),
            0.9,
            MaternNu::FiveHalves,
            false,
            None,
        )
        .expect("weighted");
        // First collocation row should scale by sqrt(4)=2.
        for j in 0..unit.d0.ncols() {
            assert!((weighted.d0[[0, j]] - 2.0 * unit.d0[[0, j]]).abs() < 1e-12);
        }
        // Second row has weight 1 -> unchanged.
        for j in 0..unit.d0.ncols() {
            assert!((weighted.d0[[1, j]] - unit.d0[[1, j]]).abs() < 1e-12);
        }
    }
}
