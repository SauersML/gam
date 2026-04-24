use crate::faer_ndarray::{
    FaerEigh, FaerLinalgError, default_rrqr_rank_alpha, fast_ab, fast_ata, fast_atb,
    rrqr_nullspace_basis,
};
use crate::linalg::utils::KahanSum;
use crate::matrix::{ChunkedKernelDesignOperator, CoefficientTransformOperator, DesignMatrix};
use crate::types::RhoPrior;
use faer::Side;
use faer::sparse::{SparseColMat, Triplet};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

/// Wrapper to send a raw pointer across thread boundaries.
/// SAFETY: the caller must ensure the pointer targets disjoint memory per thread.
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    fn add(self, offset: usize) -> *mut f64 {
        unsafe { self.0.add(offset) }
    }
}

#[cfg(test)]
use approx::assert_abs_diff_eq;

/// A comprehensive error type for all operations within the basis module.
#[derive(Error, Debug)]
pub enum BasisError {
    #[error("Spline degree must be at least 1, but was {0}.")]
    InvalidDegree(usize),

    #[error(
        "Spline degree {degree} is too low for derivative order {derivative_order}; need degree >= {minimum_degree}."
    )]
    InsufficientDegreeForDerivative {
        degree: usize,
        derivative_order: usize,
        minimum_degree: usize,
    },

    #[error("Data range is invalid: start ({0}) must be less than or equal to end ({1}).")]
    InvalidRange(f64, f64),

    #[error(
        "Data range has zero width (min equals max), which collapses the B-spline knot domain; requested {0} internal knots."
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
        "Constraint matrix must have the same number of rows as the basis: basis has {basisrows}, constraint has {constraintrows}."
    )]
    ConstraintMatrixRowMismatch {
        basisrows: usize,
        constraintrows: usize,
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

    #[error("{0}")]
    Other(String),
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

    let knotvec: Array1<f64> = match knot_source {
        KnotSource::Provided(view) => view.to_owned(),
        KnotSource::Generate {
            data_range,
            num_internal_knots,
        } => {
            if data_range.0 > data_range.1 {
                return Err(BasisError::InvalidRange(data_range.0, data_range.1));
            }
            if data_range.0 == data_range.1 {
                return Err(BasisError::DegenerateRange(num_internal_knots));
            }
            internal::generate_full_knot_vector(data_range, num_internal_knots, knot_degree)?
        }
    };
    validate_knots_for_degree(knotvec.view(), knot_degree)?;
    validate_knot_spans_nondegenerate(knotvec.view(), knot_degree)?;

    match options.basis_family {
        BasisFamily::BSpline => O::build_basis(data, degree, eval_kind, knotvec),
        BasisFamily::MSpline => {
            if O::IS_SPARSE {
                let sparse = create_mspline_sparse(data, knotvec.view(), degree)?;
                Ok((O::from_sparse(sparse)?, knotvec))
            } else {
                let dense = create_mspline_dense(data, knotvec.view(), degree)?;
                Ok((O::from_dense(dense)?, knotvec))
            }
        }
        BasisFamily::ISpline => {
            if O::IS_SPARSE {
                return Err(BasisError::InvalidInput(
                    "BasisFamily::ISpline does not support sparse output; use Dense".to_string(),
                ));
            }
            let dense = create_ispline_dense(data, knotvec.view(), degree)?;
            Ok((O::from_dense(dense)?, knotvec))
        }
    }
}

/// Applies first-order linear extension outside a knot-domain interval to a basis matrix
/// that was evaluated at clamped coordinates.
///
/// Given `z_raw` and `z_clamped = clamp(z_raw, left, right)`, this mutates
/// `basisvalues` in-place as:
/// `B_ext(z_raw) = B(z_clamped) + (z_raw - z_clamped) * B'(z_clamped)`.
pub fn apply_linear_extension_from_first_derivative(
    z_raw: ArrayView1<f64>,
    z_clamped: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    basisvalues: &mut Array2<f64>,
) -> Result<(), BasisError> {
    if z_raw.len() != z_clamped.len() {
        return Err(BasisError::DimensionMismatch(
            "z_raw and z_clamped must have equal length".to_string(),
        ));
    }
    if basisvalues.nrows() != z_raw.len() {
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
    if b_prime.nrows() != basisvalues.nrows() || b_prime.ncols() != basisvalues.ncols() {
        return Err(BasisError::DimensionMismatch(
            "basis derivative shape mismatch".to_string(),
        ));
    }

    for i in 0..z_raw.len() {
        let dz = z_raw[i] - z_clamped[i];
        if dz.abs() <= 1e-12 {
            continue;
        }
        for j in 0..basisvalues.ncols() {
            basisvalues[[i, j]] += dz * b_prime[[i, j]];
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
        knotvec: Array1<f64>,
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
        knotvec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError> {
        let knotview = knotvec.view();

        let num_basis_functions = knotview.len().saturating_sub(degree + 1);
        let basis_matrix = if should_use_sparse_basis(num_basis_functions, degree, 1) {
            let left = knotview[degree];
            let right = knotview[num_basis_functions];
            let data_clamped = data.mapv(|x| x.clamp(left, right));
            let sparse = generate_basis_internal::<SparseStorage>(
                data_clamped.view(),
                knotview,
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
            apply_dense_bspline_extrapolation(data, knotview, degree, eval_kind, &mut dense)?;
            dense
        } else {
            generate_basis_internal::<DenseStorage>(data.view(), knotview, degree, eval_kind)?
        };

        Ok((Arc::new(basis_matrix), knotvec))
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
    knotview: ArrayView1<f64>,
    degree: usize,
    eval_kind: BasisEvalKind,
    basis_matrix: &mut Array2<f64>,
) -> Result<(), BasisError> {
    let num_basis_functions = basis_matrix.ncols();
    if num_basis_functions == 0 {
        return Ok(());
    }

    let left = knotview[degree];
    let right = knotview[num_basis_functions];

    if matches!(eval_kind, BasisEvalKind::FirstDerivative) {
        let num_basis_lower = knotview.len().saturating_sub(degree);
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
                knotview,
                degree,
                row_slice,
                &mut lower_basis,
                &mut lower_scratch,
            )?;
        }
    }

    if matches!(eval_kind, BasisEvalKind::SecondDerivative) {
        for (i, &x) in data.iter().enumerate() {
            if x < left || x > right {
                basis_matrix.row_mut(i).fill(0.0);
            }
        }
    }

    if matches!(eval_kind, BasisEvalKind::Basis) {
        let z_clamped = data.mapv(|x| x.clamp(left, right));
        apply_linear_extension_from_first_derivative(
            data,
            z_clamped.view(),
            knotview,
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
        knotvec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError> {
        let knotview = knotvec.view();
        let sparse =
            generate_basis_internal::<SparseStorage>(data.view(), knotview, degree, eval_kind)?;
        Ok((sparse, knotvec))
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

fn validate_knots_for_degree(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }

    let required_knots = 2 * (degree + 1);
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

/// Rejects knot vectors whose effective basis functions have zero support
/// (i.e. `t[i+degree+1] == t[i]` for any `i`). This is stricter than the
/// structural `validate_knots_for_degree` and is only appropriate at the
/// user-facing top-level of basis construction — the recursive derivative
/// evaluators repeatedly call `validate_knots_for_degree` with a reduced
/// `degree` on the *same* (clamped) knot vector, where the outermost lower-
/// degree "basis function" always collapses to zero support by construction
/// and is harmless because the derivative recursion guards the matching
/// `1/(t_{i+k}-t_i)` denominator with an absolute-value check.
fn validate_knot_spans_nondegenerate(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    if knot_vector.len() <= degree + 1 {
        return Ok(());
    }
    let num_basis = knot_vector.len() - degree - 1;
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        if span <= 1e-12 {
            return Err(BasisError::InvalidKnotVector(format!(
                "basis function {i} has zero support: t[i+degree+1]-t[i]={span:.3e} must be > 0"
            )));
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

fn evaluate_splines_derivative_sparse_intowith_lower(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    basis_scratch: &mut internal::BsplineScratch,
    lowervalues: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> usize {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    let x_eval = if num_basis > 0 {
        let left = knotview[degree];
        let right = knotview[num_basis];
        x.clamp(left, right)
    } else {
        x
    };
    // Linear extrapolation outside the domain uses the boundary slope, so
    // first derivatives clamp to the nearest boundary derivative value.

    let start_col =
        internal::evaluate_splines_sparse_into(x_eval, degree, knotview, values, basis_scratch);
    if degree == 0 {
        values.fill(0.0);
        return start_col;
    }

    let lower_degree = degree - 1;
    let lower_support = lower_degree + 1;
    if lowervalues.len() != lower_support {
        return start_col;
    }

    let start_lower = internal::evaluate_splines_sparse_into(
        x_eval,
        lower_degree,
        knotview,
        lowervalues,
        lower_scratch,
    );

    values.fill(0.0);
    for offset in 0..=degree {
        let i = start_col + offset;
        let left_idx = i as isize - start_lower as isize;
        let right_idx = (i + 1) as isize - start_lower as isize;
        let left = if left_idx >= 0 && (left_idx as usize) < lower_support {
            lowervalues[left_idx as usize]
        } else {
            0.0
        };
        let right = if right_idx >= 0 && (right_idx as usize) < lower_support {
            lowervalues[right_idx as usize]
        } else {
            0.0
        };
        let denom_left = knotview[i + degree] - knotview[i];
        let denom_right = knotview[i + degree + 1] - knotview[i + 1];
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
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let lower_degree = degree.saturating_sub(1);
    let lower_support = lower_degree + 1;
    if scratch.lower_basis.len() != lower_support {
        scratch.lower_basis.resize(lower_support, 0.0);
        scratch.lower_scratch.ensure_degree(lower_degree);
    }
    evaluate_splines_derivative_sparse_intowith_lower(
        x,
        degree,
        knotview,
        values,
        &mut scratch.basis,
        &mut scratch.lower_basis,
        &mut scratch.lower_scratch,
    )
}

fn evaluate_splinessecond_derivative_sparse_into(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis > 0 {
        let left = knotview[degree];
        let right = knotview[num_basis];
        // Constant extrapolation outside the domain implies zero derivatives.
        if x < left || x > right {
            values.fill(0.0);
            return 0;
        }
    }

    let start_col =
        internal::evaluate_splines_sparse_into(x, degree, knotview, values, &mut scratch.basis);
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
    let start_lower = evaluate_splines_derivative_sparse_intowith_lower(
        x,
        lower_degree,
        knotview,
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
        let denom_left = knotview[i + degree] - knotview[i];
        let denom_right = knotview[i + degree + 1] - knotview[i + 1];
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

fn evaluate_splines_sparsewith_kind(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    eval_kind: BasisEvalKind,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    match eval_kind {
        BasisEvalKind::Basis => {
            internal::evaluate_splines_sparse_into(x, degree, knotview, values, &mut scratch.basis)
        }
        BasisEvalKind::FirstDerivative => {
            evaluate_splines_derivative_sparse_into(x, degree, knotview, values, scratch)
        }
        BasisEvalKind::SecondDerivative => {
            evaluate_splinessecond_derivative_sparse_into(x, degree, knotview, values, scratch)
        }
    }
}

fn evaluate_bsplinerow_entries<F>(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    eval_kind: BasisEvalKind,
    num_basis_functions: usize,
    scratch: &mut BasisEvalScratch,
    values: &mut [f64],
    mut write_entry: F,
) where
    F: FnMut(usize, f64),
{
    let start_col =
        evaluate_splines_sparsewith_kind(x, degree, knotview, eval_kind, values, scratch);
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
        knotview: ArrayView1<f64>,
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
        knotview: ArrayView1<f64>,
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
                        evaluate_bsplinerow_entries(
                            x,
                            degree,
                            knotview,
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
                evaluate_bsplinerow_entries(
                    x,
                    degree,
                    knotview,
                    eval_kind,
                    num_basis_functions,
                    &mut scratch,
                    &mut values,
                    |col_j, v| row_slice[col_j] = v,
                );
            }
        }

        apply_dense_bspline_extrapolation(data, knotview, degree, eval_kind, &mut basis_matrix)?;

        Ok(basis_matrix)
    }
}

struct SparseStorage;

impl BasisStorage for SparseStorage {
    type Output = SparseColMat<usize, f64>;

    fn build(
        data: ArrayView1<f64>,
        knotview: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        num_basis_functions: usize,
        support: usize,
        use_parallel: bool,
    ) -> Result<Self::Output, BasisError> {
        let nrows = data.len();
        let left = knotview[degree];
        let right = knotview[num_basis_functions];
        let needs_extrapolation = data.iter().any(|&x| x < left || x > right);
        if needs_extrapolation {
            let dense = DenseStorage::build(
                data,
                knotview,
                degree,
                eval_kind,
                num_basis_functions,
                support,
                use_parallel,
            )?;
            return Sparse::from_dense(dense);
        }

        let triplets: Vec<Triplet<usize, usize, f64>> =
            if let (true, Some(data_slice)) = (use_parallel, data.as_slice()) {
                const CHUNK_SIZE: usize = 1024;
                let triplet_chunks: Vec<Vec<Triplet<usize, usize, f64>>> = data_slice
                    .par_chunks(CHUNK_SIZE)
                    .enumerate()
                    .map_init(
                        || (BasisEvalScratch::new(degree), vec![0.0; support]),
                        |(scratch, values), (chunk_idx, chunk)| {
                            let baserow = chunk_idx * CHUNK_SIZE;
                            let mut local = Vec::with_capacity(chunk.len().saturating_mul(support));
                            for (i, &x) in chunk.iter().enumerate() {
                                let row_i = baserow + i;
                                evaluate_bsplinerow_entries(
                                    x,
                                    degree,
                                    knotview,
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
                    evaluate_bsplinerow_entries(
                        x,
                        degree,
                        knotview,
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
    knotview: ArrayView1<f64>,
    degree: usize,
    eval_kind: BasisEvalKind,
) -> Result<S::Output, BasisError> {
    let num_basis_functions = knotview.len().saturating_sub(degree + 1);
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
        knotview,
        degree,
        eval_kind,
        num_basis_functions,
        support,
        use_parallel,
    )
}

/// Returns true if the B-spline basis should be built in sparse form based on density.
pub fn should_use_sparse_basis(num_basis_cols: usize, degree: usize, dim: usize) -> bool {
    if num_basis_cols == 0 {
        return false;
    }

    let support_perrow = (degree + 1).saturating_pow(dim as u32) as f64;
    let density = support_perrow / num_basis_cols as f64;

    density < 0.20 && num_basis_cols > 32
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
                if span.abs() <= 1e-12 {
                    return Err(BasisError::InvalidKnotVector(format!(
                        "singular divided-difference span at order {o}, row {i}: Greville abscissae g[{}]={:.6e} and g[{i}]={:.6e} collapse",
                        i + o,
                        g[i + o],
                        g[i]
                    )));
                }
                let mut row = d.row_mut(i);
                row /= span;
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
/// - `P` is the TPS polynomial null-space block containing all monomials of
///   total degree `< m`, where `m = thin_plate_penalty_order(d)` (so `P` is
///   just `[1, x_1, ..., x_d]` for `d <= 3`)
///
/// The returned penalty matrix is block-diagonal with:
/// - upper-left `Omega_c = Z^T Omega Z` for the constrained radial block
/// - zero lower-right block for unpenalized polynomial terms.
///
/// For double-penalty GAMs, a second ridge penalty `I` is also returned so the
/// caller can optimize `(lambda_bending, lambdaridge)` jointly.
#[derive(Debug, Clone)]
pub struct ThinPlateSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_bending: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
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

/// Duchon radial basis with triple operator regularization and explicit
/// low-frequency null-space order. The penalty blocks are collocation
/// operator Gram matrices, not the native spectral Duchon seminorm.
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

#[derive(Debug, Clone)]
struct DuchonBasisDesign {
    basis: Array2<f64>,
    num_kernel_basis: usize,
    num_polynomial_basis: usize,
    dimension: usize,
    /// Effective null-space order actually used to build the basis. May
    /// differ from the requested order when auto-degraded to `Zero` because
    /// the center count could not span the requested polynomial block.
    nullspace_order: DuchonNullspaceOrder,
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
    pub knotspec: BSplineKnotSpec,
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
    Auto(Box<CenterStrategy>),
    UserProvided(Array2<f64>),
    /// Joint multidimensional equal-mass partitioning in the full smooth space.
    EqualMass {
        num_centers: usize,
    },
    /// Covariate-representative equal-mass partitioning along one selected axis.
    EqualMassCovarRepresentative {
        num_centers: usize,
    },
    FarthestPoint {
        num_centers: usize,
    },
    KMeans {
        num_centers: usize,
        max_iter: usize,
    },
    UniformGrid {
        points_per_dim: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CenterStrategyKind {
    UserProvided,
    EqualMass,
    EqualMassCovarRepresentative,
    FarthestPoint,
    KMeans,
    UniformGrid,
}

/// Adaptive default center count for spatial smooths (TPS, Duchon, Matérn).
///
/// Use this when the user has not explicitly specified a knot/center count.
/// The formula `min(2000, max(200, ceil(8 * d_factor * n^0.4)))` scales
/// sub-linearly with sample size and gives a mild boost for higher input
/// dimensionality:
///
/// | n      | d=1  | d=2  | d=5  |
/// |--------|------|------|------|
/// | 1 000  | 200  | 200  | 200  |
/// | 10 000 | 320  | 368  | 512  |
/// | 100 000| 800  | 920  | 1280 |
/// | 400 000| 1240 | 1426 | 1984 |
/// | 1 000 000| 1600 | 1840 | 2000 |
///
/// # Arguments
/// * `n` - sample size (number of observations)
/// * `d` - covariate dimensionality (number of input variables in the smooth)
pub fn default_num_centers(n: usize, d: usize) -> usize {
    const K_MIN: usize = 200;
    const K_MAX: usize = 2000;
    const ALPHA: f64 = 0.4;
    const C: f64 = 8.0;

    let d_factor = 1.0 + 0.15 * (d.max(1) - 1) as f64;
    let raw = (C * d_factor * (n as f64).powf(ALPHA)).ceil() as usize;
    let k = raw.clamp(K_MIN, K_MAX);

    // Never exceed n itself; on small datasets cap at n/4 to keep
    // penalty matrices well-conditioned relative to data.
    let small_data_cap = if n < 800 { n / 4 } else { n };
    k.min(n).min(small_data_cap)
}

/// Resource-aware plan for a spatial smooth (Duchon / Matérn / TPS).
///
/// Returned by [`plan_spatial_basis`]. Captures the resolved center count,
/// final basis dimension `p`, the dense byte cost for the value matrix and
/// each derivative tier, and a recommended storage mode that is consistent
/// with the supplied [`crate::resource::ResourcePolicy`].
#[derive(Clone, Debug)]
pub struct SpatialBasisPlan {
    pub n: usize,
    pub d: usize,
    pub centers: usize,
    pub p_final_estimate: usize,
    pub dense_design_bytes: usize,
    pub first_derivative_dense_bytes: usize,
    pub second_derivative_dense_bytes: usize,
    pub recommended_storage: SpatialStorageMode,
}

/// Storage mode recommended by [`plan_spatial_basis`].
///
/// * `DenseValueDenseDerivatives` — both the value design and its derivative
///   matrices fit under the policy's single-materialization budget.
/// * `LazyValueImplicitDerivatives` — the value design fits dense but the
///   derivative matrices do not; switch derivatives to the implicit operator.
/// * `OperatorOnly` — neither the design nor its derivatives fit; everything
///   must be operator-backed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialStorageMode {
    DenseValueDenseDerivatives,
    LazyValueImplicitDerivatives,
    OperatorOnly,
}

/// How [`plan_spatial_basis`] should pick the spatial center count.
#[derive(Clone, Copy, Debug)]
pub enum CenterCountRequest {
    /// Use the heuristic [`default_num_centers`].
    Default,
    /// Use the caller-supplied count exactly.
    Explicit(usize),
    /// Use [`default_num_centers`] but cap at `cap` to bound dense cost.
    HeuristicCapped { cap: usize },
}

/// Build a resource-aware plan for a spatial smooth basis.
///
/// Computes the resolved center count, final basis dimension, dense byte
/// estimates for the value design and first/second derivative tiers, and a
/// recommended [`SpatialStorageMode`] derived from `policy`. This is the
/// resource-aware replacement for ad-hoc calls to [`default_num_centers`] /
/// [`heuristic_centers`](crate::terms::term_builder::heuristic_centers).
pub fn plan_spatial_basis(
    n: usize,
    d: usize,
    requested_centers: CenterCountRequest,
    nullspace_order: DuchonNullspaceOrder,
    scale_dims: bool,
    policy: &crate::resource::ResourcePolicy,
) -> Result<SpatialBasisPlan, BasisError> {
    if n == 0 {
        return Err(BasisError::InvalidInput(
            "plan_spatial_basis: n must be >= 1".to_string(),
        ));
    }
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "plan_spatial_basis: d must be >= 1".to_string(),
        ));
    }

    // 1. Resolve center count.
    let centers = match requested_centers {
        CenterCountRequest::Default => default_num_centers(n, d),
        CenterCountRequest::Explicit(k) => k,
        CenterCountRequest::HeuristicCapped { cap } => default_num_centers(n, d).min(cap),
    };

    // 2. Nullspace dimension (Duchon polynomial null space of degree p-1).
    //    `duchon_p_from_nullspace_order` returns m such that the null space is
    //    polynomials of total degree < m, matching `duchon_nullspace_dimension`'s
    //    `max_total_degree = m - 1` argument.
    let m = duchon_p_from_nullspace_order(nullspace_order);
    let nullspace_dim = if m == 0 {
        0
    } else {
        duchon_nullspace_dimension(d, m - 1)
    };

    let p = centers.saturating_add(nullspace_dim);

    // 3. Dense byte estimates.
    let derivative_axes = if scale_dims { d } else { 0 };
    let bytes_per_f64 = std::mem::size_of::<f64>();
    let dense_design_bytes = bytes_per_f64.saturating_mul(n).saturating_mul(p);
    let first_derivative_dense_bytes = dense_design_bytes.saturating_mul(derivative_axes);
    // Diagonal second derivatives are also (D × n × p); off-diagonal cross terms
    // would scale as D^2 but the planner reports the diagonal tier here.
    let second_derivative_dense_bytes = first_derivative_dense_bytes;

    // 4. Pick storage mode based on policy.
    let recommended_storage = match policy.derivative_storage_mode {
        crate::resource::DerivativeStorageMode::AnalyticOperatorRequired => {
            SpatialStorageMode::OperatorOnly
        }
        crate::resource::DerivativeStorageMode::MaterializeIfSmall => {
            let budget = policy.max_single_materialization_bytes;
            if derivative_axes == 0 {
                if dense_design_bytes <= budget {
                    SpatialStorageMode::DenseValueDenseDerivatives
                } else {
                    SpatialStorageMode::LazyValueImplicitDerivatives
                }
            } else {
                let total = dense_design_bytes
                    .saturating_add(first_derivative_dense_bytes)
                    .saturating_add(second_derivative_dense_bytes);
                if total <= budget {
                    SpatialStorageMode::DenseValueDenseDerivatives
                } else if dense_design_bytes <= budget {
                    SpatialStorageMode::LazyValueImplicitDerivatives
                } else {
                    SpatialStorageMode::OperatorOnly
                }
            }
        }
        crate::resource::DerivativeStorageMode::DiagnosticsOnly => {
            // Diagnostic mode still prefers analytic storage for correctness.
            SpatialStorageMode::OperatorOnly
        }
    };

    Ok(SpatialBasisPlan {
        n,
        d,
        centers,
        p_final_estimate: p,
        dense_design_bytes,
        first_derivative_dense_bytes,
        second_derivative_dense_bytes,
        recommended_storage,
    })
}

pub fn default_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    if d >= 4 {
        CenterStrategy::EqualMassCovarRepresentative { num_centers }
    } else {
        CenterStrategy::EqualMass { num_centers }
    }
}

pub fn auto_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    CenterStrategy::Auto(Box::new(default_spatial_center_strategy(num_centers, d)))
}

pub fn center_strategy_is_auto(strategy: &CenterStrategy) -> bool {
    matches!(strategy, CenterStrategy::Auto(_))
}

fn realized_center_strategy(strategy: &CenterStrategy) -> &CenterStrategy {
    match strategy {
        CenterStrategy::Auto(inner) => inner.as_ref(),
        other => other,
    }
}

pub fn center_strategy_kind(strategy: &CenterStrategy) -> CenterStrategyKind {
    match realized_center_strategy(strategy) {
        CenterStrategy::UserProvided(_) => CenterStrategyKind::UserProvided,
        CenterStrategy::EqualMass { .. } => CenterStrategyKind::EqualMass,
        CenterStrategy::EqualMassCovarRepresentative { .. } => {
            CenterStrategyKind::EqualMassCovarRepresentative
        }
        CenterStrategy::FarthestPoint { .. } => CenterStrategyKind::FarthestPoint,
        CenterStrategy::KMeans { .. } => CenterStrategyKind::KMeans,
        CenterStrategy::UniformGrid { .. } => CenterStrategyKind::UniformGrid,
        CenterStrategy::Auto(_) => unreachable!("realized center strategy must not be nested auto"),
    }
}

pub fn center_strategy_num_centers(strategy: &CenterStrategy) -> Option<usize> {
    match realized_center_strategy(strategy) {
        CenterStrategy::UserProvided(centers) => Some(centers.nrows()),
        CenterStrategy::EqualMass { num_centers }
        | CenterStrategy::EqualMassCovarRepresentative { num_centers }
        | CenterStrategy::FarthestPoint { num_centers }
        | CenterStrategy::KMeans { num_centers, .. } => Some(*num_centers),
        CenterStrategy::UniformGrid { .. } => None,
        CenterStrategy::Auto(_) => unreachable!("realized center strategy must not be nested auto"),
    }
}

pub fn center_strategy_with_num_centers(
    strategy: &CenterStrategy,
    num_centers: usize,
) -> Result<CenterStrategy, BasisError> {
    validate_center_count(num_centers)?;
    let rebuilt = match realized_center_strategy(strategy) {
        CenterStrategy::EqualMass { .. } => CenterStrategy::EqualMass { num_centers },
        CenterStrategy::EqualMassCovarRepresentative { .. } => {
            CenterStrategy::EqualMassCovarRepresentative { num_centers }
        }
        CenterStrategy::FarthestPoint { .. } => CenterStrategy::FarthestPoint { num_centers },
        CenterStrategy::KMeans { max_iter, .. } => CenterStrategy::KMeans {
            num_centers,
            max_iter: *max_iter,
        },
        CenterStrategy::UserProvided(_) | CenterStrategy::UniformGrid { .. } => {
            Err(BasisError::InvalidInput(format!(
                "cannot replace center count for {:?} strategy",
                center_strategy_kind(strategy)
            )))?
        }
        CenterStrategy::Auto(_) => unreachable!("realized center strategy must not be nested auto"),
    };
    Ok(rebuilt)
}

/// Thin-plate basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinPlateBasisSpec {
    pub center_strategy: CenterStrategy,
    pub length_scale: f64,
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
    /// Per-axis anisotropy log-scales η_a (contrasts with Ση_a = 0).
    ///
    /// This implements geometric anisotropy: Λ = κA where A = diag(exp(η_a)),
    /// det(A) = 1. The kernel is evaluated at r = κ|Ah| instead of r = κ|h|.
    /// The decomposition preserves the isotropic scaling law for global κ
    /// and adds d−1 shape parameters for directional relevance.
    ///
    /// Conditional positive definiteness is preserved under any invertible
    /// linear coordinate transform (Schoenberg), so the kernel remains valid.
    ///
    /// When Some, the distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
    /// When None, isotropic distance r = ‖x - c‖ is used.
    #[serde(default)]
    pub aniso_log_scales: Option<Vec<f64>>,
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

/// Duchon null-space polynomial degree.
/// `0` keeps the constant null space (`m=1`), `1` keeps
/// `[1, x_1, ..., x_d]` (`m=2`), and `Degree(k)` keeps all monomials
/// with total degree <= k.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DuchonNullspaceOrder {
    Zero,
    Linear,
    Degree(usize),
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
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
    /// Per-axis anisotropy log-scales η_a.
    ///
    /// For hybrid Duchon (`length_scale=Some`), these are centered contrasts in
    /// the decomposition Λ = κA with det(A)=1. For pure Duchon
    /// (`length_scale=None`), they parameterize shape-only axis warping on the
    /// public path and are centered before basis evaluation/writeback so no
    /// global length scale is introduced.
    ///
    /// When Some, the distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
    /// When None, isotropic distance r = ‖x - c‖ is used.
    #[serde(default)]
    pub aniso_log_scales: Option<Vec<f64>>,
    #[serde(default)]
    pub operator_penalties: DuchonOperatorPenaltySpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuchonOperatorPenaltySpec {
    pub mass: OperatorPenaltySpec,
    pub tension: OperatorPenaltySpec,
    pub stiffness: OperatorPenaltySpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OperatorPenaltySpec {
    Active {
        initial_log_lambda: f64,
        prior: Option<RhoPrior>,
    },
    Disabled,
}

impl Default for DuchonOperatorPenaltySpec {
    fn default() -> Self {
        Self {
            mass: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            tension: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            stiffness: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
        }
    }
}

pub fn minimum_duchon_power_for_operator_penalties(
    dim: usize,
    nullspace_order: DuchonNullspaceOrder,
    max_operator_derivative_order: usize,
) -> usize {
    let p = duchon_p_from_nullspace_order(nullspace_order);
    let mut s = 0usize;
    while 2 * (p + s) <= dim + max_operator_derivative_order {
        s += 1;
    }
    s
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
        length_scale: f64,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
    },
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        nu: MaternNu,
        include_intercept: bool,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a for geometric anisotropy.
        /// When Some, distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
        aniso_log_scales: Option<Vec<f64>>,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        power: usize,
        nullspace_order: DuchonNullspaceOrder,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a, stored for prediction.
        aniso_log_scales: Option<Vec<f64>>,
    },
    TensorBSpline {
        feature_cols: Vec<usize>,
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        identifiability_transform: Option<Array2<f64>>,
    },
}

/// Standardized basis build result for engine-level composition.
#[derive(Clone, Debug)]
pub struct BasisBuildResult {
    pub design: DesignMatrix,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyInfo>,
    pub metadata: BasisMetadata,
    /// Optional factored rowwise-Kronecker representation for tensor-product
    /// bases. When present, downstream code can keep the design operator-backed
    /// instead of forcing a fully materialized `n x prod(q_j)` block.
    pub kronecker_factored: Option<KroneckerFactoredBasis>,
}

/// Factored tensor-product basis metadata for operator-backed downstream use.
#[derive(Debug, Clone)]
pub struct KroneckerFactoredBasis {
    /// Marginal design matrices: `marginal_designs[j]` is `(n, q_j)`.
    pub marginal_designs: Vec<Array2<f64>>,
    /// Marginal penalty matrices: `marginal_penalties[k]` is `(q_k, q_k)`.
    pub marginal_penalties: Vec<Array2<f64>>,
    /// Marginal basis dimensions: `[q_0, ..., q_{d-1}]`.
    pub marginal_dims: Vec<usize>,
    /// Whether the system includes a global ridge (double) penalty.
    pub has_double_penalty: bool,
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
    /// Kronecker factors preserved from tensor penalty construction.
    /// When present, spectral decomposition can use per-factor eigendecomposition.
    #[serde(skip)]
    pub kronecker_factors: Option<Vec<Array2<f64>>>,
}

#[derive(Debug, Clone)]
pub struct PenaltyCandidate {
    pub matrix: Array2<f64>,
    pub nullspace_dim_hint: usize,
    pub source: PenaltySource,
    pub normalization_scale: f64,
    /// Optional Kronecker factors whose product equals `matrix`.
    /// When present, spectral decomposition can be done per-factor
    /// (O(Σ q_j³) instead of O((Π q_j)³)).
    pub kronecker_factors: Option<Vec<Array2<f64>>>,
}

#[derive(Debug, Clone)]
pub struct CanonicalPenaltyBlock {
    pub sym_penalty: Array2<f64>,
    /// Eigenvalues from spectral decomposition (retained to avoid recomputation).
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors from spectral decomposition (retained to avoid recomputation).
    pub eigenvectors: Array2<f64>,
    pub rank: usize,
    pub nullity: usize,
    pub tol: f64,
    pub iszero: bool,
}

#[derive(Debug, Clone)]
pub struct BasisPsiDerivativeResult {
    pub design_derivative: Array2<f64>,
    pub penalties_derivative: Vec<Array2<f64>>,
    /// Shared operator-backed design derivative. When present, callers may
    /// avoid materializing or consuming the dense `design_derivative`.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

#[derive(Debug, Clone)]
pub struct BasisPsiSecondDerivativeResult {
    pub designsecond_derivative: Array2<f64>,
    pub penaltiessecond_derivative: Vec<Array2<f64>>,
    /// Shared operator-backed design derivative. When present, callers may
    /// avoid materializing or consuming the dense `designsecond_derivative`.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

/// Per-axis psi_a derivative package for anisotropic spatial terms.
///
/// For a d-dimensional anisotropic term, the kernel phi(r) depends on
/// the anisotropic distance r = |Lambda h| where Lambda = diag(kappa_a). Each axis a
/// has its own log-scale psi_a = log(kappa_a), yielding d first derivatives,
/// d diagonal second derivatives, and d*(d-1)/2 cross second derivatives.
///
/// The cross second derivative d2 phi/(d psi_a d psi_b) = t * s_a * s_b (a != b)
/// is rank-1, so we store the t_values and s_components vectors rather
/// than materializing d^2 matrices.
#[derive(Clone)]
pub struct AnisoBasisPsiDerivatives {
    /// d matrices, each (n x p_smooth): dX/d psi_a.
    pub design_first: Vec<Array2<f64>>,
    /// d matrices, each (n x p_smooth): d2X/d psi_a^2 (diagonal second derivatives).
    pub design_second_diag: Vec<Array2<f64>>,
    /// Cross second derivatives d2X/(d psi_a d psi_b) for a < b.
    pub design_second_cross: Vec<Array2<f64>>,
    /// Axis-pair indices corresponding to `design_second_cross`.
    pub design_second_cross_pairs: Vec<(usize, usize)>,
    /// d x num_penalties: dS_m/d psi_a for each axis a and penalty m.
    pub penalties_first: Vec<Vec<Array2<f64>>>,
    /// d x num_penalties: d2S_m/d psi_a^2 for each axis a and penalty m.
    pub penalties_second_diag: Vec<Vec<Array2<f64>>>,
    /// The (a, b) axis pairs supported by the on-demand cross-penalty
    /// provider. Only the upper triangle (a < b) is stored.
    pub penalties_cross_pairs: Vec<(usize, usize)>,
    /// On-demand cross-penalty second-derivative provider. Exact anisotropic
    /// cross-axis penalty seconds are streamed one pair at a time rather than
    /// stored as a dense upper triangle of blocks.
    pub penalties_cross_provider: Option<AnisoPenaltyCrossProvider>,
    /// Shared operator-backed representation of the anisotropic kernel-side
    /// design derivatives. When `design_first` / `design_second_diag` are empty,
    /// callers must use this operator directly; when they are present, this
    /// operator still provides exact cross-axis second derivatives without
    /// duplicating separate `t` / `s_a` storage layouts.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

#[derive(Clone)]
pub struct AnisoPenaltyCrossProvider(
    std::sync::Arc<
        dyn Fn(usize, usize) -> Result<Vec<Array2<f64>>, BasisError> + Send + Sync + 'static,
    >,
);

impl AnisoPenaltyCrossProvider {
    fn new<F>(f: F) -> Self
    where
        F: Fn(usize, usize) -> Result<Vec<Array2<f64>>, BasisError> + Send + Sync + 'static,
    {
        Self(std::sync::Arc::new(f))
    }

    pub fn evaluate(&self, axis_a: usize, axis_b: usize) -> Result<Vec<Array2<f64>>, BasisError> {
        (self.0)(axis_a, axis_b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Implicit derivative operator for scalable anisotropic REML gradients
// ═══════════════════════════════════════════════════════════════════════════

const SPATIAL_DATA_CENTER_DISTANCE_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024; // 256 MiB
const SPATIAL_CENTER_CENTER_MAX_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
const DESIGN_CROSS_CHUNK_SIZE: usize = 1024;

/// Determine whether implicit operators should be used based on problem size
/// and the supplied [`ResourcePolicy`].
///
/// Returns `true` when the dense materialization of D first-derivative
/// matrices would exceed `policy.max_single_materialization_bytes`.
///
/// For D axes with n data points and p_smooth basis columns, the dense path
/// allocates D * n * p_smooth * 8 bytes for first-derivative matrices alone
/// (plus a similar amount for second derivatives). The implicit path stores
/// only the compact (n * n_knots) radial jets plus (n * n_knots * D) axis
/// fractions, which is O(n * k * D) instead of O(n * p * D).
pub fn should_use_implicit_operators_with_policy(
    n: usize,
    p: usize,
    d: usize,
    policy: &crate::resource::ResourcePolicy,
) -> bool {
    // Each first-derivative matrix is (n x p) f64 → n*p*8 bytes.
    // We need D of them for first derivatives, D for second diag, plus
    // the cross-t matrix and s_components. Conservative estimate: 3*D matrices.
    let dense_bytes = 3usize
        .saturating_mul(n)
        .saturating_mul(p)
        .saturating_mul(d)
        .saturating_mul(8);
    dense_bytes > policy.max_single_materialization_bytes
}

/// Backwards-compatible wrapper around [`should_use_implicit_operators_with_policy`]
/// that uses the default library [`crate::resource::ResourcePolicy`].
pub fn should_use_implicit_operators(n: usize, p: usize, d: usize) -> bool {
    // TODO(resource-policy-migration): thread policy through callers
    should_use_implicit_operators_with_policy(
        n,
        p,
        d,
        &crate::resource::ResourcePolicy::default_library(),
    )
}

pub fn assert_no_dense_derivative_materialization(n: usize, p: usize, d_pc: usize) {
    let first = dense_design_bytes(n, p).saturating_mul(d_pc);
    let second = dense_design_bytes(n, p).saturating_mul(d_pc.saturating_mul(d_pc));
    panic!(
        "spatial PC Duchon derivative designs must remain operator-backed; refused persistent dense derivative materialization (n={n}, p={p}, d_pc={d_pc}, first_order={:.1} MiB, second_order={:.1} MiB)",
        first as f64 / (1024.0 * 1024.0),
        second as f64 / (1024.0 * 1024.0),
    );
}

pub fn assert_spatial_centers_below_biobank_cap(
    n: usize,
    d_pc: usize,
    centers: ArrayView2<'_, f64>,
) {
    assert_eq!(
        centers.ncols(),
        d_pc,
        "spatial PC center dimension mismatch: centers have {} columns, expected {d_pc}",
        centers.ncols()
    );
    let k = centers.nrows();
    let centers_bytes = dense_design_bytes(k, d_pc);
    let center_center_bytes = dense_design_bytes(k, k);
    let data_center_bytes = dense_design_bytes(n, k);
    assert!(
        centers_bytes <= SPATIAL_CENTER_CENTER_MAX_BYTES,
        "spatial PC centers exceed center storage cap: K={k}, d_pc={d_pc}, centers={:.1} MiB, cap={:.1} MiB",
        centers_bytes as f64 / (1024.0 * 1024.0),
        SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
    );
    assert!(
        center_center_bytes <= SPATIAL_CENTER_CENTER_MAX_BYTES,
        "spatial PC centers exceed center-center biobank cap: K={k}, d_pc={d_pc}, KxK={:.1} MiB, cap={:.1} MiB",
        center_center_bytes as f64 / (1024.0 * 1024.0),
        SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
    );
    assert!(
        data_center_bytes <= SPATIAL_DATA_CENTER_DISTANCE_CACHE_MAX_BYTES
            || !spatial_distance_cacheable_entry(n, k),
        "spatial PC n*K distance cache cap mismatch: n={n}, K={k}, d_pc={d_pc}, optional n*K cache={:.1} MiB, cap={:.1} MiB",
        data_center_bytes as f64 / (1024.0 * 1024.0),
        SPATIAL_DATA_CENTER_DISTANCE_CACHE_MAX_BYTES as f64 / (1024.0 * 1024.0),
    );
}

fn dense_design_bytes(n: usize, p: usize) -> usize {
    n.saturating_mul(p)
        .saturating_mul(std::mem::size_of::<f64>())
}

fn should_use_lazy_spatial_design(
    n: usize,
    p: usize,
    policy: &crate::resource::ResourcePolicy,
) -> bool {
    dense_design_bytes(n, p) > policy.max_single_materialization_bytes
}

fn wrap_dense_design_with_transform(
    design: DesignMatrix,
    transform: &Array2<f64>,
    label: &str,
) -> Result<DesignMatrix, BasisError> {
    match design {
        DesignMatrix::Dense(inner) => {
            let op = CoefficientTransformOperator::new(inner, transform.clone()).map_err(|e| {
                BasisError::InvalidInput(format!("{label} coefficient transform failed: {e}"))
            })?;
            Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Arc::new(op),
            )))
        }
        DesignMatrix::Sparse(_) => Err(BasisError::InvalidInput(format!(
            "{label} coefficient transform requires a dense/operator-backed design"
        ))),
    }
}

fn design_constraint_cross(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, BasisError> {
    let n = design.nrows();
    let k = design.ncols();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
        });
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(BasisError::WeightsDimensionMismatch {
            expected: n,
            found: w.len(),
        });
    }
    let q = constraint_matrix.ncols();
    let mut cross = Array2::<f64>::zeros((k, q));
    for start in (0..n).step_by(DESIGN_CROSS_CHUNK_SIZE) {
        let end = (start + DESIGN_CROSS_CHUNK_SIZE).min(n);
        let basis_chunk = design.row_chunk(start..end);
        let mut constraint_chunk = constraint_matrix.slice(s![start..end, ..]).to_owned();
        if let Some(w) = weights {
            for (mut row, &weight) in constraint_chunk
                .axis_iter_mut(Axis(0))
                .zip(w.slice(s![start..end]).iter())
            {
                row *= weight;
            }
        }
        cross += &basis_chunk.t().dot(&constraint_chunk);
    }
    Ok(cross)
}

fn design_gram_matrix(design: &DesignMatrix) -> Array2<f64> {
    let n = design.nrows();
    let p = design.ncols();
    let mut gram = Array2::<f64>::zeros((p, p));
    for start in (0..n).step_by(DESIGN_CROSS_CHUNK_SIZE) {
        let end = (start + DESIGN_CROSS_CHUNK_SIZE).min(n);
        let chunk = design.row_chunk(start..end);
        gram += &chunk.t().dot(&chunk);
    }
    gram
}

fn positive_spectral_whitener_from_gram(gram: &Array2<f64>) -> Result<Array2<f64>, BasisError> {
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(BasisError::LinalgError)?;
    let max_eval = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tol = default_rrqr_rank_alpha()
        * f64::EPSILON
        * (gram.nrows().max(gram.ncols()).max(1) as f64)
        * max_eval.max(1.0);
    let keep = eigenvalues.iter().filter(|&&ev| ev > tol).count();
    if keep == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    let eig_start = eigenvalues.len() - keep;
    let kept_vectors = eigenvectors.slice(s![.., eig_start..]).to_owned();
    let mut inv_sqrt = Array2::<f64>::zeros((keep, keep));
    for (out_i, eig_i) in (eig_start..eigenvalues.len()).enumerate() {
        inv_sqrt[[out_i, out_i]] = 1.0 / eigenvalues[eig_i].max(tol).sqrt();
    }
    Ok(fast_ab(&kept_vectors, &inv_sqrt))
}

fn stabilized_orthogonality_transform_from_gram(
    gram: &Array2<f64>,
    transform: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let constrained_gram = {
        let gt = fast_ab(gram, transform);
        fast_atb(transform, &gt)
    };
    let whitening = positive_spectral_whitener_from_gram(&constrained_gram)?;
    Ok(fast_ab(transform, &whitening))
}

fn whitening_transform_from_gram(gram: &Array2<f64>) -> Result<Array2<f64>, BasisError> {
    positive_spectral_whitener_from_gram(gram)
}

fn orthogonality_transform_from_cross_and_gram(
    constraint_cross: &Array2<f64>,
    gram: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let whitening = whitening_transform_from_gram(gram)?;
    let cross_whitened = fast_atb(&whitening, constraint_cross);
    let (transform_whitened, rank) =
        rrqr_nullspace_basis(&cross_whitened, default_rrqr_rank_alpha())
            .map_err(BasisError::LinalgError)?;
    if rank >= cross_whitened.nrows() || transform_whitened.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    let transform = fast_ab(&whitening, &transform_whitened);
    stabilized_orthogonality_transform_from_gram(gram, &transform)
}

pub(crate) fn orthogonality_transform_for_design(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, BasisError> {
    let k = design.ncols();
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok(Array2::eye(k));
    }
    let constraint_cross = design_constraint_cross(design, constraint_matrix, weights)?;
    let gram = design_gram_matrix(design);
    orthogonality_transform_from_cross_and_gram(&constraint_cross, &gram)
}

/// Which radial kernel family is being used. Stored in the streaming operator
/// so that (q, t) scalars can be recomputed on the fly without a closure.
#[derive(Debug, Clone)]
pub(crate) enum RadialScalarKind {
    /// Matern kernel: (length_scale, nu).
    Matern { length_scale: f64, nu: MaternNu },
    /// Hybrid Duchon kernel: parameters needed for `duchon_radial_jets`.
    Duchon {
        length_scale: f64,
        p_order: usize,
        s_order: usize,
        dim: usize,
        coeffs: DuchonPartialFractionCoeffs,
    },
    /// Pure Duchon kernel: a single intrinsic polyharmonic block.
    PureDuchon {
        block_order: usize,
        p_order: usize,
        s_order: usize,
        dim: usize,
    },
}

impl RadialScalarKind {
    /// Evaluate the `(phi, q, t)` radial scalars for a given distance `r`.
    fn eval_design_triplet(&self, r: f64) -> Result<(f64, f64, f64), BasisError> {
        match self {
            RadialScalarKind::Matern { length_scale, nu } => {
                let (phi, q, t, _, _) =
                    matern_aniso_extended_radial_scalars(r, *length_scale, *nu)?;
                Ok((phi, q, t))
            }
            RadialScalarKind::Duchon {
                length_scale,
                p_order,
                s_order,
                dim,
                coeffs,
            } => {
                let jets = duchon_radial_jets(r, *length_scale, *p_order, *s_order, *dim, coeffs)?;
                Ok((jets.phi, jets.q, jets.t))
            }
            RadialScalarKind::PureDuchon {
                block_order, dim, ..
            } => {
                let phi = polyharmonic_kernel(r, *block_order, *dim);
                let (q, t, _, _) = duchon_polyharmonic_operator_block_jets(r, *block_order, *dim)?;
                Ok((phi, q, t))
            }
        }
    }

    #[inline]
    fn raw_psi_isotropic_share(&self) -> f64 {
        match self {
            RadialScalarKind::Matern { .. } => 0.0,
            RadialScalarKind::Duchon {
                p_order,
                s_order,
                dim,
                ..
            } => duchon_scaling_exponent(*p_order, *s_order, *dim) / *dim as f64,
            RadialScalarKind::PureDuchon {
                p_order,
                s_order,
                dim,
                ..
            } => duchon_scaling_exponent(*p_order, *s_order, *dim) / *dim as f64,
        }
    }

    #[inline]
    fn is_duchon_family(&self) -> bool {
        matches!(
            self,
            RadialScalarKind::Duchon { .. } | RadialScalarKind::PureDuchon { .. }
        )
    }
}

/// Data stored for streaming (on-the-fly) recomputation of radial jet scalars.
/// Instead of persisting O(n*k*(d+2)) arrays, the operator stores the original
/// data/centers/eta and recomputes q/t/s per chunk during matvec operations.
#[derive(Debug, Clone)]
enum StreamingAxisMode {
    /// Per-axis anisotropic ψ_a derivatives: expose one `s_a` component per axis.
    PerAxis { eta: Vec<f64> },
    /// Scalar ψ derivative: expose a single component equal to the total
    /// scaled squared radius r² = Σ_a exp(2η_a) h_a².
    ScalarTotal { eta: Vec<f64> },
}

#[derive(Debug, Clone)]
struct StreamingRadialState {
    /// Data matrix, shape (n, d).
    data: Arc<Array2<f64>>,
    /// Center matrix, shape (k, d).
    centers: Arc<Array2<f64>>,
    /// How per-pair axis components are exposed to the derivative operator.
    axis_mode: StreamingAxisMode,
    /// Which radial kernel family to use for recomputation.
    radial_kind: RadialScalarKind,
}

impl StreamingRadialState {
    /// Compute `(phi, q, t, s_a[0..d])` for a single `(data_row i, center j)` pair.
    ///
    /// Returns `(phi, q, t)` and writes per-axis components into `s_buf` (length d).
    #[inline]
    fn compute_pair(
        &self,
        i: usize,
        j: usize,
        s_buf: &mut [f64],
    ) -> Result<(f64, f64, f64), BasisError> {
        match &self.axis_mode {
            StreamingAxisMode::PerAxis { eta } => {
                let dim = eta.len();
                debug_assert_eq!(s_buf.len(), dim);
                let eta_mean = centered_aniso_log_scale_mean(eta);
                let mut r2 = 0.0;
                for a in 0..dim {
                    let h = self.data[[i, a]] - self.centers[[j, a]];
                    let w = aniso_metric_weight(eta[a], eta_mean);
                    let s_a = w * h * h;
                    s_buf[a] = s_a;
                    r2 += s_a;
                }
                self.radial_kind.eval_design_triplet(r2.sqrt())
            }
            StreamingAxisMode::ScalarTotal { eta } => {
                debug_assert_eq!(s_buf.len(), 1);
                let eta_mean = centered_aniso_log_scale_mean(eta);
                let mut r2 = 0.0;
                for a in 0..eta.len() {
                    let h = self.data[[i, a]] - self.centers[[j, a]];
                    let w = aniso_metric_weight(eta[a], eta_mean);
                    r2 += w * h * h;
                }
                s_buf[0] = r2;
                self.radial_kind.eval_design_triplet(r2.sqrt())
            }
        }
    }
}

/// Implicit representation of ∂X/∂ψ_d that supports matrix-vector products
/// without materializing the full (n x p) derivative matrices.
///
/// For anisotropic Matern / Duchon terms with D axes, the dense path creates
/// D matrices of size (n x p_smooth) for dX/dpsi_d. At n=400K, p=2000, D=16,
/// that is ~100 GB.
///
/// Two storage modes:
///
/// **Materialized** (small-to-medium problems): stores pre-computed arrays
/// - `phi_values[i*n_knots + j]` = phi(r_{ij})
/// - `q_values[i*n_knots + j]` = phi'(r_{ij}) / r_{ij}
/// - `t_values[i*n_knots + j]` = (phi''(r_{ij}) - q_{ij}) / r_{ij}^2
/// - `axis_components[i*n_knots + j, d]` = exp(2 eta_d) * (x_{id} - c_{jd})^2
/// Memory: O(n * k * (D + 2)).
///
/// **Streaming** (biobank scale): stores only data/centers/eta/kernel params
/// and recomputes (q, t, s_a) on the fly during each matvec.
/// Memory: O(n*d + k*d) -- no per-(data,knot) storage.
///
/// The raw-psi chain rule:
///   shape_a   = q * s_a
///   shape_ab  = t * s_a * s_b + 2 q s_a 1[a=b]
///   dphi/dpsi_a         = shape_a + c * phi
///   d2phi/(dpsi_a dpsi_b) = shape_ab + c (shape_a + shape_b) + c^2 phi
/// where `c = 0` for Matérn and `c = delta / d` for hybrid Duchon.
#[derive(Debug, Clone)]
pub struct ImplicitDesignPsiDerivative {
    /// Pre-computed kernel values (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    phi_values: Array1<f64>,

    /// Pre-computed per (data, knot) pair axis components (materialized mode).
    /// Shape: (n * n_knots, D) stored in row-major order.
    /// Empty (0x0) in streaming mode.
    axis_components: Array2<f64>,

    /// Pre-computed R-operator first scalar (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    q_values: Array1<f64>,

    /// Pre-computed R-operator second scalar (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    t_values: Array1<f64>,

    /// When set, enables streaming recomputation of q/t/s from raw inputs
    /// instead of reading from the pre-computed arrays above.
    streaming: Option<StreamingRadialState>,

    /// Identifiability/constraint transform Z: (n_knots x p_constrained).
    /// Converts raw knot-space vectors to the identifiability-constrained
    /// basis. For Duchon this is the kernel-constraint nullspace Z_kernel;
    /// for Matern with identifiability constraints, it is the corresponding Z.
    /// `None` means the identity (no constraint).
    ident_transform: Option<Array2<f64>>,

    /// Optional full identifiability transform applied after Z_kernel + padding.
    /// For Duchon terms that have an additional global identifiability transform,
    /// this is applied after the kernel constraint and polynomial padding.
    /// Shape: (p_constrained + n_poly, p_final).
    full_ident_transform: Option<Array2<f64>>,

    /// Number of data points.
    n: usize,

    /// Number of knots (raw basis functions before identifiability transform).
    n_knots: usize,

    /// Number of polynomial columns appended after the smooth part.
    /// These have zero derivative with respect to psi_d.
    n_poly: usize,

    /// Number of axes (dimension D).
    n_axes: usize,

    /// Isotropic scaling contribution per raw anisotropic psi axis.
    psi_scale_share: f64,

    /// Optional exposed-axis to raw-axis linear combinations.
    /// When present, axis `a` represents Σ_i coeff_i * raw_axis_i.
    axis_combinations: Option<Vec<Vec<(usize, f64)>>>,
}

/// The rayon chunk size for parallel implicit matvec operations.
/// Each chunk processes this many data points before reducing.
const IMPLICIT_MATVEC_CHUNK_SIZE: usize = 1000;

/// Minimum data size to activate parallel iteration for implicit matvecs.
const IMPLICIT_MATVEC_PAR_THRESHOLD: usize = 10_000;

impl ImplicitDesignPsiDerivative {
    /// Construct from pre-computed radial jet scalars.
    ///
    /// # Arguments
    /// - `q_values`: (n * n_knots,) — φ'(r)/r for each (data, knot) pair.
    /// - `t_values`: (n * n_knots,) — (φ''(r) - q) / r² for each pair.
    /// - `axis_components`: (n * n_knots, D) — s_{d,ij} = exp(2η_d) · h_d² for each pair/axis.
    /// - `ident_transform`: optional (n_knots × p_constrained) constraint projection.
    /// - `full_ident_transform`: optional further projection after padding.
    /// - `n`, `n_knots`, `n_poly`, `n_axes`: dimensions.
    /// Construct from pre-computed (materialized) radial jet scalars.
    /// This is the original path for small-to-medium problems where
    /// O(n*k*(d+2)) storage is acceptable.
    pub fn new(
        phi_values: Array1<f64>,
        q_values: Array1<f64>,
        t_values: Array1<f64>,
        axis_components: Array2<f64>,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n: usize,
        n_knots: usize,
        n_poly: usize,
        n_axes: usize,
    ) -> Self {
        assert_eq!(phi_values.len(), n * n_knots);
        assert_eq!(q_values.len(), n * n_knots);
        assert_eq!(t_values.len(), n * n_knots);
        assert_eq!(axis_components.nrows(), n * n_knots);
        assert_eq!(axis_components.ncols(), n_axes);
        Self {
            phi_values,
            axis_components,
            q_values,
            t_values,
            streaming: None,
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes,
            psi_scale_share: 0.0,
            axis_combinations: None,
        }
    }

    fn with_psi_scale_share(mut self, psi_scale_share: f64) -> Self {
        self.psi_scale_share = psi_scale_share;
        self
    }

    fn with_axis_combinations(mut self, axis_combinations: Vec<Vec<(usize, f64)>>) -> Self {
        for combo in &axis_combinations {
            for &(raw_axis, _) in combo {
                assert!(raw_axis < self.n_axes);
            }
        }
        self.axis_combinations = Some(axis_combinations);
        self
    }

    /// Construct a streaming operator that recomputes (q, t, s_a) on the fly
    /// from raw data/centers/eta during each matvec. No O(n*k) arrays are stored.
    /// This is the biobank-scale path.
    pub(crate) fn new_streaming(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        eta: Vec<f64>,
        radial_kind: RadialScalarKind,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n_poly: usize,
    ) -> Self {
        let n = data.nrows();
        let n_knots = centers.nrows();
        let n_axes = data.ncols();
        let psi_scale_share = radial_kind.raw_psi_isotropic_share();
        assert_eq!(eta.len(), n_axes);
        Self {
            // Empty arrays -- not used in streaming mode.
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::PerAxis { eta },
                radial_kind,
            }),
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes,
            psi_scale_share,
            axis_combinations: None,
        }
    }

    /// Construct a streaming operator for a scalar ψ derivative. The operator
    /// exposes a single axis component equal to the full scaled squared
    /// distance r² under the fixed metric defined by `eta`.
    pub(crate) fn new_streaming_scalar(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        eta: Vec<f64>,
        radial_kind: RadialScalarKind,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n_poly: usize,
    ) -> Self {
        let n = data.nrows();
        let n_knots = centers.nrows();
        let dim = data.ncols();
        assert_eq!(eta.len(), dim);
        Self {
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::ScalarTotal { eta },
                radial_kind,
            }),
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes: 1,
            psi_scale_share: 0.0,
            axis_combinations: None,
        }
    }

    /// Whether this operator is in streaming (recompute-on-the-fly) mode.
    #[inline]
    fn is_streaming(&self) -> bool {
        self.streaming.is_some()
    }

    /// Number of data points.
    pub fn n_data(&self) -> usize {
        self.n
    }

    /// Number of axes (D).
    pub fn n_axes(&self) -> usize {
        self.axis_combinations
            .as_ref()
            .map_or(self.n_axes, Vec::len)
    }

    pub(crate) fn is_duchon_family(&self) -> bool {
        self.streaming.as_ref().is_some_and(|state| {
            matches!(
                state.radial_kind,
                RadialScalarKind::Duchon { .. } | RadialScalarKind::PureDuchon { .. }
            )
        }) || self.psi_scale_share != 0.0
    }

    /// Output dimension: total basis columns in the final space.
    pub fn p_out(&self) -> usize {
        if let Some(ref zf) = self.full_ident_transform {
            zf.ncols()
        } else {
            self.p_after_pad()
        }
    }

    /// Dimension after kernel constraint + polynomial padding (before full ident).
    fn p_after_pad(&self) -> usize {
        let p_constrained = self.p_constrained();
        p_constrained + self.n_poly
    }

    /// Dimension after kernel constraint projection (before poly padding).
    fn p_constrained(&self) -> usize {
        match &self.ident_transform {
            Some(z) => z.ncols(),
            None => self.n_knots,
        }
    }

    /// Accumulate raw knot-space vector from weighted (data, knot) contributions.
    /// Returns a vector of length n_knots: Σ_i w_i · scalar_{ij} for each knot j.
    ///
    /// This is the core primitive: for each data point i, accumulate
    /// `v[i] * per_pair_scalar(i,j)` into knot j.
    fn accumulate_knot_vector<F>(&self, v: &ArrayView1<f64>, per_pair: F) -> Array1<f64>
    where
        F: Fn(usize) -> f64 + Send + Sync,
    {
        let n = self.n;
        let k = self.n_knots;

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            // Parallel path: chunk data points and reduce.
            let n_chunks = (n + IMPLICIT_MATVEC_CHUNK_SIZE - 1) / IMPLICIT_MATVEC_CHUNK_SIZE;
            let partial_sums: Vec<Array1<f64>> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut local = Array1::<f64>::zeros(k);
                    for i in start..end {
                        let vi = v[i];
                        if vi == 0.0 {
                            continue;
                        }
                        let base = i * k;
                        for j in 0..k {
                            local[j] += vi * per_pair(base + j);
                        }
                    }
                    local
                })
                .collect();
            let mut total = Array1::<f64>::zeros(k);
            for p in partial_sums {
                total += &p;
            }
            total
        } else {
            // Sequential path.
            let mut total = Array1::<f64>::zeros(k);
            for i in 0..n {
                let vi = v[i];
                if vi == 0.0 {
                    continue;
                }
                let base = i * k;
                for j in 0..k {
                    total[j] += vi * per_pair(base + j);
                }
            }
            total
        }
    }

    /// Streaming accumulate knot vector from on-the-fly radial scalars.
    fn streaming_accumulate_knot_vector<G>(
        &self,
        v: &ArrayView1<f64>,
        deriv_fn: G,
    ) -> Result<Array1<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let err_flag = std::sync::atomic::AtomicBool::new(false);
            let nc = (n + IMPLICIT_MATVEC_CHUNK_SIZE - 1) / IMPLICIT_MATVEC_CHUNK_SIZE;
            let ps: Vec<Array1<f64>> = (0..nc)
                .into_par_iter()
                .map(|ci| {
                    let s = ci * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let e = (s + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut loc = Array1::<f64>::zeros(k);
                    let mut sb = vec![0.0; dim];
                    for i in s..e {
                        let vi = v[i];
                        if vi == 0.0 {
                            continue;
                        }
                        for j in 0..k {
                            match st.compute_pair(i, j, &mut sb) {
                                Ok((phi, q, t)) => {
                                    loc[j] += vi * deriv_fn(phi, q, t, &sb);
                                }
                                Err(_) => {
                                    err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return loc;
                                }
                            }
                        }
                    }
                    loc
                })
                .collect();
            if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(BasisError::InvalidInput(
                    "radial scalar evaluation failed during streaming accumulate_knot_vector"
                        .into(),
                ));
            }
            let mut tot = Array1::<f64>::zeros(k);
            for p in ps {
                tot += &p;
            }
            Ok(tot)
        } else {
            let mut tot = Array1::<f64>::zeros(k);
            let mut sb = vec![0.0; dim];
            for i in 0..n {
                let vi = v[i];
                if vi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    let (phi, q, t) = st.compute_pair(i,j,&mut sb).map_err(|e| BasisError::InvalidInput(
                        format!("radial scalar evaluation failed during streaming accumulate_knot_vector: {e}"),
                    ))?;
                    tot[j] += vi * deriv_fn(phi, q, t, &sb);
                }
            }
            Ok(tot)
        }
    }
    /// Streaming forward multiply.
    fn streaming_forward_mul<G>(
        &self,
        u_knot: &Array1<f64>,
        deriv_fn: G,
    ) -> Result<Array1<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let err_flag = std::sync::atomic::AtomicBool::new(false);
            let nc = (n + IMPLICIT_MATVEC_CHUNK_SIZE - 1) / IMPLICIT_MATVEC_CHUNK_SIZE;
            let cr: Vec<(usize, Vec<f64>)> = (0..nc)
                .into_par_iter()
                .map(|ci| {
                    let s = ci * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let e = (s + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut loc = vec![0.0; e - s];
                    let mut sb = vec![0.0; dim];
                    for i in s..e {
                        let mut val = 0.0;
                        for j in 0..k {
                            match st.compute_pair(i, j, &mut sb) {
                                Ok((phi, q, t)) => {
                                    val += deriv_fn(phi, q, t, &sb) * u_knot[j];
                                }
                                Err(_) => {
                                    err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                        loc[i - s] = val;
                    }
                    (s, loc)
                })
                .collect();
            if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(BasisError::InvalidInput(
                    "radial scalar evaluation failed during streaming forward_mul".into(),
                ));
            }
            let mut res = Array1::<f64>::zeros(n);
            for (s, vs) in cr {
                for (o, &v) in vs.iter().enumerate() {
                    res[s + o] = v;
                }
            }
            Ok(res)
        } else {
            let mut res = Array1::<f64>::zeros(n);
            let mut sb = vec![0.0; dim];
            for i in 0..n {
                let mut val = 0.0;
                for j in 0..k {
                    let (phi, q, t) = st.compute_pair(i, j, &mut sb).map_err(|e| {
                        BasisError::InvalidInput(format!(
                            "radial scalar evaluation failed during streaming forward_mul: {e}"
                        ))
                    })?;
                    val += deriv_fn(phi, q, t, &sb) * u_knot[j];
                }
                res[i] = val;
            }
            Ok(res)
        }
    }
    /// Streaming materialization: build (n x k) raw matrix then project.
    fn streaming_materialize<G>(&self, deriv_fn: G) -> Result<Array2<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        let mut raw = Array2::<f64>::zeros((n, k));
        let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
        let nc = (n + cs - 1) / cs;
        let err_flag = std::sync::atomic::AtomicBool::new(false);
        {
            let rp = SendPtr(raw.as_mut_ptr());
            let ef = &err_flag;
            (0..nc).into_par_iter().for_each(move |ci| {
                let s = ci * cs;
                let e = (s + cs).min(n);
                let mut sb = vec![0.0; dim];
                for i in s..e {
                    for j in 0..k {
                        match st.compute_pair(i, j, &mut sb) {
                            Ok((phi, q, t)) => unsafe {
                                *rp.add(i * k + j) = deriv_fn(phi, q, t, &sb);
                            },
                            Err(_) => {
                                ef.store(true, std::sync::atomic::Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            });
        }
        if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(BasisError::InvalidInput(
                "radial scalar evaluation failed during streaming materialize".into(),
            ));
        }
        Ok(self.project_matrix(raw))
    }

    /// Project a raw knot-space vector through the identifiability transform
    /// and pad with zeros for polynomial columns.
    fn project_and_pad(&self, raw_knot_vec: &Array1<f64>) -> Array1<f64> {
        // Step 1: apply kernel constraint Z (if present).
        let constrained = match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot_vec),
            None => raw_knot_vec.clone(),
        };

        // Step 2: pad with polynomial zeros.
        let p_padded = constrained.len() + self.n_poly;
        let mut padded = Array1::<f64>::zeros(p_padded);
        padded
            .slice_mut(s![..constrained.len()])
            .assign(&constrained);

        // Step 3: apply full identifiability transform (if present).
        match &self.full_ident_transform {
            Some(zf) => zf.t().dot(&padded),
            None => padded,
        }
    }

    /// Expand a coefficient vector from the final space back to raw knot space.
    /// This is the transpose path: p_out → (padded) → (constrained) → n_knots.
    fn unproject(&self, u: &ArrayView1<f64>) -> Array1<f64> {
        // Step 1: undo full identifiability transform.
        let after_full = match &self.full_ident_transform {
            Some(zf) => zf.dot(u),
            None => u.to_owned(),
        };

        // Step 2: extract smooth part (drop polynomial padding).
        let p_constrained = self.p_constrained();
        let smooth_part = after_full.slice(s![..p_constrained]);

        // Step 3: undo kernel constraint Z.
        match &self.ident_transform {
            Some(z) => z.dot(&smooth_part),
            None => smooth_part.to_owned(),
        }
    }

    /// Compute (∂X/∂ψ_d)^T v for a given axis d and vector v of length n.
    ///
    /// Returns a vector of length p_out (total basis dimension after all transforms).
    ///
    /// Formula in raw knot space:
    ///   [raw]_j = Σ_i v_i · q_{ij} · s_{d,ij}
    /// then project through Z and pad.
    ///
    /// Note: q = φ_r/r and s_d = exp(2ψ_d)·h_d² are UNNORMALIZED axis components.
    /// With this convention, q·s_d = (φ_r/r)·(exp(2ψ_d)·h_d²) = φ_r·(s_d/r),
    /// which equals the correct ∂φ/∂ψ_d = φ_r·∂r/∂ψ_d = φ_r·s_d/r.
    /// No r² correction is needed — that would be required only if s_d were
    /// the fractional quantity s_d/r².
    pub fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert_eq!(v.len(), self.n);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                Self::transformed_first_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    s_combo,
                    combo_sum,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw =
                self.streaming_accumulate_knot_vector(v, |phi, q, _, sb| q * sb[axis] + c * phi)?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let raw = self.accumulate_knot_vector(v, |idx| qv[idx] * af[[idx, axis]] + c * pv[idx]);
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂X/∂ψ_d) u for a given axis d and vector u of length p_out.
    ///
    /// Returns a vector of length n.
    ///
    /// Formula: for each data point i,
    ///   result_i = Σ_j q_{ij} · s_{d,ij} · u_knot_j
    /// where u_knot = Z · u_smooth (unprojected back to knot space).
    pub fn forward_mul(&self, axis: usize, u: &ArrayView1<f64>) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert_eq!(u.len(), self.p_out());
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let mut result = Array1::<f64>::zeros(n);
                let n_chunks = (n + IMPLICIT_MATVEC_CHUNK_SIZE - 1) / IMPLICIT_MATVEC_CHUNK_SIZE;
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let mut local = vec![0.0; end - start];
                        for i in start..end {
                            let base = i * k;
                            let mut val = 0.0;
                            for j in 0..k {
                                let idx = base + j;
                                let s_combo =
                                    self.transformed_combo_axis_value_materialized(idx, combo);
                                val += Self::transformed_first_kernel_value(
                                    self.phi_values[idx],
                                    self.q_values[idx],
                                    s_combo,
                                    combo_sum,
                                    c,
                                ) * u_knot[j];
                            }
                            local[i - start] = val;
                        }
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &v) in vals.iter().enumerate() {
                        result[start + offset] = v;
                    }
                }
                return Ok(result);
            }
            let mut result = Array1::<f64>::zeros(n);
            for i in 0..n {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    val += Self::transformed_first_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        s_combo,
                        combo_sum,
                        c,
                    ) * u_knot[j];
                }
                result[i] = val;
            }
            return Ok(result);
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, _, sb| q * sb[axis] + c * phi);
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let mut result = Array1::<f64>::zeros(n);
            // Parallel over chunks of data points.
            let n_chunks = (n + IMPLICIT_MATVEC_CHUNK_SIZE - 1) / IMPLICIT_MATVEC_CHUNK_SIZE;
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut local = vec![0.0; end - start];
                    for i in start..end {
                        let base = i * k;
                        let mut val = 0.0;
                        for j in 0..k {
                            val += (qv[base + j] * af[[base + j, axis]] + c * pv[base + j])
                                * u_knot[j];
                        }
                        local[i - start] = val;
                    }
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &v) in vals.iter().enumerate() {
                    result[start + offset] = v;
                }
            }
            Ok(result)
        } else {
            let mut result = Array1::<f64>::zeros(n);
            for i in 0..n {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    val += (qv[base + j] * af[[base + j, axis]] + c * pv[base + j]) * u_knot[j];
                }
                result[i] = val;
            }
            Ok(result)
        }
    }

    /// Compute (∂²X/∂ψ_d²)^T v — diagonal second derivative, same axis.
    ///
    /// Matrix-free variant of `materialize_second_diag`: avoids forming the
    /// full (n × p_out) matrix when only a single adjoint matvec is needed.
    pub fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert_eq!(v.len(), self.n);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                Self::transformed_second_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    self.t_values[idx],
                    s_combo,
                    combo_sum,
                    s_combo,
                    combo_sum,
                    overlap_s,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            })?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let raw = self.accumulate_knot_vector(v, |idx| {
            let s = af[[idx, axis]];
            2.0 * qv[idx] * s + tv[idx] * s * s + 2.0 * c * qv[idx] * s + c * c * pv[idx]
        });
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂²X/∂ψ_d∂ψ_e)^T v — cross second derivative (d ≠ e).
    pub fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis_d < self.n_axes());
        assert!(axis_e < self.n_axes());
        assert_ne!(axis_d, axis_e);
        assert_eq!(v.len(), self.n);
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                let overlap_s = self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                Self::transformed_second_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    self.t_values[idx],
                    s_d,
                    sum_d,
                    s_e,
                    sum_e,
                    overlap_s,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            })?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let raw = self.accumulate_knot_vector(v, |idx| {
            tv[idx] * af[[idx, axis_d]] * af[[idx, axis_e]]
                + c * qv[idx] * (af[[idx, axis_d]] + af[[idx, axis_e]])
                + c * c * pv[idx]
        });
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂²X/∂ψ_d²) u — forward diagonal second derivative.
    pub fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert_eq!(u.len(), self.p_out());
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let compute_row = |i: usize| -> f64 {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                    val += Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_combo,
                        combo_sum,
                        s_combo,
                        combo_sum,
                        overlap_s,
                        c,
                    ) * u_knot[j];
                }
                val
            };
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
                let mut result = Array1::<f64>::zeros(n);
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let local: Vec<f64> = (start..end).map(compute_row).collect();
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &value) in vals.iter().enumerate() {
                        result[start + offset] = value;
                    }
                }
                return Ok(result);
            }
            return Ok(Array1::from_vec((0..n).map(compute_row).collect()));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let compute_row = |i: usize| -> f64 {
            let base = i * k;
            let mut val = 0.0;
            for j in 0..k {
                let s = af[[base + j, axis]];
                val += (2.0 * qv[base + j] * s
                    + tv[base + j] * s * s
                    + 2.0 * c * qv[base + j] * s
                    + c * c * pv[base + j])
                    * u_knot[j];
            }
            val
        };

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let mut result = Array1::<f64>::zeros(n);
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let local: Vec<f64> = (start..end).map(compute_row).collect();
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &value) in vals.iter().enumerate() {
                    result[start + offset] = value;
                }
            }
            Ok(result)
        } else {
            Ok(Array1::from_vec((0..n).map(compute_row).collect()))
        }
    }

    /// Compute (∂²X/∂ψ_d∂ψ_e) u — forward cross second derivative.
    pub fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis_d < self.n_axes());
        assert!(axis_e < self.n_axes());
        assert_ne!(axis_d, axis_e);
        assert_eq!(u.len(), self.p_out());
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let compute_row = |i: usize| -> f64 {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                    let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                    let overlap_s =
                        self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                    val += Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_d,
                        sum_d,
                        s_e,
                        sum_e,
                        overlap_s,
                        c,
                    ) * u_knot[j];
                }
                val
            };
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
                let mut result = Array1::<f64>::zeros(n);
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let local: Vec<f64> = (start..end).map(compute_row).collect();
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &value) in vals.iter().enumerate() {
                        result[start + offset] = value;
                    }
                }
                return Ok(result);
            }
            return Ok(Array1::from_vec((0..n).map(compute_row).collect()));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let compute_row = |i: usize| -> f64 {
            let base = i * k;
            let mut val = 0.0;
            for j in 0..k {
                val += (tv[base + j] * af[[base + j, axis_d]] * af[[base + j, axis_e]]
                    + c * qv[base + j] * (af[[base + j, axis_d]] + af[[base + j, axis_e]])
                    + c * c * pv[base + j])
                    * u_knot[j];
            }
            val
        };

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let mut result = Array1::<f64>::zeros(n);
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let local: Vec<f64> = (start..end).map(compute_row).collect();
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &value) in vals.iter().enumerate() {
                    result[start + offset] = value;
                }
            }
            Ok(result)
        } else {
            Ok(Array1::from_vec((0..n).map(compute_row).collect()))
        }
    }

    /// Materialize the full (n × p_out) first-derivative matrix for axis d.
    ///
    /// Efficient O(n * k) construction: builds the raw (n × k) kernel derivative
    /// matrix directly, then projects through identifiability transforms.
    /// This is used when the dense matrix is needed temporarily (e.g., for
    /// HyperCoord construction) while avoiding simultaneous storage of all D axes.
    pub fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(axis < self.n_axes());
        if self.is_duchon_family() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    raw[[i, j]] = Self::transformed_first_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        s_combo,
                        combo_sum,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, _, sb| q * sb[axis] + c * phi);
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                raw[[i, j]] = self.q_values[base + j] * self.axis_components[[base + j, axis]]
                    + c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Materialize the full (n × p_out) second diagonal derivative matrix for axis d.
    pub fn materialize_second_diag(&self, axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(axis < self.n_axes());
        if self.is_duchon_family() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                    raw[[i, j]] = Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_combo,
                        combo_sum,
                        s_combo,
                        combo_sum,
                        overlap_s,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                let s = self.axis_components[[base + j, axis]];
                raw[[i, j]] = 2.0 * self.q_values[base + j] * s
                    + self.t_values[base + j] * s * s
                    + 2.0 * c * self.q_values[base + j] * s
                    + c * c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Materialize the full (n × p_out) cross second derivative matrix for axes (d, e).
    ///
    /// Dense materialization of the t · s_d · s_e cross coupling.
    pub fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(axis_d < self.n_axes());
        assert!(axis_e < self.n_axes());
        assert_ne!(axis_d, axis_e);
        if self.is_duchon_family() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                    let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                    let overlap_s =
                        self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                    raw[[i, j]] = Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_d,
                        sum_d,
                        s_e,
                        sum_e,
                        overlap_s,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                raw[[i, j]] = self.t_values[base + j]
                    * self.axis_components[[base + j, axis_d]]
                    * self.axis_components[[base + j, axis_e]]
                    + c * self.q_values[base + j]
                        * (self.axis_components[[base + j, axis_d]]
                            + self.axis_components[[base + j, axis_e]])
                    + c * c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Project a raw (n × k) kernel-space matrix through all transforms to
    /// produce an (n × p_out) matrix: Z_kernel → pad poly → full ident.
    fn project_matrix(&self, raw: Array2<f64>) -> Array2<f64> {
        // Step 1: kernel constraint projection.
        let constrained = match &self.ident_transform {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };

        // Step 2: polynomial padding.
        let padded = if self.n_poly > 0 {
            let cols = constrained.ncols();
            let mut out = Array2::<f64>::zeros((self.n, cols + self.n_poly));
            out.slice_mut(s![.., ..cols]).assign(&constrained);
            out
        } else {
            constrained
        };

        // Step 3: full identifiability transform.
        match &self.full_ident_transform {
            Some(zf) => fast_ab(&padded, zf),
            None => padded,
        }
    }

    fn project_matrix_rows(&self, raw: Array2<f64>) -> Array2<f64> {
        let nrows = raw.nrows();
        let constrained = match &self.ident_transform {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };
        let padded = if self.n_poly > 0 {
            let cols = constrained.ncols();
            let mut out = Array2::<f64>::zeros((nrows, cols + self.n_poly));
            out.slice_mut(s![.., ..cols]).assign(&constrained);
            out
        } else {
            constrained
        };
        match &self.full_ident_transform {
            Some(zf) => fast_ab(&padded, zf),
            None => padded,
        }
    }

    fn row_chunk_with_kernel<G>(
        &self,
        rows: std::ops::Range<usize>,
        deriv_fn: G,
    ) -> Result<Array2<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64], usize) -> f64,
    {
        let mut raw = Array2::<f64>::zeros((rows.end - rows.start, self.n_knots));
        if let Some(st) = self.streaming.as_ref() {
            let mut sb = vec![0.0; self.n_axes];
            for (local, i) in rows.enumerate() {
                for j in 0..self.n_knots {
                    let (phi, q, t) = st.compute_pair(i, j, &mut sb)?;
                    raw[[local, j]] = deriv_fn(phi, q, t, &sb, i * self.n_knots + j);
                }
            }
        } else {
            for (local, i) in rows.enumerate() {
                let base = i * self.n_knots;
                for j in 0..self.n_knots {
                    let idx = base + j;
                    raw[[local, j]] = deriv_fn(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        &[],
                        idx,
                    );
                }
            }
        }
        Ok(self.project_matrix_rows(raw))
    }

    pub fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(axis < self.n_axes());
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel(rows, |phi, q, _, sb, idx| {
                let s_combo = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo)
                } else {
                    combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, _, sb, idx| {
            let s = if sb.is_empty() {
                self.axis_components[[idx, axis]]
            } else {
                sb[axis]
            };
            q * s + c * phi
        })
    }

    pub fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(axis < self.n_axes());
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
                let s_combo = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo)
                } else {
                    combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let overlap = if sb.is_empty() {
                    self.transformed_combo_overlap_materialized(idx, combo, combo)
                } else {
                    Self::transformed_combo_overlap_streaming(combo, combo, sb)
                };
                Self::transformed_second_kernel_value(
                    phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap, c,
                )
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
            let s = if sb.is_empty() {
                self.axis_components[[idx, axis]]
            } else {
                sb[axis]
            };
            2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
        })
    }

    pub fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(axis_d < self.n_axes());
        assert!(axis_e < self.n_axes());
        assert_ne!(axis_d, axis_e);
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            return self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
                let s_d = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo_d)
                } else {
                    combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let s_e = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo_e)
                } else {
                    combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let overlap = if sb.is_empty() {
                    self.transformed_combo_overlap_materialized(idx, combo_d, combo_e)
                } else {
                    Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb)
                };
                Self::transformed_second_kernel_value(phi, q, t, s_d, sum_d, s_e, sum_e, overlap, c)
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
            let sd = if sb.is_empty() {
                self.axis_components[[idx, axis_d]]
            } else {
                sb[axis_d]
            };
            let se = if sb.is_empty() {
                self.axis_components[[idx, axis_e]]
            } else {
                sb[axis_e]
            };
            t * sd * se + c * q * (sd + se) + c * c * phi
        })
    }

    /// Single-row specialization of `row_chunk_first(axis, row..row+1)` that
    /// writes the length-`p_out` row directly into the caller-provided buffer.
    ///
    /// This is the row-local API used by `CustomFamilyPsiLinearMapRef::row_vector`
    /// for survival rowwise exact-Hessian paths, which previously applied a
    /// unit-vector `transpose_mul` trick (O(n·K) per row) to recover a single
    /// row. Avoids allocating a temporary (1 × p_out) matrix per row call.
    pub fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), BasisError> {
        assert!(row < self.n);
        assert_eq!(out.len(), self.p_out());
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
    }

    /// Apply the first derivative forward map to a row block: computes
    /// `result[i - rows.start] = Σ_j (∂X/∂ψ_axis)[i, j] * u[j]` for each
    /// `i ∈ rows`.
    ///
    /// The argument `u` is expressed in the final (`p_out`) basis. The returned
    /// vector has length `rows.end - rows.start`.
    pub fn forward_mul_rows(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
        u: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert_eq!(u.len(), self.p_out());
        assert!(rows.end <= self.n);
        let chunk = self.row_chunk_first(axis, rows)?;
        Ok(chunk.dot(&u))
    }

    /// Apply the first derivative adjoint to a row block: computes
    /// `result[j] = Σ_{i ∈ rows} (∂X/∂ψ_axis)[i, j] * v[i - rows.start]`
    /// expressed in the final `p_out` basis (the `row_chunk_first` output is
    /// already projected through the identifiability transforms).
    pub fn transpose_mul_rows(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
        v: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(axis < self.n_axes());
        assert!(rows.end <= self.n);
        assert_eq!(v.len(), rows.end - rows.start);
        let chunk = self.row_chunk_first(axis, rows)?;
        Ok(chunk.t().dot(&v))
    }

    fn transformed_axis_combination(&self, axis: usize) -> &[(usize, f64)] {
        self.axis_combinations
            .as_ref()
            .expect("transformed axis combinations")
            .get(axis)
            .map(Vec::as_slice)
            .expect("transformed axis index")
    }

    #[inline]
    fn transformed_combo_sum(combo: &[(usize, f64)]) -> f64 {
        combo.iter().map(|(_, coeff)| *coeff).sum()
    }

    #[inline]
    fn transformed_combo_axis_value_materialized(&self, idx: usize, combo: &[(usize, f64)]) -> f64 {
        combo
            .iter()
            .map(|(raw_axis, coeff)| coeff * self.axis_components[[idx, *raw_axis]])
            .sum()
    }

    #[inline]
    fn transformed_combo_overlap_streaming(
        combo_left: &[(usize, f64)],
        combo_right: &[(usize, f64)],
        sb: &[f64],
    ) -> f64 {
        let mut overlap = 0.0;
        for &(left_axis, left_coeff) in combo_left {
            for &(right_axis, right_coeff) in combo_right {
                if left_axis == right_axis {
                    overlap += left_coeff * right_coeff * sb[left_axis];
                }
            }
        }
        overlap
    }

    #[inline]
    fn transformed_combo_overlap_materialized(
        &self,
        idx: usize,
        combo_left: &[(usize, f64)],
        combo_right: &[(usize, f64)],
    ) -> f64 {
        let mut overlap = 0.0;
        for &(left_axis, left_coeff) in combo_left {
            for &(right_axis, right_coeff) in combo_right {
                if left_axis == right_axis {
                    overlap += left_coeff * right_coeff * self.axis_components[[idx, left_axis]];
                }
            }
        }
        overlap
    }

    #[inline]
    fn transformed_first_kernel_value(
        phi: f64,
        q: f64,
        s_combo: f64,
        coeff_sum: f64,
        psi_scale_share: f64,
    ) -> f64 {
        q * s_combo + psi_scale_share * coeff_sum * phi
    }

    #[inline]
    fn transformed_second_kernel_value(
        phi: f64,
        q: f64,
        t: f64,
        s_left: f64,
        left_sum: f64,
        s_right: f64,
        right_sum: f64,
        overlap_s: f64,
        psi_scale_share: f64,
    ) -> f64 {
        t * s_left * s_right
            + 2.0 * q * overlap_s
            + psi_scale_share * q * (right_sum * s_left + left_sum * s_right)
            + psi_scale_share * psi_scale_share * left_sum * right_sum * phi
    }
}

fn build_aniso_design_psi_derivatives_shared(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    eta: &[f64],
    p_final: usize,
    ident_transform: Option<Array2<f64>>,
    full_ident_transform: Option<Array2<f64>>,
    n_poly: usize,
    radial_kind: RadialScalarKind,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    let dim = data.ncols();
    if eta.len() != dim {
        return Err(BasisError::DimensionMismatch(format!(
            "aniso design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        )));
    }

    let force_operator = radial_kind.is_duchon_family();
    let use_implicit = force_operator || should_use_implicit_operators(n, p_final, dim);

    // ── Streaming path: biobank scale ─────────────────────────────────────
    // When dense materialization would exceed the memory threshold, build a
    // streaming operator that stores only data/centers/eta/radial_kind and
    // recomputes (q, t, s_a) on the fly during every matvec.
    if use_implicit {
        let op = ImplicitDesignPsiDerivative::new_streaming(
            shared_owned_data_matrix_from_view(data),
            shared_owned_centers_matrix_from_view(centers),
            eta.to_vec(),
            radial_kind,
            ident_transform,
            full_ident_transform,
            n_poly,
        );
        return Ok(AnisoBasisPsiDerivatives {
            design_first: Vec::new(),
            design_second_diag: Vec::new(),
            design_second_cross: Vec::new(),
            design_second_cross_pairs: Vec::new(),
            penalties_first: vec![Vec::new(); dim],
            penalties_second_diag: vec![Vec::new(); dim],
            penalties_cross_pairs: Vec::new(),
            penalties_cross_provider: None,
            implicit_operator: Some(op),
        });
    }

    // ── Materialized path: small-to-medium non-Duchon problems ────────────
    // Allocate O(n*k) arrays up front and fill with parallel chunks that
    // write directly into preallocated storage via raw pointers. No
    // intermediate Vec<(i, q_row, t_row, s_row)> collection.
    let nk = n * k;
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, dim));

    let psi_scale_share = radial_kind.raw_psi_isotropic_share();

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = (n + cs - 1) / cs;
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ef = &err_flag;
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = (start + cs).min(n);
            let mut drb = vec![0.0; dim];
            let mut cb = vec![0.0; dim];
            for i in start..end {
                for a in 0..dim {
                    drb[a] = data[[i, a]];
                }
                for j in 0..k {
                    for a in 0..dim {
                        cb[a] = centers[[j, a]];
                    }
                    let (r, sv) = aniso_distance_and_components(&drb, &cb, eta);
                    let (phi, q, t) = match radial_kind.eval_design_triplet(r) {
                        Ok(p) => p,
                        Err(_) => {
                            ef.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                    };
                    let flat = i * k + j;
                    unsafe {
                        *pp.add(flat) = phi;
                        *qp.add(flat) = q;
                        *tp.add(flat) = t;
                        for a in 0..dim {
                            *ap.add(flat * dim + a) = sv[a];
                        }
                    }
                }
            }
        });
    }
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(BasisError::InvalidInput(
            "radial scalar evaluation failed during aniso derivative construction".into(),
        ));
    }

    let op = ImplicitDesignPsiDerivative::new(
        phi_values,
        q_values,
        t_values,
        axis_components,
        ident_transform,
        full_ident_transform,
        n,
        k,
        n_poly,
        dim,
    )
    .with_psi_scale_share(psi_scale_share);
    let design_first = (0..dim)
        .map(|a| op.materialize_first(a))
        .collect::<Result<Vec<_>, _>>()?;
    let design_second_diag = (0..dim)
        .map(|a| op.materialize_second_diag(a))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AnisoBasisPsiDerivatives {
        design_first,
        design_second_diag,
        design_second_cross: Vec::new(),
        design_second_cross_pairs: Vec::new(),
        penalties_first: vec![Vec::new(); dim],
        penalties_second_diag: vec![Vec::new(); dim],
        penalties_cross_pairs: Vec::new(),
        penalties_cross_provider: None,
        implicit_operator: Some(op),
    })
}

#[derive(Debug, Clone)]
struct ScalarDesignPsiDerivatives {
    design_first: Array2<f64>,
    design_second_diag: Array2<f64>,
    implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

fn build_scalar_design_psi_derivatives_shared(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    fixed_eta: Option<&[f64]>,
    p_final: usize,
    ident_transform: Option<Array2<f64>>,
    full_ident_transform: Option<Array2<f64>>,
    n_poly: usize,
    radial_kind: RadialScalarKind,
    psi_scale_share: f64,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    let dim = data.ncols();
    if let Some(eta) = fixed_eta
        && eta.len() != dim
    {
        return Err(BasisError::DimensionMismatch(format!(
            "scalar design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        )));
    }

    let force_operator = radial_kind.is_duchon_family();
    if force_operator || should_use_implicit_operators(n, p_final, 1) {
        let metric_eta = fixed_eta
            .map(|eta| eta.to_vec())
            .unwrap_or_else(|| vec![0.0; dim]);
        let op = ImplicitDesignPsiDerivative::new_streaming_scalar(
            shared_owned_data_matrix_from_view(data),
            shared_owned_centers_matrix_from_view(centers),
            metric_eta,
            radial_kind,
            ident_transform,
            full_ident_transform,
            n_poly,
        )
        .with_psi_scale_share(psi_scale_share);
        return Ok(ScalarDesignPsiDerivatives {
            design_first: Array2::<f64>::zeros((0, 0)),
            design_second_diag: Array2::<f64>::zeros((0, 0)),
            implicit_operator: Some(op),
        });
    }

    let nk = n * k;
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, 1));

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = n.div_ceil(cs);
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ef = &err_flag;
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = (start + cs).min(n);
            let mut data_row_buf = vec![0.0; dim];
            let mut center_buf = vec![0.0; dim];
            for i in start..end {
                for a in 0..dim {
                    data_row_buf[a] = data[[i, a]];
                }
                for j in 0..k {
                    let (r, scalar_component) = if let Some(eta) = fixed_eta {
                        for a in 0..dim {
                            center_buf[a] = centers[[j, a]];
                        }
                        let (r, components) =
                            aniso_distance_and_components(&data_row_buf, &center_buf, eta);
                        (r, components.into_iter().sum::<f64>())
                    } else {
                        let r =
                            stable_euclidean_norm((0..dim).map(|a| data[[i, a]] - centers[[j, a]]));
                        (r, r * r)
                    };
                    let (phi, q, t) = match radial_kind.eval_design_triplet(r) {
                        Ok(p) => p,
                        Err(_) => {
                            ef.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                    };
                    let flat = i * k + j;
                    unsafe {
                        *pp.add(flat) = phi;
                        *qp.add(flat) = q;
                        *tp.add(flat) = t;
                        *ap.add(flat) = scalar_component;
                    }
                }
            }
        });
    }
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(BasisError::InvalidInput(
            "radial scalar evaluation failed during scalar derivative construction".into(),
        ));
    }

    let op = ImplicitDesignPsiDerivative::new(
        phi_values,
        q_values,
        t_values,
        axis_components,
        ident_transform,
        full_ident_transform,
        n,
        k,
        n_poly,
        1,
    )
    .with_psi_scale_share(psi_scale_share);

    Ok(ScalarDesignPsiDerivatives {
        design_first: op.materialize_first(0)?,
        design_second_diag: op.materialize_second_diag(0)?,
        implicit_operator: Some(op),
    })
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
    #[derive(Clone, Copy)]
    struct Leaf {
        start: usize,
        end: usize,
    }

    // Recursive equal-mass partition that always splits the leaf along its widest
    // coordinate dimension. This addresses the root cause of PC1-only slicing by
    // adapting splits to the local geometry of each partition. Keep all row indices
    // in a single buffer and sort subranges in-place so center selection stays exact
    // without allocating fresh index vectors at every split.
    let mut order: Vec<usize> = (0..n).collect();
    let mut leaves = vec![Leaf { start: 0, end: n }];

    let choose_split_dim = |slice: &[usize]| -> usize {
        let mut best_dim = 0usize;
        let mut best_span = f64::NEG_INFINITY;
        for j in 0..d {
            let mut minv = f64::INFINITY;
            let mut maxv = f64::NEG_INFINITY;
            for &idx in slice {
                let v = data[[idx, j]];
                if v < minv {
                    minv = v;
                }
                if v > maxv {
                    maxv = v;
                }
            }
            let span = maxv - minv;
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
            let leaf_size = leaf.end - leaf.start;
            if leaf_size > split_size && leaf_size > 1 {
                split_size = leaf_size;
                split_pos = Some(i);
            }
        }
        let Some(pos) = split_pos else {
            break;
        };

        let leaf = leaves.swap_remove(pos);
        let split_dim = choose_split_dim(&order[leaf.start..leaf.end]);
        order[leaf.start..leaf.end].sort_by(|&a, &b| {
            let ord = data[[a, split_dim]].total_cmp(&data[[b, split_dim]]);
            if ord.is_eq() { a.cmp(&b) } else { ord }
        });
        let mid = leaf.start + (split_size / 2);

        if mid == leaf.start || mid == leaf.end {
            leaves.push(leaf);
            break;
        }

        leaves.push(Leaf {
            start: leaf.start,
            end: mid,
        });
        leaves.push(Leaf {
            start: mid,
            end: leaf.end,
        });
    }

    if leaves.len() < num_centers {
        return Err(BasisError::InvalidInput(format!(
            "equal-mass partition produced {} leaves, expected {num_centers}",
            leaves.len()
        )));
    }

    let mut centers = Array2::<f64>::zeros((num_centers, d));
    for (c, leaf) in leaves.iter().take(num_centers).enumerate() {
        let slice = &order[leaf.start..leaf.end];
        let m = slice.len() as f64;
        let mut centroid = vec![0.0_f64; d];
        for &idx in slice {
            for j in 0..d {
                centroid[j] += data[[idx, j]];
            }
        }
        for v in &mut centroid {
            *v /= m.max(1.0);
        }

        let mut best_idx = slice[0];
        let mut best_d2 = f64::INFINITY;
        for &idx in slice {
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

fn select_equal_mass_covar_representative_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        return Err(BasisError::InvalidInput(format!(
            "equal-mass covariate-representative center selection requested {num_centers} centers but data has {n} rows"
        )));
    }
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "equal-mass covariate-representative center selection requires at least one column"
                .to_string(),
        ));
    }

    let mut split_dim = 0usize;
    let mut best_span = f64::NEG_INFINITY;
    for j in 0..d {
        let mut minv = f64::INFINITY;
        let mut maxv = f64::NEG_INFINITY;
        for i in 0..n {
            let v = data[[i, j]];
            if v < minv {
                minv = v;
            }
            if v > maxv {
                maxv = v;
            }
        }
        let span = maxv - minv;
        if span > best_span {
            best_span = span;
            split_dim = j;
        }
    }

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        let ord = data[[a, split_dim]].total_cmp(&data[[b, split_dim]]);
        if ord.is_eq() { a.cmp(&b) } else { ord }
    });

    let mut centers = Array2::<f64>::zeros((num_centers, d));
    for c in 0..num_centers {
        let lo = (c * n) / num_centers;
        let hi = ((c + 1) * n) / num_centers;
        let chunk = &sorted[lo..hi.max(lo + 1)];
        let mid = chunk[chunk.len() / 2];
        centers.row_mut(c).assign(&data.row(mid));
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
    const KMEANS_PILOT_MAX_ROWS: usize = 20_000;
    if n > KMEANS_PILOT_MAX_ROWS {
        let pilot_n = KMEANS_PILOT_MAX_ROWS.max(num_centers);
        // log::info! rather than warn! — this is a deliberate performance
        // choice (O(n·k·iter) kmeans scales badly past ~20K rows), not a
        // problem the user can act on. Surfacing it as a warning adds
        // noise to CI output and mislabels normal operation.
        log::info!(
            "kmeans center selection using {}-row pilot subsample instead of full {} rows",
            pilot_n,
            n
        );
        let pilot = select_equal_mass_covar_representative_centers(data, pilot_n)?;
        return select_kmeans_centers(pilot.view(), num_centers, max_iter);
    }
    let mut centers = select_thin_plate_knots(data, num_centers)?;
    let mut assign = vec![0usize; n];
    let iters = max_iter.max(1);

    // For large n (biobank-scale), parallelize the assignment step.
    // Each observation's nearest-center query is independent.
    let use_parallel = n >= 10_000;

    for _ in 0..iters {
        // Assignment: find nearest center for each observation.
        if use_parallel {
            const KMEANS_CHUNK: usize = 4096;
            assign
                .par_chunks_mut(KMEANS_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * KMEANS_CHUNK;
                    for (local, slot) in chunk.iter_mut().enumerate() {
                        let i = base + local;
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
                        *slot = best;
                    }
                });
        } else {
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
        }
        // Update: recompute centroids from assignments.
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
        let minv = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let maxv = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        axes.push(Array::linspace(minv, maxv, points_per_dim));
    }
    cartesian_grid_axes(&axes)
}

pub fn select_centers_by_strategy(
    data: ArrayView2<'_, f64>,
    strategy: &CenterStrategy,
) -> Result<Array2<f64>, BasisError> {
    match realized_center_strategy(strategy) {
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
        CenterStrategy::EqualMassCovarRepresentative { num_centers } => {
            select_equal_mass_covar_representative_centers(data, *num_centers)
        }
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
        CenterStrategy::Auto(_) => unreachable!("realized center strategy must not be nested auto"),
    }
}

/// Generic 1D B-spline builder returning design + penalty list.
pub fn build_bspline_basis_1d(
    data: ArrayView1<'_, f64>,
    spec: &BSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let prefer_sparse_design = matches!(
        spec.identifiability,
        BSplineIdentifiability::None | BSplineIdentifiability::WeightedSumToZero { .. }
    );
    let (design_sparse_opt, design_dense_opt, knots) = if prefer_sparse_design {
        match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => {
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Generate {
                        data_range: *data_range,
                        num_internal_knots: *num_internal_knots,
                    },
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
            BSplineKnotSpec::Provided(knots) => {
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
            BSplineKnotSpec::Automatic {
                num_internal_knots,
                placement,
            } => {
                let inferred = num_internal_knots.unwrap_or_else(|| {
                    default_internal_knot_count_for_data(data.len(), spec.degree)
                });
                let knots = match placement {
                    BSplineKnotPlacement::Uniform => {
                        let range = finite_data_range(data)?;
                        internal::generate_full_knot_vector(range, inferred, spec.degree)?
                    }
                    BSplineKnotPlacement::Quantile => {
                        internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                    }
                };
                let (basis, knots) = create_basis::<Sparse>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (Some(basis), None, knots)
            }
        }
    } else {
        match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => {
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Generate {
                        data_range: *data_range,
                        num_internal_knots: *num_internal_knots,
                    },
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
            BSplineKnotSpec::Provided(knots) => {
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
            BSplineKnotSpec::Automatic {
                num_internal_knots,
                placement,
            } => {
                let inferred = num_internal_knots.unwrap_or_else(|| {
                    default_internal_knot_count_for_data(data.len(), spec.degree)
                });
                let knots = match placement {
                    BSplineKnotPlacement::Uniform => {
                        let range = finite_data_range(data)?;
                        internal::generate_full_knot_vector(range, inferred, spec.degree)?
                    }
                    BSplineKnotPlacement::Quantile => {
                        internal::generate_full_knot_vector_quantile(data, inferred, spec.degree)?
                    }
                };
                let (basis, knots) = create_basis::<Dense>(
                    data,
                    KnotSource::Provided(knots.view()),
                    spec.degree,
                    BasisOptions::value(),
                )?;
                (None, Some((*basis).clone()), knots)
            }
        }
    };
    let p_raw = design_sparse_opt
        .as_ref()
        .map(|basis| basis.ncols())
        .or_else(|| design_dense_opt.as_ref().map(Array2::ncols))
        .expect("B-spline basis should be present");
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
        kronecker_factors: None,
    }];
    if spec.double_penalty {
        penalties_raw.push(PenaltyCandidate {
            matrix: build_nullspace_shrinkage_penalty(&s_bend_raw)?
                .map(|shrink| shrink.sym_penalty)
                .unwrap_or_else(|| Array2::<f64>::zeros(s_bend_raw.raw_dim())),
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: 1.0,
            kronecker_factors: None,
        });
    }

    let penalties_raw_mats: Vec<Array2<f64>> = penalties_raw
        .iter()
        .map(|candidate| candidate.matrix.clone())
        .collect();
    let (design, transformed_candidates, identifiability_transform) = if let Some(sparse_basis) =
        design_sparse_opt
    {
        match &spec.identifiability {
            BSplineIdentifiability::None => {
                let transformed_candidates = penalties_raw
                    .into_iter()
                    .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                        Ok(PenaltyCandidate {
                            nullspace_dim_hint: candidate.nullspace_dim_hint,
                            matrix: candidate.matrix,
                            source: candidate.source,
                            normalization_scale: candidate.normalization_scale,
                            kronecker_factors: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse_basis)),
                    transformed_candidates,
                    None,
                )
            }
            BSplineIdentifiability::WeightedSumToZero { weights } => {
                let (constrained_basis, z) = apply_sum_to_zero_constraint_sparse(
                    &sparse_basis,
                    weights.as_ref().map(|w| w.view()),
                )?;
                let transformed_candidates = penalties_raw
                    .into_iter()
                    .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                        let zt_s = fast_atb(&z, &candidate.matrix);
                        let matrix = fast_ab(&zt_s, &z);
                        Ok(PenaltyCandidate {
                            nullspace_dim_hint: candidate.nullspace_dim_hint,
                            matrix,
                            source: candidate.source,
                            normalization_scale: candidate.normalization_scale,
                            kronecker_factors: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(constrained_basis)),
                    transformed_candidates,
                    Some(z),
                )
            }
            _ => unreachable!("sparse B-spline identifiability only supports sum-to-zero"),
        }
    } else {
        let (design, penalties, identifiability_transform) = apply_bspline_identifiability_policy(
            design_dense_opt.expect("dense B-spline basis should be present"),
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
                        nullspace_dim_hint: candidate.nullspace_dim_hint,
                        matrix,
                        source: candidate.source,
                        normalization_scale: candidate.normalization_scale,
                        kronecker_factors: None,
                    })
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        (
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
            transformed_candidates,
            identifiability_transform,
        )
    };
    let (penalties, nullspace_dims, penaltyinfo) =
        filter_active_penalty_candidates(transformed_candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::BSpline1D {
            knots,
            identifiability_transform,
        },
        kronecker_factored: None,
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
            let (z, _) = compute_geometric_constraint_transform(knots, degree, 2)?;
            (design.dot(&z), Some(z))
        }
        BSplineIdentifiability::OrthogonalToDesignColumns { columns, weights } => {
            let (b_c, z) = applyweighted_orthogonality_constraint(
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
    // Keep the cutoff in eigenvalue units so uniform penalty scaling does not
    // change PSD/rank decisions for the same spectrum shape.
    (sym.nrows().max(1) as f64) * 1e-10 * max_abs_ev
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
            eigenvalues: Array1::<f64>::zeros(0),
            eigenvectors: Array2::<f64>::zeros((0, 0)),
            rank: 0,
            nullity: 0,
            tol: 1e-10,
            iszero: true,
        });
    }

    let (sym, evals, evecs) = spectral_summary(penalty)?;
    let tol = spectral_tolerance(&sym, &evals);
    let rank = evals.iter().filter(|&&ev| ev > tol).count();
    let nullity = sym.nrows().saturating_sub(rank);
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    Ok(CanonicalPenaltyBlock {
        sym_penalty: sym,
        eigenvalues: evals,
        eigenvectors: evecs,
        rank,
        nullity,
        tol,
        iszero: max_abs_eigenvalue <= tol,
    })
}

pub fn filter_active_penalty_candidates(
    candidates: Vec<PenaltyCandidate>,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let mut penalties = Vec::with_capacity(candidates.len());
    let mut nullspace_dims = Vec::with_capacity(candidates.len());
    let mut penaltyinfo = Vec::with_capacity(candidates.len());

    for (original_index, candidate) in candidates.into_iter().enumerate() {
        let analysis = analyze_penalty_block(&candidate.matrix)?;
        let dropped_reason = if analysis.rank == 0 {
            Some(if analysis.iszero {
                PenaltyDropReason::ZeroMatrix
            } else {
                PenaltyDropReason::NumericalRankZero
            })
        } else {
            None
        };
        let active = dropped_reason.is_none();
        let kronecker_factors =
            validated_kronecker_factors(candidate.kronecker_factors, &analysis.sym_penalty);
        if active {
            log::debug!(
                "Retained penalty block source={:?} original_index={} rank={} nullspace_dim_hint={}",
                candidate.source,
                original_index,
                analysis.rank,
                analysis.nullity
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
        penaltyinfo.push(PenaltyInfo {
            source: candidate.source,
            original_index,
            active,
            effective_rank: analysis.rank,
            dropped_reason,
            nullspace_dim_hint: analysis.nullity,
            normalization_scale: candidate.normalization_scale,
            kronecker_factors,
        });
    }

    Ok((penalties, nullspace_dims, penaltyinfo))
}

fn validated_kronecker_factors(
    factors: Option<Vec<Array2<f64>>>,
    matrix: &Array2<f64>,
) -> Option<Vec<Array2<f64>>> {
    let factors = factors?;
    let Some((first, rest)) = factors.split_first() else {
        return None;
    };
    let mut kron = first.clone();
    for factor in rest {
        kron = crate::construction::kronecker_product(&kron, factor);
    }
    if kron.dim() != matrix.dim() {
        return None;
    }

    let scale = kron
        .iter()
        .chain(matrix.iter())
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1.0);
    let max_abs_diff = kron
        .iter()
        .zip(matrix.iter())
        .fold(0.0_f64, |acc, (&lhs, &rhs)| acc.max((lhs - rhs).abs()));
    (max_abs_diff <= scale * 1e-10).then_some(factors)
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
        eigenvalues: evals,
        eigenvectors: evecs,
        rank: zero_idx.len(),
        nullity: 0,
        tol,
        iszero: false,
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
    let mut minv = f64::INFINITY;
    let mut maxv = f64::NEG_INFINITY;
    for &x in data {
        if x < minv {
            minv = x;
        }
        if x > maxv {
            maxv = x;
        }
    }
    Ok((minv, maxv))
}

/// Generic thin-plate builder returning design + penalty list.
pub fn build_thin_plate_basis(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basiswithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let internal_kernel_transform =
        thin_plate_kernel_constraint_nullspace(centers.view(), &mut workspace.cache)?;
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let base_cols = internal_kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
    if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "thin-plate basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            base_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    let (design, identifiability_transform, mut candidates) = if use_lazy {
        let poly_block = thin_plate_polynomial_block(data);
        let d = data.ncols();
        let length_scale_sq = spec.length_scale * spec.length_scale;
        let shared_data = shared_owned_data_matrix(data, &mut workspace.cache);
        let kernel_fn = move |data_row: &[f64], center_row: &[f64]| -> f64 {
            let mut dist2 = 0.0;
            for axis in 0..d {
                let delta = data_row[axis] - center_row[axis];
                dist2 += delta * delta;
            }
            thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
                .expect("validated thin-plate inputs should not fail")
        };
        let base_op = ChunkedKernelDesignOperator::new(
            shared_data,
            Arc::new(centers.clone()),
            kernel_fn,
            Some(Arc::new(internal_kernel_transform.clone())),
            Some(Arc::new(poly_block)),
        )
        .map_err(BasisError::InvalidInput)?;
        let base_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)));
        let identifiability_transform = spatial_identifiability_transform_from_design_matrix(
            data,
            &base_design,
            &spec.identifiability,
            "ThinPlate",
        )?;
        let design = if let Some(transform) = identifiability_transform.as_ref() {
            wrap_dense_design_with_transform(base_design, transform, "ThinPlate")?
        } else {
            base_design
        };
        let (penalty_bending, penalty_ridge) = build_thin_plate_penalty_matrices(
            centers.view(),
            spec.length_scale,
            &internal_kernel_transform,
            spec.double_penalty,
        )?;
        let mut candidates = vec![PenaltyCandidate {
            matrix: penalty_bending,
            nullspace_dim_hint: poly_cols,
            source: PenaltySource::Primary,
            normalization_scale: 1.0,
            kronecker_factors: None,
        }];
        if let Some(penalty_ridge) = penalty_ridge {
            candidates.push(PenaltyCandidate {
                matrix: penalty_ridge,
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: 1.0,
                kronecker_factors: None,
            });
        }
        (design, identifiability_transform, candidates)
    } else {
        let tps = create_thin_plate_spline_basis_scaledwithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            workspace,
        )?;
        let identifiability_transform = spatial_identifiability_transform_from_design(
            data,
            tps.basis.view(),
            &spec.identifiability,
            "ThinPlate",
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(
                &tps.basis, z,
            )))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(tps.basis.clone()))
        };
        let mut candidates = vec![PenaltyCandidate {
            matrix: tps.penalty_bending.clone(),
            nullspace_dim_hint: tps.num_polynomial_basis,
            source: PenaltySource::Primary,
            normalization_scale: 1.0,
            kronecker_factors: None,
        }];
        if spec.double_penalty {
            candidates.push(PenaltyCandidate {
                matrix: tps.penalty_ridge.clone(),
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: 1.0,
                kronecker_factors: None,
            });
        }
        (design, identifiability_transform, candidates)
    };
    if let Some(z) = identifiability_transform.as_ref() {
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = z.t().dot(&candidate.matrix);
                let matrix = zt_s.dot(z);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: candidate.nullspace_dim_hint,
                    matrix,
                    source: candidate.source,
                    normalization_scale: candidate.normalization_scale,
                    kronecker_factors: None,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    let (penalties, nullspace_dims, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::ThinPlate {
            centers,
            length_scale: spec.length_scale,
            identifiability_transform,
            input_scales: None,
        },
        kronecker_factored: None,
    })
}

#[inline(always)]
fn horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[inline(always)]
fn stable_nonnegative_poly_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    if x <= 600.0 {
        return horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
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
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0])
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
            )
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
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0 / 3.0, -1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 5.0, -1.0 / 5.0, -1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 7.0, -1.0 / 7.0, -2.0 / 35.0, -1.0 / 105.0],
            )
        }
    };
    Ok(deriv)
}

#[inline(always)]
fn matern_kernel_log_kappasecond_derivative_from_distance(
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
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0, 1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -2.0 / 5.0, -2.0 / 5.0, -1.0 / 15.0, 1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[
                    0.0,
                    0.0,
                    -2.0 / 7.0,
                    -2.0 / 7.0,
                    -3.0 / 35.0,
                    1.0 / 105.0,
                    1.0 / 105.0,
                ],
            )
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
fn matern_kernel_radial_tripletwith_safe_ratio(
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

/// Anisotropic radial scalars for Matérn kernel.
///
/// Returns (φ, q, t) — the R-operator radial scalars needed for per-axis
/// kappa derivatives under geometric anisotropy.
///
/// # R-operator hierarchy
///
/// Define the R-operator as Rg := g'(r)/r. Successive applications give:
///   q = Rφ   = φ'(r)/r
///   t = R²φ  = (φ''(r) - q) / r²  =  q'(r) / r
///   u = R³φ  = t'(r) / r           (not computed here; needed for D₂ second derivatives)
///
/// # Per-axis ψ_a derivatives via R-operators
///
/// For anisotropic distance r = |Λh| with s_a = κ_a² h_a², the kernel
/// derivatives w.r.t. per-axis log-scales ψ_a = log(κ_a) are:
///
///   ∂φ/∂ψ_a = q · s_a
///   ∂²φ/(∂ψ_a ∂ψ_b) = 2q · s_a · δ_{ab}  +  t · s_a · s_b
///
/// The Hessian is "diagonal + rank-1":  ∇²_ψ φ = 2q Diag(s) + t ss'.
/// Hessian-vector products are O(d) per point pair.
///
/// For nu = 1/2 and nu = 3/2 the scalar t diverges at r = 0 because the
/// kernel lacks sufficient smoothness (C^0 and C^1 respectively).  These
/// cases return an error when r is exactly zero; at r > 0 the formulas are
/// well defined and computed directly.
///
/// For nu >= 5/2 the quantity (phi'' - q) contains a factor of r^2 that
/// cancels analytically, yielding a closed-form t with a finite r -> 0
/// limit.
///
/// Closed-form t (a = s*r, s = sqrt(2*nu)/length_scale, E = exp(-a)):
///   nu = 5/2:  t = (s^4 / 3)  E
///   nu = 7/2:  t = (s^4 / 15) E (a + 1)
///   nu = 9/2:  t = (s^4 / 105) (a^2 + 3a + 3) E
///
/// Collision limits t(0):
///   nu = 5/2:  s^4 / 3
///   nu = 7/2:  s^4 / 15
///   nu = 9/2:  s^4 / 35  (= 3 s^4 / 105)
#[cfg(test)]
fn matern_aniso_radial_scalars(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn aniso scalar distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    let (phi, q, t) = match nu {
        // ----------------------------------------------------------------
        // nu = 1/2:  phi = exp(-a), a = r / length_scale.
        //   phi'/r diverges at r = 0 (cusp kernel).
        //   t also diverges at r = 0.
        // ----------------------------------------------------------------
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            if r < 1e-14 {
                return Err(BasisError::InvalidInput(
                    "Matérn nu=1/2 aniso scalars q and t diverge at r=0".to_string(),
                ));
            }
            // phi' = -s E, q = phi'/r = -s E / r
            let q = -s * e / r;
            // phi'' = s^2 E, t = (phi'' - q) / r^2
            let t = (s * s * e - q) / (r * r);
            (phi, q, t)
        }
        // ----------------------------------------------------------------
        // nu = 3/2:  phi = (1 + a) exp(-a), a = sqrt(3) r / length_scale.
        //   q = -s^2 E  (finite at r = 0).
        //   phi'' - q = s^2 a E = s^3 r E  =>  t = s^3 E / r  =>  diverges at r = 0.
        // ----------------------------------------------------------------
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let q = -s * s * e; // exact closed form, finite at r = 0
            if r < 1e-14 {
                return Err(BasisError::InvalidInput(
                    "Matérn nu=3/2 aniso scalar t diverges at r=0".to_string(),
                ));
            }
            // phi'' - q = s^2 a E = s^3 r E  =>  t = s^3 E / r
            let t = s * s * s * e / r;
            (phi, q, t)
        }
        // ----------------------------------------------------------------
        // nu = 5/2:  phi = (1 + a + a^2/3) exp(-a).
        //   q   = -(s^2/3) (a + 1) E
        //   phi'' - q = (s^2/3) a^2 E = (s^4/3) r^2 E
        //   t   = (s^4/3) E
        //   t(0) = s^4/3
        // ----------------------------------------------------------------
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let q = -(s * s / 3.0) * (a + 1.0) * e;
            let t = (s * s * s * s / 3.0) * e;
            (phi, q, t)
        }
        // ----------------------------------------------------------------
        // nu = 7/2:  phi = (1 + a + (2/5)a^2 + (1/15)a^3) exp(-a).
        //   q   = -(s^2/15)(a^2 + 3a + 3) E
        //   phi'' - q = (s^2/15) a^2 (a + 1) E
        //   t   = (s^4/15)(a + 1) E
        //   t(0) = s^4/15
        // ----------------------------------------------------------------
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let q = -(s * s / 15.0) * (a * a + 3.0 * a + 3.0) * e;
            let t = (s * s * s * s / 15.0) * (a + 1.0) * e;
            (phi, q, t)
        }
        // ----------------------------------------------------------------
        // nu = 9/2:  phi = (1 + a + (3/7)a^2 + (2/21)a^3 + (1/105)a^4) exp(-a).
        //   q   = -(s^2/105)(a^3 + 6a^2 + 15a + 15) E
        //   phi'' - q = (s^2/105) a^2 (a^2 + 3a + 3) E
        //   t   = (s^4/105)(a^2 + 3a + 3) E
        //   t(0) = 3 s^4 / 105 = s^4 / 35
        // ----------------------------------------------------------------
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
            let q = -(s * s / 105.0) * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0) * e;
            let t = (s * s * s * s / 105.0) * (a * a + 3.0 * a + 3.0) * e;
            (phi, q, t)
        }
    };

    if !phi.is_finite() || !q.is_finite() || !t.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Matérn aniso radial scalars at r={r}, length_scale={length_scale}, nu={nu:?}"
        )));
    }
    Ok((phi, q, t))
}

/// Extended radial scalars for exact per-axis η_a derivatives of the Matérn
/// operator collocation matrices D₁ (gradient) and D₂ (Laplacian).
///
/// Returns `(phi, q, t, dt_dr, d2t_dr2)` where:
///   - phi   = φ(r)                  (kernel value)
///   - q     = φ'(r)/r               (used in D₁)
///   - t     = (φ''(r) - q) / r²     (used in D₀/D₁ per-axis chain)
///   - dt_dr = dt/dr                 (needed for D₁ second η-derivative)
///   - d2t_dr2 = d²t/dr²            (needed for D₂ second η-derivative)
///
/// At r = 0 (center collision), the function returns zeros for all quantities
/// that would be multiplied by s_a (which also vanishes at collision).
///
/// For ν = 1/2 and ν = 3/2 where t and/or dt_dr diverge at r = 0, the
/// collision entries are safe because D₁ and D₂ derivatives at coincident
/// centers vanish via s_a = 0.
fn matern_aniso_extended_radial_scalars(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn extended radial scalar distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Matérn length_scale must be finite and positive".to_string(),
        ));
    }

    match nu {
        // ----------------------------------------------------------------
        // ν = 1/2:  φ = exp(-a), a = r / ℓ, s = 1/ℓ
        //   q = -s·E/r  (diverges at r=0)
        //   t = (s²E - q) / r²  (diverges at r=0)
        //   At r=0 all products with s_a vanish, so return 0 for dt_dr, d2t_dr2.
        // ----------------------------------------------------------------
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            if r < 1e-14 {
                // Collision: q diverges but s_a = 0 ⇒ products vanish.
                return Ok((phi, 0.0, 0.0, 0.0, 0.0));
            }
            let q = -s * e / r;
            let phi_rr = s * s * e;
            let t = (phi_rr - q) / (r * r);
            // t' from: t = f/r² where f = φ'' - q.
            //   f'  = φ''' - q' = -s³E - t·r   (since q' = t·r)
            //   t'  = (f' - 2t·r) / r²  = (-s³E - 3t·r) / r²
            let dt_dr = (-s * s * s * e - 3.0 * t * r) / (r * r);
            // t'' from: t' = g/r² where g = -s³E - 3tr.
            //   g' = s⁴E - 3(t'r + t)
            //   t'' = (g' - 2t'r) / r² = (s⁴E - 3t'r - 3t - 2t'r) / r²
            //        = (s⁴E - 5t'r - 3t) / r²
            let d2t_dr2 = (s.powi(4) * e - 5.0 * dt_dr * r - 3.0 * t) / (r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 3/2:  φ = (1 + a)E, a = √3·r/ℓ, s = √3/ℓ
        //   q  = -s²E         (finite at r=0)
        //   t  = s³E/r        (diverges at r=0)
        //   dt/dr = s³E(-sr - 1)/r²  (diverges at r=0)
        //   At r=0, s_a = 0 so all products vanish.
        // ----------------------------------------------------------------
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let q = -s * s * e;
            if r < 1e-14 {
                return Ok((phi, q, 0.0, 0.0, 0.0));
            }
            let t = s * s * s * e / r;
            // dt/dr: d/dr [s³ E / r] = s³ [-s E r - E] / r² = -s³ E (sr + 1) / r²
            let dt_dr = -s * s * s * e * (a + 1.0) / (r * r);
            // d²t/dr²: d/dr [-s³ E (a+1) / r²]
            //   = -s³ [(-s E)(a+1)r² + s E r² - 2r E(a+1)] / r⁴ ... expand
            // Let g(r) = -s³ E (a+1) / r²
            // g'(r) = -s³ [E'(a+1) + E·s] / r² + 2s³ E(a+1) / r³
            //       = -s³ [-sE(a+1) + sE] / r² + 2s³ E(a+1) / r³
            //       = -s³ · sE[-a-1+1] / r² + 2s³ E(a+1) / r³
            //       = s⁴ a E / r² + 2s³ E(a+1) / r³
            //       = s³ E [s a r + 2(a+1)] / r³
            let d2t_dr2 = s * s * s * e * (s * a * r + 2.0 * (a + 1.0)) / (r * r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 5/2:  φ = (1 + a + a²/3)E, a = √5·r/ℓ, s = √5/ℓ
        //   q = -(s²/3)(a+1)E
        //   t = (s⁴/3)E
        //   dt/dr = -(s⁵/3)E
        //   d²t/dr² = (s⁶/3)E
        // ----------------------------------------------------------------
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let q = -(s * s / 3.0) * (a + 1.0) * e;
            let t = (s * s * s * s / 3.0) * e;
            let dt_dr = -(s * s * s * s * s / 3.0) * e;
            let d2t_dr2 = (s.powi(6) / 3.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 7/2:  φ = (1 + a + (2/5)a² + (1/15)a³)E
        //   q = -(s²/15)(a² + 3a + 3)E
        //   t = (s⁴/15)(a + 1)E
        //   dt/dr = -(s⁵/15)aE
        //   d²t/dr² = (s⁶/15)(a - 1)E
        // ----------------------------------------------------------------
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let q = -(s * s / 15.0) * (a * a + 3.0 * a + 3.0) * e;
            let t = (s * s * s * s / 15.0) * (a + 1.0) * e;
            let dt_dr = -(s.powi(5) / 15.0) * a * e;
            let d2t_dr2 = (s.powi(6) / 15.0) * (a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 9/2:  φ = (1 + a + (3/7)a² + (2/21)a³ + (1/105)a⁴)E
        //   q = -(s²/105)(a³ + 6a² + 15a + 15)E
        //   t = (s⁴/105)(a² + 3a + 3)E
        //   dt/dr = -(s⁵/105)a(a + 1)E
        //   d²t/dr² = (s⁶/105)(a² - a - 1)E
        // ----------------------------------------------------------------
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
            let q = -(s * s / 105.0) * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0) * e;
            let t = (s * s * s * s / 105.0) * (a * a + 3.0 * a + 3.0) * e;
            let dt_dr = -(s.powi(5) / 105.0) * a * (a + 1.0) * e;
            let d2t_dr2 = (s.powi(6) / 105.0) * (a * a - a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
    }
}

/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic Matérn terms.
///
/// Instead of the fractional approximation `dS_op/dη_a ≈ f_a · dS_op/dψ`,
/// this computes exact first and second η_a derivatives of each operator
/// collocation matrix (D₀, D₁, D₂) and assembles the Gram product-rule
/// derivatives:
///   S_{m,a}  = D_{m,a}ᵀ D_m + D_mᵀ D_{m,a}
///   S_{m,aa} = D_{m,aa}ᵀ D_m + 2 D_{m,a}ᵀ D_{m,a} + D_mᵀ D_{m,aa}
///
/// ## Per-axis derivative formulas (y-space operators)
///
/// With r = √(Σ exp(2η_a) h_a²) and s_a = exp(2η_a) h_a²:
///
/// **D₀[k,j] = φ(r):**
///   ∂φ/∂η_a = q · s_a
///   ∂²φ/∂η_a² = t · s_a² + 2q · s_a
///
/// **D₁[(k,b),j] = q(r) · h_b** (y-space gradient):
///   ∂D₁/∂η_a = t · s_a · h_b
///   ∂²D₁/∂η_a² = (dt/dr · s_a²/r + 2t · s_a) · h_b
///
/// **D₂[k,j] = φ''(r) + (d-1)·q(r)** (y-space Laplacian):
///   ∂D₂/∂η_a = [(d+2)·t + dt/dr · r] · s_a
///   ∂²D₂/∂η_a² = [(d+3)·dt/dr/r + d²t/dr²] · s_a² + 2·[(d+2)·t + dt/dr·r] · s_a
struct MaternCrossPenaltyContext {
    centers: Array2<f64>,
    aniso_log_scales: Vec<f64>,
    length_scale: f64,
    nu: MaternNu,
    z_transform: Option<Array2<f64>>,
    penaltyinfo: Vec<PenaltyInfo>,
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    d0_eta_proj: Vec<Array2<f64>>,
    d1_eta_proj: Vec<Array2<f64>>,
    d2_eta_proj: Vec<Array2<f64>>,
    op0_s_raw: Array2<f64>,
    op1_s_raw: Array2<f64>,
    op2_s_raw: Array2<f64>,
    op0_c: f64,
    op1_c: f64,
    op2_c: f64,
    op0_s_first_raw: Vec<Array2<f64>>,
    op1_s_first_raw: Vec<Array2<f64>>,
    op2_s_first_raw: Vec<Array2<f64>>,
}

impl MaternCrossPenaltyContext {
    fn project_operator(&self, mat: &Array2<f64>, row_dim: usize) -> Array2<f64> {
        let kernel = if let Some(z) = self.z_transform.as_ref() {
            fast_ab(mat, z)
        } else {
            mat.clone()
        };
        let mut padded = Array2::<f64>::zeros((row_dim, self.d0.ncols()));
        padded.slice_mut(s![.., 0..kernel.ncols()]).assign(&kernel);
        padded
    }

    fn compute_pair(&self, axis_a: usize, axis_b: usize) -> Result<Vec<Array2<f64>>, BasisError> {
        let p = self.centers.nrows();
        let d = self.centers.ncols();
        let mut d0_cross_raw = Array2::<f64>::zeros((p, p));
        let mut d1_cross_raw = Array2::<f64>::zeros((p * d, p));
        let mut d2_cross_raw = Array2::<f64>::zeros((p, p));
        let d_f64 = d as f64;

        for k in 0..p {
            for j in 0..p {
                let ci: Vec<f64> = (0..d).map(|axis| self.centers[[k, axis]]).collect();
                let cj: Vec<f64> = (0..d).map(|axis| self.centers[[j, axis]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, &self.aniso_log_scales);
                let (_, _, t, dt_dr, d2t_dr2) =
                    matern_aniso_extended_radial_scalars(r, self.length_scale, self.nu)?;
                let s_a = s_vec[axis_a];
                let s_b = s_vec[axis_b];
                let sa_sb = s_a * s_b;

                d0_cross_raw[[k, j]] = t * sa_sb;
                for axis in 0..d {
                    let h_axis = ci[axis] - cj[axis];
                    let row = k * d + axis;
                    d1_cross_raw[[row, j]] = if r > 1e-14 {
                        dt_dr * sa_sb / r * h_axis
                    } else {
                        0.0
                    };
                }
                d2_cross_raw[[k, j]] = if r > 1e-14 {
                    let dw_dr = (d_f64 + 3.0) * dt_dr + d2t_dr2 * r;
                    dw_dr * sa_sb / r
                } else {
                    0.0
                };
            }
        }

        let d0_cross_proj = self.project_operator(&d0_cross_raw, p);
        let d1_cross_proj = self.project_operator(&d1_cross_raw, p * d);
        let d2_cross_proj = self.project_operator(&d2_cross_raw, p);

        let s0_cross = normalize_penalty_cross_psi_derivative(
            &self.op0_s_raw,
            &self.op0_s_first_raw[axis_a],
            &self.op0_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d0,
                &self.d0_eta_proj[axis_a],
                &self.d0_eta_proj[axis_b],
                &d0_cross_proj,
            ),
            self.op0_c,
        );
        let s1_cross = normalize_penalty_cross_psi_derivative(
            &self.op1_s_raw,
            &self.op1_s_first_raw[axis_a],
            &self.op1_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d1,
                &self.d1_eta_proj[axis_a],
                &self.d1_eta_proj[axis_b],
                &d1_cross_proj,
            ),
            self.op1_c,
        );
        let s2_cross = normalize_penalty_cross_psi_derivative(
            &self.op2_s_raw,
            &self.op2_s_first_raw[axis_a],
            &self.op2_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d2,
                &self.d2_eta_proj[axis_a],
                &self.d2_eta_proj[axis_b],
                &d2_cross_proj,
            ),
            self.op2_c,
        );

        active_operator_penalty_derivatives(
            &self.penaltyinfo,
            &[s0_cross, s1_cross, s2_cross],
            "Matérn-aniso-cross",
        )
    }
}

fn build_matern_operator_penalty_aniso_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    eta: &[f64],
) -> Result<
    (
        Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>,
        Vec<(usize, usize)>,
        AnisoPenaltyCrossProvider,
    ),
    BasisError,
> {
    let p = centers.nrows();
    let d = centers.ncols();
    let dim = eta.len();
    assert_eq!(dim, d);

    // Per-axis: build raw D₀, D₁, D₂ and their η_a first/second derivatives.
    // D₀: p × p
    // D₁: (p·d) × p
    // D₂: p × p
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p, p));
    let mut d0_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d0_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let num_cross = dim * (dim - 1) / 2;
    let mut cross_pairs: Vec<(usize, usize)> = Vec::with_capacity(num_cross);
    for a in 0..dim {
        for b in (a + 1)..dim {
            cross_pairs.push((a, b));
        }
    }

    let d_f64 = d as f64;

    for k in 0..p {
        for j in 0..p {
            let ci: Vec<f64> = (0..d).map(|a| centers[[k, a]]).collect();
            let cj: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
            let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);

            let (phi, q, t, dt_dr, d2t_dr2) =
                matern_aniso_extended_radial_scalars(r, length_scale, nu)?;

            // --- D₀ ---
            d0_raw[[k, j]] = phi;

            // --- D₂ (Laplacian) ---
            let lap = if r < 1e-14 {
                // At collision, φ''(0) + (d-1)·q(0).
                // For ν=1/2 this diverges but the penalty contribution at collision
                // is typically handled by the existing operator assembly.
                // Use the same scalar value:
                let (_, _, _, _, _, _, lap_c, _, _) =
                    matern_operator_psi_triplet(r, length_scale, nu, d)?;
                lap_c
            } else {
                // φ'' = q + t·r², so lap = q + t·r² + (d-1)·q = d·q + t·r²
                q + t * r * r + (d_f64 - 1.0) * q
            };
            d2_raw[[k, j]] = lap;

            // --- D₁ (gradient) ---
            for axis in 0..d {
                let h_b = ci[axis] - cj[axis];
                let row = k * d + axis;
                d1_raw[[row, j]] = q * h_b;
            }

            // --- Per-axis η_a derivatives ---
            for a in 0..dim {
                let s_a = s_vec[a];

                // ∂D₀/∂η_a = q · s_a
                d0_raw_eta[a][[k, j]] = q * s_a;
                // ∂²D₀/∂η_a² = t · s_a² + 2q · s_a
                d0_raw_eta2[a][[k, j]] = t * s_a * s_a + 2.0 * q * s_a;

                // ∂D₁/∂η_a: for each axis b, ∂(q · h_b)/∂η_a = (dq/dη_a) · h_b = t · s_a · h_b
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let row = k * d + b;
                    d1_raw_eta[a][[row, j]] = t * s_a * h_b;
                    // ∂²D₁/∂η_a² = (dt/dr · s_a/r · s_a + 2t · s_a) · h_b
                    //             = (dt/dr · s_a²/r + 2t · s_a) · h_b
                    let d2q_deta2 = if r > 1e-14 {
                        dt_dr * s_a * s_a / r + 2.0 * t * s_a
                    } else {
                        // At collision s_a = 0, so the whole expression is 0.
                        0.0
                    };
                    d1_raw_eta2[a][[row, j]] = d2q_deta2 * h_b;
                }

                // ∂D₂/∂η_a = [(d+2)·t + dt_dr · r] · s_a
                let dlap_deta = if r > 1e-14 {
                    ((d_f64 + 2.0) * t + dt_dr * r) * s_a
                } else {
                    0.0
                };
                d2_raw_eta[a][[k, j]] = dlap_deta;

                // ∂²D₂/∂η_a²:
                // Let w = (d+2)·t + dt_dr · r.
                // ∂²D₂/∂η_a² = (dw/dr · s_a/r) · s_a + w · 2 · s_a
                // dw/dr = (d+2)·dt_dr + d²t_dr²·r + dt_dr = (d+3)·dt_dr + d²t_dr²·r
                let d2lap_deta2 = if r > 1e-14 {
                    let w = (d_f64 + 2.0) * t + dt_dr * r;
                    let dw_dr = (d_f64 + 3.0) * dt_dr + d2t_dr2 * r;
                    dw_dr * s_a * s_a / r + 2.0 * w * s_a
                } else {
                    0.0
                };
                d2_raw_eta2[a][[k, j]] = d2lap_deta2;
            }
        }
    }

    // Project through identifiability transform Z (ψ-independent).
    let project = |mat: Array2<f64>| -> Array2<f64> {
        if let Some(z) = z_opt {
            fast_ab(&mat, z)
        } else {
            mat
        }
    };

    let d0_kernel = project(d0_raw);
    let d1_kernel = project(d1_raw);
    let d2_kernel = project(d2_raw);

    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);

    // Pad with intercept column.
    let pad = |kernel_mat: Array2<f64>, nrows: usize, add_intercept_ones: bool| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((nrows, total_cols));
        out.slice_mut(s![.., 0..kernel_cols]).assign(&kernel_mat);
        if add_intercept_ones && include_intercept {
            out.column_mut(kernel_cols).fill(1.0);
        }
        out
    };

    let d0 = pad(d0_kernel, p, true);
    let d1 = pad(d1_kernel, p * d, false);
    let d2 = pad(d2_kernel, p, false);

    // Project and pad all per-axis operator derivative matrices upfront,
    // so they remain available for cross-term computation.
    let d0_eta_all: Vec<Array2<f64>> = d0_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta_all: Vec<Array2<f64>> = d1_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta_all: Vec<Array2<f64>> = d2_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d0_eta2_all: Vec<Array2<f64>> = d0_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta2_all: Vec<Array2<f64>> = d1_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta2_all: Vec<Array2<f64>> = d2_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();

    // Build raw Gram penalties (axis-independent) and their per-axis
    // first/second derivatives + Frobenius norms.
    // We compute these once for axis 0 (the raw Gram S and norm c are the same
    // for all axes) and store them, then reuse c for cross-term normalization.
    struct PerOperatorInfo {
        s_raw: Array2<f64>,
        c: f64,
        s_first: Vec<Array2<f64>>, // per-axis first derivatives (normalized)
        s_second: Vec<Array2<f64>>, // per-axis second derivatives (normalized)
        s_first_raw: Vec<Array2<f64>>, // per-axis first derivatives (raw, for cross normalization)
    }

    let compute_operator_info = |d_op: &Array2<f64>,
                                 d_eta_all: &[Array2<f64>],
                                 d_eta2_all: &[Array2<f64>]|
     -> PerOperatorInfo {
        // Compute the raw Gram and its norm (same for all axes).
        let s_raw = symmetrize(&fast_ata(d_op));
        let fro2: f64 = s_raw.iter().map(|v| v * v).sum();
        let c = fro2.sqrt();

        let mut s_first = Vec::with_capacity(dim);
        let mut s_second = Vec::with_capacity(dim);
        let mut s_first_raw = Vec::with_capacity(dim);
        for a in 0..dim {
            let (_, sa, sa2) =
                gram_and_psi_derivatives_from_operator(d_op, &d_eta_all[a], &d_eta2_all[a]);
            let (_, sa_norm, sa2_norm, _) =
                normalize_penaltywith_psi_derivatives(&s_raw, &sa, &sa2);
            s_first_raw.push(sa);
            s_first.push(sa_norm);
            s_second.push(sa2_norm);
        }

        PerOperatorInfo {
            s_raw,
            c,
            s_first,
            s_second,
            s_first_raw,
        }
    };

    let op0_info = compute_operator_info(&d0, &d0_eta_all, &d0_eta2_all);
    let op1_info = compute_operator_info(&d1, &d1_eta_all, &d1_eta2_all);
    let op2_info = compute_operator_info(&d2, &d2_eta_all, &d2_eta2_all);

    // Build penalty candidates and determine which are active (using axis-0
    // normalized Gram, which is axis-independent).
    let (s0_norm, c0) = if op0_info.c > 1e-12 {
        (op0_info.s_raw.mapv(|v| v / op0_info.c), op0_info.c)
    } else {
        (op0_info.s_raw.clone(), 1.0)
    };
    let (s1_norm, c1) = if op1_info.c > 1e-12 {
        (op1_info.s_raw.mapv(|v| v / op1_info.c), op1_info.c)
    } else {
        (op1_info.s_raw.clone(), 1.0)
    };
    let (s2_norm, c2) = if op2_info.c > 1e-12 {
        (op2_info.s_raw.mapv(|v| v / op2_info.c), op2_info.c)
    } else {
        (op2_info.s_raw.clone(), 1.0)
    };

    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
        },
    ];
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;

    // Build per-axis results.
    let mut per_axis_results = Vec::with_capacity(dim);
    for a in 0..dim {
        let pen_first = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_first[a].clone(),
                op1_info.s_first[a].clone(),
                op2_info.s_first[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        let pen_second = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_second[a].clone(),
                op1_info.s_second[a].clone(),
                op2_info.s_second[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        per_axis_results.push((pen_first, pen_second));
    }

    let cross_ctx = std::sync::Arc::new(MaternCrossPenaltyContext {
        centers: centers.to_owned(),
        aniso_log_scales: eta.to_vec(),
        length_scale,
        nu,
        z_transform: z_opt.cloned(),
        penaltyinfo,
        d0,
        d1,
        d2,
        d0_eta_proj: d0_eta_all,
        d1_eta_proj: d1_eta_all,
        d2_eta_proj: d2_eta_all,
        op0_s_raw: op0_info.s_raw,
        op1_s_raw: op1_info.s_raw,
        op2_s_raw: op2_info.s_raw,
        op0_c: op0_info.c,
        op1_c: op1_info.c,
        op2_c: op2_info.c,
        op0_s_first_raw: op0_info.s_first_raw,
        op1_s_first_raw: op1_info.s_first_raw,
        op2_s_first_raw: op2_info.s_first_raw,
    });
    let cross_provider = AnisoPenaltyCrossProvider::new(move |a: usize, b: usize| {
        let (axis_a, axis_b) = if a < b { (a, b) } else { (b, a) };
        if axis_a == axis_b || axis_b >= cross_ctx.d0_eta_proj.len() {
            return Ok(Vec::new());
        }
        cross_ctx.compute_pair(axis_a, axis_b)
    });

    Ok((per_axis_results, cross_pairs, cross_provider))
}

/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic hybrid Duchon terms.
///
/// Analogous to [`build_matern_operator_penalty_aniso_derivatives`] but for
/// the Duchon kernel. Uses `duchon_radial_jets` for the full radial jet
/// `(φ, q, t, t_r, t_rr)`.
///
/// The local y-space operator shape derivatives start from the same formulas as
/// Matérn, but the raw per-axis `psi_a` coordinates also inherit the Duchon
/// isotropic scaling law. After assembling the shape-only pieces, this routine
/// adds the exact raw-`psi` isotropic-share correction implied by
/// `phi(r; kappa) = kappa^delta H(kappa r)`.
struct DuchonCrossPenaltyContext {
    centers: Array2<f64>,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    pure_block_order: usize,
    coeffs: Option<DuchonPartialFractionCoeffs>,
    aniso_log_scales: Vec<f64>,
    z_kernel: Array2<f64>,
    poly_cols: usize,
    identifiability_transform: Option<Array2<f64>>,
    penaltyinfo: Vec<PenaltyInfo>,
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    d0_eta_proj: Vec<Array2<f64>>,
    d1_eta_proj: Vec<Array2<f64>>,
    d2_eta_proj: Vec<Array2<f64>>,
    op0_s_raw: Array2<f64>,
    op1_s_raw: Array2<f64>,
    op2_s_raw: Array2<f64>,
    op0_c: f64,
    op1_c: f64,
    op2_c: f64,
    op0_s_first_raw: Vec<Array2<f64>>,
    op1_s_first_raw: Vec<Array2<f64>>,
    op2_s_first_raw: Vec<Array2<f64>>,
}

impl DuchonCrossPenaltyContext {
    fn project_operator(&self, mat: &Array2<f64>, row_dim: usize) -> Array2<f64> {
        let kernel_cols = mat.ncols();
        let total_cols = kernel_cols + self.poly_cols;
        let mut padded = Array2::<f64>::zeros((row_dim, total_cols));
        padded.slice_mut(s![.., 0..kernel_cols]).assign(mat);
        if let Some(z) = self.identifiability_transform.as_ref() {
            fast_ab(&padded, z)
        } else {
            padded
        }
    }

    fn compute_pair(&self, axis_a: usize, axis_b: usize) -> Result<Vec<Array2<f64>>, BasisError> {
        if axis_a >= self.aniso_log_scales.len() || axis_b >= self.aniso_log_scales.len() {
            return Err(BasisError::InvalidInput(format!(
                "Duchon cross-penalty pair out of bounds: ({axis_a}, {axis_b}) for dim={}",
                self.aniso_log_scales.len()
            )));
        }
        if axis_a == axis_b {
            return Err(BasisError::InvalidInput(format!(
                "Duchon cross-penalty pair must use distinct axes, got ({axis_a}, {axis_b})"
            )));
        }

        let p = self.centers.nrows();
        let d = self.centers.ncols();
        let z_cols = self.z_kernel.ncols();
        let metric_weights = centered_aniso_metric_weights(&self.aniso_log_scales);
        let sum_metric_weights: f64 = metric_weights.iter().sum();

        let mut d0_raw_eta_cross = Array2::<f64>::zeros((p, z_cols));
        let mut d1_raw_eta_cross = Array2::<f64>::zeros((p * d, z_cols));
        let mut d2_raw_eta_cross = Array2::<f64>::zeros((p, z_cols));

        for k in 0..p {
            for j in 0..p {
                let ci: Vec<f64> = (0..d).map(|a| self.centers[[k, a]]).collect();
                let cj: Vec<f64> = (0..d).map(|a| self.centers[[j, a]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, &self.aniso_log_scales);
                let (_, _q, t, dt_dr, d2t_dr2) = if let Some(length_scale) = self.length_scale {
                    let jets = duchon_radial_jets(
                        r,
                        length_scale,
                        self.p_order,
                        self.s_order,
                        d,
                        self.coeffs
                            .as_ref()
                            .expect("hybrid Duchon partial-fraction coefficients"),
                    )?;
                    (jets.phi, jets.q, jets.t, jets.t_r, jets.t_rr)
                } else {
                    let phi = polyharmonic_kernel(r, self.pure_block_order, d);
                    let (q, t, dt_dr, d2t_dr2) =
                        duchon_polyharmonic_operator_block_jets(r, self.pure_block_order, d)?;
                    (phi, q, t, dt_dr, d2t_dr2)
                };
                let sum_wb_sb: f64 = (0..d).map(|b| metric_weights[b] * s_vec[b]).sum();
                for col in 0..z_cols {
                    let z_jc = self.z_kernel[[j, col]];
                    let s_a = s_vec[axis_a];
                    let s_b = s_vec[axis_b];
                    let sa_sb = s_a * s_b;
                    let w_a = metric_weights[axis_a];
                    let w_b = metric_weights[axis_b];

                    d0_raw_eta_cross[[k, col]] += t * sa_sb * z_jc;

                    for axis in 0..d {
                        let h_l = ci[axis] - cj[axis];
                        let w_l = metric_weights[axis];
                        let row = k * d + axis;
                        let d1_cross = if r > 1e-14 {
                            let base = dt_dr * sa_sb / r;
                            if axis == axis_a {
                                (base + 2.0 * t * s_b) * w_l * h_l
                            } else if axis == axis_b {
                                (base + 2.0 * t * s_a) * w_l * h_l
                            } else {
                                base * w_l * h_l
                            }
                        } else {
                            0.0
                        };
                        d1_raw_eta_cross[[row, col]] += d1_cross * z_jc;
                    }

                    let d2_cross = if r > 1e-14 {
                        let r2 = r * r;
                        let dt_sa_r = dt_dr * s_a / r;
                        let dt_sb_r = dt_dr * s_b / r;
                        let d_dt_sa_r_b = d2t_dr2 * s_a * s_b / r2 - dt_dr * s_a * s_b / (r2 * r);
                        let term1 = d_dt_sa_r_b * sum_wb_sb + dt_sa_r * 4.0 * w_b * s_b;
                        let term2 = 4.0 * dt_sb_r * w_a * s_a;
                        let term3 = dt_sb_r * s_a * sum_metric_weights + t * s_a * 2.0 * w_b;
                        let term4 = 2.0 * t * s_b * w_a;
                        term1 + term2 + term3 + term4
                    } else {
                        0.0
                    };
                    d2_raw_eta_cross[[k, col]] += d2_cross * z_jc;
                }
            }
        }

        let value_share = duchon_scaling_exponent(self.p_order, self.s_order, d) / d as f64;
        let operator_share =
            duchon_operator_scaling_exponent(self.p_order, self.s_order, d) / d as f64;

        let mut d0_cross_proj = self.project_operator(&d0_raw_eta_cross, p);
        if value_share != 0.0 {
            d0_cross_proj +=
                &((&self.d0_eta_proj[axis_a] + &self.d0_eta_proj[axis_b]) * value_share);
            d0_cross_proj -= &(&self.d0 * (value_share * value_share));
        }
        let mut d1_cross_proj = self.project_operator(&d1_raw_eta_cross, p * d);
        if operator_share != 0.0 {
            d1_cross_proj +=
                &((&self.d1_eta_proj[axis_a] + &self.d1_eta_proj[axis_b]) * operator_share);
            d1_cross_proj -= &(&self.d1 * (operator_share * operator_share));
        }
        let mut d2_cross_proj = self.project_operator(&d2_raw_eta_cross, p);
        if operator_share != 0.0 {
            d2_cross_proj +=
                &((&self.d2_eta_proj[axis_a] + &self.d2_eta_proj[axis_b]) * operator_share);
            d2_cross_proj -= &(&self.d2 * (operator_share * operator_share));
        }

        let s0_cross = normalize_penalty_cross_psi_derivative(
            &self.op0_s_raw,
            &self.op0_s_first_raw[axis_a],
            &self.op0_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d0,
                &self.d0_eta_proj[axis_a],
                &self.d0_eta_proj[axis_b],
                &d0_cross_proj,
            ),
            self.op0_c,
        );
        let s1_cross = normalize_penalty_cross_psi_derivative(
            &self.op1_s_raw,
            &self.op1_s_first_raw[axis_a],
            &self.op1_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d1,
                &self.d1_eta_proj[axis_a],
                &self.d1_eta_proj[axis_b],
                &d1_cross_proj,
            ),
            self.op1_c,
        );
        let s2_cross = normalize_penalty_cross_psi_derivative(
            &self.op2_s_raw,
            &self.op2_s_first_raw[axis_a],
            &self.op2_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d2,
                &self.d2_eta_proj[axis_a],
                &self.d2_eta_proj[axis_b],
                &d2_cross_proj,
            ),
            self.op2_c,
        );

        active_operator_penalty_derivatives(
            &self.penaltyinfo,
            &[s0_cross, s1_cross, s2_cross],
            "Duchon-aniso-cross",
        )
    }
}

fn build_duchon_operator_penalty_aniso_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: &[f64],
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<
    (
        Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>,
        Vec<(usize, usize)>,
        AnisoPenaltyCrossProvider,
    ),
    BasisError,
> {
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    // Notation and conventions:
    //   ψ_b = aniso_log_scales[b] (log-scale parameter for axis b)
    //   w_b = exp(2ψ_b)  (metric weight for axis b)
    //   h_b = x_b - c_b  (coordinate displacement)
    //   s_b = w_b · h_b²  (UNNORMALIZED axis component; note Σ_b s_b = r²)
    //   q = φ_r / r       (radial jet scalar)
    //   t = (φ_rr - q) / r² = q_r / r  (second radial jet scalar)
    //
    // The operators D₀, D₁, D₂ are:
    //   D₀: φ(r)                              (magnitude/mass)
    //   D₁_b: ∂φ/∂x_b = q · w_b · h_b        (gradient component b)
    //   D₂: Δ_x φ = Σ_b w_b·(t·s_b + q)      (anisotropic Laplacian)
    //
    // Key ψ-derivatives (using unnormalized s_b):
    //   ∂r/∂ψ_a = s_a / r,  ∂q/∂ψ_a = t·s_a,  ∂s_b/∂ψ_a = 2·δ_{ab}·s_b,
    //   ∂w_b/∂ψ_a = 2·δ_{ab}·w_b
    let p = centers.nrows();
    let d = centers.ncols();
    let dim = aniso_log_scales.len();
    assert_eq!(dim, d);

    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    validate_duchon_collocation_orders(length_scale, p_order, s_order, d)?;
    let coeffs = length_scale
        .map(|scale| duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / scale.max(1e-300)));
    let pure_block_order = pure_duchon_block_order(p_order, s_order);

    let z_kernel = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    let z_cols = z_kernel.ncols();

    // Raw operator matrices and their per-axis derivatives (in kernel-Z space).
    let mut d0_raw = Array2::<f64>::zeros((p, z_cols));
    let mut d1_raw = Array2::<f64>::zeros((p * d, z_cols));
    let mut d2_raw = Array2::<f64>::zeros((p, z_cols));
    let mut d0_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, z_cols))).collect();
    let mut d1_raw_eta: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d, z_cols))).collect();
    let mut d2_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, z_cols))).collect();
    let mut d0_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, z_cols))).collect();
    let mut d1_raw_eta2: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d, z_cols))).collect();
    let mut d2_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, z_cols))).collect();
    // Cross-derivative operator matrices for pairs (a, b) with a < b.
    let mut cross_pairs: Vec<(usize, usize)> = Vec::with_capacity(dim * (dim - 1) / 2);
    for a_idx in 0..dim {
        for b_idx in (a_idx + 1)..dim {
            cross_pairs.push((a_idx, b_idx));
        }
    }

    // Precompute metric weights w_b = exp(2ψ_b) for each axis.
    // These are needed for the correct anisotropic gradient operator D₁
    // and anisotropic Laplacian operator D₂.
    let metric_weights = centered_aniso_metric_weights(aniso_log_scales);
    let sum_metric_weights: f64 = metric_weights.iter().sum();

    for k in 0..p {
        for j in 0..p {
            let ci: Vec<f64> = (0..d).map(|a| centers[[k, a]]).collect();
            let cj: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
            let (r, s_vec) = aniso_distance_and_components(&ci, &cj, aniso_log_scales);

            // Get the Duchon radial jets, including the exact off-origin
            // derivatives of t = R²φ needed for D₁/D₂ anisotropic penalties.
            let (phi, q, t, dt_dr, d2t_dr2) = if let Some(length_scale) = length_scale {
                let jets = duchon_radial_jets(
                    r,
                    length_scale,
                    p_order,
                    s_order,
                    d,
                    coeffs
                        .as_ref()
                        .expect("hybrid Duchon partial-fraction coefficients"),
                )?;
                (jets.phi, jets.q, jets.t, jets.t_r, jets.t_rr)
            } else {
                let phi = polyharmonic_kernel(r, pure_block_order, d);
                let (q, t, dt_dr, d2t_dr2) =
                    duchon_polyharmonic_operator_block_jets(r, pure_block_order, d)?;
                (phi, q, t, dt_dr, d2t_dr2)
            };

            // Anisotropic Laplacian:
            //   Δ_x φ = Σ_b ∂²φ/∂x_b²
            //          = Σ_b w_b · (t · s_b + q)
            //          = t · Σ_b(w_b · s_b) + q · Σ_b(w_b)
            // where w_b = exp(2ψ_b), s_b = w_b · h_b² (unnormalized), h_b = x_b - c_b.
            //
            // Derivation: ∂φ/∂x_b = q · w_b · h_b,
            //   ∂²φ/∂x_b² = (∂q/∂x_b)·w_b·h_b + q·w_b
            //              = t·w_b²·h_b² + q·w_b = w_b·(t·s_b + q).
            //
            // Note: the old isotropic formula d·q + t·r² is recovered when all w_b = 1.
            let sum_wb_sb: f64 = (0..d).map(|b| metric_weights[b] * s_vec[b]).sum();
            let lap = if r < 1e-14 {
                // At collision: s_b = 0 for all b, so t·s_b terms vanish.
                // Δφ(0) = q(0) · Σ_b w_b = φ''(0) · Σ_b exp(2ψ_b)
                q * sum_metric_weights
            } else {
                t * sum_wb_sb + q * sum_metric_weights
            };

            // Fill raw operator matrices (projected through Z_kernel).
            for col in 0..z_cols {
                let z_jc = z_kernel[[j, col]];
                d0_raw[[k, col]] += phi * z_jc;

                // D₁: anisotropic gradient components.
                //   ∂φ/∂x_b = φ_r · ∂r/∂x_b = (φ_r/r) · w_b · h_b = q · w_b · h_b
                // where w_b = exp(2ψ_b) is the metric weight for axis b.
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let w_b = metric_weights[b];
                    d1_raw[[k * d + b, col]] += q * w_b * h_b * z_jc;
                }

                d2_raw[[k, col]] += lap * z_jc;

                // Per-axis η_a derivatives.
                for a in 0..dim {
                    let s_a = s_vec[a];
                    let w_a = metric_weights[a];

                    if r <= 1e-14 {
                        // At a center collision all h_b and s_b are zero.
                        // The D0/D1 anisotropy derivatives vanish directly.
                        // The only surviving D2 terms come from differentiating
                        // the metric trace in Δφ(0)=q(0)Σ_b w_b.
                        d2_raw_eta[a][[k, col]] += 2.0 * q * w_a * z_jc;
                        d2_raw_eta2[a][[k, col]] += 4.0 * q * w_a * z_jc;
                        continue;
                    }

                    // D₀ derivatives (unchanged — chain rule through r only).
                    d0_raw_eta[a][[k, col]] += q * s_a * z_jc;
                    d0_raw_eta2[a][[k, col]] += (t * s_a * s_a + 2.0 * q * s_a) * z_jc;

                    // D₁ derivatives.
                    // Base D₁_b = q · w_b · h_b.
                    //
                    // First derivative:
                    //   ∂D₁_b/∂ψ_a = (∂q/∂ψ_a) · w_b · h_b + q · (∂w_b/∂ψ_a) · h_b
                    //   ∂q/∂ψ_a = q_r · (s_a/r) = t · s_a  (since t = q_r/r)
                    //   ∂w_b/∂ψ_a = 2 · δ_{ab} · w_b
                    //
                    //   = w_b · h_b · (t · s_a + 2 · δ_{ab} · q)
                    //
                    // Second derivative (diagonal):
                    //   ∂²D₁_b/∂ψ_a² = ∂/∂ψ_a [w_b · h_b · (t·s_a + 2·δ_{ab}·q)]
                    //   For a != b: w_b · h_b · (dt_dr·s_a²/r + 2·t·s_a)
                    //   For a == b: w_a · h_a · (dt_dr·s_a²/r + 6·t·s_a + 4·q)
                    for b in 0..d {
                        let h_b = ci[b] - cj[b];
                        let w_b = metric_weights[b];
                        let row = k * d + b;

                        let d1_first = if a == b {
                            w_b * h_b * (t * s_a + 2.0 * q)
                        } else {
                            w_b * h_b * t * s_a
                        };
                        d1_raw_eta[a][[row, col]] += d1_first * z_jc;

                        let d1_eta2_val = if a == b {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 6.0 * t * s_a + 4.0 * q)
                        } else {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 2.0 * t * s_a)
                        };
                        d1_raw_eta2[a][[row, col]] += d1_eta2_val * z_jc;
                    }

                    // D₂ first derivative (anisotropic Laplacian).
                    //   Δ = t · W₂ + q · W₁  where W₁ = Σw_b, W₂ = Σ(w_b·s_b).
                    //   ∂Δ/∂ψ_a = (∂t/∂ψ_a)·W₂ + t·(∂W₂/∂ψ_a) + (∂q/∂ψ_a)·W₁ + q·(∂W₁/∂ψ_a)
                    //
                    //   ∂t/∂ψ_a = dt_dr · s_a / r
                    //   ∂W₂/∂ψ_a = ∂/∂ψ_a[Σ w_b·s_b] = 4·w_a·s_a
                    //     (∂(w_a·s_a)/∂ψ_a = 2·w_a·s_a + w_a·2·s_a = 4·w_a·s_a)
                    //   ∂q/∂ψ_a = t · s_a
                    //   ∂W₁/∂ψ_a = 2 · w_a
                    let dlap_deta = dt_dr * s_a / r * sum_wb_sb
                        + 4.0 * t * w_a * s_a
                        + t * s_a * sum_metric_weights
                        + 2.0 * q * w_a;
                    d2_raw_eta[a][[k, col]] += dlap_deta * z_jc;

                    // D₂ second derivative (anisotropic Laplacian).
                    // ∂²Δ/∂ψ_a² = ∂/∂ψ_a of the first derivative above.
                    //
                    // Using shorthand T1..T4 for the four terms of dlap_deta:
                    //   T1 = dt_dr·s_a/r · W₂
                    //   T2 = 4·t·w_a·s_a
                    //   T3 = t·s_a·W₁
                    //   T4 = 2·q·w_a
                    let s_a2 = s_a * s_a;
                    let dt_sa_r = dt_dr * s_a / r;

                    // ∂/∂ψ_a[dt_dr·s_a/r]:
                    //   ∂t_r/∂ψ_a = d2t_dr2·s_a/r,  ∂(s_a/r)/∂ψ_a = s_a/r·(2 - s_a/r²)
                    //   product rule => d2t_dr2·s_a²/r² + dt_dr·s_a/r·(2 - s_a/r²)
                    let r2 = r * r;
                    let d_dt_sa_r = d2t_dr2 * s_a2 / r2 + dt_dr * s_a / r * (2.0 - s_a / r2);

                    // ∂T1/∂ψ_a = d_dt_sa_r · W₂ + dt_sa_r · 4·w_a·s_a
                    let dt1 = d_dt_sa_r * sum_wb_sb + dt_sa_r * 4.0 * w_a * s_a;

                    // ∂T2/∂ψ_a = 4·[(dt_dr·s_a/r)·w_a·s_a + t·(2w_a·s_a + w_a·2s_a)]
                    //           = 4·w_a·s_a·(dt_sa_r + 4·t)
                    let dt2 = 4.0 * w_a * s_a * (dt_sa_r + 4.0 * t);

                    // ∂T3/∂ψ_a = (dt_dr·s_a/r)·s_a·W₁ + t·2·s_a·W₁ + t·s_a·2·w_a
                    let dt3 = dt_sa_r * s_a * sum_metric_weights
                        + 2.0 * t * s_a * sum_metric_weights
                        + 2.0 * t * s_a * w_a;

                    // ∂T4/∂ψ_a = 2·(t·s_a·w_a + q·2·w_a) = 2·w_a·(t·s_a + 2·q)
                    let dt4 = 2.0 * w_a * (t * s_a + 2.0 * q);

                    let d2lap_deta2 = dt1 + dt2 + dt3 + dt4;
                    d2_raw_eta2[a][[k, col]] += d2lap_deta2 * z_jc;
                }
            }
        }
    }

    let apply_raw_psi_scaling = |base: &Array2<f64>,
                                 first: &mut Vec<Array2<f64>>,
                                 second: &mut Vec<Array2<f64>>,
                                 coeff: f64| {
        if coeff == 0.0 {
            return;
        }
        let first_shape = first.clone();
        for a in 0..dim {
            first[a] += &(base * coeff);
            second[a] += &(&first_shape[a] * (2.0 * coeff));
            second[a] += &(base * (coeff * coeff));
        }
    };

    let value_share = duchon_scaling_exponent(p_order, s_order, d) / d as f64;
    let operator_share = duchon_operator_scaling_exponent(p_order, s_order, d) / d as f64;
    apply_raw_psi_scaling(&d0_raw, &mut d0_raw_eta, &mut d0_raw_eta2, value_share);
    apply_raw_psi_scaling(&d1_raw, &mut d1_raw_eta, &mut d1_raw_eta2, operator_share);
    apply_raw_psi_scaling(&d2_raw, &mut d2_raw_eta, &mut d2_raw_eta2, operator_share);

    let poly_cols = polynomial_block_from_order(centers, nullspace_order).ncols();

    let project_operator = |mat: &Array2<f64>, row_dim: usize| -> Array2<f64> {
        let kernel_cols = mat.ncols();
        let total_cols = kernel_cols + poly_cols;
        let mut padded = Array2::<f64>::zeros((row_dim, total_cols));
        padded.slice_mut(s![.., 0..kernel_cols]).assign(mat);
        if let Some(z) = identifiability_transform {
            fast_ab(&padded, z)
        } else {
            padded
        }
    };

    let d0 = project_operator(&d0_raw, p);
    let d1 = project_operator(&d1_raw, p * d);
    let d2 = project_operator(&d2_raw, p);
    let d0_eta_proj: Vec<Array2<f64>> = d0_raw_eta.iter().map(|m| project_operator(m, p)).collect();
    let d1_eta_proj: Vec<Array2<f64>> = d1_raw_eta
        .iter()
        .map(|m| project_operator(m, p * d))
        .collect();
    let d2_eta_proj: Vec<Array2<f64>> = d2_raw_eta.iter().map(|m| project_operator(m, p)).collect();
    let d0_eta2_proj: Vec<Array2<f64>> =
        d0_raw_eta2.iter().map(|m| project_operator(m, p)).collect();
    let d1_eta2_proj: Vec<Array2<f64>> = d1_raw_eta2
        .iter()
        .map(|m| project_operator(m, p * d))
        .collect();
    let d2_eta2_proj: Vec<Array2<f64>> =
        d2_raw_eta2.iter().map(|m| project_operator(m, p)).collect();
    // Build raw Gram penalties (axis-independent) and per-axis derivatives + norms,
    // using the same PerOperatorInfo pattern as the Matérn case.
    struct PerOperatorInfo {
        s_raw: Array2<f64>,
        c: f64,
        s_first: Vec<Array2<f64>>,
        s_second: Vec<Array2<f64>>,
        s_first_raw: Vec<Array2<f64>>,
    }

    let compute_operator_info =
        |d_op: &Array2<f64>, d_eta: &[Array2<f64>], d_eta2: &[Array2<f64>]| -> PerOperatorInfo {
            let s_raw = symmetrize(&fast_ata(d_op));
            let fro2: f64 = s_raw.iter().map(|v| v * v).sum();
            let c = fro2.sqrt();

            let mut s_first = Vec::with_capacity(dim);
            let mut s_second = Vec::with_capacity(dim);
            let mut s_first_raw = Vec::with_capacity(dim);
            for a in 0..dim {
                let (_, sa, sa2) =
                    gram_and_psi_derivatives_from_operator(d_op, &d_eta[a], &d_eta2[a]);
                let (_, sa_norm, sa2_norm, _) =
                    normalize_penaltywith_psi_derivatives(&s_raw, &sa, &sa2);
                s_first_raw.push(sa);
                s_first.push(sa_norm);
                s_second.push(sa2_norm);
            }

            PerOperatorInfo {
                s_raw,
                c,
                s_first,
                s_second,
                s_first_raw,
            }
        };

    let op0_info = compute_operator_info(&d0, &d0_eta_proj, &d0_eta2_proj);
    let op1_info = compute_operator_info(&d1, &d1_eta_proj, &d1_eta2_proj);
    let op2_info = compute_operator_info(&d2, &d2_eta_proj, &d2_eta2_proj);

    // Build penalty candidates and determine which are active.
    let (s0_norm, c0) = if op0_info.c > 1e-12 {
        (op0_info.s_raw.mapv(|v| v / op0_info.c), op0_info.c)
    } else {
        (op0_info.s_raw.clone(), 1.0)
    };
    let (s1_norm, c1) = if op1_info.c > 1e-12 {
        (op1_info.s_raw.mapv(|v| v / op1_info.c), op1_info.c)
    } else {
        (op1_info.s_raw.clone(), 1.0)
    };
    let (s2_norm, c2) = if op2_info.c > 1e-12 {
        (op2_info.s_raw.mapv(|v| v / op2_info.c), op2_info.c)
    } else {
        (op2_info.s_raw.clone(), 1.0)
    };

    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
        },
    ];
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;

    // Build per-axis results.
    let mut per_axis_results = Vec::with_capacity(dim);
    for a in 0..dim {
        let pen_first = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_first[a].clone(),
                op1_info.s_first[a].clone(),
                op2_info.s_first[a].clone(),
            ],
            "Duchon-aniso",
        )?;
        let pen_second = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_second[a].clone(),
                op1_info.s_second[a].clone(),
                op2_info.s_second[a].clone(),
            ],
            "Duchon-aniso",
        )?;
        per_axis_results.push((pen_first, pen_second));
    }

    let cross_pairs_for_provider = cross_pairs.clone();
    let cross_ctx = DuchonCrossPenaltyContext {
        centers: centers.to_owned(),
        length_scale,
        p_order,
        s_order,
        pure_block_order,
        coeffs,
        aniso_log_scales: aniso_log_scales.to_vec(),
        z_kernel,
        poly_cols,
        identifiability_transform: identifiability_transform.cloned(),
        penaltyinfo,
        d0,
        d1,
        d2,
        d0_eta_proj,
        d1_eta_proj,
        d2_eta_proj,
        op0_s_raw: op0_info.s_raw,
        op1_s_raw: op1_info.s_raw,
        op2_s_raw: op2_info.s_raw,
        op0_c: op0_info.c,
        op1_c: op1_info.c,
        op2_c: op2_info.c,
        op0_s_first_raw: op0_info.s_first_raw,
        op1_s_first_raw: op1_info.s_first_raw,
        op2_s_first_raw: op2_info.s_first_raw,
    };
    let cross_ctx = std::sync::Arc::new(cross_ctx);
    let cross_provider = AnisoPenaltyCrossProvider::new(move |a: usize, b: usize| {
        let (axis_a, axis_b) = if a < b { (a, b) } else { (b, a) };
        if !cross_pairs_for_provider.contains(&(axis_a, axis_b)) {
            return Ok(Vec::new());
        }
        cross_ctx.compute_pair(axis_a, axis_b)
    });

    Ok((per_axis_results, cross_pairs, cross_provider))
}

fn duchon_kernel_radial_triplet(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<(f64, f64, f64), BasisError> {
    // Public Duchon (phi, phi_r, phi_rr) triplet.
    //
    // Pure Duchon keeps its direct polyharmonic path. Hybrid Duchon now
    // delegates to `duchon_radial_jets(...)` so every public radial derivative
    // shares the exact same differentiable family as q/lap/t and the operator
    // penalty code paths.
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
        let (_, mut first, second) = polyharmonic_kernel_triplet(r, block_order, k_dim)?;
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
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs_ref)?;
    let first = jets.phi_r;
    let second = jets.phi_rr;
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

fn operator_penalty_candidates_from_collocation(
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
) -> Vec<PenaltyCandidate> {
    let (s0, c0) = normalize_penalty(&symmetrize(&fast_ata(d0)));
    let (s1, c1) = normalize_penalty(&symmetrize(&fast_ata(d1)));
    let (s2, c2) = normalize_penalty(&symmetrize(&fast_ata(d2)));
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
        });
    }
    out
}

fn active_operator_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    operator_derivatives: &[Array2<f64>],
    label: &str,
) -> Result<Vec<Array2<f64>>, BasisError> {
    if operator_derivatives.len() != 3 {
        return Err(BasisError::InvalidInput(format!(
            "{label} operator derivative path requires exactly 3 canonical penalties; found {}",
            operator_derivatives.len()
        )));
    }

    penaltyinfo
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
    expectedrows: usize,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None | SpatialIdentifiability::OrthogonalToParametric => Ok(None),
        SpatialIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != expectedrows {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen {label} identifiability transform mismatch: rows={}, expected {expectedrows}",
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

fn build_thin_plate_penalty_matrices(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    kernel_transform: &Array2<f64>,
    double_penalty: bool,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let k = centers.nrows();
    let d = centers.ncols();
    let kernel_cols = kernel_transform.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(d);
    let total_cols = kernel_cols + poly_cols;
    let mut omega = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[i, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let kij = thin_plate_kernel_from_dist2(dist2 / (length_scale * length_scale), d)?;
            omega[[i, j]] = kij;
            omega[[j, i]] = kij;
        }
    }
    let omega_constrained = {
        let zt_o = fast_atb(kernel_transform, &omega);
        // `kernel_transform` spans the side-constraint nullspace, so the
        // congruence transform preserves the thin-plate PSD construction.
        // Symmetrize to remove roundoff asymmetry without paying for a full EVD
        // on the large lazy-path penalty.
        symmetrize_penalty(&fast_ab(&zt_o, kernel_transform))
    };
    let mut penalty_bending = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_bending
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_constrained);
    let penalty_ridge = if double_penalty {
        build_nullspace_shrinkage_penalty(&penalty_bending)?.map(|block| block.sym_penalty)
    } else {
        None
    };
    Ok((penalty_bending, penalty_ridge))
}

fn build_matern_kernel_penalty(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let total_cols = k + usize::from(include_intercept);
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let r = if let Some(eta) = aniso_log_scales {
                aniso_distance(
                    centers.row(i).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
            } else {
                let mut dist2 = 0.0;
                for axis in 0..centers.ncols() {
                    let delta = centers[[i, axis]] - centers[[j, axis]];
                    dist2 += delta * delta;
                }
                dist2.sqrt()
            };
            let kij = matern_kernel_from_distance(r, length_scale, nu)?;
            center_kernel[[i, j]] = kij;
            center_kernel[[j, i]] = kij;
        }
    }
    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    Ok(penalty_kernel)
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
            let (_, z) = applyweighted_orthogonality_constraint(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}

fn spatial_identifiability_transform_from_design_matrix(
    data: ArrayView2<'_, f64>,
    design: &DesignMatrix,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let z = orthogonality_transform_for_design(design, c.view(), None)?;
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

fn project_penalty_matrix(matrix: &Array2<f64>, transform: Option<&Array2<f64>>) -> Array2<f64> {
    let projected = if let Some(z) = transform {
        let zt_s = z.t().dot(matrix);
        zt_s.dot(z)
    } else {
        matrix.clone()
    };
    symmetrize(&projected)
}

fn normalize_penalty_candidate(
    matrix: Array2<f64>,
    nullspace_dim_hint: usize,
    source: PenaltySource,
) -> PenaltyCandidate {
    let (matrix, normalization_scale) = if matrix.iter().all(|v| v.abs() <= 1e-12) {
        (matrix, 1.0)
    } else {
        normalize_penalty(&matrix)
    };
    PenaltyCandidate {
        matrix,
        nullspace_dim_hint,
        source,
        normalization_scale,
        kronecker_factors: None,
    }
}

pub fn build_matern_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // Specialized Matérn operator assembly using explicit half-integer formulas:
    // - one exp(-a) and small polynomials per pair,
    // - NaN-safe phi'(r)/r without dividing by r for nu>=3/2,
    // - exact Hessian rows for the stiffness operator, not just the Laplacian.
    let p = centers.nrows();
    let d = centers.ncols();
    let row_scales = if let Some(w) = collocationweights {
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
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let metric_weights = aniso_log_scales.map(centered_aniso_metric_weights);
    const R_EPS: f64 = 1e-12;
    for k in 0..p {
        let scale_k = row_scales[k];
        for j in 0..p {
            // Distance: anisotropic r = |Ah| when eta present, isotropic |h| otherwise.
            let r = if let Some(eta) = aniso_log_scales {
                aniso_distance_and_components(
                    centers.row(k).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
                .0
            } else {
                stable_euclidean_norm((0..d).map(|c| centers[[k, c]] - centers[[j, c]]))
            };
            if matches!(nu, MaternNu::Half) && r <= R_EPS && d > 1 {
                return Err(BasisError::InvalidInput(
                    "Matérn nu=1/2 has singular Laplacian at center collisions for d>1; choose nu>=3/2 or avoid collocation at centers".to_string(),
                ));
            }
            let (phi, _, phi_rr, phi_r_over_r) =
                if matches!(nu, MaternNu::Half) && r <= R_EPS && d == 1 {
                    // In 1D: Delta phi = phi'' and the singular phi'/r term is absent.
                    let s = 1.0 / length_scale;
                    let e = 1.0;
                    (e, -s * e, s * s * e, 0.0)
                } else {
                    matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?
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
            let t = if r > R_EPS {
                (phi_rr - phi_r_over_r) / (r * r)
            } else {
                0.0
            };
            for a in 0..d {
                let h_a = centers[[k, a]] - centers[[j, a]];
                let w_a = metric_weights.as_ref().map(|w| w[a]).unwrap_or(1.0);
                for b in 0..d {
                    let h_b = centers[[k, b]] - centers[[j, b]];
                    let w_b = metric_weights.as_ref().map(|w| w[b]).unwrap_or(1.0);
                    let diagonal = if a == b { phi_r_over_r * w_a } else { 0.0 };
                    let mixed = if r > R_EPS {
                        t * w_a * h_a * w_b * h_b
                    } else {
                        0.0
                    };
                    let row = (k * d + a) * d + b;
                    d2_raw[[row, j]] = scale_k * (diagonal + mixed);
                }
            }
            if !d0_raw[[k, j]].is_finite()
                || ((k * d * d)..((k + 1) * d * d)).any(|row| !d2_raw[[row, j]].is_finite())
            {
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
    let mut d2 = Array2::<f64>::zeros((p_colloc * dim * dim, total_cols));
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
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_collocation_operator_matriceswithworkspace(
        centers,
        collocationweights,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        identifiability_transform,
        &mut workspace,
    )
}

pub fn build_duchon_collocation_operator_matriceswithworkspace(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    let p_colloc = centers.nrows();
    let dim = centers.ncols();
    validate_duchon_collocation_orders(length_scale, p_order, s_order, dim)?;
    if let Some(eta) = aniso_log_scales {
        if eta.len() != dim {
            return Err(BasisError::DimensionMismatch(format!(
                "Duchon anisotropy dimension mismatch: got {}, expected {dim}",
                eta.len()
            )));
        }
    }
    let coeffs = length_scale
        .map(|scale| duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / scale.max(1e-300)));
    let metric_weights: Option<Vec<f64>> = aniso_log_scales.map(centered_aniso_metric_weights);
    let row_scales = if let Some(w) = collocationweights {
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
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    let mut d0_raw = Array2::<f64>::zeros((p_colloc, p_colloc));
    let mut d1_raw = Array2::<f64>::zeros((p_colloc * dim, p_colloc));
    let mut d2_raw = Array2::<f64>::zeros((p_colloc * dim * dim, p_colloc));
    const R_EPS: f64 = 1e-10;
    for k in 0..p_colloc {
        let scale_k = row_scales[k];
        for j in k..p_colloc {
            let scale_j = row_scales[j];
            let r = if let Some(eta) = aniso_log_scales {
                let row_k: Vec<f64> = (0..dim).map(|a| centers[[k, a]]).collect();
                let row_j: Vec<f64> = (0..dim).map(|a| centers[[j, a]]).collect();
                aniso_distance(&row_k, &row_j, eta)
            } else {
                stable_euclidean_norm((0..dim).map(|axis| centers[[k, axis]] - centers[[j, axis]]))
            };
            let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
                r,
                length_scale,
                p_order,
                s_order,
                dim,
                coeffs.as_ref(),
            )?;
            if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() {
                return Err(BasisError::InvalidInput(format!(
                    "non-finite Duchon collocation operator derivative at rows ({k}, {j}), r={r}"
                )));
            }
            d0_raw[[k, j]] = scale_k * phi;
            d0_raw[[j, k]] = scale_j * phi;
            let (q, t) = if r > R_EPS {
                let q = phi_r / r;
                (q, (phi_rr - q) / (r * r))
            } else {
                (phi_rr, 0.0)
            };
            for axis_a in 0..dim {
                let h_a = centers[[k, axis_a]] - centers[[j, axis_a]];
                let w_a = metric_weights
                    .as_ref()
                    .map(|weights| weights[axis_a])
                    .unwrap_or(1.0);
                for axis_b in 0..dim {
                    let h_b = centers[[k, axis_b]] - centers[[j, axis_b]];
                    let w_b = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis_b])
                        .unwrap_or(1.0);
                    let diagonal = if axis_a == axis_b { q * w_a } else { 0.0 };
                    let mixed = if r > R_EPS {
                        t * w_a * h_a * w_b * h_b
                    } else {
                        0.0
                    };
                    let value = diagonal + mixed;
                    let row_k = (k * dim + axis_a) * dim + axis_b;
                    let row_j = (j * dim + axis_a) * dim + axis_b;
                    d2_raw[[row_k, j]] = scale_k * value;
                    d2_raw[[row_j, k]] = scale_j * value;
                }
            }
            if r > R_EPS {
                let grad_scale = phi_r / r;
                for axis in 0..dim {
                    let delta = centers[[k, axis]] - centers[[j, axis]];
                    let axis_scale = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis])
                        .unwrap_or(1.0);
                    d1_raw[[k * dim + axis, j]] = scale_k * grad_scale * axis_scale * delta;
                    d1_raw[[j * dim + axis, k]] = -scale_j * grad_scale * axis_scale * delta;
                }
            }
        }
    }
    let d0_kernel = fast_ab(&d0_raw, &z);
    let d1_kernel = fast_ab(&d1_raw, &z);
    let d2_kernel = fast_ab(&d2_raw, &z);
    let poly = polynomial_block_from_order(centers, nullspace_order);
    let kernel_cols = d0_kernel.ncols();
    let poly_cols = poly.ncols();
    let total_cols = kernel_cols + poly_cols;
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
    let mut d2 = Array2::<f64>::zeros((p_colloc * dim * dim, total_cols));
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    if poly_cols > 0 {
        let mut poly_hessian = polynomial_hessian_operator_block(centers, nullspace_order);
        for (k, &scale_k) in row_scales.iter().enumerate() {
            for local in 0..(dim * dim) {
                poly_hessian
                    .row_mut(k * dim * dim + local)
                    .mapv_inplace(|v| scale_k * v);
            }
        }
        d2.slice_mut(s![.., kernel_cols..]).assign(&poly_hessian);
    }
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
fn bessel_k0_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    if x_pos <= 2.0 {
        return bessel_k0_small_series(x_pos);
    }
    let y = 2.0 / x_pos;
    (-x_pos).exp() / x_pos.sqrt()
        * (1.253_314_14
            + y * (-0.078_323_58
                + y * (0.021_895_68
                    + y * (-0.010_624_46
                        + y * (0.005_878_72 + y * (-0.002_515_40 + y * 0.000_532_08))))))
}

#[inline(always)]
fn bessel_k1_stable(x: f64) -> f64 {
    let x_pos = x.max(1e-300);
    if x_pos <= 2.0 {
        return bessel_k1_small_series(x_pos);
    }
    let y = 2.0 / x_pos;
    (-x_pos).exp() / x_pos.sqrt()
        * (1.253_314_14
            + y * (0.234_986_19
                + y * (-0.036_556_20
                    + y * (0.015_042_68
                        + y * (-0.007_803_53 + y * (0.003_256_14 + y * -0.000_682_45))))))
}

#[inline(always)]
fn bessel_k0_k1_small_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let y = 0.25 * x * x;
    let log_half_plus_gamma = 0.5 * y.ln() + EULER_GAMMA;
    let mut i0 = 1.0;
    let mut i1 = 0.5 * x;
    let mut harmonic = 0.0;
    let mut y_power_over_fact_sq = 1.0;
    let mut k0_series = 0.0;
    let mut k0_series_y_derivative_times_y = 0.0;
    for k in 1..=256 {
        let kf = k as f64;
        harmonic += 1.0 / kf;
        y_power_over_fact_sq *= y / (kf * kf);
        let k0_term = harmonic * y_power_over_fact_sq;
        k0_series += k0_term;
        k0_series_y_derivative_times_y += kf * k0_term;
        i0 += y_power_over_fact_sq;
        i1 += 0.5 * x * y_power_over_fact_sq / (kf + 1.0);
        if k0_term.abs() <= f64::EPSILON * i0.abs().max(k0_series.abs()).max(1.0) {
            break;
        }
    }

    let k0 = -log_half_plus_gamma * i0 + k0_series;
    let k1 = i0 / x + log_half_plus_gamma * i1 - (2.0 / x) * k0_series_y_derivative_times_y;
    (k0, k1)
}

#[inline(always)]
fn bessel_k0_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).0
}

#[inline(always)]
fn bessel_k1_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).1
}

const DUCHON_DERIVATIVE_R_FLOOR_REL: f64 = 1e-5;
const DUCHON_COLLISION_TAYLOR_REL: f64 = 1e-4;

#[inline(always)]
fn duchon_p_from_nullspace_order(order: DuchonNullspaceOrder) -> usize {
    match order {
        // Duchon null spaces contain all polynomials of degree < m.
        // The public `order` knob chooses that polynomial degree cutoff:
        //   order=0 -> constants only  -> m=1
        //   order=1 -> constants+linear -> m=2
        DuchonNullspaceOrder::Zero => 1,
        DuchonNullspaceOrder::Linear => 2,
        DuchonNullspaceOrder::Degree(degree) => degree + 1,
    }
}

/// Returns the effective Duchon null-space order, auto-degrading to `Zero`
/// when the requested order cannot be spanned by the supplied centers.
///
/// When `order=Linear` and `centers.nrows() < d + 1`, the polynomial block
/// `[1, x_1, ..., x_d]` cannot be affinely spanned; rather than hard-erroring
/// the caller falls back to `order=Zero` (constant null space) and logs a
/// single warning so the user sees the degradation.
fn duchon_effective_nullspace_order(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> DuchonNullspaceOrder {
    if order == DuchonNullspaceOrder::Zero {
        return order;
    }
    let required = polynomial_block_from_order(centers, order).ncols();
    if centers.nrows() < required {
        // Dedup: warn only once per (rows, cols, requested_order) per process.
        // BFGS × P-IRLS × derivative callsites hit this path many times.
        static SEEN: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<(usize, usize, DuchonNullspaceOrder)>>,
        > = std::sync::OnceLock::new();
        let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let key = (centers.nrows(), centers.ncols(), order);
        let fresh = seen.lock().map(|mut s| s.insert(key)).unwrap_or(true);
        if fresh {
            log::warn!(
                "Duchon nullspace order={:?} needs >={} centers in dim={}; got {} — degrading to Zero",
                order,
                required,
                centers.ncols(),
                centers.nrows()
            );
        }
        return DuchonNullspaceOrder::Zero;
    }
    order
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

/// Precomputed coefficient for `polyharmonic_kernel` that depends only on
/// `m` and `k_dim`, not on `r`.  Avoids repeated gamma_lanczos calls in the
/// hot kernel evaluation loop (called n × k times per basis build).
#[derive(Clone, Copy)]
struct PolyharmonicBlockCoeff {
    c: f64,
    power: f64,
    is_log_case: bool,
}

impl PolyharmonicBlockCoeff {
    fn new(m: usize, k_dim: usize) -> Self {
        let k_half = 0.5 * k_dim as f64;
        let power = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
        if k_dim % 2 == 0 && m >= (k_dim / 2) {
            let c = polyharmonic_log_sign(m, k_dim)
                / (2.0_f64.powi((2 * m - 1) as i32)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m as f64)
                    * gamma_lanczos((m - k_dim / 2 + 1) as f64));
            Self {
                c,
                power,
                is_log_case: true,
            }
        } else {
            let c = gamma_lanczos(k_half - m as f64)
                / (4.0_f64.powi(m as i32)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m as f64));
            Self {
                c,
                power,
                is_log_case: false,
            }
        }
    }

    #[inline(always)]
    fn eval(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return self.origin_limit();
        }
        if self.is_log_case {
            self.c * r.powf(self.power) * r.max(1e-300).ln()
        } else {
            self.c * r.powf(self.power)
        }
    }

    #[inline(always)]
    fn origin_limit(&self) -> f64 {
        if self.is_log_case {
            log_power_origin_limit(self.c, self.power, 1.0, 0.0)
        } else {
            log_power_origin_limit(self.c, self.power, 0.0, 1.0)
        }
    }
}

fn polyharmonic_kernel(r: f64, m: usize, k_dim: usize) -> f64 {
    PolyharmonicBlockCoeff::new(m, k_dim).eval(r)
}

#[inline(always)]
fn signed_infinity(sign: f64) -> f64 {
    if sign.is_sign_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

#[inline(always)]
fn log_power_origin_limit(coeff: f64, exponent: f64, log_coeff: f64, pure_coeff: f64) -> f64 {
    if log_coeff == 0.0 && pure_coeff == 0.0 {
        return 0.0;
    }
    if exponent > 0.0 {
        return 0.0;
    }
    if exponent == 0.0 {
        if log_coeff != 0.0 {
            signed_infinity(-coeff * log_coeff)
        } else {
            coeff * pure_coeff
        }
    } else if log_coeff != 0.0 {
        signed_infinity(-coeff * log_coeff)
    } else {
        signed_infinity(coeff * pure_coeff)
    }
}

#[inline(always)]
fn polyharmonic_log_sign(m: usize, k_dim: usize) -> f64 {
    assert!(k_dim % 2 == 0);
    (-1.0_f64).powi(m as i32 - (k_dim as i32 / 2) + 1)
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
            // r^nu K_nu(kappa r) ~ 2^(nu-1) Γ(nu) kappa^(-nu).
            return Ok(c * 2.0_f64.powf(nu - 1.0) * gamma_lanczos(nu) * kappa.powf(-nu));
        }
        // Center-collision diagonal: zero by convention for nu <= 0 Matérn blocks.
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
    let (value, first, second, _, _) = duchon_matern_block_jet4(r, kappa, n_order, k_dim)?;
    Ok((value, first, second))
}

#[inline(always)]
fn polyharmonic_kernel_triplet(
    r: f64,
    m: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    let (value, first, second, _, _) = polyharmonic_block_jet4(r, m, k_dim)?;
    Ok((value, first, second))
}

/// Unified radial jet for one polyharmonic partial-fraction block.
///
/// Returns (φ, φ', φ'', φ''', φ'''') from a single consistent evaluation,
/// sharing normalization constant, r_safe, and log_r. This eliminates the
/// possibility of numerical drift between the triplet and higher-order
/// derivative paths.
fn polyharmonic_block_jet4(
    r: f64,
    m: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "polyharmonic distance must be finite and non-negative".to_string(),
        ));
    }

    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;

    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        let mut out = [0.0; 5];
        for d in 0..5 {
            let e = alpha - d as f64;
            let ff = falling_factorial(alpha, d);
            let ff_d = falling_factorial_derivative(alpha, d);
            out[d] = if r <= 0.0 {
                log_power_origin_limit(c, e, ff, ff_d)
            } else {
                c * r.powf(e) * (ff * r.ln() + ff_d)
            };
        }
        return Ok((out[0], out[1], out[2], out[3], out[4]));
    }

    let c = gamma_lanczos(k_half - m as f64)
        / (4.0_f64.powi(m as i32) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m as f64));
    let mut out = [0.0; 5];
    for d in 0..5 {
        let e = alpha - d as f64;
        let ff = falling_factorial(alpha, d);
        out[d] = if r <= 0.0 {
            log_power_origin_limit(c, e, 0.0, ff)
        } else {
            c * ff * r.powf(e)
        };
    }
    Ok((out[0], out[1], out[2], out[3], out[4]))
}

#[inline(always)]
fn log_power_family_derivative(exponent: f64, log_coeff: f64, pure_coeff: f64) -> (f64, f64, f64) {
    (
        exponent - 1.0,
        exponent * log_coeff,
        exponent * pure_coeff + log_coeff,
    )
}

#[inline(always)]
fn log_power_family_value(
    r: f64,
    coeff: f64,
    exponent: f64,
    log_coeff: f64,
    pure_coeff: f64,
) -> f64 {
    if r <= 0.0 {
        log_power_origin_limit(coeff, exponent, log_coeff, pure_coeff)
    } else {
        coeff * r.powf(exponent) * (log_coeff * r.ln() + pure_coeff)
    }
}

#[inline(always)]
fn duchon_polyharmonic_operator_block_jets(
    r: f64,
    m: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "polyharmonic distance must be finite and non-negative".to_string(),
        ));
    }

    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
    let (c, phi_log_coeff, phi_pure_coeff) = if k_dim % 2 == 0 && m >= (k_dim / 2) {
        (
            polyharmonic_log_sign(m, k_dim)
                / (2.0_f64.powi((2 * m - 1) as i32)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m as f64)
                    * gamma_lanczos((m - k_dim / 2 + 1) as f64)),
            1.0,
            0.0,
        )
    } else {
        (
            gamma_lanczos(k_half - m as f64)
                / (4.0_f64.powi(m as i32)
                    * std::f64::consts::PI.powf(k_half)
                    * gamma_lanczos(m as f64)),
            0.0,
            1.0,
        )
    };

    let (phi_r_exp, phi_r_log, phi_r_pure) =
        log_power_family_derivative(alpha, phi_log_coeff, phi_pure_coeff);
    let q_exp = phi_r_exp - 1.0;
    let q = log_power_family_value(r, c, q_exp, phi_r_log, phi_r_pure);

    let (q_r_exp_raw, q_r_log, q_r_pure) =
        log_power_family_derivative(q_exp, phi_r_log, phi_r_pure);
    let t_exp = q_r_exp_raw - 1.0;
    let t = log_power_family_value(r, c, t_exp, q_r_log, q_r_pure);

    let (t_r_exp, t_r_log, t_r_pure) = log_power_family_derivative(t_exp, q_r_log, q_r_pure);
    let t_r = log_power_family_value(r, c, t_r_exp, t_r_log, t_r_pure);

    let (t_rr_exp, t_rr_log, t_rr_pure) = log_power_family_derivative(t_r_exp, t_r_log, t_r_pure);
    let t_rr = log_power_family_value(r, c, t_rr_exp, t_rr_log, t_rr_pure);

    Ok((q, t, t_r, t_rr))
}

/// Unified radial jet for one scaled Matérn/Bessel-K family
///   coeff * r^mu * K_|mu|(kappa r).
///
/// Returns (g, g', g'', g''', g'''') from a single consistent evaluation.
/// The actual Duchon block jet and the stable operator jets for q/t reuse this
/// same helper with different `(coeff, mu)` pairs, so they share the exact same
/// Bessel evaluations and recurrence algebra.
///
/// Uses the exact recurrence derived from d/dr[r^ν K_ν(κr)] and the
/// Bessel identity dK_ν/dz = −K_{ν−1}(z) − (ν/z)K_ν(z), which gives
/// the cancellation pattern:
///
///   g^(0) = c · r^ν · K_ν(z)
///   g^(1) = −c · κ · r^ν · K_{ν−1}(z)
///   g^(2) = c·κ² r^ν K_{ν−2} − c·κ r^{ν−1} K_{ν−1}
///   g^(3) = 3c·κ² r^{ν−1} K_{ν−2} − c·κ³ r^ν K_{ν−3}
///   g^(4) = 3c·κ² r^{ν−2} K_{ν−2} − 6c·κ³ r^{ν−1} K_{ν−3} + c·κ⁴ r^ν K_{ν−4}
#[inline(always)]
fn duchon_matern_family_radial_derivative(
    r: f64,
    kappa: f64,
    coeff: f64,
    mu: f64,
    derivative_order: usize,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-family distance must be finite and non-negative".to_string(),
        ));
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-family kappa must be finite and positive".to_string(),
        ));
    }
    if r <= 0.0 && derivative_order == 0 && mu > 0.0 {
        return Ok(coeff * 2.0_f64.powf(mu - 1.0) * gamma_lanczos(mu) * kappa.powf(-mu));
    }
    if r <= 0.0 {
        return Ok(0.0);
    }

    let z = (kappa * r).max(1e-300);
    let mut terms = vec![DuchonMaternDerivativeTerm {
        coeff,
        kappa_power: 0,
        r_power: mu,
        bessel_order: mu,
    }];

    for _ in 0..derivative_order {
        let mut next_terms = Vec::with_capacity(terms.len() * 2);
        for term in terms {
            let stay_coeff = term.coeff * (term.r_power - term.bessel_order);
            if stay_coeff != 0.0 {
                next_terms.push(DuchonMaternDerivativeTerm {
                    coeff: stay_coeff,
                    kappa_power: term.kappa_power,
                    r_power: term.r_power - 1.0,
                    bessel_order: term.bessel_order,
                });
            }
            next_terms.push(DuchonMaternDerivativeTerm {
                coeff: -term.coeff,
                kappa_power: term.kappa_power + 1,
                r_power: term.r_power,
                bessel_order: term.bessel_order - 1.0,
            });
        }
        terms = next_terms;
    }

    let mut value = KahanSum::default();
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        let k_term = bessel_k_real_half_integer_or_integer(term.bessel_order.abs(), z)?;
        value.add(term.coeff * kappa.powi(term.kappa_power as i32) * r.powf(term.r_power) * k_term);
    }
    Ok(value.sum())
}

#[inline(always)]
fn duchon_matern_family_jet4(
    r: f64,
    kappa: f64,
    coeff: f64,
    mu: f64,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-family distance must be finite and non-negative".to_string(),
        ));
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon Matérn-family kappa must be finite and positive".to_string(),
        ));
    }
    Ok((
        duchon_matern_family_radial_derivative(r, kappa, coeff, mu, 0)?,
        duchon_matern_family_radial_derivative(r, kappa, coeff, mu, 1)?,
        duchon_matern_family_radial_derivative(r, kappa, coeff, mu, 2)?,
        duchon_matern_family_radial_derivative(r, kappa, coeff, mu, 3)?,
        duchon_matern_family_radial_derivative(r, kappa, coeff, mu, 4)?,
    ))
}

fn duchon_matern_block_jet4(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
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
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    if r <= 0.0 {
        let value = if nu > 0.0 {
            c * 2.0_f64.powf(nu - 1.0) * gamma_lanczos(nu) * kappa.powf(-nu)
        } else {
            0.0
        };
        return Ok((value, 0.0, 0.0, 0.0, 0.0));
    }

    duchon_matern_family_jet4(r, kappa, c, nu)
}

#[inline(always)]
fn duchon_matern_operator_block_jets(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64), BasisError> {
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
        return Ok((0.0, 0.0, 0.0, 0.0));
    }

    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    let (q, _, _, _, _) = duchon_matern_family_jet4(r, kappa, -c * kappa, nu - 1.0)?;
    let (t, t_r, t_rr, _, _) = duchon_matern_family_jet4(r, kappa, c * kappa * kappa, nu - 2.0)?;
    Ok((q, t, t_r, t_rr))
}

#[inline(always)]
fn pure_duchon_block_order(p_order: usize, s_order: usize) -> usize {
    p_order + s_order
}

fn validate_duchon_kernel_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> Result<(), BasisError> {
    if k_dim == 0 {
        return Err(BasisError::InvalidInput(
            "Duchon basis requires at least one covariate dimension".to_string(),
        ));
    }
    if let Some(scale) = length_scale
        && (!scale.is_finite() || scale <= 0.0)
    {
        return Err(BasisError::InvalidInput(
            "Duchon hybrid length_scale must be finite and positive".to_string(),
        ));
    }
    if length_scale.is_none() && 2 * s_order >= k_dim {
        return Err(BasisError::InvalidInput(format!(
            "pure Duchon requires power < dimension/2 for nullspace degree < {p_order}; got power={s_order}, dimension={k_dim}"
        )));
    }
    let spectral_order = 2 * (p_order + s_order);
    if spectral_order <= k_dim {
        return Err(BasisError::InvalidInput(format!(
            "Duchon pointwise kernel values require 2*(p+s) > dimension; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        )));
    }
    Ok(())
}

fn validate_duchon_collocation_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> Result<(), BasisError> {
    validate_duchon_kernel_orders(length_scale, p_order, s_order, k_dim)?;
    let spectral_order = 2 * (p_order + s_order);
    if spectral_order <= k_dim + 1 {
        return Err(BasisError::InvalidInput(format!(
            "Duchon D1 collocation requires 2*(p+s) > dimension+1; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        )));
    }
    if spectral_order <= k_dim + 2 {
        return Err(BasisError::InvalidInput(format!(
            "Duchon D2 collocation requires 2*(p+s) > dimension+2; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct DuchonPartialFractionCoeffs {
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
    let Some(length_scale) = length_scale else {
        return Ok(polyharmonic_kernel(
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
    if r == 0.0 {
        return duchon_hybrid_kernel_collision_value(
            length_scale,
            p_order,
            s_order,
            k_dim,
            coeffs_ref,
        );
    }
    let mut val = KahanSum::default();
    for (m, coeff) in coeffs_ref.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * polyharmonic_kernel(r, m, k_dim));
    }
    for (n, coeff) in coeffs_ref.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * duchon_matern_block(r, kappa, n, k_dim)?);
    }
    Ok(val.sum())
}

fn duchon_hybrid_kernel_collision_value(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    let spectral_order = 2 * (p_order + s_order);
    if spectral_order <= k_dim {
        return Err(BasisError::InvalidInput(format!(
            "Duchon hybrid diagonal is not finite when 2*(p+s) <= dimension; got 2*(p+s)={spectral_order}, dimension={k_dim}, p={p_order}, s={s_order}"
        )));
    }

    let kappa = 1.0 / length_scale.max(1e-300);
    let mut pure = KahanSum::default();
    let mut log_part = KahanSum::default();
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, 0);
        pure.add(a_m * block_pure);
        log_part.add(a_m * block_log);
    }
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, 0);
        pure.add(b_n * block_pure);
        log_part.add(b_n * block_log);
    }
    let value = pure.sum();
    let log_value = log_part.sum();
    if log_value.abs() > 1e-8 * value.abs().max(1e-30) {
        return Err(BasisError::InvalidInput(format!(
            "Duchon hybrid diagonal log terms did not cancel: log={log_value:.6e}, value={value:.6e}; p={p_order}, s={s_order}, d={k_dim}"
        )));
    }
    if !value.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon hybrid diagonal value for p={p_order}, s={s_order}, d={k_dim}"
        )));
    }
    Ok(value)
}

#[inline(always)]
fn stable_euclidean_norm<I>(components: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut scale = 0.0_f64;
    let mut sumsq = 1.0_f64;
    let mut has_nonzero = false;
    for component in components {
        let abs = component.abs();
        if abs == 0.0 {
            continue;
        }
        if !abs.is_finite() {
            return f64::INFINITY;
        }
        if !has_nonzero {
            scale = abs;
            has_nonzero = true;
            continue;
        }
        if scale < abs {
            let ratio = scale / abs;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = abs;
        } else {
            let ratio = abs / scale;
            sumsq += ratio * ratio;
        }
    }
    if has_nonzero {
        scale * sumsq.sqrt()
    } else {
        0.0
    }
}

#[inline]
fn centered_aniso_log_scale_mean(eta: &[f64]) -> f64 {
    if eta.len() <= 1 {
        0.0
    } else {
        eta.iter().sum::<f64>() / eta.len() as f64
    }
}

#[inline]
fn centered_aniso_log_scale(value: f64, mean: f64) -> f64 {
    (value - mean).clamp(-50.0, 50.0)
}

#[inline]
fn aniso_axis_scale(value: f64, mean: f64) -> f64 {
    centered_aniso_log_scale(value, mean).exp()
}

#[inline]
fn aniso_metric_weight(value: f64, mean: f64) -> f64 {
    (2.0 * centered_aniso_log_scale(value, mean)).exp()
}

fn centered_aniso_metric_weights(eta: &[f64]) -> Vec<f64> {
    let mean = centered_aniso_log_scale_mean(eta);
    eta.iter()
        .map(|&value| aniso_metric_weight(value, mean))
        .collect()
}

/// Compute anisotropic squared distance components and total distance.
///
/// This is the core of **geometric anisotropy**: a linear warp Λ = diag(κ_a)
/// turns ellipsoidal correlation contours into isotropic ones. Writing h = x − c,
/// z = Λh, the anisotropic distance is r = |z| = |Λh|.
///
/// We decompose Λ = κ · A where det(A) = 1, parameterized as
///   ψ_a = ψ̄ + η_a,   Σ η_a = 0
/// where ψ̄ is the global scale (existing scalar κ) and η_a are d−1 anisotropy
/// contrasts. This separates scale from shape and preserves the Duchon scaling
/// law φ(r;κ) = κ^δ H(κr) for the global part.
///
/// Given per-axis log-scales `eta`, the identifiable centered contrasts are
/// ψ_a = eta_a - mean(eta). The metric uses those contrasts so Σ_a ψ_a = 0
/// even when a caller passes an uncentered vector:
///
///   r = √( Σ_a exp(2·ψ_a) · (x_a - c_a)² )
///
/// Returns `(r, s_vec)` where `s_vec[a] = exp(2·ψ_a) · h_a²` is the
/// per-axis weighted squared displacement. These components are needed for
/// per-axis derivatives: `∂φ/∂ψ_a = q · s_a`.
///
/// The derivative chain through r gives:
///   ∇_ψ r      = s / r
///   ∇²_ψ r     = (2/r) Diag(s) − (1/r³) ss'
/// which is diagonal + rank-1, so Hessian-vector products are O(d).
#[inline]
fn aniso_distance_and_components(data_row: &[f64], center: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let d = data_row.len();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let mut s_vec = Vec::with_capacity(d);
    let mut scaled_components = Vec::with_capacity(d);
    for a in 0..d {
        let h_a = data_row[a] - center[a];
        // Clamp exp(2ψ) to avoid overflow/underflow: ψ in [-50, 50].
        let scale_a = aniso_axis_scale(eta[a], eta_mean);
        let scaled_h_a = scale_a * h_a;
        let s_a = scaled_h_a * scaled_h_a;
        scaled_components.push(scaled_h_a);
        s_vec.push(s_a);
    }
    (stable_euclidean_norm(scaled_components), s_vec)
}

/// Compute anisotropic distance without returning per-axis components.
///
/// This is the lightweight version of [`aniso_distance_and_components`] for
/// call sites that only need the scalar distance `r`.
#[inline]
fn aniso_distance(data_row: &[f64], center: &[f64], eta: &[f64]) -> f64 {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let eta_mean = centered_aniso_log_scale_mean(eta);
    stable_euclidean_norm(
        (0..data_row.len()).map(|a| aniso_axis_scale(eta[a], eta_mean) * (data_row[a] - center[a])),
    )
}

/// Return y-space points `y_{i,a} = exp(ψ_a) x_{i,a}` with
/// `ψ_a = η_a - mean(η)` so Euclidean pairwise
/// distances in y equal anisotropic kernel distances in x:
///   |y_i - y_j|² = Σ_a exp(2 ψ_a) (x_{i,a} - x_{j,a})² = aniso_distance²(x_i, x_j, η).
/// Use this before `pairwise_distance_bounds` whenever κ conditioning
/// bounds must match the kernel's actual metric (anisotropic case). For
/// isotropic terms, pass `None` and keep using the raw centers.
pub(crate) fn points_in_aniso_y_space(points: ArrayView2<'_, f64>, eta: &[f64]) -> Array2<f64> {
    assert_eq!(points.ncols(), eta.len());
    let mut y = points.to_owned();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let weights: Vec<f64> = eta.iter().map(|&e| aniso_axis_scale(e, eta_mean)).collect();
    for a in 0..eta.len() {
        let w_a = weights[a];
        y.column_mut(a).mapv_inplace(|v| v * w_a);
    }
    y
}

/// Compute per-axis standard deviations of knot center coordinates.
///
/// Returns σ_a for each axis column of `centers`. Axes with zero variance
/// (constant column) get σ_a = 1.0. All values are clamped to [1e-6, 1e6].
pub fn knot_cloud_axis_scales(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let k = centers.nrows();
    let d = centers.ncols();
    if k < 2 || d == 0 {
        return vec![1.0; d];
    }
    let n = k as f64;
    let mut scales = Vec::with_capacity(d);
    for a in 0..d {
        let col = centers.column(a);
        let mean = col.sum() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma = var.sqrt();
        // If variance is zero (constant column), use 1.0 (no scaling).
        let sigma = if sigma < 1e-12 { 1.0 } else { sigma };
        scales.push(sigma.clamp(1e-6, 1e6));
    }
    scales
}

/// Compute initial anisotropy contrasts η_a from knot center geometry.
///
/// Returns η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b), which satisfies Ση_a = 0
/// by construction. Axes with more spread get negative η_a (smaller κ_a,
/// longer correlation range), axes with less spread get positive η_a.
///
/// If d ≤ 1, returns an empty vector (anisotropy is meaningless for 1-D).
pub fn initial_aniso_contrasts(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let d = centers.ncols();
    if d <= 1 {
        return Vec::new();
    }
    let scales = knot_cloud_axis_scales(centers);
    let neg_log_scales: Vec<f64> = scales.iter().map(|&s| -s.ln()).collect();
    let mean_neg_log: f64 = neg_log_scales.iter().sum::<f64>() / d as f64;
    // η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b)
    //     = −ln(σ_a) − mean(−ln(σ_b))
    //     = neg_log_scales[a] − mean(neg_log_scales)
    neg_log_scales
        .iter()
        .map(|&nls| nls - mean_neg_log)
        .collect()
}

/// Detect the all-zero sentinel from `--scale-dimensions` and replace with
/// knot-cloud-derived contrasts. Non-zero or absent aniso is passed through.
fn maybe_initialize_aniso_contrasts(
    centers: ArrayView2<'_, f64>,
    aniso: Option<&[f64]>,
) -> Option<Vec<f64>> {
    fn center(eta: &[f64]) -> Vec<f64> {
        if eta.len() <= 1 {
            return eta.to_vec();
        }
        let mean = eta.iter().sum::<f64>() / eta.len() as f64;
        eta.iter()
            .map(|&v| {
                let centered = v - mean;
                if centered.abs() <= 1e-15 {
                    0.0
                } else {
                    centered
                }
            })
            .collect()
    }

    let eta = match aniso {
        Some(v) if v.len() > 1 => v,
        Some(v) => return Some(v.to_vec()),
        None => return None,
    };
    let all_zero = eta.iter().all(|&e| e == 0.0);
    if !all_zero {
        return Some(center(eta));
    }
    let contrasts = initial_aniso_contrasts(centers);
    if contrasts.is_empty() {
        Some(center(eta))
    } else {
        Some(center(&contrasts))
    }
}

pub(crate) fn pairwise_distance_bounds(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
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

/// Capped-sample pairwise distance bounds for large point clouds.
///
/// Returns `(r_min_hat, r_max_hat)` such that:
/// - `r_max_hat <= true r_max`  (pairwise max over a sub-sample is monotone
///    in the sample, so the sampled max underestimates the true max).
/// - `r_min_hat >= true r_min`  (pairwise min over a sub-sample can only
///    exclude some pairs, so the sampled min overestimates the true min).
///
/// Both approximations are conservative for κ-bound derivation:
///   kappa_lo = 1e-2 / r_max_hat  >=  1e-2 / true r_max  (wider window, low κ)
///   kappa_hi = 1e2  / r_min_hat  <=  1e2  / true r_min  (tighter window, high κ)
/// so no feasible κ that the exact bound would include is excluded by the
/// approximation — it can only slightly shrink the high-κ tail, which is
/// exactly the regime (κ → ∞ ⇒ degenerate kernel) that we want the outer
/// optimizer to avoid anyway.
///
/// Sampling is deterministic stride (points indexed 0, stride, 2·stride, …).
/// For a cap of `K = 1024` and n up to ~10⁹ this yields O(K²·d) work per
/// call — a few hundred μs. For n < K the exact pairwise is used.
pub(crate) fn pairwise_distance_bounds_sampled(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    const K_CAP: usize = 1024;
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    if n <= K_CAP {
        return pairwise_distance_bounds(points);
    }
    // Deterministic stride sampling: pick K_CAP evenly spaced indices.
    // This preserves any spatial stratification already present in the
    // data ordering (biobank data is typically in insertion order, not
    // spatially stratified, so stride sampling is effectively uniform).
    let stride = n / K_CAP;
    let k = K_CAP; // exactly K_CAP samples by construction (stride rounds down)
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i_idx in 0..k {
        let i = i_idx * stride;
        for j_idx in (i_idx + 1)..k {
            let j = j_idx * stride;
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
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
    datarows: usize,
    data_cols: usize,
    data_ptr: usize,
    data_stride0: isize,
    data_stride1: isize,
    centersrows: usize,
    centers_cols: usize,
    centers_hash: u64,
}

#[derive(Debug, Clone)]
struct SpatialDistanceCacheEntry {
    data_center_r: Arc<Array2<f64>>,
    center_center_r: Arc<Array2<f64>>,
}

impl crate::resource::ResidentBytes for SpatialDistanceCacheEntry {
    fn resident_bytes(&self) -> usize {
        std::mem::size_of::<f64>()
            .saturating_mul(self.data_center_r.nrows())
            .saturating_mul(self.data_center_r.ncols())
            .saturating_add(
                std::mem::size_of::<f64>()
                    .saturating_mul(self.center_center_r.nrows())
                    .saturating_mul(self.center_center_r.ncols()),
            )
    }
}

const SPATIAL_DISTANCE_CACHE_MIN_PAIRS: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConstraintNullspaceCacheKey {
    centersrows: usize,
    centers_cols: usize,
    centers_hash: u64,
    order_code: u8,
}

#[derive(Default, Clone, Debug)]
struct ConstraintNullspaceCache {
    map: HashMap<ConstraintNullspaceCacheKey, Arc<Array2<f64>>>,
    order: Vec<ConstraintNullspaceCacheKey>,
}

const CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OwnedDataCacheKey {
    rows: usize,
    cols: usize,
    ptr: usize,
    stride0: isize,
    stride1: isize,
}

#[derive(Debug)]
struct BasisCacheContext {
    spatial_distance:
        crate::resource::ByteLruCache<SpatialDistanceCacheKey, SpatialDistanceCacheEntry>,
    constraint_nullspace: ConstraintNullspaceCache,
    owned_data: crate::resource::ByteLruCache<OwnedDataCacheKey, Arc<Array2<f64>>>,
}

impl BasisCacheContext {
    fn with_policy(policy: &crate::resource::ResourcePolicy) -> Self {
        Self {
            spatial_distance: crate::resource::ByteLruCache::new(
                policy.max_spatial_distance_cache_bytes,
            ),
            constraint_nullspace: ConstraintNullspaceCache::default(),
            owned_data: crate::resource::ByteLruCache::with_max_entries(
                policy.max_owned_data_cache_bytes,
                crate::resource::OWNED_DATA_CACHE_MAX_ENTRIES,
            ),
        }
    }
}

impl Default for BasisCacheContext {
    fn default() -> Self {
        Self::with_policy(&crate::resource::ResourcePolicy::default_library())
    }
}

/// Explicit per-run workspace for basis/spatial cache reuse.
///
/// Pass one workspace through repeated basis builds to avoid global mutable state
/// and to keep caching scoped to a caller-controlled lifecycle.
///
/// The spatial-distance and owned-data caches are byte-limited via the
/// [`crate::resource::ResourcePolicy`] provided at construction; use
/// [`BasisWorkspace::with_policy`] for biobank-scale workloads where a single
/// entry can be multiple gigabytes.
#[derive(Debug)]
pub struct BasisWorkspace {
    cache: BasisCacheContext,
    policy: crate::resource::ResourcePolicy,
}

impl BasisWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: crate::resource::ResourcePolicy) -> Self {
        Self {
            cache: BasisCacheContext::with_policy(&policy),
            policy,
        }
    }

    pub fn default_library() -> Self {
        Self::with_policy(crate::resource::ResourcePolicy::default_library())
    }

    /// Returns the resource policy this workspace was configured with.
    pub fn policy(&self) -> &crate::resource::ResourcePolicy {
        &self.policy
    }
}

impl Default for BasisWorkspace {
    fn default() -> Self {
        Self::default_library()
    }
}

fn hash_arrayview2(values: ArrayView2<'_, f64>) -> u64 {
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
                row[j] = stable_euclidean_norm((0..d).map(|c| data[[i, c]] - centers[[j, c]]));
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
            let r = stable_euclidean_norm((0..d).map(|c| centers[[i, c]] - centers[[j, c]]));
            distances[[i, j]] = r;
            distances[[j, i]] = r;
        }
    }
    distances
}

#[inline(always)]
fn spatial_distance_data_center_bytes(n: usize, k: usize) -> usize {
    n.saturating_mul(k)
        .saturating_mul(std::mem::size_of::<f64>())
}

#[inline(always)]
fn spatial_distance_cacheable_entry(n: usize, k: usize) -> bool {
    spatial_distance_data_center_bytes(n, k)
        <= crate::resource::SPATIAL_DISTANCE_CACHE_SINGLE_ENTRY_MAX_BYTES
}

fn spatial_distance_matrices(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    cache: &BasisCacheContext,
) -> Result<(Arc<Array2<f64>>, Arc<Array2<f64>>), BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    if n.saturating_mul(k) < SPATIAL_DISTANCE_CACHE_MIN_PAIRS
        || !spatial_distance_cacheable_entry(n, k)
    {
        let dc = Arc::new(compute_data_center_distances(data, centers)?);
        let cc = Arc::new(compute_center_center_distances(centers));
        return Ok((dc, cc));
    }

    let key = SpatialDistanceCacheKey {
        datarows: data.nrows(),
        data_cols: data.ncols(),
        data_ptr: data.as_ptr() as usize,
        data_stride0: data.strides()[0],
        data_stride1: data.strides()[1],
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
    };

    if let Some(hit) = cache.spatial_distance.get(&key) {
        return Ok((hit.data_center_r.clone(), hit.center_center_r.clone()));
    }

    let computed_dc = Arc::new(compute_data_center_distances(data, centers)?);
    let computed_cc = Arc::new(compute_center_center_distances(centers));

    if let Some(hit) = cache.spatial_distance.get(&key) {
        return Ok((hit.data_center_r.clone(), hit.center_center_r.clone()));
    }
    cache.spatial_distance.insert(
        key,
        SpatialDistanceCacheEntry {
            data_center_r: computed_dc.clone(),
            center_center_r: computed_cc.clone(),
        },
    );
    Ok((computed_dc, computed_cc))
}

fn constraint_nullspace_order_code(order: DuchonNullspaceOrder) -> u8 {
    match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
        DuchonNullspaceOrder::Degree(degree) => degree.min(u8::MAX as usize) as u8,
    }
}

#[inline(always)]
fn thin_plate_constraint_nullspace_order_code() -> u8 {
    16
}

fn shared_owned_data_matrix(
    data: ArrayView2<'_, f64>,
    cache: &BasisCacheContext,
) -> Arc<Array2<f64>> {
    let key = OwnedDataCacheKey {
        rows: data.nrows(),
        cols: data.ncols(),
        ptr: data.as_ptr() as usize,
        stride0: data.strides()[0],
        stride1: data.strides()[1],
    };
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    let owned = Arc::new(data.to_owned());
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    cache.owned_data.insert(key, owned.clone());
    owned
}

/// Minimal cache-less intern: wraps an `ArrayView2` into an `Arc<Array2<f64>>`.
///
/// Used by derivative-operator builders that don't have a `BasisCacheContext`
/// in scope (e.g. `build_aniso_design_psi_derivatives_shared`). The goal is the
/// same as `shared_owned_data_matrix`: move the owned payload into an `Arc`
/// once so that downstream `StreamingRadialState` copies share it via
/// `Arc::clone` instead of materializing a fresh n×d `Array2<f64>` per axis.
#[inline]
fn shared_owned_data_matrix_from_view(data: ArrayView2<'_, f64>) -> Arc<Array2<f64>> {
    Arc::new(data.to_owned())
}

/// Minimal cache-less intern for knot centers; mirrors
/// `shared_owned_data_matrix_from_view`. Centers are typically k×d with k
/// much smaller than n, but the `Arc::clone` pattern still avoids a k×d
/// copy per axis when the same operator feeds multiple derivative paths.
#[inline]
fn shared_owned_centers_matrix_from_view(centers: ArrayView2<'_, f64>) -> Arc<Array2<f64>> {
    Arc::new(centers.to_owned())
}

fn kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let effective_order = duchon_effective_nullspace_order(centers, order);
    let degraded = effective_order != order;
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order_code: constraint_nullspace_order_code(effective_order),
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = polynomial_block_from_order(centers, effective_order);
    let z = Arc::new(kernel_constraint_nullspace_from_matrix(p_k.view()).map_err(|err| {
        if degraded {
            BasisError::InvalidInput(format!(
                "Duchon degraded from order={:?} to order={:?} due to insufficient centers ({} in dim={}); order={:?} construction then failed: {err}",
                order,
                effective_order,
                centers.nrows(),
                centers.ncols(),
                effective_order,
            ))
        } else {
            err
        }
    })?);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
    }

    Ok((*z).clone())
}

fn thin_plate_kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order_code: thin_plate_constraint_nullspace_order_code(),
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = thin_plate_polynomial_block(centers);
    if centers.nrows() < p_k.ncols() {
        return Err(BasisError::InvalidInput(format!(
            "thin-plate spline requires at least {} centers to span the degree-{} polynomial null space in dimension {}; got {}",
            p_k.ncols(),
            thin_plate_polynomial_degree(centers.ncols()),
            centers.ncols(),
            centers.nrows()
        )));
    }
    let (z, rank) =
        rrqr_nullspace_basis(&p_k, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank != p_k.ncols() {
        return Err(BasisError::InvalidInput(format!(
            "thin-plate spline polynomial block is rank deficient at the selected centers: expected rank {}, got {}; choose geometrically independent centers for dimension {}",
            p_k.ncols(),
            rank,
            centers.ncols()
        )));
    }
    let z = Arc::new(z);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
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
            // Mirror the Duchon path: auto-degrade to Zero (constant-only) when
            // there aren't enough centers to affinely span [1, x_1, ..., x_d].
            // kernel_constraint_nullspace_from_matrix would otherwise hard-error
            // via rrqr_nullspace_basis when centers.nrows() < d + 1.
            let effective_order =
                duchon_effective_nullspace_order(centers, DuchonNullspaceOrder::Linear);
            let q = polynomial_block_from_order(centers, effective_order);
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
    aniso_log_scales: Option<&[f64]>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let ops = build_matern_collocation_operator_matrices(
        centers,
        None,
        length_scale,
        nu,
        include_intercept,
        z_opt.map(|z| z.view()),
        aniso_log_scales,
    )?;
    Ok(operator_penalty_candidates_from_collocation(
        &ops.d0,
        &ops.d1,
        &ops.d2,
        &DuchonOperatorPenaltySpec::default(),
    ))
}

fn build_matern_double_penalty_candidates(
    spline: &MaternSplineBasis,
    full_transform: Option<&Array2<f64>>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let primary = project_penalty_matrix(&spline.penalty_kernel, full_transform);
    let mut candidates = vec![normalize_penalty_candidate(
        primary.clone(),
        0,
        PenaltySource::Primary,
    )];
    if let Some(shrinkage) = build_nullspace_shrinkage_penalty(&primary)? {
        candidates.push(normalize_penalty_candidate(
            shrinkage.sym_penalty,
            0,
            PenaltySource::DoublePenaltyNullspace,
        ));
    }
    Ok(candidates)
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
pub fn create_matern_spline_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
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
    if let Some(eta) = aniso_log_scales {
        if eta.len() != d {
            return Err(BasisError::DimensionMismatch(format!(
                "aniso_log_scales length {} does not match data dimension {d}",
                eta.len()
            )));
        }
        if eta.iter().any(|v| !v.is_finite()) {
            return Err(BasisError::InvalidInput(
                "aniso_log_scales must contain finite values".to_string(),
            ));
        }
    }

    // Practical safe operating range for κ from center geometry (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min], with κ = 1/length_scale.
    // Warn rather than silently clamp so callers keep explicit control.
    // Under anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    let warn_bounds = if let Some(eta) = aniso_log_scales {
        let y_centers = points_in_aniso_y_space(centers, eta);
        pairwise_distance_bounds(y_centers.view())
    } else {
        pairwise_distance_bounds(centers)
    };
    if let Some((r_min, r_max)) = warn_bounds {
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

    // Distance computation: anisotropic when eta is present, isotropic otherwise.
    // Under anisotropy we work in y-space (y = Ax), so r = |Ah| replaces |h|.
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    if let Some(eta) = aniso_log_scales {
        // Anisotropic path: compute distances via aniso_distance.
        let kernel_result: Result<(), BasisError> = kernel_block
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .try_for_each(|(i, mut row)| {
                let xi = data.row(i);
                for j in 0..k {
                    let r = aniso_distance(
                        xi.as_slice().unwrap(),
                        centers.row(j).as_slice().unwrap(),
                        eta,
                    );
                    row[j] = matern_kernel_from_distance(r, length_scale, nu)?;
                }
                Ok(())
            });
        kernel_result?;
        for i in 0..k {
            for j in i..k {
                let r = aniso_distance(
                    centers.row(i).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                );
                let kij = matern_kernel_from_distance(r, length_scale, nu)?;
                center_kernel[[i, j]] = kij;
                center_kernel[[j, i]] = kij;
            }
        }
    } else {
        // Isotropic path: use cached spatial distance matrices.
        let (data_center_r, center_center_r) =
            spatial_distance_matrices(data, centers, &mut workspace.cache)?;
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
        for i in 0..k {
            for j in i..k {
                let kij = matern_kernel_from_distance(center_center_r[[i, j]], length_scale, nu)?;
                center_kernel[[i, j]] = kij;
                center_kernel[[j, i]] = kij;
            }
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
    build_matern_basiswithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    // Initialize anisotropy contrasts from knot cloud geometry when the caller
    // enabled scale-dimensions but left η at the zero default.
    let aniso = maybe_initialize_aniso_contrasts(centers.view(), spec.aniso_log_scales.as_deref());
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let identifiability_transform = z_opt.clone();
    let full_transform = z_opt.as_ref().map(|z| {
        if spec.include_intercept {
            append_intercept_to_transform(z)
        } else {
            z.clone()
        }
    });
    let design_cols =
        z_opt.as_ref().map_or(centers.nrows(), Array2::ncols) + usize::from(spec.include_intercept);
    let dense_bytes = dense_design_bytes(data.nrows(), design_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), design_cols, workspace.policy());
    let (design, candidates) = if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "Matérn basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            design_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
        let shared_data = shared_owned_data_matrix(data, &mut workspace.cache);
        let d = data.ncols();
        let length_scale = spec.length_scale;
        let nu = spec.nu;
        let poly_basis = if spec.include_intercept {
            Some(Arc::new(Array2::<f64>::ones((data.nrows(), 1))))
        } else {
            None
        };
        let design = if let Some(eta) = aniso.as_ref() {
            let metric_weights = eta.iter().map(|&v| (2.0 * v).exp()).collect::<Vec<_>>();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let mut q = 0.0f64;
                for axis in 0..data_row.len() {
                    let delta = data_row[axis] - center_row[axis];
                    q += metric_weights[axis] * delta * delta;
                }
                matern_kernel_from_distance(q.sqrt(), length_scale, nu)
                    .expect("validated Matérn inputs should not fail")
            };
            let op = ChunkedKernelDesignOperator::new(
                shared_data.clone(),
                Arc::new(centers.clone()),
                kernel,
                z_opt.as_ref().map(|z| Arc::new(z.clone())),
                poly_basis.clone(),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
        } else {
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let r = stable_euclidean_norm((0..d).map(|axis| data_row[axis] - center_row[axis]));
                matern_kernel_from_distance(r, length_scale, nu)
                    .expect("validated Matérn inputs should not fail")
            };
            let op = ChunkedKernelDesignOperator::new(
                shared_data,
                Arc::new(centers.clone()),
                kernel,
                z_opt.as_ref().map(|z| Arc::new(z.clone())),
                poly_basis,
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
        };
        let candidates = if spec.double_penalty {
            let penalty_kernel = build_matern_kernel_penalty(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                aniso.as_deref(),
            )?;
            let primary = project_penalty_matrix(&penalty_kernel, full_transform.as_ref());
            let mut candidates = vec![normalize_penalty_candidate(
                primary.clone(),
                0,
                PenaltySource::Primary,
            )];
            if let Some(shrinkage) = build_nullspace_shrinkage_penalty(&primary)? {
                candidates.push(normalize_penalty_candidate(
                    shrinkage.sym_penalty,
                    0,
                    PenaltySource::DoublePenaltyNullspace,
                ));
            }
            candidates
        } else {
            build_matern_operator_penalty_candidates(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso.as_deref(),
            )?
        };
        (design, candidates)
    } else {
        let m = create_matern_spline_basiswithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            aniso.as_deref(),
            workspace,
        )?;
        let design = if let Some(transform) = full_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(
                &m.basis, transform,
            )))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(m.basis.clone()))
        };
        let candidates = if spec.double_penalty {
            build_matern_double_penalty_candidates(&m, full_transform.as_ref())?
        } else {
            build_matern_operator_penalty_candidates(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso.as_deref(),
            )?
        };
        (design, candidates)
    };
    let (penalties, nullspace_dims, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Matern {
            centers,
            length_scale: spec.length_scale,
            nu: spec.nu,
            include_intercept: spec.include_intercept,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: aniso,
        },
        kronecker_factored: None,
    })
}

#[inline(always)]
fn eval_polywith_derivatives(coeffs: &[f64], a: f64) -> (f64, f64, f64) {
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
fn maternvalue_psi_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64), BasisError> {
    // Exact value + hyper-derivatives for psi = log(kappa)
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
    // When a > 700, exp(-a) underflows to 0 while p(a) can overflow to Inf,
    // producing 0 * Inf = NaN.  All terms carry exp(-a) as a factor, so the
    // triplet is exactly zero for large a.
    if a > 700.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    let e = (-a).exp();
    let (p0, p1, p2) = eval_polywith_derivatives(p, a);
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
    // When a > 700, exp(-a) underflows to 0 while the polynomial can overflow,
    // giving 0 * Inf = NaN.  All terms carry exp(-a), so the result is exactly 0.
    if a > 700.0 {
        return (0.0, 0.0, 0.0);
    }
    let e = (-a).exp();
    let (p0, p1, p2) = eval_polywith_derivatives(coeffs, a);
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
        f64, // derivative of phi_r_over_r with respect to psi
        f64, // second derivative of phi_r_over_r with respect to psi
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
    // The `q` and `rr` arrays below are the literal coefficient arrays for
    // Q_nu(a) and R_nu(a), including the normalization factors such as 1/3,
    // 1/15, and 1/105.
    //
    // Then psi-derivatives are obtained exactly through
    // exp_poly_scaled_s2_psi_triplet, avoiding finite differences.
    let (phi, phi_psi, phi_psi_psi) = maternvalue_psi_triplet(r, length_scale, nu)?;
    let kappa = 1.0 / length_scale;
    let d = dimension as f64;
    let (s, q, rr): (f64, &[f64], &[f64]) = match nu {
        MaternNu::Half => (kappa, &[1.0], &[1.0]),
        MaternNu::ThreeHalves => (3.0_f64.sqrt() * kappa, &[1.0], &[-1.0, 1.0]),
        MaternNu::FiveHalves => (
            5.0_f64.sqrt() * kappa,
            &[1.0 / 3.0, 1.0 / 3.0],
            &[-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0],
        ),
        MaternNu::SevenHalves => (
            7.0_f64.sqrt() * kappa,
            &[1.0 / 5.0, 1.0 / 5.0, 1.0 / 15.0],
            &[-1.0 / 5.0, -1.0 / 5.0, 0.0, 1.0 / 15.0],
        ),
        MaternNu::NineHalves => (
            9.0_f64.sqrt() * kappa,
            &[1.0 / 7.0, 1.0 / 7.0, 2.0 / 35.0, 1.0 / 105.0],
            &[
                -1.0 / 7.0,
                -1.0 / 7.0,
                -1.0 / 35.0,
                2.0 / 105.0,
                1.0 / 105.0,
            ],
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

/// Cross second derivative of the Gram penalty w.r.t. two different axes a and b:
///   S_raw_{ab} = D_{ab}'^T D + D'^T D_b + D_a'^T D_b + D^T D_{ab}
/// where D_a = ∂D/∂ψ_a, D_b = ∂D/∂ψ_b, D_{ab} = ∂²D/∂ψ_a∂ψ_b.
fn gram_cross_psi_derivative_from_operator(
    d: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    d_ab: &Array2<f64>,
) -> Array2<f64> {
    symmetrize(&(d_ab.t().dot(d) + d.t().dot(d_ab) + d_a.t().dot(d_b) + d_b.t().dot(d_a)))
}

/// Normalize a cross second derivative ∂²S~_m/∂ψ_a∂ψ_b using the Frobenius norm chain rule.
///
/// Given:
///   S     = the raw Gram penalty (axis-independent)
///   S_a   = ∂S/∂ψ_a (first derivative, axis a)
///   S_b   = ∂S/∂ψ_b (first derivative, axis b)
///   S_ab  = ∂²S/∂ψ_a∂ψ_b (cross second derivative, raw)
///   c     = ||S||_F (Frobenius norm)
///
/// The normalized cross second derivative is:
///   S~_{ab} = S_{ab}/c - (c_a/c²)·S_b - (c_b/c²)·S_a + (2·c_a·c_b/c³ - c_{ab}/c²)·S
///
/// where c_a = tr(S'·S_a)/c, c_b = tr(S'·S_b)/c, and
///   c_{ab} = [tr(S_a'·S_b) + tr(S'·S_{ab})]/c - c_a·c_b/c.
fn normalize_penalty_cross_psi_derivative(
    s: &Array2<f64>,
    s_a: &Array2<f64>,
    s_b: &Array2<f64>,
    s_ab: &Array2<f64>,
    c: f64,
) -> Array2<f64> {
    if !c.is_finite() || c <= 1e-12 {
        return Array2::<f64>::zeros(s.raw_dim());
    }

    let c2 = c * c;
    let c3 = c2 * c;

    // c_a = tr(S^T S_a) / c
    let a_val = trace_of_product(s, s_a);
    let c_a = a_val / c;

    // c_b = tr(S^T S_b) / c
    let b_val = trace_of_product(s, s_b);
    let c_b = b_val / c;

    // c_{ab} = [tr(S_a^T S_b) + tr(S^T S_{ab})] / c - c_a * c_b / c
    let cross_val = trace_of_product(s_a, s_b) + trace_of_product(s, s_ab);
    let c_ab = cross_val / c - c_a * c_b / c;

    // S~_{ab} = S_{ab}/c - (c_a/c²)·S_b - (c_b/c²)·S_a + (2·c_a·c_b/c³ - c_{ab}/c²)·S
    let coeff_s = 2.0 * c_a * c_b / c3 - c_ab / c2;
    s_ab.mapv(|v| v / c) - s_b.mapv(|v| c_a / c2 * v) - s_a.mapv(|v| c_b / c2 * v)
        + s.mapv(|v| coeff_s * v)
}

#[inline(always)]
fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.t().dot(b).diag().sum()
}

fn normalize_penaltywith_psi_derivatives(
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
    aniso_log_scales: Option<&[f64]>,
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
            let r = if let Some(eta) = aniso_log_scales {
                aniso_distance(
                    centers.row(k).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
            } else {
                stable_euclidean_norm((0..d).map(|c| centers[[k, c]] - centers[[j, c]]))
            };
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

    let (s0_norm, s0_norm_psi, s0_norm_psi_psi, c0) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (s1_norm, s1_norm_psi, s1_norm_psi_psi, c1) =
        normalize_penaltywith_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (s2_norm, s2_norm_psi, s2_norm_psi_psi, c2) =
        normalize_penaltywith_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);
    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
        },
    ];
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    let penalties_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi, s1_norm_psi, s2_norm_psi],
        "Matérn",
    )?;
    let penaltiessecond_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi_psi, s1_norm_psi_psi, s2_norm_psi_psi],
        "Matérn",
    )?;
    Ok((penalties_derivative, penaltiessecond_derivative))
}

fn build_duchon_operator_penalty_psi_derivatives(
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
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power;
    validate_duchon_collocation_orders(Some(length_scale), p_order, s_order, centers.ncols())?;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let p = centers.nrows();
    let d = centers.ncols();
    let mut d0_raw = Array2::<f64>::zeros((p, z_kernel.ncols()));
    let mut d1_raw = Array2::<f64>::zeros((p * d, z_kernel.ncols()));
    let mut d2_raw = Array2::<f64>::zeros((p, z_kernel.ncols()));
    let mut d0_raw_psi = Array2::<f64>::zeros((p, z_kernel.ncols()));
    let mut d1_raw_psi = Array2::<f64>::zeros((p * d, z_kernel.ncols()));
    let mut d2_raw_psi = Array2::<f64>::zeros((p, z_kernel.ncols()));
    let mut d0_raw_psi_psi = Array2::<f64>::zeros((p, z_kernel.ncols()));
    let mut d1_raw_psi_psi = Array2::<f64>::zeros((p * d, z_kernel.ncols()));
    let mut d2_raw_psi_psi = Array2::<f64>::zeros((p, z_kernel.ncols()));

    let aniso = spec.aniso_log_scales.as_deref();
    if let Some(eta) = aniso
        && eta.len() != d
    {
        return Err(BasisError::DimensionMismatch(format!(
            "Duchon anisotropy dimension mismatch: got {}, expected {d}",
            eta.len()
        )));
    }
    let metric_weights: Option<Vec<f64>> = aniso.map(centered_aniso_metric_weights);
    let sum_metric_weights = metric_weights
        .as_ref()
        .map(|weights| weights.iter().sum::<f64>())
        .unwrap_or(d as f64);
    for k in 0..p {
        for j in k..p {
            let mut s_vec = Vec::new();
            let r = if let Some(eta) = aniso {
                let row_k: Vec<f64> = (0..d).map(|a| centers[[k, a]]).collect();
                let row_j: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
                let (r, components) = aniso_distance_and_components(&row_k, &row_j, eta);
                s_vec = components;
                r
            } else {
                stable_euclidean_norm((0..d).map(|axis| centers[[k, axis]] - centers[[j, axis]]))
            };
            let core =
                duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, d, &coeffs)?;
            for col in 0..z_kernel.ncols() {
                let z_jc = z_kernel[[j, col]];
                d0_raw[[k, col]] += core.phi.value * z_jc;
                d0_raw_psi[[k, col]] += core.phi.psi * z_jc;
                d0_raw_psi_psi[[k, col]] += core.phi.psi_psi * z_jc;
                if j != k {
                    let z_kc = z_kernel[[k, col]];
                    d0_raw[[j, col]] += core.phi.value * z_kc;
                    d0_raw_psi[[j, col]] += core.phi.psi * z_kc;
                    d0_raw_psi_psi[[j, col]] += core.phi.psi_psi * z_kc;
                }
            }
            if r > 1e-10 {
                let jets = duchon_radial_jets(r, length_scale, p_order, s_order, d, &coeffs)?;
                let q = jets.q;
                let (q_psi, q_psi_psi) =
                    duchon_q_psi_triplet_from_jets(&jets, p_order, s_order, d, r);
                let (lap, lap_psi, lap_psi_psi) = if let Some(weights) = metric_weights.as_ref() {
                    let sum_wb_sb: f64 = (0..d).map(|axis| weights[axis] * s_vec[axis]).sum();
                    let t_exponent = duchon_scaling_exponent(p_order, s_order, d) + 4.0;
                    let (t_psi, t_psi_psi) =
                        scaled_log_kappa_derivatives(jets.t, jets.t_r, jets.t_rr, t_exponent, r);
                    (
                        q * sum_metric_weights + jets.t * sum_wb_sb,
                        q_psi * sum_metric_weights + t_psi * sum_wb_sb,
                        q_psi_psi * sum_metric_weights + t_psi_psi * sum_wb_sb,
                    )
                } else {
                    let (lap_psi, lap_psi_psi) =
                        duchon_laplacian_psi_triplet_from_jets(&jets, p_order, s_order, d, r);
                    (jets.lap, lap_psi, lap_psi_psi)
                };
                for axis in 0..d {
                    let delta = centers[[k, axis]] - centers[[j, axis]];
                    let axis_scale = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis])
                        .unwrap_or(1.0);
                    let row = k * d + axis;
                    for col in 0..z_kernel.ncols() {
                        let z_jc = z_kernel[[j, col]];
                        d1_raw[[row, col]] += q * axis_scale * delta * z_jc;
                        d1_raw_psi[[row, col]] += q_psi * axis_scale * delta * z_jc;
                        d1_raw_psi_psi[[row, col]] += q_psi_psi * axis_scale * delta * z_jc;
                        if j != k {
                            let row_sym = j * d + axis;
                            let z_kc = z_kernel[[k, col]];
                            d1_raw[[row_sym, col]] -= q * axis_scale * delta * z_kc;
                            d1_raw_psi[[row_sym, col]] -= q_psi * axis_scale * delta * z_kc;
                            d1_raw_psi_psi[[row_sym, col]] -= q_psi_psi * axis_scale * delta * z_kc;
                        }
                    }
                }
                for col in 0..z_kernel.ncols() {
                    let z_jc = z_kernel[[j, col]];
                    d2_raw[[k, col]] += lap * z_jc;
                    d2_raw_psi[[k, col]] += lap_psi * z_jc;
                    d2_raw_psi_psi[[k, col]] += lap_psi_psi * z_jc;
                    if j != k {
                        let z_kc = z_kernel[[k, col]];
                        d2_raw[[j, col]] += lap * z_kc;
                        d2_raw_psi[[j, col]] += lap_psi * z_kc;
                        d2_raw_psi_psi[[j, col]] += lap_psi_psi * z_kc;
                    }
                }
            } else {
                let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
                    duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, d, &coeffs)?;
                let (_, lap_collision) = duchon_collision_operator_core_fromphi_rr(
                    phi_rr,
                    phi_rr_psi,
                    phi_rr_psi_psi,
                    d,
                );
                let lap_value = if metric_weights.is_some() {
                    sum_metric_weights * phi_rr
                } else {
                    lap_collision.value
                };
                let lap_psi = if metric_weights.is_some() {
                    sum_metric_weights * phi_rr_psi
                } else {
                    lap_collision.psi
                };
                let lap_psi_psi = if metric_weights.is_some() {
                    sum_metric_weights * phi_rr_psi_psi
                } else {
                    lap_collision.psi_psi
                };
                for col in 0..z_kernel.ncols() {
                    let z_jc = z_kernel[[j, col]];
                    d2_raw[[k, col]] += lap_value * z_jc;
                    d2_raw_psi[[k, col]] += lap_psi * z_jc;
                    d2_raw_psi_psi[[k, col]] += lap_psi_psi * z_jc;
                    if j != k {
                        let z_kc = z_kernel[[k, col]];
                        d2_raw[[j, col]] += lap_value * z_kc;
                        d2_raw_psi[[j, col]] += lap_psi * z_kc;
                        d2_raw_psi_psi[[j, col]] += lap_psi_psi * z_kc;
                    }
                }
            }
        }
    }

    let poly = polynomial_block_from_order(centers, effective_nullspace_order);
    let kernel_cols = d0_raw.ncols();
    let poly_cols = poly.ncols();
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
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_raw);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_raw);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_raw);
    d0_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d0_raw_psi);
    d1_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d1_raw_psi);
    d2_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d2_raw_psi);
    d0_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_raw_psi_psi);
    d1_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_raw_psi_psi);
    d2_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_raw_psi_psi);
    if poly_cols > 0 {
        d0.slice_mut(s![.., kernel_cols..]).assign(&poly);
        if poly_cols > 1 {
            for k in 0..p {
                for axis in 0..d {
                    d1[[k * d + axis, kernel_cols + 1 + axis]] = 1.0;
                }
            }
        }
    }

    let project = |mat: Array2<f64>| {
        if let Some(z) = identifiability_transform {
            fast_ab(&mat, z)
        } else {
            mat
        }
    };
    let d0 = project(d0);
    let d1 = project(d1);
    let d2 = project(d2);
    let d0_psi = project(d0_psi);
    let d1_psi = project(d1_psi);
    let d2_psi = project(d2_psi);
    let d0_psi_psi = project(d0_psi_psi);
    let d1_psi_psi = project(d1_psi_psi);
    let d2_psi_psi = project(d2_psi_psi);

    let (s0, s0_psi, s0_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d0, &d0_psi, &d0_psi_psi);
    let (s1, s1_psi, s1_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d1, &d1_psi, &d1_psi_psi);
    let (s2, s2_psi, s2_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d2, &d2_psi, &d2_psi_psi);

    let (s0_norm, s0_norm_psi, s0_norm_psi_psi, c0) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (s1_norm, s1_norm_psi, s1_norm_psi_psi, c1) =
        normalize_penaltywith_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (s2_norm, s2_norm_psi, s2_norm_psi_psi, c2) =
        normalize_penaltywith_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);
    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
        },
    ];
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    let penalties_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi, s1_norm_psi, s2_norm_psi],
        "Duchon",
    )?;
    let penaltiessecond_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi_psi, s1_norm_psi_psi, s2_norm_psi_psi],
        "Duchon",
    )?;
    Ok((penalties_derivative, penaltiessecond_derivative))
}

fn prepare_duchon_derivative_contextwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_biobank_cap(data.nrows(), data.ncols(), centers.view());
    let raw_design = build_duchon_basis_designwithworkspace(
        data,
        centers.view(),
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        spec.aniso_log_scales.as_deref(),
        workspace,
    )?;
    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        raw_design.basis.view(),
        &spec.identifiability,
        "Duchon",
    )?;
    Ok((centers, identifiability_transform))
}

fn build_matern_design_psi_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let k = centers.nrows();
    let kernel_cols = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let total_cols = kernel_cols + usize::from(include_intercept);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        aniso_log_scales,
        total_cols,
        z_opt.cloned(),
        None,
        usize::from(include_intercept),
        RadialScalarKind::Matern { length_scale, nu },
        0.0,
    )
}

fn build_matern_double_penalty_primarywith_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, f64), BasisError> {
    let k = centers.nrows();
    let kernel_cols = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut kernel = Array2::<f64>::zeros((k, k));
    let mut kernel_psi = Array2::<f64>::zeros((k, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((k, k));

    for i in 0..k {
        for j in i..k {
            let r = if let Some(eta) = aniso_log_scales {
                aniso_distance(
                    centers.row(i).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
            } else {
                stable_euclidean_norm(
                    (0..centers.ncols()).map(|axis| centers[[i, axis]] - centers[[j, axis]]),
                )
            };
            let value = matern_kernel_from_distance(r, length_scale, nu)?;
            let d1 = matern_kernel_log_kappa_derivative_from_distance(r, length_scale, nu)?;
            let d2 = matern_kernel_log_kappasecond_derivative_from_distance(r, length_scale, nu)?;
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
        normalize_penaltywith_psi_derivatives(&s, &s_psi, &s_psi_psi);
    Ok((s_norm, s_norm_psi, s_norm_psi_psi, c))
}

fn active_matern_double_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penaltyinfo
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
    build_matern_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    // Analytic psi derivative assembly for the Matérn basis block.
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let aniso = spec.aniso_log_scales.as_deref();
    let design_derivatives = build_matern_design_psi_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        z_opt.as_ref(),
        aniso,
    )?;
    let penalties_derivative = if spec.double_penalty {
        let base = build_matern_basiswithworkspace(data, spec, workspace)?;
        let (_, primary_derivative, _, _) =
            build_matern_double_penalty_primarywith_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso,
            )?;
        active_matern_double_penalty_derivatives(&base.penaltyinfo, &primary_derivative)?
    } else {
        let (penalties_derivative, _) = build_matern_operator_penalty_psi_derivatives(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
            aniso,
        )?;
        penalties_derivative
    };

    Ok(BasisPsiDerivativeResult {
        design_derivative: design_derivatives.design_first,
        penalties_derivative,
        implicit_operator: design_derivatives.implicit_operator,
    })
}

pub fn build_matern_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    // Analytic psi second-derivative assembly, matching the first-derivative
    // mapping logic and constrained normalized penalty geometry.
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let aniso = spec.aniso_log_scales.as_deref();
    let design_derivatives = build_matern_design_psi_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        z_opt.as_ref(),
        aniso,
    )?;
    let penaltiessecond_derivative = if spec.double_penalty {
        let base = build_matern_basiswithworkspace(data, spec, workspace)?;
        let (_, _, primarysecond_derivative, _) =
            build_matern_double_penalty_primarywith_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso,
            )?;
        active_matern_double_penalty_derivatives(&base.penaltyinfo, &primarysecond_derivative)?
    } else {
        let (_, penaltiessecond_derivative) = build_matern_operator_penalty_psi_derivatives(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
            aniso,
        )?;
        penaltiessecond_derivative
    };

    Ok(BasisPsiSecondDerivativeResult {
        designsecond_derivative: design_derivatives.design_second_diag,
        penaltiessecond_derivative,
        implicit_operator: design_derivatives.implicit_operator,
    })
}

/// Build per-axis ψ_a design-matrix derivatives for anisotropic Matérn terms.
///
/// The optimized coordinates are the raw per-axis log-scales `psi_a`, so the
/// isotropic all-ones direction is part of this coordinate system. For Matérn
/// kernels there is no extra isotropic prefactor, so the raw-`psi` derivatives
/// are exactly the familiar shape-only terms.
fn build_matern_design_psi_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    eta: &[f64],
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let k = centers.nrows();
    let p_constrained = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let n_poly = usize::from(include_intercept);
    let p_smooth = p_constrained + n_poly;
    build_aniso_design_psi_derivatives_shared(
        data,
        centers,
        eta,
        p_smooth,
        z_opt.cloned(),
        None,
        n_poly,
        RadialScalarKind::Matern { length_scale, nu },
    )
}

/// Build per-axis ψ_a derivatives for anisotropic Matérn terms, including
/// both design-matrix and penalty derivatives.
///
/// For each axis a (0..d), produces first and second derivative information.
/// The penalty derivatives use the fractional weighting approach for operator
/// penalties, and exact per-axis R-operator derivatives for double penalties.
pub fn build_matern_basis_log_kappa_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let eta = spec.aniso_log_scales.as_deref().ok_or_else(|| {
        BasisError::InvalidInput("aniso derivatives require aniso_log_scales to be set".to_string())
    })?;
    let dim = data.ncols();
    if eta.len() != dim {
        return Err(BasisError::DimensionMismatch(format!(
            "aniso_log_scales length {} != data dimension {dim}",
            eta.len()
        )));
    }

    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;

    let mut result = build_matern_design_psi_aniso_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        eta,
        spec.include_intercept,
        z_opt.as_ref(),
    )?;

    // Penalty per-axis derivatives.
    if spec.double_penalty {
        // Double-penalty path: per-axis primary penalty derivatives via R-operators.
        let k = centers.nrows();
        let kernel_cols = z_opt.as_ref().map(|z| z.ncols()).unwrap_or(k);
        let total_cols = kernel_cols + usize::from(spec.include_intercept);
        let mut primary_first = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let mut primary_second_diag = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let mut raw_first = vec![Array2::<f64>::zeros((k, k)); dim];
        let mut raw_second_diag = vec![Array2::<f64>::zeros((k, k)); dim];
        for i in 0..k {
            let ci: Vec<f64> = (0..dim).map(|a| centers[[i, a]]).collect();
            for j in i..k {
                let cj: Vec<f64> = (0..dim).map(|a| centers[[j, a]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);
                let (_, q, t, _, _) =
                    matern_aniso_extended_radial_scalars(r, spec.length_scale, spec.nu)?;
                for a in 0..dim {
                    let d1 = q * s_vec[a];
                    let d2 = 2.0 * q * s_vec[a] + t * s_vec[a] * s_vec[a];
                    raw_first[a][[i, j]] = d1;
                    raw_first[a][[j, i]] = d1;
                    raw_second_diag[a][[i, j]] = d2;
                    raw_second_diag[a][[j, i]] = d2;
                }
            }
        }
        for a in 0..dim {
            let projected_first = if let Some(z) = z_opt.as_ref() {
                z.t().dot(&raw_first[a]).dot(z)
            } else {
                raw_first[a].clone()
            };
            let projected_second = if let Some(z) = z_opt.as_ref() {
                z.t().dot(&raw_second_diag[a]).dot(z)
            } else {
                raw_second_diag[a].clone()
            };
            primary_first[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_first);
            primary_second_diag[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_second);
        }
        let mut dp_cross_pairs: Vec<(usize, usize)> = Vec::new();
        for a in 0..dim {
            for b in (a + 1)..dim {
                dp_cross_pairs.push((a, b));
            }
        }

        let base = build_matern_basiswithworkspace(data, spec, &mut BasisWorkspace::default())?;
        result.penalties_first = Vec::with_capacity(dim);
        result.penalties_second_diag = Vec::with_capacity(dim);
        for a in 0..dim {
            let pf =
                active_matern_double_penalty_derivatives(&base.penaltyinfo, &primary_first[a])?;
            let ps = active_matern_double_penalty_derivatives(
                &base.penaltyinfo,
                &primary_second_diag[a],
            )?;
            result.penalties_first.push(pf);
            result.penalties_second_diag.push(ps);
        }
        result.penalties_cross_pairs = dp_cross_pairs;
        let centers_owned = centers.to_owned();
        let eta_owned = eta.to_vec();
        let z_owned = z_opt.clone();
        let penaltyinfo = base.penaltyinfo.clone();
        let length_scale = spec.length_scale;
        let nu = spec.nu;
        result.penalties_cross_provider = Some(AnisoPenaltyCrossProvider::new(
            move |axis_a: usize, axis_b: usize| {
                let (a, b) = if axis_a < axis_b {
                    (axis_a, axis_b)
                } else {
                    (axis_b, axis_a)
                };
                if a == b || b >= eta_owned.len() {
                    return Ok(Vec::new());
                }
                let mut raw_cross = Array2::<f64>::zeros((k, k));
                for i in 0..k {
                    let ci_v: Vec<f64> = (0..dim).map(|ax| centers_owned[[i, ax]]).collect();
                    for j_idx in i..k {
                        let cj_v: Vec<f64> =
                            (0..dim).map(|ax| centers_owned[[j_idx, ax]]).collect();
                        let (r, s_vec) = aniso_distance_and_components(&ci_v, &cj_v, &eta_owned);
                        let (_, _, t_val, _, _) =
                            matern_aniso_extended_radial_scalars(r, length_scale, nu)?;
                        let val = t_val * s_vec[a] * s_vec[b];
                        raw_cross[[i, j_idx]] = val;
                        raw_cross[[j_idx, i]] = val;
                    }
                }
                let projected: Array2<f64> = if let Some(z) = z_owned.as_ref() {
                    z.t().dot(&raw_cross).dot(z)
                } else {
                    raw_cross
                };
                let mut padded = Array2::<f64>::zeros((total_cols, total_cols));
                padded
                    .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                    .assign(&projected);
                active_matern_double_penalty_derivatives(&penaltyinfo, &padded)
            },
        ));
    } else {
        // Operator penalty path: exact per-axis η_a derivatives.
        // Replaces the former fractional approximation with exact analytic
        // derivatives of D₀, D₁, D₂ w.r.t. each aniso log-scale η_a,
        // assembled via the Gram product rule into penalty derivatives.
        let (per_axis, cross_pairs, cross_provider) =
            build_matern_operator_penalty_aniso_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                eta,
            )?;

        result.penalties_first = Vec::with_capacity(dim);
        result.penalties_second_diag = Vec::with_capacity(dim);
        for (pen_first, pen_second) in per_axis {
            result.penalties_first.push(pen_first);
            result.penalties_second_diag.push(pen_second);
        }
        result.penalties_cross_pairs = cross_pairs;
        result.penalties_cross_provider = Some(cross_provider);
    }

    Ok(result)
}

/// Build per-axis ψ_a design-matrix derivatives for anisotropic Duchon terms.
///
/// Exactly parallels [`build_matern_design_psi_aniso_derivatives`] but uses
/// [`duchon_radial_jets`] to obtain the radial scalars (φ, q, t).
///
/// The per-axis chain rule is identical:
///   ∂φ/∂ψ_a         = q · s_a
///   ∂²φ/(∂ψ_a²)     = 2q · s_a + t · s_a²
///   ∂²φ/(∂ψ_a ∂ψ_b) = t · s_a · s_b   (a ≠ b)
fn build_duchon_design_psi_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "aniso Duchon derivatives require hybrid Duchon with length_scale".to_string(),
        )
    })?;
    let eta = spec.aniso_log_scales.as_deref().ok_or_else(|| {
        BasisError::InvalidInput("aniso derivatives require aniso_log_scales to be set".to_string())
    })?;
    let dim = data.ncols();
    if eta.len() != dim {
        return Err(BasisError::DimensionMismatch(format!(
            "Duchon aniso penalty derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        )));
    }

    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power;
    let kappa = 1.0 / length_scale.max(1e-300);
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);

    // Z_kernel: null-space constraint projection for Duchon polynomial conditions.
    let z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();

    // Determine output dimension to decide dense vs implicit.
    let p_constrained = z_kernel.ncols();
    let p_padded = p_constrained + poly_cols;
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_padded);

    let radial_kind = RadialScalarKind::Duchon {
        length_scale,
        p_order,
        s_order,
        dim,
        coeffs: coeffs.clone(),
    };
    build_aniso_design_psi_derivatives_shared(
        data,
        centers,
        eta,
        p_final,
        Some(z_kernel),
        identifiability_transform.cloned(),
        poly_cols,
        radial_kind,
    )
}

fn pure_duchon_axis_combinations(dim: usize) -> Vec<Vec<(usize, f64)>> {
    if dim <= 1 {
        return vec![vec![(0, 1.0)]];
    }
    let last = dim - 1;
    (0..last)
        .map(|axis| vec![(axis, 1.0), (last, -1.0)])
        .collect()
}

fn pure_duchon_reparameterize_penalty_axes(
    per_axis: Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>,
    cross_pairs: Vec<(usize, usize)>,
    cross_provider: AnisoPenaltyCrossProvider,
    dim: usize,
) -> (
    Vec<Vec<Array2<f64>>>,
    Vec<Vec<Array2<f64>>>,
    Vec<(usize, usize)>,
    Option<AnisoPenaltyCrossProvider>,
) {
    let free_dim = dim.saturating_sub(1).max(1);
    if dim <= 1 {
        let mut per_axis_iter = per_axis.into_iter();
        let (first, second_diag) = per_axis_iter.next().unwrap_or_default();
        return (vec![first], vec![second_diag], Vec::new(), None);
    }

    let last = dim - 1;
    let raw_first: Vec<Vec<Array2<f64>>> =
        per_axis.iter().map(|(first, _)| first.clone()).collect();
    let raw_second_diag: Vec<Vec<Array2<f64>>> =
        per_axis.iter().map(|(_, second)| second.clone()).collect();

    let mut penalties_first = Vec::with_capacity(free_dim);
    let mut penalties_second_diag = Vec::with_capacity(free_dim);
    for axis in 0..free_dim {
        let first_axis = raw_first[axis]
            .iter()
            .zip(raw_first[last].iter())
            .map(|(lhs, rhs)| lhs - rhs)
            .collect();
        penalties_first.push(first_axis);

        let second_axis = raw_second_diag[axis]
            .clone()
            .into_iter()
            .zip(
                cross_provider
                    .evaluate(axis, last)
                    .expect("pure Duchon raw cross-penalty derivative axis/last"),
            )
            .zip(raw_second_diag[last].clone())
            .map(|((aa, al), ll)| aa - al.mapv(|value| 2.0 * value) + ll)
            .collect();
        penalties_second_diag.push(second_axis);
    }

    let mut penalties_cross_pairs = Vec::new();
    for axis_a in 0..free_dim {
        for axis_b in (axis_a + 1)..free_dim {
            penalties_cross_pairs.push((axis_a, axis_b));
        }
    }
    let raw_second_diag = std::sync::Arc::new(raw_second_diag);
    let cross_pairs = std::sync::Arc::new(cross_pairs);
    let reparam_provider = AnisoPenaltyCrossProvider::new(move |axis_a: usize, axis_b: usize| {
        let (a, b) = if axis_a < axis_b {
            (axis_a, axis_b)
        } else {
            (axis_b, axis_a)
        };
        if a >= free_dim || b >= free_dim || !cross_pairs.contains(&(a, b)) {
            return Ok(Vec::new());
        }
        let ab = cross_provider.evaluate(a, b)?;
        let al = cross_provider.evaluate(a, last)?;
        let bl = cross_provider.evaluate(b, last)?;
        let ll = raw_second_diag[last].clone();
        Ok(ab
            .into_iter()
            .zip(al)
            .zip(bl)
            .zip(ll)
            .map(|(((ab, al), bl), ll)| ab - al - bl + ll)
            .collect())
    });

    (
        penalties_first,
        penalties_second_diag,
        penalties_cross_pairs,
        Some(reparam_provider),
    )
}

fn build_pure_duchon_basis_log_kappa_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let Some(raw_eta) = spec.aniso_log_scales.as_deref() else {
        return Err(BasisError::InvalidInput(
            "aniso derivatives require aniso_log_scales to be set".to_string(),
        ));
    };
    let dim = data.ncols();
    if raw_eta.len() != dim {
        return Err(BasisError::DimensionMismatch(format!(
            "aniso_log_scales length {} != data dimension {dim}",
            raw_eta.len()
        )));
    }
    if spec.length_scale.is_some() {
        return Err(BasisError::InvalidInput(
            "pure Duchon aniso derivative path requires length_scale=None".to_string(),
        ));
    }
    let mut workspace = BasisWorkspace::default();
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let effective_nullspace_order =
        duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let z_kernel = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let p_padded = z_kernel.ncols() + poly_cols;
    let identifiability_transform =
        frozen_spatial_identifiability_transform(&spec.identifiability, p_padded, "Duchon")?;
    let p_final = identifiability_transform
        .as_ref()
        .map(|transform| transform.ncols())
        .unwrap_or(p_padded);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power;
    validate_duchon_collocation_orders(None, p_order, s_order, dim)?;
    let block_order = pure_duchon_block_order(p_order, s_order);
    let mut design_result = build_aniso_design_psi_derivatives_shared(
        data,
        centers.view(),
        raw_eta,
        p_final,
        Some(z_kernel),
        identifiability_transform.clone(),
        poly_cols,
        RadialScalarKind::PureDuchon {
            block_order,
            p_order,
            s_order,
            dim,
        },
    )?;

    let axis_combinations = pure_duchon_axis_combinations(dim);
    if let Some(op) = design_result.implicit_operator.take() {
        design_result.implicit_operator = Some(op.with_axis_combinations(axis_combinations));
    }
    design_result.design_first.clear();
    design_result.design_second_diag.clear();
    design_result.design_second_cross.clear();
    design_result.design_second_cross_pairs.clear();

    let (per_axis, cross_terms, cross_provider) = build_duchon_operator_penalty_aniso_derivatives(
        centers.view(),
        None,
        spec.power,
        effective_nullspace_order,
        raw_eta,
        identifiability_transform.as_ref(),
        &mut workspace,
    )?;
    let (penalties_first, penalties_second_diag, penalties_cross_pairs, penalties_cross_provider) =
        pure_duchon_reparameterize_penalty_axes(per_axis, cross_terms, cross_provider, dim);
    design_result.penalties_first = penalties_first;
    design_result.penalties_second_diag = penalties_second_diag;
    design_result.penalties_cross_pairs = penalties_cross_pairs;
    design_result.penalties_cross_provider = penalties_cross_provider;
    Ok(design_result)
}

/// Build per-axis ψ_a derivatives for anisotropic Duchon terms, including
/// both design-matrix and penalty derivatives.
///
/// Uses exact per-axis operator derivatives via
/// [`build_duchon_operator_penalty_aniso_derivatives`], which computes
/// D₀, D₁, D₂ and their η_a first/second derivatives at all center pairs
/// and assembles penalty derivatives via the Gram product rule.
pub fn build_duchon_basis_log_kappa_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let eta = spec.aniso_log_scales.as_deref().ok_or_else(|| {
        BasisError::InvalidInput("aniso derivatives require aniso_log_scales to be set".to_string())
    })?;
    let dim = data.ncols();
    if eta.len() != dim {
        return Err(BasisError::DimensionMismatch(format!(
            "aniso_log_scales length {} != data dimension {dim}",
            eta.len()
        )));
    }

    if spec.length_scale.is_none() {
        return build_pure_duchon_basis_log_kappa_aniso_derivatives(data, spec);
    }
    let length_scale = spec.length_scale.expect("checked above");

    let mut workspace = BasisWorkspace::default();
    let (centers, identifiability_transform) =
        prepare_duchon_derivative_contextwithworkspace(data, spec, &mut workspace)?;
    let effective_nullspace_order =
        duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);

    let mut result = build_duchon_design_psi_aniso_derivatives(
        data,
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        &mut workspace,
    )?;

    // Exact operator penalty path: per-axis D₀, D₁, D₂ derivatives via
    // Gram product rule, replacing the former fractional approximation.
    let (per_axis, cross_pairs, cross_provider) = build_duchon_operator_penalty_aniso_derivatives(
        centers.view(),
        Some(length_scale),
        spec.power,
        effective_nullspace_order,
        eta,
        identifiability_transform.as_ref(),
        &mut workspace,
    )?;

    result.penalties_first = Vec::with_capacity(dim);
    result.penalties_second_diag = Vec::with_capacity(dim);
    for (pen_first, pen_second) in per_axis {
        result.penalties_first.push(pen_first);
        result.penalties_second_diag.push(pen_second);
    }
    result.penalties_cross_pairs = cross_pairs;
    result.penalties_cross_provider = Some(cross_provider);

    Ok(result)
}

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
fn duchon_has_classicalsecond_order_origin(p_order: usize, s_order: usize, k_dim: usize) -> bool {
    2 * (p_order + s_order) > k_dim + 2
}

#[derive(Clone, Copy)]
struct DuchonMaternDerivativeTerm {
    coeff: f64,
    kappa_power: usize,
    r_power: f64,
    bessel_order: f64,
}

#[inline(always)]
fn duchon_matern_block_radial_derivative(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    derivative_order: usize,
) -> Result<f64, BasisError> {
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
    if r <= 0.0 && derivative_order == 0 {
        return duchon_matern_block(0.0, kappa, n_order, k_dim);
    }
    if r <= 0.0 {
        return Ok(0.0);
    }
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    duchon_matern_family_radial_derivative(r, kappa, c, nu, derivative_order)
}

#[inline(always)]
fn duchon_polyharmonicsecond_collision_psi_triplet(
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
        let c = polyharmonic_log_sign(m, k_dim)
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
fn duchon_maternsecond_collision_psi_triplet(
    length_scale: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    // Same collision strategy for shifted blocks: evaluate the second radial
    // derivative at a length-scale-relative floor and propagate psi derivatives
    // analytically using the local scaling exponent.
    let kappa = 1.0 / length_scale;
    let r_eff = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let (_, _, second) = duchon_matern_block_triplet(r_eff, kappa, n_order, k_dim)?;
    let n = n_order as f64;
    let nu = n - 0.5 * k_dim as f64;
    let scale = 2.0 - 2.0 * nu;
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
    #[cfg(test)]
    gradient_ratio: PsiTriplet,
    #[cfg(test)]
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
    /// R-operator radial scalar: t = R²φ = (φ'' - q) / r² = q' / r.
    /// At collision (r = 0): t = φ''''(0) / 3, computed via assembled
    /// fourth-derivative collision limits of the partial-fraction blocks.
    t: f64,
    /// First radial derivative of t:
    ///   t_r = dt/dr = (q_rr - t) / r  for r > 0.
    /// At collision, the exact radial limit is t_r(0) = 0.
    t_r: f64,
    /// Second radial derivative of t:
    ///   t_rr = d²t/dr² = [lap_rr + 2 t - (d + 4) q_rr] / r²  for r > 0,
    /// using Delta phi = d q + r² t.
    ///
    /// At collision, the exact radial limit is
    ///   t_rr(0) = φ⁽⁶⁾(0) / 15.
    t_rr: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DuchonRegularizedOperatorCore {
    q: f64,
    t: f64,
    t_r: f64,
    t_rr: f64,
}

#[inline(always)]
fn duchon_operator_jets_from_primary_core(
    core: DuchonRegularizedOperatorCore,
    r: f64,
    d: f64,
) -> DuchonRadialJets {
    let r2 = r * r;
    let mut out = DuchonRadialJets {
        q: core.q,
        t: core.t,
        t_r: core.t_r,
        t_rr: core.t_rr,
        ..DuchonRadialJets::default()
    };
    out.q_r = r * out.t;
    out.q_rr = out.t + r * out.t_r;
    out.lap = d * out.q + r2 * out.t;
    out.lap_r = (d + 2.0) * r * out.t + r2 * out.t_r;
    out.lap_rr = (d + 2.0) * out.t + (d + 4.0) * r * out.t_r + r2 * out.t_rr;
    out.phi_r = r * out.q;
    out.phi_rr = out.q + r2 * out.t;

    assert!(((out.phi_rr - (out.q + r * out.q_r)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0));
    assert!(((out.phi_rr - (out.q + r2 * out.t)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0));
    assert!(((out.lap - (d * out.q + r2 * out.t)).abs()) <= 1e-10 * out.lap.abs().max(1.0));

    out
}

#[inline(always)]
fn scaled_log_kappa_derivatives(
    value: f64,
    radial_first: f64,
    radialsecond: f64,
    exponent: f64,
    r: f64,
) -> (f64, f64) {
    // Scaling-law differentiation template
    // For any radial quantity of the form
    //   F(r; kappa) = kappa^a G(kappa r),
    // with psi = log(kappa), one has d/dpsi = kappa d/dkappa.
    //
    // Writing t = kappa r,
    //   F_psi
    //   = kappa d/dkappa [kappa^a G(t)]
    //   = a kappa^a G(t) + kappa^a (kappa r) G'(t)
    //   = a F + r F_r.
    //
    // Differentiating again,
    //   F_psipsi
    //   = d/dpsi [a F + r F_r]
    //   = a F_psi + r (F_r)_psi
    //   = a (a F + r F_r) + r d/dr(F_psi)
    //   = a^2 F + (2a + 1) r F_r + r^2 F_rr.
    //
    // This helper is the common exact formula used for:
    //   - phi            with exponent delta
    //   - q = phi_r / r  with exponent delta + 2
    //   - Delta phi      with exponent delta + 2.
    let first = exponent * value + r * radial_first;
    let second = exponent * exponent * value
        + (2.0 * exponent + 1.0) * r * radial_first
        + r * r * radialsecond;
    (first, second)
}

#[inline(always)]
fn duchon_operator_scaling_exponent(p_order: usize, s_order: usize, k_dim: usize) -> f64 {
    // For the hybrid Duchon spectrum
    //   1 / (|w|^(2p) (kappa^2 + |w|^2)^s),
    // the spatial kernel scales as
    //   phi(r; kappa) = kappa^delta H(kappa r),
    // where
    //   delta = d - 2p - 2s.
    //
    // A first spatial derivative contributes one extra factor of kappa, so
    // phi_r scales like kappa^(delta + 1). Dividing by r gives
    //   q(r; kappa) = phi_r / r = kappa^(delta + 2) Q(kappa r).
    //
    // The Laplacian also contributes two spatial derivatives, so
    //   Delta phi(r; kappa) = kappa^(delta + 2) L(kappa r).
    //
    // Thus both Duchon operator scalars use exponent delta + 2.
    duchon_scaling_exponent(p_order, s_order, k_dim) + 2.0
}

#[inline(always)]
fn duchon_q_psi_triplet_from_jets(
    jets: &DuchonRadialJets,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    r: f64,
) -> (f64, f64) {
    // Exact scaling derivatives for
    //   q(r; kappa) = phi_r(r; kappa) / r.
    //
    // Since q scales like kappa^(delta + 2),
    //   q_psi     = (delta + 2) q + r q_r
    //   q_psipsi  = (delta + 2)^2 q + (2 delta + 5) r q_r + r^2 q_rr.
    //
    // For r > 0 this is algebraically equivalent to
    //   q_psi = (delta + 1) q + phi_rr,
    // because q_r = (phi_rr - q) / r.
    scaled_log_kappa_derivatives(
        jets.q,
        jets.q_r,
        jets.q_rr,
        duchon_operator_scaling_exponent(p_order, s_order, k_dim),
        r,
    )
}

#[inline(always)]
fn duchon_laplacian_psi_triplet_from_jets(
    jets: &DuchonRadialJets,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    r: f64,
) -> (f64, f64) {
    // Exact scaling derivatives for
    //   ell(r; kappa) = Delta phi(r; kappa).
    //
    // The Laplacian raises the scaling exponent by 2, so ell follows the same
    // exponent as q:
    //   ell_psi     = (delta + 2) ell + r ell_r
    //   ell_psipsi  = (delta + 2)^2 ell + (2 delta + 5) r ell_r + r^2 ell_rr.
    scaled_log_kappa_derivatives(
        jets.lap,
        jets.lap_r,
        jets.lap_rr,
        duchon_operator_scaling_exponent(p_order, s_order, k_dim),
        r,
    )
}

#[inline(always)]
fn duchon_collision_operator_core_fromphi_rr(
    phi_rr: f64,
    phi_rr_psi: f64,
    phi_rr_psi_psi: f64,
    k_dim: usize,
) -> (PsiTriplet, PsiTriplet) {
    // Center-collision identities for a C^2 radial kernel:
    //   q(0)       = lim_{r->0} phi_r(r)/r = phi_rr(0)
    //   Delta phi(0) = d * phi_rr(0).
    //
    // The same identities propagate to psi derivatives as long as the
    // corresponding collision limits exist:
    //   q_psi(0)        = phi_rr_psi(0)
    //   q_psipsi(0)     = phi_rr_psipsi(0)
    //   (Delta phi)_psi(0)    = d * phi_rr_psi(0)
    //   (Delta phi)_psipsi(0) = d * phi_rr_psipsi(0).
    //
    // This helper packages exactly those canonical origin values.
    (
        PsiTriplet {
            value: phi_rr,
            psi: phi_rr_psi,
            psi_psi: phi_rr_psi_psi,
        },
        PsiTriplet {
            value: k_dim as f64 * phi_rr,
            psi: k_dim as f64 * phi_rr_psi,
            psi_psi: k_dim as f64 * phi_rr_psi_psi,
        },
    )
}

fn duchon_regularized_operator_core(
    r_eval: f64,
    kappa: f64,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRegularizedOperatorCore, BasisError> {
    // Assemble the operator scalars with compensated summation because the
    // partial-fraction coefficients can alternate in sign and span many orders
    // of magnitude in higher dimensions.
    let mut q_sum = KahanSum::default();
    let mut t_sum = KahanSum::default();
    let mut t_r_sum = KahanSum::default();
    let mut t_rr_sum = KahanSum::default();

    for (m, coeff) in coeffs.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (q, t, t_r, t_rr) = duchon_polyharmonic_operator_block_jets(r_eval, m, k_dim)?;
        q_sum.add(coeff * q);
        t_sum.add(coeff * t);
        t_r_sum.add(coeff * t_r);
        t_rr_sum.add(coeff * t_rr);
    }
    for (n, coeff) in coeffs.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (q, t, t_r, t_rr) = duchon_matern_operator_block_jets(r_eval, kappa, n, k_dim)?;
        q_sum.add(coeff * q);
        t_sum.add(coeff * t);
        t_r_sum.add(coeff * t_r);
        t_rr_sum.add(coeff * t_rr);
    }
    Ok(DuchonRegularizedOperatorCore {
        q: q_sum.sum(),
        t: t_sum.sum(),
        t_r: t_r_sum.sum(),
        t_rr: t_rr_sum.sum(),
    })
}

#[inline(always)]
fn duchon_collision_taylor_operator_core(
    r: f64,
    phi_rr_collision: f64,
    t_collision: f64,
    t_rr_collision: f64,
) -> DuchonRegularizedOperatorCore {
    let r2 = r * r;
    let r4 = r2 * r2;
    DuchonRegularizedOperatorCore {
        q: phi_rr_collision + 0.5 * t_collision * r2 + 0.125 * t_rr_collision * r4,
        t: t_collision + 0.5 * t_rr_collision * r2,
        t_r: t_rr_collision * r,
        t_rr: t_rr_collision,
    }
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
    let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale.max(1e-8);
    let r_eval = r.max(r_floor);
    let d = k_dim as f64;

    // Value path keeps the intrinsic diagonal convention used by the actual basis.
    let phi = duchon_matern_kernel_general_from_distance(
        r,
        Some(length_scale),
        p_order,
        s_order,
        k_dim,
        Some(coeffs),
    )?;
    if !phi.is_finite() {
        return Err(BasisError::InvalidInput(format!(
            "non-finite Duchon radial kernel value at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        )));
    }

    // Assemble the operator scalars directly from the partial-fraction blocks.
    // This avoids the unstable off-origin subtraction
    //   t = (phi_rr - phi_r / r) / r^2
    // in high dimensions, where phi_rr and phi_r / r can be enormous and nearly
    // cancel long before the final Duchon operator stays moderate.
    let generic_jets = duchon_operator_jets_from_primary_core(
        duchon_regularized_operator_core(r_eval, kappa, k_dim, coeffs)?,
        r_eval,
        d,
    );
    let mut out = DuchonRadialJets {
        phi,
        ..generic_jets
    };

    // Smoothness check: the collision Taylor expansion requires analytic
    // collision limits (t(0) = φ''''(0)/3, etc.) which only exist when the
    // kernel is sufficiently smooth at the origin: 2(p+s) > d + 2j.
    // For the borderline case (2(p+s) == d+4), φ''''(0) diverges
    // logarithmically and the Taylor carrier cannot represent t(r) accurately.
    // In that regime, keep the generic-path values at r_eval = r_floor.
    let smoothness_order = 2 * (p_order + s_order);
    let collision_q_exists = smoothness_order > k_dim + 2;
    let collision_t_exists = smoothness_order > k_dim + 4;
    let collision_t_rr_exists = smoothness_order > k_dim + 6;

    if r <= collision_taylor_radius.max(r_floor) && collision_t_exists {
        // Tier 2+: full collision Taylor expansion using φ''(0), φ''''(0)/3,
        // and optionally φ⁽⁶⁾(0)/15.  Replaces the generic r_floor path for
        // all radial scalars in the near-origin region.
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        let analytic_t_collision =
            duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 3.0;
        let analytic_t_rr_collision = if collision_t_rr_exists {
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 15.0
        } else {
            // t_rr(0) does not exist as a finite limit for this smoothness
            // order, so the smooth-origin carrier must stop at the quadratic
            // term in t(r) and the quartic term in q(r), phi_r(r), phi_rr(r).
            0.0
        };
        let collision_jets = duchon_operator_jets_from_primary_core(
            duchon_collision_taylor_operator_core(
                r,
                analytic_phi_rr,
                analytic_t_collision,
                analytic_t_rr_collision,
            ),
            r,
            d,
        );
        out = DuchonRadialJets {
            phi: out.phi,
            ..collision_jets
        };
    } else if r < r_floor && collision_q_exists {
        // Tier 1: only lower-order collision identities exist.  φ''(0) is
        // finite but φ''''(0) diverges logarithmically at this smoothness
        // order.  Override phi_r, phi_rr, q, q_r, lap, lap_r with exact
        // values; leave t, t_r, t_rr, q_rr, lap_rr at their generic-path
        // values from r_eval = r_floor (best available for the divergent tier).
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        out.phi_r = analytic_phi_rr * r;
        out.phi_rr = analytic_phi_rr;
        out.q = analytic_phi_rr;
        out.q_r = 0.0;
        out.lap = d * analytic_phi_rr;
        out.lap_r = 0.0;
    }
    if !out.phi_r.is_finite()
        || !out.phi_rr.is_finite()
        || !out.q.is_finite()
        || !out.q_r.is_finite()
        || !out.q_rr.is_finite()
        || !out.lap.is_finite()
        || !out.lap_r.is_finite()
        || !out.lap_rr.is_finite()
        || !out.t.is_finite()
        || !out.t_r.is_finite()
        || !out.t_rr.is_finite()
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
    //
    // Representation note:
    //   When p > 0 the Duchon kernel is only conditionally positive definite, so
    //   the spatial kernel is canonical only up to polynomial additions. The
    //   formulas in this helper are therefore tied to the specific representative
    //   encoded by the partial-fraction construction and the collision rules used
    //   below. The operator penalties, exact psi derivatives, and center-collision
    //   limits all have to use that same representative or the resulting penalty
    //   geometry will drift across code paths.
    let delta = duchon_scaling_exponent(p_order, s_order, k_dim);
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs)?;
    let phi = jets.phi;
    let (phi_psi, phi_psi_psi) =
        scaled_log_kappa_derivatives(phi, jets.phi_r, jets.phi_rr, delta, r);
    if r > 1e-10 {
        #[cfg(test)]
        let (g_psi, g_psi_psi) = duchon_q_psi_triplet_from_jets(&jets, p_order, s_order, k_dim, r);
        #[cfg(test)]
        let (lap_psi, lap_psi_psi) =
            duchon_laplacian_psi_triplet_from_jets(&jets, p_order, s_order, k_dim, r);
        assert!(
            ((delta * phi + r * jets.phi_r) - phi_psi).abs() < 1e-7_f64.max(1e-7_f64 * phi.abs())
        );
        return Ok(DuchonRadialCore {
            phi: PsiTriplet {
                value: phi,
                psi: phi_psi,
                psi_psi: phi_psi_psi,
            },
            #[cfg(test)]
            gradient_ratio: PsiTriplet {
                value: jets.q,
                psi: g_psi,
                psi_psi: g_psi_psi,
            },
            #[cfg(test)]
            laplacian: PsiTriplet {
                value: jets.lap,
                psi: lap_psi,
                psi_psi: lap_psi_psi,
            },
        });
    }

    // Continuous center-collision extension for the scalar operator core:
    //   q(0; kappa) = phi_rr(0; kappa)
    //   L(0; kappa) = d * phi_rr(0; kappa).
    //
    // When 2(p+s) > d+2 (classical C^2 regime), phi_rr scales with exponent
    // delta + 2, so a closed-form shortcut is available.  In the general case
    // (including low-regularity settings where 2(p+s) <= d), the collision
    // values are computed directly from the assembled partial-fraction
    // expansion, which is equally correct and fully supported.
    #[cfg(test)]
    let (gradient_ratio, laplacian) = {
        let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        duchon_collision_operator_core_fromphi_rr(phi_rr, phi_rr_psi, phi_rr_psi_psi, k_dim)
    };
    Ok(DuchonRadialCore {
        phi: PsiTriplet {
            value: phi,
            psi: phi_psi,
            psi_psi: phi_psi_psi,
        },
        #[cfg(test)]
        gradient_ratio,
        #[cfg(test)]
        laplacian,
    })
}

fn duchonphi_rr_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(f64, f64, f64), BasisError> {
    // Center-collision rule
    // For a C^2 radial kernel one has
    //   lim_{r->0} phi_r(r)/r = phi_rr(0),
    //   lim_{r->0} Δphi(r)    = d * phi_rr(0).
    //
    // The general path assembles phi_rr and its psi derivatives by summing the
    // partial-fraction blocks directly.  When 2(p+s) > d+2 (classical C^2
    // regime), a closed-form scaling shortcut is available:
    //   phi_rr_psi     = (delta + 2) phi_rr
    //   phi_rr_psipsi  = (delta + 2)^2 phi_rr
    // and is used as an optimization.  Both paths produce consistent results;
    // the assembled path is the primary, fully supported computation.
    let mut phi_rr = KahanSum::default();
    let mut phi_rr_psi = KahanSum::default();
    let mut phi_rr_psi_psi = KahanSum::default();
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let alpha_m = duchon_coeff_exponents(p_order, s_order, m);
        let (g0, g1, g2) = duchon_polyharmonicsecond_collision_psi_triplet(length_scale, m, k_dim);
        phi_rr.add(a_m * g0);
        phi_rr_psi.add(alpha_m * a_m * g0 + a_m * g1);
        phi_rr_psi_psi.add(alpha_m * alpha_m * a_m * g0 + 2.0 * alpha_m * a_m * g1 + a_m * g2);
    }
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let beta_n = duchon_coeff_exponents(p_order, s_order, n);
        let (g0, g1, g2) = duchon_maternsecond_collision_psi_triplet(length_scale, n, k_dim)?;
        phi_rr.add(b_n * g0);
        phi_rr_psi.add(beta_n * b_n * g0 + b_n * g1);
        phi_rr_psi_psi.add(beta_n * beta_n * b_n * g0 + 2.0 * beta_n * b_n * g1 + b_n * g2);
    }
    let phi_rr = phi_rr.sum();
    let phi_rr_psi = phi_rr_psi.sum();
    let phi_rr_psi_psi = phi_rr_psi_psi.sum();
    if duchon_has_classicalsecond_order_origin(p_order, s_order, k_dim) {
        let scale = duchon_scaling_exponent(p_order, s_order, k_dim) + 2.0;
        return Ok((phi_rr, scale * phi_rr, scale * scale * phi_rr));
    }
    Ok((phi_rr, phi_rr_psi, phi_rr_psi_psi))
}

#[inline(always)]
fn falling_factorial(alpha: f64, order: usize) -> f64 {
    (0..order).fold(1.0, |acc, idx| acc * (alpha - idx as f64))
}

#[inline(always)]
fn falling_factorial_derivative(alpha: f64, order: usize) -> f64 {
    if order == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for omit in 0..order {
        let mut term = 1.0;
        for idx in 0..order {
            if idx != omit {
                term *= alpha - idx as f64;
            }
        }
        total += term;
    }
    total
}

#[inline(always)]
fn duchon_polyharmonic_radial_derivative_at_r(
    r: f64,
    m: usize,
    k_dim: usize,
    derivative_order: usize,
) -> f64 {
    let k_half = 0.5 * k_dim as f64;
    let alpha = (2_i64 * (m as i64) - (k_dim as i64)) as f64;
    let e = alpha - derivative_order as f64;
    let r_safe = r.max(1e-300);

    if k_dim % 2 == 0 && m >= (k_dim / 2) {
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        let falling = falling_factorial(alpha, derivative_order);
        let falling_derivative = falling_factorial_derivative(alpha, derivative_order);
        return c * r_safe.powf(e) * (falling * r_safe.ln() + falling_derivative);
    }

    let c = gamma_lanczos(k_half - m as f64)
        / (4.0_f64.powi(m as i32) * std::f64::consts::PI.powf(k_half) * gamma_lanczos(m as f64));
    c * falling_factorial(alpha, derivative_order) * r_safe.powf(e)
}

/// Euler-Mascheroni constant γ ≈ 0.5772.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Digamma function ψ(n) for positive integer n.
///
/// ψ(1) = −γ, ψ(n+1) = −γ + H_n where H_n = Σ_{j=1}^{n} 1/j.
#[inline(always)]
fn digamma_pos_int(n: usize) -> f64 {
    assert!(n >= 1);
    let mut h = 0.0_f64;
    for j in 1..n {
        h += 1.0 / j as f64;
    }
    -EULER_MASCHERONI + h
}

/// Extract the coefficient of r^{2j} (pure and log-r parts) from a single
/// Matérn partial-fraction block g_n(r) = c · r^ν · K_{|ν|}(κr), where
/// ν = n − d/2.
///
/// Returns `(pure_coeff, log_coeff)` such that the r^{2j} piece of g_n is
///   pure_coeff · r^{2j}  +  log_coeff · r^{2j} · ln(r).
///
/// For even d (integer ν) the expansion uses the DLMF 10.31.1 series for
/// K_n(z) at the origin, which involves digamma / harmonic-number terms.
///
/// For odd d (half-integer ν) the Bessel function is elementary; the Taylor
/// coefficients come from convolving a finite polynomial in 1/r with e^{−κr},
/// and there is no log-r contribution.
fn duchon_matern_block_taylor_r2j(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> (f64, f64) {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    // Normalization constant for the Matérn block.
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    if k_dim % 2 == 0 {
        // Integer ν.
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        duchon_matern_block_taylor_r2j_integer_nu(kappa, c, nu_int, j)
    } else {
        // Half-integer ν.
        duchon_matern_block_taylor_r2j_half_integer_nu(kappa, c, nu, j)
    }
}

/// Taylor r^{2j} coefficients for integer-ν Matérn block.
///
/// Uses the K_μ(z) expansion for integer μ = |ν| ≥ 0 (A&S 9.6.11 / DLMF 10.31.1):
///
///   K_μ(z) = (−1)^{μ+1} I_μ(z) ln(z/2)
///          + ½ Σ_{k=0}^{μ−1} (−1)^k (μ−k−1)!/k! · (z/2)^{2k−μ}   [singular]
///          + (−1)^μ · ½ Σ_{k≥0} (z/2)^{μ+2k}/(k!(μ+k)!)
///                              · [ψ(k+1)+ψ(μ+k+1)]                  [regular]
///
/// Multiplied by r^ν, the r^{2j} coefficient is assembled from the singular
/// and/or regular+log series depending on the sign and magnitude of ν.
fn duchon_matern_block_taylor_r2j_integer_nu(
    kappa: f64,
    c: f64,
    nu_int: i64,
    j: usize,
) -> (f64, f64) {
    let mu = nu_int.unsigned_abs() as usize; // |ν|

    // Helper: compute (κ/2)^p for integer p.
    let kappa_half = 0.5 * kappa;

    if nu_int >= 0 {
        let nu = nu_int as usize;
        // Two potential sources for the r^{2j} coefficient:
        //
        // 1) Singular sum:  contributes when j ≤ ν−1 (the k=j term gives r^{2j}).
        // 2) Regular+log sum: contributes when 2ν+2k = 2j, i.e. k = j−ν ≥ 0.
        let mut pure = 0.0;
        let mut log_part = 0.0;

        // Source 1: singular sum at k = j.
        if j < nu {
            // (1/2) · (−1)^j · (ν−j−1)!/j! · (κ/2)^{2j−ν}
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let coeff = sign * gamma_lanczos((nu - j) as f64) / gamma_lanczos((j + 1) as f64)
                * kappa_half.powi(2 * j as i32 - nu as i32)
                * 0.5;
            pure += coeff;
        }

        // Source 2: regular+log sum at k = j − ν.
        if j >= nu {
            let k = j - nu;
            let inv_fac =
                1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((nu + k + 1) as f64));
            let kp = kappa_half.powi(2 * k as i32 + nu as i32);
            let sign_mu = if mu % 2 == 0 { 1.0 } else { -1.0 }; // (−1)^μ

            // Log coefficient: (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!)
            log_part += -sign_mu * kp * inv_fac;

            // Pure coefficient from the log series (ln(κ/2) piece):
            //   (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!) · ln(κ/2)
            // Plus the digamma series:
            //   (−1)^μ · ½ · (κ/2)^{ν+2k} / (k!(ν+k)!) · [ψ(k+1)+ψ(ν+k+1)]
            let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(nu + k + 1);
            pure += -sign_mu * kp * inv_fac * kappa_half.ln();
            pure += sign_mu * 0.5 * kp * inv_fac * psi_sum;
        }

        (c * pure, c * log_part)
    } else {
        // ν < 0: mu = |ν| > 0.
        // Singular sum gives powers r^{2ν}, ..., r^{−2} (all negative).
        // Regular+log sum gives r^0, r^2, r^4, ... at k = j.
        let k = j;
        let inv_fac = 1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((mu + k + 1) as f64));
        let kp = kappa_half.powi(mu as i32 + 2 * k as i32);
        let sign_mu = if mu % 2 == 0 { 1.0 } else { -1.0 };

        // Log coefficient: (−1)^{μ+1} · (κ/2)^{μ+2k} / (k!(μ+k)!)
        let log_part = -sign_mu * kp * inv_fac;

        // Pure coefficient: log-series ln(κ/2) piece + digamma piece.
        let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(mu + k + 1);
        let pure =
            -sign_mu * kp * inv_fac * kappa_half.ln() + sign_mu * 0.5 * kp * inv_fac * psi_sum;

        (c * pure, c * log_part)
    }
}

/// Taylor r^{2j} coefficients for half-integer-ν Matérn block.
///
/// For half-integer |ν| = l + ½, K_{l+½}(z) is elementary:
///   K_{l+½}(z) = √(π/(2z)) · e^{−z} · Σ_{i=0}^{l} C_i · (2z)^{−i}
/// where C_i = (l+i)! / (i! · (l−i)!).
///
/// The product r^ν · K_{|ν|}(κr) expands as an explicit polynomial in r
/// (including possible negative powers) times e^{−κr}.  The r^{2j} Taylor
/// coefficient is obtained by convolving with the exponential series
/// e^{−κr} = Σ_q (−κ)^q r^q / q!.  There is never a log-r contribution.
fn duchon_matern_block_taylor_r2j_half_integer_nu(
    kappa: f64,
    c: f64,
    nu: f64,
    j: usize,
) -> (f64, f64) {
    let nu_abs = nu.abs();
    let l = (2.0 * nu_abs - 1.0).round().max(0.0) as usize;
    // Compute the polynomial coefficients C_i / (2κ)^i for each r-power.
    //
    // r^ν · K_{l+½}(κr) = √(π/(2κ)) · e^{−κr} · Σ_{i=0}^{l} C_i (2κ)^{−i} r^{ν−½−i}
    //
    // (since K_{l+½}(z) = √(π/(2z)) e^{−z} Σ C_i (2z)^{−i}, multiplying by
    // r^ν gives r^{ν−½} from the √(π/(2κr)) factor, then each (2κr)^{−i}
    // contributes r^{−i}.)
    let prefactor = (std::f64::consts::PI / (2.0 * kappa)).sqrt();

    // Polynomial term i has r-power = ν − 0.5 − i.  We need to convolve
    // each monomial with e^{−κr} = Σ_q (−κ)^q r^q / q! and extract the
    // r^{2j} coefficient.
    //
    // For monomial r^p (p = ν−½−i) times e^{−κr}: the r^{2j} coefficient is
    //   (−κ)^{2j−p} / (2j−p)!   when 2j−p is a non-negative integer.
    let target = 2 * j;
    let mut pure = 0.0;

    for i in 0..=l {
        let c_i = gamma_lanczos((l + i + 1) as f64)
            / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
        let inv_2kappa_i = (2.0 * kappa).powi(-(i as i32));

        // r-power of this polynomial term.
        let p_f64 = nu - 0.5 - i as f64;
        let p_round = p_f64.round() as i64;
        if (p_f64 - p_round as f64).abs() > 1e-12 {
            // Not integer/half-integer aligned — should not happen for half-integer ν.
            continue;
        }
        let q_needed = target as i64 - p_round;
        if q_needed < 0 {
            continue;
        }
        let q = q_needed as usize;
        let exp_coeff = (-kappa).powi(q as i32) / gamma_lanczos((q + 1) as f64);
        pure += c_i * inv_2kappa_i * exp_coeff;
    }

    (c * prefactor * pure, 0.0) // No log contribution for half-integer ν.
}

/// Extract the r^{2j} Taylor coefficient from a polyharmonic block Φ_m(r).
///
/// Non-log case (d odd, or d even with m < d/2): Φ_m = c · r^α with α = 2m − d.
///   Only contributes when α = 2j exactly: pure_coeff = c, log_coeff = 0.
///
/// Log case (d even, m ≥ d/2): Φ_m = c · r^α · ln(r).
///   Only contributes when α = 2j: pure_coeff = 0, log_coeff = c.
fn duchon_polyharmonic_block_taylor_r2j(m: usize, k_dim: usize, j: usize) -> (f64, f64) {
    let k_half = 0.5 * k_dim as f64;
    let alpha = 2 * m as i64 - k_dim as i64;

    if alpha != 2 * j as i64 {
        return (0.0, 0.0);
    }

    // α = 2j: compute the normalization constant.
    if k_dim % 2 == 0 && m >= k_dim / 2 {
        // Log case: Φ_m = c · r^α · ln(r).
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        (0.0, c)
    } else {
        // Non-log case: Φ_m = c · r^α.
        let c = gamma_lanczos(k_half - m as f64)
            / (4.0_f64.powi(m as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64));
        (c, 0.0)
    }
}

/// Compute the even-order radial derivative φ^{(2j)}(0) from analytic Taylor
/// coefficients of the partial-fraction blocks.
///
/// For a C^{2j} radial kernel with Taylor expansion φ(r) = Σ_k a_{2k} r^{2k},
/// φ^{(2j)}(0) = (2j)! · a_{2j}.  Each partial-fraction block (polyharmonic
/// and Matérn) has a computable r^{2j} Taylor coefficient (both pure and
/// ln(r) parts).  The ln(r) contributions cancel across blocks whenever the
/// kernel is sufficiently smooth; the pure coefficients sum to give a_{2j}.
///
/// Convergence conditions (kernel is C^{2j} at the origin):
///   2(p + s) > d + 2j.
///
/// When this condition fails (borderline or insufficient smoothness), the
/// function falls back to evaluating the derivative at a small floor radius.
fn duchon_phi_even_derivative_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
    j: usize,
) -> Result<f64, BasisError> {
    let smoothness_order = 2 * (p_order + s_order);
    let required = k_dim + 2 * j;

    if smoothness_order <= required {
        // Kernel is not C^{2j} at the origin (borderline or worse).
        // Fall back to evaluating the derivative at the standard small-r floor.
        // Note: for the borderline case the true limit is logarithmically
        // divergent, so any finite floor gives a regularized approximation.
        let r_eff = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
        let kappa = 1.0 / length_scale.max(1e-300);
        let mut result = KahanSum::default();
        let deriv_order = 2 * j;
        for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
            if a_m == 0.0 {
                continue;
            }
            result.add(
                a_m * duchon_polyharmonic_radial_derivative_at_r(r_eff, m, k_dim, deriv_order),
            );
        }
        for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
            if b_n == 0.0 {
                continue;
            }
            result.add(
                b_n * duchon_matern_block_radial_derivative(r_eff, kappa, n, k_dim, deriv_order)?,
            );
        }
        return Ok(result.sum());
    }

    // Analytic path: extract per-block Taylor r^{2j} coefficients and sum.
    let kappa = 1.0 / length_scale.max(1e-300);
    let mut total_pure = KahanSum::default();
    let mut total_log = KahanSum::default();

    // Polyharmonic blocks.
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        total_pure.add(a_m * pure);
        total_log.add(a_m * log);
    }

    // Matérn blocks.
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (pure, log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, j);
        total_pure.add(b_n * pure);
        total_log.add(b_n * log);
    }
    let total_pure = total_pure.sum();
    let total_log = total_log.sum();

    // The ln(r) coefficients should cancel to zero (guaranteed by the PFD
    // identity when 2(p+s) > d+2j).  Check this as a sanity guard.
    if total_log.abs() > 1e-8 * total_pure.abs().max(1e-30) {
        return Err(BasisError::InvalidInput(format!(
            "Duchon Taylor a_{} log-coefficient did not cancel: log={total_log:.6e}, pure={total_pure:.6e}; \
             p={p_order}, s={s_order}, d={k_dim}",
            2 * j
        )));
    }

    // φ^{(2j)}(0) = (2j)! · a_{2j}
    let factorial_2j = gamma_lanczos((2 * j + 1) as f64);
    Ok(factorial_2j * total_pure)
}

/// Assemble φ''''(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + ...,
/// we have φ''''(0) = 24 a₄.  This is used to compute the collision limit
/// t(0) = φ''''(0) / 3, where t = R²φ = (φ'' - q) / r².
///
/// Each partial-fraction block (polyharmonic and Matérn) has a known Taylor
/// expansion around r = 0; the r⁴ coefficient a₄ is extracted from the series
/// and summed.  This avoids the catastrophic cancellation that occurs when
/// evaluating divergent block derivatives at a small floor radius.
fn duchon_phi_rrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 2)
}

/// Assemble φ⁽⁶⁾(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + a₆r⁶ + ...,
/// we have φ⁽⁶⁾(0) = 720 a₆. This gives the collision limit
///   t_rr(0) = φ⁽⁶⁾(0) / 15
/// for t = R²φ.
///
/// Like [`duchon_phi_rrrr_collision`], this extracts per-block Taylor
/// coefficients analytically rather than evaluating divergent derivatives at
/// a small floor radius.
fn duchon_phi_rrrrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 3)
}

fn build_duchon_design_psi_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
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
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let p_padded = z_kernel.ncols() + poly_cols;
    if let Some(zf) = identifiability_transform
        && p_padded != zf.nrows()
    {
        return Err(BasisError::DimensionMismatch(format!(
            "Duchon identifiability transform mismatch in design derivatives: local cols={}, transform rows={}",
            p_padded,
            zf.nrows()
        )));
    }
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_padded);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        spec.aniso_log_scales.as_deref(),
        p_final,
        Some(z_kernel),
        identifiability_transform.cloned(),
        poly_cols,
        RadialScalarKind::Duchon {
            length_scale,
            p_order,
            s_order,
            dim: data.ncols(),
            coeffs,
        },
        duchon_scaling_exponent(p_order, s_order, data.ncols()),
    )
}

pub fn build_duchon_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let (centers, identifiability_transform) =
        prepare_duchon_derivative_contextwithworkspace(data, spec, workspace)?;
    let design_derivatives = build_duchon_design_psi_derivativeswithworkspace(
        data,
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let (penalties_derivative, _) = build_duchon_operator_penalty_psi_derivatives(
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    Ok(BasisPsiDerivativeResult {
        design_derivative: design_derivatives.design_first,
        penalties_derivative,
        implicit_operator: design_derivatives.implicit_operator,
    })
}

pub fn build_duchon_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let (centers, identifiability_transform) =
        prepare_duchon_derivative_contextwithworkspace(data, spec, workspace)?;
    let design_derivatives = build_duchon_design_psi_derivativeswithworkspace(
        data,
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let (_, penaltiessecond_derivative) = build_duchon_operator_penalty_psi_derivatives(
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    Ok(BasisPsiSecondDerivativeResult {
        designsecond_derivative: design_derivatives.design_second_diag,
        penaltiessecond_derivative,
        implicit_operator: design_derivatives.implicit_operator,
    })
}

/// Creates a Duchon radial basis whose kernel is derived from
///   P(w) = ||w||^(2p) * (kappa^2 + ||w||^2)^s
/// using:
/// - integer-parameter partial-fraction decomposition in spectral space,
/// - finite spatial kernel sum of polyharmonic + Matérn blocks,
/// - explicit polynomial null-space block determined by `nullspace_order`,
/// - side-constraint projection `P(centers)^T alpha = 0` for the selected
///   null-space degree.
///
/// The returned penalties are the canonical triple operator regularization
/// matrices built from collocation images of the final function, not the
/// native Fourier-space Duchon seminorm.
///
/// API mapping:
/// - `p` is the Duchon smoothness order `m`, determined by `nullspace_order`:
///   - `Zero`   -> m = 1 (constants)
///   - `Linear` -> m = 2 (constants + linear terms)
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
    create_duchon_spline_basiswithworkspace(
        data,
        centers,
        length_scale,
        power,
        nullspace_order,
        &mut workspace,
    )
}

pub fn create_duchon_spline_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonSplineBasis, BasisError> {
    let design = build_duchon_basis_designwithworkspace(
        data,
        centers,
        length_scale,
        power,
        nullspace_order,
        None, // create_duchon_spline_basis does not support anisotropy
        workspace,
    )?;
    // Pick up the effective order from the design (which may have been
    // auto-degraded to Zero when centers were insufficient) so the penalty
    // Gram matrix is built with the same nullspace as the design.
    let nullspace_order = design.nullspace_order;
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    let d = centers.ncols();
    let k = centers.nrows();
    let coeffs = if let Some(ls) = length_scale {
        Some(duchon_partial_fraction_coeffs(
            p_order,
            s_order,
            1.0 / ls.max(1e-300),
        ))
    } else {
        None
    };
    let center_center_r = compute_center_center_distances(centers);
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
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
    let omega_constrained = {
        let zt_k = fast_atb(&z, &center_kernel);
        fast_ab(&zt_k, &z)
    };
    let total_cols = design.basis.ncols();
    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_kernel
        .slice_mut(s![0..design.num_kernel_basis, 0..design.num_kernel_basis])
        .assign(&omega_constrained);
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_kernel)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));
    Ok(DuchonSplineBasis {
        basis: design.basis,
        penalty_kernel,
        penalty_ridge,
        num_kernel_basis: design.num_kernel_basis,
        num_polynomial_basis: design.num_polynomial_basis,
        dimension: design.dimension,
        nullspace_order: design.nullspace_order,
    })
}

fn build_duchon_basis_designwithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonBasisDesign, BasisError> {
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
    // Auto-degrade the null-space order to Zero when centers are insufficient
    // to span the requested polynomial block; emits a warning inside the helper.
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = power;
    validate_duchon_kernel_orders(length_scale, p_order, s_order, d)?;

    let poly_block = polynomial_block_from_order(data, nullspace_order);
    // Z spans null(Q^T), where Q contains polynomial side conditions at centers.
    // Reparameterizing alpha = Z gamma enforces conditional-PD constraints once
    // and yields free-parameter penalty gamma^T (Z^T K_CC Z) gamma.
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;

    let coeffs = if let Some(ls) = length_scale {
        Some(duchon_partial_fraction_coeffs(
            p_order,
            s_order,
            1.0 / ls.max(1e-300),
        ))
    } else {
        None
    };

    // Practical safe operating range (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min]
    // where r_min/r_max are pairwise center distance extrema. Under
    // anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    // We keep user-provided κ but emit a warning outside this regime.
    let warn_bounds = match (length_scale, aniso_log_scales) {
        (Some(_), Some(eta)) => {
            let y_centers = points_in_aniso_y_space(centers, eta);
            pairwise_distance_bounds(y_centers.view())
        }
        (Some(_), None) => pairwise_distance_bounds(centers),
        (None, _) => None,
    };
    if let (Some(length_scale), Some((r_min, r_max))) = (length_scale, warn_bounds) {
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

    let kernel_cols = z.ncols();
    let poly_cols = poly_block.ncols();
    let total_cols = kernel_cols + poly_cols;

    // Pre-compute polyharmonic coefficient for the pure Duchon case (no length_scale).
    // This avoids 2 gamma_lanczos calls per kernel evaluation (n × k total).
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(p_order, s_order),
            d,
        ))
    } else {
        None
    };

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    // Process rows in chunks to amortize thread-local allocation across many rows.
    // Use larger chunks (1024) for better cache utilization at biobank scale.
    let chunk_size = 1024.min(n);
    let basis_result: Result<(), BasisError> = basis
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .try_for_each(|(ci, mut chunk)| {
            let mut kernel_row = vec![0.0; k];
            // Pre-allocate row/center buffers once per thread to avoid 200M+
            // heap allocations in the anisotropic distance path.
            let mut data_row_buf = vec![0.0; d];
            let mut center_buf = vec![0.0; d];
            let chunk_start = ci * chunk_size;
            for local_i in 0..chunk.nrows() {
                let i = chunk_start + local_i;
                // Copy data row once; reuse across all k centers.
                if aniso_log_scales.is_some() {
                    for a in 0..d {
                        data_row_buf[a] = data[[i, a]];
                    }
                }
                for j in 0..k {
                    let r = if let Some(eta) = aniso_log_scales {
                        for a in 0..d {
                            center_buf[a] = centers[[j, a]];
                        }
                        aniso_distance(&data_row_buf, &center_buf, eta)
                    } else {
                        stable_euclidean_norm(
                            (0..d).map(|axis| data[[i, axis]] - centers[[j, axis]]),
                        )
                    };
                    kernel_row[j] = if let Some(ref ppc) = pure_poly_coeff {
                        // Pure Duchon: use precomputed coefficient, skip gamma calls.
                        ppc.eval(r)
                    } else {
                        duchon_matern_kernel_general_from_distance(
                            r,
                            length_scale,
                            p_order,
                            s_order,
                            d,
                            coeffs.as_ref(),
                        )?
                    };
                }
                // Write basis row = kernel_row^T × Z using scatter-accumulate
                // pattern: for each knot j with nonzero kernel, add its
                // contribution to all columns at once. This is more cache-
                // friendly than the column-by-column gather pattern since
                // Z rows are contiguous in memory.
                let mut row = chunk.row_mut(local_i);
                row.slice_mut(s![..kernel_cols]).fill(0.0);
                for j in 0..k {
                    let kv = kernel_row[j];
                    if kv != 0.0 {
                        let z_row = z.row(j);
                        for col in 0..kernel_cols {
                            row[col] += kv * z_row[col];
                        }
                    }
                }
            }
            Ok(())
        });
    basis_result?;
    if poly_cols > 0 {
        basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);
    }

    Ok(DuchonBasisDesign {
        basis,
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
    build_duchon_basiswithworkspace(data, spec, &mut workspace)
}

pub fn build_duchon_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_biobank_cap(data.nrows(), data.ncols(), centers.view());
    // Auto-degrade the requested null-space order to Zero when the selected
    // centers cannot span the requested polynomial block. Every downstream
    // consumer of `spec.nullspace_order` in this function MUST use the
    // effective order, otherwise the penalty/nullspace is built with a
    // different order than the basis.
    let effective_nullspace_order =
        duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    validate_duchon_collocation_orders(spec.length_scale, p_order, spec.power, data.ncols())?;
    // Initialize anisotropy contrasts from knot cloud geometry when the caller
    // enabled scale-dimensions but left η at the zero default.
    let aniso = maybe_initialize_aniso_contrasts(centers.view(), spec.aniso_log_scales.as_deref());
    let kernel_transform = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let base_cols = kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
    let (design, identifiability_transform) = if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "Duchon basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            base_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
        let d = data.ncols();
        let shared_data = shared_owned_data_matrix(data, &mut workspace.cache);
        let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
        let s_order = spec.power;
        let length_scale = spec.length_scale;
        let coeffs = length_scale
            .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ls.max(1e-300)));
        let pure_poly_coeff = if length_scale.is_none() {
            Some(PolyharmonicBlockCoeff::new(
                pure_duchon_block_order(p_order, s_order),
                d,
            ))
        } else {
            None
        };
        let poly_block = polynomial_block_from_order(data, effective_nullspace_order);
        let base_design = if let Some(eta) = aniso.as_ref() {
            let metric_weights = eta.iter().map(|&v| (2.0 * v).exp()).collect::<Vec<_>>();
            let coeffs = coeffs.clone();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let mut q = 0.0f64;
                for axis in 0..data_row.len() {
                    let delta = data_row[axis] - center_row[axis];
                    q += metric_weights[axis] * delta * delta;
                }
                let r = q.sqrt();
                if let Some(ppc) = pure_poly_coeff {
                    ppc.eval(r)
                } else {
                    duchon_matern_kernel_general_from_distance(
                        r,
                        length_scale,
                        p_order,
                        s_order,
                        d,
                        coeffs.as_ref(),
                    )
                    .expect("validated Duchon inputs should not fail")
                }
            };
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data.clone(),
                Arc::new(centers.clone()),
                kernel,
                Some(Arc::new(kernel_transform.clone())),
                Some(Arc::new(poly_block.clone())),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)))
        } else {
            let coeffs = coeffs.clone();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let r = stable_euclidean_norm((0..d).map(|axis| data_row[axis] - center_row[axis]));
                if let Some(ppc) = pure_poly_coeff {
                    ppc.eval(r)
                } else {
                    duchon_matern_kernel_general_from_distance(
                        r,
                        length_scale,
                        p_order,
                        s_order,
                        d,
                        coeffs.as_ref(),
                    )
                    .expect("validated Duchon inputs should not fail")
                }
            };
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data,
                Arc::new(centers.clone()),
                kernel,
                Some(Arc::new(kernel_transform.clone())),
                Some(Arc::new(poly_block)),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)))
        };
        let identifiability_transform = spatial_identifiability_transform_from_design_matrix(
            data,
            &base_design,
            &spec.identifiability,
            "Duchon",
        )?;
        let design = if let Some(transform) = identifiability_transform.as_ref() {
            wrap_dense_design_with_transform(base_design, transform, "Duchon")?
        } else {
            base_design
        };
        (design, identifiability_transform)
    } else {
        let d = build_duchon_basis_designwithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.power,
            effective_nullspace_order,
            aniso.as_deref(),
            workspace,
        )?;
        let basis = d.basis;
        let identifiability_transform = spatial_identifiability_transform_from_design(
            data,
            basis.view(),
            &spec.identifiability,
            "Duchon",
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(&basis, z)))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(basis))
        };
        (design, identifiability_transform)
    };
    let ops = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        None,
        spec.length_scale,
        spec.power,
        effective_nullspace_order,
        aniso.as_deref(),
        identifiability_transform.as_ref().map(|z| z.view()),
        workspace,
    )?;
    // Duchon radial basis with triple operator regularization. These are
    // collocation operator penalties on the fitted function, not the native
    // Fourier-space Duchon seminorm.
    let candidates = operator_penalty_candidates_from_collocation(
        &ops.d0,
        &ops.d1,
        &ops.d2,
        &spec.operator_penalties,
    );
    let (penalties, nullspace_dims, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            power: spec.power,
            nullspace_order: effective_nullspace_order,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: aniso,
        },
        kronecker_factored: None,
    })
}

fn polynomial_block_from_order(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    match order {
        DuchonNullspaceOrder::Zero => Array2::<f64>::ones((n, 1)),
        DuchonNullspaceOrder::Linear => {
            let mut poly = Array2::<f64>::zeros((n, d + 1));
            poly.column_mut(0).fill(1.0);
            for c in 0..d {
                poly.column_mut(c + 1).assign(&points.column(c));
            }
            poly
        }
        DuchonNullspaceOrder::Degree(degree) => monomial_basis_block(points, degree),
    }
}

fn monomial_exponents(dimension: usize, max_total_degree: usize) -> Vec<Vec<usize>> {
    fn recurse(
        axis: usize,
        remaining_degree: usize,
        current: &mut [usize],
        out: &mut Vec<Vec<usize>>,
    ) {
        if axis + 1 == current.len() {
            current[axis] = remaining_degree;
            out.push(current.to_vec());
            return;
        }
        for exponent in (0..=remaining_degree).rev() {
            current[axis] = exponent;
            recurse(axis + 1, remaining_degree - exponent, current, out);
        }
    }

    if dimension == 0 {
        return vec![Vec::new()];
    }

    let mut out = Vec::new();
    let mut current = vec![0usize; dimension];
    for total_degree in 0..=max_total_degree {
        recurse(0, total_degree, &mut current, &mut out);
    }
    out
}

pub fn duchon_nullspace_dimension(dimension: usize, max_total_degree: usize) -> usize {
    monomial_exponents(dimension, max_total_degree).len()
}

fn monomial_basis_block(points: ArrayView2<'_, f64>, max_total_degree: usize) -> Array2<f64> {
    let n = points.nrows();
    let exponents = monomial_exponents(points.ncols(), max_total_degree);
    let mut block = Array2::<f64>::zeros((n, exponents.len()));
    for (col, exponents) in exponents.iter().enumerate() {
        for row in 0..n {
            let mut value = 1.0;
            for axis in 0..points.ncols() {
                let exponent = exponents[axis];
                if exponent != 0 {
                    value *= points[[row, axis]].powi(exponent as i32);
                }
            }
            block[[row, col]] = value;
        }
    }
    block
}

fn monomial_hessian_operator_block(
    points: ArrayView2<'_, f64>,
    max_total_degree: usize,
) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let exponents = monomial_exponents(d, max_total_degree);
    let mut block = Array2::<f64>::zeros((n * d * d, exponents.len()));
    for (col, exponents) in exponents.iter().enumerate() {
        for row in 0..n {
            for axis_a in 0..d {
                for axis_b in 0..d {
                    let ea = exponents[axis_a];
                    let eb = exponents[axis_b];
                    let coeff = if axis_a == axis_b {
                        if ea < 2 {
                            continue;
                        }
                        (ea * (ea - 1)) as f64
                    } else {
                        if ea == 0 || eb == 0 {
                            continue;
                        }
                        (ea * eb) as f64
                    };
                    let mut value = coeff;
                    for axis in 0..d {
                        let mut exponent = exponents[axis];
                        if axis == axis_a {
                            exponent -= 1;
                        }
                        if axis == axis_b {
                            exponent -= 1;
                        }
                        if exponent != 0 {
                            value *= points[[row, axis]].powi(exponent as i32);
                        }
                    }
                    block[[(row * d + axis_a) * d + axis_b, col]] = value;
                }
            }
        }
    }
    block
}

fn polynomial_hessian_operator_block(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    match order {
        DuchonNullspaceOrder::Zero | DuchonNullspaceOrder::Linear => Array2::<f64>::zeros((
            points.nrows() * points.ncols() * points.ncols(),
            polynomial_block_from_order(points, order).ncols(),
        )),
        DuchonNullspaceOrder::Degree(degree) => monomial_hessian_operator_block(points, degree),
    }
}

#[inline(always)]
fn thin_plate_polynomial_degree(dimension: usize) -> usize {
    thin_plate_penalty_order(dimension).saturating_sub(1)
}

fn thin_plate_polynomial_block(points: ArrayView2<'_, f64>) -> Array2<f64> {
    monomial_basis_block(points, thin_plate_polynomial_degree(points.ncols()))
}

fn thin_plate_polynomial_basis_dimension(dimension: usize) -> usize {
    monomial_exponents(dimension, thin_plate_polynomial_degree(dimension)).len()
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
fn thin_plate_kernel_from_dist2(dist2: f64, dimension: usize) -> Result<f64, BasisError> {
    if !dist2.is_finite() || dist2 < 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if dist2 == 0.0 {
        return Ok(0.0);
    }
    match dimension {
        // For d≤3, the minimum penalty order m=2 (biharmonic) suffices.
        // Hand-optimized closed forms avoid the overhead of the general evaluator.
        //   d=1:  r^3
        //   d=2:  r^2 log(r)
        //   d=3: -r
        1 => Ok(dist2 * dist2.sqrt()),
        2 => Ok(0.5 * dist2 * dist2.ln()),
        3 => Ok(-dist2.sqrt()),
        _ => {
            // General case: choose the smallest penalty order m with 2m > d,
            // i.e. m = floor(d/2) + 1, and evaluate via the Duchon polyharmonic
            // kernel which handles arbitrary (m, d) combinations.
            let m = dimension / 2 + 1;
            let r = dist2.sqrt();
            Ok(polyharmonic_kernel(r, m, dimension))
        }
    }
}

#[inline(always)]
fn thin_plate_penalty_order(dimension: usize) -> usize {
    match dimension {
        1..=3 => 2,
        _ => dimension / 2 + 1,
    }
}

#[inline(always)]
fn thin_plate_kernel_triplet_from_scaled_distance(
    scaled_distance: f64,
    dimension: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !scaled_distance.is_finite() || scaled_distance < 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate scaled distance must be finite and non-negative".to_string(),
        ));
    }
    if scaled_distance == 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }

    match dimension {
        1 => {
            let value = scaled_distance.powi(3);
            let first = 3.0 * scaled_distance.powi(2);
            let second = 6.0 * scaled_distance;
            Ok((value, first, second))
        }
        2 => {
            let log_r = scaled_distance.max(1e-300).ln();
            let value = scaled_distance.powi(2) * log_r;
            let first = 2.0 * scaled_distance * log_r + scaled_distance;
            let second = 2.0 * log_r + 3.0;
            Ok((value, first, second))
        }
        3 => Ok((-scaled_distance, -1.0, 0.0)),
        _ => polyharmonic_kernel_triplet(
            scaled_distance,
            thin_plate_penalty_order(dimension),
            dimension,
        ),
    }
}

#[inline(always)]
fn thin_plate_kernel_psi_triplet_from_distance(
    distance: f64,
    length_scale: f64,
    dimension: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate length_scale must be finite and positive".to_string(),
        ));
    }

    // ThinPlate psi-derivative convention:
    // the optimizer uses psi = log(kappa) = -log(length_scale), so the scaled
    // radial argument is
    //   r(psi) = ||x - c|| / length_scale = ||x - c|| * exp(psi).
    //
    // Therefore
    //   dr/dpsi     = r
    //   d²r/dpsi²   = r
    //
    // and for any TPS radial kernel phi(r),
    //   d phi / dpsi       = phi_r(r) * r
    //   d²phi / dpsi²      = phi_rr(r) * r² + phi_r(r) * r.
    //
    // This is exactly the chain rule requested by the math spec, translated to
    // the code's stored inverse-length-scale parameterization.
    let scaled_distance = distance / length_scale;
    let (value, radial_first, radial_second) =
        thin_plate_kernel_triplet_from_scaled_distance(scaled_distance, dimension)?;
    let psi = radial_first * scaled_distance;
    let psi_psi = radial_second * scaled_distance * scaled_distance + psi;
    Ok((value, psi, psi_psi))
}

/// Creates a thin-plate regression spline basis from data and knot locations.
///
/// # Arguments
/// * `data` - `n x d` matrix of evaluation points
/// * `knots` - `k x d` matrix of knot locations
///
/// # Returns
/// `ThinPlateSplineBasis` containing:
/// - `basis`: `n x (k_c + M)` matrix (`[K_c | P]`) where `M` is the TPS
///   polynomial null-space dimension for the selected ambient dimension
/// - `penalty_bending`: constrained TPS curvature penalty
/// - `penalty_ridge`: identity penalty for null-space shrinkage
pub fn create_thin_plate_spline_basis(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
) -> Result<ThinPlateSplineBasis, BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basiswithworkspace(data, knots, &mut workspace)
}

pub fn create_thin_plate_spline_basiswithworkspace(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<ThinPlateSplineBasis, BasisError> {
    create_thin_plate_spline_basis_scaledwithworkspace(data, knots, 1.0, workspace)
}

fn create_thin_plate_spline_basis_scaledwithworkspace(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
    length_scale: f64,
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
    let poly_cols = thin_plate_polynomial_basis_dimension(d);
    if k < poly_cols {
        return Err(BasisError::InvalidInput(format!(
            "thin-plate spline requires at least {} knots to span the degree-{} polynomial null space in dimension {}; got {}",
            poly_cols,
            thin_plate_polynomial_degree(d),
            d,
            k
        )));
    }
    if data.iter().any(|v| !v.is_finite()) || knots.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "thin-plate spline requires finite data and knot values".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "thin-plate length_scale must be finite and positive".to_string(),
        ));
    }

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
                row[j] = thin_plate_kernel_from_dist2(dist2 / (length_scale * length_scale), d)?;
            }
            Ok(())
        });
    kernel_result?;

    // P block: all TPS null-space monomials of total degree < m.
    let poly_block = thin_plate_polynomial_block(data);

    // Omega block on knots
    let mut omega = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = knots[[i, c]] - knots[[j, c]];
                dist2 += delta * delta;
            }
            let kij = thin_plate_kernel_from_dist2(dist2 / (length_scale * length_scale), d)?;
            omega[[i, j]] = kij;
            omega[[j, i]] = kij;
        }
    }

    // Enforce TPS side-constraint P(knots)^T α = 0 by projecting onto
    // the nullspace of P(knots)^T.
    let z = thin_plate_kernel_constraint_nullspace(knots, &mut workspace.cache)?;
    let kernel_constrained = fast_ab(&kernel_block, &z);
    let omega_constrained = {
        let zt_o = fast_atb(&z, &omega);
        symmetrize_penalty(&fast_ab(&zt_o, &z))
    };
    let omega_psd = validate_psd_penalty(
        &omega_constrained,
        &format!("thin_plate bending penalty (dimension={d})"),
        "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
    )?;
    assert!(omega_psd.min_eigenvalue >= -omega_psd.tolerance);
    assert!(omega_psd.max_abs_eigenvalue.is_finite());
    assert!(omega_psd.effective_rank <= omega_constrained.nrows());

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

fn active_thin_plate_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected ThinPlate penalty source in psi-derivative path: {other:?}"
            ))),
        })
        .collect()
}

fn build_thin_plate_design_psi_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let z_kernel = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let n = data.nrows();
    let k = centers.nrows();
    let kernel_cols = z_kernel.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(data.ncols());
    let total_cols = kernel_cols + poly_cols;
    let mut out_psi = Array2::<f64>::zeros((n, total_cols));
    let mut out_psi_psi = Array2::<f64>::zeros((n, total_cols));

    // Exact ThinPlate design derivatives in the stored psi = log(kappa)
    // coordinates, with kappa = 1 / length_scale.
    //
    // For each design kernel entry
    //   K_ij(psi) = phi(r_ij(psi)),
    //   r_ij(psi) = ||x_i - c_j|| / length_scale = ||x_i - c_j|| exp(psi),
    //
    // the chain rule gives
    //   r_ij,psi     = r_ij
    //   r_ij,psipsi  = r_ij
    //   K_ij,psi     = phi_r(r_ij) * r_ij
    //   K_ij,psipsi  = phi_rr(r_ij) * r_ij^2 + phi_r(r_ij) * r_ij.
    //
    // The TPS polynomial null-space block is psi-independent, so its derivatives
    // are identically zero. After differentiating the raw kernel block, the
    // same frozen nullspace projection Z_kernel and frozen identifiability
    // transform Z are applied:
    //   K_c,psi     = K_psi Z_kernel
    //   K_c,psipsi  = K_psipsi Z_kernel
    //   X_psi       = [K_c,psi | 0] Z
    //   X_psipsi    = [K_c,psipsi | 0] Z.
    let derivative_result: Result<(), BasisError> = out_psi
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(out_psi_psi.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .try_for_each(|(i, (mut row_psi, mut row_psi_psi))| {
            let mut local_psi = vec![0.0; k];
            let mut local_psi_psi = vec![0.0; k];
            for j in 0..k {
                let mut dist2 = 0.0;
                for axis in 0..data.ncols() {
                    let delta = data[[i, axis]] - centers[[j, axis]];
                    dist2 += delta * delta;
                }
                let (_, phi_psi, phi_psi_psi) = thin_plate_kernel_psi_triplet_from_distance(
                    dist2.sqrt(),
                    spec.length_scale,
                    data.ncols(),
                )?;
                local_psi[j] = phi_psi;
                local_psi_psi[j] = phi_psi_psi;
            }
            for col in 0..kernel_cols {
                let mut acc_psi = 0.0;
                let mut acc_psi_psi = 0.0;
                for j in 0..k {
                    let z_jc = z_kernel[[j, col]];
                    acc_psi += local_psi[j] * z_jc;
                    acc_psi_psi += local_psi_psi[j] * z_jc;
                }
                row_psi[col] = acc_psi;
                row_psi_psi[col] = acc_psi_psi;
            }
            Ok(())
        });
    derivative_result?;

    if let Some(zf) = identifiability_transform {
        if total_cols != zf.nrows() {
            return Err(BasisError::DimensionMismatch(format!(
                "ThinPlate identifiability transform mismatch in design derivatives: local cols={}, transform rows={}",
                total_cols,
                zf.nrows()
            )));
        }
        return Ok((fast_ab(&out_psi, zf), fast_ab(&out_psi_psi, zf)));
    }

    Ok((out_psi, out_psi_psi))
}

fn build_thin_plate_penalty_psi_derivativeswithworkspace(
    centers: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let z_kernel = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let kernel_cols = z_kernel.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let total_cols = kernel_cols + poly_cols;
    let k = centers.nrows();
    let d = centers.ncols();
    let mut omega_psi = Array2::<f64>::zeros((k, k));
    let mut omega_psi_psi = Array2::<f64>::zeros((k, k));

    // Exact ThinPlate bending-penalty derivatives.
    //
    // The raw curvature block is the center Gram matrix
    //   Omega_ij(psi) = phi(r_ij(psi)),
    //   r_ij(psi) = ||c_i - c_j|| / length_scale = ||c_i - c_j|| exp(psi).
    //
    // The same chain rule as the design block applies entrywise:
    //   Omega_ij,psi     = phi_r(r_ij) * r_ij
    //   Omega_ij,psipsi  = phi_rr(r_ij) * r_ij^2 + phi_r(r_ij) * r_ij.
    //
    // Because the kernel nullspace projector Z_kernel and the frozen
    // identifiability transform Z do not depend on psi during optimization,
    // the projected penalty derivatives are just congruences:
    //   S_psi     = Z^T [diag(Z_kernel^T Omega_psi Z_kernel, 0)] Z
    //   S_psipsi  = Z^T [diag(Z_kernel^T Omega_psipsi Z_kernel, 0)] Z.
    //
    // The double-penalty nullspace block is the projector onto the fixed
    // polynomial nullspace after freezing, so its psi derivatives are zero.
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                dist2 += delta * delta;
            }
            let (_, phi_psi, phi_psi_psi) =
                thin_plate_kernel_psi_triplet_from_distance(dist2.sqrt(), spec.length_scale, d)?;
            omega_psi[[i, j]] = phi_psi;
            omega_psi[[j, i]] = phi_psi;
            omega_psi_psi[[i, j]] = phi_psi_psi;
            omega_psi_psi[[j, i]] = phi_psi_psi;
        }
    }

    let kernel_psi = {
        let zt_s = z_kernel.t().dot(&omega_psi);
        zt_s.dot(&z_kernel)
    };
    let kernel_psi_psi = {
        let zt_s = z_kernel.t().dot(&omega_psi_psi);
        zt_s.dot(&z_kernel)
    };

    let mut s_psi = Array2::<f64>::zeros((total_cols, total_cols));
    let mut s_psi_psi = Array2::<f64>::zeros((total_cols, total_cols));
    s_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi);
    s_psi_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi_psi);

    Ok((
        project_penalty_matrix(&s_psi, identifiability_transform),
        project_penalty_matrix(&s_psi_psi, identifiability_transform),
    ))
}

pub fn build_thin_plate_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let base = build_thin_plate_basiswithworkspace(data, spec, workspace)?;
    let (centers, identifiability_transform) = match &base.metadata {
        BasisMetadata::ThinPlate {
            centers,
            identifiability_transform,
            ..
        } => (centers.clone(), identifiability_transform.clone()),
        _ => {
            return Err(BasisError::InvalidInput(
                "ThinPlate derivative path expected ThinPlate metadata".to_string(),
            ));
        }
    };
    let (design_derivative, _) = build_thin_plate_design_psi_derivativeswithworkspace(
        data,
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let (primary_derivative, _) = build_thin_plate_penalty_psi_derivativeswithworkspace(
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let penalties_derivative =
        active_thin_plate_penalty_derivatives(&base.penaltyinfo, &primary_derivative)?;
    Ok(BasisPsiDerivativeResult {
        design_derivative,
        penalties_derivative,
        implicit_operator: None,
    })
}

pub fn build_thin_plate_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let base = build_thin_plate_basiswithworkspace(data, spec, workspace)?;
    let (centers, identifiability_transform) = match &base.metadata {
        BasisMetadata::ThinPlate {
            centers,
            identifiability_transform,
            ..
        } => (centers.clone(), identifiability_transform.clone()),
        _ => {
            return Err(BasisError::InvalidInput(
                "ThinPlate derivative path expected ThinPlate metadata".to_string(),
            ));
        }
    };
    let (_, designsecond_derivative) = build_thin_plate_design_psi_derivativeswithworkspace(
        data,
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let (_, primarysecond_derivative) = build_thin_plate_penalty_psi_derivativeswithworkspace(
        centers.view(),
        spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let penaltiessecond_derivative =
        active_thin_plate_penalty_derivatives(&base.penaltyinfo, &primarysecond_derivative)?;
    Ok(BasisPsiSecondDerivativeResult {
        designsecond_derivative,
        penaltiessecond_derivative,
        implicit_operator: None,
    })
}

/// High-level TPS constructor: selects knots from data, then builds basis+penalty.
pub fn create_thin_plate_spline_basis_with_knot_count(
    data: ArrayView2<f64>,
    num_knots: usize,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basis_with_knot_count_andworkspace(data, num_knots, &mut workspace)
}

pub fn create_thin_plate_spline_basis_with_knot_count_andworkspace(
    data: ArrayView2<f64>,
    num_knots: usize,
    workspace: &mut BasisWorkspace,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let knots = select_thin_plate_knots(data, num_knots)?;
    let basis = create_thin_plate_spline_basiswithworkspace(data, knots.view(), workspace)?;
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
    let constraintvector = match weights {
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
    let c = basis_matrix.t().dot(&constraintvector); // shape k

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

pub fn apply_sum_to_zero_constraint_sparse(
    basis_matrix: &SparseColMat<usize, f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(SparseColMat<usize, f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    let constraint_weights = match weights {
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

    let mut c = Array1::<f64>::zeros(k);
    let (symbolic, values) = basis_matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..k {
        let mut sum = 0.0;
        for idx in col_ptr[col]..col_ptr[col + 1] {
            sum += values[idx] * constraint_weights[row_idx[idx]];
        }
        c[col] = sum;
    }

    let (pivot, pivot_abs) = c
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .expect("non-empty constraint vector");
    if pivot_abs <= 1e-12 {
        return Ok((basis_matrix.clone(), Array2::eye(k)));
    }

    let pivot_value = c[pivot];
    let pivot_start = col_ptr[pivot];
    let pivot_end = col_ptr[pivot + 1];
    let pivot_rows = &row_idx[pivot_start..pivot_end];
    let pivot_vals = &values[pivot_start..pivot_end];

    let mut z = Array2::<f64>::zeros((k, k - 1));
    let mut triplets: Vec<Triplet<usize, usize, f64>> =
        Vec::with_capacity(values.len() + (k - 1) * pivot_rows.len());

    let mut out_col = 0usize;
    for src_col in 0..k {
        if src_col == pivot {
            continue;
        }
        z[[src_col, out_col]] = 1.0;
        let alpha = -c[src_col] / pivot_value;
        z[[pivot, out_col]] = alpha;

        let src_start = col_ptr[src_col];
        let src_end = col_ptr[src_col + 1];
        let src_rows = &row_idx[src_start..src_end];
        let src_vals = &values[src_start..src_end];

        let mut src_pos = 0usize;
        let mut pivot_pos = 0usize;
        while src_pos < src_rows.len() || pivot_pos < pivot_rows.len() {
            let (row, value) = match (src_rows.get(src_pos), pivot_rows.get(pivot_pos)) {
                (Some(&src_row), Some(&pivot_row)) if src_row == pivot_row => {
                    let value = src_vals[src_pos] + alpha * pivot_vals[pivot_pos];
                    src_pos += 1;
                    pivot_pos += 1;
                    (src_row, value)
                }
                (Some(&src_row), Some(&pivot_row)) if src_row < pivot_row => {
                    let value = src_vals[src_pos];
                    src_pos += 1;
                    (src_row, value)
                }
                (Some(_), Some(&pivot_row)) => {
                    let value = alpha * pivot_vals[pivot_pos];
                    pivot_pos += 1;
                    (pivot_row, value)
                }
                (Some(&src_row), None) => {
                    let value = src_vals[src_pos];
                    src_pos += 1;
                    (src_row, value)
                }
                (None, Some(&pivot_row)) => {
                    let value = alpha * pivot_vals[pivot_pos];
                    pivot_pos += 1;
                    (pivot_row, value)
                }
                (None, None) => unreachable!("merge loop guards ensure one side remains"),
            };
            if value.abs() > 1e-12 {
                triplets.push(Triplet::new(row, out_col, value));
            }
        }
        out_col += 1;
    }

    let constrained = SparseColMat::try_new_from_triplets(n, k - 1, &triplets).map_err(|_| {
        BasisError::SparseCreation("failed to build constrained sparse basis".into())
    })?;
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
pub fn applyweighted_orthogonality_constraint(
    basis_matrix: ArrayView2<f64>,
    constraint_matrix: ArrayView2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
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
    let gram = fast_ata(&basis_matrix.to_owned());
    let transform = orthogonality_transform_from_cross_and_gram(&constraint_cross, &gram)?;
    let basis_orthonormal = fast_ab(&basis_matrix, &transform);
    Ok((basis_orthonormal, transform))
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
    let gvar = g.iter().map(|&x| (x - g_mean).powi(2)).sum::<f64>() / (k as f64);
    let g_std = gvar.sqrt().max(1e-10);
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
        let (minval, maxval) = data_range;

        // Double-check for degenerate range - this should be caught by the public function
        // but we add it here as a defensive measure
        if minval == maxval {
            return Err(BasisError::DegenerateRange(num_internal_knots));
        }

        let h = (maxval - minval) / (num_internal_knots as f64 + 1.0);
        let total_knots = num_internal_knots + 2 * (degree + 1);

        let mut knots = Vec::with_capacity(total_knots);

        // Clamped start: repeat minval (degree + 1) times
        for _ in 0..=degree {
            knots.push(minval);
        }

        // Internal knots: uniformly spaced
        for i in 1..=num_internal_knots {
            knots.push(minval + i as f64 * h);
        }

        // Clamped end: repeat maxval (degree + 1) times
        for _ in 0..=degree {
            knots.push(maxval);
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
        let minval = sorted[0];
        let maxval = *sorted.last().unwrap_or(&minval);
        if minval == maxval {
            return Err(BasisError::DegenerateRange(num_internal_knots));
        }
        let scale = (maxval - minval).abs().max(1.0);
        let tol = 1e-12 * scale;

        let total_knots = num_internal_knots + 2 * (degree + 1);
        let mut knots = Vec::with_capacity(total_knots);
        for _ in 0..=degree {
            knots.push(minval);
        }

        if num_internal_knots > 0 {
            let mut support = Vec::with_capacity(sorted.len());
            let mut last: Option<f64> = None;
            for &x in &sorted {
                if x <= minval + tol || x >= maxval - tol {
                    continue;
                }
                if last.map(|prev| (x - prev).abs() <= tol).unwrap_or(false) {
                    continue;
                }
                support.push(x);
                last = Some(x);
            }
            if support.is_empty() {
                return Err(BasisError::InvalidInput(format!(
                    "quantile knot placement requires distinct interior support between {:.6e} and {:.6e}",
                    minval, maxval
                )));
            }
            let n = support.len();
            let mut prev_q = minval;
            for j in 1..=num_internal_knots {
                let p = j as f64 / (num_internal_knots + 1) as f64;
                let pos = p * (n.saturating_sub(1) as f64);
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                let q = if lo == hi {
                    support[lo]
                } else {
                    support[lo] * (1.0 - frac) + support[hi] * frac
                };
                let q = q.clamp(minval, maxval);
                if q <= prev_q + tol || q >= maxval - tol {
                    return Err(BasisError::InvalidInput(format!(
                        "quantile knot placement produced a non-interior knot at index {}: {:.6e}",
                        j - 1,
                        q
                    )));
                }
                knots.push(q);
                prev_q = q;
            }
        }

        for _ in 0..=degree {
            knots.push(maxval);
        }

        Ok(Array::from_vec(knots))
    }

    /// Evaluates all B-spline basis functions at a single point `x`.
    /// This uses a numerically stable implementation of the Cox-de Boor algorithm,
    /// based on Algorithm A2.2 from "The NURBS Book" by Piegl and Tiller.
    ///
    /// For x outside the spline domain [t_degree, tnum_basis], we apply constant
    /// boundary extrapolation by clamping x to the nearest boundary before running
    /// Cox-de Boor recursion.
    #[inline]
    pub(super) fn evaluate_splines_at_point_into(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        basisvalues: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        match degree {
            3 => evaluate_splines_at_point_fixed::<3>(x, knots, basisvalues, scratch),
            2 => evaluate_splines_at_point_fixed::<2>(x, knots, basisvalues, scratch),
            1 => evaluate_splines_at_point_fixed::<1>(x, knots, basisvalues, scratch),
            _ => evaluate_splines_at_point_dynamic(x, degree, knots, basisvalues, scratch),
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
        basisvalues: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        let (mu, num_basis) = evaluate_spline_local_values(x, DEGREE, knots, scratch);
        assert_eq!(basisvalues.len(), num_basis);
        let n = &scratch.n;
        basisvalues.fill(0.0);
        for i in 0..=DEGREE {
            let gi = mu as isize + i as isize - DEGREE as isize;
            if gi >= 0 {
                let global_idx = gi as usize;
                if global_idx < num_basis {
                    basisvalues[global_idx] = n[i];
                }
            }
        }
    }

    #[inline]
    fn evaluate_splines_at_point_dynamic(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        basisvalues: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        let (mu, num_basis) = evaluate_spline_local_values(x, degree, knots, scratch);
        assert_eq!(basisvalues.len(), num_basis);
        let n = &scratch.n;
        basisvalues.fill(0.0);
        for i in 0..=degree {
            let gi = mu as isize + i as isize - degree as isize;
            if gi >= 0 {
                let global_idx = gi as usize;
                if global_idx < num_basis {
                    basisvalues[global_idx] = n[i];
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
        let (mu, _) = evaluate_spline_local_values(x, degree, knots, scratch);
        assert_eq!(values.len(), degree + 1);
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
        let mut basisvalues = Array1::zeros(num_basis);
        let mut scratch = BsplineScratch::new(degree);
        evaluate_splines_at_point_into(
            x,
            degree,
            knots,
            basisvalues
                .as_slice_mut()
                .expect("basis row should be contiguous"),
            &mut scratch,
        );
        basisvalues
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
    validate_knots_for_degree(knot_vector, degree)?;

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
pub fn evaluate_ispline_scalarwith_scratch(
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
    let num_bspline_basis = knot_vector.len() - bs_degree - 1;
    let num_ispline_basis = num_bspline_basis.saturating_sub(1);
    if out.len() != num_ispline_basis {
        return Err(BasisError::DimensionMismatch(format!(
            "I-spline output buffer length {} does not match basis size {}",
            out.len(),
            num_ispline_basis
        )));
    }

    // Domain for B_{., degree+1} is [t_{degree+1}, t_{num_basis}].
    let left = knot_vector[bs_degree];
    let right = knot_vector[num_bspline_basis];
    let support = bs_degree + 1;
    if x < left {
        out.fill(0.0);
        return Ok(());
    }
    if x >= right {
        if scratch.left_local.len() < support {
            scratch.left_local.resize(support, 0.0);
        }
        if scratch.left_offsets.len() < num_bspline_basis {
            scratch.left_offsets.resize(num_bspline_basis, 0.0);
        }
        scratch.left_offsets[..num_bspline_basis].fill(0.0);
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
        let left_offsets = &mut scratch.left_offsets[..num_bspline_basis];
        let mut left_running = 0.0_f64;
        for offset in (0..support).rev() {
            let j = left_start + offset;
            if j >= num_bspline_basis {
                continue;
            }
            left_running += left_local[offset];
            left_offsets[j] = left_running;
        }
        for j in 1..num_bspline_basis {
            let value = 1.0 - left_offsets[j];
            out[j - 1] = if value.abs() <= 1e-15 { 0.0 } else { value };
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
    let lead_end = start.min(num_bspline_basis);
    if lead_end > 1 {
        out[..(lead_end - 1)].fill(total);
    }

    let mut running = 0.0f64;
    for offset in (0..support).rev() {
        let j = start + offset;
        if j >= num_bspline_basis {
            continue;
        }
        running += local[offset];
        if j > 0 {
            out[j - 1] = running;
        }
    }

    // Subtract left-boundary constants so I_j(left) = 0 exactly.
    if scratch.left_local.len() < support {
        scratch.left_local.resize(support, 0.0);
    }
    if scratch.left_offsets.len() < num_bspline_basis {
        scratch.left_offsets.resize(num_bspline_basis, 0.0);
    }
    scratch.left_offsets[..num_bspline_basis].fill(0.0);
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
    let left_offsets = &mut scratch.left_offsets[..num_bspline_basis];
    let mut left_running = 0.0_f64;
    for offset in (0..support).rev() {
        let j = left_start + offset;
        if j >= num_bspline_basis {
            continue;
        }
        left_running += left_local[offset];
        left_offsets[j] = left_running;
    }
    for j in 1..num_bspline_basis {
        let out_idx = j - 1;
        out[out_idx] -= left_offsets[j];
        if out[out_idx].abs() <= 1e-15 {
            out[out_idx] = 0.0;
        }
    }
    Ok(())
}

/// Compute the k-th derivative of an I-spline basis as a dense matrix.
///
/// The I-spline of degree `degree` uses internal B-splines of degree `degree+1`.
/// The k-th derivative of I-spline j is the right-cumulative sum of the k-th
/// derivatives of those B-splines, starting from column j+1 down to j.
///
/// This produces `num_bspline_basis - 1` columns (same as the I-spline value
/// basis), where `num_bspline_basis = len(knot_vector) - degree - 2`.
pub fn create_ispline_derivative_dense(
    data: ArrayView1<'_, f64>,
    knot_vector: &Array1<f64>,
    degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, BasisError> {
    if derivative_order == 0 {
        // For order 0, return the I-spline value basis.
        let (basis_arc, _) = create_basis::<Dense>(
            data,
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::i_spline(),
        )?;
        return Ok(basis_arc.as_ref().clone());
    }
    let bs_degree = degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    if derivative_order > bs_degree {
        // Derivative order exceeds basis degree — result is identically zero.
        let num_bspline_basis = knot_vector.len().checked_sub(bs_degree + 1).unwrap_or(0);
        let num_ispline_basis = num_bspline_basis.saturating_sub(1);
        return Ok(Array2::zeros((data.len(), num_ispline_basis)));
    }
    let num_bspline_cols = knot_vector.len().saturating_sub(bs_degree + 1);
    let db = match derivative_order {
        1 | 2 => {
            let bspline_options = match derivative_order {
                1 => BasisOptions::first_derivative(),
                2 => BasisOptions::second_derivative(),
                _ => unreachable!(),
            };
            let (db_arc, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knot_vector.view()),
                bs_degree,
                bspline_options,
            )?;
            db_arc.as_ref().clone()
        }
        3 | 4 => {
            let mut db = Array2::<f64>::zeros((data.len(), num_bspline_cols));
            for (row_idx, &x) in data.iter().enumerate() {
                let row = db.slice_mut(s![row_idx, ..]).into_slice().ok_or_else(|| {
                    BasisError::InvalidInput(
                        "I-spline derivative row is not contiguous".to_string(),
                    )
                })?;
                match derivative_order {
                    3 => evaluate_bsplinethird_derivative_scalar(
                        x,
                        knot_vector.view(),
                        bs_degree,
                        row,
                    )?,
                    4 => evaluate_bspline_fourth_derivative_scalar(
                        x,
                        knot_vector.view(),
                        bs_degree,
                        row,
                    )?,
                    _ => unreachable!(),
                }
            }
            db
        }
        _ => unreachable!(),
    };
    let num_ispline_cols = num_bspline_cols.saturating_sub(1);
    if num_ispline_cols == 0 {
        return Ok(Array2::zeros((data.len(), 0)));
    }
    // Right-cumulative sum: I-spline derivative column j = sum_{m=j+1..end} dB_m.
    // In our indexing: output column j (0-based) = sum of dB columns j+1..num_bspline_cols.
    let mut out = Array2::<f64>::zeros((data.len(), num_ispline_cols));
    for i in 0..data.len() {
        let mut running = 0.0_f64;
        for j in (1..num_bspline_cols).rev() {
            let term = db[[i, j]];
            if term.is_finite() {
                running += term;
            }
            out[[i, j - 1]] = running;
        }
    }
    // Apply numerical floor for near-zero values.
    for val in out.iter_mut() {
        if val.abs() <= 1e-12 {
            *val = 0.0;
        }
    }
    Ok(out)
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
    evaluate_ispline_scalarwith_scratch(x, knot_vector, degree, out, &mut scratch)
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
    validate_knots_for_degree(knot_vector, degree)?;

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
    let num_bspline_basis = knot_vector.len() - bs_degree - 1;
    let num_ispline_basis = num_bspline_basis.saturating_sub(1);
    let mut out = Array2::<f64>::zeros((data.len(), num_ispline_basis));
    let mut scratch = internal::BsplineScratch::new(bs_degree);
    let support = bs_degree + 1;
    let mut local = vec![0.0; support];
    let left = knot_vector[bs_degree];
    let right = knot_vector[num_bspline_basis];

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
    let mut left_offsets = vec![0.0_f64; num_bspline_basis];
    let mut left_running = 0.0_f64;
    for offset in (0..support).rev() {
        let j = left_start + offset;
        if j >= num_bspline_basis {
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
            for j in 1..num_bspline_basis {
                let value = 1.0 - left_offsets[j];
                out[[row_i, j - 1]] = if value.abs() <= 1e-15 { 0.0 } else { value };
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
        let lead_end = start.min(num_bspline_basis);
        if lead_end > 1 {
            out.slice_mut(s![row_i, 0..(lead_end - 1)]).fill(total);
        }
        let mut running = 0.0f64;
        for offset in (0..support).rev() {
            let j = start + offset;
            if j >= num_bspline_basis {
                continue;
            }
            running += local[offset];
            if j > 0 {
                let value = running - left_offsets[j];
                out[[row_i, j - 1]] = if value.abs() <= 1e-15 { 0.0 } else { value };
            }
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
pub fn evaluate_bsplinesecond_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 2 {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 2,
            minimum_degree: 2,
        });
    }
    let num_basis_lower = knot_vector
        .len()
        .saturating_sub(degree - 1)
        .saturating_sub(1);
    let mut deriv_lower = vec![0.0; num_basis_lower];
    let mut lower_basis = vec![0.0; knot_vector.len().saturating_sub(degree - 1)];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(2));
    evaluate_bsplinesecond_derivative_scalar_into(
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
pub fn evaluate_bsplinesecond_derivative_scalar_into(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    deriv_lower: &mut [f64],
    lower_basis: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> Result<(), BasisError> {
    if degree < 2 {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 2,
            minimum_degree: 2,
        });
    }
    validate_knots_for_degree(knot_vector, degree)?;

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }
    if num_basis > 0 {
        let left = knot_vector[degree];
        let right = knot_vector[num_basis];
        if x < left || x > right {
            out.fill(0.0);
            return Ok(());
        }
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
pub fn evaluate_bsplinethird_derivative_scalar(
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if degree < 3 {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 3,
            minimum_degree: 3,
        });
    }
    let numsecond_lower = knot_vector.len().saturating_sub(degree);
    let mut second_lower = vec![0.0; numsecond_lower];
    let mut deriv_lower = vec![0.0; knot_vector.len().saturating_sub(degree - 1)];
    let mut lower_basis = vec![0.0; knot_vector.len().saturating_sub(degree - 2)];
    let mut lower_scratch = internal::BsplineScratch::new(degree.saturating_sub(3));
    evaluate_bsplinethird_derivative_scalar_into(
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
pub fn evaluate_bsplinethird_derivative_scalar_into(
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
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 3,
            minimum_degree: 3,
        });
    }
    validate_knots_for_degree(knot_vector, degree)?;

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }
    if num_basis > 0 {
        let left = knot_vector[degree];
        let right = knot_vector[num_basis];
        if x < left || x > right {
            out.fill(0.0);
            return Ok(());
        }
    }

    let expectedsecond_lower = knot_vector.len().saturating_sub(degree);
    if second_lower.len() != expectedsecond_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-second-derivative buffer length {} does not match expected length {}",
            second_lower.len(),
            expectedsecond_lower
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

    evaluate_bsplinesecond_derivative_scalar_into(
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
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 4,
            minimum_degree: 4,
        });
    }
    let numthird_lower = knot_vector.len().saturating_sub(degree);
    let mut third_lower = vec![0.0; numthird_lower];
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
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: 4,
            minimum_degree: 4,
        });
    }
    validate_knots_for_degree(knot_vector, degree)?;

    let num_basis = knot_vector.len() - degree - 1;
    if out.len() != num_basis {
        return Err(BasisError::InvalidKnotVector(format!(
            "Output buffer length {} does not match number of basis functions {}",
            out.len(),
            num_basis
        )));
    }
    if num_basis > 0 {
        let left = knot_vector[degree];
        let right = knot_vector[num_basis];
        if x < left || x > right {
            out.fill(0.0);
            return Ok(());
        }
    }

    let expectedthird_lower = knot_vector.len().saturating_sub(degree);
    if third_lower.len() != expectedthird_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-third-derivative buffer length {} does not match expected length {}",
            third_lower.len(),
            expectedthird_lower
        )));
    }
    let expectedsecond_lower = knot_vector.len().saturating_sub(degree - 1);
    if second_lower.len() != expectedsecond_lower {
        return Err(BasisError::InvalidKnotVector(format!(
            "Lower-second-derivative buffer length {} does not match expected length {}",
            second_lower.len(),
            expectedsecond_lower
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

    evaluate_bsplinethird_derivative_scalar_into(
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
    use ndarray::{Array1, Array2, array};
    use num_dual::{DualNum, second_derivative};
    use std::sync::Arc;

    fn dense_orthogonality_relative_residual(
        basis_matrix: ArrayView2<'_, f64>,
        constraint_matrix: ArrayView2<'_, f64>,
    ) -> f64 {
        let cross = basis_matrix.t().dot(&constraint_matrix);
        let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm = basis_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
        let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
        num / (b_norm * c_norm).max(1e-300)
    }

    fn scaling_test_profile<D: DualNum<f64> + Copy>(t: D) -> D {
        D::one() + t * t + t.powi(4)
    }

    fn scaling_testphi<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
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

    #[test]
    fn polynomial_hessian_operator_penalizes_trace_free_curvature() {
        let points = array![[0.0, 0.0]];
        let hessian = monomial_hessian_operator_block(points.view(), 2);
        let beta = array![0.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let signal = hessian.dot(&beta);
        let frobenius_sq = signal.iter().map(|value| value * value).sum::<f64>();
        let laplacian = signal[0] + signal[3];

        assert_eq!(signal.to_vec(), vec![2.0, 0.0, 0.0, -2.0]);
        assert_eq!(laplacian, 0.0);
        assert_eq!(frobenius_sq, 8.0);
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
    fn shared_owned_data_matrix_reuses_cached_arc_for_same_view() {
        let data =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("data");
        let cache = BasisCacheContext::default();

        let first = shared_owned_data_matrix(data.view(), &cache);
        let second = shared_owned_data_matrix(data.view(), &cache);

        assert!(Arc::ptr_eq(&first, &second));
        assert!(cache.owned_data.resident_bytes() > 0);
    }

    #[test]
    fn owned_data_cache_respects_byte_budget() {
        // Tiny budget: only one 2x2 matrix fits.
        let policy = crate::resource::ResourcePolicy {
            max_owned_data_cache_bytes: 8 * 2 * 2,
            ..crate::resource::ResourcePolicy::default_library()
        };
        let cache = BasisCacheContext::with_policy(&policy);

        let first = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("first data");
        let second = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("second data");
        let third =
            Array2::from_shape_vec((2, 2), vec![9.0, 10.0, 11.0, 12.0]).expect("third data");

        let _ = shared_owned_data_matrix(first.view(), &cache);
        let _ = shared_owned_data_matrix(second.view(), &cache);
        let _ = shared_owned_data_matrix(third.view(), &cache);

        // At most one 2x2 f64 matrix (32 bytes) resident.
        assert!(cache.owned_data.resident_bytes() <= 8 * 2 * 2);
    }

    #[test]
    fn owned_data_cache_respects_entry_cap() {
        let cache = BasisCacheContext::default();

        let first = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("first data");
        let second = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("second data");
        let third =
            Array2::from_shape_vec((2, 2), vec![9.0, 10.0, 11.0, 12.0]).expect("third data");

        let first_cached = shared_owned_data_matrix(first.view(), &cache);
        let second_cached = shared_owned_data_matrix(second.view(), &cache);
        let third_cached = shared_owned_data_matrix(third.view(), &cache);

        assert_eq!(cache.owned_data.len(), crate::resource::OWNED_DATA_CACHE_MAX_ENTRIES);
        assert!(cache.owned_data.get(&OwnedDataCacheKey {
            rows: first.nrows(),
            cols: first.ncols(),
            ptr: first.as_ptr() as usize,
            stride0: first.strides()[0],
            stride1: first.strides()[1],
        }).is_none());
        assert!(Arc::ptr_eq(
            &second_cached,
            &shared_owned_data_matrix(second.view(), &cache)
        ));
        assert!(Arc::ptr_eq(
            &third_cached,
            &shared_owned_data_matrix(third.view(), &cache)
        ));
        drop(first_cached);
    }

    #[test]
    fn spatial_distance_cacheability_is_byte_capped() {
        let n = 400_000;

        assert_eq!(spatial_distance_data_center_bytes(n, 24), 76_800_000);
        assert_eq!(spatial_distance_data_center_bytes(n, 32), 102_400_000);
        assert_eq!(spatial_distance_data_center_bytes(n, 64), 204_800_000);
        assert_eq!(spatial_distance_data_center_bytes(n, 128), 409_600_000);
        assert_eq!(spatial_distance_data_center_bytes(n, 1400), 4_480_000_000);

        assert!(spatial_distance_cacheable_entry(n, 24));
        assert!(spatial_distance_cacheable_entry(n, 32));
        assert!(spatial_distance_cacheable_entry(n, 64));
        assert!(!spatial_distance_cacheable_entry(n, 128));
        assert!(!spatial_distance_cacheable_entry(n, 1400));
    }

    #[test]
    fn spatial_distance_cache_evicts_by_total_bytes() {
        fn key(id: usize) -> SpatialDistanceCacheKey {
            SpatialDistanceCacheKey {
                datarows: id,
                data_cols: 1,
                data_ptr: id,
                data_stride0: 1,
                data_stride1: 1,
                centersrows: 1,
                centers_cols: 1,
                centers_hash: id as u64,
            }
        }

        // Each entry reports 200 MiB via its dense arrays.
        fn entry() -> SpatialDistanceCacheEntry {
            // 25 * 1024 * 1024 f64 = 200 MiB per field; we only need one field
            // to hit the target, but keep both populated to match production.
            let pair_bytes: usize = 200 * 1024 * 1024 / std::mem::size_of::<f64>();
            SpatialDistanceCacheEntry {
                data_center_r: Arc::new(Array2::zeros((pair_bytes, 1))),
                center_center_r: Arc::new(Array2::zeros((1, 1))),
            }
        }

        let cache: crate::resource::ByteLruCache<
            SpatialDistanceCacheKey,
            SpatialDistanceCacheEntry,
        > = crate::resource::ByteLruCache::new(512 * 1024 * 1024);

        cache.insert(key(1), entry());
        cache.insert(key(2), entry());
        cache.insert(key(3), entry());

        // Total resident bytes must stay at or below the 512 MiB cap, and the
        // oldest entry must have been evicted.
        assert!(cache.resident_bytes() <= 512 * 1024 * 1024);
        assert!(cache.get(&key(1)).is_none());
        assert!(cache.get(&key(2)).is_some());
        assert!(cache.get(&key(3)).is_some());
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
    fn test_knot_generationwith_training_data_falls_back_to_uniform() {
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
    fn test_thin_plate_kernel_matches_dimensionspecific_forms() {
        let dist2 = 4.0;
        assert_abs_diff_eq!(thin_plate_kernel_from_dist2(dist2, 1).unwrap(), 8.0);
        assert_abs_diff_eq!(
            thin_plate_kernel_from_dist2(dist2, 2).unwrap(),
            0.5 * dist2 * dist2.ln(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(thin_plate_kernel_from_dist2(dist2, 3).unwrap(), -2.0);
        assert_abs_diff_eq!(thin_plate_kernel_from_dist2(0.0, 3).unwrap(), 0.0);

        // d=4: general kernel with m=3, power = 2*3-4 = 2, d even & m>=d/2
        let val4 = thin_plate_kernel_from_dist2(dist2, 4).unwrap();
        assert!(val4.is_finite(), "d=4 kernel should be finite, got {val4}");
        assert_ne!(val4, 0.0, "d=4 kernel at dist2=4 should be nonzero");

        // d=5: m=3, power = 2*3-5 = 1 (odd) → c * r^1
        let val5 = thin_plate_kernel_from_dist2(dist2, 5).unwrap();
        assert!(val5.is_finite(), "d=5 kernel should be finite, got {val5}");

        // d=19: m=10, power = 1 → c * r
        let val19 = thin_plate_kernel_from_dist2(dist2, 19).unwrap();
        assert!(
            val19.is_finite(),
            "d=19 kernel should be finite, got {val19}"
        );

        // Zero distance always returns zero
        assert_abs_diff_eq!(thin_plate_kernel_from_dist2(0.0, 7).unwrap(), 0.0);
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
    fn test_thin_platewith_knot_count_constructor() {
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
    fn test_thin_plate_dimension4_uses_quadratic_polynomial_nullspace() {
        let data = array![[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]];
        let mut knots = Array2::<f64>::zeros((16, 4));
        let mut seed = 7u64;
        for i in 0..knots.nrows() {
            for j in 0..knots.ncols() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                knots[[i, j]] = ((seed >> 33) as f64) / ((1u64 << 31) as f64)
                    + 0.05 * i as f64
                    + 0.01 * j as f64;
            }
        }
        let tps = create_thin_plate_spline_basis(data.view(), knots.view())
            .expect("dimension-4 TPS should build with a quadratic null space");
        assert_eq!(tps.dimension, 4);
        assert_eq!(tps.num_polynomial_basis, 15);
        assert_eq!(tps.num_kernel_basis, 1);
        assert_eq!(tps.basis.nrows(), data.nrows());
        assert_eq!(tps.basis.ncols(), 16);
        assert!(tps.basis.iter().all(|v| v.is_finite()));
        assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));
        assert!(tps.penalty_ridge.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_thin_plate_dimension4_rejects_insufficient_knots_for_quadratic_nullspace() {
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
                assert!(msg.contains("requires at least 15 knots"));
                assert!(msg.contains("degree-2 polynomial null space"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn testvalidate_psd_penalty_rejects_materially_indefinite_matrix() {
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
    fn testvalidate_psd_penalty_keeps_rank_for_uniformly_scaled_psd_penalty() {
        let penalty = array![[4.0, 0.0], [0.0, 1.0]];
        let scaled_penalty = penalty.mapv(|v| v * 1e-12);

        let summary = validate_psd_penalty(
            &penalty,
            "unit test penalty",
            "uniform scaling should not change the positive eigenspace",
        )
        .unwrap();
        let scaled_summary = validate_psd_penalty(
            &scaled_penalty,
            "unit test penalty",
            "uniform scaling should not change the positive eigenspace",
        )
        .unwrap();

        assert_eq!(summary.effective_rank, 2);
        assert_eq!(scaled_summary.effective_rank, summary.effective_rank);
        assert!(scaled_summary.max_abs_eigenvalue > scaled_summary.tolerance);
    }

    #[test]
    fn test_thin_plate_3d_bending_penalty_is_psdwith_positive_rank() {
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
            length_scale: 1.0,
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
    fn test_build_thin_plate_basis_switches_to_lazy_design_for_large_blocks() {
        let n = 17_000usize;
        let k = 2_000usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut centers = Array2::<f64>::zeros((k, 1));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n - 1) as f64;
        }
        for j in 0..k {
            centers[[j, 0]] = j as f64 / (k - 1) as f64;
        }
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
        };
        let result = build_thin_plate_basis(data.view(), &spec).expect("large thin-plate basis");
        assert!(matches!(
            result.design,
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::Lazy(_))
        ));
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
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
        };
        let result = build_thin_plate_basis(data.view(), &spec).unwrap();
        let result_design = result.design.to_dense();

        let mut c = Array2::<f64>::ones((data.nrows(), data.ncols() + 1));
        c.slice_mut(s![.., 1..]).assign(&data);
        let cross = result_design.t().dot(&c);
        let rel = dense_orthogonality_relative_residual(result_design.view(), c.view());

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
                length_scale: 1.0,
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::KMeans {
                    num_centers: 4,
                    max_iter: 5,
                },
                length_scale: 1.0,
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::UniformGrid { points_per_dim: 2 },
                length_scale: 1.0,
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
                length_scale: 1.0,
                double_penalty: false,
                identifiability: SpatialIdentifiability::default(),
            },
        ];
        for spec in specs {
            let result = build_thin_plate_basis(data.view(), &spec).unwrap();
            assert!(result.design.nrows() > 0);
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
            knotspec: BSplineKnotSpec::Generate {
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
            knotspec: BSplineKnotSpec::Automatic {
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
            knotspec: BSplineKnotSpec::Automatic {
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
    fn test_penalty_greville_selectornone_for_uniform_breakpoints() {
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
            knotspec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let built_design = built.design.to_dense();
        let knots = match &built.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => panic!("expected BSpline1D metadata"),
        };
        let g = penalty_greville_abscissae_for_knots(knots, spec.degree)
            .unwrap()
            .expect("quantile knots should trigger Greville scaling");
        let expected = create_difference_penalty_matrix(
            built_design.ncols(),
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
    fn test_build_bspline_basis_1d_none_identifiability_prefers_sparse_design() {
        let x = Array::linspace(0.0, 1.0, 32);
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(6),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };

        let built = build_bspline_basis_1d(x.view(), &spec).expect("build sparse bspline");
        assert!(matches!(built.design, DesignMatrix::Sparse(_)));
    }

    #[test]
    fn test_build_bspline_basis_1d_default_identifiability_prefers_sparse_design() {
        let x = Array::linspace(0.0, 1.0, 32);
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(6),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let built = build_bspline_basis_1d(x.view(), &spec).expect("build centered sparse bspline");
        assert!(matches!(built.design, DesignMatrix::Sparse(_)));
    }

    #[test]
    fn test_build_bspline_basis_1d_quantile_rejects_missing_interior_support() {
        let x = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let spec = BSplineBasisSpec {
            degree: 2,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(3),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };

        match build_bspline_basis_1d(x.view(), &spec).unwrap_err() {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("distinct interior support"));
            }
            err => panic!("expected InvalidInput for missing interior support, got {err:?}"),
        }
    }

    #[test]
    fn test_quantile_knot_generation_excludes_boundary_point_masses() {
        let mut x = vec![1e-9; 16];
        x.extend([4.0, 7.0, 10.0, 20.0, 40.0, 80.0, 160.0, 285.0]);
        let x = Array1::from_vec(x).mapv(f64::ln);
        let knots = internal::generate_full_knot_vector_quantile(x.view(), 6, 3)
            .expect("quantile knots should be inferred from strict interior support");

        let lower = knots[0];
        let upper = knots[knots.len() - 1];
        for &k in knots.iter().skip(4).take(6) {
            assert!(
                k > lower,
                "internal knot should be strictly above lower boundary"
            );
            assert!(
                k < upper,
                "internal knot should be strictly below upper boundary"
            );
        }

        let g = compute_greville_abscissae(&knots, 3).expect("Greville abscissae should be valid");
        for i in 1..g.len() {
            assert!(
                g[i] > g[i - 1],
                "Greville abscissae must be strictly increasing to support divided differences"
            );
        }
    }

    #[test]
    fn test_bspline_identifiability_defaultweighted_sum_tozero() {
        let x = Array::linspace(0.0, 1.0, 40);
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        let built_design = built.design.to_dense();
        for j in 0..built_design.ncols() {
            let col_sum = built_design.column(j).sum();
            assert!(
                col_sum.abs() < 1e-8,
                "default weighted-sum-to-zero failed for column {j}: {col_sum}"
            );
        }

        let (raw_basis, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            spec.degree,
            BasisOptions::value(),
        )
        .unwrap();
        let z = match &built.metadata {
            BasisMetadata::BSpline1D {
                identifiability_transform: Some(z),
                ..
            } => z,
            _ => panic!("expected frozen B-spline identifiability transform"),
        };
        let expected = raw_basis.dot(z);
        assert_eq!(built_design.dim(), expected.dim());
        for i in 0..built_design.nrows() {
            for j in 0..built_design.ncols() {
                assert_abs_diff_eq!(built_design[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bspline_identifiability_weighted_sum_tozero_respects_weights_with_sparse_design() {
        let x = Array::linspace(0.0, 1.0, 30);
        let weights = Array1::from_iter((0..x.len()).map(|idx| 1.0 + idx as f64 / 10.0));
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 4,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::WeightedSumToZero {
                weights: Some(weights.clone()),
            },
        };

        let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
        assert!(matches!(built.design, DesignMatrix::Sparse(_)));
        let built_design = built.design.to_dense();
        for j in 0..built_design.ncols() {
            let weighted_sum = built_design.column(j).dot(&weights);
            assert!(
                weighted_sum.abs() < 1e-8,
                "weighted sum-to-zero failed for column {j}: {weighted_sum}"
            );
        }
    }

    #[test]
    fn test_bspline_identifiability_remove_linear_trend_reduces_two_dims() {
        let x = Array::linspace(0.0, 1.0, 50);
        let raw = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
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
            knotspec: BSplineKnotSpec::Generate {
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
        let built_design = built.design.to_dense();
        let cross = built_design.t().dot(&constraints);
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
    fn test_bspline_basis_sums_to_onewith_uniform_knots() {
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
    fn test_boundaryvalue_handling() {
        // Test for proper boundary value handling at the upper boundary.
        // This test ensures that evaluation at the upper boundary works correctly.

        // Test the internal function directly with the problematic case
        let knots = array![
            0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0
        ];
        let x = 10.0; // This is the value that caused the panic
        let degree = 3;

        let basisvalues = internal::evaluate_splines_at_point(x, degree, knots.view());

        // Should not panic and should return valid results
        assert_eq!(basisvalues.len(), 8); // num_basis = 12 - 3 - 1 = 8

        let sum = basisvalues.sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis functions should sum to 1.0 at boundary, got {}",
            sum
        );
    }

    #[test]
    fn test_basis_boundaryvalues() {
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
    fn test_constant_extrapolation_matches_boundary_basisvalues() {
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
    fn test_dense_basis_preserves_linear_extensionwhen_internal_builder_goes_sparse() {
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

        evaluate_splinessecond_derivative_sparse_into(
            -10.0,
            degree,
            knots.view(),
            &mut d2,
            &mut scratch,
        );
        assert!(d2.iter().all(|v| v.abs() < 1e-12));
        evaluate_splinessecond_derivative_sparse_into(
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
        let n_ispline = knots.len() - (degree + 1) - 2;
        let mut out = vec![0.0; n_ispline];

        evaluate_ispline_scalar(-10.0, knots.view(), degree, &mut out).expect("left boundary eval");
        assert!(out.iter().all(|&v| v.abs() <= 1e-12));

        evaluate_ispline_scalar(10.0, knots.view(), degree, &mut out).expect("right boundary eval");
        for &v in &out {
            assert!((v - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_ispline_scalar_is_monotone_in_x() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 1usize;
        let n_ispline = knots.len() - (degree + 1) - 2;
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
    fn test_create_basis_msplinezero_outside_domain() {
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
    fn test_create_basis_ispline_boundaryrows() {
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
        for j in 0..i_basis.ncols() {
            assert!((i_basis[[2, j]] - 1.0).abs() <= 1e-12);
        }
        for &v in i_basis.row(1) {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_ispline_basis_drops_identicallyzero_leading_column() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let degree = 1usize;
        let x = array![0.0, 0.5, 1.5, 2.5, 3.0];
        let (i_basis, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("create ispline basis");
        let i_basis = i_basis.as_ref();
        assert_eq!(i_basis.ncols(), knots.len() - (degree + 1) - 2);
        for j in 0..i_basis.ncols() {
            assert!(
                i_basis.column(j).iter().any(|&v| v.abs() > 1e-12),
                "I-spline column {j} should not be identically zero"
            );
        }
    }

    #[test]
    fn test_ispline_derivative_matches_cumulative_bspline_derivative_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let degree = 2usize;
        let bs_degree = degree + 1;
        let n_i = knots.len() - bs_degree - 2;

        // Check at interior points away from knot boundaries for stable central differences.
        let xs = [0.35, 0.8, 1.4, 2.2];
        let h = 1e-6;
        for &x in &xs {
            let mut db = vec![0.0; n_i + 1];
            evaluate_bspline_derivative_scalar(x, knots.view(), bs_degree, &mut db).expect("B'(x)");
            let mut d_i = vec![0.0; n_i];
            let mut running = 0.0_f64;
            for j in (1..(n_i + 1)).rev() {
                running += db[j];
                d_i[j - 1] = running;
            }

            crate::assert_central_difference_array!(
                x,
                h,
                |x_eval| {
                    let mut iv = vec![0.0; n_i];
                    evaluate_ispline_scalar(x_eval, knots.view(), degree, &mut iv).unwrap();
                    iv
                },
                d_i,
                2e-5
            );
        }
    }

    #[test]
    fn testvalidate_knots_for_degree_rejects_too_few_knots_for_degree_domain() {
        let knots = array![0.0, 0.0, 1.0, 1.0];
        let err = create_basis::<Dense>(
            array![0.5].view(),
            KnotSource::Provided(knots.view()),
            2,
            BasisOptions::value(),
        )
        .expect_err("degree-2 basis should reject knot vectors with too few knots");
        match err {
            BasisError::InsufficientKnotsForDegree {
                degree,
                required,
                provided,
            } => {
                assert_eq!(degree, 2);
                assert_eq!(required, 6);
                assert_eq!(provided, 4);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn testvalidate_knots_for_degree_rejectszero_support_boundary_basis() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let err = create_basis::<Dense>(
            array![0.5].view(),
            KnotSource::Provided(knots.view()),
            3,
            BasisOptions::value(),
        )
        .expect_err("over-repeated boundary knots should be rejected");
        match err {
            BasisError::InvalidKnotVector(msg) => {
                assert!(msg.contains("zero support"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_densesecond_derivativezeroes_outside_domain_even_on_sparse_heuristic_path() {
        let degree = 3usize;
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0];
        let mut xs = Vec::with_capacity(128);
        xs.push(-0.2);
        for i in 0..126 {
            xs.push(i as f64 / 125.0);
        }
        xs.push(1.2);
        let x = Array1::from_vec(xs);
        let (basis, _) = create_basis::<Dense>(
            x.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::second_derivative(),
        )
        .expect("dense second derivative");
        let basis = basis.as_ref();
        assert!(basis.row(0).iter().all(|v| v.abs() <= 1e-12));
        assert!(
            basis
                .row(basis.nrows() - 1)
                .iter()
                .all(|v| v.abs() <= 1e-12)
        );
    }

    #[test]
    fn test_scalar_higher_derivatives_arezero_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0];
        let mut second = vec![1.0; knots.len() - 3 - 1];
        evaluate_bsplinesecond_derivative_scalar(-0.1, knots.view(), 3, &mut second)
            .expect("second derivative");
        assert!(second.iter().all(|v| v.abs() <= 1e-12));

        let mut third = vec![1.0; knots.len() - 3 - 1];
        evaluate_bsplinethird_derivative_scalar(1.1, knots.view(), 3, &mut third)
            .expect("third derivative");
        assert!(third.iter().all(|v| v.abs() <= 1e-12));

        let knots4 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut fourth = vec![1.0; knots4.len() - 4 - 1];
        evaluate_bspline_fourth_derivative_scalar(-0.2, knots4.view(), 4, &mut fourth)
            .expect("fourth derivative");
        assert!(fourth.iter().all(|v| v.abs() <= 1e-12));
    }

    #[test]
    fn test_higher_derivative_degree_errors_arespecific() {
        let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; knots.len() - 1 - 1];
        let err = evaluate_bsplinesecond_derivative_scalar(0.5, knots.view(), 1, &mut out)
            .expect_err("degree-1 second derivative should fail");
        match err {
            BasisError::InsufficientDegreeForDerivative {
                degree,
                derivative_order,
                minimum_degree,
            } => {
                assert_eq!(degree, 1);
                assert_eq!(derivative_order, 2);
                assert_eq!(minimum_degree, 2);
            }
            other => panic!("unexpected error: {other:?}"),
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
    fn test_mspline_rejectszero_normalization_spans() {
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
        assert!(matches!(err, BasisError::InvalidKnotVector(_)));
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
            let intervalwidth = knots[i + 1] - knots[i];
            let expected = if intervalwidth.abs() < EPS {
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

        let recursivevalues: Vec<f64> = (0..num_basis)
            .map(|i| evaluate_bspline(x, &knots, i, degree))
            .collect();
        let expected = [0.0, 0.0, 1.0];

        assert_eq!(
            recursivevalues.len(),
            expected.len(),
            "Recursive evaluation length mismatch"
        );

        for (i, (&recursive, &expectedvalue)) in
            recursivevalues.iter().zip(expected.iter()).enumerate()
        {
            assert_abs_diff_eq!(recursive, expectedvalue, epsilon = 1e-12);
            assert_abs_diff_eq!(iterative_basis[i], expectedvalue, epsilon = 1e-12);
        }

        let recursive_sum: f64 = recursivevalues.iter().sum();
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
        let nonzero_count = basis_at_1_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            nonzero_count, 2,
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
        let nonzero_count_2_5 = basis_at_2_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            nonzero_count_2_5, 2,
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

        match create_basis::<Dense>(
            array![].view(),
            KnotSource::Generate {
                data_range: (5.0, 5.0),
                num_internal_knots: 0,
            },
            1,
            BasisOptions::value(),
        )
        .unwrap_err()
        {
            BasisError::DegenerateRange(num_knots) => {
                assert_eq!(num_knots, 0);
            }
            err => panic!("Expected DegenerateRange error, got {:?}", err),
        }

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
    fn testsecond_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut d1 = vec![0.0; num_basis];
        let mut d2 = vec![0.0; num_basis];

        let x = 0.37;
        let h = 1e-5;

        evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut d1)
            .expect("first derivative");
        evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, &mut d2)
            .expect("second derivative");

        crate::assert_central_difference_array!(
            x,
            h,
            |x_eval| {
                let mut v = vec![0.0; num_basis];
                evaluate_bspline_derivative_scalar(x_eval, knots.view(), degree, &mut v).unwrap();
                v
            },
            d2,
            1e-3
        );
    }

    #[test]
    fn testthird_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut d3 = vec![0.0; num_basis];

        let x = 0.37;
        let h = 1e-4;

        evaluate_bsplinethird_derivative_scalar(x, knots.view(), degree, &mut d3)
            .expect("third derivative");

        crate::assert_central_difference_array!(
            x,
            h,
            |x_eval| {
                let mut v = vec![0.0; num_basis];
                evaluate_bsplinesecond_derivative_scalar(x_eval, knots.view(), degree, &mut v)
                    .unwrap();
                v
            },
            d3,
            5e-3
        );
    }

    #[test]
    fn test_fourth_derivative_matches_finite_difference() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 4;
        let num_basis = knots.len() - degree - 1;
        let mut d4 = vec![0.0; num_basis];

        let x = 0.47;
        let h = 1e-4;

        evaluate_bspline_fourth_derivative_scalar(x, knots.view(), degree, &mut d4)
            .expect("fourth derivative");

        crate::assert_central_difference_array!(
            x,
            h,
            |x_eval| {
                let mut v = vec![0.0; num_basis];
                evaluate_bsplinethird_derivative_scalar(x_eval, knots.view(), degree, &mut v)
                    .unwrap();
                v
            },
            d4,
            3e-2
        );
    }

    #[test]
    fn test_sparsesecond_derivative_matches_scalar() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1;
        let mut sparsevalues = vec![0.0; degree + 1];
        let mut scalarvalues = vec![0.0; num_basis];
        let mut scratch = BasisEvalScratch::new(degree);

        let xs = [0.05, 0.2, 0.37, 0.61, 0.9];
        for &x in &xs {
            let start = evaluate_splinessecond_derivative_sparse_into(
                x,
                degree,
                knots.view(),
                &mut sparsevalues,
                &mut scratch,
            );

            evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, &mut scalarvalues)
                .expect("scalar second derivative");

            let mut reconstructed = vec![0.0; num_basis];
            for (offset, &value) in sparsevalues.iter().enumerate() {
                let col = start + offset;
                if col < num_basis {
                    reconstructed[col] = value;
                }
            }

            for j in 0..num_basis {
                assert!(
                    (reconstructed[j] - scalarvalues[j]).abs() < 1e-11,
                    "sparse second derivative mismatch at x={}, basis {}: sparse={}, scalar={}",
                    x,
                    j,
                    reconstructed[j],
                    scalarvalues[j]
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
        let n = 12usize;
        let d = 10usize;
        // DuchonNullspaceOrder::Linear requires at least d + 1 = 11 centers,
        // and those centers must be affinely independent so the null-space
        // polynomial block [1, x_1, ..., x_d] has full column rank.
        let k = 12usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut centers = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64 + 1.0) * (j as f64 + 0.5) * 0.01;
            }
        }
        for i in 0..k {
            for j in 0..d {
                let jitter = ((i * 7 + j * 11) % 13) as f64 * 0.011;
                centers[[i, j]] = (i as f64 + 0.25) * (j as f64 + 1.0) * 0.02 + jitter;
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
    fn test_duchon_non_primary_case_buildswith_general_kernel() {
        // DuchonNullspaceOrder::Linear in dimension d=3 needs at least d+1=4
        // affinely independent centers to span [1, x_1, x_2, x_3].
        let data = Array2::<f64>::zeros((4, 3));
        let centers = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
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
    fn test_build_duchon_basisfreezes_default_spatial_identifiability() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            length_scale: Some(1.0),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
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
            identifiability: SpatialIdentifiability::OrthogonalToParametric,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let out = build_duchon_basis(data.view(), &spec).unwrap();
        let out_design = out.design.to_dense();

        let mut c = Array2::<f64>::ones((data.nrows(), data.ncols() + 1));
        c.slice_mut(s![.., 1..]).assign(&data);
        let cross = out_design.t().dot(&c);
        let rel = dense_orthogonality_relative_residual(out_design.view(), c.view());

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
    fn test_build_duchon_basis_uses_operator_penalty_triplet() {
        let data = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
            length_scale: None,
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let out = build_duchon_basis(data.view(), &spec).expect("Duchon basis should build");
        assert_eq!(out.penalties.len(), 3);
        assert_eq!(out.penaltyinfo.len(), 3);
        assert!(out.penaltyinfo.iter().all(|info| info.active));
        assert!(matches!(
            out.penaltyinfo[0].source,
            PenaltySource::OperatorMass
        ));
        assert!(matches!(
            out.penaltyinfo[1].source,
            PenaltySource::OperatorTension
        ));
        assert!(matches!(
            out.penaltyinfo[2].source,
            PenaltySource::OperatorStiffness
        ));
    }

    #[test]
    fn test_build_duchon_basis_linear_nullspace_uses_operator_penalty_triplet() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            length_scale: Some(1.0),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let out = build_duchon_basis(data.view(), &spec).expect("Duchon basis should build");
        assert_eq!(out.penaltyinfo.len(), 3);
        assert!(out.penaltyinfo.iter().all(|info| info.active));
        assert!(matches!(
            out.penaltyinfo[0].source,
            PenaltySource::OperatorMass
        ));
        assert!(matches!(
            out.penaltyinfo[1].source,
            PenaltySource::OperatorTension
        ));
        assert!(matches!(
            out.penaltyinfo[2].source,
            PenaltySource::OperatorStiffness
        ));
    }

    #[test]
    fn filter_active_penalty_candidates_preserves_matching_kronecker_factors() {
        let s = array![[1.0, -1.0], [-1.0, 1.0]];
        let identity = Array2::<f64>::eye(2);
        let kron = crate::construction::kronecker_product(&s, &identity);
        let (_, _, penaltyinfo) = filter_active_penalty_candidates(vec![PenaltyCandidate {
            matrix: kron,
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorMarginal { dim: 0 },
            normalization_scale: 1.0,
            kronecker_factors: Some(vec![s.clone(), identity.clone()]),
        }])
        .expect("matching Kronecker factors should be retained");

        assert_eq!(penaltyinfo.len(), 1);
        assert!(penaltyinfo[0].kronecker_factors.is_some());
    }

    #[test]
    fn filter_active_penalty_candidates_drops_stale_kronecker_factors_after_projection() {
        let s = array![[1.0, -1.0], [-1.0, 1.0]];
        let identity = Array2::<f64>::eye(2);
        let kron = crate::construction::kronecker_product(&s, &identity);
        let z = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        let projected = z.t().dot(&kron).dot(&z);
        let (_, _, penaltyinfo) = filter_active_penalty_candidates(vec![PenaltyCandidate {
            matrix: projected,
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorMarginal { dim: 0 },
            normalization_scale: 1.0,
            kronecker_factors: Some(vec![s, identity]),
        }])
        .expect("projected penalty should still analyze");

        assert_eq!(penaltyinfo.len(), 1);
        assert!(penaltyinfo[0].active);
        assert!(penaltyinfo[0].kronecker_factors.is_none());
    }

    #[test]
    fn test_pairwise_distance_bounds_helper() {
        let pts = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
        let (r_min, r_max) = pairwise_distance_bounds(pts.view()).expect("bounds should exist");
        assert!((r_min - 5.0).abs() < 1e-12);
        assert!((r_max - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_pairwise_distance_bounds_handles_large_finite_coordinates() {
        let pts = array![[0.0], [3.0e200], [6.0e200]];
        let (r_min, r_max) =
            pairwise_distance_bounds(pts.view()).expect("large finite bounds should exist");
        assert!((r_min - 3.0e200).abs() / 3.0e200 < 1e-12);
        assert!((r_max - 6.0e200).abs() / 6.0e200 < 1e-12);
    }

    #[test]
    fn test_pairwise_distance_bounds_sampled_matches_exact_small() {
        // For n <= K_CAP (=1024), sampled path delegates to the exact path.
        let pts = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [-1.0, 1.0]];
        let exact = pairwise_distance_bounds(pts.view()).unwrap();
        let sampled = pairwise_distance_bounds_sampled(pts.view()).unwrap();
        assert!((exact.0 - sampled.0).abs() < 1e-15);
        assert!((exact.1 - sampled.1).abs() < 1e-15);
    }

    #[test]
    fn test_pairwise_distance_bounds_sampled_conservative_on_large() {
        // On a point cloud larger than K_CAP, verify the mathematical
        // conservativeness invariants of the sampled bounds:
        //   sampled r_max <= exact r_max   (sampled max can only shrink)
        //   sampled r_min >= exact r_min   (sampled min can only grow)
        // These guarantees are the correctness contract that lets the sampled
        // path back outer-κ bounds without excluding any feasible κ that the
        // exact method would include.
        let n = 2000usize;
        let mut pts = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            // Deterministic scatter in [0, 100] × [0, 100].
            let x = ((i * 37 + 7) % 1000) as f64 * 0.1;
            let y = ((i * 53 + 11) % 1000) as f64 * 0.1;
            pts[[i, 0]] = x;
            pts[[i, 1]] = y;
        }
        let exact = pairwise_distance_bounds(pts.view()).unwrap();
        let sampled = pairwise_distance_bounds_sampled(pts.view()).unwrap();
        assert!(
            sampled.1 <= exact.1 + 1e-12,
            "sampled r_max {} must not exceed exact r_max {}",
            sampled.1,
            exact.1
        );
        assert!(
            sampled.0 >= exact.0 - 1e-12,
            "sampled r_min {} must not be below exact r_min {}",
            sampled.0,
            exact.0
        );
    }

    #[test]
    fn test_compute_data_center_distances_preserves_tiny_nonzero_separations() {
        let data = array![[0.0, 0.0, 0.0]];
        let centers = array![[1.0e-200, 0.0, 0.0]];
        let distances = compute_data_center_distances(data.view(), centers.view())
            .expect("distance matrix should build");
        assert!(
            distances[[0, 0]] > 0.0,
            "tiny finite separations should not collapse to an exact collision"
        );
        assert!((distances[[0, 0]] - 1.0e-200).abs() / 1.0e-200 < 1e-12);
    }

    #[test]
    fn test_duchon_general_kernel_symmetric_and_finite() {
        let n = 7usize;
        let d = 5usize;
        // DuchonNullspaceOrder::Linear requires k >= d + 1 = 6 affinely
        // independent centers to span [1, x_1, ..., x_d].
        let k = 7usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut centers = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = 0.03 * (i as f64 + 1.0) * (j as f64 + 0.5);
            }
        }
        for i in 0..k {
            for j in 0..d {
                let jitter = ((i * 5 + j * 3) % 11) as f64 * 0.013;
                centers[[i, j]] = 0.07 * (i as f64 + 0.2) * (j as f64 + 0.8) + jitter;
            }
        }
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(0.9),
            5,
            DuchonNullspaceOrder::Linear, // order=1 => m=2
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
    fn test_duchon_polyharmonic_log_branch_sign_depends_on_dimension() {
        let r = 1.7;

        // In 2D the legacy (-1)^m sign happens to agree with the correct formula.
        let m_2d = 2usize;
        let d_2d = 2usize;
        let c_2d = polyharmonic_log_sign(m_2d, d_2d)
            / (2.0_f64.powi((2 * m_2d - 1) as i32)
                * std::f64::consts::PI.powf(0.5 * d_2d as f64)
                * gamma_lanczos(m_2d as f64)
                * gamma_lanczos((m_2d - d_2d / 2 + 1) as f64));
        let expected_2d = c_2d * r.powi((2 * m_2d - d_2d) as i32) * r.ln();
        let got_2d = polyharmonic_kernel(r, m_2d, d_2d);
        assert!((got_2d - expected_2d).abs() < 1e-12);

        // In 4D the correct log-branch sign differs from (-1)^m and must be positive for m=3.
        let m_4d = 3usize;
        let d_4d = 4usize;
        let legacy_sign = (-1.0_f64).powi(m_4d as i32);
        let fixed_sign = polyharmonic_log_sign(m_4d, d_4d);
        assert_eq!(legacy_sign, -1.0);
        assert_eq!(fixed_sign, 1.0);

        let c_4d = fixed_sign
            / (2.0_f64.powi((2 * m_4d - 1) as i32)
                * std::f64::consts::PI.powf(0.5 * d_4d as f64)
                * gamma_lanczos(m_4d as f64)
                * gamma_lanczos((m_4d - d_4d / 2 + 1) as f64));
        let expected_4d = c_4d * r.powi((2 * m_4d - d_4d) as i32) * r.ln();
        let got_4d = polyharmonic_kernel(r, m_4d, d_4d);
        assert!((got_4d - expected_4d).abs() < 1e-12);
        assert!(got_4d > 0.0);
    }

    #[test]
    fn test_pure_duchon_rejects_undefined_gradient_collocation() {
        let centers = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let err = match build_duchon_collocation_operator_matrices(
            centers.view(),
            None,
            None,
            1,
            DuchonNullspaceOrder::Zero,
            None,
            None,
        ) {
            Ok(_) => panic!("d=3, p=1, s=1 has no well-defined collision gradient"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("D1 collocation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pure_duchon_rejects_divergent_laplacian_collocation() {
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let err = match build_duchon_collocation_operator_matrices(
            centers.view(),
            None,
            None,
            0,
            DuchonNullspaceOrder::Linear,
            None,
            None,
        ) {
            Ok(_) => panic!("2D thin-plate Duchon collocation has no finite collision Laplacian"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("D2 collocation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pure_polyharmonic_origin_jets_preserve_derivative_singularities() {
        let (_, _, tps_phi_rr) = polyharmonic_kernel_triplet(0.0, 2, 2).expect("thin-plate jet");
        assert!(
            tps_phi_rr.is_infinite() && tps_phi_rr.is_sign_negative(),
            "2D thin-plate phi_rr(0) should diverge to -inf, got {tps_phi_rr}"
        );
        let (q, _, _, _) =
            duchon_polyharmonic_operator_block_jets(0.0, 2, 2).expect("thin-plate operator jet");
        assert!(
            q.is_infinite() && q.is_sign_negative(),
            "2D thin-plate phi_r/r at collision should diverge to -inf, got {q}"
        );

        let (_, gradient_first, gradient_second) =
            polyharmonic_kernel_triplet(0.0, 2, 3).expect("3D first-derivative jet");
        assert_abs_diff_eq!(
            gradient_first,
            -1.0 / (8.0 * std::f64::consts::PI),
            epsilon = 1e-14
        );
        assert_abs_diff_eq!(gradient_second, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_duchon_hybrid_collision_uses_combined_partial_fraction_limit() {
        let p_order = 1usize;
        let s_order = 1usize;
        let dim = 3usize;
        let length_scale = 1.0;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let got = duchon_matern_kernel_general_from_distance(
            0.0,
            Some(length_scale),
            p_order,
            s_order,
            dim,
            Some(&coeffs),
        )
        .expect("finite hybrid diagonal");
        let expected = 1.0 / (4.0 * std::f64::consts::PI);
        assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_duchon_hybrid_public_basis_uses_nonzero_collision_diagonal() {
        let centers = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let out = create_duchon_spline_basis(
            centers.view(),
            centers.view(),
            Some(1.0),
            1,
            DuchonNullspaceOrder::Zero,
        )
        .expect("hybrid Duchon basis");

        assert_eq!(out.num_kernel_basis, 1);
        let expected_collision = 1.0 / (4.0 * std::f64::consts::PI);
        let expected_offdiag = (1.0 - (-1.0_f64).exp()) / (4.0 * std::f64::consts::PI);
        let expected_projected = expected_collision - expected_offdiag;
        assert_abs_diff_eq!(
            out.penalty_kernel[[0, 0]],
            expected_projected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_duchon_matern_block_origin_includes_kappa_power() {
        let kappa = 4.0;
        let value = duchon_matern_block(0.0, kappa, 1, 1).expect("block value");
        let (jet_value, _, _, _, _) =
            duchon_matern_block_jet4(0.0, kappa, 1, 1).expect("block jet");
        let radial_value =
            duchon_matern_block_radial_derivative(0.0, kappa, 1, 1, 0).expect("radial value");
        assert_abs_diff_eq!(value, 1.0 / 8.0, epsilon = 1e-14);
        assert_abs_diff_eq!(jet_value, value, epsilon = 1e-14);
        assert_abs_diff_eq!(radial_value, value, epsilon = 1e-14);
    }

    #[test]
    fn test_duchon_aniso_collocation_uses_metric_weights() {
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let eta = vec![2.0_f64.ln(), -2.0_f64.ln()];
        let ops = build_duchon_collocation_operator_matrices(
            centers.view(),
            None,
            Some(1.0),
            2,
            DuchonNullspaceOrder::Linear,
            Some(&eta),
            None,
        )
        .expect("anisotropic Duchon collocation");

        let mut workspace = BasisWorkspace::default();
        let z = kernel_constraint_nullspace(
            centers.view(),
            DuchonNullspaceOrder::Linear,
            &mut workspace.cache,
        )
        .expect("kernel constraint nullspace");
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 2usize;
        let dim = 2usize;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0);
        let weights = [4.0, 0.25];
        let sum_weights = weights.iter().sum::<f64>();

        for k in 0..centers.nrows() {
            for col in 0..z.ncols() {
                let mut expected_d2 = 0.0;
                let mut expected_d1 = [0.0; 2];
                for j in 0..centers.nrows() {
                    let h = [
                        centers[[k, 0]] - centers[[j, 0]],
                        centers[[k, 1]] - centers[[j, 1]],
                    ];
                    let s_vec = [weights[0] * h[0] * h[0], weights[1] * h[1] * h[1]];
                    let r = (s_vec[0] + s_vec[1]).sqrt();
                    let (_, phi_r, phi_rr) = duchon_kernel_radial_triplet(
                        r,
                        Some(1.0),
                        p_order,
                        s_order,
                        dim,
                        Some(&coeffs),
                    )
                    .expect("radial triplet");
                    let lap = if r > 1e-10 {
                        let q = phi_r / r;
                        let t = (phi_rr - q) / (r * r);
                        let sum_wb_sb = weights[0] * s_vec[0] + weights[1] * s_vec[1];
                        for axis in 0..dim {
                            expected_d1[axis] += q * weights[axis] * h[axis] * z[[j, col]];
                        }
                        q * sum_weights + t * sum_wb_sb
                    } else {
                        sum_weights * phi_rr
                    };
                    expected_d2 += lap * z[[j, col]];
                }

                for axis in 0..dim {
                    assert_abs_diff_eq!(
                        ops.d1[[k * dim + axis, col]],
                        expected_d1[axis],
                        epsilon = 1e-9
                    );
                }
                assert_abs_diff_eq!(ops.d2[[k, col]], expected_d2, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_matern_center_sum_tozero_produces_kernel_transform() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 0.7,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
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
            aniso_log_scales: None,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        // (k-1) constrained kernel cols + explicit intercept.
        assert_eq!(out.design.ncols(), centers.nrows());
        assert_eq!(out.penalties.len(), 3);
        assert_eq!(out.nullspace_dims.len(), 3);
    }

    #[test]
    fn test_matern_double_penalty_drops_inactive_nullspace_blockwithout_intercept() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 1.1,
            nu: MaternNu::ThreeHalves,
            include_intercept: false,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        assert_eq!(out.penalties.len(), 1);
        assert_eq!(out.nullspace_dims.len(), 1);
        assert_eq!(out.penaltyinfo.len(), 1);
        assert!(out.penaltyinfo.iter().all(|info| info.active));
        assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
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
            aniso_log_scales: None,
        };
        let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
        assert_eq!(out.penalties.len(), 2);
        assert_eq!(out.nullspace_dims.len(), 2);
        assert_eq!(out.penaltyinfo.len(), 2);
        assert!(out.penaltyinfo.iter().all(|info| info.active));
        assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
        assert!(matches!(
            out.penaltyinfo[1].source,
            PenaltySource::DoublePenaltyNullspace
        ));
    }

    #[test]
    fn test_matern_log_kappa_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
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

        let plus_design = plus.design.to_dense();
        let minus_design = minus.design.to_dense();
        let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
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
    fn test_matern_double_penalty_log_kappa_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: true,
            double_penalty: true,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
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
    fn test_thin_plate_log_kappa_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            double_penalty: true,
            identifiability: SpatialIdentifiability::None,
        };
        let deriv = build_thin_plate_basis_log_kappa_derivative(data.view(), &spec)
            .expect("analytic ThinPlate derivative should build");

        let eps: f64 = 1e-6;
        let kappa = 1.0 / spec.length_scale;
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = ls_plus;
        spec_minus.length_scale = ls_minus;
        let plus = build_thin_plate_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_thin_plate_basis(data.view(), &spec_minus).expect("minus build");

        let plus_design = plus.design.to_dense();
        let minus_design = minus.design.to_dense();
        let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
        let fd_primary = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
        let design_err = (&deriv.design_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let primary_err = (&deriv.penalties_derivative[0] - &fd_primary)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(
            design_err < 1e-5,
            "ThinPlate design derivative mismatch: {design_err}"
        );
        assert!(
            primary_err < 1e-5,
            "ThinPlate primary penalty derivative mismatch: {primary_err}"
        );
        assert_eq!(deriv.penalties_derivative.len(), 2);
        assert!(
            deriv.penalties_derivative[1]
                .iter()
                .all(|v| v.abs() < 1e-12),
            "ThinPlate nullspace shrinkage derivative should be zero"
        );
    }

    #[test]
    fn test_thin_plate_log_kappasecond_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            double_penalty: true,
            identifiability: SpatialIdentifiability::None,
        };
        let analytic = build_thin_plate_basis_log_kappasecond_derivative(data.view(), &spec)
            .expect("analytic ThinPlate second derivative should build");
        let base = build_thin_plate_basis(data.view(), &spec).expect("base build");

        let eps: f64 = 2e-5;
        let kappa = 1.0 / spec.length_scale;
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = ls_plus;
        spec_minus.length_scale = ls_minus;
        let plus = build_thin_plate_basis(data.view(), &spec_plus).expect("plus build");
        let minus = build_thin_plate_basis(data.view(), &spec_minus).expect("minus build");

        let plus_design = plus.design.to_dense();
        let base_design = base.design.to_dense();
        let minus_design = minus.design.to_dense();
        let fd_design = (&plus_design - &(base_design.clone() * 2.0) + &minus_design) / (eps * eps);
        let fd_primary = (&plus.penalties[0] - &(base.penalties[0].clone() * 2.0)
            + &minus.penalties[0])
            / (eps * eps);
        let design_err = (&analytic.designsecond_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let primary_err = (&analytic.penaltiessecond_derivative[0] - &fd_primary)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(
            design_err < 5e-3,
            "ThinPlate design second derivative mismatch: {design_err}"
        );
        assert!(
            primary_err < 5e-3,
            "ThinPlate primary penalty second derivative mismatch: {primary_err}"
        );
        assert_eq!(analytic.penaltiessecond_derivative.len(), 2);
        assert!(
            analytic.penaltiessecond_derivative[1]
                .iter()
                .all(|v| v.abs() < 1e-12),
            "ThinPlate nullspace shrinkage second derivative should be zero"
        );
    }

    #[test]
    fn test_duchon_log_kappa_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: Some(0.9),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let mut workspace = BasisWorkspace::default();
        let derivative = build_duchon_basis_log_kappa_derivativewithworkspace(
            data.view(),
            &spec,
            &mut workspace,
        )
        .expect("analytic Duchon derivative should build");

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

        let plus_design = plus.design.to_dense();
        let minus_design = minus.design.to_dense();
        let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
        let design_err = (&derivative.design_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            design_err < 1e-4,
            "Duchon design derivative mismatch too large: {design_err}"
        );

        assert_eq!(derivative.penalties_derivative.len(), plus.penalties.len());
        let fd_primary_penalty = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
        let primary_penalty_err = (&derivative.penalties_derivative[0] - &fd_primary_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            primary_penalty_err < 1e-4,
            "Duchon mass penalty derivative mismatch too large: {primary_penalty_err}"
        );
        for penalty_idx in 1..derivative.penalties_derivative.len() {
            let fd_penalty =
                (&plus.penalties[penalty_idx] - &minus.penalties[penalty_idx]) / (2.0 * eps);
            let penalty_err = (&derivative.penalties_derivative[penalty_idx] - &fd_penalty)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!(
                penalty_err < 1e-4,
                "Duchon operator penalty derivative mismatch too large at block {penalty_idx}: {penalty_err}"
            );
        }
    }

    #[test]
    fn test_duchon_log_kappasecond_derivative_matchesfd() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: Some(0.9),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let mut workspace = BasisWorkspace::default();
        let second_derivative = build_duchon_basis_log_kappasecond_derivativewithworkspace(
            data.view(),
            &spec,
            &mut workspace,
        )
        .expect("analytic Duchon second derivative should build");
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

        let plus_design = plus.design.to_dense();
        let base_design = base.design.to_dense();
        let minus_design = minus.design.to_dense();
        let fd_design = (&plus_design - &(base_design.clone() * 2.0) + &minus_design) / (eps * eps);
        let design_err = (&second_derivative.designsecond_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            design_err < 5e-3,
            "Duchon design second derivative mismatch too large: {design_err}"
        );

        assert_eq!(
            second_derivative.penaltiessecond_derivative.len(),
            base.penalties.len()
        );
        let fd_primary_penalty = (&plus.penalties[0] - &(base.penalties[0].clone() * 2.0)
            + &minus.penalties[0])
            / (eps * eps);
        let primary_penalty_err = (&second_derivative.penaltiessecond_derivative[0]
            - &fd_primary_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            primary_penalty_err < 5e-3,
            "Duchon mass penalty second derivative mismatch too large: {primary_penalty_err}"
        );
        for penalty_idx in 1..second_derivative.penaltiessecond_derivative.len() {
            let fd_penalty = (&plus.penalties[penalty_idx]
                - &(base.penalties[penalty_idx].clone() * 2.0)
                + &minus.penalties[penalty_idx])
                / (eps * eps);
            let penalty_err = (&second_derivative.penaltiessecond_derivative[penalty_idx]
                - &fd_penalty)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!(
                penalty_err < 5e-3,
                "Duchon operator penalty second derivative mismatch too large at block {penalty_idx}: {penalty_err}"
            );
        }
    }

    #[test]
    fn test_gram_and_psi_derivatives_from_operator_matchesfd() {
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
        let sfd = (&s_plus - &s_minus) / (2.0 * h);
        let s2fd = (&s_plus - &(s.mapv(|v| 2.0 * v)) + &s_minus) / (h * h);

        let err1 = (&s_psi - &sfd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let err2 = (&s_psi_psi - &s2fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for i in 0..s_psi.nrows() {
            for j in 0..s_psi.ncols() {
                assert_eq!(s_psi[[i, j]].signum(), sfd[[i, j]].signum());
                assert_eq!(s_psi_psi[[i, j]].signum(), s2fd[[i, j]].signum());
            }
        }

        assert!(err1 < 2e-6, "S' mismatch too large: {err1}");
        assert!(err2 < 5e-4, "S'' mismatch too large: {err2}");
    }

    #[test]
    fn test_normalize_penaltywith_psi_derivatives_matchesfd() {
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

        let (_, sn_psi, sn_psi_psi, _) =
            normalize_penaltywith_psi_derivatives(&s, &s_psi, &s_psi_psi);

        let h = 1e-6;
        let eval_snorm = |psi: f64| {
            let s_eval = &s0 + &(s1.mapv(|v| psi * v)) + &(s2.mapv(|v| 0.5 * psi * psi * v));
            let c = trace_of_product(&s_eval, &s_eval).sqrt();
            s_eval.mapv(|v| v / c)
        };
        let sn = eval_snorm(psi0);
        let sn_plus = eval_snorm(psi0 + h);
        let sn_minus = eval_snorm(psi0 - h);
        let snfd = (&sn_plus - &sn_minus) / (2.0 * h);
        let sn2fd = (&sn_plus - &(sn.mapv(|v| 2.0 * v)) + &sn_minus) / (h * h);

        let err1 = (&sn_psi - &snfd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let err2 = (&sn_psi_psi - &sn2fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for i in 0..sn_psi.nrows() {
            for j in 0..sn_psi.ncols() {
                assert_eq!(sn_psi[[i, j]].signum(), snfd[[i, j]].signum());
                assert_eq!(sn_psi_psi[[i, j]].signum(), sn2fd[[i, j]].signum());
            }
        }

        assert!(err1 < 2e-6, "normalized S' mismatch too large: {err1}");
        assert!(err2 < 5e-4, "normalized S'' mismatch too large: {err2}");
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
            second_derivative(|psi| scaling_testphi(psi, r, eta), psi0);
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
    fn test_duchonspectral_scaling_matches_implementation() {
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
        // phi(r;κ) = κ^δ·H(κr) holds for the spectral Fourier transform, but
        // only modulo a κ-dependent additive constant that reflects the
        // IR-divergence ambiguity of the polyharmonic log branch at even
        // dimension with 2m == d. The code's polyharmonic_kernel picks the
        // log(r) convention (reference scale 1), so under s = κ_2/κ_1 the
        // m=2, d=4 block leaves the residue
        //   s^δ · a_m(κ_1) · c_p · log(s),
        // with c_p = polyharmonic_log_sign(m,d) / (2^{2m-1} π^{d/2} Γ(m) Γ(m-d/2+1)).
        // Operator scalars (q, Δphi) are differentiated and have no residue;
        // they are still checked tightly below.
        let m_log = 2usize;
        let c_p = polyharmonic_log_sign(m_log, k_dim)
            / (2.0_f64.powi((2 * m_log - 1) as i32)
                * std::f64::consts::PI.powf(0.5 * k_dim as f64)
                * gamma_lanczos(m_log as f64)
                * gamma_lanczos((m_log - k_dim / 2 + 1) as f64));
        let a_m_kappa_1 = kappa_1.powf(-2.0 * (s_order + p_order - m_log) as f64);
        let log_branch_residue = (phi_scale * a_m_kappa_1 * c_p * scale.ln()).abs();
        let phi_tol = (log_branch_residue * 1.5).max(1e-12);
        assert!(
            (phi_2 - phi_scale * phi_1).abs() < phi_tol,
            "phi scaling residue {} exceeds expected log-branch bound {}",
            (phi_2 - phi_scale * phi_1).abs(),
            phi_tol,
        );
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
    fn test_duchon_collision_operator_limits_matchphi_rr_identities() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let delta = duchon_scaling_exponent(p_order, s_order, k_dim);

        let core =
            duchon_radial_core_psi_triplet(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision core");
        let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision phi_rr");

        assert!((core.gradient_ratio.value - phi_rr).abs() < 1e-12);
        assert!((core.gradient_ratio.psi - phi_rr_psi).abs() < 1e-12);
        assert!((core.gradient_ratio.psi_psi - phi_rr_psi_psi).abs() < 1e-12);

        let lap = k_dim as f64 * phi_rr;
        let lap_psi = k_dim as f64 * phi_rr_psi;
        let lap_psi_psi = k_dim as f64 * phi_rr_psi_psi;
        assert!((core.laplacian.value - lap).abs() < 1e-12);
        assert!((core.laplacian.psi - lap_psi).abs() < 1e-12);
        assert!((core.laplacian.psi_psi - lap_psi_psi).abs() < 1e-12);

        assert!((phi_rr_psi - (delta + 2.0) * phi_rr).abs() < 1e-10);
        assert!((phi_rr_psi_psi - (delta + 2.0) * (delta + 2.0) * phi_rr).abs() < 1e-9);
    }

    #[test]
    fn test_duchon_radial_jets_use_collision_limits_at_origin() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 4usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let jets = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets at origin");
        let (phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision phi_rr");
        let t_collision = duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi''''")
            / 3.0;

        assert!(jets.phi_r.abs() < 1e-12);
        assert!((jets.q - phi_rr).abs() < 1e-12);
        assert!((jets.lap - k_dim as f64 * phi_rr).abs() < 1e-12);
        assert!(jets.q_r.abs() < 1e-12);
        assert!((jets.q_rr - t_collision).abs() < 1e-12);
        assert!(jets.lap_r.abs() < 1e-12);
        assert!((jets.lap_rr - (k_dim as f64 + 2.0) * t_collision).abs() < 1e-12);
        // t(0) should be finite (= φ''''(0) / 3) and is checked more
        // thoroughly in the dedicated t-field tests below.
        assert!((jets.t - t_collision).abs() < 1e-12);
    }

    #[test]
    fn test_duchon_radial_jets_use_lower_order_collision_limits_at_origin() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let jets = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets at origin");
        let (phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision phi_rr");

        assert!(jets.phi_r.abs() < 1e-12);
        assert!((jets.q - phi_rr).abs() < 1e-12);
        assert!((jets.lap - k_dim as f64 * phi_rr).abs() < 1e-12);
        assert!(jets.q_r.abs() < 1e-12);
        assert!(jets.lap_r.abs() < 1e-12);
    }

    #[test]
    fn test_duchon_radial_jets_t_equals_phi_rr_minus_q_over_r2() {
        // Verify t = (φ'' - q) / r² at several r values.
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

        for &r in &[0.01, 0.1, 0.5, 1.0, 2.0] {
            let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets");
            let t_expected = (jets.phi_rr - jets.q) / (r * r);
            let rel = if t_expected.abs() > 1e-15 {
                ((jets.t - t_expected) / t_expected).abs()
            } else {
                (jets.t - t_expected).abs()
            };
            assert!(
                rel < 1e-10,
                "t mismatch at r={r}: jets.t={}, expected={}, rel_err={rel}",
                jets.t,
                t_expected,
            );
        }
    }

    #[test]
    fn test_duchon_radial_jets_t_equals_q_r_over_r_fd() {
        // Finite-difference check: t = q' / r, so
        //   t ≈ (q(r+ε) - q(r-ε)) / (2ε·r).
        // Uses a 4-point Richardson-extrapolated central stencil so the
        // truncation error is O(h^4) rather than O(h^2), which keeps the
        // relative error below 1e-3 even at r = 0.1 where q''' is steep.
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

        for &r in &[0.1, 0.5, 1.0, 2.0] {
            let eps = 1e-3 * r;
            let jets_2p = duchon_radial_jets(
                r + 2.0 * eps,
                length_scale,
                p_order,
                s_order,
                k_dim,
                &coeffs,
            )
            .expect("jets+2h");
            let jets_p =
                duchon_radial_jets(r + eps, length_scale, p_order, s_order, k_dim, &coeffs)
                    .expect("jets+h");
            let jets_m =
                duchon_radial_jets(r - eps, length_scale, p_order, s_order, k_dim, &coeffs)
                    .expect("jets-h");
            let jets_2m = duchon_radial_jets(
                r - 2.0 * eps,
                length_scale,
                p_order,
                s_order,
                k_dim,
                &coeffs,
            )
            .expect("jets-2h");
            let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets");
            // 5-point central difference: (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h).
            let q_prime_fd =
                (-jets_2p.q + 8.0 * jets_p.q - 8.0 * jets_m.q + jets_2m.q) / (12.0 * eps);
            let t_fd = q_prime_fd / r;
            let rel = if jets.t.abs() > 1e-15 {
                ((jets.t - t_fd) / jets.t).abs()
            } else {
                (jets.t - t_fd).abs()
            };
            assert!(
                rel < 1e-3,
                "t FD mismatch at r={r}: jets.t={}, fd={t_fd}, rel_err={rel}",
                jets.t,
            );
        }
    }

    #[test]
    fn test_duchon_radial_jets_t_derivatives_match_finite_difference() {
        // Uses a 5-point central stencil (O(h^4) truncation) for t_r. The
        // step must be large enough that partial-fraction cancellation in
        // the double-precision t values does not dominate the difference.
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

        for &r in &[0.1_f64, 0.5, 1.0, 2.0] {
            let h = 1e-2 * r.max(1e-6);
            let jets_2p =
                duchon_radial_jets(r + 2.0 * h, length_scale, p_order, s_order, k_dim, &coeffs)
                    .expect("jets+2h");
            let jets_p = duchon_radial_jets(r + h, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets+h");
            let jets_m = duchon_radial_jets(r - h, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets-h");
            let jets_2m =
                duchon_radial_jets(r - 2.0 * h, length_scale, p_order, s_order, k_dim, &coeffs)
                    .expect("jets-2h");
            let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets");

            // 5-point central first derivative:
            //   f'(x) ≈ (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h).
            let t_r_fd = (-jets_2p.t + 8.0 * jets_p.t - 8.0 * jets_m.t + jets_2m.t) / (12.0 * h);
            let rel_t_r = if jets.t_r.abs() > 1e-15 {
                ((jets.t_r - t_r_fd) / jets.t_r).abs()
            } else {
                (jets.t_r - t_r_fd).abs()
            };
            assert!(
                rel_t_r < 1e-2,
                "t_r FD mismatch at r={r}: jets.t_r={}, fd={t_r_fd}, rel_err={rel_t_r}",
                jets.t_r,
            );
            assert!(jets.t_rr.is_finite(), "expected finite t_rr at r={r}");
        }
    }

    #[test]
    fn test_duchon_radial_jets_t_collision_matches_nearby() {
        // The collision limit t(0) should be close to t at small r.
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 4usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

        let jets_0 = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets at origin");
        // Evaluate at a small radius
        let r_small = 1e-4 * length_scale;
        let jets_small =
            duchon_radial_jets(r_small, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("jets at small r");

        let rel = if jets_0.t.abs() > 1e-15 {
            ((jets_0.t - jets_small.t) / jets_0.t).abs()
        } else {
            (jets_0.t - jets_small.t).abs()
        };
        assert!(
            rel < 1e-2,
            "t collision limit should be close to nearby value: t(0)={}, t(r_small)={}, rel_err={rel}",
            jets_0.t,
            jets_small.t,
        );
    }

    #[test]
    fn test_duchon_radial_jets_t_derivative_collision_limits_are_exact() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 5usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let t_rr_collision =
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision phi''''''")
                / 15.0;

        let jets_0 = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("jets at origin");
        assert!(
            jets_0.t_r.abs() < 1e-12,
            "expected t_r(0)=0, got {}",
            jets_0.t_r
        );
        assert!(
            (jets_0.t_rr - t_rr_collision).abs() < 1e-12,
            "expected exact t_rr(0) collision limit, got {} vs {}",
            jets_0.t_rr,
            t_rr_collision
        );
    }

    #[test]
    fn test_duchon_high_dim_single_matern_block_operator_jets_are_stable() {
        let p_order = 0usize;
        let s_order = 1usize;
        let k_dim = 16usize;
        let length_scale = 1.0;
        let kappa = 1.0 / length_scale;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
        let r = 1e-5;

        let jets =
            duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");
        let (q_expected, t_expected, t_r_expected, t_rr_expected) =
            duchon_matern_operator_block_jets(r, kappa, 1, k_dim).expect("block operator jets");

        assert!((jets.q - q_expected).abs() <= 1e-12 * q_expected.abs().max(1.0));
        assert!((jets.t - t_expected).abs() <= 1e-12 * t_expected.abs().max(1.0));
        assert!((jets.t_r - t_r_expected).abs() <= 1e-12 * t_r_expected.abs().max(1.0));
        assert!((jets.t_rr - t_rr_expected).abs() <= 1e-12 * t_rr_expected.abs().max(1.0));
        assert!(
            ((jets.phi_rr - (jets.q + r * r * jets.t)).abs()) <= 1e-12 * jets.phi_rr.abs().max(1.0)
        );
        assert!(
            ((jets.lap - (k_dim as f64 * jets.q + r * r * jets.t)).abs())
                <= 1e-12 * jets.lap.abs().max(1.0)
        );
    }

    #[test]
    fn test_duchon_high_dim_single_matern_block_subfloor_jets_stay_stable() {
        let p_order = 0usize;
        let s_order = 1usize;
        let k_dim = 16usize;
        let length_scale = 1.0;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale;
        let r_small = 0.25 * r_floor;

        let jets_small =
            duchon_radial_jets(r_small, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("sub-floor jets");
        let jets_floor =
            duchon_radial_jets(r_floor, length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("floor jets");

        assert!(jets_small.q.is_finite());
        assert!(jets_small.t.is_finite());
        assert!(jets_small.lap.is_finite());
        assert!((jets_small.q - jets_floor.q).abs() <= 1e-12 * jets_floor.q.abs().max(1.0));
        assert!((jets_small.t - jets_floor.t).abs() <= 1e-12 * jets_floor.t.abs().max(1.0));
        assert!((jets_small.lap - jets_floor.lap).abs() <= 1e-12 * jets_floor.lap.abs().max(1.0));
    }

    #[test]
    fn test_duchon_high_dim_mixed_operator_jets_remain_finite_and_consistent() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 4usize;
        let k_dim = 16usize;
        let length_scale = 1.0;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let r = 1e-5;

        let jets =
            duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");

        assert!(jets.q.is_finite());
        assert!(jets.t.is_finite());
        assert!(jets.t_r.is_finite());
        assert!(jets.t_rr.is_finite());
        assert!(
            ((jets.phi_rr - (jets.q + r * r * jets.t)).abs()) <= 1e-10 * jets.phi_rr.abs().max(1.0)
        );
        assert!(
            ((jets.lap - (k_dim as f64 * jets.q + r * r * jets.t)).abs())
                <= 1e-10 * jets.lap.abs().max(1.0)
        );
    }

    #[test]
    fn test_duchon_kernel_radial_triplet_uses_collisionphi_rr_at_origin() {
        let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
        let s_order = 3usize;
        let k_dim = 4usize;
        let length_scale = 0.85;
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
        let (_, phi_r, phi_rr) = duchon_kernel_radial_triplet(
            0.0,
            Some(length_scale),
            p_order,
            s_order,
            k_dim,
            Some(&coeffs),
        )
        .expect("radial triplet at origin");
        let (phi_rr_collision, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
                .expect("collision phi_rr");

        assert!(phi_r.abs() < 1e-12);
        assert!((phi_rr - phi_rr_collision).abs() < 1e-12);
    }

    #[test]
    fn test_matern_public_second_derivative_matchesfd_of_public_first_derivative() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: None,
        };
        let analytic = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
            .expect("analytic Matérn second derivative should build");

        let eps: f64 = 1e-5;
        let kappa = 1.0 / spec.length_scale;
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = ls_plus;
        spec_minus.length_scale = ls_minus;
        let plus = build_matern_basis_log_kappa_derivative(data.view(), &spec_plus).expect("plus");
        let minus =
            build_matern_basis_log_kappa_derivative(data.view(), &spec_minus).expect("minus");

        let fd_design = (&plus.design_derivative - &minus.design_derivative) / (2.0 * eps);
        let fd_penalty =
            (&plus.penalties_derivative[0] - &minus.penalties_derivative[0]) / (2.0 * eps);

        let design_err = (&analytic.designsecond_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let penalty_err = (&analytic.penaltiessecond_derivative[0] - &fd_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(
            design_err < 5e-3,
            "Matérn public second-derivative design mismatch: {design_err}"
        );
        assert!(
            penalty_err < 5e-3,
            "Matérn public second-derivative penalty mismatch: {penalty_err}"
        );
    }

    #[test]
    fn test_matern_aniso_operator_penalties_use_cross_provider() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.2],
            [0.3, 1.1],
            [0.9, 0.8],
            [0.4, 0.5],
            [0.7, 0.1]
        ];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: 0.9,
            nu: MaternNu::FiveHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::CenterSumToZero,
            aniso_log_scales: Some(vec![0.1, -0.1]),
        };

        let basis = build_matern_basis(data.view(), &spec).expect("aniso Matérn basis");
        let derivs = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec)
            .expect("aniso Matérn derivatives");
        let expected_cols = basis.design.ncols();

        assert_eq!(derivs.penalties_cross_pairs, vec![(0, 1)]);
        let cross_penalties = derivs
            .penalties_cross_provider
            .as_ref()
            .expect("aniso Matérn cross penalties should be provider-backed")
            .evaluate(0, 1)
            .expect("aniso Matérn cross penalties");
        assert!(!cross_penalties.is_empty());
        for penalty in &cross_penalties {
            assert_eq!(penalty.nrows(), expected_cols);
            assert_eq!(penalty.ncols(), expected_cols);
        }
    }

    #[test]
    fn test_duchon_public_second_derivative_matchesfd_of_public_first_derivative() {
        let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: Some(0.9),
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let analytic = build_duchon_basis_log_kappasecond_derivative(data.view(), &spec)
            .expect("analytic Duchon second derivative should build");

        let eps: f64 = 2e-5;
        let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
        let ls_plus = 1.0 / (kappa * eps.exp());
        let ls_minus = 1.0 / (kappa * (-eps).exp());
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.length_scale = Some(ls_plus);
        spec_minus.length_scale = Some(ls_minus);
        let plus = build_duchon_basis_log_kappa_derivative(data.view(), &spec_plus).expect("plus");
        let minus =
            build_duchon_basis_log_kappa_derivative(data.view(), &spec_minus).expect("minus");

        let fd_design = (&plus.design_derivative - &minus.design_derivative) / (2.0 * eps);
        let fd_penalty =
            (&plus.penalties_derivative[0] - &minus.penalties_derivative[0]) / (2.0 * eps);

        let design_err = (&analytic.designsecond_derivative - &fd_design)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let penalty_err = (&analytic.penaltiessecond_derivative[0] - &fd_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(
            design_err < 5e-3,
            "Duchon public second-derivative design mismatch: {design_err}"
        );
        assert!(
            penalty_err < 5e-3,
            "Duchon public second-derivative penalty mismatch: {penalty_err}"
        );
    }

    #[test]
    fn test_duchon_aniso_derivative_blocks_match_realized_smooth_width() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.2],
            [0.3, 1.1],
            [0.9, 0.8],
            [0.4, 0.5],
            [0.7, 0.1]
        ];
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: Some(0.9),
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::default(),
            aniso_log_scales: Some(vec![0.0, 0.0]),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };

        let basis = build_duchon_basis(data.view(), &spec).expect("aniso Duchon basis");
        let derivs = build_duchon_basis_log_kappa_aniso_derivatives(data.view(), &spec)
            .expect("aniso Duchon derivatives");
        let expected_cols = basis.design.ncols();
        assert_eq!(expected_cols, basis.penalties[0].ncols());

        if let Some(op) = derivs.implicit_operator.as_ref() {
            assert_eq!(op.p_out(), expected_cols);
        }
        for design in &derivs.design_first {
            assert_eq!(design.ncols(), expected_cols);
        }
        for design in &derivs.design_second_diag {
            assert_eq!(design.ncols(), expected_cols);
        }
        for penalties in &derivs.penalties_first {
            for penalty in penalties {
                assert_eq!(penalty.nrows(), expected_cols);
                assert_eq!(penalty.ncols(), expected_cols);
            }
        }
        for penalties in &derivs.penalties_second_diag {
            for penalty in penalties {
                assert_eq!(penalty.nrows(), expected_cols);
                assert_eq!(penalty.ncols(), expected_cols);
            }
        }
        assert_eq!(derivs.penalties_cross_pairs, vec![(0, 1)]);
        let cross_penalties = derivs
            .penalties_cross_provider
            .as_ref()
            .expect("aniso Duchon cross penalties should be provider-backed")
            .evaluate(0, 1)
            .expect("aniso Duchon cross penalties");
        for penalty in &cross_penalties {
            assert_eq!(penalty.nrows(), expected_cols);
            assert_eq!(penalty.ncols(), expected_cols);
        }
    }

    #[test]
    fn test_pure_duchon_aniso_derivatives_use_contrast_operator_axes() {
        let data = array![
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.2, 0.8],
            [0.9, 0.7, 0.1],
            [0.2, 0.8, 0.6],
        ];
        let centers = data.clone();
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: None,
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: Some(vec![0.2, -0.1, -0.1]),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };

        let basis = build_duchon_basis(data.view(), &spec).expect("pure Duchon basis");
        let derivs = build_duchon_basis_log_kappa_aniso_derivatives(data.view(), &spec)
            .expect("pure Duchon anisotropic derivatives");

        assert!(derivs.design_first.is_empty());
        assert!(derivs.design_second_diag.is_empty());
        let op = derivs
            .implicit_operator
            .as_ref()
            .expect("pure Duchon should expose an operator-backed anisotropic derivative path");
        assert_eq!(op.n_axes(), 2);
        assert_eq!(op.p_out(), basis.design.ncols());
        assert_eq!(derivs.penalties_first.len(), 2);
        assert_eq!(derivs.penalties_second_diag.len(), 2);
        assert_eq!(derivs.penalties_cross_pairs, vec![(0, 1)]);
        let cross_penalties = derivs
            .penalties_cross_provider
            .as_ref()
            .expect("pure Duchon cross penalties should be provider-backed")
            .evaluate(0, 1)
            .expect("pure Duchon cross penalties");
        assert!(!cross_penalties.is_empty());
    }

    fn pure_duchon_design_for_eta(
        data: &Array2<f64>,
        centers: &Array2<f64>,
        eta: Vec<f64>,
    ) -> Array2<f64> {
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: None,
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: Some(eta),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        build_duchon_basis(data.view(), &spec)
            .expect("pure Duchon basis")
            .design
            .to_dense()
    }

    fn perturb_contrast_eta(base_eta: &[f64], perturbations: &[(usize, f64)]) -> Vec<f64> {
        let mut eta = base_eta.to_vec();
        let last = eta.len() - 1;
        for &(axis, amount) in perturbations {
            eta[axis] += amount;
            eta[last] -= amount;
        }
        eta
    }

    fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
        assert_eq!(actual.dim(), expected.dim());
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_abs <= tolerance,
            "matrix mismatch: max_abs={max_abs:.3e}, tolerance={tolerance:.3e}"
        );
    }

    fn assert_pure_duchon_contrast_hessian_matches_finite_difference(
        data: Array2<f64>,
        centers: Array2<f64>,
        eta: Vec<f64>,
    ) {
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: None,
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: Some(eta.clone()),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let derivs = build_duchon_basis_log_kappa_aniso_derivatives(data.view(), &spec)
            .expect("pure Duchon anisotropic derivatives");
        let op = derivs
            .implicit_operator
            .as_ref()
            .expect("pure Duchon contrast operator");
        let h = 1e-4;
        let x0 = pure_duchon_design_for_eta(&data, &centers, eta.clone());

        for axis in 0..op.n_axes() {
            let x_plus = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(axis, h)]),
            );
            let x_minus = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(axis, -h)]),
            );
            let finite_diff = (&x_plus - &(x0.mapv(|value| 2.0 * value)) + &x_minus)
                .mapv(|value| value / (h * h));
            let analytic = op
                .materialize_second_diag(axis)
                .expect("contrast diagonal Hessian");
            assert_matrix_close(&analytic, &finite_diff, 2e-5);
        }

        if op.n_axes() >= 2 {
            let x_pp = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(0, h), (1, h)]),
            );
            let x_pm = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(0, h), (1, -h)]),
            );
            let x_mp = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(0, -h), (1, h)]),
            );
            let x_mm = pure_duchon_design_for_eta(
                &data,
                &centers,
                perturb_contrast_eta(&eta, &[(0, -h), (1, -h)]),
            );
            let finite_diff = (&x_pp - &x_pm - &x_mp + &x_mm).mapv(|value| value / (4.0 * h * h));
            let analytic = op
                .materialize_second_cross(0, 1)
                .expect("contrast cross Hessian");
            assert_matrix_close(&analytic, &finite_diff, 2e-5);
        }
    }

    #[test]
    fn test_pure_duchon_dim2_contrast_hessian_matches_finite_difference() {
        let data = array![[0.1, 0.2], [0.4, 0.8], [0.9, 0.3], [1.2, 0.7]];
        let centers = array![[0.0, 0.0], [0.8, 0.1], [0.2, 1.0], [1.1, 0.9]];
        assert_pure_duchon_contrast_hessian_matches_finite_difference(
            data,
            centers,
            vec![0.17, -0.17],
        );
    }

    #[test]
    fn test_pure_duchon_dim3_contrast_hessian_matches_finite_difference() {
        let data = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.8, 0.2],
            [0.9, 0.3, 0.7],
            [1.2, 0.7, 0.4],
            [0.6, 1.1, 0.9]
        ];
        let centers = array![
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.2],
            [0.2, 1.0, 0.4],
            [1.1, 0.9, 0.8],
            [0.5, 0.6, 1.2]
        ];
        assert_pure_duchon_contrast_hessian_matches_finite_difference(
            data,
            centers,
            vec![0.2, -0.05, -0.15],
        );
    }

    #[test]
    fn test_pure_duchon_contrast_hessian_matches_raw_axis_reparameterization() {
        let data = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.8, 0.2],
            [0.9, 0.3, 0.7],
            [1.2, 0.7, 0.4],
            [0.6, 1.1, 0.9]
        ];
        let centers = array![
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.2],
            [0.2, 1.0, 0.4],
            [1.1, 0.9, 0.8],
            [0.5, 0.6, 1.2]
        ];
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: None,
            power: 1,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: Some(vec![0.2, -0.05, -0.15]),
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let derivs = build_duchon_basis_log_kappa_aniso_derivatives(data.view(), &spec)
            .expect("pure Duchon anisotropic derivatives");
        let contrast_op = derivs
            .implicit_operator
            .as_ref()
            .expect("pure Duchon contrast operator");
        let mut raw_op = contrast_op.clone();
        raw_op.axis_combinations = None;
        let last = 2;

        for axis in 0..2 {
            let contrast = contrast_op
                .materialize_second_diag(axis)
                .expect("contrast diagonal Hessian");
            let expected = raw_op
                .materialize_second_diag(axis)
                .expect("raw diagonal Hessian")
                - raw_op
                    .materialize_second_cross(axis, last)
                    .expect("raw axis/last Hessian")
                    .mapv(|value| 2.0 * value)
                + raw_op
                    .materialize_second_diag(last)
                    .expect("raw last diagonal Hessian");
            assert_matrix_close(&contrast, &expected, 1e-12);
        }

        let contrast_cross = contrast_op
            .materialize_second_cross(0, 1)
            .expect("contrast cross Hessian");
        let expected_cross = raw_op
            .materialize_second_cross(0, 1)
            .expect("raw cross Hessian")
            - raw_op
                .materialize_second_cross(0, last)
                .expect("raw first/last Hessian")
            - raw_op
                .materialize_second_cross(1, last)
                .expect("raw second/last Hessian")
            + raw_op
                .materialize_second_diag(last)
                .expect("raw last diagonal Hessian");
        assert_matrix_close(&contrast_cross, &expected_cross, 1e-12);
    }

    #[test]
    fn test_duchon_order_zero_builds_constant_nullspace() {
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
            DuchonNullspaceOrder::Zero,
        )
        .expect("order=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, 1);
        assert!(
            out.basis
                .column(out.basis.ncols() - 1)
                .iter()
                .all(|&v| v == 1.0)
        );
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_duchon_order_zero_16d_case_rejects_infinite_diagonal() {
        let data = array![
            [
                0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
                1.40, 1.50
            ],
            [
                0.05, 0.15, 0.18, 0.28, 0.43, 0.47, 0.58, 0.73, 0.82, 0.88, 1.04, 1.08, 1.21, 1.27,
                1.43, 1.48
            ],
            [
                0.12, 0.22, 0.32, 0.27, 0.38, 0.49, 0.63, 0.69, 0.86, 0.95, 1.02, 1.16, 1.18, 1.34,
                1.37, 1.53
            ],
            [
                0.18, 0.19, 0.29, 0.36, 0.41, 0.53, 0.57, 0.76, 0.84, 0.93, 1.08, 1.12, 1.24, 1.31,
                1.46, 1.57
            ],
            [
                0.27, 0.14, 0.24, 0.33, 0.46, 0.55, 0.61, 0.74, 0.91, 0.97, 1.01, 1.19, 1.29, 1.36,
                1.44, 1.60
            ],
            [
                0.31, 0.24, 0.34, 0.41, 0.48, 0.57, 0.68, 0.78, 0.87, 0.99, 1.07, 1.22, 1.26, 1.39,
                1.49, 1.63
            ]
        ];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let err = match create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            1,
            DuchonNullspaceOrder::Zero,
        ) {
            Ok(_) => panic!("16D rough Duchon case has an infinite diagonal"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("pointwise kernel values"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pure_duchon_default_tuple_rejects_insufficient_nullspace() {
        let data = array![[0.0, 0.1], [0.2, 0.0], [0.4, 0.2], [0.6, 0.4], [0.8, 0.5]];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers),
            length_scale: None,
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Zero,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let err = match build_duchon_basis(data.view(), &spec) {
            Ok(_) => panic!("pure Duchon default tuple violates the nullspace-order condition"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("power < dimension/2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pure_duchon_default_counterexample_is_rejected() {
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let block_order =
            pure_duchon_block_order(duchon_p_from_nullspace_order(DuchonNullspaceOrder::Zero), 2);
        let k23 = polyharmonic_kernel(2.0_f64.sqrt(), block_order, 2);
        let alpha = [-2.0, 1.0, 1.0];
        let qform = 2.0 * alpha[1] * alpha[2] * k23;
        assert!(
            qform < 0.0,
            "the raw pure Duchon default tuple is indefinite under the constant-only side condition"
        );

        let spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: None,
            power: 2,
            nullspace_order: DuchonNullspaceOrder::Zero,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
        };
        let err = match build_duchon_basis(centers.view(), &spec) {
            Ok(_) => panic!("indefinite pure Duchon counterexample should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("power < dimension/2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pure_duchon_10d_a2_diagonal_is_infinite_not_zero() {
        let value = polyharmonic_kernel(0.0, 2, 10);
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "d=10, a=2 pure polyharmonic diagonal should be +inf, got {value}"
        );

        let near_zero = polyharmonic_kernel(1.0e-3, 2, 10);
        let expected = 1.0 / (8.0 * std::f64::consts::PI.powi(5)) * 1.0e18;
        assert!(
            ((near_zero - expected) / expected).abs() < 1e-12,
            "unexpected d=10, a=2 near-origin value: got {near_zero}, expected {expected}"
        );
    }

    #[test]
    fn test_duchon_order_one_builds_linear_nullspace() {
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
        .expect("order=1, s=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, data.ncols() + 1);
        assert!(out.basis.iter().all(|v| v.is_finite()));
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_duchon_order_zero_s0_case_builds_constant_nullspace() {
        let data = array![[0.0], [0.2], [0.4], [0.6], [0.8]];
        let centers = data.slice(s![0..4, ..]).to_owned();
        let out = create_duchon_spline_basis(
            data.view(),
            centers.view(),
            Some(1.0),
            0,
            DuchonNullspaceOrder::Zero,
        )
        .expect("order=0, s=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, 1);
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
        let firstfd = (fp - fm) / (2.0 * h);
        let secondfd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), firstfd.signum());
        assert_eq!(phi_rr.signum(), secondfd.signum());
        assert!((phi_r - firstfd).abs() < 5e-5);
        assert!((phi_rr - secondfd).abs() < 1e-3);
    }

    #[test]
    fn test_matern_safe_ratio_matches_closed_form_limits_atzero() {
        let ls = 1.7;
        let kappa = 1.0 / ls;
        let (_, _, _, r32) =
            matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::ThreeHalves)
                .expect("three-halves");
        let (_, _, _, r52) =
            matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::FiveHalves)
                .expect("five-halves");
        let (_, _, _, r72) =
            matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::SevenHalves)
                .expect("seven-halves");
        let (_, _, _, r92) =
            matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::NineHalves)
                .expect("nine-halves");
        assert!((r32 - (-3.0 * kappa * kappa)).abs() < 1e-12);
        assert!((r52 - (-(5.0 / 3.0) * kappa * kappa)).abs() < 1e-12);
        assert!((r72 - (-(7.0 / 5.0) * kappa * kappa)).abs() < 1e-12);
        assert!((r92 - (-(9.0 / 7.0) * kappa * kappa)).abs() < 1e-12);
    }

    #[test]
    fn test_matern_safe_ratio_half_is_finitewith_floor() {
        let ls = 1.3;
        let (_, _, _, ratio) =
            matern_kernel_radial_tripletwith_safe_ratio(0.0, ls, MaternNu::Half).expect("half");
        assert!(ratio.is_finite());
        assert!(ratio < 0.0);
    }

    // ---------------------------------------------------------------
    // Tests for matern_aniso_radial_scalars
    // ---------------------------------------------------------------

    #[test]
    fn test_aniso_scalars_q_matches_phi_prime_over_r() {
        // For several (nu, r) pairs, verify q == phi'/r by comparing against
        // the radial triplet which gives us phi'.
        let ls = 1.4;
        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            for &r in &[0.01, 0.1, 0.5, 1.0, 2.5] {
                let (_, phi_r, _) = matern_kernel_radial_triplet(r, ls, nu).expect("triplet");
                let (_, q, _) = matern_aniso_radial_scalars(r, ls, nu).expect("aniso");
                let q_ref = phi_r / r;
                assert!(
                    (q - q_ref).abs() < 1e-12 * q_ref.abs().max(1.0),
                    "q mismatch for nu={nu:?}, r={r}: q={q}, phi'/r={q_ref}"
                );
            }
        }
    }

    #[test]
    fn test_aniso_scalars_t_matches_definition() {
        // Verify t == (phi'' - q) / r^2 at several r values by computing
        // phi'' from the radial triplet.
        let ls = 1.7;
        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            for &r in &[0.05, 0.2, 0.7, 1.5, 3.0] {
                let (_, phi_r, phi_rr) = matern_kernel_radial_triplet(r, ls, nu).expect("triplet");
                let (_, _, t) = matern_aniso_radial_scalars(r, ls, nu).expect("aniso");
                let q_check = phi_r / r;
                let t_ref = (phi_rr - q_check) / (r * r);
                assert!(
                    (t - t_ref).abs() < 1e-10 * t_ref.abs().max(1.0),
                    "t mismatch for nu={nu:?}, r={r}: t={t}, ref={t_ref}"
                );
            }
        }
    }

    #[test]
    fn test_aniso_scalars_collision_limits() {
        // At r = 0, q(0) = phi''(0), t(0) = phi''''(0) / 3 for smooth nus.
        //   nu=5/2: q(0) = -s^2/3, t(0) = s^4/3
        //   nu=7/2: q(0) = -s^2·(1/5), t(0) = s^4/15
        //   nu=9/2: q(0) = -s^2·(1/7), t(0) = 3·s^4/105 = s^4/35
        let ls = 2.1;
        {
            let s2 = 5.0 / (ls * ls);
            let s4 = s2 * s2;
            let (phi, q, t) =
                matern_aniso_radial_scalars(0.0, ls, MaternNu::FiveHalves).expect("5/2 at 0");
            assert!((phi - 1.0).abs() < 1e-14, "phi(0) should be 1");
            assert!((q - (-s2 / 3.0)).abs() < 1e-12, "q(0) for 5/2: got {q}");
            assert!((t - s4 / 3.0).abs() < 1e-10, "t(0) for 5/2: got {t}");
        }
        {
            let s2 = 7.0 / (ls * ls);
            let s4 = s2 * s2;
            let (phi, q, t) =
                matern_aniso_radial_scalars(0.0, ls, MaternNu::SevenHalves).expect("7/2 at 0");
            assert!((phi - 1.0).abs() < 1e-14);
            assert!((q - (-s2 / 5.0)).abs() < 1e-12, "q(0) for 7/2: got {q}");
            // t(0) = s^4/15
            assert!((t - s4 / 15.0).abs() < 1e-10, "t(0) for 7/2: got {t}");
        }
        {
            let s2 = 9.0 / (ls * ls);
            let s4 = s2 * s2;
            let (phi, q, t) =
                matern_aniso_radial_scalars(0.0, ls, MaternNu::NineHalves).expect("9/2 at 0");
            assert!((phi - 1.0).abs() < 1e-14);
            assert!((q - (-s2 / 7.0)).abs() < 1e-12, "q(0) for 9/2: got {q}");
            // t(0) = 3 s^4 / 105 = s^4 / 35
            assert!((t - s4 / 35.0).abs() < 1e-10, "t(0) for 9/2: got {t}");
        }
    }

    #[test]
    fn test_aniso_scalars_half_and_three_halves_diverge_at_zero() {
        let ls = 1.0;
        assert!(matern_aniso_radial_scalars(0.0, ls, MaternNu::Half).is_err());
        assert!(matern_aniso_radial_scalars(0.0, ls, MaternNu::ThreeHalves).is_err());
    }

    #[test]
    fn test_aniso_scalars_half_and_three_halves_finite_away_from_zero() {
        let ls = 1.5;
        let r = 0.3;
        let (phi, q, t) = matern_aniso_radial_scalars(r, ls, MaternNu::Half).expect("half r>0");
        assert!(phi.is_finite() && q.is_finite() && t.is_finite());
        let (phi, q, t) =
            matern_aniso_radial_scalars(r, ls, MaternNu::ThreeHalves).expect("3/2 r>0");
        assert!(phi.is_finite() && q.is_finite() && t.is_finite());
    }

    #[test]
    fn test_aniso_scalars_t_finite_difference_validation() {
        // Validate t via finite differences on q:
        //   t(r) = d/dr[q(r)] / r  (by differentiating q = phi'/r).
        // Actually, t = (phi'' - q)/r^2. We can also check by finite-diff on
        // the function f(r) = phi'(r)/r:
        //   f'(r) = (phi''(r) - phi'(r)/r) / r = (phi''(r) - q(r)) / r = t(r) * r
        // So t(r) = f'(r) / r = (q(r+h) - q(r-h)) / (2h r).
        let ls = 1.3;
        let h = 1e-6;
        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            for &r in &[0.1, 0.5, 1.0, 2.0] {
                let (_, q_plus, _) = matern_aniso_radial_scalars(r + h, ls, nu).expect("q+");
                let (_, q_minus, _) = matern_aniso_radial_scalars(r - h, ls, nu).expect("q-");
                let (_, _, t) = matern_aniso_radial_scalars(r, ls, nu).expect("t");
                // f'(r) ≈ (q(r+h) - q(r-h)) / (2h), and t = f'(r) / r
                let t_fd = (q_plus - q_minus) / (2.0 * h * r);
                assert!(
                    (t - t_fd).abs() < 1e-4 * t.abs().max(1e-10),
                    "t finite-diff mismatch for nu={nu:?}, r={r}: t={t}, fd={t_fd}"
                );
            }
        }
    }

    #[test]
    fn test_aniso_scalars_phi_matches_kernel_evaluator() {
        // Verify that phi from the aniso scalars matches matern_kernel_from_distance.
        let ls = 0.8;
        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            for &r in &[0.0, 0.1, 0.5, 1.0, 3.0] {
                let phi_ref = matern_kernel_from_distance(r, ls, nu).expect("kern");
                let (phi, _, _) = matern_aniso_radial_scalars(r, ls, nu).expect("aniso");
                assert!(
                    (phi - phi_ref).abs() < 1e-14,
                    "phi mismatch for nu={nu:?}, r={r}: {phi} vs {phi_ref}"
                );
            }
        }
    }

    #[test]
    fn test_aniso_scalars_invalid_inputs() {
        assert!(matern_aniso_radial_scalars(-1.0, 1.0, MaternNu::FiveHalves).is_err());
        assert!(matern_aniso_radial_scalars(1.0, 0.0, MaternNu::FiveHalves).is_err());
        assert!(matern_aniso_radial_scalars(1.0, -1.0, MaternNu::FiveHalves).is_err());
        assert!(matern_aniso_radial_scalars(f64::NAN, 1.0, MaternNu::FiveHalves).is_err());
        assert!(matern_aniso_radial_scalars(1.0, f64::INFINITY, MaternNu::FiveHalves).is_err());
    }

    #[test]
    fn test_duchon_radial_triplet_matches_finite_difference_away_fromzero() {
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
        let firstfd = (fp - fm) / (2.0 * h);
        let secondfd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), firstfd.signum());
        assert_eq!(phi_rr.signum(), secondfd.signum());
        assert!((phi_r - firstfd).abs() < 1e-3);
        assert!((phi_rr - secondfd).abs() < 1e-1);
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
        let firstfd = (fp - fm) / (2.0 * h);
        let secondfd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), firstfd.signum());
        assert_eq!(phi_rr.signum(), secondfd.signum());
        assert!((phi_r - firstfd).abs() < 2e-3);
        assert!(phi_rr.is_finite());
        assert!(secondfd.is_finite());
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
        let firstfd = (fp - fm) / (2.0 * h);
        let secondfd = (fp - 2.0 * phi + fm) / (h * h);
        assert_eq!(phi_r.signum(), firstfd.signum());
        assert_eq!(phi_rr.signum(), secondfd.signum());
        assert!((phi_r - firstfd).abs() < 1e-6);
        assert!((phi_rr - secondfd).abs() < 1e-4);
    }

    #[test]
    fn test_collocation_derivatives_are_finite_at_rzero() {
        let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let m_ops = build_matern_collocation_operator_matrices(
            centers.view(),
            None,
            0.8,
            MaternNu::FiveHalves,
            false,
            None,
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
            None,
        )
        .expect("duchon ops");
        assert!(d_ops.d1.iter().all(|v| v.is_finite()));
        assert!(d_ops.d2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matern_collocationweights_scalerows_by_sqrtweight() {
        let centers = array![[0.0, 0.0], [1.0, 0.0]];
        let unit = build_matern_collocation_operator_matrices(
            centers.view(),
            None,
            0.9,
            MaternNu::FiveHalves,
            false,
            None,
            None, // aniso_log_scales
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
            None, // aniso_log_scales
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

    #[test]
    fn matern_closed_form_should_decay_to_zero_not_nan_at_huge_distance() {
        let r = 1.0e308;
        let value = matern_kernel_from_distance(r, 1.0, MaternNu::NineHalves).expect("kernel");
        assert!(
            value == 0.0,
            "the Matérn kernel should decay to 0 for enormous finite distances, not produce NaN/Inf; got {value}"
        );

        let dpsi = matern_kernel_log_kappa_derivative_from_distance(r, 1.0, MaternNu::NineHalves)
            .expect("kernel first hyper-derivative");
        assert!(
            dpsi == 0.0,
            "the log-kappa derivative should also decay to 0 for enormous finite distances; got {dpsi}"
        );
        let d2psi =
            matern_kernel_log_kappasecond_derivative_from_distance(r, 1.0, MaternNu::NineHalves)
                .expect("kernel second hyper-derivative");
        assert!(
            d2psi == 0.0,
            "the second log-kappa derivative should also decay to 0 for enormous finite distances; got {d2psi}"
        );
    }

    #[test]
    fn maternvalue_psi_triplet_should_decay_to_zero_not_nan_at_huge_distance() {
        let r = 1.0e308;
        let (value, dpsi, d2psi) =
            maternvalue_psi_triplet(r, 1.0, MaternNu::NineHalves).expect("psi triplet");
        assert!(
            value == 0.0,
            "maternvalue_psi_triplet value should decay to 0 for enormous finite distances; got {value}"
        );
        assert!(
            dpsi == 0.0,
            "maternvalue_psi_triplet first psi derivative should decay to 0 for enormous finite distances; got {dpsi}"
        );
        assert!(
            d2psi == 0.0,
            "maternvalue_psi_triplet second psi derivative should decay to 0 for enormous finite distances; got {d2psi}"
        );
    }

    #[test]
    fn matern_operator_psi_triplet_should_decay_to_zero_not_nan_at_huge_distance() {
        let r = 1.0e308;
        let triplet = matern_operator_psi_triplet(r, 1.0, MaternNu::NineHalves, 3)
            .expect("operator psi triplet");
        for (idx, value) in [
            triplet.0, triplet.1, triplet.2, triplet.3, triplet.4, triplet.5, triplet.6, triplet.7,
            triplet.8,
        ]
        .into_iter()
        .enumerate()
        {
            assert!(
                value == 0.0,
                "matern_operator_psi_triplet component {idx} should decay to 0 for enormous finite distances; got {value}"
            );
        }
    }

    #[test]
    fn matern_nine_halves_log_kappasecond_derivative_matches_closed_form() {
        let r = 1.0_f64;
        let length_scale = 1.0_f64;
        let a = 3.0 * r / length_scale;
        let expected = (-a).exp()
            * (-(2.0 / 7.0) * a * a - (2.0 / 7.0) * a.powi(3) - (3.0 / 35.0) * a.powi(4)
                + (1.0 / 105.0) * a.powi(5)
                + (1.0 / 105.0) * a.powi(6));
        let actual = matern_kernel_log_kappasecond_derivative_from_distance(
            r,
            length_scale,
            MaternNu::NineHalves,
        )
        .expect("9/2 second log-kappa derivative");
        assert!(
            (actual - expected).abs() < 1e-15,
            "nu=9/2 second log-kappa derivative should match the closed form at r={r}, length_scale={length_scale}; got {actual} vs {expected}"
        );
    }

    #[test]
    fn matern_operator_psi_triplet_should_match_closed_form_polynomials() {
        let r = 1.0_f64;
        let length_scale = 1.0_f64;

        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            let a = match nu {
                MaternNu::FiveHalves => 5.0_f64.sqrt() * r / length_scale,
                MaternNu::SevenHalves => 7.0_f64.sqrt() * r / length_scale,
                MaternNu::NineHalves => 3.0 * r / length_scale,
                _ => unreachable!("test only covers nu >= 5/2"),
            };
            let (expected_ratio, expected_lap) = match nu {
                MaternNu::FiveHalves => (
                    -(5.0 / 3.0) * (-a).exp() * (a + 1.0),
                    (5.0 / 3.0) * (-a).exp() * (a * a - a - 1.0),
                ),
                MaternNu::SevenHalves => (
                    -(7.0 / 15.0) * (-a).exp() * (a * a + 3.0 * a + 3.0),
                    (7.0 / 15.0) * (-a).exp() * (a.powi(3) - 3.0 * a - 3.0),
                ),
                MaternNu::NineHalves => (
                    -(3.0 / 35.0) * (-a).exp() * (a.powi(3) + 6.0 * a * a + 15.0 * a + 15.0),
                    (3.0 / 35.0)
                        * (-a).exp()
                        * (a.powi(4) + 2.0 * a.powi(3) - 3.0 * a * a - 15.0 * a - 15.0),
                ),
                _ => unreachable!("test only covers nu >= 5/2"),
            };
            let triplet =
                matern_operator_psi_triplet(r, length_scale, nu, 1).expect("operator psi triplet");
            let ratio = triplet.3;
            let lap = triplet.6;
            assert!(
                (ratio - expected_ratio).abs() < 1e-14,
                "phi'(r)/r closed form mismatch for nu={nu:?}: got {ratio} vs {expected_ratio}"
            );
            assert!(
                (lap - expected_lap).abs() < 1e-14,
                "phi'' closed form mismatch for nu={nu:?}: got {lap} vs {expected_lap}"
            );
        }
    }

    #[test]
    fn matern_collocation_operator_matrices_should_match_closed_forms_in_1d() {
        let centers = array![[0.0], [1.0]];
        let length_scale = 1.0_f64;

        for &nu in &[
            MaternNu::FiveHalves,
            MaternNu::SevenHalves,
            MaternNu::NineHalves,
        ] {
            let ops = build_matern_collocation_operator_matrices(
                centers.view(),
                None,
                length_scale,
                nu,
                false,
                None,
                None,
            )
            .expect("matern collocation operators");

            let r = 1.0_f64;
            let a = match nu {
                MaternNu::FiveHalves => 5.0_f64.sqrt() * r / length_scale,
                MaternNu::SevenHalves => 7.0_f64.sqrt() * r / length_scale,
                MaternNu::NineHalves => 3.0 * r / length_scale,
                _ => unreachable!("test only covers nu >= 5/2"),
            };
            let (expected_phi, expected_ratio, expected_second) = match nu {
                MaternNu::FiveHalves => (
                    (1.0 + a + a * a / 3.0) * (-a).exp(),
                    -(5.0 / 3.0) * (-a).exp() * (a + 1.0),
                    (5.0 / 3.0) * (-a).exp() * (a * a - a - 1.0),
                ),
                MaternNu::SevenHalves => (
                    (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a.powi(3)) * (-a).exp(),
                    -(7.0 / 15.0) * (-a).exp() * (a * a + 3.0 * a + 3.0),
                    (7.0 / 15.0) * (-a).exp() * (a.powi(3) - 3.0 * a - 3.0),
                ),
                MaternNu::NineHalves => (
                    (1.0 + a
                        + (3.0 / 7.0) * a * a
                        + (2.0 / 21.0) * a.powi(3)
                        + (1.0 / 105.0) * a.powi(4))
                        * (-a).exp(),
                    -(3.0 / 35.0) * (-a).exp() * (a.powi(3) + 6.0 * a * a + 15.0 * a + 15.0),
                    (3.0 / 35.0)
                        * (-a).exp()
                        * (a.powi(4) + 2.0 * a.powi(3) - 3.0 * a * a - 15.0 * a - 15.0),
                ),
                _ => unreachable!("test only covers nu >= 5/2"),
            };

            assert!(
                (ops.d0[[1, 0]] - expected_phi).abs() < 1e-14,
                "D0 off-diagonal mismatch for nu={nu:?}: got {} vs {expected_phi}",
                ops.d0[[1, 0]]
            );
            assert!(
                (ops.d1[[1, 0]] - expected_ratio).abs() < 1e-14,
                "D1 off-diagonal mismatch for nu={nu:?}: got {} vs {expected_ratio}",
                ops.d1[[1, 0]]
            );
            assert!(
                (ops.d2[[4, 0]] - expected_second).abs() < 1e-14,
                "D2 xx off-diagonal mismatch for nu={nu:?}: got {} vs {expected_second}",
                ops.d2[[4, 0]]
            );
            assert!(
                ops.d2[[5, 0]].abs() < 1e-14
                    && ops.d2[[6, 0]].abs() < 1e-14
                    && ops.d2[[7, 0]].abs() < 1e-14,
                "D2 must expose full Hessian rows with zero transverse/off-diagonal components on x-axis"
            );
        }
    }

    // ---- anisotropic distance helper tests ----

    #[test]
    fn aniso_distance_isotropic_when_eta_zero() {
        // When all η_a = 0, exp(2·0) = 1, so aniso distance == Euclidean distance.
        let x = [1.0, 2.0, 3.0];
        let c = [4.0, 5.0, 6.0];
        let eta = [0.0, 0.0, 0.0];
        let iso_r = {
            let mut d2 = 0.0;
            for a in 0..3 {
                let h = x[a] - c[a];
                d2 += h * h;
            }
            d2.sqrt()
        };
        let (r, s) = aniso_distance_and_components(&x, &c, &eta);
        assert_abs_diff_eq!(r, iso_r, epsilon = 1e-14);
        assert_abs_diff_eq!(aniso_distance(&x, &c, &eta), iso_r, epsilon = 1e-14);
        // s_a components should sum to r²
        let s_sum: f64 = s.iter().sum();
        assert_abs_diff_eq!(s_sum, r * r, epsilon = 1e-14);
    }

    #[test]
    fn aniso_distance_weighted_correctly() {
        // Two axes, η = [ln2, -ln2] so exp(2η) = [4, 1/4].
        // h = [1, 2], so s = [4·1, 0.25·4] = [4, 1], r = √5.
        let x = [3.0, 5.0];
        let c = [2.0, 3.0];
        let eta = [2.0_f64.ln(), -(2.0_f64.ln())];
        let (r, s) = aniso_distance_and_components(&x, &c, &eta);
        assert_abs_diff_eq!(s[0], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r, 5.0_f64.sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(
            aniso_distance(&x, &c, &eta),
            5.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn aniso_distance_components_sum_to_r_squared() {
        let x = [1.5, -0.3, 2.7, 0.1];
        let c = [0.2, 1.1, -0.5, 3.3];
        let eta = [0.5, -0.2, 0.1, -0.4];
        let (r, s) = aniso_distance_and_components(&x, &c, &eta);
        let s_sum: f64 = s.iter().sum();
        assert_abs_diff_eq!(s_sum, r * r, epsilon = 1e-12);
    }

    #[test]
    fn aniso_distance_zero_displacement_gives_zero_component() {
        // When h_a = 0 for some axis, that s_a must be exactly 0.
        let x = [1.0, 5.0, 3.0];
        let c = [1.0, 2.0, 3.0]; // axis 0 and 2 have h=0
        let eta = [10.0, -5.0, -5.0]; // large eta on axis 0 should not matter
        let (r, s) = aniso_distance_and_components(&x, &c, &eta);
        assert_eq!(s[0], 0.0, "s_a should be exactly 0 when h_a = 0");
        assert_eq!(s[2], 0.0, "s_a should be exactly 0 when h_a = 0");
        assert!(s[1] > 0.0);
        // r should equal sqrt(s[1])
        assert_abs_diff_eq!(r, s[1].sqrt(), epsilon = 1e-14);
    }

    // ── knot_cloud_axis_scales tests ─────────────────────────────────────

    #[test]
    fn test_knot_cloud_axis_scales_basic() {
        // 5x3 center matrix with known std devs per axis.
        // Axis 0: values 1,2,3,4,5 → std = sqrt(2.5) ≈ 1.5811
        // Axis 1: values 10,20,30,40,50 → std = sqrt(250) ≈ 15.811
        // Axis 2: values 0,0,0,0,1 → std = sqrt(0.2) ≈ 0.4472
        use ndarray::Array2;
        let centers = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 10.0, 0.0, 2.0, 20.0, 0.0, 3.0, 30.0, 0.0, 4.0, 40.0, 0.0, 5.0, 50.0, 1.0,
            ],
        )
        .unwrap();
        let scales = knot_cloud_axis_scales(centers.view());
        assert_eq!(scales.len(), 3);
        // Axis 0: sample std of [1,2,3,4,5]
        let expected_0 = (2.5_f64).sqrt(); // sqrt(10/4)
        assert_abs_diff_eq!(scales[0], expected_0, epsilon = 1e-10);
        // Axis 1: 10x axis 0
        assert_abs_diff_eq!(scales[1], expected_0 * 10.0, epsilon = 1e-10);
        // Axis 2: sample std of [0,0,0,0,1]
        let var2 = (4.0 * 0.04 + 0.64) / 4.0; // mean=0.2, var = sum((xi-0.2)^2)/4
        let expected_2 = var2.sqrt();
        // Re-derive: mean=0.2, deviations: -0.2,-0.2,-0.2,-0.2,0.8
        // sum of sq = 4*0.04 + 0.64 = 0.8, var = 0.8/4 = 0.2, std = sqrt(0.2)
        assert_abs_diff_eq!(scales[2], expected_2, epsilon = 1e-10);
    }

    #[test]
    fn test_knot_cloud_axis_scales_zero_variance() {
        // One axis is constant → should return sigma=1.0 for that axis.
        use ndarray::Array2;
        let centers =
            Array2::from_shape_vec((4, 2), vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0]).unwrap();
        let scales = knot_cloud_axis_scales(centers.view());
        assert_eq!(scales.len(), 2);
        // Axis 0 has nonzero variance
        assert!(scales[0] > 1e-6);
        // Axis 1 is constant → sigma clamped to 1.0
        assert_abs_diff_eq!(scales[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_knot_cloud_axis_scales_single_center() {
        // Fewer than 2 centers → returns vec![1.0; d].
        use ndarray::Array2;
        let centers = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let scales = knot_cloud_axis_scales(centers.view());
        assert_eq!(scales, vec![1.0, 1.0, 1.0]);
    }

    // ── initial_aniso_contrasts tests ────────────────────────────────────

    #[test]
    fn test_initial_aniso_contrasts_sum_to_zero() {
        // Create centers with different axis scales; verify sum of η ≈ 0.
        use ndarray::Array2;
        let centers = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
                500.0,
            ],
        )
        .unwrap();
        let eta = initial_aniso_contrasts(centers.view());
        assert_eq!(eta.len(), 3);
        let sum: f64 = eta.iter().sum();
        assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_initial_aniso_contrasts_1d_returns_empty() {
        // 1-D centers → empty vec (anisotropy meaningless).
        use ndarray::Array2;
        let centers = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let eta = initial_aniso_contrasts(centers.view());
        assert!(eta.is_empty());
    }

    #[test]
    fn test_initial_aniso_contrasts_equal_scales() {
        // All axes have same std dev → all η should be ~0.
        use ndarray::Array2;
        let centers = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        )
        .unwrap();
        let eta = initial_aniso_contrasts(centers.view());
        assert_eq!(eta.len(), 3);
        for &e in &eta {
            assert_abs_diff_eq!(e, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_initial_aniso_contrasts_unequal_scales() {
        // Axis 0 has 10x the std dev of axis 1.
        // η_a = −ln(σ_a) + mean(−ln(σ_b))
        // Axis with larger σ → more negative −ln(σ) → η_a < 0
        // Axis with smaller σ → more positive −ln(σ) → η_a > 0
        use ndarray::Array2;
        let centers =
            Array2::from_shape_vec((4, 2), vec![10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0])
                .unwrap();
        let eta = initial_aniso_contrasts(centers.view());
        assert_eq!(eta.len(), 2);
        // Axis 0 has 10x spread → negative η (larger scale → smaller κ)
        assert!(
            eta[0] < 0.0,
            "axis with larger spread should have negative η, got {}",
            eta[0]
        );
        // Axis 1 has smaller spread → positive η
        assert!(
            eta[1] > 0.0,
            "axis with smaller spread should have positive η, got {}",
            eta[1]
        );
        // Sum should be zero
        assert_abs_diff_eq!(eta[0] + eta[1], 0.0, epsilon = 1e-12);
        // |η| should be ln(10)/2 for d=2 zero-sum contrasts
        assert_abs_diff_eq!(eta[0].abs(), 10.0_f64.ln() / 2.0, epsilon = 1e-12);
    }

    // ── maybe_initialize_aniso_contrasts tests ──────────────────────────

    #[test]
    fn test_maybe_initialize_replaces_zeros() {
        // Input: Some(&[0.0, 0.0, 0.0]) → should be replaced with knot-derived values.
        use ndarray::Array2;
        let centers = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
                500.0,
            ],
        )
        .unwrap();
        let zeros = vec![0.0, 0.0, 0.0];
        let result = maybe_initialize_aniso_contrasts(centers.view(), Some(&zeros));
        let eta = result.expect("should return Some");
        assert_eq!(eta.len(), 3);
        // Should NOT be all zeros any more — should match initial_aniso_contrasts
        let expected = initial_aniso_contrasts(centers.view());
        for (a, b) in eta.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_maybe_initialize_preserves_nonzero() {
        // Input: Some(&[0.1, -0.05, -0.05]) → should be returned unchanged.
        use ndarray::Array2;
        let centers = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let input = vec![0.1, -0.05, -0.05];
        let result = maybe_initialize_aniso_contrasts(centers.view(), Some(&input));
        let eta = result.expect("should return Some");
        assert_eq!(eta, input);
    }

    #[test]
    fn test_maybe_initialize_preserves_none() {
        // Input: None → should remain None.
        use ndarray::Array2;
        let centers =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = maybe_initialize_aniso_contrasts(centers.view(), None);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Duchon anisotropic exposed-axis second-derivative kernel tests.
    //
    // These tests target the `2 q G_{LR}` overlap term in
    // `ImplicitDesignPsiDerivative::transformed_second_kernel_value`. A bug
    // in that term (e.g. missing the overlap contribution) would be silently
    // masked by finite-difference probes because the radial kernel still
    // produces smooth output — the overlap only shows up as an additive
    // correction in the exact analytic second derivative.
    //
    // The full formula for linear combinations
    //   L = Σ_a l_a ∂/∂ψ_a,   R = Σ_a r_a ∂/∂ψ_a
    // applied to the radial kernel φ (which depends on the distance shaped
    // by per-axis scales s_a = exp(2 η_a)) is
    //
    //   D²_{L,R} φ = t S_L S_R + 2 q G_{LR}
    //                + c q (C_R S_L + C_L S_R)
    //                + c² C_L C_R φ
    //
    // where S_L = Σ l_a s_a, C_L = Σ l_a, G_{LR} = Σ l_a r_a s_a, and
    // c = psi_scale_share. The closed-form values below pin each piece of
    // the formula — in particular the overlap contribution 2 q G_{LR}.
    // -----------------------------------------------------------------------

    #[test]
    fn overlap_diag_contrast_e0_minus_elast_matches_closed_form() {
        // Pure Duchon contrast L = R = e_0 - e_last in d = 3:
        //   C_L = 0, so the `cq` and `c²` terms vanish.
        //   S_L = s_0 - s_last, G_LL = s_0 + s_last.
        //   Correct kernel value = t (s_0 - s_last)² + 2 q (s_0 + s_last).
        // The `2 q (s_0 + s_last)` piece is the overlap term — missing it
        // would leave only the first summand.
        let s = [1.3_f64, 0.7, 2.1];
        let phi = 3.0;
        let q = -0.5;
        let t = 0.9;
        let c = 0.0;
        let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];
        let r = l;

        let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
        let s_r: f64 = r.iter().map(|&(a, ra)| ra * s[a]).sum();
        let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
        let c_r: f64 = r.iter().map(|&(_, ra)| ra).sum();
        let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, r, &s);

        let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
            phi, q, t, s_l, c_l, s_r, c_r, overlap, c,
        );
        let expected = t * (s[0] - s[2]).powi(2) + 2.0 * q * (s[0] + s[2]);

        assert!(
            (got - expected).abs() < 1e-12,
            "diag overlap mismatch: got={got} expected={expected}"
        );

        // Buggy "no-overlap" value misses the `2 q G_LL` correction. Pin
        // that the correction is non-trivial and equals the overlap term.
        let no_overlap = t * (s[0] - s[2]).powi(2);
        assert!(
            (got - no_overlap).abs() > 1e-6,
            "overlap term contributes no correction: got={got} no_overlap={no_overlap}"
        );
        assert!(
            (got - no_overlap - 2.0 * q * (s[0] + s[2])).abs() < 1e-12,
            "overlap correction should equal 2 q (s_0 + s_last) exactly"
        );
    }

    #[test]
    fn overlap_cross_contrast_matches_closed_form() {
        // Pure Duchon cross contrast L = e_0 - e_last, R = e_1 - e_last.
        //   C_L = C_R = 0, so only the first two terms of the formula
        //   survive. Only the `last` axis is shared between L and R:
        //     G_{LR} = l_last * r_last * s_last = (-1)(-1) s_last = s_last.
        //   Correct kernel value = t (s_0 - s_last)(s_1 - s_last) + 2 q s_last.
        // The `2 q s_last` piece is the overlap term.
        let s = [1.3_f64, 0.7, 2.1];
        let phi = 3.0;
        let q = -0.5;
        let t = 0.9;
        let c = 0.0;
        let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];
        let r: &[(usize, f64)] = &[(1, 1.0), (2, -1.0)];

        let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
        let s_r: f64 = r.iter().map(|&(a, ra)| ra * s[a]).sum();
        let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
        let c_r: f64 = r.iter().map(|&(_, ra)| ra).sum();
        let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, r, &s);

        let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
            phi, q, t, s_l, c_l, s_r, c_r, overlap, c,
        );
        let expected = t * (s[0] - s[2]) * (s[1] - s[2]) + 2.0 * q * s[2];

        assert!(
            (got - expected).abs() < 1e-12,
            "cross overlap mismatch: got={got} expected={expected}"
        );

        // Confirm the overlap term is exactly what separates the correct
        // value from the buggy "no-overlap" version: 2 q s_last.
        let no_overlap = t * (s[0] - s[2]) * (s[1] - s[2]);
        assert!(
            (got - no_overlap - 2.0 * q * s[2]).abs() < 1e-12,
            "overlap correction should equal 2 q s_last exactly"
        );
    }

    #[test]
    fn overlap_vs_no_overlap_diag_differs_by_2q_sum() {
        // Pin the streaming-overlap helper itself: for L = R = e_0 - e_last,
        // each l_a² ∈ {0, 1}, so overlap = Σ_a l_a² s_a = s_0 + s_last.
        let s = [1.3_f64, 0.7, 2.1];
        let q = -0.5;
        let l: &[(usize, f64)] = &[(0, 1.0), (2, -1.0)];

        let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, l, &s);
        let expected_overlap = s[0] + s[2];
        assert!(
            (overlap - expected_overlap).abs() < 1e-14,
            "overlap helper mismatch: got={overlap} expected={expected_overlap}"
        );

        let overlap_contribution = 2.0 * q * overlap;
        let expected = 2.0 * q * (s[0] + s[2]);
        assert!(
            (overlap_contribution - expected).abs() < 1e-14,
            "overlap contribution mismatch: got={overlap_contribution} expected={expected}"
        );
    }

    #[test]
    fn overlap_psi_scale_share_nonzero_matches_full_formula() {
        // With c ≠ 0 and L = R = e_0 + e_1 (so C_L = C_R = 2), every term
        // of the full formula is active. Pin the helper against the
        // hand-written expression.
        let s = [1.3_f64, 0.7];
        let phi = 3.0;
        let q = -0.5;
        let t = 0.9;
        let c = 0.25;
        let l: &[(usize, f64)] = &[(0, 1.0), (1, 1.0)];

        let s_l: f64 = l.iter().map(|&(a, la)| la * s[a]).sum();
        let c_l: f64 = l.iter().map(|&(_, la)| la).sum();
        let overlap = ImplicitDesignPsiDerivative::transformed_combo_overlap_streaming(l, l, &s);

        // G_LL = Σ l_a² s_a = s_0 + s_1 for all-ones combo.
        assert!((overlap - (s[0] + s[1])).abs() < 1e-14);

        let got = ImplicitDesignPsiDerivative::transformed_second_kernel_value(
            phi, q, t, s_l, c_l, s_l, c_l, overlap, c,
        );
        let expected = t * s_l * s_l
            + 2.0 * q * overlap
            + c * q * (c_l * s_l + c_l * s_l)
            + c * c * c_l * c_l * phi;

        assert!(
            (got - expected).abs() < 1e-12,
            "psi-scale-share full-formula mismatch: got={got} expected={expected}"
        );
    }
}
