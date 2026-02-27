use crate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd, fast_ab, fast_ata, fast_atb};
use faer::Side;
use faer::sparse::{SparseColMat, Triplet};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::ParallelSlice;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use thiserror::Error;

#[cfg(test)]
use approx::assert_abs_diff_eq;

fn bspline_thread_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .build()
            .expect("bspline thread pool initialization should succeed")
    })
}

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

    #[error(
        "Failed to identify nullspace for sum-to-zero constraint; matrix is ill-conditioned or SVD returned no basis."
    )]
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
}

impl BasisOptions {
    /// Create options for evaluating basis functions (no derivative).
    pub fn value() -> Self {
        Self {
            derivative_order: 0,
        }
    }

    /// Create options for evaluating first derivatives of basis functions.
    pub fn first_derivative() -> Self {
        Self {
            derivative_order: 1,
        }
    }

    /// Create options for evaluating second derivatives of basis functions.
    pub fn second_derivative() -> Self {
        Self {
            derivative_order: 2,
        }
    }
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

    let knot_vec: Array1<f64> = match knot_source {
        KnotSource::Provided(view) => {
            validate_knots_for_degree(view, degree)?;
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
            internal::generate_full_knot_vector(data_range, num_internal_knots, degree)?
        }
    };

    O::build_basis(data, degree, eval_kind, knot_vec)
}

/// Applies first-order linear extension outside `[0, 1]` to a basis matrix
/// that was evaluated at clamped coordinates.
///
/// Given `z_raw` and `z_clamped = clamp(z_raw, 0, 1)`, this mutates
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

    fn build_basis(
        data: ArrayView1<f64>,
        degree: usize,
        eval_kind: BasisEvalKind,
        knot_vec: Array1<f64>,
    ) -> Result<(Self::Output, Array1<f64>), BasisError>;
}

impl BasisOutputFormat for Dense {
    type Output = Arc<Array2<f64>>;

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
                    dense[[row_idx[idx], col]] = values[idx];
                }
            }
            dense
        } else {
            generate_basis_internal::<DenseStorage>(data.view(), knot_view, degree, eval_kind)?
        };

        Ok((Arc::new(basis_matrix), knot_vec))
    }
}

impl BasisOutputFormat for Sparse {
    type Output = SparseColMat<usize, f64>;

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
        internal::evaluate_splines_sparse_into(x, degree, knot_view, values, basis_scratch);
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
        x,
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
            bspline_thread_pool().install(|| {
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
            });
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
                let triplet_chunks: Vec<Vec<Triplet<usize, usize, f64>>> = bspline_thread_pool()
                    .install(|| {
                        data_slice
                            .par_chunks(CHUNK_SIZE)
                            .enumerate()
                            .map_init(
                                || (BasisEvalScratch::new(degree), vec![0.0; support]),
                                |(scratch, values), (chunk_idx, chunk)| {
                                    let base_row = chunk_idx * CHUNK_SIZE;
                                    let mut local =
                                        Vec::with_capacity(chunk.len().saturating_mul(support));
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
                            .collect()
                    });

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
        bspline_thread_pool().install(|| {
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
        });
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
        bspline_thread_pool().install(|| {
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
    /// Recommend a conservative default for high-dimensional settings.
    /// This corresponds to choosing `s = ceil(k/2) + 0.5`, giving `nu = s - k/2`.
    /// For even `k`, this yields `nu = 0.5` (exponential Matérn), avoiding TPS-style
    /// null-space inflation while preserving rotation invariance.
    pub fn recommended_for_dimension(_dimension: usize) -> Self {
        Self::Half
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

impl MaternSplineBasis {
    pub fn penalty_matrices(&self) -> Vec<Array2<f64>> {
        vec![self.penalty_kernel.clone(), self.penalty_ridge.clone()]
    }
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

/// Duchon null-space order. `0` matches fully-penalized Matérn-like behavior,
/// `1` keeps `[1, x_1, ..., x_d]` unpenalized by the primary curvature penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuchonNullspaceOrder {
    Zero,
    Linear,
}

/// Duchon-like basis configuration using a Matérn high-frequency backbone and
/// explicit low-frequency null-space control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuchonBasisSpec {
    pub center_strategy: CenterStrategy,
    pub length_scale: f64,
    pub nu: MaternNu,
    pub nullspace_order: DuchonNullspaceOrder,
    pub double_penalty: bool,
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
        length_scale: f64,
        nu: MaternNu,
        nullspace_order: DuchonNullspaceOrder,
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
    pub metadata: BasisMetadata,
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
    let s_bend_raw = create_difference_penalty_matrix(p_raw, spec.penalty_order, None)?;
    let mut penalties_raw = vec![s_bend_raw.clone()];
    if spec.double_penalty {
        penalties_raw.push(nullspace_shrinkage_penalty(&s_bend_raw)?);
    }

    let (design, penalties, identifiability_transform) = apply_bspline_identifiability_policy(
        design_raw,
        penalties_raw,
        &knots,
        spec.degree,
        &spec.identifiability,
    )?;
    let nullspace_dims = penalties
        .iter()
        .map(estimate_penalty_nullity)
        .collect::<Result<Vec<_>, _>>()?;

    if penalties.len() != nullspace_dims.len() {
        return Err(BasisError::InvalidInput(
            "penalty/nullspace dimension mismatch after identifiability transform".to_string(),
        ));
    }
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
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

    let mut sym = penalty.clone();
    for i in 0..sym.nrows() {
        for j in 0..i {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }

    let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    let max_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (sym.nrows().max(1) as f64) * 1e-10 * max_ev.max(1.0);
    Ok(evals.iter().filter(|&&ev| ev.abs() <= tol).count())
}

fn nullspace_shrinkage_penalty(penalty: &Array2<f64>) -> Result<Array2<f64>, BasisError> {
    if penalty.nrows() != penalty.ncols() {
        return Err(BasisError::DimensionMismatch(
            "penalty matrix must be square when building nullspace shrinkage penalty".to_string(),
        ));
    }
    if penalty.nrows() == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }

    let mut sym = penalty.clone();
    for i in 0..sym.nrows() {
        for j in 0..i {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }

    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    let max_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (sym.nrows().max(1) as f64) * 1e-10 * max_ev.max(1.0);

    let zero_idx: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &ev)| (ev.abs() <= tol).then_some(i))
        .collect();
    let mut shrink = Array2::<f64>::eye(sym.nrows()) * 1e-6;
    if !zero_idx.is_empty() {
        let z = evecs.select(Axis(1), &zero_idx);
        shrink += &fast_ab(&z, &z.t().to_owned());
    }
    Ok(shrink)
}

fn with_fixed_shrinkage(base: &Array2<f64>, shrink: &Array2<f64>) -> Array2<f64> {
    let scale = base
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1e-6;
    base + &shrink.mapv(|v| v * scale)
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
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let tps = create_thin_plate_spline_basis(data, centers.view())?;
    let penalties = vec![if spec.double_penalty {
        with_fixed_shrinkage(&tps.penalty_bending, &tps.penalty_ridge)
    } else {
        tps.penalty_bending.clone()
    }];
    let nullspace_dims = vec![if spec.double_penalty {
        0
    } else {
        tps.num_polynomial_basis
    }];
    Ok(BasisBuildResult {
        design: tps.basis,
        penalties,
        nullspace_dims,
        metadata: BasisMetadata::ThinPlate { centers },
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
fn bessel_i0_manual(x: f64) -> f64 {
    // Manual Cephes-style approximation with two regions:
    //  - |x| < 3.75: polynomial in y=(x/3.75)^2
    //  - otherwise : asymptotic exp(|x|)/sqrt(|x|) times polynomial in y=3.75/|x|
    //
    // This avoids external dependencies and is numerically stable for the
    // argument ranges used by Duchon-Matern K0/K1 evaluation.
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

const DUCHON_SERIES_Z_CUTOFF: f64 = 1.0;
const DUCHON_ASYMPTOTIC_Z_CUTOFF: f64 = 35.0;

#[inline(always)]
fn duchon_p_from_nullspace_order(order: DuchonNullspaceOrder) -> usize {
    match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
    }
}

#[inline(always)]
fn duchon_s_from_nu(nu: MaternNu) -> usize {
    // Map half-integer nu values to integer s in the spectral factor
    // (kappa^2 + |w|^2)^s. This preserves a compact API while enabling the
    // general integer (p,s,k) construction.
    match nu {
        MaternNu::Half => 1,
        MaternNu::ThreeHalves => 2,
        MaternNu::FiveHalves => 3,
        MaternNu::SevenHalves => 4,
        MaternNu::NineHalves => 5,
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
fn duchon_partial_fraction_coeffs(
    p_order: usize,
    s_order: usize,
    kappa: f64,
) -> (Vec<f64>, Vec<f64>) {
    // 1/(ρ^{2p}(κ²+ρ²)^s) = Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)^n
    let mut a = vec![0.0_f64; p_order + 1]; // 1-based m
    let mut b = vec![0.0_f64; s_order + 1]; // 1-based n
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
    (a, b)
}

fn duchon_matern_kernel_general_from_distance(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon-Matern kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon-Matern length_scale must be finite and positive".to_string(),
        ));
    }
    let kappa = 1.0 / length_scale;
    // Preserve the exact validated closed-form branch for the original primary case.
    if p_order == 1 && s_order == 4 && k_dim == 10 {
        return duchon_matern_kernel_p1_s4_k10_from_distance(r, length_scale);
    }

    // For intrinsic p>0 kernels, diagonal values are not uniquely defined in the
    // generalized-kernel sense; use intrinsic convention.
    if r == 0.0 && p_order > 0 {
        return Ok(0.0);
    }

    let (a, b) = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let mut val = 0.0_f64;
    for (m, coeff) in a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val += coeff * duchon_polyharmonic_block(r, m, k_dim);
    }
    for (n, coeff) in b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val += coeff * duchon_matern_block(r, kappa, n, k_dim)?;
    }
    Ok(val)
}

#[inline(always)]
fn duchon_matern_p1_s4_k10_asymptotic(z: f64) -> f64 {
    // For large z = κr, Bessel terms in the exact closed form are exponentially
    // suppressed, leaving the algebraic leading term.
    let invz = 1.0 / z.max(1e-300);
    let invz2 = invz * invz;
    let invz4 = invz2 * invz2;
    48.0 * invz4 * invz4
}

#[inline(always)]
fn duchon_matern_p1_s4_k10_closed_form(r: f64, kappa: f64) -> f64 {
    // Mathematical derivation (primary case: p=1, s=4, k=10):
    //
    // Spectral kernel:
    //   K^(ω) ∝ 1 / (|ω|^(2p) * (κ^2 + |ω|^2)^s)
    // so here:
    //   K^(ω) ∝ 1 / (|ω|^2 * (κ^2 + |ω|^2)^4).
    //
    // For this specific tuple (p,s,k), the radial kernel is equivalently:
    //   K(a) = (1/96) ∫_0^1 u^3 K0(a*sqrt(u)) du,   a = κr.
    //
    // This function evaluates an exact algebraic reduction of that integral:
    //   K(a)= 48/a^8
    //         - (1/48) K1(a)/a
    //         - (1/8)  K0(a)/a^2
    //         - (3/4)  K1(a)/a^3
    //         - 3      K0(a)/a^4
    //         - 12     K1(a)/a^5
    //         - 24     K0(a)/a^6
    //         - 48     K1(a)/a^7.
    //
    // IMPORTANT: this representation is exact but numerically ill-conditioned as a->0,
    // because each term is large and cancellation leaves an O(log a) remainder.
    // We therefore call this only in the moderate-a regime from the dispatcher.
    let z = (kappa * r).max(1e-300);
    // Hot-path speedup: for very large z the exact Bessel terms are negligible.
    if z > DUCHON_ASYMPTOTIC_Z_CUTOFF {
        return duchon_matern_p1_s4_k10_asymptotic(z);
    }
    let k0 = bessel_k0_stable(z);
    let k1 = bessel_k1_stable(z);
    // Faster and slightly more accurate than repeated powi/division:
    // build reciprocal powers once and evaluate grouped terms.
    let invz = 1.0 / z;
    let invz2 = invz * invz;
    let invz3 = invz2 * invz;
    let invz4 = invz2 * invz2;
    let invz5 = invz4 * invz;
    let invz6 = invz3 * invz3;
    let invz7 = invz6 * invz;
    let invz8 = invz4 * invz4;
    let k1_block = (1.0 / 48.0) * invz + (3.0 / 4.0) * invz3 + 12.0 * invz5 + 48.0 * invz7;
    let k0_block = (1.0 / 8.0) * invz2 + 3.0 * invz4 + 24.0 * invz6;
    48.0 * invz8 - k1 * k1_block - k0 * k0_block
}

#[inline(always)]
fn duchon_matern_p1_s4_k10_small_a_series(a: f64) -> f64 {
    // Cancellation-free small-a expansion:
    //
    // Start from:
    //   K(a) = (1/96) ∫_0^1 u^3 K0(a*sqrt(u)) du.
    //
    // Expand K0(z) for z->0:
    //   K0(z) = -log(z/2) - γ + O(z^2 log z),
    // then integrate term-by-term in u. This yields:
    //   K(a) =
    //     L/384 + 1/3072
    //     + a^2( L/1920 + 11/19200 )
    //     + a^4( L/36864 + 19/442368 )
    //     + a^6( L/1548288 + 5/4064256 )
    //     + a^8( L/113246208 + 103/5435817984 )
    //   where L = -log(a/2) - γ.
    //
    // Remainder after retained a^8 term: O(a^10 log a).
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let aa = a.max(1e-300);
    let l = -(aa * 0.5).ln() - EULER_GAMMA;
    let a2 = aa * aa;
    let a4 = a2 * a2;
    let a6 = a4 * a2;
    let a8 = a4 * a4;

    l / 384.0
        + 1.0 / 3072.0
        + a2 * (l / 1920.0 + 11.0 / 19_200.0)
        + a4 * (l / 36_864.0 + 19.0 / 442_368.0)
        + a6 * (l / 1_548_288.0 + 5.0 / 4_064_256.0)
        + a8 * (l / 113_246_208.0 + 103.0 / 5_435_817_984.0)
}

#[inline(always)]
#[cfg(test)]
fn duchon_matern_p1_s4_k10_integral(r: f64, kappa: f64) -> f64 {
    #[inline(always)]
    fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
        let c = 0.5 * (a + b);
        (b - a) * (f(a) + 4.0 * f(c) + f(b)) / 6.0
    }

    fn adaptive_simpson<F: Fn(f64) -> f64>(
        f: &F,
        a: f64,
        b: f64,
        eps: f64,
        whole: f64,
        depth: usize,
    ) -> f64 {
        let c = 0.5 * (a + b);
        let left = simpson(f, a, c);
        let right = simpson(f, c, b);
        let delta = left + right - whole;
        if depth == 0 || delta.abs() <= 15.0 * eps {
            // Richardson extrapolation.
            left + right + delta / 15.0
        } else {
            adaptive_simpson(f, a, c, eps * 0.5, left, depth - 1)
                + adaptive_simpson(f, c, b, eps * 0.5, right, depth - 1)
        }
    }

    let a = kappa * r;
    // Substitute u = t^2 to eliminate the sqrt in argument and make the
    // endpoint behavior smooth: du = 2 t dt, u^3 = t^6, so integrand is
    // 2 t^7 K0(a t) over t in [0,1].
    let f_t = |t: f64| -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        2.0 * t.powi(7) * bessel_k0_stable(a * t)
    };
    let whole = simpson(&f_t, 0.0, 1.0);
    let integral = adaptive_simpson(&f_t, 0.0, 1.0, 1e-10, whole, 20);
    integral / 96.0
}

fn duchon_matern_kernel_p1_s4_k10_from_distance(
    r: f64,
    length_scale: f64,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon-Matern kernel distance must be finite and non-negative".to_string(),
        ));
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(BasisError::InvalidInput(
            "Duchon-Matern length_scale must be finite and positive".to_string(),
        ));
    }
    if r == 0.0 {
        // Borderline regularity note for the current primary case:
        //   p=1, s=4, k=10  =>  p+s = k/2 (equivalently 2p+2s = k).
        // In this regime point-evaluation is not strictly proper (log-type
        // singularity at r->0 in the underlying generalized kernel). For the
        // spline penalty workflow we therefore use an intrinsic convention on
        // the diagonal and rely on constrained coefficients + lambda scaling.
        return Ok(0.0);
    }

    let kappa = 1.0 / length_scale;
    let z = kappa * r;
    // Numerically stable regime split:
    //
    // 1) z < 1: series branch.
    //    Avoids catastrophic cancellation in the closed form near the Sobolev
    //    boundary p+s=k/2 where the true behavior is logarithmic.
    //
    // 2) 1 <= z <= 50: exact closed form with K0/K1.
    //    Stable in this window and significantly faster than quadrature.
    //
    // 3) z > 50: asymptotic branch.
    //    Kν(z) terms are exponentially small, so dominant algebraic term is 48/z^8.
    //
    // We intentionally do NOT evaluate the raw Hankel/inverse-Fourier integral
    // in production:
    //   - it is oscillatory with slowly decaying tails,
    //   - per-entry quadrature cost is too high for O(M^2) kernel builds, and
    //   - numerical quadrature noise degrades K_CC conditioning and REML gradients.
    // The integral implementation is retained under #[cfg(test)] only as a
    // verification oracle against this branch-wise closed-form evaluator.
    if z < DUCHON_SERIES_Z_CUTOFF {
        Ok(duchon_matern_p1_s4_k10_small_a_series(z))
    } else if z > DUCHON_ASYMPTOTIC_Z_CUTOFF {
        Ok(duchon_matern_p1_s4_k10_asymptotic(z))
    } else {
        Ok(duchon_matern_p1_s4_k10_closed_form(r, kappa))
    }
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

    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
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
                row[j] = matern_kernel_from_distance(dist2.sqrt(), length_scale, nu)?;
            }
            Ok(())
        });
    kernel_result?;

    // Center-center Gram matrix K_CC. In RKHS form, the kernel penalty on
    // radial coefficients is alpha^T K_CC alpha.
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[i, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let kij = matern_kernel_from_distance(dist2.sqrt(), length_scale, nu)?;
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
    let penalty_ridge = nullspace_shrinkage_penalty(&penalty_kernel)?;

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
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let k = centers.nrows();
    let z_opt = match &spec.identifiability {
        MaternIdentifiability::None => None,
        MaternIdentifiability::CenterSumToZero => {
            let q = Array2::<f64>::ones((k, 1));
            Some(kernel_constraint_nullspace_from_matrix(q.view())?)
        }
        MaternIdentifiability::CenterLinearOrthogonal => {
            let q = polynomial_block_from_order(centers.view(), DuchonNullspaceOrder::Linear);
            Some(kernel_constraint_nullspace_from_matrix(q.view())?)
        }
        MaternIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != k {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen Matérn identifiability transform mismatch: centers={k}, transform rows={}",
                    transform.nrows()
                )));
            }
            Some(transform.clone())
        }
    };

    // Build an unconstrained kernel block first, then apply center-based
    // coefficient constraints in a deterministic way.
    let m = create_matern_spline_basis(data, centers.view(), spec.length_scale, spec.nu, false)?;
    let (design, penalty_kernel, identifiability_transform) = if let Some(z) = z_opt {
        let kernel_raw = m.basis.slice(s![.., 0..m.num_kernel_basis]).to_owned();
        let kernel_constrained = fast_ab(&kernel_raw, &z);
        let omega_raw = m
            .penalty_kernel
            .slice(s![0..m.num_kernel_basis, 0..m.num_kernel_basis])
            .to_owned();
        let omega_constrained = {
            let zt_k = fast_atb(&z, &omega_raw);
            fast_ab(&zt_k, &z)
        };
        let mut design = kernel_constrained;
        if spec.include_intercept {
            let n = design.nrows();
            let p = design.ncols();
            let mut design_with_intercept = Array2::<f64>::zeros((n, p + 1));
            design_with_intercept
                .slice_mut(s![.., 0..p])
                .assign(&design);
            design_with_intercept.column_mut(p).fill(1.0);
            design = design_with_intercept;
        }
        let total_cols = design.ncols();
        let mut penalty = Array2::<f64>::zeros((total_cols, total_cols));
        let kernel_cols = omega_constrained.ncols();
        penalty
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&omega_constrained);
        (design, penalty, Some(z))
    } else {
        (m.basis, m.penalty_kernel, None)
    };
    let penalty_ridge = nullspace_shrinkage_penalty(&penalty_kernel)?;
    let penalties = vec![if spec.double_penalty {
        with_fixed_shrinkage(&penalty_kernel, &penalty_ridge)
    } else {
        penalty_kernel.clone()
    }];
    let nullspace_dims = vec![if spec.double_penalty {
        0
    } else if spec.include_intercept {
        1
    } else {
        0
    }];
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        metadata: BasisMetadata::Matern {
            centers,
            length_scale: spec.length_scale,
            nu: spec.nu,
            include_intercept: spec.include_intercept,
            identifiability_transform,
        },
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
/// - `s` is determined by `nu`:
///   - `Half`         -> s = 1
///   - `ThreeHalves`  -> s = 2
///   - `FiveHalves`   -> s = 3
///   - `SevenHalves`  -> s = 4
///   - `NineHalves`   -> s = 5
///
/// For the historical primary case `(p=1, s=4, k=10)` we retain the exact
/// specialized branch (series + closed-form + asymptotic) for speed and
/// numerical stability near the Sobolev boundary.
pub fn create_duchon_spline_basis(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    nullspace_order: DuchonNullspaceOrder,
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
    let z = kernel_constraint_nullspace(centers, nullspace_order)?;

    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order = duchon_s_from_nu(nu);

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
    if let Some((r_min, r_max)) = pairwise_distance_bounds(centers) {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::warn!(
                "Duchon-Matern κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); numerical conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
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
                row[j] = duchon_matern_kernel_general_from_distance(
                    dist2.sqrt(),
                    length_scale,
                    p_order,
                    s_order,
                    d,
                )?;
            }
            Ok(())
        });
    kernel_result?;

    let mut center_kernel = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = centers[[i, c]] - centers[[j, c]];
                dist2 += delta * delta;
            }
            let kij = duchon_matern_kernel_general_from_distance(
                dist2.sqrt(),
                length_scale,
                p_order,
                s_order,
                d,
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
    let penalty_ridge = nullspace_shrinkage_penalty(&penalty_kernel)?;

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
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let d = create_duchon_spline_basis(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.nullspace_order,
    )?;
    let penalties = vec![if spec.double_penalty {
        with_fixed_shrinkage(&d.penalty_kernel, &d.penalty_ridge)
    } else {
        d.penalty_kernel.clone()
    }];
    let nullspace_dims = vec![if spec.double_penalty {
        0
    } else {
        d.num_polynomial_basis
    }];
    Ok(BasisBuildResult {
        design: d.basis,
        penalties,
        nullspace_dims,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            nu: spec.nu,
            nullspace_order: spec.nullspace_order,
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
    // Constraint system Q^T alpha = 0. The nullspace of Q^T gives feasible
    // kernel coefficients that cannot represent constrained polynomial content.
    let q_t = constraint_matrix.t().to_owned(); // q x k

    let (_, singular_values, vt_opt) = q_t.svd(false, true).map_err(BasisError::LinalgError)?;
    let vt = match vt_opt {
        Some(vt) => vt,
        None => return Err(BasisError::ConstraintNullspaceNotFound),
    };
    let v = vt.t().to_owned(); // k x k

    let max_sigma = singular_values
        .iter()
        .fold(0.0_f64, |max_val, &sigma| max_val.max(sigma));
    let tol = (k as f64) * 1e-12 * max_sigma.max(1.0);
    let rank = singular_values.iter().filter(|&&sigma| sigma > tol).count();
    let z = if rank >= k {
        Array2::<f64>::zeros((k, 0))
    } else {
        v.slice(s![.., rank..]).to_owned()
    };
    Ok(z)
}

fn kernel_constraint_nullspace(
    knots: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Result<Array2<f64>, BasisError> {
    let p_k = polynomial_block_from_order(knots, order);
    kernel_constraint_nullspace_from_matrix(p_k.view())
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

    for i in 0..n {
        let mut d2 = 0.0;
        for c in 0..d {
            let delta = data[[i, c]] - data[[seed_idx, c]];
            d2 += delta * delta;
        }
        min_dist2[i] = d2;
    }
    min_dist2[seed_idx] = 0.0;

    while selected.len() < num_knots {
        let mut best_idx = None;
        let mut best_dist2 = f64::NEG_INFINITY;
        for i in 0..n {
            if chosen[i] {
                continue;
            }
            let cand = min_dist2[i];
            if cand > best_dist2 {
                best_dist2 = cand;
                best_idx = Some(i);
            }
        }
        let next_idx = match best_idx {
            Some(i) => i,
            None => break,
        };
        selected.push(next_idx);
        chosen[next_idx] = true;

        for i in 0..n {
            if chosen[i] {
                continue;
            }
            let mut d2 = 0.0;
            for c in 0..d {
                let delta = data[[i, c]] - data[[next_idx, c]];
                d2 += delta * delta;
            }
            if d2 < min_dist2[i] {
                min_dist2[i] = d2;
            }
        }
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
        // m = 2 => 2m-d = 3 (odd): r^3 = (r^2) * r
        1 => Ok(dist2 * dist2.sqrt()),
        // m = 2 => 2m-d = 2 (even): r^2 log(r) = 0.5 * r^2 * log(r^2)
        2 => Ok(0.5 * dist2 * dist2.ln()),
        // m = 2 => 2m-d = 1 (odd): r = sqrt(r^2)
        3 => Ok(dist2.sqrt()),
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
    let z = kernel_constraint_nullspace(knots, DuchonNullspaceOrder::Linear)?;
    let kernel_constrained = fast_ab(&kernel_block, &z);
    let omega_constrained = {
        let zt_o = fast_atb(&z, &omega);
        fast_ab(&zt_o, &z)
    };

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
    let penalty_ridge = nullspace_shrinkage_penalty(&penalty_bending)?;

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
    let knots = select_thin_plate_knots(data, num_knots)?;
    let basis = create_thin_plate_spline_basis(data, knots.view())?;
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

    // Orthonormal basis for nullspace of c^T.
    // Build a k×1 matrix and compute its SVD; the trailing columns of U after
    // the numerical rank span the nullspace.
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(&c);

    use crate::faer_ndarray::FaerSvd;
    let (u_opt, singular_values, _) = c_mat.svd(true, false).map_err(BasisError::LinalgError)?;
    let u = match u_opt {
        Some(u) => u,
        None => return Err(BasisError::ConstraintNullspaceNotFound),
    };
    let max_sigma = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (k.max(1) as f64) * f64::EPSILON * max_sigma.max(1.0);
    let rank = singular_values
        .iter()
        .filter(|&&sigma| sigma.abs() > tol)
        .count();
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }
    if rank == 0 {
        // Already orthogonal to the intercept constraint; keep full basis unchanged.
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }
    let z = u.slice(s![.., rank..]).to_owned();

    // Constrained basis
    let constrained = basis_matrix.dot(&z);
    Ok((constrained, z))
}

/// Reparameterizes a basis matrix so its columns are orthogonal (with optional weights)
/// to a supplied constraint matrix.
///
/// Given a basis `B` (n×k), a constraint matrix `Z` (n×q), and optional observation weights
/// `w`, this function returns a new basis `B_c = B K` where the columns of `B_c` satisfy
/// `(B_c)^T W Z = 0`. The transformation matrix `K` spans the nullspace of `B^T W Z`, so the
/// constrained basis cannot express any function correlated with the provided constraints.
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

    let constraint_cross = basis_matrix.t().dot(&weighted_constraints); // k×q

    let constraint_cross_t = constraint_cross.t().to_owned();

    let (_, singular_values, vt_opt) = constraint_cross_t
        .svd(false, true)
        .map_err(BasisError::LinalgError)?;
    let vt = match vt_opt {
        Some(vt) => vt,
        None => return Err(BasisError::ConstraintNullspaceNotFound),
    };
    let v = vt.t().to_owned();

    let max_sigma = singular_values
        .iter()
        .fold(0.0_f64, |max_val, &sigma| max_val.max(sigma));
    let tol = if max_sigma > 0.0 {
        (k.max(q) as f64) * 1e-12 * max_sigma
    } else {
        1e-12
    };
    let rank = singular_values.iter().filter(|&&sigma| sigma > tol).count();

    let total_cols = v.ncols();
    if rank >= total_cols {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    let transform = v.slice(s![.., rank..]).to_owned();

    if transform.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    let constrained_basis = basis_matrix.dot(&transform);
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

    // 4. SVD to find nullspace of C_geom
    // C_geom is 2×k, so we want right singular vectors corresponding to zero singular values
    use crate::faer_ndarray::FaerSvd;
    let (_, singular_values, vt_opt) = c_geom.svd(false, true).map_err(BasisError::LinalgError)?;
    let vt = match vt_opt {
        Some(vt) => vt,
        None => return Err(BasisError::ConstraintNullspaceNotFound),
    };

    // 5. Identify nullspace columns (singular values ≈ 0)
    let max_sigma = singular_values.iter().fold(0.0_f64, |a, &b| a.max(b));
    let tol = (k as f64) * 1e-12 * max_sigma.max(1.0);
    let rank = singular_values.iter().filter(|&&s| s > tol).count();

    if rank >= k {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    // 6. Z = columns of V corresponding to near-zero singular values
    // V = Vt^T, and we want columns rank..k
    let v = vt.t();
    let z = v.slice(s![.., rank..]).to_owned();

    if z.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceNotFound);
    }

    // 7. Build raw penalty and project: S_c = Z' S Z
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
}

impl SplineScratch {
    pub fn new(degree: usize) -> Self {
        Self {
            inner: internal::BsplineScratch::new(degree),
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

        out[i] = k * (left_term - right_term);
    }

    Ok(())
}

/// Evaluates B-spline second derivatives at a single scalar point `x` into a provided buffer.
///
/// Uses the derivative recursion:
/// B''_{i,k}(x) = k * (B'_{i,k-1}(x)/(t_{i+k}-t_i) - B'_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
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

// Unit tests are crucial for a mathematical library like this.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

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
    fn test_build_thin_plate_basis_double_penalty_outputs_two_blocks() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
            double_penalty: true,
        };
        let result = build_thin_plate_basis(data.view(), &spec).unwrap();
        assert_eq!(result.penalties.len(), 1);
        assert_eq!(result.nullspace_dims, vec![0]);
        assert_eq!(result.design.nrows(), data.nrows());
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
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::KMeans {
                    num_centers: 4,
                    max_iter: 5,
                },
                double_penalty: false,
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::UniformGrid { points_per_dim: 2 },
                double_penalty: false,
            },
            ThinPlateBasisSpec {
                center_strategy: CenterStrategy::UserProvided(array![
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]
                ]),
                double_penalty: false,
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
        assert_eq!(result.nullspace_dims, vec![1, 0]);
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
    fn test_sparse_derivatives_are_zero_outside_domain() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3usize;
        let support = degree + 1;
        let mut scratch = BasisEvalScratch::new(degree);
        let mut d1 = vec![0.0; support];
        let mut d2 = vec![0.0; support];

        let _ = evaluate_splines_derivative_sparse_into(
            -10.0,
            degree,
            knots.view(),
            &mut d1,
            &mut scratch,
        );
        assert!(d1.iter().all(|v| v.abs() < 1e-12));
        let _ = evaluate_splines_derivative_sparse_into(
            10.0,
            degree,
            knots.view(),
            &mut d1,
            &mut scratch,
        );
        assert!(d1.iter().all(|v| v.abs() < 1e-12));

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
            1.0,
            MaternNu::Half,
            DuchonNullspaceOrder::Linear,
        )
        .expect("primary Duchon-Matern case should build");
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
            1.0,
            MaternNu::Half,
            DuchonNullspaceOrder::Linear,
        )
        .expect("general integer (p,s,k) Duchon kernel should build");
        assert_eq!(out.dimension, 3);
        assert_eq!(out.basis.nrows(), 4);
        assert_eq!(out.penalty_kernel.nrows(), out.penalty_kernel.ncols());
    }

    #[test]
    fn test_duchon_primary_kernel_branch_policy() {
        // For z < 1, runtime should use the small-a series branch.
        let z_small = 0.1;
        let k_boundary_expected = duchon_matern_p1_s4_k10_small_a_series(z_small);
        let k_boundary_actual = duchon_matern_kernel_p1_s4_k10_from_distance(z_small, 1.0).unwrap();
        let rel =
            (k_boundary_actual - k_boundary_expected).abs() / k_boundary_expected.abs().max(1e-12);
        assert!(
            rel < 1e-10,
            "small-a path should use series branch: expected={}, actual={}, rel={}",
            k_boundary_expected,
            k_boundary_actual,
            rel
        );

        // In the closed-form branch regime z>=1, result should stay finite.
        let k_closed = duchon_matern_kernel_p1_s4_k10_from_distance(1.0, 1.0).unwrap();
        assert!(k_closed.is_finite());
    }

    #[test]
    fn test_duchon_closed_form_matches_integral_at_moderate_scale() {
        let kappa = 1.0;
        let r = 1.0; // z=1, safely in closed-form branch
        let k_int = duchon_matern_p1_s4_k10_integral(r, kappa);
        let k_cf = duchon_matern_p1_s4_k10_closed_form(r, kappa);
        let rel = (k_int - k_cf).abs() / k_int.abs().max(1e-12);
        assert!(
            rel < 2e-2,
            "closed form should match integral at moderate scale: int={}, cf={}, rel={}",
            k_int,
            k_cf,
            rel
        );
    }

    #[test]
    fn test_duchon_asymptotic_matches_closed_form_large_z() {
        let z = DUCHON_ASYMPTOTIC_Z_CUTOFF;
        let k_cf = duchon_matern_p1_s4_k10_closed_form(z, 1.0);
        let k_asym = duchon_matern_p1_s4_k10_asymptotic(z);
        let rel = (k_cf - k_asym).abs() / k_cf.abs().max(1e-14);
        assert!(
            rel < 1e-8,
            "asymptotic shortcut should match closed form at cutoff: cf={}, asym={}, rel={}",
            k_cf,
            k_asym,
            rel
        );
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
            0.9,
            MaternNu::NineHalves,         // s=5
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
        assert_eq!(out.nullspace_dims, vec![1]);
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
            1.2,
            MaternNu::SevenHalves,      // s=4
            DuchonNullspaceOrder::Zero, // p=0
        )
        .expect("p=0 Duchon case should build");
        assert_eq!(out.num_polynomial_basis, 0);
        assert!(out.penalty_kernel.iter().all(|v| v.is_finite()));
    }
}
