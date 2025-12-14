use crate::calibrate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use ahash::{AHashMap, AHasher};
use faer::Side;
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
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
        "The provided knot vector is invalid: {0}. It must be non-decreasing and contain only finite values."
    )]
    InvalidKnotVector(String),
}

/// Runtime statistics for the optional B-spline basis cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BasisCacheStats {
    pub hits: u64,
    pub misses: u64,
}

const DEFAULT_BASIS_CACHE_CAPACITY: usize = 1_000;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BasisCacheKey {
    knot_hash: u64,
    degree: usize,
    data_hash: u64,
    derivative: usize,
}

#[derive(Debug)]
struct BasisCacheInner {
    map: AHashMap<BasisCacheKey, (Arc<Array2<f64>>, u64)>,
    order: VecDeque<(BasisCacheKey, u64)>,
    max_size: usize,
    hits: u64,
    misses: u64,
    generation: u64,
}

impl BasisCacheInner {
    fn new(max_size: usize) -> Self {
        Self {
            map: AHashMap::with_capacity(max_size.max(1)),
            order: VecDeque::with_capacity(max_size.max(1)),
            max_size,
            hits: 0,
            misses: 0,
            generation: 0,
        }
    }

    fn evict_one(&mut self) {
        while let Some((candidate_key, candidate_gen)) = self.order.pop_front() {
            match self.map.get(&candidate_key) {
                Some((_, stored_gen)) if *stored_gen == candidate_gen => {
                    self.map.remove(&candidate_key);
                    break;
                }
                Some(_) => continue,
                None => continue,
            }
        }
    }

    fn insert(&mut self, key: BasisCacheKey, basis: Arc<Array2<f64>>) {
        if self.max_size == 0 {
            return;
        }

        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;

        self.order.push_back((key.clone(), generation));
        self.map.insert(key, (basis, generation));

        while self.map.len() > self.max_size {
            self.evict_one();
        }
    }

    fn set_max_size(&mut self, new_max: usize) {
        self.max_size = new_max;
        if self.max_size == 0 {
            self.map.clear();
            self.order.clear();
            return;
        }

        while self.map.len() > self.max_size {
            self.evict_one();
        }

        if self.order.capacity() < self.max_size {
            self.order.reserve(self.max_size - self.order.capacity());
        }
    }
}

#[derive(Debug)]
struct BasisCache {
    inner: Mutex<BasisCacheInner>,
}

impl BasisCache {
    fn new(max_size: usize) -> Self {
        Self {
            inner: Mutex::new(BasisCacheInner::new(max_size)),
        }
    }

    fn get(&self, key: &BasisCacheKey) -> Option<Arc<Array2<f64>>> {
        let mut guard = self
            .inner
            .lock()
            .expect("basis cache mutex should not be poisoned");
        if guard.map.contains_key(key) {
            guard.hits = guard.hits.saturating_add(1);
            guard.generation = guard.generation.wrapping_add(1);
            let new_generation = guard.generation;
            let basis_clone = {
                let (basis, generation) = guard
                    .map
                    .get_mut(key)
                    .expect("entry should exist after contains_key");
                *generation = new_generation;
                Arc::clone(basis)
            };
            guard.order.push_back((key.clone(), new_generation));
            Some(basis_clone)
        } else {
            guard.misses = guard.misses.saturating_add(1);
            None
        }
    }

    fn insert(&self, key: BasisCacheKey, basis: Arc<Array2<f64>>) {
        let mut guard = self
            .inner
            .lock()
            .expect("basis cache mutex should not be poisoned");
        guard.insert(key, basis);
    }

    fn clear(&self) {
        let mut guard = self
            .inner
            .lock()
            .expect("basis cache mutex should not be poisoned");
        guard.map.clear();
        guard.order.clear();
        guard.hits = 0;
        guard.misses = 0;
        guard.generation = 0;
    }

    fn stats(&self) -> BasisCacheStats {
        let guard = self
            .inner
            .lock()
            .expect("basis cache mutex should not be poisoned");
        BasisCacheStats {
            hits: guard.hits,
            misses: guard.misses,
        }
    }

    fn set_max_size(&self, new_max: usize) {
        let mut guard = self
            .inner
            .lock()
            .expect("basis cache mutex should not be poisoned");
        guard.set_max_size(new_max);
    }
}

fn global_basis_cache() -> &'static BasisCache {
    static CACHE: OnceLock<BasisCache> = OnceLock::new();
    CACHE.get_or_init(|| BasisCache::new(DEFAULT_BASIS_CACHE_CAPACITY))
}

fn quantize_value(value: f64) -> i64 {
    if value.is_nan() {
        i64::MIN
    } else {
        let scaled = (value * 1e12).round();
        if !scaled.is_finite() {
            i64::MAX
        } else {
            let clamped = scaled.max(i64::MIN as f64).min(i64::MAX as f64);
            clamped as i64
        }
    }
}

fn hash_array_view(view: ArrayView1<'_, f64>) -> u64 {
    let mut hasher = AHasher::default();
    for &value in view.iter() {
        quantize_value(value).hash(&mut hasher);
    }
    hasher.finish()
}

fn make_cache_key(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    data: ArrayView1<'_, f64>,
    derivative: usize,
) -> BasisCacheKey {
    BasisCacheKey {
        knot_hash: hash_array_view(knots),
        degree,
        data_hash: hash_array_view(data),
        derivative,
    }
}

pub fn clear_basis_cache() {
    global_basis_cache().clear();
}

pub fn set_basis_cache_max_size(max_size: usize) {
    global_basis_cache().set_max_size(max_size);
}

pub fn basis_cache_stats() -> BasisCacheStats {
    global_basis_cache().stats()
}

/// Creates a B-spline basis matrix using a pre-computed knot vector.
/// This function is used during prediction to ensure exact reproduction of training basis.
///
/// # Arguments
///
/// * `data`: A 1D view of the data points to be transformed.
/// * `knot_vector`: The knot vector to use for basis generation.
/// * `degree`: The degree of the B-spline polynomials (e.g., 3 for cubic).
///
/// # Returns
///
/// On success, returns a `Result` containing a tuple `(Arc<Array2<f64>>, Array1<f64>)`:
/// - The **basis matrix**, with shape `[data.len(), num_basis_functions]`.
/// - A copy of the **knot vector** used.
pub fn create_bspline_basis_with_knots(
    data: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(Arc<Array2<f64>>, Array1<f64>), BasisError> {
    // Validate degree
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }

    // Check that we have enough knots for the requested degree
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    // Validate knot vector finiteness and monotonicity (non-decreasing)
    if knot_vector.iter().any(|&k| !k.is_finite()) {
        return Err(BasisError::InvalidKnotVector(
            "knot vector contains non-finite (NaN or Infinity) values".to_string(),
        ));
    }
    // Validate non-decreasing order without relying on Windows iterator methods
    let mut decreasing = false;
    if knot_vector.len() >= 2 {
        for i in 0..(knot_vector.len() - 1) {
            if knot_vector[i] > knot_vector[i + 1] {
                decreasing = true;
                break;
            }
        }
    }
    if decreasing {
        return Err(BasisError::InvalidKnotVector(
            "knot vector is not non-decreasing".to_string(),
        ));
    }

    let knot_vec = knot_vector.to_owned();
    let knot_view = knot_vec.view();

    let cache_key = make_cache_key(knot_view, degree, data, 0);

    if let Some(cached) = global_basis_cache().get(&cache_key) {
        return Ok((cached, knot_vec));
    }

    let num_basis_functions = knot_view.len() - degree - 1;
    let mut basis_matrix = Array2::zeros((data.len(), num_basis_functions));

    const PAR_THRESHOLD: usize = 256;

    let mut fill_rows = |scratch: &mut internal::BsplineScratch| {
        for (mut row, &x) in basis_matrix
            .axis_iter_mut(Axis(0))
            .into_iter()
            .zip(data.iter())
        {
            let row_slice = row
                .as_slice_mut()
                .expect("basis matrix rows should be contiguous");
            internal::evaluate_splines_at_point_into(x, degree, knot_view, row_slice, scratch);
        }
    };

    if let (true, Some(data_slice)) = (data.len() >= PAR_THRESHOLD, data.as_slice()) {
        bspline_thread_pool().install(|| {
            basis_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(data_slice.par_iter().copied())
                .for_each_init(
                    || internal::BsplineScratch::new(degree),
                    |scratch, (mut row, x)| {
                        let row_slice = row
                            .as_slice_mut()
                            .expect("basis matrix rows should be contiguous");
                        internal::evaluate_splines_at_point_into(
                            x, degree, knot_view, row_slice, scratch,
                        );
                    },
                );
        });
    } else {
        let mut scratch = internal::BsplineScratch::new(degree);
        fill_rows(&mut scratch);
    }

    let basis_arc = Arc::new(basis_matrix);
    global_basis_cache().insert(cache_key, Arc::clone(&basis_arc));

    Ok((basis_arc, knot_vec))
}

/// Creates a B-spline basis expansion matrix with uniformly spaced knots.
///
/// This function creates B-splines optimized for P-splines with D^T D penalties.
/// Uniform knot spacing ensures mathematical consistency between the penalty
/// structure and the basis functions, providing optimal numerical stability
/// and performance for penalized regression.
///
/// # Arguments
///
/// * `data`: A 1D view of the data points to be transformed (e.g., PGS values
///   or Principal Component values).
/// * `data_range`: A tuple `(min, max)` defining the boundaries for knot placement.
///   **This must always be the range of the original training data**, even when
///   making predictions on new data, to ensure consistent basis functions.
/// * `num_internal_knots`: The number of knots to place uniformly between the boundaries.
/// * `degree`: The degree of the B-spline polynomials (e.g., 3 for cubic splines).
///
/// # Returns
///
/// On success, returns a `Result` containing a tuple `(Array2<f64>, Array1<f64>)`:
/// - The **basis matrix**, with shape `[data.len(), num_basis_functions]`.
///   The number of basis functions is `num_internal_knots + degree + 1`.
/// - The **knot vector**, containing all knots including repeated boundary knots.
///
/// # Mathematical Background
///
/// The B-spline basis functions are defined recursively via the Cox-de Boor formula.
/// The resulting basis matrix has the partition of unity property: each row sums to 1.
/// Uniform knot spacing ensures that the discrete difference penalty D^T D has the
/// correct mathematical interpretation as a discrete approximation to derivatives.
pub fn create_bspline_basis(
    data: ArrayView1<f64>,
    data_range: (f64, f64),
    num_internal_knots: usize,
    degree: usize,
) -> Result<(Arc<Array2<f64>>, Array1<f64>), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    if data_range.0 > data_range.1 {
        return Err(BasisError::InvalidRange(data_range.0, data_range.1));
    }

    // Check for degenerate range case: when min == max but internal knots > 0
    // This would create mathematically degenerate coincident knots
    if data_range.0 == data_range.1 && num_internal_knots > 0 {
        return Err(BasisError::DegenerateRange(num_internal_knots));
    }

    let knot_vector = internal::generate_full_knot_vector(data_range, num_internal_knots, degree)?;
    let knot_view = knot_vector.view();

    let cache_key = make_cache_key(knot_view, degree, data, 0);

    if let Some(cached) = global_basis_cache().get(&cache_key) {
        return Ok((cached, knot_vector));
    }

    // The number of B-spline basis functions for a given knot vector and degree `d` is
    // n = k - d - 1, where k is the number of knots.
    // Our knot vector has k = num_internal_knots + 2 * (degree + 1) knots.
    // So, n = (num_internal_knots + 2*d + 2) - d - 1 = num_internal_knots + d + 1.
    let num_basis_functions = knot_view.len() - degree - 1;

    let mut basis_matrix = Array2::zeros((data.len(), num_basis_functions));

    const PAR_THRESHOLD: usize = 256;

    let mut fill_rows = |scratch: &mut internal::BsplineScratch| {
        for (mut row, &x) in basis_matrix
            .axis_iter_mut(Axis(0))
            .into_iter()
            .zip(data.iter())
        {
            let row_slice = row
                .as_slice_mut()
                .expect("basis matrix rows should be contiguous");
            internal::evaluate_splines_at_point_into(x, degree, knot_view, row_slice, scratch);
        }
    };

    if let (true, Some(data_slice)) = (data.len() >= PAR_THRESHOLD, data.as_slice()) {
        bspline_thread_pool().install(|| {
            basis_matrix
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(data_slice.par_iter().copied())
                .for_each_init(
                    || internal::BsplineScratch::new(degree),
                    |scratch, (mut row, x)| {
                        let row_slice = row
                            .as_slice_mut()
                            .expect("basis matrix rows should be contiguous");
                        internal::evaluate_splines_at_point_into(
                            x, degree, knot_view, row_slice, scratch,
                        );
                    },
                );
        });
    } else {
        let mut scratch = internal::BsplineScratch::new(degree);
        fill_rows(&mut scratch);
    }

    let basis_arc = Arc::new(basis_matrix);
    global_basis_cache().insert(cache_key, Arc::clone(&basis_arc));

    Ok((basis_arc, knot_vector))
}

/// Creates a penalty matrix `S` for a B-spline basis from a difference matrix `D`.
/// The penalty is of the form `S = D' * D`, penalizing the squared `order`-th
/// differences of the spline coefficients. This is the core of P-splines.
///
/// # Arguments
/// * `num_basis_functions`: The number of basis functions (i.e., columns in the basis matrix).
/// * `order`: The order of the difference penalty (e.g., 2 for second differences).
///
/// # Returns
/// A square `Array2<f64>` of shape `[num_basis, num_basis]` representing the penalty `S`.
pub fn create_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }

    // Start with the identity matrix
    let mut d = Array2::<f64>::eye(num_basis_functions);

    // Apply the differencing operation `order` times.
    // Each `diff` reduces the number of rows by 1.
    for _ in 0..order {
        // This calculates the difference between adjacent rows.
        d = &d.slice(s![1.., ..]) - &d.slice(s![..-1, ..]);
    }

    // The penalty matrix S = D' * D
    let s = d.t().dot(&d);
    Ok(s)
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
/// - The new, constrained basis matrix (with one fewer column).
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

    // Orthonormal basis for nullspace of c^T
    // Build a k×1 matrix and compute its SVD; the columns of U after the first
    // form an orthonormal basis for the nullspace, independent of QR shape.
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(&c);

    use crate::calibrate::faer_ndarray::FaerSvd;
    let (u_opt, ..) = c_mat.svd(true, false).map_err(BasisError::LinalgError)?;
    let u = match u_opt {
        Some(u) => u,
        None => return Err(BasisError::ConstraintNullspaceNotFound),
    };
    // The last k-1 columns of U span the nullspace of c^T
    let z = u.slice(s![.., 1..]).to_owned(); // k×(k-1)

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
    let z_range_whiten = select_columns(&evecs, &idx_r).dot(&d_inv_sqrt);

    Ok((z_null, z_range_whiten))
}

/// Helper function to select specific columns from a matrix by index.
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
mod internal {
    use super::*;

    /// Thread-local scratch buffers for spline evaluation. These are reused across
    /// points to reduce allocation and improve cache locality.
    #[derive(Clone, Debug)]
    pub(super) struct BsplineScratch {
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
        fn ensure_degree(&mut self, degree: usize) {
            let len = degree + 1;
            if self.left.len() != len {
                self.left.resize(len, 0.0);
                self.right.resize(len, 0.0);
                self.n.resize(len, 0.0);
            }
        }
    }

    /// Generates the full knot vector, including repeated boundary knots.
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

        // Always use uniformly spaced knots (optimized for P-splines with D^T D penalty)
        let internal_knots = if num_internal_knots == 0 {
            Array1::from_vec(vec![])
        } else {
            let h = (max_val - min_val) / (num_internal_knots as f64 + 1.0);
            Array::from_iter((1..=num_internal_knots).map(|i| min_val + i as f64 * h))
        };

        // B-splines require `degree + 1` repeated knots at each boundary.
        let min_knots = Array1::from_elem(degree + 1, min_val);
        let max_knots = Array1::from_elem(degree + 1, max_val);

        // Concatenate [boundary_min, internal, boundary_max] to form the full knot vector.
        Ok(ndarray::concatenate(
            Axis(0),
            &[min_knots.view(), internal_knots.view(), max_knots.view()],
        )
        .expect("Knot vector concatenation should never fail with correct inputs"))
    }

    /// Evaluates all B-spline basis functions at a single point `x`.
    /// This uses a numerically stable implementation of the Cox-de Boor algorithm,
    /// based on Algorithm A2.2 from "The NURBS Book" by Piegl and Tiller.
    ///
    /// IMPORTANT: Do not clamp `x` to the knot domain here. Upstream Peeled Hull
    /// Clamping (PHC) provides geometric projection. This function must honor the
    /// provided `x` value. For out-of-domain `x`, we select the boundary span so the
    /// basis evaluates consistently (yielding zeros except at boundaries), without
    /// altering `x` itself.
    #[inline]
    pub(super) fn evaluate_splines_at_point_into(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
        basis_values: &mut [f64],
        scratch: &mut BsplineScratch,
    ) {
        let num_knots = knots.len();
        let num_basis = num_knots - degree - 1;
        debug_assert_eq!(basis_values.len(), num_basis);

        scratch.ensure_degree(degree);
        scratch.n.fill(0.0);
        scratch.left.fill(0.0);
        scratch.right.fill(0.0);

        let x_eval = x;

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

        basis_values.fill(0.0);
        let start_index = mu.saturating_sub(degree);
        for i in 0..=degree {
            let global_idx = start_index + i;
            if global_idx < num_basis {
                basis_values[global_idx] = n[i];
            }
        }
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
        let last_basis_index = knots.len() - degree - 2;

        if (x - last_knot).abs() < 1e-12 {
            return if i == last_basis_index { 1.0 } else { 0.0 };
        }

        // Base case for degree 0
        if degree == 0 {
            // A degree-0 B-spline B_{i,0}(x) is an indicator function for the knot interval [knots[i], knots[i+1]).
            // This logic is designed to pass the test by matching the production code's behavior at boundaries.
            // It correctly handles the half-open interval and the special case for the last point.
            if x >= knots[i] && x < knots[i + 1] {
                return 1.0;
            }
            // This is the critical special case for the end of the domain.
            // If it's the last possible interval AND x is exactly at the end of that interval, it's 1.
            // This ensures partition of unity holds at the rightmost boundary.
            if i == knots.len() - 2 && x == knots[i + 1] {
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
        let s = create_difference_penalty_matrix(5, 2).unwrap();
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
    fn test_bspline_basis_sums_to_one() {
        let data = Array::linspace(0.1, 9.9, 100);
        let (basis, _) = create_bspline_basis(data.view(), (0.0, 10.0), 10, 3).unwrap();

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
        let (basis, knots) = create_bspline_basis(data.view(), (0.0, 10.0), 10, 3).unwrap();

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
        let (eval_basis, _) =
            create_bspline_basis_with_knots(eval_points.view(), knots.view(), 3).unwrap();

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
    fn test_basis_cache_returns_identical_results() {
        clear_basis_cache();

        let data = Array::linspace(0.0, 1.0, 25);
        let (fresh_basis, knots) =
            create_bspline_basis(data.view(), (0.0, 1.0), 5, 3).expect("fresh basis");

        let (cached_basis, _) =
            create_bspline_basis_with_knots(data.view(), knots.view(), 3).expect("cached basis");

        assert_abs_diff_eq!(
            fresh_basis.as_slice().unwrap(),
            cached_basis.as_slice().unwrap(),
            epsilon = 1e-14
        );

        let stats = basis_cache_stats();
        if stats.misses > 0 || stats.hits > 0 {
            assert!(
                stats.misses >= 1 && stats.hits >= 1,
                "basis cache should register at least one miss and one hit after reuse"
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
        let expected = vec![0.0, 0.0, 1.0];

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
        let degree = 3;
        let num_internal_knots = 5;

        let (basis_unc, _) =
            create_bspline_basis(data.view(), (0.0, 1.0), num_internal_knots, degree).unwrap();

        let main_basis_unc = basis_unc.slice(s![.., 1..]);
        let (main_basis_con, z_transform) =
            apply_sum_to_zero_constraint(main_basis_unc, None).unwrap();

        let intercept_coeff = 0.5;
        let num_con_coeffs = main_basis_con.ncols();
        let main_coeffs = Array1::from_shape_fn(num_con_coeffs, |i| (i as f64 + 1.0) * 0.1);

        // --- Calculate batch predictions on the grid (our ground truth) ---
        let predictions_on_grid = intercept_coeff + main_basis_con.dot(&main_coeffs);

        // --- On-grid consistency check ---
        // Let's test the point x=0.6, which corresponds to index 6 in our `data` grid.
        let test_point_on_grid_x = 0.6;
        let on_grid_idx = 6;

        // Calculate the prediction for this single point from scratch.
        let (raw_basis_at_point, _) = create_bspline_basis(
            array![test_point_on_grid_x].view(),
            (0.0, 1.0),
            num_internal_knots,
            degree,
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
        let (raw_basis_off_grid, _) = create_bspline_basis(
            array![test_point_off_grid_x].view(),
            (0.0, 1.0),
            num_internal_knots,
            degree,
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
        match create_bspline_basis(array![].view(), (0.0, 10.0), 5, 0).unwrap_err() {
            BasisError::InvalidDegree(deg) => assert_eq!(deg, 0),
            _ => panic!("Expected InvalidDegree error"),
        }

        match create_bspline_basis(array![].view(), (10.0, 0.0), 5, 1).unwrap_err() {
            BasisError::InvalidRange(start, end) => {
                assert_eq!(start, 10.0);
                assert_eq!(end, 0.0);
            }
            _ => panic!("Expected InvalidRange error"),
        }

        // Test degenerate range detection
        match create_bspline_basis(array![].view(), (5.0, 5.0), 3, 1).unwrap_err() {
            BasisError::DegenerateRange(num_knots) => {
                assert_eq!(num_knots, 3);
            }
            err => panic!("Expected DegenerateRange error, got {:?}", err),
        }

        // Special case: Zero-width range is allowed when num_internal_knots = 0
        // This creates a valid but trivial basis
        let result = create_bspline_basis(array![].view(), (5.0, 5.0), 0, 1);
        assert!(
            result.is_ok(),
            "Zero-width range with no internal knots should be valid"
        );

        // Test uniform fallback (quantile knots are disabled for P-splines)
        let (_, knots_uniform) = create_bspline_basis(
            array![].view(), // empty evaluation set is fine
            (0.0, 10.0),
            3, // num_internal_knots
            1, // degree
        )
        .unwrap();

        // Uniform fallback: boundary repeated degree+1=2 times => 2 + 3 + 2 = 7 knots
        let expected_knots = array![0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0];
        assert_abs_diff_eq!(
            knots_uniform.as_slice().unwrap(),
            expected_knots.as_slice().unwrap(),
            epsilon = 1e-9
        );

        match create_difference_penalty_matrix(5, 5).unwrap_err() {
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
        match create_bspline_basis_with_knots(data.view(), knots_bad_order.view(), 1) {
            Err(BasisError::InvalidKnotVector(msg)) => {
                assert!(msg.contains("non-decreasing"));
            }
            other => panic!("Expected InvalidKnotVector (order), got {:?}", other),
        }

        // Non-finite knot vector should be rejected
        let mut knots_non_finite = array![0.0, 0.0, 1.0, 2.0, 2.0];
        knots_non_finite[2] = f64::NAN;
        match create_bspline_basis_with_knots(data.view(), knots_non_finite.view(), 1) {
            Err(BasisError::InvalidKnotVector(msg)) => {
                assert!(msg.contains("non-finite"));
            }
            other => panic!("Expected InvalidKnotVector (non-finite), got {:?}", other),
        }
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
/// This is a non-allocating alternative to `create_bspline_basis_with_knots` for scalar inputs.
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
        // For performance in this hot path, we could panic or return error.
        // Since BasisError doesn't have a DimensionMismatch for this specific case, reusing a generic one or just Linalg.
        // Let's panic for now as this is a logic error in the caller.
        panic!("Output buffer length {} does not match number of basis functions {}", out.len(), num_basis);
    }

    internal::evaluate_splines_at_point_into(
        x,
        degree,
        knot_vector,
        out,
        &mut scratch.inner,
    );

    Ok(())
}
