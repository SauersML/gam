
pub use constant_curvature_smooth::{
    ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, build_constant_curvature_basis,
    build_constant_curvature_basis_kappa_derivatives, constant_curvature_kernel_kappa_jets,
    constant_curvature_kernel_matrix, realized_constant_curvature_length_scale,
};

pub use measure_jet_moments::{
    MeasureJetJetStats, MeasureJetMomentTable, accumulate_moment_table, jet_sufficient_stats,
    merge_moment_tables, recenter_moment_table,
};

pub use measure_jet_predict::{
    MeasureJetExtrapolationSpectrum, measure_jet_extrapolation_variance,
};

pub use measure_jet_smooth::{
    MeasureJetBand, MeasureJetBasisSpec, MeasureJetEnergyJets, MeasureJetFrozenQuadrature,
    MeasureJetIdentifiability, build_measure_jet_basis, build_measure_jet_basis_psi_derivatives,
    measure_jet_band, measure_jet_center_masses, measure_jet_design_matrix,
    measure_jet_energy_form, measure_jet_energy_form_with_jets, measure_jet_energy_forms_per_scale,
    measure_jet_multiscale_mode, measure_jet_quadrature_nodes, measure_jet_scale_spectrum,
    measure_jet_support_curve, realized_measure_jet_length_scale,
};

pub use sphere_spec::{
    SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
};


pub use cyclic::{
    create_closure_difference_penalty_jet, create_cyclic_difference_penalty_matrix,
    create_open_difference_penalty_matrix,
};

pub(crate) use cyclic::{
    create_cyclic_bspline_basis_dense, cyclic_distance_1d, cyclic_uniform_knot_vector,
    wrap_to_period,
};


/// Absolute floor below which a B-spline knot span (`t_{i+k} - t_i`) is treated
/// as degenerate: the corresponding Cox–de Boor / derivative-recurrence
/// denominator is then skipped (its term contributes zero), and a zero-support
/// basis function is rejected. Set well above `f64::EPSILON` so that knot
/// vectors with near-coincident knots are caught before the division amplifies
/// rounding noise, yet far below any meaningful covariate-scale knot spacing.
const KNOT_SPAN_DEGENERACY_FLOOR: f64 = 1e-12;


/// Absolute distance by which a covariate value must lie outside the clamped
/// B-spline domain before the linear extrapolation correction is applied; below
/// this the point is treated as on-boundary and no extrapolation term is added.
const BSPLINE_EXTRAPOLATION_THRESHOLD: f64 = 1e-12;


/// Default number of rows in each block the streaming design evaluators
/// materialize at a time when the caller does not supply an explicit chunk
/// size. Bounds the transient working set (one `chunk_rows × p` dense block)
/// while staying large enough to amortize per-chunk kernel-column setup.
const DEFAULT_STREAMING_CHUNK_ROWS: usize = 2048;


/// Wrapper to send a raw pointer across thread boundaries for parallel buffer fills.
/// SAFETY: every `SendPtr` value must be built from live, properly aligned `f64`
/// storage whose mutable borrow is held until all worker threads finish; callers
/// may only dereference offsets that are in-bounds and disjoint across workers.
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);

// SAFETY: SendPtr only grants raw-pointer transport. Actual dereferences occur
// at call sites after row-chunk partitioning proves each thread writes a
// distinct in-bounds element of the backing Array/Vec allocation.
unsafe impl Send for SendPtr {}

// SAFETY: shared references to SendPtr are sound because the pointee is never
// accessed through the wrapper without the call-site disjoint-offset proof.
unsafe impl Sync for SendPtr {}


impl SendPtr {
    #[inline(always)]
    fn add(self, offset: usize) -> *mut f64 {
        // SAFETY: callers pass offsets within the backing allocation and only
        // dereference the returned pointer after proving the target element is
        // uniquely owned by that worker's chunk for the whole parallel region.
        unsafe { self.0.add(offset) }
    }
}


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

    #[error(
        "Failed to identify a constraint nullspace basis at {site}: \
         coefficient dim {coeff_dim}, cross-rank {cross_rank}, \
         constraint Frobenius {cross_frobenius:.3e}, \
         constrained Gram max eigenvalue {constrained_gram_max_eigenvalue:.3e} \
         (min {constrained_gram_min_eigenvalue:.3e}, \
         spectral tolerance {spectral_tolerance:.3e}). \
         The smooth basis collapses onto the parametric block — typical causes: \
         (a) the smooth's evaluated kernel underflows after projecting out the \
         polynomial nullspace, leaving only floating-point noise (Duchon hybrid \
         in moderate-to-high d with length_scale near pairwise center distances); \
         (b) the parametric block already spans the smooth's column space \
         (over-restrictive identifiability constraint); \
         (c) the smooth has effective rank ≤ parametric-block size on this data."
    )]
    ConstraintNullspaceCollapsed {
        site: &'static str,
        cross_rank: usize,
        coeff_dim: usize,
        cross_frobenius: f64,
        constrained_gram_max_eigenvalue: f64,
        constrained_gram_min_eigenvalue: f64,
        spectral_tolerance: f64,
    },

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

    #[error(
        "Radial basis derivative is undefined at center collision (r = 0) for {kernel} \
         with dim = {dim}, m = {m}: {message}. The first/second derivative of the \
         underlying φ(r) does not have a finite limit as r → 0+, so the design-row \
         gradient and Hessian have no well-defined value at coincident points."
    )]
    DegenerateAtCollision {
        kernel: &'static str,
        dim: usize,
        m: f64,
        message: &'static str,
    },

    #[error(
        "Periodic radial basis derivative is undefined at the wrap branch cut \
         (signed displacement = ±period/2) for raw delta = {raw}, period = {period}: \
         the wrapped displacement jumps between ±period/2 and the first derivative \
         w.r.t. the input has a one-sided discontinuity. Move the evaluation point \
         off the branch cut or define a one-sided convention."
    )]
    PeriodicWrapBranchCut { raw: f64, period: f64 },

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
    pub const fn value() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating first derivatives of basis functions.
    pub const fn first_derivative() -> Self {
        Self {
            derivative_order: 1,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating second derivatives of basis functions.
    pub const fn second_derivative() -> Self {
        Self {
            derivative_order: 2,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating M-spline basis values.
    pub const fn m_spline() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::MSpline,
        }
    }

    /// Create options for evaluating I-spline basis values.
    pub const fn i_spline() -> Self {
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
        crate::bail_invalid_basis!("derivatives are only supported for BasisFamily::BSpline");
    }

    let eval_kind = match options.derivative_order {
        0 => BasisEvalKind::Basis,
        1 => BasisEvalKind::FirstDerivative,
        2 => BasisEvalKind::SecondDerivative,
        n => {
            crate::bail_invalid_basis!(
                "unsupported derivative order {n}; only 0, 1, 2 are supported"
            );
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
            if O::LAYOUT.is_sparse() {
                let sparse = create_mspline_sparse(data, knotvec.view(), degree)?;
                Ok((O::from_sparse(sparse)?, knotvec))
            } else {
                let dense = create_mspline_dense(data, knotvec.view(), degree)?;
                Ok((O::from_dense(dense)?, knotvec))
            }
        }
        BasisFamily::ISpline => {
            if O::LAYOUT.is_sparse() {
                crate::bail_invalid_basis!(
                    "BasisFamily::ISpline does not support sparse output; use Dense"
                );
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
        crate::bail_dim_basis!("z_raw and z_clamped must have equal length");
    }
    if basisvalues.nrows() != z_raw.len() {
        crate::bail_dim_basis!("basis row count must match z length");
    }

    let mut needs_ext = false;
    for i in 0..z_raw.len() {
        if (z_raw[i] - z_clamped[i]).abs() > BSPLINE_EXTRAPOLATION_THRESHOLD {
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
        crate::bail_dim_basis!("basis derivative shape mismatch");
    }

    for i in 0..z_raw.len() {
        let dz = z_raw[i] - z_clamped[i];
        if dz.abs() <= BSPLINE_EXTRAPOLATION_THRESHOLD {
            continue;
        }
        for j in 0..basisvalues.ncols() {
            basisvalues[[i, j]] += dz * b_prime[[i, j]];
        }
    }
    Ok(())
}


/// Storage layout discriminant for [`BasisOutputFormat`] impls. Encoded as an
/// enum rather than a bool so the type-level distinction reads as
/// "Dense vs Sparse" at call sites instead of a polarity-sensitive flag.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BasisStorageLayout {
    Dense,
    Sparse,
}


impl BasisStorageLayout {
    #[inline]
    pub const fn is_sparse(self) -> bool {
        matches!(self, Self::Sparse)
    }
}


/// Trait for building basis matrices with different storage formats.
/// This is an implementation detail for the unified `create_basis` function.
pub trait BasisOutputFormat {
    type Output;
    const LAYOUT: BasisStorageLayout;

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
    const LAYOUT: BasisStorageLayout = BasisStorageLayout::Dense;

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


#[inline]
fn one_sided_derivative_eval_point(x: f64, knotview: ArrayView1<f64>, degree: usize) -> f64 {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis == 0 {
        return x;
    }
    let left = knotview[degree];
    let right = knotview[num_basis];
    if !left.is_finite() || !right.is_finite() || left >= right {
        return x;
    }
    if x <= left {
        let next = left.next_up();
        if next < right {
            next
        } else {
            left + 0.5 * (right - left)
        }
    } else if x >= right {
        let prev = right.next_down();
        if prev > left {
            prev
        } else {
            left + 0.5 * (right - left)
        }
    } else {
        x
    }
}


impl BasisOutputFormat for Sparse {
    type Output = SparseColMat<usize, f64>;
    const LAYOUT: BasisStorageLayout = BasisStorageLayout::Sparse;

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
        if span <= KNOT_SPAN_DEGENERACY_FLOOR {
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
        one_sided_derivative_eval_point(x.clamp(left, right), knotview, degree)
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
        let left_term = if denom_left.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            left / denom_left
        } else {
            0.0
        };
        let right_term = if denom_right.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            right / denom_right
        } else {
            0.0
        };
        values[offset] = (degree as f64) * (left_term - right_term);
    }

    start_col
}


#[inline]
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
        let left_term = if denom_left.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            left / denom_left
        } else {
            0.0
        };
        let right_term = if denom_right.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            right / denom_right
        } else {
            0.0
        };
        values[offset] = (degree as f64) * (left_term - right_term);
    }

    start_col
}


#[inline]
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


#[inline]
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

    if let Some(g) = greville_abscissae
        && g.len() != num_basis_functions
    {
        crate::bail_dim_basis!(
            "Greville abscissae length {} does not match num_basis_functions {}",
            g.len(),
            num_basis_functions
        );
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
                if span.abs() <= KNOT_SPAN_DEGENERACY_FLOOR {
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


fn bspline_raw_column_count(
    knots: &Array1<f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
) -> Result<usize, String> {
    if let Some((_, _, num_basis)) = periodic {
        if num_basis <= degree {
            return Err(format!(
                "streaming cyclic B-spline basis requires more basis functions ({num_basis}) than degree ({degree})"
            ));
        }
        return Ok(num_basis);
    }
    knots
        .len()
        .checked_sub(degree + 1)
        .filter(|&p| p > 0)
        .ok_or_else(|| {
            format!(
                "streaming B-spline knots length {} is too short for degree {}",
                knots.len(),
                degree
            )
        })
}


fn bspline_raw_row_chunk(
    data: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    start: usize,
    end: usize,
) -> Result<Array2<f64>, BasisError> {
    if start > end || end > data.len() {
        crate::bail_dim_basis!(
            "B-spline row chunk [{start}, {end}) is out of bounds for {} rows",
            data.len()
        );
    }
    let chunk = data.slice(s![start..end]);
    if let Some((domain_start, period, num_basis)) = periodic {
        if period <= 0.0 {
            crate::bail_invalid_basis!("periodic B-spline period must be positive, got {period}");
        }
        let wrapped = chunk.mapv(|x| wrap_to_period(x, domain_start, period));
        let (extended, _) = create_basis::<Dense>(
            wrapped.view(),
            KnotSource::Provided(knots),
            degree,
            BasisOptions::value(),
        )?;
        let mut cyclic = Array2::<f64>::zeros((chunk.len(), num_basis));
        for i in 0..extended.nrows() {
            for j in 0..extended.ncols() {
                cyclic[[i, j % num_basis]] += extended[[i, j]];
            }
        }
        Ok(cyclic)
    } else {
        let (basis, _) = create_basis::<Dense>(
            chunk,
            KnotSource::Provided(knots),
            degree,
            BasisOptions::value(),
        )?;
        Ok((*basis).clone())
    }
}


/// Selects Greville abscissae for difference-penalty scaling.
///
/// The classical P-spline integer-difference penalty `D'D` penalizes the squared
/// `m`-th differences of the *coefficients*. Those differences represent the
/// squared `m`-th derivative of the *function* — with the correct polynomial null
/// space `{1, x, …, x^{m-1}}` — only when the basis has *evenly spaced Greville
/// abscissae*, because a coefficient sequence that is linear in its index then
/// reproduces a function linear in `x` (B-spline linear precision, `Σ ξ_i B_i(x)
/// = x`).
///
/// Equally spaced *breakpoints* are **not** sufficient. gam's B-splines are
/// clamped (boundary knots repeated `degree + 1` times), so even on a uniform
/// interior grid the Greville abscissae `ξ_i = (1/m)·Σ_{k=1}^{degree} t_{i+k}`
/// cluster toward the ends. With such a basis the integer-difference null space
/// is a *rotated approximation* of the polynomial space rather than the exact
/// `{1, x, …}`. That tilts the direction REML shrinks toward as `λ → ∞`, so the
/// recovered surface is biased and the selected smoothing parameters land off
/// the true optimum (e.g. anisotropic tensor `te`/`ti` recovery degrades).
///
/// We therefore gate the integer-difference fast path on uniformity of the
/// **Greville abscissae** and otherwise return them, so divided-difference
/// scaling in [`create_difference_penalty_matrix`] restores the exact polynomial
/// null space for any knot geometry (clamped, quantile, or otherwise). When the
/// abscissae are already uniform (e.g. a non-clamped Eilers–Marx grid), the
/// divided differences reduce to the integer differences up to an overall scale,
/// so returning `None` there is exact and cheaper.
pub fn penalty_greville_abscissae_for_knots(
    knot_vector: &Array1<f64>,
    degree: usize,
) -> Result<Option<Array1<f64>>, BasisError> {
    // Degenerate / too-short knot vectors have no meaningful divided-difference
    // scaling (and `compute_greville_abscissae` rejects them); fall back to the
    // plain integer-difference penalty exactly as before.
    let greville = match compute_greville_abscissae(knot_vector, degree) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    if is_uniformly_spaced_sequence(greville.view()) {
        Ok(None)
    } else {
        Ok(Some(greville))
    }
}


/// True when the entries of `values` are (numerically) evenly spaced. Used to
/// decide whether classical integer-difference penalties coincide with the
/// geometry-correct divided-difference penalty for a basis.
fn is_uniformly_spaced_sequence(values: ArrayView1<'_, f64>) -> bool {
    let n = values.len();
    if n <= 2 {
        return true;
    }
    let span = (values[n - 1] - values[0]).abs().max(1.0);
    let h0 = values[1] - values[0];
    for i in 2..n {
        let hi = values[i] - values[i - 1];
        if (hi - h0).abs() > 1e-8 * span {
            return false;
        }
    }
    true
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
    /// Wood-TPRS radial reparameterization (k-M) × (k-M) matrix V whose columns are
    /// eigenvectors of the constrained kernel penalty Z' Ω Z, sorted so the leading
    /// columns carry the most penalized directions. `kernel_constrained` columns and
    /// the radial penalty block are expressed in this rotated basis: design columns
    /// are `Φ Z V` and the radial penalty is `diag(Λ)`.
    pub radial_reparam: Array2<f64>,
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
    /// The half-integer smoothness value ν as an `f64` (0.5, 1.5, …).
    pub const fn half_integer_value(self) -> f64 {
        match self {
            MaternNu::Half => 0.5,
            MaternNu::ThreeHalves => 1.5,
            MaternNu::FiveHalves => 2.5,
            MaternNu::SevenHalves => 3.5,
            MaternNu::NineHalves => 4.5,
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


#[derive(Debug, Clone)]
struct DuchonBasisDesign {
    basis: Array2<f64>,
}


/// Boundary-condition policy for one-dimensional smooth bases.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OneDimensionalBoundary {
    /// Ordinary open interval basis with clamped endpoint behavior.
    #[default]
    Open,
    /// Periodic/cyclic basis over the half-open interval `[start, end)`.
    ///
    /// Values are evaluated modulo `period = end - start`; the basis and its
    /// first `degree - 1` derivatives agree at the two endpoints for B-splines.
    Cyclic { start: f64, end: f64 },
}


impl OneDimensionalBoundary {
    fn period(&self) -> Option<(f64, f64, f64)> {
        match *self {
            OneDimensionalBoundary::Open => None,
            OneDimensionalBoundary::Cyclic { start, end } if end > start => {
                Some((start, end, end - start))
            }
            OneDimensionalBoundary::Cyclic { .. } => None,
        }
    }
}


/// Which knot strategy to use for 1D B-spline bases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BSplineKnotSpec {
    Generate {
        data_range: (f64, f64),
        num_internal_knots: usize,
    },
    /// Uniform cyclic B-spline basis on `[data_range.0, data_range.1)`.
    ///
    /// The first and last endpoints are identified, so evaluating at `x` and
    /// `x + m * period` gives identical rows. `num_basis` is the number of
    /// periodic control sites around the loop and must be at least
    /// `degree + 1` for an unaliased local support stencil.
    PeriodicUniform {
        data_range: (f64, f64),
        num_basis: usize,
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
    #[serde(default)]
    pub boundary: OneDimensionalBoundary,
    /// Optional endpoint boundary constraints (Hermite-style pin of value and/or
    /// derivative at the left/right knot extents). Default = `Free` on both
    /// sides which is a no-op.
    #[serde(default)]
    pub boundary_conditions: BSplineBoundaryConditions,
}


/// Per-endpoint boundary constraint policy for B-spline 1D bases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum BSplineEndpointBoundaryCondition {
    /// No endpoint constraint.
    #[default]
    Free,
    /// Pin the first derivative to zero at this endpoint.
    Clamped,
    /// Pin the value at this endpoint to `value` (currently only `value == 0`
    /// is accepted in the builder; non-zero anchors require an affine offset).
    Anchored { value: f64 },
}


/// Left/right pair of B-spline endpoint constraints.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BSplineBoundaryConditions {
    #[serde(default)]
    pub left: BSplineEndpointBoundaryCondition,
    #[serde(default)]
    pub right: BSplineEndpointBoundaryCondition,
}


impl BSplineBoundaryConditions {
    pub const fn is_free(&self) -> bool {
        matches!(self.left, BSplineEndpointBoundaryCondition::Free)
            && matches!(self.right, BSplineEndpointBoundaryCondition::Free)
    }
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
    // Smooth terms are centered by default to avoid intercept confounding.
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
        BSplineIdentifiability::WeightedSumToZero { weights: None }
    }
}


/// Spatial center selection strategy.
///
/// `num_centers` is the exact number of knot/center rows selected by the
/// strategy. Polynomial nullspace columns are added separately by each basis
/// builder and must never be folded into this count.
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
/// The basis size is the sub-linear `ceil(8 * d_factor * n^0.4)`, clamped above
/// at `K_MAX = 2000` and below at a *data-proportional* floor `min(200, n/8)` so
/// the floor only engages once there are enough observations to support a rich
/// basis. The result is additionally capped at `n/4` so the penalty matrices
/// stay well-conditioned relative to the data:
///
/// | n      | d=1  | d=2  | d=5  |
/// |--------|------|------|------|
/// | 800    | 116  | 134  | 186  |
/// | 1 000  | 127  | 146  | 200  |
/// | 2 000  | 200  | 200  | 268  |
/// | 10 000 | 319  | 367  | 510  |
/// | 100 000| 801  | 921  | 1281 |
/// | 400 000| 1393 | 1602 | 2000 |
/// | 1 000 000| 2000 | 2000 | 2000 |
///
/// The flat `200` floor used to inflate moderate-`n` spatial smooths (a few
/// hundred to ~2000 rows) up to a dense 200-column design even though the raw
/// sub-linear count — and the mesh/knot density that mgcv and R-INLA use on the
/// same data — is far smaller. On ~800 rows that turned a single 2-D thin-plate
/// REML fit into an `O(n·p² + p³)` grind at `p ≈ 200` (#718). Smoothness is
/// already controlled by REML's penalty weight λ, not by the center count, so a
/// data-proportional floor recovers the same surface at a fraction of the cost.
///
/// # Arguments
/// * `n` - sample size (number of observations)
/// * `d` - covariate dimensionality (number of input variables in the smooth)
pub fn default_num_centers(n: usize, d: usize) -> usize {
    const K_MIN: usize = 200;
    const K_MAX: usize = 2000;
    const ALPHA: f64 = 0.4;
    const C: f64 = 8.0;
    /// Per-extra-dimension growth in the center count: each covariate axis
    /// beyond the first widens the basis by 15% to keep the per-axis mesh
    /// density roughly constant as the smooth's domain dimensionality grows.
    const PER_DIM_GROWTH: f64 = 0.15;
    /// Divisor for the data-proportional floor: the `K_MIN` floor only engages
    /// once `n` exceeds `K_MIN * FLOOR_N_DIVISOR`, so small samples are not
    /// forced up to a dense `K_MIN`-column design.
    const FLOOR_N_DIVISOR: usize = 8;
    /// Divisor for the conditioning cap: the center count never exceeds `n /
    /// COND_N_DIVISOR`, keeping the penalty matrices well-conditioned relative
    /// to the data.
    const COND_N_DIVISOR: usize = 4;

    let d_factor = 1.0 + PER_DIM_GROWTH * (d.max(1) - 1) as f64;
    let raw = (C * d_factor * (n as f64).powf(ALPHA)).ceil() as usize;

    // Data-proportional floor: never inflate beyond n/FLOOR_N_DIVISOR, so the
    // K_MIN-center floor only takes effect once n is large enough (~1600) to
    // genuinely support that many basis columns.
    let floor = K_MIN.min(n / FLOOR_N_DIVISOR);
    let k = raw.clamp(floor, K_MAX);

    // Never exceed n itself; cap at n/COND_N_DIVISOR to keep the penalty
    // matrices well-conditioned relative to the data.
    k.min(n).min(n / COND_N_DIVISOR)
}


/// Conservative center count for a *secondary* (distributional) predictor's
/// spatial smooth — e.g. the log-σ scale model in a Gaussian location-scale
/// fit.
///
/// The mean is identified directly by the response, so it warrants the
/// generous [`default_num_centers`] basis. A scale/shape predictor is
/// identified only through (noisy) squared residuals: handing it a basis sized
/// for the mean lets REML/LAML smoothing selection over-fit it, because where
/// the fitted scale is driven small the *observed* information collapses and
/// the determinant penalty stops holding the wiggle down (#501). This mirrors
/// standard GAMLSS/mgcv practice of giving distribution parameters a modest
/// default (mgcv's `k = 10` for a 1-D `s()`), grown gently with dimensionality
/// and never exceeding the generous primary-predictor default.
pub fn conservative_secondary_centers(n: usize, d: usize) -> usize {
    const BASE_1D_CENTERS: usize = 10;
    let modest = BASE_1D_CENTERS.saturating_mul(d.max(1));
    default_num_centers(n, d).min(modest).max(1)
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
        crate::bail_invalid_basis!("plan_spatial_basis: n must be >= 1");
    }
    if d == 0 {
        crate::bail_invalid_basis!("plan_spatial_basis: d must be >= 1");
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


pub const fn default_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    if d >= 4 {
        CenterStrategy::EqualMassCovarRepresentative { num_centers }
    } else {
        CenterStrategy::EqualMass { num_centers }
    }
}


pub fn auto_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    let strategy = if d == 1 {
        // In one dimension, farthest-point selection is the deterministic
        // maximin grid over the observed domain. Equal-mass midpoints leave the
        // low-frequency Duchon radial block slightly under-resolved at the
        // boundaries, and REML then compensates with an over-smooth λ on
        // low-noise signals (#504). The maximin grid matches the native
        // reproducing-kernel interpolation geometry while keeping the existing
        // equal-mass defaults for genuinely multivariate smooths.
        CenterStrategy::FarthestPoint { num_centers }
    } else {
        default_spatial_center_strategy(num_centers, d)
    };
    CenterStrategy::Auto(Box::new(strategy))
}


pub const fn center_strategy_is_auto(strategy: &CenterStrategy) -> bool {
    matches!(strategy, CenterStrategy::Auto(_))
}


fn realized_center_strategy(strategy: &CenterStrategy) -> &CenterStrategy {
    match strategy {
        CenterStrategy::Auto(inner) => inner.as_ref(),
        other => other,
    }
}


pub fn center_strategy_kind(strategy: &CenterStrategy) -> CenterStrategyKind {
    match strategy {
        CenterStrategy::Auto(inner) => center_strategy_kind(inner.as_ref()),
        CenterStrategy::UserProvided(_) => CenterStrategyKind::UserProvided,
        CenterStrategy::EqualMass { .. } => CenterStrategyKind::EqualMass,
        CenterStrategy::EqualMassCovarRepresentative { .. } => {
            CenterStrategyKind::EqualMassCovarRepresentative
        }
        CenterStrategy::FarthestPoint { .. } => CenterStrategyKind::FarthestPoint,
        CenterStrategy::KMeans { .. } => CenterStrategyKind::KMeans,
        CenterStrategy::UniformGrid { .. } => CenterStrategyKind::UniformGrid,
    }
}


pub fn center_strategy_num_centers(strategy: &CenterStrategy) -> Option<usize> {
    match strategy {
        CenterStrategy::Auto(inner) => center_strategy_num_centers(inner.as_ref()),
        CenterStrategy::UserProvided(centers) => Some(centers.nrows()),
        CenterStrategy::EqualMass { num_centers }
        | CenterStrategy::EqualMassCovarRepresentative { num_centers }
        | CenterStrategy::FarthestPoint { num_centers }
        | CenterStrategy::KMeans { num_centers, .. } => Some(*num_centers),
        CenterStrategy::UniformGrid { .. } => None,
    }
}


pub fn center_strategy_with_num_centers(
    strategy: &CenterStrategy,
    num_centers: usize,
) -> Result<CenterStrategy, BasisError> {
    validate_center_count(num_centers)?;
    fn rebuild_inner(
        strategy: &CenterStrategy,
        num_centers: usize,
    ) -> Result<CenterStrategy, BasisError> {
        match strategy {
            CenterStrategy::Auto(inner) => rebuild_inner(inner.as_ref(), num_centers),
            CenterStrategy::EqualMass { .. } => Ok(CenterStrategy::EqualMass { num_centers }),
            CenterStrategy::EqualMassCovarRepresentative { .. } => {
                Ok(CenterStrategy::EqualMassCovarRepresentative { num_centers })
            }
            CenterStrategy::FarthestPoint { .. } => {
                Ok(CenterStrategy::FarthestPoint { num_centers })
            }
            CenterStrategy::KMeans { max_iter, .. } => Ok(CenterStrategy::KMeans {
                num_centers,
                max_iter: *max_iter,
            }),
            CenterStrategy::UserProvided(_) | CenterStrategy::UniformGrid { .. } => {
                Err(BasisError::InvalidInput(format!(
                    "cannot replace center count for {:?} strategy",
                    center_strategy_kind(strategy)
                )))
            }
        }
    }
    let rebuilt = rebuild_inner(strategy, num_centers)?;
    Ok(match strategy {
        CenterStrategy::Auto(_) => CenterStrategy::Auto(Box::new(rebuilt)),
        _ => rebuilt,
    })
}


/// Thin-plate basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinPlateBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
    pub length_scale: f64,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
    /// Frozen Wood-TPRS radial reparameterization. When `Some`, the builder
    /// reuses this `(raw_radial_cols) × (kept_radial_cols)` matrix instead of
    /// recomputing it from the constrained kernel penalty eigensystem. The
    /// rectangular case is the truncated regression-spline path; carrying it
    /// into prediction guarantees identical radial modes to fit-time.
    #[serde(default)]
    pub radial_reparam: Option<Array2<f64>>,
}


/// Per-smooth identifiability policy for spatial (TPS / Duchon) bases.
///
/// For a raw local basis `B` and parametric design block `C`, the orthogonalized
/// basis is `B_c = B Z` where columns of `Z` span `null((B^T C)^T)`. This enforces:
///   `B_c^T C = 0`
/// in the unweighted inner product, so spatial effects cannot absorb parametric
/// directions that actually exist in the model. The standalone basis builder has
/// only an implicit intercept available, so it centers smooths against that
/// intercept. The term-collection builder augments `C` with explicit linear
/// terms when those terms are present in the formula.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum SpatialIdentifiability {
    /// Keep unconstrained basis columns.
    None,
    /// Orthogonalize the smooth against model-owned parametric columns.
    // "Magic" default for modular GAMs with explicit parametric block:
    // keep spatial smooth orthogonal to intercept/linear terms.
    // ApproxKind: Exact (orthogonalization is an exact projection).
    #[default]
    OrthogonalToParametric,
    /// Freeze a fit-time transform `Z`; prediction uses `B_new * Z` unchanged.
    FrozenTransform { transform: Array2<f64> },
}


pub(crate) use sphere_kernels::{
    wahba_sphere_kernel_derivative_dcos_kind, wahba_sphere_kernel_from_cos_kind,
    wahba_sphere_kernel_from_cos_simd_kind, wahba_sphere_kernel_sobolev_derivative_dcos,
};

pub use sphere_spectral::{
    pseudo_s2_truncated_coefficients, sobolev_s2_truncated_coefficients,
    sphere_truncated_spectral_eval,
};


/// Matérn basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaternBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
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
    /// Frozen double-penalty nullspace-shrinkage decision (gam#787/#860).
    ///
    /// `None` (the default, and the cold-build value) = decide whether to emit
    /// the `DoublePenaltyNullspace` candidate via the κ-dependent spectral test in
    /// `build_nullspace_shrinkage_penalty`. `Some(b)` = force the decision (set by
    /// the freeze step from the bootstrap-κ build, mirrored from
    /// `MaternIdentifiability::FrozenTransform`) so the learned-penalty count stays
    /// invariant as the κ-optimizer rebuilds the design at each trial length-scale.
    /// Only consulted when `double_penalty` is true.
    #[serde(default)]
    pub nullspace_shrinkage_survived: Option<bool>,
}


/// Per-smooth identifiability policy for Matérn kernel coefficients.
///
/// These constraints are geometric (center-based), so they are stable across
/// train/predict and do not depend on response weights.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum MaternIdentifiability {
    /// Keep the unconstrained kernel coefficient space.
    None,
    /// Enforce `1^T alpha = 0` at center locations (removes constant drift).
    // Safe default with model intercepts: prevent kernel block from absorbing
    // a global mean level.
    #[default]
    CenterSumToZero,
    /// Enforce orthogonality to `[1, c_1, ..., c_d]` at centers.
    /// Use this when explicit linear terms should own global trends.
    CenterLinearOrthogonal,
    /// Freeze a fit-time transform `Z` so prediction cannot drift.
    ///
    /// `nullspace_shrinkage_survived` freezes the double-penalty
    /// nullspace-shrinkage decision alongside the transform (gam#787/#860). The
    /// matern double-penalty path emits a `DoublePenaltyNullspace` candidate iff
    /// `build_nullspace_shrinkage_penalty(&projected_kernel)` finds a near-zero
    /// eigenvalue — but that spectral test is κ-DEPENDENT (its tolerance scales
    /// with `λ_max`), so a near-zero eigenvalue can cross the threshold as the
    /// κ-optimizer rebuilds the design at each trial length-scale. That flips the
    /// learned-penalty count 6↔7 across the rebuild and the rebuilt design's ρ
    /// dimension then disagrees with the frozen joint setup ("joint hyper rho
    /// dimension mismatch" → every κ seed fails startup validation). Freezing the
    /// bootstrap-κ decision here (`Some(true)` = always emit the shrinkage
    /// candidate, `Some(false)` = never) keeps the penalty count INVARIANT across
    /// the κ rebuild so κ actually optimizes. `None` = decide via the spectral
    /// test (the non-frozen / cold-build behavior; also the serde back-compat
    /// default for transforms frozen before this field existed).
    FrozenTransform {
        transform: Array2<f64>,
        #[serde(default)]
        nullspace_shrinkage_survived: Option<bool>,
    },
}


/// Duchon null-space polynomial degree.
///
/// Controls the polynomial null space of the Duchon / polyharmonic spline. The
/// Duchon seminorm `‖D^m f‖²` annihilates all polynomials of total degree
/// `< m`, so those polynomials must be handled as explicit unpenalized columns.
///
/// The user-facing `order` knob selects the polynomial degree cutoff `r`, and
/// the resulting polynomial null space has dimension `C(d + r, r)` where `d`
/// is the covariate dimension.  In the `duchon(...)` formula DSL:
///
/// | `order=` | Variant         | max total degree | null-space dim  |
/// |----------|-----------------|------------------|-----------------|
/// | `0`      | `Zero`          | 0                | `C(d+0,0) = 1`  |
/// | `1`      | `Linear`        | 1                | `C(d+1,1) = d+1`|
/// | `k≥2`    | `Degree(k)`     | k                | `C(d+k,k)`      |
///
/// **How the polynomial null space is consumed during basis construction:**
///
/// 1. `polynomial_block_from_order` materialises an `(n, C(d+r,r))` block `P`
///    of monomials up to total degree `r` at the selected `centers`.
/// 2. `kernel_constraint_nullspace` computes `Z = null(P_centers^T)`, a
///    `(k, k − C(d+r,r))` matrix. Reparameterising the radial kernel
///    coefficients as `α = Z γ` enforces the side condition `P_centers^T α = 0`
///    and yields `k − C(d+r,r)` free kernel parameters.
/// 3. The polynomial block `P_data` evaluated at the data rows is appended to
///    the kernel block `Φ Z`, giving a total of
///    `(k − C(d+r,r)) + C(d+r,r) = k` columns before the spatial
///    identifiability transform.  Crucially, the total width equals the
///    requested center count `k`, **not** `k + C(d+r,r)`.
///
/// **Example — `duchon(PC1, PC2, PC3, centers=10, order=1)` (d=3):**
///
/// - Polynomial null space: `C(3+1,1) = 4` monomials `{1, x₁, x₂, x₃}`.
/// - Kernel columns after constraint: `10 − 4 = 6`.
/// - Appended polynomial block: 4 columns.
/// - Pre-identifiability total: `6 + 4 = 10` columns, i.e. exactly `centers`.
///
/// The variant naming matches the Duchon `m` parameter:
/// `Zero` → `m=1`, `Linear` → `m=2`, `Degree(k)` → `m=k+1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DuchonNullspaceOrder {
    Zero,
    Linear,
    Degree(usize),
}


/// Duchon-like basis configuration with explicit low-frequency null-space
/// control and explicit spectral power.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuchonBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
    /// Optional hybrid Matérn width. `None` means pure scale-free Duchon with
    /// spectrum `||w||^(2p + 2s)`. `Some(length_scale)` enables the hybrid
    /// spectrum `||w||^(2p) * (kappa^2 + ||w||^2)^s`, `kappa = 1/length_scale`.
    pub length_scale: Option<f64>,
    /// Literal Duchon spectral power `s` (`f64`, fractional values fully
    /// threaded end-to-end). The pure-Duchon kernel exponent is `2(p + s) − d`,
    /// so this is the knob that sets `φ(r)`: `s = 0` is the integer-order Duchon
    /// kernel `r^{2p−d}` (its `r²·log r` log case in even `d`, ≡ the thin-plate
    /// kernel); `s = (d − 1)/2` gives the cubic `r³` in every dimension.
    ///
    /// This field is taken LITERALLY by the basis builder — `power = 0` means
    /// `s = 0`, NOT "use a default". The magic cubic default (applied when the
    /// user gives no explicit power) is a request-layer choice resolved by the
    /// formula / CLI / pyffi front-ends via [`duchon_cubic_default`]; by the time
    /// a spec reaches the builder this value is the final intended `s`. The
    /// hybrid Duchon–Matérn path (`length_scale = Some`) still requires an
    /// integer `s` (read via `spec.power_as_usize()`).
    #[serde(alias = "power_int")]
    pub power: f64,
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
    #[serde(default)]
    pub boundary: OneDimensionalBoundary,
}


impl DuchonBasisSpec {
    /// Integer view of `power` for the existing integer-only downstream chain.
    /// Non-finite or non-integer values fall back to `0` (the integer-only
    /// validators downstream already reject this case with a clear message).
    pub fn power_as_usize(&self) -> usize {
        duchon_power_to_usize(self.power)
    }
}


/// Convert a Duchon spectral-power `f64` into the integer view used by the
/// closed-form code paths. Non-finite, negative, or fractional values clamp to
/// `0` so the validator downstream emits the canonical error.
pub fn duchon_power_to_usize(power: f64) -> usize {
    if !power.is_finite() || power < 0.0 {
        return 0;
    }
    let rounded = power.round();
    if (rounded - power).abs() > 1e-9 {
        return 0;
    }
    rounded as usize
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
        // ALL ON. The Duchon penalty is a Hilbert scale: curvature is the
        // always-on exact RKHS `Primary` Gram and the trend ridge is always on;
        // the lower orders — mass (amplitude `Σ(f−f̄)²`) and tension (first-order
        // roughness `Σ‖∇f‖²`) — are active here, collocated on a density-blind
        // data-support sample. REML deselects any the data don't support (SPEC:
        // recover the null by default, opt INTO overfitting). Stiffness (`D2`)
        // stays off — `Primary` is the exact, superior curvature. (The Matérn
        // collocation overlay builds its own `all_active()`; SAE atoms, which
        // ship only `Primary`, use `all_disabled()`.)
        Self {
            mass: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            tension: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            stiffness: OperatorPenaltySpec::Disabled,
        }
    }
}


impl DuchonOperatorPenaltySpec {
    pub fn all_disabled() -> Self {
        Self {
            mass: OperatorPenaltySpec::Disabled,
            tension: OperatorPenaltySpec::Disabled,
            stiffness: OperatorPenaltySpec::Disabled,
        }
    }

    /// All three operator dials active — used by the Matérn collocation overlay.
    pub fn all_active() -> Self {
        let active = || OperatorPenaltySpec::Active {
            initial_log_lambda: 0.0,
            prior: None,
        };
        Self {
            mass: active(),
            tension: active(),
            stiffness: active(),
        }
    }

    /// Operator-penalty dials appropriate for a Matérn-ν kernel in dimension `d`.
    ///
    /// The Matérn-ν RKHS is the Sobolev space `H^m` with `m = ν + d/2`: its
    /// squared norm controls the order-`j` derivative in L2 exactly when
    /// `j ≤ m`. The collocation overlay penalizes the squared L2 norms of the
    /// value (mass, `D0`, j=0), gradient (tension, `D1`, j=1) and Hessian
    /// (stiffness, `D2`, j=2). Activating a penalty whose derivative order
    /// exceeds the RKHS smoothness (`j > m`) imposes a roughness constraint the
    /// true kernel does NOT — it over-smooths the reduced-rank fit relative to
    /// the exact GP (mgcv `bs="gp"`, GpGp).
    ///
    /// Concretely the roughest Matérn, ν=1/2 in d=1 (`m = 1`), is the
    /// Ornstein–Uhlenbeck/exponential kernel: an H¹ process whose sample paths
    /// are continuous but non-differentiable. Although `∫(f')²` is finite on
    /// its RKHS, the kernel itself already encodes the H¹ control; layering an
    /// extra tension dial on top biases the reduced-rank fit toward the smooth
    /// `C¹` functions the kernel does not favour (and stiffness `D2` toward
    /// `C²`), collapsing held-out oscillation (#707). We therefore gate each
    /// operator on `j < m` STRICTLY: mass (j=0) is always on, tension (j=1) is
    /// on for `m > 1`, stiffness (j=2) is on for `m > 2`. For ν ≥ 3/2 (or any
    /// d ≥ 2) every dial is active, recovering `all_active`; only the
    /// genuinely rough ν=1/2 (d=1) kernel — where the Sobolev order sits
    /// exactly on a derivative boundary — drops the higher operators.
    pub fn matern_for_smoothness(nu: MaternNu, d: usize) -> Self {
        let m = nu.half_integer_value() + 0.5 * d as f64;
        // Tolerance so an exact half-integer Sobolev order (e.g. m = 1.0 for
        // ν=1/2, d=1) reliably DISABLES the matching-order operator instead
        // of flipping on a float-equality knife-edge.
        const ORDER_EPS: f64 = 1e-9;
        let active = || OperatorPenaltySpec::Active {
            initial_log_lambda: 0.0,
            prior: None,
        };
        let gate = |order: f64| {
            if m > order + ORDER_EPS {
                active()
            } else {
                OperatorPenaltySpec::Disabled
            }
        };
        Self {
            mass: active(),
            tension: gate(1.0),
            stiffness: gate(2.0),
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


/// Resolve a fully admissible Duchon `(nullspace_order, power)` pair.
///
/// Three constraints fold into one resolution:
///   (a) operator collocation up to `max_op`:        `2(p + s) > d + max_op`
///   (b) pure-mode CPD vs polynomial nullspace P_p:  `2s < d`
///       (Wendland Thm 8.17: pure polyharmonic kernel of order m = p+s in
///        R^d is CPD of order `m − ⌊d/2⌋ + 1[d even, log] / m − (d−1)/2
///        [d odd]`, and Duchon interpolation against P_p is well-posed iff
///        CPD order ≤ p, which collapses to `2s < d` since 2s, d are
///        integers and 2s is even.)
///   (a) implies the kernel-existence condition `2(p + s) > d`.
///   (b) is dropped when `length_scale` is `Some` (hybrid Matérn-blended
///       kernel is strictly PD, CPD order 0).
///
/// Strategy: at the requested `nullspace_order`, take the smallest `s`
/// satisfying (a). If that `s` violates (b) in pure mode, escalate the
/// nullspace order by one and retry. Termination: at `p ≥ ⌈(d+max_op)/2⌉ + 1`
/// the operator constraint (a) admits `s = 0`, and `0 < d` satisfies (b)
/// for any `d ≥ 1`, so escalation always converges.
///
/// The returned nullspace order is monotone in the request: it never
/// decreases the user's requested order — only strengthens it when pure-mode
/// CPD requires a richer polynomial absorption space.
pub fn resolve_duchon_orders(
    dim: usize,
    requested_nullspace_order: DuchonNullspaceOrder,
    max_operator_derivative_order: usize,
    length_scale: Option<f64>,
) -> (DuchonNullspaceOrder, usize) {
    assert!(dim >= 1, "Duchon basis requires dim >= 1");
    let pure = length_scale.is_none();
    let mut nullspace = requested_nullspace_order;
    // Bounded loop: escalation terminates by the argument above.
    for _ in 0..=(dim + max_operator_derivative_order + 1) {
        let p = duchon_p_from_nullspace_order(nullspace);
        // Smallest s with 2(p + s) > d + max_op:
        //   2p > d + max_op            ⇒ s = 0
        //   else s = ⌈(d + max_op + 1 − 2p) / 2⌉ = (d + max_op + 2 − 2p) / 2
        let s_op = if 2 * p > dim + max_operator_derivative_order {
            0
        } else {
            (dim + max_operator_derivative_order + 2 - 2 * p) / 2
        };
        if !pure || 2 * s_op < dim {
            return (nullspace, s_op);
        }
        nullspace = duchon_next_nullspace_order(nullspace);
    }
    // Bounded-loop fallback: by the analysis in the docstring, for
    // `p >= ceil((dim + max_op) / 2) + 1` the operator constraint admits
    // `s = 0` and (in pure mode) `0 < dim` satisfies the kernel-existence
    // condition. The loop above always reaches that regime within the bound,
    // so returning the last `nullspace` with `s = 0` is a valid answer.
    (nullspace, 0)
}


#[inline]
fn duchon_next_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Degree(2),
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k + 1),
    }
}


fn duchon_previous_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Degree(2) => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k - 1),
    }
}


/// Returns the maximum derivative order required by the *active* operator
/// penalties: 2 if stiffness is Active, else 1 if tension is Active, else 0.
/// Mass-only (or no active operator) penalties only require kernel validity
/// (`2(p+s) > d`), tension requires D1 collocation (`2(p+s) > d+1`), and
/// stiffness requires D2 collocation (`2(p+s) > d+2`).
pub fn duchon_max_active_operator_derivative_order(
    operator_penalties: &DuchonOperatorPenaltySpec,
) -> usize {
    if matches!(
        operator_penalties.stiffness,
        OperatorPenaltySpec::Active { .. }
    ) {
        2
    } else if matches!(
        operator_penalties.tension,
        OperatorPenaltySpec::Active { .. }
    ) {
        1
    } else {
        0
    }
}


/// Metadata returned by generic basis builders.
#[derive(Debug, Clone)]
pub enum BasisMetadata {
    BSpline1D {
        knots: Array1<f64>,
        identifiability_transform: Option<Array2<f64>>,
        periodic: Option<(f64, f64, usize)>,
        /// Effective B-spline polynomial degree carried by `knots`.
        ///
        /// Persisted alongside `knots` so prediction can reconstruct an
        /// evaluator that matches fit-time geometry, even when the fit-time
        /// auto-shrink (issue #340) reduced the user's requested degree to
        /// fit the available data (`n` too small for cubic ⇒ quadratic ⇒
        /// linear). When `None` the consumer should fall back to the
        /// upstream `BSplineBasisSpec.degree` (legacy / non-shrunk path).
        degree: Option<usize>,
        /// Human-readable description of an automatic basis shrink (issue #340)
        /// when the user's requested `(degree, num_internal_knots)` exceeded the
        /// available evaluation count `n`. `Some(note)` records the before→after
        /// configuration; `None` means no auto-shrink occurred for this basis.
        auto_shrink_note: Option<String>,
    },
    ThinPlate {
        centers: Array2<f64>,
        length_scale: f64,
        periodic: Option<Vec<Option<f64>>>,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Wood-TPRS radial reparameterization carried into prediction so the
        /// rotated radial basis at predict-time matches fit-time exactly. `None`
        /// in the lazy/streaming path which retains the original basis.
        radial_reparam: Option<Array2<f64>>,
    },
    Sphere {
        centers: Array2<f64>,
        penalty_order: usize,
        method: SphereMethod,
        max_degree: Option<usize>,
        wahba_kernel: SphereWahbaKernel,
        constraint_transform: Option<Array2<f64>>,
    },
    /// Constant-curvature (`M_κ`) geodesic-kernel smooth (#944). `kappa` and
    /// the realized `length_scale` are persisted so predict-time (and the
    /// future ψ-channel per-trial) rebuilds replay the exact fit-time
    /// geometry; `constraint_transform` is the composed `z · z_parametric`
    /// frozen by the global identifiability pipeline (#532 pattern).
    ConstantCurvature {
        centers: Array2<f64>,
        kappa: f64,
        length_scale: f64,
        constraint_transform: Option<Array2<f64>>,
    },
    /// Measure-jet spline smooth: multiscale local-jet-residual energy of the
    /// empirical measure, quadratured on the center set. `centers` are the
    /// REALIZED barycenter nodes; `order_s` stores the spec's order sentinel
    /// verbatim as the mode marker (0.0 = per-level/spectral, > 0 = fused
    /// pin — persisting a realized default would flip the rebuilt mode). The
    /// penalty depends on the FIT data through `masses`, the realized
    /// `eps_band`, the support anchors, and the normalization scales, so all
    /// are persisted and replayed verbatim by
    /// predict-time (and per-ψ-trial) rebuilds — recomputing either from
    /// predict rows would change the penalty the coefficients were estimated
    /// under. `constraint_transform` is the composed `z · z_parametric`
    /// frozen by the global identifiability pipeline (#532 pattern).
    MeasureJet {
        centers: Array2<f64>,
        input_scales: Option<Vec<f64>>,
        length_scale: f64,
        eps_band: Vec<f64>,
        order_s: f64,
        alpha: f64,
        tau0: f64,
        masses: Array1<f64>,
        support_means: Vec<f64>,
        penalty_normalization_scales: Vec<f64>,
        raw_penalty_normalization_scales: Vec<f64>,
        fused_penalty_normalization_scale: Option<f64>,
        constraint_transform: Option<Array2<f64>>,
    },
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        periodic: Option<Vec<Option<f64>>>,
        nu: MaternNu,
        include_intercept: bool,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a for geometric anisotropy.
        /// When Some, distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
        aniso_log_scales: Option<Vec<f64>>,
        /// Realized double-penalty nullspace-shrinkage decision at this build
        /// (gam#787/#860). The freeze step pins this into
        /// `MaternIdentifiability::FrozenTransform::nullspace_shrinkage_survived`
        /// so the κ-optimizer's per-trial rebuilds keep the learned-penalty count
        /// invariant (otherwise the κ-dependent spectral test flips it 6↔7 → "joint
        /// hyper rho dimension mismatch").
        nullspace_shrinkage_survived: bool,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        periodic: Option<Vec<Option<f64>>>,
        power: f64,
        nullspace_order: DuchonNullspaceOrder,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a, stored for prediction.
        aniso_log_scales: Option<Vec<f64>>,
        /// Support points used to build the active lower-order operator
        /// penalties (mass/tension/stiffness). Stored so runtime adaptive
        /// caches can rebuild the exact same operator rows instead of guessing
        /// from centers.
        operator_collocation_points: Option<Array2<f64>>,
    },
    Pca {
        feature_cols: Vec<usize>,
        basis_matrix: Array2<f64>,
        centered: bool,
        smooth_penalty: f64,
        center_mean: Option<Array1<f64>>,
        pca_basis_path: Option<std::path::PathBuf>,
        chunk_size: usize,
    },
    TensorBSpline {
        feature_cols: Vec<usize>,
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        periods: Vec<Option<f64>>,
        identifiability_transform: Option<Array2<f64>>,
    },
    SphereHarmonics {
        max_degree: usize,
        radians: bool,
    },
    /// Wrap an inner basis metadata to record a multiplicative `by` (continuous or
    /// factor) along a column of the dataset.
    BySmooth {
        inner: Box<BasisMetadata>,
        by_col: usize,
        levels: Option<Vec<u64>>,
        ordered: bool,
    },
    /// Factor-by-smooth (mgcv-style `s(x, by=g, bs="fs"|"sz"|"re")`).
    FactorSmooth {
        continuous_cols: Vec<usize>,
        group_col: usize,
        knots: Array1<f64>,
        degree: usize,
        periodic: Option<(f64, f64, usize)>,
        group_levels: Vec<u64>,
        flavour: String,
    },
}


/// Standardized basis build result for engine-level composition.
#[derive(Clone)]
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
    /// Per-active-penalty operator handles (parallel to `penalties`). Each
    /// entry is `Some(op)` when the closed-form factory emitted an op-form
    /// penalty bit-equivalent to the dense matrix, `None` for ordinary dense
    /// penalties. Downstream consumers route through the `Some` entries to
    /// avoid materializing dense `p x p` Grams in exact operator algebra.
    pub ops: Vec<Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>>,
    /// Per-active-penalty null-space eigenvector matrices (parallel to
    /// `penalties`). Each entry is `Some(U_null)` with `U_null.ncols() ==
    /// nullspace_dims[k]` when the active block has a non-trivial null space
    /// (eigenvalues ≤ spectral tolerance), and `None` when the block is
    /// already full-rank. The columns of `U_null` are the eigenvectors of
    /// `sym_penalty` at the (near-)zero eigenvalues — i.e., an orthonormal
    /// basis of `null(S_block)` in the block's own coordinate system.
    ///
    /// This is the raw spectral data that the construction pipeline uses to
    /// absorb each smooth's penalty null space into the parametric block
    /// (reparameterize-and-split). Without absorption the inner Newton solve
    /// cannot converge on data whose unpenalized signal lies along a null
    /// direction of `S` (phantom-multiplier refusal at the KKT certificate).
    pub null_eigenvectors: Vec<Option<Array2<f64>>>,
    /// Joint-null absorption rotation for this basis, when the basis carries
    /// any penalties with a non-trivial joint null space.
    ///
    /// `Some(rotation)` records `Q = [U_range | U_null]` where `U_null` spans
    /// the joint null space `null(Σ_k S_k)` over this basis's active
    /// penalties (unscaled — the structural joint null is independent of
    /// `λ`). After the basis pipeline applies this rotation, the design
    /// becomes `X · Q` and each penalty becomes `Qᵀ S_k Q`, block-diagonal
    /// with a guaranteed-zero null tail. The same `Q` must be replayed at
    /// prediction time, so it is persisted in the fitted model. `None`
    /// indicates either no penalties on this basis, or a full-rank joint
    /// penalty (joint nullity = 0). A `Some` value is never recorded with
    /// `joint_nullity == 0` — the `None` discriminant is canonical for
    /// "nothing to absorb".
    ///
    /// Stage-2 commit A: this field is plumbed into the struct but neither
    /// computed nor applied yet. Stage-2 commit B populates it; Stage-2
    /// commit D applies the rotation to `design` and `penalties`.
    pub joint_null_rotation: Option<JointNullRotation>,
}


/// Joint-null absorption rotation, attached to a smooth's basis when the
/// basis's joint penalty `Σ_k S_k` has a non-trivial null space.
///
/// The `rotation` field stores the orthonormal eigenvector matrix
/// `Q = [U_range | U_null]` of the symmetric joint penalty: the first
/// `range_dim = rotation.ncols() - joint_nullity` columns span
/// `range(Σ_k S_k)`; the remaining `joint_nullity` columns span
/// `null(Σ_k S_k)`. After the pipeline applies the rotation, the smooth's
/// coefficient vector satisfies `β = Q · γ`, the design becomes `X · Q`,
/// and each per-block penalty `S_k` becomes `Qᵀ S_k Q`, which is guaranteed
/// block-diagonal with a zero `(joint_nullity × joint_nullity)` tail
/// (because the joint null annihilates every active `S_k`).
#[derive(Clone, Serialize, Deserialize)]
pub struct JointNullRotation {
    /// `(p_smooth × p_smooth)` orthonormal matrix; range columns first,
    /// joint-null columns last.
    pub rotation: Array2<f64>,
    /// Number of columns at the tail of `rotation` that span the joint
    /// null space. Always `> 0` when wrapped in `Some` — the value `0`
    /// is encoded as `None`.
    pub joint_nullity: usize,
}


impl std::fmt::Debug for JointNullRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointNullRotation")
            .field(
                "rotation",
                &format_args!("{}×{}", self.rotation.nrows(), self.rotation.ncols()),
            )
            .field("joint_nullity", &self.joint_nullity)
            .finish()
    }
}


impl std::fmt::Debug for BasisBuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasisBuildResult")
            .field("design", &self.design)
            .field("penalties", &self.penalties)
            .field("nullspace_dims", &self.nullspace_dims)
            .field("penaltyinfo", &self.penaltyinfo)
            .field("metadata", &self.metadata)
            .field("kronecker_factored", &self.kronecker_factored)
            .field(
                "ops",
                &format_args!(
                    "[{}]",
                    self.ops
                        .iter()
                        .map(|o| if o.is_some() { "Some" } else { "None" })
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .field(
                "null_eigenvectors",
                &format_args!(
                    "[{}]",
                    self.null_eigenvectors
                        .iter()
                        .map(|u| match u {
                            Some(m) => format!("Some({}x{})", m.nrows(), m.ncols()),
                            None => "None".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .field("joint_null_rotation", &self.joint_null_rotation)
            .finish()
    }
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
    /// One per input axis `a` of a multivariate Duchon smooth: the gradient
    /// energy along axis `a`, `Σ(∂f/∂x_a)²`, each with its own REML λ_a. REML
    /// shrinks an axis's contribution toward flat only when it does not earn
    /// its keep — penalty-based ARD / variable relevance, the replacement for
    /// brittle kernel-η optimization. Emitted when `scale_dims` is on.
    OperatorRelevance {
        axis: usize,
    },
    TensorMarginal {
        dim: usize,
    },
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


#[derive(Clone)]
pub struct PenaltyCandidate {
    pub matrix: Array2<f64>,
    pub nullspace_dim_hint: usize,
    pub source: PenaltySource,
    pub normalization_scale: f64,
    /// Optional Kronecker factors whose product equals `matrix`.
    /// When present, spectral decomposition can be done per-factor
    /// (O(Σ q_j³) instead of O((Π q_j)³)).
    pub kronecker_factors: Option<Vec<Array2<f64>>>,
    /// Optional operator-form handle whose `as_dense()` matches `matrix`. When
    /// populated by the closed-form factories, this is propagated through to
    /// `CanonicalPenaltyBlock` so downstream consumers can use exact matvec
    /// algebra without rebuilding the dense Gram. When `None`, only the dense
    /// `matrix` path is available.
    pub op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
}


impl std::fmt::Debug for PenaltyCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PenaltyCandidate")
            .field(
                "matrix",
                &format_args!("{}×{}", self.matrix.nrows(), self.matrix.ncols()),
            )
            .field("nullspace_dim_hint", &self.nullspace_dim_hint)
            .field("source", &self.source)
            .field("normalization_scale", &self.normalization_scale)
            .field(
                "kronecker_factors",
                &self.kronecker_factors.as_ref().map(|v| v.len()),
            )
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}


#[derive(Clone)]
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
    /// Optional operator-form handle that is bit-equivalent to `sym_penalty`.
    /// Propagated from `PenaltyCandidate.op` when present so downstream
    /// consumers can use matvec without rebuilding the dense Gram.
    pub op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
}


impl std::fmt::Debug for CanonicalPenaltyBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalPenaltyBlock")
            .field(
                "sym_penalty",
                &format_args!("{}×{}", self.sym_penalty.nrows(), self.sym_penalty.ncols()),
            )
            .field("eigenvalues", &self.eigenvalues)
            .field(
                "eigenvectors",
                &format_args!(
                    "{}×{}",
                    self.eigenvectors.nrows(),
                    self.eigenvectors.ncols()
                ),
            )
            .field("rank", &self.rank)
            .field("nullity", &self.nullity)
            .field("tol", &self.tol)
            .field("iszero", &self.iszero)
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}


#[derive(Debug)]
pub struct BasisPsiDerivativeResult {
    pub design_derivative: Array2<f64>,
    pub penalties_derivative: Vec<Array2<f64>>,
    /// Operator-backed design derivative for standalone first-derivative
    /// callers. Bundled first+second callers receive the shared operator on
    /// `BasisPsiDerivativeBundle` instead.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}


#[derive(Debug)]
pub struct BasisPsiSecondDerivativeResult {
    pub designsecond_derivative: Array2<f64>,
    pub penaltiessecond_derivative: Vec<Array2<f64>>,
    /// Operator-backed design derivative for standalone second-derivative
    /// callers. Bundled first+second callers receive the shared operator on
    /// `BasisPsiDerivativeBundle` instead.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}


#[derive(Debug)]
pub struct BasisPsiDerivativeBundle {
    pub first: BasisPsiDerivativeResult,
    pub second: BasisPsiSecondDerivativeResult,
    /// Shared operator-backed design derivative for the first and second
    /// psi derivatives. Bundled callers consume this once instead of storing
    /// duplicate materialized/streaming operators in both derivative payloads.
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


fn implicit_radial_cache_bytes(n: usize, k: usize, n_axes: usize) -> usize {
    n.saturating_mul(k)
        .saturating_mul(n_axes.saturating_add(3))
        .saturating_mul(8)
}


fn should_cache_implicit_radial_components(
    n: usize,
    k: usize,
    n_axes: usize,
    policy: &crate::resource::ResourcePolicy,
) -> bool {
    implicit_radial_cache_bytes(n, k, n_axes) <= policy.max_operator_cache_bytes
}


pub fn assert_no_dense_derivative_materialization(n: usize, p: usize, d_pc: usize) {
    let first = dense_design_bytes(n, p).saturating_mul(d_pc);
    let second = dense_design_bytes(n, p).saturating_mul(d_pc.saturating_mul(d_pc));
    // Consult the library default ResourcePolicy. Production large-scale runs
    // configure `AnalyticOperatorRequired`, which still refuses every dense
    // materialization here. The default `MaterializeIfSmall` mode lets tiny
    // problems (and small-data/test usage) materialize as long as the combined
    // first- and second-order dense bytes fit under the single-materialization
    // byte budget. `DiagnosticsOnly` is treated like `MaterializeIfSmall` for
    // this guard: it permits dense materialization under the same byte cap.
    let policy = crate::resource::ResourcePolicy::default_library();
    let budget = policy.max_single_materialization_bytes;
    let needed = first.saturating_add(second);
    match policy.derivative_storage_mode {
        crate::resource::DerivativeStorageMode::AnalyticOperatorRequired => {
            // SAFETY: this assertion helper exists specifically to enforce
            // the large-scale invariant that spatial-PC Duchon derivative
            // designs never persist as dense `Array2<f64>` storage. When the
            // resource policy is `AnalyticOperatorRequired`, any caller that
            // reached this point has materialized something the strict
            // operator contract forbids.
            // SAFETY: AnalyticOperatorRequired forbids dense derivative materialization.
            panic!(
                "spatial PC Duchon derivative designs must remain operator-backed; refused persistent dense derivative materialization (n={n}, p={p}, d_pc={d_pc}, first_order={:.1} MiB, second_order={:.1} MiB)",
                first as f64 / (1024.0 * 1024.0),
                second as f64 / (1024.0 * 1024.0),
            );
        }
        crate::resource::DerivativeStorageMode::MaterializeIfSmall
        | crate::resource::DerivativeStorageMode::DiagnosticsOnly => {
            assert!(
                needed <= budget,
                "spatial PC Duchon derivative designs would exceed the single-materialization budget; refused persistent dense derivative materialization (n={n}, p={p}, d_pc={d_pc}, first_order={:.1} MiB, second_order={:.1} MiB, budget={:.1} MiB)",
                first as f64 / (1024.0 * 1024.0),
                second as f64 / (1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0),
            );
        }
    }
}


pub fn assert_spatial_centers_below_large_scale_cap(
    d_pc: usize,
    centers: ArrayView2<'_, f64>,
) -> Result<(), BasisError> {
    if centers.ncols() != d_pc {
        crate::bail_dim_basis!(
            "spatial PC center dimension mismatch: centers have {} columns, expected {d_pc}",
            centers.ncols()
        );
    }
    let k = centers.nrows();
    let centers_bytes = dense_design_bytes(k, d_pc);
    let center_center_bytes = dense_design_bytes(k, k);
    if centers_bytes > SPATIAL_CENTER_CENTER_MAX_BYTES {
        crate::bail_invalid_basis!(
            "spatial PC centers exceed center storage cap: K={k}, d_pc={d_pc}, centers={:.1} MiB, cap={:.1} MiB",
            centers_bytes as f64 / (1024.0 * 1024.0),
            SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
        );
    }
    if center_center_bytes > SPATIAL_CENTER_CENTER_MAX_BYTES {
        crate::bail_invalid_basis!(
            "spatial PC centers exceed center-center large-scale cap: K={k}, d_pc={d_pc}, KxK={:.1} MiB, cap={:.1} MiB",
            center_center_bytes as f64 / (1024.0 * 1024.0),
            SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
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


/// Single-pass `(Bᵀ(W·C), BᵀB)` accumulation over the streamed design.
///
/// Materialises each row chunk of the design **once** and reuses it for both
/// the constraint cross `Bᵀ(W·C)` and the Gram `BᵀB`. On the lazy chunked
/// spatial path each `try_row_chunk` re-evaluates all kernel columns for the
/// chunk, so accumulating both products in a single sweep halves the per-build
/// kernel re-evaluation work (the dominant cost at large scale) versus two
/// independent streaming passes — without changing the result beyond
/// floating-point reassociation. The cross is masked off (`q == 0`) by the
/// caller, which never invokes this when there is no constraint block.
fn design_cross_and_gram(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
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
    let mut gram = Array2::<f64>::zeros((k, k));
    for start in (0..n).step_by(DESIGN_CROSS_CHUNK_SIZE) {
        let end = (start + DESIGN_CROSS_CHUNK_SIZE).min(n);
        let basis_chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let mut constraint_chunk = constraint_matrix.slice(s![start..end, ..]).to_owned();
        if let Some(w) = weights {
            for (mut row, &weight) in constraint_chunk
                .axis_iter_mut(Axis(0))
                .zip(w.slice(s![start..end]).iter())
            {
                row *= weight;
            }
        }
        cross += &fast_atb(&basis_chunk, &constraint_chunk);
        gram += &fast_atb(&basis_chunk, &basis_chunk);
    }
    Ok((cross, gram))
}


fn positive_spectral_whitener_from_gram(gram: &Array2<f64>) -> Result<Array2<f64>, BasisError> {
    // Inverse-square-root for the positive part of `gram`. Eigenvalues at or
    // below the relative rank tolerance `α·ε·n·max_eval` are *dropped*: the
    // returned whitener has shape `(n × keep)` where `keep` counts strictly
    // positive eigendirections of `gram`.
    //
    // Dropping (rather than ridging) is what makes the result a true
    // square-root inverse on the column space of `gram`. This whitener is
    // used by `stabilized_orthogonality_transform_from_gram` to make a
    // pre-existing transform `K_raw` orthonormal under the W-inner product:
    // when some columns of `K_raw` map to zero (or near-zero) under `B`, the
    // constrained Gram `K_raw^T G K_raw` is rank-deficient. Ridging those
    // tail directions with `1/sqrt(ε)` produced spurious basis columns
    // whose coefficient norms blew up to `~1/sqrt(ε)` while their image in
    // `B` was floating-point zero, contaminating downstream linear algebra
    // (in particular it forced `smooth.rs` to widen the post-transform
    // orthogonality residual tolerance to absorb a `cond ≈ 1/sqrt(ε)`
    // rounding floor). Dropping these directions is the right behavior:
    // they contribute nothing to `B`'s column space, and removing them
    // tightens the orthogonality residual back down to the genuine
    // floating-point limit.
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(BasisError::LinalgError)?;
    let n = gram.nrows();
    let max_eval = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tol = (default_rrqr_rank_alpha() * f64::EPSILON * (n.max(1) as f64) * max_eval.max(1.0))
        .max(f64::EPSILON);
    let keep = eigenvalues.iter().filter(|&&ev| ev > tol).count();
    if keep == 0 {
        let min_ev = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "positive_spectral_whitener_from_gram",
            cross_rank: 0,
            coeff_dim: gram.nrows(),
            cross_frobenius: gram.iter().map(|v| v * v).sum::<f64>().sqrt(),
            constrained_gram_max_eigenvalue: max_eval,
            constrained_gram_min_eigenvalue: min_ev,
            spectral_tolerance: tol,
        });
    }
    // `eigh` returns eigenvalues in ascending order, so the largest `keep`
    // eigenvalues live at the tail.
    let eig_start = eigenvalues.len() - keep;
    let kept_vectors = eigenvectors.slice(s![.., eig_start..]).to_owned();
    let mut inv_sqrt = Array2::<f64>::zeros((keep, keep));
    for (out_i, eig_i) in (eig_start..eigenvalues.len()).enumerate() {
        inv_sqrt[[out_i, out_i]] = 1.0 / eigenvalues[eig_i].sqrt();
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


fn orthogonality_transform_from_cross_and_gram(
    constraint_cross: &Array2<f64>,
    gram: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    // Compute null(M^T) directly on M = B^T W C (k × q) via column-pivoted QR.
    // Working in the original k-dim coefficient space rather than first
    // whitening by B^T B avoids a fundamental failure mode: when B is heavily
    // collinear, `positive_spectral_whitener_from_gram` truncates the design
    // column-space to a `keep`-dim subspace, and if `keep <= q` the subsequent
    // nullspace search has no room — even though dim null(M^T) = k - rank(M)
    // ≥ k - q is always positive when k > q. The constraint nullspace is a
    // property of M alone; conditioning of the design only matters for the
    // downstream stabilization of B*K_raw.
    let k = constraint_cross.nrows();
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let (transform_raw, rank) = rrqr_nullspace_basis(constraint_cross, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rank >= k || transform_raw.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "orthogonality_transform_from_cross_and_gram",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: constraint_cross.iter().map(|v| v * v).sum::<f64>().sqrt(),
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
        });
    }

    // Make the constrained design B*K_raw orthonormal under the W-inner product.
    // If the constrained Gram K_raw^T G K_raw is rank-deficient (because some
    // directions in null(M^T) collapse under B), the spectral whitener drops
    // them — that is the right behavior: a degenerate column never contributes
    // to B's column space and shouldn't appear in the reparameterized basis.
    stabilized_orthogonality_transform_from_gram(gram, &transform_raw)
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
    let (constraint_cross, gram) = design_cross_and_gram(design, constraint_matrix, weights)?;
    orthogonality_transform_from_cross_and_gram(&constraint_cross, &gram)
}


/// Which radial kernel family is being used. Stored in the streaming operator
/// so that (q, t) scalars can be recomputed on the fly without a closure.
#[derive(Debug, Clone)]
pub enum RadialScalarKind {
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
    /// Thin-Plate Spline kernel: isotropic with a scalar length-scale, used
    /// only with `n_axes = 1` (`ScalarTotal` streaming mode). The chain rule
    /// for ψ = log κ = -log(length_scale) and r̃ = ‖x − c‖ gives
    /// ∂φ/∂ψ   = φ'(z)·z
    /// ∂²φ/∂ψ² = φ''(z)·z² + φ'(z)·z
    /// where z = r̃ / length_scale. With the operator's c=0, s_0 = r̃²,
    /// `shape_0 = q·s_0` and `shape_00 = t·s_0² + 2 q s_0` reproduce both
    /// derivatives exactly when q = φ'(r̃)/r̃ and t = (φ''(r̃) − q)/r̃².
    ThinPlate { length_scale: f64, dim: usize },
}


impl RadialScalarKind {
    /// Evaluate the `(phi, q, t)` radial scalars for a given distance `r`.
    ///
    /// `q = φ'(r)/r` and `t = (φ''(r) - q)/r²` (with the appropriate
    /// finite limits at `r → 0`). This is exactly the scalar pair needed
    /// to assemble the first and second derivatives of `Φ(t) = φ(‖t − c‖)`
    /// with respect to the input location `t`:
    ///
    /// ```text
    /// ∂Φ/∂t_a       = q · (t − c)_a
    /// ∂²Φ/∂t_a∂t_b  = q · δ_ab + t · (t − c)_a (t − c)_b
    /// ```
    ///
    /// Re-pointing the existing ψ-derivative machinery at the first kernel
    /// argument t (see `crate::terms::input_loc_derivatives`).
    ///
    /// Returns `true` iff both `q = φ'(r)/r` and `t = (φ''(r) − q)/r²` have
    /// finite limits as `r → 0+` for this kernel. When this returns `false`
    /// the design-row gradient/Hessian at a center collision (`r = 0`) is not
    /// defined by a single finite value; callers must either move off the
    /// collision or surface a `BasisError::DegenerateAtCollision`.
    ///
    /// Smoothness criteria used here (matching the analytic limits derived
    /// in this file and the comments on `eval_design_triplet`):
    ///   - Matérn ν = 1/2: `q = -s·E/r → -∞`, not smooth.
    ///   - Matérn ν = 3/2: `q` finite but `t = s³E/r → ∞`, not smooth.
    ///   - Matérn ν = 5/2, 7/2, 9/2: both finite, smooth.
    ///   - Duchon hybrid (`Duchon`): finite via the hybrid PFD identity;
    ///     the radial-jets routine produces a finite limit, so smooth.
    ///   - PureDuchon (raw polyharmonic block, exponent α = 2m − d):
    ///       non-log case and α ≥ 4 ⇒ both `q` and `t` vanish (smooth);
    ///       log case at any α, or α < 4 ⇒ at least one derivative diverges.
    ///   - ThinPlate dim = 1: φ = r³, `q = 3r → 0`, but `t = 3/r → ∞`. The
    ///     1-D Hessian formula `q·δ + t·s·s` at r = 0 has the only diagonal
    ///     entry contracted by `s_a = 0`, but the bare scalar limit is still
    ///     not finite, so we report it as non-smooth and let callers in 1-D
    ///     (where `s_a` literally vanishes) opt in by handling the error.
    ///     Dim 2 (log r), Dim 3 (-r) both diverge.
    #[inline]
    pub(crate) fn is_smooth_at_collision(&self) -> bool {
        match self {
            RadialScalarKind::Matern { nu, .. } => matches!(
                nu,
                MaternNu::FiveHalves | MaternNu::SevenHalves | MaternNu::NineHalves
            ),
            RadialScalarKind::Duchon { .. } => true,
            RadialScalarKind::PureDuchon {
                p_order,
                s_order,
                dim,
                ..
            } => {
                let alpha = duchon_scaling_exponent(*p_order, *s_order, *dim);
                let is_log = (*dim) % 2 == 0 && {
                    let half = (alpha / 2.0).round();
                    half >= 0.0 && (half * 2.0 - alpha).abs() < 1e-12
                };
                !is_log && alpha >= 4.0
            }
            RadialScalarKind::ThinPlate { .. } => false,
        }
    }

    pub(crate) fn eval_design_triplet(&self, r: f64) -> Result<(f64, f64, f64), BasisError> {
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
                let phi = polyharmonic_kernel(r, (*block_order) as f64, *dim);
                if r < 1e-14 {
                    // Collision: q = φ'/r and t = (φ'' − q)/r² generally
                    // diverge here. Only the non-log, α = 2m − d ≥ 4 case
                    // gives finite limits (both 0). Otherwise the design
                    // gradient/Hessian at r = 0 is undefined: surface a
                    // `DegenerateAtCollision` so callers can detect it.
                    if !self.is_smooth_at_collision() {
                        return Err(BasisError::DegenerateAtCollision {
                            kernel: "PureDuchon (polyharmonic)",
                            dim: *dim,
                            m: *block_order as f64,
                            message: "raw polyharmonic block φ(r) = c r^α (log r) is \
                                      not C² at r = 0 for α = 2m − d < 4 or for log \
                                      cases; first/second radial derivatives diverge",
                        });
                    }
                    return Ok((phi, 0.0, 0.0));
                }
                let (q, t, _, _) =
                    duchon_polyharmonic_operator_block_jets(r, *block_order as f64, *dim)?;
                Ok((phi, q, t))
            }
            RadialScalarKind::ThinPlate { length_scale, dim } => {
                // (q, t) individually diverge at r = 0 for ThinPlate
                // (q = 2 log r + 1 → −∞ in dim 2, q = −1/r → −∞ in dim 3,
                // t = 3/r → ∞ in dim 1, …) but the chain-rule coefficient
                // `c = raw_psi_isotropic_share` is 0 for ThinPlate, so every
                // consumer multiplies q by a squared displacement s_a and t
                // by s_a · s_b before use (design row uses φ alone, and
                // φ(0) = 0). The products
                //   q · s_a = (φ'(r) · r) · (s_a / r²),
                //   t · s_a · s_b = (φ''(r) · r² − φ'(r) · r) · (s_a/r²)·(s_b/r²)
                // both vanish as r → 0+, since r · φ'(r) → 0 and r² · φ''(r) → 0
                // for every standard TPS kernel (φ = r³ in dim 1, r² log r in
                // dim 2, −r in dim 3, and the general polyharmonic case for
                // d ≥ 4) and the ratios s_a/r², s_b/r² are bounded. The
                // closed-form ψ-derivative limit at the collision is
                // therefore (0, 0, 0).
                if r < 1e-14 {
                    return Ok((0.0, 0.0, 0.0));
                }
                let scaled_r = r / *length_scale;
                let (phi, phi_kernel_first, phi_kernel_second) =
                    thin_plate_kernel_triplet_from_scaled_distance(scaled_r, *dim)?;
                // The implicit operator uses derivatives w.r.t. the unscaled r
                // (the operator's chain rule will rescale them to ψ-derivatives
                // via s_0 = r²). Convert φ'(z), φ''(z) → φ'(r), φ''(r) by the
                // length-scale chain rule:
                //   φ'(r)  = φ'(z) / length_scale
                //   φ''(r) = φ''(z) / length_scale²
                let phi_r = phi_kernel_first / *length_scale;
                let phi_rr = phi_kernel_second / (*length_scale * *length_scale);
                let q = phi_r / r;
                let t = (phi_rr - q) / (r * r);
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
            // ThinPlate is a pure radial kernel φ(z) with no κ^δ prefactor;
            // the chain rule has no isotropic share term.
            RadialScalarKind::ThinPlate { .. } => 0.0,
        }
    }

    #[inline]
    fn is_duchon_family(&self) -> bool {
        matches!(
            self,
            RadialScalarKind::Duchon { .. } | RadialScalarKind::PureDuchon { .. }
        )
    }

    /// Whether the radial-kind enforces a hard guard against accidental
    /// dense `(n × p)` ψ-derivative materialization. Duchon-family terms
    /// always do (they are streaming-only at any scale). ThinPlate joins
    /// the guard list because the new scalar-streaming routing makes it
    /// genuine to rely on the implicit operator at large scale, and a
    /// downstream consumer that sneaks in a `materialize_dense()` call
    /// would silently re-introduce the same `n × p` allocation we wired
    /// streaming to avoid. The guard panics only when the resource
    /// policy says the materialization would exceed budget — small `n`
    /// problems still get the dense fast path.
    #[inline]
    fn enforces_dense_materialization_budget(&self) -> bool {
        matches!(
            self,
            RadialScalarKind::Duchon { .. }
                | RadialScalarKind::PureDuchon { .. }
                | RadialScalarKind::ThinPlate { .. }
        )
    }
}


/// Shared chunked-operator machinery for the streaming basis evaluators.
///
/// `StreamingMaternEvaluator`, `StreamingSphereEvaluator` and
/// `StreamingBSplineEvaluator` differ only in how a single row chunk of the
/// design is materialized (`for_row_chunk`) and the chunk size policy
/// (`chunk_rows`); every other operator method — the chunked matvec,
/// transpose-matvec, weighted Gram and dense materialization — is identical
/// boilerplate over that one primitive. This trait carries those shared
/// methods as defaults keyed on `for_row_chunk`, so each evaluator implements
/// only the per-basis pieces. The `NAME` const bakes the struct name into the
/// panic/error strings so diagnostics stay per-evaluator.
trait ChunkedDesign {
    /// Struct name used in assertion / error messages.
    const NAME: &'static str;

    /// Number of design rows (observations).
    fn op_nrows(&self) -> usize;

    /// Number of design columns (basis functions after any transform).
    fn op_ncols(&self) -> usize;

    /// Row-block size used to bound the per-chunk working set.
    fn chunk_rows(&self) -> usize;

    /// Materialize the dense design rows `[start, end)` — the only genuinely
    /// per-evaluator computation.
    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64>;

    /// Chunked matvec `output = X · theta`.
    fn chunked_gradient_into(&self, theta: ArrayView1<'_, f64>, output: &mut Array1<f64>) {
        assert_eq!(
            theta.len(),
            self.op_ncols(),
            "{} theta width mismatch",
            Self::NAME
        );
        assert_eq!(
            output.len(),
            self.op_nrows(),
            "{} output length mismatch",
            Self::NAME
        );
        output.fill(0.0);
        let nrows = self.op_nrows();
        for start in (0..nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(nrows);
            let chunk = self.for_row_chunk(start, end);
            let values = chunk.dot(&theta);
            output.slice_mut(s![start..end]).assign(&values);
        }
    }

    /// Chunked matvec returning a fresh vector (`LinearOperator::apply`).
    fn chunked_apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.op_nrows());
        self.chunked_gradient_into(vector.view(), &mut out);
        out
    }

    /// Chunked transpose-matvec `out = Xᵀ · vector`
    /// (`LinearOperator::apply_transpose`).
    fn chunked_apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.op_nrows(),
            "{} transpose vector length mismatch",
            Self::NAME
        );
        let nrows = self.op_nrows();
        let mut out = Array1::<f64>::zeros(self.op_ncols());
        for start in (0..nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(nrows);
            let chunk = self.for_row_chunk(start, end);
            let partial = chunk.t().dot(&vector.slice(s![start..end]));
            out += &partial;
        }
        out
    }

    /// Chunked weighted Gram `XᵀWX` (`LinearOperator::diag_xtw_x`).
    fn chunked_diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>
    where
        Self: Sync,
    {
        let nrows = self.op_nrows();
        if weights.len() != nrows {
            return Err(format!(
                "{} diag_xtw_x weight length mismatch: weights={}, nrows={}",
                Self::NAME,
                weights.len(),
                nrows
            ));
        }
        let p = self.op_ncols();
        let chunk_rows = self.chunk_rows();
        let starts = (0..nrows).step_by(chunk_rows).collect::<Vec<_>>();
        Ok(starts
            .into_par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut acc, start| {
                    let end = (start + chunk_rows).min(nrows);
                    let chunk = self.for_row_chunk(start, end);
                    let mut weighted = chunk.clone();
                    for local in 0..(end - start) {
                        let w = weights[start + local];
                        weighted.row_mut(local).mapv_inplace(|v| v * w);
                    }
                    acc += &chunk.t().dot(&weighted);
                    acc
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut a, b| {
                    a += &b;
                    a
                },
            ))
    }

    /// Chunked dense row fill (`DenseDesignOperator::row_chunk_into`).
    fn chunked_row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if rows.end > self.op_nrows() || rows.start > rows.end {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: Self::ROW_RANGE_OOB,
            });
        }
        if out.nrows() != rows.end - rows.start || out.ncols() != self.op_ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: Self::ROW_CHUNK_SHAPE_MISMATCH,
            });
        }
        out.assign(&self.for_row_chunk(rows.start, rows.end));
        Ok(())
    }

    /// Full dense materialization (`DenseDesignOperator::to_dense`).
    fn chunked_to_dense(&self) -> Array2<f64> {
        self.for_row_chunk(0, self.op_nrows())
    }

    /// Static `&str` context strings for the row-chunk errors — kept as
    /// associated consts because `MatrixMaterializationError::MissingRowChunk`
    /// stores `&'static str`, so a runtime-formatted name cannot be used.
    const ROW_RANGE_OOB: &'static str;
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str;
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingMaternEvaluator {
    data: Arc<Array2<f64>>,
    centers: Arc<Array2<f64>>,
    length_scale: f64,
    nu: MaternNu,
    metric_weights: Arc<[f64]>,
    ident_transform: Option<Arc<Array2<f64>>>,
    include_intercept: bool,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingMaternEvaluator {
    pub(crate) fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        length_scale: f64,
        nu: MaternNu,
        aniso_log_scales: Option<Vec<f64>>,
        ident_transform: Option<Arc<Array2<f64>>>,
        include_intercept: bool,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        if data.ncols() != centers.ncols() {
            return Err(format!(
                "StreamingMaternEvaluator: data dim {} != centers dim {}",
                data.ncols(),
                centers.ncols()
            ));
        }
        let metric_weights = match aniso_log_scales {
            Some(eta) => {
                if eta.len() != data.ncols() {
                    return Err(format!(
                        "StreamingMaternEvaluator: aniso_log_scales len {} != data dim {}",
                        eta.len(),
                        data.ncols()
                    ));
                }
                eta.into_iter().map(|v| (2.0 * v).exp()).collect::<Vec<_>>()
            }
            None => vec![1.0; data.ncols()],
        };
        if let Some(z) = ident_transform.as_ref()
            && z.nrows() != centers.nrows()
        {
            return Err(format!(
                "StreamingMaternEvaluator: identifiability transform rows {} != centers {}",
                z.nrows(),
                centers.nrows()
            ));
        }
        let kernel_cols = ident_transform
            .as_ref()
            .map_or(centers.nrows(), |z| z.ncols());
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            centers: Arc::new(centers.as_standard_layout().to_owned()),
            length_scale,
            nu,
            metric_weights: Arc::from(metric_weights),
            ident_transform,
            include_intercept,
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
            total_cols: kernel_cols + usize::from(include_intercept),
        })
    }

    fn raw_kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_raw = self.centers.nrows();
        let dim = self.data.ncols();
        let data = self
            .data
            .as_slice()
            .expect("StreamingMaternEvaluator stores standard-layout data");
        let centers = self
            .centers
            .as_slice()
            .expect("StreamingMaternEvaluator stores standard-layout centers");
        let mut values = vec![0.0_f64; chunk_n * k_raw];
        values
            .par_chunks_mut(k_raw)
            .enumerate()
            .for_each(|(local, out_row)| {
                let global = rows.start + local;
                let x = &data[global * dim..(global + 1) * dim];
                for j in 0..k_raw {
                    let c = &centers[j * dim..(j + 1) * dim];
                    let mut r2 = 0.0_f64;
                    for axis in 0..dim {
                        let h = x[axis] - c[axis];
                        r2 += self.metric_weights[axis] * h * h;
                    }
                    out_row[j] = matern_kernel_from_distance(r2.sqrt(), self.length_scale, self.nu)
                        .expect("validated Matérn inputs should not fail");
                }
            });
        Array2::from_shape_vec((chunk_n, k_raw), values)
            .expect("StreamingMaternEvaluator chunk shape should match generated values")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_kernel_chunk(start..end);
        let kernel = match self.ident_transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };
        if !self.include_intercept {
            return kernel;
        }
        let mut out = Array2::<f64>::ones((end - start, kernel.ncols() + 1));
        out.slice_mut(s![.., ..kernel.ncols()]).assign(&kernel);
        out
    }
}


impl ChunkedDesign for StreamingMaternEvaluator {
    const NAME: &'static str = "StreamingMaternEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingMaternEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingMaternEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.nrows()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.nrows().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.nrows(),
            "StreamingMaternEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingMaternEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingMaternEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingSphereEvaluator {
    data: Arc<Array2<f64>>,
    centers: Arc<Array2<f64>>,
    penalty_order: usize,
    radians: bool,
    wahba_kernel: SphereWahbaKernel,
    constraint_transform: Option<Arc<Array2<f64>>>,
    sin_lat_c: Arc<[f64]>,
    cos_lat_c: Arc<[f64]>,
    sin_lon_c: Arc<[f64]>,
    cos_lon_c: Arc<[f64]>,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingSphereEvaluator {
    pub(crate) fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        penalty_order: usize,
        radians: bool,
        wahba_kernel: SphereWahbaKernel,
        constraint_transform: Option<Arc<Array2<f64>>>,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        validate_lat_lon_matrix(data.view(), "StreamingSphereEvaluator data", radians)
            .map_err(|e| e.to_string())?;
        validate_lat_lon_matrix(centers.view(), "StreamingSphereEvaluator centers", radians)
            .map_err(|e| e.to_string())?;
        if !(1..=4).contains(&penalty_order) {
            return Err(format!(
                "StreamingSphereEvaluator: penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
            ));
        }
        if let Some(z) = constraint_transform.as_ref()
            && z.nrows() != centers.nrows()
        {
            return Err(format!(
                "StreamingSphereEvaluator: constraint transform rows {} != centers {}",
                z.nrows(),
                centers.nrows()
            ));
        }
        let deg = if radians {
            1.0
        } else {
            std::f64::consts::PI / 180.0
        };
        let mut sin_lat_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut cos_lat_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut sin_lon_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut cos_lon_c = Vec::<f64>::with_capacity(centers.nrows());
        for c in centers.outer_iter() {
            let (s_lat, c_lat) = (c[0] * deg).sin_cos();
            let (s_lon, c_lon) = (c[1] * deg).sin_cos();
            sin_lat_c.push(s_lat);
            cos_lat_c.push(c_lat);
            sin_lon_c.push(s_lon);
            cos_lon_c.push(c_lon);
        }
        let total_cols = constraint_transform
            .as_ref()
            .map_or(centers.nrows(), |z| z.ncols());
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            centers: Arc::new(centers.as_standard_layout().to_owned()),
            penalty_order,
            radians,
            wahba_kernel,
            constraint_transform,
            sin_lat_c: Arc::from(sin_lat_c),
            cos_lat_c: Arc::from(cos_lat_c),
            sin_lon_c: Arc::from(sin_lon_c),
            cos_lon_c: Arc::from(cos_lon_c),
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
            total_cols,
        })
    }

    fn raw_kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k = self.centers.nrows();
        let deg = if self.radians {
            1.0
        } else {
            std::f64::consts::PI / 180.0
        };
        let mut values = vec![0.0_f64; chunk_n * k];
        values
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(local, out_row)| {
                use wide::f64x4;
                let row = rows.start + local;
                let lat = self.data[[row, 0]] * deg;
                let lon = self.data[[row, 1]] * deg;
                let (sin_lat, cos_lat) = lat.sin_cos();
                let (sin_lon, cos_lon) = lon.sin_cos();
                let sin_lat_v = f64x4::from(sin_lat);
                let cos_lat_v = f64x4::from(cos_lat);
                let sin_lon_v = f64x4::from(sin_lon);
                let cos_lon_v = f64x4::from(cos_lon);
                for cidx in 0..(k / 4) {
                    let base = cidx * 4;
                    let sl_c = f64x4::from([
                        self.sin_lat_c[base],
                        self.sin_lat_c[base + 1],
                        self.sin_lat_c[base + 2],
                        self.sin_lat_c[base + 3],
                    ]);
                    let cl_c = f64x4::from([
                        self.cos_lat_c[base],
                        self.cos_lat_c[base + 1],
                        self.cos_lat_c[base + 2],
                        self.cos_lat_c[base + 3],
                    ]);
                    let sn_c = f64x4::from([
                        self.sin_lon_c[base],
                        self.sin_lon_c[base + 1],
                        self.sin_lon_c[base + 2],
                        self.sin_lon_c[base + 3],
                    ]);
                    let cn_c = f64x4::from([
                        self.cos_lon_c[base],
                        self.cos_lon_c[base + 1],
                        self.cos_lon_c[base + 2],
                        self.cos_lon_c[base + 3],
                    ]);
                    let dlon_cos = cos_lon_v * cn_c + sin_lon_v * sn_c;
                    let cos_gamma = sin_lat_v * sl_c + cos_lat_v * cl_c * dlon_cos;
                    let arr = wahba_sphere_kernel_from_cos_simd_kind(
                        cos_gamma,
                        self.penalty_order,
                        self.wahba_kernel,
                    )
                    .to_array();
                    out_row[base..base + 4].copy_from_slice(&arr);
                }
                let tail_start = (k / 4) * 4;
                for j in tail_start..k {
                    let dlon_cos = cos_lon * self.cos_lon_c[j] + sin_lon * self.sin_lon_c[j];
                    let cos_gamma =
                        sin_lat * self.sin_lat_c[j] + cos_lat * self.cos_lat_c[j] * dlon_cos;
                    out_row[j] = wahba_sphere_kernel_from_cos_kind(
                        cos_gamma,
                        self.penalty_order,
                        self.wahba_kernel,
                    )
                    .expect("validated sphere kernel inputs should not fail");
                }
            });
        Array2::from_shape_vec((chunk_n, k), values)
            .expect("StreamingSphereEvaluator chunk shape should match generated values")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_kernel_chunk(start..end);
        match self.constraint_transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        }
    }
}


impl ChunkedDesign for StreamingSphereEvaluator {
    const NAME: &'static str = "StreamingSphereEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingSphereEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingSphereEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.nrows()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.nrows().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.nrows(),
            "StreamingSphereEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingSphereEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingSphereEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingBSplineEvaluator {
    data: Arc<Array1<f64>>,
    knots: Arc<Array1<f64>>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    transform: Option<Arc<Array2<f64>>>,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingBSplineEvaluator {
    pub(crate) fn new(
        data: Arc<Array1<f64>>,
        knots: Arc<Array1<f64>>,
        degree: usize,
        periodic: Option<(f64, f64, usize)>,
        transform: Option<Arc<Array2<f64>>>,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        let raw_cols = bspline_raw_column_count(knots.as_ref(), degree, periodic)?;
        if let Some(z) = transform.as_ref()
            && z.nrows() != raw_cols
        {
            return Err(format!(
                "StreamingBSplineEvaluator: transform rows {} != raw basis columns {}",
                z.nrows(),
                raw_cols
            ));
        }
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            knots: Arc::new(knots.as_standard_layout().to_owned()),
            degree,
            periodic,
            total_cols: transform.as_ref().map_or(raw_cols, |z| z.ncols()),
            transform,
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
        })
    }

    fn raw_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        bspline_raw_row_chunk(
            self.data.view(),
            self.knots.view(),
            self.degree,
            self.periodic,
            start,
            end,
        )
        .expect("StreamingBSplineEvaluator validated inputs should build row chunks")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_chunk(start, end);
        match self.transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        }
    }
}


impl ChunkedDesign for StreamingBSplineEvaluator {
    const NAME: &'static str = "StreamingBSplineEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingBSplineEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingBSplineEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.len()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.len().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.len(),
            "StreamingBSplineEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingBSplineEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingBSplineEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


/// Data stored for streaming (on-the-fly) recomputation of radial jet scalars.
/// Instead of persisting O(n*k*(d+2)) arrays, the operator stores the original
/// data/centers/eta and recomputes q/t/s per chunk during matvec operations.
#[derive(Debug, Clone)]
enum StreamingAxisMode {
    /// Per-axis anisotropic ψ_a derivatives: expose one `s_a` component per axis.
    PerAxis { metric_weights: Arc<[f64]> },
    /// Scalar ψ derivative: expose a single component equal to the total
    /// scaled squared radius r² = Σ_a exp(2η_a) h_a².
    ScalarTotal { metric_weights: Arc<[f64]> },
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
    /// Lazily materialized radial-scalar cache. (phi, q, t) per (i, j) pair
    /// — independent of axis, identical across every per-axis chunk loop —
    /// so collapses (axes × calls × chunks × n × n_knots) streaming radial
    /// evaluations into a single O(n × n_knots) sweep per operator. The
    /// inner `Option` is `None` when the parallel fill encountered a radial
    /// evaluation error (e.g. a non-finite r); callers fall back to the
    /// streaming path which propagates the error through `compute_pair`.
    triplet_cache: Arc<std::sync::OnceLock<Option<StreamingTripletCache>>>,
}


#[derive(Debug)]
struct StreamingTripletCache {
    phi: Vec<f64>,
    q: Vec<f64>,
    t: Vec<f64>,
}


/// Memory cap (bytes) above which we keep streaming the radial scalars
/// instead of materializing the (phi, q, t) triplet cache. Three `Vec<f64>`
/// arrays of length `n × n_knots` consume `24 × n × n_knots` bytes; the cap
/// keeps the resident footprint bounded for designs that would blow past a
/// few hundred MiB.
const STREAMING_TRIPLET_CACHE_BYTE_BUDGET: usize = 1 << 30;


impl StreamingRadialState {
    fn cache_fits_budget(&self) -> bool {
        let total = self
            .data
            .nrows()
            .saturating_mul(self.centers.nrows())
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_mul(3);
        total <= STREAMING_TRIPLET_CACHE_BYTE_BUDGET
    }

    fn ensure_triplet_cache(&self) -> Option<&StreamingTripletCache> {
        if !self.cache_fits_budget() {
            return None;
        }
        let n = self.data.nrows();
        let n_knots = self.centers.nrows();
        if n == 0 || n_knots == 0 {
            return None;
        }
        // The OnceLock holds `Option<StreamingTripletCache>` so a fill that
        // hits an invalid `eval_design_triplet` (e.g. a non-finite r) does
        // not poison the cache silently — consumers see `None` and fall back
        // to the streaming `compute_pair` path that propagates the error
        // through `Result<…, BasisError>`.
        self.triplet_cache
            .get_or_init(|| self.materialize_triplet_cache())
            .as_ref()
    }

    fn materialize_triplet_cache(&self) -> Option<StreamingTripletCache> {
        let n = self.data.nrows();
        let n_knots = self.centers.nrows();
        let total = n * n_knots;
        let mut phi = vec![0.0_f64; total];
        let mut q = vec![0.0_f64; total];
        let mut t = vec![0.0_f64; total];

        let metric_weights: &[f64] = match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights }
            | StreamingAxisMode::ScalarTotal { metric_weights } => metric_weights,
        };
        let dim = metric_weights.len();
        assert_eq!(dim, self.data.ncols());
        assert_eq!(dim, self.centers.ncols());

        // SERIAL fill: `ensure_triplet_cache` is called from inside outer
        // `into_par_iter` workers (e.g. the per-axis cross-trace sweep at
        // `projected_operator_terms_batched`). A nested `par_chunks_mut`
        // inside this `OnceLock::get_or_init` closure would deadlock the
        // global rayon pool — every outer worker blocks on the OnceLock
        // while the one that won the race tries to schedule child tasks no
        // worker is free to pick up (see `feedback_oncelock_rayon_deadlock`).
        //
        // The serial sweep is only affordable when the per-pair radial
        // evaluation is cheap. For the 16-D power-9 hybrid Duchon kernel a
        // single exact `eval_design_triplet` costs tens of microseconds
        // across its partial-fraction blocks, and at the large-scale
        // conditional-PGS shape (n·k ≈ 480k pairs) this loop was ~15–20 s
        // of single-threaded work per κ-trial — the dominant cost of the
        // whole CTN stage-1 fit (#979; the cost model in the previous
        // version of this comment assumed a cheap kernel). For large sweeps
        // we therefore build a certified 1-D Chebyshev radial profile once
        // (a few hundred exact evaluations, see `radial_profile`) from a
        // distance-only pre-pass over the radius range, and answer per-pair
        // queries with a Clenshaw contraction; out-of-range or uncertified
        // cases fall back to the exact evaluator per pair.
        let pair_radius = |i: usize, j: usize| -> f64 {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                // Streaming constructors set n=data.nrows(), n_knots=centers.nrows(),
                // and require dim=data.ncols()=centers.ncols(); the loop ranges
                // therefore keep both uget reads in-bounds.
                let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) }; // SAFETY: bounds per the comment immediately above
                r2 += metric_weights[a] * h * h;
            }
            r2.sqrt()
        };
        let profile = if total >= RADIAL_PROFILE_MIN_PAIRS {
            let mut r_lo = f64::INFINITY;
            let mut r_hi = 0.0_f64;
            for i in 0..n {
                for j in 0..n_knots {
                    let r = pair_radius(i, j);
                    if r > 0.0 {
                        r_lo = r_lo.min(r);
                        r_hi = r_hi.max(r);
                    }
                }
            }
            if r_lo.is_finite() && r_hi > r_lo {
                radial_profile::RadialProfile::build(&self.radial_kind, r_lo, r_hi)
            } else {
                None
            }
        } else {
            None
        };
        for i in 0..n {
            let row_off = i * n_knots;
            for j in 0..n_knots {
                let r = pair_radius(i, j);
                let triplet = match profile.as_ref() {
                    Some(profile) => profile.eval_or_exact(&self.radial_kind, r),
                    None => self.radial_kind.eval_design_triplet(r),
                };
                match triplet {
                    Ok((pv, qv, tv)) => {
                        phi[row_off + j] = pv;
                        q[row_off + j] = qv;
                        t[row_off + j] = tv;
                    }
                    Err(_) => return None,
                }
            }
        }
        Some(StreamingTripletCache { phi, q, t })
    }

    #[inline]
    fn fill_s_buf(&self, i: usize, j: usize, s_buf: &mut [f64]) {
        match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights } => {
                let dim = metric_weights.len();
                assert_eq!(s_buf.len(), dim);
                for a in 0..dim {
                    // SAFETY: compute_pair/ensure_triplet_cache callers pass i <
                    // data.nrows() and j < centers.nrows(); streaming constructors
                    // require dim=data.ncols()=centers.ncols(), and this loop has a < dim.
                    let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) };
                    s_buf[a] = metric_weights[a] * h * h;
                }
            }
            StreamingAxisMode::ScalarTotal { metric_weights } => {
                assert_eq!(s_buf.len(), 1);
                let dim = metric_weights.len();
                let mut r2 = 0.0;
                for a in 0..dim {
                    // SAFETY: compute_pair/ensure_triplet_cache callers pass i <
                    // data.nrows() and j < centers.nrows(); streaming constructors
                    // require dim=data.ncols()=centers.ncols(), and this loop has a < dim.
                    let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) };
                    r2 += metric_weights[a] * h * h;
                }
                s_buf[0] = r2;
            }
        }
    }

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
        assert!(i < self.data.nrows() && j < self.centers.nrows());
        self.fill_s_buf(i, j, s_buf);
        match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights } => {
                let r2: f64 = (0..metric_weights.len()).map(|a| s_buf[a]).sum();
                self.radial_kind.eval_design_triplet(r2.sqrt())
            }
            StreamingAxisMode::ScalarTotal { .. } => {
                let r2 = s_buf[0];
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
/// **Streaming** (large scale): stores only data/centers/eta/kernel params
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


/// Streaming design derivative for one per-row latent coordinate `t[n, a]`.
///
/// The operator stores the shared latent matrix plus either radial-kernel
/// ingredients or a precomputed non-radial derivative jet. Individual REML
/// hyper-directions carry only a flat coordinate index and call
/// `forward_mul_axis` / `transpose_mul_axis` to expose the corresponding
/// one-row design derivative on demand.
pub struct LatentCoordDesignDerivative {
    provider: Arc<dyn LocalDesignJacobianProvider>,
}


#[derive(Debug, Clone)]
struct RadialLatentCoordLocalDesignJacobian {
    latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
    centers: Arc<Array2<f64>>,
    radial_kind: RadialScalarKind,
    ident_transform: Option<Array2<f64>>,
    full_ident_transform: Option<Array2<f64>>,
    n_poly: usize,
    polynomial_order: Option<DuchonNullspaceOrder>,
}


#[derive(Debug, Clone)]
struct JetLatentCoordLocalDesignJacobian {
    latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
    jet: Arc<Array3<f64>>,
    ident_transform: Option<Array2<f64>>,
}


impl std::fmt::Debug for LatentCoordDesignDerivative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatentCoordDesignDerivative")
            .field("n_data", &self.n_data())
            .field("latent_dim", &self.latent_dim())
            .field("n_axes", &self.n_axes())
            .field("p_out", &self.p_out())
            .field("provider", &self.provider)
            .finish()
    }
}


impl Clone for LatentCoordDesignDerivative {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
        }
    }
}


impl RadialLatentCoordLocalDesignJacobian {
    fn p_constrained(&self) -> usize {
        self.ident_transform
            .as_ref()
            .map_or(self.centers.nrows(), Array2::ncols)
    }

    fn p_after_pad(&self) -> usize {
        self.p_constrained() + self.n_poly
    }

    fn p_out(&self) -> usize {
        self.full_ident_transform
            .as_ref()
            .map_or(self.p_after_pad(), Array2::ncols)
    }
}


impl JetLatentCoordLocalDesignJacobian {
    fn p_out(&self) -> usize {
        self.ident_transform
            .as_ref()
            .map_or(self.jet.shape()[1], Array2::ncols)
    }
}


/// The complete contract a per-row latent / novel-manifold coordinate type must
/// supply to participate in the REML design-derivative operator surface.
///
/// Onboarding a new coordinate type (the SAE / novel-manifold frontier) reduces
/// to implementing the small set of *required* methods below — the coordinate
/// geometry (`n_data`, `latent_dim`, `n_axes`) plus the single genuinely-new
/// payload `local_design_jacobian_row` (the local block ∂(design row)/∂(coord)).
/// The streaming operator surface consumed by `LatentCoordDerivativeOp` in
/// `src/solver/reml/mod.rs` — forward matvec, transpose matvec, and dense
/// materialization, together with the flat-axis → (row, axis) decode — is
/// inherited as *default* methods and never re-implemented per coordinate type.
///
/// This is the close condition for #767: a new coordinate type touches zero
/// operator-surface code; it provides only its local Jacobian and geometry.
pub(crate) trait LocalDesignJacobianProvider: Send + Sync + std::fmt::Debug {
    /// Number of data rows `n` the operator spans.
    fn n_data(&self) -> usize;

    /// Latent coordinate dimension `d` (perturbation axes per row).
    fn latent_dim(&self) -> usize;

    /// Number of flat hyper-axes `n · d` (one per (row, coordinate-axis) pair).
    fn n_axes(&self) -> usize;

    /// Number of output-basis columns in each local design-Jacobian row.
    fn p_out(&self) -> usize;

    /// The only per-coordinate payload: the projected local design-Jacobian row
    /// ∂(design row `row`)/∂(coordinate axis `axis`) in output-basis columns.
    fn local_design_jacobian_row(&self, row: usize, axis: usize)
    -> Result<Array1<f64>, BasisError>;

    /// Decode a flat hyper-axis into its `(row, coordinate axis)`. Row-major over
    /// `(row, axis)` with stride `latent_dim`; uniform across coordinate types.
    fn row_axis(&self, flat_axis: usize) -> (usize, usize) {
        let d = self.latent_dim();
        (flat_axis / d, flat_axis % d)
    }

    /// Forward matvec for one flat hyper-axis: place `J_row · u` at `row`.
    fn forward_mul_axis(
        &self,
        flat_axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in forward_mul_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        let (row, axis) = self.row_axis(flat_axis);
        let local_jacobian = self.local_design_jacobian_row(row, axis)?;
        assert_eq!(
            u.len(),
            local_jacobian.len(),
            "latent-coordinate derivative coefficient length mismatch in forward_mul_axis"
        );
        let value = local_jacobian.dot(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        out[row] = value;
        Ok(out)
    }

    /// Transpose matvec for one flat hyper-axis: scatter `v[row] · J_rowᵀ`.
    fn transpose_mul_axis(
        &self,
        flat_axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in transpose_mul_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n_data(),
            "latent-coordinate derivative row-adjoint length mismatch in transpose_mul_axis"
        );
        let (row, axis) = self.row_axis(flat_axis);
        let scale = v[row];
        Ok(self
            .local_design_jacobian_row(row, axis)?
            .mapv(|value| scale * value))
    }

    /// Dense `(n_data × p_out)` materialization of one flat hyper-axis: the local
    /// Jacobian row placed at `row`, all other rows zero.
    fn materialize_axis(&self, flat_axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in materialize_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        let (row, axis) = self.row_axis(flat_axis);
        let projected = self.local_design_jacobian_row(row, axis)?;
        let mut out = Array2::<f64>::zeros((self.n_data(), projected.len()));
        out.row_mut(row).assign(&projected);
        Ok(out)
    }
}


/// The rayon chunk size for parallel implicit matvec operations.
/// Each chunk processes this many data points before reducing.
const IMPLICIT_MATVEC_CHUNK_SIZE: usize = 1000;


/// Minimum data size to activate parallel iteration for implicit matvecs.
const IMPLICIT_MATVEC_PAR_THRESHOLD: usize = 10_000;


/// Number of lower-triangular center rows per tile when assembling dense
/// ThinPlate penalty ψ-derivative kernel blocks.
const THIN_PLATE_PENALTY_PSI_TILE_ROWS: usize = 32;


impl LatentCoordDesignDerivative {
    pub(crate) fn from_local_design_jacobian_provider(
        provider: Arc<dyn LocalDesignJacobianProvider>,
    ) -> Self {
        Self { provider }
    }

    pub(crate) fn new_matern(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        length_scale: f64,
        nu: MaternNu,
        include_intercept: bool,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Matérn dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            RadialLatentCoordLocalDesignJacobian {
                latent,
                centers,
                radial_kind: RadialScalarKind::Matern { length_scale, nu },
                ident_transform,
                full_ident_transform: None,
                n_poly: usize::from(include_intercept),
                polynomial_order: None,
            },
        )))
    }

    pub(crate) fn new_duchon(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        length_scale: Option<f64>,
        power: f64,
        nullspace_order: DuchonNullspaceOrder,
        full_ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Duchon dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        let effective_order = duchon_effective_nullspace_order(centers.view(), nullspace_order);
        let p_order = duchon_p_from_nullspace_order(effective_order);
        let s_order = power.max(0.0).round() as usize;
        let radial_kind = if let Some(length_scale) = length_scale {
            RadialScalarKind::Duchon {
                length_scale,
                p_order,
                s_order,
                dim: centers.ncols(),
                coeffs: duchon_partial_fraction_coeffs(
                    p_order,
                    s_order,
                    1.0 / length_scale.max(1e-300),
                ),
            }
        } else {
            RadialScalarKind::PureDuchon {
                block_order: pure_duchon_block_order(p_order, power).max(1.0) as usize,
                p_order,
                s_order,
                dim: centers.ncols(),
            }
        };
        let mut workspace = BasisWorkspace::default();
        let ident_transform =
            kernel_constraint_nullspace(centers.view(), effective_order, &mut workspace.cache)?;
        let n_poly = polynomial_block_from_order(centers.view(), effective_order).ncols();
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            RadialLatentCoordLocalDesignJacobian {
                latent,
                centers,
                radial_kind,
                ident_transform: Some(ident_transform),
                full_ident_transform,
                n_poly,
                polynomial_order: Some(effective_order),
            },
        )))
    }

    pub(crate) fn new_sphere(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        penalty_order: usize,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative sphere dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        let raw_jet = sphere_first_derivative_nd(
            latent.as_matrix().view(),
            centers.view(),
            penalty_order,
            true,
        )?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent_coord::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_periodic_bspline(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        data_range: (f64, f64),
        degree: usize,
        num_basis: usize,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        let raw_jet = periodic_bspline_first_derivative_nd(
            latent.as_matrix().view(),
            data_range,
            degree,
            num_basis,
        )?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent_coord::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_tensor_bspline(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        knots_per_axis: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        let knot_views = knots_per_axis
            .iter()
            .map(|knots| knots.view())
            .collect::<Vec<_>>();
        let raw_jet =
            bspline_tensor_first_derivative(latent.as_matrix().view(), &knot_views, &degrees)?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent_coord::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_pca(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        basis_matrix: Arc<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != basis_matrix.nrows() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Pca dimension mismatch: latent d={} basis rows={}",
                latent.latent_dim(),
                basis_matrix.nrows()
            );
        }
        let mut jet =
            Array3::<f64>::zeros((latent.n_obs(), basis_matrix.ncols(), basis_matrix.nrows()));
        for row in 0..latent.n_obs() {
            for axis in 0..basis_matrix.nrows() {
                for col in 0..basis_matrix.ncols() {
                    jet[[row, col, axis]] = basis_matrix[[axis, col]];
                }
            }
        }
        Self::from_jet(latent, jet, None)
    }

    fn from_jet(
        latent: Arc<crate::terms::latent_coord::LatentCoordValues>,
        jet: Array3<f64>,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if jet.shape()[0] != latent.n_obs() || jet.shape()[2] != latent.latent_dim() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative jet shape {:?} does not match latent shape ({}, {}, {})",
                jet.shape(),
                latent.n_obs(),
                jet.shape()[1],
                latent.latent_dim()
            );
        }
        if let Some(z) = ident_transform.as_ref()
            && z.nrows() != jet.shape()[1]
        {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative identifiability transform has {} rows but derivative jet has {} basis columns",
                z.nrows(),
                jet.shape()[1]
            );
        }
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            JetLatentCoordLocalDesignJacobian {
                latent,
                jet: Arc::new(jet),
                ident_transform,
            },
        )))
    }

    pub(crate) fn n_data(&self) -> usize {
        self.provider.n_data()
    }

    pub(crate) fn latent_dim(&self) -> usize {
        self.provider.latent_dim()
    }

    pub(crate) fn n_axes(&self) -> usize {
        self.provider.n_axes()
    }

    pub(crate) fn p_out(&self) -> usize {
        self.provider.p_out()
    }
}


impl RadialLatentCoordLocalDesignJacobian {
    fn project_and_pad(
        &self,
        raw_knot: &Array1<f64>,
        raw_poly: &Array1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        let constrained = match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot),
            None => raw_knot.clone(),
        };
        let mut padded = Array1::<f64>::zeros(constrained.len() + self.n_poly);
        padded
            .slice_mut(s![..constrained.len()])
            .assign(&constrained);
        if self.n_poly > 0 {
            padded.slice_mut(s![constrained.len()..]).assign(raw_poly);
        }
        Ok(match &self.full_ident_transform {
            Some(zf) => zf.t().dot(&padded),
            None => padded,
        })
    }

    fn kernel_axis_scalar(
        &self,
        row: usize,
        center: usize,
        axis: usize,
    ) -> Result<f64, BasisError> {
        let t_row = self.latent.row(row);
        let mut r2 = 0.0_f64;
        for a in 0..self.latent.latent_dim() {
            let delta = t_row[a] - self.centers[[center, a]];
            r2 += delta * delta;
        }
        let r = r2.sqrt();
        if r == 0.0 {
            // At a center collision the axis component s_axis = (t − c)_axis
            // is exactly zero. The product q · s_axis is therefore 0 for any
            // kernel whose q has a finite limit; for kernels where q diverges
            // the value is genuinely indeterminate (0 · ∞) and we must not
            // pretend it is zero. Defer to the kernel's classification.
            if self.radial_kind.is_smooth_at_collision() {
                return Ok(0.0);
            }
            return Err(BasisError::DegenerateAtCollision {
                kernel: "RadialScalarKind (design axis)",
                dim: self.latent.latent_dim(),
                m: 0.0,
                message: "radial scalar q = φ'/r has no finite limit at r = 0; \
                          the design row axis component is undefined",
            });
        }
        let (_, q, _) = self.radial_kind.eval_design_triplet(r)?;
        Ok(q * (t_row[axis] - self.centers[[center, axis]]))
    }

    fn polynomial_axis_values(&self, row: usize, axis: usize) -> Array1<f64> {
        let Some(order) = self.polynomial_order else {
            return Array1::<f64>::zeros(self.n_poly);
        };
        let max_degree = match order {
            DuchonNullspaceOrder::Zero => 0usize,
            DuchonNullspaceOrder::Linear => 1usize,
            DuchonNullspaceOrder::Degree(k) => k,
        };
        let t_row = self.latent.row(row);
        let exponents = monomial_exponents(self.latent.latent_dim(), max_degree);
        let mut out = Array1::<f64>::zeros(exponents.len());
        for (col, alpha) in exponents.iter().enumerate() {
            let a_axis = alpha[axis];
            if a_axis == 0 {
                continue;
            }
            let mut value = a_axis as f64;
            for a in 0..self.latent.latent_dim() {
                let exp_a = if a == axis { a_axis - 1 } else { alpha[a] };
                if exp_a != 0 {
                    value *= t_row[a].powi(exp_a as i32);
                }
            }
            out[col] = value;
        }
        out
    }
}


impl JetLatentCoordLocalDesignJacobian {
    fn project_jet(&self, raw_knot: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
        Ok(match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot),
            None => raw_knot.clone(),
        })
    }
}


impl LocalDesignJacobianProvider for LatentCoordDesignDerivative {
    fn n_data(&self) -> usize {
        self.provider.n_data()
    }

    fn latent_dim(&self) -> usize {
        self.provider.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.provider.n_axes()
    }

    fn p_out(&self) -> usize {
        self.provider.p_out()
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        self.provider.local_design_jacobian_row(row, axis)
    }
}


impl LocalDesignJacobianProvider for RadialLatentCoordLocalDesignJacobian {
    fn n_data(&self) -> usize {
        self.latent.n_obs()
    }

    fn latent_dim(&self) -> usize {
        self.latent.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.latent.len()
    }

    fn p_out(&self) -> usize {
        Self::p_out(self)
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        let mut raw_knot = Array1::<f64>::zeros(self.centers.nrows());
        for center in 0..self.centers.nrows() {
            raw_knot[center] = self.kernel_axis_scalar(row, center, axis)?;
        }
        let raw_poly = self.polynomial_axis_values(row, axis);
        self.project_and_pad(&raw_knot, &raw_poly)
    }
}


impl LocalDesignJacobianProvider for JetLatentCoordLocalDesignJacobian {
    fn n_data(&self) -> usize {
        self.latent.n_obs()
    }

    fn latent_dim(&self) -> usize {
        self.latent.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.latent.len()
    }

    fn p_out(&self) -> usize {
        Self::p_out(self)
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        let mut raw_knot = Array1::<f64>::zeros(self.jet.shape()[1]);
        for basis_col in 0..self.jet.shape()[1] {
            raw_knot[basis_col] = self.jet[[row, basis_col, axis]];
        }
        self.project_jet(&raw_knot)
    }
}


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
        assert_eq!(
            phi_values.len(),
            n * n_knots,
            "implicit psi derivative phi length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            phi_values.len()
        );
        assert_eq!(
            q_values.len(),
            n * n_knots,
            "implicit psi derivative q length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            q_values.len()
        );
        assert_eq!(
            t_values.len(),
            n * n_knots,
            "implicit psi derivative t length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            t_values.len()
        );
        assert_eq!(
            axis_components.nrows(),
            n * n_knots,
            "implicit psi derivative axis-component row mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            axis_components.nrows()
        );
        assert_eq!(
            axis_components.ncols(),
            n_axes,
            "implicit psi derivative axis-component column mismatch: expected n_axes={n_axes}, got {}",
            axis_components.ncols()
        );
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

    /// Construct a streaming operator that recomputes (q, t, s_a) on the fly
    /// from raw data/centers/eta during each matvec. No O(n*k) arrays are stored.
    /// This is the large-scale path.
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
        assert_eq!(
            centers.ncols(),
            n_axes,
            "streaming radial centers have {} columns but data/eta have {n_axes}",
            centers.ncols()
        );
        let metric_weights: Arc<[f64]> = Arc::from(centered_aniso_metric_weights(&eta));
        Self {
            // Empty arrays -- not used in streaming mode.
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::PerAxis { metric_weights },
                radial_kind,
                triplet_cache: Arc::new(std::sync::OnceLock::new()),
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
        assert_eq!(
            centers.ncols(),
            dim,
            "streaming scalar radial centers have {} columns but data/eta have {dim}",
            centers.ncols()
        );
        let metric_weights: Arc<[f64]> = Arc::from(centered_aniso_metric_weights(&eta));
        Self {
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::ScalarTotal { metric_weights },
                radial_kind,
                triplet_cache: Arc::new(std::sync::OnceLock::new()),
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

    /// Whether this operator is wired up by a basis whose large-scale path
    /// is supposed to stay implicit, so a dense `(n × p)` materialization
    /// here is a regression rather than a normal compute path. Duchon-family
    /// terms qualify because they are streaming-only at any scale; ThinPlate
    /// qualifies because the new scalar-streaming routing relies on the
    /// implicit operator above the policy threshold and a sneaky
    /// `materialize_dense()` would silently re-introduce the n × p
    /// allocation we just removed. The flag is consulted by the
    /// materialize_first / materialize_second_diag / materialize_second_cross
    /// guards to fire `assert_no_dense_derivative_materialization` for these
    /// kinds whenever the resource policy says the materialization would
    /// exceed budget. Small-n problems still pass the assertion and get the
    /// dense fast path.
    pub(crate) fn enforces_dense_materialization_budget(&self) -> bool {
        if self
            .streaming
            .as_ref()
            .is_some_and(|state| state.radial_kind.enforces_dense_materialization_budget())
        {
            return true;
        }
        // The materialized-mode path keeps no `radial_kind` to inspect, but
        // a non-zero psi_scale_share is the unambiguous Duchon-family
        // signature there (Matern uses 0, ThinPlate uses 0). Materialized
        // ThinPlate / Matern terms are in the dense fast path and the
        // guard does not need to fire for them.
        self.psi_scale_share != 0.0
    }

    /// Output dimension: total basis columns in the final space.
    pub fn p_out(&self) -> usize {
        if let Some(ref zf) = self.full_ident_transform {
            zf.ncols()
        } else {
            self.p_after_pad()
        }
    }

    pub(crate) fn append_full_transform(
        mut self,
        transform: &Array2<f64>,
    ) -> Result<Self, BasisError> {
        if transform.nrows() != self.p_out() {
            crate::bail_dim_basis!(
                "implicit psi derivative transform has {} rows but operator has {} output columns",
                transform.nrows(),
                self.p_out()
            );
        }
        self.full_ident_transform = Some(match self.full_ident_transform.take() {
            Some(existing) => fast_ab(&existing, transform),
            None => transform.clone(),
        });
        Ok(self)
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
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
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
            let nc = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
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
                crate::bail_invalid_basis!(
                    "radial scalar evaluation failed during streaming accumulate_knot_vector"
                        .into(),
                );
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
            let nc = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
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
                crate::bail_invalid_basis!(
                    "radial scalar evaluation failed during streaming forward_mul".into(),
                );
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
        let nc = n.div_ceil(cs);
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
                            // SAFETY: chunk ci owns rows [s..e) of the raw n×k buffer,
                            // so offsets i*k+j for i ∈ [s,e), j ∈ [0,k) are pairwise
                            // disjoint across workers and stay within n*k = raw.len().
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
            crate::bail_invalid_basis!(
                "radial scalar evaluation failed during streaming materialize".into(),
            );
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

    /// Batched `unproject` for a (p_out × rank) coefficient matrix.
    /// Returns (n_knots × rank) via two BLAS3 matmuls — the same algebra as
    /// `unproject`, but amortized across all rank columns of `u`. Used by
    /// `forward_mul_matrix` so per-axis trace evaluations can be a single
    /// chunked GEMM rather than rank-many `forward_mul` calls.
    pub fn unproject_matrix(&self, u: &ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(u.nrows(), self.p_out());
        // Step 1: undo full identifiability transform → (p_after_pad, rank).
        let after_full = match &self.full_ident_transform {
            Some(zf) => fast_ab(zf, u),
            None => u.to_owned(),
        };
        // Step 2: drop polynomial padding rows → (p_constrained, rank).
        let p_constrained = self.p_constrained();
        let smooth_part = after_full.slice(s![..p_constrained, ..]);
        // Step 3: undo kernel constraint Z → (n_knots, rank).
        match &self.ident_transform {
            Some(z) => fast_ab(z, &smooth_part),
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
        assert!(
            axis < self.n_axes(),
            "implicit psi first transpose axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi first transpose row-adjoint length mismatch"
        );
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
        assert!(
            axis < self.n_axes(),
            "implicit psi first forward axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi first forward coefficient length mismatch"
        );
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
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
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
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
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
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal transpose axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi second diagonal transpose row-adjoint length mismatch"
        );
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
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross transpose first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross transpose second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross transpose requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi second cross transpose row-adjoint length mismatch"
        );
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
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal forward axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi second diagonal forward coefficient length mismatch"
        );
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
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross forward first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross forward second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross forward requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi second cross forward coefficient length mismatch"
        );
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
        assert!(
            axis < self.n_axes(),
            "implicit psi first materialization axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        if self.enforces_dense_materialization_budget() {
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
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal materialization axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        if self.enforces_dense_materialization_budget() {
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
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross materialization first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross materialization second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross materialization requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        if self.enforces_dense_materialization_budget() {
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
        let raw = self.row_chunk_with_kernel_raw(rows, deriv_fn)?;
        Ok(self.project_matrix_rows(raw))
    }

    /// Like `row_chunk_with_kernel` but returns the raw (chunk × n_knots)
    /// kernel scalars without the identifiability/padding projection. Used
    /// by `forward_mul_matrix`, which does the projection on the rank side
    /// instead (`unproject_matrix(F)`) so the (n × p_out) projected
    /// derivative is never materialized for large-scale row counts.
    fn row_chunk_with_kernel_raw<G>(
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
            if let Some(cache) = st.ensure_triplet_cache() {
                for (local, i) in rows.enumerate() {
                    let base = i * self.n_knots;
                    for j in 0..self.n_knots {
                        let idx = base + j;
                        st.fill_s_buf(i, j, &mut sb);
                        raw[[local, j]] =
                            deriv_fn(cache.phi[idx], cache.q[idx], cache.t[idx], &sb, idx);
                    }
                }
            } else {
                for (local, i) in rows.enumerate() {
                    for j in 0..self.n_knots {
                        let (phi, q, t) = st.compute_pair(i, j, &mut sb)?;
                        raw[[local, j]] = deriv_fn(phi, q, t, &sb, i * self.n_knots + j);
                    }
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
        Ok(raw)
    }

    pub fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
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

    /// Raw (chunk × n_knots) first-order kernel scalars for axis d, without
    /// the identifiability/padding projection. Pairs with `unproject_matrix`
    /// in `forward_mul_matrix`: the kernel scalars stay in raw knot space
    /// while the rank side (F) is unprojected to knot space, so the per-chunk
    /// GEMM is (chunk × n_knots) · (n_knots × rank) rather than (chunk × p_out)
    /// · (p_out × rank). Saves both flops and a (chunk × p_out) intermediate.
    pub fn row_chunk_first_raw(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first raw row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel_raw(rows, |phi, q, _, sb, idx| {
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
        self.row_chunk_with_kernel_raw(rows, |phi, q, _, sb, idx| {
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
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
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
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross row chunk first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross row chunk second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross row chunk requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
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
        assert!(
            row < self.n,
            "implicit psi row-vector request out of bounds: row={row}, n={}",
            self.n
        );
        assert_eq!(
            out.len(),
            self.p_out(),
            "implicit psi row-vector output length mismatch"
        );
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
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
        crate::bail_dim_basis!(
            "aniso design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        );
    }

    let policy = crate::resource::ResourcePolicy::default_library();
    let force_operator = radial_kind.is_duchon_family();
    let dense_derivatives_exceed_budget =
        should_use_implicit_operators_with_policy(n, p_final, dim, &policy);
    let operator_only = force_operator || dense_derivatives_exceed_budget;
    let cache_radial_components = should_cache_implicit_radial_components(n, k, dim, &policy);

    // ── Streaming path: large scale ─────────────────────────────────────
    // When even the compact radial cache would exceed the operator-cache
    // budget, store only data/centers/eta/radial_kind and recompute
    // (q, t, s_a) chunkwise during each matvec. Otherwise the operator-only
    // path below caches phi/q/t/s_a without materializing dense derivative
    // matrices.
    if operator_only && !cache_radial_components {
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

    // ── Materialized radial-cache path ────────────────────────────────────
    // Allocate O(n*k) arrays up front and fill with parallel chunks that
    // write directly into preallocated storage via raw pointers. No
    // intermediate Vec<(i, q_row, t_row, s_row)> collection.
    let nk = n.checked_mul(k).ok_or_else(|| {
        BasisError::InvalidInput("aniso radial cache has too many data-center pairs".to_string())
    })?;
    if nk.checked_mul(dim).is_none() {
        crate::bail_invalid_basis!("aniso radial cache axis component storage is too large");
    }
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, dim));

    let psi_scale_share = radial_kind.raw_psi_isotropic_share();

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = n.div_ceil(cs);
    // Capture the *first* underlying radial-evaluation error rather than a
    // bare boolean: at an extreme trial hyperparameter the anisotropic
    // distance `r` can push the Duchon/Matérn radial kernel out of its
    // evaluable range, and the caller (the spatial-κ optimizer) needs the
    // real cause to decide whether the trial point is merely infeasible
    // (retreat) versus a genuine invariant violation (abort). Swallowing it
    // as "radial scalar evaluation failed" hid both the cause and the
    // recoverability.
    let first_err: std::sync::Mutex<Option<BasisError>> = std::sync::Mutex::new(None);
    // For large sweeps, replace per-pair exact radial evaluation with a
    // certified 1-D Chebyshev profile built once from a distance-only
    // pre-pass over the radius range (see `radial_profile`): at the 16-D
    // power-9 hybrid Duchon configuration a single exact triplet costs tens
    // of microseconds across its partial-fraction blocks, and this n·k
    // sweep was the dominant per-κ-trial cost of large-scale fits (#979).
    // Out-of-range radii and uncertified builds fall back to the exact
    // evaluator per pair.
    let profile = if nk >= RADIAL_PROFILE_MIN_PAIRS {
        let mut r_lo = f64::INFINITY;
        let mut r_hi = 0.0_f64;
        let mut drb = vec![0.0; dim];
        let mut cb = vec![0.0; dim];
        for i in 0..n {
            for a in 0..dim {
                drb[a] = data[[i, a]];
            }
            for j in 0..k {
                for a in 0..dim {
                    cb[a] = centers[[j, a]];
                }
                let (r, _) = aniso_distance_and_components(&drb, &cb, eta);
                if r > 0.0 {
                    r_lo = r_lo.min(r);
                    r_hi = r_hi.max(r);
                }
            }
        }
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(&radial_kind, r_lo, r_hi)
        } else {
            None
        }
    } else {
        None
    };
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ferr = &first_err;
        let profile_ref = profile.as_ref();
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = start.saturating_add(cs).min(n);
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
                    let triplet = match profile_ref {
                        Some(profile) => profile.eval_or_exact(&radial_kind, r),
                        None => radial_kind.eval_design_triplet(r),
                    };
                    let (phi, q, t) = match triplet {
                        Ok(p) => p,
                        Err(e) => {
                            let mut slot = ferr.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    };
                    let flat = i * k + j;
                    // SAFETY: each Rayon chunk owns a disjoint i-row range,
                    // so flat=i*k+j stays in 0..nk for phi/q/t and
                    // flat*dim+a stays in 0..nk*dim for axis_components.
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
    if let Some(cause) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(BasisError::InvalidInput(format!(
            "radial scalar evaluation failed during aniso derivative construction \
             (eta={eta:?}): {cause}"
        )));
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

    if operator_only {
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
        crate::bail_dim_basis!(
            "scalar design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        );
    }

    let policy = crate::resource::ResourcePolicy::default_library();
    let force_operator = radial_kind.is_duchon_family();
    let dense_derivatives_exceed_budget =
        should_use_implicit_operators_with_policy(n, p_final, 1, &policy);
    let operator_only = force_operator || dense_derivatives_exceed_budget;
    let cache_radial_components = should_cache_implicit_radial_components(n, k, 1, &policy);
    if operator_only && !cache_radial_components {
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

    let nk = n.checked_mul(k).ok_or_else(|| {
        BasisError::InvalidInput("scalar radial cache has too many data-center pairs".to_string())
    })?;
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, 1));

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = n.div_ceil(cs);
    let first_err: std::sync::Mutex<Option<BasisError>> = std::sync::Mutex::new(None);
    // Same certified radial-profile amortization as the per-axis sweep
    // above: one distance-only pre-pass for the radius range, one profile
    // build, Clenshaw per pair, exact fallback out of range (#979).
    let pair_r = |i: usize, j: usize, drb: &mut [f64], cb: &mut [f64]| -> f64 {
        if let Some(eta) = fixed_eta {
            for a in 0..dim {
                drb[a] = data[[i, a]];
                cb[a] = centers[[j, a]];
            }
            aniso_distance_and_components(drb, cb, eta).0
        } else {
            stable_euclidean_norm((0..dim).map(|a| data[[i, a]] - centers[[j, a]]))
        }
    };
    let profile = if nk >= RADIAL_PROFILE_MIN_PAIRS {
        let mut r_lo = f64::INFINITY;
        let mut r_hi = 0.0_f64;
        let mut drb = vec![0.0; dim];
        let mut cb = vec![0.0; dim];
        for i in 0..n {
            for j in 0..k {
                let r = pair_r(i, j, &mut drb, &mut cb);
                if r > 0.0 {
                    r_lo = r_lo.min(r);
                    r_hi = r_hi.max(r);
                }
            }
        }
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(&radial_kind, r_lo, r_hi)
        } else {
            None
        }
    } else {
        None
    };
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ferr = &first_err;
        let profile_ref = profile.as_ref();
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = start.saturating_add(cs).min(n);
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
                    let triplet = match profile_ref {
                        Some(profile) => profile.eval_or_exact(&radial_kind, r),
                        None => radial_kind.eval_design_triplet(r),
                    };
                    let (phi, q, t) = match triplet {
                        Ok(p) => p,
                        Err(e) => {
                            let mut slot = ferr.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    };
                    let flat = i * k + j;
                    // SAFETY: each Rayon chunk owns a disjoint i-row range
                    // of the nk-long phi/q/t/axis buffers, so flat=i*k+j is
                    // in-bounds for every write and never aliases another worker.
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
    if let Some(cause) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(BasisError::InvalidInput(format!(
            "radial scalar evaluation failed during scalar derivative construction: {cause}"
        )));
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

    if operator_only {
        return Ok(ScalarDesignPsiDerivatives {
            design_first: Array2::<f64>::zeros((0, 0)),
            design_second_diag: Array2::<f64>::zeros((0, 0)),
            implicit_operator: Some(op),
        });
    }

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
    /// Kernel-constraint nullspace transform `Z` applied internally to the
    /// raw kernel-basis K×K operator matrices (Some for Duchon, None for
    /// Matérn which uses a different basis).
    pub kernel_nullspace_transform: Option<Array2<f64>>,
    /// Polynomial block columns appended after the kernel block (Duchon
    /// polynomial null space). Zero for Matérn.
    pub polynomial_block_cols: usize,
}


#[derive(Debug, Clone)]
pub struct DuchonOperatorPenaltyMatrices {
    pub mass: Array2<f64>,
    pub tension: Array2<f64>,
    pub stiffness: Array2<f64>,
}


#[derive(Debug, Clone)]
pub struct ThinPlatePenaltyMatrix {
    pub penalty: Array2<f64>,
}


fn default_normalization_scale() -> f64 {
    1.0
}


fn validate_center_count(num_centers: usize) -> Result<(), BasisError> {
    if num_centers == 0 {
        crate::bail_invalid_basis!("center count must be positive");
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
        crate::bail_invalid_basis!(
            "equal-mass center selection requested {num_centers} centers but data has {n} rows"
        );
    }
    if d == 0 {
        crate::bail_invalid_basis!("equal-mass center selection requires at least one column");
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
        // Score candidate split dimensions in parallel, but keep each dimension's
        // row scan in serial row order and use the same strict-`>` update rule
        // (with lowest-dimension tie breaking) as the original greedy splitter.
        (0..d)
            .into_par_iter()
            .map(|j| {
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
                let span = if span.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    span
                };
                (j, span)
            })
            .reduce_with(|a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(j, _)| j)
            .unwrap_or(0)
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
        crate::bail_invalid_basis!(
            "equal-mass partition produced {} leaves, expected {num_centers}",
            leaves.len()
        );
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

        let best_idx = slice
            .par_iter()
            .filter_map(|&idx| {
                let mut d2 = 0.0;
                for j in 0..d {
                    let delta = data[[idx, j]] - centroid[j];
                    d2 += delta * delta;
                }
                if d2.is_finite() {
                    Some((idx, d2))
                } else {
                    None
                }
            })
            .reduce_with(|a, b| {
                if b.1 < a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(idx, _)| idx)
            .unwrap_or(slice[0]);
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
        crate::bail_invalid_basis!(
            "equal-mass covariate-representative center selection requested {num_centers} centers but data has {n} rows"
        );
    }
    if d == 0 {
        crate::bail_invalid_basis!(
            "equal-mass covariate-representative center selection requires at least one column"
                .to_string(),
        );
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
        crate::bail_invalid_basis!("kmeans requested {num_centers} centers but data has {n} rows");
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

    // For large n (large-scale), parallelize the assignment step.
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
        crate::bail_invalid_basis!("uniform grid requires at least one axis");
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
        crate::bail_invalid_basis!("uniform-grid points_per_dim must be positive");
    }
    let d = data.ncols();
    if d == 0 {
        crate::bail_invalid_basis!("uniform-grid center selection requires at least one column");
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
