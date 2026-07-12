use super::*;

/// Whether a knot difference is structurally degenerate.
///
/// Cox–de Boor terms are scale-free ratios. The only undefined denominator in
/// a validated nondecreasing knot vector is therefore an exactly repeated knot,
/// whose conventional contribution is zero. A magnitude tolerance—absolute or
/// relative—changes the represented spline when physical units change or when
/// a valid nonuniform knot pattern contains a narrow span.
#[inline]
pub(crate) fn knot_span_is_degenerate(span: f64) -> bool {
    span == 0.0
}

/// Default number of rows in each block the streaming design evaluators
/// materialize at a time when the caller does not supply an explicit chunk
/// size. Bounds the transient working set (one `chunk_rows × p` dense block)
/// while staying large enough to amortize per-chunk kernel-column setup.
pub(crate) const DEFAULT_STREAMING_CHUNK_ROWS: usize = 2048;
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
        if z_raw[i] != z_clamped[i] {
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
        if dz == 0.0 {
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

pub(crate) fn apply_dense_bspline_extrapolation(
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
    if !(left.is_finite() && right.is_finite() && left < right) {
        return Ok(());
    }

    // Open (unclamped) knots: the value evaluator clamps the eval point to the
    // modeling interval `[knots[degree], knots[num_basis]]` (constant extension —
    // there is no linear extension because `has_clamped_bspline_boundaries` is
    // false). A constant function has zero derivative, so BOTH the first and
    // second derivative must be zero in the exterior spans. Without this, the
    // dense derivative path leaves the raw mathematical B-spline derivative in the
    // boundary spans (nonzero), which no longer matches a finite difference of the
    // constant-extended value basis (gam#1348). The genuine cyclic basis never
    // reaches here (it pre-wraps its input into the base period).
    if !has_clamped_bspline_boundaries(knotview, degree) {
        if matches!(
            eval_kind,
            BasisEvalKind::FirstDerivative | BasisEvalKind::SecondDerivative
        ) {
            for (i, &x) in data.iter().enumerate() {
                if x < left || x > right {
                    basis_matrix.row_mut(i).fill(0.0);
                }
            }
        }
        return Ok(());
    }

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
pub(crate) fn has_clamped_bspline_boundaries(knotview: ArrayView1<f64>, degree: usize) -> bool {
    let clamp_count = degree + 1;
    if knotview.len() < 2 * clamp_count {
        return false;
    }
    let left = knotview[0];
    let right = knotview[knotview.len() - 1];
    let left_clamped = knotview
        .iter()
        .take(clamp_count)
        .all(|&k| k == left);
    let right_clamped = knotview
        .iter()
        .rev()
        .take(clamp_count)
        .all(|&k| k == right);
    left_clamped && right_clamped
}

/// Clamp a B-spline derivative evaluation point to the modeling interval
/// `[knots[degree], knots[num_basis]]`, mirroring the value evaluator's clamp
/// (`evaluate_splines_at_point_into`). Outside that interval the non-periodic
/// value basis is a linear extension, so its derivative is the constant boundary
/// derivative — which is exactly what evaluating at the clamped endpoint yields.
/// Keeping the derivative's boundary semantics identical to the value's is what
/// makes the analytic derivative equal a finite difference of the value (gam#1348).
#[inline]
pub(crate) fn clamp_eval_point_to_modeling_interval(
    x: f64,
    knotview: ArrayView1<f64>,
    degree: usize,
) -> f64 {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis == 0 {
        return x;
    }
    let left = knotview[degree];
    let right = knotview[num_basis];
    if !left.is_finite() || !right.is_finite() || left >= right {
        return x;
    }
    x.clamp(left, right)
}

/// True when `x` lies strictly outside the modeling interval of an *open*
/// (non-clamped) knot vector, where the analytic B-spline derivative of every
/// order must be zero.
///
/// The boundary extension differs by knot geometry, and the derivative has to
/// follow whatever the value basis does so that it equals a finite difference of
/// the value (gam#1348):
///
/// * **Open / unclamped** knots — the value evaluator clamps its argument to
///   `[t[degree], t[num_basis]]` and holds the value *constant* outside it
///   (`has_clamped_bspline_boundaries` is false, so the dense builder applies no
///   linear extension). A constant has zero derivative, so the exterior
///   derivative is zero — this returns `true`.
/// * **Clamped** knots — the value is extended *linearly* past the boundary, so
///   the exterior derivative is the nonzero boundary slope obtained by evaluating
///   at the clamped endpoint. This returns `false`, leaving the existing
///   clamp-and-evaluate path in charge.
///
/// The dense builder already zeroes the open-knot exterior in
/// [`apply_dense_bspline_extrapolation`]; this predicate lets the *per-point*
/// sparse, scalar, and recurrence evaluators do the same, so every derivative
/// path agrees with the value basis — not just the dense one the public
/// `bspline_basis_derivative` happens to use. (Genuinely cyclic bases pre-wrap
/// their input into the base period and never reach here.)
#[inline]
pub(crate) fn open_knot_derivative_exterior_is_zero(
    x: f64,
    knotview: ArrayView1<f64>,
    degree: usize,
) -> bool {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis == 0 {
        return false;
    }
    let left = knotview[degree];
    let right = knotview[num_basis];
    if !(left.is_finite() && right.is_finite() && left < right) {
        return false;
    }
    (x < left || x > right) && !has_clamped_bspline_boundaries(knotview, degree)
}

/// True when the *linear* (clamped-knot) exterior extension forces the order-`k`
/// derivative to vanish outside the modeling interval.
///
/// On a clamped knot vector the value basis is extended past the boundary as the
/// affine function `B(x_b) + (x − x_b)·B'(x_b)` (see
/// [`apply_dense_bspline_extrapolation`] and the value clamp in
/// `clamp_eval_point_to_modeling_interval`). An affine function has a constant
/// first derivative (the boundary slope) and **identically zero** second and
/// higher derivatives, so for `derivative_order ≥ 2` the exterior derivative is
/// zero — *not* the boundary's own `B^{(k)}(x_b)`, which is what naively
/// clamping the eval point and evaluating the order-`k` recurrence returns.
///
/// This is the clamped-knot counterpart of
/// [`open_knot_derivative_exterior_is_zero`] (which zeroes *every* order for the
/// *constant* open-knot extension). The dense builder already enforces the
/// affine-exterior contract in [`apply_dense_bspline_extrapolation`]; this
/// predicate lets the per-point scalar/recurrence evaluators agree with it so a
/// higher-derivative design equals a finite difference of the value in the
/// boundary spans.
#[inline]
pub(crate) fn linear_extension_higher_derivative_is_zero(
    x: f64,
    knotview: ArrayView1<f64>,
    degree: usize,
    derivative_order: usize,
) -> bool {
    if derivative_order < 2 {
        return false;
    }
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis == 0 {
        return false;
    }
    let left = knotview[degree];
    let right = knotview[num_basis];
    if !(left.is_finite() && right.is_finite() && left < right) {
        return false;
    }
    x < left || x > right
}

#[inline]
pub(crate) fn one_sided_derivative_eval_point(
    x: f64,
    knotview: ArrayView1<f64>,
    degree: usize,
) -> f64 {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if num_basis == 0 {
        return x;
    }
    let left = knotview[degree];
    let right = knotview[num_basis];
    if !left.is_finite() || !right.is_finite() || left >= right {
        return x;
    }
    if x == left {
        let next = left.next_up();
        if next < right {
            next
        } else {
            left + 0.5 * (right - left)
        }
    } else if x == right {
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

pub(crate) fn validate_knots_for_degree(
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
pub(crate) fn validate_knot_spans_nondegenerate(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    if knot_vector.len() <= degree + 1 {
        return Ok(());
    }
    let num_basis = knot_vector.len() - degree - 1;
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        if span <= 0.0 {
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

pub(crate) struct BasisEvalScratch {
    pub(crate) basis: internal::BsplineScratch,
    pub(crate) lower_basis: Vec<f64>,
    pub(crate) lower_scratch: internal::BsplineScratch,
    pub(crate) derivative_workspace: BsplineDerivativeWorkspace,
}

impl BasisEvalScratch {
    pub(crate) fn new(degree: usize) -> Self {
        let lower_degree = degree.saturating_sub(1);
        Self {
            basis: internal::BsplineScratch::new(degree),
            lower_basis: vec![0.0; lower_degree + 1],
            lower_scratch: internal::BsplineScratch::new(lower_degree),
            derivative_workspace: BsplineDerivativeWorkspace::new(),
        }
    }
}

#[inline]
pub(crate) fn copy_full_row_to_sparse_window(full: &[f64], values: &mut [f64]) -> usize {
    values.fill(0.0);
    let Some(start_col) = full.iter().position(|&v| v != 0.0) else {
        return 0;
    };
    for (offset, value_slot) in values.iter_mut().enumerate() {
        if let Some(&v) = full.get(start_col + offset) {
            *value_slot = v;
        }
    }
    start_col
}

pub(crate) fn evaluate_splines_derivative_sparse_intowith_lower(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    lowervalues: &mut [f64],
    lower_scratch: &mut internal::BsplineScratch,
) -> usize {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if degree == 0 {
        values.fill(0.0);
        return 0;
    }

    let num_basis_lower = knotview.len().saturating_sub(degree);
    if lowervalues.len() < num_basis_lower {
        values.fill(0.0);
        return 0;
    }
    lowervalues[..num_basis_lower].fill(0.0);

    // Non-periodic (open/clamped) B-spline derivative, kept consistent with the
    // value basis so it equals a finite difference of the value (gam#1348). On an
    // *open* knot vector the value is held constant outside the modeling interval,
    // so the exterior derivative is zero — the dense builder enforces this in
    // `apply_dense_bspline_extrapolation`, and the per-point sparse path (used
    // directly for open knots, which never take the clamped extrapolation
    // fallback in `SparseStorage::build`) must do the same or a P-spline
    // derivative design disagrees with its own value in the boundary spans.
    // Clamped knots extend linearly and keep their nonzero boundary slope, so the
    // guard intentionally fires only for the open-knot exterior. No periodic wrap:
    // wrapping moved boundary-span points onto unrelated interior columns;
    // genuinely cyclic bases pre-wrap their input into the base period upstream.
    if open_knot_derivative_exterior_is_zero(x, knotview, degree) {
        values.fill(0.0);
        return 0;
    }
    // Clamped knots extend the value linearly past the boundary, so the exterior
    // first derivative is the constant boundary slope obtained by evaluating at
    // the clamped endpoint — mirror the value clamp (and the scalar derivative
    // path) here so the sparse derivative agrees with a finite difference of the
    // value in the boundary spans. Without the clamp a far-exterior point lands
    // outside the degree-(d−1) support and the recurrence reads zero, breaking
    // the boundary-slope contract.
    let x_clamped = clamp_eval_point_to_modeling_interval(x, knotview, degree);
    let x_eval = one_sided_derivative_eval_point(x_clamped, knotview, degree);
    internal::evaluate_splines_at_point_full_support_into(
        x_eval,
        degree - 1,
        knotview,
        &mut lowervalues[..num_basis_lower],
        lower_scratch,
    );

    let mut full_derivative = vec![0.0; num_basis];
    for i in 0..num_basis {
        let denom_left = knotview[i + degree] - knotview[i];
        let denom_right = knotview[i + degree + 1] - knotview[i + 1];
        let left_term = if !knot_span_is_degenerate(denom_left) {
            lowervalues[i] / denom_left
        } else {
            0.0
        };
        let right_term = if !knot_span_is_degenerate(denom_right) {
            lowervalues[i + 1] / denom_right
        } else {
            0.0
        };
        let value = (degree as f64) * (left_term - right_term);
        full_derivative[i] = value;
    }

    copy_full_row_to_sparse_window(&full_derivative, values)
}

#[inline]
pub(crate) fn evaluate_splines_derivative_sparse_into(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let num_basis_lower = knotview.len().saturating_sub(degree);
    if scratch.lower_basis.len() != num_basis_lower {
        scratch.lower_basis.resize(num_basis_lower, 0.0);
        scratch
            .lower_scratch
            .ensure_degree(degree.saturating_sub(1));
    }
    evaluate_splines_derivative_sparse_intowith_lower(
        x,
        degree,
        knotview,
        values,
        &mut scratch.lower_basis,
        &mut scratch.lower_scratch,
    )
}

pub(crate) fn evaluate_splinessecond_derivative_sparse_into(
    x: f64,
    degree: usize,
    knotview: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BasisEvalScratch,
) -> usize {
    let num_basis = knotview.len().saturating_sub(degree + 1);
    if degree < 2 {
        values.fill(0.0);
        return 0;
    }

    if scratch.lower_basis.len() != num_basis {
        scratch.lower_basis.resize(num_basis, 0.0);
    }
    evaluate_bspline_derivative_recurrence_into(
        2,
        x,
        knotview,
        degree,
        &mut scratch.lower_basis,
        &mut scratch.derivative_workspace,
        0,
    )
    .expect("validated B-spline second-derivative inputs");

    copy_full_row_to_sparse_window(&scratch.lower_basis, values)
}

#[inline]
pub(crate) fn evaluate_splines_sparsewith_kind(
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
pub(crate) fn evaluate_bsplinerow_entries<F>(
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

pub(crate) trait BasisStorage {
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

pub(crate) struct DenseStorage;

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

pub(crate) struct SparseStorage;

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
        let needs_extrapolation = has_clamped_bspline_boundaries(knotview, degree)
            && data.iter().any(|&x| x < left || x > right);
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

pub(crate) fn generate_basis_internal<S: BasisStorage>(
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

/// Creates the discrete coefficient-sequence operator `S = D' * D`, penalizing
/// squared `order`-th differences of the supplied coordinates.
///
/// This is **not** a B-spline roughness functional: for a spline use
/// [`bspline_derivative_penalty_matrix`], which assembles
/// `S_ij = ∫ B_i^(order) B_j^(order)` from the actual knot vector. This
/// discrete operator remains for models whose coordinates themselves form the
/// object being differenced (including explicitly declared latent priors).
///
/// # Arguments
/// * `num_coefficients`: Length of the coefficient sequence.
/// * `order`: Order of the discrete difference.
/// * `coefficient_abscissae`: Optional coordinate attached to each coefficient.
///   `None` uses ordinary integer-grid differences; `Some` uses divided
///   differences scaled by relative coordinate spans.
///
/// # Returns
/// A square `Array2<f64>` of shape `[num_coefficients, num_coefficients]`.
pub fn create_difference_penalty_matrix(
    num_coefficients: usize,
    order: usize,
    coefficient_abscissae: Option<ArrayView1<f64>>,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_coefficients {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_coefficients,
        });
    }

    if let Some(g) = coefficient_abscissae
        && g.len() != num_coefficients
    {
        crate::bail_dim_basis!(
            "coefficient abscissae length {} does not match coefficient count {}",
            g.len(),
            num_coefficients
        );
    }

    // Start with the identity matrix
    let mut d = Array2::<f64>::eye(num_coefficients);

    // Apply the differencing operation `order` times.
    // Each `diff` reduces the number of rows by 1.
    for o in 1..=order {
        // Calculate the difference between adjacent rows: D^{(o)} = Delta * D^{(o-1)}
        d = &d.slice(s![1.., ..]) - &d.slice(s![..-1, ..]);

        // If using non-uniform coefficient coordinates, apply divided-difference scaling:
        // D^{(o)}_i = D^{(o)}_i / (xi_{i+o} - xi_i)
        //
        // The raw divided-difference divisor `g[i+o] - g[i]` carries the units
        // of the covariate, so a pure rescaling `g -> c*g` (a change of physical
        // units for `x`) would multiply every divisor by `c` and hence scale
        // `S = DᵀD` by `c^(-2*order)`. This discrete operator uses only relative
        // coefficient spacing, so normalize each order's spans by their
        // geometric mean. The divisor is then invariant to a global rescaling
        // and identically one on a uniform coordinate grid.
        if let Some(g) = coefficient_abscissae {
            let nrows = d.nrows();
            let mut log_span_sum = 0.0_f64;
            for i in 0..nrows {
                let span = g[i + o] - g[i];
                if !span.is_finite() || span <= 0.0 {
                    return Err(BasisError::InvalidKnotVector(format!(
                        "divided-difference coordinates must be finite and strictly increasing at order {o}, row {i}: g[{}]={:.6e}, g[{i}]={:.6e}",
                        i + o,
                        g[i + o],
                        g[i]
                    )));
                }
                log_span_sum += span.abs().ln();
            }
            // Geometric mean of the spans at this order; scales as `c` under
            // `g -> c*g`, so dividing each span by it cancels the units exactly.
            let ref_span = (log_span_sum / nrows as f64).exp();
            for i in 0..nrows {
                let span = (g[i + o] - g[i]) / ref_span;
                let mut row = d.row_mut(i);
                row /= span;
            }
        }
    }

    // The penalty matrix S = D' * D
    let s = fast_ata(&d);
    Ok(s)
}

pub(crate) fn bspline_raw_column_count(
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

pub(crate) fn bspline_raw_row_chunk(
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
        // Wrap into the ANCHORED window so the data align with the anchored
        // knot grid (`cyclic_knot_anchor`, the domain origin). The dense
        // builder resolves the anchor through the identical helper; both must
        // agree for predict-time streaming to reproduce the fit-time design.
        let (anchor, _) = crate::basis::cyclic_knot_anchor(domain_start, period, num_basis);
        let wrapped = chunk.mapv(|x| wrap_to_period(x, anchor, period));
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
