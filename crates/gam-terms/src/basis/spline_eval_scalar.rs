use super::*;

/// Scratch memory for B-spline evaluation to avoid allocations in tight loops.
pub struct SplineScratch {
    pub(crate) inner: internal::BsplineScratch,
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

/// Configuration for a dense one-dimensional periodic B-spline basis.
///
/// The basis lives on a circle parameterized by `origin + [0, period)`.  It is
/// vector-valued agnostic: the same scalar periodic design can be shared by any
/// number of ambient output coordinates, so a single fitted curve
/// `u -> R^d_ambient` can trace ellipses, ovals, and skewed/distorted closed
/// loops without assuming a unit circle embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicBSplineBasisSpec {
    /// Polynomial degree of the cardinal B-spline pieces.
    pub degree: usize,
    /// Number of periodic basis functions around the circle.
    pub num_basis: usize,
    /// Period of the parameter coordinate.
    pub period: f64,
    /// Parameter value identified with zero phase.
    pub origin: f64,
    /// Derivative order in the periodic function roughness
    /// `∮(f^(penalty_order))²` used by curve fitting.
    pub penalty_order: usize,
}

impl PeriodicBSplineBasisSpec {
    /// Construct a validated-looking spec. Full semantic validation is still
    /// performed by builders so deserialized specs receive identical checks.
    pub fn new(
        degree: usize,
        num_basis: usize,
        period: f64,
        origin: f64,
        penalty_order: usize,
    ) -> Self {
        Self {
            degree,
            num_basis,
            period,
            origin,
            penalty_order,
        }
    }
}

/// Fitted vector-valued periodic spline curve.
///
/// `coefficients` has shape `(num_basis, ambient_dim)`. Evaluation multiplies
/// the periodic scalar basis row by every output column, preserving any
/// anisotropic stretching, skew, or non-circular shape present in the training
/// coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicSplineCurve {
    pub spec: PeriodicBSplineBasisSpec,
    pub coefficients: Array2<f64>,
}

impl PeriodicSplineCurve {
    /// Number of coordinates in the ambient output space.
    pub fn ambient_dim(&self) -> usize {
        self.coefficients.ncols()
    }

    /// Evaluate the fitted curve at arbitrary parameter values. Values outside
    /// the base interval are wrapped modulo `period`.
    pub fn evaluate(&self, u: ArrayView1<'_, f64>) -> Result<Array2<f64>, BasisError> {
        if self.coefficients.nrows() != self.spec.num_basis {
            crate::bail_dim_basis!(
                "curve coefficient rows ({}) must equal periodic basis size ({})",
                self.coefficients.nrows(),
                self.spec.num_basis
            );
        }
        let basis = build_periodic_bspline_basis_1d(u, &self.spec)?;
        Ok(basis.dot(&self.coefficients))
    }

}

pub(crate) fn validate_periodic_bspline_spec(
    spec: &PeriodicBSplineBasisSpec,
) -> Result<(), BasisError> {
    if spec.degree < 1 {
        return Err(BasisError::InvalidDegree(spec.degree));
    }
    if spec.num_basis < spec.degree + 1 {
        crate::bail_invalid_basis!(
            "periodic B-spline basis requires num_basis >= degree + 1 (got num_basis={}, degree={})",
            spec.num_basis,
            spec.degree
        );
    }
    if !spec.period.is_finite() || spec.period <= 0.0 {
        crate::bail_invalid_basis!(
            "periodic B-spline period must be finite and positive, got {}",
            spec.period
        );
    }
    if !spec.origin.is_finite() {
        crate::bail_invalid_basis!(
            "periodic B-spline origin must be finite, got {}",
            spec.origin
        );
    }
    if spec.penalty_order == 0 || spec.penalty_order >= spec.num_basis {
        return Err(BasisError::InvalidPenaltyOrder {
            order: spec.penalty_order,
            num_basis: spec.num_basis,
        });
    }
    if spec.penalty_order > spec.degree {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree: spec.degree,
            derivative_order: spec.penalty_order,
            minimum_degree: spec.penalty_order,
        });
    }
    Ok(())
}

#[inline]
pub(crate) fn wrap_periodic_phase(u: f64, origin: f64, period: f64) -> f64 {
    let wrapped = (u - origin).rem_euclid(period);
    // Keep values numerically on the half-open interval even when rem_euclid
    // returns period after extreme-roundoff cancellation.
    if wrapped >= period { 0.0 } else { wrapped }
}

pub(crate) fn cardinal_bspline_value(x: f64, degree: usize) -> f64 {
    if degree == 0 {
        return if (0.0..1.0).contains(&x) { 1.0 } else { 0.0 };
    }
    if x <= 0.0 || x >= (degree + 1) as f64 {
        return 0.0;
    }
    let p = degree as f64;
    (x / p) * cardinal_bspline_value(x, degree - 1)
        + (((degree + 1) as f64 - x) / p) * cardinal_bspline_value(x - 1.0, degree - 1)
}

pub(crate) fn fill_periodic_bspline_unnormalized_value_row(
    u: f64,
    origin: f64,
    period: f64,
    degree: usize,
    row: &mut [f64],
) -> f64 {
    let m = row.len();
    let m_f = m as f64;
    let h = period / m_f;
    let t = wrap_periodic_phase(u, origin, period) / h;
    let mut rowsum = 0.0_f64;
    for (col, value_slot) in row.iter_mut().enumerate() {
        let base = t - col as f64;
        let k_min = ((-base) / m_f).floor() as isize - 1;
        let k_max = (((degree + 1) as f64 - base) / m_f).ceil() as isize + 1;
        let mut value = 0.0_f64;
        for k in k_min..=k_max {
            value += cardinal_bspline_value(base + (k as f64) * m_f, degree);
        }
        *value_slot = value;
        rowsum += value;
    }
    rowsum
}

pub(crate) fn fill_periodic_bspline_unnormalized_derivative_row(
    u: f64,
    origin: f64,
    period: f64,
    degree: usize,
    row: &mut [f64],
) -> f64 {
    let m = row.len();
    let m_f = m as f64;
    let h = period / m_f;
    let tau = wrap_periodic_phase(u, origin, period) / h;
    let mut rowsum_derivative = 0.0_f64;
    for (col, value_slot) in row.iter_mut().enumerate() {
        let base = tau - col as f64;
        let k_min = ((-base) / m_f).floor() as isize - 1;
        let k_max = (((degree + 1) as f64 - base) / m_f).ceil() as isize + 1;
        let mut value = 0.0_f64;
        for k in k_min..=k_max {
            let x_arg = base + (k as f64) * m_f;
            value += cardinal_bspline_value(x_arg, degree - 1)
                - cardinal_bspline_value(x_arg - 1.0, degree - 1);
        }
        let derivative = value / h;
        *value_slot = derivative;
        rowsum_derivative += derivative;
    }
    rowsum_derivative
}

/// Build a dense periodic cardinal B-spline design for one circular parameter.
///
/// Row `i` contains `num_basis` periodic basis functions evaluated at `u[i]`.
/// The rows form a partition of unity and are exactly periodic in `period`.
/// No output-space normalization is performed; use the same design matrix for
/// each coordinate of a vector-valued curve to preserve arbitrary anisotropic
/// stretching in ambient space.
pub fn build_periodic_bspline_basis_1d(
    u: ArrayView1<'_, f64>,
    spec: &PeriodicBSplineBasisSpec,
) -> Result<Array2<f64>, BasisError> {
    validate_periodic_bspline_spec(spec)?;
    if u.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("periodic B-spline inputs must all be finite");
    }

    let n = u.len();
    let m = spec.num_basis;
    let mut out = Array2::<f64>::zeros((n, m));
    let mut value_row = vec![0.0_f64; m];
    for (row_idx, &ui) in u.iter().enumerate() {
        let rowsum = fill_periodic_bspline_unnormalized_value_row(
            ui,
            spec.origin,
            spec.period,
            spec.degree,
            &mut value_row,
        );
        if !rowsum.is_finite() || rowsum <= 0.0 {
            crate::bail_invalid_basis!(
                "periodic B-spline row has non-positive rowsum at row {row_idx}: {rowsum}"
            );
        }
        for col in 0..m {
            out[[row_idx, col]] = value_row[col] / rowsum;
        }
    }
    Ok(out)
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
        let num_bspline_basis = knot_vector.len().saturating_sub(bs_degree + 1);
        let num_ispline_basis = num_bspline_basis.saturating_sub(1);
        return Ok(Array2::zeros((data.len(), num_ispline_basis)));
    }
    let num_bspline_cols = knot_vector.len().saturating_sub(bs_degree + 1);
    let db = match derivative_order {
        1 => {
            let (db_arc, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knot_vector.view()),
                bs_degree,
                BasisOptions::first_derivative(),
            )?;
            db_arc.as_ref().clone()
        }
        2 => {
            let (db_arc, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knot_vector.view()),
                bs_degree,
                BasisOptions::second_derivative(),
            )?;
            db_arc.as_ref().clone()
        }
        3 => {
            let mut db = Array2::<f64>::zeros((data.len(), num_bspline_cols));
            for (row_idx, &x) in data.iter().enumerate() {
                let row = db.slice_mut(s![row_idx, ..]).into_slice().ok_or_else(|| {
                    BasisError::InvalidInput(
                        "I-spline derivative row is not contiguous".to_string(),
                    )
                })?;
                evaluate_bsplinethird_derivative_scalar(x, knot_vector.view(), bs_degree, row)?;
            }
            db
        }
        4 => {
            let mut db = Array2::<f64>::zeros((data.len(), num_bspline_cols));
            for (row_idx, &x) in data.iter().enumerate() {
                let row = db.slice_mut(s![row_idx, ..]).into_slice().ok_or_else(|| {
                    BasisError::InvalidInput(
                        "I-spline derivative row is not contiguous".to_string(),
                    )
                })?;
                evaluate_bspline_fourth_derivative_scalar(x, knot_vector.view(), bs_degree, row)?;
            }
            db
        }
        other => {
            crate::bail_invalid_basis!(
                "I-spline derivative supports orders 1..=4; got order={other}"
            );
        }
    };
    let num_ispline_cols = num_bspline_cols.saturating_sub(1);
    if num_ispline_cols == 0 {
        return Ok(Array2::zeros((data.len(), 0)));
    }
    // The exterior of the modelling interval, on the I-spline's OWN convention
    // (gam#2695).
    //
    // `create_ispline_dense` saturates: `I_j(x) = 0` for `x < left` and
    // `I_j(x) = 1 − offset_j` for `x >= right`, both CONSTANT in `x`. Its
    // comment states that outright and justifies it — a linear extension would
    // make I-spline entries negative below `left` and greater than one above
    // `right`, breaking non-negativity and the [0, 1] range the basis exists to
    // guarantee. A constant function has zero derivative, so every order of the
    // exterior derivative of an I-spline is exactly zero.
    //
    // The B-spline machinery this function differentiates through obeys the
    // opposite convention. `apply_dense_bspline_extrapolation` already zeroes
    // the exterior for an OPEN knot vector on exactly this argument (gam#1348,
    // "A constant function has zero derivative, so BOTH the first and second
    // derivative must be zero in the exterior spans"), but on a CLAMPED vector
    // — which is what an I-spline knot vector always is — it evaluates the
    // derivative AT the clamped endpoint and returns the boundary slope,
    // because a clamped *B*-spline's value extends linearly. So before this,
    // `create_ispline_dense` and `create_ispline_derivative_dense` described two
    // different functions outside `[left, right]`, and only the value's
    // convention was written down.
    //
    // Measured consequence (gam#2695): the survival link warp is
    // `q = q0 + Σ_j βw_j·I_j(q0)`, so the threshold and log-sigma blocks reach
    // `q` only through `m1 = 1 + Σ_j βw_j·I'_j(q0)`. Outside the knot domain the
    // warp value is flat while `m1` picked up a slope it does not have, every
    // chain-rule channel through `q0` was scaled by it, and the joint-Newton RHS
    // asserted a first-order change the objective does not make — at any step
    // size. The wiggle block's own gradient (`∂q/∂βw_j = I_j(q0)`, the VALUE)
    // was correct throughout, which is why the disagreement looked
    // state-dependent rather than structural.
    let left = knot_vector[bs_degree];
    let right = knot_vector[num_bspline_cols];
    let interval_is_usable = left.is_finite() && right.is_finite() && left < right;

    // Right-cumulative sum: I-spline derivative column j = sum_{m=j+1..end} dB_m.
    // In our indexing: output column j (0-based) = sum of dB columns j+1..num_bspline_cols.
    let mut out = Array2::<f64>::zeros((data.len(), num_ispline_cols));
    for i in 0..data.len() {
        // Strictly outside, matching `apply_dense_bspline_extrapolation`'s own
        // open-knot branch (`x < left || x > right`). The endpoints keep the
        // interior one-sided slope on purpose: `right` is routinely the largest
        // observed value (knot vectors are built from the data range), and the
        // transformation-normal shape derivative `h'(y)` must stay positive
        // there. The written form is `!(in range)` so a NaN evaluation point
        // zeroes the row instead of propagating through the cumulative sum.
        if interval_is_usable && !(data[i] >= left && data[i] <= right) {
            continue;
        }
        let mut running = 0.0_f64;
        for j in (1..num_bspline_cols).rev() {
            let term = db[[i, j]];
            if term.is_finite() {
                running += term;
            }
            out[[i, j - 1]] = running;
        }
    }
    Ok(out)
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

    // Non-periodic (open/clamped) B-spline derivative, kept consistent with the
    // value basis so it equals a finite difference of the value (gam#1348). The
    // exterior boundary treatment follows the value basis and depends on the knot
    // geometry: an *open* knot vector holds the value constant outside the
    // modeling interval, so its exterior derivative is zero; a *clamped* knot
    // vector extends the value linearly, so its exterior derivative is the nonzero
    // boundary slope obtained by evaluating at the clamped endpoint. Handle the
    // open-knot exterior explicitly; otherwise clamp to the interval and evaluate
    // (interior points are unchanged; clamped exterior points get the boundary
    // slope). The eval point must NOT be wrapped modulo a period for an open basis
    // — a periodic wrap moved a boundary-span point onto unrelated interior
    // columns; genuinely cyclic bases pre-wrap their input upstream.
    if open_knot_derivative_exterior_is_zero(x, knot_vector, degree) {
        out.fill(0.0);
        return Ok(());
    }
    let x_clamped = clamp_eval_point_to_modeling_interval(x, knot_vector, degree);
    let x_eval = one_sided_derivative_eval_point(x_clamped, knot_vector, degree);

    // Evaluate lower-degree (k-1) basis functions on the full knot support.
    internal::evaluate_splines_at_point_full_support_into(
        x_eval,
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

        let left_term = if !knot_span_is_degenerate(denom_left) && i < num_basis_lower {
            lower_basis[i] / denom_left
        } else {
            0.0
        };

        let right_term = if !knot_span_is_degenerate(denom_right) && (i + 1) < num_basis_lower {
            lower_basis[i + 1] / denom_right
        } else {
            0.0
        };

        out[i] = k * (left_term - right_term);
    }

    Ok(())
}

/// Per-basis M-spline normalization scales `(degree + 1) / (t_{i+d+1} - t_i)`.
///
/// The M-spline is the B-spline rescaled so each basis integrates to one over
/// its support; this factor is the shared normalization used by both the dense
/// and sparse builders.
fn mspline_scales(knot_vector: ArrayView1<f64>, degree: usize, num_basis: usize) -> Vec<f64> {
    let order = (degree + 1) as f64;
    (0..num_basis)
        .map(|i| order / (knot_vector[i + degree + 1] - knot_vector[i]))
        .collect()
}

pub(crate) fn create_mspline_dense(
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
    let scales = mspline_scales(knot_vector, degree, num_basis);

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

pub(crate) fn create_mspline_sparse(
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
    let scales = mspline_scales(knot_vector, degree, ncols);

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

pub(crate) fn validate_mspline_normalization_spans(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    let num_basis = knot_vector.len().saturating_sub(degree + 1);
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        if span <= 0.0 {
            crate::bail_invalid_basis!(
                "invalid M-spline normalization span at i={i}: t[i+degree+1]-t[i]={span:.3e} must be > 0"
            );
        }
    }
    Ok(())
}

pub(crate) fn create_ispline_dense(
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
    let mut left_offsets = vec![0.0_f64; num_bspline_basis];
    internal::cumulative_bspline_offsets_into(
        left,
        bs_degree,
        knot_vector,
        &mut left_local,
        &mut left_scratch,
        &mut left_offsets,
    );

    // Outside the knot domain the I-spline saturates: every basis is anchored
    // at 0 at `left` and reaches its right-cumulative mass (≈ 1 minus the
    // left-boundary offset) by `right`. Saturation is the definition of the
    // cumulative integral of an M-spline whose support is `[left, right]`, and
    // it preserves the I-spline value range [0, 1] — linearly extending past
    // the boundary would produce NEGATIVE basis entries for `x < left` and
    // entries `> 1` for `x > right`, violating both monotonicity inside [0, 1]
    // and the constraint that an I-spline is itself non-negative everywhere.
    // Callers that need a different out-of-domain behavior (e.g. survival
    // log-Λ that must keep growing past the right-most observation time) must
    // clamp inputs and add their own extrapolation correction — the basis
    // evaluator's contract is the same on the scalar and dense paths.
    for (row_i, &x) in data.iter().enumerate() {
        if x < left {
            // No cumulative mass yet — I_j(x) = 0 for every column.
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

/// Reusable scratch arena for the shared B-spline higher-derivative recurrence.
///
/// The derivative recursion
/// `B^{(m)}_{degree} = degree · (B^{(m-1)}_{degree-1}/Δ_left − B^{(m-1)}_{degree-1}/Δ_right)`
/// peels one order and one degree per level until it bottoms out in the first
/// derivative (which itself is evaluated from the plain degree-`d` basis).
/// Each level needs one lower-order output buffer; the base case additionally
/// needs a plain-basis buffer and a `internal::BsplineScratch`. This arena
/// owns that whole chain so a tight evaluation loop can amortise the
/// allocations across many points. Buffers grow on demand and are reused.
#[derive(Default)]
pub struct BsplineDerivativeWorkspace {
    /// Lower-order derivative buffers, one per recursion level (`chain[depth]`
    /// holds the order-`m-1` derivative consumed by the order-`m` step).
    pub(crate) chain: Vec<Vec<f64>>,
    /// Plain (non-derivative) basis buffer for the order-1 base case.
    pub(crate) lower_basis: Vec<f64>,
    /// Cox–de Boor scratch for the order-1 base case.
    pub(crate) lower_scratch: internal::BsplineScratch,
}

impl BsplineDerivativeWorkspace {
    /// Creates an empty workspace; buffers are sized lazily on first use.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a level-`depth` lower-order buffer of length `len`, zero-filled,
    /// growing the chain and the buffer in place as needed.
    #[inline]
    pub(crate) fn chain_buffer(&mut self, depth: usize, len: usize) -> &mut [f64] {
        if self.chain.len() <= depth {
            self.chain.resize_with(depth + 1, Vec::new);
        }
        let buf = &mut self.chain[depth];
        if buf.len() != len {
            buf.resize(len, 0.0);
        }
        for v in buf.iter_mut() {
            *v = 0.0;
        }
        buf
    }
}

/// Shared engine for B-spline derivatives of order `derivative_order ≥ 1`.
///
/// Implements the single de-Boor derivative recurrence
/// `B^{(m)}_{i,degree}(x) = degree · ( B^{(m-1)}_{i,degree-1}(x)/(t_{i+degree}−t_i)
///                                    − B^{(m-1)}_{i+1,degree-1}(x)/(t_{i+degree+1}−t_{i+1}) )`
/// recursively: order `m` is obtained from order `m−1` on degree `degree−1`,
/// bottoming out at order 1, which delegates to
/// [`evaluate_bspline_derivative_scalar_into`]. The order-2/3/4 public entry
/// points are thin adapters over this function — the recurrence body lives here
/// exactly once.
///
/// `depth` is the recursion level used to pick a distinct reusable buffer in
/// `workspace`; top-level callers pass `0`.
///
/// Returns derivatives in the raw spline basis. If a model uses an
/// identifiability/constrained basis `BZ`, the caller must apply that same
/// constraint transform in derivative space.
pub(crate) fn evaluate_bspline_derivative_recurrence_into(
    derivative_order: usize,
    x: f64,
    knot_vector: ArrayView1<f64>,
    degree: usize,
    out: &mut [f64],
    workspace: &mut BsplineDerivativeWorkspace,
    depth: usize,
) -> Result<(), BasisError> {
    if degree < derivative_order {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order,
            minimum_degree: derivative_order,
        });
    }
    // Resolve the top-level eval point's boundary treatment once, at `depth == 0`,
    // matching the value basis so every higher-order derivative agrees with a
    // finite difference of the value (gam#1348). On an *open* knot vector the value
    // is constant outside the modeling interval, so every derivative order is zero
    // there; on a *clamped* vector the value extends LINEARLY, so the exterior
    // first derivative is the constant boundary slope (obtained by clamping the
    // eval point to the interval) while every order ≥ 2 is identically zero — an
    // affine extension has no curvature. The earlier code clamped for all orders
    // and so returned the boundary's nonzero `B^{(k)}` for k ≥ 2 outside the
    // domain, disagreeing with both the dense builder
    // (`apply_dense_bspline_extrapolation`) and a finite difference of the value.
    // No periodic wrap for an open/clamped basis: wrapping is only correct for a
    // cyclic basis (whose evaluator pre-wraps its input) and corrupted the
    // boundary spans here.
    if depth == 0
        && (open_knot_derivative_exterior_is_zero(x, knot_vector, degree)
            || linear_extension_higher_derivative_is_zero(x, knot_vector, degree, derivative_order))
    {
        out.fill(0.0);
        return Ok(());
    }
    let x = if depth == 0 {
        clamp_eval_point_to_modeling_interval(x, knot_vector, degree)
    } else {
        x
    };

    // Order 1 is the base case: it is computed directly from the plain
    // degree-`degree` basis rather than from a lower-order derivative.
    if derivative_order <= 1 {
        let num_basis_lower = knot_vector.len().saturating_sub(degree);
        if workspace.lower_basis.len() < num_basis_lower {
            workspace.lower_basis.resize(num_basis_lower, 0.0);
        }
        return evaluate_bspline_derivative_scalar_into(
            x,
            knot_vector,
            degree,
            out,
            &mut workspace.lower_basis,
            &mut workspace.lower_scratch,
        );
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
    // Evaluate the order-(m-1) derivative on degree-1 into this level's buffer.
    // Length matches `num_basis` of the degree-(degree-1) basis:
    // `knot_vector.len() - (degree - 1) - 1 = knot_vector.len() - degree`.
    let num_basis_lower = knot_vector.len() - degree;

    // Move this level's buffer out of the workspace so the recursive call (which
    // needs `&mut workspace` for deeper levels and the base-case scratch) cannot
    // alias it; swap it back afterwards to preserve buffer reuse across points.
    workspace.chain_buffer(depth, num_basis_lower);
    let mut lower = std::mem::take(&mut workspace.chain[depth]);

    let recurse = evaluate_bspline_derivative_recurrence_into(
        derivative_order - 1,
        x,
        knot_vector,
        degree - 1,
        &mut lower,
        workspace,
        depth + 1,
    );
    workspace.chain[depth] = lower;
    recurse?;

    let lower = &workspace.chain[depth];
    let k = degree as f64;
    for i in 0..num_basis {
        let denom1 = knot_vector[i + degree] - knot_vector[i];
        let denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];
        let term1 = if !knot_span_is_degenerate(denom1) {
            k * lower[i] / denom1
        } else {
            0.0
        };
        let term2 = if !knot_span_is_degenerate(denom2) {
            k * lower[i + 1] / denom2
        } else {
            0.0
        };
        out[i] = term1 - term2;
    }

    Ok(())
}

/// Evaluates B-spline third derivatives at a single scalar point `x` into `out`.
///
/// Thin adapter over `evaluate_bspline_derivative_recurrence_into` with
/// `derivative_order = 3`; the de-Boor recurrence body lives there exactly once.
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
    let mut workspace = BsplineDerivativeWorkspace::new();
    evaluate_bspline_derivative_recurrence_into(3, x, knot_vector, degree, out, &mut workspace, 0)
}

/// Evaluates B-spline fourth derivatives at a single scalar point `x` into `out`.
///
/// Thin adapter over `evaluate_bspline_derivative_recurrence_into` with
/// `derivative_order = 4`; the de-Boor recurrence body lives there exactly once.
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
    let mut workspace = BsplineDerivativeWorkspace::new();
    evaluate_bspline_derivative_recurrence_into(4, x, knot_vector, degree, out, &mut workspace, 0)
}

/// gam#2695 — an I-spline and its own derivative tower must be ONE function.
///
/// `create_ispline_dense` saturates outside the modelling interval
/// `[knots[bs_degree], knots[num_bspline_basis]]`: the value is the all-zero row
/// below `left` and a constant row at and above `right`. That convention is
/// deliberate — a linear extension would produce negative I-spline entries below
/// `left` and entries above one past `right` — and it is written down at the
/// value site. Nothing enforced it on the derivative, which is built from a
/// CLAMPED B-spline whose own exterior convention is linear extension, so
/// `apply_dense_bspline_extrapolation` returned the boundary slope there.
///
/// The consequence #2695 measures: the survival link warp is
/// `q = q0 + Σ_j βw_j·I_j(q0)`, so every block that reaches `q` only through
/// `q0` carries `m1 = 1 + Σ_j βw_j·I'_j(q0)`. Outside the knot domain the warp
/// value is flat and `m1` was not, so the joint-Newton RHS asserted a
/// first-order change the objective does not make — invisible at `βw ≈ 0`,
/// which is the amplitude every existing oracle ran at.
#[cfg(test)]
mod ispline_exterior_derivative_2695_tests {
    use super::*;

    /// A clamped cubic knot vector: `has_clamped_bspline_boundaries` is TRUE,
    /// which is precisely the branch that extended linearly.
    fn clamped_knots() -> Array1<f64> {
        Array1::from_vec(vec![
            -3.0, -3.0, -3.0, -3.0, -1.5, 0.0, 1.5, 3.0, 3.0, 3.0, 3.0,
        ])
    }

    /// I-spline degree; the internal B-spline runs at `DEGREE + 1`.
    const DEGREE: usize = 2;

    fn value_row(x: f64) -> Vec<f64> {
        let knots = clamped_knots();
        let data = Array1::from_vec(vec![x]);
        create_ispline_dense(data.view(), knots.view(), DEGREE)
            .expect("i-spline value")
            .row(0)
            .to_vec()
    }

    fn derivative_row(x: f64, order: usize) -> Vec<f64> {
        let knots = clamped_knots();
        let data = Array1::from_vec(vec![x]);
        create_ispline_derivative_dense(data.view(), &knots, DEGREE, order)
            .expect("i-spline derivative")
            .row(0)
            .to_vec()
    }

    /// The premise, stated as a measurement rather than assumed: the value
    /// really is constant out there, so its derivative really is zero.
    #[test]
    fn the_ispline_value_is_constant_outside_the_modelling_interval() {
        for (a, b) in [(-4.0, -8.0), (4.0, 9.0)] {
            let left = value_row(a);
            let right = value_row(b);
            assert_eq!(
                left.len(),
                right.len(),
                "the basis width must not depend on the evaluation point"
            );
            for (j, (lo, hi)) in left.iter().zip(right.iter()).enumerate() {
                assert_eq!(
                    lo.to_bits(),
                    hi.to_bits(),
                    "I_{j}({a}) = {lo} but I_{j}({b}) = {hi}; the I-spline value is \
                     documented as saturating outside the knot domain"
                );
            }
        }
    }

    /// Positive control: INSIDE the interval the derivative is the derivative,
    /// so the assertion below is about the exterior and not about the routine
    /// being zero everywhere.
    #[test]
    fn the_ispline_derivative_matches_a_finite_difference_inside_the_interval() {
        let x = 0.4_f64;
        let h = 1.0e-5;
        let plus = value_row(x + h);
        let minus = value_row(x - h);
        let analytic = derivative_row(x, 1);
        let mut any_nonzero = false;
        for (j, value) in analytic.iter().enumerate() {
            let fd = (plus[j] - minus[j]) / (2.0 * h);
            assert!(
                (fd - value).abs() <= 1.0e-6 * (1.0 + value.abs()),
                "interior column {j}: analytic I'_{j}({x}) = {value:.9e} but the central \
                 difference of the value is {fd:.9e}"
            );
            any_nonzero |= value.abs() > 1.0e-6;
        }
        assert!(
            any_nonzero,
            "the interior control must exercise a non-zero derivative"
        );
    }

    /// The defect. Every order, both sides.
    #[test]
    fn the_ispline_derivative_is_zero_where_its_value_saturates() {
        for x in [-4.0_f64, -3.5, 3.5, 4.0, 12.0] {
            for order in 1..=4 {
                for (j, value) in derivative_row(x, order).iter().enumerate() {
                    assert_eq!(
                        *value, 0.0,
                        "order-{order} I-spline derivative at x={x} (outside the knot domain \
                         [-3, 3], where the value is constant) reports {value:.9e} in column \
                         {j}; a constant function has zero derivative"
                    );
                }
            }
        }
    }

    /// The warp factor #2695 is about, stated in its own terms: with
    /// non-negative coefficients the monotone warp multiplier
    /// `m1 = 1 + Σ_j βw_j·I'_j(q0)` must be EXACTLY 1 wherever the warp itself
    /// is flat, or the chain rule through `q0` invents a slope.
    #[test]
    fn the_monotone_warp_multiplier_is_one_where_the_warp_is_flat() {
        let beta_w = [0.30_f64, 0.40, 0.50, 0.60, 0.70, 0.80];
        for x in [-5.0_f64, 5.0] {
            let d1 = derivative_row(x, 1);
            assert_eq!(
                d1.len(),
                beta_w.len(),
                "fixture coefficient width must match the basis"
            );
            let m1: f64 = 1.0 + d1.iter().zip(beta_w.iter()).map(|(b, c)| b * c).sum::<f64>();
            assert_eq!(
                m1, 1.0,
                "at x={x} the warp value is constant, so its multiplier must be exactly 1"
            );
        }
    }
}
