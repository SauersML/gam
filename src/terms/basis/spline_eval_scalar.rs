use super::*;

/// Scratch memory for B-spline evaluation to avoid allocations in tight loops.
pub struct SplineScratch {
    pub(crate) inner: internal::BsplineScratch,
    pub(crate) local: Vec<f64>,
    pub(crate) left_inner: internal::BsplineScratch,
    pub(crate) left_local: Vec<f64>,
    pub(crate) left_offsets: Vec<f64>,
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
    /// Cyclic finite-difference order used by curve fitting penalties.
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

    /// Evaluate the derivative of the fitted curve with respect to its scalar
    /// periodic parameter.
    pub fn evaluate_derivative(&self, u: ArrayView1<'_, f64>) -> Result<Array2<f64>, BasisError> {
        if self.coefficients.nrows() != self.spec.num_basis {
            crate::bail_dim_basis!(
                "curve coefficient rows ({}) must equal periodic basis size ({})",
                self.coefficients.nrows(),
                self.spec.num_basis
            );
        }
        let t = u.to_owned().insert_axis(Axis(1));
        let derivative = periodic_bspline_first_derivative_nd(
            t.view(),
            (self.spec.origin, self.spec.origin + self.spec.period),
            self.spec.degree,
            self.spec.num_basis,
        )?
        .index_axis(Axis(2), 0)
        .to_owned();
        Ok(derivative.dot(&self.coefficients))
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

pub(crate) fn solve_spd_cholesky(
    a: Array2<f64>,
    b: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n {
        crate::bail_dim_basis!(
            "normal-equation solve shape mismatch: A is {}x{}, B is {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        );
    }
    let mut jitter = 0.0_f64;
    for attempt in 0..8 {
        let mut l = a.clone();
        if jitter > 0.0 {
            for i in 0..n {
                l[[i, i]] += jitter;
            }
        }
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = l[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    if sum <= 0.0 || !sum.is_finite() {
                        ok = false;
                        break;
                    }
                    l[[i, j]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
            if !ok {
                break;
            }
            for j in (i + 1)..n {
                l[[i, j]] = 0.0;
            }
        }
        if ok {
            let mut y = Array2::<f64>::zeros(b.raw_dim());
            for i in 0..n {
                for rhs in 0..b.ncols() {
                    let mut sum = b[[i, rhs]];
                    for k in 0..i {
                        sum -= l[[i, k]] * y[[k, rhs]];
                    }
                    y[[i, rhs]] = sum / l[[i, i]];
                }
            }
            let mut x = Array2::<f64>::zeros(b.raw_dim());
            for i_rev in 0..n {
                let i = n - 1 - i_rev;
                for rhs in 0..b.ncols() {
                    let mut sum = y[[i, rhs]];
                    for k in (i + 1)..n {
                        sum -= l[[k, i]] * x[[k, rhs]];
                    }
                    x[[i, rhs]] = sum / l[[i, i]];
                }
            }
            return Ok(x);
        }
        let diag_scale = (0..n)
            .map(|i| a[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        jitter = if attempt == 0 {
            1e-12 * diag_scale
        } else {
            jitter * 10.0
        };
    }
    Err(BasisError::InvalidInput(
        "periodic spline normal equations were not positive definite even after jitter".to_string(),
    ))
}

/// Fit a vector-valued 1D periodic spline curve by penalized least squares.
///
/// `y` may have any positive number of columns. Each column is solved with the
/// same periodic basis and smoothing penalty, so the result is a single closed
/// curve `u -> R^d_ambient`. This deliberately makes no circularity or
/// isotropy assumption: ellipses, ovals, sheared loops, and other anisotropic
/// embeddings are represented by the learned multi-output coefficients.
pub fn fit_periodic_bspline_curve(
    u: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    spec: &PeriodicBSplineBasisSpec,
    smoothing_lambda: f64,
) -> Result<PeriodicSplineCurve, BasisError> {
    validate_periodic_bspline_spec(spec)?;
    if y.nrows() != u.len() {
        crate::bail_dim_basis!(
            "periodic curve fit requires y rows ({}) to match u length ({})",
            y.nrows(),
            u.len()
        );
    }
    if y.ncols() == 0 {
        crate::bail_invalid_basis!(
            "periodic curve fit requires at least one ambient output column"
        );
    }
    if !smoothing_lambda.is_finite() || smoothing_lambda < 0.0 {
        crate::bail_invalid_basis!(
            "smoothing_lambda must be finite and nonnegative, got {smoothing_lambda}"
        );
    }
    if y.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("periodic curve outputs must all be finite");
    }

    let basis = build_periodic_bspline_basis_1d(u, spec)?;
    let mut lhs = basis.t().dot(&basis);
    if smoothing_lambda > 0.0 {
        let penalty = create_cyclic_difference_penalty_matrix(spec.num_basis, spec.penalty_order)?;
        lhs = lhs + smoothing_lambda * penalty;
    }
    // A tiny ridge selects a stable coefficient representative in the rare case
    // of undersampled or exactly aliased parameter grids while leaving ordinary
    // fits unchanged at test tolerances.
    let diag_scale = (0..lhs.nrows())
        .map(|i| lhs[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    for i in 0..lhs.nrows() {
        lhs[[i, i]] += 1e-12 * diag_scale;
    }
    let rhs = basis.t().dot(&y);
    let coefficients = solve_spd_cholesky(lhs, &rhs)?;
    Ok(PeriodicSplineCurve {
        spec: spec.clone(),
        coefficients,
    })
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
        crate::bail_dim_basis!(
            "M-spline output buffer length {} does not match basis size {}",
            out.len(),
            num_basis
        );
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
        crate::bail_dim_basis!(
            "I-spline output buffer length {} does not match basis size {}",
            out.len(),
            num_ispline_basis
        );
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

    let x_eval = one_sided_derivative_eval_point(x, knot_vector, degree);

    // Evaluate lower-degree (k-1) basis functions
    internal::evaluate_splines_at_point_into(
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

        let left_term = if denom_left.abs() > KNOT_SPAN_DEGENERACY_FLOOR && i < num_basis_lower {
            lower_basis[i] / denom_left
        } else {
            0.0
        };

        let right_term =
            if denom_right.abs() > KNOT_SPAN_DEGENERACY_FLOOR && (i + 1) < num_basis_lower {
                lower_basis[i + 1] / denom_right
            } else {
                0.0
            };

        out[i] = k * (left_term - right_term);
    }

    Ok(())
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

pub(crate) fn validate_mspline_normalization_spans(
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(), BasisError> {
    let num_basis = knot_vector.len().saturating_sub(degree + 1);
    for i in 0..num_basis {
        let span = knot_vector[i + degree + 1] - knot_vector[i];
        if span <= KNOT_SPAN_DEGENERACY_FLOOR {
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
/// needs a plain-basis buffer and a [`internal::BsplineScratch`]. This arena
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
    if num_basis > 0 {
        let left = knot_vector[degree];
        let right = knot_vector[num_basis];
        if x < left || x > right {
            out.fill(0.0);
            return Ok(());
        }
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
        let term1 = if denom1.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            k * lower[i] / denom1
        } else {
            0.0
        };
        let term2 = if denom2.abs() > KNOT_SPAN_DEGENERACY_FLOOR {
            k * lower[i + 1] / denom2
        } else {
            0.0
        };
        out[i] = term1 - term2;
    }

    Ok(())
}

/// Evaluates B-spline second derivatives at a single scalar point `x` into `out`.
///
/// Thin adapter over [`evaluate_bspline_derivative_recurrence_into`] with
/// `derivative_order = 2`; the de-Boor recurrence body lives there exactly once.
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
    let mut workspace = BsplineDerivativeWorkspace::new();
    evaluate_bspline_derivative_recurrence_into(2, x, knot_vector, degree, out, &mut workspace, 0)
}

/// Evaluates B-spline third derivatives at a single scalar point `x` into `out`.
///
/// Thin adapter over [`evaluate_bspline_derivative_recurrence_into`] with
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
/// Thin adapter over [`evaluate_bspline_derivative_recurrence_into`] with
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
