//! Exact shape-constraint (monotone / convex / concave) machinery.
//!
//! Shape constraints are admitted only for open, untransformed B-spline
//! control coefficients.  In that chart, non-negative first control-point
//! differences certify monotonicity on every knot span, while non-decreasing
//! Greville-scaled control-polygon slopes certify convexity.  The smooth
//! builder realizes those cones with an invertible coefficient transform and
//! coordinate lower bounds; no sampled evaluation grid is involved.

use super::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
use crate::basis::{BSplineKnotSpec, BasisError, OneDimensionalBoundary};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1, s};

pub(super) fn shape_order_and_sign(shape: ShapeConstraint) -> Option<(usize, f64)> {
    match shape {
        ShapeConstraint::None => None,
        ShapeConstraint::MonotoneIncreasing => Some((1, 1.0)),
        ShapeConstraint::MonotoneDecreasing => Some((1, -1.0)),
        ShapeConstraint::Convex => Some((2, 1.0)),
        ShapeConstraint::Concave => Some((2, -1.0)),
    }
}

pub fn shape_lower_bounds_local(shape: ShapeConstraint, dim: usize) -> Option<Array1<f64>> {
    let (order, _) = shape_order_and_sign(shape)?;
    if dim <= order {
        return None;
    }
    let mut lb = Array1::<f64>::from_elem(dim, f64::NEG_INFINITY);
    for j in order..dim {
        lb[j] = 0.0;
    }
    Some(lb)
}

pub(super) fn shape_supports_basis(term: &SmoothTermSpec) -> bool {
    let SmoothBasisSpec::BSpline1D { spec, .. } = &term.basis else {
        return false;
    };

    // A cyclic spline cannot be globally monotone unless it is constant, and
    // the periodic coefficient chart requires a wrap-around constraint that
    // the open cumulative transform does not encode.  Natural cubic regression
    // coefficients are knot values rather than raw B-spline control points, so
    // coefficient differences do not certify against cubic overshoot.  Endpoint
    // boundary conditions likewise introduce a raw-basis nullspace transform.
    // Reject all three instead of pretending their transformed coordinates have
    // the control-polygon geometry used by the exact cone below.
    !matches!(
        &spec.knotspec,
        BSplineKnotSpec::PeriodicUniform { .. } | BSplineKnotSpec::NaturalCubicRegression { .. }
    ) && matches!(&spec.boundary, OneDimensionalBoundary::Open)
        && spec.boundary_conditions.is_free()
}

pub(super) fn shape_uses_box_reparameterization(basis: &SmoothBasisSpec) -> bool {
    matches!(basis, SmoothBasisSpec::BSpline1D { .. })
}

/// First-derivative control denominators for an open B-spline basis.
///
/// For `f = sum_i beta_i N_{i,d}`, the derivative control multiplying the
/// degree-`d - 1` basis function is
///
/// `d * (beta[i + 1] - beta[i]) / (t[i + d + 1] - t[i + 1])`.
///
/// Dividing every denominator by `d` gives the adjacent Greville-abscissa
/// gaps, but computing the gaps directly from knot differences avoids loss of
/// translation invariance from subtracting two separately averaged abscissae.
pub(crate) fn bspline_first_derivative_control_spans(
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    if degree == 0 {
        return Err(BasisError::InvalidDegree(degree));
    }
    let required = degree
        .checked_add(1)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| BasisError::InvalidInput("B-spline degree overflows usize".to_string()))?;
    if knots.len() < required {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required,
            provided: knots.len(),
        });
    }
    if knots.iter().any(|value| !value.is_finite()) {
        return Err(BasisError::InvalidKnotVector(
            "knot vector contains non-finite (NaN or Infinity) values".to_string(),
        ));
    }
    for pair in knots.windows(2) {
        if pair[0] > pair[1] {
            return Err(BasisError::InvalidKnotVector(
                "knot vector is not non-decreasing".to_string(),
            ));
        }
    }

    let coefficient_count = knots.len() - degree - 1;
    let mut spans = Array1::<f64>::zeros(coefficient_count.saturating_sub(1));
    let degree_scale = degree as f64;
    for i in 0..spans.len() {
        let width = knots[i + degree + 1] - knots[i + 1];
        if !width.is_finite() || width <= 0.0 {
            return Err(BasisError::InvalidKnotVector(format!(
                "shape-constrained derivative control span t[{}]-t[{}]={width:.3e} must be finite and positive",
                i + degree + 1,
                i + 1,
            )));
        }
        spans[i] = width / degree_scale;
    }
    Ok(spans)
}

/// Exact continuum shape cone for raw open-B-spline control coefficients.
///
/// The returned rows describe `A * beta >= 0`. Monotonicity rows constrain
/// consecutive control-point differences. Curvature rows constrain consecutive
/// first-derivative controls, using the exact knot-dependent denominators. As a
/// result, the number and values of the rows depend only on the realized spline
/// chart and never on an evaluation grid.
///
/// `ShapeConstraint::None` returns `None`. A nontrivial shape on an affine basis
/// can legitimately return a zero-row constraint: an affine function is both
/// convex and concave without imposing a coefficient restriction.
pub fn bspline_shape_linear_constraints(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    shape: ShapeConstraint,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let Some((order, sign)) = shape_order_and_sign(shape) else {
        return Ok(None);
    };
    let spans = bspline_first_derivative_control_spans(knots, degree)?;
    let coefficient_count = spans.len() + 1;
    let row_count = coefficient_count.saturating_sub(order);
    let mut a = Array2::<f64>::zeros((row_count, coefficient_count));

    match order {
        1 => {
            for row in 0..row_count {
                a[[row, row]] = -sign;
                a[[row, row + 1]] = sign;
            }
        }
        2 => {
            for row in 0..row_count {
                let left = spans[row];
                let right = spans[row + 1];
                // Positive scaling by left*right turns the reciprocal form
                // [1/left, -(1/left + 1/right), 1/right] into this stable row.
                // Divide both spans by their maximum before adding them so a
                // valid very-large-domain spline cannot overflow in `left + right`.
                let span_scale = left.max(right);
                let left = left / span_scale;
                let right = right / span_scale;
                a[[row, row]] = sign * right;
                a[[row, row + 1]] = -sign * (left + right);
                a[[row, row + 2]] = sign * left;
            }
        }
        _ => {
            return Err(BasisError::InvalidInput(format!(
                "unsupported B-spline shape derivative order {order}"
            )));
        }
    }

    for mut row in a.rows_mut() {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(BasisError::InvalidInput(
                "shape-constraint row has no finite direction".to_string(),
            ));
        }
        row.mapv_inplace(|value| value / norm);
    }

    Ok(Some(LinearInequalityConstraints {
        b: Array1::zeros(row_count),
        a,
    }))
}

pub fn linear_constraints_from_lower_bounds_global(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

pub fn merge_linear_constraints_global(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(c), None) | (None, Some(c)) => {
            if c.a.nrows() != c.b.len() {
                return Err(BasisError::DimensionMismatch(format!(
                    "linear constraint has {} rows but {} right-hand-side values",
                    c.a.nrows(),
                    c.b.len()
                )));
            }
            Ok(Some(c))
        }
        (Some(a), Some(b)) => {
            if a.a.ncols() != b.a.ncols() {
                return Err(BasisError::DimensionMismatch(format!(
                    "cannot merge linear constraints with {} and {} columns",
                    a.a.ncols(),
                    b.a.ncols()
                )));
            }
            if a.a.nrows() != a.b.len() || b.a.nrows() != b.b.len() {
                return Err(BasisError::DimensionMismatch(format!(
                    "cannot merge linear constraints with row/RHS shapes {}x{}/{} and {}x{}/{}",
                    a.a.nrows(),
                    a.a.ncols(),
                    a.b.len(),
                    b.a.nrows(),
                    b.a.ncols(),
                    b.b.len()
                )));
            }
            let m1 = a.a.nrows();
            let m2 = b.a.nrows();
            let p = a.a.ncols();
            let mut mat = Array2::<f64>::zeros((m1 + m2, p));
            mat.slice_mut(s![0..m1, ..]).assign(&a.a);
            mat.slice_mut(s![m1..(m1 + m2), ..]).assign(&b.a);
            let mut rhs = Array1::<f64>::zeros(m1 + m2);
            rhs.slice_mut(s![0..m1]).assign(&a.b);
            rhs.slice_mut(s![m1..(m1 + m2)]).assign(&b.b);
            Ok(Some(LinearInequalityConstraints { a: mat, b: rhs }))
        }
    }
}

#[cfg(test)]
mod exact_bspline_shape_tests {
    use super::*;
    use ndarray::array;

    fn irregular_cubic_knots() -> Array1<f64> {
        array![0.0, 0.0, 0.0, 0.0, 0.08, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0]
    }

    #[test]
    fn all_shape_rows_are_exact_derivative_control_cones() {
        let knots = irregular_cubic_knots();
        let p = knots.len() - 4;
        let increasing =
            bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::MonotoneIncreasing)
                .unwrap()
                .unwrap();
        let decreasing =
            bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::MonotoneDecreasing)
                .unwrap()
                .unwrap();
        let convex = bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::Convex)
            .unwrap()
            .unwrap();
        let concave = bspline_shape_linear_constraints(knots.view(), 3, ShapeConstraint::Concave)
            .unwrap()
            .unwrap();

        assert_eq!(increasing.a.dim(), (p - 1, p));
        assert_eq!(convex.a.dim(), (p - 2, p));
        assert_eq!(decreasing.a, -&increasing.a);
        assert_eq!(concave.a, -&convex.a);
        assert_eq!(increasing.b, Array1::zeros(p - 1));
        assert_eq!(convex.b, Array1::zeros(p - 2));

        let spans = bspline_first_derivative_control_spans(knots.view(), 3).unwrap();
        for row in 0..convex.a.nrows() {
            let scale = spans[row].max(spans[row + 1]);
            let left = spans[row] / scale;
            let right = spans[row + 1] / scale;
            let expected = array![right, -(left + right), left];
            let expected = &expected / expected.dot(&expected).sqrt();
            assert_eq!(convex.a.slice(s![row, row..row + 3]), expected.view());
        }
    }

    #[test]
    fn rows_are_invariant_to_positive_affine_knot_units() {
        let knots = irregular_cubic_knots();
        let shifted = knots.mapv(|value| 17.0 + 9.0 * value);
        for shape in [
            ShapeConstraint::MonotoneIncreasing,
            ShapeConstraint::MonotoneDecreasing,
            ShapeConstraint::Convex,
            ShapeConstraint::Concave,
        ] {
            let base = bspline_shape_linear_constraints(knots.view(), 3, shape)
                .unwrap()
                .unwrap();
            let transformed = bspline_shape_linear_constraints(shifted.view(), 3, shape)
                .unwrap()
                .unwrap();
            for (left, right) in base.a.iter().zip(transformed.a.iter()) {
                assert!((left - right).abs() <= 32.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn affine_linear_spline_has_vacuous_curvature_cone() {
        let knots = array![0.0, 0.0, 1.0, 1.0];
        for shape in [ShapeConstraint::Convex, ShapeConstraint::Concave] {
            let constraints = bspline_shape_linear_constraints(knots.view(), 1, shape)
                .unwrap()
                .unwrap();
            assert_eq!(constraints.a.dim(), (0, 2));
            assert!(constraints.b.is_empty());
            assert!(shape_lower_bounds_local(shape, 2).is_none());
        }
    }

    #[test]
    fn derivative_control_geometry_rejects_collapsed_spans() {
        let knots = array![0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
        let err =
            bspline_shape_linear_constraints(knots.view(), 1, ShapeConstraint::MonotoneIncreasing)
                .unwrap_err();
        assert!(err.to_string().contains("must be finite and positive"));
    }

    #[test]
    fn incompatible_constraint_blocks_are_errors_not_dropped_cones() {
        let left = LinearInequalityConstraints {
            a: Array2::eye(2),
            b: Array1::zeros(2),
        };
        let right = LinearInequalityConstraints {
            a: Array2::eye(3),
            b: Array1::zeros(3),
        };
        let err = merge_linear_constraints_global(Some(left), Some(right)).unwrap_err();
        assert!(err.to_string().contains("cannot merge linear constraints"));
    }
}
