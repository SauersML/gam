//! Exact shape-constraint (monotone / convex / concave) machinery.
//!
//! Shape constraints are admitted only for open, untransformed B-spline
//! control coefficients.  In that chart, non-negative first control-point
//! differences certify monotonicity on every knot span, while non-decreasing
//! Greville-scaled control-polygon slopes certify convexity.  The smooth
//! builder realizes those cones with an invertible coefficient transform and
//! coordinate lower bounds; no sampled evaluation grid is involved.

use super::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
use crate::basis::{BSplineKnotSpec, OneDimensionalBoundary};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, s};

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
        BSplineKnotSpec::PeriodicUniform { .. }
            | BSplineKnotSpec::NaturalCubicRegression { .. }
    ) && matches!(&spec.boundary, OneDimensionalBoundary::Open)
        && spec.boundary_conditions.is_free()
}

pub(super) fn shape_uses_box_reparameterization(basis: &SmoothBasisSpec) -> bool {
    matches!(basis, SmoothBasisSpec::BSpline1D { .. })
}

pub fn linear_constraints_from_lower_bounds_global(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

pub fn merge_linear_constraints_global(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    match (first, second) {
        (None, None) => None,
        (Some(c), None) | (None, Some(c)) => Some(c),
        (Some(a), Some(b)) => {
            if a.a.ncols() != b.a.ncols() {
                return None;
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
            Some(LinearInequalityConstraints { a: mat, b: rhs })
        }
    }
}
