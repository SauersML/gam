//! B-spline boundary-condition linear constraints for the 1-D smooth arm.
//!
//! Pure relocation from `smooth.rs` (issue #780 decomposition): the clamped /
//! anchored endpoint boundary-row machinery and the equality→two-sided-
//! inequality assembly that turns B-spline boundary conditions into a
//! `LinearInequalityConstraints`. No behavior change — bodies are
//! byte-identical and the single externally-called entry point is re-imported
//! by the parent so every call site is unchanged.

use crate::basis::{
    BSplineBoundaryConditions, BSplineEndpointBoundaryCondition, BasisError, BasisMetadata,
    BasisOptions, Dense, KnotSource,
};
use crate::pirls::LinearInequalityConstraints;
use ndarray::{Array1, Array2};

fn bspline_boundary_endpoint(
    knots: &Array1<f64>,
    degree: usize,
    right: bool,
) -> Result<f64, BasisError> {
    if knots.len() <= degree + 1 {
        crate::bail_invalid_basis!("B-spline boundary condition requires a valid knot vector");
    }
    let n_basis = knots.len() - degree - 1;
    Ok(if right { knots[n_basis] } else { knots[degree] })
}

fn bspline_endpoint_row(
    knots: &Array1<f64>,
    degree: usize,
    condition: BSplineEndpointBoundaryCondition,
    right: bool,
    identifiability_transform: Option<&Array2<f64>>,
    coefficient_transform: Option<&Array2<f64>>,
) -> Result<Option<(Array1<f64>, f64)>, BasisError> {
    let derivative_order = match condition {
        BSplineEndpointBoundaryCondition::Free => return Ok(None),
        BSplineEndpointBoundaryCondition::Clamped => 1,
        BSplineEndpointBoundaryCondition::Anchored { .. } => 0,
    };
    let target = match condition {
        BSplineEndpointBoundaryCondition::Free | BSplineEndpointBoundaryCondition::Clamped => 0.0,
        BSplineEndpointBoundaryCondition::Anchored { value } => value,
    };
    if !target.is_finite() {
        crate::bail_invalid_basis!("anchored B-spline boundary value must be finite");
    }
    let endpoint = bspline_boundary_endpoint(knots, degree, right)?;
    let point = Array1::from_vec(vec![endpoint]);
    let options = if derivative_order == 1 {
        BasisOptions::first_derivative()
    } else {
        BasisOptions::value()
    };
    let (raw, _) = crate::basis::create_basis::<Dense>(
        point.view(),
        KnotSource::Provided(knots.view()),
        degree,
        options,
    )?;
    let mut row = raw.row(0).to_owned();
    if let Some(z) = identifiability_transform {
        if row.len() != z.nrows() {
            crate::bail_dim_basis!(
                "B-spline boundary constraint transform mismatch: row has {} columns but transform has {} rows",
                row.len(),
                z.nrows()
            );
        }
        row = row.dot(z);
    }
    if let Some(t) = coefficient_transform {
        if row.len() != t.nrows() {
            crate::bail_dim_basis!(
                "B-spline boundary constraint coefficient transform mismatch: row has {} columns but transform has {} rows",
                row.len(),
                t.nrows()
            );
        }
        row = row.dot(t);
    }
    Ok(Some((row, target)))
}

pub(crate) fn bspline_boundary_linear_constraints(
    boundary_conditions: BSplineBoundaryConditions,
    metadata: &BasisMetadata,
    degree: usize,
    coefficient_transform: Option<&Array2<f64>>,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    if boundary_conditions.is_free() {
        return Ok(None);
    }
    let BasisMetadata::BSpline1D {
        knots,
        identifiability_transform,
        periodic: _,
        ..
    } = metadata
    else {
        crate::bail_invalid_basis!("B-spline boundary constraints require B-spline metadata");
    };

    let mut eq_rows = Vec::<Array1<f64>>::new();
    let mut eq_targets = Vec::<f64>::new();
    for (cond, right) in [
        (boundary_conditions.left, false),
        (boundary_conditions.right, true),
    ] {
        if let Some((row, target)) = bspline_endpoint_row(
            knots,
            degree,
            cond,
            right,
            identifiability_transform.as_ref(),
            coefficient_transform,
        )? {
            let norm = row.dot(&row).sqrt();
            if norm <= 1e-12 {
                if target.abs() > 1e-12 {
                    crate::bail_invalid_basis!(
                        "anchored B-spline boundary value {target} is infeasible because the endpoint row is zero"
                    );
                }
                continue;
            }
            eq_rows.push(row);
            eq_targets.push(target);
        }
    }
    if eq_rows.is_empty() {
        return Ok(None);
    }
    let p = eq_rows[0].len();
    let mut a = Array2::<f64>::zeros((2 * eq_rows.len(), p));
    let mut b = Array1::<f64>::zeros(2 * eq_rows.len());
    for (i, (row, &target)) in eq_rows.iter().zip(eq_targets.iter()).enumerate() {
        a.row_mut(2 * i).assign(row);
        b[2 * i] = target;
        a.row_mut(2 * i + 1).assign(&row.mapv(|v| -v));
        b[2 * i + 1] = -target;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}
