//! Shape-constraint (monotone / convex / concave) machinery for the 1-D smooth
//! arm.
//!
//! Pure relocation from `smooth.rs` (issue #780 decomposition): the
//! shape-constraint order/sign lookup, per-coefficient lower-bound vector,
//! basis-support gate, box-reparameterization gate, the 1-D grid + design
//! reconstruction used to assemble the inequality rows, and the
//! `LinearInequalityConstraints` assembly/merge helpers. No behavior change —
//! bodies are byte-identical and the entry points are re-imported by the parent
//! so every call site is unchanged.

use super::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
use crate::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    BasisError, BasisMetadata, DuchonBasisSpec, MaternBasisSpec, MaternIdentifiability,
    SpatialIdentifiability, ThinPlateBasisSpec, build_bspline_basis_1d, build_duchon_basis,
    build_matern_basis, build_thin_plate_basis,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

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
    !matches!(
        term.basis,
        SmoothBasisSpec::TensorBSpline { .. } | SmoothBasisSpec::Pca { .. }
    )
}

pub(super) fn shape_uses_box_reparameterization(basis: &SmoothBasisSpec) -> bool {
    matches!(basis, SmoothBasisSpec::BSpline1D { .. })
}

pub(super) fn build_shape_constraint_grid_1d(
    x: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, BasisError> {
    if x.is_empty() {
        crate::bail_invalid_basis!("shape-constrained smooth requires non-empty covariate values");
    }
    if x.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("shape-constrained smooth requires finite covariate values");
    }

    let mut x_sorted: Vec<f64> = x.iter().copied().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut x_unique: Vec<f64> = Vec::with_capacity(x_sorted.len());
    let mut last: Option<f64> = None;
    for v in x_sorted {
        let take = match last {
            None => true,
            Some(prev) => (v - prev).abs() > 1e-12 * prev.abs().max(v.abs()).max(1.0),
        };
        if take {
            x_unique.push(v);
            last = Some(v);
        }
    }
    if x_unique.len() < 2 {
        crate::bail_invalid_basis!(
            "shape-constrained smooth requires at least two unique covariate values"
        );
    }

    let min_x = x_unique[0];
    let max_x = *x_unique
        .last()
        .expect("x_unique has at least two elements by construction");
    if (max_x - min_x).abs() <= 1e-12 {
        crate::bail_invalid_basis!(
            "shape-constrained smooth requires non-degenerate covariate range"
        );
    }

    let target_points = x_unique.len().clamp(96, 320);
    let mut grid = Array1::<f64>::zeros(target_points);
    let denom = (target_points - 1) as f64;
    for i in 0..target_points {
        let t = i as f64 / denom;
        grid[i] = min_x + t * (max_x - min_x);
    }
    Ok(grid)
}

pub(super) fn build_shape_constraint_design_1d(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    metadata: &BasisMetadata,
    axis_col: usize,
) -> Result<(Array1<f64>, Array2<f64>), BasisError> {
    let x_grid = build_shape_constraint_grid_1d(data.column(axis_col))?;
    let grid_2d = x_grid
        .clone()
        .into_shape_with_order((x_grid.len(), 1))
        .map_err(|e| {
            BasisError::InvalidInput(format!(
                "failed to construct 1D shape grid matrix for term '{}': {e}",
                term.name
            ))
        })?;

    let design = match (&term.basis, metadata) {
        (
            SmoothBasisSpec::BSpline1D { spec, .. },
            BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic,
                degree: meta_degree,
                ..
            },
        ) => {
            // Issue #340: predict against the metadata-recorded effective
            // degree so fit-time auto-shrink (cubic → linear for small n) is
            // honoured at prediction time too.
            let effective_degree = meta_degree.unwrap_or(spec.degree);
            let evalspec = BSplineBasisSpec {
                degree: effective_degree,
                penalty_order: spec.penalty_order,
                knotspec: periodic
                    .map(
                        |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        },
                    )
                    .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone())),
                double_penalty: false,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| BSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or(BSplineIdentifiability::None),
                boundary: spec.boundary.clone(),
                boundary_conditions: BSplineBoundaryConditions::default(),
            };
            build_bspline_basis_1d(x_grid.view(), &evalspec)?
                .design
                .to_dense()
        }
        (
            SmoothBasisSpec::ThinPlate { .. },
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                identifiability_transform,
                radial_reparam,
                ..
            },
        ) => {
            let evalspec = ThinPlateBasisSpec {
                periodic: None,
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                double_penalty: false,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or(SpatialIdentifiability::None),
                radial_reparam: radial_reparam.clone(),
            };
            build_thin_plate_basis(grid_2d.view(), &evalspec)?
                .design
                .to_dense()
        }
        (
            SmoothBasisSpec::Matern { .. },
            BasisMetadata::Matern {
                centers,
                length_scale,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                ..
            },
        ) => {
            let ident = identifiability_transform
                .as_ref()
                .map(|z| MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                    // Predict-time design rebuild: penalties are not assembled
                    // here (`double_penalty: false` below), so the frozen
                    // shrinkage decision is irrelevant on this path.
                    nullspace_shrinkage_survived: None,
                })
                .unwrap_or(MaternIdentifiability::None);
            let evalspec = MaternBasisSpec {
                periodic: None,
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                nu: *nu,
                include_intercept: *include_intercept,
                double_penalty: false,
                identifiability: ident,
                aniso_log_scales: aniso_log_scales.clone(),
                nullspace_shrinkage_survived: None,
            };
            build_matern_basis(grid_2d.view(), &evalspec)?
                .design
                .to_dense()
        }
        (
            SmoothBasisSpec::Duchon { spec, .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                power,
                nullspace_order,
                identifiability_transform,
                aniso_log_scales,
                radial_reparam,
                ..
            },
        ) => {
            let evalspec = DuchonBasisSpec {
                periodic: None,
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                power: *power,
                nullspace_order: *nullspace_order,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or_else(|| spec.identifiability.clone()),
                aniso_log_scales: aniso_log_scales.clone(),
                operator_penalties: spec.operator_penalties.clone(),
                boundary: spec.boundary.clone(),
                radial_reparam: radial_reparam.clone(),
            };
            build_duchon_basis(grid_2d.view(), &evalspec)?
                .design
                .to_dense()
        }
        _ => {
            crate::bail_invalid_basis!(
                "shape-constraint grid reconstruction metadata mismatch for term '{}'",
                term.name
            );
        }
    };

    Ok((x_grid, design))
}

pub(super) fn build_shape_linear_constraints_1d(
    x: ArrayView1<'_, f64>,
    design_local: ArrayView2<'_, f64>,
    shape: ShapeConstraint,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let Some((order, sign)) = shape_order_and_sign(shape) else {
        return Ok(None);
    };
    let n = x.len();
    let p = design_local.ncols();
    if n == 0 || p == 0 {
        return Ok(None);
    }
    if x.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("shape-constrained smooth requires finite covariate values");
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));

    let x_scale = x.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let x_tol = 1e-12 * x_scale;
    let mut collapsedrows: Vec<Array1<f64>> = Vec::new();
    let mut group_sum = Array1::<f64>::zeros(p);
    let mut group_count = 0usize;
    let mut last_x: Option<f64> = None;
    for &r in &idx {
        let xr = x[r];
        let start_new = match last_x {
            None => false,
            Some(prev) => (xr - prev).abs() > x_tol,
        };
        if start_new {
            if group_count > 0 {
                collapsedrows.push(group_sum.mapv(|v| v / group_count as f64));
            }
            group_sum.fill(0.0);
            group_count = 0;
        }
        group_sum.scaled_add(1.0, &design_local.row(r));
        group_count += 1;
        last_x = Some(xr);
    }
    if group_count > 0 {
        collapsedrows.push(group_sum.mapv(|v| v / group_count as f64));
    }

    let m = collapsedrows.len();
    if m <= order {
        crate::bail_invalid_basis!(
            "shape-constrained smooth requires at least {} unique covariate locations; found {}",
            order + 1,
            m
        );
    }

    let q_raw = m - order;
    let mut arows: Vec<Array1<f64>> = Vec::with_capacity(q_raw);
    for i in 0..q_raw {
        let row = if order == 1 {
            &collapsedrows[i + 1] - &collapsedrows[i]
        } else {
            &collapsedrows[i + 2] - &collapsedrows[i + 1].mapv(|v| 2.0 * v) + &collapsedrows[i]
        };
        let mut row_signed = row;
        if sign < 0.0 {
            row_signed.mapv_inplace(|v| -v);
        }
        let norm = row_signed.dot(&row_signed).sqrt();
        if norm > 1e-12 {
            arows.push(row_signed);
        }
    }
    if arows.is_empty() {
        return Ok(None);
    }

    let mut a = Array2::<f64>::zeros((arows.len(), p));
    for (i, row) in arows.iter().enumerate() {
        a.row_mut(i).assign(row);
    }
    let b = Array1::<f64>::zeros(a.nrows());
    Ok(Some(LinearInequalityConstraints { a, b }))
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
