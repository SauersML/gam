use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisBuildResult, BasisError,
    BasisMetadata, DuchonBasisSpec, MaternBasisSpec, MaternIdentifiability, SpatialIdentifiability,
    ThinPlateBasisSpec, apply_sum_to_zero_constraint, apply_weighted_orthogonality_constraint,
    build_bspline_basis_1d, build_duchon_basis, build_matern_basis, build_thin_plate_basis,
    create_bspline_basis_nd_with_knots, estimate_penalty_nullity,
};
use crate::construction::kronecker_product;
use crate::estimate::{EstimationError, FitOptions, FitResult, fit_gam};
use crate::pirls::LinearInequalityConstraints;
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f64;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeConstraint {
    None,
    MonotoneIncreasing,
    MonotoneDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothBasisSpec {
    BSpline1D {
        feature_col: usize,
        spec: BSplineBasisSpec,
    },
    ThinPlate {
        feature_cols: Vec<usize>,
        spec: ThinPlateBasisSpec,
    },
    Matern {
        feature_cols: Vec<usize>,
        spec: MaternBasisSpec,
    },
    Duchon {
        feature_cols: Vec<usize>,
        spec: DuchonBasisSpec,
    },
    /// Tensor-product smooth built from 1D B-spline marginals.
    ///
    /// This is the `te()`-style construction used when axes have different units/scales
    /// (for example, space x time) and isotropic radial kernels are not appropriate.
    TensorBSpline {
        feature_cols: Vec<usize>,
        spec: TensorBSplineSpec,
    },
}

/// Tensor-product B-spline smooth specification.
///
/// `marginal_specs[i]` is the 1D B-spline setup for `feature_cols[i]`.
/// The final penalty set is one Kronecker penalty per margin:
/// `S_i = I ⊗ ... ⊗ S_marginal_i ⊗ ... ⊗ I`, plus optional global ridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBSplineSpec {
    pub marginal_specs: Vec<BSplineBasisSpec>,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: TensorBSplineIdentifiability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorBSplineIdentifiability {
    None,
    SumToZero,
    FrozenTransform { transform: Array2<f64> },
}

impl Default for TensorBSplineIdentifiability {
    fn default() -> Self {
        Self::SumToZero
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothTermSpec {
    pub name: String,
    pub basis: SmoothBasisSpec,
    pub shape: ShapeConstraint,
}

#[derive(Debug, Clone)]
pub struct SmoothTerm {
    pub name: String,
    pub coeff_range: Range<usize>,
    pub shape: ShapeConstraint,
    pub penalties_local: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub metadata: BasisMetadata,
    /// Optional term-local lower bounds for constrained coefficients.
    /// `-inf` means unconstrained.
    pub lower_bounds_local: Option<Array1<f64>>,
    /// Optional term-local inequality constraints in local coefficient coordinates.
    /// `A_local * beta_local >= b_local`.
    pub linear_constraints_local: Option<LinearInequalityConstraints>,
}

#[derive(Debug, Clone)]
pub struct SmoothDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub terms: Vec<SmoothTerm>,
    /// Optional smooth-block lower bounds in smooth coefficient coordinates.
    /// Length equals `design.ncols()` when present.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional smooth-block inequality constraints:
    /// `A_smooth * beta_smooth >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// Optional double-penalty ridge on this linear coefficient.
    /// If true, emits an identity penalty block for this 1D term.
    pub double_penalty: bool,
}

/// Random-effects term specification.
///
/// The selected feature column is interpreted as a categorical grouping variable.
/// The term contributes a one-hot dummy block with an identity penalty on group
/// coefficients, equivalent to i.i.d. Gaussian random effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffectTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// If true, drop the lexicographically first group level to use treatment coding.
    /// If false, keep all levels (full one-hot block, still identifiable under ridge).
    pub drop_first_level: bool,
    /// Optional fixed kept-level set (sorted by f64 bit pattern) captured at fit time.
    /// When present, prediction uses exactly these columns to avoid design drift.
    #[serde(default)]
    pub frozen_levels: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCollectionSpec {
    pub linear_terms: Vec<LinearTermSpec>,
    pub random_effect_terms: Vec<RandomEffectTermSpec>,
    pub smooth_terms: Vec<SmoothTermSpec>,
}

#[derive(Debug, Clone)]
pub struct TermCollectionDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    /// Optional global coefficient lower bounds for constrained fitting.
    /// Length equals `design.ncols()` when present. Unconstrained entries are `-inf`.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional global inequality constraints:
    /// `A * beta >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    pub intercept_range: Range<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_levels: Vec<(String, Vec<u64>)>,
    pub smooth: SmoothDesign,
}

pub struct FittedTermCollection {
    pub fit: FitResult,
    pub design: TermCollectionDesign,
}

pub struct FittedTermCollectionWithSpec {
    pub fit: FitResult,
    pub design: TermCollectionDesign,
    pub resolved_spec: TermCollectionSpec,
}

pub struct TwoBlockMaternKappaOptimizationResult<FitOut> {
    pub resolved_mean_spec: TermCollectionSpec,
    pub resolved_noise_spec: TermCollectionSpec,
    pub mean_design: TermCollectionDesign,
    pub noise_design: TermCollectionDesign,
    pub fit: FitOut,
}

#[derive(Debug, Clone)]
pub struct MaternKappaOptimizationOptions {
    /// Enable outer-loop optimization over Matérn κ (= 1 / length_scale).
    pub enabled: bool,
    /// Maximum number of coordinate-descent passes over Matérn terms.
    pub max_outer_iter: usize,
    /// Relative improvement threshold for accepting a κ update.
    pub rel_tol: f64,
    /// Half-width of local search bracket in log(length_scale) units.
    pub log_step: f64,
    /// Minimum allowed length_scale during κ search.
    pub min_length_scale: f64,
    /// Maximum allowed length_scale during κ search.
    pub max_length_scale: f64,
}

impl Default for MaternKappaOptimizationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_outer_iter: 3,
            rel_tol: 1e-4,
            // Search around current scale by approximately x0.5 and x2.0.
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
        }
    }
}

#[derive(Debug, Clone)]
struct RandomEffectBlock {
    name: String,
    design: Array2<f64>,
    kept_levels: Vec<u64>,
}

fn select_columns(data: ArrayView2<'_, f64>, cols: &[usize]) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    for &c in cols {
        if c >= p {
            return Err(BasisError::DimensionMismatch(format!(
                "feature column {c} is out of bounds for data with {p} columns"
            )));
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&data.column(c));
    }
    Ok(out)
}

fn cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += values[i].exp();
        out[i] = sign * run;
    }
    out
}

fn second_cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let first = cumulative_exp(values, sign);
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += first[i];
        out[i] = run;
    }
    out
}

fn cumulative_sum_transform_matrix(dim: usize, order: usize, sign: f64) -> Array2<f64> {
    let mut t = Array2::<f64>::eye(dim);
    for _ in 0..order {
        let mut next = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..=i {
                next[[i, j]] = 1.0;
            }
        }
        t = t.dot(&next);
    }
    if sign < 0.0 {
        t.mapv_inplace(|v| -v);
    }
    t
}

fn shape_order_and_sign(shape: ShapeConstraint) -> Option<(usize, f64)> {
    match shape {
        ShapeConstraint::None => None,
        ShapeConstraint::MonotoneIncreasing => Some((1, 1.0)),
        ShapeConstraint::MonotoneDecreasing => Some((1, -1.0)),
        ShapeConstraint::Convex => Some((2, 1.0)),
        ShapeConstraint::Concave => Some((2, -1.0)),
    }
}

fn shape_lower_bounds_local(shape: ShapeConstraint, dim: usize) -> Option<Array1<f64>> {
    let (order, _) = shape_order_and_sign(shape)?;
    let mut lb = Array1::<f64>::from_elem(dim, f64::NEG_INFINITY);
    for j in order..dim {
        lb[j] = 0.0;
    }
    Some(lb)
}

fn shape_supports_basis(term: &SmoothTermSpec) -> bool {
    let _ = term;
    true
}

fn shape_uses_box_reparameterization(basis: &SmoothBasisSpec) -> bool {
    matches!(
        basis,
        SmoothBasisSpec::BSpline1D { .. } | SmoothBasisSpec::TensorBSpline { .. }
    )
}

fn build_shape_constraint_grid_1d(x: ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError> {
    if x.is_empty() {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires non-empty covariate values".to_string(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires finite covariate values".to_string(),
        ));
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
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires at least two unique covariate values".to_string(),
        ));
    }

    let min_x = x_unique[0];
    let max_x = *x_unique
        .last()
        .expect("x_unique has at least two elements by construction");
    if (max_x - min_x).abs() <= 1e-12 {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires non-degenerate covariate range".to_string(),
        ));
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

fn build_shape_constraint_design_1d(
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
            },
        ) => {
            let eval_spec = BSplineBasisSpec {
                degree: spec.degree,
                penalty_order: spec.penalty_order,
                knot_spec: BSplineKnotSpec::Provided(knots.clone()),
                double_penalty: false,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| BSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or(BSplineIdentifiability::None),
            };
            build_bspline_basis_1d(x_grid.view(), &eval_spec)?.design
        }
        (SmoothBasisSpec::ThinPlate { .. }, BasisMetadata::ThinPlate { centers, .. }) => {
            let eval_spec = ThinPlateBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                double_penalty: false,
                identifiability: SpatialIdentifiability::None,
            };
            build_thin_plate_basis(grid_2d.view(), &eval_spec)?.design
        }
        (
            SmoothBasisSpec::Matern { .. },
            BasisMetadata::Matern {
                centers,
                length_scale,
                nu,
                include_intercept,
                identifiability_transform,
            },
        ) => {
            let ident = identifiability_transform
                .as_ref()
                .map(|z| MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                })
                .unwrap_or(MaternIdentifiability::None);
            let eval_spec = MaternBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                nu: *nu,
                include_intercept: *include_intercept,
                double_penalty: false,
                identifiability: ident,
            };
            build_matern_basis(grid_2d.view(), &eval_spec)?.design
        }
        (
            SmoothBasisSpec::Duchon { spec, .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                nu,
                nullspace_order,
                ..
            },
        ) => {
            let eval_spec = DuchonBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                nu: *nu,
                nullspace_order: *nullspace_order,
                double_penalty: false,
                identifiability: spec.identifiability.clone(),
            };
            build_duchon_basis(grid_2d.view(), &eval_spec)?.design
        }
        _ => {
            return Err(BasisError::InvalidInput(format!(
                "shape-constraint grid reconstruction metadata mismatch for term '{}'",
                term.name
            )));
        }
    };

    Ok((x_grid, design))
}

fn build_shape_linear_constraints_1d(
    x: ArrayView1<'_, f64>,
    design_local: ArrayView2<'_, f64>,
    shape: ShapeConstraint,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let (order, sign) = match shape_order_and_sign(shape) {
        Some(v) => v,
        None => return Ok(None),
    };
    let n = x.len();
    let p = design_local.ncols();
    if n == 0 || p == 0 {
        return Ok(None);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires finite covariate values".to_string(),
        ));
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));

    let x_scale = x.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let x_tol = 1e-12 * x_scale;
    let mut collapsed_rows: Vec<Array1<f64>> = Vec::new();
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
                collapsed_rows.push(group_sum.mapv(|v| v / group_count as f64));
            }
            group_sum.fill(0.0);
            group_count = 0;
        }
        group_sum += &design_local.row(r).to_owned();
        group_count += 1;
        last_x = Some(xr);
    }
    if group_count > 0 {
        collapsed_rows.push(group_sum.mapv(|v| v / group_count as f64));
    }

    let m = collapsed_rows.len();
    if m <= order {
        return Err(BasisError::InvalidInput(format!(
            "shape-constrained smooth requires at least {} unique covariate locations; found {}",
            order + 1,
            m
        )));
    }

    let q_raw = m - order;
    let mut a_rows: Vec<Array1<f64>> = Vec::with_capacity(q_raw);
    for i in 0..q_raw {
        let row = if order == 1 {
            &collapsed_rows[i + 1] - &collapsed_rows[i]
        } else {
            &collapsed_rows[i + 2] - &collapsed_rows[i + 1].mapv(|v| 2.0 * v) + &collapsed_rows[i]
        };
        let mut row_signed = row;
        if sign < 0.0 {
            row_signed.mapv_inplace(|v| -v);
        }
        let norm = row_signed.dot(&row_signed).sqrt();
        if norm > 1e-12 {
            a_rows.push(row_signed);
        }
    }
    if a_rows.is_empty() {
        return Ok(None);
    }

    let mut a = Array2::<f64>::zeros((a_rows.len(), p));
    for (i, row) in a_rows.iter().enumerate() {
        a.row_mut(i).assign(row);
    }
    let b = Array1::<f64>::zeros(a.nrows());
    Ok(Some(LinearInequalityConstraints { a, b }))
}

fn linear_constraints_from_lower_bounds_global(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    let rows: Vec<usize> = (0..lower_bounds.len())
        .filter(|&i| lower_bounds[i].is_finite())
        .collect();
    if rows.is_empty() {
        return None;
    }
    let p = lower_bounds.len();
    let mut a = Array2::<f64>::zeros((rows.len(), p));
    let mut b = Array1::<f64>::zeros(rows.len());
    for (r, &idx) in rows.iter().enumerate() {
        a[[r, idx]] = 1.0;
        b[r] = lower_bounds[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn merge_linear_constraints_global(
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

fn build_tensor_bspline_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    spec: &TensorBSplineSpec,
) -> Result<BasisBuildResult, BasisError> {
    if feature_cols.is_empty() {
        return Err(BasisError::InvalidInput(
            "TensorBSpline requires at least one feature column".to_string(),
        ));
    }
    if feature_cols.len() != spec.marginal_specs.len() {
        return Err(BasisError::DimensionMismatch(format!(
            "TensorBSpline feature/spec mismatch: feature_cols={}, marginal_specs={}",
            feature_cols.len(),
            spec.marginal_specs.len()
        )));
    }
    let p = data.ncols();
    for &c in feature_cols {
        if c >= p {
            return Err(BasisError::DimensionMismatch(format!(
                "tensor feature column {c} is out of bounds for data with {p} columns"
            )));
        }
    }

    let mut marginal_knots = Vec::<Array1<f64>>::with_capacity(feature_cols.len());
    let mut marginal_degrees = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginal_num_basis = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginal_penalties = Vec::<Array2<f64>>::with_capacity(feature_cols.len());

    // Reuse the robust 1D builder to ensure the same knot validation and
    // marginal difference-penalty construction as standalone smooth terms.
    for (dim, (&col, marginal_spec)) in feature_cols
        .iter()
        .zip(spec.marginal_specs.iter())
        .enumerate()
    {
        // Tensor basis uses raw marginal knot-product columns. Applying 1D
        // identifiability constraints here would change marginal penalty sizes
        // without changing the tensor design construction, causing dimension
        // mismatch. Keep marginal builders unconstrained at this stage.
        let mut marginal_unconstrained = marginal_spec.clone();
        marginal_unconstrained.identifiability = BSplineIdentifiability::None;
        let built = build_bspline_basis_1d(data.column(col), &marginal_unconstrained)?;
        let knots = match built.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => {
                return Err(BasisError::InvalidInput(format!(
                    "internal TensorBSpline error at dim {dim}: expected BSpline1D metadata"
                )));
            }
        };
        marginal_knots.push(knots);
        marginal_degrees.push(marginal_spec.degree);
        marginal_num_basis.push(built.design.ncols());
        marginal_penalties.push(
            built
                .penalties
                .first()
                .ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "internal TensorBSpline error at dim {dim}: missing marginal penalty"
                    ))
                })?
                .clone(),
        );
        let _ = built.nullspace_dims.first().ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "internal TensorBSpline error at dim {dim}: missing marginal nullspace dim"
            ))
        })?;
    }

    let data_views: Vec<_> = feature_cols.iter().map(|&c| data.column(c)).collect();
    let knot_views: Vec<_> = marginal_knots.iter().map(|k| k.view()).collect();
    let (basis, _) =
        create_bspline_basis_nd_with_knots(&data_views, &knot_views, &marginal_degrees)?;
    let mut design = (*basis).clone();

    let total_cols = design.ncols();
    let mut penalties = Vec::<Array2<f64>>::with_capacity(
        marginal_penalties.len() + if spec.double_penalty { 1 } else { 0 },
    );

    for dim in 0..marginal_penalties.len() {
        let mut s_dim = Array2::<f64>::eye(1);
        for (j, &qj) in marginal_num_basis.iter().enumerate() {
            let factor = if j == dim {
                marginal_penalties[j].clone()
            } else {
                Array2::<f64>::eye(qj)
            };
            s_dim = kronecker_product(&s_dim, &factor);
        }

        penalties.push(s_dim);
    }

    if spec.double_penalty {
        penalties.push(Array2::<f64>::eye(total_cols));
    }

    let z_opt = match &spec.identifiability {
        TensorBSplineIdentifiability::None => None,
        TensorBSplineIdentifiability::SumToZero => {
            if total_cols < 2 {
                return Err(BasisError::InvalidInput(
                    "TensorBSpline requires at least 2 basis coefficients to enforce sum-to-zero identifiability".to_string(),
                ));
            }
            let (_design_constrained, z) = apply_sum_to_zero_constraint(design.view(), None)?;
            Some(z)
        }
        TensorBSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != total_cols {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen tensor identifiability transform mismatch: design has {} columns but transform has {} rows",
                    total_cols,
                    transform.nrows()
                )));
            }
            Some(transform.clone())
        }
    };

    if let Some(z) = z_opt.as_ref() {
        design = design.dot(z);
        penalties = penalties
            .into_iter()
            .map(|s| {
                let zt_s = z.t().dot(&s);
                zt_s.dot(z)
            })
            .collect();
    }

    let nullspace_dims = penalties
        .iter()
        .map(estimate_penalty_nullity)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        metadata: BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.to_vec(),
            knots: marginal_knots,
            degrees: marginal_degrees,
            identifiability_transform: z_opt,
        },
    })
}

fn build_random_effect_block(
    data: ArrayView2<'_, f64>,
    spec: &RandomEffectTermSpec,
) -> Result<RandomEffectBlock, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    if spec.feature_col >= p {
        return Err(BasisError::DimensionMismatch(format!(
            "random-effect term '{}' feature column {} out of bounds for {} columns",
            spec.name, spec.feature_col, p
        )));
    }

    let col = data.column(spec.feature_col);
    if col.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(format!(
            "random-effect term '{}' contains non-finite group values",
            spec.name
        )));
    }

    let mut kept_levels: Vec<u64> = if let Some(levels) = spec.frozen_levels.as_ref() {
        if levels.is_empty() {
            return Err(BasisError::InvalidInput(format!(
                "random-effect term '{}' has empty frozen_levels",
                spec.name
            )));
        }
        levels.clone()
    } else {
        let mut levels_set = BTreeSet::<u64>::new();
        for &v in col {
            levels_set.insert(v.to_bits());
        }
        if levels_set.is_empty() {
            return Err(BasisError::InvalidInput(format!(
                "random-effect term '{}' has no observed levels",
                spec.name
            )));
        }
        let levels: Vec<u64> = levels_set.into_iter().collect();
        let start_idx = if spec.drop_first_level && levels.len() > 1 {
            1usize
        } else {
            0usize
        };
        levels[start_idx..].to_vec()
    };
    kept_levels.sort_unstable();
    kept_levels.dedup();

    if kept_levels.is_empty() {
        return Err(BasisError::InvalidInput(format!(
            "random-effect term '{}' drops all levels; keep at least one level",
            spec.name
        )));
    }

    let q = kept_levels.len();
    let mut design = Array2::<f64>::zeros((n, q));
    for (i, &v) in col.iter().enumerate() {
        let bits = v.to_bits();
        let pos = kept_levels.binary_search(&bits).ok();
        if let Some(j) = pos {
            design[[i, j]] = 1.0;
        }
    }

    Ok(RandomEffectBlock {
        name: spec.name.clone(),
        design,
        kept_levels,
    })
}

impl SmoothDesign {
    /// Map an unconstrained term coefficient vector to its constrained shape space.
    /// This is useful for nonlinear fits that optimize unconstrained parameters.
    pub fn map_term_coefficients(
        unconstrained: &Array1<f64>,
        shape: ShapeConstraint,
    ) -> Result<Array1<f64>, BasisError> {
        if unconstrained.is_empty() {
            return Err(BasisError::InvalidInput(
                "unconstrained coefficient vector cannot be empty".to_string(),
            ));
        }
        let mapped = match shape {
            ShapeConstraint::None => unconstrained.clone(),
            ShapeConstraint::MonotoneIncreasing => cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::MonotoneDecreasing => cumulative_exp(unconstrained, -1.0),
            ShapeConstraint::Convex => second_cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::Concave => second_cumulative_exp(unconstrained, -1.0),
        };
        Ok(mapped)
    }
}

pub fn build_smooth_design(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<SmoothDesign, BasisError> {
    let n = data.nrows();
    let mut local_designs = Vec::<Array2<f64>>::with_capacity(terms.len());
    let mut local_penalties = Vec::<Vec<Array2<f64>>>::with_capacity(terms.len());
    let mut local_nullspaces = Vec::<Vec<usize>>::with_capacity(terms.len());
    let mut local_metadata = Vec::<BasisMetadata>::with_capacity(terms.len());
    let mut local_dims = Vec::<usize>::with_capacity(terms.len());
    let mut local_linear_constraints =
        Vec::<Option<LinearInequalityConstraints>>::with_capacity(terms.len());
    let mut local_box_reparam = Vec::<bool>::with_capacity(terms.len());

    for term in terms {
        if !shape_supports_basis(term) {
            return Err(BasisError::InvalidInput(format!(
                "ShapeConstraint::{:?} is unsupported for term '{}'",
                term.shape, term.name
            )));
        }
        let mut shape_axis_col: Option<usize> = None;
        let built: BasisBuildResult = match &term.basis {
            SmoothBasisSpec::BSpline1D { feature_col, spec } => {
                if *feature_col >= data.ncols() {
                    return Err(BasisError::DimensionMismatch(format!(
                        "term '{}' feature column {} out of bounds for {} columns",
                        term.name,
                        feature_col,
                        data.ncols()
                    )));
                }
                let mut spec_local = spec.clone();
                if term.shape != ShapeConstraint::None {
                    // Shape-constrained B-splines are anchored by construction.
                    // Sum-to-zero side constraints conflict with monotonic/convex cones.
                    spec_local.identifiability = BSplineIdentifiability::None;
                }
                build_bspline_basis_1d(data.column(*feature_col), &spec_local)?
            }
            SmoothBasisSpec::ThinPlate { feature_cols, spec } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on ThinPlate basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let x = select_columns(data, feature_cols)?;
                build_thin_plate_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Matern { feature_cols, spec } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on Matern basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let x = select_columns(data, feature_cols)?;
                build_matern_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Duchon { feature_cols, spec } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on Duchon basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let x = select_columns(data, feature_cols)?;
                build_duchon_basis(x.view(), spec)?
            }
            SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
                let mut spec_local = spec.clone();
                if term.shape != ShapeConstraint::None {
                    spec_local.identifiability = TensorBSplineIdentifiability::None;
                }
                build_tensor_bspline_basis(data, feature_cols, &spec_local)?
            }
        };

        let p_local = built.design.ncols();
        let metadata = built.metadata.clone();
        let mut design_t = built.design;
        let mut penalties_t: Vec<Array2<f64>> = built.penalties;
        let use_box_reparam =
            term.shape != ShapeConstraint::None && shape_uses_box_reparameterization(&term.basis);
        if let Some((order, sign)) = shape_order_and_sign(term.shape)
            && use_box_reparam
        {
            let t = cumulative_sum_transform_matrix(p_local, order, sign);
            design_t = design_t.dot(&t);
            penalties_t = penalties_t
                .into_iter()
                .map(|s_local| {
                    // Congruence transform preserves PSD:
                    //   S_new = T^T S T.
                    let tt_s = t.t().dot(&s_local);
                    tt_s.dot(&t)
                })
                .collect();
        }
        let linear_constraints_local = if term.shape != ShapeConstraint::None && !use_box_reparam {
            let axis = shape_axis_col.ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "internal shape-constraint axis missing for term '{}'",
                    term.name
                ))
            })?;
            let (x_shape_eval, design_shape_eval) =
                build_shape_constraint_design_1d(data, term, &metadata, axis)?;
            build_shape_linear_constraints_1d(
                x_shape_eval.view(),
                design_shape_eval.view(),
                term.shape,
            )?
        } else {
            None
        };

        let nullspaces_t = penalties_t
            .iter()
            .map(estimate_penalty_nullity)
            .collect::<Result<Vec<_>, _>>()?;

        local_dims.push(p_local);
        local_designs.push(design_t);
        local_penalties.push(penalties_t);
        local_nullspaces.push(nullspaces_t);
        local_metadata.push(metadata);
        local_linear_constraints.push(linear_constraints_local);
        local_box_reparam.push(use_box_reparam);
    }

    let total_p: usize = local_dims.iter().sum();
    let mut design = Array2::<f64>::zeros((n, total_p));
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraints_rows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for (idx, term) in terms.iter().enumerate() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;
        let lb_local = if local_box_reparam[idx] {
            shape_lower_bounds_local(term.shape, p_local)
        } else {
            None
        };

        design
            .slice_mut(s![.., col_start..col_end])
            .assign(&local_designs[idx]);

        for (s_local, &ns) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
        {
            let mut s_global = Array2::<f64>::zeros((total_p, total_p));
            s_global
                .slice_mut(s![col_start..col_end, col_start..col_end])
                .assign(s_local);
            penalties_global.push(s_global);
            nullspace_dims_global.push(ns);
        }

        terms_out.push(SmoothTerm {
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            metadata: local_metadata[idx].clone(),
            lower_bounds_local: lb_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraints_rows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = lb_local {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(&lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    Ok(SmoothDesign {
        design,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraints_rows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraints_rows.len(), total_p));
            for (i, row) in linear_constraints_rows.iter().enumerate() {
                a.row_mut(i).assign(row);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();
    let smooth_raw = build_smooth_design(data, &spec.smooth_terms)?;
    let random_blocks: Vec<RandomEffectBlock> = spec
        .random_effect_terms
        .iter()
        .map(|term| build_random_effect_block(data, term))
        .collect::<Result<_, _>>()?;

    for linear in &spec.linear_terms {
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
    }

    let smooth = apply_spatial_orthogonality_to_parametric(
        smooth_raw,
        data,
        &spec.linear_terms,
        &spec.smooth_terms,
    )?;

    let p_intercept = 1usize;
    let p_lin = spec.linear_terms.len();
    let p_rand: usize = random_blocks.iter().map(|b| b.design.ncols()).sum();
    let p_smooth = smooth.design.ncols();
    let p_total = p_intercept + p_lin + p_rand + p_smooth;
    let mut design = Array2::<f64>::zeros((n, p_total));
    design.column_mut(0).fill(1.0);

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        design
            .column_mut(col)
            .assign(&data.column(linear.feature_col));
        linear_ranges.push((linear.name.clone(), col..(col + 1)));
    }
    let mut random_effect_ranges =
        Vec::<(String, Range<usize>)>::with_capacity(random_blocks.len());
    let mut random_effect_levels = Vec::<(String, Vec<u64>)>::with_capacity(random_blocks.len());
    let mut col_cursor = p_intercept + p_lin;
    for block in &random_blocks {
        let q = block.design.ncols();
        let end = col_cursor + q;
        design
            .slice_mut(s![.., col_cursor..end])
            .assign(&block.design);
        random_effect_ranges.push((block.name.clone(), col_cursor..end));
        random_effect_levels.push((block.name.clone(), block.kept_levels.clone()));
        col_cursor = end;
    }
    if p_smooth > 0 {
        design
            .slice_mut(s![.., (p_intercept + p_lin + p_rand)..])
            .assign(&smooth.design);
    }

    let mut penalties = Vec::<Array2<f64>>::new();
    let mut nullspace_dims = Vec::<usize>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraints = None;

    for (j, linear) in spec.linear_terms.iter().enumerate() {
        if !linear.double_penalty {
            continue;
        }
        let col = p_intercept + j;
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        s[[col, col]] = 1.0;
        penalties.push(s);
        nullspace_dims.push(0);
    }

    for (_name, range) in &random_effect_ranges {
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        for j in range.clone() {
            s[[j, j]] = 1.0;
        }
        penalties.push(s);
        nullspace_dims.push(0);
    }

    for (s_local, &ns) in smooth.penalties.iter().zip(smooth.nullspace_dims.iter()) {
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        let start = p_intercept + p_lin + p_rand;
        s.slice_mut(s![start..(start + p_smooth), start..(start + p_smooth)])
            .assign(s_local);
        penalties.push(s);
        nullspace_dims.push(ns);
    }

    if let Some(lb_smooth) = smooth.coefficient_lower_bounds.as_ref() {
        let start = p_intercept + p_lin + p_rand;
        coefficient_lower_bounds
            .slice_mut(s![start..(start + p_smooth)])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = smooth.linear_constraints.as_ref() {
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        let start = p_intercept + p_lin + p_rand;
        a_global
            .slice_mut(s![.., start..(start + p_smooth)])
            .assign(&lin_smooth.a);
        linear_constraints = Some(LinearInequalityConstraints {
            a: a_global,
            b: lin_smooth.b.clone(),
        });
    }

    // Canonical constraint path: convert any explicit lower bounds into linear
    // inequalities and merge into the global constraint matrix. This keeps fitting
    // behavior independent of user-facing lower-bound options.
    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    linear_constraints =
        merge_linear_constraints_global(linear_constraints, lower_bound_constraints);

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints,
        intercept_range: 0..1,
        linear_ranges,
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
}

fn apply_spatial_orthogonality_to_parametric(
    smooth: SmoothDesign,
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    smooth_specs: &[SmoothTermSpec],
) -> Result<SmoothDesign, BasisError> {
    // Option 5 identifiability policy:
    //
    // Build a term-local parametric confounding block C_j = [1 | X_lin,overlap(j)],
    // then for each spatial smooth basis B_j enforce orthogonality to C_j in the
    // unweighted inner product:
    //   B_con^T C = 0.
    //
    // Reparameterization derivation:
    //   M = B^T C.
    //   If columns of Z span null(M^T), then
    //      (B Z)^T C = Z^T (B^T C) = Z^T M = 0.
    //
    // So B_con = B Z has no component in the parametric column space, eliminating
    // intercept/linear confounding without hand-picking polynomial columns.
    //
    // Penalties transform by congruence:
    //   S_con = Z^T S Z.
    // This preserves PSD and keeps curvature geometry consistent in constrained coords.
    if smooth_specs.len() != smooth.terms.len() {
        return Err(BasisError::DimensionMismatch(format!(
            "smooth spec count ({}) does not match built term count ({})",
            smooth_specs.len(),
            smooth.terms.len()
        )));
    }

    // Fast-path: if no spatial term participates in Option 5 (orthogonal or frozen),
    // return the smooth bundle unchanged and skip all matrix work.
    let any_spatial_transform = smooth_specs.iter().any(|t| {
        !matches!(
            spatial_identifiability_policy(t),
            Some(SpatialIdentifiability::None) | None
        )
    });
    if !any_spatial_transform {
        return Ok(smooth);
    }

    let n = smooth.design.nrows();
    let mut local_designs = Vec::<Array2<f64>>::with_capacity(smooth.terms.len());
    let mut local_penalties = Vec::<Vec<Array2<f64>>>::with_capacity(smooth.terms.len());
    let mut local_nullspaces = Vec::<Vec<usize>>::with_capacity(smooth.terms.len());
    let mut local_metadata = Vec::<BasisMetadata>::with_capacity(smooth.terms.len());
    let mut local_dims = Vec::<usize>::with_capacity(smooth.terms.len());
    let mut local_linear_constraints =
        Vec::<Option<LinearInequalityConstraints>>::with_capacity(smooth.terms.len());

    for (idx, term) in smooth.terms.iter().enumerate() {
        let term_spec = &smooth_specs[idx];
        let design_local = smooth
            .design
            .slice(s![.., term.coeff_range.clone()])
            .to_owned();
        let c_local = if matches!(
            spatial_identifiability_policy(term_spec),
            Some(SpatialIdentifiability::OrthogonalToParametric)
        ) {
            Some(build_parametric_constraint_block_for_term(
                data,
                linear_terms,
                term_spec,
            )?)
        } else {
            None
        };
        let (design_constrained, z_opt) = maybe_spatial_identifiability_transform(
            term_spec,
            design_local.view(),
            c_local.as_ref().map(|mat| mat.view()),
        )?;

        // Mathematical acceptance criterion:
        //   ||B_con^T C||_F / (||B_con||_F ||C||_F) <= tol.
        if matches!(
            spatial_identifiability_policy(term_spec),
            Some(SpatialIdentifiability::OrthogonalToParametric)
        ) {
            let c_ref = c_local
                .as_ref()
                .expect("parametric constraint block must exist for orthogonal policy");
            let rel = orthogonality_relative_residual(design_constrained.view(), c_ref.view());
            let tol = 1e-8;
            if rel > tol {
                return Err(BasisError::InvalidInput(format!(
                    "spatial orthogonality residual too large for term '{}': {:.3e} > {:.1e}",
                    term.name, rel, tol
                )));
            }
        }

        let mut penalties_constrained =
            Vec::<Array2<f64>>::with_capacity(term.penalties_local.len());
        let mut nullspace_constrained = Vec::<usize>::with_capacity(term.penalties_local.len());
        for s_local in &term.penalties_local {
            let s_con = if let Some(z) = z_opt.as_ref() {
                let zt_s = z.t().dot(s_local);
                zt_s.dot(z)
            } else {
                s_local.clone()
            };
            let ns = estimate_penalty_nullity(&s_con)?;
            penalties_constrained.push(s_con);
            nullspace_constrained.push(ns);
        }
        let linear_constraints_constrained =
            if let Some(lin_local) = term.linear_constraints_local.as_ref() {
                if let Some(z) = z_opt.as_ref() {
                    Some(LinearInequalityConstraints {
                        a: lin_local.a.dot(z),
                        b: lin_local.b.clone(),
                    })
                } else {
                    Some(lin_local.clone())
                }
            } else {
                None
            };

        local_dims.push(design_constrained.ncols());
        local_designs.push(design_constrained);
        local_penalties.push(penalties_constrained);
        local_nullspaces.push(nullspace_constrained);
        local_linear_constraints.push(linear_constraints_constrained);
        local_metadata.push(with_spatial_identifiability_transform(
            &term.metadata,
            z_opt.as_ref(),
        ));
    }

    let total_p: usize = local_dims.iter().sum();
    let mut design = Array2::<f64>::zeros((n, total_p));
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(smooth.terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraints_rows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for idx in 0..smooth.terms.len() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;
        design
            .slice_mut(s![.., col_start..col_end])
            .assign(&local_designs[idx]);

        for (s_local, &ns) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
        {
            let mut s_global = Array2::<f64>::zeros((total_p, total_p));
            s_global
                .slice_mut(s![col_start..col_end, col_start..col_end])
                .assign(s_local);
            penalties_global.push(s_global);
            nullspace_dims_global.push(ns);
        }

        terms_out.push(SmoothTerm {
            name: smooth.terms[idx].name.clone(),
            coeff_range: col_start..col_end,
            shape: smooth.terms[idx].shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            metadata: local_metadata[idx].clone(),
            lower_bounds_local: smooth.terms[idx].lower_bounds_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraints_rows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = smooth.terms[idx].lower_bounds_local.as_ref()
            && lb_local.len() == p_local
        {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    Ok(SmoothDesign {
        design,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraints_rows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraints_rows.len(), total_p));
            for (i, row) in linear_constraints_rows.iter().enumerate() {
                a.row_mut(i).assign(row);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}

fn build_parametric_constraint_block_for_term(
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    term_spec: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();

    let overlapping_linear_term_indices: Vec<usize> = match &term_spec.basis {
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => linear_terms
            .iter()
            .enumerate()
            .filter_map(|(idx, linear)| {
                if feature_cols.contains(&linear.feature_col) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
        SmoothBasisSpec::Duchon { feature_cols, .. } => linear_terms
            .iter()
            .enumerate()
            .filter_map(|(idx, linear)| {
                if feature_cols.contains(&linear.feature_col) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    };

    let mut c = Array2::<f64>::zeros((n, 1 + overlapping_linear_term_indices.len()));
    c.column_mut(0).fill(1.0);
    for (j, &lin_idx) in overlapping_linear_term_indices.iter().enumerate() {
        let linear = &linear_terms[lin_idx];
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
        c.column_mut(j + 1).assign(&data.column(linear.feature_col));
    }
    Ok(c)
}

fn maybe_spatial_identifiability_transform(
    term_spec: &SmoothTermSpec,
    design_local: ArrayView2<'_, f64>,
    parametric_block: Option<ArrayView2<'_, f64>>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let maybe_policy = spatial_identifiability_policy(term_spec);
    let Some(policy) = maybe_policy else {
        return Ok((design_local.to_owned(), None));
    };

    match policy {
        SpatialIdentifiability::None => Ok((design_local.to_owned(), None)),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = parametric_block.ok_or_else(|| {
                BasisError::InvalidInput(
                    "missing parametric constraint block for OrthogonalToParametric policy"
                        .to_string(),
                )
            })?;
            let (b_c, z) = apply_weighted_orthogonality_constraint(
                design_local,
                c,
                None, // fixed subspace: do not use iteration-varying PIRLS weights
            )?;
            Ok((b_c, Some(z)))
        }
        SpatialIdentifiability::FrozenTransform { transform } => {
            if design_local.ncols() != transform.nrows() {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen spatial identifiability transform mismatch: design has {} columns but transform has {} rows",
                    design_local.ncols(),
                    transform.nrows()
                )));
            }
            let z = transform.clone();
            Ok((design_local.dot(&z), Some(z)))
        }
    }
}

fn spatial_identifiability_policy(term_spec: &SmoothTermSpec) -> Option<&SpatialIdentifiability> {
    match &term_spec.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.identifiability),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.identifiability),
        _ => None,
    }
}

fn orthogonality_relative_residual(
    basis_matrix: ArrayView2<'_, f64>,
    constraint_matrix: ArrayView2<'_, f64>,
) -> f64 {
    let cross = basis_matrix.t().dot(&constraint_matrix);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = basis_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (b_norm * c_norm).max(1e-300);
    num / denom
}

fn with_spatial_identifiability_transform(
    metadata: &BasisMetadata,
    transform: Option<&Array2<f64>>,
) -> BasisMetadata {
    match metadata {
        BasisMetadata::ThinPlate { centers, .. } => BasisMetadata::ThinPlate {
            centers: centers.clone(),
            identifiability_transform: transform.cloned(),
        },
        BasisMetadata::Duchon {
            centers,
            length_scale,
            nu,
            nullspace_order,
            ..
        } => BasisMetadata::Duchon {
            centers: centers.clone(),
            length_scale: *length_scale,
            nu: *nu,
            nullspace_order: *nullspace_order,
            identifiability_transform: transform.cloned(),
        },
        _ => metadata.clone(),
    }
}

pub fn fit_term_collection(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let out = fit_term_collection_with_matern_kappa_optimization(
        data,
        y,
        weights,
        offset,
        spec,
        family,
        options,
        &MaternKappaOptimizationOptions::default(),
    )?;
    Ok(FittedTermCollection {
        fit: out.fit,
        design: out.design,
    })
}

fn fit_term_collection_for_spec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let design = build_term_collection_design(data, spec)?;
    let fit = fit_gam(
        design.design.view(),
        y,
        weights,
        offset,
        &design.penalties,
        family,
        &FitOptions {
            max_iter: options.max_iter,
            tol: options.tol,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
        },
    )?;
    enforce_term_constraint_feasibility(&design, &fit)?;
    Ok(FittedTermCollection { fit, design })
}

fn enforce_term_constraint_feasibility(
    design: &TermCollectionDesign,
    fit: &FitResult,
) -> Result<(), EstimationError> {
    let tol = 1e-7;
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.design.ncols());
    let mut violations: Vec<String> = Vec::new();
    for term in &design.smooth.terms {
        let gr = (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let beta_local = fit.beta.slice(s![gr.clone()]).to_owned();
        if let Some(lb) = term.lower_bounds_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worst_idx = 0usize;
            for i in 0..lb.len().min(beta_local.len()) {
                if lb[i].is_finite() {
                    let viol = (lb[i] - beta_local[i]).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_idx = i;
                    }
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=lower-bound max_violation={:.3e} coeff_index={}",
                    term.name, worst, worst_idx
                ));
            }
        }
        if let Some(lin) = term.linear_constraints_local.as_ref() {
            let slack = lin.a.dot(&beta_local) - &lin.b;
            let mut worst = 0.0_f64;
            let mut worst_row = 0usize;
            for (i, &v) in slack.iter().enumerate() {
                let viol = (-v).max(0.0);
                if viol > worst {
                    worst = viol;
                    worst_row = i;
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=linear-inequality max_violation={:.3e} row={}",
                    term.name, worst, worst_row
                ));
            }
        }
    }

    if !violations.is_empty() {
        let mut msg = format!(
            "constraint violation after fit ({} violating term constraints): {}",
            violations.len(),
            violations.join(" | ")
        );
        if let Some(kkt) = fit.artifacts.pirls.constraint_kkt.as_ref() {
            msg.push_str(&format!(
                "; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}]",
                kkt.primal_feasibility, kkt.dual_feasibility, kkt.complementarity, kkt.stationarity
            ));
        }
        return Err(EstimationError::ParameterConstraintViolation(msg));
    }
    Ok(())
}

fn matern_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    spec.smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, term)| match term.basis {
            SmoothBasisSpec::Matern { .. } => Some(idx),
            _ => None,
        })
        .collect()
}

fn fit_score(fit: &FitResult) -> f64 {
    let pirls = &fit.artifacts.pirls;
    // Use the penalized objective from the converged inner fit as a practical
    // comparison score across κ candidates. Each candidate has its own B and S,
    // so we compare fully refit objectives rather than reusing lambda estimates.
    let score = 0.5 * pirls.deviance + 0.5 * pirls.stable_penalty_term;
    if score.is_finite() {
        score
    } else {
        f64::INFINITY
    }
}

fn set_matern_length_scale(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    length_scale: f64,
) -> Result<(), EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        return Err(EstimationError::InvalidInput(format!(
            "matérn term index {term_idx} out of range"
        )));
    };
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' is not Matérn",
            term.name
        ))),
    }
}

fn get_matern_length_scale(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => Some(spec.length_scale),
            _ => None,
        })
}

fn get_matern_double_penalty(spec: &TermCollectionSpec, term_idx: usize) -> Option<bool> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => Some(spec.double_penalty),
            _ => None,
        })
}

pub fn optimize_two_block_matern_kappa<FitOut, FitFn, ScoreFn>(
    data: ArrayView2<'_, f64>,
    mean_spec: &TermCollectionSpec,
    noise_spec: &TermCollectionSpec,
    kappa_options: &MaternKappaOptimizationOptions,
    mut fit_fn: FitFn,
    score_fn: ScoreFn,
) -> Result<TwoBlockMaternKappaOptimizationResult<FitOut>, String>
where
    FitFn: FnMut(&TermCollectionDesign, &TermCollectionDesign) -> Result<FitOut, String>,
    ScoreFn: Fn(&FitOut) -> f64,
{
    // For location-scale models, κ (Matérn length_scale) is block-specific.
    // We optimize κ for mean/noise blocks separately, while each candidate
    // evaluation re-runs fitting so λ/δ are re-optimized.
    let mut best_mean_spec = mean_spec.clone();
    let mut best_noise_spec = noise_spec.clone();
    let mean_terms = matern_term_indices(&best_mean_spec);
    let noise_terms = matern_term_indices(&best_noise_spec);

    let build_pair = |ms: &TermCollectionSpec,
                      ns: &TermCollectionSpec|
     -> Result<(TermCollectionDesign, TermCollectionDesign), String> {
        let d_mean = build_term_collection_design(data, ms)
            .map_err(|e| format!("failed to build mean design during κ optimization: {e}"))?;
        let d_noise = build_term_collection_design(data, ns)
            .map_err(|e| format!("failed to build noise design during κ optimization: {e}"))?;
        Ok((d_mean, d_noise))
    };

    let (mut best_mean_design, mut best_noise_design) =
        build_pair(&best_mean_spec, &best_noise_spec)?;
    let mut best_fit = fit_fn(&best_mean_design, &best_noise_design)?;
    let mut best_score = score_fn(&best_fit);
    if !best_score.is_finite() {
        best_score = f64::INFINITY;
    }

    if !kappa_options.enabled || (mean_terms.is_empty() && noise_terms.is_empty()) {
        return Ok(TwoBlockMaternKappaOptimizationResult {
            resolved_mean_spec: best_mean_spec,
            resolved_noise_spec: best_noise_spec,
            mean_design: best_mean_design,
            noise_design: best_noise_design,
            fit: best_fit,
        });
    }
    if kappa_options.max_outer_iter == 0 {
        return Err("Matérn κ optimization requires max_outer_iter >= 1".to_string());
    }
    if !(kappa_options.log_step.is_finite() && kappa_options.log_step > 0.0) {
        return Err("Matérn κ optimization requires log_step > 0".to_string());
    }
    if !(kappa_options.min_length_scale.is_finite()
        && kappa_options.max_length_scale.is_finite()
        && kappa_options.min_length_scale > 0.0
        && kappa_options.max_length_scale >= kappa_options.min_length_scale)
    {
        return Err(
            "Matérn κ optimization requires valid positive length_scale bounds".to_string(),
        );
    }

    let rel_tol = kappa_options.rel_tol.max(0.0);
    let mut blocks: Vec<(bool, usize)> = Vec::new();
    for idx in &mean_terms {
        blocks.push((true, *idx));
    }
    for idx in &noise_terms {
        blocks.push((false, *idx));
    }

    for _outer in 0..kappa_options.max_outer_iter {
        let mut any_improvement = false;
        for (is_mean_block, term_idx) in &blocks {
            let spec_ref = if *is_mean_block {
                &best_mean_spec
            } else {
                &best_noise_spec
            };
            let Some(current_ls) = get_matern_length_scale(spec_ref, *term_idx) else {
                continue;
            };
            let current_ls = current_ls.clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
            let double_penalty = get_matern_double_penalty(spec_ref, *term_idx).unwrap_or(false);
            let step = if double_penalty {
                kappa_options.log_step
            } else {
                0.5 * kappa_options.log_step
            };
            let log0 = current_ls.ln();
            let mut candidates = if double_penalty {
                vec![
                    log0 - 2.0 * step,
                    log0 - step,
                    log0,
                    log0 + step,
                    log0 + 2.0 * step,
                ]
            } else {
                vec![log0 - step, log0, log0 + step]
            };
            candidates.sort_by(|a, b| a.total_cmp(b));
            candidates.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

            let term_rel_tol = if double_penalty {
                rel_tol
            } else {
                rel_tol.max(5e-4)
            };
            let mut local_best_score = best_score;
            let mut local_best_fit: Option<FitOut> = None;
            let mut local_best_mean_spec: Option<TermCollectionSpec> = None;
            let mut local_best_noise_spec: Option<TermCollectionSpec> = None;
            let mut local_best_mean_design: Option<TermCollectionDesign> = None;
            let mut local_best_noise_design: Option<TermCollectionDesign> = None;

            for cand_log in candidates {
                let cand_ls = cand_log.exp().clamp(
                    kappa_options.min_length_scale,
                    kappa_options.max_length_scale,
                );
                if (cand_ls - current_ls).abs() <= 1e-15 {
                    continue;
                }
                let mut cand_mean_spec = best_mean_spec.clone();
                let mut cand_noise_spec = best_noise_spec.clone();
                if *is_mean_block {
                    set_matern_length_scale(&mut cand_mean_spec, *term_idx, cand_ls)
                        .map_err(|e| e.to_string())?;
                } else {
                    set_matern_length_scale(&mut cand_noise_spec, *term_idx, cand_ls)
                        .map_err(|e| e.to_string())?;
                }
                let (cand_mean_design, cand_noise_design) =
                    build_pair(&cand_mean_spec, &cand_noise_spec)?;
                let cand_fit = match fit_fn(&cand_mean_design, &cand_noise_design) {
                    Ok(v) => v,
                    Err(err) => {
                        log::warn!(
                            "[location-scale][Matern-kappa] block={} term={} length_scale={:.6e} fit failed: {}",
                            if *is_mean_block { "mean" } else { "noise" },
                            term_idx,
                            cand_ls,
                            err
                        );
                        continue;
                    }
                };
                let cand_score = score_fn(&cand_fit);
                if cand_score + term_rel_tol * local_best_score.abs().max(1.0) < local_best_score {
                    local_best_score = cand_score;
                    local_best_fit = Some(cand_fit);
                    local_best_mean_spec = Some(cand_mean_spec);
                    local_best_noise_spec = Some(cand_noise_spec);
                    local_best_mean_design = Some(cand_mean_design);
                    local_best_noise_design = Some(cand_noise_design);
                }
            }

            if let (
                Some(next_fit),
                Some(next_mean_spec),
                Some(next_noise_spec),
                Some(next_mean_design),
                Some(next_noise_design),
            ) = (
                local_best_fit,
                local_best_mean_spec,
                local_best_noise_spec,
                local_best_mean_design,
                local_best_noise_design,
            ) {
                best_fit = next_fit;
                best_score = local_best_score;
                best_mean_spec = next_mean_spec;
                best_noise_spec = next_noise_spec;
                best_mean_design = next_mean_design;
                best_noise_design = next_noise_design;
                any_improvement = true;
            }
        }
        if !any_improvement {
            break;
        }
    }

    Ok(TwoBlockMaternKappaOptimizationResult {
        resolved_mean_spec: best_mean_spec,
        resolved_noise_spec: best_noise_spec,
        mean_design: best_mean_design,
        noise_design: best_noise_design,
        fit: best_fit,
    })
}

pub fn fit_term_collection_with_matern_kappa_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
    kappa_options: &MaternKappaOptimizationOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    // κ (= 1/length_scale) changes kernel geometry nonlinearly.
    // That means both basis values B and penalty blocks S change, so each κ
    // proposal requires a full basis rebuild and a fresh lambda optimization.
    let mut resolved_spec = spec.clone();
    let matern_terms = matern_term_indices(&resolved_spec);
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        return Err(EstimationError::InvalidInput(format!(
            "fit_term_collection_with_matern_kappa_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        )));
    }
    if !kappa_options.enabled || matern_terms.is_empty() {
        let out = fit_term_collection_for_spec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolved_spec,
            family,
            options,
        )?;
        return Ok(FittedTermCollectionWithSpec {
            fit: out.fit,
            design: out.design,
            resolved_spec,
        });
    }
    if kappa_options.max_outer_iter == 0 {
        return Err(EstimationError::InvalidInput(
            "Matern kappa optimization requires max_outer_iter >= 1".to_string(),
        ));
    }
    if !(kappa_options.log_step.is_finite() && kappa_options.log_step > 0.0) {
        return Err(EstimationError::InvalidInput(
            "Matern kappa optimization requires log_step > 0".to_string(),
        ));
    }
    if !(kappa_options.min_length_scale.is_finite()
        && kappa_options.max_length_scale.is_finite()
        && kappa_options.min_length_scale > 0.0
        && kappa_options.max_length_scale >= kappa_options.min_length_scale)
    {
        return Err(EstimationError::InvalidInput(
            "Matern kappa optimization requires valid positive length_scale bounds".to_string(),
        ));
    }

    let mut best = fit_term_collection_for_spec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolved_spec,
        family,
        options,
    )?;
    let mut best_score = fit_score(&best.fit);
    if !best_score.is_finite() {
        best_score = f64::INFINITY;
    }
    let rel_tol = kappa_options.rel_tol.max(0.0);

    for outer in 0..kappa_options.max_outer_iter {
        let mut any_improvement = false;
        for &term_idx in &matern_terms {
            let Some(current_ls) = get_matern_length_scale(&resolved_spec, term_idx) else {
                continue;
            };
            let double_penalty =
                get_matern_double_penalty(&resolved_spec, term_idx).unwrap_or(false);
            let current_ls = current_ls.clamp(
                kappa_options.min_length_scale,
                kappa_options.max_length_scale,
            );
            let log0 = current_ls.ln();
            // λ and κ are partially confounded for single-penalty Matérn terms.
            // With double-penalty enabled we gain an extra shrinkage degree of freedom,
            // so we can search κ more aggressively (wider local bracket).
            let step = if double_penalty {
                kappa_options.log_step
            } else {
                0.5 * kappa_options.log_step
            };
            let mut candidates = if double_penalty {
                vec![
                    log0 - 2.0 * step,
                    log0 - step,
                    log0,
                    log0 + step,
                    log0 + 2.0 * step,
                ]
            } else {
                vec![log0 - step, log0, log0 + step]
            };
            candidates.sort_by(|a, b| a.total_cmp(b));
            candidates.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

            let mut local_best_score = best_score;
            let mut local_best_fit: Option<FittedTermCollection> = None;
            let mut local_best_spec: Option<TermCollectionSpec> = None;
            let term_rel_tol = if double_penalty {
                rel_tol
            } else {
                // Require a clearer gain when only one penalty knob is available.
                rel_tol.max(5e-4)
            };
            // Coordinate update on one Matérn term at a time in log(length_scale)
            // to keep search stable under λ/κ partial confounding.
            for cand_log in candidates {
                let cand_ls = cand_log.exp().clamp(
                    kappa_options.min_length_scale,
                    kappa_options.max_length_scale,
                );
                if (cand_ls - current_ls).abs() <= 1e-15 {
                    continue;
                }
                let mut cand_spec = resolved_spec.clone();
                set_matern_length_scale(&mut cand_spec, term_idx, cand_ls)?;
                // Full refit at candidate κ: rebuild design/penalties, then run
                // standard REML/LAML outer optimization for λ on that basis.
                let cand_fit = match fit_term_collection_for_spec(
                    data,
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &cand_spec,
                    family,
                    options,
                ) {
                    Ok(v) => v,
                    Err(err) => {
                        log::warn!(
                            "[Matern-kappa] term={} length_scale={:.6e} fit failed: {}",
                            term_idx,
                            cand_ls,
                            err
                        );
                        continue;
                    }
                };
                let cand_score = fit_score(&cand_fit.fit);
                if cand_score + term_rel_tol * local_best_score.abs().max(1.0) < local_best_score {
                    local_best_score = cand_score;
                    local_best_fit = Some(cand_fit);
                    local_best_spec = Some(cand_spec);
                }
            }

            if let (Some(next_fit), Some(next_spec)) = (local_best_fit, local_best_spec) {
                best = next_fit;
                best_score = local_best_score;
                resolved_spec = next_spec;
                any_improvement = true;
                if let Some(new_ls) = get_matern_length_scale(&resolved_spec, term_idx) {
                    log::info!(
                        "[Matern-kappa] outer={} term={} accepted length_scale={:.6e} (score={:.6e})",
                        outer + 1,
                        term_idx,
                        new_ls,
                        best_score
                    );
                }
            }
        }
        if !any_improvement {
            break;
        }
    }

    Ok(FittedTermCollectionWithSpec {
        fit: best.fit,
        design: best.design,
        resolved_spec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec,
        DuchonNullspaceOrder, MaternBasisSpec, MaternIdentifiability, MaternNu,
        SpatialIdentifiability, ThinPlateBasisSpec,
    };
    use crate::faer_ndarray::FaerSvd;
    use ndarray::array;

    fn numerical_rank(x: &Array2<f64>) -> usize {
        let (_, s, _) = x
            .svd(false, false)
            .expect("SVD should succeed in rank test");
        let sigma_max = s.iter().copied().fold(0.0_f64, f64::max);
        let tol = (x.nrows().max(x.ncols()).max(1) as f64) * f64::EPSILON * sigma_max.max(1.0);
        s.iter().filter(|&&sv| sv > tol).count()
    }

    fn residual_norm_to_column_space(x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let (u_opt, _, _) = x
            .svd(true, false)
            .expect("SVD should succeed in projection residual test");
        let u = u_opt.expect("left singular vectors should be present");
        let rank = numerical_rank(x);
        let mut proj = Array1::<f64>::zeros(y.len());
        for j in 0..rank.min(u.ncols()) {
            let uj = u.column(j);
            let coeff = uj.dot(y);
            proj += &(&uj.to_owned() * coeff);
        }
        let resid = y - &proj;
        resid.dot(&resid).sqrt()
    }

    #[test]
    fn smooth_design_assembles_terms_and_penalties() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];

        let terms = vec![
            SmoothTermSpec {
                name: "s_x0".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: true,
                        identifiability: BSplineIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            },
            SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            },
        ];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), data.nrows());
        assert_eq!(sd.terms.len(), 2);
        // bspline double-penalty contributes two blocks; tps double-penalty is
        // represented as one fully-shrunk block.
        assert_eq!(sd.penalties.len(), 3);
        assert_eq!(sd.nullspace_dims.len(), 3);
        for s in &sd.penalties {
            assert_eq!(s.nrows(), sd.design.ncols());
            assert_eq!(s.ncols(), sd.design.ncols());
        }
    }

    #[test]
    fn shape_mapping_monotone_increasing_is_non_decreasing() {
        let theta = array![-1.0, 0.5, -0.2, 0.3];
        let beta = SmoothDesign::map_term_coefficients(&theta, ShapeConstraint::MonotoneIncreasing)
            .unwrap();
        for i in 1..beta.len() {
            assert!(beta[i] >= beta[i - 1]);
        }
    }

    #[test]
    fn build_smooth_design_rejects_multiaxis_spatial_shape_constraints() {
        let data = array![[0.0, 0.0], [0.5, 0.2], [1.0, 0.4], [1.5, 0.6],];
        let terms = vec![SmoothTermSpec {
            name: "tps_shape".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 3 },
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];

        let err = build_smooth_design(data.view(), &terms).expect_err("shape should be rejected");
        match err {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("requires exactly 1 feature axis"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn build_smooth_design_accepts_monotone_thin_plate_1d_with_linear_constraints() {
        let data = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_tps".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained thin-plate");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.design.ncols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_accepts_monotone_matern_1d_with_linear_constraints() {
        let data = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 0.7,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained Matérn");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.design.ncols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_accepts_monotone_duchon_1d_with_linear_constraints() {
        let data = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 0.9,
                    nu: MaternNu::FiveHalves,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained Duchon");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.design.ncols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_accepts_monotone_bspline_with_bounds() {
        let data = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_bs".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knot_spec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 3,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained bspline");
        let lb = sd
            .coefficient_lower_bounds
            .as_ref()
            .expect("lower bounds should be generated");
        assert_eq!(lb.len(), sd.design.ncols());
        assert!(lb[0].is_infinite() && lb[0].is_sign_negative());
        for j in 1..lb.len() {
            assert_eq!(lb[j], 0.0);
        }
    }

    #[test]
    fn term_collection_design_combines_linear_and_smooth() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: true,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };
        let design = build_term_collection_design(data.view(), &spec).unwrap();
        assert_eq!(design.design.nrows(), data.nrows());
        assert_eq!(design.intercept_range, 0..1);
        assert!(
            design
                .design
                .column(design.intercept_range.start)
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-12)
        );
        assert!(design.design.ncols() >= 2);
        assert_eq!(design.linear_ranges.len(), 1);
        assert_eq!(design.random_effect_ranges.len(), 0);
        assert_eq!(design.penalties.len(), 2); // linear ridge + 1 smooth penalty
        assert_eq!(design.nullspace_dims.len(), 2);
    }

    #[test]
    fn spatial_smooth_columns_do_not_duplicate_global_intercept() {
        let data = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.6],
            [0.8, 0.7],
            [1.0, 1.0],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();
        let smooth_start = 1usize;
        let smooth_end = smooth_start + design.smooth.design.ncols();
        for col in smooth_start..smooth_end {
            let is_all_ones = design
                .design
                .column(col)
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-12);
            assert!(
                !is_all_ones,
                "smooth column {col} unexpectedly duplicated intercept"
            );
        }
    }

    #[test]
    fn spatial_smooth_drops_matching_linear_trend_columns() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();

        // Raw TPS width for k=4,d=2 is 4; we drop intercept + matching x0 linear component.
        assert_eq!(design.smooth.design.ncols(), 2);

        let lin_col = design.linear_ranges[0].1.start;
        let lin_values = design.design.column(lin_col).to_owned();
        let smooth_start = 1 + spec.linear_terms.len();
        let smooth_end = smooth_start + design.smooth.design.ncols();
        for col in smooth_start..smooth_end {
            let same_as_linear = design
                .design
                .column(col)
                .iter()
                .zip(lin_values.iter())
                .all(|(&a, &b)| (a - b).abs() < 1e-12);
            assert!(
                !same_as_linear,
                "smooth column {col} unexpectedly duplicated linear term column"
            );
        }
    }

    #[test]
    fn spatial_option5_is_orthogonal_to_parametric_block() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();
        let n = data.nrows();
        let mut c = Array2::<f64>::zeros((n, 2));
        c.column_mut(0).fill(1.0);
        c.column_mut(1).assign(&data.column(0));
        let smooth_start = 1 + spec.linear_terms.len();
        let b = design
            .design
            .slice(s![
                ..,
                smooth_start..(smooth_start + design.smooth.design.ncols())
            ])
            .to_owned();
        let rel = orthogonality_relative_residual(b.view(), c.view());
        assert!(
            rel <= 1e-10,
            "Option 5 orthogonality residual too large: {rel}"
        );
    }

    #[test]
    fn spatial_option5_does_not_overconstrain_on_nonoverlapping_linear_terms() {
        let n = 40usize;
        let p = 16usize;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                // Deterministic, non-collinear synthetic PCs.
                data[[i, j]] = (i as f64) * 0.03 + (j as f64) * 0.11 + ((i * (j + 1)) as f64) * 1e-3;
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: (5..16)
                .map(|j| LinearTermSpec {
                    name: format!("pc{j}"),
                    feature_col: j,
                    double_penalty: false,
                })
                .collect(),
            random_effect_terms: vec![],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "tps_pc1".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![1],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        },
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "tps_pc2".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![2],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        },
                    },
                    shape: ShapeConstraint::None,
                },
            ],
        };

        let out = build_term_collection_design(data.view(), &spec);
        assert!(
            out.is_ok(),
            "term-local Option 5 should not over-constrain non-overlapping smooth/linear terms: {:?}",
            out.err()
        );
    }

    #[test]
    fn spatial_frozen_transform_rebuild_is_exact_on_training_rows() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let fit_spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_design = build_term_collection_design(data.view(), &fit_spec).unwrap();
        let term_meta = &fit_design.smooth.terms[0].metadata;
        let (centers, z) = match term_meta {
            BasisMetadata::ThinPlate {
                centers,
                identifiability_transform,
            } => (
                centers.clone(),
                identifiability_transform
                    .clone()
                    .expect("fit-time Option 5 should store transform"),
            ),
            other => panic!("unexpected metadata variant: {other:?}"),
        };

        let frozen_spec = TermCollectionSpec {
            linear_terms: fit_spec.linear_terms.clone(),
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::UserProvided(centers),
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::FrozenTransform { transform: z },
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };
        let frozen_design = build_term_collection_design(data.view(), &frozen_spec).unwrap();

        let a = fit_design.smooth.design;
        let b = frozen_design.smooth.design;
        assert_eq!(a.dim(), b.dim());
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs <= 1e-12,
            "frozen transform rebuild mismatch max_abs={max_abs}"
        );
    }

    #[test]
    fn term_collection_design_adds_random_effect_dummy_block_with_ridge() {
        let data = array![
            [0.1, 0.0],
            [0.2, 1.0],
            [0.3, 0.0],
            [0.4, 2.0],
            [0.5, 1.0],
            [0.6, 2.0],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "id".to_string(),
                feature_col: 1,
                drop_first_level: false,
                frozen_levels: None,
            }],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).unwrap();
        assert_eq!(design.intercept_range, 0..1);
        // 3 observed levels -> 3 dummy columns
        assert_eq!(design.design.ncols(), 4);
        assert_eq!(design.random_effect_ranges.len(), 1);
        assert_eq!(design.penalties.len(), 1);
        assert_eq!(design.nullspace_dims, vec![0]);
        let (_name, range) = &design.random_effect_ranges[0];
        for i in 0..design.design.nrows() {
            let row_sum: f64 = design.design.slice(s![i, range.clone()]).sum();
            assert!((row_sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn matern_smooth_builds_with_double_penalty_in_high_dim() {
        let n = 12usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.1 + (j as f64) * 0.03;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "matern_x".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale: 0.75,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // Matérn double-penalty is represented as one fully-shrunk block.
        assert_eq!(sd.penalties.len(), 1);
        assert_eq!(sd.nullspace_dims.len(), 1);
        assert_eq!(sd.nullspace_dims[0], 0);
    }

    #[test]
    fn duchon_linear_nullspace_builds_and_reports_nullspace_dim() {
        let n = 14usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.07 + (j as f64) * 0.05;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "duchon_x".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: 0.9,
                    nu: MaternNu::FiveHalves,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    double_penalty: true,
                    identifiability: SpatialIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // Duchon double-penalty is represented as one fully-shrunk block.
        assert_eq!(sd.penalties.len(), 1);
        assert_eq!(sd.nullspace_dims.len(), 1);
        assert_eq!(sd.nullspace_dims[0], 0);
    }

    #[test]
    fn tensor_bspline_term_builds_te_style_design_and_penalties() {
        let n = 10usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (i as f64 / (n as f64 - 1.0)).powi(2);
        }

        let spec_x = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 3,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };
        let spec_y = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 2,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let terms = vec![SmoothTermSpec {
            name: "te_xy".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginal_specs: vec![spec_x, spec_y],
                    double_penalty: true,
                    identifiability: TensorBSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // one Kronecker penalty per marginal + optional ridge
        assert_eq!(sd.penalties.len(), 3);
        assert_eq!(sd.nullspace_dims.len(), 3);
        assert!(sd.penalties.iter().all(|s| s.nrows() == sd.design.ncols()));
        assert!(sd.penalties.iter().all(|s| s.ncols() == sd.design.ncols()));
    }

    #[test]
    fn tensor_bspline_design_is_identifiable_against_global_intercept() {
        let n = 120usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (3.0 * t).sin();
        }

        let tensor_term = SmoothTermSpec {
            name: "te_xy".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginal_specs: vec![
                        BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knot_spec: BSplineKnotSpec::Generate {
                                data_range: (0.0, 1.0),
                                num_internal_knots: 6,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::default(),
                        },
                        BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knot_spec: BSplineKnotSpec::Generate {
                                data_range: (-1.0, 1.0),
                                num_internal_knots: 6,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::default(),
                        },
                    ],
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::None,
        };

        let sd = build_smooth_design(data.view(), &[tensor_term.clone()]).unwrap();
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![tensor_term],
        };
        let full = build_term_collection_design(data.view(), &spec).unwrap();
        let ones = Array1::<f64>::ones(n);
        let residual_vs_tensor = residual_norm_to_column_space(&sd.design, &ones);
        let residual_vs_full = residual_norm_to_column_space(&full.design, &ones);

        // Tensor block alone must not be able to represent the constant surface.
        assert!(residual_vs_tensor > 1e-6);
        // With explicit intercept, constants should be represented (near) exactly.
        assert!(residual_vs_full < 1e-8);
    }

    #[test]
    fn matern_kappa_optimization_monotone_improves_or_keeps_score() {
        let n = 60usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.13).sin();
            let x2 = (i as f64 * 0.07).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = (2.5 * x0).sin() + 0.4 * x1 - 0.2 * x2;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 20.0,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let baseline = fit_term_collection_for_spec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
        )
        .expect("baseline fit should succeed");
        let baseline_score = fit_score(&baseline.fit);

        let optimized = fit_term_collection_with_matern_kappa_optimization(
            data.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
            &MaternKappaOptimizationOptions {
                enabled: true,
                max_outer_iter: 2,
                rel_tol: 1e-5,
                log_step: std::f64::consts::LN_2,
                min_length_scale: 1e-3,
                max_length_scale: 1e3,
            },
        )
        .expect("optimized fit should succeed");
        let optimized_score = fit_score(&optimized.fit);
        assert!(optimized_score <= baseline_score + 1e-10);

        let ls = match &optimized.resolved_spec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale,
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));
    }
}
