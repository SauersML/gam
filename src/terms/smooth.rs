use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BasisBuildResult, BasisError, BasisMetadata,
    DuchonBasisSpec, MaternBasisSpec, ThinPlateBasisSpec, build_bspline_basis_1d,
    build_duchon_basis, build_matern_basis, build_thin_plate_basis,
    create_bspline_basis_nd_with_knots,
};
#[cfg(test)]
use crate::basis::MaternIdentifiability;
use crate::construction::kronecker_product;
use crate::estimate::{EstimationError, FitOptions, FitResult, fit_gam};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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
}

#[derive(Debug, Clone)]
pub struct SmoothDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub terms: Vec<SmoothTerm>,
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
    let mut marginal_null_dims = Vec::<usize>::with_capacity(feature_cols.len());

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
        marginal_null_dims.push(*built.nullspace_dims.first().ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "internal TensorBSpline error at dim {dim}: missing marginal nullspace dim"
            ))
        })?);
    }

    let data_views: Vec<_> = feature_cols.iter().map(|&c| data.column(c)).collect();
    let knot_views: Vec<_> = marginal_knots.iter().map(|k| k.view()).collect();
    let (basis, _) =
        create_bspline_basis_nd_with_knots(&data_views, &knot_views, &marginal_degrees)?;
    let design = (*basis).clone();

    let total_cols = design.ncols();
    let mut penalties = Vec::<Array2<f64>>::with_capacity(
        marginal_penalties.len() + if spec.double_penalty { 1 } else { 0 },
    );
    let mut nullspace_dims = Vec::<usize>::with_capacity(
        marginal_null_dims.len() + if spec.double_penalty { 1 } else { 0 },
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

        let mut null_dim = marginal_null_dims[dim];
        for (j, &qj) in marginal_num_basis.iter().enumerate() {
            if j == dim {
                continue;
            }
            null_dim = null_dim.checked_mul(qj).ok_or_else(|| {
                BasisError::DimensionMismatch(
                    "TensorBSpline null-space dimension overflow".to_string(),
                )
            })?;
        }

        penalties.push(s_dim);
        nullspace_dims.push(null_dim);
    }

    if spec.double_penalty {
        penalties.push(Array2::<f64>::eye(total_cols));
        nullspace_dims.push(0);
    }

    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        metadata: BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.to_vec(),
            knots: marginal_knots,
            degrees: marginal_degrees,
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

    for term in terms {
        if term.shape != ShapeConstraint::None {
            return Err(BasisError::InvalidInput(format!(
                "ShapeConstraint::{:?} is not enforced in the current linear fit pipeline; use ShapeConstraint::None or a constrained/nonlinear solver",
                term.shape
            )));
        }
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
                build_bspline_basis_1d(data.column(*feature_col), spec)?
            }
            SmoothBasisSpec::ThinPlate { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_thin_plate_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Matern { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_matern_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Duchon { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_duchon_basis(x.view(), spec)?
            }
            SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
                build_tensor_bspline_basis(data, feature_cols, spec)?
            }
        };

        let p_local = built.design.ncols();
        let design_t = built.design;
        let penalties_t: Vec<Array2<f64>> = built.penalties;

        local_dims.push(p_local);
        local_designs.push(design_t);
        local_penalties.push(penalties_t);
        local_nullspaces.push(built.nullspace_dims);
        local_metadata.push(built.metadata);
    }

    let total_p: usize = local_dims.iter().sum();
    let mut design = Array2::<f64>::zeros((n, total_p));
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();

    let mut col_start = 0usize;
    for (idx, term) in terms.iter().enumerate() {
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
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            metadata: local_metadata[idx].clone(),
        });

        col_start = col_end;
    }

    Ok(SmoothDesign {
        design,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        terms: terms_out,
    })
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();
    let smooth = build_smooth_design(data, &spec.smooth_terms)?;
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

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        intercept_range: 0..1,
        linear_ranges,
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
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
        },
    )?;
    Ok(FittedTermCollection { fit, design })
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

    for outer in 0..kappa_options.max_outer_iter {
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
        DuchonNullspaceOrder, MaternBasisSpec, MaternIdentifiability, MaternNu, ThinPlateBasisSpec,
    };
    use ndarray::array;

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
                    },
                },
                shape: ShapeConstraint::None,
            },
        ];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), data.nrows());
        assert_eq!(sd.terms.len(), 2);
        assert_eq!(sd.penalties.len(), 4);
        assert_eq!(sd.nullspace_dims.len(), 4);
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
    fn build_smooth_design_rejects_non_none_shape_constraints() {
        let data = array![[0.0, 0.0], [0.5, 0.2], [1.0, 0.4], [1.5, 0.6],];
        let terms = vec![SmoothTermSpec {
            name: "tps_shape".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 3 },
                    double_penalty: false,
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];

        let err = build_smooth_design(data.view(), &terms).expect_err("shape should be rejected");
        match err {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("not enforced"));
            }
            other => panic!("unexpected error: {other:?}"),
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
        assert_eq!(design.penalties.len(), 3); // linear ridge + 2 smooth penalties
        assert_eq!(design.nullspace_dims.len(), 3);
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
        // kernel + ridge penalties
        assert_eq!(sd.penalties.len(), 2);
        assert_eq!(sd.nullspace_dims.len(), 2);
        // centered Matérn smooth has no unpenalized intercept mode
        assert_eq!(sd.nullspace_dims[0], 0);
        assert_eq!(sd.nullspace_dims[1], 0);
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
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        assert_eq!(sd.penalties.len(), 2);
        assert_eq!(sd.nullspace_dims.len(), 2);
        // Linear null space in d dimensions -> d+1 free polynomial terms
        assert_eq!(sd.nullspace_dims[0], d + 1);
        assert_eq!(sd.nullspace_dims[1], 0);
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
