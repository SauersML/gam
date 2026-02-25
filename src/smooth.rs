use crate::basis::{
    BSplineBasisSpec, BasisBuildResult, BasisError, BasisMetadata, DuchonBasisSpec,
    MaternBasisSpec, ThinPlateBasisSpec, build_bspline_basis_1d, build_duchon_basis,
    build_matern_basis, build_thin_plate_basis,
};
use crate::estimate::{
    EstimationError, ExternalOptimOptions, FitOptions, FitResult, evaluate_external_cost_and_ridge,
    evaluate_external_gradients, fit_gam,
};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView2, s};
use std::ops::Range;
use wolfe_bfgs::Bfgs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeConstraint {
    None,
    MonotoneIncreasing,
    MonotoneDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct LinearTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// Optional double-penalty ridge on this linear coefficient.
    /// If true, emits an identity penalty block for this 1D term.
    pub double_penalty: bool,
}

#[derive(Debug, Clone)]
pub struct TermCollectionSpec {
    pub linear_terms: Vec<LinearTermSpec>,
    pub smooth_terms: Vec<SmoothTermSpec>,
}

#[derive(Debug, Clone)]
pub struct TermCollectionDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub smooth: SmoothDesign,
}

pub struct FittedTermCollection {
    pub fit: FitResult,
    pub design: TermCollectionDesign,
}

#[derive(Clone, Debug)]
struct ShapeBlock {
    range: Range<usize>,
    shape: ShapeConstraint,
}

#[derive(Debug, Clone)]
pub enum MaternLengthScaleOptimization {
    Fixed,
    HeuristicDefault,
    ProfileGrid {
        num_points: usize,
        kappa_min_factor: f64,
        kappa_max_factor: f64,
    },
    JointBfgs {
        max_iter: usize,
        tol: f64,
        kappa_fd_step: f64,
    },
}

impl Default for MaternLengthScaleOptimization {
    fn default() -> Self {
        Self::HeuristicDefault
    }
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

fn nu_to_value(nu: crate::basis::MaternNu) -> f64 {
    match nu {
        crate::basis::MaternNu::Half => 0.5,
        crate::basis::MaternNu::ThreeHalves => 1.5,
        crate::basis::MaternNu::FiveHalves => 2.5,
        crate::basis::MaternNu::SevenHalves => 3.5,
        crate::basis::MaternNu::NineHalves => 4.5,
    }
}

fn median_pairwise_distance(x: ArrayView2<'_, f64>) -> Option<f64> {
    let n = x.nrows();
    let d = x.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    let mut dists = Vec::<f64>::with_capacity(n * (n.saturating_sub(1)) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist2 = 0.0;
            for c in 0..d {
                let delta = x[[i, c]] - x[[j, c]];
                dist2 += delta * delta;
            }
            let dist = dist2.sqrt();
            if dist.is_finite() && dist > 0.0 {
                dists.push(dist);
            }
        }
    }
    if dists.is_empty() {
        return None;
    }
    dists.sort_by(f64::total_cmp);
    let m = dists.len();
    Some(if m % 2 == 1 {
        dists[m / 2]
    } else {
        0.5 * (dists[m / 2 - 1] + dists[m / 2])
    })
}

fn matern_heuristic_length_scale(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    nu: crate::basis::MaternNu,
) -> Option<f64> {
    let x = select_columns(data, feature_cols).ok()?;
    let med = median_pairwise_distance(x.view())?;
    let kappa = (2.0 * nu_to_value(nu)).sqrt() / med.max(1e-12);
    Some(1.0 / kappa.max(1e-12))
}

fn collect_matern_terms(spec: &TermCollectionSpec) -> Vec<(usize, Vec<usize>, MaternBasisSpec)> {
    let mut out = Vec::new();
    for (idx, term) in spec.smooth_terms.iter().enumerate() {
        if let SmoothBasisSpec::Matern {
            feature_cols,
            spec: mspec,
        } = &term.basis
        {
            out.push((idx, feature_cols.clone(), mspec.clone()));
        }
    }
    out
}

fn apply_matern_length_scales(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    mode: &MaternLengthScaleOptimization,
    override_kappas: Option<&[f64]>,
) -> Result<TermCollectionSpec, EstimationError> {
    let mut updated = spec.clone();
    let mut override_i = 0usize;
    for term in &mut updated.smooth_terms {
        let SmoothBasisSpec::Matern {
            feature_cols,
            spec: mspec,
        } = &mut term.basis
        else {
            continue;
        };

        if let Some(kappas) = override_kappas {
            if override_i >= kappas.len() {
                return Err(EstimationError::InvalidInput(
                    "override kappas length is smaller than number of Matérn terms".to_string(),
                ));
            }
            let kappa = kappas[override_i];
            mspec.length_scale = 1.0 / kappa.max(1e-12);
            override_i += 1;
            continue;
        }

        match mode {
            MaternLengthScaleOptimization::HeuristicDefault => {
                if !(mspec.length_scale.is_finite() && mspec.length_scale > 0.0)
                    && let Some(ls) = matern_heuristic_length_scale(data, feature_cols, mspec.nu)
                {
                    mspec.length_scale = ls;
                }
            }
            MaternLengthScaleOptimization::Fixed
            | MaternLengthScaleOptimization::ProfileGrid { .. }
            | MaternLengthScaleOptimization::JointBfgs { .. } => {}
        }
    }
    if let Some(kappas) = override_kappas
        && override_i != kappas.len()
    {
        return Err(EstimationError::InvalidInput(format!(
            "override kappas length mismatch: consumed {}, provided {}",
            override_i,
            kappas.len()
        )));
    }
    Ok(updated)
}

fn single_matern_term(spec: &TermCollectionSpec) -> Option<(usize, Vec<usize>, MaternBasisSpec)> {
    let all = collect_matern_terms(spec);
    if all.len() == 1 {
        Some(all[0].clone())
    } else {
        None
    }
}

fn fit_with_spec(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let design = build_term_collection_design(data, spec)?;
    let mut fit = fit_gam(
        design.design.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &design.penalties,
        family,
        &FitOptions {
            max_iter: options.max_iter,
            tol: options.tol,
            nullspace_dims: design.nullspace_dims.clone(),
        },
    )?;
    apply_shape_constraints_if_needed(
        &mut fit,
        &design,
        y.view(),
        weights.view(),
        offset.view(),
        family,
        options,
    )?;
    Ok(FittedTermCollection { fit, design })
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
        Ok(theta_to_beta_range(unconstrained.view(), shape))
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

fn stable_exp(theta: f64) -> f64 {
    theta.clamp(-40.0, 40.0).exp()
}

fn theta_to_beta_range(theta: ndarray::ArrayView1<'_, f64>, shape: ShapeConstraint) -> Array1<f64> {
    let n = theta.len();
    let mut beta = Array1::<f64>::zeros(n);
    match shape {
        ShapeConstraint::None => beta.assign(&theta),
        ShapeConstraint::MonotoneIncreasing | ShapeConstraint::MonotoneDecreasing => {
            let sign = if shape == ShapeConstraint::MonotoneIncreasing {
                1.0
            } else {
                -1.0
            };
            if n == 0 {
                return beta;
            }
            let base = theta[0];
            let mut run = 0.0;
            beta[0] = base;
            for i in 1..n {
                run += stable_exp(theta[i]);
                beta[i] = base + sign * run;
            }
        }
        ShapeConstraint::Convex | ShapeConstraint::Concave => {
            let sign = if shape == ShapeConstraint::Convex {
                1.0
            } else {
                -1.0
            };
            if n == 0 {
                return beta;
            }
            beta[0] = theta[0];
            if n == 1 {
                return beta;
            }
            let mut first = theta[1];
            beta[1] = beta[0] + first;
            for i in 2..n {
                first += sign * stable_exp(theta[i]);
                beta[i] = beta[i - 1] + first;
            }
        }
    }
    beta
}

fn add_shape_chain_rule_grad(
    grad_theta: &mut Array1<f64>,
    grad_beta: ndarray::ArrayView1<'_, f64>,
    theta: ndarray::ArrayView1<'_, f64>,
    block: &ShapeBlock,
) {
    match block.shape {
        ShapeConstraint::None => {
            for i in 0..block.range.len() {
                grad_theta[block.range.start + i] = grad_beta[i];
            }
        }
        ShapeConstraint::MonotoneIncreasing | ShapeConstraint::MonotoneDecreasing => {
            let sign = if block.shape == ShapeConstraint::MonotoneIncreasing {
                1.0
            } else {
                -1.0
            };
            let n = block.range.len();
            if n == 0 {
                return;
            }
            grad_theta[block.range.start] = grad_beta.sum();
            let mut suffix = 0.0;
            for k_rev in 0..(n - 1) {
                let k = n - 1 - k_rev;
                suffix += grad_beta[k];
                grad_theta[block.range.start + k] = sign * stable_exp(theta[k]) * suffix;
            }
        }
        ShapeConstraint::Convex | ShapeConstraint::Concave => {
            let sign = if block.shape == ShapeConstraint::Convex {
                1.0
            } else {
                -1.0
            };
            let n = block.range.len();
            if n == 0 {
                return;
            }
            // Base intercept contribution.
            grad_theta[block.range.start] = grad_beta.sum();
            if n == 1 {
                return;
            }
            // Initial slope (theta[1]) contributes to beta[i] for i>=1 with weight i.
            let mut slope_grad = 0.0;
            for i in 1..n {
                slope_grad += (i as f64) * grad_beta[i];
            }
            grad_theta[block.range.start + 1] = slope_grad;

            let mut s1 = 0.0; // Σ_{i>=k} grad_beta[i]
            let mut s2 = 0.0; // Σ_{i>=k} i*grad_beta[i] (local index)
            for k_rev in 0..(n - 2) {
                let k = n - 1 - k_rev;
                let g = grad_beta[k];
                s1 += g;
                s2 += (k as f64) * g;
                let weighted = s2 - ((k as f64) - 1.0) * s1; // Σ_{i>=k} (i-k+1) g_i
                grad_theta[block.range.start + k] = sign * stable_exp(theta[k]) * weighted;
            }
        }
    }
}

fn initial_theta_from_beta(beta: ndarray::ArrayView1<'_, f64>, block: &ShapeBlock) -> Array1<f64> {
    let n = block.range.len();
    let mut theta = Array1::<f64>::zeros(n);
    let eps = 1e-6;
    match block.shape {
        ShapeConstraint::None => theta.assign(&beta),
        ShapeConstraint::MonotoneIncreasing => {
            if n == 0 {
                return theta;
            }
            theta[0] = beta[0];
            let mut prev = beta[0];
            for i in 1..n {
                let inc = (beta[i] - prev).max(eps);
                theta[i] = inc.ln();
                prev = beta[i];
            }
        }
        ShapeConstraint::MonotoneDecreasing => {
            if n == 0 {
                return theta;
            }
            theta[0] = beta[0];
            let mut prev = beta[0];
            for i in 1..n {
                let inc = (-(beta[i] - prev)).max(eps);
                theta[i] = inc.ln();
                prev = beta[i];
            }
        }
        ShapeConstraint::Convex => {
            if n == 0 {
                return theta;
            }
            theta[0] = beta[0];
            if n == 1 {
                return theta;
            }
            let mut prev_first = beta[1] - beta[0];
            theta[1] = prev_first;
            for i in 2..n {
                let first = beta[i] - beta[i - 1];
                let inc = (first - prev_first).max(eps);
                theta[i] = inc.ln();
                prev_first = first;
            }
        }
        ShapeConstraint::Concave => {
            if n == 0 {
                return theta;
            }
            theta[0] = beta[0];
            if n == 1 {
                return theta;
            }
            let mut prev_first = beta[1] - beta[0];
            theta[1] = prev_first;
            for i in 2..n {
                let first = beta[i] - beta[i - 1];
                let inc = (-(first - prev_first)).max(eps);
                theta[i] = inc.ln();
                prev_first = first;
            }
        }
    }
    theta
}

fn apply_shape_constraints_if_needed(
    fit: &mut FitResult,
    design: &TermCollectionDesign,
    y: ndarray::ArrayView1<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    offset: ndarray::ArrayView1<'_, f64>,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<(), EstimationError> {
    let p_lin = design.linear_ranges.len();
    let mut shape_blocks: Vec<ShapeBlock> = Vec::new();
    for term in &design.smooth.terms {
        if term.shape == ShapeConstraint::None {
            continue;
        }
        let start = p_lin + term.coeff_range.start;
        let end = p_lin + term.coeff_range.end;
        shape_blocks.push(ShapeBlock {
            range: start..end,
            shape: term.shape,
        });
    }
    if shape_blocks.is_empty() {
        return Ok(());
    }
    if !matches!(family, LikelihoodFamily::GaussianIdentity) {
        return Err(EstimationError::InvalidInput(
            "Shape constraints are currently supported for GaussianIdentity in fit_term_collection"
                .to_string(),
        ));
    }

    let x = &design.design;
    let n = x.nrows();
    let p = x.ncols();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    for (k, s) in design.penalties.iter().enumerate() {
        let lambda = fit.lambdas.get(k).copied().unwrap_or(0.0);
        if lambda == 0.0 {
            continue;
        }
        s_lambda = s_lambda + &(s * lambda);
    }

    let mut theta0 = fit.beta.clone();
    for block in &shape_blocks {
        let beta_local = fit.beta.slice(s![block.range.start..block.range.end]);
        let t_local = initial_theta_from_beta(beta_local, block);
        theta0
            .slice_mut(s![block.range.start..block.range.end])
            .assign(&t_local);
    }

    let eval = |theta: &Array1<f64>| -> (f64, Array1<f64>) {
        let mut beta = theta.clone();
        for block in &shape_blocks {
            let local_theta = theta.slice(s![block.range.start..block.range.end]);
            let beta_local = theta_to_beta_range(local_theta, block.shape);
            beta.slice_mut(s![block.range.start..block.range.end])
                .assign(&beta_local);
        }

        let mut eta = x.dot(&beta);
        eta += &offset;
        let resid = y.to_owned() - &eta;
        let w_resid = &resid * &weights;
        let data_term = 0.5 * weights.dot(&resid.mapv(|r| r * r));
        let penalty_term = 0.5 * beta.dot(&s_lambda.dot(&beta));
        let cost = data_term + penalty_term;

        let grad_beta = -x.t().dot(&w_resid) + s_lambda.dot(&beta);
        let mut grad_theta = grad_beta.clone();
        for block in &shape_blocks {
            let gb = grad_beta.slice(s![block.range.start..block.range.end]);
            let lt = theta.slice(s![block.range.start..block.range.end]);
            add_shape_chain_rule_grad(&mut grad_theta, gb, lt, block);
        }
        (cost, grad_theta)
    };

    let mut solver = Bfgs::new(theta0, eval)
        .with_tolerance(options.tol.max(1e-8))
        .with_max_iterations(options.max_iter.max(100))
        .with_rng_seed(0x5A1E_u64);
    let solution = match solver.run() {
        Ok(sol) => sol,
        Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
        Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(e) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "shape-constrained coefficient optimization failed: {e:?}"
            )));
        }
    };
    let theta_star = solution.final_point;
    let mut beta_star = theta_star.clone();
    for block in &shape_blocks {
        let local_theta = theta_star.slice(s![block.range.start..block.range.end]);
        let beta_local = theta_to_beta_range(local_theta, block.shape);
        beta_star
            .slice_mut(s![block.range.start..block.range.end])
            .assign(&beta_local);
    }
    fit.beta = beta_star;

    // Recompute Gaussian scale after constrained coefficient update.
    let mut eta = x.dot(&fit.beta);
    eta += &offset;
    let resid = y.to_owned() - &eta;
    let wrss = weights.dot(&resid.mapv(|r| r * r));
    fit.scale = wrss / ((n as f64 - fit.edf_total).max(1.0));

    // Existing covariance fields come from unconstrained PIRLS and are invalid after
    // nonlinear constrained re-optimization. Clear them to avoid misleading inference.
    fit.beta_covariance = None;
    fit.beta_standard_errors = None;
    fit.beta_covariance_corrected = None;
    fit.beta_standard_errors_corrected = None;

    Ok(())
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();
    let smooth = build_smooth_design(data, &spec.smooth_terms)?;

    for linear in &spec.linear_terms {
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
    }

    let p_lin = spec.linear_terms.len();
    let p_smooth = smooth.design.ncols();
    let p_total = p_lin + p_smooth;
    let mut design = Array2::<f64>::zeros((n, p_total));

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        design
            .column_mut(j)
            .assign(&data.column(linear.feature_col));
        linear_ranges.push((linear.name.clone(), j..(j + 1)));
    }
    if p_smooth > 0 {
        design.slice_mut(s![.., p_lin..]).assign(&smooth.design);
    }

    let mut penalties = Vec::<Array2<f64>>::new();
    let mut nullspace_dims = Vec::<usize>::new();

    for (j, linear) in spec.linear_terms.iter().enumerate() {
        if !linear.double_penalty {
            continue;
        }
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        s[[j, j]] = 1.0;
        penalties.push(s);
        nullspace_dims.push(0);
    }

    for (s_local, &ns) in smooth.penalties.iter().zip(smooth.nullspace_dims.iter()) {
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        let start = p_lin;
        let end = p_lin + p_smooth;
        s.slice_mut(s![start..end, start..end]).assign(s_local);
        penalties.push(s);
        nullspace_dims.push(ns);
    }

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        linear_ranges,
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
    fit_term_collection_with_matern_optimization(
        data,
        y,
        weights,
        offset,
        spec,
        family,
        options,
        &MaternLengthScaleOptimization::HeuristicDefault,
    )
}

pub fn fit_term_collection_with_matern_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
    tuning: &MaternLengthScaleOptimization,
) -> Result<FittedTermCollection, EstimationError> {
    match tuning {
        MaternLengthScaleOptimization::Fixed | MaternLengthScaleOptimization::HeuristicDefault => {
            let resolved = apply_matern_length_scales(data, spec, tuning, None)?;
            fit_with_spec(data, y, weights, offset, &resolved, family, options)
        }
        MaternLengthScaleOptimization::ProfileGrid {
            num_points,
            kappa_min_factor,
            kappa_max_factor,
        } => {
            let Some((_, feature_cols, mspec)) = single_matern_term(spec) else {
                return Err(EstimationError::InvalidInput(
                    "ProfileGrid requires exactly one Matérn smooth term".to_string(),
                ));
            };
            let base_kappa = if mspec.length_scale.is_finite() && mspec.length_scale > 0.0 {
                1.0 / mspec.length_scale
            } else {
                let ls = matern_heuristic_length_scale(data, &feature_cols, mspec.nu).ok_or_else(
                    || {
                        EstimationError::InvalidInput(
                            "Unable to compute Matérn heuristic length scale".to_string(),
                        )
                    },
                )?;
                1.0 / ls
            };
            let m = (*num_points).clamp(5, 8);
            let k_lo = base_kappa * kappa_min_factor.max(1e-6);
            let k_hi = base_kappa * kappa_max_factor.max(kappa_min_factor + 1e-6);
            let log_lo = k_lo.ln();
            let log_hi = k_hi.ln();
            let mut best: Option<(f64, FittedTermCollection)> = None;

            for i in 0..m {
                let t = if m == 1 {
                    0.0
                } else {
                    i as f64 / (m - 1) as f64
                };
                let kappa = (log_lo + t * (log_hi - log_lo)).exp();
                let one = [kappa];
                let resolved = apply_matern_length_scales(data, spec, tuning, Some(&one))?;
                let candidate = fit_with_spec(
                    data,
                    y.clone(),
                    weights.clone(),
                    offset.clone(),
                    &resolved,
                    family,
                    options,
                )?;
                let rho = candidate.fit.lambdas.mapv(|v| v.max(1e-300).ln());
                let ext_opts = ExternalOptimOptions {
                    family,
                    max_iter: options.max_iter,
                    tol: options.tol,
                    nullspace_dims: candidate.design.nullspace_dims.clone(),
                };
                let (cost, _) = evaluate_external_cost_and_ridge(
                    y.view(),
                    weights.view(),
                    candidate.design.design.view(),
                    offset.view(),
                    &candidate.design.penalties,
                    &ext_opts,
                    &rho,
                )?;
                if best.as_ref().map(|(c, _)| cost < *c).unwrap_or(true) {
                    best = Some((cost, candidate));
                }
            }

            best.map(|(_, fit)| fit).ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Matérn profile grid produced no candidates".to_string(),
                )
            })
        }
        MaternLengthScaleOptimization::JointBfgs {
            max_iter,
            tol,
            kappa_fd_step,
        } => {
            let matern_terms = collect_matern_terms(spec);
            if matern_terms.is_empty() {
                return Err(EstimationError::InvalidInput(
                    "JointBfgs requested but no Matérn smooth terms are present".to_string(),
                ));
            }

            let mut base_kappas = Vec::with_capacity(matern_terms.len());
            for (_, feature_cols, mspec) in &matern_terms {
                let kappa = if mspec.length_scale.is_finite() && mspec.length_scale > 0.0 {
                    1.0 / mspec.length_scale
                } else {
                    let ls = matern_heuristic_length_scale(data, feature_cols, mspec.nu)
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "Unable to compute Matérn heuristic length scale".to_string(),
                            )
                        })?;
                    1.0 / ls
                };
                base_kappas.push(kappa.max(1e-12));
            }

            let seed = apply_matern_length_scales(data, spec, tuning, Some(&base_kappas))?;
            let seeded_fit = fit_with_spec(
                data,
                y.clone(),
                weights.clone(),
                offset.clone(),
                &seed,
                family,
                options,
            )?;
            let k = seeded_fit.design.penalties.len();
            let m = base_kappas.len();
            let mut theta0 = Array1::<f64>::zeros(k + m);
            for i in 0..k {
                theta0[i] = seeded_fit.fit.lambdas[i].max(1e-300).ln();
            }
            for j in 0..m {
                theta0[k + j] = base_kappas[j].ln();
            }

            let evaluate_theta =
                |theta: &Array1<f64>| -> Result<(f64, Array1<f64>), EstimationError> {
                    let k = seeded_fit.design.penalties.len();
                    let m = matern_terms.len();
                    if theta.len() != k + m {
                        return Err(EstimationError::InvalidInput(format!(
                            "joint theta length mismatch: got {}, expected {}",
                            theta.len(),
                            k + m
                        )));
                    }
                    let rho = theta.slice(s![..k]).to_owned();
                    let mut kappas = Vec::with_capacity(m);
                    for j in 0..m {
                        kappas.push(theta[k + j].exp().max(1e-12));
                    }
                    let resolved = apply_matern_length_scales(data, spec, tuning, Some(&kappas))?;
                    let design = build_term_collection_design(data, &resolved)?;
                    let ext_opts = ExternalOptimOptions {
                        family,
                        max_iter: options.max_iter,
                        tol: options.tol,
                        nullspace_dims: design.nullspace_dims.clone(),
                    };
                    let (cost, _) = evaluate_external_cost_and_ridge(
                        y.view(),
                        weights.view(),
                        design.design.view(),
                        offset.view(),
                        &design.penalties,
                        &ext_opts,
                        &rho,
                    )?;
                    let (analytic_grad_rho, _) = evaluate_external_gradients(
                        y.view(),
                        weights.view(),
                        design.design.view(),
                        offset.view(),
                        &design.penalties,
                        &ext_opts,
                        &rho,
                    )?;

                    let h = (*kappa_fd_step).max(1e-4);
                    let mut grad = Array1::<f64>::zeros(k + m);
                    grad.slice_mut(s![..k]).assign(&analytic_grad_rho);
                    for j in 0..m {
                        let mut kappas_p = kappas.clone();
                        let mut kappas_m = kappas.clone();
                        kappas_p[j] = (theta[k + j] + h).exp().max(1e-12);
                        kappas_m[j] = (theta[k + j] - h).exp().max(1e-12);
                        let resolved_p =
                            apply_matern_length_scales(data, spec, tuning, Some(&kappas_p))?;
                        let resolved_m =
                            apply_matern_length_scales(data, spec, tuning, Some(&kappas_m))?;
                        let design_p = build_term_collection_design(data, &resolved_p)?;
                        let design_m = build_term_collection_design(data, &resolved_m)?;
                        let ext_opts_p = ExternalOptimOptions {
                            family,
                            max_iter: options.max_iter,
                            tol: options.tol,
                            nullspace_dims: design_p.nullspace_dims.clone(),
                        };
                        let ext_opts_m = ExternalOptimOptions {
                            family,
                            max_iter: options.max_iter,
                            tol: options.tol,
                            nullspace_dims: design_m.nullspace_dims.clone(),
                        };
                        let (cp, _) = evaluate_external_cost_and_ridge(
                            y.view(),
                            weights.view(),
                            design_p.design.view(),
                            offset.view(),
                            &design_p.penalties,
                            &ext_opts_p,
                            &rho,
                        )?;
                        let (cm, _) = evaluate_external_cost_and_ridge(
                            y.view(),
                            weights.view(),
                            design_m.design.view(),
                            offset.view(),
                            &design_m.penalties,
                            &ext_opts_m,
                            &rho,
                        )?;
                        grad[k + j] = (cp - cm) / (2.0 * h);
                    }
                    Ok((cost, grad))
                };

            let mut solver = Bfgs::new(theta0, |theta| match evaluate_theta(theta) {
                Ok(v) => v,
                Err(_) => {
                    let mut grad = Array1::<f64>::zeros(theta.len());
                    for g in &mut grad {
                        *g = -1.0;
                    }
                    (f64::INFINITY, grad)
                }
            })
            .with_tolerance(*tol)
            .with_max_iterations(*max_iter)
            .with_rng_seed(0xBADC0DE_u64);

            let solution = match solver.run() {
                Ok(sol) => sol,
                Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => {
                    *last_solution
                }
                Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => {
                    *last_solution
                }
                Err(e) => {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "joint Matérn (rho, log-kappa[]) BFGS failed: {e:?}"
                    )));
                }
            };
            let theta_star = solution.final_point;
            let mut kappas_star = Vec::with_capacity(matern_terms.len());
            for j in 0..matern_terms.len() {
                kappas_star.push(theta_star[k + j].exp().max(1e-12));
            }
            let resolved = apply_matern_length_scales(data, spec, tuning, Some(&kappas_star))?;
            fit_with_spec(data, y, weights, offset, &resolved, family, options)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, MaternBasisSpec,
        MaternNu, ThinPlateBasisSpec,
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
    fn build_smooth_design_accepts_shape_constraints() {
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
        let sd = build_smooth_design(data.view(), &terms).expect("shape should be accepted");
        assert_eq!(sd.terms.len(), 1);
        assert_eq!(sd.terms[0].shape, ShapeConstraint::MonotoneIncreasing);
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
        assert!(design.design.ncols() >= 1);
        assert_eq!(design.linear_ranges.len(), 1);
        assert_eq!(design.penalties.len(), 3); // linear ridge + 2 smooth penalties
        assert_eq!(design.nullspace_dims.len(), 3);
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
                    include_intercept: true,
                    double_penalty: true,
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
        // first penalty keeps 1 "almost null" intercept dimension
        assert_eq!(sd.nullspace_dims[0], 1);
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
    fn fit_term_collection_matern_heuristic_default_handles_nonpositive_length_scale() {
        let n = 20usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            let x2 = (i as f64 * 0.11).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = 0.5 + 1.2 * x0 - 0.7 * x1 + 0.3 * x2;
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern_auto".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        length_scale: 0.0, // invalid sentinel -> heuristic default
                        nu: MaternNu::FiveHalves,
                        include_intercept: true,
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let fit = fit_term_collection_with_matern_optimization(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                max_iter: 25,
                tol: 1e-4,
                nullspace_dims: vec![],
            },
            &MaternLengthScaleOptimization::HeuristicDefault,
        )
        .expect("heuristic default Matérn tuning should succeed");

        assert!(fit.fit.scale.is_finite());
        assert!(fit.fit.lambdas.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn fit_term_collection_matern_joint_bfgs_runs() {
        let n = 24usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.23).sin();
            let x2 = (i as f64 * 0.19).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = 0.2 + 1.1 * x0 - 0.4 * x1 + 0.25 * x2;
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern_joint".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        length_scale: 0.0, // let joint tuning choose kappa from heuristic seed
                        nu: MaternNu::FiveHalves,
                        include_intercept: true,
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let fit = fit_term_collection_with_matern_optimization(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                max_iter: 20,
                tol: 1e-4,
                nullspace_dims: vec![],
            },
            &MaternLengthScaleOptimization::JointBfgs {
                max_iter: 6,
                tol: 1e-3,
                kappa_fd_step: 5e-3,
            },
        )
        .expect("joint Matérn BFGS tuning should run");

        assert!(fit.fit.scale.is_finite());
        assert!(fit.fit.lambdas.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn fit_term_collection_matern_profile_grid_runs() {
        let n = 30usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.21).sin();
            let x2 = (i as f64 * 0.13).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = 0.3 + 0.8 * x0 - 0.5 * x1 + 0.2 * x2;
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern_grid".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 0.0, // force heuristic seed for grid center
                        nu: MaternNu::FiveHalves,
                        include_intercept: true,
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };

        let fit = fit_term_collection_with_matern_optimization(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                max_iter: 25,
                tol: 1e-4,
                nullspace_dims: vec![],
            },
            &MaternLengthScaleOptimization::ProfileGrid {
                num_points: 6,
                kappa_min_factor: 0.2,
                kappa_max_factor: 5.0,
            },
        )
        .expect("Matérn profile-grid tuning should run");

        assert!(fit.fit.scale.is_finite());
        assert!(fit.fit.lambdas.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn fit_term_collection_enforces_monotone_shape_gaussian() {
        let n = 80usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            // Intentionally mildly wiggly signal; monotone constraint should smooth
            // out local decreases in fitted term coefficients.
            y[i] = 0.2 + 0.8 * x + 0.08 * (12.0 * x).sin();
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "mono_bspline".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 8,
                        },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::MonotoneIncreasing,
            }],
        };

        let fitted = fit_term_collection(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                max_iter: 80,
                tol: 1e-6,
                nullspace_dims: vec![],
            },
        )
        .expect("shape-constrained fit should succeed");

        let term = &fitted.design.smooth.terms[0];
        let beta = fitted
            .fit
            .beta
            .slice(s![term.coeff_range.start..term.coeff_range.end]);
        for i in 1..beta.len() {
            assert!(
                beta[i] >= beta[i - 1] - 1e-9,
                "monotone coefficients violated at {}: {} < {}",
                i,
                beta[i],
                beta[i - 1]
            );
        }
    }

    #[test]
    fn fit_term_collection_enforces_convex_shape_gaussian() {
        let n = 90usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = -0.2 + 0.4 * x + 0.9 * x * x + 0.05 * (14.0 * x).sin();
        }
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "conv_bspline".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 10,
                        },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::Convex,
            }],
        };

        let fitted = fit_term_collection(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                max_iter: 100,
                tol: 1e-6,
                nullspace_dims: vec![],
            },
        )
        .expect("convex constrained fit should succeed");

        let term = &fitted.design.smooth.terms[0];
        let beta = fitted
            .fit
            .beta
            .slice(s![term.coeff_range.start..term.coeff_range.end]);
        if beta.len() >= 3 {
            for i in 2..beta.len() {
                let d2 = beta[i] - 2.0 * beta[i - 1] + beta[i - 2];
                assert!(
                    d2 >= -1e-8,
                    "convex coefficients violated at {}: d2={}",
                    i,
                    d2
                );
            }
        }
    }

    #[test]
    fn fit_term_collection_rejects_shape_constraints_for_non_gaussian() {
        let data = array![[0.0], [0.3], [0.6], [1.0], [1.3], [1.7], [2.0]];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let weights = Array1::<f64>::ones(data.nrows());
        let offset = Array1::<f64>::zeros(data.nrows());
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "mono_logit".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 2.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::MonotoneIncreasing,
            }],
        };

        let result = fit_term_collection(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::BinomialLogit,
            &FitOptions {
                max_iter: 30,
                tol: 1e-6,
                nullspace_dims: vec![],
            },
        );
        match result {
            Err(EstimationError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("Shape constraints are currently supported for GaussianIdentity")
                );
            }
            Err(other) => panic!("unexpected error type: {other:?}"),
            Ok(_) => panic!("expected non-gaussian shaped fit to be rejected"),
        }
    }
}
