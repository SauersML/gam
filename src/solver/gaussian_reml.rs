use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerCholesky, FaerEigh, fast_ab, fast_atb, fast_xt_diag_x, fast_xt_diag_y,
};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use rayon::prelude::*;

const RHO_LOWER: f64 = -30.0;
const RHO_UPPER: f64 = 30.0;
const EIGEN_REL_TOL: f64 = 1.0e-10;
const GRAD_TOL: f64 = 1.0e-12;
const MIN_DEVIANCE: f64 = 1.0e-300;

#[derive(Clone, Debug)]
pub struct GaussianRemlEigenCache {
    pub penalty_eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub coefficient_basis: Array2<f64>,
    pub xtwx_fingerprint: u64,
    pub penalty_fingerprint: u64,
    pub logdet_xtwx: f64,
    pub logdet_penalty_positive: f64,
    pub penalty_rank: usize,
    pub nullity: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GaussianRemlWarmStart {
    pub lambda: Option<f64>,
    pub eigen_cache: Option<GaussianRemlEigenCache>,
}

impl GaussianRemlWarmStart {
    pub fn from_result(result: &GaussianRemlResult) -> Self {
        Self {
            lambda: Some(result.lambda),
            eigen_cache: Some(result.cache.clone()),
        }
    }

    pub fn from_multi_result(result: &GaussianRemlMultiResult) -> Self {
        Self {
            lambda: Some(result.lambda),
            eigen_cache: Some(result.cache.clone()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GaussianRemlResult {
    pub lambda: f64,
    pub rho: f64,
    pub coefficients: Array1<f64>,
    pub fitted: Array1<f64>,
    pub reml_score: f64,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
    pub sigma2: f64,
    pub cache: GaussianRemlEigenCache,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiResult {
    pub lambda: f64,
    pub rho: f64,
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub reml_score: f64,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
    pub sigma2: Array1<f64>,
    pub cache: GaussianRemlEigenCache,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlScoreDerivatives {
    pub reml_score: f64,
    pub grad_lambda: f64,
    pub hess_lambda: f64,
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub sigma2: Array1<f64>,
    pub edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlBackwardResult {
    pub grad_x: Array2<f64>,
    pub grad_y: Array2<f64>,
    pub grad_weights: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiBackwardProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub fit: &'a GaussianRemlMultiResult,
    pub grad_lambda: f64,
    pub grad_coefficients: Option<ArrayView2<'a, f64>>,
    pub grad_fitted: Option<ArrayView2<'a, f64>>,
    pub grad_reml_score: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlNoAllocWorkspace {
    pub xtwy: Array2<f64>,
    pub ywy: Array1<f64>,
    pub projected_rhs: Array2<f64>,
    pub projected_rhs_squared: Array2<f64>,
    pub scaled_projected_rhs: Array2<f64>,
}

impl GaussianRemlNoAllocWorkspace {
    pub fn new(n_coefficients: usize, n_outputs: usize) -> Self {
        Self {
            xtwy: Array2::zeros((n_coefficients, n_outputs)),
            ywy: Array1::zeros(n_outputs),
            projected_rhs: Array2::zeros((n_coefficients, n_outputs)),
            projected_rhs_squared: Array2::zeros((n_coefficients, n_outputs)),
            scaled_projected_rhs: Array2::zeros((n_coefficients, n_outputs)),
        }
    }

    fn validate(&self, p: usize, d: usize) -> Result<(), EstimationError> {
        if self.xtwy.dim() != (p, d)
            || self.ywy.len() != d
            || self.projected_rhs.dim() != (p, d)
            || self.projected_rhs_squared.dim() != (p, d)
            || self.scaled_projected_rhs.dim() != (p, d)
        {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML no-alloc workspace shape mismatch: expected p={p}, d={d}"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GaussianRemlNoAllocFit {
    pub lambda: f64,
    pub rho: f64,
    pub reml_score: f64,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlScalarBatchProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub init_rho: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiBatchProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub init_rho: Option<f64>,
}

#[derive(Clone)]
struct GaussianRemlPrepared {
    cache: GaussianRemlEigenCache,
    ywy: Array1<f64>,
    projected_rhs_squared: Array2<f64>,
    projected_rhs: Array2<f64>,
    n_observations: usize,
    n_outputs: usize,
}

#[derive(Clone, Copy)]
struct ObjectiveEval {
    cost: f64,
    grad: f64,
    hess: f64,
    edf: f64,
}

pub fn gaussian_reml_closed_form(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlResult, EstimationError> {
    gaussian_reml_closed_form_with_nullspace_dim(x, y, penalty, None, weights, init_rho)
}

pub fn gaussian_reml_closed_form_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlResult, EstimationError> {
    let y2 = y.insert_axis(Axis(1));
    let result = gaussian_reml_multi_closed_form_with_nullspace_dim(
        x,
        y2,
        penalty,
        nullspace_dim,
        weights,
        init_rho,
    )?;
    scalar_result_from_multi(result)
}

pub fn gaussian_reml_closed_form_warm_started(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    warm_start: Option<&GaussianRemlWarmStart>,
) -> Result<GaussianRemlResult, EstimationError> {
    gaussian_reml_closed_form_warm_started_with_nullspace_dim(
        x, y, penalty, None, weights, warm_start,
    )
}

pub fn gaussian_reml_closed_form_warm_started_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    warm_start: Option<&GaussianRemlWarmStart>,
) -> Result<GaussianRemlResult, EstimationError> {
    let y2 = y.insert_axis(Axis(1));
    let result = gaussian_reml_multi_closed_form_warm_started_with_nullspace_dim(
        x,
        y2,
        penalty,
        nullspace_dim,
        weights,
        warm_start,
    )?;
    scalar_result_from_multi(result)
}

fn scalar_result_from_multi(
    result: GaussianRemlMultiResult,
) -> Result<GaussianRemlResult, EstimationError> {
    Ok(GaussianRemlResult {
        lambda: result.lambda,
        rho: result.rho,
        coefficients: result.coefficients.column(0).to_owned(),
        fitted: result.fitted.column(0).to_owned(),
        reml_score: result.reml_score,
        reml_grad_lambda: result.reml_grad_lambda,
        reml_hess_lambda: result.reml_hess_lambda,
        reml_grad_rho: result.reml_grad_rho,
        reml_hess_rho: result.reml_hess_rho,
        edf: result.edf,
        sigma2: result.sigma2[0],
        cache: result.cache,
    })
}

pub fn gaussian_reml_multi_closed_form(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    gaussian_reml_multi_closed_form_with_nullspace_dim(x, y, penalty, None, weights, init_rho)
}

pub fn gaussian_reml_multi_closed_form_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    let init_lambda = init_rho.map(f64::exp);
    gaussian_reml_multi_closed_form_from_parts(
        x,
        y,
        penalty,
        nullspace_dim,
        weights,
        init_lambda,
        None,
    )
}

pub fn gaussian_reml_multi_closed_form_warm_started(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    warm_start: Option<&GaussianRemlWarmStart>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    gaussian_reml_multi_closed_form_warm_started_with_nullspace_dim(
        x, y, penalty, None, weights, warm_start,
    )
}

pub fn gaussian_reml_multi_closed_form_warm_started_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    warm_start: Option<&GaussianRemlWarmStart>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    let init_lambda = warm_start.and_then(|start| start.lambda);
    let eigen_cache = warm_start.and_then(|start| start.eigen_cache.as_ref());
    gaussian_reml_multi_closed_form_from_parts(
        x,
        y,
        penalty,
        nullspace_dim,
        weights,
        init_lambda,
        eigen_cache,
    )
}

pub fn gaussian_reml_multi_closed_form_with_cache(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    gaussian_reml_multi_closed_form_from_parts(
        x,
        y,
        penalty,
        None,
        weights,
        init_lambda,
        eigen_cache,
    )
}

pub fn gaussian_reml_multi_closed_form_with_cache_no_alloc(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    eigen_cache: &GaussianRemlEigenCache,
    workspace: &mut GaussianRemlNoAllocWorkspace,
    mut coefficients: ArrayViewMut2<'_, f64>,
    mut fitted: ArrayViewMut2<'_, f64>,
    mut sigma2: ArrayViewMut1<'_, f64>,
) -> Result<GaussianRemlNoAllocFit, EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    validate_gaussian_reml_eigen_cache(eigen_cache, p)?;
    if y.nrows() != n {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        )));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML inputs must be finite".to_string(),
        ));
    }
    if n <= eigen_cache.nullity {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
            eigen_cache.nullity
        )));
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    if eigen_cache.penalty_fingerprint != penalty_fingerprint {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML eigen cache penalty mismatch".to_string(),
        ));
    }
    workspace.validate(p, d)?;
    if coefficients.dim() != (p, d) || fitted.dim() != (n, d) || sigma2.len() != d {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML no-alloc output shape mismatch: expected coefficients=({p},{d}), fitted=({n},{d}), sigma2={d}"
        )));
    }
    if let Some(lambda) = init_lambda {
        validate_initial_lambda(lambda)?;
    }

    fill_weighted_rhs_no_alloc(x, y, weights, workspace)?;
    project_rhs_no_alloc(eigen_cache, workspace);

    let init_rho = init_lambda.map(f64::ln);
    let rho = optimize_rho_no_alloc(
        eigen_cache,
        workspace.ywy.view(),
        workspace.projected_rhs_squared.view(),
        n,
        d,
        init_rho,
    )?;
    let eval = evaluate_reml_parts(
        eigen_cache,
        workspace.ywy.view(),
        workspace.projected_rhs_squared.view(),
        n,
        d,
        rho,
    );
    let lambda = rho.exp();
    fill_coefficients_no_alloc(eigen_cache, workspace, lambda, coefficients.view_mut());
    fill_fitted_no_alloc(x, coefficients.view(), fitted.view_mut());
    fill_sigma2_no_alloc(
        eigen_cache,
        workspace.ywy.view(),
        workspace.projected_rhs_squared.view(),
        n,
        d,
        lambda,
        sigma2.view_mut(),
    );
    let (reml_grad_lambda, reml_hess_lambda) =
        rho_derivatives_to_lambda(lambda, eval.grad, eval.hess);
    Ok(GaussianRemlNoAllocFit {
        lambda,
        rho,
        reml_score: eval.cost,
        reml_grad_lambda,
        reml_hess_lambda,
        reml_grad_rho: eval.grad,
        reml_hess_rho: eval.hess,
        edf: eval.edf,
    })
}

pub fn gaussian_reml_closed_form_batch<'a>(
    problems: &[GaussianRemlScalarBatchProblem<'a>],
    penalty: ArrayView2<'a, f64>,
    nullspace_dim: Option<usize>,
) -> Result<Vec<GaussianRemlResult>, EstimationError> {
    let fits: Vec<Result<GaussianRemlResult, EstimationError>> = problems
        .par_iter()
        .map(|problem| {
            gaussian_reml_closed_form_with_nullspace_dim(
                problem.x.view(),
                problem.y.view(),
                penalty.view(),
                nullspace_dim,
                problem.weights.as_ref().map(|weights| weights.view()),
                problem.init_rho,
            )
        })
        .collect();
    fits.into_iter().collect()
}

pub fn gaussian_reml_multi_closed_form_batch<'a>(
    problems: &[GaussianRemlMultiBatchProblem<'a>],
    penalty: ArrayView2<'a, f64>,
    nullspace_dim: Option<usize>,
) -> Result<Vec<GaussianRemlMultiResult>, EstimationError> {
    let fits: Vec<Result<GaussianRemlMultiResult, EstimationError>> = problems
        .par_iter()
        .map(|problem| {
            gaussian_reml_multi_closed_form_with_nullspace_dim(
                problem.x.view(),
                problem.y.view(),
                penalty.view(),
                nullspace_dim,
                problem.weights.as_ref().map(|weights| weights.view()),
                problem.init_rho,
            )
        })
        .collect();
    fits.into_iter().collect()
}

fn gaussian_reml_multi_closed_form_from_parts(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    let prepared = prepare_gaussian_reml(x, y, penalty, nullspace_dim, weights, eigen_cache)?;
    let init_rho = init_lambda
        .map(validate_initial_lambda)
        .transpose()?
        .map(f64::ln);
    let rho = optimize_rho(&prepared, init_rho)?;
    let eval = prepared.evaluate(rho);
    let lambda = rho.exp();
    let coefficients = prepared.coefficients(lambda);
    let fitted = dense_ab(x, coefficients.view());
    let sigma2 = prepared.sigma2(lambda);
    let (reml_grad_lambda, reml_hess_lambda) =
        rho_derivatives_to_lambda(lambda, eval.grad, eval.hess);
    Ok(GaussianRemlMultiResult {
        lambda,
        rho,
        coefficients,
        fitted,
        reml_score: eval.cost,
        reml_grad_lambda,
        reml_hess_lambda,
        reml_grad_rho: eval.grad,
        reml_hess_rho: eval.hess,
        edf: eval.edf,
        sigma2,
        cache: prepared.cache,
    })
}

pub fn gaussian_reml_score_derivatives(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    lambda: f64,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlScoreDerivatives, EstimationError> {
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML lambda must be finite and positive; got {lambda}"
        )));
    }
    let prepared = prepare_gaussian_reml(x, y, penalty, None, weights, None)?;
    let eval = prepared.evaluate(lambda.ln());
    let coefficients = prepared.coefficients(lambda);
    let fitted = dense_ab(x, coefficients.view());
    let sigma2 = prepared.sigma2(lambda);
    let (grad_lambda, hess_lambda) = rho_derivatives_to_lambda(lambda, eval.grad, eval.hess);
    Ok(GaussianRemlScoreDerivatives {
        reml_score: eval.cost,
        grad_lambda,
        hess_lambda,
        coefficients,
        fitted,
        sigma2,
        edf: eval.edf,
    })
}

pub fn gaussian_reml_multi_closed_form_backward(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    let fit =
        gaussian_reml_multi_closed_form_with_cache(x, y, penalty, weights, init_lambda, None)?;
    gaussian_reml_multi_closed_form_backward_from_fit(
        x,
        y,
        penalty,
        weights,
        &fit,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gaussian_reml_multi_closed_form_backward_from_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &GaussianRemlMultiResult,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    validate_gaussian_reml_backward_upstreams(
        x,
        y,
        penalty,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
    )?;
    validate_gaussian_reml_forward_fit(x, y, penalty, weights, fit)?;
    let lambda = fit.lambda;
    if !(fit.reml_hess_rho.is_finite() && fit.reml_hess_rho.abs() > 1.0e-14) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    let weight = gaussian_reml_weights(n, weights)?;
    let inverse_hessian = gaussian_reml_inverse_hessian_from_cache(&fit.cache, lambda)?;
    let beta = &fit.coefficients;
    let residual = y.to_owned() - &fit.fitted;
    let nu = n as f64 - fit.cache.nullity as f64;

    let mut grad_x = Array2::<f64>::zeros((n, p));
    let mut grad_y = Array2::<f64>::zeros((n, d));
    let mut grad_weights = Array1::<f64>::zeros(n);

    let mut upstream_beta = Array2::<f64>::zeros((p, d));
    if let Some(upstream_coefficients) = upstream_coefficients {
        upstream_beta += &upstream_coefficients;
    }
    if let Some(upstream_fitted) = upstream_fitted {
        upstream_beta += &dense_atb(x, upstream_fitted);
        grad_x += &dense_ab(upstream_fitted, beta.t());
    }

    let mut lambda_adjoint = upstream_lambda;
    if upstream_beta.iter().any(|value| *value != 0.0) {
        lambda_adjoint += add_ridge_profile_vjp(
            1.0,
            x,
            y,
            penalty,
            &weight,
            &inverse_hessian,
            beta,
            upstream_beta.view(),
            &mut grad_x,
            &mut grad_y,
            &mut grad_weights,
        );
    }

    if upstream_reml_score != 0.0 {
        add_reml_score_vjp(
            upstream_reml_score,
            x,
            &weight,
            &inverse_hessian,
            beta,
            &residual,
            &fit.sigma2,
            nu,
            &mut grad_x,
            &mut grad_y,
            &mut grad_weights,
        );
        lambda_adjoint += upstream_reml_score * fit.reml_grad_lambda;
    }

    if lambda_adjoint != 0.0 {
        let root_scale = -lambda_adjoint * lambda / fit.reml_hess_rho;
        add_reml_rho_gradient_vjp(
            root_scale,
            x,
            y,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            beta,
            &residual,
            &fit.sigma2,
            nu,
            &mut grad_x,
            &mut grad_y,
            &mut grad_weights,
        );
    }

    Ok(GaussianRemlBackwardResult {
        grad_x,
        grad_y,
        grad_weights,
    })
}

pub fn gaussian_reml_multi_closed_form_backward_batch<'a>(
    problems: &[GaussianRemlMultiBackwardProblem<'a>],
    penalty: ArrayView2<'a, f64>,
) -> Result<Vec<GaussianRemlBackwardResult>, EstimationError> {
    let results: Vec<Result<GaussianRemlBackwardResult, EstimationError>> = problems
        .par_iter()
        .map(|problem| {
            gaussian_reml_multi_closed_form_backward_from_fit(
                problem.x.view(),
                problem.y.view(),
                penalty,
                problem.weights.as_ref().map(|weights| weights.view()),
                problem.fit,
                problem.grad_lambda,
                problem.grad_coefficients.as_ref().map(|grad| grad.view()),
                problem.grad_fitted.as_ref().map(|grad| grad.view()),
                problem.grad_reml_score,
            )
        })
        .collect();
    results.into_iter().collect()
}

fn rho_derivatives_to_lambda(lambda: f64, grad_rho: f64, hess_rho: f64) -> (f64, f64) {
    (grad_rho / lambda, (hess_rho - grad_rho) / (lambda * lambda))
}

fn validate_gaussian_reml_backward_upstreams(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
) -> Result<(), EstimationError> {
    if !(upstream_lambda.is_finite() && upstream_reml_score.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML backward upstream scalars must be finite".to_string(),
        ));
    }
    if let Some(upstream_coefficients) = upstream_coefficients {
        if upstream_coefficients.dim() != (x.ncols(), y.ncols()) {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML backward coefficient upstream shape mismatch: expected {}x{}, got {}x{}",
                x.ncols(),
                y.ncols(),
                upstream_coefficients.nrows(),
                upstream_coefficients.ncols()
            )));
        }
        if upstream_coefficients.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(
                "Gaussian REML backward coefficient upstream must be finite".to_string(),
            ));
        }
    }
    if let Some(upstream_fitted) = upstream_fitted {
        if upstream_fitted.dim() != y.dim() {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML backward fitted upstream shape mismatch: expected {}x{}, got {}x{}",
                y.nrows(),
                y.ncols(),
                upstream_fitted.nrows(),
                upstream_fitted.ncols()
            )));
        }
        if upstream_fitted.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(
                "Gaussian REML backward fitted upstream must be finite".to_string(),
            ));
        }
    }
    validate_gaussian_reml_design(x, penalty, None)?;
    Ok(())
}

fn validate_gaussian_reml_forward_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &GaussianRemlMultiResult,
) -> Result<(), EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    validate_gaussian_reml_eigen_cache(&fit.cache, p)?;
    if y.nrows() != n
        || fit.coefficients.dim() != (p, d)
        || fit.fitted.dim() != (n, d)
        || fit.sigma2.len() != d
    {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML backward forward-state shape mismatch: expected coefficients=({p},{d}), fitted=({n},{d}), sigma2={d}"
        )));
    }
    if !(fit.lambda.is_finite()
        && fit.lambda > 0.0
        && fit.rho.is_finite()
        && fit.reml_score.is_finite()
        && fit.reml_hess_rho.is_finite()
        && fit.edf.is_finite())
        || fit.coefficients.iter().any(|value| !value.is_finite())
        || fit.fitted.iter().any(|value| !value.is_finite())
        || fit.sigma2.iter().any(|value| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML backward forward state must be finite".to_string(),
        ));
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    if fit.cache.penalty_fingerprint != penalty_fingerprint {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML backward forward-state penalty mismatch".to_string(),
        ));
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let xtwx = dense_xt_diag_x(x, weight.view());
    if fit.cache.xtwx_fingerprint != matrix_fingerprint(xtwx.view()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML backward forward-state X'WX mismatch".to_string(),
        ));
    }
    Ok(())
}

fn gaussian_reml_inverse_hessian_from_cache(
    cache: &GaussianRemlEigenCache,
    lambda: f64,
) -> Result<Array2<f64>, EstimationError> {
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML lambda must be finite and positive; got {lambda}"
        )));
    }
    let p = cache.penalty_eigenvalues.len();
    let mut scaled_basis = cache.coefficient_basis.clone();
    for eig in 0..p {
        let scale = 1.0 / (1.0 + lambda * cache.penalty_eigenvalues[eig]);
        for row in 0..p {
            scaled_basis[[row, eig]] *= scale;
        }
    }
    let inverse = dense_ab(scaled_basis.view(), cache.coefficient_basis.t());
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok(inverse)
}

fn add_ridge_profile_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> f64 {
    let m = dense_ab(inverse_hessian.view(), upstream_beta);
    let c = dense_ab(m.view(), beta.t());
    let c_sym = &c + &c.t();
    let ymt = dense_ab(y, m.t());
    let xcs = dense_ab(x, c_sym.view());
    for i in 0..x.nrows() {
        let wi = weights[i] * scale;
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * (ymt[[i, k]] - xcs[[i, k]]);
        }
    }

    let xm = dense_ab(x, m.view());
    for i in 0..x.nrows() {
        let wi = weights[i] * scale;
        for j in 0..y.ncols() {
            grad_y[[i, j]] += wi * xm[[i, j]];
        }
    }

    let xc = dense_ab(x, c.view());
    for i in 0..x.nrows() {
        let mut from_b = 0.0;
        for j in 0..y.ncols() {
            from_b += y[[i, j]] * xm[[i, j]];
        }
        let mut from_a = 0.0;
        for k in 0..x.ncols() {
            from_a += x[[i, k]] * xc[[i, k]];
        }
        grad_weights[i] += scale * (from_b - from_a);
    }

    let penalty_beta = dense_ab(penalty, beta.view());
    -scale
        * m.iter()
            .zip(penalty_beta.iter())
            .map(|(left, right)| left * right)
            .sum::<f64>()
}

fn add_reml_score_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    sigma2: &Array1<f64>,
    nu: f64,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    let d = beta.ncols() as f64;
    let xp = dense_ab(x, inverse_hessian.view());
    for i in 0..x.nrows() {
        let wi = weights[i] * scale * d;
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * xp[[i, k]];
        }
        let mut leverage = 0.0;
        for k in 0..x.ncols() {
            leverage += x[[i, k]] * xp[[i, k]];
        }
        grad_weights[i] += scale * 0.5 * d * leverage;
    }

    for j in 0..beta.ncols() {
        let dp = (sigma2[j] * nu).max(MIN_DEVIANCE);
        let coef = scale * 0.5 * nu / dp;
        add_deviance_profile_vjp(
            coef,
            j,
            x,
            weights,
            beta,
            residual,
            grad_x,
            grad_y,
            grad_weights,
        );
    }
}

fn add_reml_rho_gradient_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    sigma2: &Array1<f64>,
    nu: f64,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    let d = beta.ncols() as f64;
    let k_matrix = penalty.to_owned() * lambda;
    let inverse_k = dense_ab(inverse_hessian.view(), k_matrix.view());
    let trace_kernel = dense_ab(inverse_k.view(), inverse_hessian.view());
    let xt = dense_ab(x, trace_kernel.view());
    for i in 0..x.nrows() {
        let wi = -scale * d * weights[i];
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * xt[[i, k]];
        }
        let mut quad = 0.0;
        for k in 0..x.ncols() {
            quad += x[[i, k]] * xt[[i, k]];
        }
        grad_weights[i] -= scale * 0.5 * d * quad;
    }

    let k_beta = dense_ab(k_matrix.view(), beta.view());
    let mut upstream_beta = Array2::<f64>::zeros(beta.dim());
    for j in 0..beta.ncols() {
        let dp = (sigma2[j] * nu).max(MIN_DEVIANCE);
        let q = beta.column(j).dot(&k_beta.column(j));
        let q_coef = scale * nu / dp;
        for row in 0..beta.nrows() {
            upstream_beta[[row, j]] = q_coef * k_beta[[row, j]];
        }
        let dp_coef = -scale * 0.5 * nu * q / (dp * dp);
        add_deviance_profile_vjp(
            dp_coef,
            j,
            x,
            weights,
            beta,
            residual,
            grad_x,
            grad_y,
            grad_weights,
        );
    }
    let _ = add_ridge_profile_vjp(
        1.0,
        x,
        y,
        penalty,
        weights,
        inverse_hessian,
        beta,
        upstream_beta.view(),
        grad_x,
        grad_y,
        grad_weights,
    );
}

fn add_deviance_profile_vjp(
    scale: f64,
    output: usize,
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    for i in 0..x.nrows() {
        let r = residual[[i, output]];
        let wr_scale = scale * weights[i] * r;
        grad_y[[i, output]] += 2.0 * wr_scale;
        for k in 0..x.ncols() {
            grad_x[[i, k]] -= 2.0 * wr_scale * beta[[k, output]];
        }
        grad_weights[i] += scale * r * r;
    }
}

fn validate_initial_lambda(lambda: f64) -> Result<f64, EstimationError> {
    if lambda.is_finite() && lambda > 0.0 {
        Ok(lambda)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "Gaussian REML initial lambda must be finite and positive; got {lambda}"
        )))
    }
}

fn dense_ab(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    fast_ab(&a, &b)
}

fn dense_atb(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    fast_atb(&a, &b)
}

fn dense_xt_diag_x(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Array2<f64> {
    fast_xt_diag_x(&x, &w)
}

fn dense_xt_diag_y(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Array2<f64> {
    fast_xt_diag_y(&x, &w, &y)
}

fn matrix_fingerprint(matrix: ArrayView2<'_, f64>) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    hash = fnv1a_mix(hash, matrix.nrows() as u64);
    hash = fnv1a_mix(hash, matrix.ncols() as u64);
    for &value in matrix {
        hash = fnv1a_mix(hash, value.to_bits());
    }
    hash
}

fn fnv1a_mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x100000001b3)
}

pub fn build_gaussian_reml_eigen_cache(
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    build_gaussian_reml_eigen_cache_with_nullspace_dim(x, penalty, None, weights)
}

pub fn build_gaussian_reml_eigen_cache_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let n = x.nrows();
    validate_gaussian_reml_design(x, penalty, weights)?;
    let weight = gaussian_reml_weights(n, weights)?;

    let xtwx = dense_xt_diag_x(x, weight.view());
    gaussian_reml_eigen_cache_from_xtwx(xtwx, penalty, nullspace_dim)
}

fn validate_gaussian_reml_design(
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(), EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        )));
    }
    if x.iter().chain(penalty.iter()).any(|v| !v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML inputs must be finite".to_string(),
        ));
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML weights length mismatch: expected {n}, got {}",
                w.len()
            )));
        }
        if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(EstimationError::InvalidInput(
                "Gaussian REML weights must be finite and non-negative".to_string(),
            ));
        }
    }
    Ok(())
}

fn gaussian_reml_weights(
    n: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    match weights {
        Some(w) => {
            if w.len() != n {
                return Err(EstimationError::InvalidInput(format!(
                    "Gaussian REML weights length mismatch: expected {n}, got {}",
                    w.len()
                )));
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(EstimationError::InvalidInput(
                    "Gaussian REML weights must be finite and non-negative".to_string(),
                ));
            }
            Ok(w.to_owned())
        }
        None => Ok(Array1::ones(n)),
    }
}

fn gaussian_reml_eigen_cache_from_xtwx(
    xtwx: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let p = xtwx.nrows();
    if xtwx.ncols() != p {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML X'WX must be square".to_string(),
        ));
    }
    let xtwx_fingerprint = matrix_fingerprint(xtwx.view());
    let penalty_fingerprint = matrix_fingerprint(penalty);
    let chol = xtwx
        .cholesky(Side::Lower)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    let lower = chol.lower_triangular();
    let logdet_xtwx = 2.0 * lower.diag().iter().map(|v| v.ln()).sum::<f64>();
    let l_inv = invert_lower_triangular(&lower)?;
    let penalty_in_metric = dense_ab(l_inv.view(), penalty);
    let transformed_penalty = dense_ab(penalty_in_metric.view(), l_inv.t());
    let (mut penalty_eigenvalues, eigenvectors) =
        transformed_penalty.eigh(Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?;
    // Rank tolerance must be RELATIVE to the largest eigenvalue — never
    // floored at an absolute value. The old `.max(1.0)` clamped the
    // tolerance up whenever max|eig| < 1, classifying genuine modes as
    // null for small-scale penalties (e.g. Wahba pseudo-spline `m=4`
    // with `K(p,p) ≈ 3e-4`). That broke REML's invariance under
    // `S → c·S` — the optimum λ rescales but the score landscape
    // diverges from the true marginal likelihood, and the smooth
    // contribution collapsed to ~0 on smooth truths.
    // Fully scale-invariant form: `safety · max|eig| · eps`.
    let max_abs_eig = penalty_eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let eig_tol = max_abs_eig * EIGEN_REL_TOL;
    for value in &mut penalty_eigenvalues {
        if *value < 0.0 && value.abs() <= eig_tol {
            *value = 0.0;
        }
        if *value < 0.0 {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML penalty is not positive semidefinite; eigenvalue={value:.3e}"
            )));
        }
    }
    let penalty_rank = penalty_eigenvalues
        .iter()
        .filter(|&&value| value > eig_tol)
        .count();
    let nullity = p - penalty_rank;
    if let Some(expected_nullity) = nullspace_dim {
        if expected_nullity != nullity {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML penalty nullspace mismatch: expected {expected_nullity}, inferred {nullity}"
            )));
        }
    }
    let logdet_penalty_positive = gaussian_penalty_positive_logdet(penalty, penalty_rank)?;
    let coefficient_basis = solve_upper_triangular_matrix(&lower.t().to_owned(), &eigenvectors)?;

    Ok(GaussianRemlEigenCache {
        penalty_eigenvalues,
        eigenvectors,
        coefficient_basis,
        xtwx_fingerprint,
        penalty_fingerprint,
        logdet_xtwx,
        logdet_penalty_positive,
        penalty_rank,
        nullity,
    })
}

fn gaussian_penalty_positive_logdet(
    penalty: ArrayView2<'_, f64>,
    penalty_rank: usize,
) -> Result<f64, EstimationError> {
    if penalty_rank == 0 {
        return Ok(0.0);
    }
    let (pen_eigs, _) = penalty.to_owned().eigh(Side::Lower).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    // Scale-invariant relative tolerance — see the cousin site for the
    // rationale. Same `.max(1.0)` floor used to live here and corrupted
    // the positive-eigenvalue count for small-scale penalties.
    let pen_scale = pen_eigs
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let pen_tol = pen_scale * EIGEN_REL_TOL;
    let mut positive_eigs: Vec<f64> = pen_eigs
        .iter()
        .copied()
        .filter(|&value| value > pen_tol)
        .collect();
    if positive_eigs.len() != penalty_rank {
        positive_eigs = pen_eigs
            .iter()
            .copied()
            .filter(|&value| value > 0.0)
            .collect();
        positive_eigs.sort_by(|a, b| b.total_cmp(a));
        if positive_eigs.len() < penalty_rank {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        positive_eigs.truncate(penalty_rank);
    }
    Ok(positive_eigs.iter().map(|value| value.ln()).sum())
}

fn validate_gaussian_reml_eigen_cache(
    cache: &GaussianRemlEigenCache,
    p: usize,
) -> Result<(), EstimationError> {
    if cache.penalty_eigenvalues.len() != p
        || cache.eigenvectors.dim() != (p, p)
        || cache.coefficient_basis.dim() != (p, p)
    {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML eigen cache dimension mismatch: expected {p} coefficients"
        )));
    }
    if cache.penalty_rank > p || cache.nullity > p || cache.penalty_rank + cache.nullity != p {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML eigen cache rank/nullity mismatch: rank={}, nullity={}, p={p}",
            cache.penalty_rank, cache.nullity
        )));
    }
    if !(cache.logdet_xtwx.is_finite() && cache.logdet_penalty_positive.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML eigen cache log-determinants must be finite".to_string(),
        ));
    }
    if cache
        .penalty_eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || cache.eigenvectors.iter().any(|value| !value.is_finite())
        || cache
            .coefficient_basis
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML eigen cache entries must be finite with non-negative eigenvalues"
                .to_string(),
        ));
    }
    Ok(())
}

fn prepare_gaussian_reml(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlPrepared, EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    if y.nrows() != n {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        )));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML inputs must be finite".to_string(),
        ));
    }
    let weight = gaussian_reml_weights(n, weights)?;

    let xtwy = dense_xt_diag_y(x, weight.view(), y);
    let ywy = Array1::from_iter((0..d).map(|j| {
        let mut value = 0.0;
        for row in 0..n {
            value += weight[row] * y[[row, j]] * y[[row, j]];
        }
        value
    }));
    let xtwx = dense_xt_diag_x(x, weight.view());

    if let Some(cache) = eigen_cache {
        validate_gaussian_reml_eigen_cache(cache, p)?;
        let xtwx_fingerprint = matrix_fingerprint(xtwx.view());
        if cache.xtwx_fingerprint != xtwx_fingerprint {
            return Err(EstimationError::InvalidInput(
                "Gaussian REML eigen cache X'WX mismatch".to_string(),
            ));
        }
        let penalty_fingerprint = matrix_fingerprint(penalty);
        if cache.penalty_fingerprint != penalty_fingerprint {
            return Err(EstimationError::InvalidInput(
                "Gaussian REML eigen cache penalty mismatch".to_string(),
            ));
        }
        if let Some(expected_nullity) = nullspace_dim {
            if expected_nullity != cache.nullity {
                return Err(EstimationError::InvalidInput(format!(
                    "Gaussian REML eigen cache nullspace mismatch: expected {expected_nullity}, got {}",
                    cache.nullity
                )));
            }
        }
        if n <= cache.nullity {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
                cache.nullity
            )));
        }
        let projected_rhs = dense_atb(cache.coefficient_basis.view(), xtwy.view());
        let projected_rhs_squared = projected_rhs.mapv(|value| value * value);
        return Ok(GaussianRemlPrepared {
            cache: cache.clone(),
            ywy,
            projected_rhs_squared,
            projected_rhs,
            n_observations: n,
            n_outputs: d,
        });
    }

    let cache = gaussian_reml_eigen_cache_from_xtwx(xtwx, penalty, nullspace_dim)?;
    if n <= cache.nullity {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
            cache.nullity
        )));
    }
    let projected_rhs = dense_atb(cache.coefficient_basis.view(), xtwy.view());
    let projected_rhs_squared = projected_rhs.mapv(|value| value * value);

    Ok(GaussianRemlPrepared {
        cache,
        ywy,
        projected_rhs_squared,
        projected_rhs,
        n_observations: n,
        n_outputs: d,
    })
}

impl GaussianRemlPrepared {
    fn nu(&self) -> f64 {
        self.n_observations as f64 - self.cache.nullity as f64
    }

    fn evaluate(&self, rho: f64) -> ObjectiveEval {
        evaluate_reml_parts(
            &self.cache,
            self.ywy.view(),
            self.projected_rhs_squared.view(),
            self.n_observations,
            self.n_outputs,
            rho,
        )
    }

    fn coefficients(&self, lambda: f64) -> Array2<f64> {
        let mut scaled = self.projected_rhs.clone();
        for i in 0..self.cache.penalty_eigenvalues.len() {
            let scale = 1.0 / (1.0 + lambda * self.cache.penalty_eigenvalues[i]);
            for value in scaled.row_mut(i) {
                *value *= scale;
            }
        }
        dense_ab(self.cache.coefficient_basis.view(), scaled.view())
    }

    fn sigma2(&self, lambda: f64) -> Array1<f64> {
        let nu = self.nu();
        Array1::from_iter((0..self.n_outputs).map(|j| {
            let mut fitted_quadratic = 0.0;
            for i in 0..self.cache.penalty_eigenvalues.len() {
                let denom = 1.0 + lambda * self.cache.penalty_eigenvalues[i];
                fitted_quadratic += self.projected_rhs_squared[[i, j]] / denom;
            }
            ((self.ywy[j] - fitted_quadratic).max(MIN_DEVIANCE)) / nu
        }))
    }
}

fn optimize_rho(
    prepared: &GaussianRemlPrepared,
    init_rho: Option<f64>,
) -> Result<f64, EstimationError> {
    if prepared.cache.penalty_rank == 0 {
        return Ok(init_rho.unwrap_or(0.0).clamp(RHO_LOWER, RHO_UPPER));
    }

    const GRID_INTERVALS: usize = 96;
    let mut stationary = Vec::<f64>::new();
    let mut prev_rho = RHO_LOWER;
    let mut prev_eval = prepared.evaluate(prev_rho);
    for i in 1..=GRID_INTERVALS {
        let rho = RHO_LOWER + (RHO_UPPER - RHO_LOWER) * (i as f64) / (GRID_INTERVALS as f64);
        let eval = prepared.evaluate(rho);
        if prev_eval.grad <= 0.0 && eval.grad >= 0.0 {
            push_candidate(
                &mut stationary,
                refine_stationary_rho(prepared, prev_rho, rho, 0.5 * (prev_rho + rho)),
            );
        }
        prev_rho = rho;
        prev_eval = eval;
    }

    // Prefer interior stationary points (smooth under X perturbation). Only fall
    // back to boundaries when no stationary point exists — this avoids a kink
    // where a grid candidate's cost crosses below a refined stationary point's
    // cost, which would make d(reml_score, lambda)/dX discontinuous and break
    // analytic-vs-FD agreement at strict tolerance.
    let candidates = if stationary.is_empty() {
        let mut boundary = Vec::<f64>::new();
        push_candidate(&mut boundary, RHO_LOWER);
        push_candidate(&mut boundary, RHO_UPPER);
        if let Some(rho0) = init_rho {
            push_candidate(&mut boundary, rho0);
        }
        boundary
    } else {
        stationary
    };

    candidates
        .into_iter()
        .min_by(|&a, &b| {
            prepared
                .evaluate(a)
                .cost
                .total_cmp(&prepared.evaluate(b).cost)
        })
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian REML optimizer produced no candidates".to_string(),
            )
        })
}

fn fill_weighted_rhs_no_alloc(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    workspace: &mut GaussianRemlNoAllocWorkspace,
) -> Result<(), EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    workspace.xtwy.fill(0.0);
    workspace.ywy.fill(0.0);

    for row in 0..n {
        let wi = weights.map_or(1.0, |w| w[row]);
        for output in 0..d {
            let weighted_y = wi * y[[row, output]];
            workspace.ywy[output] += y[[row, output]] * weighted_y;
            for col in 0..p {
                workspace.xtwy[[col, output]] += x[[row, col]] * weighted_y;
            }
        }
    }

    if workspace
        .xtwy
        .iter()
        .chain(workspace.ywy.iter())
        .any(|value| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML weighted cross-products must be finite".to_string(),
        ));
    }
    Ok(())
}

fn project_rhs_no_alloc(
    cache: &GaussianRemlEigenCache,
    workspace: &mut GaussianRemlNoAllocWorkspace,
) {
    let p = cache.penalty_eigenvalues.len();
    let d = workspace.ywy.len();
    for eig in 0..p {
        for output in 0..d {
            let mut value = 0.0;
            for col in 0..p {
                value += cache.coefficient_basis[[col, eig]] * workspace.xtwy[[col, output]];
            }
            workspace.projected_rhs[[eig, output]] = value;
            workspace.projected_rhs_squared[[eig, output]] = value * value;
        }
    }
}

fn evaluate_reml_parts(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_observations: usize,
    n_outputs: usize,
    rho: f64,
) -> ObjectiveEval {
    let lambda = rho.exp();
    let nu = n_observations as f64 - cache.nullity as f64;
    let d = n_outputs as f64;
    let mut logdet_h = cache.logdet_xtwx;
    let mut trace_h = 0.0;
    let mut trace_h_deriv = 0.0;
    let mut edf = 0.0;
    for &delta in &cache.penalty_eigenvalues {
        let t = lambda * delta;
        logdet_h += (1.0 + t).ln();
        if delta > 0.0 {
            trace_h += t / (1.0 + t);
            trace_h_deriv += t / ((1.0 + t) * (1.0 + t));
        }
        edf += 1.0 / (1.0 + t);
    }
    let logdet_s = cache.logdet_penalty_positive + (cache.penalty_rank as f64) * rho;

    let mut cost = 0.5 * d * (logdet_h - logdet_s);
    let mut grad = 0.5 * d * (trace_h - cache.penalty_rank as f64);
    let mut hess = 0.5 * d * trace_h_deriv;
    for output in 0..n_outputs {
        let mut fitted_quadratic = 0.0;
        let mut dp_grad = 0.0;
        let mut dp_hess = 0.0;
        for eig in 0..cache.penalty_eigenvalues.len() {
            let c2 = projected_rhs_squared[[eig, output]];
            let t = lambda * cache.penalty_eigenvalues[eig];
            let denom = 1.0 + t;
            fitted_quadratic += c2 / denom;
            dp_grad += c2 * t / (denom * denom);
            dp_hess += c2 * t * (1.0 - t) / (denom * denom * denom);
        }
        let dp = (ywy[output] - fitted_quadratic).max(MIN_DEVIANCE);
        cost += 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
        grad += 0.5 * nu * dp_grad / dp;
        hess += 0.5 * nu * (dp_hess / dp - (dp_grad * dp_grad) / (dp * dp));
    }
    ObjectiveEval {
        cost,
        grad,
        hess,
        edf,
    }
}

fn optimize_rho_no_alloc(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_observations: usize,
    n_outputs: usize,
    init_rho: Option<f64>,
) -> Result<f64, EstimationError> {
    if cache.penalty_rank == 0 {
        return Ok(init_rho.unwrap_or(0.0).clamp(RHO_LOWER, RHO_UPPER));
    }

    let lower_eval = evaluate_reml_parts(
        cache,
        ywy,
        projected_rhs_squared,
        n_observations,
        n_outputs,
        RHO_LOWER,
    );
    let mut best_rho = RHO_LOWER;
    let mut best_cost = lower_eval.cost;

    consider_rho_no_alloc(
        cache,
        ywy,
        projected_rhs_squared,
        n_observations,
        n_outputs,
        RHO_UPPER,
        &mut best_rho,
        &mut best_cost,
    );
    if let Some(rho0) = init_rho {
        consider_rho_no_alloc(
            cache,
            ywy,
            projected_rhs_squared,
            n_observations,
            n_outputs,
            rho0,
            &mut best_rho,
            &mut best_cost,
        );
    }

    const GRID_INTERVALS: usize = 96;
    let mut prev_rho = RHO_LOWER;
    let mut prev_eval = lower_eval;
    for i in 1..=GRID_INTERVALS {
        let rho = RHO_LOWER + (RHO_UPPER - RHO_LOWER) * (i as f64) / (GRID_INTERVALS as f64);
        let eval = evaluate_reml_parts(
            cache,
            ywy,
            projected_rhs_squared,
            n_observations,
            n_outputs,
            rho,
        );
        if eval.cost < best_cost {
            best_rho = rho;
            best_cost = eval.cost;
        }
        if prev_eval.grad <= 0.0 && eval.grad >= 0.0 {
            let stationary = refine_stationary_rho_no_alloc(
                cache,
                ywy,
                projected_rhs_squared,
                n_observations,
                n_outputs,
                prev_rho,
                rho,
                0.5 * (prev_rho + rho),
            );
            consider_rho_no_alloc(
                cache,
                ywy,
                projected_rhs_squared,
                n_observations,
                n_outputs,
                stationary,
                &mut best_rho,
                &mut best_cost,
            );
        }
        prev_rho = rho;
        prev_eval = eval;
    }

    if best_cost.is_finite() {
        Ok(best_rho)
    } else {
        Err(EstimationError::InvalidInput(
            "Gaussian REML optimizer produced no finite candidates".to_string(),
        ))
    }
}

fn consider_rho_no_alloc(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_observations: usize,
    n_outputs: usize,
    rho: f64,
    best_rho: &mut f64,
    best_cost: &mut f64,
) {
    if !rho.is_finite() {
        return;
    }
    let candidate = rho.clamp(RHO_LOWER, RHO_UPPER);
    let eval = evaluate_reml_parts(
        cache,
        ywy,
        projected_rhs_squared,
        n_observations,
        n_outputs,
        candidate,
    );
    if eval.cost < *best_cost {
        *best_rho = candidate;
        *best_cost = eval.cost;
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_stationary_rho_no_alloc(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_observations: usize,
    n_outputs: usize,
    mut lo: f64,
    mut hi: f64,
    mut rho: f64,
) -> f64 {
    for _ in 0..80 {
        let eval = evaluate_reml_parts(
            cache,
            ywy,
            projected_rhs_squared,
            n_observations,
            n_outputs,
            rho,
        );
        if eval.grad.abs() <= GRAD_TOL * (1.0 + eval.cost.abs()) {
            return rho;
        }
        if eval.grad >= 0.0 {
            hi = rho;
        } else {
            lo = rho;
        }
        let newton = if eval.hess > 0.0 {
            let candidate = rho - eval.grad / eval.hess;
            (candidate > lo && candidate < hi).then_some(candidate)
        } else {
            None
        };
        if (hi - lo).abs() <= 1e-12 * (1.0 + rho.abs()) {
            break;
        }
        rho = newton.unwrap_or(0.5 * (lo + hi));
    }
    0.5 * (lo + hi)
}

fn fill_coefficients_no_alloc(
    cache: &GaussianRemlEigenCache,
    workspace: &mut GaussianRemlNoAllocWorkspace,
    lambda: f64,
    mut coefficients: ArrayViewMut2<'_, f64>,
) {
    let p = cache.penalty_eigenvalues.len();
    let d = workspace.ywy.len();
    for eig in 0..p {
        let scale = 1.0 / (1.0 + lambda * cache.penalty_eigenvalues[eig]);
        for output in 0..d {
            workspace.scaled_projected_rhs[[eig, output]] =
                workspace.projected_rhs[[eig, output]] * scale;
        }
    }

    for col in 0..p {
        for output in 0..d {
            let mut value = 0.0;
            for eig in 0..p {
                value += cache.coefficient_basis[[col, eig]]
                    * workspace.scaled_projected_rhs[[eig, output]];
            }
            coefficients[[col, output]] = value;
        }
    }
}

fn fill_fitted_no_alloc(
    x: ArrayView2<'_, f64>,
    coefficients: ArrayView2<'_, f64>,
    mut fitted: ArrayViewMut2<'_, f64>,
) {
    let n = x.nrows();
    let p = x.ncols();
    let d = coefficients.ncols();
    for row in 0..n {
        for output in 0..d {
            let mut value = 0.0;
            for col in 0..p {
                value += x[[row, col]] * coefficients[[col, output]];
            }
            fitted[[row, output]] = value;
        }
    }
}

fn fill_sigma2_no_alloc(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_observations: usize,
    n_outputs: usize,
    lambda: f64,
    mut sigma2: ArrayViewMut1<'_, f64>,
) {
    let nu = n_observations as f64 - cache.nullity as f64;
    for output in 0..n_outputs {
        let mut fitted_quadratic = 0.0;
        for eig in 0..cache.penalty_eigenvalues.len() {
            let denom = 1.0 + lambda * cache.penalty_eigenvalues[eig];
            fitted_quadratic += projected_rhs_squared[[eig, output]] / denom;
        }
        sigma2[output] = ((ywy[output] - fitted_quadratic).max(MIN_DEVIANCE)) / nu;
    }
}

fn push_candidate(candidates: &mut Vec<f64>, rho: f64) {
    if rho.is_finite() {
        candidates.push(rho.clamp(RHO_LOWER, RHO_UPPER));
    }
}

fn refine_stationary_rho(
    prepared: &GaussianRemlPrepared,
    mut lo: f64,
    mut hi: f64,
    mut rho: f64,
) -> f64 {
    for _ in 0..80 {
        let eval = prepared.evaluate(rho);
        if eval.grad.abs() <= GRAD_TOL * (1.0 + eval.cost.abs()) {
            return rho;
        }
        if eval.grad >= 0.0 {
            hi = rho;
        } else {
            lo = rho;
        }
        let newton = if eval.hess > 0.0 {
            let candidate = rho - eval.grad / eval.hess;
            (candidate > lo && candidate < hi).then_some(candidate)
        } else {
            None
        };
        if (hi - lo).abs() <= 1e-12 * (1.0 + rho.abs()) {
            break;
        }
        rho = newton.unwrap_or(0.5 * (lo + hi));
    }
    0.5 * (lo + hi)
}

fn invert_lower_triangular(lower: &Array2<f64>) -> Result<Array2<f64>, EstimationError> {
    let n = lower.nrows();
    if lower.ncols() != n {
        return Err(EstimationError::InvalidInput(
            "lower-triangular solve requires a square matrix".to_string(),
        ));
    }
    let eye = Array2::eye(n);
    solve_lower_triangular_matrix(lower, &eye)
}

fn solve_lower_triangular_matrix(
    lower: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let n = lower.nrows();
    if lower.ncols() != n || rhs.nrows() != n {
        return Err(EstimationError::InvalidInput(
            "lower-triangular solve dimension mismatch".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros(rhs.dim());
    for col in 0..rhs.ncols() {
        for i in 0..n {
            let mut value = rhs[[i, col]];
            for k in 0..i {
                value -= lower[[i, k]] * out[[k, col]];
            }
            let diag = lower[[i, i]];
            if !(diag.is_finite() && diag.abs() > 0.0) {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
            out[[i, col]] = value / diag;
        }
    }
    Ok(out)
}

fn solve_upper_triangular_matrix(
    upper: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let n = upper.nrows();
    if upper.ncols() != n || rhs.nrows() != n {
        return Err(EstimationError::InvalidInput(
            "upper-triangular solve dimension mismatch".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros(rhs.dim());
    for col in 0..rhs.ncols() {
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut value = rhs[[i, col]];
            for k in (i + 1)..n {
                value -= upper[[i, k]] * out[[k, col]];
            }
            let diag = upper[[i, i]];
            if !(diag.is_finite() && diag.abs() > 0.0) {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
            out[[i, col]] = value / diag;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn edf_does_not_double_count_penalty_nullspace() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
        let y = array![[0.0], [1.0], [1.8], [3.2], [4.1]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let result =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("small full-rank Gaussian REML fit");

        assert!(result.edf >= result.cache.nullity as f64);
        assert!(result.edf <= x.ncols() as f64 + 1.0e-10);
    }

    #[test]
    fn multi_output_duplicate_columns_match_scalar_fit() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
        ];
        let y1 = array![0.5, 0.2, 0.0, 0.3, 1.1, 2.0];
        let y = Array2::from_shape_fn(
            (y1.len(), 2),
            |(i, j)| if j == 0 { y1[i] } else { 2.0 * y1[i] },
        );
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        let scalar =
            gaussian_reml_closed_form(x.view(), y1.view(), penalty.view(), None, Some(0.0))
                .expect("scalar Gaussian REML fit");
        let multi =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("multi-output Gaussian REML fit");

        assert!((multi.rho - scalar.rho).abs() <= 1.0e-8);
        for i in 0..x.ncols() {
            assert!((multi.coefficients[[i, 0]] - scalar.coefficients[i]).abs() <= 1.0e-8);
            assert!((multi.coefficients[[i, 1]] - 2.0 * scalar.coefficients[i]).abs() <= 1.0e-8);
        }
    }

    #[test]
    fn warm_start_reuses_cache_and_lambda_seed() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.25],
            [1.0, 0.5],
            [1.0, 1.25],
            [1.0, 2.0],
        ];
        let y = array![[0.1], [0.4], [0.7], [1.4], [2.2]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        let cold =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("cold fit");
        let warm_start = GaussianRemlWarmStart::from_multi_result(&cold);
        let warm = gaussian_reml_multi_closed_form_warm_started(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            Some(&warm_start),
        )
        .expect("warm-started fit");

        assert!((cold.lambda - warm.lambda).abs() <= 1.0e-10);
        assert_eq!(cold.cache.xtwx_fingerprint, warm.cache.xtwx_fingerprint);
        for i in 0..x.ncols() {
            assert!((cold.coefficients[[i, 0]] - warm.coefficients[[i, 0]]).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn warm_start_cache_rejects_different_penalty_geometry() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.25],
            [1.0, 0.5],
            [1.0, 1.25],
            [1.0, 2.0],
        ];
        let y = array![[0.1], [0.4], [0.7], [1.4], [2.2]];
        let penalty_a = array![[0.0, 0.0], [0.0, 1.0]];
        let penalty_b = array![[1.0, -1.0], [-1.0, 1.0]];

        let first =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty_a.view(), None, Some(0.0))
                .expect("first fit");
        let warm_start = GaussianRemlWarmStart::from_multi_result(&first);
        let err = gaussian_reml_multi_closed_form_warm_started(
            x.view(),
            y.view(),
            penalty_b.view(),
            None,
            Some(&warm_start),
        )
        .expect_err("penalty-mismatched cache must be rejected");

        assert!(err.to_string().contains("penalty mismatch"));
    }

    #[test]
    fn no_alloc_cache_path_matches_allocating_fit() {
        let x = array![
            [1.0, -1.0, 0.25],
            [1.0, -0.5, 0.10],
            [1.0, 0.0, -0.20],
            [1.0, 0.5, -0.05],
            [1.0, 1.0, 0.30],
            [1.0, 1.5, 0.60],
        ];
        let y = array![
            [0.0, 0.2],
            [0.3, 0.1],
            [0.4, -0.1],
            [0.9, 0.3],
            [1.6, 0.8],
            [2.2, 1.2],
        ];
        let weights = array![1.0, 0.8, 1.2, 1.1, 0.9, 1.3];
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]];

        let allocating = gaussian_reml_multi_closed_form_with_cache(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(1.0),
            None,
        )
        .expect("allocating fit");
        let mut workspace = GaussianRemlNoAllocWorkspace::new(x.ncols(), y.ncols());
        let mut coefficients = Array2::zeros((x.ncols(), y.ncols()));
        let mut fitted = Array2::zeros(y.dim());
        let mut sigma2 = Array1::zeros(y.ncols());

        let no_alloc = gaussian_reml_multi_closed_form_with_cache_no_alloc(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(allocating.lambda),
            &allocating.cache,
            &mut workspace,
            coefficients.view_mut(),
            fitted.view_mut(),
            sigma2.view_mut(),
        )
        .expect("no-alloc cached fit");

        assert!((no_alloc.lambda - allocating.lambda).abs() <= 1.0e-10);
        assert!((no_alloc.reml_score - allocating.reml_score).abs() <= 1.0e-8);
        assert!((no_alloc.reml_grad_rho - allocating.reml_grad_rho).abs() <= 1.0e-8);
        assert!((no_alloc.reml_hess_rho - allocating.reml_hess_rho).abs() <= 1.0e-8);
        assert!((no_alloc.edf - allocating.edf).abs() <= 1.0e-10);
        for i in 0..x.ncols() {
            for j in 0..y.ncols() {
                assert!((coefficients[[i, j]] - allocating.coefficients[[i, j]]).abs() <= 1.0e-8);
            }
        }
        for i in 0..x.nrows() {
            for j in 0..y.ncols() {
                assert!((fitted[[i, j]] - allocating.fitted[[i, j]]).abs() <= 1.0e-8);
            }
        }
        for j in 0..y.ncols() {
            assert!((sigma2[j] - allocating.sigma2[j]).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn no_alloc_cache_path_rejects_bad_shapes_and_penalty_mismatch() {
        let x = array![[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let y = array![[0.0], [0.2], [0.9], [1.8]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let cache = build_gaussian_reml_eigen_cache(x.view(), penalty.view(), None)
            .expect("Gaussian REML cache");

        let mut bad_workspace = GaussianRemlNoAllocWorkspace::new(x.ncols(), y.ncols() + 1);
        let mut coefficients = Array2::zeros((x.ncols(), y.ncols()));
        let mut fitted = Array2::zeros(y.dim());
        let mut sigma2 = Array1::zeros(y.ncols());
        let err = gaussian_reml_multi_closed_form_with_cache_no_alloc(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            Some(1.0),
            &cache,
            &mut bad_workspace,
            coefficients.view_mut(),
            fitted.view_mut(),
            sigma2.view_mut(),
        )
        .expect_err("workspace shape mismatch must be rejected");
        assert!(err.to_string().contains("workspace shape mismatch"));

        let penalty_mismatch = array![[1.0, -1.0], [-1.0, 1.0]];
        let mut workspace = GaussianRemlNoAllocWorkspace::new(x.ncols(), y.ncols());
        let err = gaussian_reml_multi_closed_form_with_cache_no_alloc(
            x.view(),
            y.view(),
            penalty_mismatch.view(),
            None,
            Some(1.0),
            &cache,
            &mut workspace,
            coefficients.view_mut(),
            fitted.view_mut(),
            sigma2.view_mut(),
        )
        .expect_err("penalty mismatch must be rejected");
        assert!(err.to_string().contains("penalty mismatch"));
    }

    #[derive(Clone, Copy, Debug)]
    enum ForwardScalar {
        Coefficient(usize, usize),
        Fitted(usize, usize),
    }

    fn finite_difference_design() -> Array2<f64> {
        Array2::from_shape_fn((20, 5), |(row, col)| {
            let t = (row as f64 - 9.5) / 10.0;
            match col {
                0 => 1.0,
                1 => t,
                2 => 0.5 * (3.0 * t * t - 1.0),
                3 => 0.5 * (5.0 * t * t * t - 3.0 * t),
                4 => (35.0 * t.powi(4) - 30.0 * t * t + 3.0) / 8.0,
                _ => unreachable!(),
            }
        })
    }

    fn finite_difference_response(outputs: usize) -> Array2<f64> {
        Array2::from_shape_fn((20, outputs), |(row, output)| {
            let t = (row as f64 - 9.5) / 10.0;
            let phase = output as f64 + 1.0;
            0.2 + 0.25 * phase * t - 0.12 * t * t
                + (0.08 + 0.03 * phase) * (1.1 * t + 0.3 * phase).sin()
        })
    }

    fn finite_difference_penalty() -> Array2<f64> {
        Array2::from_diag(&array![0.0, 0.8, 1.2, 1.7, 2.3])
    }

    fn finite_difference_weights() -> Array1<f64> {
        Array1::from_shape_fn(20, |row| {
            let t = (row as f64 - 9.5) / 10.0;
            1.0 + 0.025 * (1.1 * t).sin() + 0.01 * t
        })
    }

    fn one_hot_objective(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> f64 {
        let fit = gaussian_reml_multi_closed_form_with_cache(
            x,
            y,
            penalty,
            Some(weights),
            Some(0.85),
            None,
        )
        .expect("finite-difference forward fit");
        match target {
            ForwardScalar::Coefficient(row, col) => fit.coefficients[[row, col]],
            ForwardScalar::Fitted(row, col) => fit.fitted[[row, col]],
        }
    }

    fn one_hot_backward(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> GaussianRemlBackwardResult {
        let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
        let mut grad_fitted = Array2::<f64>::zeros(y.dim());
        let (grad_lambda, grad_score, coefficient_upstream, fitted_upstream) = match target {
            ForwardScalar::Coefficient(row, col) => {
                grad_coefficients[[row, col]] = 1.0;
                (0.0, 0.0, Some(grad_coefficients.view()), None)
            }
            ForwardScalar::Fitted(row, col) => {
                grad_fitted[[row, col]] = 1.0;
                (0.0, 0.0, None, Some(grad_fitted.view()))
            }
        };
        gaussian_reml_multi_closed_form_backward(
            x,
            y,
            penalty,
            Some(weights),
            Some(0.85),
            grad_lambda,
            coefficient_upstream,
            fitted_upstream,
            grad_score,
        )
        .expect("analytic backward VJP")
    }

    fn assert_fd_close(label: &str, analytic: f64, finite_difference: f64) {
        let rel_tol = 1.0e-6_f64;
        let abs_tol = 1.0e-6_f64;
        let tol = abs_tol.max(rel_tol * analytic.abs().max(finite_difference.abs()));
        let diff = (analytic - finite_difference).abs();
        assert!(
            diff <= tol,
            "{label}: analytic={analytic:.12e}, finite_difference={finite_difference:.12e}, diff={diff:.3e}, tol={tol:.3e}"
        );
    }

    fn adaptive_central_difference(mut eval: impl FnMut(f64) -> f64) -> f64 {
        let steps: [f64; 5] = [1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5];
        let mut best = f64::NAN;
        let mut best_delta = f64::INFINITY;
        let mut previous: Option<f64> = None;
        for h in steps {
            let d1 = (eval(h) - eval(-h)) / (2.0 * h);
            let half_h = 0.5 * h;
            let d2 = (eval(half_h) - eval(-half_h)) / (2.0 * half_h);
            let estimate: f64 = d2 + (d2 - d1) / 3.0;
            if let Some(prev) = previous {
                let delta = (estimate - prev).abs();
                if delta < best_delta {
                    best_delta = delta;
                    best = estimate;
                }
            } else {
                best = estimate;
            }
            previous = Some(estimate);
        }
        best
    }

    fn assert_backward_matches_forward_finite_difference(outputs: usize) {
        let x = finite_difference_design();
        let y = finite_difference_response(outputs);
        let penalty = finite_difference_penalty();
        let weights = finite_difference_weights();
        let targets = [
            ForwardScalar::Coefficient(3, outputs - 1),
            ForwardScalar::Fitted(12, outputs - 1),
        ];
        for target in targets {
            let backward =
                one_hot_backward(x.view(), y.view(), penalty.view(), weights.view(), target);

            for row in 0..x.nrows() {
                for col in 0..x.ncols() {
                    let eval = |delta: f64| {
                        let mut candidate = x.clone();
                        candidate[[row, col]] += delta;
                        one_hot_objective(
                            candidate.view(),
                            y.view(),
                            penalty.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} x[{row},{col}]"),
                        backward.grad_x[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..y.nrows() {
                for col in 0..y.ncols() {
                    let eval = |delta: f64| {
                        let mut candidate = y.clone();
                        candidate[[row, col]] += delta;
                        one_hot_objective(
                            x.view(),
                            candidate.view(),
                            penalty.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} y[{row},{col}]"),
                        backward.grad_y[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..weights.len() {
                let eval = |delta: f64| {
                    let mut candidate = weights.clone();
                    candidate[row] += delta;
                    one_hot_objective(x.view(), y.view(), penalty.view(), candidate.view(), target)
                };
                let fd = adaptive_central_difference(eval);
                assert_fd_close(
                    &format!("target={target:?} weights[{row}]"),
                    backward.grad_weights[row],
                    fd,
                );
            }
        }
    }

    #[test]
    fn scalar_backward_matches_forward_finite_difference_for_all_x_y_and_weight_entries() {
        assert_backward_matches_forward_finite_difference(1);
    }

    #[test]
    fn multi_output_backward_matches_forward_finite_difference_for_all_x_y_and_weight_entries() {
        assert_backward_matches_forward_finite_difference(3);
    }

    #[test]
    fn backward_vjp_matches_finite_difference() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.3, -0.1],
            [1.0, 0.2, 0.4],
            [1.0, 0.8, 0.1],
            [1.0, 1.4, 0.5],
            [1.0, 2.0, 0.9],
        ];
        let y = array![
            [0.1, -0.2],
            [0.2, 0.1],
            [0.7, 0.0],
            [1.1, 0.3],
            [1.8, 0.9],
            [2.4, 1.4],
        ];
        let weights = array![1.0, 0.9, 1.1, 1.2, 0.8, 1.3];
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.2, 1.7]];
        let upstream_coefficients = array![[0.2, -0.1], [0.05, 0.03], [-0.04, 0.07]];
        let upstream_fitted = array![
            [0.01, -0.02],
            [0.03, 0.01],
            [-0.01, 0.02],
            [0.04, -0.03],
            [0.02, 0.05],
            [-0.02, 0.01],
        ];
        let upstream_lambda = 0.17;
        let upstream_score = -0.11;

        let backward = gaussian_reml_multi_closed_form_backward(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.8),
            upstream_lambda,
            Some(upstream_coefficients.view()),
            Some(upstream_fitted.view()),
            upstream_score,
        )
        .expect("backward VJP");

        let objective = |x_eval: &Array2<f64>, y_eval: &Array2<f64>, w_eval: &Array1<f64>| {
            let fit = gaussian_reml_multi_closed_form_with_cache(
                x_eval.view(),
                y_eval.view(),
                penalty.view(),
                Some(w_eval.view()),
                Some(0.8),
                None,
            )
            .expect("fit for objective");
            upstream_lambda * fit.lambda
                + upstream_score * fit.reml_score
                + (&fit.coefficients * &upstream_coefficients).sum()
                + (&fit.fitted * &upstream_fitted).sum()
        };
        let eps = 1.0e-6;
        assert!(objective(&x, &y, &weights).is_finite());

        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[[3, 2]] += eps;
        x_minus[[3, 2]] -= eps;
        let fd_x =
            (objective(&x_plus, &y, &weights) - objective(&x_minus, &y, &weights)) / (2.0 * eps);
        assert!(
            (fd_x - backward.grad_x[[3, 2]]).abs() <= 2.0e-4,
            "grad_x mismatch: analytic={} fd={}",
            backward.grad_x[[3, 2]],
            fd_x
        );

        let mut y_plus = y.clone();
        let mut y_minus = y.clone();
        y_plus[[4, 1]] += eps;
        y_minus[[4, 1]] -= eps;
        let fd_y =
            (objective(&x, &y_plus, &weights) - objective(&x, &y_minus, &weights)) / (2.0 * eps);
        assert!(
            (fd_y - backward.grad_y[[4, 1]]).abs() <= 2.0e-4,
            "grad_y mismatch: analytic={} fd={}",
            backward.grad_y[[4, 1]],
            fd_y
        );

        let mut w_plus = weights.clone();
        let mut w_minus = weights.clone();
        w_plus[2] += eps;
        w_minus[2] -= eps;
        let fd_w = (objective(&x, &y, &w_plus) - objective(&x, &y, &w_minus)) / (2.0 * eps);
        assert!(
            (fd_w - backward.grad_weights[2]).abs() <= 2.0e-4,
            "grad_weight mismatch: analytic={} fd={}",
            backward.grad_weights[2],
            fd_w
        );
    }
}
