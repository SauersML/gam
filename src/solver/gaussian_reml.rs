use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

const RHO_LOWER: f64 = -30.0;
const RHO_UPPER: f64 = 30.0;
const EIGEN_REL_TOL: f64 = 1.0e-10;
const GRAD_TOL: f64 = 1.0e-10;
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
    let fitted = x.dot(&coefficients);
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

fn rho_derivatives_to_lambda(lambda: f64, grad_rho: f64, hess_rho: f64) -> (f64, f64) {
    (grad_rho / lambda, (hess_rho - grad_rho) / (lambda * lambda))
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

    let mut wx = x.to_owned();
    for i in 0..n {
        let wi = weight[i];
        for value in wx.row_mut(i) {
            *value *= wi;
        }
    }
    let xtwx = x.t().dot(&wx);
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
    let transformed_penalty = l_inv.dot(&penalty.to_owned()).dot(&l_inv.t());
    let (mut penalty_eigenvalues, eigenvectors) =
        transformed_penalty.eigh(Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?;
    let scale = penalty_eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1.0);
    let eig_tol = scale * EIGEN_REL_TOL;
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
    let pen_scale = pen_eigs
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1.0);
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

    let mut wy = y.to_owned();
    for i in 0..n {
        let wi = weight[i];
        for value in wy.row_mut(i) {
            *value *= wi;
        }
    }
    let xtwy = x.t().dot(&wy);
    let ywy = Array1::from_iter((0..d).map(|j| y.column(j).dot(&wy.column(j))));

    let mut wx = x.to_owned();
    for i in 0..n {
        let wi = weight[i];
        for value in wx.row_mut(i) {
            *value *= wi;
        }
    }
    let xtwx = x.t().dot(&wx);

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
        let projected_rhs = cache.coefficient_basis.t().dot(&xtwy);
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
    let projected_rhs = cache.coefficient_basis.t().dot(&xtwy);
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
        let lambda = rho.exp();
        let nu = self.nu();
        let d = self.n_outputs as f64;
        let mut logdet_h = self.cache.logdet_xtwx;
        let mut trace_h = 0.0;
        let mut trace_h_deriv = 0.0;
        let mut edf = 0.0;
        for &delta in &self.cache.penalty_eigenvalues {
            let t = lambda * delta;
            logdet_h += (1.0 + t).ln();
            if delta > 0.0 {
                trace_h += t / (1.0 + t);
                trace_h_deriv += t / ((1.0 + t) * (1.0 + t));
            }
            edf += 1.0 / (1.0 + t);
        }
        let logdet_s = self.cache.logdet_penalty_positive + (self.cache.penalty_rank as f64) * rho;

        let mut cost = 0.5 * d * (logdet_h - logdet_s);
        let mut grad = 0.5 * d * (trace_h - self.cache.penalty_rank as f64);
        let mut hess = 0.5 * d * trace_h_deriv;
        for j in 0..self.n_outputs {
            let mut fitted_quadratic = 0.0;
            let mut dp_grad = 0.0;
            let mut dp_hess = 0.0;
            for i in 0..self.cache.penalty_eigenvalues.len() {
                let c2 = self.projected_rhs_squared[[i, j]];
                let t = lambda * self.cache.penalty_eigenvalues[i];
                let denom = 1.0 + t;
                fitted_quadratic += c2 / denom;
                dp_grad += c2 * t / (denom * denom);
                dp_hess += c2 * t * (1.0 - t) / (denom * denom * denom);
            }
            let dp = (self.ywy[j] - fitted_quadratic).max(MIN_DEVIANCE);
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

    fn coefficients(&self, lambda: f64) -> Array2<f64> {
        let mut scaled = self.projected_rhs.clone();
        for i in 0..self.cache.penalty_eigenvalues.len() {
            let scale = 1.0 / (1.0 + lambda * self.cache.penalty_eigenvalues[i]);
            for value in scaled.row_mut(i) {
                *value *= scale;
            }
        }
        self.cache.coefficient_basis.dot(&scaled)
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
    let mut candidates = Vec::<f64>::new();
    push_candidate(&mut candidates, RHO_LOWER);
    push_candidate(&mut candidates, RHO_UPPER);
    if let Some(rho0) = init_rho {
        push_candidate(&mut candidates, rho0);
    }

    const GRID_INTERVALS: usize = 96;
    let mut prev_rho = RHO_LOWER;
    let mut prev_eval = prepared.evaluate(prev_rho);
    for i in 1..=GRID_INTERVALS {
        let rho = RHO_LOWER + (RHO_UPPER - RHO_LOWER) * (i as f64) / (GRID_INTERVALS as f64);
        let eval = prepared.evaluate(rho);
        push_candidate(&mut candidates, rho);
        if prev_eval.grad <= 0.0 && eval.grad >= 0.0 {
            push_candidate(
                &mut candidates,
                refine_stationary_rho(prepared, prev_rho, rho, 0.5 * (prev_rho + rho)),
            );
        }
        prev_rho = rho;
        prev_eval = eval;
    }

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
}
