use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerCholesky, FaerEigh, fast_ab, fast_atb, fast_xt_diag_x, fast_xt_diag_y,
};
use faer::Side;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, Axis,
    s,
};
use rayon::prelude::*;
use std::sync::Once;

/// One-time warning latch for backward-pass graceful degradation on a
/// near-singular penalized Hessian `K = XᵀWX + λS`. When `λ_k` saturates
/// (e.g. 1e10+), `K` becomes effectively rank-deficient and the analytic VJP
/// cannot be evaluated. Rather than raising, the backward returns zero
/// gradients of the correct shape: this is the statistically correct
/// "shrink-out" gradient — when `λ` has saturated, the atom is unused, so
/// every input's contribution to the loss is zero in the limit.
static ILL_CONDITIONED_BACKWARD_WARNED: Once = Once::new();

fn warn_ill_conditioned_backward_once(p: usize, d: usize, condition_number: f64) {
    ILL_CONDITIONED_BACKWARD_WARNED.call_once(|| {
        log::warn!(
            "gaussian_reml_fit_backward: K = XᵀWX + λS is near-singular \
             (p={p}, d={d}, cond≈{condition_number:.2e}); returning zero gradients \
             for this fit (λ has saturated, atom is effectively unused). \
             Further occurrences are silent."
        );
    });
}

fn zero_backward_result(n: usize, p: usize, d: usize) -> GaussianRemlBackwardResult {
    GaussianRemlBackwardResult {
        grad_x: Array2::<f64>::zeros((n, p)),
        grad_y: Array2::<f64>::zeros((n, d)),
        grad_penalty: Array2::<f64>::zeros((p, p)),
        grad_weights: Array1::<f64>::zeros(n),
    }
}

const RHO_LOWER: f64 = -30.0;
const RHO_UPPER: f64 = 30.0;
const EIGEN_REL_TOL: f64 = 1.0e-10;
const GRAD_TOL: f64 = 1.0e-12;
const MIN_DEVIANCE: f64 = 1.0e-300;

/// Canonicalize a penalty matrix to its symmetric average.
///
/// Closed-form Gaussian REML treats `S` as symmetric throughout — the
/// eigendecomposition, the pseudo-determinant `log|S|₊`, the rank detector,
/// and every per-helper VJP all assume `S = Sᵀ`. To make that contract
/// explicit (rather than implicit in `eigh(Side::Lower)` reading the lower
/// triangle and silently ignoring the upper), every entry point that takes a
/// penalty matrix replaces it with `0.5 (S + Sᵀ)` before any downstream use.
/// For symmetric input this is a numerical no-op; for asymmetric input it
/// defines the function as operating on the symmetric average.
fn canonicalize_penalty(penalty: ArrayView2<'_, f64>) -> Array2<f64> {
    let p = penalty.nrows();
    let mut out = penalty.to_owned();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

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
pub struct GaussianRemlFreeBScore {
    pub reml_score: f64,
    pub grad_coefficients: Array2<f64>,
    pub grad_penalty: Array2<f64>,
    pub grad_log_lambda: f64,
    pub fitted: Array2<f64>,
    pub sigma2: Array1<f64>,
    pub edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlBackwardResult {
    pub grad_x: Array2<f64>,
    pub grad_y: Array2<f64>,
    pub grad_penalty: Array2<f64>,
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
    pub grad_edf: f64,
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
            crate::bail_invalid_estim!(
                "Gaussian REML no-alloc workspace shape mismatch: expected p={p}, d={d}"
            );
        }
        Ok::<(), _>(())
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

#[derive(Clone, Debug)]
pub struct GaussianRemlBlockOrthogonalResult {
    pub coefficients: Vec<Array2<f64>>,
    pub fitted: Array2<f64>,
    pub lambdas: Array1<f64>,
    pub log_lambdas: Array1<f64>,
    pub reml_score: f64,
    pub edf: Array1<f64>,
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

/// A single Gaussian closed-form REML objective term, carrying its analytic
/// VALUE together with its analytic ρ-GRADIENT and ρ-HESSIAN.
///
/// Single source of truth: each term's value and its (already hand-derived,
/// closed-form) ρ-derivatives are returned from ONE function body, so a future
/// edit to the value formula cannot silently leave the derivatives stale.
/// Mirrors the `PenaltyLogdetDerivs`-returning-tuple pattern used by the
/// unified outer evaluator — the structural cure for the objective↔gradient
/// desync class (#752/#748/#808). The three contributions are accumulated
/// through [`ObjectiveEval`] at one site, so they cannot drift apart.
#[derive(Clone, Copy)]
struct TermDerivs {
    value: f64,
    grad: f64,
    hess: f64,
}

impl std::ops::AddAssign<TermDerivs> for ObjectiveEval {
    /// Fold a term's `(value, grad, hess)` triple into the running totals in
    /// lock-step, so value and derivative can never be added at separate sites.
    fn add_assign(&mut self, rhs: TermDerivs) {
        self.cost += rhs.value;
        self.grad += rhs.grad;
        self.hess += rhs.hess;
    }
}

/// `½d·(log|H| − log|S|_+)` value with its analytic ρ-gradient/Hessian.
///
/// The penalty-eigenvalue sum produces all three quantities from the SAME
/// `t = λδ` intermediates in one pass, so the value (`log|1+t|`) and its
/// derivatives (`t/(1+t)`, `t/(1+t)²`) are single-sourced.
fn gaussian_reml_logdet_term(
    cache: &GaussianRemlEigenCache,
    rho: f64,
    n_outputs: f64,
) -> (TermDerivs, f64) {
    let lambda = rho.exp();
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
    let term = TermDerivs {
        value: 0.5 * n_outputs * (logdet_h - logdet_s),
        grad: 0.5 * n_outputs * (trace_h - cache.penalty_rank as f64),
        hess: 0.5 * n_outputs * trace_h_deriv,
    };
    (term, edf)
}

/// Per-output dispersion-prior term `½ν·(1 + log(2π·dp/ν))` with its analytic
/// ρ-gradient/Hessian.
///
/// `dp`, `dp_grad`, `dp_hess` are computed from the SAME eigenvalue sum, then
/// the value `log(dp)` and its derivatives `dp_grad/dp`,
/// `dp_hess/dp − (dp_grad/dp)²` are returned together so they cannot desync.
fn gaussian_reml_dispersion_term(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    output: usize,
    nu: f64,
    lambda: f64,
) -> TermDerivs {
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
    TermDerivs {
        value: 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln()),
        grad: 0.5 * nu * dp_grad / dp,
        hess: 0.5 * nu * (dp_hess / dp - (dp_grad * dp_grad) / (dp * dp)),
    }
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
    // Match the symmetric-S contract used by the cache builder: the
    // fingerprint check below compares against a fingerprint computed on the
    // canonicalized penalty, so the input must be canonicalized first.
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    validate_gaussian_reml_eigen_cache(eigen_cache, p)?;
    if y.nrows() != n {
        crate::bail_invalid_estim!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        );
    }
    if y.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }
    if n <= eigen_cache.nullity {
        crate::bail_invalid_estim!(
            "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
            eigen_cache.nullity
        );
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    if eigen_cache.penalty_fingerprint != penalty_fingerprint {
        crate::bail_invalid_estim!("Gaussian REML eigen cache penalty mismatch");
    }
    workspace.validate(p, d)?;
    if coefficients.dim() != (p, d) || fitted.dim() != (n, d) || sigma2.len() != d {
        crate::bail_invalid_estim!(
            "Gaussian REML no-alloc output shape mismatch: expected coefficients=({p},{d}), fitted=({n},{d}), sigma2={d}"
        );
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
    if problems.is_empty() {
        return Ok(Vec::new());
    }
    // Phase A: par_iter compute X'WX per problem (the only per-fit step that
    // depends on `n_b`; remaining work is `O(p)` and can amortize through
    // `_with_cache`).
    let xtwx_per_problem: Vec<Array2<f64>> = problems
        .par_iter()
        .map(|problem| {
            let weight = match problem.weights.as_ref() {
                Some(w) => w.to_owned(),
                None => Array1::ones(problem.x.nrows()),
            };
            dense_xt_diag_x(problem.x.view(), weight.view())
        })
        .collect();
    // Phase B: one batched cuSOLVER Cholesky when policy approves uniform p
    // and K aggregate FLOPs; otherwise the cache builder uses the normal
    // per-fit non-GPU factorization path.
    let caches =
        build_gaussian_reml_eigen_cache_batched(xtwx_per_problem, penalty.view(), nullspace_dim);
    // Phase C: par_iter finish each fit with its prebuilt cache. A cache-build
    // error is a real per-problem error, not a signal to rebuild through a
    // second path.
    let fits: Vec<Result<GaussianRemlMultiResult, EstimationError>> = problems
        .par_iter()
        .zip(caches.into_par_iter())
        .map(|(problem, cache_result)| {
            let init_lambda = problem.init_rho.map(f64::exp);
            let cache = cache_result?;
            gaussian_reml_multi_closed_form_from_parts(
                problem.x.view(),
                problem.y.view(),
                penalty.view(),
                nullspace_dim,
                problem.weights.as_ref().map(|weights| weights.view()),
                init_lambda,
                Some(&cache),
            )
        })
        .collect();
    fits.into_iter().collect()
}

struct BlockOrthogonalEval {
    beta: Array2<f64>,
    logdet: f64,
    trace: f64,
    trace_pair: f64,
    fitted_energy: Array1<f64>,
    penalty_energy: Array1<f64>,
    curvature_energy: Array1<f64>,
    edf: f64,
}

fn block_penalty_rank_logdet(
    penalty: ArrayView2<'_, f64>,
) -> Result<(usize, f64), EstimationError> {
    let eigs = penalty
        .to_owned()
        .eigh(Side::Lower)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?
        .0;
    let max_abs = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let tol = (EIGEN_REL_TOL * max_abs).max(1.0e-14);
    let mut rank = 0_usize;
    let mut logdet = 0.0;
    for eig in eigs.iter().copied() {
        if eig > tol {
            rank += 1;
            logdet += eig.ln();
        }
    }
    Ok((rank, logdet))
}

fn block_orthogonal_eval(
    gram: &Array2<f64>,
    rhs: &Array2<f64>,
    penalty: &Array2<f64>,
    rho: f64,
) -> Result<BlockOrthogonalEval, EstimationError> {
    let lambda = rho.exp();
    validate_initial_lambda(lambda)?;
    let scaled_penalty = penalty * lambda;
    let hessian = canonicalize_penalty((gram + &scaled_penalty).view());
    let chol = gaussian_reml_cholesky_lower(hessian)?;
    let beta = solve_spd_from_lower_factor(&chol, rhs)?;
    let solved_penalty = solve_spd_from_lower_factor(&chol, &scaled_penalty)?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    let trace = (0..solved_penalty.nrows())
        .map(|i| solved_penalty[[i, i]])
        .sum::<f64>();
    let trace_pair =
        crate::linalg::utils::trace_of_product(solved_penalty.view(), solved_penalty.view());
    let fitted_energy = (rhs * &beta).sum_axis(Axis(0));
    let p_beta = scaled_penalty.dot(&beta);
    let penalty_energy = (&beta * &p_beta).sum_axis(Axis(0));
    let solved_p_beta = solve_spd_from_lower_factor(&chol, &p_beta)?;
    let curvature_energy = (&p_beta * &solved_p_beta).sum_axis(Axis(0));
    Ok(BlockOrthogonalEval {
        beta,
        logdet,
        trace,
        trace_pair,
        fitted_energy,
        penalty_energy,
        curvature_energy,
        edf: penalty.nrows() as f64 - trace,
    })
}

/// Block-orthogonal shared-scale REML objective VALUE together with its
/// analytic ρ-gradient and ρ-Hessian.
///
/// Single source of truth: the value `½d·logdet − ½·fit − ½d·rank·ρ` and its
/// ρ-derivatives are returned from ONE function body, so a future edit to the
/// objective cannot leave the Newton gradient/Hessian (previously written at a
/// physically separate site inside `solve_block_orthogonal_rho`) stale. This
/// closes a genuine `(value_here, gradient_there)` loose pair. Mirrors the
/// `PenaltyLogdetDerivs` single-source pattern; behavior is identical (the same
/// closed-form formulas, reorganized).
struct BlockOrthogonalScaleDerivs {
    value: f64,
    grad: f64,
    hess: f64,
}

fn block_orthogonal_scale_objective(
    eval: &BlockOrthogonalEval,
    rho: f64,
    scale_precision: ArrayView1<'_, f64>,
    rank: usize,
) -> BlockOrthogonalScaleDerivs {
    let d = scale_precision.len() as f64;
    let fit_term = scale_precision
        .iter()
        .zip(eval.fitted_energy.iter())
        .map(|(scale, energy)| scale * energy)
        .sum::<f64>();
    // VALUE: ½d·log|H| − ½ Σ_o w_o ⟨y_o, fit_o⟩ − ½d·rank·ρ.
    let value = 0.5 * d * eval.logdet - 0.5 * fit_term - 0.5 * d * (rank as f64) * rho;
    // ρ-GRADIENT: d/dρ of the same scalar. The logdet term contributes
    // ½d·(tr(H⁻¹λS) − rank); the (data-independent-at-fixed-β envelope) fit term
    // contributes +½ Σ_o w_o βᵀ(λS)β. Both share `eval`'s cached energies.
    let grad = 0.5 * d * (eval.trace - rank as f64)
        + 0.5
            * scale_precision
                .iter()
                .zip(eval.penalty_energy.iter())
                .map(|(scale, energy)| scale * energy)
                .sum::<f64>();
    // ρ-HESSIAN: d²/dρ². Logdet term: ½d·(tr(H⁻¹λS) − tr((H⁻¹λS)²)); penalty
    // term: ½ Σ_o w_o (βᵀλSβ − 2 βᵀλS H⁻¹ λS β).
    let hess = 0.5 * d * (eval.trace - eval.trace_pair)
        + 0.5
            * scale_precision
                .iter()
                .zip(eval.penalty_energy.iter().zip(eval.curvature_energy.iter()))
                .map(|(scale, (energy, curvature))| scale * (energy - 2.0 * curvature))
                .sum::<f64>();
    BlockOrthogonalScaleDerivs { value, grad, hess }
}

fn solve_block_orthogonal_rho(
    gram: &Array2<f64>,
    rhs: &Array2<f64>,
    penalty: &Array2<f64>,
    rho0: f64,
    scale_precision: ArrayView1<'_, f64>,
    rank: usize,
    max_iter: usize,
) -> Result<(f64, BlockOrthogonalEval), EstimationError> {
    let mut rho = rho0;
    let mut current = block_orthogonal_eval(gram, rhs, penalty, rho)?;
    for _ in 0..max_iter {
        // Value, ρ-gradient, and ρ-Hessian all come from the SINGLE
        // single-source objective evaluation — they cannot desync.
        let derivs = block_orthogonal_scale_objective(&current, rho, scale_precision, rank);
        let grad = derivs.grad;
        let hess = derivs.hess;
        if !(grad.is_finite() && hess.is_finite()) {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        // Newton step where the curvature is reliably positive; a unit
        // gradient-descent direction where it is not (non-convex / near-flat
        // region) so we never step ALONG negative curvature (which ascends).
        // There is deliberately NO magic step clamp: the line search below
        // globalizes and SKIPS any candidate ρ that is infeasible (e.g. λ =
        // exp(ρ) overflows `validate_initial_lambda`), so an over-long Newton
        // step is simply rejected rather than bounded by an arbitrary constant
        // or crashing the solve. This is the root fix the old `.clamp(-2,2)`
        // was masking: the clamp existed only to keep an over-long step from
        // reaching `block_orthogonal_eval`, which errors on a non-finite λ.
        let descent = grad.signum();
        let step = if hess > 1.0e-10 {
            grad / hess
        } else {
            descent
        };
        let mut best_rho = rho;
        let mut best_eval = current;
        let mut best_phi =
            block_orthogonal_scale_objective(&best_eval, best_rho, scale_precision, rank).value;
        for candidate_rho in [
            rho - step,
            rho - 0.5 * step,
            rho - 0.25 * step,
            rho - descent,
            rho - 0.25 * descent,
        ] {
            // Skip infeasible candidates (λ overflow / ill-conditioned Gram)
            // instead of failing the whole solve — the bounded gradient
            // candidates (`rho - descent`, `rho - 0.25·descent`) remain valid,
            // so a too-long Newton step degrades to gradient descent.
            let Ok(candidate_eval) = block_orthogonal_eval(gram, rhs, penalty, candidate_rho)
            else {
                continue;
            };
            let candidate_phi = block_orthogonal_scale_objective(
                &candidate_eval,
                candidate_rho,
                scale_precision,
                rank,
            )
            .value;
            if candidate_phi < best_phi {
                best_rho = candidate_rho;
                best_eval = candidate_eval;
                best_phi = candidate_phi;
            }
        }
        let delta = (best_rho - rho).abs();
        rho = best_rho;
        current = best_eval;
        if delta < 1.0e-12 || step.abs() < 1.0e-7 {
            break;
        }
    }
    Ok((rho, current))
}

pub fn gaussian_reml_blocks_orthogonal_shared_scale(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rhos: Option<&[f64]>,
) -> Result<GaussianRemlBlockOrthogonalResult, EstimationError> {
    if designs.is_empty() {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML requires at least one block");
    }
    if designs.len() != penalties.len() {
        crate::bail_invalid_estim!(
            "block-orthogonal Gaussian REML block mismatch: {} designs, {} penalties",
            designs.len(),
            penalties.len()
        );
    }
    let n = y.nrows();
    let d = y.ncols();
    if d == 0 {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML requires at least one output");
    }
    if y.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML response must be finite");
    }
    let weight = gaussian_reml_weights(n, weights)?;
    if let Some(rhos) = init_rhos {
        if rhos.len() != designs.len() {
            crate::bail_invalid_estim!(
                "block-orthogonal Gaussian REML init_rhos length mismatch: expected {}, got {}",
                designs.len(),
                rhos.len()
            );
        }
        if rhos.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("block-orthogonal Gaussian REML init_rhos must be finite");
        }
    }

    let mut ywy = Array1::<f64>::zeros(d);
    for row in 0..n {
        for output in 0..d {
            ywy[output] += weight[row] * y[[row, output]] * y[[row, output]];
        }
    }
    let mut grams = Vec::with_capacity(designs.len());
    let mut rhs_blocks = Vec::with_capacity(designs.len());
    let mut penalties_owned = Vec::with_capacity(penalties.len());
    let mut ranks = Vec::with_capacity(penalties.len());
    let mut penalty_logdets = Vec::with_capacity(penalties.len());
    let mut nullity_total = 0_usize;
    for (block, (design, penalty)) in designs.iter().zip(penalties.iter()).enumerate() {
        let penalty_owned = canonicalize_penalty(penalty.view());
        validate_gaussian_reml_design(design.view(), penalty_owned.view(), Some(weight.view()))?;
        if design.nrows() != n {
            crate::bail_invalid_estim!(
                "block-orthogonal Gaussian REML designs[{block}] has {} rows, expected {n}",
                design.nrows()
            );
        }
        let gram = dense_xt_diag_x(design.view(), weight.view());
        let rhs = dense_xt_diag_y(design.view(), weight.view(), y);
        let (rank, logdet) = block_penalty_rank_logdet(penalty_owned.view())?;
        nullity_total += penalty_owned.nrows().saturating_sub(rank);
        grams.push(canonicalize_penalty(gram.view()));
        rhs_blocks.push(rhs);
        penalties_owned.push(penalty_owned);
        ranks.push(rank);
        penalty_logdets.push(logdet);
    }
    if n <= nullity_total {
        crate::bail_invalid_estim!(
            "block-orthogonal Gaussian REML requires n > total penalty nullity; got n={n}, nullity={nullity_total}"
        );
    }
    let nu = (n - nullity_total) as f64;
    let mut rhos = match init_rhos {
        Some(values) => Array1::from_vec(values.to_vec()),
        None => Array1::zeros(designs.len()),
    };
    let mut scale_precision = ywy.mapv(|value| nu / value.max(MIN_DEVIANCE));
    let mut evals = Vec::new();
    for _ in 0..40 {
        evals.clear();
        for block in 0..designs.len() {
            let (rho, eval) = solve_block_orthogonal_rho(
                &grams[block],
                &rhs_blocks[block],
                &penalties_owned[block],
                rhos[block],
                scale_precision.view(),
                ranks[block],
                32,
            )?;
            rhos[block] = rho;
            evals.push(eval);
        }
        let mut explained = Array1::<f64>::zeros(d);
        for eval in evals.iter() {
            explained += &eval.fitted_energy;
        }
        let q = &ywy - &explained;
        if q.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let next_scale = q.mapv(|value| nu / value);
        let scale_step = next_scale
            .iter()
            .zip(scale_precision.iter())
            .map(|(next, old)| (next.ln() - old.ln()).abs())
            .fold(0.0_f64, f64::max);
        scale_precision = next_scale;
        if scale_step < 1.0e-7 {
            break;
        }
    }
    evals.clear();
    for block in 0..designs.len() {
        let (rho, eval) = solve_block_orthogonal_rho(
            &grams[block],
            &rhs_blocks[block],
            &penalties_owned[block],
            rhos[block],
            scale_precision.view(),
            ranks[block],
            16,
        )?;
        rhos[block] = rho;
        evals.push(eval);
    }

    let coefficients = evals
        .iter()
        .map(|eval| eval.beta.clone())
        .collect::<Vec<_>>();
    let mut fitted = Array2::<f64>::zeros((n, d));
    for (design, coef) in designs.iter().zip(coefficients.iter()) {
        fitted += &fast_ab(&design.view(), &coef.view());
    }
    let mut explained = Array1::<f64>::zeros(d);
    for eval in evals.iter() {
        explained += &eval.fitted_energy;
    }
    let q = &ywy - &explained;
    if q.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let lambdas = rhos.mapv(f64::exp);
    let edf = Array1::from_iter(evals.iter().map(|eval| eval.edf));
    let logdet_term = evals
        .iter()
        .enumerate()
        .map(|(block, eval)| {
            eval.logdet - penalty_logdets[block] - (ranks[block] as f64) * rhos[block]
        })
        .sum::<f64>();
    let scale_term = q
        .iter()
        .map(|value| nu * (1.0 + (2.0 * std::f64::consts::PI * value / nu).ln()))
        .sum::<f64>();
    Ok(GaussianRemlBlockOrthogonalResult {
        coefficients,
        fitted,
        lambdas,
        log_lambdas: rhos,
        reml_score: 0.5 * (d as f64) * logdet_term + 0.5 * scale_term,
        edf,
    })
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

pub fn gaussian_reml_free_b_score(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    coefficients: ArrayView2<'_, f64>,
    log_lambda: f64,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlFreeBScore, EstimationError> {
    if !log_lambda.is_finite() {
        crate::bail_invalid_estim!("Gaussian REML log_lambda must be finite; got {log_lambda}");
    }
    let lambda = log_lambda.exp();
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    if y.nrows() != n {
        crate::bail_invalid_estim!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        );
    }
    if coefficients.dim() != (p, d) {
        crate::bail_invalid_estim!(
            "Gaussian REML coefficient shape mismatch: expected {p}x{d}, got {}x{}",
            coefficients.nrows(),
            coefficients.ncols()
        );
    }
    if y.iter().chain(coefficients.iter()).any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }

    let weight = gaussian_reml_weights(n, weights)?;
    let cache =
        build_gaussian_reml_eigen_cache_with_nullspace_dim(x, penalty, None, Some(weight.view()))?;
    if n <= cache.nullity {
        crate::bail_invalid_estim!(
            "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
            cache.nullity
        );
    }
    let nu = n as f64 - cache.nullity as f64;
    let fitted = dense_ab(x, coefficients);
    let residual = y.to_owned() - &fitted;
    let xtw_residual = dense_xt_diag_y(x, weight.view(), residual.view());
    let s_beta = dense_ab(penalty, coefficients);

    let mut logdet_h = cache.logdet_xtwx;
    let mut trace_h = 0.0;
    let mut edf = 0.0;
    for &delta in &cache.penalty_eigenvalues {
        let t = lambda * delta;
        logdet_h += (1.0 + t).ln();
        if delta > 0.0 {
            trace_h += t / (1.0 + t);
        }
        edf += 1.0 / (1.0 + t);
    }
    let logdet_s = cache.logdet_penalty_positive + (cache.penalty_rank as f64) * log_lambda;
    let mut reml_score = 0.5 * (d as f64) * (logdet_h - logdet_s);
    let mut grad_log_lambda = 0.5 * (d as f64) * (trace_h - cache.penalty_rank as f64);
    let mut grad_coefficients = Array2::<f64>::zeros((p, d));
    let inverse_hessian = {
        let xtwx = dense_xt_diag_x(x, weight.view());
        let mut hessian = xtwx;
        hessian += &(penalty.to_owned() * lambda);
        hessian
            .cholesky(Side::Lower)
            .map_err(EstimationError::LinearSystemSolveFailed)?
            .solve_mat(&Array2::<f64>::eye(p))
    };
    let penalty_pinv = gaussian_reml_penalty_pseudoinverse_from_cache(&cache);
    let mut grad_penalty = Array2::<f64>::zeros((p, p));
    for row in 0..p {
        for col in 0..p {
            grad_penalty[[row, col]] += 0.5
                * (d as f64)
                * (lambda * inverse_hessian[[col, row]] - penalty_pinv[[col, row]]);
        }
    }
    let mut sigma2 = Array1::<f64>::zeros(d);

    for output in 0..d {
        let mut weighted_rss = 0.0;
        for row in 0..n {
            let r = residual[[row, output]];
            weighted_rss += weight[row] * r * r;
        }
        let beta_col = coefficients.column(output);
        let s_beta_col = s_beta.column(output);
        let penalty_quadratic = beta_col.dot(&s_beta_col);
        let dp = (weighted_rss + lambda * penalty_quadratic).max(MIN_DEVIANCE);
        sigma2[output] = dp / nu;
        reml_score += 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
        grad_log_lambda += 0.5 * nu * lambda * penalty_quadratic / dp;
        let scale = nu / dp;
        for coeff in 0..p {
            grad_coefficients[[coeff, output]] =
                scale * (-xtw_residual[[coeff, output]] + lambda * s_beta[[coeff, output]]);
        }
        add_rank_one_penalty_vjp(0.5 * scale * lambda, beta_col, &mut grad_penalty);
    }
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (grad_penalty[[i, j]] + grad_penalty[[j, i]]);
            grad_penalty[[i, j]] = avg;
            grad_penalty[[j, i]] = avg;
        }
    }

    Ok(GaussianRemlFreeBScore {
        reml_score,
        grad_coefficients,
        grad_penalty,
        grad_log_lambda,
        fitted,
        sigma2,
        edf,
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
    upstream_edf: f64,
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
        upstream_edf,
    )
}

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
    upstream_edf: f64,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    validate_gaussian_reml_backward_upstreams(
        x,
        y,
        penalty,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
        upstream_edf,
    )?;
    validate_gaussian_reml_forward_fit(x, y, penalty, weights, fit)?;
    let lambda = fit.lambda;
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    if !(fit.reml_hess_rho.is_finite() && fit.reml_hess_rho.abs() > 1.0e-14) {
        // Graceful degradation: when λ saturates, K = XᵀWX + λS is
        // effectively rank-deficient and the analytic VJP is undefined.
        // Return zero gradients (the correct shrink-out limit) instead of
        // raising — production training at large F can have individual
        // atoms saturate λ in early batches and must not blow up here.
        warn_ill_conditioned_backward_once(p, d, f64::INFINITY);
        return Ok(zero_backward_result(n, p, d));
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let inverse_hessian = match gaussian_reml_inverse_hessian_from_cache(&fit.cache, lambda) {
        Ok(inv) => inv,
        Err(EstimationError::ModelIsIllConditioned { condition_number }) => {
            warn_ill_conditioned_backward_once(p, d, condition_number);
            return Ok(zero_backward_result(n, p, d));
        }
        Err(err) => return Err(err),
    };
    gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
        x,
        y,
        penalty,
        weight,
        fit,
        inverse_hessian,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
        upstream_edf,
        n,
        p,
        d,
    )
}

fn gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weight: Array1<f64>,
    fit: &GaussianRemlMultiResult,
    inverse_hessian: Array2<f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
    upstream_edf: f64,
    n: usize,
    p: usize,
    d: usize,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    // Backward sees the same symmetric S the forward used. Canonicalize on
    // entry so an asymmetric input (e.g. a single-entry gradcheck perturbation
    // around a symmetric base) cannot leak into the per-helper VJPs.
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let lambda = fit.lambda;
    let beta = &fit.coefficients;
    let residual = y.to_owned() - &fit.fitted;
    let nu = n as f64 - fit.cache.nullity as f64;

    let mut grad_x = Array2::<f64>::zeros((n, p));
    let mut grad_y = Array2::<f64>::zeros((n, d));
    let mut grad_penalty = Array2::<f64>::zeros((p, p));
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
        // A downstream loss that explicitly uses beta_hat or fitted = X beta_hat
        // cannot use the REML envelope shortcut.  Route those seeds through
        // the fixed-rho KKT adjoint M u = upstream_beta, then differentiate
        // X, y, weights, and S through the ridge solve.
        add_ridge_profile_vjp_with_lambda_grad(
            1.0,
            x,
            y,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            beta,
            upstream_beta.view(),
            &mut grad_x,
            &mut grad_y,
            &mut grad_penalty,
            &mut grad_weights,
            &mut lambda_adjoint,
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
            lambda,
            &fit.cache,
            &mut grad_x,
            &mut grad_y,
            &mut grad_penalty,
            &mut grad_weights,
        );
        lambda_adjoint += upstream_reml_score * fit.reml_grad_lambda;
    }

    if upstream_edf != 0.0 {
        lambda_adjoint += add_edf_vjp(
            upstream_edf,
            x,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            &mut grad_x,
            &mut grad_penalty,
            &mut grad_weights,
        );
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
            &mut grad_penalty,
            &mut grad_weights,
        );
    }

    // The forward consumes `S` only through the canonicalization
    // `S_canon = 0.5 (S + Sᵀ)`. By the chain rule, the gradient w.r.t. an
    // input `S_input` is `0.5 (G + Gᵀ)` where `G = ∂L/∂S_canon` is what the
    // per-helper VJPs accumulate. Symmetrize the full matrix here so a
    // single-entry perturbation `δS = ε E_{i,j}` (asymmetric, as
    // `torch.autograd.gradcheck` produces) sees the gradient component
    // `0.5 (G[i,j] + G[j,i])` it expects from FD — no caller-side
    // bookkeeping required.
    let p = grad_penalty.nrows();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (grad_penalty[[i, j]] + grad_penalty[[j, i]]);
            grad_penalty[[i, j]] = avg;
            grad_penalty[[j, i]] = avg;
        }
    }
    Ok(GaussianRemlBackwardResult {
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    })
}

pub fn gaussian_reml_multi_closed_form_backward_batch<'a>(
    problems: &[GaussianRemlMultiBackwardProblem<'a>],
    penalty: ArrayView2<'a, f64>,
) -> Vec<Result<GaussianRemlBackwardResult, EstimationError>> {
    let inverse_hessians = batched_inverse_hessians_from_caches(problems);
    let results: Vec<Result<GaussianRemlBackwardResult, EstimationError>> = problems
        .par_iter()
        .zip(inverse_hessians.into_par_iter())
        .map(|(problem, inverse_hessian_result)| {
            validate_gaussian_reml_backward_upstreams(
                problem.x.view(),
                problem.y.view(),
                penalty,
                problem.grad_lambda,
                problem.grad_coefficients.as_ref().map(|g| g.view()),
                problem.grad_fitted.as_ref().map(|g| g.view()),
                problem.grad_reml_score,
                problem.grad_edf,
            )?;
            validate_gaussian_reml_forward_fit(
                problem.x.view(),
                problem.y.view(),
                penalty,
                problem.weights.as_ref().map(|w| w.view()),
                problem.fit,
            )?;
            let n = problem.x.nrows();
            let p = problem.x.ncols();
            let d = problem.y.ncols();
            if !(problem.fit.reml_hess_rho.is_finite() && problem.fit.reml_hess_rho.abs() > 1.0e-14)
            {
                // Graceful degradation — see `gaussian_reml_multi_closed_form_backward_from_fit`.
                warn_ill_conditioned_backward_once(p, d, f64::INFINITY);
                return Ok(zero_backward_result(n, p, d));
            }
            let weight = gaussian_reml_weights(n, problem.weights.as_ref().map(|w| w.view()))?;
            let inverse_hessian = match inverse_hessian_result {
                Ok(inv) => inv,
                Err(EstimationError::ModelIsIllConditioned { condition_number }) => {
                    warn_ill_conditioned_backward_once(p, d, condition_number);
                    return Ok(zero_backward_result(n, p, d));
                }
                Err(err) => return Err(err),
            };
            gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
                problem.x.view(),
                problem.y.view(),
                penalty,
                weight,
                problem.fit,
                inverse_hessian,
                problem.grad_lambda,
                problem.grad_coefficients.as_ref().map(|g| g.view()),
                problem.grad_fitted.as_ref().map(|g| g.view()),
                problem.grad_reml_score,
                problem.grad_edf,
                n,
                p,
                d,
            )
        })
        .collect();
    results
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
    upstream_edf: f64,
) -> Result<(), EstimationError> {
    if !(upstream_lambda.is_finite() && upstream_reml_score.is_finite() && upstream_edf.is_finite())
    {
        crate::bail_invalid_estim!("Gaussian REML backward upstream scalars must be finite");
    }
    if let Some(upstream_coefficients) = upstream_coefficients {
        if upstream_coefficients.dim() != (x.ncols(), y.ncols()) {
            crate::bail_invalid_estim!(
                "Gaussian REML backward coefficient upstream shape mismatch: expected {}x{}, got {}x{}",
                x.ncols(),
                y.ncols(),
                upstream_coefficients.nrows(),
                upstream_coefficients.ncols()
            );
        }
        if upstream_coefficients.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!(
                "Gaussian REML backward coefficient upstream must be finite"
            );
        }
    }
    if let Some(upstream_fitted) = upstream_fitted {
        if upstream_fitted.dim() != y.dim() {
            crate::bail_invalid_estim!(
                "Gaussian REML backward fitted upstream shape mismatch: expected {}x{}, got {}x{}",
                y.nrows(),
                y.ncols(),
                upstream_fitted.nrows(),
                upstream_fitted.ncols()
            );
        }
        if upstream_fitted.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("Gaussian REML backward fitted upstream must be finite");
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
    // Fingerprint the canonicalized penalty: caches are keyed on the
    // symmetric average, and the caller may hand us a raw input (e.g. a
    // single-entry-perturbed matrix produced by ``torch.autograd.gradcheck``).
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
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
        crate::bail_invalid_estim!(
            "Gaussian REML backward forward-state shape mismatch: expected coefficients=({p},{d}), fitted=({n},{d}), sigma2={d}"
        );
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
        crate::bail_invalid_estim!("Gaussian REML backward forward state must be finite");
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    if fit.cache.penalty_fingerprint != penalty_fingerprint {
        crate::bail_invalid_estim!("Gaussian REML backward forward-state penalty mismatch");
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let xtwx = dense_xt_diag_x(x, weight.view());
    if fit.cache.xtwx_fingerprint != matrix_fingerprint(xtwx.view()) {
        crate::bail_invalid_estim!("Gaussian REML backward forward-state X'WX mismatch");
    }
    Ok(())
}

fn gaussian_reml_inverse_hessian_from_cache(
    cache: &GaussianRemlEigenCache,
    lambda: f64,
) -> Result<Array2<f64>, EstimationError> {
    if !(lambda.is_finite() && lambda > 0.0) {
        crate::bail_invalid_estim!(
            "Gaussian REML lambda must be finite and positive; got {lambda}"
        );
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

fn batched_inverse_hessians_from_caches(
    problems: &[GaussianRemlMultiBackwardProblem<'_>],
) -> Vec<Result<Array2<f64>, EstimationError>> {
    if problems.is_empty() {
        return Vec::new();
    }
    let p = problems[0].fit.cache.coefficient_basis.nrows();
    let uniform = p > 0
        && problems.iter().all(|problem| {
            let cache = &problem.fit.cache;
            cache.coefficient_basis.dim() == (p, p) && cache.penalty_eigenvalues.len() == p
        });
    if uniform && problems.len() > 1 {
        let mut scaled_basis = Array3::<f64>::zeros((problems.len(), p, p));
        let mut basis = Array3::<f64>::zeros((problems.len(), p, p));
        let mut valid = true;
        for (idx, problem) in problems.iter().enumerate() {
            let lambda = problem.fit.lambda;
            if !(lambda.is_finite() && lambda > 0.0) {
                valid = false;
                break;
            }
            let cache = &problem.fit.cache;
            basis
                .slice_mut(s![idx, .., ..])
                .assign(&cache.coefficient_basis);
            for eig in 0..p {
                let scale = 1.0 / (1.0 + lambda * cache.penalty_eigenvalues[eig]);
                for row in 0..p {
                    scaled_basis[[idx, row, eig]] = cache.coefficient_basis[[row, eig]] * scale;
                }
            }
        }
        if valid
            && let Some(inverses) =
                crate::gpu::try_fast_abt_strided_batched(scaled_basis.view(), basis.view())
        {
            return inverses
                .axis_iter(Axis(0))
                .map(|inverse| Ok(inverse.to_owned()))
                .collect();
        }
    }
    problems
        .iter()
        .map(|problem| {
            gaussian_reml_inverse_hessian_from_cache(&problem.fit.cache, problem.fit.lambda)
        })
        .collect()
}

/// Side-effects of the ridge-profile VJP that are independent of λ.
///
/// Computes the KKT adjoint `m = M^{-1} u` for `u = upstream_beta` and accumulates
/// the partials w.r.t. `X`, `y`, `S`, and `w` into the provided gradient buffers.
/// Returns `m` so callers that also need `∂L/∂λ` can fold in the λ-adjoint dot
/// product `−scale · ⟨m, S β⟩` without recomputing the adjoint solve.
fn ridge_profile_vjp_data_partials(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> Array2<f64> {
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

    for row in 0..penalty.nrows() {
        for col in 0..penalty.ncols() {
            let mut value = 0.0;
            for output in 0..beta.ncols() {
                value += m[[row, output]] * beta[[col, output]];
            }
            grad_penalty[[row, col]] -= scale * lambda * value;
        }
    }
    m
}

/// Ridge-profile VJP for callers that also need `∂L/∂λ`.
///
/// Accumulates the data/penalty/weight partials and adds the implicit-function
/// λ-adjoint contribution `−scale · ⟨M^{-1} u, S β⟩` into `lambda_adjoint_out`.
fn add_ridge_profile_vjp_with_lambda_grad(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
    lambda_adjoint_out: &mut f64,
) {
    let m = ridge_profile_vjp_data_partials(
        scale,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta,
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
    let penalty_beta = dense_ab(penalty, beta.view());
    let dot = m
        .iter()
        .zip(penalty_beta.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    *lambda_adjoint_out += -scale * dot;
}

/// Ridge-profile VJP for callers that hold λ fixed (e.g. the implicit-root
/// partial inside `add_reml_rho_gradient_vjp`). The λ-adjoint dot product is
/// skipped entirely — it would be unused work in this branch.
fn add_ridge_profile_vjp_fixed_lambda(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    ridge_profile_vjp_data_partials(
        scale,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta,
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
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
    lambda: f64,
    cache: &GaussianRemlEigenCache,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    let d = beta.ncols() as f64;
    let xp = dense_ab(x, inverse_hessian.view());
    let penalty_pinv = gaussian_reml_penalty_pseudoinverse_from_cache(cache);
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] +=
                scale * 0.5 * d * (lambda * inverse_hessian[[col, row]] - penalty_pinv[[col, row]]);
        }
    }
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
        add_rank_one_penalty_vjp(coef * lambda, beta.column(j), grad_penalty);
    }
}

/// VJP contribution from an upstream gradient on `edf`.
///
/// With `M = X^T W X + λ S`, `edf = trace(M^{-1} · X^T W X) = p - λ trace(M^{-1} S)`.
/// Holding `λ` fixed, the direct partials are
///   ∂edf/∂A = λ M^{-1} S M^{-1}      (A = X^T W X, symmetric)
///   ∂edf/∂S = −λ M^{-1} A M^{-1} = −λ M^{-1} + λ² M^{-1} S M^{-1}
///   ∂edf/∂λ = −trace(M^{-1} S) + λ trace((M^{-1} S)²)
/// The λ-component is returned as the lambda_adjoint contribution and routed
/// through the implicit-function chain by the caller (same path as
/// `upstream_lambda` and `upstream_reml_score`).
fn add_edf_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    grad_x: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> f64 {
    // m_inv_s = M^{-1} S, then g_a = λ M^{-1} S M^{-1} = ∂edf/∂A.
    let m_inv_s = dense_ab(inverse_hessian.view(), penalty);
    let mut g_a = dense_ab(m_inv_s.view(), inverse_hessian.view());
    g_a.mapv_inplace(|v| v * lambda);

    // Chain ∂edf/∂A through A = X^T W X.
    //   grad_X += scale · 2 · (W X) · G_A
    //   grad_w_i += scale · (X G_A X^T)_{ii}
    let xg = dense_ab(x, g_a.view());
    // Row-scaled dense accumulate: grad_x[i,:] += (2·scale·weights[i]) · xg[i,:].
    // (Inlined here — the former `assembly::add_row_scaled_dense_into` helper was
    // removed as "unused" by 0cb722d, which missed this gam-pyffi-reachable caller.)
    let leading_scale = 2.0 * scale;
    for i in 0..xg.nrows() {
        let row_scale = leading_scale * weights[i];
        for k in 0..xg.ncols() {
            grad_x[[i, k]] += row_scale * xg[[i, k]];
        }
    }
    for i in 0..x.nrows() {
        let mut quad = 0.0;
        for k in 0..x.ncols() {
            quad += x[[i, k]] * xg[[i, k]];
        }
        grad_weights[i] += scale * quad;
    }

    // ∂edf/∂S = -λ M^{-1} + λ² M^{-1} S M^{-1} = -λ M^{-1} + λ · g_a
    // (since g_a = λ M^{-1} S M^{-1}, so λ · g_a = λ² M^{-1} S M^{-1}).
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] +=
                scale * (-lambda * inverse_hessian[[row, col]] + lambda * g_a[[row, col]]);
        }
    }

    // ∂edf/∂λ (with A, S fixed) = -tr(M^{-1} S) + λ tr((M^{-1} S)²).
    let p_dim = m_inv_s.nrows();
    let mut tr_m_inv_s = 0.0;
    for i in 0..p_dim {
        tr_m_inv_s += m_inv_s[[i, i]];
    }
    let mut tr_squared = 0.0;
    for i in 0..p_dim {
        for j in 0..p_dim {
            tr_squared += m_inv_s[[i, j]] * m_inv_s[[j, i]];
        }
    }
    scale * (-tr_m_inv_s + lambda * tr_squared)
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
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    let d = beta.ncols() as f64;
    let inverse_s = dense_ab(inverse_hessian.view(), penalty);
    let trace_kernel = dense_ab(inverse_s.view(), inverse_hessian.view());
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] += scale
                * 0.5
                * d
                * lambda
                * (inverse_hessian[[col, row]] - lambda * trace_kernel[[col, row]]);
        }
    }
    let xt = dense_ab(x, trace_kernel.view());
    for i in 0..x.nrows() {
        let wi = -scale * d * lambda * weights[i];
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * xt[[i, k]];
        }
        let mut quad = 0.0;
        for k in 0..x.ncols() {
            quad += x[[i, k]] * xt[[i, k]];
        }
        grad_weights[i] -= scale * 0.5 * d * lambda * quad;
    }

    let s_beta = dense_ab(penalty, beta.view());
    let mut upstream_beta = Array2::<f64>::zeros(beta.dim());
    for j in 0..beta.ncols() {
        let dp = (sigma2[j] * nu).max(MIN_DEVIANCE);
        let q = lambda * beta.column(j).dot(&s_beta.column(j));
        let q_coef = scale * nu / dp;
        for row in 0..beta.nrows() {
            upstream_beta[[row, j]] = q_coef * lambda * s_beta[[row, j]];
        }
        let dp_coef = -scale * 0.5 * nu * q / (dp * dp);
        add_rank_one_penalty_vjp(
            (0.5 * q_coef + dp_coef) * lambda,
            beta.column(j),
            grad_penalty,
        );
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
    // The implicit-root VJP holds lambda fixed inside this partial; only the
    // data, penalty, and weight side effects from the ridge solve are needed.
    add_ridge_profile_vjp_fixed_lambda(
        1.0,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta.view(),
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
}

fn add_rank_one_penalty_vjp(
    scale: f64,
    beta_col: ArrayView1<'_, f64>,
    grad_penalty: &mut Array2<f64>,
) {
    for row in 0..beta_col.len() {
        for col in 0..beta_col.len() {
            grad_penalty[[row, col]] += scale * beta_col[row] * beta_col[col];
        }
    }
}

fn gaussian_reml_penalty_pseudoinverse_from_cache(cache: &GaussianRemlEigenCache) -> Array2<f64> {
    let p = cache.penalty_eigenvalues.len();
    let mut scaled_basis = Array2::<f64>::zeros((p, p));
    for eig in 0..p {
        let delta = cache.penalty_eigenvalues[eig];
        if delta > 0.0 {
            for row in 0..p {
                scaled_basis[[row, eig]] = cache.coefficient_basis[[row, eig]] / delta;
            }
        }
    }
    dense_ab(scaled_basis.view(), cache.coefficient_basis.t())
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

/// Build eigen caches for K problems that share the same penalty matrix in a
/// single phased pipeline. X'WX construction is batched by the caller; each
/// cache then uses the same Cholesky/eigendecomposition implementation as the
/// single-fit path.
pub fn build_gaussian_reml_eigen_cache_batched(
    xtwx_matrices: Vec<Array2<f64>>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
) -> Vec<Result<GaussianRemlEigenCache, EstimationError>> {
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let k = xtwx_matrices.len();
    if k == 0 {
        return Vec::new();
    }
    let fingerprints: Vec<u64> = xtwx_matrices
        .iter()
        .map(|m| matrix_fingerprint(m.view()))
        .collect();

    let p = xtwx_matrices[0].nrows();
    let uniform_square = p > 0 && xtwx_matrices.iter().all(|matrix| matrix.dim() == (p, p));
    if uniform_square && k > 1 {
        let mut lower_matrices = xtwx_matrices.clone();
        if crate::gpu::try_cholesky_batched_lower_inplace(&mut lower_matrices).is_some() {
            // The batched penalty transform is an optional accelerator. On
            // failure we must NOT fabricate an empty Vec (indexing it per-block
            // would silently drop the transform for every block and could index
            // out of range) — instead route every block through the same
            // no-GPU-transform path used when the batched transform is
            // unavailable, which recomputes the whitened penalty on CPU from the
            // already-valid Cholesky factor `lower`.
            let transforms = batched_whitened_penalty_transforms(&lower_matrices, penalty);
            return lower_matrices
                .into_iter()
                .enumerate()
                .map(|(b, lower)| {
                    let precomputed_transform = transforms.as_ref().map(|t| t[b].clone());
                    gaussian_reml_eigen_cache_from_lower_with_transform(
                        lower,
                        penalty,
                        nullspace_dim,
                        fingerprints[b],
                        precomputed_transform,
                    )
                })
                .collect();
        }
    }

    let mut results = Vec::with_capacity(k);
    for (b, xtwx) in xtwx_matrices.into_iter().enumerate() {
        let lower = match gaussian_reml_cholesky_lower(xtwx) {
            Ok(l) => l,
            Err(err) => {
                results.push(Err(err));
                continue;
            }
        };
        results.push(gaussian_reml_eigen_cache_from_lower_with_transform(
            lower,
            penalty,
            nullspace_dim,
            fingerprints[b],
            None,
        ));
    }
    results
}

fn batched_whitened_penalty_transforms(
    lowers: &[Array2<f64>],
    penalty: ArrayView2<'_, f64>,
) -> Option<Vec<Array2<f64>>> {
    let first = lowers.first()?;
    let p = first.nrows();
    if p == 0 || first.ncols() != p || lowers.iter().any(|lower| lower.dim() != (p, p)) {
        return None;
    }
    let mut linv_stack = Array3::<f64>::zeros((lowers.len(), p, p));
    for (idx, lower) in lowers.iter().enumerate() {
        let l_inv = invert_lower_triangular(lower).ok()?;
        linv_stack.slice_mut(s![idx, .., ..]).assign(&l_inv);
    }
    let penalty_in_metric =
        crate::gpu::try_fast_ab_broadcast_b_batched(linv_stack.view(), penalty)?;
    let transformed =
        crate::gpu::try_fast_abt_strided_batched(penalty_in_metric.view(), linv_stack.view())?;
    Some(
        transformed
            .axis_iter(Axis(0))
            .map(|matrix| matrix.to_owned())
            .collect(),
    )
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
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
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
        crate::bail_invalid_estim!(
            "Gaussian REML penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        );
    }
    if x.iter().chain(penalty.iter()).any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }
    if let Some(w) = weights {
        if w.len() != n {
            crate::bail_invalid_estim!(
                "Gaussian REML weights length mismatch: expected {n}, got {}",
                w.len()
            );
        }
        if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
            crate::bail_invalid_estim!("Gaussian REML weights must be finite and non-negative");
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
                crate::bail_invalid_estim!(
                    "Gaussian REML weights length mismatch: expected {n}, got {}",
                    w.len()
                );
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                crate::bail_invalid_estim!("Gaussian REML weights must be finite and non-negative");
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
    let xtwx_fingerprint = matrix_fingerprint(xtwx.view());
    let lower = gaussian_reml_cholesky_lower(xtwx)?;
    gaussian_reml_eigen_cache_from_lower(lower, penalty, nullspace_dim, xtwx_fingerprint)
}

/// Cache-build entry point for callers that have already computed `L =
/// chol(X'WX, lower)`. Used by the batched K-way fit path so a single
/// `cusolverDnDpotrfBatched` call factors all K matrices, then each cache
/// finishes per-fit without re-doing the Cholesky.
fn gaussian_reml_eigen_cache_from_lower(
    lower: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    xtwx_fingerprint: u64,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    gaussian_reml_eigen_cache_from_lower_with_transform(
        lower,
        penalty,
        nullspace_dim,
        xtwx_fingerprint,
        None,
    )
}

/// Cache-build variant that accepts a pre-computed whitened penalty
/// `L⁻¹·S·L⁻ᵀ`. Callers pass `None` to compute it from the Cholesky factor.
fn gaussian_reml_eigen_cache_from_lower_with_transform(
    lower: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    xtwx_fingerprint: u64,
    precomputed_transform: Option<Array2<f64>>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let p = lower.nrows();
    if lower.ncols() != p {
        crate::bail_invalid_estim!("Gaussian REML Cholesky factor must be square");
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    let logdet_xtwx = 2.0 * lower.diag().iter().map(|v| v.ln()).sum::<f64>();
    let transformed_penalty = match precomputed_transform {
        Some(transformed) => transformed,
        None => {
            let l_inv = invert_lower_triangular(&lower)?;
            let penalty_in_metric = dense_ab(l_inv.view(), penalty);
            dense_ab(penalty_in_metric.view(), l_inv.t())
        }
    };
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
            crate::bail_invalid_estim!(
                "Gaussian REML penalty is not positive semidefinite; eigenvalue={value:.3e}"
            );
        }
    }
    let penalty_rank = penalty_eigenvalues
        .iter()
        .filter(|&&value| value > eig_tol)
        .count();
    let nullity = p - penalty_rank;
    if let Some(expected_nullity) = nullspace_dim
        && expected_nullity != nullity
    {
        crate::bail_invalid_estim!(
            "Gaussian REML penalty nullspace mismatch: expected {expected_nullity}, inferred {nullity}"
        );
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

fn gaussian_reml_cholesky_lower(xtwx: Array2<f64>) -> Result<Array2<f64>, EstimationError> {
    // Attempt Cholesky directly; on failure, retry with a tiny diagonal jitter
    // proportional to the matrix trace. X'WX is symmetric positive semidefinite
    // by construction, but FP noise (e.g. in a basis whose kernel block is only
    // FP-orthogonal to its explicit polynomial nullspace columns, as the
    // periodic Duchon basis is) can push the smallest eigenvalue slightly
    // negative on adversarial inputs, intermittently failing Cholesky. A
    // jitter of 1e-12 * trace/p shifts every eigenvalue up by an amount well
    // below the natural scale of the well-conditioned eigenvalues but well
    // above f64 FP noise, eliminating the spurious-failure regime.
    let mut gpu_candidate = xtwx.clone();
    if crate::gpu::try_cholesky_lower_inplace(&mut gpu_candidate).is_some() {
        return Ok(gpu_candidate);
    }
    if let Ok(chol) = xtwx.cholesky(Side::Lower) {
        return Ok(chol.lower_triangular());
    }
    let p = xtwx.nrows();
    let trace: f64 = (0..p).map(|i| xtwx[[i, i]]).sum();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let mut jitter = 1e-12 * trace / (p as f64);
    for _ in 0..6 {
        let mut jittered = xtwx.clone();
        for i in 0..p {
            jittered[[i, i]] += jitter;
        }
        let mut gpu_candidate = jittered.clone();
        if crate::gpu::try_cholesky_lower_inplace(&mut gpu_candidate).is_some() {
            return Ok(gpu_candidate);
        }
        if let Ok(chol) = jittered.cholesky(Side::Lower) {
            return Ok(chol.lower_triangular());
        }
        jitter *= 10.0;
    }
    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
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
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache dimension mismatch: expected {p} coefficients"
        );
    }
    if cache.penalty_rank > p || cache.nullity > p || cache.penalty_rank + cache.nullity != p {
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache rank/nullity mismatch: rank={}, nullity={}, p={p}",
            cache.penalty_rank,
            cache.nullity
        );
    }
    if !(cache.logdet_xtwx.is_finite() && cache.logdet_penalty_positive.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML eigen cache log-determinants must be finite");
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
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache entries must be finite with non-negative eigenvalues"
                .to_string(),
        );
    }
    Ok::<(), _>(())
}

fn prepare_gaussian_reml(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlPrepared, EstimationError> {
    // Enforce the symmetric-S contract once at the central forward chokepoint;
    // every closed-form forward path funnels through here.
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    if y.nrows() != n {
        crate::bail_invalid_estim!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        );
    }
    if y.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
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
            crate::bail_invalid_estim!("Gaussian REML eigen cache X'WX mismatch");
        }
        let penalty_fingerprint = matrix_fingerprint(penalty);
        if cache.penalty_fingerprint != penalty_fingerprint {
            crate::bail_invalid_estim!("Gaussian REML eigen cache penalty mismatch");
        }
        if let Some(expected_nullity) = nullspace_dim
            && expected_nullity != cache.nullity
        {
            crate::bail_invalid_estim!(
                "Gaussian REML eigen cache nullspace mismatch: expected {expected_nullity}, got {}",
                cache.nullity
            );
        }
        if n <= cache.nullity {
            crate::bail_invalid_estim!(
                "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
                cache.nullity
            );
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
        crate::bail_invalid_estim!(
            "Gaussian REML requires n > nullspace dimension; got n={n}, nullity={}",
            cache.nullity
        );
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
    let mut grid = Vec::<(f64, f64)>::with_capacity(GRID_INTERVALS + 1);
    let mut prev_rho = RHO_LOWER;
    let mut prev_eval = prepared.evaluate(prev_rho);
    grid.push((prev_rho, prev_eval.cost));
    for i in 1..=GRID_INTERVALS {
        let rho = RHO_LOWER + (RHO_UPPER - RHO_LOWER) * (i as f64) / (GRID_INTERVALS as f64);
        let eval = prepared.evaluate(rho);
        grid.push((rho, eval.cost));
        if prev_eval.grad <= 0.0 && eval.grad >= 0.0 {
            push_candidate(
                &mut stationary,
                refine_stationary_rho(prepared, prev_rho, rho, 0.5 * (prev_rho + rho)),
            );
        }
        prev_rho = rho;
        prev_eval = eval;
    }

    let mut candidates = stationary;
    push_candidate(&mut candidates, RHO_LOWER);
    push_candidate(&mut candidates, RHO_UPPER);
    if let Some(rho0) = init_rho {
        push_candidate(&mut candidates, rho0);
    }
    if let Some(rho) = refine_best_grid_cell(prepared, &grid) {
        push_candidate(&mut candidates, rho);
    }

    // Evaluate each candidate exactly once. `min_by` over a comparator that
    // re-evaluates would do O(n log n) extra `prepared.evaluate` calls during
    // the sort.
    candidates
        .into_iter()
        .map(|rho| (rho, prepared.evaluate(rho).cost))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(rho, _)| rho)
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian REML optimizer produced no candidates".to_string(),
            )
        })
}

fn refine_best_grid_cell(prepared: &GaussianRemlPrepared, grid: &[(f64, f64)]) -> Option<f64> {
    let best_idx = grid
        .iter()
        .enumerate()
        .filter(|(_, (_, cost))| cost.is_finite())
        .min_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))
        .map(|(idx, _)| idx)?;
    if best_idx == 0 || best_idx + 1 == grid.len() {
        return Some(grid[best_idx].0);
    }
    // The best interior grid cell brackets a genuine REML minimum (its cost is
    // below both neighbours), so the objective gradient changes sign across
    // `[grid[i-1], grid[i+1]]`. Refine to that stationary point (∂V/∂ρ = 0)
    // rather than minimising the cost with a golden section: the cost-based
    // search only locates ρ to ~√ε of the cell (~1e-8), whereas the
    // grad-sign-change branch already contributes stationary candidates
    // converged to GRAD_TOL (~1e-12). When both target the same minimum, the
    // ~1e-16 cost ordering between a 1e-8-accurate and a 1e-12-accurate ρ is
    // numerical noise, so `min_by(cost)` used to pick between two ρ values
    // ~1e-8 apart essentially at random — making the selected λ̂ a
    // non-smooth function of the design X (its ~1e-8 jumps wrecked the
    // closed-form REML reverse-mode VJP's agreement with finite differences).
    // Returning the stationary point makes every interior candidate a
    // GRAD_TOL-accurate root, so the residual selection jitter collapses to
    // ~1e-12 and λ̂(X) is smooth to the IFT gradient.
    Some(refine_stationary_rho(
        prepared,
        grid[best_idx - 1].0,
        grid[best_idx + 1].0,
        grid[best_idx].0,
    ))
}

fn fill_weighted_rhs_no_alloc(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    workspace: &mut GaussianRemlNoAllocWorkspace,
) -> Result<(), EstimationError> {
    let d = y.ncols();

    // XᵀWY and YᵀWY via faer BLAS. Both `fast_xt_diag_y` and `fast_atb`
    // dispatch to faer's SIMD-optimized GEMM (with chunked weight scaling
    // when weights are present), replacing the previous scalar triple loop
    // over (n, p, d). For YᵀWY we only need the diagonal entries, but d is
    // small (typically 1–10) so computing the full d×d Gram is negligible.
    let (xtwy, ywy_full) = match weights {
        Some(w) => (fast_xt_diag_y(&x, &w, &y), fast_xt_diag_y(&y, &w, &y)),
        None => (fast_atb(&x, &y), fast_atb(&y, &y)),
    };
    workspace.xtwy.assign(&xtwy);
    for output in 0..d {
        workspace.ywy[output] = ywy_full[[output, output]];
    }

    if workspace
        .xtwy
        .iter()
        .chain(workspace.ywy.iter())
        .any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!("Gaussian REML weighted cross-products must be finite");
    }
    Ok(())
}

fn project_rhs_no_alloc(
    cache: &GaussianRemlEigenCache,
    workspace: &mut GaussianRemlNoAllocWorkspace,
) {
    // projected_rhs = coefficient_basisᵀ · xtwy, computed via faer BLAS
    // (was previously a scalar triple loop over (p, d, p)).
    let projected = fast_atb(&cache.coefficient_basis, &workspace.xtwy);
    workspace.projected_rhs.assign(&projected);
    let p = cache.penalty_eigenvalues.len();
    let d = workspace.ywy.len();
    for eig in 0..p {
        for output in 0..d {
            let value = workspace.projected_rhs[[eig, output]];
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

    // Each term's value and its ρ-derivatives come back from ONE function so
    // they cannot be edited independently; `+=` folds the triple in lock-step.
    let (logdet_term, edf) = gaussian_reml_logdet_term(cache, rho, d);
    let mut eval = ObjectiveEval {
        cost: 0.0,
        grad: 0.0,
        hess: 0.0,
        edf,
    };
    eval += logdet_term;
    for output in 0..n_outputs {
        eval +=
            gaussian_reml_dispersion_term(cache, ywy, projected_rhs_squared, output, nu, lambda);
    }
    eval
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

    const GRID_INTERVALS: usize = 96;
    let mut grid = Vec::<(f64, f64)>::with_capacity(GRID_INTERVALS + 1);
    let mut prev_rho = RHO_LOWER;
    let mut prev_eval = lower_eval;
    grid.push((prev_rho, prev_eval.cost));
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
        grid.push((rho, eval.cost));
        if prev_eval.grad <= 0.0 && eval.grad >= 0.0 {
            let stationary_rho = refine_stationary_rho_no_alloc(
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
                stationary_rho,
                &mut best_rho,
                &mut best_cost,
            );
        }
        prev_rho = rho;
        prev_eval = eval;
    }
    if let Some(best_idx) = grid
        .iter()
        .enumerate()
        .filter(|(_, (_, cost))| cost.is_finite())
        .min_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))
        .map(|(idx, _)| idx)
    {
        let refined = if best_idx == 0 || best_idx + 1 == grid.len() {
            grid[best_idx].0
        } else {
            // Refine the best interior grid cell to the REML stationary point
            // (∂V/∂ρ = 0) rather than the golden-section cost minimum, mirroring
            // the allocating `refine_best_grid_cell`. A cost-based search locates
            // ρ only to ~1e-8, which competed against the GRAD_TOL-accurate
            // (~1e-12) stationary candidates in the cost `min_by` below and made
            // the selected λ̂ jump ~1e-8 with the design — a non-smoothness the
            // closed-form REML VJP could not match under finite differences.
            // (Keeping both optimizers' refinement identical preserves their
            // allocating/no-alloc bit-for-bit parity.)
            refine_stationary_rho_no_alloc(
                cache,
                ywy,
                projected_rhs_squared,
                n_observations,
                n_outputs,
                grid[best_idx - 1].0,
                grid[best_idx + 1].0,
                grid[best_idx].0,
            )
        };
        consider_rho_no_alloc(
            cache,
            ywy,
            projected_rhs_squared,
            n_observations,
            n_outputs,
            refined,
            &mut best_rho,
            &mut best_cost,
        );
    }

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
        crate::bail_invalid_estim!("lower-triangular solve requires a square matrix");
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
        crate::bail_invalid_estim!("lower-triangular solve dimension mismatch");
    }
    if let Some(out) = crate::gpu::try_solve_lower_triangular_matrix(lower.view(), rhs.view()) {
        return Ok(out);
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

/// Solve the SPD system `L Lᵀ X = rhs` for `X` given the lower Cholesky factor
/// `L` (as returned by [`gaussian_reml_cholesky_lower`]): a forward solve
/// against `L` followed by a back solve against `Lᵀ`.
fn solve_spd_from_lower_factor(
    lower: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let forward = solve_lower_triangular_matrix(lower, rhs)?;
    solve_upper_triangular_matrix(&lower.t().to_owned(), &forward)
}

fn solve_upper_triangular_matrix(
    upper: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let n = upper.nrows();
    if upper.ncols() != n || rhs.nrows() != n {
        crate::bail_invalid_estim!("upper-triangular solve dimension mismatch");
    }
    if let Some(out) = crate::gpu::try_solve_upper_triangular_matrix(upper.view(), rhs.view()) {
        return Ok(out);
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
        Lambda,
        RemlScore,
        Coefficient(usize, usize),
        Fitted(usize, usize),
        Edf,
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
        // The truth must NOT lie (essentially) in span(X). The 5-column design
        // is Legendre P_0..P_4, so a low-order polynomial + low-frequency sin
        // would be fit to near machine precision — driving σ² → 0, dp → 0,
        // and ∂score/∂y ≈ ν w r / dp → ∞. Central finite differences with
        // Richardson extrapolation cannot resolve such steep, highly-nonlinear
        // surfaces at 1e-6 relative because the truncation term scales with
        // f^(5)(y), which explodes in that regime. The high-frequency sin
        // below is well outside span(P_0..P_4) on t ∈ [-0.95, 0.95], leaving
        // a genuine residual (σ² ≈ 1e-3) and an interior REML optimum
        // (ρ ≈ -3) at which the analytic-vs-FD comparison is meaningful.
        Array2::from_shape_fn((20, outputs), |(row, output)| {
            let t = (row as f64 - 9.5) / 10.0;
            let phase = output as f64 + 1.0;
            0.2 + 0.25 * phase * t - 0.12 * t * t
                + (0.08 + 0.03 * phase) * (1.1 * t + 0.3 * phase).sin()
                + 0.05 * (7.0 * t + 0.5 * phase).sin()
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

    /// Fallible forward-scalar probe. Returns `None` when the closed-form fit
    /// rejects the inputs — the relevant case being a penalty perturbation that
    /// pushes `S` out of the PSD cone (a single-entry central bump on a
    /// null-direction entry drives one eigenvalue slightly negative). Such a
    /// point has no well-defined REML objective, so the caller skips it rather
    /// than panicking.
    fn one_hot_objective_try(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> Option<f64> {
        let fit = gaussian_reml_multi_closed_form_with_cache(
            x,
            y,
            penalty,
            Some(weights),
            Some(0.85),
            None,
        )
        .ok()?;
        Some(match target {
            ForwardScalar::Lambda => fit.lambda,
            ForwardScalar::RemlScore => fit.reml_score,
            ForwardScalar::Coefficient(row, col) => fit.coefficients[[row, col]],
            ForwardScalar::Fitted(row, col) => fit.fitted[[row, col]],
            ForwardScalar::Edf => fit.edf,
        })
    }

    fn one_hot_objective(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> f64 {
        one_hot_objective_try(x, y, penalty, weights, target)
            .expect("finite-difference forward fit")
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
        let (grad_lambda, grad_score, grad_edf, coefficient_upstream, fitted_upstream) =
            match target {
                ForwardScalar::Lambda => (1.0, 0.0, 0.0, None, None),
                ForwardScalar::RemlScore => (0.0, 1.0, 0.0, None, None),
                ForwardScalar::Coefficient(row, col) => {
                    grad_coefficients[[row, col]] = 1.0;
                    (0.0, 0.0, 0.0, Some(grad_coefficients.view()), None)
                }
                ForwardScalar::Fitted(row, col) => {
                    grad_fitted[[row, col]] = 1.0;
                    (0.0, 0.0, 0.0, None, Some(grad_fitted.view()))
                }
                ForwardScalar::Edf => (0.0, 0.0, 1.0, None, None),
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
            grad_edf,
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
            ForwardScalar::Lambda,
            ForwardScalar::RemlScore,
            ForwardScalar::Coefficient(3, outputs - 1),
            ForwardScalar::Fitted(12, outputs - 1),
            ForwardScalar::Edf,
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

            // ∂L/∂S over the RANGE-SPACE penalty entries. The REML objective
            // carries −½d·log|S|₊ (the pseudo-determinant over the NONZERO
            // eigenvalues), so ∂L/∂S is only a finite, FD-verifiable derivative
            // where a central ±h bump keeps S inside the PSD cone WITHOUT
            // changing its rank. A single-entry bump touching the null
            // direction violates both: the −h side drives an eigenvalue
            // slightly negative (leaves the cone → fit Err) and the +h side
            // turns the zero eigenvalue into a tiny positive one that joins
            // log|S|₊ as a −log(ε) term (a rank-change discontinuity in L).
            // The null-direction component of the analytic S-gradient is a
            // gauge convention for the null space (the L-metric pseudoinverse
            // `penalty_pinv` = L⁻ᵀ T⁺ L⁻¹), validated by algebra/consumer, not
            // FD. So restrict to the strictly-positive diagonal block (both
            // indices in 1..p for the diag([0, 0.8, 1.2, 1.7, 2.3]) fixture,
            // where S_rr > 0 and ±h stays PSD at full rank). The forward
            // consumes only `S_canon = 0.5(S + Sᵀ)` and the backward returns
            // the symmetrized gradient, so a single-entry bump of S[r, c]
            // (asymmetric) compares directly against `grad_penalty[r, c]` =
            // 0.5(G[r, c] + G[c, r]). Defensively, any entry whose largest ±h
            // probe leaves the cone is skipped (cone membership is monotone in
            // |h| here, so probing the largest step suffices).
            let null_index = 0usize; // diag([0.0, ...]) ⇒ coordinate 0 is the null direction.
            let probe_h = 1.0e-3_f64; // matches the largest adaptive_central_difference step.
            for r in 0..penalty.nrows() {
                for c in 0..penalty.ncols() {
                    if r == null_index || c == null_index {
                        continue;
                    }
                    let eval = |delta: f64| {
                        let mut candidate = penalty.clone();
                        candidate[[r, c]] += delta;
                        one_hot_objective(
                            x.view(),
                            y.view(),
                            candidate.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let cone_safe = {
                        let mut s_plus = penalty.clone();
                        let mut s_minus = penalty.clone();
                        s_plus[[r, c]] += probe_h;
                        s_minus[[r, c]] -= probe_h;
                        one_hot_objective_try(
                            x.view(),
                            y.view(),
                            s_plus.view(),
                            weights.view(),
                            target,
                        )
                        .is_some()
                            && one_hot_objective_try(
                                x.view(),
                                y.view(),
                                s_minus.view(),
                                weights.view(),
                                target,
                            )
                            .is_some()
                    };
                    if !cone_safe {
                        continue;
                    }
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} penalty[{r},{c}]"),
                        backward.grad_penalty[[r, c]],
                        fd,
                    );
                }
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
            0.0,
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

        // Combined-seed ∂L/∂S spot-check: perturb individual penalty entries with
        // x/y/w held at base, under mixed (λ, score, β, fitted) seeds. The penalty
        // [[0,0,0],[0,1,0.2],[0,0.2,1.7]] is nullity 1 (coordinate 0 is the null
        // direction); ∂L/∂S is FD-verifiable only on the strictly-positive
        // RANGE block (indices 1,2), where a central ±h bump keeps S PSD at full
        // rank. Null-touching entries (any index 0) are non-FD-verifiable — the
        // −½d·log|S|₊ pseudo-determinant term makes L either cone-leaving or
        // rank-change-discontinuous there (see the exhaustive S loop above). A
        // single-entry asymmetric bump of S[r, c] compares directly to
        // grad_penalty[[r, c]] = 0.5(G[r,c] + G[c,r]), exercising the backward
        // symmetrization.
        let objective_s = |s_eval: &Array2<f64>| {
            let fit = gaussian_reml_multi_closed_form_with_cache(
                x.view(),
                y.view(),
                s_eval.view(),
                Some(weights.view()),
                Some(0.8),
                None,
            )
            .expect("fit for penalty objective");
            upstream_lambda * fit.lambda
                + upstream_score * fit.reml_score
                + (&fit.coefficients * &upstream_coefficients).sum()
                + (&fit.fitted * &upstream_fitted).sum()
        };
        // (1,1) full-rank diagonal; (1,2) pure off-diagonal between two penalized
        // directions; (2,2) full-rank diagonal. All in the strictly-positive
        // range block, so ±h stays PSD at full rank.
        for (r, c) in [(1usize, 1usize), (1, 2), (2, 2)] {
            let mut s_plus = penalty.clone();
            let mut s_minus = penalty.clone();
            s_plus[[r, c]] += eps;
            s_minus[[r, c]] -= eps;
            let fd_s = (objective_s(&s_plus) - objective_s(&s_minus)) / (2.0 * eps);
            assert!(
                (fd_s - backward.grad_penalty[[r, c]]).abs() <= 2.0e-4,
                "grad_penalty[{r},{c}] mismatch: analytic={} fd={}",
                backward.grad_penalty[[r, c]],
                fd_s
            );
        }
    }

    #[test]
    fn batched_eigen_cache_matches_per_fit_build() {
        // Three K=3 problems sharing the same penalty matrix. The batched
        // pipeline must produce caches that are bit-exact identical to what
        // the per-fit `gaussian_reml_eigen_cache_from_xtwx` builder produces,
        // regardless of whether the GPU batched Cholesky kicks in or the
        // helper falls through to per-fit Cholesky.
        let xtwx_a = array![[4.0, 1.0], [1.0, 3.0]];
        let xtwx_b = array![[2.5, -0.5], [-0.5, 1.7]];
        let xtwx_c = array![[7.2, 0.3], [0.3, 5.1]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        let batched = build_gaussian_reml_eigen_cache_batched(
            vec![xtwx_a.clone(), xtwx_b.clone(), xtwx_c.clone()],
            penalty.view(),
            None,
        );
        assert_eq!(batched.len(), 3);

        for (xtwx, batched_cache) in [&xtwx_a, &xtwx_b, &xtwx_c].into_iter().zip(batched.iter()) {
            let single = gaussian_reml_eigen_cache_from_xtwx(xtwx.clone(), penalty.view(), None)
                .expect("per-fit cache");
            let batched_cache = batched_cache.as_ref().expect("batched cache");
            assert_eq!(batched_cache.penalty_rank, single.penalty_rank);
            assert_eq!(batched_cache.nullity, single.nullity);
            assert_eq!(batched_cache.xtwx_fingerprint, single.xtwx_fingerprint);
            assert_eq!(
                batched_cache.penalty_fingerprint,
                single.penalty_fingerprint
            );
            assert!((batched_cache.logdet_xtwx - single.logdet_xtwx).abs() <= 1.0e-12);
            assert!(
                (batched_cache.logdet_penalty_positive - single.logdet_penalty_positive).abs()
                    <= 1.0e-12
            );
            for (a, b) in batched_cache
                .penalty_eigenvalues
                .iter()
                .zip(single.penalty_eigenvalues.iter())
            {
                assert!((a - b).abs() <= 1.0e-12);
            }
            for ((a, b), _) in batched_cache
                .coefficient_basis
                .iter()
                .zip(single.coefficient_basis.iter())
                .zip(0..)
            {
                assert!((a - b).abs() <= 1.0e-12);
            }
        }
    }

    #[test]
    fn scalar_rho_optimizer_chooses_lowest_cost_stationary_point() {
        let cache = GaussianRemlEigenCache {
            penalty_eigenvalues: array![5.2430192311066924e-05, 81734184.18548436],
            eigenvectors: Array2::eye(2),
            coefficient_basis: Array2::eye(2),
            xtwx_fingerprint: 0,
            penalty_fingerprint: 0,
            logdet_xtwx: 0.0,
            logdet_penalty_positive: 0.0,
            penalty_rank: 2,
            nullity: 0,
        };
        let prepared = GaussianRemlPrepared {
            cache: cache.clone(),
            ywy: array![0.5021347226586624],
            projected_rhs_squared: array![[0.361060218768292], [0.01014486085547482]],
            projected_rhs: array![
                [0.361060218768292_f64.sqrt()],
                [0.01014486085547482_f64.sqrt()]
            ],
            n_observations: 100,
            n_outputs: 1,
        };

        let rho = optimize_rho(&prepared, None).expect("allocating rho optimizer");
        let no_alloc_rho = optimize_rho_no_alloc(
            &cache,
            prepared.ywy.view(),
            prepared.projected_rhs_squared.view(),
            prepared.n_observations,
            prepared.n_outputs,
            None,
        )
        .expect("no-alloc rho optimizer");

        assert!(
            (rho - 4.3251059890).abs() < 1.0e-6,
            "rho optimizer selected {rho}, expected the lower-cost later stationary point"
        );
        assert!(
            (no_alloc_rho - rho).abs() < 1.0e-8,
            "no-alloc optimizer selected {no_alloc_rho}, allocating selected {rho}"
        );
        assert!(prepared.evaluate(rho).cost < prepared.evaluate(-18.9277503549).cost);
    }

    #[test]
    fn backward_from_fit_matches_backward_with_refit() {
        // The Task 3 state round-trip in pyffi calls `_from_fit`; that path
        // must be numerically identical to the refitting `_backward` entry
        // when fed the same forward result. This guards the optimization
        // against drift when either path is touched.
        let x = array![[1.0, -0.9], [1.0, -0.4], [1.0, 0.1], [1.0, 0.6], [1.0, 1.1],];
        let y = array![[0.2, -0.1], [0.4, 0.1], [0.7, 0.3], [1.0, 0.5], [1.5, 0.8]];
        let penalty = array![[0.0, 0.0], [0.0, 1.5]];
        let weights = array![1.05, 0.95, 1.01, 0.99, 1.03];

        let refit = gaussian_reml_multi_closed_form_backward(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.85),
            0.2,
            None,
            None,
            -0.1,
            0.0,
        )
        .expect("refit backward");

        let fit = gaussian_reml_multi_closed_form_with_cache(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.85),
            None,
        )
        .expect("forward fit");
        let from_fit = gaussian_reml_multi_closed_form_backward_from_fit(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            &fit,
            0.2,
            None,
            None,
            -0.1,
            0.0,
        )
        .expect("from_fit backward");

        for (a, b) in refit.grad_x.iter().zip(from_fit.grad_x.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_y.iter().zip(from_fit.grad_y.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_weights.iter().zip(from_fit.grad_weights.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
    }

    /// Regression: when `K = XᵀWX + λS` is effectively rank-deficient (e.g.
    /// `λ` has saturated very large), the backward must NOT error — it must
    /// degrade gracefully and return zero gradients of the correct shape.
    /// This is the production-training scenario where individual atoms can
    /// saturate `λ_k` in early batches; raising here would crash an entire
    /// step. We construct the degenerate state by running a real forward
    /// fit and then corrupting `reml_hess_rho` to 0 (the gate variable the
    /// backward checks). We assert: (a) no error, (b) all gradients finite,
    /// (c) shapes match the inputs.
    #[test]
    fn backward_degrades_gracefully_when_k_is_near_singular() {
        // Small, full-rank S with a moderately-conditioned X. The exact
        // numbers don't matter; what matters is that we then force the
        // ill-conditioned gate to fire.
        let x = array![
            [1.0, -1.0, 0.5],
            [1.0, -0.5, 0.2],
            [1.0, 0.0, -0.1],
            [1.0, 0.5, 0.3],
            [1.0, 1.0, 0.8],
            [1.0, 1.5, 1.1],
            [1.0, 2.0, 1.5],
            [1.0, 2.5, 2.0],
            [1.0, 3.0, 2.6],
            [1.0, 3.5, 3.1],
        ];
        let y = array![
            [0.1],
            [0.3],
            [0.4],
            [0.7],
            [1.0],
            [1.5],
            [2.0],
            [2.7],
            [3.3],
            [4.0]
        ];
        // Full-rank S to keep the forward well-posed.
        let penalty = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut fit =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("forward fit must succeed for well-posed input");
        // Force the ill-conditioned gate to fire by zeroing the REML
        // Hessian w.r.t. rho — this is exactly what happens in production
        // when `λ` saturates to 1e10+ and `d²ℓ/dρ² → 0`.
        fit.reml_hess_rho = 0.0;

        let result = gaussian_reml_multi_closed_form_backward_from_fit(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            &fit,
            // Nonzero upstreams to force the backward to actually try to
            // populate gradients (rather than short-circuit on zero seeds).
            1.0,
            None,
            None,
            1.0,
            1.0,
        )
        .expect("backward must NOT error on near-singular K");

        assert_eq!(result.grad_x.dim(), (x.nrows(), x.ncols()));
        assert_eq!(result.grad_y.dim(), (y.nrows(), y.ncols()));
        assert_eq!(result.grad_penalty.dim(), (x.ncols(), x.ncols()));
        assert_eq!(result.grad_weights.dim(), x.nrows());
        for v in result.grad_x.iter() {
            assert!(v.is_finite(), "grad_x must be finite, got {v}");
        }
        for v in result.grad_y.iter() {
            assert!(v.is_finite(), "grad_y must be finite, got {v}");
        }
        for v in result.grad_penalty.iter() {
            assert!(v.is_finite(), "grad_penalty must be finite, got {v}");
        }
        for v in result.grad_weights.iter() {
            assert!(v.is_finite(), "grad_weights must be finite, got {v}");
        }
    }
}

/// Vector–Jacobian products of the multi-block per-smooth-λ Gaussian REML
/// forward fit ([`gaussian_reml_blocks_orthogonal_shared_scale`]), back to the
/// design blocks, penalty blocks, response, and weights.
pub struct GaussianRemlBlocksBackwardAnalytic {
    pub grad_designs: Vec<Array2<f64>>,
    pub grad_penalties: Vec<Array2<f64>>,
    pub grad_y: Array2<f64>,
    pub grad_weights: Array1<f64>,
}

/// Analytic backward for the multi-block per-smooth-λ Gaussian REML forward.
///
/// Computes VJPs of (coefficients, fitted, lambdas, log_lambdas, reml_score,
/// edf) back to (design_blocks, penalty_blocks, y, weights). The VJP is
/// assembled at the converged log-λ vector: fixed-ρ β/fitted/profiled-REML/EDF
/// terms are accumulated first, then the smoothing-parameter sensitivity is
/// routed through the F×F profiled REML score Hessian from the implicit optimum.
/// Pairs with the forward [`gaussian_reml_blocks_orthogonal_shared_scale`].
pub fn gaussian_reml_fit_blocks_backward_analytic(
    designs: &[Array2<f64>],
    penalties_raw: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    rhos: &[f64],
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_lambdas: Option<ArrayView1<'_, f64>>,
    grad_log_lambdas: Option<ArrayView1<'_, f64>>,
    grad_reml_score: f64,
    grad_edf: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlBlocksBackwardAnalytic, EstimationError> {
    let n = y.len();
    let f_blocks = designs.len();
    let mut offsets = Vec::with_capacity(f_blocks + 1);
    offsets.push(0_usize);
    for design in designs {
        offsets.push(offsets.last().copied().unwrap() + design.ncols());
    }
    let p_total = *offsets.last().unwrap();
    if n == 0 || p_total == 0 {
        return Err(EstimationError::InvalidInput(
            "gaussian_reml_fit_blocks_backward requires non-empty rows and at least one coefficient column"
                .to_string(),
        ));
    }

    if rhos.len() != f_blocks {
        return Err(EstimationError::InvalidInput(format!(
            "log_lambdas length mismatch: expected {f_blocks}, got {}",
            rhos.len()
        )));
    }
    if let Some(gc) = grad_coefficients {
        if gc.dim() != (p_total, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients shape mismatch: expected {}x1, got {}x{}",
                p_total,
                gc.nrows(),
                gc.ncols()
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if gf.dim() != (n, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted shape mismatch: expected {}x1, got {}x{}",
                n,
                gf.nrows(),
                gf.ncols()
            )));
        }
    }
    if !grad_reml_score.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "grad_reml_score must be finite; got {grad_reml_score}"
        )));
    }
    if let Some(vec) = grad_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(gc) = grad_coefficients {
        if let Some(((row, col), value)) = gc.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if let Some(((row, col), value)) = gf.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf[{block}] must be finite; got {value}"
            )));
        }
    }
    for (block, design) in designs.iter().enumerate() {
        if let Some(((row, col), value)) =
            design.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "designs[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    for (block, penalty) in penalties_raw.iter().enumerate() {
        if let Some(((row, col), value)) =
            penalty.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "penalties[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some((row, value)) = y.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "y[{row}] must be finite; got {value}"
        )));
    }
    if let Some((row, value)) = weights
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || **value < 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "weights[{row}] must be finite and non-negative; got {value}"
        )));
    }

    let mut z = Array2::<f64>::zeros((n, p_total));
    for k in 0..f_blocks {
        z.slice_mut(s![.., offsets[k]..offsets[k + 1]])
            .assign(&designs[k]);
    }

    let penalties: Vec<Array2<f64>> = penalties_raw
        .iter()
        .map(|p| {
            let mut out = p.clone();
            crate::matrix::symmetrize_in_place(&mut out);
            out
        })
        .collect();
    let mut ranks = Vec::with_capacity(f_blocks);
    let mut pinvs = Vec::with_capacity(f_blocks);
    for penalty in &penalties {
        let (rank, pinv) = crate::linalg::utils::block_penalty_rank_and_pinv(penalty)?;
        ranks.push(rank);
        pinvs.push(pinv);
    }

    let lambdas = Array1::from_iter(rhos.iter().map(|rho| rho.exp()));
    if let Some((block, lambda)) = lambdas
        .iter()
        .enumerate()
        .find(|(_, lambda)| !lambda.is_finite() || **lambda <= 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "exp(log_lambdas[{block}]) must be finite and positive; got {lambda}"
        )));
    }
    let mut k_matrix = fast_xt_diag_x(&z.view(), &weights);
    for block in 0..f_blocks {
        let lambda = lambdas[block];
        for local_i in 0..penalties[block].nrows() {
            let global_i = offsets[block] + local_i;
            for local_j in 0..penalties[block].ncols() {
                let global_j = offsets[block] + local_j;
                k_matrix[[global_i, global_j]] += lambda * penalties[block][[local_i, local_j]];
            }
        }
    }
    let r = crate::linalg::utils::invert_spd_with_ridge(&k_matrix, 0.0)?;

    let mut xtwy = Array1::<f64>::zeros(p_total);
    for row in 0..n {
        let wy = weights[row] * y[row];
        for col in 0..p_total {
            xtwy[col] += z[[row, col]] * wy;
        }
    }
    let beta = r.dot(&xtwy);
    let fitted = z.dot(&beta);
    if let Some((col, value)) = beta
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "solved coefficient {col} is non-finite: {value}"
        )));
    }
    let residual = &y.to_owned() - &fitted;
    let weighted_residual = &residual * &weights.to_owned();
    let ywy = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| wi * yi * yi)
        .sum::<f64>();
    let q_raw = ywy - xtwy.dot(&beta);
    if !q_raw.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual quadratic form must be finite; got {q_raw}"
        )));
    }
    let q = q_raw.max(1.0e-300);
    let nullity = penalties
        .iter()
        .zip(ranks.iter())
        .map(|(penalty, rank)| penalty.nrows().saturating_sub(*rank))
        .sum::<usize>();
    let nu = n as f64 - nullity as f64;
    if !(nu.is_finite() && nu > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual degrees of freedom must be positive; got {nu}"
        )));
    }
    let tau = nu / q;
    let tau_q = -nu / (q * q);
    if !(tau.is_finite() && tau_q.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML scale derivatives are non-finite: tau={tau}, tau_q={tau_q}"
        )));
    }

    let mut grad_z = Array2::<f64>::zeros((n, p_total));
    let mut g_kernel = Array2::<f64>::zeros((p_total, p_total));
    let mut h_kernel = Array1::<f64>::zeros(p_total);
    let mut q_kernel = 0.0_f64;
    let mut j_blocks: Vec<Array2<f64>> = penalties
        .iter()
        .map(|p| Array2::<f64>::zeros(p.dim()))
        .collect();

    let mut beta_tilde = Array1::<f64>::zeros(p_total);
    if let Some(gc) = grad_coefficients {
        beta_tilde += &gc.column(0).to_owned();
    }
    if let Some(gf) = grad_fitted {
        let gf_col = gf.column(0).to_owned();
        beta_tilde += &z.t().dot(&gf_col);
        for row in 0..n {
            for col in 0..p_total {
                grad_z[[row, col]] += gf_col[row] * beta[col];
            }
        }
    }

    // Generic downstream losses that explicitly seed beta_hat or fitted
    // values cannot use the REML envelope shortcut. Route those seeds through
    // the fixed-rho KKT adjoint K u = beta_tilde before differentiating
    // designs, penalties, y, weights, and rho.
    let u = r.dot(&beta_tilde);
    h_kernel += &u;
    for i in 0..p_total {
        for j in 0..p_total {
            g_kernel[[i, j]] -= 0.5 * (beta[i] * u[j] + u[i] * beta[j]);
        }
    }

    let mut alpha = Array1::<f64>::zeros(f_blocks);
    if let Some(gl) = grad_lambdas {
        for block in 0..f_blocks {
            alpha[block] += gl[block] * lambdas[block];
        }
    }
    if let Some(grho) = grad_log_lambdas {
        alpha += &grho.to_owned();
    }

    let mut p_betas = Vec::with_capacity(f_blocks);
    let mut m_vectors = Vec::with_capacity(f_blocks);
    let mut rp_matrices = Vec::with_capacity(f_blocks);
    let mut rpr_matrices = Vec::with_capacity(f_blocks);
    let mut b_values = Array1::<f64>::zeros(f_blocks);
    let mut t_values = Array1::<f64>::zeros(f_blocks);

    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let beta_k = beta.slice(s![start..end]).to_owned();
        let s_beta = penalties[block].dot(&beta_k);
        let lambda = lambdas[block];
        let lambda_s_beta = s_beta.mapv(|value| lambda * value);
        let mut p_beta = Array1::<f64>::zeros(p_total);
        for local_i in 0..(end - start) {
            p_beta[start + local_i] = lambda_s_beta[local_i];
        }
        let weighted_penalty = penalties[block].mapv(|value| lambda * value);
        let rp_block = r.slice(s![.., start..end]).dot(&weighted_penalty);
        let mut rp = Array2::<f64>::zeros((p_total, p_total));
        rp.slice_mut(s![.., start..end]).assign(&rp_block);
        let rpr = rp_block.dot(&r.slice(s![start..end, ..]));
        let m = r.slice(s![.., start..end]).dot(&lambda_s_beta);
        b_values[block] = beta.dot(&p_beta);
        t_values[block] = (0..(end - start))
            .map(|local_i| rp_block[[start + local_i, local_i]])
            .sum::<f64>();
        alpha[block] -= u.dot(&p_beta);
        p_betas.push(p_beta);
        m_vectors.push(m);
        rp_matrices.push(rp);
        rpr_matrices.push(rpr);
    }

    if grad_reml_score != 0.0 {
        q_kernel += 0.5 * grad_reml_score * tau;
        g_kernel += &(r.clone() * (0.5 * grad_reml_score));
        for block in 0..f_blocks {
            j_blocks[block] -= &(pinvs[block].clone() * (0.5 * grad_reml_score / lambdas[block]));
        }
    }

    let mut trace_pairs = Array2::<f64>::zeros((f_blocks, f_blocks));
    for i in 0..f_blocks {
        for j in 0..f_blocks {
            trace_pairs[[i, j]] = crate::linalg::utils::trace_of_product(
                rp_matrices[i].view(),
                rp_matrices[j].view(),
            );
        }
    }

    if let Some(ge) = grad_edf {
        for edf_block in 0..f_blocks {
            let scale = ge[edf_block];
            if scale == 0.0 {
                continue;
            }
            let start = offsets[edf_block];
            let end = offsets[edf_block + 1];
            g_kernel += &(rpr_matrices[edf_block].clone() * scale);
            j_blocks[edf_block] -= &(r.slice(s![start..end, start..end]).to_owned() * scale);
            for rho_block in 0..f_blocks {
                alpha[rho_block] += scale * trace_pairs[[edf_block, rho_block]];
                if rho_block == edf_block {
                    alpha[rho_block] -= scale * t_values[edf_block];
                }
            }
        }
    }

    if let Some((block, value)) = alpha
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "rho adjoint seed for block {block} is non-finite: {value}"
        )));
    }

    if alpha.iter().any(|value| *value != 0.0) {
        let mut outer_h = Array2::<f64>::zeros((f_blocks, f_blocks));
        for k in 0..f_blocks {
            for j in 0..f_blocks {
                let beta_pk_r_pj_beta = p_betas[k].dot(&m_vectors[j]);
                outer_h[[k, j]] = 0.5 * trace_pairs[[k, j]] + tau * beta_pk_r_pj_beta
                    - if k == j {
                        0.5 * (t_values[k] + tau * b_values[k])
                    } else {
                        0.0
                    }
                    - 0.5 * tau_q * b_values[k] * b_values[j];
            }
        }
        // `outer_h` is the Jacobian of the negative profiled REML estimating
        // equation. Preserve signed curvature directions while flooring
        // near-zero modes; flipping negative eigenvalues would change the VJP.
        crate::matrix::symmetrize_in_place(&mut outer_h);
        if let Some(((row, col), value)) =
            outer_h.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho curvature entry ({row},{col}) is non-finite: {value}"
            )));
        }
        let rho_adj =
            crate::linalg::utils::solve_symmetric_vector_with_floor(&outer_h, &alpha, 1.0e-10)?;
        if let Some((block, value)) = rho_adj
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho adjoint for block {block} is non-finite: {value}"
            )));
        }
        let weighted_b_sum = rho_adj
            .iter()
            .zip(b_values.iter())
            .map(|(&zk, &bk)| zk * bk)
            .sum::<f64>();
        q_kernel += 0.5 * tau_q * weighted_b_sum;
        for block in 0..f_blocks {
            let zk = rho_adj[block];
            if zk == 0.0 {
                continue;
            }
            g_kernel -= &(rpr_matrices[block].clone() * (0.5 * zk));
            let m = &m_vectors[block];
            for i in 0..p_total {
                h_kernel[i] += tau * zk * m[i];
                for j in 0..p_total {
                    g_kernel[[i, j]] -= 0.5 * tau * zk * (beta[i] * m[j] + m[i] * beta[j]);
                }
            }
            let start = offsets[block];
            let end = offsets[block + 1];
            j_blocks[block] += &(r.slice(s![start..end, start..end]).to_owned() * (0.5 * zk));
            for i in 0..(end - start) {
                for j in 0..(end - start) {
                    j_blocks[block][[i, j]] += 0.5 * tau * zk * beta[start + i] * beta[start + j];
                }
            }
        }
    }

    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += -2.0 * q_kernel * weighted_residual[row] * beta[col];
        }
    }
    let zg = z.dot(&g_kernel);
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += 2.0 * weights[row] * zg[[row, col]];
        }
    }
    let wy = y.to_owned() * &weights.to_owned();
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += wy[row] * h_kernel[col];
        }
    }

    let mut grad_y = Array2::<f64>::zeros((n, 1));
    let zh = z.dot(&h_kernel);
    for row in 0..n {
        grad_y[[row, 0]] = 2.0 * q_kernel * weighted_residual[row] + weights[row] * zh[row];
    }

    let mut grad_weights = Array1::<f64>::zeros(n);
    for row in 0..n {
        let diag_zgz = (0..p_total)
            .map(|col| z[[row, col]] * zg[[row, col]])
            .sum::<f64>();
        grad_weights[row] = q_kernel * residual[row] * residual[row] + diag_zgz + y[row] * zh[row];
    }

    let mut grad_penalties = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let mut local = g_kernel.slice(s![start..end, start..end]).to_owned();
        for i in 0..(end - start) {
            for j in 0..(end - start) {
                local[[i, j]] += q_kernel * beta[start + i] * beta[start + j];
            }
        }
        local += &j_blocks[block];
        local *= lambdas[block];
        crate::matrix::symmetrize_in_place(&mut local);
        grad_penalties.push(local);
    }

    let mut grad_designs = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        grad_designs.push(
            grad_z
                .slice(s![.., offsets[block]..offsets[block + 1]])
                .to_owned(),
        );
    }

    Ok(GaussianRemlBlocksBackwardAnalytic {
        grad_designs,
        grad_penalties,
        grad_y,
        grad_weights,
    })
}

/// Fixed-λ multi-output Gaussian fit under a per-row dense Fisher–Rao precision
/// metric: coefficients, fitted values, per-output residual scale, and the
/// penalized Fisher-weighted objective.
pub struct DenseFisherGaussianFit {
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub sigma2: Array1<f64>,
    pub objective: f64,
}

/// Add a block-diagonal `λ·S` penalty (one `S` block per output) into a stacked
/// `(k·n_outputs)` Hessian in place, symmetrizing `S`.
pub fn add_block_diagonal_penalty(
    hessian: &mut Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    lambda: f64,
    n_outputs: usize,
) -> Result<(), EstimationError> {
    let k = penalty.ncols();
    if penalty.nrows() != k {
        return Err(EstimationError::InvalidInput(format!(
            "penalty must be square for dense Fisher fit; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        )));
    }
    if hessian.dim() != (k * n_outputs, k * n_outputs) {
        return Err(EstimationError::InvalidInput(
            "dense Fisher Hessian shape mismatch while adding penalty".to_string(),
        ));
    }
    for output in 0..n_outputs {
        let offset = output * k;
        for row in 0..k {
            for col in 0..k {
                let s_sym = 0.5 * (penalty[[row, col]] + penalty[[col, row]]);
                hessian[[offset + row, offset + col]] += lambda * s_sym;
            }
        }
    }
    Ok(())
}

/// Closed-form fixed-λ multi-output Gaussian fit with a per-row dense Fisher–Rao
/// precision metric. Assembles the block `XᵀWX` (+ block-diagonal `λS`) and
/// `XᵀWY` via the dense Fisher block kernels, solves, then forms fitted values,
/// per-output residual scale `sigma2`, and the penalized Fisher-weighted
/// objective seeded by `latent_prior_score`. `row_weights` are the (already
/// resolved) per-observation likelihood weights.
pub fn dense_fisher_gaussian_fit(
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    row_weights: ArrayView1<'_, f64>,
    fisher_w: ArrayView3<'_, f64>,
    lambda: f64,
    latent_prior_score: f64,
) -> Result<DenseFisherGaussianFit, EstimationError> {
    let n_obs = design.nrows();
    let k = design.ncols();
    let n_outputs = y.ncols();
    let mut hessian = crate::pirls::dense_block_xtwx(design, fisher_w, Some(row_weights))?;
    add_block_diagonal_penalty(&mut hessian, penalty, lambda, n_outputs)?;
    let rhs = crate::pirls::dense_block_xtwy(design, fisher_w, y, Some(row_weights))?;
    let beta_vec =
        crate::linalg::utils::solve_dense_block_system(&hessian, &rhs, "dense Fisher Gaussian")
            .map_err(EstimationError::InvalidInput)?;
    let mut coefficients = Array2::<f64>::zeros((k, n_outputs));
    for output in 0..n_outputs {
        for col in 0..k {
            coefficients[[col, output]] = beta_vec[output * k + col];
        }
    }
    let fitted = design.dot(&coefficients);
    let mut sigma2 = Array1::<f64>::zeros(n_outputs);
    let mut objective = latent_prior_score;
    for row in 0..n_obs {
        for a in 0..n_outputs {
            let ra = y[[row, a]] - fitted[[row, a]];
            sigma2[a] += row_weights[row] * ra * ra;
            for b in 0..n_outputs {
                objective += 0.5
                    * row_weights[row]
                    * ra
                    * fisher_w[[row, a, b]]
                    * (y[[row, b]] - fitted[[row, b]]);
            }
        }
    }
    for output in 0..n_outputs {
        sigma2[output] /= (n_obs.saturating_sub(k).max(1)) as f64;
        let beta_col = coefficients.column(output);
        let s_beta = penalty.dot(&beta_col);
        objective += 0.5 * lambda * beta_col.dot(&s_beta);
    }
    Ok(DenseFisherGaussianFit {
        coefficients,
        fitted,
        sigma2,
        objective,
    })
}
