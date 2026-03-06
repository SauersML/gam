use crate::estimate::{EstimationError, RHO_BOUND};
use crate::seeding::{SeedConfig, SeedRiskProfile, generate_rho_candidates};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use wolfe_bfgs::{
    Arc as ArcOptimizer, ArcError, NewtonTrustRegion, NewtonTrustRegionError, ObjectiveEvalError,
    ObjectiveRequest, ObjectiveSample,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingOptimizerKind {
    TrustRegion,
    Arc,
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub finite_diff_step: f64,
    /// Retained for API compatibility.
    /// This setting is only used by finite-difference fallback paths.
    pub fd_hessian_max_dim: usize,
    pub optimizer_kind: SmoothingOptimizerKind,
    pub seed_config: SeedConfig,
}

impl Default for SmoothingBfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            finite_diff_step: 1e-3,
            fd_hessian_max_dim: usize::MAX,
            optimizer_kind: SmoothingOptimizerKind::TrustRegion,
            seed_config: SeedConfig {
                risk_profile: SeedRiskProfile::GeneralizedLinear,
                ..SeedConfig::default()
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsResult {
    pub rho: Array1<f64>,
    pub final_value: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub stationary: bool,
}

fn finite_diff_gradient_external<F>(
    rho: &Array1<f64>,
    step: f64,
    objective: &mut F,
) -> Result<Array1<f64>, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    // Central-difference gradient in rho-space:
    //   g_k ≈ [V(rho + h e_k) - V(rho - h e_k)] / (2h).
    //
    // This is intentionally objective-level (black-box) differentiation, so the returned
    // gradient is exactly consistent with whatever nonlinearities the objective currently
    // includes (Laplace terms, truncation conventions, ridge policies, survival constraints,
    // etc.). That consistency is often more robust than brittle closed-form expressions.
    let mut grad = Array1::<f64>::zeros(rho.len());
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    for i in 0..rho.len() {
        rp[i] += step;
        let fp = objective(&rp)?;
        rm[i] -= step;
        let fm = objective(&rm)?;
        grad[i] = (fp - fm) / (2.0 * step);
        rp[i] = rho[i];
        rm[i] = rho[i];
    }
    Ok(grad)
}

fn finite_diff_gradient_external_parallel<F>(
    rho: &Array1<f64>,
    step: f64,
    objective: &F,
) -> Result<Array1<f64>, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError> + Sync,
{
    let grad_vals: Result<Vec<f64>, EstimationError> = (0..rho.len())
        .into_par_iter()
        .map(|i| {
            let mut rp = rho.clone();
            rp[i] += step;
            let mut rm = rho.clone();
            rm[i] -= step;
            let (fp_res, fm_res) = rayon::join(|| objective(&rp), || objective(&rm));
            let fp = fp_res?;
            let fm = fm_res?;
            Ok((fp - fm) / (2.0 * step))
        })
        .collect();
    Ok(Array1::from_vec(grad_vals?))
}

fn approx_same_rho_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-12 {
            return false;
        }
    }
    true
}

fn should_replace_smoothing_candidate(
    best: &Option<SmoothingBfgsResult>,
    candidate: &SmoothingBfgsResult,
) -> bool {
    match best {
        None => true,
        Some(current) => {
            if candidate.stationary != current.stationary {
                candidate.stationary
            } else if candidate.stationary {
                candidate.final_value < current.final_value
            } else {
                candidate.final_grad_norm < current.final_grad_norm
            }
        }
    }
}

#[inline]
fn should_parallelize_smoothing_candidates(
    num_penalties: usize,
    options: &SmoothingBfgsOptions,
) -> bool {
    if rayon::current_num_threads() <= 1 {
        return false;
    }
    let screening_budget = options.seed_config.screening_budget.max(1);
    let max_seeds = options.seed_config.max_seeds.max(screening_budget);
    let workload = num_penalties
        .saturating_mul(screening_budget)
        .saturating_mul(max_seeds);
    workload >= 64 || num_penalties >= 8 || screening_budget >= 4
}

fn screened_seeds<C, Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<Vec<(usize, Array1<f64>)>, EstimationError>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    let seeds = generate_rho_candidates(num_penalties, heuristic_lambdas, &options.seed_config);
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(
            "no smoothing seeds produced".to_string(),
        ));
    }
    let candidate_seeds: Vec<(usize, Array1<f64>)> = seeds.into_iter().enumerate().collect();

    // Screen seeds: evaluate cost at each seed point, sort by cost, and only
    // run full trust-region solves on the best screening_budget candidates.
    let screening_budget = options.seed_config.screening_budget.max(1);
    let screened_seeds = if candidate_seeds.len() > screening_budget {
        let mut scored: Vec<(usize, Array1<f64>, f64)> = candidate_seeds
            .into_iter()
            .map(|(idx, rho)| {
                let cost = match eval_cost_grad_rho(context, &rho) {
                    Ok((c, _)) => c,
                    Err(_) => f64::INFINITY,
                };
                (idx, rho, cost)
            })
            .collect();
        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(screening_budget);
        scored
            .into_iter()
            .map(|(i, r, _)| (i, r))
            .collect::<Vec<_>>()
    } else {
        candidate_seeds
    };

    Ok(screened_seeds)
}

fn run_single_seed_trust_region<C, Eval>(
    context: &mut C,
    rho_seed: &Array1<f64>,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Option<SmoothingBfgsResult>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>)> = None;
    let mut optimizer = NewtonTrustRegion::new(rho_seed.clone(), |rho, request| {
        if let Some((rho_c, cost_c, grad_c)) = &last_eval
            && approx_same_rho_point(rho, rho_c)
        {
            return Ok(match request {
                ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(*cost_c),
                ObjectiveRequest::CostAndGradient
                | ObjectiveRequest::GradientAndHessian
                | ObjectiveRequest::CostGradientHessian => {
                    ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                }
            });
        }

        let (cost, grad_rho) = match eval_cost_grad_rho(context, rho) {
            Ok((cost, grad_rho)) if cost.is_finite() && grad_rho.iter().all(|v| v.is_finite()) => {
                (cost, grad_rho)
            }
            _ => {
                return Err(ObjectiveEvalError::recoverable(
                    "outer objective returned non-finite cost/gradient",
                ));
            }
        };
        last_eval = Some((rho.clone(), cost, grad_rho.clone()));
        Ok(match request {
            ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(cost),
            ObjectiveRequest::CostAndGradient
            | ObjectiveRequest::GradientAndHessian
            | ObjectiveRequest::CostGradientHessian => {
                ObjectiveSample::cost_and_gradient(cost, grad_rho)
            }
        })
    })
    .with_bounds(lower, upper, 1e-6)
    .with_tolerance(options.tol)
    .with_max_iterations(options.max_iter)
    .with_bfgs_fallback(true)
    .with_fallback_history(12);

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(_) => return None,
    };

    let rho = solution.final_point.clone();
    let mut grad_rho = match &last_eval {
        Some((rho_cached, _cost_cached, grad_cached))
            if approx_same_rho_point(&rho, rho_cached) =>
        {
            grad_cached.clone()
        }
        _ => match eval_cost_grad_rho(context, &rho) {
            Ok((_, grad)) => grad,
            Err(_) => Array1::<f64>::from_elem(rho.len(), f64::NAN),
        },
    };
    for g in grad_rho.iter_mut() {
        if !g.is_finite() {
            *g = f64::NAN;
        }
    }
    let grad_norm = grad_rho.dot(&grad_rho).sqrt();
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        stationary: grad_norm <= options.tol.max(1e-6),
    })
}

fn run_multistart_trust_region<C, Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    let screened_seeds = screened_seeds(
        num_penalties,
        heuristic_lambdas,
        context,
        eval_cost_grad_rho,
        options,
    )?;
    let mut best: Option<SmoothingBfgsResult> = None;
    let near_stationary_tol = (options.tol.max(1e-8)) * 2.0;
    let mut best_grad_norm = f64::INFINITY;
    for (_seed_idx, rho_seed) in screened_seeds.iter() {
        let Some(candidate) =
            run_single_seed_trust_region(context, rho_seed, eval_cost_grad_rho, options)
        else {
            continue;
        };
        let grad_norm = candidate.final_grad_norm;
        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
        best_grad_norm = best_grad_norm.min(grad_norm);
        if best.as_ref().is_some_and(|s| s.stationary) && best_grad_norm <= near_stationary_tol {
            break;
        }
    }

    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "all smoothing outer starts failed before producing a candidate".to_string(),
        )
    })
}

fn run_single_seed_newton<C, Eval>(
    context: &mut C,
    rho_seed: &Array1<f64>,
    eval_cost_grad_hess_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Option<SmoothingBfgsResult>
where
    Eval: FnMut(
        &mut C,
        &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), EstimationError>,
{
    let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>, Option<Array2<f64>>)> = None;
    let mut optimizer = ArcOptimizer::new(rho_seed.clone(), |rho, request| {
        if let Some((rho_c, cost_c, grad_c, hess_c)) = &last_eval
            && approx_same_rho_point(rho, rho_c)
        {
            return Ok(match request {
                ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(*cost_c),
                ObjectiveRequest::CostAndGradient => {
                    ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                }
                ObjectiveRequest::GradientAndHessian | ObjectiveRequest::CostGradientHessian => {
                    if let Some(h) = hess_c {
                        ObjectiveSample::cost_gradient_hessian(*cost_c, grad_c.clone(), h.clone())
                    } else {
                        ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                    }
                }
            });
        }

        let (cost, grad, hess) = eval_cost_grad_hess_rho(context, rho).map_err(|e| {
            ObjectiveEvalError::recoverable(format!("outer objective evaluation failed: {e}"))
        })?;
        if !cost.is_finite() || grad.iter().any(|v| !v.is_finite()) {
            return Err(ObjectiveEvalError::recoverable(
                "outer objective returned non-finite cost/gradient",
            ));
        }
        if let Some(ref h) = hess
            && (h.nrows() != rho.len()
                || h.ncols() != rho.len()
                || h.iter().any(|v| !v.is_finite()))
        {
            return Err(ObjectiveEvalError::recoverable(
                "outer objective returned invalid Hessian",
            ));
        }

        last_eval = Some((rho.clone(), cost, grad.clone(), hess.clone()));
        Ok(match request {
            ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(cost),
            ObjectiveRequest::CostAndGradient => ObjectiveSample::cost_and_gradient(cost, grad),
            ObjectiveRequest::GradientAndHessian | ObjectiveRequest::CostGradientHessian => {
                if let Some(h) = hess {
                    ObjectiveSample::cost_gradient_hessian(cost, grad, h)
                } else {
                    ObjectiveSample::cost_and_gradient(cost, grad)
                }
            }
        })
    })
    .with_bounds(lower, upper, 1e-6)
    .with_tolerance(options.tol)
    .with_max_iterations(options.max_iter)
    .with_bfgs_fallback(true)
    .with_fallback_history(12);

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(ArcError::MaxIterationsReached { last_solution, .. }) => *last_solution,
        Err(_) => return None,
    };

    let rho = solution.final_point.clone();
    let mut grad_rho = match &last_eval {
        Some((rho_cached, _cost_cached, grad_cached, _h_cached))
            if approx_same_rho_point(&rho, rho_cached) =>
        {
            grad_cached.clone()
        }
        _ => match eval_cost_grad_hess_rho(context, &rho) {
            Ok((_, grad, _)) => grad,
            Err(_) => Array1::<f64>::from_elem(rho.len(), f64::NAN),
        },
    };
    for g in grad_rho.iter_mut() {
        if !g.is_finite() {
            *g = f64::NAN;
        }
    }
    let grad_norm = grad_rho.dot(&grad_rho).sqrt();
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        stationary: grad_norm <= options.tol.max(1e-6),
    })
}

fn run_multistart_newton<C, Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    eval_cost_grad_hess_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    Eval: FnMut(
        &mut C,
        &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), EstimationError>,
{
    let mut eval_cost_grad_rho =
        |ctx: &mut C, rho: &Array1<f64>| -> Result<(f64, Array1<f64>), EstimationError> {
            let (cost, grad, _) = eval_cost_grad_hess_rho(ctx, rho)?;
            Ok((cost, grad))
        };
    let screened_seeds = screened_seeds(
        num_penalties,
        heuristic_lambdas,
        context,
        &mut eval_cost_grad_rho,
        options,
    )?;
    let mut best: Option<SmoothingBfgsResult> = None;
    let near_stationary_tol = (options.tol.max(1e-8)) * 2.0;
    let mut best_grad_norm = f64::INFINITY;
    for (_seed_idx, rho_seed) in screened_seeds.iter() {
        let Some(candidate) =
            run_single_seed_newton(context, rho_seed, eval_cost_grad_hess_rho, options)
        else {
            continue;
        };
        let grad_norm = candidate.final_grad_norm;
        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
        best_grad_norm = best_grad_norm.min(grad_norm);
        if best.as_ref().is_some_and(|s| s.stationary) && best_grad_norm <= near_stationary_tol {
            break;
        }
    }

    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "all smoothing outer starts failed before producing a candidate".to_string(),
        )
    })
}

fn screened_seeds_parallel<EvalCost>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    eval_cost_rho: &EvalCost,
    options: &SmoothingBfgsOptions,
) -> Result<Vec<(usize, Array1<f64>)>, EstimationError>
where
    EvalCost: Fn(&Array1<f64>) -> Result<f64, EstimationError> + Sync,
{
    let seeds = generate_rho_candidates(num_penalties, heuristic_lambdas, &options.seed_config);
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(
            "no smoothing seeds produced".to_string(),
        ));
    }
    let candidate_seeds: Vec<(usize, Array1<f64>)> = seeds.into_iter().enumerate().collect();
    let screening_budget = options.seed_config.screening_budget.max(1);
    if candidate_seeds.len() <= screening_budget {
        return Ok(candidate_seeds);
    }
    let mut scored: Vec<(usize, Array1<f64>, f64)> = candidate_seeds
        .into_par_iter()
        .map(|(idx, rho)| {
            let cost = eval_cost_rho(&rho).unwrap_or(f64::INFINITY);
            (idx, rho, cost)
        })
        .collect();
    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(screening_budget);
    Ok(scored.into_iter().map(|(i, r, _)| (i, r)).collect())
}

fn run_single_seed_trust_region_parallel<Eval>(
    rho_seed: &Array1<f64>,
    eval_cost_grad_rho: &Eval,
    options: &SmoothingBfgsOptions,
) -> Option<SmoothingBfgsResult>
where
    Eval: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError> + Sync,
{
    let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>)> = None;
    let mut optimizer = NewtonTrustRegion::new(rho_seed.clone(), |rho, request| {
        if let Some((rho_c, cost_c, grad_c)) = &last_eval
            && approx_same_rho_point(rho, rho_c)
        {
            return Ok(match request {
                ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(*cost_c),
                ObjectiveRequest::CostAndGradient
                | ObjectiveRequest::GradientAndHessian
                | ObjectiveRequest::CostGradientHessian => {
                    ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                }
            });
        }

        let (cost, grad_rho) = match eval_cost_grad_rho(rho) {
            Ok((cost, grad_rho)) if cost.is_finite() && grad_rho.iter().all(|v| v.is_finite()) => {
                (cost, grad_rho)
            }
            _ => {
                return Err(ObjectiveEvalError::recoverable(
                    "outer objective returned non-finite cost/gradient",
                ));
            }
        };
        last_eval = Some((rho.clone(), cost, grad_rho.clone()));
        Ok(match request {
            ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(cost),
            ObjectiveRequest::CostAndGradient
            | ObjectiveRequest::GradientAndHessian
            | ObjectiveRequest::CostGradientHessian => {
                ObjectiveSample::cost_and_gradient(cost, grad_rho)
            }
        })
    })
    .with_bounds(lower, upper, 1e-6)
    .with_tolerance(options.tol)
    .with_max_iterations(options.max_iter)
    .with_bfgs_fallback(true)
    .with_fallback_history(12);

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(_) => return None,
    };

    let rho = solution.final_point.clone();
    let mut grad_rho = match &last_eval {
        Some((rho_cached, _cost_cached, grad_cached))
            if approx_same_rho_point(&rho, rho_cached) =>
        {
            grad_cached.clone()
        }
        _ => match eval_cost_grad_rho(&rho) {
            Ok((_, grad)) => grad,
            Err(_) => Array1::<f64>::from_elem(rho.len(), f64::NAN),
        },
    };
    for g in grad_rho.iter_mut() {
        if !g.is_finite() {
            *g = f64::NAN;
        }
    }
    let grad_norm = grad_rho.dot(&grad_rho).sqrt();
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        stationary: grad_norm <= options.tol.max(1e-6),
    })
}

fn run_multistart_trust_region_parallel<Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    eval_cost_grad_rho: &Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    Eval: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError> + Sync,
{
    let screened_seeds = screened_seeds_parallel(
        num_penalties,
        heuristic_lambdas,
        &|rho: &Array1<f64>| eval_cost_grad_rho(rho).map(|(c, _)| c),
        options,
    )?;
    let mut candidates: Vec<(usize, SmoothingBfgsResult)> = screened_seeds
        .into_par_iter()
        .filter_map(|(seed_idx, rho_seed)| {
            let res =
                run_single_seed_trust_region_parallel(&rho_seed, eval_cost_grad_rho, options)?;
            Some((seed_idx, res))
        })
        .collect();
    if candidates.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(
            "all smoothing outer starts failed before producing a candidate".to_string(),
        ));
    }
    candidates.sort_by_key(|(seed_idx, _)| *seed_idx);
    let mut best: Option<SmoothingBfgsResult> = None;
    for (_, candidate) in candidates {
        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "all smoothing outer starts failed before producing a candidate".to_string(),
        )
    })
}

/// Generic multi-start trust-region smoothing optimizer over log-smoothing parameters (`rho`).
///
/// This is intended for likelihoods whose outer objective is exposed as a scalar
/// function of `rho` (for example survival workflows built on working-model PIRLS).
///
/// Mathematically, this optimizer searches:
///   rho* = argmin_rho V(rho),
/// where `V` is supplied by the caller.
///
/// The gradient seen by the trust-region optimizer is computed by finite differences on `V`:
///   grad_k = dV/drho_k ≈ [V(rho+h e_k)-V(rho-h e_k)]/(2h).
/// This makes the direction field fully consistent with the exact scalar objective,
/// which is particularly useful for complicated non-Gaussian/survival objectives where
/// exact analytic outer derivatives are either expensive or error-prone.
pub fn optimize_log_smoothing_with_multistart<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    mut objective: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: objective(&Array1::<f64>::zeros(0))?,
            iterations: 0,
            final_grad_norm: 0.0,
            stationary: true,
        });
    }

    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| {
        let cost = objective(rho)?;
        let grad_rho = finite_diff_gradient_external(rho, options.finite_diff_step, objective)?;
        Ok((cost, grad_rho))
    };
    match options.optimizer_kind {
        SmoothingOptimizerKind::TrustRegion => run_multistart_trust_region(
            num_penalties,
            heuristic_lambdas,
            &mut objective,
            &mut eval_cost_grad_rho,
            options,
        ),
        SmoothingOptimizerKind::Arc => {
            let mut objective_with_gradient_hessian = |rho: &Array1<f64>| -> Result<
                (f64, Array1<f64>, Option<Array2<f64>>),
                EstimationError,
            > {
                let cost = objective(rho)?;
                let grad_rho =
                    finite_diff_gradient_external(rho, options.finite_diff_step, &mut objective)?;
                Ok((cost, grad_rho, None))
            };
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                &mut objective_with_gradient_hessian,
                &mut |obj, rho| obj(rho),
                options,
            )
        }
    }
}

/// Generic multi-start trust-region smoothing optimizer over log-smoothing parameters (`rho`)
/// when the caller can provide an exact objective gradient in rho-space.
///
/// The callback must return:
/// - `value = V(rho)`
/// - `grad_rho = dV/drho` (same dimension/order as `rho`)
///
/// Internally we optimize directly in `rho` coordinates with no additional
/// reparameterization layer.
///
/// Why this exists:
/// - finite-difference outer gradients require repeated inner solves per coordinate,
/// - exact outer gradients can be injected directly here,
/// - multi-start seed handling and stationarity ranking remain identical to the
///   FD-based optimizer, so behavior is comparable while much faster when exact
///   gradients are available.
///
/// Mathematical contract for callers:
/// - `rho_j = log(lambda_j)` for each smoothing parameter.
/// - The callback returns the scalar outer objective `V(rho)` and its exact
///   gradient `g(rho) = dV/drho`.
/// - the trust-region solver then builds its internal quasi-Newton model from
///   successive `(rho, g)` pairs; it does not need the caller's exact Hessian.
///
/// In particular, when the outer objective is
///   V(rho) = Phi(beta_hat(rho), rho)
///          + 0.5 log|H(rho)|
///          - 0.5 log|S(rho)|_+ ,
/// with `beta_hat(rho)` defined by the inner stationarity equations, the caller
/// should already have used the envelope theorem to eliminate explicit
/// `d beta_hat / d rho` terms from the objective derivative and return the final
/// exact `dV/drho` here.
pub fn optimize_log_smoothing_with_multistart_with_gradient<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    mut objective_with_gradient: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        let (value, grad) = objective_with_gradient(&rho)?;
        let grad_norm = grad.dot(&grad).sqrt();
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: value,
            iterations: 0,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }

    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
    match options.optimizer_kind {
        SmoothingOptimizerKind::TrustRegion => run_multistart_trust_region(
            num_penalties,
            heuristic_lambdas,
            &mut objective_with_gradient,
            &mut eval_cost_grad_rho,
            options,
        ),
        SmoothingOptimizerKind::Arc => {
            let mut eval_cost_grad_hess_rho = |objective: &mut F,
                                               rho: &Array1<f64>|
             -> Result<
                (f64, Array1<f64>, Option<Array2<f64>>),
                EstimationError,
            > {
                let (cost, grad) = objective(rho)?;
                Ok((cost, grad, None))
            };
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                &mut objective_with_gradient,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

/// Parallelized multi-start exact-gradient optimizer.
///
/// This variant runs seed screening and candidate trust-region probes concurrently.
/// It requires a thread-safe objective callback.
pub fn optimize_log_smoothing_with_multistart_with_gradient_parallel<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective_with_gradient: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError> + Sync,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        let (value, grad) = objective_with_gradient(&rho)?;
        let grad_norm = grad.dot(&grad).sqrt();
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: value,
            iterations: 0,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }
    if options.optimizer_kind == SmoothingOptimizerKind::Arc {
        return optimize_log_smoothing_with_multistart_with_gradient(
            num_penalties,
            heuristic_lambdas,
            |rho| objective_with_gradient(rho),
            options,
        );
    }
    if !should_parallelize_smoothing_candidates(num_penalties, options) {
        return optimize_log_smoothing_with_multistart_with_gradient(
            num_penalties,
            heuristic_lambdas,
            |rho| objective_with_gradient(rho),
            options,
        );
    }
    run_multistart_trust_region_parallel(
        num_penalties,
        heuristic_lambdas,
        &objective_with_gradient,
        options,
    )
}

/// Parallelized multi-start finite-difference optimizer.
///
/// This variant runs seed screening and candidate trust-region probes concurrently and
/// computes each coordinate's central-difference gradient in parallel.
pub fn optimize_log_smoothing_with_multistart_parallel_fd<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError> + Sync,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: objective(&Array1::<f64>::zeros(0))?,
            iterations: 0,
            final_grad_norm: 0.0,
            stationary: true,
        });
    }
    if options.optimizer_kind == SmoothingOptimizerKind::Arc {
        return optimize_log_smoothing_with_multistart(
            num_penalties,
            heuristic_lambdas,
            |rho| objective(rho),
            options,
        );
    }
    if !should_parallelize_smoothing_candidates(num_penalties, options) {
        return optimize_log_smoothing_with_multistart(
            num_penalties,
            heuristic_lambdas,
            |rho| objective(rho),
            options,
        );
    }
    run_multistart_trust_region_parallel(
        num_penalties,
        heuristic_lambdas,
        &|rho: &Array1<f64>| {
            let cost = objective(rho)?;
            let grad_rho =
                finite_diff_gradient_external_parallel(rho, options.finite_diff_step, &objective)?;
            Ok((cost, grad_rho))
        },
        options,
    )
}

/// Multi-start smoothing optimizer when the caller can also provide exact Hessians.
///
/// This path uses `wolfe_bfgs::NewtonTrustRegion` and feeds Hessians directly via
/// `ObjectiveSample::cost_gradient_hessian`. If a sample omits a Hessian, the solver
/// falls back to internal BFGS updates for robustness.
pub fn optimize_log_smoothing_with_multistart_with_gradient_and_hessian<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    mut objective_with_gradient_hessian: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), EstimationError>,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        let (value, grad, _) = objective_with_gradient_hessian(&rho)?;
        let grad_norm = grad.dot(&grad).sqrt();
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: value,
            iterations: 0,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }

    match options.optimizer_kind {
        SmoothingOptimizerKind::TrustRegion => {
            let mut eval_cost_grad_rho =
                |objective: &mut F,
                 rho: &Array1<f64>|
                 -> Result<(f64, Array1<f64>), EstimationError> {
                    let (cost, grad, _) = objective(rho)?;
                    Ok((cost, grad))
                };
            run_multistart_trust_region(
                num_penalties,
                heuristic_lambdas,
                &mut objective_with_gradient_hessian,
                &mut eval_cost_grad_rho,
                options,
            )
        }
        SmoothingOptimizerKind::Arc => {
            let mut eval_cost_grad_hess_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                &mut objective_with_gradient_hessian,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convex_value(rho: &Array1<f64>) -> f64 {
        let target = [0.7, -1.1, 0.3];
        let weight = [1.0, 2.0, 0.5];
        rho.iter()
            .enumerate()
            .map(|(i, &r)| 0.5 * weight[i] * (r - target[i]).powi(2))
            .sum()
    }

    #[test]
    fn exact_gradient_parallel_matches_sequential() {
        let mut options = SmoothingBfgsOptions {
            tol: 1e-8,
            max_iter: 120,
            ..SmoothingBfgsOptions::default()
        };
        options.seed_config.screening_budget = 4;
        let heur = [1.0, 1.0, 1.0];

        let seq = optimize_log_smoothing_with_multistart_with_gradient(
            3,
            Some(&heur),
            |rho: &Array1<f64>| {
                let mut g = Array1::<f64>::zeros(rho.len());
                g[0] = rho[0] - 0.7;
                g[1] = 2.0 * (rho[1] + 1.1);
                g[2] = 0.5 * (rho[2] - 0.3);
                Ok((convex_value(rho), g))
            },
            &options,
        )
        .expect("sequential exact-gradient optimization should succeed");

        let par = optimize_log_smoothing_with_multistart_with_gradient_parallel(
            3,
            Some(&heur),
            |rho: &Array1<f64>| {
                let mut g = Array1::<f64>::zeros(rho.len());
                g[0] = rho[0] - 0.7;
                g[1] = 2.0 * (rho[1] + 1.1);
                g[2] = 0.5 * (rho[2] - 0.3);
                Ok((convex_value(rho), g))
            },
            &options,
        )
        .expect("parallel exact-gradient optimization should succeed");

        assert!((seq.final_value - par.final_value).abs() < 1e-9);
        assert_eq!(seq.rho.len(), par.rho.len());
        for i in 0..seq.rho.len() {
            assert!((seq.rho[i] - par.rho[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn finite_difference_parallel_matches_sequential() {
        let mut options = SmoothingBfgsOptions {
            tol: 1e-7,
            max_iter: 120,
            finite_diff_step: 1e-5,
            ..SmoothingBfgsOptions::default()
        };
        options.seed_config.screening_budget = 4;
        let heur = [1.0, 1.0, 1.0];

        let seq = optimize_log_smoothing_with_multistart(
            3,
            Some(&heur),
            |rho: &Array1<f64>| Ok(convex_value(rho)),
            &options,
        )
        .expect("sequential FD optimization should succeed");

        let par = optimize_log_smoothing_with_multistart_parallel_fd(
            3,
            Some(&heur),
            |rho: &Array1<f64>| Ok(convex_value(rho)),
            &options,
        )
        .expect("parallel FD optimization should succeed");

        assert!((seq.final_value - par.final_value).abs() < 1e-8);
        assert_eq!(seq.rho.len(), par.rho.len());
        for i in 0..seq.rho.len() {
            assert!((seq.rho[i] - par.rho[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn optimizer_kind_arc_and_trust_region_agree_on_exact_gradient_problem() {
        let heur = [1.0, 1.0, 1.0];
        let trust = optimize_log_smoothing_with_multistart_with_gradient(
            3,
            Some(&heur),
            |rho: &Array1<f64>| {
                let mut g = Array1::<f64>::zeros(rho.len());
                g[0] = rho[0] - 0.7;
                g[1] = 2.0 * (rho[1] + 1.1);
                g[2] = 0.5 * (rho[2] - 0.3);
                Ok((convex_value(rho), g))
            },
            &SmoothingBfgsOptions {
                tol: 1e-8,
                max_iter: 120,
                optimizer_kind: SmoothingOptimizerKind::TrustRegion,
                ..SmoothingBfgsOptions::default()
            },
        )
        .expect("trust-region exact-gradient optimization should succeed");

        let arc = optimize_log_smoothing_with_multistart_with_gradient(
            3,
            Some(&heur),
            |rho: &Array1<f64>| {
                let mut g = Array1::<f64>::zeros(rho.len());
                g[0] = rho[0] - 0.7;
                g[1] = 2.0 * (rho[1] + 1.1);
                g[2] = 0.5 * (rho[2] - 0.3);
                Ok((convex_value(rho), g))
            },
            &SmoothingBfgsOptions {
                tol: 1e-8,
                max_iter: 120,
                optimizer_kind: SmoothingOptimizerKind::Arc,
                ..SmoothingBfgsOptions::default()
            },
        )
        .expect("arc exact-gradient optimization should succeed");

        assert!((trust.final_value - arc.final_value).abs() < 1e-8);
        for i in 0..trust.rho.len() {
            assert!((trust.rho[i] - arc.rho[i]).abs() < 1e-6);
        }
    }
}
