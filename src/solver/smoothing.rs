use crate::estimate::{EstimationError, RHO_BOUND};
use crate::seeding::{SeedConfig, SeedRiskProfile, generate_rho_candidates};
use ndarray::{Array1, Array2};
use wolfe_bfgs::{NewtonTrustRegion, ObjectiveEvalError, ObjectiveRequest, ObjectiveSample};

#[derive(Clone, Debug)]
pub struct SmoothingBfgsOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub finite_diff_step: f64,
    pub seed_config: SeedConfig,
}

impl Default for SmoothingBfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            finite_diff_step: 1e-3,
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

fn finite_diff_hessian_from_gradient_external<C, Eval>(
    context: &mut C,
    rho: &Array1<f64>,
    step: f64,
    eval_cost_grad_rho: &mut Eval,
) -> Result<Array2<f64>, EstimationError>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    let k = rho.len();
    let mut h = Array2::<f64>::zeros((k, k));
    if k == 0 {
        return Ok(h);
    }
    for j in 0..k {
        let hj = (step * (1.0 + rho[j].abs())).max(1e-6);
        let mut rho_p = rho.clone();
        rho_p[j] += hj;
        let mut rho_m = rho.clone();
        rho_m[j] -= hj;
        let g_p = eval_cost_grad_rho(context, &rho_p)?.1;
        let g_m = eval_cost_grad_rho(context, &rho_m)?.1;
        if g_p.len() != k || g_m.len() != k {
            return Err(EstimationError::RemlOptimizationFailed(
                "outer FD Hessian gradient length mismatch".to_string(),
            ));
        }
        for i in 0..k {
            h[[i, j]] = (g_p[i] - g_m[i]) / (2.0 * hj);
        }
    }
    for i in 0..k {
        for j in 0..i {
            let v = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = v;
            h[[j, i]] = v;
        }
    }
    if h.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::RemlOptimizationFailed(
            "outer FD Hessian produced non-finite values".to_string(),
        ));
    }
    Ok(h)
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

fn run_multistart_bfgs<C, Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
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
    // run full BFGS on the best screening_budget candidates.
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

    let mut best: Option<SmoothingBfgsResult> = None;
    let near_stationary_tol = (options.tol.max(1e-8)) * 2.0;
    let mut best_grad_norm = f64::INFINITY;
    for (_seed_idx, rho_seed) in screened_seeds.iter() {
        let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
        let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
        let mut last_eval: Option<(Array1<f64>, Array1<f64>)> = None;
        let mut full_cache: Option<(Array1<f64>, f64, Array1<f64>, Array2<f64>)> = None;
        let mut cg_cache: Option<(Array1<f64>, f64, Array1<f64>)> = None;
        let mut optimizer = NewtonTrustRegion::new(rho_seed.clone(), |rho, request| {
            if let Some((rho_c, cost_c, grad_c, hess_c)) = &full_cache
                && approx_same_rho_point(rho, rho_c)
            {
                return Ok(match request {
                    ObjectiveRequest::CostOnly => ObjectiveSample::cost_only(*cost_c),
                    ObjectiveRequest::CostAndGradient => {
                        ObjectiveSample::cost_and_gradient(*cost_c, grad_c.clone())
                    }
                    ObjectiveRequest::GradientAndHessian
                    | ObjectiveRequest::CostGradientHessian => {
                        ObjectiveSample::cost_gradient_hessian(
                            *cost_c,
                            grad_c.clone(),
                            hess_c.clone(),
                        )
                    }
                });
            }

            let eval_cost_grad = |rho: &Array1<f64>,
                                  context: &mut C,
                                  eval_cost_grad_rho: &mut Eval|
             -> Result<(f64, Array1<f64>), ObjectiveEvalError> {
                let (cost, grad_rho) = eval_cost_grad_rho(context, rho)
                    .map_err(|e| ObjectiveEvalError::recoverable(format!("{e}")))?;
                if !cost.is_finite() || grad_rho.iter().any(|v| !v.is_finite()) {
                    return Err(ObjectiveEvalError::recoverable(
                        "non-finite smoothing objective/gradient",
                    ));
                }
                Ok((cost, grad_rho))
            };
            match request {
                ObjectiveRequest::CostOnly => {
                    let (cost, grad_rho) = eval_cost_grad(rho, context, eval_cost_grad_rho)?;
                    last_eval = Some((rho.clone(), grad_rho.clone()));
                    cg_cache = Some((rho.clone(), cost, grad_rho));
                    Ok(ObjectiveSample::cost_only(cost))
                }
                ObjectiveRequest::CostAndGradient => {
                    let (cost, grad_rho) = if let Some((rho_c, cost_c, grad_c)) = &cg_cache {
                        if approx_same_rho_point(rho, rho_c) {
                            (*cost_c, grad_c.clone())
                        } else {
                            eval_cost_grad(rho, context, eval_cost_grad_rho)?
                        }
                    } else {
                        eval_cost_grad(rho, context, eval_cost_grad_rho)?
                    };
                    last_eval = Some((rho.clone(), grad_rho.clone()));
                    cg_cache = Some((rho.clone(), cost, grad_rho.clone()));
                    Ok(ObjectiveSample::cost_and_gradient(cost, grad_rho))
                }
                ObjectiveRequest::GradientAndHessian | ObjectiveRequest::CostGradientHessian => {
                    let (cost, grad_rho) = if let Some((rho_c, cost_c, grad_c)) = &cg_cache {
                        if approx_same_rho_point(rho, rho_c) {
                            (*cost_c, grad_c.clone())
                        } else {
                            eval_cost_grad(rho, context, eval_cost_grad_rho)?
                        }
                    } else {
                        eval_cost_grad(rho, context, eval_cost_grad_rho)?
                    };
                    last_eval = Some((rho.clone(), grad_rho.clone()));
                    cg_cache = Some((rho.clone(), cost, grad_rho.clone()));
                    let hess_rho = match finite_diff_hessian_from_gradient_external(
                        context,
                        rho,
                        options.finite_diff_step,
                        eval_cost_grad_rho,
                    ) {
                        Ok(h) => h,
                        Err(e) => return Err(ObjectiveEvalError::recoverable(format!("{e}"))),
                    };
                    full_cache = Some((rho.clone(), cost, grad_rho.clone(), hess_rho.clone()));
                    Ok(ObjectiveSample::cost_gradient_hessian(
                        cost, grad_rho, hess_rho,
                    ))
                }
            }
        })
        .with_bounds(lower, upper, 1e-6)
        .with_tolerance(options.tol)
        .with_max_iterations(options.max_iter)
        .with_initial_trust_radius(1.0)
        .with_max_trust_radius(1e6)
        .with_acceptance_threshold(0.1);

        let solution = match optimizer.run() {
            Ok(sol) => sol,
            Err(wolfe_bfgs::NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
                *last_solution
            }
            Err(_) => continue,
        };

        let rho = solution.final_point.clone();
        let mut grad_rho = match &last_eval {
            Some((rho_cached, grad_cached)) if approx_same_rho_point(&rho, rho_cached) => {
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
        let candidate = SmoothingBfgsResult {
            rho,
            final_value: solution.final_value,
            iterations: solution.iterations,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        };

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

/// Generic multi-start BFGS smoothing optimizer over log-smoothing parameters (`rho`).
///
/// This is intended for likelihoods whose outer objective is exposed as a scalar
/// function of `rho` (for example survival workflows built on working-model PIRLS).
///
/// Mathematically, this optimizer searches:
///   rho* = argmin_rho V(rho),
/// where `V` is supplied by the caller.
///
/// The gradient seen by BFGS is always computed by finite differences on `V`:
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
    run_multistart_bfgs(
        num_penalties,
        heuristic_lambdas,
        &mut objective,
        &mut eval_cost_grad_rho,
        options,
    )
}

/// Generic multi-start BFGS smoothing optimizer over log-smoothing parameters (`rho`)
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
    run_multistart_bfgs(
        num_penalties,
        heuristic_lambdas,
        &mut objective_with_gradient,
        &mut eval_cost_grad_rho,
        options,
    )
}
