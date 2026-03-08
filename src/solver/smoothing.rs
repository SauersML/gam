use crate::estimate::{EstimationError, RHO_BOUND};
use crate::seeding::{SeedConfig, SeedRiskProfile, generate_rho_candidates};
use crate::solver::opt_objective::{CachedFirstOrderObjective, CachedSecondOrderObjective};
use ndarray::{Array1, Array2};
use opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, MaxIterations, ObjectiveEvalError,
    Profile, Tolerance,
};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingOptimizerKind {
    Bfgs,
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
            optimizer_kind: SmoothingOptimizerKind::Bfgs,
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
    pub final_stationarity_residual: f64,
    pub final_bound_violation: f64,
    pub stationary: bool,
}

fn smoothing_debug_enabled() -> bool {
    matches!(
        std::env::var("GAM_DEBUG_SMOOTHING").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn no_smoothing_seeds_error(
    solver: &str,
    num_penalties: usize,
    options: &SmoothingBfgsOptions,
) -> EstimationError {
    EstimationError::RemlOptimizationFailed(format!(
        "{solver}: no smoothing seeds were generated for {num_penalties} penalty parameter(s) \
         (seed bounds [{:.1}, {:.1}], max_seeds={}, screening_budget={})",
        options.seed_config.bounds.0,
        options.seed_config.bounds.1,
        options.seed_config.max_seeds,
        options.seed_config.screening_budget.max(1),
    ))
}

fn no_smoothing_candidate_error(
    solver: &str,
    num_penalties: usize,
    screened_seed_count: usize,
) -> EstimationError {
    let start_summary = if screened_seed_count == 1 {
        "the only screened outer-start failed before producing a valid candidate".to_string()
    } else {
        "every screened outer-start failed before producing a valid candidate".to_string()
    };
    let implication = if screened_seed_count == 1 {
        "This run only screened one start, so the failure is narrower than a general multistart search failure.".to_string()
    } else {
        "This means each start either errored during objective evaluation or produced non-finite cost/gradient values.".to_string()
    };
    EstimationError::RemlOptimizationFailed(format!(
        "{solver}: {start_summary} (num_penalties={num_penalties}, \
         screened_starts={screened_seed_count}). {implication}"
    ))
}

fn sanitize_screening_cost(cost: f64) -> f64 {
    if cost.is_finite() {
        cost
    } else {
        f64::INFINITY
    }
}

fn max_bound_violation(rho: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> f64 {
    rho.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|((&x, &lb), &ub)| {
            if x < lb {
                lb - x
            } else if x > ub {
                x - ub
            } else {
                0.0
            }
        })
        .fold(0.0_f64, f64::max)
}

fn projected_stationarity_residual(
    rho: &Array1<f64>,
    grad: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> f64 {
    if rho.len() != grad.len() || rho.len() != lower.len() || rho.len() != upper.len() {
        return f64::NAN;
    }
    let mut sq_norm = 0.0;
    for i in 0..rho.len() {
        let x = rho[i].clamp(lower[i], upper[i]);
        let projected = (x - grad[i]).clamp(lower[i], upper[i]);
        let residual = x - projected;
        sq_norm += residual * residual;
    }
    sq_norm.sqrt()
}

fn unequal_central_difference(f0: f64, fp: f64, fm: f64, h_plus: f64, h_minus: f64) -> f64 {
    (h_minus * h_minus * fp - h_plus * h_plus * fm + (h_plus * h_plus - h_minus * h_minus) * f0)
        / (h_plus * h_minus * (h_plus + h_minus))
}

fn finite_diff_gradient_external<C, EvalCost, Reset>(
    rho: &Array1<f64>,
    step: f64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    context: &mut C,
    reset_context: &mut Reset,
    eval_cost_rho: &mut EvalCost,
    base_value: f64,
) -> Result<Array1<f64>, EstimationError>
where
    EvalCost: FnMut(&mut C, &Array1<f64>) -> Result<f64, EstimationError>,
    Reset: FnMut(&mut C),
{
    let mut grad = Array1::<f64>::zeros(rho.len());
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    for i in 0..rho.len() {
        let h_plus = (upper[i] - rho[i]).clamp(0.0, step);
        let h_minus = (rho[i] - lower[i]).clamp(0.0, step);
        let use_central = h_plus >= 0.5 * step && h_minus >= 0.5 * step;
        if use_central {
            rp[i] = rho[i] + h_plus;
            reset_context(context);
            let fp = eval_cost_rho(context, &rp)?;
            rm[i] = rho[i] - h_minus;
            reset_context(context);
            let fm = eval_cost_rho(context, &rm)?;
            grad[i] = unequal_central_difference(base_value, fp, fm, h_plus, h_minus);
        } else if h_plus >= h_minus && h_plus > 0.0 {
            rp[i] = rho[i] + h_plus;
            reset_context(context);
            let fp = eval_cost_rho(context, &rp)?;
            grad[i] = (fp - base_value) / h_plus;
        } else if h_minus > 0.0 {
            rm[i] = rho[i] - h_minus;
            reset_context(context);
            let fm = eval_cost_rho(context, &rm)?;
            grad[i] = (base_value - fm) / h_minus;
        } else {
            grad[i] = 0.0;
        }
        rp[i] = rho[i];
        rm[i] = rho[i];
    }
    Ok(grad)
}

fn finite_diff_gradient_external_parallel<F>(
    rho: &Array1<f64>,
    step: f64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    objective: &F,
    base_value: f64,
) -> Result<Array1<f64>, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError> + Sync,
{
    let grad_vals: Result<Vec<f64>, EstimationError> = (0..rho.len())
        .into_par_iter()
        .map(|i| {
            let h_plus = (upper[i] - rho[i]).clamp(0.0, step);
            let h_minus = (rho[i] - lower[i]).clamp(0.0, step);
            let use_central = h_plus >= 0.5 * step && h_minus >= 0.5 * step;
            if use_central {
                let mut rp = rho.clone();
                rp[i] = rho[i] + h_plus;
                let mut rm = rho.clone();
                rm[i] = rho[i] - h_minus;
                let (fp_res, fm_res) = rayon::join(|| objective(&rp), || objective(&rm));
                let fp = fp_res?;
                let fm = fm_res?;
                Ok(unequal_central_difference(
                    base_value, fp, fm, h_plus, h_minus,
                ))
            } else if h_plus >= h_minus && h_plus > 0.0 {
                let mut rp = rho.clone();
                rp[i] = rho[i] + h_plus;
                let fp = objective(&rp)?;
                Ok((fp - base_value) / h_plus)
            } else if h_minus > 0.0 {
                let mut rm = rho.clone();
                rm[i] = rho[i] - h_minus;
                let fm = objective(&rm)?;
                Ok((base_value - fm) / h_minus)
            } else {
                Ok(0.0)
            }
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
            if candidate.final_bound_violation != current.final_bound_violation {
                candidate.final_bound_violation < current.final_bound_violation
            } else if candidate.stationary != current.stationary {
                candidate.stationary
            } else if candidate.stationary {
                candidate.final_value < current.final_value
            } else if candidate.final_stationarity_residual != current.final_stationarity_residual {
                candidate.final_stationarity_residual < current.final_stationarity_residual
            } else if candidate.final_value != current.final_value {
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

fn screened_seeds<C, EvalCost, Reset>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    reset_context: &mut Reset,
    eval_cost_rho: &mut EvalCost,
    options: &SmoothingBfgsOptions,
) -> Result<Vec<(usize, Array1<f64>)>, EstimationError>
where
    EvalCost: FnMut(&mut C, &Array1<f64>) -> Result<f64, EstimationError>,
    Reset: FnMut(&mut C),
{
    let seeds = generate_rho_candidates(num_penalties, heuristic_lambdas, &options.seed_config);
    if seeds.is_empty() {
        return Err(no_smoothing_seeds_error(
            "bfgs outer optimizer",
            num_penalties,
            options,
        ));
    }
    let candidate_seeds: Vec<(usize, Array1<f64>)> = seeds.into_iter().enumerate().collect();

    // Screen seeds: evaluate cost at each seed point, sort by cost, and only
    // run full outer solves on the best screening_budget candidates.
    let screening_budget = options.seed_config.screening_budget.max(1);
    let screened_seeds = if candidate_seeds.len() > screening_budget {
        let mut scored: Vec<(usize, Array1<f64>, f64)> = candidate_seeds
            .into_iter()
            .map(|(idx, rho)| {
                reset_context(context);
                let cost = match eval_cost_rho(context, &rho) {
                    Ok(c) => sanitize_screening_cost(c),
                    Err(_) => f64::INFINITY,
                };
                (idx, rho, cost)
            })
            .collect();
        scored.sort_by(|a, b| a.2.total_cmp(&b.2));
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

fn run_single_seed_bfgs<C, Eval, Reset>(
    context: &mut C,
    rho_seed: &Array1<f64>,
    reset_context: &mut Reset,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Option<SmoothingBfgsResult>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
    Reset: FnMut(&mut C),
{
    let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>)> = None;
    let objective = CachedFirstOrderObjective::new(|rho: &Array1<f64>| {
        reset_context(context);
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
        Ok((cost, grad_rho))
    });
    let mut optimizer = Bfgs::new(rho_seed.clone(), objective)
        .with_bounds(
            Bounds::new(lower.clone(), upper.clone(), 1e-6)
                .expect("smoothing rho bounds must be valid"),
        )
        .with_tolerance(Tolerance::new(options.tol).expect("smoothing tolerance must be valid"))
        .with_profile(Profile::Robust)
        .with_max_iterations(
            MaxIterations::new(options.max_iter).expect("smoothing max_iter must be valid"),
        );

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
        Err(err) => {
            if smoothing_debug_enabled() {
                eprintln!(
                    "[smoothing-debug] bfgs seed failed: rho_seed={:?} err={err:?}",
                    rho_seed.to_vec()
                );
            }
            return None;
        }
    };

    let rho = solution.final_point.clone();
    let mut grad_rho = match &last_eval {
        Some((rho_cached, _cost_cached, grad_cached))
            if approx_same_rho_point(&rho, rho_cached) =>
        {
            grad_cached.clone()
        }
        _ => {
            reset_context(context);
            match eval_cost_grad_rho(context, &rho) {
                Ok((_, grad)) => grad,
                Err(_) => Array1::<f64>::from_elem(rho.len(), f64::NAN),
            }
        }
    };
    for g in grad_rho.iter_mut() {
        if !g.is_finite() {
            *g = f64::NAN;
        }
    }
    let grad_norm = grad_rho.dot(&grad_rho).sqrt();
    let bound_violation = max_bound_violation(&rho, &lower, &upper);
    let stationarity_residual = projected_stationarity_residual(&rho, &grad_rho, &lower, &upper);
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        final_stationarity_residual: stationarity_residual,
        final_bound_violation: bound_violation,
        stationary: bound_violation <= 1e-6 && stationarity_residual <= options.tol.max(1e-6),
    })
}

fn run_multistart_bfgs<C, EvalCost, EvalGrad, Reset>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    reset_context: &mut Reset,
    eval_cost_rho: &mut EvalCost,
    eval_cost_grad_rho: &mut EvalGrad,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    EvalCost: FnMut(&mut C, &Array1<f64>) -> Result<f64, EstimationError>,
    EvalGrad: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
    Reset: FnMut(&mut C),
{
    let screened_seeds = screened_seeds(
        num_penalties,
        heuristic_lambdas,
        context,
        reset_context,
        eval_cost_rho,
        options,
    )?;
    let mut best: Option<SmoothingBfgsResult> = None;
    let near_stationary_tol = (options.tol.max(1e-8)) * 2.0;
    let mut best_stationarity_residual = f64::INFINITY;
    for (_seed_idx, rho_seed) in screened_seeds.iter() {
        let Some(candidate) = run_single_seed_bfgs(
            context,
            rho_seed,
            reset_context,
            eval_cost_grad_rho,
            options,
        ) else {
            continue;
        };
        let stationarity_residual = candidate.final_stationarity_residual;
        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
        best_stationarity_residual = best_stationarity_residual.min(stationarity_residual);
        if best.as_ref().is_some_and(|s| s.stationary)
            && best_stationarity_residual <= near_stationary_tol
        {
            break;
        }
    }

    best.ok_or_else(|| {
        no_smoothing_candidate_error("bfgs outer optimizer", num_penalties, screened_seeds.len())
    })
}

fn run_single_seed_newton<C, Eval, Reset>(
    context: &mut C,
    rho_seed: &Array1<f64>,
    reset_context: &mut Reset,
    eval_cost_grad_hess_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Option<SmoothingBfgsResult>
where
    Eval: FnMut(
        &mut C,
        &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), EstimationError>,
    Reset: FnMut(&mut C),
{
    let lower = Array1::<f64>::from_elem(rho_seed.len(), -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(rho_seed.len(), RHO_BOUND);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>, Option<Array2<f64>>)> = None;
    let objective = CachedSecondOrderObjective::new(
        |rho: &Array1<f64>| {
            reset_context(context);
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
            Ok((cost, grad, hess))
        },
        options.finite_diff_step,
    );
    let mut optimizer = ArcOptimizer::new(rho_seed.clone(), objective)
        .with_bounds(
            Bounds::new(lower.clone(), upper.clone(), 1e-6)
                .expect("smoothing rho bounds must be valid"),
        )
        .with_tolerance(Tolerance::new(options.tol).expect("smoothing tolerance must be valid"))
        .with_max_iterations(
            MaxIterations::new(options.max_iter).expect("smoothing max_iter must be valid"),
        );

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(ArcError::MaxIterationsReached { last_solution, .. }) => *last_solution,
        Err(err) => {
            if smoothing_debug_enabled() {
                eprintln!(
                    "[smoothing-debug] ARC seed failed: rho_seed={:?} err={err:?}",
                    rho_seed.to_vec()
                );
            }
            return None;
        }
    };

    let rho = solution.final_point.clone();
    let mut grad_rho = match &last_eval {
        Some((rho_cached, _cost_cached, grad_cached, _h_cached))
            if approx_same_rho_point(&rho, rho_cached) =>
        {
            grad_cached.clone()
        }
        _ => {
            reset_context(context);
            match eval_cost_grad_hess_rho(context, &rho) {
                Ok((_, grad, _)) => grad,
                Err(_) => Array1::<f64>::from_elem(rho.len(), f64::NAN),
            }
        }
    };
    for g in grad_rho.iter_mut() {
        if !g.is_finite() {
            *g = f64::NAN;
        }
    }
    let grad_norm = grad_rho.dot(&grad_rho).sqrt();
    let bound_violation = max_bound_violation(&rho, &lower, &upper);
    let stationarity_residual = projected_stationarity_residual(&rho, &grad_rho, &lower, &upper);
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        final_stationarity_residual: stationarity_residual,
        final_bound_violation: bound_violation,
        stationary: bound_violation <= 1e-6 && stationarity_residual <= options.tol.max(1e-6),
    })
}

fn run_multistart_newton<C, EvalCost, Eval, Reset>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    reset_context: &mut Reset,
    eval_cost_rho: &mut EvalCost,
    eval_cost_grad_hess_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    EvalCost: FnMut(&mut C, &Array1<f64>) -> Result<f64, EstimationError>,
    Eval: FnMut(
        &mut C,
        &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), EstimationError>,
    Reset: FnMut(&mut C),
{
    let screened_seeds = screened_seeds(
        num_penalties,
        heuristic_lambdas,
        context,
        reset_context,
        eval_cost_rho,
        options,
    )?;
    let mut best: Option<SmoothingBfgsResult> = None;
    let near_stationary_tol = (options.tol.max(1e-8)) * 2.0;
    let mut best_stationarity_residual = f64::INFINITY;
    for (_seed_idx, rho_seed) in screened_seeds.iter() {
        let Some(candidate) = run_single_seed_newton(
            context,
            rho_seed,
            reset_context,
            eval_cost_grad_hess_rho,
            options,
        ) else {
            continue;
        };
        let stationarity_residual = candidate.final_stationarity_residual;
        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
        best_stationarity_residual = best_stationarity_residual.min(stationarity_residual);
        if best.as_ref().is_some_and(|s| s.stationary)
            && best_stationarity_residual <= near_stationary_tol
        {
            break;
        }
    }

    best.ok_or_else(|| {
        no_smoothing_candidate_error("ARC outer optimizer", num_penalties, screened_seeds.len())
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
        return Err(no_smoothing_seeds_error(
            "parallel bfgs outer optimizer",
            num_penalties,
            options,
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
            let cost = eval_cost_rho(&rho)
                .map(sanitize_screening_cost)
                .unwrap_or(f64::INFINITY);
            (idx, rho, cost)
        })
        .collect();
    scored.sort_by(|a, b| a.2.total_cmp(&b.2));
    scored.truncate(screening_budget);
    Ok(scored.into_iter().map(|(i, r, _)| (i, r)).collect())
}

fn run_single_seed_bfgs_parallel<Eval>(
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
    let objective = CachedFirstOrderObjective::new(|rho: &Array1<f64>| {
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
        Ok((cost, grad_rho))
    });
    let mut optimizer = Bfgs::new(rho_seed.clone(), objective)
        .with_bounds(
            Bounds::new(lower.clone(), upper.clone(), 1e-6)
                .expect("smoothing rho bounds must be valid"),
        )
        .with_tolerance(Tolerance::new(options.tol).expect("smoothing tolerance must be valid"))
        .with_profile(Profile::Robust)
        .with_max_iterations(
            MaxIterations::new(options.max_iter).expect("smoothing max_iter must be valid"),
        );

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
        Err(err) => {
            if smoothing_debug_enabled() {
                eprintln!(
                    "[smoothing-debug] parallel bfgs seed failed: rho_seed={:?} err={err:?}",
                    rho_seed.to_vec()
                );
            }
            return None;
        }
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
    let bound_violation = max_bound_violation(&rho, &lower, &upper);
    let stationarity_residual = projected_stationarity_residual(&rho, &grad_rho, &lower, &upper);
    Some(SmoothingBfgsResult {
        rho,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: grad_norm,
        final_stationarity_residual: stationarity_residual,
        final_bound_violation: bound_violation,
        stationary: bound_violation <= 1e-6 && stationarity_residual <= options.tol.max(1e-6),
    })
}

fn run_multistart_bfgs_parallel<Eval>(
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
    let screened_seed_count = screened_seeds.len();
    let mut candidates: Vec<(usize, SmoothingBfgsResult)> = screened_seeds
        .into_par_iter()
        .filter_map(|(seed_idx, rho_seed)| {
            let res = run_single_seed_bfgs_parallel(&rho_seed, eval_cost_grad_rho, options)?;
            Some((seed_idx, res))
        })
        .collect();
    if candidates.is_empty() {
        return Err(no_smoothing_candidate_error(
            "parallel bfgs outer optimizer",
            num_penalties,
            screened_seed_count,
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
        no_smoothing_candidate_error(
            "parallel bfgs outer optimizer",
            num_penalties,
            screened_seed_count,
        )
    })
}

/// Generic multi-start smoothing optimizer over log-smoothing parameters (`rho`).
///
/// This is intended for likelihoods whose outer objective is exposed as a scalar
/// function of `rho` (for example survival workflows built on working-model PIRLS).
///
/// Mathematically, this optimizer searches:
///   rho* = argmin_rho V(rho),
/// where `V` is supplied by the caller.
///
/// The gradient seen by the optimizer is computed by finite differences on `V`:
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
            final_stationarity_residual: 0.0,
            final_bound_violation: 0.0,
            stationary: true,
        });
    }

    let lower = Array1::<f64>::from_elem(num_penalties, -RHO_BOUND);
    let upper = Array1::<f64>::from_elem(num_penalties, RHO_BOUND);
    let mut reset_objective = |_objective: &mut F| {};
    let mut eval_cost_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| {
        let cost = objective(rho)?;
        let mut reset_fd_context = |_objective: &mut F| {};
        let mut eval_cost_fd = |objective: &mut F, rho: &Array1<f64>| objective(rho);
        let grad_rho = finite_diff_gradient_external(
            rho,
            options.finite_diff_step,
            &lower,
            &upper,
            objective,
            &mut reset_fd_context,
            &mut eval_cost_fd,
            cost,
        )?;
        Ok((cost, grad_rho))
    };
    match options.optimizer_kind {
        SmoothingOptimizerKind::Bfgs => run_multistart_bfgs(
            num_penalties,
            heuristic_lambdas,
            &mut objective,
            &mut reset_objective,
            &mut eval_cost_rho,
            &mut eval_cost_grad_rho,
            options,
        ),
        SmoothingOptimizerKind::Arc => {
            let mut eval_cost_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
            let mut eval_cost_grad_hess_rho = |objective: &mut F,
                                               rho: &Array1<f64>|
             -> Result<
                (f64, Array1<f64>, Option<Array2<f64>>),
                EstimationError,
            > {
                let cost = objective(rho)?;
                let mut reset_fd_context = |_objective: &mut F| {};
                let mut eval_cost_fd = |objective: &mut F, rho: &Array1<f64>| objective(rho);
                let grad_rho = finite_diff_gradient_external(
                    rho,
                    options.finite_diff_step,
                    &lower,
                    &upper,
                    objective,
                    &mut reset_fd_context,
                    &mut eval_cost_fd,
                    cost,
                )?;
                Ok((cost, grad_rho, None))
            };
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                &mut objective,
                &mut reset_objective,
                &mut eval_cost_rho,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

/// Generic multi-start smoothing optimizer over log-smoothing parameters (`rho`)
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
/// - the BFGS solver then builds its internal quasi-Newton model from
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
            final_stationarity_residual: grad_norm,
            final_bound_violation: 0.0,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }

    let mut reset_objective = |_objective: &mut F| {};
    let mut eval_cost_rho =
        |objective: &mut F, rho: &Array1<f64>| objective(rho).map(|(cost, _)| cost);
    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
    match options.optimizer_kind {
        SmoothingOptimizerKind::Bfgs => run_multistart_bfgs(
            num_penalties,
            heuristic_lambdas,
            &mut objective_with_gradient,
            &mut reset_objective,
            &mut eval_cost_rho,
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
                &mut reset_objective,
                &mut eval_cost_rho,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

/// Parallelized multi-start exact-gradient optimizer.
///
/// This variant runs seed screening and candidate BFGS probes concurrently.
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
            final_stationarity_residual: grad_norm,
            final_bound_violation: 0.0,
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
    run_multistart_bfgs_parallel(
        num_penalties,
        heuristic_lambdas,
        &objective_with_gradient,
        options,
    )
}

/// Parallelized multi-start finite-difference optimizer.
///
/// This variant runs seed screening and candidate outer probes concurrently and
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
            final_stationarity_residual: 0.0,
            final_bound_violation: 0.0,
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
    run_multistart_bfgs_parallel(
        num_penalties,
        heuristic_lambdas,
        &|rho: &Array1<f64>| {
            let cost = objective(rho)?;
            let lower = Array1::<f64>::from_elem(num_penalties, -RHO_BOUND);
            let upper = Array1::<f64>::from_elem(num_penalties, RHO_BOUND);
            let grad_rho = finite_diff_gradient_external_parallel(
                rho,
                options.finite_diff_step,
                &lower,
                &upper,
                &objective,
                cost,
            )?;
            Ok((cost, grad_rho))
        },
        options,
    )
}

/// Multi-start smoothing optimizer when the caller can also provide exact Hessians.
///
/// This path uses `opt::NewtonTrustRegion` / `opt::Arc` through the trait-based
/// second-order objective API.
///
/// When the caller does not provide an analytic Hessian, the shared objective
/// adapter builds a symmetric finite-difference Hessian from exact gradients.
/// That keeps the outer solver on the same `opt` path instead of maintaining a
/// duplicate fallback optimizer API locally.
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
            final_stationarity_residual: grad_norm,
            final_bound_violation: 0.0,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }

    match options.optimizer_kind {
        SmoothingOptimizerKind::Bfgs => optimize_log_smoothing_with_multistart_with_gradient(
            num_penalties,
            heuristic_lambdas,
            |rho| {
                let (cost, grad, _) = objective_with_gradient_hessian(rho)?;
                Ok((cost, grad))
            },
            options,
        ),
        SmoothingOptimizerKind::Arc => {
            let mut reset_objective = |_objective: &mut F| {};
            let mut eval_cost_rho =
                |objective: &mut F, rho: &Array1<f64>| objective(rho).map(|(cost, _, _)| cost);
            let mut eval_cost_grad_hess_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                &mut objective_with_gradient_hessian,
                &mut reset_objective,
                &mut eval_cost_rho,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

pub(crate) fn optimize_log_smoothing_with_multistart_with_gradient_state<
    C,
    EvalCost,
    EvalGrad,
    Reset,
>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    mut reset_context: Reset,
    mut eval_cost_rho: EvalCost,
    mut eval_cost_grad_rho: EvalGrad,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    EvalCost: FnMut(&mut C, &Array1<f64>) -> Result<f64, EstimationError>,
    EvalGrad: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
    Reset: FnMut(&mut C),
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        reset_context(context);
        let value = eval_cost_rho(context, &rho)?;
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: value,
            iterations: 0,
            final_grad_norm: 0.0,
            final_stationarity_residual: 0.0,
            final_bound_violation: 0.0,
            stationary: true,
        });
    }
    match options.optimizer_kind {
        SmoothingOptimizerKind::Bfgs => run_multistart_bfgs(
            num_penalties,
            heuristic_lambdas,
            context,
            &mut reset_context,
            &mut eval_cost_rho,
            &mut eval_cost_grad_rho,
            options,
        ),
        SmoothingOptimizerKind::Arc => {
            let mut eval_cost_grad_hess_rho = |ctx: &mut C,
                                               rho: &Array1<f64>|
             -> Result<
                (f64, Array1<f64>, Option<Array2<f64>>),
                EstimationError,
            > {
                let (cost, grad) = eval_cost_grad_rho(ctx, rho)?;
                Ok((cost, grad, None))
            };
            run_multistart_newton(
                num_penalties,
                heuristic_lambdas,
                context,
                &mut reset_context,
                &mut eval_cost_rho,
                &mut eval_cost_grad_hess_rho,
                options,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn convex_value(rho: &Array1<f64>) -> f64 {
        let target = [0.7, -1.1, 0.3];
        let weight = [1.0, 2.0, 0.5];
        rho.iter()
            .enumerate()
            .map(|(i, &r)| 0.5 * weight[i] * (r - target[i]).powi(2))
            .sum()
    }

    #[test]
    fn projected_stationarity_accepts_bound_optimum_with_nonzero_raw_gradient() {
        let options = SmoothingBfgsOptions {
            tol: 1e-8,
            max_iter: 80,
            ..SmoothingBfgsOptions::default()
        };
        let res = optimize_log_smoothing_with_multistart_with_gradient(
            1,
            Some(&[1.0]),
            |rho: &Array1<f64>| {
                let target = RHO_BOUND + 2.0;
                let delta = rho[0] - target;
                Ok((0.5 * delta * delta, array![delta]))
            },
            &options,
        )
        .expect("bound-constrained exact-gradient optimization should succeed");
        assert!((res.rho[0] - RHO_BOUND).abs() <= 1e-6);
        assert!(res.stationary);
        assert!(res.final_stationarity_residual <= options.tol.max(1e-6));
        assert!(res.final_grad_norm > 1.0);
    }

    #[test]
    fn bound_aware_fd_gradient_never_steps_outside_rho_bounds() {
        let rho = array![RHO_BOUND];
        let lower = array![-RHO_BOUND];
        let upper = array![RHO_BOUND];
        let mut objective = |x: &Array1<f64>| -> Result<f64, EstimationError> {
            if x[0] < -RHO_BOUND - 1e-12 || x[0] > RHO_BOUND + 1e-12 {
                return Err(EstimationError::InvalidInput(
                    "finite-difference step left rho bounds".to_string(),
                ));
            }
            Ok(-x[0])
        };
        let mut reset = |_objective: &mut _| {};
        let mut eval_cost = |objective: &mut _, x: &Array1<f64>| objective(x);
        let grad = finite_diff_gradient_external(
            &rho,
            1e-3,
            &lower,
            &upper,
            &mut objective,
            &mut reset,
            &mut eval_cost,
            -RHO_BOUND,
        )
        .expect("bound-aware finite differences should stay feasible");
        assert!((grad[0] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn screening_drops_nan_costs() {
        let mut options = SmoothingBfgsOptions::default();
        options.seed_config.max_seeds = 4;
        options.seed_config.screening_budget = 1;
        let mut unit = ();
        let screened = screened_seeds(
            1,
            Some(&[1.0]),
            &mut unit,
            &mut |_ctx: &mut ()| {},
            &mut |_ctx: &mut (), rho: &Array1<f64>| {
                if rho.iter().all(|v| v.abs() < 1e-12) {
                    Ok(f64::NAN)
                } else {
                    Ok(rho[0].abs())
                }
            },
            &options,
        )
        .expect("screening should succeed");
        assert_eq!(screened.len(), 1);
        assert!(screened[0].1.iter().any(|v| v.abs() > 1e-12));
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
    fn optimizer_kind_arc_and_bfgs_agree_on_exact_gradient_problem() {
        let heur = [1.0, 1.0, 1.0];
        let bfgs = optimize_log_smoothing_with_multistart_with_gradient(
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
                optimizer_kind: SmoothingOptimizerKind::Bfgs,
                ..SmoothingBfgsOptions::default()
            },
        )
        .expect("bfgs exact-gradient optimization should succeed");

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

        assert!((bfgs.final_value - arc.final_value).abs() < 1e-8);
        for i in 0..bfgs.rho.len() {
            assert!((bfgs.rho[i] - arc.rho[i]).abs() < 1e-6);
        }
    }
}
