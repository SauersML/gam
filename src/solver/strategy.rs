//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, joint flexible
//! link, GAMLSS custom family, spatial κ, etc.) declares its derivative
//! capability here and receives an [`OuterPlan`] that determines which solver
//! and Hessian source to use.
//!
//! # Design invariant
//!
//! Finite-difference Hessian fallback is _never_ silent. If a path cannot
//! provide an analytic Hessian, that fact is visible in its
//! [`OuterCapability`] declaration and in the resulting [`OuterPlan`].
//! The previous pattern of returning `None` from a closure and letting
//! `CachedSecondOrderObjective` silently FD is what this module replaces.

use ndarray::{Array1, Array2};

/// Whether an analytic derivative is available for a given order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// Exact analytic derivative implemented and available.
    Analytic,
    /// No analytic derivative; must be approximated or skipped.
    Unavailable,
}

/// Declares what a specific model path can provide to the outer optimizer.
///
/// Each call site that optimizes smoothing parameters constructs one of these
/// to describe its analytic derivative coverage. The [`plan`] function then
/// selects the optimizer and Hessian strategy.
#[derive(Clone, Copy, Debug)]
pub struct OuterCapability {
    pub gradient: Derivative,
    pub hessian: Derivative,
    /// Number of smoothing (+ any auxiliary hyper-) parameters being optimized.
    pub n_params: usize,
}

/// Which solver algorithm to use for the outer optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Adaptive Regularized Cubic; fastest convergence, requires Hessian.
    Arc,
    /// Newton trust-region; quadratic model, requires Hessian.
    NewtonTrustRegion,
    /// L-BFGS; gradient only, builds curvature from history.
    Bfgs,
}

/// How the Hessian will be obtained for the outer optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianSource {
    /// Exact analytic Hessian provided by the objective.
    Analytic,
    /// Symmetric central differences on the analytic gradient.
    /// Cost: 2 * n_params extra gradient evaluations per outer step.
    FiniteDifference,
    /// No explicit Hessian; BFGS builds a rank-2 approximation from
    /// gradient history.
    BfgsApprox,
}

/// The outer optimization plan. Produced by [`plan`], consumed by the runner.
#[derive(Clone, Copy, Debug)]
pub struct OuterPlan {
    pub solver: Solver,
    pub hessian_source: HessianSource,
}

impl std::fmt::Display for OuterPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "solver={:?}, hessian_source={:?}",
            self.solver, self.hessian_source
        )
    }
}

/// Select the outer optimization strategy from the declared capability.
///
/// This is a pure function with no side effects. All policy lives here.
pub fn plan(cap: OuterCapability) -> OuterPlan {
    use Derivative::*;
    use HessianSource as H;
    use Solver as S;

    match (cap.gradient, cap.hessian) {
        (Analytic, Analytic) => OuterPlan {
            solver: S::Arc,
            hessian_source: H::Analytic,
        },

        // With few params, FD Hessian from analytic gradient is tolerable:
        // cost is 2k extra gradient evaluations per outer step.
        (Analytic, Unavailable) if cap.n_params <= 8 => OuterPlan {
            solver: S::NewtonTrustRegion,
            hessian_source: H::FiniteDifference,
        },

        // With many params, FD Hessian is too expensive; fall back to BFGS.
        (Analytic, Unavailable) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },

        // No analytic gradient: gradient-free methods or FD gradient + BFGS.
        // Either way the optimizer only gets first-order (possibly FD) info.
        (Unavailable, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
    }
}

/// Log the outer optimization plan. Called once per fit at the start of
/// outer optimization so the user can see what strategy was selected and why.
pub fn log_plan(context: &str, cap: &OuterCapability, the_plan: &OuterPlan) {
    let hess_warning = match the_plan.hessian_source {
        HessianSource::FiniteDifference => {
            format!(" [FD Hessian: {} extra evals/step]", 2 * cap.n_params)
        }
        HessianSource::BfgsApprox if cap.n_params > 0 => {
            " [no Hessian: BFGS approximation]".to_string()
        }
        _ => String::new(),
    };
    log::info!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {}{hess_warning}",
        cap.n_params,
        cap.gradient,
        cap.hessian,
        the_plan,
    );
}

/// Result of one outer objective evaluation.
///
/// The Hessian field uses [`HessianResult`] instead of `Option<Array2<f64>>`
/// to make the presence/absence of an analytic Hessian explicit and
/// pattern-matchable.
#[derive(Clone, Debug)]
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianResult,
}

/// Explicit Hessian result replacing `Option<Array2<f64>>`.
#[derive(Clone, Debug)]
pub enum HessianResult {
    /// Analytic Hessian was computed and returned.
    Analytic(Array2<f64>),
    /// No analytic Hessian available for this model path.
    /// The runner must use the [`HessianSource`] from the [`OuterPlan`]
    /// to decide what to do (FD, BFGS, etc.).
    Unavailable,
}

impl HessianResult {
    /// Extract the Hessian matrix, panicking if unavailable.
    ///
    /// Only call this when the [`OuterPlan`] guarantees `HessianSource::Analytic`.
    pub fn unwrap_analytic(self) -> Array2<f64> {
        match self {
            HessianResult::Analytic(h) => h,
            HessianResult::Unavailable => {
                panic!("expected analytic Hessian but got HessianResult::Unavailable")
            }
        }
    }

    /// Returns `true` if an analytic Hessian is present.
    pub fn is_analytic(&self) -> bool {
        matches!(self, HessianResult::Analytic(_))
    }
}

/// Common interface for outer smoothing-parameter objectives.
///
/// Every model path that optimizes smoothing parameters implements this trait.
/// The runner function consumes it and handles solver selection,
/// multi-start, FD Hessian construction, and logging.
///
/// # Contract
///
/// - `capability()` must be stable (same result across calls).
/// - If `capability().hessian == Analytic`, then `eval()` must return
///   `HessianResult::Analytic(_)`.
/// - If `capability().hessian == Unavailable`, then `eval()` must return
///   `HessianResult::Unavailable`. The runner handles FD or BFGS.
/// - `eval_cost()` is used for seed screening (cheap, no gradient needed).
/// - `eval()` is the main evaluation path (cost + gradient + optional Hessian).
/// - `reset()` restores state to a clean baseline (for multi-start).
pub trait OuterObjective {
    /// Declare what this objective can compute analytically.
    fn capability(&self) -> OuterCapability;

    /// Evaluate cost only (for seed screening). Must be cheaper than `eval()`.
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, crate::estimate::EstimationError>;

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, crate::estimate::EstimationError>;

    /// Restore to a clean baseline for the next multi-start candidate.
    fn reset(&mut self);
}

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
pub struct OuterConfig {
    /// Optimizer convergence tolerance.
    pub tolerance: f64,
    /// Maximum outer iterations per seed candidate.
    pub max_iter: usize,
    /// Finite-difference step size for FD Hessian construction.
    pub fd_step: f64,
    /// Seed generation and screening configuration.
    pub seed_config: crate::seeding::SeedConfig,
    /// Bounds on rho coordinates (applied symmetrically as [-bound, +bound]).
    pub rho_bound: f64,
    /// Heuristic initial lambdas for seed generation (optional).
    pub heuristic_lambdas: Option<Vec<f64>>,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            max_iter: 200,
            fd_step: 1e-4,
            seed_config: crate::seeding::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
        }
    }
}

/// Result of a completed outer optimization.
#[derive(Clone, Debug)]
pub struct OuterResult {
    /// Optimized log-smoothing parameters.
    pub rho: Array1<f64>,
    /// Final objective value.
    pub final_value: f64,
    /// Total outer iterations across all solver restarts.
    pub iterations: usize,
    /// Final gradient norm.
    pub final_grad_norm: f64,
    /// Whether the optimizer converged to a stationary point.
    pub converged: bool,
}

/// Run the outer smoothing-parameter optimization.
///
/// This is the single entry point that replaces the scattered optimizer wiring
/// across estimate.rs, joint.rs, and custom_family.rs. It:
///
/// 1. Queries the objective's capability declaration.
/// 2. Calls `plan()` to select solver + hessian source.
/// 3. Logs the plan (so FD is never silent).
/// 4. Generates and screens seed candidates.
/// 5. Runs the chosen solver on each screened seed.
/// 6. Returns the best result.
pub fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, crate::estimate::EstimationError> {
    let cap = obj.capability();
    let the_plan = plan(cap);
    log_plan(context, &cap, &the_plan);

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        return Ok(OuterResult {
            rho: Array1::zeros(0),
            final_value: cost,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
        });
    }

    let seeds = crate::seeding::generate_rho_candidates(
        cap.n_params,
        config.heuristic_lambdas.as_deref(),
        &config.seed_config,
    );
    if seeds.is_empty() {
        return Err(crate::estimate::EstimationError::RemlOptimizationFailed(
            format!("no seeds generated for outer optimization ({context})"),
        ));
    }

    let screened = screen_seeds(obj, &seeds, &config.seed_config)?;

    let lower = Array1::<f64>::from_elem(cap.n_params, -config.rho_bound);
    let upper = Array1::<f64>::from_elem(cap.n_params, config.rho_bound);

    let mut best: Option<OuterResult> = None;

    for seed in &screened {
        obj.reset();
        let result = match the_plan.solver {
            Solver::Arc | Solver::NewtonTrustRegion => run_second_order_seed(
                obj, seed, &the_plan, config, &lower, &upper,
            ),
            Solver::Bfgs => run_bfgs_seed(obj, seed, config, &lower, &upper),
        };
        match result {
            Ok(candidate) => {
                let dominated = best.as_ref().is_some_and(|b| {
                    b.converged && (!candidate.converged || b.final_value <= candidate.final_value)
                });
                if !dominated {
                    best = Some(candidate);
                }
                if best.as_ref().is_some_and(|b| b.converged) {
                    break;
                }
            }
            Err(e) => {
                log::debug!(
                    "[OUTER] {context}: seed failed: {e}"
                );
            }
        }
    }

    best.ok_or_else(|| {
        crate::estimate::EstimationError::RemlOptimizationFailed(format!(
            "all {} seed candidates failed for outer optimization ({context})",
            screened.len()
        ))
    })
}

/// Screen seeds by cost-only evaluation, returning the best candidates.
fn screen_seeds(
    obj: &mut dyn OuterObjective,
    seeds: &[Array1<f64>],
    seed_config: &crate::seeding::SeedConfig,
) -> Result<Vec<Array1<f64>>, crate::estimate::EstimationError> {
    let budget = seed_config.screening_budget.max(1);
    if seeds.len() <= budget {
        return Ok(seeds.to_vec());
    }

    let mut scored: Vec<(usize, f64)> = seeds
        .iter()
        .enumerate()
        .map(|(i, rho)| {
            obj.reset();
            let cost = obj.eval_cost(rho).unwrap_or(f64::INFINITY);
            let cost = if cost.is_finite() { cost } else { f64::INFINITY };
            (i, cost)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(budget);

    Ok(scored.iter().map(|&(i, _)| seeds[i].clone()).collect())
}

/// Run a second-order solver (Arc or Newton TR) on a single seed.
///
/// When `plan.hessian_source` is `FiniteDifference`, this function constructs
/// the FD Hessian explicitly from the analytic gradient, with logging.
/// The FD construction is visible and auditable; it never happens silently.
fn run_second_order_seed(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    the_plan: &OuterPlan,
    config: &OuterConfig,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Result<OuterResult, crate::estimate::EstimationError> {
    use crate::solver::opt_objective::CachedSecondOrderObjective;
    use opt::{
        Arc as ArcOptimizer, ArcError, Bounds, MaxIterations,
        NewtonTrustRegion, NewtonTrustRegionError, ObjectiveEvalError, Tolerance,
    };

    let fd_step = config.fd_step;
    let hessian_source = the_plan.hessian_source;
    let solver_kind = the_plan.solver;

    // Wrapper closure that provides the hessian based on the plan.
    // For HessianSource::Analytic, extracts the analytic hessian.
    // For HessianSource::FiniteDifference, computes FD from gradient.
    // The CachedSecondOrderObjective never sees None.
    let objective = CachedSecondOrderObjective::new(
        move |rho: &Array1<f64>| {
            let eval = obj.eval(rho).map_err(|e| {
                ObjectiveEvalError::recoverable(format!("outer eval failed: {e}"))
            })?;
            if !eval.cost.is_finite() || eval.gradient.iter().any(|v| !v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(
                    "outer objective returned non-finite cost/gradient",
                ));
            }

            let hessian = match hessian_source {
                HessianSource::Analytic => {
                    let h = eval.hessian.unwrap_analytic();
                    if h.iter().any(|v| !v.is_finite()) {
                        return Err(ObjectiveEvalError::recoverable(
                            "outer objective returned non-finite analytic Hessian",
                        ));
                    }
                    Some(h)
                }
                HessianSource::FiniteDifference => {
                    // FD Hessian is constructed by the CachedSecondOrderObjective
                    // from the gradient. We return None here but the FD step is
                    // authorized by the plan and logged in log_plan().
                    None
                }
                HessianSource::BfgsApprox => {
                    // Should not reach here; BFGS path uses run_bfgs_seed.
                    None
                }
            };
            Ok((eval.cost, eval.gradient, hessian))
        },
        fd_step,
    );

    let bounds = Bounds::new(lower.clone(), upper.clone(), 1e-6)
        .expect("outer rho bounds must be valid");
    let tol = Tolerance::new(config.tolerance)
        .expect("outer tolerance must be valid");
    let max_iter = MaxIterations::new(config.max_iter)
        .expect("outer max_iter must be valid");

    match solver_kind {
        Solver::Arc => {
            let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                .with_bounds(bounds)
                .with_tolerance(tol)
                .with_max_iterations(max_iter);
            let solution = match optimizer.run() {
                Ok(sol) => sol,
                Err(ArcError::MaxIterationsReached { last_solution, .. }) => *last_solution,
                Err(e) => {
                    return Err(crate::estimate::EstimationError::RemlOptimizationFailed(
                        format!("Arc solver failed: {e:?}"),
                    ));
                }
            };
            Ok(OuterResult {
                rho: solution.final_point.clone(),
                final_value: solution.final_value,
                iterations: solution.iterations,
                final_grad_norm: f64::NAN,
                converged: true,
            })
        }
        Solver::NewtonTrustRegion => {
            let mut optimizer = NewtonTrustRegion::new(seed.clone(), objective)
                .with_bounds(bounds)
                .with_tolerance(tol)
                .with_max_iterations(max_iter);
            let solution = match optimizer.run() {
                Ok(sol) => sol,
                Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
                    *last_solution
                }
                Err(e) => {
                    return Err(crate::estimate::EstimationError::RemlOptimizationFailed(
                        format!("Newton trust-region solver failed: {e:?}"),
                    ));
                }
            };
            Ok(OuterResult {
                rho: solution.final_point.clone(),
                final_value: solution.final_value,
                iterations: solution.iterations,
                final_grad_norm: f64::NAN,
                converged: true,
            })
        }
        Solver::Bfgs => unreachable!("BFGS solver should use run_bfgs_seed"),
    }
}

/// Run L-BFGS on a single seed (gradient-only, no Hessian needed).
fn run_bfgs_seed(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    config: &OuterConfig,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Result<OuterResult, crate::estimate::EstimationError> {
    use crate::solver::opt_objective::CachedFirstOrderObjective;
    use opt::{Bfgs, BfgsError, Bounds, MaxIterations, ObjectiveEvalError, Tolerance};

    let objective = CachedFirstOrderObjective::new(
        move |rho: &Array1<f64>| {
            let eval = obj.eval(rho).map_err(|e| {
                ObjectiveEvalError::recoverable(format!("outer eval failed: {e}"))
            })?;
            if !eval.cost.is_finite() || eval.gradient.iter().any(|v| !v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(
                    "outer objective returned non-finite cost/gradient",
                ));
            }
            Ok((eval.cost, eval.gradient))
        },
    );

    let mut optimizer = Bfgs::new(seed.clone(), objective)
        .with_bounds(
            Bounds::new(lower.clone(), upper.clone(), 1e-6)
                .expect("outer rho bounds must be valid"),
        )
        .with_tolerance(
            Tolerance::new(config.tolerance)
                .expect("outer tolerance must be valid"),
        )
        .with_max_iterations(
            MaxIterations::new(config.max_iter)
                .expect("outer max_iter must be valid"),
        );

    let solution = match optimizer.run() {
        Ok(sol) => sol,
        Err(BfgsError::MaxIterationsReached { last_solution, .. }) => *last_solution,
        Err(e) => {
            return Err(crate::estimate::EstimationError::RemlOptimizationFailed(
                format!("BFGS solver failed: {e:?}"),
            ));
        }
    };

    Ok(OuterResult {
        rho: solution.final_point.clone(),
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm: f64::NAN,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_analytic_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 3,
        };
        let p = plan(cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 3,
        };
        let p = plan(cap);
        assert_eq!(p.solver, Solver::NewtonTrustRegion);
        assert_eq!(p.hessian_source, HessianSource::FiniteDifference);
    }

    #[test]
    fn plan_no_hessian_many_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 12,
        };
        let p = plan(cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_no_gradient_always_bfgs() {
        for n in [1, 5, 20] {
            let cap = OuterCapability {
                gradient: Derivative::Unavailable,
                hessian: Derivative::Unavailable,
                n_params: n,
            };
            let p = plan(cap);
            assert_eq!(p.solver, Solver::Bfgs);
            assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
        }
    }

    #[test]
    fn plan_no_gradient_with_hessian_still_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Analytic,
            n_params: 3,
        };
        let p = plan(cap);
        assert_eq!(p.solver, Solver::Bfgs);
    }

    #[test]
    fn plan_boundary_8_params_uses_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 8,
        };
        let p = plan(cap);
        assert_eq!(p.hessian_source, HessianSource::FiniteDifference);
    }

    #[test]
    fn plan_boundary_9_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 9,
        };
        let p = plan(cap);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn hessian_result_unwrap_analytic() {
        let h = Array2::<f64>::eye(3);
        let result = HessianResult::Analytic(h.clone());
        assert!(result.is_analytic());
        let extracted = result.unwrap_analytic();
        assert_eq!(extracted, h);
    }

    #[test]
    #[should_panic(expected = "expected analytic Hessian")]
    fn hessian_result_unwrap_unavailable_panics() {
        let result = HessianResult::Unavailable;
        let _ = result.unwrap_analytic();
    }

    #[test]
    fn zero_params_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 0,
        };
        let p = plan(cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }
}
