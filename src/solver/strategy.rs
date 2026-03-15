//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, joint flexible
//! link, GAMLSS custom family, spatial kappa, etc.) declares its derivative
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

use crate::estimate::EstimationError;
use crate::solver::estimate::reml::unified::BarrierConfig;
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
#[derive(Clone, Debug)]
pub struct OuterCapability {
    pub gradient: Derivative,
    pub hessian: Derivative,
    /// Number of smoothing (+ any auxiliary hyper-) parameters being optimized.
    pub n_params: usize,
    /// Whether all hyperparameter coordinates (both rho and any extended coords)
    /// are penalty-like. When true, the EFS (Extended Fellner-Schall) fixed-point
    /// optimizer is eligible. When false (e.g. psi/design-moving coordinates),
    /// EFS cannot be used because the multiplicative update structure breaks down.
    pub all_penalty_like: bool,
    /// Optional log-barrier configuration for structural monotonicity constraints.
    /// When present, EFS is still eligible at plan time, but the EFS iteration
    /// loop performs a quantitative check each step: if
    /// `barrier_curvature_is_significant(β, ref_diag, threshold)` fires, EFS
    /// bails out early and the result is finalized at the current rho.
    ///
    /// Previously this was a binary `barrier_active: bool` that unconditionally
    /// blocked EFS. The quantitative check allows EFS when constraints exist but
    /// the barrier curvature is negligible (coefficients far from their bounds).
    pub barrier_config: Option<BarrierConfig>,
    /// When set, `plan()` bypasses its normal selection logic and returns the
    /// specified solver directly. Only used in fallback_sequence entries to
    /// force `CoordinateSearch` (which `plan()` never auto-selects).
    pub force_solver: Option<Solver>,
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
    /// Extended Fellner-Schall; multiplicative fixed-point iteration.
    /// Only valid when all hyperparameter coordinates are penalty-like.
    /// Needs no gradient or Hessian — only traces tr(H^{-1} A_k) and
    /// Frobenius norms from the inner solution.
    Efs,
    /// Derivative-free coordinate search. Never auto-selected by `plan()`;
    /// only reachable via `fallback_sequence` in `OuterConfig`.
    CoordinateSearch,
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
    /// No explicit Hessian or gradient needed. EFS uses traces and
    /// Frobenius norms from the inner solution directly.
    EfsFixedPoint,
    /// No gradient or Hessian needed. CoordinateSearch uses only `eval_cost`.
    None,
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
pub fn plan(cap: &OuterCapability) -> OuterPlan {
    use Derivative::*;
    use HessianSource as H;
    use Solver as S;

    if let Some(forced) = cap.force_solver {
        let hessian_source = match forced {
            S::Arc => H::Analytic,
            S::NewtonTrustRegion => {
                if cap.hessian == Analytic { H::Analytic } else { H::FiniteDifference }
            }
            S::Bfgs => H::BfgsApprox,
            S::Efs => H::EfsFixedPoint,
            S::CoordinateSearch => H::None,
        };
        return OuterPlan { solver: forced, hessian_source };
    }

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

        // EFS: all penalty-like coords, no analytic Hessian, many params.
        // Multiplicative fixed-point needs only traces — no gradient evals.
        // Much cheaper than BFGS for k=10-50 smoothing parameters.
        //
        // When a log-barrier is present (monotonicity constraints), EFS is
        // still selected here. The EFS iteration loop in `run_outer` performs
        // a quantitative check each step via `barrier_curvature_is_significant`
        // and bails out early if the barrier curvature becomes non-negligible
        // relative to the penalized Hessian diagonal.
        (_, Unavailable) if cap.all_penalty_like && cap.n_params > 8 => {
            OuterPlan {
                solver: S::Efs,
                hessian_source: H::EfsFixedPoint,
            }
        }

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
    let barrier_note = if cap.barrier_config.is_some() && cap.all_penalty_like && cap.n_params > 8 {
        " [EFS with runtime barrier-curvature guard]"
    } else {
        ""
    };
    log::info!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {}{hess_warning}{barrier_note}",
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

    /// Convert to `Option<Array2<f64>>` for interop with legacy code.
    pub fn into_option(self) -> Option<Array2<f64>> {
        match self {
            HessianResult::Analytic(h) => Some(h),
            HessianResult::Unavailable => None,
        }
    }
}

/// Result of an EFS (Extended Fellner-Schall) evaluation at a given rho.
///
/// Contains the REML/LAML cost at the current rho and the additive step
/// vector produced by `compute_efs_update`. The caller applies the step as
/// `rho_new[i] = rho[i] + steps[i]`.
#[derive(Clone, Debug)]
pub struct EfsEval {
    /// REML/LAML cost at the current rho (for convergence monitoring and
    /// comparing candidates).
    pub cost: f64,
    /// Additive EFS steps. Length = n_rho + n_ext_coords.
    /// Steps for non-penalty-like coordinates are 0.0.
    pub steps: Vec<f64>,
    /// Current coefficient vector β̂ from the inner P-IRLS solve.
    /// Used by the EFS loop for the runtime barrier-curvature significance
    /// check when monotonicity constraints are present.
    pub beta: Option<Array1<f64>>,
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
/// - `eval_efs()` is used only by the EFS solver. It runs the inner solve,
///   builds the `InnerSolution`, and computes the EFS step vector. The default
///   implementation returns an error; only objectives that support EFS need
///   to override it.
/// - `reset()` restores state to a clean baseline (for multi-start).
pub trait OuterObjective {
    /// Declare what this objective can compute analytically.
    fn capability(&self) -> OuterCapability;

    /// Evaluate cost only (for seed screening). Must be cheaper than `eval()`.
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError>;

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError>;

    /// Evaluate cost + EFS step vector. Only needed when the plan selects
    /// `Solver::Efs`. The default returns an error indicating EFS is not
    /// supported by this objective.
    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        let _ = rho;
        Err(EstimationError::RemlOptimizationFailed(
            "EFS evaluation not implemented for this objective".to_string(),
        ))
    }

    /// Restore to a clean baseline for the next multi-start candidate.
    fn reset(&mut self);
}

/// Closure-based adapter for [`OuterObjective`].
///
/// This allows any call site to construct an `OuterObjective` from closures
/// without needing to define a wrapper struct or modify the state type.
/// Each call site wraps its existing methods into closures and passes them here.
pub struct ClosureObjective<S, Fc, Fe, Fr, Fefs = fn(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>> {
    pub state: S,
    pub cap: OuterCapability,
    pub cost_fn: Fc,
    pub eval_fn: Fe,
    pub reset_fn: Fr,
    /// Optional EFS evaluation closure. When `None`, the default
    /// `OuterObjective::eval_efs` returns an error.
    pub efs_fn: Option<Fefs>,
}

impl<S, Fc, Fe, Fr, Fefs> OuterObjective for ClosureObjective<S, Fc, Fe, Fr, Fefs>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
{
    fn capability(&self) -> OuterCapability {
        self.cap.clone()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        (self.cost_fn)(&mut self.state, rho)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        (self.eval_fn)(&mut self.state, rho)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        match self.efs_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => Err(EstimationError::RemlOptimizationFailed(
                "EFS evaluation not implemented for this objective".to_string(),
            )),
        }
    }

    fn reset(&mut self) {
        (self.reset_fn)(&mut self.state);
    }
}

impl OuterResult {
    /// Convert to legacy `SmoothingBfgsResult` for backwards compatibility.
    pub fn into_smoothing_result(self) -> crate::solver::smoothing::SmoothingBfgsResult {
        crate::solver::smoothing::SmoothingBfgsResult {
            rho: self.rho,
            final_value: self.final_value,
            iterations: self.iterations,
            finalgrad_norm: self.final_grad_norm,
            final_stationarity_residual: if self.converged { 0.0 } else { f64::NAN },
            final_boundviolation: 0.0,
            stationary: self.converged,
        }
    }
}

/// Configuration for coordinate search (derivative-free optimizer).
#[derive(Clone, Debug)]
pub struct CoordinateSearchConfig {
    /// Step size in log-space for probing each coordinate.
    pub log_step: f64,
    /// Maximum number of full passes over all coordinates.
    pub max_coord_iters: usize,
    /// Relative tolerance: an improvement counts only if
    /// `new_cost < old_cost * (1 - rel_tol)`.
    pub rel_tol: f64,
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
    /// If provided, use this as the sole starting point (skip seed generation
    /// and screening). Useful when the caller already has a good initial rho.
    pub initial_rho: Option<Array1<f64>>,
    /// Ordered list of degraded capabilities to try when the primary plan
    /// fails. Each entry triggers `plan()` with the degraded capability and
    /// a fresh solver run. Empty by default (no fallback).
    pub fallback_sequence: Vec<OuterCapability>,
    /// Configuration for coordinate search. Only used when `plan()` selects
    /// `Solver::CoordinateSearch` (via fallback).
    pub coordinate_search: Option<CoordinateSearchConfig>,
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
            initial_rho: None,
            fallback_sequence: Vec::new(),
            coordinate_search: None,
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
    /// Which plan was actually used (may differ from initial if fallback fired).
    pub plan_used: OuterPlan,
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
/// 6. If all seeds fail and `fallback_sequence` is non-empty, re-plans
///    with degraded capability and retries.
/// 7. Returns the best result (including which plan was actually used).
pub fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let cap = obj.capability();

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan(&cap);
        return Ok(OuterResult {
            rho: Array1::zeros(0),
            final_value: cost,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            plan_used: the_plan,
        });
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // each entry from fallback_sequence.
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + config.fallback_sequence.len());
    attempts.push(cap.clone());
    for fb in &config.fallback_sequence {
        let mut degraded = fb.clone();
        degraded.n_params = cap.n_params;
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
        if attempt_idx > 0 {
            log::info!(
                "[OUTER] {context}: primary plan failed; falling back to {the_plan}"
            );
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        match run_outer_with_plan(obj, config, context, attempt_cap, &the_plan) {
            Ok(mut result) => {
                result.plan_used = the_plan;
                return Ok(result);
            }
            Err(e) => {
                log::debug!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) failed: {e}",
                    attempt_idx + 1
                );
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!(
            "all plan attempts exhausted ({context})"
        ))
    }))
}

/// Execute a single plan attempt (seed generation → solver loop → best result).
fn run_outer_with_plan(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    cap: &OuterCapability,
    the_plan: &OuterPlan,
) -> Result<OuterResult, EstimationError> {
    use crate::solver::opt_objective::{CachedFirstOrderObjective, CachedSecondOrderObjective};
    use opt::{
        Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, MaxIterations, NewtonTrustRegion,
        NewtonTrustRegionError, ObjectiveEvalError, Tolerance,
    };

    let seeds = if let Some(ref rho) = config.initial_rho {
        vec![rho.clone()]
    } else {
        let generated = crate::seeding::generate_rho_candidates(
            cap.n_params,
            config.heuristic_lambdas.as_deref(),
            &config.seed_config,
        );
        if generated.is_empty() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "no seeds generated for outer optimization ({context})"
            )));
        }
        generated
    };

    // Screen seeds by cost-only evaluation.
    let budget = config.seed_config.screening_budget.max(1);
    let screened = if seeds.len() <= budget {
        seeds
    } else {
        let mut scored: Vec<(usize, f64)> = seeds
            .iter()
            .enumerate()
            .map(|(i, rho)| {
                obj.reset();
                let cost = obj.eval_cost(rho).unwrap_or(f64::INFINITY);
                (
                    i,
                    if cost.is_finite() {
                        cost
                    } else {
                        f64::INFINITY
                    },
                )
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(budget);
        scored.iter().map(|&(i, _)| seeds[i].clone()).collect()
    };

    let lower = Array1::<f64>::from_elem(cap.n_params, -config.rho_bound);
    let upper = Array1::<f64>::from_elem(cap.n_params, config.rho_bound);
    let bounds_template = (lower, upper);

    struct RawOuterCandidate {
        rho: Array1<f64>,
        final_value: f64,
        iterations: usize,
        converged: bool,
    }

    let mut best: Option<OuterResult> = None;

    let finalize_candidate = |obj: &mut dyn OuterObjective,
                              candidate: RawOuterCandidate|
     -> Result<OuterResult, EstimationError> {
        obj.reset();
        let (final_value, final_grad_norm) = if cap.gradient == Derivative::Analytic {
            let eval = obj.eval(&candidate.rho)?;
            if !eval.cost.is_finite() || eval.gradient.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "final outer evaluation returned non-finite cost/gradient".to_string(),
                ));
            }
            let grad_norm = eval.gradient.dot(&eval.gradient).sqrt();
            if !grad_norm.is_finite() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "final outer gradient norm was non-finite".to_string(),
                ));
            }
            (eval.cost, grad_norm)
        } else {
            let cost = obj.eval_cost(&candidate.rho)?;
            if !cost.is_finite() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "final outer cost was non-finite".to_string(),
                ));
            }
            let mut grad = Array1::zeros(cap.n_params);
            for i in 0..cap.n_params {
                let h = config.fd_step * (1.0 + candidate.rho[i].abs());
                let mut rp = candidate.rho.clone();
                let mut rm = candidate.rho.clone();
                rp[i] += h;
                rm[i] -= h;
                let fp = obj.eval_cost(&rp)?;
                let fm = obj.eval_cost(&rm)?;
                if !fp.is_finite() || !fm.is_finite() {
                    return Err(EstimationError::RemlOptimizationFailed(
                        "final outer finite-difference gradient used non-finite costs".to_string(),
                    ));
                }
                grad[i] = (fp - fm) / (2.0 * h);
            }
            let grad_norm = grad.dot(&grad).sqrt();
            if !grad_norm.is_finite() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "final outer finite-difference gradient norm was non-finite".to_string(),
                ));
            }
            (cost, grad_norm)
        };
        Ok(OuterResult {
            rho: candidate.rho,
            final_value,
            iterations: candidate.iterations,
            final_grad_norm,
            converged: candidate.converged,
            plan_used: *the_plan,
        })
    };

    for seed in &screened {
        obj.reset();

        let result: Result<RawOuterCandidate, EstimationError> = match the_plan.solver {
            Solver::Arc | Solver::NewtonTrustRegion => {
                let hessian_source = the_plan.hessian_source;
                let fd_step = config.fd_step;
                let objective = CachedSecondOrderObjective::new(
                    |rho: &Array1<f64>| {
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
                                match eval.hessian {
                                    HessianResult::Analytic(h) => {
                                        if h.iter().any(|v| !v.is_finite()) {
                                            return Err(ObjectiveEvalError::recoverable(
                                                "analytic Hessian contained non-finite values",
                                            ));
                                        }
                                        Some(h)
                                    }
                                    HessianResult::Unavailable => {
                                        log::debug!(
                                            "[OUTER] analytic Hessian expected but unavailable; falling back to FD for this step"
                                        );
                                        None
                                    }
                                }
                            }
                            HessianSource::FiniteDifference => None,
                            HessianSource::BfgsApprox
                            | HessianSource::EfsFixedPoint
                            | HessianSource::None => None,
                        };
                        Ok((eval.cost, eval.gradient, hessian))
                    },
                    fd_step,
                );

                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");

                if the_plan.solver == Solver::Arc {
                    let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter);
                    match optimizer.run() {
                        Ok(sol) => Ok(RawOuterCandidate {
                            rho: sol.final_point.clone(),
                            final_value: sol.final_value,
                            iterations: sol.iterations,
                            converged: true,
                        }),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            Ok(RawOuterCandidate {
                                rho: last_solution.final_point.clone(),
                                final_value: last_solution.final_value,
                                iterations: last_solution.iterations,
                                converged: false,
                            })
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Arc solver failed: {e:?}"
                        ))),
                    }
                } else {
                    let mut optimizer = NewtonTrustRegion::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter);
                    match optimizer.run() {
                        Ok(sol) => Ok(RawOuterCandidate {
                            rho: sol.final_point.clone(),
                            final_value: sol.final_value,
                            iterations: sol.iterations,
                            converged: true,
                        }),
                        Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
                            Ok(RawOuterCandidate {
                                rho: last_solution.final_point.clone(),
                                final_value: last_solution.final_value,
                                iterations: last_solution.iterations,
                                converged: false,
                            })
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Newton trust-region solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Bfgs => {
                let gradient_available = cap.gradient == Derivative::Analytic;
                let fd_step = config.fd_step;
                let n_params = cap.n_params;
                let objective = CachedFirstOrderObjective::new(|rho: &Array1<f64>| {
                    if gradient_available {
                        let eval = obj.eval(rho).map_err(|e| {
                            ObjectiveEvalError::recoverable(format!("outer eval failed: {e}"))
                        })?;
                        if !eval.cost.is_finite() || eval.gradient.iter().any(|v| !v.is_finite()) {
                            return Err(ObjectiveEvalError::recoverable(
                                "outer objective returned non-finite cost/gradient",
                            ));
                        }
                        Ok((eval.cost, eval.gradient))
                    } else {
                        let cost = obj.eval_cost(rho).map_err(|e| {
                            ObjectiveEvalError::recoverable(format!("outer eval_cost failed: {e}"))
                        })?;
                        if !cost.is_finite() {
                            return Err(ObjectiveEvalError::recoverable(
                                "outer objective returned non-finite cost",
                            ));
                        }
                        let mut grad = Array1::zeros(n_params);
                        for i in 0..n_params {
                            let h = fd_step * (1.0 + rho[i].abs());
                            let mut rp = rho.clone();
                            let mut rm = rho.clone();
                            rp[i] += h;
                            rm[i] -= h;
                            let fp = obj.eval_cost(&rp).map_err(|e| {
                                ObjectiveEvalError::recoverable(format!(
                                    "outer FD eval_cost failed: {e}"
                                ))
                            })?;
                            let fm = obj.eval_cost(&rm).map_err(|e| {
                                ObjectiveEvalError::recoverable(format!(
                                    "outer FD eval_cost failed: {e}"
                                ))
                            })?;
                            grad[i] = if fp.is_finite() && fm.is_finite() {
                                (fp - fm) / (2.0 * h)
                            } else {
                                0.0
                            };
                        }
                        Ok((cost, grad))
                    }
                });
                let (lo, hi) = &bounds_template;
                let mut optimizer = Bfgs::new(seed.clone(), objective)
                    .with_bounds(
                        Bounds::new(lo.clone(), hi.clone(), 1e-6)
                            .expect("outer rho bounds must be valid"),
                    )
                    .with_tolerance(
                        Tolerance::new(config.tolerance).expect("outer tolerance must be valid"),
                    )
                    .with_max_iterations(
                        MaxIterations::new(config.max_iter).expect("outer max_iter must be valid"),
                    );
                match optimizer.run() {
                    Ok(sol) => Ok(RawOuterCandidate {
                        rho: sol.final_point.clone(),
                        final_value: sol.final_value,
                        iterations: sol.iterations,
                        converged: true,
                    }),
                    Err(BfgsError::MaxIterationsReached { last_solution }) => {
                        Ok(RawOuterCandidate {
                            rho: last_solution.final_point.clone(),
                            final_value: last_solution.final_value,
                            iterations: last_solution.iterations,
                            converged: false,
                        })
                    }
                    Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
                        Ok(RawOuterCandidate {
                            rho: last_solution.final_point.clone(),
                            final_value: last_solution.final_value,
                            iterations: last_solution.iterations,
                            converged: false,
                        })
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "BFGS solver failed: {e:?}"
                    ))),
                }
            }
            Solver::Efs => {
                let max_efs_iter = config.max_iter;
                let efs_tol = config.tolerance;
                let (lo, hi) = &bounds_template;

                let mut rho = seed.clone();
                let mut last_cost = f64::INFINITY;
                let mut total_iter = 0_usize;
                let mut converged = false;

                for iter in 0..max_efs_iter {
                    total_iter = iter + 1;

                    let efs_eval = match obj.eval_efs(&rho) {
                        Ok(e) => e,
                        Err(e) => {
                            log::debug!(
                                "[OUTER] EFS iteration {iter} failed: {e}; using last rho"
                            );
                            break;
                        }
                    };

                    if !efs_eval.cost.is_finite() {
                        log::debug!(
                            "[OUTER] EFS iteration {iter}: non-finite cost; stopping"
                        );
                        break;
                    }
                    last_cost = efs_eval.cost;

                    if let Some(ref barrier_cfg) = cap.barrier_config {
                        if let Some(ref beta) = efs_eval.beta {
                            let ref_diag = 1.0;
                            let threshold = 0.01;
                            if barrier_cfg.barrier_curvature_is_significant(
                                beta, ref_diag, threshold,
                            ) {
                                log::info!(
                                    "[OUTER] EFS iter {iter}: barrier curvature significant \
                                     (τ/(β-l)² > {threshold} × ref_diag); stopping EFS early"
                                );
                                break;
                            }
                        }
                    }

                    let mut step_sq_sum = 0.0_f64;
                    for i in 0..cap.n_params {
                        let step_i = if i < efs_eval.steps.len() {
                            efs_eval.steps[i]
                        } else {
                            0.0
                        };
                        let new_val = (rho[i] + step_i).clamp(lo[i], hi[i]);
                        let actual_step = new_val - rho[i];
                        step_sq_sum += actual_step * actual_step;
                        rho[i] = new_val;
                    }

                    let step_norm = step_sq_sum.sqrt();
                    log::trace!(
                        "[OUTER] EFS iter {iter}: cost={:.6e}, step_norm={:.4e}",
                        last_cost,
                        step_norm
                    );

                    if step_norm < efs_tol {
                        converged = true;
                        break;
                    }
                }

                Ok(RawOuterCandidate {
                    rho,
                    final_value: last_cost,
                    iterations: total_iter,
                    converged,
                })
            }
            Solver::CoordinateSearch => {
                let cs = config.coordinate_search.as_ref().ok_or_else(|| {
                    EstimationError::RemlOptimizationFailed(
                        "CoordinateSearch selected but no coordinate_search config provided"
                            .to_string(),
                    )
                })?;
                let (lo, hi) = &bounds_template;
                run_coordinate_search(obj, seed, lo, hi, cs, cap.n_params)
                    .map(|r| RawOuterCandidate {
                        rho: r.rho,
                        final_value: r.final_value,
                        iterations: r.iterations,
                        converged: true,
                    })
            }
        };

        match result.and_then(|candidate| finalize_candidate(obj, candidate)) {
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
                log::debug!("[OUTER] {context}: seed failed: {e}");
            }
        }
    }

    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!(
            "all {} seed candidates failed ({context})",
            screened.len()
        ))
    })
}

/// Coordinate search result: (rho, final_value, iterations, converged).
struct CoordSearchResult {
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
}

/// Derivative-free coordinate search over bounded log-space parameters.
///
/// This is the `run_outer` equivalent of the former `coordinate_search_spatial`
/// from smooth.rs. It only uses `obj.eval_cost()`.
fn run_coordinate_search(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    cs: &CoordinateSearchConfig,
    n_params: usize,
) -> Result<CoordSearchResult, EstimationError> {
    log::warn!(
        "[coordinate_search] Using derivative-free coordinate search over {} parameters. \
         This is slow — analytic REML derivatives should be wired through ext_coords instead.",
        n_params
    );

    let mut current_theta = seed.clone();
    let mut current_cost = obj
        .eval_cost(&current_theta)
        .unwrap_or(f64::INFINITY);
    let mut total_iters = 0_usize;

    for pass in 0..cs.max_coord_iters {
        total_iters = pass + 1;
        let mut pass_improved = false;

        for coord in 0..current_theta.len() {
            let base_value = current_theta[coord];
            let base_cost = current_cost;
            let left_value = (base_value - cs.log_step).max(lower[coord]);
            let right_value = (base_value + cs.log_step).min(upper[coord]);

            let left_cost = if (left_value - base_value).abs() <= 1e-12 {
                f64::INFINITY
            } else {
                let mut probe = current_theta.clone();
                probe[coord] = left_value;
                obj.eval_cost(&probe).unwrap_or(f64::INFINITY)
            };
            let right_cost = if (right_value - base_value).abs() <= 1e-12 {
                f64::INFINITY
            } else {
                let mut probe = current_theta.clone();
                probe[coord] = right_value;
                obj.eval_cost(&probe).unwrap_or(f64::INFINITY)
            };

            // Quadratic interpolation attempt.
            if left_cost.is_finite() && right_cost.is_finite() {
                let d_left = left_value - base_value;
                let d_right = right_value - base_value;
                if d_left.abs() > 1e-15 && d_right.abs() > 1e-15 && (d_left - d_right).abs() > 1e-15 {
                    let a_left = (left_cost - base_cost) / d_left;
                    let a_right = (right_cost - base_cost) / d_right;
                    let curvature = (a_left - a_right) / (d_left - d_right);
                    if curvature.is_finite() && curvature > 0.0 {
                        let slope = a_left - curvature * d_left;
                        let x_star = base_value - slope / (2.0 * curvature);
                        if x_star.is_finite() {
                            let interior = x_star.clamp(lower[coord], upper[coord]);
                            let lo_bound = left_value.min(right_value) + 1e-12;
                            let hi_bound = left_value.max(right_value) - 1e-12;
                            if interior > lo_bound && interior < hi_bound {
                                let mut probe = current_theta.clone();
                                probe[coord] = interior;
                                let interior_cost = obj.eval_cost(&probe).unwrap_or(f64::INFINITY);
                                if coord_search_improves(interior_cost, current_cost, cs.rel_tol) {
                                    current_theta[coord] = interior;
                                    current_cost = interior_cost;
                                    pass_improved = true;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // Greedy extension in improving direction.
            let (mut cand_value, mut cand_cost, direction) =
                if coord_search_improves(left_cost, base_cost, cs.rel_tol)
                    && left_cost <= right_cost
                {
                    (left_value, left_cost, -1.0_f64)
                } else if coord_search_improves(right_cost, base_cost, cs.rel_tol) {
                    (right_value, right_cost, 1.0_f64)
                } else {
                    continue;
                };

            loop {
                let next_value = (cand_value + direction * cs.log_step)
                    .clamp(lower[coord], upper[coord]);
                if (next_value - cand_value).abs() <= 1e-12 {
                    break;
                }
                let mut probe = current_theta.clone();
                probe[coord] = next_value;
                let next_cost = obj.eval_cost(&probe).unwrap_or(f64::INFINITY);
                if !coord_search_improves(next_cost, cand_cost, cs.rel_tol) {
                    break;
                }
                cand_value = next_value;
                cand_cost = next_cost;
            }

            if coord_search_improves(cand_cost, current_cost, cs.rel_tol) {
                current_theta[coord] = cand_value;
                current_cost = cand_cost;
                pass_improved = true;
            }
        }

        if !pass_improved {
            break;
        }
    }

    Ok(CoordSearchResult {
        rho: current_theta,
        final_value: current_cost,
        iterations: total_iters,
    })
}

fn coord_search_improves(new_cost: f64, old_cost: f64, rel_tol: f64) -> bool {
    new_cost.is_finite() && new_cost < old_cost * (1.0 - rel_tol)
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
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 3,
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::NewtonTrustRegion);
        assert_eq!(p.hessian_source, HessianSource::FiniteDifference);
    }

    #[test]
    fn plan_no_hessian_many_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 12,
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
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
                all_penalty_like: false,
                barrier_config: None,
                force_solver: None,
            };
            let p = plan(&cap);
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
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
    }

    #[test]
    fn plan_boundary_8_params_uses_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 8,
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.hessian_source, HessianSource::FiniteDifference);
    }

    #[test]
    fn plan_boundary_9_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 9,
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_selected_for_penalty_like_many_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: true,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_not_selected_few_params_even_if_penalty_like() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 5,
            all_penalty_like: true,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        // With few params and analytic gradient, FD Newton is better.
        assert_eq!(p.solver, Solver::NewtonTrustRegion);
    }

    #[test]
    fn plan_efs_not_selected_with_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 20,
            all_penalty_like: true,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        // Arc is always preferred when analytic Hessian is available.
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_efs_with_no_gradient_penalty_like_many_params() {
        // Even without analytic gradient, EFS works because it doesn't
        // need the gradient at all.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            n_params: 20,
            all_penalty_like: true,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config() {
        // When barrier_config is present (monotonicity constraints), EFS is
        // still selected at plan time. The runtime barrier-curvature guard
        // in the EFS loop handles safety.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0, 1],
            lower_bounds: vec![0.0, 0.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: true,
            barrier_config: Some(barrier),
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config_no_gradient() {
        // Even without analytic gradient, EFS is selected when all coords
        // are penalty-like and n_params > 8, regardless of barrier presence.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            n_params: 20,
            all_penalty_like: true,
            barrier_config: Some(barrier),
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn barrier_curvature_significant_blocks_efs_at_runtime() {
        // Verify that barrier_curvature_is_significant correctly detects
        // when coefficients are near their bounds.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
        };
        // β very close to bound → curvature is large
        let beta_near = Array1::from_vec(vec![0.001]);
        assert!(barrier.barrier_curvature_is_significant(&beta_near, 1.0, 0.01));

        // β far from bound → curvature is negligible
        let beta_far = Array1::from_vec(vec![10.0]);
        assert!(!barrier.barrier_curvature_is_significant(&beta_far, 1.0, 0.01));
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
            all_penalty_like: false,
            barrier_config: None,
            force_solver: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn hessian_result_into_option() {
        let h = Array2::<f64>::eye(2);
        let result = HessianResult::Analytic(h.clone());
        assert_eq!(result.into_option(), Some(h));

        let result = HessianResult::Unavailable;
        assert_eq!(result.into_option(), None);
    }

    #[test]
    fn closure_objective_delegates() {
        let mut obj = ClosureObjective {
            state: 42_i32,
            cap: OuterCapability {
                gradient: Derivative::Analytic,
                hessian: Derivative::Unavailable,
                n_params: 1,
                all_penalty_like: false,
                barrier_config: None,
                force_solver: None,
            },
            cost_fn: |st: &mut i32, rho: &Array1<f64>| {
                let _ = (*st, rho.len());
                Ok(1.0)
            },
            eval_fn: |st: &mut i32, rho: &Array1<f64>| {
                let _ = (*st, rho.len());
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                })
            },
            reset_fn: |st: &mut i32| {
                *st = 42;
            },
            efs_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        };
        assert_eq!(obj.capability().n_params, 1);
        assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
    }

    #[test]
    fn outer_config_default() {
        let cfg = OuterConfig::default();
        assert_eq!(cfg.tolerance, 1e-5);
        assert_eq!(cfg.max_iter, 200);
        assert_eq!(cfg.rho_bound, 30.0);
    }
}
