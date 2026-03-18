//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, link-wiggle,
//! GAMLSS custom family, spatial kappa, etc.) declares its derivative
//! capability here and receives an [`OuterPlan`] that determines which solver
//! and Hessian source to use.
//!
//! # Design invariant
//!
//! Finite-difference Hessian fallback is _never_ silent. If a path cannot
//! provide an analytic Hessian, that fact is visible in its
//! [`OuterCapability`] declaration and in the resulting [`OuterPlan`].
//! The runner passes Hessian availability through to `opt`, which performs
//! the finite-difference fill internally when the selected plan requires it.

use crate::estimate::EstimationError;
use crate::solver::estimate::reml::unified::BarrierConfig;
use ::opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, FiniteDiffGradient,
    FirstOrderObjective, FirstOrderSample, FixedPoint, FixedPointError, FixedPointObjective,
    FixedPointSample, FixedPointStatus, MaxIterations, NewtonTrustRegion, NewtonTrustRegionError,
    ObjectiveEvalError, SecondOrderObjective, SecondOrderSample, Solution, Tolerance,
    ZerothOrderObjective,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Whether an analytic derivative is available for a given order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// Exact analytic derivative implemented and available.
    Analytic,
    /// Derivative is available only via finite differences computed inside
    /// `opt` from a lower-order objective interface.
    FiniteDifference,
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
    /// are penalty-like. When true, the pure EFS (Extended Fellner-Schall)
    /// fixed-point optimizer can be eligible when `fixed_point_available` is
    /// also true. When false (e.g. psi/design-moving coordinates), the
    /// fallback is not automatically hybrid EFS; that still requires
    /// `fixed_point_available`.
    ///
    /// # Hybrid EFS strategy (when `has_psi_coords` is true)
    ///
    /// The hybrid is enabled when `has_psi_coords = true` and `n_params > 8`.
    /// It combines:
    /// - Standard EFS multiplicative fixed-point updates for ρ coordinates
    /// - Safeguarded preconditioned gradient steps for ψ coordinates:
    ///   `Δψ = -α G⁺ g_ψ` where G is the trace Gram matrix
    ///
    /// This is mathematically necessary because no EFS-type fixed-point
    /// iteration exists for indefinite B_ψ (see response.md Section 2 for
    /// the counterexample proof). The structural requirement for EFS is
    /// `H^{-1/2} B_d H^{-1/2} ≽ 0` (PSD) plus fixed nullspace — exactly
    /// what penalty-like coords satisfy and design-moving coords do not.
    pub all_penalty_like: bool,
    /// Whether ψ (design-moving) coordinates are present among the extended
    /// hyperparameter coordinates. When true together with `n_params > 8` and
    /// `fixed_point_available`, the planner can select hybrid EFS instead of
    /// falling back to full BFGS.
    ///
    /// The hybrid strategy is O(1) H⁻¹ solves per iteration (same as pure
    /// EFS), compared to O(dim(θ)) for BFGS which needs finite-difference
    /// gradient evaluations. This makes the hybrid dramatically cheaper for
    /// high-dimensional smoothing parameter spaces.
    pub has_psi_coords: bool,
    /// Whether the objective actually implements `eval_efs()` for fixed-point
    /// plans. Structural eligibility (`all_penalty_like` / `has_psi_coords`)
    /// is not sufficient by itself: if this is false, the planner must stay on
    /// Newton/BFGS-style plans even when EFS or Hybrid-EFS would otherwise be
    /// mathematically admissible.
    pub fixed_point_available: bool,
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
    /// Hybrid EFS + preconditioned gradient.
    ///
    /// Used when ψ (design-moving) coordinates are present alongside ρ
    /// (penalty-like) coordinates. Combines:
    /// - Standard EFS multiplicative fixed-point steps for ρ coords
    /// - Safeguarded preconditioned gradient steps for ψ coords:
    ///   `Δψ = -α G⁺ g_ψ` where `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)`
    ///
    /// This hybrid exists because no EFS-type fixed-point iteration can
    /// guarantee convergence for indefinite B_ψ (proven by counterexample
    /// in response.md Section 2). The key structural property that EFS
    /// needs — `H^{-1/2} B_d H^{-1/2} ≽ 0` plus parameter-independent
    /// nullspace — holds for penalty-like coords but fails for
    /// design-moving coords where B_ψ has mixed inertia.
    ///
    /// The preconditioned gradient uses the same trace Gram matrix that
    /// EFS already computes, so the cost is O(1) H⁻¹ solves per iteration
    /// (same as pure EFS), compared to O(dim(θ)) for full BFGS.
    HybridEfs,
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
    /// Hybrid EFS + preconditioned gradient for ψ coordinates.
    /// EFS traces for ρ coords, trace Gram matrix + gradient for ψ coords.
    HybridEfsFixedPoint,
}

/// The outer optimization plan. Produced by [`plan`], consumed by the runner.
#[derive(Clone, Copy, Debug)]
pub struct OuterPlan {
    pub solver: Solver,
    pub hessian_source: HessianSource,
}

/// Whether outer_strategy should automatically derive a downgrade ladder from
/// the primary capability, or disable retries entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Centralized degradation path chosen from the declared capability.
    Automatic,
    /// No retries; use only the primary plan.
    Disabled,
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

    match (cap.gradient, cap.hessian) {
        (Analytic, Analytic) => OuterPlan {
            solver: S::Arc,
            hessian_source: H::Analytic,
        },

        (Analytic, FiniteDifference) => OuterPlan {
            solver: S::NewtonTrustRegion,
            hessian_source: H::FiniteDifference,
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
        (Analytic, Unavailable)
            if cap.fixed_point_available && cap.all_penalty_like && cap.n_params > 8 =>
        {
            OuterPlan {
                solver: S::Efs,
                hessian_source: H::EfsFixedPoint,
            }
        }
        (Unavailable, Unavailable)
            if cap.fixed_point_available && cap.all_penalty_like && cap.n_params > 8 =>
        {
            OuterPlan {
                solver: S::Efs,
                hessian_source: H::EfsFixedPoint,
            }
        }

        // Hybrid EFS: ψ (design-moving) coords present alongside ρ coords.
        //
        // When ψ coords are present, pure EFS is invalid because B_ψ can be
        // indefinite (see response.md Section 2 for the counterexample). But
        // falling back to full BFGS wastes the cheap EFS structure for ρ coords.
        //
        // The hybrid strategy uses EFS for ρ-coords and a safeguarded
        // preconditioned gradient step for ψ-coords:
        //   Δψ = -α G⁺ g_ψ,  G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)
        //
        // This stays O(1) H⁻¹ solves per iteration (vs O(dim(θ)) for BFGS)
        // and uses the same trace Gram matrix that EFS already computes.
        (Analytic, Unavailable)
            if cap.fixed_point_available && cap.has_psi_coords && cap.n_params > 8 =>
        {
            OuterPlan {
                solver: S::HybridEfs,
                hessian_source: H::HybridEfsFixedPoint,
            }
        }
        (Unavailable, Unavailable)
            if cap.fixed_point_available && cap.has_psi_coords && cap.n_params > 8 =>
        {
            OuterPlan {
                solver: S::HybridEfs,
                hessian_source: H::HybridEfsFixedPoint,
            }
        }

        // With many params, FD Hessian is too expensive; fall back to BFGS.
        (Analytic, Unavailable) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },

        (FiniteDifference, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },

        // No declared gradient: the runner will still approximate one from
        // `eval_cost()` when BFGS is the only viable non-EFS strategy.
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
    let grad_warning = match cap.gradient {
        Derivative::FiniteDifference => " [FD gradient from cost-only objective]",
        _ => "",
    };
    let barrier_note = if cap.barrier_config.is_some() && cap.all_penalty_like && cap.n_params > 8 {
        " [EFS with runtime barrier-curvature guard]"
    } else {
        ""
    };
    let hybrid_note = if the_plan.solver == Solver::HybridEfs {
        " [hybrid EFS(ρ) + preconditioned-gradient(ψ)]"
    } else {
        ""
    };
    log::info!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {}{grad_warning}{hess_warning}{barrier_note}{hybrid_note}",
        cap.n_params,
        cap.gradient,
        cap.hessian,
        the_plan,
    );
}

fn downgrade_hessian(cap: &OuterCapability) -> Option<OuterCapability> {
    (cap.hessian == Derivative::Analytic).then(|| {
        let mut degraded = cap.clone();
        degraded.hessian = Derivative::Unavailable;
        degraded
    })
}

fn downgrade_gradient(cap: &OuterCapability) -> Option<OuterCapability> {
    (cap.gradient == Derivative::Analytic).then(|| {
        let mut degraded = cap.clone();
        degraded.gradient = Derivative::FiniteDifference;
        degraded.hessian = Derivative::Unavailable;
        degraded
    })
}

fn automatic_fallback_attempts(cap: &OuterCapability) -> Vec<OuterCapability> {
    let mut attempts = Vec::new();
    if let Some(grad_cap) = downgrade_hessian(cap) {
        attempts.push(grad_cap.clone());
        if let Some(fd_cap) = downgrade_gradient(&grad_cap) {
            attempts.push(fd_cap);
        }
    } else if let Some(fd_cap) = downgrade_gradient(cap) {
        attempts.push(fd_cap);
    }
    attempts
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

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    ///
    /// `opt` translates the non-finite objective into a recoverable trial
    /// failure so trust-region/line-search solvers retreat without the caller
    /// needing to special-case infeasible regions locally.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
        }
    }
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

    /// Convert to the optional Hessian shape used by the opt bridge.
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
///
/// For the hybrid EFS+preconditioned-gradient strategy, the steps vector
/// contains both EFS steps (for ρ coords) and preconditioned gradient steps
/// (for ψ coords). The `psi_gradient` field carries the raw ψ-block gradient
/// for optional backtracking.
#[derive(Clone, Debug)]
pub struct EfsEval {
    /// REML/LAML cost at the current rho (for convergence monitoring and
    /// comparing candidates).
    pub cost: f64,
    /// Additive steps. Length = n_rho + n_ext_coords.
    ///
    /// For pure EFS: steps for non-penalty-like coordinates are 0.0.
    /// For hybrid EFS: ρ-coords get standard EFS multiplicative steps,
    /// ψ-coords get preconditioned gradient steps `Δψ = -α G⁺ g_ψ`.
    pub steps: Vec<f64>,
    /// Current coefficient vector β̂ from the inner P-IRLS solve.
    /// Used by the EFS loop for the runtime barrier-curvature significance
    /// check when monotonicity constraints are present.
    pub beta: Option<Array1<f64>>,
    /// Raw REML/LAML gradient restricted to the ψ block (design-moving coords).
    ///
    /// Present only when the hybrid EFS strategy is active. Used by the
    /// outer iteration for backtracking on the ψ step: if the combined
    /// (ρ-EFS, ψ-gradient) step does not decrease V(θ), the ψ step size
    /// α is halved while keeping the ρ-EFS step fixed.
    ///
    /// This avoids re-evaluating the gradient during backtracking since
    /// the gradient was already computed as part of the hybrid EFS eval.
    pub psi_gradient: Option<Array1<f64>>,
    /// Indices into the full θ vector that correspond to ψ (design-moving)
    /// coordinates. Used by the backtracking logic to selectively scale
    /// only the ψ portion of the step.
    pub psi_indices: Option<Vec<usize>>,
}

/// Common interface for outer smoothing-parameter objectives.
///
/// Every model path that optimizes smoothing parameters implements this trait.
/// The runner function consumes it and handles solver selection,
/// multi-start, and logging while delegating derivative fallback policy to
/// `opt`.
///
/// # Contract
///
/// - `capability()` must be stable (same result across calls).
/// - `eval()` may return `HessianResult::Unavailable` at individual trial
///   points even when `capability().hessian == Analytic`; `opt` degrades that
///   step to an FD Hessian instead of requiring the objective to fake a stale
///   or non-finite Hessian.
/// - Use `eval_cost()` / `OuterEval::infeasible()` for infeasible trial points.
///   Return `Err(...)` for genuine evaluation breakdowns so the runner can mark
///   the step as a recoverable solver failure and escalate to the next declared
///   fallback plan if the full attempt still fails.
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
    fn eval_efs(&mut self, _: &Array1<f64>) -> Result<EfsEval, EstimationError> {
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
pub struct ClosureObjective<
    S,
    Fc,
    Fe,
    Fr = fn(&mut S),
    Fefs = fn(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
> {
    pub state: S,
    pub cap: OuterCapability,
    pub cost_fn: Fc,
    pub eval_fn: Fe,
    /// Optional reset closure. When `None`, `reset()` is a no-op.
    pub reset_fn: Option<Fr>,
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
        if let Some(f) = self.reset_fn.as_mut() {
            f(&mut self.state);
        }
    }
}

fn into_objective_error(context: &str, err: EstimationError) -> ObjectiveEvalError {
    ObjectiveEvalError::recoverable(format!("{context}: {err}"))
}

struct OuterCostBridge<'a> {
    obj: &'a mut dyn OuterObjective,
}

impl ZerothOrderObjective for OuterCostBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))
    }
}

struct OuterFirstOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
}

impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))
    }
}

impl FirstOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        let eval = self
            .obj
            .eval(x)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

struct OuterSecondOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    hessian_source: HessianSource,
}

impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))
    }
}

impl FirstOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        let eval = self
            .obj
            .eval(x)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl SecondOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        let eval = self
            .obj
            .eval(x)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let hessian = match self.hessian_source {
            HessianSource::Analytic => eval.hessian.into_option(),
            HessianSource::FiniteDifference
            | HessianSource::BfgsApprox
            | HessianSource::EfsFixedPoint
            | HessianSource::HybridEfsFixedPoint => None,
        };
        Ok(SecondOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian,
        })
    }
}

struct OuterFixedPointBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    barrier_config: Option<BarrierConfig>,
}

/// Maximum number of backtracking halvings for the ψ block in the hybrid
/// EFS+preconditioned-gradient iteration.
///
/// If after this many halvings the combined (ρ-EFS, ψ-gradient) step still
/// doesn't decrease V(θ), the ψ step is zeroed out and only the ρ-EFS step
/// is applied. This preserves the EFS convergence guarantee for ρ coords
/// even when the ψ step is too aggressive.
const MAX_PSI_BACKTRACK: usize = 8;

impl FixedPointObjective for OuterFixedPointBridge<'_> {
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        let eval = self
            .obj
            .eval_efs(x)
            .map_err(|err| into_objective_error("outer EFS eval failed", err))?;
        let status = if let Some(ref barrier_cfg) = self.barrier_config {
            if let Some(ref beta) = eval.beta {
                let ref_diag = 1.0;
                let threshold = 0.01;
                if barrier_cfg.barrier_curvature_is_significant(beta, ref_diag, threshold) {
                    FixedPointStatus::Stop
                } else {
                    FixedPointStatus::Continue
                }
            } else {
                FixedPointStatus::Continue
            }
        } else {
            FixedPointStatus::Continue
        };

        let step = if matches!(status, FixedPointStatus::Stop) {
            Array1::zeros(x.len())
        } else if eval.psi_indices.is_some() && eval.psi_gradient.is_some() {
            // ── Hybrid EFS+preconditioned-gradient path ──
            //
            // The step vector contains EFS steps for ρ/τ coordinates and
            // preconditioned gradient steps for ψ coordinates. We perform
            // backtracking on the ψ block to ensure the combined step
            // decreases V(θ).
            //
            // Backtracking strategy:
            // 1. Try the full combined step.
            // 2. If it doesn't decrease V(θ), halve only the ψ portion.
            // 3. Repeat up to MAX_PSI_BACKTRACK times.
            // 4. If still no decrease, zero out the ψ step entirely.
            //
            // This preserves the EFS convergence guarantee for ρ coords:
            // the ρ-EFS step is always applied in full, and only the ψ
            // portion is adjusted. The ρ-EFS step has its own monotonicity
            // guarantee from the Wood-Fasiolo theorem (valid because ρ
            // coords satisfy the PSD + fixed-nullspace structural property).
            let psi_indices = eval.psi_indices.as_ref().unwrap();
            let current_cost = eval.cost;
            let mut combined_step = Array1::from_vec(eval.steps);

            // Save the original ψ step magnitudes for halving.
            let original_psi_steps: Vec<f64> =
                psi_indices.iter().map(|&i| combined_step[i]).collect();

            let mut accepted = false;
            for bt in 0..MAX_PSI_BACKTRACK {
                // Evaluate cost at the trial point.
                let trial = x + &combined_step;
                match self.obj.eval_cost(&trial) {
                    Ok(trial_cost) if trial_cost < current_cost => {
                        // Step accepted — the combined step decreases V(θ).
                        if bt > 0 {
                            log::debug!(
                                "[HYBRID-EFS] ψ backtrack accepted after {bt} halvings \
                                 (cost: {current_cost:.6} → {trial_cost:.6})"
                            );
                        }
                        accepted = true;
                        break;
                    }
                    Ok(trial_cost) => {
                        // Step rejected — halve the ψ portion.
                        log::debug!(
                            "[HYBRID-EFS] ψ backtrack {bt}: trial cost {trial_cost:.6} >= \
                             current {current_cost:.6}, halving ψ step"
                        );
                        for (j, &i) in psi_indices.iter().enumerate() {
                            let halved = original_psi_steps[j] * 0.5_f64.powi((bt + 2) as i32);
                            combined_step[i] = halved;
                        }
                    }
                    Err(_) => {
                        // Evaluation failed — halve ψ step and retry.
                        log::debug!(
                            "[HYBRID-EFS] ψ backtrack {bt}: trial eval failed, halving ψ step"
                        );
                        for (j, &i) in psi_indices.iter().enumerate() {
                            let halved = original_psi_steps[j] * 0.5_f64.powi((bt + 2) as i32);
                            combined_step[i] = halved;
                        }
                    }
                }
            }

            if !accepted {
                // All backtracking attempts exhausted. Zero out the ψ step
                // and rely solely on the ρ-EFS step for this iteration.
                log::info!(
                    "[HYBRID-EFS] ψ backtrack exhausted ({MAX_PSI_BACKTRACK} halvings). \
                     Zeroing ψ step; applying ρ-EFS step only."
                );
                for &i in psi_indices {
                    combined_step[i] = 0.0;
                }
            }

            combined_step
        } else {
            // Pure EFS path: no ψ coordinates, no backtracking needed.
            Array1::from_vec(eval.steps)
        };

        Ok(FixedPointSample {
            value: eval.cost,
            step,
            status,
        })
    }
}

fn solution_into_outer_result(
    solution: Solution,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let final_grad_norm = solution
        .final_gradient_norm
        .or(solution.final_step_norm)
        .unwrap_or(0.0);
    OuterResult {
        rho: solution.final_point,
        final_value: solution.final_value,
        iterations: solution.iterations,
        final_grad_norm,
        final_gradient: solution.final_gradient,
        final_hessian: solution.final_hessian,
        converged,
        plan_used,
    }
}

impl OuterResult {
    /// Convert to the smoothing summary type consumed by downstream estimate
    /// code.
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

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
pub struct OuterConfig {
    /// Optimizer convergence tolerance.
    pub tolerance: f64,
    /// Maximum outer iterations per seed candidate.
    pub max_iter: usize,
    /// Finite-difference step size for FD Hessian construction.
    pub fd_step: f64,
    /// Optional per-coordinate lower/upper bounds for the outer coordinates.
    /// When absent, the runner uses symmetric `[-rho_bound, +rho_bound]`.
    pub bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Seed generation and screening configuration.
    pub seed_config: crate::seeding::SeedConfig,
    /// Bounds on rho coordinates (applied symmetrically as [-bound, +bound]).
    pub rho_bound: f64,
    /// Heuristic initial lambdas for seed generation (optional).
    pub heuristic_lambdas: Option<Vec<f64>>,
    /// If provided, use this as the sole starting point (skip seed generation
    /// and screening). Useful when the caller already has a good initial rho.
    pub initial_rho: Option<Array1<f64>>,
    /// Centralized retry policy for degraded outer plans.
    pub fallback_policy: FallbackPolicy,
    /// Shared atomic for screening iteration cap. When present, the screening
    /// loop sets this to `screen_max_inner_iterations` before eval_cost calls
    /// and resets to 0 afterward. The inner solver reads it atomically.
    pub screening_cap: Option<Arc<AtomicUsize>>,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            max_iter: 200,
            fd_step: 1e-4,
            bounds: None,
            seed_config: crate::seeding::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
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
    /// Final gradient when the solver is gradient-based.
    pub final_gradient: Option<Array1<f64>>,
    /// Final Hessian when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
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
/// 6. If the configured fallback policy allows it, re-plans with degraded
///    capabilities chosen centrally inside outer_strategy and retries.
/// 7. Returns the best result (including which plan was actually used).
///
/// Do not wrap `run_outer` calls in try/catch with ad-hoc solver recovery.
/// Callers should declare only the primary capability and, at most, whether
/// automatic fallback is enabled at all.
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
            final_gradient: None,
            final_hessian: None,
            converged: true,
            plan_used: the_plan,
        });
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // any centrally-derived degraded capabilities.
    let fallback_attempts = match config.fallback_policy {
        FallbackPolicy::Automatic => automatic_fallback_attempts(&cap),
        FallbackPolicy::Disabled => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
        if attempt_idx > 0 {
            log::info!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
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
        EstimationError::RemlOptimizationFailed(format!("all plan attempts exhausted ({context})"))
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
    // Use cheap partial-PIRLS (capped inner iterations) when available.
    let budget = config.seed_config.screening_budget.max(1);
    let screen_max = config.seed_config.screen_max_inner_iterations;
    let screened = if seeds.len() <= budget {
        seeds
    } else {
        log::info!(
            "[OUTER] {context}: screening {}/{} seeds (budget={budget}, pirls_cap={screen_max})",
            seeds.len(),
            seeds.len(),
        );
        let screen_start = std::time::Instant::now();
        struct ScreeningCapGuard<'a> {
            cap: Option<&'a Arc<AtomicUsize>>,
        }
        impl Drop for ScreeningCapGuard<'_> {
            fn drop(&mut self) {
                if let Some(cap) = self.cap {
                    cap.store(0, Ordering::Relaxed);
                }
            }
        }
        let screening_cap_guard = if screen_max > 0 {
            if let Some(cap) = config.screening_cap.as_ref() {
                cap.store(screen_max, Ordering::Relaxed);
                Some(ScreeningCapGuard { cap: Some(cap) })
            } else {
                None
            }
        } else {
            None
        };
        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(seeds.len());
        for (i, rho) in seeds.iter().enumerate() {
            obj.reset();
            let cost = obj.eval_cost(rho).unwrap_or(f64::INFINITY);
            scored.push((
                i,
                if cost.is_finite() {
                    cost
                } else {
                    f64::INFINITY
                },
            ));
        }
        drop(screening_cap_guard);
        // Reset after screening so cached partial-iteration results don't
        // leak into the real optimizer.
        obj.reset();
        log::info!(
            "[OUTER] {context}: screening done in {:.3}s",
            screen_start.elapsed().as_secs_f64(),
        );
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(budget);
        scored.iter().map(|&(i, _)| seeds[i].clone()).collect()
    };

    let (lower, upper) = config.bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(cap.n_params, -config.rho_bound),
            Array1::<f64>::from_elem(cap.n_params, config.rho_bound),
        )
    });
    let bounds_template = (lower, upper);

    let mut best: Option<OuterResult> = None;

    for seed in &screened {
        obj.reset();

        let result: Result<OuterResult, EstimationError> = match the_plan.solver {
            Solver::Arc | Solver::NewtonTrustRegion => {
                let hessian_source = the_plan.hessian_source;
                let objective = OuterSecondOrderBridge {
                    obj,
                    hessian_source,
                };

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
                        .with_max_iterations(max_iter)
                        .with_fd_hessian_step(config.fd_step);
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Arc solver failed: {e:?}"
                        ))),
                    }
                } else {
                    let mut optimizer = NewtonTrustRegion::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter)
                        .with_fd_hessian_step(config.fd_step);
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(NewtonTrustRegionError::MaxIterationsReached { last_solution }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Newton trust-region solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Bfgs => {
                let gradient_available = cap.gradient == Derivative::Analytic;
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                if gradient_available {
                    let objective = OuterFirstOrderBridge { obj };
                    let mut optimizer = Bfgs::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter);
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(BfgsError::MaxIterationsReached { last_solution }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "BFGS solver failed: {e:?}"
                        ))),
                    }
                } else {
                    let objective =
                        FiniteDiffGradient::new(OuterCostBridge { obj }).with_step(config.fd_step);
                    let mut optimizer = Bfgs::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter);
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(BfgsError::MaxIterationsReached { last_solution }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "BFGS solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Efs => {
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                let objective = OuterFixedPointBridge {
                    obj,
                    barrier_config: cap.barrier_config.clone(),
                };
                let mut optimizer = FixedPoint::new(seed.clone(), objective)
                    .with_bounds(bounds)
                    .with_tolerance(tol)
                    .with_max_iterations(max_iter);
                match optimizer.run() {
                    Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                    Err(FixedPointError::MaxIterationsReached { last_solution }) => {
                        Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "fixed-point solver failed: {e:?}"
                    ))),
                }
            }
            // ── Hybrid EFS + preconditioned gradient for ψ coords ──
            //
            // This solver combines EFS multiplicative steps for ρ (penalty-like)
            // coordinates with safeguarded preconditioned gradient steps for ψ
            // (design-moving) coordinates.
            //
            // The hybrid is needed because EFS is mathematically invalid for
            // indefinite B_ψ: Wood-Fasiolo's ascent proof requires
            // H^{-1/2} B_d H^{-1/2} ≽ 0 with parameter-independent nullspace,
            // which holds for penalty-like coords but fails for design-moving
            // coords. A concrete counterexample (response.md Section 2) shows
            // that any fixed-point map using only {a_d, tr(H⁻¹B_d),
            // tr(H⁻¹B_dH⁻¹B_e), v_d} can be made ascent or descent by varying
            // the local curvature c, so no universal convergence guarantee exists.
            //
            // The preconditioned gradient Δψ = -α G⁺ g_ψ uses the same trace
            // Gram matrix G_{de} = tr(H⁻¹ B_d H⁻¹ B_e) that EFS computes,
            // staying O(1) H⁻¹ solves per iteration (vs O(dim(θ)) for BFGS).
            //
            // Backtracking: if the combined step doesn't decrease V(θ), the ψ
            // step size α is halved (up to MAX_PSI_BACKTRACK times) while the
            // ρ-EFS step is kept fixed.
            Solver::HybridEfs => {
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                let objective = OuterFixedPointBridge {
                    obj,
                    barrier_config: cap.barrier_config.clone(),
                };
                let mut optimizer = FixedPoint::new(seed.clone(), objective)
                    .with_bounds(bounds)
                    .with_tolerance(tol)
                    .with_max_iterations(max_iter);
                match optimizer.run() {
                    Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                    Err(FixedPointError::MaxIterationsReached { last_solution }) => {
                        Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "hybrid EFS solver failed: {e:?}"
                    ))),
                }
            }
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
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_fd_hessian_selects_newton() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::FiniteDifference,
            n_params: 3,
            all_penalty_like: false,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::NewtonTrustRegion);
        assert_eq!(p.hessian_source, HessianSource::FiniteDifference);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 3,
            all_penalty_like: false,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_fd_gradient_always_bfgs() {
        for n in [1, 5, 20] {
            let cap = OuterCapability {
                gradient: Derivative::FiniteDifference,
                hessian: Derivative::Unavailable,
                n_params: n,
                all_penalty_like: false,
                has_psi_coords: false,
                fixed_point_available: false,
                barrier_config: None,
            };
            let p = plan(&cap);
            assert_eq!(p.solver, Solver::Bfgs);
            assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
        }
    }

    #[test]
    fn plan_fd_gradient_with_hessian_still_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::FiniteDifference,
            hessian: Derivative::Analytic,
            n_params: 3,
            all_penalty_like: false,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_penalty_like_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: true,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_few_params_even_if_penalty_like() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 5,
            all_penalty_like: true,
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: None,
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
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: Some(barrier),
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
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: Some(barrier),
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
        result.unwrap_analytic();
    }

    #[test]
    fn zero_params_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 0,
            all_penalty_like: false,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
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
                has_psi_coords: false,
                fixed_point_available: false,
                barrier_config: None,
            },
            cost_fn: |_: &mut i32, _: &Array1<f64>| Ok(1.0),
            eval_fn: |_: &mut i32, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                })
            },
            reset_fn: Some(|st: &mut i32| {
                *st = 42;
            }),
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

    #[test]
    fn plan_hybrid_efs_selected_for_psi_coords_many_params() {
        // When ψ (design-moving) coords are present and n_params > 8,
        // the planner should select HybridEfs instead of falling back to BFGS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    #[test]
    fn plan_psi_without_fixed_point_stays_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: false,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_hybrid_efs_no_gradient_selected_for_psi_coords() {
        // Even without analytic gradient, hybrid EFS works because the
        // gradient is computed internally by the unified evaluator.
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_few_params() {
        // With few params, FD Hessian is preferred over hybrid EFS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 5,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        // Few params + analytic gradient → FD Newton is better.
        assert_eq!(p.solver, Solver::NewtonTrustRegion);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_with_analytic_hessian() {
        // Arc is always preferred when analytic Hessian is available,
        // even with ψ coordinates.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 20,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_pure_efs_not_hybrid_when_all_penalty_like() {
        // When all coords are penalty-like (no ψ), pure EFS is selected
        // even if has_psi_coords is false.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            all_penalty_like: true,
            has_psi_coords: false,
            fixed_point_available: true,
            barrier_config: None,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn automatic_fallbacks_degrade_hessian_then_gradient() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 12,
            all_penalty_like: false,
            has_psi_coords: false,
            fixed_point_available: false,
            barrier_config: None,
        };
        let attempts = automatic_fallback_attempts(&cap);
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert_eq!(attempts[0].hessian, Derivative::Unavailable);
        assert_eq!(attempts[1].gradient, Derivative::FiniteDifference);
        assert_eq!(attempts[1].hessian, Derivative::Unavailable);
    }
}
