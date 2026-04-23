//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, link-wiggle,
//! GAMLSS custom family, spatial kappa, etc.) declares its derivative
//! capability here and receives an [`OuterPlan`] that determines which solver
//! and Hessian source to use.
//!
//! # Design invariant
//!
//! The planner never selects a finite-difference Hessian policy. If a path
//! cannot provide an analytic Hessian, that fact is visible in its
//! [`OuterCapability`] declaration and in the resulting [`OuterPlan`], which
//! falls back to BFGS or an EFS variant instead of synthesizing second-order
//! curvature numerically.

use crate::estimate::EstimationError;
use crate::solver::estimate::reml::unified::BarrierConfig;
use ::opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, FirstOrderObjective, FirstOrderSample,
    FixedPoint, FixedPointError, FixedPointObjective, FixedPointSample, FixedPointStatus,
    MaxIterations, ObjectiveEvalError, SecondOrderObjective, SecondOrderSample, Solution,
    Tolerance, ZerothOrderObjective,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Matrix-free outer Hessian operator.
///
/// This is the exact outer Hessian action `H_outer * v` evaluated at the
/// current outer point, without requiring dense materialization.
pub trait OuterHessianOperator: Send + Sync {
    fn dim(&self) -> usize;
    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String>;

    fn materialize_dense(&self) -> Result<Array2<f64>, String> {
        let dim = self.dim();
        let mut dense = Array2::<f64>::zeros((dim, dim));
        let mut basis = Array1::<f64>::zeros(dim);
        for col in 0..dim {
            basis[col] = 1.0;
            let hv = self.matvec(&basis)?;
            basis[col] = 0.0;
            if hv.len() != dim {
                return Err(format!(
                    "outer Hessian operator matvec length mismatch: got {}, expected {}",
                    hv.len(),
                    dim
                ));
            }
            dense.column_mut(col).assign(&hv);
        }
        for row in 0..dim {
            for col in (row + 1)..dim {
                let sym = 0.5 * (dense[[row, col]] + dense[[col, row]]);
                dense[[row, col]] = sym;
                dense[[col, row]] = sym;
            }
        }
        Ok(dense)
    }
}

struct RhoBlockAdditiveOuterHessian {
    base: Arc<dyn OuterHessianOperator>,
    rho_block: Array2<f64>,
    dim: usize,
}

impl OuterHessianOperator for RhoBlockAdditiveOuterHessian {
    fn dim(&self) -> usize {
        self.dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.dim {
            return Err(format!(
                "outer Hessian operator input length mismatch: got {}, expected {}",
                v.len(),
                self.dim
            ));
        }
        let mut out = self.base.matvec(v)?;
        let k = self.rho_block.nrows();
        if k > 0 {
            let rho_v = v.slice(ndarray::s![..k]).to_owned();
            let rho_out = self.rho_block.dot(&rho_v);
            out.slice_mut(ndarray::s![..k]).scaled_add(1.0, &rho_out);
        }
        Ok(out)
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, String> {
        let mut dense = self.base.materialize_dense()?;
        let k = self.rho_block.nrows();
        if k > dense.nrows() || k > dense.ncols() {
            return Err(format!(
                "rho-block Hessian update shape mismatch: got {}x{}, dense operator is {}x{}",
                self.rho_block.nrows(),
                self.rho_block.ncols(),
                dense.nrows(),
                dense.ncols()
            ));
        }
        dense
            .slice_mut(ndarray::s![..k, ..k])
            .scaled_add(1.0, &self.rho_block);
        Ok(dense)
    }
}

/// Whether an analytic derivative is available for a given order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// Exact analytic derivative implemented and available.
    Analytic,
    /// Derivative is available only via finite differences computed inside
    /// `opt` from a lower-order objective interface. This is a real capability
    /// for gradients; Hessians are normalized to `Unavailable` at planning time
    /// because finite-differenced Hessians are a last resort and should not be
    /// selected ahead of a gradient-only solver.
    FiniteDifference,
    /// No analytic derivative; must be approximated or skipped.
    Unavailable,
}

/// Declares what a specific model path can provide to the outer optimizer.
///
/// Each call site that optimizes smoothing parameters constructs one of these
/// to describe its analytic derivative coverage. The [`plan`] function then
/// selects the optimizer and Hessian strategy.
const SMALL_OUTER_BFGS_MAX_PARAMS: usize = 8;
const SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterThetaLayout {
    pub n_params: usize,
    pub psi_dim: usize,
}

impl OuterThetaLayout {
    pub fn new(n_params: usize, psi_dim: usize) -> Self {
        Self { n_params, psi_dim }
    }

    pub fn rho_dim(&self) -> usize {
        self.n_params.saturating_sub(self.psi_dim)
    }

    fn validate_capability(&self, context: &str) -> Result<(), EstimationError> {
        if self.psi_dim > self.n_params {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "{context}: invalid outer theta layout (psi_dim={} exceeds n_params={})",
                self.psi_dim, self.n_params
            )));
        }
        Ok(())
    }

    fn validate_point_len(
        &self,
        theta: &Array1<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if theta.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer theta length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                theta.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_gradient_len(
        &self,
        gradient: &Array1<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if gradient.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer gradient length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                gradient.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_hessian_shape(
        &self,
        hessian: &Array2<f64>,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if hessian.nrows() != self.n_params || hessian.ncols() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer Hessian shape mismatch: got {}x{}, expected {}x{} (rho_dim={}, psi_dim={})",
                hessian.nrows(),
                hessian.ncols(),
                self.n_params,
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        Ok(())
    }

    fn validate_efs_eval(&self, eval: &EfsEval, context: &str) -> Result<(), ObjectiveEvalError> {
        if eval.steps.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer EFS step length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                eval.steps.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        if let Some(ref psi_gradient) = eval.psi_gradient {
            if psi_gradient.len() != self.psi_dim {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer EFS psi-gradient length mismatch: got {}, expected {}",
                    psi_gradient.len(),
                    self.psi_dim
                )));
            }
        }
        if let Some(ref psi_indices) = eval.psi_indices {
            if psi_indices.len() != self.psi_dim {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer EFS psi-index count mismatch: got {}, expected {}",
                    psi_indices.len(),
                    self.psi_dim
                )));
            }
            if psi_indices.iter().any(|&idx| idx >= self.n_params) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer EFS psi index out of range for n_params={}",
                    self.n_params
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct OuterCapability {
    pub gradient: Derivative,
    pub hessian: Derivative,
    /// Number of smoothing (+ any auxiliary hyper-) parameters being optimized.
    pub n_params: usize,
    /// Number of ψ (design-moving) coordinates among the extended
    /// hyperparameter coordinates. When 0, all coords are penalty-like and
    /// pure EFS is eligible (given `fixed_point_available`). When > 0,
    /// hybrid EFS is eligible instead: EFS for ρ + preconditioned gradient
    /// for ψ.
    ///
    /// # Hybrid EFS strategy (when `psi_dim > 0`)
    ///
    /// Enabled when `psi_dim > 0`,
    /// `n_params > SMALL_OUTER_BFGS_MAX_PARAMS`, and
    /// `fixed_point_available`.
    /// Combines:
    /// - Standard EFS multiplicative fixed-point updates for ρ coordinates
    /// - Safeguarded preconditioned gradient steps for ψ coordinates:
    ///   `Δψ = -α G⁺ g_ψ` where G is the trace Gram matrix
    ///
    /// Mathematically necessary because no EFS-type fixed-point iteration
    /// exists for indefinite B_ψ (see response.md Section 2). The structural
    /// requirement for EFS is `H^{-1/2} B_d H^{-1/2} ≽ 0` (PSD) plus fixed
    /// nullspace — exactly what penalty-like coords satisfy and design-moving
    /// coords do not.
    ///
    /// The hybrid is O(1) H⁻¹ solves per iteration (same as pure EFS),
    /// compared to O(dim(θ)) for BFGS.
    pub psi_dim: usize,
    /// Whether the objective actually implements `eval_efs()` for fixed-point
    /// plans. Structural eligibility (`psi_dim == 0` / `psi_dim > 0`)
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
    /// Policy hint: even when an analytic Hessian exists, prefer a gradient-only
    /// outer solver because each second-order evaluation is too expensive.
    pub prefer_gradient_only: bool,
    /// Policy hint: even when the objective implements `eval_efs()` and the
    /// coordinate structure is penalty-like, the planner must NOT select
    /// EFS/HybridEfs for this problem.
    ///
    /// Set by the caller for problem classes where the Wood-Fasiolo structural
    /// property (`H^{-1/2} B_k H^{-1/2} ≽ 0` plus parameter-independent
    /// nullspace) is known not to hold — e.g. GAMLSS/location-scale families
    /// where the joint Hessian is β-dependent and cross-block smoothers
    /// induce non-diagonal curvature that the EFS multiplicative fixed-point
    /// cannot resolve. Also set by the automatic fallback cascade when an
    /// EFS/HybridEfs attempt failed to converge, so the next attempt falls
    /// back to analytic-gradient BFGS rather than retrying EFS.
    pub disable_fixed_point: bool,
}

impl OuterCapability {
    pub fn theta_layout(&self) -> OuterThetaLayout {
        OuterThetaLayout::new(self.n_params, self.psi_dim)
    }

    pub fn validate_layout(&self, context: &str) -> Result<(), EstimationError> {
        self.theta_layout().validate_capability(context)
    }

    /// True when all coordinates are penalty-like (no ψ coords).
    pub fn all_penalty_like(&self) -> bool {
        self.psi_dim == 0
    }
    /// True when ψ (design-moving) coordinates are present.
    pub fn has_psi_coords(&self) -> bool {
        self.psi_dim > 0
    }

    fn efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.all_penalty_like()
            && self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS
    }

    fn hybrid_efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.has_psi_coords()
            && self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS
    }

    fn declared_hessian_for_planning(&self) -> Derivative {
        match self.hessian {
            Derivative::Analytic => Derivative::Analytic,
            Derivative::FiniteDifference | Derivative::Unavailable => Derivative::Unavailable,
        }
    }
}

/// Which solver algorithm to use for the outer optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Adaptive Regularized Cubic; fastest convergence, requires Hessian.
    Arc,
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
    /// Opportunistic coordinate compass search (positive basis {±e_i} with
    /// step contraction). Derivative-free by construction — no gradient
    /// and no finite differences — so categorically distinct from the
    /// deprecated FD-BFGS degradation rung.
    ///
    /// Reserved for genuinely-derivative-free auxiliary searches
    /// (baseline-theta for parametric survival baselines, SAS/BetaLogistic
    /// /Mixture inverse-link parameters) where no analytic
    /// ∂cost/∂θ is available and the dimension is small (≤ ~5).
    ///
    /// The planner only selects this variant when the caller has opted in
    /// via [`SolverClass::AuxiliaryGradientFree`]; it is NEVER selected
    /// for the main REML outer. This gate is explicit to prevent the
    /// silent FD-grad masking failure mode that `automatic_fallback_attempts`
    /// was tightened to avoid: for the big REML outer, declared-analytic
    /// gradients must converge on their own merits.
    ///
    /// Convergence to a stationary point on any continuously-differentiable
    /// cost bounded below on a compact box follows from
    /// Kolda-Lewis-Torczon, SIAM Review 45:385, 2003, Thm 3.3. The theorem
    /// requires that all 2·dim basis directions are polled before step
    /// contraction; the dispatcher's sweep loop satisfies this by the
    /// `!improved ⇒ step /= 2` branch.
    CompassSearch,
}

/// Declares which "class" of outer optimization the caller is doing.
///
/// The default `Primary` class applies to the main REML outer — the
/// canonical smoothing-parameter optimization — and has access to
/// Arc/Bfgs/Efs/HybridEfs according to declared derivatives.
///
/// `AuxiliaryGradientFree` unlocks `Solver::CompassSearch` for small-dim
/// auxiliary pre-optimizations where no analytic ∂cost/∂θ exists (survival
/// baseline theta, non-standard inverse-link parameters). The planner gates
/// selection of CompassSearch strictly on this flag; REML builders never
/// set it, so REML can never be routed to compass search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SolverClass {
    /// The main REML outer — smoothing parameters and ψ-coords where
    /// analytic gradient (and typically analytic Hessian) is the contract.
    #[default]
    Primary,
    /// A genuinely-derivative-free low-dim auxiliary search (e.g. survival
    /// baseline theta). Opts into `Solver::CompassSearch` when gradient is
    /// Unavailable. Must not be set by REML builders.
    AuxiliaryGradientFree,
}

#[inline]
fn effective_seed_budget(
    requested_budget: usize,
    solver: Solver,
    risk_profile: crate::seeding::SeedRiskProfile,
    screening_enabled: bool,
) -> usize {
    let requested_budget = requested_budget.max(1);
    let capped = match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) if screening_enabled => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => 2,
        // Aux direct-search is a single-start low-dim local method; restarting
        // from another seed would just re-explore the same basin.
        (Solver::CompassSearch, _) => 1,
        _ => requested_budget,
    };
    requested_budget.min(capped)
}

#[inline]
fn should_screen_seeds(
    config: &OuterConfig,
    solver: Solver,
    generated_seed_count: usize,
    seed_budget: usize,
) -> bool {
    config.screening_cap.is_some()
        && generated_seed_count > seed_budget
        && matches!(solver, Solver::Arc | Solver::Efs | Solver::HybridEfs)
}

#[inline]
fn expensive_unsuccessful_seed_limit(
    solver: Solver,
    risk_profile: crate::seeding::SeedRiskProfile,
) -> Option<usize> {
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => Some(2),
        (Solver::CompassSearch, _) => Some(1),
        _ => None,
    }
}

fn rank_seeds_with_screening(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    seeds: &[Array1<f64>],
) -> Vec<Array1<f64>> {
    let Some(screening_cap) = config.screening_cap.as_ref() else {
        return seeds.to_vec();
    };

    let previous_cap = screening_cap.swap(
        config.seed_config.screen_max_inner_iterations.max(1),
        Ordering::Relaxed,
    );
    let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(seeds.len());
    let mut rejected = 0usize;

    for (idx, seed) in seeds.iter().enumerate() {
        obj.reset();
        match obj.eval_cost(seed) {
            Ok(cost) if cost.is_finite() => ranked.push((idx, cost)),
            Ok(_) | Err(_) => rejected += 1,
        }
    }

    screening_cap.store(previous_cap, Ordering::Relaxed);
    obj.reset();

    if ranked.is_empty() {
        log::info!(
            "[OUTER] {context}: seed screening produced no finite cheap costs; keeping heuristic order"
        );
        return seeds.to_vec();
    }

    ranked.sort_by(|(idx_a, cost_a), (idx_b, cost_b)| {
        cost_a.total_cmp(cost_b).then_with(|| idx_a.cmp(idx_b))
    });

    let mut ordered = Vec::with_capacity(seeds.len());
    let mut seen = vec![false; seeds.len()];
    for (idx, _) in ranked {
        seen[idx] = true;
        ordered.push(seeds[idx].clone());
    }
    for (idx, seed) in seeds.iter().enumerate() {
        if !seen[idx] {
            ordered.push(seed.clone());
        }
    }

    log::info!(
        "[OUTER] {context}: seed screening ranked {}/{} candidates with capped eval_cost \
         (screen_max_inner_iterations={}); rejected={}",
        ordered.len() - rejected,
        seeds.len(),
        config.seed_config.screen_max_inner_iterations.max(1),
        rejected,
    );

    ordered
}

#[inline]
fn candidate_improves_best(candidate: &OuterResult, best: Option<&OuterResult>) -> bool {
    match best {
        None => true,
        Some(best) if candidate.converged != best.converged => candidate.converged,
        Some(best) => candidate.final_value < best.final_value,
    }
}

/// How the Hessian will be obtained for the outer optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianSource {
    /// Exact analytic Hessian provided by the objective.
    Analytic,
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

/// Requested derivative order for an outer objective evaluation.
///
/// Cost-only paths continue to use [`OuterObjective::eval_cost`]. This enum is
/// for the shared `eval` bridge where the runner needs either first-order or
/// second-order information depending on the active plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterEvalOrder {
    /// Compute value and gradient only.
    ValueAndGradient,
    /// Compute value, gradient, and analytic Hessian when available.
    ValueGradientHessian,
}

/// The outer optimization plan. Produced by [`plan`], consumed by the runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterPlan {
    pub solver: Solver,
    pub hessian_source: HessianSource,
}

pub(crate) const EFS_FIRST_ORDER_FALLBACK_MARKER: &str = "[outer-efs-first-order-fallback]";

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

    match (cap.gradient, cap.declared_hessian_for_planning()) {
        (Analytic, Analytic) => OuterPlan {
            solver: if cap.prefer_gradient_only {
                S::Bfgs
            } else {
                S::Arc
            },
            hessian_source: if cap.prefer_gradient_only {
                H::BfgsApprox
            } else {
                H::Analytic
            },
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
        (Analytic, Unavailable) if cap.efs_plan_eligible() => OuterPlan {
            solver: S::Efs,
            hessian_source: H::EfsFixedPoint,
        },
        (Unavailable, Unavailable) if cap.efs_plan_eligible() => OuterPlan {
            solver: S::Efs,
            hessian_source: H::EfsFixedPoint,
        },

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
        (Analytic, Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },
        (Unavailable, Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },

        // Gradient-only problems should use a gradient-only optimizer.
        // Finite-differencing a Hessian multiplies objective cost by ~2p per
        // outer step and is a bad default for expensive likelihoods.
        (Analytic, Unavailable) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
        // FD-gradient planning is a production footgun: the runner now rejects
        // non-analytic gradients in `Solver::Bfgs`, but we still emit a plan
        // here so the error surfaces with context rather than as a panic on an
        // unmatched arm. The caller must supply an analytic gradient or use an
        // EFS-eligible capability.
        (Analytic, FiniteDifference) | (FiniteDifference, _) | (Unavailable, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
    }
}

/// Plan selection with an explicit [`SolverClass`] opt-in.
///
/// For `SolverClass::Primary` this is identical to [`plan`] — the main REML
/// outer dispatch never changes behavior.
///
/// For `SolverClass::AuxiliaryGradientFree` with no declared gradient or
/// Hessian capability, returns a `Solver::CompassSearch` plan. This is the
/// sole path by which compass search can be dispatched; the primary REML
/// builder never sets the aux class, so the direct-search variant cannot
/// leak into the big REML outer or the automatic fallback cascade.
///
/// If the aux class is set but analytic gradient IS available, that is a
/// caller error (the caller should have used `Primary` and let Arc/Bfgs
/// handle it); we defer to the standard `plan` in that case so the caller
/// still gets a well-formed plan rather than a silent mis-dispatch.
pub fn plan_with_class(cap: &OuterCapability, class: SolverClass) -> OuterPlan {
    use Derivative::*;
    if class == SolverClass::AuxiliaryGradientFree
        && cap.gradient == Unavailable
        && cap.declared_hessian_for_planning() == Unavailable
        && !cap.efs_plan_eligible()
        && !cap.hybrid_efs_plan_eligible()
    {
        return OuterPlan {
            solver: Solver::CompassSearch,
            hessian_source: HessianSource::BfgsApprox,
        };
    }
    plan(cap)
}

/// Log the outer optimization plan. Called once per fit at the start of
/// outer optimization so the user can see what strategy was selected and why.
pub fn log_plan(context: &str, cap: &OuterCapability, the_plan: &OuterPlan) {
    let hess_warning = match the_plan.hessian_source {
        HessianSource::BfgsApprox if cap.n_params > 0 => {
            " [no Hessian: BFGS approximation]".to_string()
        }
        _ => String::new(),
    };
    let grad_warning = match cap.gradient {
        Derivative::FiniteDifference => " [FD gradient from cost-only objective]",
        _ => "",
    };
    let barrier_note = if cap.barrier_config.is_some() && cap.efs_plan_eligible() {
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

fn requests_immediate_first_order_fallback(message: &str) -> bool {
    message.contains(EFS_FIRST_ORDER_FALLBACK_MARKER)
}

fn downgrade_hessian(cap: &OuterCapability) -> Option<OuterCapability> {
    (cap.hessian == Derivative::Analytic).then(|| {
        let mut degraded = cap.clone();
        degraded.hessian = Derivative::Unavailable;
        degraded
    })
}

/// Disable the EFS/HybridEfs planner path, forcing BFGS-class solvers on the
/// next attempt. Returns `None` if fixed-point is already disabled.
fn disable_fixed_point(cap: &OuterCapability) -> Option<OuterCapability> {
    (!cap.disable_fixed_point && (cap.efs_plan_eligible() || cap.hybrid_efs_plan_eligible())).then(
        || {
            let mut degraded = cap.clone();
            degraded.disable_fixed_point = true;
            degraded
        },
    )
}

fn automatic_fallback_attempts(cap: &OuterCapability) -> Vec<OuterCapability> {
    // Production fallback ladder is strictly analytic-gradient.
    //
    // Silently degrading an `Analytic` gradient capability to
    // `FiniteDifference` was a loaded footgun: the FD path costs 2k inner
    // PIRLS solves per outer gradient evaluation, and for large-n binomial
    // biobank problems (n=320k, p≈71) it would burn the entire benchmark
    // budget spinning on warm-started PIRLS (`iters=1 max_eta=5.9`) without
    // ever reporting failure. The primary analytic path must converge on its
    // own merits — masking non-convergence with FD just hid the real problem.
    //
    // The cascade is now:
    //   1. If EFS/HybridEFS was the primary plan, retry with fixed-point
    //      disabled so BFGS can use the declared analytic gradient directly.
    //   2. If analytic Hessian was declared, retry with it downgraded to
    //      BFGS-approximation (still analytic gradient).
    //   3. No further retries — the caller surfaces the RemlOptimizationFailed
    //      error so the non-convergence is visible.
    let mut attempts = Vec::new();

    if let Some(no_fp_cap) = disable_fixed_point(cap) {
        attempts.push(no_fp_cap.clone());
        if let Some(grad_cap) = downgrade_hessian(&no_fp_cap) {
            attempts.push(grad_cap);
        }
        return attempts;
    }

    if let Some(grad_cap) = downgrade_hessian(cap) {
        attempts.push(grad_cap);
    }
    attempts
}

/// Result of one outer objective evaluation.
///
/// The Hessian field uses [`HessianResult`] instead of `Option<Array2<f64>>`
/// to make the presence/absence of an analytic Hessian explicit and
/// pattern-matchable.
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
pub enum HessianResult {
    /// Analytic Hessian was computed and returned.
    Analytic(Array2<f64>),
    /// Analytic Hessian is available as an exact Hessian-vector product.
    Operator(Arc<dyn OuterHessianOperator>),
    /// No analytic Hessian available for this model path.
    /// The runner must use the [`HessianSource`] from the [`OuterPlan`]
    /// to decide what to do (FD, BFGS, etc.).
    Unavailable,
}

impl Clone for OuterEval {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            gradient: self.gradient.clone(),
            hessian: self.hessian.clone(),
        }
    }
}

impl std::fmt::Debug for OuterEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterEval")
            .field("cost", &self.cost)
            .field("gradient", &self.gradient)
            .field("hessian", &self.hessian)
            .finish()
    }
}

impl Clone for HessianResult {
    fn clone(&self) -> Self {
        match self {
            Self::Analytic(h) => Self::Analytic(h.clone()),
            Self::Operator(op) => Self::Operator(Arc::clone(op)),
            Self::Unavailable => Self::Unavailable,
        }
    }
}

impl std::fmt::Debug for HessianResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analytic(h) => f
                .debug_tuple("Analytic")
                .field(&format!("{}x{}", h.nrows(), h.ncols()))
                .finish(),
            Self::Operator(op) => f
                .debug_tuple("Operator")
                .field(&format!("dim={}", op.dim()))
                .finish(),
            Self::Unavailable => f.write_str("Unavailable"),
        }
    }
}

impl HessianResult {
    /// Extract the Hessian matrix, panicking if unavailable.
    ///
    /// Only call this when the [`OuterPlan`] guarantees `HessianSource::Analytic`.
    pub fn unwrap_analytic(self) -> Array2<f64> {
        match self {
            HessianResult::Analytic(h) => h,
            HessianResult::Operator(_) => {
                panic!("expected dense analytic Hessian but got HessianResult::Operator")
            }
            HessianResult::Unavailable => {
                panic!("expected analytic Hessian but got HessianResult::Unavailable")
            }
        }
    }

    /// Returns `true` if an analytic Hessian is present in any exact form.
    pub fn is_analytic(&self) -> bool {
        matches!(
            self,
            HessianResult::Analytic(_) | HessianResult::Operator(_)
        )
    }

    /// Convert to the optional Hessian shape used by the opt bridge.
    pub fn into_option(self) -> Option<Array2<f64>> {
        match self {
            HessianResult::Analytic(h) => Some(h),
            HessianResult::Operator(_) => None,
            HessianResult::Unavailable => None,
        }
    }

    pub fn dim(&self) -> Option<usize> {
        match self {
            HessianResult::Analytic(h) => Some(h.nrows()),
            HessianResult::Operator(op) => Some(op.dim()),
            HessianResult::Unavailable => None,
        }
    }

    pub fn materialize_dense(&self) -> Result<Option<Array2<f64>>, String> {
        match self {
            HessianResult::Analytic(h) => Ok(Some(h.clone())),
            HessianResult::Operator(op) => op.materialize_dense().map(Some),
            HessianResult::Unavailable => Ok(None),
        }
    }

    pub fn add_rho_block_dense(&mut self, rho_block: &Array2<f64>) -> Result<(), String> {
        if rho_block.nrows() != rho_block.ncols() {
            return Err(format!(
                "rho-block Hessian update must be square, got {}x{}",
                rho_block.nrows(),
                rho_block.ncols()
            ));
        }
        match self {
            HessianResult::Analytic(h) => {
                if rho_block.nrows() > h.nrows() || rho_block.ncols() > h.ncols() {
                    return Err(format!(
                        "rho-block Hessian update shape mismatch: got {}x{}, outer Hessian is {}x{}",
                        rho_block.nrows(),
                        rho_block.ncols(),
                        h.nrows(),
                        h.ncols()
                    ));
                }
                let k = rho_block.nrows();
                let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                sl += rho_block;
                Ok(())
            }
            HessianResult::Operator(op) => {
                let base = Arc::clone(op);
                let dim = base.dim();
                if rho_block.nrows() > dim {
                    return Err(format!(
                        "rho-block Hessian update dimension mismatch: got {}x{}, operator dim is {}",
                        rho_block.nrows(),
                        rho_block.ncols(),
                        dim
                    ));
                }
                *self = HessianResult::Operator(Arc::new(RhoBlockAdditiveOuterHessian {
                    base,
                    rho_block: rho_block.clone(),
                    dim,
                }));
                Ok(())
            }
            HessianResult::Unavailable => Ok(()),
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
/// - `eval_cost()` is used only for cost-based optimization paths.
/// - `eval()` is the main evaluation path (cost + gradient + optional Hessian).
/// - `eval_efs()` is used only by the EFS solver. It runs the inner solve,
///   builds the `InnerSolution`, and computes the EFS step vector. The default
///   implementation returns an error; only objectives that support EFS need
///   to override it.
/// - `reset()` restores state to a clean baseline (for multi-start).
pub trait OuterObjective {
    /// Declare what this objective can compute analytically.
    fn capability(&self) -> OuterCapability;

    /// Evaluate cost only for cost-based optimization paths.
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError>;

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError>;

    /// Evaluate the outer objective at the order requested by the active plan.
    ///
    /// The default preserves legacy behavior by delegating to
    /// [`OuterObjective::eval`].
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
        }
    }

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
    Feo = fn(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
> {
    pub(crate) state: S,
    pub(crate) cap: OuterCapability,
    pub(crate) cost_fn: Fc,
    pub(crate) eval_fn: Fe,
    /// Optional order-aware eval closure. When `None`, `eval_with_order()`
    /// falls back to `eval()`.
    pub(crate) eval_order_fn: Option<Feo>,
    /// Optional reset closure. When `None`, `reset()` is a no-op.
    pub(crate) reset_fn: Option<Fr>,
    /// Optional EFS evaluation closure. When `None`, the default
    /// `OuterObjective::eval_efs` returns an error.
    pub(crate) efs_fn: Option<Fefs>,
}

impl<S, Fc, Fe, Fr, Fefs, Feo> OuterObjective for ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
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

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match self.eval_order_fn.as_mut() {
            Some(f) => f(&mut self.state, rho, order),
            None => (self.eval_fn)(&mut self.state, rho),
        }
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

fn finite_cost_or_error(context: &str, cost: f64) -> Result<f64, ObjectiveEvalError> {
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )))
    }
}

fn finite_outer_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    match &eval.hessian {
        HessianResult::Analytic(hessian) => {
            layout.validate_hessian_shape(hessian, context)?;
            if !hessian.iter().all(|v| v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: objective returned a non-finite Hessian"
                )));
            }
        }
        HessianResult::Operator(op) => {
            if op.dim() != layout.n_params {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer Hessian operator dimension mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                    op.dim(),
                    layout.n_params,
                    layout.rho_dim(),
                    layout.psi_dim
                )));
            }
        }
        HessianResult::Unavailable => {}
    }
    Ok(eval)
}

fn finite_outer_first_order_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    Ok(eval)
}

fn validate_second_order_seed_hessian(
    context: &str,
    layout: OuterThetaLayout,
    eval: &OuterEval,
) -> Result<(), ObjectiveEvalError> {
    if layout.n_params > SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS || !eval.hessian.is_analytic() {
        return Ok(());
    }

    let Some(hessian) = eval.hessian.materialize_dense().map_err(|message| {
        ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian materialization failed during second-order seed validation: {message}"
        ))
    })?
    else {
        return Ok(());
    };

    layout.validate_hessian_shape(&hessian, context)?;
    if !hessian.iter().all(|value| value.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian probe encountered non-finite entries"
        )));
    }

    Ok(())
}

fn finite_efs_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: EfsEval,
) -> Result<EfsEval, ObjectiveEvalError> {
    layout.validate_efs_eval(&eval, context)?;
    finite_cost_or_error(context, eval.cost)?;
    Ok(eval)
}

struct OuterFirstOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
}

impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let cost = self
            .obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        finite_cost_or_error("outer eval_cost failed", cost)
    }
}

impl FirstOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

struct OuterSecondOrderBridge<'a> {
    obj: &'a mut dyn OuterObjective,
    layout: OuterThetaLayout,
    hessian_source: HessianSource,
}

impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let cost = self
            .obj
            .eval_cost(x)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        finite_cost_or_error("outer eval_cost failed", cost)
    }
}

impl FirstOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl SecondOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        let hessian = match self.hessian_source {
            HessianSource::Analytic => eval.hessian.into_option(),
            HessianSource::BfgsApprox
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
    layout: OuterThetaLayout,
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
        self.layout.validate_point_len(x, "outer EFS eval failed")?;
        let eval = self
            .obj
            .eval_efs(x)
            .map_err(|err| into_objective_error("outer EFS eval failed", err))?;
        self.layout
            .validate_efs_eval(&eval, "outer EFS eval failed")?;
        if !eval.cost.is_finite() {
            return Err(ObjectiveEvalError::recoverable(
                "outer EFS eval failed: objective returned a non-finite cost".to_string(),
            ));
        }
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
                    Ok(trial_cost) if trial_cost.is_finite() && trial_cost < current_cost => {
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
                            let halved = original_psi_steps[j] * 0.5_f64.powi((bt + 1) as i32);
                            combined_step[i] = halved;
                        }
                    }
                    Err(_) => {
                        // Evaluation failed — halve ψ step and retry.
                        log::debug!(
                            "[HYBRID-EFS] ψ backtrack {bt}: trial eval failed, halving ψ step"
                        );
                        for (j, &i) in psi_indices.iter().enumerate() {
                            let halved = original_psi_steps[j] * 0.5_f64.powi((bt + 1) as i32);
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

/// Outcome of an auxiliary compass-search run.
enum CompassSearchOutcome {
    /// The step length contracted below tolerance with no further improvement
    /// — i.e. the iterate is a step-minimizer over the positive basis
    /// {±step·e_i} at scale < step_tol. By Kolda-Lewis-Torczon Thm 3.3 this
    /// implies first-order stationarity up to the step-tol grid.
    Converged { point: Array1<f64>, cost: f64, polls: usize },
    /// The poll budget was exhausted before step contraction reached the
    /// tolerance. Return the best-seen iterate; caller treats as
    /// non-converged so log/diagnostics surface the truncation.
    BudgetExhausted { point: Array1<f64>, cost: f64, polls: usize },
}

/// Coordinate compass search with bound clamping.
///
/// Why this method is correct for derivative-free aux optimization:
/// the algorithm only compares cost values at polled points. It never
/// finite-differences a gradient and never feeds approximations into a
/// gradient-based optimizer. For any continuously differentiable cost
/// bounded below on the compact box [lower, upper], compass search
/// converges to a stationary point (Kolda-Lewis-Torczon, SIAM Review
/// 45:385, 2003, Thm 3.3). The theorem's polling requirement — that
/// all 2·dim directions ±step·e_i are evaluated before the step
/// contracts — is satisfied explicitly by the `!improved ⇒ step /= 2`
/// branch below: if no coordinate probe improved, every probe was
/// evaluated and rejected.
///
/// Error policy: `obj.eval_cost` errors at a probe are treated as
/// infeasible (the search simply does not accept that point). A genuine
/// error at the seed itself is surfaced by the caller via the initial
/// `eval_cost` check, so the helper only runs against a finite seed cost.
fn compass_search_outer(
    obj: &mut dyn OuterObjective,
    mut x: Array1<f64>,
    mut best_cost: f64,
    lower: ndarray::ArrayView1<'_, f64>,
    upper: ndarray::ArrayView1<'_, f64>,
    init_step: f64,
    step_tol: f64,
    max_polls: usize,
) -> CompassSearchOutcome {
    for i in 0..x.len() {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
    let mut step = init_step;
    let mut polls: usize = 0;
    while step > step_tol && polls < max_polls {
        let mut improved = false;
        'sweep: for i in 0..x.len() {
            for &sign in &[1.0, -1.0] {
                if polls >= max_polls {
                    break 'sweep;
                }
                polls += 1;
                let candidate_i = (x[i] + sign * step).clamp(lower[i], upper[i]);
                if (candidate_i - x[i]).abs() < step_tol {
                    continue;
                }
                let mut candidate = x.clone();
                candidate[i] = candidate_i;
                let probe = obj
                    .eval_cost(&candidate)
                    .ok()
                    .filter(|v| v.is_finite());
                if let Some(c) = probe
                    && c < best_cost
                {
                    x = candidate;
                    best_cost = c;
                    improved = true;
                    break 'sweep;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    if step <= step_tol {
        CompassSearchOutcome::Converged { point: x, cost: best_cost, polls }
    } else {
        CompassSearchOutcome::BudgetExhausted { point: x, cost: best_cost, polls }
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

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
struct OuterConfig {
    tolerance: f64,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    seed_config: crate::seeding::SeedConfig,
    rho_bound: f64,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    solver_class: SolverClass,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            max_iter: 200,
            bounds: None,
            seed_config: crate::seeding::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            solver_class: SolverClass::Primary,
        }
    }
}

// ─── OuterProblem builder ─────────────────────────────────────────────
//
// Declarative builder for outer optimization problems.  Derives
// OuterCapability flags from high-level inputs (gradient/hessian
// availability, psi dimension, EFS eligibility) so call sites never
// hand-copy capability flags.

/// Declarative outer-problem builder.  Produces both the
/// [`OuterCapability`] (what the objective can provide) and the
/// [`OuterConfig`] (how the runner should behave) from a small set
/// of high-level declarations.
pub struct OuterProblem {
    n_params: usize,
    gradient: Derivative,
    hessian: Derivative,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    psi_dim: usize,
    barrier_config: Option<BarrierConfig>,
    tolerance: f64,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    rho_bound: f64,
    seed_config: crate::seeding::SeedConfig,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    solver_class: SolverClass,
}

impl OuterProblem {
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            prefer_gradient_only: false,
            disable_fixed_point: false,
            psi_dim: 0,
            barrier_config: None,
            tolerance: 1e-5,
            max_iter: 200,
            bounds: None,
            rho_bound: 30.0,
            seed_config: crate::seeding::SeedConfig::default(),
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            solver_class: SolverClass::Primary,
        }
    }

    pub fn with_gradient(mut self, d: Derivative) -> Self {
        self.gradient = d;
        self
    }
    pub fn with_hessian(mut self, d: Derivative) -> Self {
        self.hessian = d;
        self
    }
    pub fn with_prefer_gradient_only(mut self, prefer_gradient_only: bool) -> Self {
        self.prefer_gradient_only = prefer_gradient_only;
        self
    }
    /// Forbid the planner from selecting EFS/HybridEfs, even when the
    /// objective implements `eval_efs()` and the coordinate structure would
    /// otherwise make pure/hybrid EFS eligible.
    ///
    /// Callers use this for families where the Wood-Fasiolo structural
    /// property is known not to hold (e.g. GAMLSS/location-scale with
    /// β-dependent joint Hessian), so EFS would stagnate and burn budget
    /// before the automatic cascade falls back to gradient-based BFGS.
    pub fn with_disable_fixed_point(mut self, disable: bool) -> Self {
        self.disable_fixed_point = disable;
        self
    }
    pub fn with_psi_dim(mut self, dim: usize) -> Self {
        self.psi_dim = dim;
        self
    }
    pub fn with_barrier(mut self, cfg: Option<BarrierConfig>) -> Self {
        self.barrier_config = cfg;
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_bounds(mut self, lo: Array1<f64>, hi: Array1<f64>) -> Self {
        self.bounds = Some((lo, hi));
        self
    }
    pub fn with_rho_bound(mut self, b: f64) -> Self {
        self.rho_bound = b;
        self
    }
    pub fn with_seed_config(mut self, sc: crate::seeding::SeedConfig) -> Self {
        self.seed_config = sc;
        self
    }
    pub fn with_heuristic_lambdas(mut self, h: Vec<f64>) -> Self {
        self.heuristic_lambdas = Some(h);
        self
    }
    pub fn with_initial_rho(mut self, rho: Array1<f64>) -> Self {
        self.initial_rho = Some(rho);
        self
    }
    pub fn with_screening_cap(mut self, screening_cap: Arc<AtomicUsize>) -> Self {
        self.screening_cap = Some(screening_cap);
        self
    }
    /// Opt into a specific solver class. The default is
    /// [`SolverClass::Primary`] (the main REML outer). Setting
    /// [`SolverClass::AuxiliaryGradientFree`] unlocks
    /// [`Solver::CompassSearch`] dispatch for small-dim problems with no
    /// analytic gradient (survival baseline theta, inverse-link params).
    /// REML builders must not set this.
    pub fn with_solver_class(mut self, class: SolverClass) -> Self {
        self.solver_class = class;
        self
    }

    /// Derive the capability flags from the builder state.
    /// `fixed_point_available` is set to `false` here; `build_objective`
    /// overrides it based on whether an EFS closure is actually provided.
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: self.gradient,
            hessian: self.hessian,
            prefer_gradient_only: self.prefer_gradient_only,
            disable_fixed_point: self.disable_fixed_point,
            n_params: self.n_params,
            psi_dim: self.psi_dim,
            fixed_point_available: false,
            barrier_config: self.barrier_config.clone(),
        }
    }

    /// Derive the runner configuration from the builder state.
    fn config(&self) -> OuterConfig {
        OuterConfig {
            tolerance: self.tolerance,
            max_iter: self.max_iter,
            bounds: self.bounds.clone(),
            seed_config: self.seed_config.clone(),
            rho_bound: self.rho_bound,
            heuristic_lambdas: self.heuristic_lambdas.clone(),
            initial_rho: self.initial_rho.clone(),
            fallback_policy: self.fallback_policy,
            screening_cap: self.screening_cap.clone(),
            solver_class: self.solver_class,
        }
    }

    /// Construct a [`ClosureObjective`] with capability flags derived from the
    /// builder state **and** the closures actually provided.
    ///
    /// `fixed_point_available` is set to `true` when `efs_fn` is `Some`,
    /// regardless of whether `.with_efs()` was called.  This is the canonical
    /// way to create production objectives — it eliminates the drift risk of
    /// manually entering capability flags.
    pub fn build_objective<S, Fc, Fe, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        // Derive fixed_point_available from whether the caller actually
        // provided an EFS hook, rather than relying on manual flags.
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: None,
            reset_fn,
            efs_fn,
        }
    }

    /// Construct a [`ClosureObjective`] with an order-aware evaluation hook.
    ///
    /// This lets the runner request first-order vs second-order work based on
    /// the active outer plan while preserving the legacy eager `eval_fn`.
    pub fn build_objective_with_eval_order<S, Fc, Fe, Feo, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
        }
    }

    /// Run the outer optimization with a given objective.
    pub fn run(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<OuterResult, EstimationError> {
        run_outer(obj, &self.config(), context)
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

const OPERATOR_TRUST_RADIUS_INIT: f64 = 1.0;
const OPERATOR_TRUST_RADIUS_MAX: f64 = 1.0e6;
const OPERATOR_ETA_ACCEPT: f64 = 1.0e-4;

fn project_to_bounds(x: &Array1<f64>, bounds: Option<&(Array1<f64>, Array1<f64>)>) -> Array1<f64> {
    match bounds {
        Some((lower, upper)) => {
            let mut out = x.clone();
            for idx in 0..out.len() {
                out[idx] = out[idx].clamp(lower[idx], upper[idx]);
            }
            out
        }
        None => x.clone(),
    }
}

fn projected_gradient(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> Array1<f64> {
    let mut out = gradient.clone();
    if let Some((lower, upper)) = bounds {
        for idx in 0..out.len() {
            let at_lower = x[idx] <= lower[idx] + 1e-10;
            let at_upper = x[idx] >= upper[idx] - 1e-10;
            if (at_lower && gradient[idx] > 0.0) || (at_upper && gradient[idx] < 0.0) {
                out[idx] = 0.0;
            }
        }
    }
    out
}

fn active_mask(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> Vec<bool> {
    match bounds {
        Some((lower, upper)) => (0..x.len())
            .map(|idx| {
                let at_lower = x[idx] <= lower[idx] + 1e-10;
                let at_upper = x[idx] >= upper[idx] - 1e-10;
                (at_lower && gradient[idx] > 0.0) || (at_upper && gradient[idx] < 0.0)
            })
            .collect(),
        None => vec![false; x.len()],
    }
}

fn masked_matvec(
    op: &dyn OuterHessianOperator,
    vector: &Array1<f64>,
    active: &[bool],
) -> Result<Array1<f64>, String> {
    let mut masked = vector.clone();
    for idx in 0..masked.len() {
        if active[idx] {
            masked[idx] = 0.0;
        }
    }
    let mut out = op.matvec(&masked)?;
    if out.len() != masked.len() {
        return Err(format!(
            "outer Hessian operator matvec length mismatch: got {}, expected {}",
            out.len(),
            masked.len()
        ));
    }
    for idx in 0..out.len() {
        if active[idx] {
            out[idx] = 0.0;
        }
    }
    Ok(out)
}

fn predicted_decrease_from_operator(
    op: &dyn OuterHessianOperator,
    gradient: &Array1<f64>,
    step: &Array1<f64>,
    active: &[bool],
) -> Result<f64, String> {
    let hs = if active.iter().copied().any(|flag| flag) {
        masked_matvec(op, step, active)?
    } else {
        op.matvec(step)?
    };
    Ok(-(gradient.dot(step) + 0.5 * step.dot(&hs)))
}

fn steihaug_toint_step_operator(
    op: &dyn OuterHessianOperator,
    gradient: &Array1<f64>,
    trust_radius: f64,
    active: &[bool],
) -> Result<Option<(Array1<f64>, f64)>, String> {
    let n = gradient.len();
    let g_norm = gradient.dot(gradient).sqrt();
    if !g_norm.is_finite() || g_norm <= 0.0 {
        return Ok(None);
    }

    let mut p = Array1::<f64>::zeros(n);
    let mut r = gradient.clone();
    for idx in 0..n {
        if active[idx] {
            r[idx] = 0.0;
        }
    }
    let mut d = r.mapv(|value| -value);
    let mut rtr = r.dot(&r);
    if !rtr.is_finite() || rtr <= 0.0 {
        return Ok(None);
    }

    let cg_tol = (1e-6 * g_norm).max(1e-12);
    let max_iter = (2 * n).max(10);

    for _ in 0..max_iter {
        let bd = if active.iter().copied().any(|flag| flag) {
            masked_matvec(op, &d, active)?
        } else {
            op.matvec(&d)?
        };
        let d_bd = d.dot(&bd);
        if !d_bd.is_finite() || d_bd <= 1e-14 * d.dot(&d).max(1.0) {
            let d_norm_sq = d.dot(&d);
            if d_norm_sq <= 0.0 {
                return Ok(None);
            }
            let p_dot_d = p.dot(&d);
            let p_norm_sq = p.dot(&p);
            let disc = p_dot_d * p_dot_d - d_norm_sq * (p_norm_sq - trust_radius * trust_radius);
            if disc < 0.0 {
                return Ok(None);
            }
            let tau = (-p_dot_d + disc.sqrt()) / d_norm_sq;
            let mut boundary = p.clone();
            boundary.scaled_add(tau, &d);
            let pred = predicted_decrease_from_operator(op, gradient, &boundary, active)?;
            return Ok((pred.is_finite() && pred > 0.0).then_some((boundary, pred)));
        }

        let alpha = rtr / d_bd;
        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(None);
        }

        let mut p_next = p.clone();
        p_next.scaled_add(alpha, &d);
        let p_next_norm = p_next.dot(&p_next).sqrt();
        if p_next_norm >= trust_radius {
            let d_norm_sq = d.dot(&d);
            if d_norm_sq <= 0.0 {
                return Ok(None);
            }
            let p_dot_d = p.dot(&d);
            let p_norm_sq = p.dot(&p);
            let disc = p_dot_d * p_dot_d - d_norm_sq * (p_norm_sq - trust_radius * trust_radius);
            if disc < 0.0 {
                return Ok(None);
            }
            let tau = (-p_dot_d + disc.sqrt()) / d_norm_sq;
            let mut boundary = p.clone();
            boundary.scaled_add(tau, &d);
            let pred = predicted_decrease_from_operator(op, gradient, &boundary, active)?;
            return Ok((pred.is_finite() && pred > 0.0).then_some((boundary, pred)));
        }

        r.scaled_add(alpha, &bd);
        for idx in 0..n {
            if active[idx] {
                r[idx] = 0.0;
            }
        }
        let rtr_next = r.dot(&r);
        if !rtr_next.is_finite() {
            return Ok(None);
        }

        p = p_next;
        if rtr_next.sqrt() <= cg_tol {
            let pred = predicted_decrease_from_operator(op, gradient, &p, active)?;
            return Ok((pred.is_finite() && pred > 0.0).then_some((p, pred)));
        }

        let beta = rtr_next / rtr.max(1e-32);
        if !beta.is_finite() || beta < 0.0 {
            return Ok(None);
        }
        d *= beta;
        d -= &r;
        for idx in 0..n {
            if active[idx] {
                d[idx] = 0.0;
            }
        }
        rtr = rtr_next;
    }

    let pred = predicted_decrease_from_operator(op, gradient, &p, active)?;
    Ok((pred.is_finite() && pred > 0.0).then_some((p, pred)))
}

fn run_operator_trust_region(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    layout: OuterThetaLayout,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
    tolerance: f64,
    max_iter: usize,
    initial_eval: OuterEval,
    plan: OuterPlan,
) -> Result<OuterResult, EstimationError> {
    let mut x_k = project_to_bounds(seed, bounds);
    let mut eval_k = initial_eval;
    let mut trust_radius = OPERATOR_TRUST_RADIUS_INIT;

    for iter in 0..max_iter {
        let g_proj = projected_gradient(&x_k, &eval_k.gradient, bounds);
        let g_norm = g_proj.dot(&g_proj).sqrt();
        if g_norm.is_finite() && g_norm <= tolerance {
            return Ok(OuterResult {
                rho: x_k,
                final_value: eval_k.cost,
                iterations: iter,
                final_grad_norm: g_norm,
                final_gradient: Some(eval_k.gradient),
                final_hessian: None,
                converged: true,
                plan_used: plan,
            });
        }

        let HessianResult::Operator(op_arc) = &eval_k.hessian else {
            return Err(EstimationError::RemlOptimizationFailed(
                "operator trust-region received a non-operator Hessian".to_string(),
            ));
        };
        let active = active_mask(&x_k, &eval_k.gradient, bounds);
        let Some((trial_step, pred_dec_free)) =
            steihaug_toint_step_operator(op_arc.as_ref(), &g_proj, trust_radius, &active)
                .map_err(EstimationError::RemlOptimizationFailed)?
        else {
            trust_radius = (trust_radius * 0.5).max(1e-12);
            continue;
        };

        let x_trial_raw = &x_k + &trial_step;
        let x_trial = project_to_bounds(&x_trial_raw, bounds);
        let s_trial = &x_trial - &x_k;
        let s_norm = s_trial.dot(&s_trial).sqrt();
        if !s_norm.is_finite() || s_norm <= 1e-16 {
            trust_radius = (trust_radius * 0.5).max(1e-12);
            continue;
        }

        let pred_dec = if (&s_trial - &trial_step)
            .dot(&(&s_trial - &trial_step))
            .sqrt()
            > 1e-8 * (1.0 + trial_step.dot(&trial_step).sqrt())
        {
            predicted_decrease_from_operator(op_arc.as_ref(), &g_proj, &s_trial, &active)
                .map_err(EstimationError::RemlOptimizationFailed)?
        } else {
            pred_dec_free
        };
        if !pred_dec.is_finite() || pred_dec <= 0.0 {
            trust_radius = (trust_radius * 0.5).max(1e-12);
            continue;
        }

        let eval_trial = obj.eval_with_order(&x_trial, OuterEvalOrder::ValueGradientHessian)?;
        let eval_trial =
            finite_outer_eval_or_error("outer operator eval failed", layout, eval_trial).map_err(
                |err| match err {
                    ObjectiveEvalError::Recoverable { message }
                    | ObjectiveEvalError::Fatal { message } => {
                        EstimationError::RemlOptimizationFailed(message)
                    }
                },
            )?;
        let act_dec = eval_k.cost - eval_trial.cost;
        let rho = act_dec / pred_dec;
        if rho > 0.75 && s_norm > 0.99 * trust_radius {
            trust_radius = (trust_radius * 2.0).min(OPERATOR_TRUST_RADIUS_MAX);
        } else if rho < 0.25 {
            trust_radius = (trust_radius * 0.5).max(1e-12);
        }

        if rho > OPERATOR_ETA_ACCEPT {
            x_k = x_trial;
            eval_k = eval_trial;
        }
    }

    let final_grad = projected_gradient(&x_k, &eval_k.gradient, bounds);
    let final_grad_norm = final_grad.dot(&final_grad).sqrt();
    Ok(OuterResult {
        rho: x_k,
        final_value: eval_k.cost,
        iterations: max_iter,
        final_grad_norm,
        final_gradient: Some(eval_k.gradient),
        final_hessian: None,
        converged: false,
        plan_used: plan,
    })
}

/// Run the outer smoothing-parameter optimization.
///
/// This is the single entry point that replaces the scattered optimizer wiring
/// across estimate.rs, joint.rs, and custom_family.rs. It:
///
/// 1. Queries the objective's capability declaration.
/// 2. Calls `plan()` to select solver + hessian source.
/// 3. Logs the plan (so FD is never silent).
/// 4. Generates seed candidates.
/// 5. Runs the chosen solver on candidates in heuristic order up to budget.
/// 6. If the configured fallback policy allows it, re-plans with degraded
///    capabilities chosen centrally inside outer_strategy and retries.
/// 7. Returns the best result (including which plan was actually used).
///
/// Do not wrap `run_outer` calls in try/catch with ad-hoc solver recovery.
/// Callers should declare only the primary capability and, at most, whether
/// automatic fallback is enabled at all.
fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let cap = obj.capability();
    cap.validate_layout(context)?;
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        cap.theta_layout()
            .validate_point_len(initial_rho, "initial outer seed")
            .map_err(|err| match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => {
                    EstimationError::RemlOptimizationFailed(format!("{context}: {message}"))
                }
            })?;
    }

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan_with_class(&cap, config.solver_class);
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
    // any centrally-derived degraded capabilities. Aux direct-search has no
    // degraded ladder — a single attempt either succeeds or the failure is
    // surfaced to the caller.
    let fallback_attempts = match (config.fallback_policy, config.solver_class) {
        (FallbackPolicy::Automatic, SolverClass::Primary) => automatic_fallback_attempts(&cap),
        (FallbackPolicy::Automatic, SolverClass::AuxiliaryGradientFree)
        | (FallbackPolicy::Disabled, _) => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan_with_class(attempt_cap, config.solver_class);
        if attempt_idx > 0 {
            log::info!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        match run_outer_with_plan(obj, config, context, attempt_cap, &the_plan) {
            Ok(result) => {
                if result.converged || attempt_idx + 1 == attempts.len() {
                    return Ok(result);
                }

                let message = format!(
                    "{context}: attempt {} (plan={the_plan}) exhausted without convergence",
                    attempt_idx + 1
                );
                log::info!("[OUTER] {message}; trying degraded fallback plan");
                last_error = Some(EstimationError::RemlOptimizationFailed(message));
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
    let mut seeds = {
        let generated = crate::seeding::generate_rho_candidates(
            cap.n_params,
            config.heuristic_lambdas.as_deref(),
            &config.seed_config,
        );
        if generated.is_empty() {
            Vec::new()
        } else {
            generated
        }
    };
    if let Some(initial_rho) = config.initial_rho.as_ref()
        && !seeds.iter().any(|seed| seed == initial_rho)
    {
        seeds.insert(0, initial_rho.clone());
    }
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "no seeds generated for outer optimization ({context})"
        )));
    }

    let screening_enabled = config.screening_cap.is_some();
    let seed_budget = effective_seed_budget(
        config.seed_config.seed_budget,
        the_plan.solver,
        config.seed_config.risk_profile,
        screening_enabled,
    )
    .min(seeds.len());
    if should_screen_seeds(config, the_plan.solver, seeds.len(), seed_budget) {
        seeds = rank_seeds_with_screening(obj, config, context, &seeds);
    }
    log::info!(
        "[OUTER] {context}: trying generated seeds directly (generated={}, budget={})",
        seeds.len(),
        seed_budget,
    );
    if seed_budget < config.seed_config.seed_budget.max(1) {
        log::info!(
            "[OUTER] {context}: capped requested seed budget {} -> {} for {:?} ({:?})",
            config.seed_config.seed_budget.max(1),
            seed_budget,
            the_plan.solver,
            config.seed_config.risk_profile,
        );
    }
    if seeds.len() > seed_budget {
        log::info!(
            "[OUTER] {context}: trying up to {seed_budget}/{} generated seeds in heuristic order",
            seeds.len(),
        );
    }

    let (lower, upper) = config.bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(cap.n_params, -config.rho_bound),
            Array1::<f64>::from_elem(cap.n_params, config.rho_bound),
        )
    });
    let bounds_template = (lower, upper);

    let mut best: Option<OuterResult> = None;
    let mut last_seed_error: Option<String> = None;
    let layout = cap.theta_layout();
    let mut started_seeds = 0usize;
    let expensive_seed_limit =
        expensive_unsuccessful_seed_limit(the_plan.solver, config.seed_config.risk_profile);
    let mut unsuccessful_expensive_seeds = 0usize;

    'seed_attempts: for seed in &seeds {
        if started_seeds == seed_budget {
            break;
        }
        obj.reset();
        let t_seed_start = std::time::Instant::now();
        let seed_slot;
        let result: Result<OuterResult, EstimationError> = match the_plan.solver {
            Solver::Arc => {
                let seed_eval = obj
                    .eval_with_order(&seed, OuterEvalOrder::ValueGradientHessian)
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                        last_seed_error = Some(err.to_string());
                        continue 'seed_attempts;
                    }
                };
                let seed_eval = finite_outer_eval_or_error("outer eval failed", layout, seed_eval)
                    .map_err(|err| match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    });
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                        last_seed_error = Some(err.to_string());
                        continue 'seed_attempts;
                    }
                };
                validate_second_order_seed_hessian(context, layout, &seed_eval).map_err(|err| {
                    match err {
                        ObjectiveEvalError::Recoverable { message }
                        | ObjectiveEvalError::Fatal { message } => {
                            EstimationError::RemlOptimizationFailed(message)
                        }
                    }
                })?;
                started_seeds += 1;
                seed_slot = started_seeds;

                if matches!(seed_eval.hessian, HessianResult::Operator(_)) {
                    log::debug!(
                        "[OUTER] {context}: analytic Hessian provided as Hv operator; \
                         routing to internal trust-region CG"
                    );
                    run_operator_trust_region(
                        obj,
                        &seed,
                        layout,
                        Some(&bounds_template),
                        config.tolerance,
                        config.max_iter,
                        seed_eval,
                        *the_plan,
                    )
                } else {
                    let hessian_source = the_plan.hessian_source;
                    let objective = OuterSecondOrderBridge {
                        obj,
                        layout,
                        hessian_source,
                    };

                    let (lo, hi) = &bounds_template;
                    let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                        .expect("outer rho bounds must be valid");
                    let tol =
                        Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                    let max_iter =
                        MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");

                    let mut optimizer = ArcOptimizer::new(seed.clone(), objective)
                        .with_bounds(bounds)
                        .with_tolerance(tol)
                        .with_max_iterations(max_iter);
                    match optimizer.run() {
                        Ok(sol) => Ok(solution_into_outer_result(sol, true, *the_plan)),
                        Err(ArcError::MaxIterationsReached { last_solution, .. }) => {
                            Ok(solution_into_outer_result(*last_solution, false, *the_plan))
                        }
                        Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Arc solver failed: {e:?}"
                        ))),
                    }
                }
            }
            Solver::Bfgs => {
                // Production invariant: the outer BFGS runner never accepts a
                // non-analytic gradient capability. The FD-BFGS wrapping path
                // was a silent footgun that could spin for the entire benchmark
                // budget on large-n binomial problems (~2k PIRLS solves per
                // outer gradient at n=320k, masking a non-converging primary
                // analytic path). Fail loudly at the top of the seed loop so
                // the caller surfaces the underlying capability/plan mismatch
                // instead of degrading correctness behind the scenes.
                if cap.gradient != Derivative::Analytic {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "{context}: outer BFGS requires an analytic gradient capability; \
                         FD-gradient fallback is disabled in production (plan={the_plan}, \
                         declared gradient={:?})",
                        cap.gradient,
                    )));
                }
                let seed_eval = obj
                    .eval_with_order(&seed, OuterEvalOrder::ValueAndGradient)
                    .map_err(|err| into_objective_error("outer eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                        last_seed_error = Some(err.to_string());
                        continue 'seed_attempts;
                    }
                };
                let seed_eval =
                    finite_outer_first_order_eval_or_error("outer eval failed", layout, seed_eval)
                        .map_err(|err| match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        });
                if let Err(err) = seed_eval {
                    log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                    last_seed_error = Some(err.to_string());
                    continue 'seed_attempts;
                }
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                let objective = OuterFirstOrderBridge { obj, layout };
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
            Solver::Efs => {
                let seed_eval = obj
                    .eval_efs(&seed)
                    .map_err(|err| into_objective_error("outer EFS eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                        last_seed_error = Some(err.to_string());
                        continue 'seed_attempts;
                    }
                };
                let seed_eval =
                    finite_efs_eval_or_error("outer EFS eval failed", layout, seed_eval).map_err(
                        |err| match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        },
                    );
                if let Err(err) = seed_eval {
                    if requests_immediate_first_order_fallback(&err.to_string()) {
                        return Err(err);
                    }
                    log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                    last_seed_error = Some(err.to_string());
                    continue 'seed_attempts;
                }
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                let objective = OuterFixedPointBridge {
                    obj,
                    layout,
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
            Solver::HybridEfs => {
                let seed_eval = obj
                    .eval_efs(&seed)
                    .map_err(|err| into_objective_error("outer EFS eval failed", err));
                let seed_eval = match seed_eval {
                    Ok(seed_eval) => seed_eval,
                    Err(err) => {
                        let err = match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        };
                        if requests_immediate_first_order_fallback(&err.to_string()) {
                            return Err(err);
                        }
                        log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                        last_seed_error = Some(err.to_string());
                        continue 'seed_attempts;
                    }
                };
                let seed_eval =
                    finite_efs_eval_or_error("outer EFS eval failed", layout, seed_eval).map_err(
                        |err| match err {
                            ObjectiveEvalError::Recoverable { message }
                            | ObjectiveEvalError::Fatal { message } => {
                                EstimationError::RemlOptimizationFailed(message)
                            }
                        },
                    );
                if let Err(err) = seed_eval {
                    if requests_immediate_first_order_fallback(&err.to_string()) {
                        return Err(err);
                    }
                    log::warn!("[OUTER] {context}: rejecting seed before solver start: {err}");
                    last_seed_error = Some(err.to_string());
                    continue 'seed_attempts;
                }
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let bounds = Bounds::new(lo.clone(), hi.clone(), 1e-6)
                    .expect("outer rho bounds must be valid");
                let tol = Tolerance::new(config.tolerance).expect("outer tolerance must be valid");
                let max_iter =
                    MaxIterations::new(config.max_iter).expect("outer max_iter must be valid");
                let objective = OuterFixedPointBridge {
                    obj,
                    layout,
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
            Solver::CompassSearch => {
                // Aux direct-search: uses cost values only, never queries
                // gradient or Hessian. config.tolerance is the step-length
                // floor, config.max_iter is the total-poll budget.
                let seed_cost = obj
                    .eval_cost(&seed)
                    .map_err(|err| EstimationError::RemlOptimizationFailed(
                        format!("aux direct-search seed cost failed ({context}): {err}")
                    ))?;
                if !seed_cost.is_finite() {
                    last_seed_error = Some(format!(
                        "aux direct-search rejects non-finite seed cost ({seed_cost})"
                    ));
                    continue 'seed_attempts;
                }
                started_seeds += 1;
                seed_slot = started_seeds;
                let (lo, hi) = &bounds_template;
                let outcome = compass_search_outer(
                    obj,
                    seed.clone(),
                    seed_cost,
                    lo.view(),
                    hi.view(),
                    1.0,
                    config.tolerance,
                    config.max_iter,
                );
                match outcome {
                    CompassSearchOutcome::Converged { point, cost, polls } => {
                        Ok(OuterResult {
                            rho: point,
                            final_value: cost,
                            iterations: polls,
                            final_grad_norm: 0.0,
                            final_gradient: None,
                            final_hessian: None,
                            converged: true,
                            plan_used: *the_plan,
                        })
                    }
                    CompassSearchOutcome::BudgetExhausted { point, cost, polls } => {
                        Ok(OuterResult {
                            rho: point,
                            final_value: cost,
                            iterations: polls,
                            final_grad_norm: 0.0,
                            final_gradient: None,
                            final_hessian: None,
                            converged: false,
                            plan_used: *the_plan,
                        })
                    }
                }
            }
        };

        let seed_elapsed = t_seed_start.elapsed().as_secs_f64();
        match result {
            Ok(candidate) => {
                let candidate_converged = candidate.converged;
                log::info!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  cost={:.6e}  converged={}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    candidate.final_value,
                    candidate.converged,
                );
                if candidate_improves_best(&candidate, best.as_ref()) {
                    best = Some(candidate);
                }
                if best.as_ref().is_some_and(|b| b.converged) {
                    break;
                }
                if !candidate_converged && matches!(expensive_seed_limit, Some(limit) if limit > 0)
                {
                    unsuccessful_expensive_seeds += 1;
                    if let Some(limit) = expensive_seed_limit
                        && unsuccessful_expensive_seeds >= limit
                    {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} non-converged {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        break;
                    }
                }
            }
            Err(e) => {
                if requests_immediate_first_order_fallback(&e.to_string()) {
                    return Err(e);
                }
                log::info!(
                    "[outer-timing] seed {}/{} ({:?}): {:.3}s  FAILED: {}",
                    seed_slot,
                    seed_budget,
                    the_plan.solver,
                    seed_elapsed,
                    e,
                );
                last_seed_error = Some(e.to_string());
                if let Some(limit) = expensive_seed_limit {
                    unsuccessful_expensive_seeds += 1;
                    if unsuccessful_expensive_seeds >= limit {
                        log::info!(
                            "[OUTER] {context}: stopping expensive multi-start after {} failed {:?} seed(s)",
                            unsuccessful_expensive_seeds,
                            the_plan.solver,
                        );
                        break;
                    }
                }
            }
        }
    }

    best.ok_or_else(|| {
        if started_seeds == 0 {
            return match last_seed_error {
                Some(err) => EstimationError::RemlOptimizationFailed(format!(
                    "no candidate seeds passed outer startup validation ({context}); last_error: {err}"
                )),
                None => EstimationError::RemlOptimizationFailed(format!(
                    "no candidate seeds passed outer startup validation ({context})"
                )),
            };
        }
        match last_seed_error {
            Some(err) => EstimationError::RemlOptimizationFailed(format!(
                "all {} seed candidates failed ({context}); last_error: {err}",
                started_seeds
            )),
            None => EstimationError::RemlOptimizationFailed(format!(
                "all {} seed candidates failed ({context})",
                started_seeds
            )),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::opt::FixedPointObjective;
    use ndarray::array;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    struct FailingSeedMaterializationOperator {
        dim: usize,
    }

    impl OuterHessianOperator for FailingSeedMaterializationOperator {
        fn dim(&self) -> usize {
            self.dim
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(v.clone())
        }

        fn materialize_dense(&self) -> Result<Array2<f64>, String> {
            Err("seed materialization failed".to_string())
        }
    }

    #[test]
    fn plan_analytic_hessian_selects_arc() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Arc);
        assert_eq!(p.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn plan_prefer_gradient_only_downgrades_analytic_hessian_to_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 3,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_no_hessian_few_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_fd_hessian_capability_is_treated_as_gradient_only() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::FiniteDifference,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_no_hessian_many_params_selects_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_fd_gradient_still_maps_to_bfgs_for_error_context() {
        // FD-gradient capabilities must not be declared by production callers;
        // the runner now errors out on the Bfgs path. The planner still
        // produces a `Solver::Bfgs` plan so the eventual error can format a
        // meaningful "plan={plan}" context string instead of panicking on an
        // unmatched match arm.
        for n in [1, 5, 20] {
            let cap = OuterCapability {
                gradient: Derivative::FiniteDifference,
                hessian: Derivative::Unavailable,
                n_params: n,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            };
            let p = plan(&cap);
            assert_eq!(p.solver, Solver::Bfgs);
            assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
        }
    }

    #[test]
    fn plan_fd_gradient_with_hessian_still_bfgs_for_error_context() {
        let cap = OuterCapability {
            gradient: Derivative::FiniteDifference,
            hessian: Derivative::Analytic,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
    }

    #[test]
    fn plan_boundary_8_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_boundary_9_params_uses_bfgs() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: SMALL_OUTER_BFGS_MAX_PARAMS + 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_selected_for_penalty_like_many_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_efs_not_selected_with_analytic_hessian() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn plan_efs_allowed_with_barrier_config_no_gradient() {
        // Even without analytic gradient, EFS is selected when all coords
        // are penalty-like and the problem is above the small-problem
        // BFGS cutoff, regardless of barrier presence.
        let barrier = BarrierConfig {
            tau: 1e-6,
            constrained_indices: vec![0],
            lower_bounds: vec![0.0],
        };
        let cap = OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            n_params: 20,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: Some(barrier),
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut i32, _: &Array1<f64>| Ok(1.0),
            eval_fn: |_: &mut i32, _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 1.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                })
            },
            eval_order_fn: None::<
                fn(&mut i32, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: Some(|st: &mut i32| {
                *st = 42;
            }),
            efs_fn: None::<fn(&mut i32, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        };
        assert_eq!(obj.capability().n_params, 1);
        assert_eq!(obj.eval_cost(&Array1::zeros(1)).unwrap(), 1.0);
    }

    #[test]
    fn hybrid_efs_backtracking_uses_half_step_after_first_rejection() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 12,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let mut obj = ClosureObjective {
            state: (),
            cap: cap.clone(),
            cost_fn: |_: &mut (), theta: &Array1<f64>| {
                let psi = theta[11];
                let cost = if (psi - 0.0).abs() < 1e-12 {
                    1.0
                } else if (psi - 0.5).abs() < 1e-12 {
                    0.5
                } else {
                    2.0
                };
                Ok(cost)
            },
            eval_fn: |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[11].abs(),
                    gradient: Array1::zeros(theta.len()),
                    hessian: HessianResult::Unavailable,
                })
            },
            eval_order_fn: None::<
                fn(&mut (), &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut ())>,
            efs_fn: Some(|_: &mut (), theta: &Array1<f64>| {
                let mut steps = vec![0.0; theta.len()];
                steps[11] = 1.0;
                Ok(EfsEval {
                    cost: 1.0,
                    steps,
                    beta: None,
                    psi_gradient: Some(array![1.0]),
                    psi_indices: Some(vec![11]),
                })
            }),
        };
        let mut bridge = OuterFixedPointBridge {
            obj: &mut obj,
            layout: cap.theta_layout(),
            barrier_config: None,
        };

        let sample = bridge
            .eval_step(&Array1::zeros(cap.n_params))
            .expect("hybrid EFS step should backtrack cleanly");

        assert_eq!(sample.status, FixedPointStatus::Continue);
        assert_eq!(sample.step.len(), cap.n_params);
        assert_eq!(sample.step[11], 0.5);
        assert!(
            sample
                .step
                .iter()
                .enumerate()
                .all(|(idx, &value)| idx == 11 || value == 0.0)
        );
    }

    #[test]
    fn run_bfgs_mode_aware_eval_skips_hessian_work() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_initial_rho(array![1.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run on BFGS".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Unavailable,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "mode-aware bfgs first order")
            .expect("BFGS should use the order-aware first-order bridge");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            !seen_orders.is_empty(),
            "mode-aware eval hook should have been used"
        );
        assert!(
            seen_orders
                .iter()
                .all(|order| *order == OuterEvalOrder::ValueAndGradient),
            "BFGS should request only value+gradient, saw {seen_orders:?}"
        );
    }

    #[test]
    fn outer_second_order_bridge_separates_first_and_second_order_requests() {
        let seen_orders = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic);
        let mut obj = problem.build_objective_with_eval_order(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Err(EstimationError::InvalidInput(
                    "legacy eager eval should not run".to_string(),
                ))
            },
            {
                let seen_orders = Arc::clone(&seen_orders);
                move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
                    seen_orders.lock().unwrap().push(order);
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: match order {
                            OuterEvalOrder::ValueAndGradient => HessianResult::Unavailable,
                            OuterEvalOrder::ValueGradientHessian => {
                                HessianResult::Analytic(array![[2.0]])
                            }
                        },
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut bridge = OuterSecondOrderBridge {
            obj: &mut obj,
            layout: OuterThetaLayout::new(1, 0),
            hessian_source: HessianSource::Analytic,
        };
        let grad_sample =
            FirstOrderObjective::eval_grad(&mut bridge, &array![1.0]).expect("grad eval");
        assert_eq!(grad_sample.value, 1.0);
        assert_eq!(grad_sample.gradient, array![2.0]);
        let hess_sample =
            SecondOrderObjective::eval_hessian(&mut bridge, &array![1.0]).expect("hessian eval");
        assert_eq!(hess_sample.value, 1.0);
        assert_eq!(hess_sample.gradient, array![2.0]);
        assert_eq!(hess_sample.hessian, Some(array![[2.0]]));
        let seen_orders = seen_orders.lock().unwrap();
        assert!(
            *seen_orders
                == vec![
                    OuterEvalOrder::ValueAndGradient,
                    OuterEvalOrder::ValueGradientHessian
                ],
            "second-order bridge should split first-order and second-order requests, saw {seen_orders:?}"
        );
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
        // When ψ (design-moving) coords are present and the problem is above
        // the small-problem BFGS cutoff, the planner should select HybridEfs
        // instead of falling back to BFGS.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 1,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::HybridEfs);
        assert_eq!(p.hessian_source, HessianSource::HybridEfsFixedPoint);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_few_params() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 5,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn plan_hybrid_efs_not_selected_with_analytic_hessian() {
        // Arc is always preferred when analytic Hessian is available,
        // even with ψ coordinates.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 20,
            psi_dim: 1,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
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
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Efs);
        assert_eq!(p.hessian_source, HessianSource::EfsFixedPoint);
    }

    #[test]
    fn automatic_fallbacks_only_degrade_hessian_not_gradient() {
        // The analytic gradient MUST NOT be silently degraded to FD; see
        // `automatic_fallback_attempts` for the production-safety rationale.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let attempts = automatic_fallback_attempts(&cap);
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert_eq!(attempts[0].hessian, Derivative::Unavailable);
    }

    #[test]
    fn automatic_fallbacks_from_efs_prefer_analytic_bfgs_over_fd() {
        // When the primary plan is EFS, the first fallback must keep the
        // analytic gradient and just disable the fixed-point path so the
        // planner picks gradient-based BFGS. Silently downgrading to finite
        // differences here was the long-standing production bug we are
        // guarding against.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::Efs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty(), "EFS failure must have a fallback");
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert_eq!(attempts[0].hessian, Derivative::Unavailable);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);

        // FD-gradient fallback was removed in production (silent BFGS-FD
        // spin was a footgun on large-n binomial biobank problems); the
        // cascade must stay on analytic-gradient attempts.
        assert!(
            attempts.iter().all(|c| c.gradient == Derivative::Analytic),
            "fallback cascade must not introduce a finite-difference gradient",
        );
    }

    #[test]
    fn automatic_fallbacks_from_hybrid_efs_prefer_analytic_bfgs_over_fd() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            psi_dim: 2,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        assert_eq!(plan(&cap).solver, Solver::HybridEfs);

        let attempts = automatic_fallback_attempts(&cap);
        assert!(!attempts.is_empty());
        assert_eq!(attempts[0].gradient, Derivative::Analytic);
        assert!(attempts[0].disable_fixed_point);
        assert_eq!(plan(&attempts[0]).solver, Solver::Bfgs);
    }

    #[test]
    fn plan_disable_fixed_point_forces_bfgs_even_when_efs_eligible() {
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 15,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        };
        let p = plan(&cap);
        assert_eq!(p.solver, Solver::Bfgs);
        assert_eq!(p.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn run_malformed_gradient_seed_surfaces_as_error() {
        // A capability that declares Analytic gradient but returns a malformed
        // one must fail loudly. The previous FD-fallback behaviour masked the
        // underlying bug by silently spinning a cost-only BFGS with
        // finite-difference gradient — that path is disabled in production.
        let problem = OuterProblem::new(2)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_initial_rho(Array1::zeros(2))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test gradient mismatch")
            .expect_err("malformed analytic gradient must surface as error");
        assert!(
            matches!(err, EstimationError::RemlOptimizationFailed(_)),
            "unexpected error variant: {err:?}",
        );
    }

    #[test]
    fn run_bfgs_ignores_malformed_hessian_payload() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    // First-order paths must ignore Hessian payload quality.
                    hessian: HessianResult::Analytic(array![[f64::NAN, 0.0], [0.0, 1.0]]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "bfgs should ignore malformed hessian payload")
            .expect("valid first-order data should be enough for BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn finite_outer_eval_reports_gradient_length_mismatch() {
        let err = finite_outer_eval_or_error(
            "test gradient mismatch",
            OuterThetaLayout::new(2, 0),
            OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianResult::Unavailable,
            },
        )
        .expect_err("gradient mismatch should be rejected");
        let message = match err {
            ObjectiveEvalError::Recoverable { message } | ObjectiveEvalError::Fatal { message } => {
                message
            }
        };
        assert!(
            message.contains("outer gradient length mismatch"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn run_with_initial_seed_still_considers_generated_candidates() {
        let generated = crate::seeding::generate_rho_candidates(
            1,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let expected_seed = valid_seed.clone();
        let initial_seed = array![9.0];
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(initial_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == &valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(f64::INFINITY)
                    }
                }
            },
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == &valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: Array1::zeros(1),
                        hessian: HessianResult::Unavailable,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "generated seed should remain reachable")
            .expect("generated seed should still be eligible when an initial seed is provided");
        assert_eq!(result.rho, expected_seed);
    }

    #[test]
    fn run_indefinite_analytic_seed_stays_on_arc() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Analytic(array![[-1.0]]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "indefinite seed geometry")
            .expect("indefinite analytic seed geometry should stay on the second-order plan");
        assert_eq!(result.plan_used.solver, Solver::Arc);
        assert_eq!(result.plan_used.hessian_source, HessianSource::Analytic);
    }

    #[test]
    fn run_seed_materialization_failure_falls_back_from_arc() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_seed_config(seed_config)
            .with_initial_rho(array![0.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: array![0.0],
                    hessian: HessianResult::Operator(Arc::new(
                        FailingSeedMaterializationOperator { dim: 1 },
                    )),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "seed materialization failure")
            .expect("materialization failure should trigger the degraded fallback plan");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn run_nonconverged_arc_falls_back_to_first_order_plan() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_seed_config(seed_config)
            .with_initial_rho(array![5.0])
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(theta[0].powi(4)),
            |_: &mut (), theta: &Array1<f64>| {
                let x = theta[0];
                Ok(OuterEval {
                    cost: x.powi(4),
                    gradient: array![4.0 * x.powi(3)],
                    hessian: HessianResult::Analytic(array![[12.0 * x.powi(2)]]),
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "nonconverged arc should fall back")
            .expect("automatic fallback should continue after a nonconverged Arc attempt");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(result.plan_used.hessian_source, HessianSource::BfgsApprox);
    }

    #[test]
    fn candidate_selection_prefers_lower_cost_within_same_convergence_class() {
        let nonconverged_hi = OuterResult {
            rho: array![0.0],
            final_value: 9.0,
            iterations: 1,
            final_grad_norm: 1.0,
            final_gradient: None,
            final_hessian: None,
            converged: false,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        };
        let nonconverged_lo = OuterResult {
            rho: array![1.0],
            final_value: 1.0,
            iterations: 1,
            final_grad_norm: 1.0,
            final_gradient: None,
            final_hessian: None,
            converged: false,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        };
        let converged = OuterResult {
            rho: array![2.0],
            final_value: 5.0,
            iterations: 1,
            final_grad_norm: 0.0,
            final_gradient: None,
            final_hessian: None,
            converged: true,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        };

        assert!(candidate_improves_best(&nonconverged_hi, None));
        assert!(candidate_improves_best(
            &nonconverged_lo,
            Some(&nonconverged_hi)
        ));
        assert!(!candidate_improves_best(
            &nonconverged_hi,
            Some(&nonconverged_lo)
        ));
        assert!(candidate_improves_best(&converged, Some(&nonconverged_lo)));
        assert!(!candidate_improves_best(&nonconverged_lo, Some(&converged)));
    }

    #[test]
    fn run_starts_solver_with_direct_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("cost");
                    Ok(theta[0] * theta[0])
                }
            },
            {
                let calls = Arc::clone(&calls);
                move |_: &mut (), theta: &Array1<f64>| {
                    calls.lock().unwrap().push("eval");
                    Ok(OuterEval {
                        cost: theta[0] * theta[0],
                        gradient: array![2.0 * theta[0]],
                        hessian: HessianResult::Analytic(array![[2.0]]),
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        problem
            .run(&mut obj, "solver should start from a direct startup eval")
            .expect("analytic plans should start with a direct full evaluation");
        let calls = calls.lock().unwrap();
        let first_eval_idx = calls
            .iter()
            .position(|call| *call == "eval")
            .expect("solver should eventually request a full eval");
        assert!(
            first_eval_idx == 0,
            "startup should not perform a separate cost-screening pass first: {calls:?}"
        );
    }

    #[test]
    fn run_screening_reorders_expensive_seeds_before_full_startup_eval() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        seed_config.risk_profile = crate::seeding::SeedRiskProfile::GeneralizedLinear;
        let screening_cap = Arc::new(AtomicUsize::new(0));
        let initial_seed = array![9.0];
        let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let started = Arc::new(Mutex::new(Vec::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_seed_config(seed_config)
            .with_screening_cap(Arc::clone(&screening_cap))
            .with_initial_rho(initial_seed.clone())
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let valid_seed = valid_seed.clone();
                move |_: &mut (), theta: &Array1<f64>| {
                    if theta == &valid_seed {
                        Ok(0.0)
                    } else {
                        Ok(1000.0)
                    }
                }
            },
            {
                let valid_seed = valid_seed.clone();
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    if theta == &valid_seed {
                        Ok(OuterEval {
                            cost: 0.0,
                            gradient: array![0.0],
                            hessian: HessianResult::Analytic(array![[1.0]]),
                        })
                    } else {
                        Ok(OuterEval::infeasible(theta.len()))
                    }
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "screening should reorder expensive seeds")
            .expect("screened startup should reach the best generated seed");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(
            started.lock().unwrap().first().cloned(),
            Some(valid_seed),
            "screening should move the lowest-cost seed to the front before full startup eval",
        );
        assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn run_efs_skips_global_cost_screening() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.max_seeds = 6;
        seed_config.seed_budget = 1;
        let screening_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Unavailable)
            .with_hessian(Derivative::Unavailable)
            .with_seed_config(seed_config)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            {
                let screening_calls = Arc::clone(&screening_calls);
                move |_: &mut (), _: &Array1<f64>| {
                    screening_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(0.0)
                }
            },
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            Some(|_: &mut (), theta: &Array1<f64>| {
                Ok(EfsEval {
                    cost: 0.0,
                    steps: vec![0.0; theta.len()],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                })
            }),
        );
        problem
            .run(
                &mut obj,
                "EFS should not use a separate global cost-screening pass",
            )
            .expect("first generated EFS seed should be sufficient");
        assert_eq!(
            screening_calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "EFS startup should not call eval_cost just to screen seeds"
        );
    }

    #[test]
    fn run_efs_skips_invalid_leading_seed_without_spending_budget() {
        let generated = crate::seeding::generate_rho_candidates(
            15,
            None,
            &crate::seeding::SeedConfig::default(),
        );
        let valid_seed = generated
            .first()
            .expect("seed generator should yield at least one candidate")
            .clone();
        let invalid_seed = Array1::from_elem(15, 9.0);
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 1;
        let problem = OuterProblem::new(15)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(invalid_seed)
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), theta: &Array1<f64>| Ok(OuterEval::infeasible(theta.len())),
            None::<fn(&mut ())>,
            {
                let valid_seed = valid_seed.clone();
                Some(move |_: &mut (), theta: &Array1<f64>| {
                    if theta == &valid_seed {
                        Ok(EfsEval {
                            cost: 0.0,
                            steps: vec![0.0; theta.len()],
                            beta: None,
                            psi_gradient: None,
                            psi_indices: None,
                        })
                    } else {
                        Err(EstimationError::RemlOptimizationFailed(
                            "invalid EFS seed".to_string(),
                        ))
                    }
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs generated seed should remain reachable")
            .expect("invalid startup seeds should not consume the only EFS seed slot");
        assert_eq!(result.rho, valid_seed);
        assert_eq!(result.plan_used.solver, Solver::Efs);
    }

    #[test]
    fn run_efs_runtime_fallback_marker_degrades_to_bfgs_immediately() {
        let mut seed_config = crate::seeding::SeedConfig::default();
        seed_config.seed_budget = 2;
        let efs_calls = Arc::new(AtomicUsize::new(0));
        let problem = OuterProblem::new(12)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_seed_config(seed_config)
            .with_initial_rho(Array1::zeros(12))
            .with_max_iter(5);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianResult::Unavailable,
                })
            },
            None::<fn(&mut ())>,
            {
                let efs_calls = Arc::clone(&efs_calls);
                Some(move |_: &mut (), _: &Array1<f64>| {
                    efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{} synthetic runtime escape hatch",
                        EFS_FIRST_ORDER_FALLBACK_MARKER,
                    )))
                })
            },
        );
        let result = problem
            .run(&mut obj, "efs runtime fallback marker")
            .expect("runtime EFS escape hatch should degrade to BFGS");
        assert_eq!(result.plan_used.solver, Solver::Bfgs);
        assert_eq!(
            efs_calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "runtime fallback marker should abort the EFS attempt immediately"
        );
    }

    #[test]
    fn run_rejects_invalid_theta_layout() {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_psi_dim(2)
            .with_initial_rho(Array1::zeros(1))
            .with_max_iter(1);
        let mut obj = problem.build_objective(
            (),
            |_: &mut (), _: &Array1<f64>| Ok(0.0),
            |_: &mut (), _: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.0,
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let err = problem
            .run(&mut obj, "test invalid layout")
            .expect_err("invalid theta layout should fail cleanly");
        assert!(
            err.to_string().contains("invalid outer theta layout"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn effective_seed_budget_caps_expensive_solver_retries() {
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::Efs,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                4,
                Solver::HybridEfs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::GeneralizedLinear,
                true,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Arc,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            1
        );
        assert_eq!(
            effective_seed_budget(
                3,
                Solver::Bfgs,
                crate::seeding::SeedRiskProfile::Survival,
                false,
            ),
            3
        );
    }

    // ─── Gated SolverClass / CompassSearch dispatch ──────────────────────

    fn aux_cap_unavailable(n_params: usize) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Unavailable,
            hessian: Derivative::Unavailable,
            n_params,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    #[test]
    fn plan_with_class_primary_is_identical_to_plan_for_unavailable_grad() {
        // Unavailable-grad capability must still route to BFGS for the
        // primary class (the runner will then hard-error on the missing
        // analytic gradient — that behavior is tested elsewhere).
        let cap = aux_cap_unavailable(3);
        assert_eq!(plan_with_class(&cap, SolverClass::Primary), plan(&cap));
    }

    #[test]
    fn plan_with_class_aux_unavailable_routes_to_compass_search() {
        let cap = aux_cap_unavailable(3);
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::CompassSearch);
    }

    #[test]
    fn plan_with_class_aux_analytic_grad_defers_to_primary_plan() {
        // Aux class + analytic gradient is a misuse: the caller should
        // have used Primary. We defer to the standard plan so the caller
        // still gets a well-formed result rather than silently being
        // routed to direct search when a derivative-based solver exists.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 3,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::Arc);
    }

    #[test]
    fn plan_with_class_aux_efs_eligible_defers_to_primary() {
        // If the coordinate structure is EFS-eligible, use EFS even if
        // the caller set Auxiliary — EFS is strictly better than compass
        // search whenever it applies.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Unavailable,
            n_params: 12,
            psi_dim: 0,
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let p = plan_with_class(&cap, SolverClass::AuxiliaryGradientFree);
        assert_eq!(p.solver, Solver::Efs);
    }

    #[test]
    fn automatic_fallback_never_includes_compass_search() {
        // The fallback cascade must not introduce direct-search for the
        // primary REML path. Aux direct-search is a single-attempt
        // method; its dispatch is orthogonal to the fallback ladder.
        let cap = OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: 5,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        };
        let attempts = automatic_fallback_attempts(&cap);
        for attempt_cap in &attempts {
            let p = plan_with_class(attempt_cap, SolverClass::Primary);
            assert_ne!(p.solver, Solver::CompassSearch);
        }
    }

    #[test]
    fn compass_search_budget_accounts_for_single_seed() {
        // Aux direct-search is intrinsically a single-seed local method;
        // generating extra seeds would just duplicate cost.
        let b = effective_seed_budget(
            8,
            Solver::CompassSearch,
            crate::seeding::SeedRiskProfile::Survival,
            false,
        );
        assert_eq!(b, 1);
    }
}
