use super::*;

/// Declares what a specific model path can provide to the outer optimizer.
///
/// Each call site that optimizes smoothing parameters constructs one of these
/// to describe its analytic derivative coverage. The [`plan`] function then
/// selects the optimizer and Hessian strategy.
///
/// HISTORY: this crossover used to be 8 — a "performance choice" that routed
/// every small-dimensional problem WITH an analytic gradient to BFGS on the
/// theory that a dense quasi-Newton is cheaper below the cutoff. On the
/// criteria that actually fail (the SAE manifold Laplace evidence: 2–7 ρ
/// coordinates, piecewise-smooth basin-envelope value, inner-solve truncation
/// noise), BFGS is not cheaper — its Strong-Wolfe line search is the consumer
/// of the entire probe-lane / wall / escape / rescue apparatus, and every cost
/// probe is a full inner re-convergence. EFS is the declared canonical REML
/// method, needs only the traces `tr(H⁻¹S_k)` (no line search, no Wolfe, no
/// value/gradient-lane agreement), and already drives every large fit. The
/// crossover is therefore 0: a fixed-point-capable objective routes to
/// EFS/HybridEfs at EVERY dimension, and BFGS remains the fallback for
/// objectives with no fixed-point hook (or after `disable_fixed_point`).
pub(crate) const SMALL_OUTER_BFGS_MAX_PARAMS: usize = 0;

pub(crate) const SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterThetaLayout {
    pub n_params: usize,
    pub psi_dim: usize,
}

impl OuterThetaLayout {
    pub const fn new(n_params: usize, psi_dim: usize) -> Self {
        Self { n_params, psi_dim }
    }

    pub const fn rho_dim(&self) -> usize {
        self.n_params.saturating_sub(self.psi_dim)
    }

    fn validate_capability(&self, context: &str) -> Result<(), EstimationError> {
        if self.psi_dim > self.n_params {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "{context}: invalid outer theta layout (psi_dim={} exceeds n_params={})",
                self.psi_dim, self.n_params
            )));
        }
        Ok::<(), _>(())
    }

    pub(crate) fn validate_point_len(
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
        Ok::<(), _>(())
    }

    pub(crate) fn validate_gradient_len(
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
        Ok::<(), _>(())
    }

    pub(crate) fn validate_hessian_shape(
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
        Ok::<(), _>(())
    }

    pub(crate) fn validate_efs_eval(
        &self,
        eval: &EfsEval,
        context: &str,
    ) -> Result<(), ObjectiveEvalError> {
        if eval.steps.len() != self.n_params {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer EFS step length mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                eval.steps.len(),
                self.n_params,
                self.rho_dim(),
                self.psi_dim
            )));
        }
        if let Some(ref psi_gradient) = eval.psi_gradient
            && psi_gradient.len() != self.psi_dim
        {
            return Err(ObjectiveEvalError::recoverable(format!(
                "{context}: outer EFS psi-gradient length mismatch: got {}, expected {}",
                psi_gradient.len(),
                self.psi_dim
            )));
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
    /// Declared shape of the analytic Hessian (or its absence). Replaces
    /// the binary `Derivative` so the planner can route between dense
    /// ARC and matrix-free trust-region *before* seed evaluation. See
    /// [`DeclaredHessianForm`].
    pub hessian: DeclaredHessianForm,
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
    /// Enabled when `psi_dim > 0`, `fixed_point_available`, and either the
    /// analytic gradient is unavailable or the problem is above the small-
    /// dimensional BFGS crossover.
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
    /// is abandoned and the fallback ladder routes to a first-order joint
    /// optimizer.
    ///
    /// Previously this was a binary `barrier_active: bool` that unconditionally
    /// blocked EFS. The quantitative check allows EFS when constraints exist but
    /// the barrier curvature is negligible (coefficients far from their bounds).
    pub barrier_config: Option<BarrierConfig>,
    /// Policy hint for derivative-free auxiliary optimizers only. Primary REML
    /// optimization ignores this flag when an analytic Hessian exists: exact
    /// second-order geometry must not be hidden behind a quasi-Newton policy.
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
    pub const fn theta_layout(&self) -> OuterThetaLayout {
        OuterThetaLayout::new(self.n_params, self.psi_dim)
    }

    pub fn validate_layout(&self, context: &str) -> Result<(), EstimationError> {
        self.theta_layout().validate_capability(context)
    }

    /// True when all coordinates are penalty-like (no ψ coords).
    pub const fn all_penalty_like(&self) -> bool {
        self.psi_dim == 0
    }
    /// True when ψ (design-moving) coordinates are present.
    pub const fn has_psi_coords(&self) -> bool {
        self.psi_dim > 0
    }

    fn efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.all_penalty_like()
            // A fixed-point-capable objective routes to EFS at every dimension
            // (see `SMALL_OUTER_BFGS_MAX_PARAMS`): the former ≤8-coordinate
            // BFGS crossover sent exactly the failing small fits into the
            // fragile Wolfe/probe lane while large fits got the robust
            // trace-based fixed point.
            && (self.gradient == Derivative::Unavailable
                || self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS)
    }

    fn hybrid_efs_plan_eligible(&self) -> bool {
        self.fixed_point_available
            && !self.disable_fixed_point
            && self.has_psi_coords()
            && (self.gradient == Derivative::Unavailable
                || self.n_params > SMALL_OUTER_BFGS_MAX_PARAMS)
    }

    fn declared_hessian_for_planning(&self) -> Derivative {
        if self.hessian.is_analytic() {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        }
    }
}

/// Which solver algorithm to use for the outer optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Adaptive Regularized Cubic; fastest convergence, requires Hessian.
    Arc,
    /// BFGS; gradient only, builds a dense curvature approximation.
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
/// This enum is for the shared `eval` bridge where the runner needs value-only,
/// first-order, or second-order information depending on the active plan.
///
/// Single-sourced on the lower `gam-model-api` crate so the gam-models
/// fit_orchestration drivers and the gam-solve runner share one type (#1521).
pub use gam_model_api::OuterEvalOrder;

/// The outer optimization plan. Produced by [`plan`], consumed by the runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterPlan {
    pub solver: Solver,
    pub hessian_source: HessianSource,
}

pub(crate) const EFS_FIRST_ORDER_FALLBACK_MARKER: &str = "[outer-efs-first-order-fallback]";

/// Whether outer_strategy should automatically derive a retry ladder from the
/// primary capability, or disable retries entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Centralized retry path chosen from the declared capability.
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

impl OuterPlan {
    /// Stable, grep-friendly routing token for large-scale/log regression
    /// assertions. Emits `solver=<Solver>;hessian=<Source>;matrix-free=<bool>`.
    /// Planning alone does not prove the runtime Hessian representation;
    /// matrix-free routing is decided after the seed evaluation returns an
    /// operator Hessian, so the static plan token reports `false`.
    pub fn routing_log_line(&self) -> String {
        let matrix_free = false;
        format!(
            "solver={:?};hessian={:?};matrix-free={}",
            self.solver, self.hessian_source, matrix_free
        )
    }
}

/// Select the outer optimization strategy from the declared capability.
///
/// This is a pure function with no side effects. All policy lives here.
pub fn plan(cap: &OuterCapability) -> OuterPlan {
    use Derivative as D;
    use HessianSource as H;
    use Solver as S;

    match (cap.gradient, cap.declared_hessian_for_planning()) {
        (D::Analytic, D::Analytic) => OuterPlan {
            solver: S::Arc,
            hessian_source: H::Analytic,
        },
        // EFS: all penalty-like coords and no analytic Hessian. With an
        // analytic gradient this is the many-parameter fast path; without one
        // it is the only declared analytic solver at any dimension.
        // Multiplicative fixed-point needs only traces — no gradient evals.
        // Much cheaper than BFGS for k=10-50 smoothing parameters.
        //
        // When a log-barrier is present (monotonicity constraints), EFS is
        // still selected here. The EFS iteration loop in `run_outer` performs
        // a quantitative check each step via `barrier_curvature_is_significant`
        // and bails out early if the barrier curvature becomes non-negligible
        // relative to the penalized Hessian diagonal.
        (D::Analytic, D::Unavailable) if cap.efs_plan_eligible() => OuterPlan {
            solver: S::Efs,
            hessian_source: H::EfsFixedPoint,
        },
        (D::Unavailable, D::Unavailable) if cap.efs_plan_eligible() => OuterPlan {
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
        (D::Analytic, D::Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },
        (D::Unavailable, D::Unavailable) if cap.hybrid_efs_plan_eligible() => OuterPlan {
            solver: S::HybridEfs,
            hessian_source: H::HybridEfsFixedPoint,
        },

        // Gradient-only problems should use a gradient-only optimizer.
        (D::Analytic, D::Unavailable) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
        // No analytic gradient (with or without a declared Hessian), and the
        // EFS/HybridEFS fixed-point lane ruled out above. Every outer objective
        // in the tree now supplies an analytic gradient, so a cost-only
        // capability is a programming error. Emit a BFGS plan so it surfaces
        // loudly with context: the runner rejects it because BFGS requires the
        // analytic gradient this capability declares is absent. We deliberately
        // do NOT invent a working primary here — a cost-only objective has no
        // solver, by design.
        (D::Unavailable, _) => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
    }
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
    // Promoted to info: this fires once per outer optimization dispatch and
    // tells the user immediately whether ARC, BFGS, EFS, etc. was selected
    // and why. That information is otherwise inferred only from the per-iter
    // log tag prefix once the loop has started.
    log::info!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {} [{}]{hess_warning}{barrier_note}{hybrid_note}",
        cap.n_params,
        cap.gradient,
        cap.hessian,
        the_plan,
        the_plan.routing_log_line(),
    );
}

pub(crate) fn requests_immediate_first_order_fallback(message: &str) -> bool {
    message.contains(EFS_FIRST_ORDER_FALLBACK_MARKER)
}

/// Disable the EFS/HybridEfs planner path, forcing BFGS-class solvers on the
/// next attempt. Returns `None` if fixed-point is already disabled.
pub(crate) fn disable_fixed_point(cap: &OuterCapability) -> Option<OuterCapability> {
    (!cap.disable_fixed_point && (cap.efs_plan_eligible() || cap.hybrid_efs_plan_eligible())).then(
        || {
            let mut degraded = cap.clone();
            degraded.disable_fixed_point = true;
            degraded
        },
    )
}

pub(crate) fn automatic_fallback_attempts(cap: &OuterCapability) -> Vec<OuterCapability> {
    // Production fallback ladder is strictly analytic-gradient.
    //
    // The cascade is:
    //   1. If the primary plan is EFS/HybridEFS AND an analytic gradient is
    //      available, retry with fixed-point disabled so the analytic
    //      derivative declaration is evaluated directly.
    //   2. If the primary plan is Arc (declared (Analytic, Analytic)
    //      capability), do NOT add a degraded fallback. Demoting to
    //      BFGS+BfgsApprox in this case discards the analytic outer Hessian
    //      ARC was using — a strictly weaker geometry — and silently masks
    //      ARC's actual failure mode (e.g. budget exhaustion, indefinite
    //      curvature) under a BFGS Strong-Wolfe plateau on a flat surface.
    //      ARC retries are handled by the per-attempt budget-bump retry
    //      ladder in `run_outer_with_strategy`; once that is exhausted, the
    //      caller surfaces the underlying ARC failure verbatim.
    //   3. Otherwise (e.g. (Analytic, Unavailable) without EFS eligibility,
    //      which is the BFGS primary), there is nothing to degrade further
    //      — the caller surfaces the RemlOptimizationFailed error so the
    //      non-convergence is visible.
    let mut attempts = Vec::new();

    if cap.gradient == Derivative::Analytic
        && matches!(plan(cap).solver, Solver::Efs | Solver::HybridEfs)
        && let Some(no_fp_cap) = disable_fixed_point(cap)
    {
        attempts.push(no_fp_cap.clone());
        return attempts;
    }

    // Arc primary: no lateral demotion to BFGS. The runner's ARC-budget-bump
    // retry covers cases where ARC needed more iterations; if even that is
    // exhausted, the caller sees the genuine analytic-Hessian non-convergence
    // rather than a misleading BFGS-on-flat-surface plateau.
    if matches!(plan(cap).solver, Solver::Arc) {
        return attempts;
    }

    attempts
}

pub(crate) fn disabled_fallback_hybrid_efs_has_standalone_bfgs_primary(
    cap: &OuterCapability,
    config: &OuterConfig,
) -> bool {
    config.fallback_policy == FallbackPolicy::Disabled
        && cap.gradient == Derivative::Analytic
        && matches!(plan(cap).solver, Solver::HybridEfs)
}

pub(crate) fn primary_capability_for_config(
    mut cap: OuterCapability,
    config: &OuterConfig,
    context: &str,
) -> OuterCapability {
    if disabled_fallback_hybrid_efs_has_standalone_bfgs_primary(&cap, config) {
        // HybridEFS is not a standalone first-order method for ψ coordinates:
        // when ψ backtracking proves non-descent, the bridge intentionally
        // surfaces `EFS_FIRST_ORDER_FALLBACK_MARKER` so the runner can switch
        // to a joint gradient solver that enforces ∇ψ V = 0. With fallback
        // disabled and an analytic gradient available, selecting HybridEFS as
        // the only primary attempt is internally inconsistent; BFGS is the
        // standalone first-order primary for that capability.
        log::info!(
            "[OUTER] {context}: HybridEFS requires the automatic first-order \
             escape path for ψ coordinates; fallback is disabled, so routing the \
             primary attempt to analytic-gradient BFGS"
        );
        cap.disable_fixed_point = true;
    }
    cap
}
