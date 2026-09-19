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

    /// Does outer coordinate `k` parameterize a smoothing parameter as
    /// `λ = e^ρ`?
    ///
    /// The θ vector is one contiguous array under one box, and every layer
    /// that reasons about a *bound* — the projector, the active set, the PSD
    /// sub-block — is right to treat all of it alike: a box constraint is a
    /// statement about a coordinate, not about what the coordinate means.
    ///
    /// The asymptote/tail law is not that kind of rule. `ĉ = ∓e^{±ρ}·∂V/∂ρ`
    /// is derived from `λ = e^ρ` together with the REML criterion's `O(1/λ)`
    /// behaviour, and it exists because the ρ box is a **proxy** for a limit
    /// the search can never actually reach: `λ = ∞` is at `ρ = ∞`, so the
    /// remaining gap to the optimum has to be extrapolated rather than
    /// observed. A ψ coordinate is a different quantity — a sectional
    /// curvature, a log length-scale, a measure exponent — and its box is a
    /// **real** constraint (chart validity, `s ∈ (0, 2)`) whose endpoint a
    /// point can sit exactly on. There the ordinary complementarity the
    /// projector already applies IS the first-order certificate, and
    /// exponentiating the coordinate has no derivation at all (#2453).
    ///
    /// Answering this question by position is exactly what makes it a
    /// question about the quantity: `psi_dim` is declared by the call site
    /// that knows what it put in each slot, and the trailing block is the ψ
    /// block by construction.
    pub(crate) const fn coordinate_is_log_smoothing(&self, k: usize) -> bool {
        k < self.rho_dim()
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
            return Err(ObjectiveEvalError::fatal(format!(
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
            return Err(ObjectiveEvalError::fatal(format!(
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
            return Err(ObjectiveEvalError::fatal(format!(
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
    /// Reserve analytic Hessian work for the terminal mint certificate.
    ///
    /// The generic REML/LAML Hessian consumes the row-family derivative ladder
    /// through order four, while its analytic gradient stops at order three.
    /// When this is true, search therefore uses analytic-gradient BFGS and the
    /// exact Hessian remains declared for the one terminal
    /// `ValueGradientHessian` certification. Closed-form objectives whose
    /// Hessian does not consume an order-four family tower may set this false
    /// and use ARC during search.
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
    pub(crate) const fn theta_layout(&self) -> OuterThetaLayout {
        OuterThetaLayout::new(self.n_params, self.psi_dim)
    }

    pub fn validate_layout(&self, context: &str) -> Result<(), EstimationError> {
        self.theta_layout().validate_capability(context)
    }

    /// True when all coordinates are penalty-like (no ψ coords).
    pub(crate) const fn all_penalty_like(&self) -> bool {
        self.psi_dim == 0
    }
    /// True when ψ (design-moving) coordinates are present.
    pub(crate) const fn has_psi_coords(&self) -> bool {
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
    /// Curvature used by the search algorithm, not by the mandatory terminal
    /// certificate. A gradient-only BFGS search can still be audited by the
    /// objective's exact analytic Hessian at mint time (#2359, #2641).
    pub hessian_source: HessianSource,
}

/// A fixed-point evaluator's typed request to hand the current outer attempt
/// to a joint first-order solver.
///
/// This is control flow, not a trial-point failure category. It therefore
/// travels as the source of an [`ObjectiveEvalError`] and is recognized by
/// downcast; neither its wording nor any sentinel embedded in that wording can
/// change which plan runs next.
#[derive(Clone, Debug)]
pub(crate) struct FirstOrderFallbackRequest {
    reason: String,
    /// Iterations the plan attempt had spent when the request left it: every
    /// seed it started before the one whose evaluation asked (#2817).
    spent_iterations: usize,
}

impl FirstOrderFallbackRequest {
    pub(crate) fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            spent_iterations: 0,
        }
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }

    pub(crate) fn with_spent_iterations(mut self, spent_iterations: usize) -> Self {
        self.spent_iterations = spent_iterations;
        self
    }

    pub(crate) fn spent_iterations(&self) -> usize {
        self.spent_iterations
    }
}

impl std::fmt::Display for FirstOrderFallbackRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for FirstOrderFallbackRequest {}

pub(crate) fn first_order_fallback_error(reason: impl Into<String>) -> ObjectiveEvalError {
    ObjectiveEvalError::recoverable_from(FirstOrderFallbackRequest::new(reason))
}

pub(crate) fn first_order_fallback_request(
    error: &ObjectiveEvalError,
) -> Option<&FirstOrderFallbackRequest> {
    error.downcast_ref::<FirstOrderFallbackRequest>()
}

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
            "solver={:?}, search_hessian_source={:?}",
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
    pub(crate) fn routing_log_line(&self) -> String {
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
        (D::Analytic, D::Analytic) if cap.prefer_gradient_only => OuterPlan {
            solver: S::Bfgs,
            hessian_source: H::BfgsApprox,
        },
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
pub(crate) fn log_plan(context: &str, cap: &OuterCapability, the_plan: &OuterPlan) {
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
    log::debug!(
        "[OUTER] {context}: n_params={}, gradient={:?}, hessian={:?} -> {} [{}]{hess_warning}{barrier_note}{hybrid_note}",
        cap.n_params,
        cap.gradient,
        cap.hessian,
        the_plan,
        the_plan.routing_log_line(),
    );
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
    //   3. If the primary plan is BFGS only because the capability prefers
    //      gradient-only search over a declared analytic Hessian (#2359), retry
    //      with that preference cleared, i.e. ARC on the declared curvature.
    //   4. Otherwise (e.g. (Analytic, Unavailable) without EFS eligibility,
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

    // Gradient-only primary over a declared analytic Hessian: the search keeps
    // order four out of every fit it certifies, and a search that exhausts
    // without claiming convergence gets the declared curvature once before the
    // refusal surfaces (#2898). A positive-definite secant model cannot follow
    // negative curvature, which a nonconvex criterion such as the Firth-armed
    // multinomial refit presents along its search path (#2627).
    if cap.prefer_gradient_only
        && cap.gradient == Derivative::Analytic
        && cap.declared_hessian_for_planning() == Derivative::Analytic
    {
        let mut exact_curvature = cap.clone();
        exact_curvature.prefer_gradient_only = false;
        attempts.push(exact_curvature);
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
        // surfaces a typed `FirstOrderFallbackRequest` so the runner can switch
        // to a joint gradient solver that enforces ∇ψ V = 0. With fallback
        // disabled and an analytic gradient available, selecting HybridEFS as
        // the only primary attempt is internally inconsistent; BFGS is the
        // standalone first-order primary for that capability.
        log::debug!(
            "[OUTER] {context}: HybridEFS requires the automatic first-order \
             escape path for ψ coordinates; fallback is disabled, so routing the \
             primary attempt to analytic-gradient BFGS"
        );
        cap.disable_fixed_point = true;
    }
    if config.curvature_search_latched
        && cap.prefer_gradient_only
        && cap.gradient == Derivative::Analytic
        && cap.declared_hessian_for_planning() == Derivative::Analytic
    {
        // A positive-definite secant model cannot follow the negative curvature
        // the mint just measured, and restarted inside its own gradient band it
        // stops at iteration 0 (#2939). The declared Hessian takes over the
        // search for the rest of this solve.
        log::debug!(
            "[OUTER] {context}: a certified strict saddle latched the declared analytic \
             Hessian into the search; planning ARC instead of gradient-only BFGS (#2939)"
        );
        cap.prefer_gradient_only = false;
    }
    cap
}

#[cfg(test)]
mod typed_fallback_tests {
    use super::*;

    #[test]
    fn first_order_fallback_routing_depends_on_source_type_not_message_2658() {
        let typed = first_order_fallback_error("arbitrary human-readable reason");
        let request =
            first_order_fallback_request(&typed).expect("typed request must survive opt boundary");
        assert_eq!(request.reason(), "arbitrary human-readable reason");

        let marker_shaped_prose = ObjectiveEvalError::recoverable(
            "[outer-efs-first-order-fallback] legacy-looking prose",
        );
        assert!(
            first_order_fallback_request(&marker_shaped_prose).is_none(),
            "prose must have no authority to select a solver"
        );
    }
}
