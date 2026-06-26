//! Fit-time configuration and cost accounting: `BlockwiseFitOptions`, the
//! outer-derivative policy + order selection, coefficient cost models, and the
//! argument-validation asserts shared by the solver entry points.

use crate::families::custom_family::family_trait::{CustomFamily, OuterEvalContext};
use crate::families::custom_family::psi_design::{
    CustomFamilyBlockPsiDerivative, ExactNewtonJointHessianWorkspace,
};
use gam_linalg::RidgePolicy;
use gam_problem::{CustomFamilyError, ParameterBlockSpec, ParameterBlockState};
use ndarray::Array1;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

// Moved to `gam-problem` (#1521 CustomFamily-cone inversion): the neutral,
// dependency-free outer-objective + exact-derivative-order capability enums now
// live in `gam_problem::family_options` and are re-exported here so every
// `custom_family::ExactNewtonOuterObjective` / `ExactOuterDerivativeOrder` path
// keeps resolving byte-for-byte.
pub use gam_problem::{ExactNewtonOuterObjective, ExactOuterDerivativeOrder};

pub const CUSTOM_FAMILY_WEIGHT_FLOOR: f64 = 1e-12;
pub const CUSTOM_FAMILY_RIDGE_FLOOR: f64 = 1e-12;

pub(crate) fn validate_blockspec_consistency(
    specs: &[ParameterBlockSpec],
) -> Result<Vec<usize>, String> {
    let mut seen_names = BTreeMap::<String, usize>::new();
    for (b, spec) in specs.iter().enumerate() {
        if let Some(prev) = seen_names.insert(spec.name.clone(), b) {
            return Err(CustomFamilyError::ConstraintViolation {
                reason: format!(
                    "duplicate parameter block name '{}' at indices {prev} and {b}: block names must be unique so coefficient labels resolved by name are unambiguous",
                    spec.name
                ),
            }
            .into());
        }
    }
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let n = spec.design.nrows();
        if spec.offset.len() != n {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} offset length mismatch: got {}, expected {}",
                    spec.offset.len(),
                    n
                ),
            }
            .into());
        }
        match (&spec.stacked_design, &spec.stacked_offset) {
            (Some(sd), Some(so)) => {
                if sd.nrows() != so.len() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design/stacked_offset row mismatch: \
                             stacked_design.nrows()={}, stacked_offset.len()={}",
                            sd.nrows(),
                            so.len(),
                        ),
                    }
                    .into());
                }
                if sd.ncols() != spec.design.ncols() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design column count {} disagrees with \
                             design column count {}",
                            sd.ncols(),
                            spec.design.ncols(),
                        ),
                    }
                    .into());
                }
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {b} stacked_design and stacked_offset must be Some together \
                         or both None"
                    ),
                }
                .into());
            }
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta
            && beta0.len() != p
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_beta length mismatch: got {}, expected {p}",
                    beta0.len()
                ),
            }
            .into());
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_log_lambdas length {} does not match penalties {}",
                    spec.initial_log_lambdas.len(),
                    spec.penalties.len()
                ),
            }
            .into());
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!("block {b} penalty {k} must be {p}x{p}, got {r}x{c}"),
                }
                .into());
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}

/// Exact outer derivative order for families that expose second-order
/// coefficient geometry.
///
/// This used to be a cost gate that demoted large large-scale problems to
/// first-order BFGS. That was a policy leak into the math layer: if the family
/// supplies analytic dense Hessian blocks or an analytic profiled-Hessian HVP,
/// the outer optimizer should see the exact second-order objective. Runtime
/// representation choices (dense vs operator) belong below this declaration,
/// not in a first-order downgrade.
/// Precondition check for the family capability / operator hooks (e.g.
/// `batched_outer_hessian_terms`, `outer_hyper_hessian_operator`).
///
/// These hooks operate on whatever block geometry the caller has assembled and
/// must validate the *consistency* of the specs they are handed — never the
/// fit-level "at least one block" precondition, which belongs to the fit entry
/// points (`validate_blockspecs`). An empty, self-consistent argument set is a
/// valid no-op probe of the operator path (the operator may ignore the specs
/// entirely), so it must not panic here.
pub(crate) fn assert_valid_blockspecs(specs: &[ParameterBlockSpec], context: &str) {
    assert!(
        validate_blockspec_consistency(specs).is_ok(),
        "{context}: inconsistent parameter block specs"
    );
}

pub(crate) fn assert_valid_options(options: &BlockwiseFitOptions, context: &str) {
    assert!(
        options.inner_tol.is_finite() && options.inner_tol >= 0.0,
        "{context}: inner_tol must be finite and non-negative"
    );
    assert!(
        options.outer_tol.is_finite() && options.outer_tol >= 0.0,
        "{context}: outer_tol must be finite and non-negative"
    );
    assert!(
        options.minweight.is_finite() && options.minweight >= 0.0,
        "{context}: minweight must be finite and non-negative"
    );
    assert!(
        options.ridge_floor.is_finite() && options.ridge_floor >= 0.0,
        "{context}: ridge_floor must be finite and non-negative"
    );
    if let Some(threshold) = options.early_exit_threshold {
        assert!(
            threshold.is_finite(),
            "{context}: early_exit_threshold must be finite"
        );
    }
}

pub(crate) fn assert_states_match_specs(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    assert_eq!(
        states.len(),
        specs.len(),
        "{context}: state/spec block count mismatch"
    );
    for (block, (state, spec)) in states.iter().zip(specs).enumerate() {
        assert_eq!(
            state.beta.len(),
            spec.design.ncols(),
            "{context}: beta length mismatch in block {block}"
        );
        // `state.eta` is produced from `solver_design()` (see
        // `refresh_all_block_etas`), which is `stacked_design` when set
        // (3·n_obs rows for survival LS time-varying blocks) and `design`
        // (n_obs rows) otherwise. Use the same accessor here.
        assert_eq!(
            state.eta.len(),
            spec.solver_design().nrows(),
            "{context}: eta length mismatch in block {block}"
        );
    }
}

pub(crate) fn assert_derivative_blocks_match_specs(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    assert_eq!(
        derivative_blocks.len(),
        specs.len(),
        "{context}: derivative/spec block count mismatch"
    );
}

pub(crate) fn assert_rho_matches_specs(
    rho: &Array1<f64>,
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    let expected = specs.iter().map(|spec| spec.penalties.len()).sum::<usize>();
    assert_eq!(
        rho.len(),
        expected,
        "{context}: rho length does not match penalty count"
    );
}

pub(crate) fn validate_hessian_workspace_ready(
    hessian_workspace: &Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    context: &str,
) -> Result<(), String> {
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace
            .warm_up_outer_caches()
            .map_err(|err| format!("{context}: failed to warm Hessian workspace caches: {err}"))?;
    }
    Ok(())
}

pub fn exact_outer_order_from_capability(
    specs: &[ParameterBlockSpec],
    coefficient_cost: u64,
) -> ExactOuterDerivativeOrder {
    assert_valid_blockspecs(specs, "exact outer derivative order");
    match coefficient_cost {
        0 => ExactOuterDerivativeOrder::Second,
        _ => ExactOuterDerivativeOrder::Second,
    }
}

/// Capability-aware variant of [`exact_outer_order_from_capability`].
///
/// Kept as the public declaration helper for existing family impls, but it no
/// longer gates by cost. Once a caller has established dense or HVP analytic
/// second-order support, the correct derivative order is `Second`.
pub fn exact_outer_order_with_outer_hvp(
    specs: &[ParameterBlockSpec],
    coefficient_cost: u64,
    outer_hyper_hessian_hvp_available: bool,
) -> ExactOuterDerivativeOrder {
    if outer_hyper_hessian_hvp_available {
        assert_valid_blockspecs(specs, "exact outer derivative order with HVP");
        match coefficient_cost {
            0 => ExactOuterDerivativeOrder::Second,
            _ => ExactOuterDerivativeOrder::Second,
        }
    } else {
        exact_outer_order_from_capability(specs, coefficient_cost)
    }
}

/// Realized outer-derivative policy at the current problem size.
///
/// Capability (the family can produce exact second-order calculus) controls
/// whether the Hessian is declared. Runtime cost controls only representation
/// and staging choices below this layer. Large problems must stay on the exact
/// analytic Hessian path and use an operator representation when dense assembly
/// is too expensive; they are not demoted to first-order BFGS here.
///
/// `OuterDerivativePolicy` records the family's *capability*, the *predicted
/// per-eval cost* for both gradient-only and Hessian paths, and exposes the
/// two policy queries the outer optimizer actually needs:
///
/// * [`order_for_evaluation`](Self::order_for_evaluation) — clamp a requested
///   evaluation order against the policy gate.
/// * [`declared_hessian_form`](Self::declared_hessian_form) — what shape the
///   outer-strategy planner should declare to its plan ladder.
/// * [`should_use_staged_kappa`](Self::should_use_staged_kappa) — auto-route
///   the κ optimizer through the pilot/polish schedule at large `n`.
///
/// All thresholds are *const* — no env vars, no CLI flags. The cost model is
/// the family's own `coefficient_gradient_cost` / `coefficient_hessian_cost`
/// scaled by the joint outer-coordinate dimension, with `saturating_mul` so
/// overflow rounds up to the budget ceiling rather than wrapping silently.
#[derive(Clone, Copy, Debug)]
pub struct OuterDerivativePolicy {
    /// What exact calculus the family advertises in principle.
    pub capability: ExactOuterDerivativeOrder,
    /// Predicted per-eval work for one `ValueGradientHessian` evaluation.
    /// Rounded conservatively *up* via `saturating_mul`. Informational for
    /// representation and diagnostics; it does not disable Hessian capability.
    pub predicted_hessian_work: u128,
    /// Predicted per-eval work for one `ValueAndGradient` evaluation.
    /// Rounded conservatively *up* via `saturating_mul`.
    pub predicted_gradient_work: u128,
    /// True when the family's outer-only paths consume
    /// [`BlockwiseFitOptions::outer_score_subsample`] and produce
    /// Horvitz-Thompson-weighted partial sums (i.e. the family overrides
    /// `log_likelihood_only_with_options`,
    /// `exact_newton_joint_psi_workspace_with_options`, and any other
    /// outer-only hooks reached by `evaluate_custom_family_joint_hyper`).
    ///
    /// Determines whether the κ optimizer's pilot/polish staging schedule
    /// engages: when this is `false`, [`Self::should_use_staged_kappa`]
    /// returns `false` regardless of `n`. Engaging the schedule on a
    /// family that ignores the subsample is strictly worse than not
    /// engaging it — the schedule builds a `RowSet::Subsample` and the
    /// boundary plumbing installs an `OuterScoreSubsample` on options,
    /// but the family's default outer-only paths fall back to full-data
    /// sums, so the pilot evaluation costs the same as the polish but
    /// adds a Vec allocation per eval.
    ///
    /// Families that do **not** consume the subsample (default for new
    /// implementations, including the GAMLSS location-scale families
    /// today) leave this `false`. Families that do consume (today:
    /// `BernoulliMarginalSlopeFamily`) override `outer_derivative_policy`
    /// to set this `true`.
    pub subsample_capable: bool,
}

impl OuterDerivativePolicy {
    /// Per-eval gradient work ceiling above which the κ schedule switches
    /// to the staged pilot/polish path. At large scale (n ≳ 100 k) even
    /// the gradient sweep takes minutes per outer iter; subsampling the
    /// pilot stage cuts that to seconds and leaves the final polish on
    /// full data to recover the MLE.
    pub const OUTER_GRADIENT_WORK_BUDGET: u128 = 50_000_000_000;

    /// Pilot subsample auto-engages when full-data `n` exceeds this. Below
    /// this the κ schedule collapses to a single full-data stage —
    /// behaviour identical to the pre-P7 path.
    pub const STAGED_KAPPA_TRIGGER_N: usize = 30_000;

    /// Clamp a requested evaluation order against the policy gate.
    ///
    /// Returns the highest order this policy permits for the requested order:
    /// * `ValueGradientHessian` requested → keep only if `declared_hessian_form`
    ///   is something other than `Unavailable`.
    /// * `ValueAndGradient` requested → always permitted (gradient-only is
    ///   universal).
    pub fn order_for_evaluation(&self, requested: crate::OuterEvalOrder) -> crate::OuterEvalOrder {
        use crate::OuterEvalOrder;
        match requested {
            // Value-only is universal: every policy can evaluate the bare
            // objective, so the request passes through unclamped.
            OuterEvalOrder::Value => OuterEvalOrder::Value,
            OuterEvalOrder::ValueAndGradient => OuterEvalOrder::ValueAndGradient,
            OuterEvalOrder::ValueGradientHessian => {
                if matches!(
                    self.declared_hessian_form(),
                    gam_problem::DeclaredHessianForm::Unavailable
                ) {
                    OuterEvalOrder::ValueAndGradient
                } else {
                    OuterEvalOrder::ValueGradientHessian
                }
            }
        }
    }

    /// Outer Hessian declaration for the outer-strategy planner.
    ///
    /// `Either` ⇔ capability has Hessian. Work estimates select dense vs
    /// operator assembly later; they must not erase analytic second-order
    /// capability from the planner.
    pub fn declared_hessian_form(&self) -> gam_problem::DeclaredHessianForm {
        use gam_problem::DeclaredHessianForm;
        if !self.capability.has_hessian() {
            return DeclaredHessianForm::Unavailable;
        }
        DeclaredHessianForm::Either
    }

    /// True when the κ optimizer should auto-route through the staged
    /// pilot/polish schedule. Triggers when **either** the data is big
    /// (`n ≥ STAGED_KAPPA_TRIGGER_N`) **or** the per-eval gradient work
    /// exceeds `OUTER_GRADIENT_WORK_BUDGET`. The second clause catches
    /// problems with moderate `n` but very wide design (large `p_total`
    /// or `psi_dim`) where a single full-data gradient sweep still
    /// dominates the κ trajectory.
    pub fn should_use_staged_kappa(&self, n: usize) -> bool {
        if !self.subsample_capable {
            // Family does not consume `outer_score_subsample` on its
            // outer-only paths. Engaging the schedule would build a
            // pilot `RowSet::Subsample` whose only effect is per-eval
            // Vec/Arc bookkeeping — the underlying coefficient gradient
            // would still sum every row. Gate the schedule off until
            // the family override declares consumption.
            return false;
        }
        n >= Self::STAGED_KAPPA_TRIGGER_N
            || self.predicted_gradient_work > Self::OUTER_GRADIENT_WORK_BUDGET
    }
}

/// Total outer-coordinate dimensionality used by the default policy work
/// model: `rho_dim + psi_dim`. Each outer evaluation propagates one
/// directional derivative per outer coordinate through the inner solve.
#[inline]
pub(crate) fn outer_coord_dim_for_policy(specs: &[ParameterBlockSpec], psi_dim: usize) -> u128 {
    let rho_total: u128 = specs
        .iter()
        .map(|s| s.penalties.len() as u128)
        .fold(0u128, |acc, k| acc.saturating_add(k));
    rho_total.saturating_add(psi_dim as u128)
}

/// Default predicted-cost model for [`OuterDerivativePolicy`]:
///
/// * gradient work ≈ `coefficient_gradient_cost · (rho_dim + psi_dim)`
/// * Hessian work  ≈ `coefficient_hessian_cost  · (rho_dim + psi_dim)`
///
/// Each outer coordinate triggers one analytic directional derivative
/// through the inner solve; the dense Hessian assembly carries the extra
/// `O(p_total)` factor already captured by `coefficient_hessian_cost`.
///
/// All multiplications saturate so an overflow rounds *up* to the gate
/// ceiling: we'd rather drop one Hessian evaluation that we could have
/// afforded than crash on a 600 s eval.
pub fn default_outer_derivative_policy_costs(
    specs: &[ParameterBlockSpec],
    psi_dim: usize,
    grad_cost: u64,
    hess_cost: u64,
) -> (u128, u128) {
    let k = outer_coord_dim_for_policy(specs, psi_dim);
    let grad = (grad_cost as u128).saturating_mul(k.max(1));
    let hess = (hess_cost as u128).saturating_mul(k.max(1));
    (grad, hess)
}

/// Default coefficient-space Hessian cost: `Σ_b n_b · p_b²`, summed across
/// blocks. Represents the work to assemble or apply the dense block-diagonal
/// inner Hessian once.
pub fn default_coefficient_hessian_cost(specs: &[ParameterBlockSpec]) -> u64 {
    specs
        .iter()
        .map(|s| {
            let n = s.design.nrows() as u64;
            let p = s.design.ncols() as u64;
            n.saturating_mul(p.saturating_mul(p))
        })
        .fold(0u64, |acc, c| acc.saturating_add(c))
}

/// Joint-coupled coefficient-space Hessian cost: `n · (Σ_b p_b)²`. The honest
/// per-evaluation work for any family whose row likelihood couples every block
/// (every observation contributes a rank-`m` outer-product update to the full
/// joint Hessian over `Σ p_b` coefficients), as opposed to the block-diagonal
/// `default_coefficient_hessian_cost` which assumes each `X_b' W_b X_b` is
/// assembled independently.
///
/// Used by all GAMLSS, marginal-slope, and joint-latent families. CTN does
/// not delegate here — it uses its Khatri–Rao factor dimensions internally.
pub fn joint_coupled_coefficient_hessian_cost(n: u64, specs: &[ParameterBlockSpec]) -> u64 {
    let p_total: u64 = specs
        .iter()
        .map(|s| s.design.ncols() as u64)
        .fold(0u64, |acc, p| acc.saturating_add(p));
    n.saturating_mul(p_total.saturating_mul(p_total))
}

/// Default coefficient-space gradient cost: half the Hessian cost.
///
/// The first-order analytic gradient in the unified evaluator runs the same
/// inner Newton solve as the second-order path but skips the `K`-fold
/// pairwise Hessian assembly (`B_{j,k}` blocks) and the `K`-fold inner
/// derivative solves; what remains is the inner solve plus a single
/// gradient-only sweep through the data. Empirically this is roughly half
/// the per-evaluation arithmetic of forming the dense Hessian, hence the
/// `/2` default. Families whose gradient assembly differs structurally
/// (e.g. matrix-free Hv operators with no dense Hessian assembly to halve)
/// should override [`CustomFamily::coefficient_gradient_cost`] explicitly.
pub fn default_coefficient_gradient_cost(specs: &[ParameterBlockSpec]) -> u64 {
    default_coefficient_hessian_cost(specs) / 2
}

/// Compute β-block column ranges from a slice of `ParameterBlockSpec`s.
///
/// Returns one `Range<usize>` per spec, covering the spec's columns in the
/// concatenated β vector (i.e. `offset .. offset + p_block` where `p_block =
/// spec.design.ncols()`). The ranges are non-overlapping, sorted, and their
/// union covers `0..Σ p_block`.
///
/// This is the canonical source of `block_offsets` for every
/// [`crate::solver::arrow_schur::ArrowSchurSystem`] built for a custom family
/// (survival, GAMLSS, transformation-normal, latent-survival, marginal-slope,
/// …). Pass the result to
/// [`crate::solver::arrow_schur::ArrowSchurSystem::set_block_offsets`] before
/// calling `solve` or `solve_with_options` whenever the system will use
/// [`crate::solver::arrow_schur::ArrowSolverMode::InexactPCG`].
///
/// Specs with zero columns produce a zero-width range; callers that want to
/// skip trivial blocks may filter on `r.start < r.end` after calling this
/// function.
pub fn block_offsets_from_specs(specs: &[ParameterBlockSpec]) -> Arc<[Range<usize>]> {
    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(specs.len());
    let mut cursor = 0usize;
    for spec in specs {
        let p = spec.design.ncols();
        ranges.push(cursor..cursor + p);
        cursor += p;
    }
    Arc::from(ranges.into_boxed_slice())
}

/// Bound first-order outer iterations when each analytic-gradient evaluation is
/// already large-scale work. This is only applied after the planner has
/// selected a gradient-only route; second-order/ARC plans keep their requested
/// iteration budget.
pub fn cost_gated_first_order_max_iter(
    requested: usize,
    coefficient_gradient_cost: u64,
    has_outer_hessian: bool,
) -> usize {
    const FIRST_ORDER_OUTER_WORK_BUDGET: u64 = 80_000_000_000;
    const MIN_FIRST_ORDER_ITERS: usize = 4;

    if has_outer_hessian || requested <= 1 || coefficient_gradient_cost == 0 {
        return requested;
    }

    let affordable = (FIRST_ORDER_OUTER_WORK_BUDGET / coefficient_gradient_cost) as usize;
    requested.min(affordable.max(MIN_FIRST_ORDER_ITERS))
}

/// Local trust budget for first-order outer BFGS on log-smoothing parameters.
///
/// One unit in `rho = log(lambda)` is an `e`-fold smoothing-parameter change.
/// Previously this cap was `1.0`, which throttled BFGS to ~1/5 of its
/// quasi-Newton step on flat REML surfaces (the natural BFGS direction has
/// `|d|_inf` of ~5 in log-λ for large-scale survival fits). Probes whose
/// `step_inf > cap` are rejected for free in `OuterFirstOrderBridge::eval_cost`
/// (returning `BFGS_LINE_SEARCH_REJECT_COST` without running an inner solve),
/// so a larger cap costs nothing on rejection — it only lets Strong-Wolfe
/// accept bigger steps that the inner-PIRLS divergence guard can already
/// validate. `5.0` allows up to `e^5 ≈ 148`-fold smoothing-parameter change
/// per accepted outer iter, which matches the typical quasi-Newton direction
/// magnitude while still bounding pathological probes.
pub const fn first_order_bfgs_loglambda_step_cap(has_outer_hessian: bool) -> Option<f64> {
    if has_outer_hessian { None } else { Some(5.0) }
}

pub fn exact_newton_outer_geometry_supports_second_order_solver<F: CustomFamily + ?Sized>(
    family: &F,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}

/// Stable public API for installing outer-score subsampling.
#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    /// Optional override for the OUTER smoothing optimizer's
    /// *relative-cost-decrease* convergence stop, decoupled from `outer_tol`.
    ///
    /// The outer convergence test derives BOTH the absolute projected-gradient
    /// floor (`max(outer_tol, n·1e-9)`) AND the relative-cost stop
    /// (`rel_cost = outer_tol`) from the single `outer_tol`. A caller that needs
    /// a *tight absolute floor* to resolve λ to the genuine REML optimum at
    /// large `n` (where the floor is `n·1e-9`) is then forced to also accept a
    /// *tight rel-cost stop*, which on a flat REML ridge never trips and grinds
    /// the optimizer to `outer_max_iter` — dozens of surplus O(D·p³)
    /// Laplace-derivative outer iterations (the #1082 multinomial
    /// smooth-by-factor wall-clock blow-up). When `Some(r)`, the rel-cost stop
    /// uses `r` while the absolute floor keeps using `outer_tol`, so accuracy
    /// (absolute floor) and perf (loose rel-cost) are selected independently.
    /// `None` preserves the legacy coupling (`rel_cost = outer_tol`) for every
    /// existing caller byte-for-byte.
    pub outer_rel_cost_tol: Option<f64>,
    /// Lower box bound for smoothing coordinates ρ = log λ.
    ///
    /// The default preserves the historical custom-family domain
    /// `λ >= exp(-10)`. Families with known calibration failures at the
    /// near-zero penalty boundary can raise this lower bound without changing
    /// the upper effective-df cap or adding family-specific branches inside the
    /// optimizer.
    pub rho_lower_bound: f64,
    pub minweight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_remlobjective: bool,
    /// If false, the outer smoothing optimizer uses exact gradients but does
    /// not request an analytic outer Hessian from the family.
    pub use_outer_hessian: bool,
    /// If false, skip post-fit joint covariance assembly.
    pub compute_covariance: bool,
    /// Shared cap engaged during seed screening so cost-only evaluations can
    /// stop inner iterations early without affecting the full solve.
    pub screening_max_inner_iterations: Option<Arc<AtomicUsize>>,
    /// Shared cap engaged during regular outer iterations. Unlike screening,
    /// this is only a budget: capped solves still have to earn the ordinary
    /// KKT certificate before derivatives may be exposed.
    pub outer_inner_max_iterations: Option<Arc<AtomicUsize>>,
    /// Optional line-search objective ceiling for lazy log-likelihood-only
    /// evaluations. Families whose per-row log-likelihood contributions are
    /// non-positive may stop once the partial negative log-likelihood is already
    /// above this ceiling, because the unvisited rows cannot improve the trial
    /// objective enough to be accepted. Default `None` preserves exact full-sum
    /// behavior and is the only mode used outside backtracking rejection tests.
    pub early_exit_threshold: Option<f64>,
    /// Stable public API for installing outer-score subsampling.
    ///
    /// Optional stratified row subsample used by outer-only score/gradient
    /// passes. When `Some(s)`, outer score/gradient hot loops should iterate
    /// only over `s.rows` and multiply each contribution by that row's
    /// Horvitz-Thompson inverse-inclusion weight. Inner-PIRLS and final
    /// covariance passes always run on the full data, so this field is
    /// consulted only by outer-only call sites. Default `None` preserves the
    /// full-data behavior. Wrapping in `Arc` keeps `Clone` cheap across the
    /// many places `BlockwiseFitOptions` is duplicated per-eval.
    pub outer_score_subsample: Option<Arc<crate::OuterScoreSubsample>>,
    /// Gate for marginal-slope families to auto-derive a stratified
    /// outer-score subsample at large scale (see
    /// [`crate::families::marginal_slope_shared::auto_outer_score_subsample`]).
    ///
    /// **Default `true`.** Auto-subsampling makes the early rho-gradient
    /// evaluations unbiased stochastic estimators with bounded relative
    /// variance (≈ 1 % at the conservative defaults), then the family switches
    /// back to full-data gradients for the remaining outer iterations. That
    /// keeps large marginal-slope fits fast during the high-motion part of the
    /// trajectory while preserving the default tight `outer_tol` polish on
    /// exact gradients. For small datasets the auto path declines to install a
    /// mask and the fit remains full-data throughout.
    ///
    /// When `outer_score_subsample` is already `Some(...)` the auto
    /// path is bypassed entirely (caller-provided masks always win).
    pub auto_outer_subsample: bool,
    /// Outer-evaluation context populated by the smoothing optimizer at
    /// the top of each real outer derivative evaluation. Used by
    /// auto-subsample install paths to key the stratified mask on the
    /// outer ρ rather than the inner β proxy: during the inner trust-
    /// region / coefficient line search β changes on every trial step,
    /// so keying on β re-fires phase prints (and re-shuffles the mask)
    /// inside a single outer eval. Keying on (rho, eval_id) instead
    /// keeps the mask stable across the inner Newton at one ρ, and
    /// suppresses auto-subsample entirely on inner trial evaluations via
    /// the [`EvalScope::InnerCoefficient`] tag set by
    /// [`coefficient_line_search_options`].
    ///
    /// `None` preserves legacy behavior (no context — install paths fall
    /// back to "no auto-subsample"). Default `None`.
    pub outer_eval_context: Option<OuterEvalContext>,
    /// Optional persistent warm-start cache session. When `Some`, the
    /// outer smoothing optimizer consults the on-disk cache before
    /// starting (to seed θ from the last accepted iterate) and writes
    /// checkpoints + a final entry on completion. When `None`, the fit
    /// runs cold and writes nothing — the default for unit tests and
    /// any caller that pinned a deterministic optimum.
    ///
    /// Callers that need cross-process reuse must supply the session
    /// explicitly; ordinary workflow fits leave this empty so refit-heavy
    /// loops do not touch the shared on-disk store.
    pub cache_session: Option<Arc<gam_runtime::warm_start::Session>>,
    /// Optional mirror sessions that receive a copy of the final-result
    /// finalize() write. Callers can use this to broadcast a converged ρ to
    /// additional keyspace(s) so future fits with related structure can
    /// warm-start from this run. Writes still pass through the session rate
    /// limiter, so mirroring checkpoints does not add unbounded I/O.
    pub cache_mirror_sessions: Vec<Arc<gam_runtime::warm_start::Session>>,
    /// Optional bundle of cross-block (full-width) penalties, paired with
    /// their current `log λ` values from the outer ρ vector. When `Some`,
    /// the inner joint-Newton primitives add the contributions
    ///
    /// * objective: `½ Σ_j exp(ρ_j) βᵀ S_j β`
    /// * gradient:  `Σ_j exp(ρ_j) S_j β`
    /// * Hessian:   `Σ_j exp(ρ_j) S_j`
    ///
    /// in addition to the per-block penalty stack assembled from
    /// `ParameterBlockSpec.penalties`. The per-block path is unchanged.
    /// `None` preserves legacy behaviour for every existing caller.
    pub joint_penalties: Option<Arc<crate::JointPenaltyBundle>>,
    /// Whether the outer smoothing optimizer screens the explicit
    /// `initial_rho` seed through the seed-screening cascade before the
    /// solver starts.
    ///
    /// **Default `true`** — the general path benefits from ranking the
    /// initial seed against the generated exploration seeds via cheap
    /// capped proxy fits.
    ///
    /// A caller sets this `false` when `initial_rho` is already the correct,
    /// identified optimum for its regime so that re-screening it adds only
    /// cost. The survival location-scale constant-scale (parametric-AFT)
    /// path uses this: its time-warp ρ seed is pinned AT the inner ρ box
    /// bound (the affine-baseline limit), where the REML/LAML profile is a
    /// dead-flat unidentified ridge. Running the screening cascade there
    /// drives each proxy fit (and, when every capped stage collapses to
    /// non-finite cost, the uncapped final stage) into a full inner solve on
    /// the near-singular flat Hessian — the source of the multi-minute
    /// no-iteration-log stall (#736, #735, #721). Skipping screening lets the
    /// already-correct seed flow straight to the outer solver, which certifies
    /// box-constraint stationarity at iteration 0. Genuinely flexible regimes
    /// (smooth scale / spatial) leave this `true` and keep full screening.
    pub screen_initial_rho: bool,
    /// Set ONLY while the inner solve is invoked from the seed-screening proxy
    /// (`custom_family_seed_screening_proxy_labeled`), which RANKS candidate
    /// seeds by their penalized objective and never produces the final fit.
    ///
    /// When `true`, the inner joint-Newton skips the full per-axis
    /// Jeffreys/Firth curvature (`custom_family_joint_jeffreys_term`'s
    /// `for k in 0..p` directional-derivative loop, O(p · per-axis-Hdot) per
    /// cycle), keeping ONLY the cheap value-only Jeffreys term
    /// (`custom_family_joint_jeffreys_value`, one reduced-info eigendecomposition)
    /// in the screening score. The per-axis gradient/curvature is what the inner
    /// Newton step needs to *converge* a near-separating fit; the screening proxy
    /// is capped and only ranks, so it does not need step convergence — it needs
    /// a finite, separation-aware score cheaply. For a K-block coupled family
    /// (Dirichlet/multinomial) each per-axis directional derivative is itself
    /// O(K²·n·p), so running the full term for every cascade candidate over the
    /// joint width `p` is the wrong cost class and made the coupled fit
    /// non-completing during screening alone (gam#729/#808). The actual fit
    /// (after a seed is selected) runs with this `false`, so the load-bearing
    /// Firth curvature is fully present where it matters.
    ///
    /// **Default `false`** — only the screening proxy sets it `true`.
    pub seed_screening: bool,
}

pub const DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES: usize = 1200;

impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            // Large-scale custom-family marginal-slope fits can have a
            // long, monotone joint-Newton tail: objective and step size keep
            // shrinking, but the exact KKT residual may need several hundred
            // additional cycles after the old 300-cycle cap. The outer
            // REML/LAML derivative path is correct only at a stationary inner
            // mode, so a merely descended iterate must not be accepted as
            // converged. Use a production-sized cap by default and rely on the
            // KKT/objective certificates to exit early for well-conditioned
            // Gaussian, logistic, and small-n fits.
            inner_max_cycles: DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            outer_rel_cost_tol: None,
            rho_lower_bound: -10.0,
            minweight: CUSTOM_FAMILY_WEIGHT_FLOOR,
            // `ridge_floor` is an ExplicitPrior in the canonical
            // stabilization ledger taxonomy (`StabilizationKind::ExplicitPrior`):
            // its δ enters the quadratic term, the Laplace Hessian, and the
            // penalty log-determinant — `ridge_policy` below is the live
            // policy that confirms which terms it lands in. The default
            // pos-part policy enables every inclusion flag, so callers
            // wanting solver-only damping should construct a custom policy
            // (or, preferably, a `StabilizationLedger::numerical_perturbation`)
            // rather than reusing this field.
            ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: true,
            // Default ON: families expose exact outer Hessians whenever their
            // analytic dense or operator representation is implemented.
            use_outer_hessian: true,
            compute_covariance: false,
            screening_max_inner_iterations: None,
            outer_inner_max_iterations: None,
            seed_screening: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
            auto_outer_subsample: true,
            outer_eval_context: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            joint_penalties: None,
            screen_initial_rho: true,
        }
    }
}
