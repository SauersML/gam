//! Fit-time configuration and cost accounting: `BlockwiseFitOptions`, the
//! outer-derivative policy + order selection, coefficient cost models, and the
//! argument-validation asserts shared by the solver entry points.

use crate::families::custom_family::psi_design::{
    CustomFamilyHyperLayout, ExactNewtonJointHessianWorkspace,
};
use gam_problem::{ParameterBlockSpec, ParameterBlockState};
use ndarray::Array1;
use std::ops::Range;
use std::sync::Arc;

// Moved to `gam-problem` (#1521 CustomFamily-cone inversion): the neutral,
// dependency-free outer-objective + exact-derivative-order capability enums now
// live in `gam_problem::family_options` and are re-exported here so every
// `custom_family::ExactNewtonOuterObjective` / `ExactOuterDerivativeOrder` path
// keeps resolving byte-for-byte.
pub use gam_problem::{ExactNewtonOuterObjective, ExactOuterDerivativeOrder};

// The block-spec consistency validator is neutral (no `CustomFamily`
// dependency), so it lives in `gam-problem`. Coefficient damping
// deliberately has no model-level default: the custom-family solver derives
// any transient shift from the current curvature and must converge the
// undamped score equation.
pub use gam_problem::validate_blockspec_consistency;

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

/// Precondition check for the hooks that receive block states *without* the
/// matching specs (`exact_newton_outer_curvature`,
/// `exact_newton_joint_loglik_gradient`, ...).
///
/// Those hooks are asked for curvature/score *at the supplied point*, so the
/// point has to be one: a NaN beta or eta is not an iterate, and a family that
/// returns `None` (the trait default) would otherwise let the NaN travel on to
/// whichever path the caller falls back to, surfacing as an unattributable
/// non-finite outer derivative several layers up.
pub(crate) fn assert_blockstates_are_a_point(states: &[ParameterBlockState], context: &str) {
    for (block, state) in states.iter().enumerate() {
        assert!(
            state.beta.iter().all(|v| !v.is_nan()),
            "{context}: NaN beta in block {block}"
        );
        assert!(
            state.eta.iter().all(|v| !v.is_nan()),
            "{context}: NaN eta in block {block}"
        );
    }
}

/// Precondition check for the per-block hooks that take `(states, block_index,
/// spec)`: the index must select a block of `states`, and the spec handed
/// alongside it must be *that* block's spec, not a different block's.
///
/// The index and the spec are two independent ways of naming the same block, so
/// a caller that advances one without the other silently asks the family to
/// project block `i`'s coefficients onto block `j`'s geometry.
pub(crate) fn assert_block_index_matches_spec(
    states: &[ParameterBlockState],
    block_index: usize,
    spec: &ParameterBlockSpec,
    context: &str,
) {
    assert!(
        block_index < states.len(),
        "{context}: block index {block_index} out of range for {} blocks",
        states.len()
    );
    assert_eq!(
        states[block_index].beta.len(),
        spec.design.ncols(),
        "{context}: spec does not describe block {block_index}"
    );
}

/// Precondition check for the per-block hooks that take `(states, block_index,
/// direction)` where `direction` lives in the block's *coefficient* space
/// (`exact_newton_hessian_directional_derivative`, `max_feasible_step_size`, ...).
pub(crate) fn assert_block_local_beta_direction(
    states: &[ParameterBlockState],
    block_index: usize,
    direction: &Array1<f64>,
    context: &str,
) {
    assert!(
        block_index < states.len(),
        "{context}: block index {block_index} out of range for {} blocks",
        states.len()
    );
    assert_eq!(
        direction.len(),
        states[block_index].beta.len(),
        "{context}: direction is not in block {block_index}'s coefficient space"
    );
    assert!(
        direction.iter().all(|v| !v.is_nan()),
        "{context}: NaN entry in block {block_index} coefficient direction"
    );
}

/// Precondition check for the per-block hooks that take `(states, block_index,
/// d_eta)` where the direction lives in the block's *predictor* space
/// (`diagonalworking_weights_directional_derivative` and its second-order
/// sibling): `dw` is indexed by row, so a coefficient-space vector slipped in
/// here would silently produce a weight derivative of the wrong length.
pub(crate) fn assert_block_local_eta_direction(
    states: &[ParameterBlockState],
    block_index: usize,
    d_eta: &Array1<f64>,
    context: &str,
) {
    assert!(
        block_index < states.len(),
        "{context}: block index {block_index} out of range for {} blocks",
        states.len()
    );
    assert_eq!(
        d_eta.len(),
        states[block_index].eta.len(),
        "{context}: direction is not in block {block_index}'s predictor space"
    );
    assert!(
        d_eta.iter().all(|v| !v.is_nan()),
        "{context}: NaN entry in block {block_index} predictor direction"
    );
}

/// Precondition check for the psi hooks that name an outer coordinate by its
/// global index into the hyper-layout: the index must resolve to a real axis,
/// otherwise the family is being asked to differentiate a coordinate that the
/// layout does not carry.
pub(crate) fn assert_psi_index_in_layout(
    hyper_layout: &CustomFamilyHyperLayout,
    psi_index: usize,
    context: &str,
) {
    assert!(
        hyper_layout.axis(psi_index).is_some(),
        "{context}: psi index {psi_index} is not an axis of a layout with {} coordinates",
        hyper_layout.len()
    );
}

pub(crate) fn assert_hyper_layout_matches_specs(
    hyper_layout: &CustomFamilyHyperLayout,
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    assert_eq!(
        hyper_layout.block_count(),
        specs.len(),
        "{context}: hyper-layout/spec block count mismatch"
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
    eval_mode: gam_problem::EvalMode,
) -> Result<(), String> {
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace
            .warm_up_outer_caches_for_mode(eval_mode)
            .map_err(|err| format!("{context}: failed to warm Hessian workspace caches: {err}"))?;
    }
    Ok(())
}

/// Realized outer-derivative policy: the family's capability, wrapped for the
/// outer planner.
///
/// Capability (the family can produce exact second-order calculus) controls
/// whether the Hessian is declared. Problem size selects only representation
/// below this layer. Large problems stay on the exact analytic Hessian path and
/// use an operator representation when dense assembly is too expensive; they
/// are not demoted to first-order BFGS here.
///
/// `OuterDerivativePolicy` exposes the two policy queries the outer optimizer
/// needs:
///
/// * [`order_for_evaluation`](Self::order_for_evaluation) — clamp a requested
///   evaluation order against the policy gate.
/// * [`declared_hessian_form`](Self::declared_hessian_form) — what shape the
///   outer-strategy planner should declare to its plan ladder.
#[derive(Clone, Copy, Debug)]
pub struct OuterDerivativePolicy {
    /// What exact calculus the family advertises in principle.
    pub capability: ExactOuterDerivativeOrder,
}

impl OuterDerivativePolicy {
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
    /// `Either` ⇔ capability has Hessian. Representation routing happens later
    /// and must not erase analytic second-order capability from the planner.
    pub fn declared_hessian_form(&self) -> gam_problem::DeclaredHessianForm {
        use gam_problem::DeclaredHessianForm;
        if !self.capability.has_hessian() {
            return DeclaredHessianForm::Unavailable;
        }
        DeclaredHessianForm::Either
    }
}

/// Declared work of the two routes for solving with the joint coefficient
/// Hessian, in flops.
///
/// The dense route assembles the `p × p` Hessian once (`build`) and factors it
/// (`p³/3`). The matrix-free route runs preconditioned CG, one operator product
/// (`apply`) per iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointHessianWork {
    /// Flops to assemble the dense joint Hessian once.
    pub build: u64,
    /// Flops of one operator Hessian-vector product.
    pub apply: u64,
}

impl JointHessianWork {
    /// Work of a row pullback `H = Σ_i J_iᵀ W_i J_i` over `n` rows and `p`
    /// coefficients: assembly is `n·p²`, and one product streams the rows
    /// forward and back, `2·n·p`.
    pub fn row_pullback(n: u64, p: u64) -> Self {
        Self {
            build: n.saturating_mul(p.saturating_mul(p)),
            apply: n.saturating_mul(p).saturating_mul(2),
        }
    }

    /// How the inner Newton step may solve with preconditioned CG before the
    /// dense route: the same attempt the penalized normal equations make
    /// ([`gam_linalg::pcg::DenseRouteWork::pcg_attempt`]), priced by this work.
    ///
    /// CG may spend what the dense route costs, `(build + p³/3) / apply`
    /// products, and the dense route takes over if CG has not converged by then,
    /// so no CG iteration count has to be predicted. Past the memory governor's
    /// single-materialization cap CG is the only solve.
    pub fn pcg_attempt(&self, p: usize) -> gam_linalg::pcg::PcgAttempt {
        gam_linalg::pcg::DenseRouteWork {
            build: self.build,
            apply: self.apply,
        }
        .pcg_attempt(p)
    }
}

/// Compute β-block column ranges from a slice of `ParameterBlockSpec`s.
///
/// Returns one `Range<usize>` per spec, covering the spec's columns in the
/// concatenated β vector (i.e. `offset .. offset + p_block` where `p_block =
/// spec.design.ncols()`). The ranges are non-overlapping, sorted, and their
/// union covers `0..Σ p_block`.
///
/// This is the canonical source of `block_offsets` for every
/// `crate::solver::arrow_schur::ArrowSchurSystem` built for a custom family
/// (survival, GAMLSS, transformation-normal, latent-survival, marginal-slope,
/// …). Pass the result to
/// `crate::solver::arrow_schur::ArrowSchurSystem::set_block_offsets` before
/// calling `solve` or `solve_with_options` whenever the system will use
/// `crate::solver::arrow_schur::ArrowSolverMode::InexactPCG`.
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

/// A prior fit's certified outer point, handed to a new fit as its start
/// (gam#3002). It is the one warm-start mechanism: every source (a saved model's
/// `warm_start_from`) becomes this, and every outer search takes it through
/// `OuterProblem::with_warm_start`.
///
/// A certificate is a property of a point AND a criterion, so a warm start never
/// changes which point a search reports for its own criterion V:
/// - With [`same_inputs`](Self::same_inputs), every outer search of the fit is
///   offered `theta` as a prior certificate. A search accepts it, with no outer
///   iteration, only where it is certified for that search's own V: V(theta)
///   agrees with [`value`](Self::value) within the rounding envelope and the
///   projected gradient is inside the band. Any other search of the fit (a
///   pilot, an unarmed evidence fit, an earlier alternation round, whose V is
///   not the one `theta` was certified for) declines it and runs exactly as it
///   runs cold. The fit's inputs are the parent's, so it replays the parent's
///   searches, and the one that accepts is the parent's published search.
/// - On other inputs, `theta` only joins a search whose result is the argmin over
///   its seed set (the independent multistart), as one more seed, so that
///   search's published V is at most its cold one. A search that stops at its
///   first certified seed does not use it, because a new first seed would change
///   which point it certifies. The search records that as
///   [`WarmStartOutcome::NotUsed`].
#[derive(Clone, Debug)]
pub struct WarmStart {
    /// The certified outer coordinates, in the order the fit's outer problem
    /// holds them: ρ, then the log length scales, then the auxiliary coordinates.
    pub theta: Array1<f64>,
    /// The inner coefficient mode at `theta`, flattened across blocks.
    pub beta: Array1<f64>,
    /// The outer criterion the parent certified at `theta`.
    pub value: f64,
    /// This fit's inputs (data, weights, offsets and model request) are the
    /// parent fit's, by their input fingerprint.
    pub same_inputs: bool,
    /// What the fit's outer searches did with the point: the most any of them
    /// did ([`WarmStart::record`]). `None` after the fit means no search received
    /// it.
    pub outcome: Arc<std::sync::Mutex<Option<WarmStartOutcome>>>,
}

/// What an outer search did with a [`WarmStart`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WarmStartOutcome {
    /// The point was accepted where it stands, as a prior certificate for this
    /// search's criterion.
    Resumed,
    /// The point was one more seed of an argmin-over-seeds multistart.
    JoinedMultistart,
    /// The point was not used; the reason says why.
    NotUsed(&'static str),
}

impl WarmStart {
    /// Record what one outer search did with the point. A fit can run several
    /// searches (a pilot, an evidence fit, alternation rounds) and each records,
    /// so the slot keeps the most any of them did: a search that resumed or
    /// joined is not hidden by a later one that declined.
    pub fn record(&self, outcome: WarmStartOutcome) {
        let rank = |outcome: &WarmStartOutcome| match outcome {
            WarmStartOutcome::NotUsed(_) => 0,
            WarmStartOutcome::JoinedMultistart => 1,
            WarmStartOutcome::Resumed => 2,
        };
        if let Ok(mut slot) = self.outcome.lock()
            && slot
                .as_ref()
                .is_none_or(|held| rank(&outcome) >= rank(held))
        {
            *slot = Some(outcome);
        }
    }

    /// What the fit's outer searches did with the point, if any received it.
    pub fn recorded(&self) -> Option<WarmStartOutcome> {
        self.outcome.lock().ok().and_then(|slot| slot.clone())
    }
}

/// Stable public API for installing outer-score subsampling.
#[derive(Clone)]
pub struct BlockwiseFitOptions {
    /// Maximum coefficient-optimizer cycles allowed before non-convergence is returned.
    pub inner_max_cycles: usize,
    /// Absolute coefficient stationarity tolerance; must be finite and positive.
    pub inner_tol: f64,
    /// Maximum REML/LAML outer iterations allowed before non-convergence is returned.
    pub outer_max_iter: usize,
    /// Absolute outer stationarity tolerance; must be finite and positive.
    pub outer_tol: f64,
    /// A family-declared floor for the smoothing coordinates ρ = log λ.
    ///
    /// `None` (the default) leaves the λ-selection domain to the engine's
    /// derived resolvability domain (#2812): per coordinate, the ρ interval on
    /// which the penalty is resolvable against the term's own design
    /// curvature. A family with a known calibration limit at the near-zero
    /// penalty boundary (the multinomial's derived minimum strength) raises
    /// the lower edge here; nothing here supplies an upper wall.
    pub rho_lower_bound: Option<f64>,
    /// Optional seed for transient solver damping. The default is zero. The
    /// damping enters only inner linear solves, never the quadratic objective,
    /// the penalty determinant, or the Laplace Hessian, so the converged
    /// estimand is the stationary point of the undamped statistical objective.
    pub ridge_floor: f64,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_remlobjective: bool,
    /// If false, the outer smoothing optimizer uses exact gradients but does
    /// not request an analytic outer Hessian from the family.
    pub use_outer_hessian: bool,
    /// If false, skip post-fit joint covariance assembly.
    pub compute_covariance: bool,
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
    /// outer-score subsample whenever the shared size rule picks `K < n` (see
    /// `gam_models::marginal_slope_shared::auto_outer_score_subsample`).
    ///
    /// **Default `true`.** Auto-subsampling makes the early rho-gradient
    /// evaluations unbiased stochastic estimators with relative noise about
    /// `1/√K` (1 % at `K = 10_000`, 2 % at `K = 2_000`), then the family switches
    /// back to full-data gradients for the remaining outer iterations. That
    /// keeps large marginal-slope fits fast during the high-motion part of the
    /// trajectory while preserving the default tight `outer_tol` polish on
    /// exact gradients. For small datasets the auto path declines to install a
    /// mask and the fit remains full-data throughout.
    ///
    /// When `outer_score_subsample` is already `Some(...)` the auto
    /// path is bypassed entirely (caller-provided masks always win).
    pub auto_outer_subsample: bool,
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
    /// A prior fit's certified outer point to start from (`warm_start_from`,
    /// gam#3002). Every outer driver that reads options takes it through
    /// `OuterProblem::with_warm_start`; see [`WarmStart`].
    pub warm_start: Option<WarmStart>,
    /// Explicit fit-owned cross-process store. Unlike `cache_session`, which is
    /// one caller-keyed outer-iterate stream, this capability owns the shared
    /// response-keyed record and descriptor-keyed artifact namespaces too.
    ///
    /// `None` is disk-silent. The high-level configuration constructs this
    /// lazily from an explicit root; no custom-family fit discovers ambient
    /// temp/cache state.
    pub persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
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
}

/// Default maximum coefficient cycles for a custom-family fit.
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
            rho_lower_bound: None,
            // Conditioning is solver state, not a coefficient prior. Start at
            // the exact Hessian (zero shift); rank/curvature-aware damping may
            // regularize rejected Newton steps, but none of it enters the
            // objective or its derivatives and convergence is certified on the
            // undamped KKT residual.
            ridge_floor: 0.0,
            use_remlobjective: true,
            // Default ON: families expose exact outer Hessians whenever their
            // analytic dense or operator representation is implemented.
            use_outer_hessian: true,
            compute_covariance: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
            auto_outer_subsample: true,
            cache_session: None,
            warm_start: None,
            persistent_warm_start_store: None,
            cache_mirror_sessions: Vec::new(),
            joint_penalties: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::Array2;

    fn make_spec(nrows: usize, ncols: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            design: DesignMatrix::from(Array2::<f64>::zeros((nrows, ncols))),
            ..ParameterBlockSpec::defaults()
        }
    }

    // -----------------------------------------------------------------------
    // block_offsets_from_specs
    // -----------------------------------------------------------------------

    #[test]
    fn block_offsets_empty_is_empty() {
        let offsets = block_offsets_from_specs(&[]);
        assert_eq!(offsets.len(), 0);
    }

    #[test]
    fn block_offsets_three_blocks() {
        // p = [2, 3, 1] → [0..2, 2..5, 5..6]
        let specs = [make_spec(1, 2), make_spec(1, 3), make_spec(1, 1)];
        let offsets = block_offsets_from_specs(&specs);
        assert_eq!(&offsets[0], &(0..2));
        assert_eq!(&offsets[1], &(2..5));
        assert_eq!(&offsets[2], &(5..6));
    }

    #[test]
    fn block_offsets_zero_width_block() {
        // p = [2, 0, 1] → [0..2, 2..2, 2..3]
        let specs = [make_spec(1, 2), make_spec(1, 0), make_spec(1, 1)];
        let offsets = block_offsets_from_specs(&specs);
        assert_eq!(&offsets[0], &(0..2));
        assert_eq!(&offsets[1], &(2..2));
        assert_eq!(&offsets[2], &(2..3));
    }

    #[test]
    fn default_custom_family_objective_is_coefficient_ridge_free() {
        let options = BlockwiseFitOptions::default();
        assert_eq!(options.ridge_floor, 0.0);
    }
}
