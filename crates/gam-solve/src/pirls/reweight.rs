//! CPU row-reweight orchestration for the inner PIRLS loop.
//!
//! Owns:
//! - `madsen_lm_accept_factor` — Madsen-Nielsen-Tingleff smooth Marquardt
//!   trust-region update factor for accepted steps.
//! - `runworking_model_pirls` — the inner-loop kernel that takes a
//!   WorkingModel, a starting beta, and an options bundle, and runs the
//!   accept/reject LM iteration until convergence, max-iters, or LM
//!   exhaustion.
//! - `test_support` — thread-local penalized-deviance trace harness.

use super::{
    ExportedLaplaceCurvature, HessianCurvatureKind, PIRLS_ETA_ABS_CAP, PirlsAcceptedStateCacheKey,
    PirlsStatus, SoftAcceptProgress, WorkingModel, WorkingModelIterationInfo,
    WorkingModelPirlsOptions, WorkingModelPirlsResult, WorkingState,
    add_scaled_diagonal_to_upper_sparse, commit_pending_arrow_latent,
    compute_constraint_kkt_diagnostics, compute_lm_d2, constrained_stationarity_norm,
    effective_kkt_tolerance, linear_constraints_from_lower_bounds, pirls_soft_acceptance,
    project_coefficients_to_lower_bounds, restore_pending_arrow_latent_if_needed,
    solve_direction_with_dense_factor, solve_newton_direction_dense,
    solve_newton_directionwith_linear_constraints, solve_newton_directionwith_lower_bounds,
    update_scaled_diagonal_in_place,
};
use crate::estimate::EstimationError;
use crate::loop_guard::{FlatStreak, IterationBound, LoopVerdict, RejectEscalator};
use faer::sparse::SparseColMat;
use gam_linalg::sparse_exact::{
    factorize_sparse_spd, solve_sparse_spd_into, sparse_symmetric_upper_matvec_public,
};
use gam_linalg::utils::{StableSolver, array_is_finite, inf_norm};
use gam_problem::Coefficients;
use ndarray::{Array1, Zip};

/// Madsen-Nielsen-Tingleff smooth Marquardt trust-region update (eq 3.17 in
/// "Methods for non-linear least squares problems", IMM Tech Univ Denmark,
/// 2nd ed 2004) for the LM accept branch. Replaces the older binary
/// `if rho > 0.25 { /10 } else { keep }` rule.
///
/// The accepted-step damping update is
///   λ_next = λ_loop · max(1/3, 1 − (2ρ − 1)³)
/// where ρ = actual_reduction / predicted_reduction is the gain ratio.
/// The cubic expression interpolates smoothly across the gain-ratio scale:
///
/// | ρ      | factor | rationale                                        |
/// |--------|-------:|--------------------------------------------------|
/// | 1.0    | 1/3    | good Newton match — mild shrink (was ÷10 before, |
/// |        |        | which over-shot to 1e-9 and forced the next      |
/// |        |        | iter to discover the over-trust by rejection)    |
/// | 0.75   | 0.875  | slight shrink                                    |
/// | 0.5    | 1.0    | no change — the model is "fine"                  |
/// | 0.25   | 1.125  | slight EXPAND on marginal accepts (new behavior; |
/// |        |        | the binary rule held lambda flat here, then the  |
/// |        |        | next iter often needed extra halvings)           |
/// | →0⁺    | →2.0   | capped at 2.0 so a just-barely-accepted step     |
/// |        |        | bumps λ toward gradient-descent at most ×2       |
///
/// Cap at 2.0 (vs unbounded `1 − (2ρ − 1)³` which diverges as ρ→−∞) is
/// safety: this branch only fires for accepted (ρ > 0) steps, but the
/// post-accept ρ can be as small as `noise_floor / predicted_reduction`,
/// where the cubic blow-up isn't physical.
#[inline]
pub(super) fn madsen_lm_accept_factor(rho: f64) -> f64 {
    let two_rho_minus_one = 2.0 * rho - 1.0;
    let cube = two_rho_minus_one * two_rho_minus_one * two_rho_minus_one;
    (1.0 - cube).clamp(1.0 / 3.0, 2.0)
}

/// Exact squared Newton decrement `gᵀH⁻¹g` for the bare penalized Hessian.
///
/// The ordinary in-loop bound derives this quantity from the damped LM step.
/// That bound is intentionally conservative, but becomes vacuous when the
/// statistical Hessian is extremely stiff and the LM damping is large: using
/// the generic stabilization floor as a lower eigenvalue bound can inflate a
/// machine-small decrement by many orders of magnitude. At a numerical
/// objective plateau we pay for one bare factorization and certify the actual
/// local quadratic geometry instead. Failure to factorize is not a
/// certificate; callers simply continue the iteration.
pub(super) fn exact_newton_decrement_sq(state: &WorkingState) -> Option<f64> {
    if !state.gradient.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut solution = Array1::<f64>::zeros(state.gradient.len());
    match &state.hessian {
        gam_linalg::matrix::SymmetricMatrix::Dense(hessian) => {
            if !hessian.iter().all(|value| value.is_finite()) {
                return None;
            }
            let factor = StableSolver::new().factorize(hessian).ok()?;
            solve_direction_with_dense_factor(&factor, &state.gradient, &mut solution);
            solution.mapv_inplace(|value| -value);
        }
        gam_linalg::matrix::SymmetricMatrix::Sparse(hessian) => {
            let factor = factorize_sparse_spd(hessian).ok()?;
            solve_sparse_spd_into(&factor, &state.gradient, &mut solution).ok()?;
        }
    }
    let decrement_sq = state.gradient.dot(&solution);
    (decrement_sq.is_finite() && decrement_sq >= 0.0).then_some(decrement_sq)
}

/// Whether a constrained iterate `(beta, gradient)` sits within the SAME
/// degeneracy-aware constraint-KKT acceptance band the outer REML startup gate
/// (`enforce_constraint_kkt`) applies, and so may be soft-accepted as a valid
/// constrained minimum.
///
/// A soft acceptance based on an objective plateau or an `objective_scale`-
/// relative gradient band is not, on its own, a constrained-stationarity
/// certificate: the inner solve can flatten the penalized deviance on a
/// near-vertex face of a curvature cone (shape=convex/concave) while the
/// constrained KKT residual still stalls far above the outer gate's *absolute*
/// tolerance. Accepting such a point lets the fit's success hinge on which ρ
/// the seed loop started from — the warm-start cache vs cold-cache divergence
/// of #873. Gating every constrained soft-acceptance on this band keeps the
/// inner verdict in lockstep with the outer gate, so the fit is cache-
/// independent.
///
/// The band mirrors the outer gate exactly: a genuinely rank-deficient active
/// face (the underdetermined I-spline case the plateau branch was built for)
/// gets the relaxed `ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL`; a
/// non-degenerate face is held to a strict band (`10 · kkt_tolerance`, the same
/// near-stationary band `near_stationary_kkt` uses). Stationarity is checked
/// scale-invariantly — absolute residual OR the relative ratio
/// `stationarity / max(‖grad‖∞, 1)` within the band — exactly as the outer gate
/// and the inner active-set solver do, so an O(n) gradient scale (issue #879)
/// does not leave a converged optimum stranded above a fixed absolute band
/// (issue #989). Returns `true` for an unconstrained fit (no constraint-KKT
/// gate to honour) and when no constraint rows can be derived (no bound is
/// finite).
pub(crate) fn constraint_kkt_admits_soft_accept(
    options: &WorkingModelPirlsOptions,
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    kkt_tolerance: f64,
) -> bool {
    // Mirror the exported-diagnostic construction in the result assembly (and
    // the outer gate's input): prefer explicit linear constraints, else derive
    // the constraint rows from the coordinate lower bounds.
    let diag = match options.linear_constraints.as_ref() {
        Some(lin) => Some(compute_constraint_kkt_diagnostics(beta, gradient, lin)),
        None => options.coefficient_lower_bounds.as_ref().and_then(|lb| {
            linear_constraints_from_lower_bounds(lb)
                .map(|lin| compute_constraint_kkt_diagnostics(beta, gradient, &lin))
        }),
    };
    match diag {
        None => true,
        Some(kkt) => {
            let stationarity_band = if kkt.working_set_rank_deficient {
                crate::active_set::ACTIVE_SET_KKT_DEGENERATE_STATIONARITY_TOL
            } else {
                kkt_tolerance * 10.0
            };
            // Scale-invariant stationarity, in lockstep with the outer gate
            // (`enforce_constraint_kkt`) and the inner active-set solver: accept
            // when EITHER the absolute residual OR the relative ratio
            // `stationarity / max(‖grad‖∞, 1)` is within the band. An O(n)
            // gradient scale (issue #879) leaves the absolute residual above any
            // fixed band at a genuine optimum; gating only on it would refuse a
            // converged soft-accept the outer gate now admits (issue #989).
            let stationarity_rel = kkt.stationarity / kkt.gradient_scale.max(1.0);
            kkt.primal_feasibility <= crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
                && (kkt.stationarity <= stationarity_band || stationarity_rel <= stationarity_band)
        }
    }
}

/// Constraint-KKT cleanliness gate for the LONG (20-iteration) constrained
/// objective-plateau certificate — the stall exit for fits whose objective is
/// genuinely exhausted while the raw stationarity residual stays above the
/// `near_stationary_kkt` band (e.g. a shallow, almost-linear direction where
/// the gradient is small-but-not-tiny and every Newton step buys progress far
/// below the convergence tolerance).
///
/// Deliberately DIFFERENT from [`constraint_kkt_admits_soft_accept`]: the
/// stationarity band is *not* required here. What discriminates a legitimate
/// progress-exhausted stall from the failure modes the stationarity band
/// protects against is carried by the certificate's other conjuncts:
///
/// * **Value↔gradient desync** (the recurring objective/gradient drift class)
///   cannot certify: its quadratic model keeps PREDICTING above-tolerance
///   progress that the value never realizes, and the long-plateau branch
///   requires the model-predicted reduction itself to be sub-tolerance for
///   the whole streak (see `model_progress_exhausted` at the call site).
/// * **The #873 degenerate-vertex stall** cannot certify: a rank-deficient
///   working set is refused outright here (the fast 2-iteration path keeps
///   its relaxed degenerate band — that path still demands stationarity).
/// * **A wrong-side or infeasible iterate** cannot certify: primal
///   feasibility, dual feasibility (no wrong-sign multipliers), and
///   complementarity must all sit inside the same bands the outer gate uses.
///
/// At a strictly-interior iterate every constraint-KKT component except
/// stationarity is exactly zero, so this gate reduces to "feasible, clean,
/// non-degenerate" — which is precisely the set of states for which a
/// 20-iteration monotone sub-tolerance plateau with a sub-tolerance model
/// prediction is an honest "no useful progress is available" certificate.
pub(crate) fn constraint_kkt_admits_progress_exhausted_stall(
    options: &WorkingModelPirlsOptions,
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    kkt_tolerance: f64,
) -> bool {
    let diag = match options.linear_constraints.as_ref() {
        Some(lin) => Some(compute_constraint_kkt_diagnostics(beta, gradient, lin)),
        None => options.coefficient_lower_bounds.as_ref().and_then(|lb| {
            linear_constraints_from_lower_bounds(lb)
                .map(|lin| compute_constraint_kkt_diagnostics(beta, gradient, &lin))
        }),
    };
    match diag {
        None => true,
        Some(kkt) => {
            let cleanliness_band = kkt_tolerance * 10.0;
            !kkt.working_set_rank_deficient
                && kkt.primal_feasibility <= crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
                && kkt.dual_feasibility <= cleanliness_band
                && kkt.complementarity <= cleanliness_band
        }
    }
}

pub fn runworking_model_pirls<M, F>(
    model: &mut M,
    mut beta: Coefficients,
    options: &WorkingModelPirlsOptions,
    mut iteration_callback: F,
) -> Result<WorkingModelPirlsResult, EstimationError>
where
    M: WorkingModel + ?Sized,
    F: FnMut(&WorkingModelIterationInfo),
{
    const CONSTRAINED_OBJECTIVE_PLATEAU_STREAK: usize = 20;

    // ── Anderson acceleration of depth 1 (AA(1)) for the Fisher fixed-point ──
    // PIRLS normally uses observed-information Newton (already super-linear, no
    // help available from AA). When `force_fisher_for_rest` engages, the inner
    // iteration becomes the linearly-convergent Fisher contraction — exactly
    // the regime where AA(1) provably improves the rate. State is local to
    // this PIRLS call; costs nothing while `force_fisher_for_rest` stays
    // false because the mixing branch is never entered.
    const AA1_DAMPING_FLOOR: f64 = 1e-12;
    const AA1_DISABLE_REJECT_THRESHOLD: usize = 3;

    struct AndersonOneState {
        pub(crate) prev_beta: Option<Array1<f64>>,
        pub(crate) prev_residual: Option<Array1<f64>>,
        pub(crate) r_k: Array1<f64>,
        pub(crate) dr: Array1<f64>,
        pub(crate) dx: Array1<f64>,
        pub(crate) beta_accel: Array1<f64>,
        pub(crate) consecutive_accepts: usize,
        pub(crate) consecutive_rejects: usize,
        pub(crate) disabled: bool,
        pub(crate) engaged_logged: bool,
    }

    impl AndersonOneState {
        pub(crate) fn new() -> Self {
            Self {
                prev_beta: None,
                prev_residual: None,
                r_k: Array1::zeros(0),
                dr: Array1::zeros(0),
                dx: Array1::zeros(0),
                beta_accel: Array1::zeros(0),
                consecutive_accepts: 0,
                consecutive_rejects: 0,
                disabled: false,
                engaged_logged: false,
            }
        }

        pub(crate) fn ensure_len(buf: &mut Array1<f64>, len: usize) {
            if buf.len() != len {
                *buf = Array1::zeros(len);
            }
        }

        /// Try to produce an accelerated candidate from the plain Fisher
        /// fixed-point step `beta_new = beta_old + direction`. The fixed-point
        /// residual at this iteration is `r_k = beta_new - beta_old`.
        ///
        /// Returns `Some(beta_accel)` when a finite acceleration is available,
        /// `None` when AA should be skipped (no history yet, disabled, or
        /// numerical floor hit).
        pub(crate) fn aa1_mix(
            &mut self,
            beta_old: &Array1<f64>,
            beta_new: &Array1<f64>,
        ) -> Option<&Array1<f64>> {
            if self.disabled {
                return None;
            }
            let prev_beta = self.prev_beta.as_ref()?;
            let prev_residual = self.prev_residual.as_ref()?;
            if prev_beta.len() != beta_old.len() || prev_residual.len() != beta_old.len() {
                return None;
            }
            let len = beta_old.len();
            Self::ensure_len(&mut self.r_k, len);
            Self::ensure_len(&mut self.dr, len);
            Self::ensure_len(&mut self.dx, len);
            Self::ensure_len(&mut self.beta_accel, len);
            // r_k = beta_new - beta_old
            Zip::from(&mut self.r_k)
                .and(beta_new)
                .and(beta_old)
                .for_each(|r, &new, &old| *r = new - old);
            // dr = r_k - prev_residual
            Zip::from(&mut self.dr)
                .and(&self.r_k)
                .and(prev_residual)
                .for_each(|dr, &r, &prev| *dr = r - prev);
            // dx = beta_old - prev_beta
            Zip::from(&mut self.dx)
                .and(beta_old)
                .and(prev_beta)
                .for_each(|dx, &old, &prev| *dx = old - prev);
            let den = self.dr.dot(&self.dr);
            if !den.is_finite() || den < AA1_DAMPING_FLOOR {
                return None;
            }
            let alpha = (self.dr.dot(&self.r_k) / den).clamp(-1.0, 1.0);
            // beta_accel = beta_new - alpha * (dx + dr)
            for i in 0..len {
                self.beta_accel[i] = beta_new[i] - alpha * (self.dx[i] + self.dr[i]);
            }
            if !array_is_finite(&self.beta_accel) {
                return None;
            }
            // The caller copies this borrow into its candidate buffer before the next AA mutation.
            Some(&self.beta_accel)
        }

        pub(crate) fn note_accept(&mut self, iter: usize) {
            self.consecutive_accepts = self.consecutive_accepts.saturating_add(1);
            self.consecutive_rejects = 0;
            if !self.engaged_logged {
                log::info!("[PIRLS-AA1] engaged at iter={}", iter);
                self.engaged_logged = true;
            }
        }

        pub(crate) fn note_reject(&mut self, iter: usize) {
            self.consecutive_rejects = self.consecutive_rejects.saturating_add(1);
            self.consecutive_accepts = 0;
            if !self.disabled
                && self.consecutive_rejects >= AA1_DISABLE_REJECT_THRESHOLD
                && self.consecutive_accepts < 1
            {
                self.disabled = true;
                log::info!(
                    "[PIRLS-AA1] disabled at iter={} reason=consecutive_rejects",
                    iter
                );
            }
        }

        pub(crate) fn update_history(&mut self, beta_old: &Array1<f64>, residual: &Array1<f64>) {
            // AA history must outlive this LM attempt; assign into retained
            // buffers so accepted Fisher steps do not allocate two O(p) clones.
            match self.prev_beta.as_mut() {
                Some(prev) if prev.len() == beta_old.len() => prev.assign(beta_old),
                _ => self.prev_beta = Some(beta_old.to_owned()),
            }
            match self.prev_residual.as_mut() {
                Some(prev) if prev.len() == residual.len() => prev.assign(residual),
                _ => self.prev_residual = Some(residual.to_owned()),
            }
        }
    }

    fn reuse_regularized_hessian_buffer(
        existing: Option<gam_linalg::matrix::SymmetricMatrix>,
        source: &gam_linalg::matrix::SymmetricMatrix,
    ) -> gam_linalg::matrix::SymmetricMatrix {
        match (existing, source.as_dense()) {
            (Some(gam_linalg::matrix::SymmetricMatrix::Dense(mut buf)), Some(src))
                if buf.nrows() == src.nrows() && buf.ncols() == src.ncols() =>
            {
                buf.assign(src);
                gam_linalg::matrix::SymmetricMatrix::Dense(buf)
            }
            _ => source.clone(),
        }
    }

    fn is_lm_retriable_candidate_error(err: &EstimationError) -> bool {
        match err {
            EstimationError::LinearSystemSolveFailed(_)
            | EstimationError::HessianNotPositiveDefinite { .. } => true,
            // The shared log-link jet is an exact function on a declared eta
            // domain. A trial step outside that domain is infeasible, so LM must
            // reject and damp the step rather than either projecting the link or
            // terminating the entire fit.
            EstimationError::InverseLinkDomainViolation { .. }
            | EstimationError::PirlsRowGeometryUnrepresentable { .. } => true,
            EstimationError::InvalidInput(message) => {
                let message = message.to_ascii_lowercase();
                message.contains("nan")
                    || message.contains("non-finite")
                    || message.contains("infinite")
                    || message.contains("overflow")
                    || message.contains("exceeds f64 range")
            }
            // A candidate step that drives the linear predictor into a region
            // where the model's likelihood is structurally infeasible (e.g.
            // survival monotonicity violated, cumulative hazard decreasing) is
            // the natural LM-halving trigger: the proposed step is too
            // aggressive and the gain-ratio guard never gets to see it.
            // Retrying with a larger damping factor collapses towards a
            // feasible region rather than hard-failing at the first infeasible
            // candidate.
            EstimationError::ParameterConstraintViolation(_) => true,
            _ => false,
        }
    }
    // Exhaustion policy is owned by the shared loop guard (#968) — the #874
    // hang was guard drift between sibling branches. The damping-window
    // question delegates here; the count-or-window exhaustion question is
    // `IterationBound::exhausted_at`, answered from guard-owned state, so
    // no branch in this file can re-derive either predicate locally.
    fn lm_can_retry(loop_lambda: f64) -> bool {
        crate::loop_guard::madsen_can_retry(loop_lambda)
    }
    fn lm_nonconvergence_error(
        options: &WorkingModelPirlsOptions,
        last_change: f64,
    ) -> EstimationError {
        EstimationError::PirlsDidNotConverge {
            max_iterations: options.max_iterations,
            last_change,
        }
    }

    if let Some(lb) = options.coefficient_lower_bounds.as_ref() {
        project_coefficients_to_lower_bounds(&mut beta.0, lb);
    }
    let mut lastgradient_norm = f64::INFINITY;
    let mut last_deviance_change = f64::INFINITY;
    let mut last_step_size = 0.0;
    let mut last_step_halving = 0usize;
    // Tracks the gain ratio of the most-recently-accepted step across
    // PIRLS iters. Populates the result's `final_accept_rho` field so
    // outer consumers (cap schedule, convergence guard) can query the
    // inner Newton's last model-fidelity measurement programmatically.
    let mut last_iter_accept_rho: Option<f64> = None;
    let mut max_abs_eta = 0.0;
    let mut status = PirlsStatus::MaxIterationsReached;
    let mut iterations = 0usize;
    // Streak counter for the soft-acceptance plateau check. Every soft
    // criterion the post-loop rescue would apply to a fit that has hit
    // MaxIterations is also evaluated per-iter via [`pirls_soft_acceptance`]
    // — a fit which has functionally converged exits at the iteration it
    // first satisfies the criterion, instead of grinding through the rest
    // of the budget only to be rescued with the same conditions. A single
    // iteration meeting the band is not robust evidence (one noisy step
    // can fake it), so we require two consecutive matches before exiting
    // — virtually free when the optimizer has truly settled, and a
    // principled defence against false positives otherwise.
    let mut plateau_streak = FlatStreak::new(2);
    let mut constrained_objective_plateau_streak =
        FlatStreak::new(CONSTRAINED_OBJECTIVE_PLATEAU_STREAK);
    let has_explicit_constraints =
        options.coefficient_lower_bounds.is_some() || options.linear_constraints.is_some();
    let mut min_penalized_deviance = f64::INFINITY;
    let mut final_state: Option<WorkingState> = None;
    let mut final_state_cache_key: Option<PirlsAcceptedStateCacheKey> = None;
    // Initial gradient norm captured at iter 1 so the post-loop
    // `[PIRLS solve-end]` summary can report the geometric reduction
    // factor `(g_end / g_start)^(1/iters)` — the per-iter convergence
    // rate. r ≪ 0.5 means the inner Newton is converging fast; r > 0.7
    // means it's struggling. Bench runner aggregates this across
    // accepted PIRLS solves to give a per-fit diagnostic.
    let mut initial_gradient_norm: Option<f64> = None;
    let inner_solve_start = std::time::Instant::now();
    let mut newton_direction = Array1::<f64>::zeros(beta.len());
    let mut linear_active_hint: Option<Vec<usize>> =
        options.linear_constraints.as_ref().map(|_| Vec::new());
    let mut bound_active_hint: Option<Vec<usize>> = options
        .coefficient_lower_bounds
        .as_ref()
        .map(|_| Vec::new());
    let mut consecutive_fisher_fallbacks = 0usize;
    // Exact bare-Hessian decrement factorization is paid at most once per
    // contiguous numerical plateau. Meaningful objective progress rearms it.
    let mut exact_decrement_checked_at_plateau = false;
    // AA(1) state — engages only while `force_fisher_for_rest == true`. The
    // initial allocations stay None until the first Fisher-regime iteration,
    // so this is free when PIRLS stays on the observed-information Newton
    // path the whole way through.
    let mut aa_state = AndersonOneState::new();
    let mut force_fisher_for_rest = false;
    // Reused across LM attempts and PIRLS iterations. On acceptance we swap
    // the old beta allocation back into this buffer, so the hot path keeps
    // one O(p) candidate allocation for the whole solve.
    let mut candidate_buf: Array1<f64> = Array1::zeros(beta.len());
    let kkt_tolerance = effective_kkt_tolerance(options);
    if let Some(adaptive) = options.adaptive_kkt_tolerance {
        log::info!(
            "[ADAPTIVE-KKT] outer_g_norm={:.3e} effective_tol={:.3e} floor={:.3e} ceiling={:.3e}",
            adaptive.outer_grad_norm,
            kkt_tolerance,
            adaptive.floor,
            adaptive.ceiling,
        );
    }
    // Pre-allocated buffer for the regularized hessian to avoid O(p²) clone
    // per PIRLS iteration. Reused across iterations when dimensions match.
    let mut regularized_buf: Option<gam_linalg::matrix::SymmetricMatrix> = None;

    // The penalized objective the LM gain ratio and stall detector compare.
    // `dev_scale` is the family's dispersion factor `k` (Gamma shape, Tweedie /
    // fixed-φ Gaussian 1/φ, else 1): the inner gradient and Hessian are built
    // from the `k`-scaled working weight, so the objective value must be
    // `½(k·deviance + penalty)` for the gain ratio (actual ÷ predicted reduction)
    // and the accept test to compare like with like. Using the bare (unscaled)
    // deviance while the step targets `k·D + penalty` freezes the Gamma smooth
    // at heavily-penalized ρ (issue #2128).
    let penalizedobjective = |state: &WorkingState, dev_scale: f64| {
        let mut value = 0.5 * (dev_scale * state.deviance + state.penalty_term);
        if options.firth_bias_reduction
            && let Some(jeffreys_logdet) = state.jeffreys_logdet()
        {
            // Jeffreys/Firth adds the identifiable-subspace Jeffreys term
            // Φ to the log-likelihood,
            // so the PIRLS deviance is reduced by 2 * Φ.
            value -= jeffreys_logdet;
        }
        value
    };

    // Initial Levenberg-Marquardt damping. Seeded from the caller's
    // `initial_lm_lambda` hint when present, with a safety clamp into
    // [1e-9, 1.0]:
    //   * floor 1e-9 matches the LM-internal accept-side floor
    //     (madsen_lm_accept_factor caps shrink at λ → λ/3, and the
    //     post-multiply `.max(1e-9)` enforces this absolute lower bound),
    //     so any positive cached value gets through unchanged.
    //   * ceiling 1.0 covers the gradient-descent regime; values above
    //     that are pathological (the MADSEN_DAMPING_CAP = 1e12 ceiling is the
    //     LM exit condition, well above any sensible warm-start).
    // The runtime layer (`solver/reml/runtime.rs::execute_pirls_if_needed`)
    // applies an *adaptive* clamp before this one, narrowing the range
    // based on the previous solve's halving history (Newton-friendly →
    // [1e-9, 1e-3], default → [1e-6, 1e-3], hard-fit → [1e-3, 1.0]).
    // This PIRLS clamp is defense in depth — it catches a pathological
    // hint from any caller that bypasses the runtime adaptive layer.
    // Cold default `1e-6` matches the original.
    let mut lambda = options
        .initial_lm_lambda
        .map(|v| v.clamp(1e-9, 1.0))
        .unwrap_or(1e-6);
    let lm_max_attempts = options.max_step_halving.max(1);
    // Convergence is decided by `WorkingState::certifies_kkt` /
    // `WorkingState::near_stationary_kkt`, which combine a dimension-based
    // bound  ‖g‖ < τ · √n · max(1, √p)  with a data-driven natural-scale
    // bound  ‖g‖ / (1 + ‖score‖ + ‖S·β‖) < τ  and accept under either.
    // Both certificates are scale-invariant under F → c·F (the additive 1
    // is a NaN-safe floor; for non-trivial fits the natural scale dominates
    // it within one PIRLS iteration). The absolute test ‖g‖ < τ that this
    // replaces was systematically too tight at large-scale n because ‖g‖₂ grows
    // as O(√n) for standardized columns.

    // ─── Observed vs expected information in PIRLS (see response.md Section 3) ───
    //
    // The mixed strategy is used here:
    // - The inner PIRLS iteration uses observed-information curvature when
    //   available (preferred_curvature = Observed for non-canonical links).
    //   This gives faster convergence than Fisher scoring for non-canonical
    //   links, but either choice finds the same mode.
    // - Fisher scoring internally is FINE --- any convergent algorithm works.
    //   If observed curvature fails (non-SPD), we fall back to Fisher scoring.
    // - The requirement is that the output Hessian (which flows
    //   into the outer REML log|H| and trace terms) uses observed information.
    //   This is ensured by `into_final_state()` which stores the
    //   `lasthessian_weights` (observed when available) as `finalweights`.
    //
    // The Laplace approximation int exp(-F(beta)) dbeta uses the actual
    // Hessian nabla^2 F at the actual mode. Replacing with expected Fisher
    // changes the approximation itself --- it becomes a PQL-type surrogate.
    'pirls_loop: for iter in 1..=options.max_iterations {
        iterations = iter;
        // Per-iter wall-clock anchor: lets the [PIRLS iter-end] log below
        // report exactly how long this iteration took. Useful for the
        // adaptive-convergence work (replacing the path #3 schedule
        // bandaid) — we need to see what fraction of inner cost is
        // curvature update vs LM solve vs deviance check, plus per-iter
        // timing distribution at large scale.
        // ApproxKind: TemporarySolverDamping — LM ridge + step-halving
        // schedule are inactive at convergence; fixed point is exact Newton.
        let iter_start = std::time::Instant::now();
        // Start-of-iteration beacon: emits one line BEFORE the curvature-sensitive
        // inner work begins, so CI logs show *which* PIRLS iteration is in flight
        // if the process is killed during `update_with_curvature` or the LM solve.
        log::debug!(
            "[PIRLS] start iter {:>3} | lm_lambda {:.2e} | last_halving {} | last_dev_change {:.3e}",
            iter,
            lambda,
            last_step_halving,
            last_deviance_change
        );
        let preferred_curvature =
            if model.supports_observed_information_curvature() && !force_fisher_for_rest {
                HessianCurvatureKind::Observed
            } else {
                HessianCurvatureKind::Fisher
            };
        let mut used_fisher_fallback_this_iter = false;
        let curvature_start = std::time::Instant::now();
        // The previous iter's LM accept path computed `accepted_state` via
        // `update_candidate(candidate_beta, state.hessian_curvature)` and
        // stored it as `final_state`. The new iter starts at exactly that
        // candidate_beta (line `beta = candidate_beta` in the accept branch),
        // and the working model's `last_*` buffers (eta, mu, weights, ...)
        // are already populated at this beta. Rebuilding the curvature here
        // would reproduce identical numbers at the cost of a full sweep
        // (XᵀWX assembly + PD ridge + gradient) — measured 23 s / iter on
        // the large-scale duchon60 lane (n=320 K, p_eff=42), where it doubled
        // wall-clock per iter on top of the candidate eval that already paid
        // the same cost. Reuse `final_state` only when the cached curvature
        // kind, Firth mode, exact coefficient bits, and any arrow-Schur latent
        // snapshot bits match what this iter requests; the Hessian depends on
        // the working weights/linearization point, so curvature kind alone is
        // not a sufficient freshness predicate.
        // Otherwise (e.g. force_fisher_for_rest just engaged, flipping
        // preferred from Observed → Fisher) fall through to the rebuild path.
        // Smoothing ρ/λ and penalty structure are fixed for this PIRLS call;
        // outer REML constructs a new working model when they change.
        // Iter 1 always rebuilds because no prior accept has populated
        // `final_state`.
        let requested_cache_key =
            PirlsAcceptedStateCacheKey::requested(&beta, preferred_curvature, options);
        let cached_state_matches = iter > 1
            && final_state.is_some()
            && final_state_cache_key.as_ref() == Some(&requested_cache_key);
        let mut state = if cached_state_matches {
            final_state
                .take()
                .expect("cached_state_matches implies final_state.is_some()")
        } else {
            match model.update_with_curvature(&beta, preferred_curvature) {
                Ok(state) => state,
                Err(
                    err @ (EstimationError::InverseLinkDomainViolation { .. }
                    | EstimationError::PirlsRowGeometryUnrepresentable { .. }),
                ) => return Err(err),
                Err(_) if preferred_curvature == HessianCurvatureKind::Observed => {
                    used_fisher_fallback_this_iter = true;
                    consecutive_fisher_fallbacks += 1;
                    if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                        log::info!(
                            "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=iter_start",
                            iter,
                            consecutive_fisher_fallbacks,
                        );
                        force_fisher_for_rest = true;
                    }
                    model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?
                }
                Err(err) => return Err(err),
            }
        };
        let mut curvature_total = curvature_start.elapsed();
        // Log the ACTUAL curvature used, not the preferred one. When
        // Fisher-fallback fires (Observed assembly failed → retried with
        // Fisher), `state.hessian_curvature` correctly reports `Fisher`
        // while `preferred_curvature` is still `Observed`. Logging
        // preferred_curvature here would systematically under-count
        // Fisher fallbacks for the bench runner's `pirls_fisher_frac`
        // diagnostic (commit 971e67ad), masking observed-Hessian PD
        // failures at large scale.
        log::info!(
            "[STAGE] PIRLS update_with_curvature iter={} curvature={:?} elapsed={:.3}s source={}",
            iter,
            state.hessian_curvature,
            curvature_total.as_secs_f64(),
            if cached_state_matches {
                "reused_prev_accept"
            } else {
                "rebuilt"
            },
        );
        // Per-iter LM-loop accumulators. Surface where the inner Newton
        // spends time when the LM has to halve repeatedly: solve-direction
        // work (assemble + factorize + back-solve), candidate evaluation
        // (model.update_candidate — for FLEX margslope this is the per-row
        // sextic-kernel intercept root-find, the dominant cost at large-scale
        // shape per memory/scaling_law_margslope_inner_pirls.md), and the
        // predicted-reduction quadratic form. The breakdown emits at
        // iter-end alongside the existing [PIRLS iter-end] line, giving a
        // structured signal we can aggregate across the run to identify
        // which sub-phase to optimize next.
        let mut lm_solve_total = std::time::Duration::ZERO;
        let mut lm_candidate_total = std::time::Duration::ZERO;
        let mut lm_predred_total = std::time::Duration::ZERO;
        let mut lm_attempts_done = 0usize;
        // Dispersion factor `k` baked into the working weight (Gamma shape,
        // Tweedie/fixed-φ Gaussian 1/φ, else 1). Read AFTER the iter-start
        // `update_with_curvature` above so the once-per-solve Gamma-shape lock is
        // in place; it is constant for the rest of this inner solve.
        let penalized_dev_scale = model.penalized_deviance_scale()?;
        let current_penalized = penalizedobjective(&state, penalized_dev_scale);
        if current_penalized.is_finite() && current_penalized < min_penalized_deviance {
            min_penalized_deviance = current_penalized;
        }
        test_support::record_penalized_deviance(current_penalized);

        // Capture the initial gradient norm at iter 1 (the first iter
        // where `state.gradient` has been computed by `update_with_curvature`).
        // Used by the [PIRLS solve-end] summary log's convergence-rate report.
        if initial_gradient_norm.is_none() {
            let g0_sq: f64 = state
                .gradient
                .iter()
                .map(|g| if g.is_finite() { g * g } else { 0.0 })
                .sum();
            let g0 = g0_sq.sqrt();
            if g0.is_finite() && g0 > 0.0 {
                initial_gradient_norm = Some(g0);
            }
        }

        // Early exit: if the current state has non-finite gradient, the
        // model evaluation has overflowed (eta too extreme).  No Newton
        // step can recover — accept the best state we have.
        let current_grad_finite = state.gradient.iter().all(|g| g.is_finite());
        if !current_grad_finite {
            lastgradient_norm = f64::INFINITY;
            max_abs_eta = inf_norm(state.eta.iter().copied());
            final_state = Some(state);
            // Non-finite-gradient rescue is deviance-plateau based, not a KKT certificate.
            if last_deviance_change.abs() < options.convergence_tolerance {
                status = PirlsStatus::StalledAtValidMinimum;
            }
            break 'pirls_loop;
        }

        // --- Levenberg-Marquardt Step ---

        // Loop to adjust lambda until we accept a step or fail
        // In standard LM, we solve (H + λI)δ = -g
        let mut loop_lambda = lambda;
        // Per-iteration hard bound (#968): ticks at the top of EVERY LM
        // pass — including `continue` paths that reach no reject ritual
        // (Fisher fallback, special cases) — so the unbounded `loop {}`
        // below is structurally bounded no matter which branch a pass
        // takes. Distinct from the reject escalator by design; see the
        // loop_guard module docs.
        let mut lm_bound = IterationBound::new(lm_max_attempts);
        // Snapshot the LM trajectory's starting λ for the
        // `[PIRLS lm-trajectory]` log emitted at iter-end. This is what
        // the runtime-layer adaptive clamp (commit 43be42be) selected for
        // this iter, and the iter-end final λ (after Madsen accept-side
        // shrink/expand) reveals how the LM trajectory moved this iter.
        // Aggregating start→final ratios across a fit tells us whether
        // the textbook LM updates (commits 58ae42d1, d37626e6) are
        // actually moving λ in useful directions at large scale.
        let lm_start_lambda = lambda;
        // Track the gain ratio of the accepted step. Aggregating ρ
        // accepted across iters tells us whether the LM model is
        // well-calibrated: ρ ≈ 1 throughout = healthy Newton;
        // ρ << 1 = model over-states predicted reduction. The
        // trajectory log is emitted only on iter-end fall-through
        // (via `break;` from the LM loop), which always passes
        // through the accept-step assignment; no initial value is
        // needed because rustc proves definite assignment before
        // the trajectory log reads it.
        let lm_accept_rho: Option<f64>;
        // Madsen-Nielsen-Tingleff stateful rejection factor (eq 3.16 in
        // "Methods for non-linear least squares problems", IMM Tech Univ
        // Denmark, 2nd ed 2004): v starts at 2 and doubles on every
        // rejection, so successive bumps are ×2, ×4, ×8, ×16, ... vs the
        // older fixed ×10 every time. The textbook progression gives
        // more chances to find a usable trust radius before
        // `lm_can_retry` declares MADSEN_DAMPING_CAP exhausted; the older ×10
        // hit the ceiling in just 12 rejections (10^12 = MADSEN_DAMPING_CAP),
        // while ×2 doubling needs 40 rejections to exceed the same
        // ceiling — well past `lm_max_attempts`. Restarts on
        // Fisher-fallback (different problem, restart the LM trajectory).
        // The doubling discipline itself is owned by the shared escalator
        // (#968) so no reject branch can apply the damping bump without
        // advancing the schedule.
        let mut madsen_escalator = RejectEscalator::new();
        let mut pending_arrow_latent_restore: Option<Array1<f64>> = None;
        let mut pending_arrow_predicted_reduction: Option<f64>;

        // Copy the hessian into the reusable buffer (avoids allocation after first iteration).
        let mut regularized =
            reuse_regularized_hessian_buffer(regularized_buf.take(), &state.hessian);
        let mut applied_lambda = 0.0_f64;
        // Cache for sparse regularized hessian (reuse for predicted reduction
        // and for in-place diagonal updates on subsequent LM attempts).
        let mut cached_sparse_regularized: Option<SparseColMat<usize, f64>> = None;
        // Track which lambda is currently baked into the cached sparse matrix's
        // diagonal so we can apply a delta update (loop_lambda - sparse_applied_lambda)
        // rather than rebuilding the regularized matrix each attempt.
        let mut sparse_applied_lambda = 0.0_f64;
        // Per-coordinate LM damping scale: D²[i] = clamp(max(H_diag[i], ε),
        // D2_MIN, D2_MAX). Held constant within an LM rejection cluster (only λ
        // varies); recomputed whenever state.hessian changes (Fisher fallback).
        // Using the full penalized-Hessian diagonal avoids the artificial
        // anisotropy that scalar λI introduces across basis/penalty/latent blocks
        // with very different column scales.
        let mut lm_d2 = compute_lm_d2(&state.hessian);
        loop {
            restore_pending_arrow_latent_if_needed(options, &mut pending_arrow_latent_restore);
            pending_arrow_predicted_reduction = None;
            lm_bound.tick();
            lm_attempts_done += 1;
            let attempt_solve_start = std::time::Instant::now();

            // 1. Solve (H + λD²)δ = -g
            // Update diagonal in-place: add (loop_lambda - applied_lambda)*D²[i]
            // per coordinate instead of a scalar shift. This makes LM damping
            // invariant to column-scale anisotropy across basis/penalty blocks.
            if let gam_linalg::matrix::SymmetricMatrix::Dense(ref mut dense) = regularized {
                let delta_lambda = loop_lambda - applied_lambda;
                let dim = dense.nrows();
                for i in 0..dim {
                    dense[[i, i]] += delta_lambda * lm_d2[i];
                }
                applied_lambda = loop_lambda;
            }

            let has_constraints =
                options.linear_constraints.is_some() || options.coefficient_lower_bounds.is_some();
            let direction = match if let Some(h_sparse) = state.hessian.as_sparse() {
                if has_constraints {
                    Err(EstimationError::InvalidInput(
                        "sparse-native PIRLS does not support constrained solves".to_string(),
                    ))
                } else {
                    // First attempt this LM round: materialize the regularized
                    // matrix by rebuilding (ensures every diagonal entry is
                    // present). Subsequent attempts mutate the cached matrix's
                    // diagonal in place — symbolic structure (col_ptr/row_idx)
                    // is identical, only diagonal values change.
                    if cached_sparse_regularized.is_none() {
                        let sparse_reg = add_scaled_diagonal_to_upper_sparse(
                            h_sparse,
                            loop_lambda,
                            lm_d2.as_slice().unwrap(),
                        )?;
                        cached_sparse_regularized = Some(sparse_reg);
                        sparse_applied_lambda = loop_lambda;
                    } else {
                        let delta = loop_lambda - sparse_applied_lambda;
                        if delta != 0.0 {
                            let cached = cached_sparse_regularized.as_mut().unwrap();
                            update_scaled_diagonal_in_place(
                                cached,
                                delta,
                                lm_d2.as_slice().unwrap(),
                            )
                            .map_err(|e| {
                                EstimationError::InvalidInput(format!(
                                    "sparse diagonal in-place update failed: {e}"
                                ))
                            })?;
                            sparse_applied_lambda = loop_lambda;
                        }
                    }
                    let sparse_reg_ref = cached_sparse_regularized.as_ref().unwrap();
                    let factor = factorize_sparse_spd(sparse_reg_ref)?;
                    solve_sparse_spd_into(&factor, &state.gradient, &mut newton_direction)?;
                    newton_direction.mapv_inplace(|g| -g);
                    Ok(())
                }
            } else {
                // Dense path: extract the concrete Array2 once — the compiler
                // ensures we never pass an unresolved SymmetricMatrix downstream.
                let dense_reg = regularized.as_dense().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "PIRLS Newton step requires a dense Hessian but got a non-dense variant"
                            .to_string(),
                    )
                })?;
                if let Some(lin) = options.linear_constraints.as_ref() {
                    solve_newton_directionwith_linear_constraints(
                        dense_reg,
                        &state.gradient,
                        beta.as_ref(),
                        lin,
                        &mut newton_direction,
                        linear_active_hint.as_mut(),
                    )
                } else if let Some(lb) = options.coefficient_lower_bounds.as_ref() {
                    solve_newton_directionwith_lower_bounds(
                        dense_reg,
                        &state.gradient,
                        beta.as_ref(),
                        lb,
                        &mut newton_direction,
                        bound_active_hint.as_mut(),
                    )
                } else if let Some(arrow_cfg) = options.arrow_schur.as_ref() {
                    // Arrow-Schur structured-inner-solve path. The
                    // driver-supplied closure assembles
                    // the bordered (t, β) system at the current β and
                    // current latent t; we solve via per-row d×d
                    // Cholesky + one K×K Schur factor, write the
                    // β-direction into `newton_direction` so the
                    // existing LM gain test / line search evaluates the
                    // same joint candidate `(t + Δt, β + Δβ)`, and push
                    // the latent increment back into the driver via the
                    // `apply_delta_t` callback. Rejected trials are
                    // restored through the driver-owned snapshot/restore
                    // callbacks before the next LM attempt.
                    //
                    // NOTE: this branch exploits the inner-GN
                    // block-diagonality of `H_tt`. The REML outer
                    // gradient w.r.t. `t` carries a shared `Schur⁻¹`
                    // factor — that's a separate plumbing change
                    // handled at the REML driver level, NOT here.
                    assert_eq!(arrow_cfg.n_beta, beta.as_ref().len());
                    match arrow_cfg.build.as_ref()(beta.as_ref()) {
                        None => {
                            // Driver opted out (e.g. latent not yet
                            // initialized). Fall through to β-only path.
                            solve_newton_direction_dense(
                                dense_reg,
                                &state.gradient,
                                &mut newton_direction,
                            )
                        }
                        Some(mut arrow_system) => {
                            // Apply spec-derived block offsets for block-Jacobi
                            // preconditioner — a single call here covers every
                            // family that supplies block_offsets via the config
                            // rather than baking the call into each `build`
                            // closure (issue #287).
                            if let Some(offsets) = arrow_cfg.block_offsets.as_ref() {
                                arrow_system.set_block_offsets(offsets.clone());
                            }
                            let mut solve_options =
                                crate::arrow_schur::ArrowSolveOptions::automatic(arrow_system.k);
                            if let Some(mode) = arrow_cfg.solver_mode {
                                solve_options.mode = mode;
                            } else if arrow_cfg.streaming_chunk_size.is_some() {
                                solve_options.mode = crate::arrow_schur::ArrowSolverMode::Direct;
                            }
                            solve_options.streaming_chunk_size = arrow_cfg.streaming_chunk_size;
                            solve_options.trust_region.radius = arrow_cfg.trust_region_radius;
                            let latent_snapshot = arrow_cfg.snapshot_t.as_ref()();
                            let arrow_solve_result =
                                arrow_system.solve_with_options(0.0, loop_lambda, &solve_options);
                            match arrow_solve_result {
                                Ok((delta_t, delta_beta, pcg_diag)) => {
                                    log::debug!(
                                        "[arrow-Schur] iter {:>3} | k={} | pcg_iters={} | \
                                         precond_calls={} | ridge_escalations={} | \
                                         final_residual={:.3e} | stop={:?}",
                                        iter,
                                        arrow_system.k,
                                        pcg_diag.iterations,
                                        pcg_diag.precond_apply_calls,
                                        pcg_diag.ridge_escalations,
                                        pcg_diag.final_relative_residual,
                                        pcg_diag.stopping_reason,
                                    );
                                    let arrow_predicted_reduction =
                                        match crate::arrow_schur::arrow_bare_quadratic_model_reduction(
                                            &arrow_system,
                                            delta_t.view(),
                                            delta_beta.view(),
                                            0.0,
                                            loop_lambda,
                                        ) {
                                            Ok(value) => value,
                                            Err(e) => {
                                                crate::bail_invalid_estim!(
                                                    "arrow-Schur predicted reduction failed at iter {iter} \
                                                     (loop_lambda={loop_lambda:.3e}): {e}"
                                                );
                                            }
                                        };
                                    // Apply the latent half of the joint
                                    // trial before screening β + Δβ so the
                                    // merit test evaluates the same pair
                                    // that will be committed on acceptance.
                                    arrow_cfg.apply_delta_t.as_ref()(&delta_t);
                                    pending_arrow_latent_restore = Some(latent_snapshot);
                                    pending_arrow_predicted_reduction =
                                        Some(arrow_predicted_reduction);
                                    // Write β-step into the existing
                                    // direction buffer so the rest of
                                    // the LM loop proceeds unchanged.
                                    newton_direction.assign(&delta_beta);
                                    Ok(())
                                }
                                Err(e) => Err(EstimationError::InvalidInput(format!(
                                    "arrow-Schur inner solve failed at iter {iter} \
                                     (loop_lambda={loop_lambda:.3e}): {e}"
                                ))),
                            }
                        }
                    }
                } else {
                    model.solve_unconstrained_direction(
                        &state,
                        loop_lambda,
                        &lm_d2,
                        dense_reg,
                        &mut newton_direction,
                    )
                }
            } {
                Ok(()) => &newton_direction,
                Err(e) => {
                    // The constrained active-set Newton inner solve can fail to
                    // reach a feasible constrained stationary point when the
                    // working-weight Hessian is near-singular — the #1786 low-count
                    // log-link regime, where `W ≈ μ → 0` leaves the KKT system
                    // ill-conditioned (KKT primal feasible but stationarity/dual
                    // blown up). The FIRST recourse is exactly the recourse an
                    // unconstrained singular solve takes: escalate the LM damping
                    // `λ` and retry. `H + λD²` grows better-conditioned as `λ`
                    // increases, so the same active-set solve on the damped system
                    // converges — this is resolution #1 (regularize the
                    // near-singular Hessian) applied through the existing trust
                    // machinery, and it keeps the constrained and unconstrained
                    // singular-solve paths in lockstep.
                    if lm_can_retry(loop_lambda) {
                        lm_solve_total += attempt_solve_start.elapsed();
                        madsen_escalator.escalate(&mut loop_lambda);
                        continue;
                    }
                    if has_constraints {
                        // Damping is exhausted and the constrained solve still can
                        // not produce a step. Rather than hard-error here (which
                        // bubbles to a NON-CONVERGED outer REML whose keep-best then
                        // silently ships the last, possibly INFEASIBLE β — the exact
                        // #1786 defect), fall back to a FEASIBILITY-RESTORING step:
                        // project the current β strictly into the feasible cone and
                        // move toward the projection. This never leaves the feasible
                        // region and lets the PIRLS loop reach the post-loop
                        // feasibility guard, which certifies (and, if needed,
                        // re-projects) the returned coefficients so the shipped model
                        // honors `A·β ≥ 0`. If even the projection is unavailable we
                        // surface the failure honestly instead of shipping garbage.
                        let restored = options.linear_constraints.as_ref().and_then(|lin| {
                            crate::active_set::project_point_strictly_into_feasible_cone(
                                beta.as_ref(),
                                lin,
                            )
                        });
                        match restored {
                            Some(feasible_beta) => {
                                newton_direction.assign(&feasible_beta);
                                newton_direction -= beta.as_ref();
                                &newton_direction
                            }
                            None => {
                                return Err(EstimationError::ParameterConstraintViolation(
                                    format!(
                                        "constrained PIRLS step solve failed at iteration {iter} with damping λ={loop_lambda:.3e} and no feasible projection onto the constraint cone was available: {e}"
                                    ),
                                ));
                            }
                        }
                    } else {
                        // Fallback to gradient descent
                        newton_direction.assign(&state.gradient);
                        newton_direction.mapv_inplace(|g| -g);
                        &newton_direction
                    }
                }
            };
            lm_solve_total += attempt_solve_start.elapsed();
            if !array_is_finite(direction) {
                if lm_can_retry(loop_lambda) {
                    madsen_escalator.escalate(&mut loop_lambda);
                    continue;
                }
                let detail = if has_constraints {
                    "constrained PIRLS produced non-finite step direction"
                } else {
                    "PIRLS produced non-finite step direction"
                };
                restore_pending_arrow_latent_if_needed(options, &mut pending_arrow_latent_restore);
                crate::bail_invalid_estim!(
                    "{detail} at iteration {iter} with damping λ={loop_lambda:.3e}"
                );
            }

            // 2. Compute Predicted Reduction
            // The quadratic model of the penalized objective F at the current
            // β uses the BARE (penalized) Hessian H — the LM damping λ·I is
            // a solver-internal regularizer, not part of F. So:
            //     m(δ) = F(β) + g'δ + 0.5 δ'Hδ
            //     Reduction = m(0) − m(δ) = -(g'δ + 0.5 δ'Hδ)
            // The cached `regularized` / `cached_sparse_regularized` matrices
            // are H + loop_lambda·I (built for the LM step solve), so
            //     δ'(H+λD²)δ = δ'Hδ + λ·Σᵢ D²[i]·δᵢ²
            //               ⇒  δ'Hδ = δ'(H+λD²)δ − λ·Σᵢ D²[i]·δᵢ².
            // Subtracting `0.5·λ·Σᵢ D²[i]·δᵢ²` from the regularized-matrix
            // quadratic recovers the bare-H quadratic without re-doing the
            // matvec on `state.hessian`. With diagonal damping (H + λD²) the
            // correction term generalises the scalar `λ‖δ‖²` to a D²-weighted
            // norm; this keeps the gain-ratio numerator calibrated regardless
            // of the column-scale anisotropy that exists across
            // basis/penalty/latent/tensor blocks. Cross-coordinate issue #295
            // (Arrow-Schur predicted reduction) needs the same generalisation.
            let predred_start = std::time::Instant::now();
            let lin = state.gradient.dot(direction);
            let predicted_reduction =
                if let Some(arrow_reduction) = pending_arrow_predicted_reduction {
                    arrow_reduction
                } else {
                    let q_term = if let Some(sparse_reg) = cached_sparse_regularized.as_ref() {
                        sparse_symmetric_upper_matvec_public(sparse_reg, direction)
                    } else {
                        regularized.dot(direction)
                    };
                    // Σᵢ D²[i]·δᵢ²  (D²-weighted squared norm of the step)
                    let d2_weighted_sq: f64 = direction
                        .iter()
                        .zip(lm_d2.iter())
                        .map(|(di, d2i)| d2i * di * di)
                        .sum();
                    // δ'(H+λD²)δ − λ·Σᵢ D²[i]·δᵢ² = δ'Hδ
                    let quad = 0.5 * (direction.dot(&q_term) - loop_lambda * d2_weighted_sq);
                    -(lin + quad)
                };
            lm_predred_total += predred_start.elapsed();

            // 3. Compute Actual Reduction
            // Reuse the hoisted candidate buffer: fill via assign + in-place add
            // rather than allocating a fresh Array1 per LM attempt.
            if candidate_buf.len() != beta.len() {
                candidate_buf = Array1::zeros(beta.len());
            }
            candidate_buf.assign(beta.as_ref());
            candidate_buf += direction;
            if options.linear_constraints.is_none()
                && let Some(lb) = options.coefficient_lower_bounds.as_ref()
            {
                project_coefficients_to_lower_bounds(&mut candidate_buf, lb);
            }
            // ── AA(1) Anderson acceleration ──────────────────────────────────
            // Active only in the Fisher fixed-point regime (linearly
            // convergent contraction). Treats the LM attempt as a fixed-point
            // step F(beta_old) = beta_old + direction; mixes against the
            // previous iteration's residual to produce an accelerated
            // candidate. If the existing bound-projection / finiteness checks
            // reject the accelerated candidate, fall back to the plain Fisher
            // candidate transparently — no change to the rest of the loop.
            let mut aa_attempt = false;
            if force_fisher_for_rest && !aa_state.disabled {
                let beta_old_ref: &Array1<f64> = beta.as_ref();
                if let Some(beta_accel) = aa_state.aa1_mix(beta_old_ref, &candidate_buf) {
                    candidate_buf.assign(beta_accel);
                    // Apply the same bound projection the loop already runs on
                    // the plain candidate. Treat this as the "existing
                    // validity check": if projection moves the accelerated
                    // candidate (i.e. it would have left the feasible region)
                    // we keep the projected version; finiteness was already
                    // validated inside aa1_mix. No new gates are introduced
                    // here.
                    if options.linear_constraints.is_none()
                        && let Some(lb) = options.coefficient_lower_bounds.as_ref()
                    {
                        project_coefficients_to_lower_bounds(&mut candidate_buf, lb);
                    }
                    if array_is_finite(&candidate_buf) {
                        aa_attempt = true;
                    }
                }
            }
            let candidate_beta = Coefficients::new(std::mem::take(&mut candidate_buf));
            let candidate_eval_start = std::time::Instant::now();
            let candidate_eval_result = model.screen_candidate(
                &candidate_beta,
                direction,
                &state.eta,
                state.hessian_curvature,
            );
            lm_candidate_total += candidate_eval_start.elapsed();
            match candidate_eval_result {
                Ok(candidate_eval) => {
                    let screening_penalized = candidate_eval
                        .penalized_objective(options.firth_bias_reduction, penalized_dev_scale);
                    let screening_reduction = current_penalized - screening_penalized;

                    // 4. Gain Ratio
                    // When predicted reduction is at floating-point noise level
                    // relative to the objective, both predicted and actual are
                    // meaningless — treat as a neutral step (rho = 1) rather
                    // than hard-rejecting on the sign of noise. The floor tracks
                    // the penalized objective's own magnitude (issue #1127): the
                    // predicted/screening reductions and `current_penalized` all
                    // scale as `O(a²)` under `y → a·y`, so a relative floor is
                    // scale-equivariant. The previous `.max(1.0)` absolute floor
                    // pinned it at `1e-14` for a micro-unit response, mismatching
                    // genuine `O(a²)` reductions and biasing the step screening
                    // toward the over-smoothed iterate. A converged objective
                    // (`current_penalized == 0`) yields a `0` floor, so the
                    // `predicted_reduction > floor` branch still governs.
                    let noise_floor = current_penalized.abs() * 1e-14;
                    let screening_rho = if predicted_reduction > noise_floor {
                        screening_reduction / predicted_reduction
                    } else if screening_reduction >= -noise_floor {
                        // Both reductions are noise — accept the step
                        1.0
                    } else {
                        // Genuine increase despite tiny predicted reduction
                        -1.0
                    };

                    // Guard: reject steps that produce non-finite gradients
                    // or extreme linear predictors. Tail size alone is not a
                    // reason to reject a candidate; only non-finite objective
                    // or gradient arithmetic is. Saturation is diagnosed later
                    // by `pirls_soft_acceptance`.
                    let candidate_arithmetic_finite = candidate_eval.arithmetic_finite();

                    if screening_rho > 0.0
                        && screening_penalized.is_finite()
                        && candidate_arithmetic_finite
                    {
                        let accepted_state = if let Some(candidate_state) =
                            candidate_eval.into_full()
                        {
                            candidate_state
                        } else {
                            let accepted_curv_start = std::time::Instant::now();
                            let accepted_curv_result = model
                                .update_with_curvature(&candidate_beta, state.hessian_curvature);
                            curvature_total += accepted_curv_start.elapsed();
                            match accepted_curv_result {
                                Ok(state) => state,
                                Err(err) => {
                                    if !is_lm_retriable_candidate_error(&err) {
                                        restore_pending_arrow_latent_if_needed(
                                            options,
                                            &mut pending_arrow_latent_restore,
                                        );
                                        return Err(err);
                                    }
                                    if lm_bound.exhausted_at(loop_lambda) {
                                        restore_pending_arrow_latent_if_needed(
                                            options,
                                            &mut pending_arrow_latent_restore,
                                        );
                                        return Err(lm_nonconvergence_error(
                                            options,
                                            constrained_stationarity_norm(
                                                &state.gradient,
                                                beta.as_ref(),
                                                options.coefficient_lower_bounds.as_ref(),
                                                options.linear_constraints.as_ref(),
                                            ),
                                        ));
                                    }
                                    candidate_buf = candidate_beta.into();
                                    // Exhaustion was ruled out just above, so
                                    // the retry is unconditional; the two
                                    // historical branches here were identical
                                    // (one indivisible escalation either way).
                                    madsen_escalator.escalate(&mut loop_lambda);
                                    continue;
                                }
                            }
                        };
                        let candidate_penalized =
                            penalizedobjective(&accepted_state, penalized_dev_scale);
                        if candidate_penalized.is_finite()
                            && candidate_penalized < min_penalized_deviance
                        {
                            min_penalized_deviance = candidate_penalized;
                        }
                        let actual_reduction = current_penalized - candidate_penalized;
                        let rho = if predicted_reduction > noise_floor {
                            actual_reduction / predicted_reduction
                        } else if actual_reduction >= -noise_floor {
                            1.0
                        } else {
                            -1.0
                        };
                        if rho > 100.0 && actual_reduction > noise_floor {
                            log::info!(
                                "[PIRLS gain-ratio audit] rho={:.6e} actual_reduction={:.6e} predicted_reduction={:.6e} linear_model_term={:.6e} direction_norm={:.6e} data_reduction={:.6e} penalty_reduction={:.6e} current_deviance={:.6e} candidate_deviance={:.6e} current_penalty={:.6e} candidate_penalty={:.6e}",
                                rho,
                                actual_reduction,
                                predicted_reduction,
                                lin,
                                direction.dot(direction).sqrt(),
                                0.5
                                    * penalized_dev_scale
                                    * (state.deviance - accepted_state.deviance),
                                0.5 * (state.penalty_term - accepted_state.penalty_term),
                                state.deviance,
                                accepted_state.deviance,
                                state.penalty_term,
                                accepted_state.penalty_term,
                            );
                        }
                        if !(rho > 0.0 && candidate_penalized.is_finite()) {
                            if aa_attempt {
                                aa_state.note_reject(iter);
                            }
                            candidate_buf = candidate_beta.into();
                            // Exhaustion guard, identical to the screening-reject
                            // branch below. The screening test admitted this trial
                            // (cheap forward eval looked like a descent) but the full
                            // penalized gain `rho` came back ≤ 0 — i.e. the step is a
                            // genuine non-improver at the current iterate. Without
                            // this guard the loop would bump λ and `continue` forever:
                            // at a flat optimum (e.g. the larger null space of a
                            // cyclic `bs='cc'` penalty, where the gradient is at
                            // machine zero and every trial step lies in a direction
                            // the objective is flat along) `rho` never crosses 0, λ
                            // grows until it overflows to `inf`, the damped solve then
                            // returns δ≈0 so the candidate equals the current iterate,
                            // and `rho` stays ≤ 0 — an unbounded ~0%-CPU spin that
                            // never increments the outer iteration counter (gam#874).
                            // When the LM budget is spent we instead recognize a
                            // genuinely-converged iterate (near-stationary KKT ⇒
                            // StalledAtValidMinimum) or surface the honest
                            // non-convergence, mirroring the guarded reject path.
                            let projected_grad = constrained_stationarity_norm(
                                &state.gradient,
                                beta.as_ref(),
                                options.coefficient_lower_bounds.as_ref(),
                                options.linear_constraints.as_ref(),
                            );
                            let lm_rejection_soft = pirls_soft_acceptance(
                                &state,
                                projected_grad,
                                SoftAcceptProgress::Predicted {
                                    predicted_reduction,
                                    current_penalized,
                                },
                                f64::NAN,
                                options.convergence_tolerance,
                                kkt_tolerance,
                            );
                            if let Some(reason) = lm_rejection_soft {
                                log::debug!(
                                    "[PIRLS] gain-rejection soft acceptance: {reason:?} \
                                     (‖g‖={projected_grad:.3e}, \
                                     predicted_reduction={predicted_reduction:.3e})"
                                );
                                lastgradient_norm = projected_grad;
                                last_deviance_change = 0.0;
                                last_step_size = 0.0;
                                last_step_halving = lm_bound.used();
                                max_abs_eta = inf_norm(state.eta.iter().copied());
                                restore_pending_arrow_latent_if_needed(
                                    options,
                                    &mut pending_arrow_latent_restore,
                                );
                                final_state = Some(state);
                                status = PirlsStatus::StalledAtValidMinimum;
                                break 'pirls_loop;
                            }
                            if lm_bound.exhausted_at(loop_lambda) {
                                lastgradient_norm = projected_grad;
                                if state.near_stationary_kkt(projected_grad, kkt_tolerance) {
                                    status = PirlsStatus::StalledAtValidMinimum;
                                } else {
                                    status = PirlsStatus::LmStepSearchExhausted;
                                }
                                restore_pending_arrow_latent_if_needed(
                                    options,
                                    &mut pending_arrow_latent_restore,
                                );
                                final_state = Some(state);
                                break 'pirls_loop;
                            }
                            madsen_escalator.escalate(&mut loop_lambda);
                            continue;
                        }
                        if preferred_curvature == HessianCurvatureKind::Observed
                            && state.hessian_curvature == HessianCurvatureKind::Observed
                            && !used_fisher_fallback_this_iter
                        {
                            consecutive_fisher_fallbacks = 0;
                        }
                        if aa_attempt {
                            aa_state.note_accept(iter);
                        }
                        // Refresh AA(1) history with the plain Fisher residual
                        // before `beta` is replaced; borrowing here avoids the
                        // speculative O(p) beta/direction clones on rejected LM attempts.
                        if force_fisher_for_rest {
                            aa_state.update_history(beta.as_ref(), direction);
                        }
                        // Accept Step.
                        // Stash the accepted gain ratio for the
                        // [PIRLS lm-trajectory] log emitted at iter-end
                        // AND the result's `final_accept_rho` field so
                        // outer consumers can query model fidelity
                        // programmatically.
                        lm_accept_rho = Some(rho);
                        last_iter_accept_rho = Some(rho);
                        // Update Trust Region (Lambda) — Madsen-Nielsen-Tingleff
                        // smooth Marquardt update. See `madsen_lm_accept_factor`
                        // for the textbook derivation and canonical values.
                        lambda = (loop_lambda * madsen_lm_accept_factor(rho)).max(1e-9);
                        // Accepting commits the latent trial together with β,
                        // so there is no rejected snapshot left to restore.
                        commit_pending_arrow_latent(&mut pending_arrow_latent_restore);

                        // Updates for next iteration. Recycle the previous beta
                        // allocation as the next candidate buffer instead of
                        // allocating a fresh O(p) Array1 on the following iter.
                        let old_beta = std::mem::replace(&mut beta, candidate_beta);
                        candidate_buf = old_beta.into();

                        // Update Iteration Info
                        let candidategrad_norm = constrained_stationarity_norm(
                            &accepted_state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                            options.linear_constraints.as_ref(),
                        );
                        let deviance_change = actual_reduction;

                        iteration_callback(&WorkingModelIterationInfo {
                            iteration: iter,
                            deviance: accepted_state.deviance,
                            gradient_norm: candidategrad_norm,
                            step_size: 1.0,
                            step_halving: lm_bound.used(), // repurpose as attempt count
                        });

                        lastgradient_norm = candidategrad_norm;
                        last_deviance_change = deviance_change;
                        last_step_size = 1.0;
                        last_step_halving = lm_bound.used();
                        max_abs_eta = accepted_state
                            .eta
                            .iter()
                            .copied()
                            .map(f64::abs)
                            .fold(0.0, f64::max);

                        // Check convergence in the current PIRLS coefficient frame.
                        // For active inequality constraints, valid KKT multipliers
                        // must be projected out rather than counted as defects.
                        // The inputs (`accepted_state.gradient`, `beta`, the two
                        // constraint slots) are byte-identical to the call above
                        // that produced `candidategrad_norm` — reuse to avoid a
                        // duplicate active-set projection per accepted LM step.
                        let convergence_grad_norm = candidategrad_norm;

                        // Preserve the structural ridge computed by the model.
                        // LM damping is a transient solver detail and must not
                        // redefine the objective's stabilization ridge.
                        final_state_cache_key = Some(PirlsAcceptedStateCacheKey::accepted(
                            &beta,
                            &accepted_state,
                            options,
                        ));
                        final_state = Some(accepted_state);
                        let final_state_ref = final_state
                            .as_ref()
                            .expect("final_state set immediately above");

                        // Newton-decrement acceptance (Boyd & Vandenberghe §9.5.1):
                        // at the pre-step iterate the squared decrement is
                        //     λ_N²(β) = gᵀ H⁻¹ g  =  −g · d_N,
                        // where d_N = −H⁻¹g is the pure Newton step.
                        //
                        // The direction we just solved is d = −(H + λ_lm·I)⁻¹g,
                        // so −lin = gᵀ(H + λ_lm·I)⁻¹g UNDER-estimates λ_N².
                        // From the resolvent identity
                        //     H⁻¹ = (H + λ_lm·I)⁻¹ + λ_lm·H⁻¹·(H + λ_lm·I)⁻¹,
                        // applied between gᵀ and g and bounded by ‖H⁻¹‖₂ ≤
                        // 1/λ_min(H), we get the *exact* upper bound
                        //     λ_N² ≤ (−lin) · (1 + λ_lm/λ_min(H)).
                        // PIRLS's `ensure_positive_definite_with_ridge` step
                        // guarantees λ_min(H) ≥ `ridge_used` after the ridge
                        // is folded in (with a 1e-12 absolute floor for the
                        // ridge-free case). Multiplying −lin by that
                        // correction makes the test a *provably* faithful
                        // upper bound on the true Newton decrement, removing
                        // the prior heuristic gate `loop_lambda ≤ 1.0`.
                        //
                        // The scale-invariant criterion
                        //     λ_N² / (1 + |F(β)|) ≤ τ²
                        // is the textbook Newton stopping rule: ½λ_N² is the
                        // model's predicted decrease in F from this iterate,
                        // so when it falls below the objective's natural
                        // rounding scale, further inner iterations cannot
                        // improve the certificate. This is an *additional*
                        // acceptance — it never weakens the gradient-norm
                        // tests, only certifies convergence in problems where
                        // ‖g‖ is intrinsically large (very ill-conditioned
                        // designs) but H⁻¹g is already tiny.
                        let f_scale = 1.0 + current_penalized.abs();
                        let lambda_floor = final_state_ref.ridge_used.max(1.0e-12);
                        let nd_correction = 1.0 + loop_lambda / lambda_floor;
                        let newton_decrement_sq_upper = (-lin).max(0.0) * nd_correction;
                        let nd_threshold = kkt_tolerance * kkt_tolerance * f_scale;
                        let nd_pass = newton_decrement_sq_upper <= nd_threshold;
                        // Once realized progress is below the objective's
                        // floating-point noise floor, the damped resolvent bound
                        // above can be arbitrarily loose even though the exact
                        // bare-Hessian decrement is decisive. Compute that exact
                        // certificate once at the plateau. This is restricted to
                        // unconstrained coefficient geometry: constrained KKT
                        // points and Arrow-Schur joint states require their own
                        // projected/block decrement, so the raw beta Hessian is
                        // not a valid certificate for them.
                        let numerical_plateau = actual_reduction.abs() <= noise_floor;
                        let should_check_exact_nd =
                            numerical_plateau && !exact_decrement_checked_at_plateau;
                        exact_decrement_checked_at_plateau = numerical_plateau;
                        let exact_decrement_sq = if should_check_exact_nd
                            && !has_explicit_constraints
                            && options.arrow_schur.is_none()
                        {
                            exact_newton_decrement_sq(final_state_ref)
                        } else {
                            None
                        };
                        let exact_nd_pass = exact_decrement_sq
                            .is_some_and(|decrement_sq| decrement_sq <= nd_threshold);
                        if should_check_exact_nd {
                            log::info!(
                                "[PIRLS exact-decrement] applicable={} decrement_sq={:.6e} threshold={:.6e} pass={} gradient_norm={:.6e} relative_gradient={:.6e} dimension_scale={:.6e} natural_scale={:.6e} objective={:.6e} actual_reduction={:.6e} predicted_reduction={:.6e} linear_model_term={:.6e} direction_norm={:.6e} data_reduction={:.6e} penalty_reduction={:.6e}",
                                !has_explicit_constraints && options.arrow_schur.is_none(),
                                exact_decrement_sq.unwrap_or(f64::NAN),
                                nd_threshold,
                                exact_nd_pass,
                                convergence_grad_norm,
                                final_state_ref.relative_gradient_norm(convergence_grad_norm),
                                final_state_ref.kkt_dimension_scale(),
                                final_state_ref.gradient_natural_scale,
                                final_state_ref.penalized_objective(),
                                actual_reduction,
                                predicted_reduction,
                                lin,
                                direction.dot(direction).sqrt(),
                                0.5
                                    * penalized_dev_scale
                                    * (state.deviance - final_state_ref.deviance),
                                0.5 * (state.penalty_term - final_state_ref.penalty_term),
                            );
                        }

                        // Strict KKT: scale-invariant under EITHER the
                        // dimension-based bound ‖g‖ < τ·√n·max(1,√p) OR the
                        // data-driven natural-scale bound
                        //     ‖g‖ / (1 + ‖score‖ + ‖S·β‖) < τ.
                        // Newton decrement is an independent additional
                        // acceptance for ill-conditioned problems where ‖g‖
                        // is intrinsically large but H⁻¹g is already tiny.
                        if final_state_ref.certifies_kkt(convergence_grad_norm, kkt_tolerance)
                            || nd_pass
                            || exact_nd_pass
                        {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }

                        // An objective plateau or a relative-band soft
                        // acceptance is NOT, on its own, a constrained
                        // stationarity certificate: the inner solve can flatten
                        // the penalized deviance — or drive the projected
                        // gradient below an `objective_scale`-relative band — on
                        // a near-vertex face of a curvature cone
                        // (shape=convex/concave) while the constrained KKT
                        // residual still stalls at ~1e-4, far above the outer
                        // startup gate's *absolute* acceptance tolerance.
                        // Soft-accepting such a point makes the fit's *success*
                        // depend on which ρ the seed loop happens to start from
                        // (a warm-start cache hands a pre-converged ρ whose
                        // constrained optimum binds fewer rows and certifies
                        // cleanly; a cold cache starts at an over-smoothed ρ
                        // whose optimum sits on the stalled face), so the same
                        // data+formula returns a different curve — or aborts —
                        // depending on cache state (#873). Gate every
                        // soft-acceptance exit of a *constrained* fit on the SAME
                        // degeneracy-aware constraint-KKT band the outer
                        // validation gate (`enforce_constraint_kkt`) applies, so
                        // a point the inner solve declares a valid minimum is one
                        // the outer gate will also accept regardless of the seed
                        // ρ. A genuinely rank-deficient active face (the
                        // underdetermined I-spline case the plateau branch was
                        // built for) keeps its relaxed stationarity band; a
                        // non-degenerate stall (the #873 face) is held to the
                        // strict band and the solve keeps iterating.
                        let soft_accept_kkt_ok = !has_explicit_constraints
                            || constraint_kkt_admits_soft_accept(
                                options,
                                beta.as_ref(),
                                &final_state_ref.gradient,
                                kkt_tolerance,
                            );

                        // Soft acceptance: every criterion the post-loop
                        // rescue would apply to a fit that has hit
                        // MaxIterations, evaluated per-iter so a fit that
                        // has functionally converged exits at the iteration
                        // it first satisfies the criterion instead of
                        // grinding through the rest of the budget only to
                        // be rescued with the same conditions. The streak
                        // requirement (≥2 consecutive matches) defends
                        // against a single noisy step briefly satisfying
                        // the band — when the optimizer has truly settled,
                        // two consecutive matches cost only one extra
                        // iteration of inner work and give principled
                        // protection against false positives. For a
                        // constrained fit the soft acceptance additionally
                        // requires the constraint-KKT band above, so a
                        // relative-band / boundary-saturation plateau on a
                        // non-degenerate cone face cannot mask a stalled,
                        // outer-gate-rejected iterate (#873).
                        match pirls_soft_acceptance(
                            final_state_ref,
                            convergence_grad_norm,
                            SoftAcceptProgress::Realized {
                                dev_change: deviance_change,
                            },
                            max_abs_eta,
                            options.convergence_tolerance,
                            kkt_tolerance,
                        )
                        .filter(|_| soft_accept_kkt_ok)
                        {
                            Some(reason) => {
                                if plateau_streak.note(true) == LoopVerdict::Plateaued {
                                    log::debug!(
                                        "[PIRLS] iter {iter} early-exit on soft acceptance: \
                                         {reason:?} (‖g‖={convergence_grad_norm:.3e}, \
                                         Δdev={deviance_change:.3e})"
                                    );
                                    status = PirlsStatus::StalledAtValidMinimum;
                                    break 'pirls_loop;
                                }
                            }
                            None => {
                                plateau_streak.note(false);
                            }
                        }

                        // Explicitly constrained fits can reach a valid
                        // bounded optimum with a flat objective trace while
                        // the raw projected-gradient certificate remains
                        // noisy, especially when small monotone I-spline
                        // bases are underdetermined. Accept only a long
                        // streak of finite, monotone, sub-tolerance
                        // objective movement with eta safely away from the
                        // clipping boundary. This is deliberately separate
                        // from the two-iteration soft-acceptance gate above:
                        // unconstrained one-off plateaus must still run out
                        // as MaxIterationsReached.
                        //
                        // Gate composition for the long streak (vs the fast
                        // path's stationarity band):
                        //
                        // * `model_progress_exhausted` — the accepted step's
                        //   OWN quadratic model predicted only sub-tolerance
                        //   progress. This is the discriminator that keeps a
                        //   value↔gradient desync (analytic gradient
                        //   promising progress the value never realizes) from
                        //   ever certifying: a desynced model keeps predicting
                        //   above-tolerance reductions, breaking the streak,
                        //   while a genuinely exhausted direction predicts
                        //   next-to-nothing for 20 consecutive accepted steps.
                        //   A small-but-not-tiny gradient along an almost-flat
                        //   ray (e.g. ‖g‖ ~ 50× the near-stationary band with
                        //   per-step gains ~10⁻⁹·scale) is exactly the
                        //   progress-exhausted stall this branch certifies —
                        //   the stationarity band would starve it into
                        //   MaxIterationsReached after burning the entire
                        //   budget on numerically invisible progress.
                        // * `constraint_kkt_admits_progress_exhausted_stall`
                        //   — primal/dual/complementarity cleanliness inside
                        //   the outer gate's bands plus a hard refusal of
                        //   rank-deficient working sets, so the #873
                        //   degenerate-vertex stall still cannot be accepted
                        //   here (it remains confined to the fast path's
                        //   relaxed degenerate band, which demands
                        //   stationarity).
                        // Scale-equivariant objective magnitude (issue #1127):
                        // the predicted reduction and Δdeviance compared against
                        // this band both scale as `O(a²)` under a response
                        // rescaling `y → a·y`, so the band must track the
                        // objective's own magnitude rather than the absolute
                        // `.max(1.0)` floor, which pinned it at `1.0` for a
                        // micro-unit response and accepted a non-converged
                        // constrained iterate. Well-scaled fits (`|obj| ≳ 1`)
                        // are unchanged.
                        let objective_scale =
                            final_state_ref.deviance.abs() + final_state_ref.penalty_term.abs();
                        let plateau_band = options.convergence_tolerance * objective_scale * 0.1;
                        let model_progress_exhausted = predicted_reduction.is_finite()
                            && predicted_reduction.abs() <= plateau_band;
                        let strict_objective_plateau = has_explicit_constraints
                            && deviance_change.is_finite()
                            && deviance_change >= 0.0
                            && deviance_change.abs()
                                // Objective-plateau progress test, not a KKT certificate.
                                <= plateau_band
                            && model_progress_exhausted
                            && max_abs_eta.is_finite()
                            && max_abs_eta < PIRLS_ETA_ABS_CAP * 0.5
                            && constraint_kkt_admits_progress_exhausted_stall(
                                options,
                                beta.as_ref(),
                                &final_state_ref.gradient,
                                kkt_tolerance,
                            );
                        if constrained_objective_plateau_streak.note(strict_objective_plateau)
                            == LoopVerdict::Plateaued
                        {
                            log::debug!(
                                "[PIRLS] iter {iter} early-exit on constrained objective \
                                 plateau (streak={}, ‖g‖={convergence_grad_norm:.3e}, \
                                 Δdev={deviance_change:.3e})",
                                constrained_objective_plateau_streak.streak(),
                            );
                            status = PirlsStatus::StalledAtValidMinimum;
                            break 'pirls_loop;
                        }

                        break; // Break inner lambda loop, continue outer pirls loop
                    } else {
                        candidate_buf = candidate_beta.into();
                        if aa_attempt {
                            aa_state.note_reject(iter);
                        }
                        if state.hessian_curvature == HessianCurvatureKind::Observed
                            && !used_fisher_fallback_this_iter
                        {
                            used_fisher_fallback_this_iter = true;
                            consecutive_fisher_fallbacks += 1;
                            if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                                log::info!(
                                    "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=gain_rejection",
                                    iter,
                                    consecutive_fisher_fallbacks,
                                );
                                force_fisher_for_rest = true;
                            }
                            // Mid-LM-loop Fisher fallback: the Observed
                            // curvature succeeded at iter-start but the
                            // candidate evaluation produced a bad gain
                            // ratio (or non-finite gradient / extreme
                            // eta), suggesting the Observed Hessian is
                            // unreliable for this β region. Distinct
                            // signal from iter-start Fisher fallback
                            // (Observed assembly itself failed). Tagged
                            // with `gain_rejection` so the runner
                            // aggregator can count both reasons.
                            log::info!(
                                "[PIRLS] mid-iter Fisher fallback iter={} reason=gain_rejection",
                                iter,
                            );
                            restore_pending_arrow_latent_if_needed(
                                options,
                                &mut pending_arrow_latent_restore,
                            );
                            let fisher_fallback_start = std::time::Instant::now();
                            state =
                                model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                            curvature_total += fisher_fallback_start.elapsed();
                            regularized =
                                reuse_regularized_hessian_buffer(Some(regularized), &state.hessian);
                            applied_lambda = 0.0;
                            cached_sparse_regularized = None;
                            sparse_applied_lambda = 0.0;
                            loop_lambda = lambda;
                            lm_d2 = compute_lm_d2(&state.hessian);
                            // Different problem (Hessian curvature changed):
                            // restart the Madsen rejection-factor trajectory.
                            madsen_escalator.restart();
                            continue;
                        }
                        // Reject Step
                        let stategrad_norm = constrained_stationarity_norm(
                            &state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                            options.linear_constraints.as_ref(),
                        );
                        let projected_grad = stategrad_norm;
                        // Near stationarity with a noise-floor predicted reduction:
                        // the screening rejected all candidates, but the gradient
                        // is tiny and the model predicts an essentially-zero step.
                        // Routed through the unified soft-acceptance helper so
                        // this branch stays in lockstep with the per-iter and
                        // post-loop checks. Only the NearStationaryPlateau branch
                        // can fire here — the helper gates the η-cap and
                        // relative-band branches behind a Realized Δdev signal,
                        // which we don't have without an accepted step.
                        let lm_rejection_soft = pirls_soft_acceptance(
                            &state,
                            projected_grad,
                            SoftAcceptProgress::Predicted {
                                predicted_reduction,
                                current_penalized,
                            },
                            // `pirls_soft_acceptance` returns early on the
                            // `Predicted` arm before reading `max_abs_eta`, so
                            // skip the redundant `O(n)` |η| sweep here. The
                            // accept-branch below recomputes it when needed.
                            f64::NAN,
                            options.convergence_tolerance,
                            kkt_tolerance,
                        );
                        let near_stationary_pass =
                            state.near_stationary_kkt(projected_grad, kkt_tolerance);

                        if let Some(reason) = lm_rejection_soft {
                            log::debug!(
                                "[PIRLS] LM-rejection soft acceptance: {reason:?} \
                                 (‖g‖={projected_grad:.3e}, \
                                 predicted_reduction={predicted_reduction:.3e})"
                            );
                            lastgradient_norm = stategrad_norm;
                            last_deviance_change = 0.0;
                            last_step_size = 0.0;
                            last_step_halving = lm_bound.used();
                            max_abs_eta = inf_norm(state.eta.iter().copied());
                            // `state` is unused after `break 'pirls_loop` — move it
                            // instead of cloning to avoid an n+p² full-state copy.
                            restore_pending_arrow_latent_if_needed(
                                options,
                                &mut pending_arrow_latent_restore,
                            );
                            final_state = Some(state);
                            status = PirlsStatus::StalledAtValidMinimum;
                            break 'pirls_loop;
                        }

                        if lm_bound.exhausted_at(loop_lambda) {
                            lastgradient_norm = stategrad_norm;
                            // Only accept "stalled but valid" when we are near stationarity.
                            // Otherwise report MaxIterationsReached so callers can fail fast.
                            if near_stationary_pass {
                                status = PirlsStatus::StalledAtValidMinimum;
                            } else {
                                // Surface what actually exhausted: damping reached its
                                // ceiling, retry counter exhausted, or lambda went non-
                                // finite. The collapsed status hides that distinction;
                                // this debug log restores it.
                                let ceiling = !lm_can_retry(loop_lambda);
                                let attempts_used = lm_bound.count_exhausted();
                                let max_abs_eta_now = state
                                    .eta
                                    .iter()
                                    .copied()
                                    .map(f64::abs)
                                    .fold(0.0_f64, f64::max);
                                let relative_grad = state.relative_gradient_norm(projected_grad);
                                log::debug!(
                                    "[PIRLS] LM step search exhausted at iter={}: \
                                     attempts={}/{} lambda={:.3e} (ceiling={}) \
                                     projected_grad={:.3e} (relative={:.3e}) \
                                     current_pen={:.6e} predicted_reduction={:.3e} \
                                     max|eta|={:.1} attempts_exhausted={}",
                                    iter,
                                    lm_bound.used(),
                                    lm_bound.max(),
                                    loop_lambda,
                                    ceiling,
                                    projected_grad,
                                    relative_grad,
                                    current_penalized,
                                    predicted_reduction,
                                    max_abs_eta_now,
                                    attempts_used,
                                );
                                status = PirlsStatus::LmStepSearchExhausted;
                            }
                            // Preserve the structural ridge from the model state.
                            // `state` is unused after `break 'pirls_loop` — move it
                            // instead of cloning to avoid an n+p² full-state copy.
                            restore_pending_arrow_latent_if_needed(
                                options,
                                &mut pending_arrow_latent_restore,
                            );
                            final_state = Some(state);
                            break 'pirls_loop;
                        }
                        madsen_escalator.escalate(&mut loop_lambda);
                    }
                }
                Err(err) => {
                    candidate_buf = candidate_beta.into();
                    if state.hessian_curvature == HessianCurvatureKind::Observed
                        && !used_fisher_fallback_this_iter
                    {
                        used_fisher_fallback_this_iter = true;
                        consecutive_fisher_fallbacks += 1;
                        if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                            log::info!(
                                "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=candidate_err",
                                iter,
                                consecutive_fisher_fallbacks,
                            );
                            force_fisher_for_rest = true;
                        }
                        // Mid-LM-loop Fisher fallback: the candidate
                        // evaluation itself returned Err (e.g., model
                        // overflowed at the proposed β + δ). Tagged with
                        // `candidate_err` to distinguish from the
                        // gain-rejection variant above; both indicate
                        // mid-iter Observed-curvature unreliability,
                        // but candidate_err is a stronger signal
                        // (numerical breakdown, not just bad gain).
                        log::info!(
                            "[PIRLS] mid-iter Fisher fallback iter={} reason=candidate_err",
                            iter,
                        );
                        restore_pending_arrow_latent_if_needed(
                            options,
                            &mut pending_arrow_latent_restore,
                        );
                        let fisher_err_start = std::time::Instant::now();
                        state = model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                        curvature_total += fisher_err_start.elapsed();
                        regularized =
                            reuse_regularized_hessian_buffer(Some(regularized), &state.hessian);
                        applied_lambda = 0.0;
                        cached_sparse_regularized = None;
                        sparse_applied_lambda = 0.0;
                        loop_lambda = lambda;
                        lm_d2 = compute_lm_d2(&state.hessian);
                        // Different problem (Hessian curvature changed):
                        // restart the Madsen rejection-factor trajectory.
                        madsen_escalator.restart();
                        continue;
                    }
                    if !is_lm_retriable_candidate_error(&err) {
                        restore_pending_arrow_latent_if_needed(
                            options,
                            &mut pending_arrow_latent_restore,
                        );
                        return Err(err);
                    }
                    if lm_bound.exhausted_at(loop_lambda) {
                        restore_pending_arrow_latent_if_needed(
                            options,
                            &mut pending_arrow_latent_restore,
                        );
                        return Err(lm_nonconvergence_error(
                            options,
                            constrained_stationarity_norm(
                                &state.gradient,
                                beta.as_ref(),
                                options.coefficient_lower_bounds.as_ref(),
                                options.linear_constraints.as_ref(),
                            ),
                        ));
                    }
                    // Retry only clearly numerical candidate-evaluation failures.
                    madsen_escalator.escalate(&mut loop_lambda);
                }
            }
        } // end loop (lambda search)
        // Recycle the regularized hessian buffer for the next iteration.
        regularized_buf = Some(regularized);
        // Per-iter wall-clock log: lets us see in CI logs how each
        // inner-Newton iter spent its time. Includes the final LM
        // damping (lambda) so we can see when the LM step search has
        // re-stabilized vs is still struggling. Foundation for the
        // adaptive-convergence work (task #3) — the path #3 schedule
        // is currently hardcoded; once we have per-iter timing we can
        // exit early when the iteration is cheap (small change) AND
        // the residual is small.
        let iter_elapsed = iter_start.elapsed();
        log::info!(
            "[PIRLS iter-end] iter={:>3} elapsed={:.4}s lm_lambda={:.2e} g_norm={:.3e} last_dev_change={:.3e} last_halving={}",
            iter,
            iter_elapsed.as_secs_f64(),
            lambda,
            lastgradient_norm,
            last_deviance_change,
            last_step_halving,
        );
        // Per-iter LM-loop breakdown: tells us where the inner Newton
        // spent time. Sum of (curvature + solve + predred + candidate) is
        // a lower bound on iter_elapsed; the residual is everything else
        // (bookkeeping, soft-acceptance checks, KKT certification, etc).
        // For FLEX margslope at large-scale shape we expect candidate to
        // dominate (per-row sextic-kernel intercept root-find, see
        // memory/scaling_law_margslope_inner_pirls.md). For dense
        // standard-GAM with no per-row Newton inner, solve typically
        // dominates because of the O(p³) Cholesky in the LM solve.
        // Knowing which path is hot tells us where the next principled
        // optimization should land.
        if log::log_enabled!(log::Level::Info) {
            let timed_total =
                curvature_total + lm_solve_total + lm_predred_total + lm_candidate_total;
            let other_total = iter_elapsed.saturating_sub(timed_total);
            log::info!(
                "[PIRLS iter-breakdown] iter={:>3} attempts={} curvature={:.3}s solve={:.3}s predred={:.3}s candidate={:.3}s other={:.3}s",
                iter,
                lm_attempts_done,
                curvature_total.as_secs_f64(),
                lm_solve_total.as_secs_f64(),
                lm_predred_total.as_secs_f64(),
                lm_candidate_total.as_secs_f64(),
                other_total.as_secs_f64(),
            );
        }
        // Per-iter LM trajectory: validates that the textbook Madsen
        // accept (commit 58ae42d1) and reject (d37626e6) updates plus
        // the runtime adaptive λ clamp (43be42be) are moving the trust
        // region in useful directions.
        //   start_lambda  : λ at iter start (after runtime adaptive
        //                   clamp; matches PIRLS's own safety clamp)
        //   final_lambda  : λ written for the next iter (after Madsen
        //                   accept-side shrink/expand, OR last
        //                   loop_lambda if rejection-exhausted)
        //   ratio         : final/start, log10 — distribution tells
        //                   us the per-iter trust-region trajectory.
        //                   Healthy Newton: ratio < 0 (shrink). Hard:
        //                   ratio > 0 (expand). Mostly stationary:
        //                   ratio ≈ 0.
        //   accept_rho    : gain ratio of the accepted step. ≈1 means
        //                   the quadratic model was faithful; <0.5
        //                   means it overstated the predicted
        //                   reduction. NaN on rejection-exhausted.
        if log::log_enabled!(log::Level::Info) {
            let lambda_ratio_log10 = if lm_start_lambda > 0.0 && lambda > 0.0 {
                (lambda / lm_start_lambda).log10()
            } else {
                f64::NAN
            };
            log::info!(
                "[PIRLS lm-trajectory] iter={:>3} start_lambda={:.3e} final_lambda={:.3e} \
                 log10_ratio={:.3} accept_rho={:.3} attempts={}",
                iter,
                lm_start_lambda,
                lambda,
                lambda_ratio_log10,
                lm_accept_rho.unwrap_or(f64::NAN),
                lm_attempts_done,
            );
        }

        // NOTE: there is deliberately NO wall-clock-driven "adaptive
        // early-exit" here (formerly: accept convergence when an iter ran
        // in <25% of the per-iter EMA while deviance/gradient sat within a
        // 10× relaxed band). A convergence verdict keyed on wall-clock is
        // non-deterministic under CPU contention — the same fit converges to
        // a different β in a parallel sweep than it does run alone, which
        // cascades into different outer seed screening and load-unstable
        // fire/collapse decisions downstream (gam#979). It also accepted
        // iterates up to 10× outside `convergence_tolerance`, an
        // unrequested weakening of the inner certificate. Convergence is
        // certified only by the deterministic mathematical tests above.
    }

    // Solve-end summary: one line per accepted (or rescued) PIRLS solve
    // capturing the per-iter geometric convergence rate and total
    // wall-clock. The bench runner aggregates these across all PIRLS
    // solves in a fit so CI logs end with a per-fit verdict on inner-
    // Newton convergence health: rate < 0.5 = healthy Newton; rate ≥ 0.7
    // = struggling (likely stuck near singular geometry or a
    // flat-warm-start that the predictor failed to refine).
    if log::log_enabled!(log::Level::Info) {
        let total_iters = iterations.max(1) as f64;
        let convergence_rate = match initial_gradient_norm {
            Some(g0) if g0 > 0.0 && lastgradient_norm.is_finite() => {
                let ratio = (lastgradient_norm / g0).max(1e-30);
                ratio.powf(1.0 / total_iters)
            }
            _ => f64::NAN,
        };
        log::info!(
            "[PIRLS solve-end] iters={} elapsed={:.4}s g_norm_initial={:.3e} g_norm_final={:.3e} convergence_rate={:.3e} status={:?}",
            iterations,
            inner_solve_start.elapsed().as_secs_f64(),
            initial_gradient_norm.unwrap_or(f64::NAN),
            lastgradient_norm,
            convergence_rate,
            status,
        );
    }

    let mut state = final_state.ok_or(EstimationError::PirlsDidNotConverge {
        max_iterations: options.max_iterations,
        last_change: lastgradient_norm,
    })?;

    // ── Undamped Newton polish (#1122) ──────────────────────────────────────
    //
    // Convergence is certified by `certifies_kkt` / the Newton-decrement bound,
    // both of which are *relative* tests. For a Gaussian-identity (exactly
    // quadratic) problem they fire at the very first accepted LM step — while
    // the cold-start damping `loop_lambda` (default 1e-6) is still folded into
    // the solve. The accepted iterate therefore satisfies the *damped* normal
    // equations `(H + λ_lm·D²)β̂ = X'Wz`, not the bare `Hβ̂ = X'Wz`, leaving a
    // stationarity residual
    //
    //     g_bare = Hβ̂ − X'Wz = λ_lm·D²·β̂ ≈ δ·β̂,   δ ≈ λ_lm·mean(diag H),
    //
    // which is parallel to β̂ and tiny (here ‖g‖≈2e-5, δ≈1.7e-6). The inner
    // solver's own contract (see the iter-loop note: "LM ridge + step-halving
    // schedule are inactive at convergence; fixed point is exact Newton") and
    // `WorkingState::near_stationary_kkt`'s comment both call this out as the
    // LM-ridge bias. Inner-only it is negligible; but the OUTER REML gradient
    // is assembled by the envelope theorem `∂V/∂ψ = ∂V/∂ψ|_β̂` which assumes
    // EXACT stationarity. Any residual `g_bare` makes the analytic outer
    // gradient omit the term `g_bare·(dβ̂/dψ)` that a finite difference of the
    // criterion (which re-solves PIRLS at ψ±h) does see — the #1122 iso-κ
    // Matérn DESYNC, amplified by the profiled-REML datafit prefactor
    // `(denom/2)·(dp_c'/dp_c)` (~259 at θ₀).
    //
    // Restore the documented invariant: take ONE exact, *undamped* (λ=0)
    // Newton step `β̂ ← β̂ − H⁻¹ g_bare` on the bare penalized Hessian once the
    // loop has converged. For a quadratic problem this lands on the exact
    // optimum (g_bare → 0); for a genuinely nonlinear family it is a pure
    // Newton refinement of an already-converged point. It is gated on three
    // safety conditions so it can NEVER worsen a fit:
    //   (1) unconstrained only — a constrained KKT point carries multipliers,
    //       so its residual is not a plain gradient to be zeroed;
    //   (2) the bare Hessian factorizes (PD) and the step is finite;
    //   (3) the re-evaluated state STRICTLY reduces the stationarity residual.
    // Condition (3) makes the polish Pareto-safe: a fit that is already
    // machine-stationary (residual at round-off) sees no accepted step, so
    // existing golden values are untouched; only LM-ridge-biased iterates move.
    let polish_allowed = options.linear_constraints.is_none()
        && options.coefficient_lower_bounds.is_none()
        && options.arrow_schur.is_none();
    if polish_allowed {
        if let Some(bare_h) = state.hessian.as_dense() {
            let g_norm_before =
                constrained_stationarity_norm(&state.gradient, beta.as_ref(), None, None);
            // Only bother when there is a residual worth removing and the
            // gradient/Hessian are finite — skip the work for already-exact fits.
            let bare_finite = state.gradient.iter().all(|v| v.is_finite())
                && bare_h.iter().all(|v| v.is_finite());
            if g_norm_before > 0.0 && bare_finite {
                // `solve_direction_with_dense_factor` returns the Newton
                // DIRECTION d = −H⁻¹g (sign already applied), so the polished
                // iterate is β̂ + d. Factorize the BARE (undamped) penalized
                // Hessian — `state.hessian` carries no LM ridge (the damping
                // lived only on the throwaway `regularized` clone in the loop).
                let direction = StableSolver::new()
                    .factorize(bare_h)
                    .ok()
                    .map(|factor| {
                        let mut d = Array1::<f64>::zeros(state.gradient.len());
                        solve_direction_with_dense_factor(&factor, &state.gradient, &mut d);
                        d
                    });
                if let Some(direction) = direction {
                    let step_finite = direction.iter().all(|v| v.is_finite());
                    // Guard against a runaway step: an exact Newton refinement
                    // of a converged iterate is small relative to ‖β̂‖. A large
                    // step signals a near-singular bare H (the LM damping was
                    // load-bearing) — decline rather than risk a worse point.
                    let beta_norm_sq = beta.as_ref().dot(beta.as_ref());
                    let step_norm_sq = direction.dot(&direction);
                    let step_reasonable =
                        step_finite && (step_norm_sq <= 0.25 * beta_norm_sq.max(1.0));
                    if step_reasonable {
                        let polished: Array1<f64> = beta.as_ref() + &direction;
                        if polished.iter().all(|v| v.is_finite()) {
                            let polished_beta = Coefficients::new(polished);
                            if let Ok(polished_state) =
                                model.update_with_curvature(&polished_beta, state.hessian_curvature)
                            {
                                let g_norm_after = constrained_stationarity_norm(
                                    &polished_state.gradient,
                                    polished_beta.as_ref(),
                                    None,
                                    None,
                                );
                                // Commit ONLY on a strict improvement, and only
                                // when the polished objective did not increase
                                // (a quadratic Newton step cannot increase F, so
                                // this rejects only nonlinear-family overshoot).
                                // Same dispersion scale `k` as the gain-ratio
                                // objective above (the shape lock is constant
                                // across this inner solve); read locally since
                                // this polish branch sits outside the iter-start
                                // binding's scope.
                                let polish_dev_scale = model.penalized_deviance_scale()?;
                                let obj_before = penalizedobjective(&state, polish_dev_scale);
                                let obj_after =
                                    penalizedobjective(&polished_state, polish_dev_scale);
                                let objective_ok = !obj_after.is_finite()
                                    || !obj_before.is_finite()
                                    || obj_after <= obj_before + obj_before.abs().max(1.0) * 1e-12;
                                if g_norm_after.is_finite()
                                    && g_norm_after < g_norm_before
                                    && objective_ok
                                {
                                    log::debug!(
                                        "[PIRLS] undamped Newton polish (#1122): \
                                         ‖g‖ {g_norm_before:.3e} → {g_norm_after:.3e} \
                                         (‖step‖={:.3e})",
                                        step_norm_sq.sqrt()
                                    );
                                    beta = polished_beta;
                                    state = polished_state;
                                    lastgradient_norm = g_norm_after;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Post-loop rescue: use the constrained stationarity residual in the
    // current PIRLS basis, not the raw gradient norm.
    let final_projected_grad = constrained_stationarity_norm(
        &state.gradient,
        beta.as_ref(),
        options.coefficient_lower_bounds.as_ref(),
        options.linear_constraints.as_ref(),
    );
    if status.is_failed_max_iterations() {
        // Strict KKT met after the loop bailed: reclassify as a valid
        // (if non-strictly-converged) minimum. The remaining soft-acceptance
        // criteria (near-stationary plateau, boundary saturation, relative
        // band) are checked uniformly through `pirls_soft_acceptance` so the
        // post-loop rescue and the per-iter early-exit stay in lockstep —
        // anything accepted here is also a candidate for early-exit, and
        // anything that meets the early criterion would have been rescued
        // here.
        if state.certifies_kkt(final_projected_grad, kkt_tolerance) {
            log::debug!(
                "[PIRLS] post-loop rescue: strict KKT after MaxIterations \
                 (‖g‖={final_projected_grad:.3e})"
            );
            status = PirlsStatus::StalledAtValidMinimum;
        } else if pirls_soft_acceptance(
            &state,
            final_projected_grad,
            SoftAcceptProgress::Realized {
                dev_change: last_deviance_change,
            },
            max_abs_eta,
            options.convergence_tolerance,
            kkt_tolerance,
        )
        // A constrained fit may only be rescued onto a plateau / relative-band /
        // boundary-saturation soft acceptance when its constraint-KKT residual
        // is within the SAME degeneracy-aware band the outer gate applies, so
        // the rescue cannot certify a stalled non-degenerate cone face the outer
        // gate would reject (#873). Unconstrained fits keep the existing rescue.
        .filter(|_| {
            !has_explicit_constraints
                || constraint_kkt_admits_soft_accept(
                    options,
                    beta.as_ref(),
                    &state.gradient,
                    kkt_tolerance,
                )
        })
        .is_some()
        {
            log::debug!(
                "[PIRLS] post-loop rescue on soft acceptance \
                 (‖g‖={final_projected_grad:.3e}, \
                 Δdev={last_deviance_change:.3e})"
            );
            status = PirlsStatus::StalledAtValidMinimum;
        }
    }

    // Post-loop constraint-feasibility guard (#1786).
    //
    // A shape-constrained smooth (`monotone_increasing`, `convex`, `concave`)
    // under a NON-canonical log-link family (Poisson / Gamma) in the low-count
    // regime has IRLS working weights `W ≈ μ` that collapse toward zero, so the
    // constrained active-set Newton inner solve can be too ill-conditioned to
    // reach a feasible constrained stationary point. When that happens the inner
    // solve either failed (bubbling up as `ParameterConstraintViolation`) or the
    // outer keep-best/best-iterate substitution shipped the LAST iterate — whose
    // β may VIOLATE the constraints `A·β ≥ 0`, producing point predictions that
    // are NOT monotone. The contract is family/link independent: a returned
    // `monotone_increasing` model MUST have non-decreasing point predictions.
    //
    // So before finalizing, if this fit carries explicit linear-inequality
    // constraints and the converged β is primal-INFEASIBLE beyond the published
    // active-set tolerance, PROJECT β onto the feasible cone (the minimal-L2 move
    // into `A·β ≥ 0`). The projection is exact and always monotone-feasible when
    // it succeeds, so the shipped β honors the constraint. If no feasible repair
    // can be reached we surface the violation to the caller as an error rather
    // than silently shipping an infeasible model. Because the projected β differs
    // from the pre-projection β, the working `state` (gradient / Hessian) is
    // re-evaluated at the projected β below so the exported curvature and the
    // shipped coefficients stay consistent; feasible fits (the common case) never
    // enter this branch and pay only one cheap feasibility check.
    if let Some(lin) = options.linear_constraints.as_ref() {
        let primal_feasibility =
            compute_constraint_kkt_diagnostics(beta.as_ref(), &state.gradient, lin)
                .primal_feasibility;
        if primal_feasibility > crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL {
            let projected =
                crate::active_set::project_point_strictly_into_feasible_cone(beta.as_ref(), lin)
                    .filter(|candidate| {
                        compute_constraint_kkt_diagnostics(candidate, &state.gradient, lin)
                            .primal_feasibility
                            <= crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
                    });
            match projected {
                Some(feasible_beta) => {
                    log::warn!(
                        "[PIRLS] constrained fit converged to an INFEASIBLE β \
                         (primal={primal_feasibility:.3e} > \
                         {tol:.3e}); projecting onto the feasible cone so the \
                         returned model honors A·β ≥ 0 (#1786)",
                        tol = crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
                    );
                    beta = Coefficients::new(feasible_beta);
                    // Re-evaluate the working state at the projected β so the
                    // exported gradient / Hessian match the shipped coefficients.
                    state = model.update_with_curvature(&beta, state.hessian_curvature)?;
                }
                None => {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "constrained PIRLS converged to an infeasible coefficient vector \
                         (primal feasibility {primal_feasibility:.3e} exceeds tolerance \
                         {tol:.3e}) and no feasible projection onto the constraint cone \
                         could be found; refusing to return a model whose point predictions \
                         violate the requested shape constraint (#1786)",
                        tol = crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
                    )));
                }
            }
        }
    }

    // Post-convergence Laplace curvature finalization (Issue 4).
    //
    // The Laplace approximation ∫ exp(-F(β)) dβ requires the *actual*
    // Hessian H_F = ∇²F at the mode. The inner LM step search may have
    // accepted steps under Fisher curvature (when observed went non-SPD or
    // produced a bad gain ratio mid-iteration), but that decision must NOT
    // leak into the exported H — Fisher → Observed substitution turns the
    // exact Laplace criterion into a silent PQL surrogate.
    //
    // Always re-evaluate observed curvature at β̂. If the model supports it
    // and the resulting Hessian is SPD within tolerance, export
    // `ObservedExact`. If it's indefinite, surface the witness via
    // `InvalidObservedCurvature` with diagnostics so the outer caller can
    // decide loudly. If the model does not support observed curvature
    // (canonical-link case where Observed = Fisher, or by-design surrogate
    // family), export `ExpectedInformationSurrogate` — never silently
    // mislabel a Fisher fallback as exact.
    let exported_laplace_curvature: ExportedLaplaceCurvature =
        if model.supports_observed_information_curvature() {
            match model.update_with_curvature(&beta, HessianCurvatureKind::Observed) {
                Ok(observed_state) => {
                    // Inertia check via the dense Hessian. Use the symmetric
                    // eigensolver (matches the indefinite-safe stabilization
                    // path elsewhere in PIRLS). If the Hessian is sparse-only
                    // and we cannot densify, conservatively label SPD when
                    // assembly succeeded; the symbolic pattern of the sparse
                    // path enforces SPD assembly upstream.
                    let inertia = observed_state
                        .hessian
                        .as_dense()
                        .and_then(gam_linalg::utils::symmetric_extremes);
                    let (label, accept_observed) = match inertia {
                        Some((min_eig, max_eig)) => {
                            let pd_tolerance = max_eig.abs().max(1.0) * 1e-12;
                            if min_eig > -pd_tolerance {
                                (ExportedLaplaceCurvature::ObservedExact, true)
                            } else {
                                let g_norm = constrained_stationarity_norm(
                                    &observed_state.gradient,
                                    beta.as_ref(),
                                    options.coefficient_lower_bounds.as_ref(),
                                    options.linear_constraints.as_ref(),
                                );
                                log::warn!(
                                    "[PIRLS] post-convergence observed Hessian indefinite: \
                                 λ_min={min_eig:.3e}, pd_tol={pd_tolerance:.3e}, ‖g‖={g_norm:.3e}"
                                );
                                (
                                    ExportedLaplaceCurvature::InvalidObservedCurvature {
                                        min_eigenvalue: min_eig,
                                        pd_tolerance,
                                        gradient_norm: g_norm,
                                    },
                                    // Indefinite observed Hessian: we still
                                    // promote it into `state` so the exported
                                    // Hessian matches the diagnostic label —
                                    // the caller is told loudly via
                                    // InvalidObservedCurvature that downstream
                                    // log|H| is not trustworthy.
                                    true,
                                )
                            }
                        }
                        None => {
                            // Sparse-native path or eigensolver failure: rely on
                            // the upstream SPD-assembly invariant. Treat as
                            // observed-exact since the model accepted the assembly.
                            (ExportedLaplaceCurvature::ObservedExact, true)
                        }
                    };
                    // WHY: promote the observed_state into the exported `state`
                    // when the post-convergence Observed assembly succeeded.
                    // Without this swap, when the inner LM loop ended on
                    // Fisher (force_fisher_for_rest engaged or mid-iter
                    // gain-rejection fallback fired), the exported
                    // `penalized_hessian_transformed` would still carry Fisher
                    // weights even though `exported_laplace_curvature` claims
                    // ObservedExact. The label and the matrix must agree.
                    if accept_observed {
                        state = observed_state;
                    }
                    label
                }
                Err(err) => {
                    let g_norm = constrained_stationarity_norm(
                        &state.gradient,
                        beta.as_ref(),
                        options.coefficient_lower_bounds.as_ref(),
                        options.linear_constraints.as_ref(),
                    );
                    log::warn!(
                        "[PIRLS] post-convergence observed Hessian assembly failed: {err}; \
                     exporting InvalidObservedCurvature with ‖g‖={g_norm:.3e}"
                    );
                    ExportedLaplaceCurvature::InvalidObservedCurvature {
                        min_eigenvalue: f64::NAN,
                        pd_tolerance: f64::NAN,
                        gradient_norm: g_norm,
                    }
                }
            }
        } else {
            // Canonical link or surrogate-by-design: Observed = Fisher (canonical)
            // or Fisher used by explicit family choice. Either way, the exported
            // curvature is the Fisher information, labeled honestly.
            ExportedLaplaceCurvature::ExpectedInformationSurrogate
        };

    Ok(WorkingModelPirlsResult {
        constraint_kkt: options
            .linear_constraints
            .as_ref()
            .map(|lin| compute_constraint_kkt_diagnostics(beta.as_ref(), &state.gradient, lin))
            .or_else(|| {
                options.coefficient_lower_bounds.as_ref().and_then(|lb| {
                    linear_constraints_from_lower_bounds(lb).map(|lin| {
                        compute_constraint_kkt_diagnostics(beta.as_ref(), &state.gradient, &lin)
                    })
                })
            }),
        beta,
        state,
        status,
        iterations,
        lastgradient_norm,
        last_deviance_change,
        last_step_size,
        last_step_halving,
        max_abs_eta,
        min_penalized_deviance,
        final_lm_lambda: lambda,
        final_accept_rho: last_iter_accept_rho,
        exported_laplace_curvature,
    })
}

pub(super) mod test_support {
    //! Thread-local diagnostic trace for PIRLS penalized-deviance sequences.
    //! `record_penalized_deviance` is always on so the per-iteration call site
    //! in the PIRLS hot loop does not need a `#[cfg(test)]` gate; the trace
    //! defaults to `None`, so the call is a near-free thread-local read in
    //! production. The capture entry point lives next to the test that uses
    //! it.
    thread_local! {
        pub(crate) static PIRLS_PENALIZED_DEVIANCE_TRACE: std::cell::RefCell<Option<Vec<f64>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub fn record_penalized_deviance(value: f64) {
        PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
            if let Some(ref mut buf) = *trace.borrow_mut() {
                buf.push(value);
            }
        });
    }
}
