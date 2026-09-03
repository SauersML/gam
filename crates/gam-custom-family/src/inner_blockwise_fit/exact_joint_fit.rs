//! Authoritative coupled exact-joint inner engine.
//!
//! Route selection and common setup remain in the parent blockwise driver.
//! Once selected, this engine owns the full trust-region/KKT lifecycle and
//! returns a complete result; it cannot silently fall through to coordinate
//! iteration after a failed exact-joint certificate.

use super::*;

/// Clear every cross-cycle statistic whose inference assumes a FIXED step model
/// (gam#2612).
///
/// The joint-Newton loop's stall verdicts — "the residual has not improved for
/// `N` cycles", "the trailing window projects more cycles than the budget has"
/// — are inferences about a sequence of steps taken from ONE model. When the
/// model changes underneath them that evidence describes a solver that no
/// longer exists, exactly as it does when an accepted step changes the critical
/// cone. Nothing here accepts an iterate or loosens a tolerance: the next cycle
/// recomputes the residual and must build fresh evidence against the unchanged
/// KKT certificate.
fn clear_stall_evidence_collected_under_the_previous_model(
    best_residual_seen: &mut f64,
    cycles_since_residual_improved: &mut usize,
    tr_clamped_during_stall: &mut bool,
    residual_descent_history: &mut std::collections::VecDeque<f64>,
    residual_rate_history: &mut std::collections::VecDeque<f64>,
    merit_window: &mut std::collections::VecDeque<f64>,
    geometric_tail_history: &mut std::collections::VecDeque<f64>,
) {
    *best_residual_seen = f64::INFINITY;
    *cycles_since_residual_improved = 0;
    *tr_clamped_during_stall = false;
    residual_descent_history.clear();
    residual_rate_history.clear();
    merit_window.clear();
    geometric_tail_history.clear();
}

pub(super) fn fit_exact_joint<F: CustomFamily + Clone + Send + Sync + 'static>(
    context: ExactJointFitContext<'_, F>,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    let ExactJointFitContext {
        family,
        specs,
        block_log_lambdas,
        options,
        mut states,
        s_lambdas,
        ridge,
        joint_bundle,
        mut lastobjective,
        mut converged,
        mut cycles_done,
        mut terminal_convergence_state,
        inner_tol,
        inner_max_cycles,
        mut cached_active_sets,
        mut current_log_likelihood,
        mut cached_eval,
        mut cached_joint_gradient,
        mut cached_joint_workspace,
        mut cached_joint_hessian_source,
        objective_state,
        joint_workspace_requested,
        matrix_free_joint_requested,
        total_joint_n,
        prelude_log,
        inner_started,
        mut last_residual_tol,
        product,
    } = context;
    let mut current_penalty: f64;

    // Build block ranges for the joint system.
    let ranges: Vec<(usize, usize)> = {
        let mut offset = 0;
        specs
            .iter()
            .map(|s| {
                let start = offset;
                offset += s.design.ncols();
                (start, offset)
            })
            .collect()
    };
    let total_p: usize = ranges.last().map_or(0, |r| r.1);

    // Universal full-span Jeffreys/Firth robustness. Build `Z_J` once and
    // use the same term in the coupled Newton step, objective value, and
    // stationarity checks so a near-separating coefficient is bounded by
    // the likelihood's own Fisher geometry instead of an ad-hoc ridge.
    // `None` (empty coefficient system) leaves every step and objective at
    // the un-augmented inner Newton.
    //
    // Continuous-response families (the canonical example: transformation-
    // normal h(Y|x) ~ N(0,1)) opt out via
    // `joint_jeffreys_term_required() = false`. They have no separation
    // regime, the Fisher information is `O(n)` on every identified
    // direction by construction, and each Jeffreys evaluation costs
    // `p` directional-derivative calls into the family's exact joint
    // Hessian — at large scale (CTN duchon16d, p=144, n=20000) that
    // is the dominant per-cycle cost (~200 s/cycle on three calls per
    // cycle), exhausting the inner budget before the algorithm converges
    // while contributing essentially zero to the gradient/curvature.
    let joint_jeffreys_subspace = if family.joint_jeffreys_term_required() {
        build_joint_jeffreys_subspace(family, specs, &ranges)?
    } else {
        None
    };
    // FIRTH MERIT BOOKKEEPING (gam#826/#872 — per-cycle Φ fold, not a carried
    // value). `current_penalty` / `lastobjective` hold ONLY the quadratic
    // penalty `½βᵀSβ` (NO Φ). The Firth value `−Φ` is folded into the
    // accept/reject comparison FRESH at each β under the same
    // `jeffreys_skippable_this_cycle` gate the step and KKT residual use, so
    // `old_objective` (old β) and `trialobjective` (trial β) are always on the
    // same objective `−ℓ + ½βᵀSβ − Φ` regardless of whether a cycle skips the
    // term. Carrying Φ in `current_penalty` (the previous design) desynced
    // old-vs-trial by ±Φ whenever the per-cycle skippable decision flipped —
    // and the cycle-0 baseline folded Φ UNCONDITIONALLY while the trial folded
    // it gated, so a skippable cycle 0 saw a spurious `Δobj = ±Φ`, rejected
    // every backtrack, and refused as a `phantom_multiplier` at a zero step
    // (the binomial location-scale coupled non-convergence). SIGN: Firth ADDS
    // ½log|I| to the log-likelihood ⇒ the NLL objective SUBTRACTS Φ, matching
    // the Newton step rhs / KKT residual which ADD `∇Φ` to `∇L − Sβ`.

    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.accounts_for_objective()
    {
        ridge
    } else {
        0.0
    };

    // Exact joint Newton steps are guarded by two independent mechanisms:
    // family-owned feasibility (`max_feasible_step_size`) and the adaptive
    // trust region below. There is intentionally no family hook for a
    // hard per-attempt coefficient-space clamp; keeping the policy local
    // avoids stale no-op configuration and makes the trust-region behavior
    // explicit at the only place it is used.

    // Cross-cycle convergence carry-over: set at the end of every
    // accepted cycle so the next cycle can distinguish a true KKT
    // optimum on a rank-deficient null mode (objective stuck
    // because every direction is along the null space) from
    // genuine non-convergence. The residual signal does not need
    // a carry-over — `residual <= residual_tol` is the canonical
    // KKT certificate and the end-of-cycle test consumes it
    // directly when it fires.

    // Predicted-reduction tracker for the principled trust-region
    // stopping criterion (Conn-Gould-Toint, *Trust-Region Methods*,
    // Theorem 6.4.6). The Newton model at the accepted step has a
    // predicted decrease `m(0) − m(δ) = −g·δ − 0.5·δ·H·δ`. For an
    // unclipped Newton step (H·δ = −g) this is `0.5·g·H⁻¹·g`, the
    // Newton decrement squared / 2. When the model itself predicts
    // a decrease smaller than the objective tolerance, no descent
    // direction the Hessian can resolve will lower the objective
    // by more than `objective_tol`, and continuing is wall-clock
    // waste regardless of whether the raw gradient residual or
    // step-norm gates have closed.
    //
    // Cross-cycle convergence carry-over: set at the end of every
    // accepted cycle so the next cycle's line-search-failure path
    // can distinguish a true KKT optimum on a rank-deficient
    // Hessian (no meaningful trial step, even though step_inf is
    // O(1) along the null mode) from genuine non-convergence.
    let mut last_cycle_residual_below_tol = false;
    let mut last_cycle_obj_change_below_tol = false;

    let mut joint_trust_radius = 1.0_f64;
    let mut joint_block_trust_radii = vec![1.0_f64; ranges.len()];
    let mut last_accepted_hit_joint_trust_boundary = false;
    // Hard upper bound for the for-loop's range. The cap is fixed at
    // `inner_max_cycles` for the lifetime of this outer call (the
    // earlier mid-loop cap extension was removed in favor of the
    // plateau-flat-objective convergence certificate), but the
    // sentinel pattern is retained — the `.max(200)` floor is a
    // harmless safety pad and the explicit `cycle >= inner_max_cycles`
    // break keeps the existing `continue` statements in the body
    // working
    // (they advance `cycle` via the iterator), unlike a `while` +
    // manual-counter rewrite.
    let inner_loop_hard_ceiling = inner_max_cycles.max(200);
    // Verbose cadence for the inner joint-Newton log block. Boring cycles
    // (first-attempt accepts with no convergence event) emit ONE compact
    // one-liner instead of the 4-line pre-cycle/TR/cycle-summary/convergence
    // block. Verbose cycles (first, last, every 20th, all rejections,
    // convergence events) keep the full detail. JOINT_LOG_VERBOSE_PERIOD is
    // tuned so a 200-cycle inner solve emits ~10 detailed waypoints plus
    // 1 compact line per remaining cycle (~210 lines), down from ~800.
    const JOINT_LOG_VERBOSE_PERIOD: usize = 50;
    // Residual-stall detector for joint Newton. Distinct from the
    // blockwise loglik-frozen divergence detector lower in the file:
    // that one requires the log-likelihood to be unchanged for K
    // cycles AND the per-block Newton step pinned at the cap.
    //
    // Large-scale survival marginal-slope hits a different pattern —
    // the joint objective decreases monotonically by O(1) per cycle
    // (so loglik is NOT frozen), the TR repeatedly clamps proposals
    // with |prop|∞ >> trust_radius, and the post-step KKT residual
    // oscillates in a band orders of magnitude above residual_tol
    // without trending down. Burning the rest of the cycle budget on
    // this pattern reaches inner_max_cycles "non-converged", which
    // then drops the outer optimizer into the first-order bridge
    // fallback with a stale-mode gradient that ‖g‖ ≈ 10⁷ kills BFGS
    // line search at iter 0.
    //
    // Track the best residual seen and the number of cycles since
    // any meaningful improvement (≥10% drop). Once we've burned at
    // least RESIDUAL_STALL_MIN_CYCLES with no improvement AND the
    // TR has been clamping aggressively, exit `converged=false` so
    // the outer optimizer sees a non-converged signal while we still
    // have a finite, in-range β to return (instead of running to the
    // hard ceiling and then handing BFGS a junk gradient).
    const RESIDUAL_STALL_NO_IMPROVE_CYCLES: usize = 30;
    const RESIDUAL_STALL_MIN_CYCLES: usize = 40;
    const RESIDUAL_STALL_IMPROVEMENT_FACTOR: f64 = 0.9;
    // Upper bound on how long a still-descending Φ-merit may VETO the
    // flat-residual stall exit (gam#979 survival marginal-slope hang). The
    // merit-descent veto below (`merit_still_descending_over_window`) was
    // added to protect the gam#1607 transient wiggle — a few cycles where
    // the line search drives the objective down while the KKT residual
    // re-anchors through a gauge null before catching up. That is a
    // TRANSIENT: a healthy or wiggling solve reaches KKT tolerance in a
    // handful of cycles. On the survival marginal-slope monotone-cone DGP,
    // by contrast, the joint block carries a free warp/gauge direction
    // (the #892 flexible-regime family) along which the penalized objective
    // drifts DOWN by O(1) every cycle indefinitely while the KKT residual
    // sits orders of magnitude above tol and never trends toward it. The
    // veto then reads that unbounded drift as "still making progress" and
    // suppresses the flat-residual exit for the ENTIRE cycle budget — the
    // loop grinds to `inner_loop_hard_ceiling` on every one of the ~60
    // outer ρ-evaluations (the ~900s #979 hang), then hands the outer
    // optimizer a non-converged result anyway. Once the residual has been
    // flat (no ≥10% drop) for this many cycles — a large multiple of the
    // stall window, far beyond any legitimate wiggle transient or healthy
    // convergence — the drifting merit is no longer credible evidence of
    // reachable convergence and the veto yields to the honest non-converged
    // exit. This changes nothing for solves that actually converge or
    // wiggle briefly (they exit long before the counter climbs this high);
    // it only rejects a provably-non-stationary ρ sooner.
    const RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES: usize = 4 * RESIDUAL_STALL_NO_IMPROVE_CYCLES;
    let mut best_residual_seen: f64 = f64::INFINITY;
    // Smallest *certified* stationarity residual the solve actually computed,
    // tracked independently of `best_residual_seen` (whose updates are bound
    // to the residual-stall counters at the post-step site below and so are
    // skipped by every head-of-cycle / pre-line-search certificate exit). The
    // terminal verdict reports THIS so a legitimate early-certificate exit
    // (e.g. the cycle-0 pre-line-search KKT exit on intercept-only / already-
    // stationary data) reports the finite residual it certified on instead of
    // the sentinel `inf` — converged=true must never be paired with a non-
    // finite residual in the log (#1040 inner-report truthfulness).
    let mut min_certified_residual: f64 = f64::INFINITY;
    // What ONE evaluation of the inner objective carries in rounding, measured
    // from the trust region's own backtracking ladders (gam#2612). Lives at
    // SOLVE scope, not cycle scope: rounding does not get smaller because the
    // iterate moved, the ladder long enough to measure it appears once, and
    // every cycle after it is two attempts at the radius floor. See
    // [`ObjectiveResolutionWitness`].
    let mut objective_resolution_witness = ObjectiveResolutionWitness::default();
    // The penalty's own ACCUMULATION scale, which is not its value (gam#2612's
    // central observation, gam#2748's ceiling). `block_quadratic_penalty`
    // evaluates `½β·(S_λβ)` with signed `S_ij`, so one evaluation accumulates at
    // scale `max|S_λ|·‖β‖₁²` while returning something that can be many orders
    // smaller. `S_λ` is a function of ρ alone and ρ is fixed for this solve, so
    // the matrix factor is read once here; the `‖β‖₁²` factor is per trial point.
    let penalty_entry_magnitude: f64 = s_lambdas
        .iter()
        .flat_map(|s_lambda| s_lambda.iter())
        .filter(|value| value.is_finite())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let mut cycles_since_residual_improved: usize = 0;
    // Number of consecutive non-improving cycles after which the
    // conditioning-based self-vanishing Levenberg–Marquardt damping is
    // ARMED inside the spectral-range Newton solve, for EVERY family
    // (#826/#808). The undamped range-restricted Newton step oscillates on a
    // full-rank-but-ill-conditioned penalized Hessian at the oversmoothed-ρ
    // operating point: the tiny-but-above-cutoff curvature of the lightly
    // identified mean/threshold/wiggle block takes an enormous `component/λ`
    // proposal that the trust region clips every cycle, so the residual on
    // that block freezes while its β stays ≈0 (the exact #826 signature).
    // The conditioning-gated `μ = c·‖∇L − Sβ‖∞` caps that component into a
    // bounded descent step. It is SELF-VANISHING (μ → 0 as the residual → 0)
    // so the converged β and the KKT certificate are byte-identical to the
    // undamped solve — zero REML/LAML bias. Arming it on OBSERVED non-
    // progress rather than a static per-family flag keeps the AFT /
    // constant-scale endgame (which converges quadratically and never
    // stalls) byte-identical: a quadratically-converging solve reaches
    // tolerance in a handful of cycles and never trips this threshold, so μ
    // is never engaged there. Only a genuinely oscillating ill-conditioned
    // solve crosses it, which is exactly when the damping is sound. Set a
    // few cycles below the stall-exit window so the damping gets a chance to
    // rescue the solve well before the early-exit / budget tripwire fires.
    // (The conditioning-gated self-vanishing μ this armed now lives ONLY in the
    // test-retained `solve_joint_newton_step_on_spectral_range`; the production
    // joint step takes the exact trust-region multiplier λ instead — gam#979.)
    // Recent KKT-residual values (oldest→newest) used to detect STEADY
    // geometric descent at the certificate-refusal gate. A still-converging
    // Newton direction (residual dropping by a steady factor < 1 each cycle)
    // must not be misclassified as a multiplier/null plateau and exited
    // early (gam#787 duchon centers≥20: the slope block converges
    // geometrically — residual ~0.33×/cycle — but `linearized_rel ≥ 0.5`
    // routed it into the plateau-refusal break a few cycles short of tol).
    const RESIDUAL_DESCENT_WINDOW: usize = 3;
    let mut residual_descent_history: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(RESIDUAL_DESCENT_WINDOW);
    let mut tr_clamped_during_stall: bool = false;
    // Deterministic slow-geometric-rate stall guard (gam#979 survival
    // marginal-slope). The flat-residual guard below resets its no-improve
    // counter whenever the residual drops ≥10% versus the running best, and
    // the Newton-decrement certificate refuses while the decrement sits a
    // hair above `objective_tol`. A residual crawling down by a small fixed
    // fraction each cycle — the survival marginal-slope oversmoothed-ρ
    // endgame: a stiff penalized Hessian (penalty dominates, eigenvalues
    // ~1e6) yields Newton steps ~1e-5 far INSIDE a large trust radius, so
    // the KKT residual descends geometrically but very slowly (~0.99×/cycle,
    // halving only every ~80 cycles) — clears that 10% bar every ~12 cycles,
    // so NEITHER guard ever fires and the solve grinds ~10³ cycles at ~p³
    // each: minutes-to-hours per outer ρ-evaluation, the measured #979
    // survival "hang" (n≈2500, centers=12 runs past a 900 s wall with no
    // result). This is NOT divergence and NOT a flat stall — the residual is
    // genuinely (geometrically) descending, just far too slowly to reach tol
    // in a practical cycle count. Track a trailing window of residuals so
    // the post-step site can PROJECT, from the window's geometric rate
    // (cycle indices and residual ratios only — fully deterministic, NO
    // wall-clock; cf. the explicit no-wall-clock note at the bottom of the
    // cycle loop), how many more cycles reaching `residual_tol` would take.
    const LINEAR_RATE_WINDOW: usize = 16;
    // Floor on the slow-geometric-rate projection cap. The EFFECTIVE cap is
    // `max(this floor, remaining budget = inner_max_cycles − cycle)`: if the
    // geometric projection says tol is reachable within the caller's own
    // remaining `inner_max_cycles` budget, the solve is allowed to run to it
    // rather than being killed at a fixed 100 (gam#979 survival marginal-
    // slope). The derivative-quality outer eval deliberately sets a large
    // budget (`inner_max_cycles=1200`, psi_hyper.rs) BECAUSE the analytic
    // LAML trace-gradient is only consistent at a joint-stationary β̂ with a
    // very tight `residual_tol` (~1e-9): the 3-block survival-MS constrained
    // solve descends geometrically (~0.94×/cycle) and reaches that tol in
    // ~200 cycles — well inside 1200 — but a fixed cap of 100 cut it off at
    // ~cycle 107, handed the outer a non-stationary mode, and the resulting
    // IFT-inconsistent outer gradient stranded ARC at a spurious strict
    // saddle it could not escape (the measured n=400/2500 centers=12 hard
    // failure). Deferring the cap to the caller's budget lets the solve
    // finish; a genuinely non-converging solve (rate ≥ 1, or projected past
    // the remaining budget) still exits, and `inner_max_cycles` remains the
    // hard backstop. Never SMALLER than the historic 100 (the floor), so no
    // previously-surviving solve is cut off earlier.
    const LINEAR_RATE_PROJECTION_CAP: usize = 100;
    let mut residual_rate_history: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(LINEAR_RATE_WINDOW + 1);
    // Trailing window of the Φ-augmented merit objective, parallel to
    // `residual_rate_history`, for the merit-descent veto on the two
    // residual-trend stall guards (gam#1607 binomial location-scale-WIGGLE).
    //
    // Both residual-trend guards below (flat-residual no-improve, and the
    // slow-geometric-rate projection) use the trend of the KKT *residual*
    // as a proxy for "can this solve still reach tol in a practical
    // budget". On the wiggle family that proxy is unsound: the model
    // carries an exact additive gauge null (the threshold βₜ and the
    // wiggle-intercept `βwᵀB(q₀)` both shift q = q₀ + Bᵀβw), and as the
    // dynamic basis `B(q₀)` re-anchors during PIRLS the KKT residual is
    // genuinely NON-monotone — it humps up (0.2→0.4) for ~150 cycles
    // before descending to tol — even though the merit objective the line
    // search actually minimizes descends monotonically the whole way and
    // the solve DOES converge (measured: cycle 638, β bounded ≈1.9). A
    // residual-trend guard then reads the transient rise as "diverging /
    // can't reach tol" and bails ~cycle 40, handing the outer optimizer a
    // false non-convergence (the #1607 wiggle fullhessian failure).
    //
    // The merit is the real Lyapunov function: a descending merit IS
    // progress, regardless of the residual's transient shape. So veto a
    // residual-trend stall exit while the merit is still descending
    // robustly over the SAME trailing window. This preserves termination —
    // the merit is bounded below and monotone-nonincreasing under the
    // line search, so it cannot keep clearing a fixed relative-descent bar
    // forever; once it genuinely flattens (the true #979 survival stall:
    // ~1e-5 steps ⇒ merit flat to f64) the veto lifts and the guard fires
    // exactly as before.
    let mut merit_window: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(LINEAR_RATE_WINDOW + 1);
    // Fully-rejected stall guard. The residual-stall guard below
    // (post-grad-reload) only fires on cycles that produced an accepted
    // step, because every termination check it gates lives after the
    // `if !accepted { continue; }` exit at the bottom of the trust-region
    // attempt loop. When every cycle in a row is fully rejected — all
    // JOINT_TRUST_MAX_ATTEMPTS trial steps fail the line-search check —
    // none of those guards ever see the iterate, the cycle loop spins
    // up to `inner_loop_hard_ceiling` cycles, and the inner solver burns
    // ~120 s of wall-clock per outer ρ-evaluation that the outer
    // optimizer will reject anyway. The signature is exact and local:
    // (i) every trust attempt this cycle was rejected by SOME path —
    // model, likelihood, objective, OR feasibility (the four counters
    // partition the JOINT_TRUST_MAX_ATTEMPTS attempts), so `model_rejects +
    // likelihood_rejects + objective_rejects + feasibility_rejects ==
    // JOINT_TRUST_MAX_ATTEMPTS`,
    // AND (ii) the joint trust radius has NOT shrunk relative to the
    // previous fully-rejected cycle. Condition (i) was originally
    // objective-only (`objective_rejects == MAX`, others 0), which never
    // fired on the biobank gauge-flat marginal/slope fit: there the
    // objective is flat to f64 precision along the residual direction and
    // the BMS line search rejects every trial on the LIKELIHOOD early-exit
    // path, so the guard's increment was unreachable and the loop spun to
    // the cap. A full likelihood-path rejection at a collapsed radius is
    // the same no-descent stall, so any-path full rejection counts.
    // Condition (ii) is what proves no progress is possible: β is
    // reverted to its pre-cycle value on every fully-rejected cycle, so
    // with an identical Newton system AND an identical trust radius the
    // next cycle's trust-region search is byte-deterministically the
    // same as this one's. The radius can stall above the 1e-12 floor
    // when `shrink_active_joint_block_trust_radii` only shrinks blocks
    // that hit their per-block boundary — an interior block keeps its
    // radius forever, so `max(block_radii)` is held by that block while
    // the boundary block's radius collapses to 1e-12 without changing
    // the max. After `FULLY_REJECTED_STALL_MAX_CYCLES` consecutive cycles
    // with both conditions, judge convergence on the identified (range)
    // subspace: a stall at a collapsed radius proves the descent direction
    // is gauge-flat, so if the range-projected KKT residual is at tolerance
    // the fit is at a numerically-stationary penalized optimum and is
    // returned converged; only when the identified-subspace residual is
    // ALSO above tol is this a genuine non-convergence the outer optimizer
    // should reject — exit non-converged so it rejects this ρ cleanly
    // instead of waiting for the cycle cap.
    // A rejected-cycle fixed point must describe the state that actually
    // determines the next trust-region search. The former detector compared
    // only the first trial objective; that scalar omits the joint and
    // per-block trust radii, so shrinking-radius globalization could produce
    // the same rounded objective while still making essential progress.
    // Keep exact f64 bit patterns for the pre-cycle iterate and every radius,
    // plus the realized first proposal and rejection partition. Equality is
    // therefore an equality of solver state and observed transition, not a
    // coincidental equality of one rounded output.
    #[derive(PartialEq, Eq)]
    struct FullyRejectedCycleSignature {
        beta_bits: Vec<u64>,
        joint_trust_radius_bits: u64,
        block_trust_radius_bits: Vec<u64>,
        first_trial_delta_bits: Option<Vec<u64>>,
        first_trial_objective_bits: Option<u64>,
        rejection_counts: [usize; 4],
    }
    let mut prev_fully_rejected_cycle_signature: Option<FullyRejectedCycleSignature> = None;
    let mut consecutive_identical_rejected_cycles: usize = 0;
    const IDENTICAL_REJECTED_STALL_MAX_CYCLES: usize = 2;
    // Collapsed-trust-region all-reject-at-floor guard (gam#979 survival
    // hang / binary high-`centers` `IntegrationError`). DISTINCT from the
    // two detectors above:
    //   * `consecutive_held_rejected_cycles` requires the radius to be HELD
    //     relative to the *previous reject* — which it is at any pinned
    //     value, floor or not — and only fires after 8 cycles.
    //   * `consecutive_identical_rejected_cycles` requires the trial
    //     objective to repeat BIT-FOR-BIT, which a near-singular coupled
    //     marginal↔slope system need not do: tiny non-deterministic
    //     round-off in the per-row tower contraction perturbs the trial
    //     objective in its last ULPs even while the step is otherwise
    //     stuck, so the byte-identical detector never latches.
    // The unambiguous deterministic signal of "stuck and cannot recover" is
    // the trust radius sitting at its absolute `1e-12` floor WHILE every
    // line-search attempt is rejected: no smaller step is representable, so
    // the radius cannot shrink further, and the all-reject means the step
    // makes no progress. After `JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES`
    // consecutive such cycles the loop is provably grinding to its budget on
    // a near-singular system (`phantom_multiplier_with_well_conditioned_H`),
    // so exit cleanly through the SAME identified-subspace / fixed-point
    // certificate path the other two detectors use — converged if the
    // range-space residual is stationary, give-best non-converged otherwise
    // — instead of spinning out the full `inner_max_cycles`. The absolute-
    // floor requirement is why this CANNOT fire on a genuinely progressing
    // fit: a fit that is descending keeps the radius well above `1e-12`
    // (it grows on `rho>0.75`/boundary and only collapses to the floor after
    // a sustained reject streak), so the counter resets on every accepted
    // cycle and never reaches the threshold.
    // Threshold + floor ceiling live in `joint_newton.rs` so the loop and
    // the `joint_newton_collapsed_trust_region_all_reject_exits_before_grinding_budget`
    // unit test assert against one source of truth.
    let mut consecutive_all_reject_at_floor_cycles: usize = 0;
    // Set by the most recent fully-rejected cycle whose trust ratio failed to
    // approach 1 under refinement. Carried out of the cycle loop so the inner
    // solve's refusal can name the fault as a model/objective disagreement
    // instead of leaving the four reject counters to imply a step-length
    // problem that shrinking would have fixed (gam#2695).
    let mut trust_ratio_model_inconsistency: Option<String> = None;
    let mut last_joint_math: Option<JointNewtonMathDiagnostic> = None;
    // Cross-cycle cache of the joint Jeffreys/Firth triple `(β_key, ∇Φ, H_Φ)`
    // (gam#729/#826/#808). Computing `(∇Φ, H_Φ)` costs `p` family
    // directional-derivative calls plus the `½ S Sᵀ` GEMM; for a K-block
    // coupled family that is the dominant per-inner-cycle cost. The post-step
    // KKT residual recomputes the triple at the just-accepted β; the NEXT
    // cycle's head needs the SAME triple at that SAME β. Carry it forward
    // keyed on the flattened β so the head reuses the post-step result instead
    // of recomputing — collapsing two O(p)-directional-derivative evaluations
    // per accepted cycle to one. The key is an exact-equality check on the
    // flattened β (β is byte-identical between an accepted post-step residual
    // and the next head), so the reused term is the exact term at the current
    // iterate — no staleness, no tolerance fudge.
    let mut jeffreys_triple_cache: Option<(Array1<f64>, Array1<f64>, Array2<f64>)> = None;
    // Stash for the structured cert-REFUSED report computed inside the
    // cycle loop, so the post-loop bubbled error (`coupled exact-joint
    // inner solve exited the joint Newton path …`) can emit the same
    // per-block + spectrum breakdown without re-materializing H_pen.
    let mut last_kkt_refusal_report: Option<KktRefusalReport> = None;
    let mut prev_kkt_norm: Option<f64> = None;
    // Convergence-endgame flag for the Jeffreys second-order completion
    // (gam#979): set once the post-step KKT residual enters
    // `JEFFREYS_COMPLETION_RESIDUAL_BAND × residual_tol`, consumed by the
    // next cycle's dense-spectral step assembly.
    let mut jeffreys_completion_endgame = false;
    // Exact completion accounting is part of the causal #979 evidence: a
    // correct quadratic endgame should form the completion only after the
    // residual-band latch (or for a returned-mode certificate), then need
    // only a handful of true-Hessian cycles.
    let mut jeffreys_completion_calls = 0usize;
    let mut jeffreys_true_hessian_probe_logged = false;
    // ONE INVARIANT FOR EVERY CONCEDING EXIT (gam#2612): this solve may not
    // give up while its step model is still the DIVIDED-DIFFERENCE Jeffreys
    // surrogate.
    //
    // `H_Φ` is the Daleckii–Krein divided-difference part of `−∇²Φ`; the exact
    // second-order completion `−½ tr(K D_ab)` is the rest of it, and until it
    // is formed the Newton step is built on a matrix that is NOT the Hessian
    // of the objective the certificate is taken against.
    // [`JEFFREYS_COMPLETION_RESIDUAL_BAND`] arms it on a PROXIMITY proxy — the
    // residual reaching `300 × residual_tol` — and that proxy is circular
    // wherever the distance from tolerance is CAUSED by the inexact model.
    // Measured on the #2612 penguins armed refit, once the trust region was
    // repaired enough to stop being the binding constraint: cycle 155 takes
    // the FULL Newton step (`|δ|∞ = |prop|∞ = 1.069e-4`, interior at
    // `r = 9.290e-3`) and the residual still does not contract — it drifts at
    // `1.0031×/cycle` at `2.398e-6`, five hundred times outside a band of
    // `4.3e-9`, with `jeffreys_completion_calls = 0`. The step is exactly as
    // long as the model wants; the model is the wrong matrix. And the in-tree
    // `[979-TRUE-HESSIAN]` probe prices the difference on an armed iterate of
    // this same fit: the divided-difference step's linearized residual
    // contraction is `5.791e-1` where the true Hessian's is `4.285e-11`.
    //
    // So a stall verdict taken on the surrogate is a statement about the
    // model, not about the problem. Every conceding path asks this question
    // first, and if the completion has never been formed it arms it and takes
    // another cycle. Latched, so it fires at most once per solve and every
    // later concession is honest.
    //
    // The cross-cycle stall evidence is cleared with it (
    // [`clear_stall_evidence_collected_under_the_previous_model`]) for the same
    // reason the accepted-active-face transition clears it: every statistic in
    // that set is an inference about a FIXED step model, and the model just
    // changed.
    // Total descent budget across the joint-Newton loop, used by
    // the end-of-loop summary to report `descent_total`.
    let initial_joint_objective: f64 = lastobjective;
    // Per-cycle |Δobjective| history for the geometric-tail trigger of
    // the constrained-stationary certificate below. When the cycles
    // settle into a linear-rate plateau (|Δobj_next| / |Δobj_prev|
    // approaching 1 monotonically over the window), the total
    // *remaining* objective descent is rigorously bounded above by the
    // geometric series sum |Δobj_now| / (1 − max_ratio). When that
    // bound is below `objective_tol` the cert can fire many cycles
    // earlier than waiting for any single |Δobj| to individually
    // cross obj_tol — the bound is mathematically the same precision
    // contract, applied to the asymptotic tail rather than one step.
    const GEOMETRIC_TAIL_WINDOW: usize = 5;
    let mut geometric_tail_history: std::collections::VecDeque<f64> =
        std::collections::VecDeque::with_capacity(GEOMETRIC_TAIL_WINDOW);
    // A first-order convergence event after an accepted step is tentative
    // until exact curvature at that returned beta proves second-order
    // stationarity. The next ordinary cycle owns that proof and, when it
    // exposes a strict saddle, immediately runs the existing finite-radius
    // More-Sorensen hard case from the same beta.
    let mut returned_mode_curvature_pending = false;
    let mut returned_mode_curvature_certified = false;
    // Constrained analogue of `returned_mode_curvature_pending`: a first-order
    // KKT point on an active face is tentative until the next cycle head
    // certifies its active-face-tangent curvature. On a strict face-tangent
    // saddle that head escapes along the negative-curvature direction and
    // resumes; `saddle_escapes_used` bounds those escapes at
    // `MAX_SADDLE_ESCAPES` before the honest typed refusal.
    let mut returned_constrained_mode_pending = false;
    let mut saddle_escapes_used = 0usize;
    // The curvature the last accepted escape left behind, so the next
    // certificate can tell whether that escape was worth anything (#2587).
    let mut previous_escape_lambda_min: Option<f64> = None;

    // #2627 — there is NO wall-clock deadline in this loop, at entry or per
    // cycle. gam#2055 removed both, and what survived was the orphaned first
    // half of the comment that described them: it announced a "fit-level
    // wall-clock budget guard at inner-solve ENTRY" and a "per-cycle guard
    // below", then ran off mid-sentence into the paragraph beneath. A reader
    // auditing why a coupled joint-Newton solve ran past a 300 s cap found a
    // comment promising the guard that would have stopped it. Deleted rather
    // than reimplemented: the loop's bounds are deterministic and iteration-
    // counted by design — `inner_loop_hard_ceiling` and the caller's
    // `inner_max_cycles` bound the cycle count, and the residual-trend,
    // merit-descent, and fully-rejected-stall guards below decide
    // termination from cycle indices and residual ratios ONLY, with no
    // wall-clock (see the no-wall-clock note at the bottom of the loop).
    // Reintroducing a deadline here would make termination host-dependent
    // and non-reproducible; the honest fix for a slow cycle is a smaller
    // per-cycle cost or a tighter deterministic guard, not a clock.
    // The exact joint-Hessian route solves the penalized Newton system
    // directly. Extra damping must be wired through an accepted/rejected
    // step policy before it belongs here; keep the matvec faithful to the
    // objective until then.
    'joint_newton_cycles: for cycle in 0..inner_loop_hard_ceiling {
        if cycle >= inner_max_cycles {
            break;
        }
        // Constrained returned-mode second-order certification (gam#979).
        //
        // A constrained first-order KKT point reached last cycle is tentative
        // until its active-face-tangent curvature is certified. Do that here,
        // at the cycle head, before rebuilding the step: a strict face-tangent
        // saddle (curvature the first-order certificate cannot see, the #979
        // CTN witness) is ESCAPED along its negative-curvature direction and
        // the solve resumes at the escaped, feasible point — the standard
        // second-order response to an indefinite stationary point, not a
        // refusal. Only when `MAX_SADDLE_ESCAPES` feasible escapes still land
        // on a saddle does the honest typed refusal fire (inside the helper).
        if returned_constrained_mode_pending {
            returned_constrained_mode_pending = false;
            let escape_block_constraints =
                collect_block_linear_constraints(family, &states, specs)?;
            let escape_objective_tol = inner_tol * (1.0 + lastobjective.abs());
            match resolve_constrained_converged_mode(
                family,
                &states,
                specs,
                options,
                &ranges,
                &s_lambdas,
                joint_mode_diagonal_ridge,
                joint_bundle,
                total_p,
                &escape_block_constraints,
                &cached_active_sets,
                saddle_escapes_used,
                previous_escape_lambda_min,
                escape_objective_tol,
                &mut jeffreys_completion_calls,
            )? {
                ConstrainedModeResolution::Certified { workspace } => {
                    cached_joint_workspace = workspace;
                    returned_mode_curvature_certified = true;
                    converged = true;
                    cycles_done = cycle;
                    break;
                }
                ConstrainedModeResolution::Escape {
                    direction,
                    alpha,
                    lambda_min,
                } => {
                    for (block_idx, (start, _)) in ranges.iter().copied().enumerate() {
                        for (coefficient_idx, coefficient) in
                            states[block_idx].beta.iter_mut().enumerate()
                        {
                            *coefficient += alpha * direction[start + coefficient_idx];
                        }
                    }
                    refresh_all_block_etas(family, specs, &mut states)?;
                    saddle_escapes_used += 1;
                    previous_escape_lambda_min = Some(lambda_min);
                    log::info!(
                        "[PIRLS/joint-Newton saddle-escape] attempt={} lambda_min={:.6e} alpha={:.6e}",
                        saddle_escapes_used,
                        lambda_min,
                        alpha,
                    );
                    // The escaped point is a fresh iterate; every cross-cycle
                    // progress statistic collected at the saddle is stale.
                    converged = false;
                    returned_mode_curvature_certified = false;
                    last_cycle_residual_below_tol = false;
                    last_cycle_obj_change_below_tol = false;
                    min_certified_residual = f64::INFINITY;
                    best_residual_seen = f64::INFINITY;
                    cycles_since_residual_improved = 0;
                    residual_descent_history.clear();
                    tr_clamped_during_stall = false;
                    residual_rate_history.clear();
                    merit_window.clear();
                    prev_fully_rejected_cycle_signature = None;
                    consecutive_identical_rejected_cycles = 0;
                    consecutive_all_reject_at_floor_cycles = 0;
                    last_joint_math = None;
                    last_kkt_refusal_report = None;
                    prev_kkt_norm = None;
                    geometric_tail_history.clear();
                }
            }
        }
        let verbose_cycle = cycle == 0
            || cycle + 1 == inner_max_cycles
            || (cycle + 1) % JOINT_LOG_VERBOSE_PERIOD == 0;
        // Pre-cycle header line removed: the post-cycle one-liner below
        // carries cycle/objective/Δobj/step/residual/time and on verbose
        // cadence the expanded convergence line additionally carries
        // -loglik and penalty. Suppressing this avoids emitting a second
        // info-level line per cycle just to repeat numbers we already
        // log at end of cycle.
        // Per-cycle phase-timing accumulators. Surface where the inner
        // joint-Newton spends time so a 18-min silent cycle 0 (the
        // bernoulli marginal-slope FLEX large-scale failure mode) becomes a
        // logged timeline at the end of the cycle. Phases:
        //   * hessian: joint Hessian source build (matrix-free workspace
        //     OR dense fallback assembly)
        //   * pcg:     matrix-free QP solve via solve_spd_pcg_with_info_into
        //              (already logs its own diagnostics; we accumulate
        //              here for the end-of-cycle summary)
        //   * line_search: backtracking step-size search (up to 8 attempts)
        //   * grad_reload: post-accept joint gradient + workspace refresh
        let cycle_started = std::time::Instant::now();
        // Top-of-cycle row-measure capture. The trust-region ratio
        // ρ = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ] is only meaningful when
        // every input (Hessian, gradient, objective at β, trial objective
        // at β + δ) is evaluated against the same row measure. We freeze
        // the measure here and re-read it at each of the four sites later
        // in the cycle, then hard-fail (Err) just before ρ if any of them
        // diverged. Cf. `src/solver/row_measure.rs`.
        let tr_row_measure_top =
            gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
        let hessian_started = std::time::Instant::now();
        let hessian_scope_guard = gam_runtime::process_monitor::track_scope(format!(
            "joint Newton hessian_qp cycle={cycle} n={total_joint_n} p={total_p}"
        ));
        log::info!(
            "[joint-newton-tr] phase=hessian_qp cycle={} r={:.3e}",
            cycle,
            joint_trust_radius,
        );
        let cycle_log = prelude_log;
        let constraints_started = std::time::Instant::now();
        let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
        let joint_constraints =
            assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
        // gam#979: joint simple lower bounds, when the joint constraints are
        // all axis-aligned lower bounds (the survival monotone-baseline-hazard
        // / monotone-smooth case). Threaded into the stationarity certificate
        // so ACTIVE simple-lower-bound multipliers are projected out (the
        // box-bound analog of the linear-constraint projection), instead of
        // their multiplier mass being mis-read as a stationarity defect and
        // mis-refusing a genuinely-optimal constrained iterate.
        let joint_lower_bounds: Option<Array1<f64>> = joint_constraints
            .as_ref()
            .and_then(|c| extract_simple_lower_bounds(c, total_p).ok().flatten())
            .map(|b| b.lower_bounds);

        // Every convergence exit from this loop routes through here. The
        // definition must sit AFTER `joint_constraints` is bound (macro_rules
        // hygiene resolves free locals in the DEFINITION scope, not at the call
        // site) and BEFORE the first call site — which since #2485 is the
        // stall-guard certificate below, earlier in the body than the post-step
        // sites this used to sit beside.
        //
        // Note it does not simply `break`: for an unconstrained, non-Jeffreys
        // fit, first-order convergence is TENTATIVE. The macro hands control
        // back to the next cycle head to certify the returned mode's curvature
        // and, on a strict saddle, escape along it (gam#979). Anything that
        // certifies convergence must inherit that — which is exactly why the
        // stall-guard site calls this instead of assigning `converged` itself.
        macro_rules! finish_post_step_convergence {
            () => {{
                converged = true;
                if joint_constraints.is_none() {
                    returned_mode_curvature_pending = true;
                    returned_mode_curvature_certified = false;
                    continue 'joint_newton_cycles;
                }
                // Every constrained first-order convergence event is
                // tentative until the next cycle head certifies M_true on
                // the numerically-tight active-face tangent. Jeffreys modes
                // are not a separate objective or a certificate exemption.
                returned_constrained_mode_pending = true;
                returned_mode_curvature_certified = false;
                continue 'joint_newton_cycles;
            }};
        }
        if cycle_log && cycle == 0 {
            log::info!(
                "[STAGE] PIRLS/inner step=cycle0 block+joint constraints elapsed={:.3}s n={} p={}",
                constraints_started.elapsed().as_secs_f64(),
                total_joint_n,
                total_p,
            );
        }
        let workspace_build_started = std::time::Instant::now();
        // Get joint Hessian and block gradients from the current evaluation.
        // Hold the cycle's exact-Newton workspace (cache of per-row kernel
        // evaluations at the current β) so a REJECTED cycle can hand it back
        // to `cached_joint_workspace` for the next cycle. After a reject the
        // line search restores β to `old_beta` — exactly the β this workspace
        // was built at — so reusing the cache is bit-identical and skips the
        // O(n) row-kernel re-evaluation (`build_row_kernel_cache`) that
        // otherwise reruns the full data through the per-row CDF/derivative
        // math on every rejected cycle. The converged-exit paths below null
        // this (no carry-forward needed once the inner solve returns).
        let mut hessian_workspace_for_cycle: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
            None;
        let joint_hessian_source = if joint_workspace_requested {
            let cached_hit = cached_joint_workspace.is_some();
            let workspace = match cached_joint_workspace.take() {
                Some(workspace) => workspace,
                None => family
                    .exact_newton_joint_hessian_workspace_with_options(
                        &states, specs, options,
                    )?
                    .ok_or_else(|| {
                        "joint Newton requested an exact Hessian workspace, but the family returned none"
                            .to_string()
                    })?,
            };
            if cycle_log && cycle == 0 {
                log::info!(
                    "[STAGE] PIRLS/inner step=cycle0 hessian-workspace cached_hit={} elapsed={:.3}s n={} p={}",
                    cached_hit,
                    workspace_build_started.elapsed().as_secs_f64(),
                    total_joint_n,
                    total_p,
                );
            }
            hessian_workspace_for_cycle = Some(Arc::clone(&workspace));
            Some(match cached_joint_hessian_source.take() {
                Some(source) => source,
                None => exact_newton_joint_hessian_source_from_workspace(
                    &workspace,
                    total_p,
                    MaterializationIntent::InnerSolve,
                    "joint Newton inner exact-newton operator mismatch",
                )?
                .ok_or_else(|| {
                    "joint Newton exact Hessian workspace supplied no inner-solve curvature source"
                        .to_string()
                })?,
            })
        } else {
            None
        };
        // Row measure observed by the Hessian build above.
        let tr_row_measure_hessian =
            gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
        let joint_hessian_source = match joint_hessian_source {
            Some(source) => source,
            None => {
                // Spec-aware joint Hessian: canonical coupled-curvature
                // source (see the availability gate above). Families that
                // only override `_with_specs` (Dirichlet common-parameter)
                // would otherwise hand back `None` from the spec-less
                // default and silently drop off the joint-Newton path.
                let h_joint_opt = family.exact_newton_joint_hessian_with_specs(&states, specs)?;
                let Some(h_joint) = h_joint_opt else {
                    break; // Fall back to blockwise if joint Hessian unavailable
                };
                match symmetrized_square_matrix(
                    h_joint,
                    total_p,
                    "joint Newton inner exact-newton Hessian shape mismatch",
                ) {
                    Ok(matrix) => JointHessianSource::Dense(matrix),
                    Err(_) => break,
                }
            }
        };
        let hessian_source_elapsed = workspace_build_started.elapsed();
        if hessian_source_elapsed.as_secs_f64() >= 1.0 || (cycle_log && cycle == 0) {
            let source_kind = if matches!(&joint_hessian_source, JointHessianSource::Dense(_)) {
                "dense"
            } else {
                "operator"
            };
            log::info!(
                "[STAGE] PIRLS/inner step=cycle{} hessian-source joint_workspace_requested={} source={} elapsed={:.3}s n={} p={}",
                cycle,
                joint_workspace_requested,
                source_kind,
                hessian_source_elapsed.as_secs_f64(),
                total_joint_n,
                total_p,
            );
        }

        // Concatenate block gradients and betas.
        let Some(grad_joint) = cached_joint_gradient.clone() else {
            break;
        };
        // Row measure observed by the gradient at β. `cached_joint_gradient`
        // was loaded earlier under `options`; if the auto-subsample
        // installer or any sibling path swapped the mask between then and
        // now, the id captured here will diverge from the rest and the
        // pre-ρ check below will Err. Cf. `src/solver/row_measure.rs`.
        let tr_row_measure_gradient =
            gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
        if grad_joint.len() != total_p {
            break;
        }
        let mut beta_joint = Array1::<f64>::zeros(total_p);
        for b in 0..specs.len() {
            let (start, end) = ranges[b];
            beta_joint
                .slice_mut(ndarray::s![start..end])
                .assign(&states[b].beta);
        }

        // Non-finite-curvature guard (gam#1088). A `NaN`/`Inf` in the
        // family curvature `H` makes the penalized Hessian `H_pen = H +
        // S(λ)` — and therefore its spectrum — degenerate, so the KKT
        // certificate is structurally unreachable: the spectral step
        // solve produces garbage, the projected residual neither converges
        // nor trends down, and the residual-based divergence/stall guards
        // below (gated on a *finite* residual that a corrupted-but-not-yet-
        // propagated curvature can still leave finite) do not catch it.
        // Left unguarded the loop then burns the full `inner_loop_hard_
        // ceiling` (1200 cycles) on every outer ρ-eval / seed — the
        // multi-hour link-wiggle & location-scale benchmark timeouts. The
        // penalty is finite by construction, so this is a curvature defect:
        // the trial is degenerate. Exit immediately as non-converged with
        // the current finite β so the outer optimizer rejects this ρ-eval
        // cleanly (mirrors the residual divergence guard below), rather
        // than grinding to the ceiling and reporting a `NaN` H_pen
        // spectrum at the refusal point.
        if !joint_hessian_source_curvature_is_finite(&joint_hessian_source) {
            // A non-finite entry at the STARTING iterate (cycle 0) is a
            // contract violation against the family's analytic joint second
            // derivative — the coupled solve cannot even begin — so it is a
            // typed hard failure at the same smooth-regularized logdet
            // boundary that `validate_block_hessians_finite` enforces for a
            // per-block exact-Newton Hessian (gam#1088 fail-loudly contract).
            // A non-finite entry that only emerges at a LATER cycle, after
            // the coupled Newton loop has driven β to an overflowing
            // operating point during outer optimization, is a genuine
            // ρ-degeneracy: exit non-converged with the current finite β so
            // the outer optimizer rejects this ρ cleanly instead of grinding
            // to inner_max_cycles (the multi-hour link-wiggle & location-
            // scale timeouts). Both exit immediately; only the initial-iterate
            // case aborts, because there is no finite progress to hand back.
            if cycle == 0 {
                joint_hessian_source_finite_check(&joint_hessian_source)?;
            }
            cycles_done = cycle + 1;
            log::warn!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | non-finite-curvature guard (gam#1088): the joint Hessian source carries a non-finite entry, so the penalized Hessian H_pen = H + S(λ) and its spectrum (λ_max/λ_min/cond) are degenerate and the KKT certificate can never be issued; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation instead of grinding to inner_max_cycles={}.",
                cycle,
                inner_max_cycles,
            );
            converged = false;
            break;
        }

        let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;
        let joint_hessian_is_dense = matches!(&joint_hessian_source, JointHessianSource::Dense(_));
        let joint_solver_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
            family,
            &joint_hessian_source,
            &ranges,
            &s_lambdas,
            trace_diagonal_ridge,
            options.ridge_floor,
            joint_bundle,
        );
        // CHEAP CONDITIONING PRE-CHECK (always-on robustness, zero-cost on
        // easy/large fits). Before paying for the dense joint-Hessian
        // materialization + `O(p³)` reduced eigendecomposition inside the
        // Jeffreys term, ask whether the term is PROVABLY skippable from a few
        // matrix-free Hessian-vector products against the source we just built.
        // When `true`, the exact conditioning gate is certain to return the
        // zero term, so every Jeffreys call this cycle short-circuits to the
        // exact-zero contribution WITHOUT forming anything dense — byte-
        // identical to the gated-off path, and preserving the matrix-free path
        // on wide well-conditioned fits. Only runs the estimate when a Jeffreys
        // subspace exists and `total_p` is wide enough that the dense eigh is
        // the cost we want to avoid (the helper itself gates on the size
        // threshold and conservatively returns `false` if unsure). Computed
        // once per inner cycle and reused across the cycle's head-KKT, step,
        // and trial-value calls; the conditioning changes slowly across cycles
        // so re-estimating per cycle (one `O(p·k)` burst) is already cheap
        // against the work it guards.
        let jeffreys_skippable_this_cycle: bool = if options.seed_screening {
            // Seed screening only ranks seeds: skip the O(p · per-axis-Hdot)
            // full Jeffreys gradient/curvature loop. The value-only Jeffreys
            // term (folded into the objective baseline / trial penalties via
            // `custom_family_joint_jeffreys_value`, gated independently on
            // `joint_jeffreys_subspace.is_some()`) still bounds the screening
            // score on separating directions; only the per-axis step curvature
            // — the wrong cost class for ranking on a K-block coupled family —
            // is dropped here (gam#729/#808).
            true
        } else if joint_jeffreys_subspace.is_some() {
            // EXPECTED-INFORMATION GUARD (gam#1020): the skippable
            // certificate probes the OBSERVED Hessian source; it only
            // transfers to the Jeffreys gate when the family's Jeffreys
            // information IS the observed Hessian. Expected-information
            // families (probit-class) bypass the pre-check — observed
            // information grows on saturated rows exactly where the
            // expected information collapses and the gate must arm.
            family.joint_jeffreys_information_matches_observed_hessian()
                && jeffreys_term_skippable_for_source(&joint_hessian_source, total_p)
                    .unwrap_or(false)
        } else {
            false
        };
        let joint_trust_metric_diag = match &joint_hessian_source {
            JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                &h_joint.diag().to_owned(),
                &ranges,
                &s_lambdas,
                joint_solver_diagonal_ridge,
                joint_bundle,
            ),
            JointHessianSource::Operator { diagonal, .. } => joint_penalty_preconditioner_diag(
                diagonal,
                &ranges,
                &s_lambdas,
                joint_solver_diagonal_ridge,
                joint_bundle,
            ),
        };
        // Scale-aware trust-metric floor for a free-scale-coupled block
        // (#1569). A coupled location-scale survival fit drives some rows to
        // small σ (large `exp(−η_σ)`), which inflates the scale-coupled
        // (location / log-σ) block's likelihood-Hessian diagonal on the rows
        // it loads but UNDERSTATES the per-coordinate curvature scale for
        // coefficients loading mostly on large-σ rows. The affine-covariant
        // Moré–Sorensen step then over-reaches on those coordinates (a tiny
        // metric entry blows up the whitened component `c_k/(γ_k+λ)`), the
        // gain ratio never justifies growing the radius, and the inner solve
        // grinds. The family supplies a per-coordinate floor auto-derived from
        // the scale-predictor magnitude (no knob); we take `max(D_i, floor_i)`,
        // so the floor can only tighten the metric and is a no-op for every
        // family that returns `None`. It shapes the trajectory only — the
        // converged β, the KKT certificate, and the REML/LAML the residual
        // feeds are unchanged.
        let mut joint_trust_metric_diag = joint_trust_metric_diag;
        if let Some(floor) = family.joint_trust_metric_block_floor(&states, specs)?
            && floor.len() == joint_trust_metric_diag.len()
        {
            for (d, f) in joint_trust_metric_diag.iter_mut().zip(floor.iter()) {
                if f.is_finite() && *f > *d {
                    *d = *f;
                }
            }
        }
        // HEAD-β JEFFREYS CACHE (gam#729/#808). The full Jeffreys/Firth triple
        // `(Φ, ∇Φ, H_Φ)` costs `p` family directional-derivative calls (the
        // `for k in 0..p` loop in `joint_jeffreys_term`); for a K-block coupled
        // family (Dirichlet/multinomial) that is the dominant per-cycle cost.
        // The head-of-cycle KKT residual, the constrained-QP step, and the
        // spectral/dense Newton step are ALL built at the SAME cycle-start β
        // (`&states`, before any step is accepted), so they need the SAME
        // triple. Compute it ONCE here and reuse, instead of three independent
        // O(p)-directional-derivative evaluations per cycle. The post-step
        // residual below is at the accepted β, so it correctly recomputes.
        // `None` when the term is condition-gated/skippable (∇Φ=0, H_Φ=0).
        let head_beta_key: Array1<f64> = flatten_state_betas(&states, specs);
        let head_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
            if jeffreys_skippable_this_cycle {
                None
            } else if let Some((_, grad_phi, hphi)) = jeffreys_triple_cache
                .as_ref()
                .filter(|(key, _, _)| beta_cache_keys_match_bitwise(key, &head_beta_key))
            {
                // Cross-cycle cache hit: the previous cycle's post-step KKT
                // residual already computed the exact triple at this β. Reuse.
                Some((grad_phi.clone(), hphi.clone()))
            } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
                // The cycle workspace is the authoritative exact-β row cache
                // that supplied this very Hessian source. Ask it for the
                // Jeffreys triple before reconstructing the same information
                // matrix and all-axes derivatives through `family + states`.
                // A workspace without the optional batched derivative retains
                // the generic exact family assembly.
                let workspace_term = match hessian_workspace_for_cycle.as_ref() {
                    Some(workspace) => custom_family_joint_jeffreys_term_from_workspace(
                        workspace.as_ref(),
                        total_p,
                        z_joint,
                        family.joint_jeffreys_term_strength(),
                    )?,
                    None => None,
                };
                let exact_term = match workspace_term {
                    Some(term) => Some(term),
                    None => {
                        custom_family_joint_jeffreys_term(family, &states, specs, &ranges, z_joint)?
                    }
                };
                let term = match exact_term {
                    Some((_phi, grad_phi, hphi))
                        if grad_phi.len() == grad_joint.len()
                            && hphi.nrows() == total_p
                            && hphi.ncols() == total_p =>
                    {
                        Some((grad_phi, hphi))
                    }
                    _ => None,
                };
                if let Some((grad_phi, hphi)) = term.as_ref() {
                    jeffreys_triple_cache =
                        Some((head_beta_key.clone(), grad_phi.clone(), hphi.clone()));
                }
                term
            } else {
                None
            };
        // The divided-difference H_Φ is sufficient for globalization away
        // from the mode. Once the residual-band latch arms the Newton
        // endgame, or a first-order exit asks to certify the returned beta,
        // form the exact second-order remainder exactly once at this beta.
        // There is no component-wise PSD gate: only the spectrum of the
        // complete objective Hessian has mathematical authority.
        let true_jeffreys_hessian_required = (jeffreys_completion_endgame
            || returned_mode_curvature_pending)
            && head_jeffreys_term.is_some();
        let head_jeffreys_completion = if true_jeffreys_hessian_required {
            let z_joint = joint_jeffreys_subspace.as_ref().ok_or_else(|| {
                "joint Newton true Jeffreys Hessian requested without a coefficient subspace"
                    .to_string()
            })?;
            jeffreys_completion_calls += 1;
            Some(exact_joint_jeffreys_completion_at(
                family,
                &states,
                specs,
                z_joint,
                total_p,
                "joint Newton true-Hessian endgame",
            )?)
        } else {
            None
        };
        let head_jeffreys_curvature = head_jeffreys_term.as_ref().map(|(_, hphi)| {
            let mut curvature = hphi.clone();
            if let Some(completion) = head_jeffreys_completion.as_ref() {
                curvature += completion;
            }
            curvature
        });
        // Fold the Firth/Jeffreys score `∇Φ` into the head-of-cycle KKT
        // residual when the term is armed, for the same reason as the
        // post-step residual below: the inner objective is `−ℓ + ½βᵀSβ − Φ`,
        // so the certifiable stationarity is `∇L − Sβ + ∇Φ = 0`. Without
        // this the head-of-cycle KKT exit (`current_stationarity_residual ≤
        // residual_tol`) can never fire on the near-separating span, even
        // when the iterate is the Firth optimum. No-op when the Jeffreys
        // term is unavailable or condition-gated to zero.
        let head_kkt_gradient: Option<Array1<f64>> = head_jeffreys_term
            .as_ref()
            .map(|(grad_phi, _hphi)| &grad_joint + grad_phi);
        let current_kkt_norm = exact_newton_joint_stationarity_inf_norm_from_gradient(
            head_kkt_gradient.as_ref().unwrap_or(&grad_joint),
            &states,
            specs,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            &block_constraints,
            Some(cached_active_sets.as_slice()),
            joint_lower_bounds.as_ref(),
            joint_penalty_stationarity_score(options, specs, &states).as_ref(),
        )?;
        if current_kkt_norm.is_finite() {
            min_certified_residual = min_certified_residual.min(current_kkt_norm);
        }
        let pcg_rel_tol = joint_pcg_eisenstat_walker_forcing(prev_kkt_norm, current_kkt_norm);

        {
            let grad_phi_inf = head_jeffreys_term
                .as_ref()
                .map(|(g, _)| g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
                .unwrap_or(0.0);
            let beta_inf_probe = states
                .iter()
                .flat_map(|s| s.beta.iter())
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            log::info!(
                "[979-PROBE] cyc={:>3} firth_armed={} skippable={} |gradPhi|inf={:.3e} kkt={:.3e} |beta|inf={:.3e} endgame={}",
                cycle,
                head_jeffreys_term.is_some(),
                jeffreys_skippable_this_cycle,
                grad_phi_inf,
                current_kkt_norm,
                beta_inf_probe,
                jeffreys_completion_endgame,
            );
        }

        let solve_joint_constraints_dense =
            joint_constraints.is_some() || !matrix_free_joint_requested || joint_hessian_is_dense;
        if cycle == 0 {
            log::info!(
                "[JN-BRANCH-DIAG #1040] cycle=0 joint_constraints_is_some={} matrix_free_joint_requested={} joint_hessian_is_dense={} solve_joint_constraints_dense={} -> branch={} total_p={} levenberg_on_ill_cond={}",
                joint_constraints.is_some(),
                matrix_free_joint_requested,
                joint_hessian_is_dense,
                solve_joint_constraints_dense,
                if solve_joint_constraints_dense && joint_constraints.is_some() {
                    "CONSTRAINED_QP"
                } else if matrix_free_joint_requested && !joint_hessian_is_dense {
                    "MATRIX_FREE_PCG"
                } else {
                    "DENSE_SPECTRAL"
                },
                total_p,
                family.levenberg_on_ill_conditioning(),
            );
        }
        // Exact trust-region subproblem factorization (gam#979). Populated on
        // the unconstrained dense-spectral path with the metric-whitened
        // eigendecomposition of the penalized Hessian, so the trust loop below
        // re-solves the *exact* Moré–Sorensen subproblem at each trust radius
        // from one factorization — replacing the dogleg/Cauchy/box-truncation
        // globalization with the single object they all approximate. `None` on
        // the constrained-QP and matrix-free PCG paths, which keep their
        // existing globalization untouched.
        let mut joint_spectrum: Option<whitened_spectrum::WhitenedHessianSpectrum> = None;
        // DENSE-FALLBACK OPERATOR MATERIALIZATION REUSE (gam#1040). On the
        // DENSE_SPECTRAL path the inner Hessian `source` can be a matrix-free
        // `Operator` (BMS flex, large n, p below the matrix-free joint-dim
        // threshold so PCG is not requested): the dense-fallback below then
        // calls `materialize_joint_hessian_source` to form the unpenalized
        // dense `H` ONCE for the spectral `decompose`. Without capturing it,
        // the per-cycle Cauchy leg and the up-to-`JOINT_TRUST_MAX_ATTEMPTS`
        // predicted-reduction matvecs each re-apply the operator's `apply_into`
        // — an `O(n·p)` row sweep over n≈196k rows, ~25× per cycle — when the
        // identical action is already available as an `O(p²)` dense matvec.
        // Capturing the unpenalized dense here and routing those matvecs
        // through a `Dense` source is byte-identical (the dense build IS the
        // operator's action by construction of `materialize_joint_hessian_source`)
        // and removes the dominant residual per-cycle row work on this path.
        let mut materialized_dense_unpenalized: Option<Array2<f64>> = None;
        let (
            candidate_beta,
            joint_active_set,
            joint_step_spectral_nullity,
            joint_reduced_face_kind,
        ) = if solve_joint_constraints_dense && let Some(constraints) = joint_constraints.as_ref() {
            let mut lhs = match materialize_joint_hessian_source(
                &joint_hessian_source,
                total_p,
                "joint Newton inner constrained Hessian materialization",
            ) {
                Ok(matrix) => matrix,
                Err(_) => break,
            };
            let exact_lhs = if true_jeffreys_hessian_required {
                assemble_true_joint_objective_hessian(
                    lhs.clone(),
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                    head_jeffreys_term.as_ref().map(|(_, hphi)| hphi),
                    head_jeffreys_completion.as_ref(),
                    "joint Newton constrained true-Hessian endgame",
                )?
            } else {
                let mut matrix = lhs.clone();
                add_joint_penalty_to_matrix(
                    &mut matrix,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                );
                if let Some(curvature) = head_jeffreys_curvature.as_ref() {
                    matrix += curvature;
                }
                matrix
            };
            add_joint_penalty_to_matrix(
                &mut lhs,
                &ranges,
                &s_lambdas,
                trace_diagonal_ridge,
                joint_bundle,
            );
            if joint_solver_diagonal_ridge != trace_diagonal_ridge {
                for d in 0..lhs.nrows() {
                    lhs[[d, d]] += joint_solver_diagonal_ridge - trace_diagonal_ridge;
                }
            }
            check_linear_feasibility(&beta_joint, constraints).map_err(|e| {
                CustomFamilyError::trial_point(format!(
                    "joint Newton constrained solve [cycle={cycle}]: {e}"
                ))
            })?;
            let warm_joint_active =
                flatten_joint_active_set(&cached_active_sets, &block_constraints);
            let lower_bounds = match extract_simple_lower_bounds(constraints, total_p) {
                Ok(bounds) => bounds,
                Err(_) => break,
            };
            // Newton IRLS step in absolute-β space:
            //
            //   β_new = H_pen⁻¹ (H_L β + ∇ℓ)
            //
            // where H_pen = H_L + S, derived from Newton's update
            //   β_new = β + H_pen⁻¹(∇ℓ − Sβ)
            //         = H_pen⁻¹(H_pen β + ∇ℓ − Sβ)
            //         = H_pen⁻¹(H_L β + ∇ℓ).
            //
            // The QP `min 0.5 β' H_pen β − rhs_beta' β` has unconstrained
            // optimum β = H_pen⁻¹ rhs_beta, so rhs_beta = H_pen β + (∇ℓ − Sβ)
            // gives the correct Newton update. Passing raw grad_joint (=∇ℓ)
            // would collapse to β = H_pen⁻¹ ∇ℓ, which at the true optimum
            // (∇ℓ = Sβ̂) gives H_pen⁻¹ Sβ̂ ≠ β̂ — wrong fixed point.
            let penalty_beta_joint = apply_joint_block_penalty(
                &ranges,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
                joint_bundle,
            );
            let mut rhs_step = &grad_joint - &penalty_beta_joint;
            // Reuse the head-β Jeffreys triple (consistently attenuated in
            // `head_jeffreys_term` — both ∇Φ and H_Φ scaled by one scalar,
            // gam#826/#872/#715). Skipped when the cheap pre-check certifies
            // well-conditioning: ∇Φ = 0 and H_Φ = 0 there, so neither
            // rhs_step nor lhs change.
            // The QP is convexified only after every objective component is
            // assembled. Projecting H_Φ by itself is not legitimate:
            // H_Φ+completion may be indefinite while H+S stabilizes the full
            // objective, or vice versa. The reduced-face and ambient spectra
            // below retain `exact_lhs`; the QP receives a full-matrix
            // modified-Newton geometry solely for globalization.
            if let Some((grad_phi, hphi)) = head_jeffreys_term.as_ref()
                && grad_phi.len() == rhs_step.len()
            {
                rhs_step += grad_phi;
                let curvature = head_jeffreys_curvature.as_ref().unwrap_or(hphi);
                lhs += curvature;
            }
            // The constrained QP cannot drop ker(H_pen) the way the
            // spectral range solve does. A numerical gauge therefore
            // needs positive curvature so the minimizer is unique, but
            // adding the residual-scaled μ to every diagonal also damps
            // weak IDENTIFIED modes. The 4,800-row CTN measured the result:
            // a stable 79-row face, flat objective, and residual contraction
            // of 0.9998 per cycle. `symmetric_constrained_hessian_geometry`
            // installs μ only on the certified null projector, leaving
            // range(H_pen) on the exact Newton equation. A family that
            // explicitly owns the separate full-rank ill-conditioning case
            // retains its ambient policy.
            //
            // Scale gauge curvature by the PROJECTED stationarity residual,
            // not the raw RHS: at a constrained optimum the raw RHS includes
            // the non-vanishing multiplier mass Aᵀλ, while the projected
            // residual is the actual distance from KKT and tends to zero.
            let rhs_inf = rhs_step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let floor_scale = if current_kkt_norm.is_finite() {
                current_kkt_norm.min(rhs_inf)
            } else {
                rhs_inf
            };
            let constrained_levenberg_mu = JOINT_SPECTRAL_LEVENBERG_FACTOR * floor_scale;
            // MODIFIED-NEWTON CONVEXIFICATION (gam#1040 / gam#979). The
            // exact survival marginal-slope joint NLL Hessian is INDEFINITE
            // on the flat baseline-hazard λ valley (the linear baseline +
            // the z·exp(slope) cross-coupling carry genuine negative
            // curvature away from the optimum). The active-set QP below
            // minimizes `½βᵀHβ − rhs_betaᵀβ`; with an indefinite `H` that
            // model has a direction that LOWERS the local quadratic
            // objective while moving AWAY from the KKT point. The
            // trust-region wrapper gates acceptance on the objective-
            // reduction ratio ρ — NOT on the stationarity residual — so it
            // accepts every such step at ρ≈1 and GROWS its radius while the
            // stationarity residual DIVERGES (the measured 3.5e4 → 9.5e6
            // blow-up on the time block). The unconstrained dense-spectral
            // path never exhibits this because `WhitenedHessianSpectrum`
            // already reflects negative-curvature modes to `|γ|`; the
            // constrained branch must do the same to its dense `lhs`.
            // Reflecting (not clamping-to-zero) keeps the curvature
            // magnitude so the QP stays bounded and the step length matches
            // the dense path; at a genuine constrained optimum the reduced
            // Hessian is PSD so this is a no-op and the converged β is
            // unchanged.
            //
            // NEWTON-DECREMENT CERTIFICATE ON THE CONSTRAINED PATH
            // (gam#1040 / gam#1088). The dense-spectral branch populates
            // `joint_spectrum` (line ~1493) so the convergence loop's
            // Newton-decrement exit can terminate the geometric/linear tail
            // when the achievable model descent `½ Σ c_k²/|γ_k|` drops below
            // `objective_tol`. The constrained branch never set it, so a
            // weakly-identified survival-MS fit (the n≈2e5 slope block,
            // step clamped by the trust region, residual creeping ~7%/cycle)
            // had no early-exit and ground the whole budget. Build the same
            // D-whitened spectrum from the penalized `lhs` (decrement reflects
            // negative modes via `.abs()` internally, so the pre-reflection
            // `lhs` is the right input) and the augmented stationarity RHS, so
            // the decrement read is consistent with the dense path. Diagnostic
            // only for the convergence test — it does NOT change the QP step.
            let spectrum_started = std::time::Instant::now();
            let spectrum_scope = gam_runtime::process_monitor::track_scope(format!(
                "joint Newton ambient spectrum cycle={cycle} p={total_p}"
            ));
            if let Ok(spectrum) = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                &exact_lhs,
                &rhs_step,
                &joint_trust_metric_diag,
                KKT_REFUSAL_RANK_TOL,
            ) {
                joint_spectrum = Some(spectrum);
            }
            drop(spectrum_scope);
            let spectrum_elapsed = spectrum_started.elapsed();
            if spectrum_elapsed >= std::time::Duration::from_secs(1) {
                log::warn!(
                    "[gam#979 constrained-QP phase] cycle={cycle} phase=ambient-spectrum elapsed_s={:.3}",
                    spectrum_elapsed.as_secs_f64(),
                );
            }
            let convexification_started = std::time::Instant::now();
            let convexification_scope = gam_runtime::process_monitor::track_scope(format!(
                "joint Newton convexification cycle={cycle} p={total_p}"
            ));
            let constrained_geometry = symmetric_constrained_hessian_geometry(
                &lhs,
                constrained_levenberg_mu,
                family.levenberg_on_ill_conditioning(),
            )?;
            drop(convexification_scope);
            let convexification_elapsed = convexification_started.elapsed();
            if convexification_elapsed >= std::time::Duration::from_secs(1) {
                log::warn!(
                    "[gam#979 constrained-QP phase] cycle={cycle} phase=convexification elapsed_s={:.3}",
                    convexification_elapsed.as_secs_f64(),
                );
            }
            if cycle <= 2 {
                let min_eval_raw = constrained_geometry.raw_min_eigenvalue;
                let min_eval_refl = constrained_geometry.stabilized_min_eigenvalue;
                log::info!(
                    "[JN-REFLECT-DIAG #1040] cycle={cycle} CONSTRAINED_QP lambda_min_signed_raw={min_eval_raw:.3e} lambda_min_signed_reflected={min_eval_refl:.3e} nullity={} condition={:.3e} (reflection {})",
                    constrained_geometry.nullity,
                    constrained_geometry.condition,
                    if min_eval_refl > min_eval_raw + min_eval_raw.abs() * 1e-9 {
                        "CHANGED the spectrum"
                    } else {
                        "NO-OP (already PSD)"
                    },
                );
            }
            // The free solve and bound-multiplier KKT test must use this
            // same convexified Hessian. Mixing the reflected step model
            // with the original indefinite curvature or the bare gradient
            // makes release and entry contradict each other (gam#979).
            let lhs = constrained_geometry.matrix;
            let rhs_beta = &lhs.dot(&beta_joint) + &rhs_step;
            let exact_face_candidate = if lower_bounds.is_none() {
                if let Some(active_rows) = warm_joint_active.as_deref() {
                    let reduced_face_started = std::time::Instant::now();
                    let reduced_face_scope = gam_runtime::process_monitor::track_scope(format!(
                        "joint Newton reduced face cycle={cycle} p={total_p} warm_rows={}",
                        active_rows.len(),
                    ));
                    let result = certified_reduced_face_candidate(
                        &exact_lhs,
                        &rhs_step,
                        &beta_joint,
                        constraints,
                        active_rows,
                        &joint_trust_metric_diag,
                        joint_trust_radius,
                    );
                    drop(reduced_face_scope);
                    let reduced_face_elapsed = reduced_face_started.elapsed();
                    if reduced_face_elapsed >= std::time::Duration::from_secs(1) {
                        log::warn!(
                            "[gam#979 constrained-QP phase] cycle={cycle} phase=reduced-face elapsed_s={:.3} warm_rows={}",
                            reduced_face_elapsed.as_secs_f64(),
                            active_rows.len(),
                        );
                    }
                    result?
                } else {
                    None
                }
            } else {
                None
            };
            let reduced_face_kind = exact_face_candidate.as_ref().map(|(_, _, kind)| *kind);
            let metric_projection_started = std::time::Instant::now();
            let metric_projection_scope = gam_runtime::process_monitor::track_scope(format!(
                "joint Newton metric projection cycle={cycle} p={total_p}"
            ));
            let solve_result = if let Some((candidate, active, _)) = exact_face_candidate {
                Ok((candidate, active))
            } else if let Some(bounds) = lower_bounds.as_ref() {
                solve_quadratic_with_simple_lower_bounds(
                    &lhs,
                    &rhs_beta,
                    &beta_joint,
                    bounds,
                    warm_joint_active.as_deref(),
                )
            } else {
                gam_solve::active_set::solve_quadratic_with_constraint_set(
                    &lhs,
                    &rhs_beta,
                    &beta_joint,
                    constraints,
                    warm_joint_active.as_deref(),
                )
                .map_err(|error| CustomFamilyError::trial_point(error.to_string()))
            };
            drop(metric_projection_scope);
            let metric_projection_elapsed = metric_projection_started.elapsed();
            if metric_projection_elapsed >= std::time::Duration::from_secs(1) {
                log::warn!(
                    "[gam#979 constrained-QP phase] cycle={cycle} phase=metric-projection elapsed_s={:.3}",
                    metric_projection_elapsed.as_secs_f64(),
                );
            }
            match solve_result {
                Ok((beta_new, active_set)) => {
                    // Durable constrained-QP liveness: per-cycle active-face
                    // size + ‖β‖∞ distinguishes a changing working set from a
                    // stable face with a blowing-up free direction
                    // (near-separation). WARN reaches bounded workflow captures;
                    // cond/nullity/diagnosis come from `format_structured_log`
                    // at any refused exit.
                    log::warn!(
                        "[gam#979 constrained-QP] cycle={} path={} warm_rows={} active_set_rows={} beta_inf={:.4e}",
                        cycle,
                        match reduced_face_kind {
                            Some(ReducedFaceCandidateKind::ExactNewton) => "exact-face",
                            Some(ReducedFaceCandidateKind::ReducedNewton) => "reduced-newton",
                            Some(ReducedFaceCandidateKind::RegularizedNewton) => {
                                "regularized-newton"
                            }
                            None if lower_bounds.is_some() => "simple",
                            None => "linear",
                        },
                        warm_joint_active.as_ref().map_or(0, |v| v.len()),
                        active_set.len(),
                        beta_new.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
                    );
                    (beta_new, Some(active_set), 0usize, reduced_face_kind)
                }
                Err(error) => {
                    return Err(CustomFamilyError::trial_point(format!(
                        "joint constrained Newton QP failed at cycle {cycle} \
                             (constraint_rows={}, warm_active_rows={}, beta_inf={:.6e}, \
                             rhs_inf={:.6e}): {error}",
                        constraints.nrows(),
                        warm_joint_active.as_ref().map_or(0, Vec::len),
                        beta_joint
                            .iter()
                            .map(|value| value.abs())
                            .fold(0.0_f64, f64::max),
                        rhs_step
                            .iter()
                            .map(|value| value.abs())
                            .fold(0.0_f64, f64::max),
                    )));
                }
            }
        } else {
            // Stationarity residual: r = S*beta - gradient (for penalized NLL)
            let penalty_beta = apply_joint_block_penalty(
                &ranges,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
                joint_bundle,
            );
            let mut rhs = &grad_joint - &penalty_beta;
            // Universal robustness: fold the family-general
            // Jeffreys/Firth curvature `H_Φ` and score `∇Φ` into the dense
            // spectral step below, scoped to the full-span basis `Z_J`.
            // Computed ONCE here so the RHS and curvature share the SAME
            // term. The inner objective is
            // `−ℓ + ½βᵀSβ − Φ`, so the Newton system the step must solve is
            //   (H + S_λ + H_Φ) δ = (∇ℓ − S_λβ) + ∇Φ.
            // An active H_Φ can be indefinite, so it cannot be projected by
            // itself and handed to SPD-only CG. Active Jeffreys cycles route
            // to the exact dense Moré–Sorensen spectrum; well-conditioned
            // cycles are certified as exact-zero by the pre-check and keep
            // the matrix-free PCG path.
            // Cheap pre-check certified well-conditioned ⇒ the exact term
            // is the zero contribution (∇Φ = 0, H_Φ = 0). Short-circuit to
            // `None` WITHOUT materializing the dense joint Hessian or running
            // the O(p³) reduced eigendecomposition — this is the matrix-free
            // PCG hot path, where forming a dense p×p H_Φ every cycle was the
            // regression. Byte-identical to the gated-off dense path: `rhs`
            // is left as `∇ℓ − S_λβ` and no H_Φ is folded into the matvec.
            // Reuse the head-β Jeffreys triple (computed once this cycle);
            // this Newton step is built at the same cycle-start β.
            let inner_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
                match head_jeffreys_term.as_ref() {
                    Some((grad_phi, hphi)) if grad_phi.len() == rhs.len() => {
                        rhs += grad_phi;
                        Some((grad_phi.clone(), hphi.clone()))
                    }
                    _ => None,
                };
            let pcg_started = std::time::Instant::now();
            let pcg_requested = matrix_free_joint_requested
                && !joint_hessian_is_dense
                && !returned_mode_curvature_pending
                && !true_jeffreys_hessian_required
                && inner_jeffreys_term.is_none();
            let mut spectral_nullity_for_step = 0usize;
            let mut delta = if pcg_requested {
                let preconditioner_diag = match &joint_hessian_source {
                    JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                        &h_joint.diag().to_owned(),
                        &ranges,
                        &s_lambdas,
                        joint_solver_diagonal_ridge,
                        joint_bundle,
                    ),
                    JointHessianSource::Operator { diagonal, .. } => {
                        joint_penalty_preconditioner_diag(
                            diagonal,
                            &ranges,
                            &s_lambdas,
                            joint_solver_diagonal_ridge,
                            joint_bundle,
                        )
                    }
                };
                // Pre-allocate the penalty workspace ONCE outside the
                // PCG closure so each CG iter (called hundreds-to-
                // thousands of times per outer iter at large scale)
                // reuses the buffer instead of allocating per call.
                // RefCell because solve_spd_pcg* expects `Fn` (immutable
                // borrow of captures) and we need interior mutability
                // to write into the workspace.
                let penalty_workspace = RefCell::new(Array1::<f64>::zeros(total_p));
                match &joint_hessian_source {
                    JointHessianSource::Dense(h_joint) => {
                        gam_linalg::utils::solve_spd_pcg_with_info_into(
                            |v, out| {
                                // h_joint * v -> out (faer-backed, no alloc)
                                gam_linalg::faer_ndarray::fast_av_view_into(
                                    h_joint,
                                    v,
                                    out.view_mut(),
                                );
                                let mut pen = penalty_workspace.borrow_mut();
                                apply_joint_block_penalty_into(
                                    &ranges,
                                    &s_lambdas,
                                    v,
                                    joint_solver_diagonal_ridge,
                                    &mut pen,
                                    joint_bundle,
                                );
                                *out += &*pen;
                            },
                            &rhs,
                            &preconditioner_diag,
                            pcg_rel_tol,
                            JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                        )
                        .map(|(solution, info)| {
                            log_joint_pcg_diagnostics(
                                cycle,
                                total_p,
                                total_joint_n,
                                &preconditioner_diag,
                                &info,
                            );
                            solution
                        })
                    }
                    JointHessianSource::Operator { apply_into, .. } => {
                        let apply_h_into = Arc::clone(apply_into);
                        gam_linalg::utils::solve_spd_pcg_with_info_into(
                            |v, out| {
                                if let Err(error) = apply_h_into(v, out) {
                                    log::warn!(
                                        "joint Newton inner operator matvec failed: {error}"
                                    );
                                    out.fill(0.0);
                                }
                                let mut pen = penalty_workspace.borrow_mut();
                                apply_joint_block_penalty_into(
                                    &ranges,
                                    &s_lambdas,
                                    v,
                                    joint_solver_diagonal_ridge,
                                    &mut pen,
                                    joint_bundle,
                                );
                                *out += &*pen;
                            },
                            &rhs,
                            &preconditioner_diag,
                            pcg_rel_tol,
                            JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                        )
                        .map(|(solution, info)| {
                            log_joint_pcg_diagnostics(
                                cycle,
                                total_p,
                                total_joint_n,
                                &preconditioner_diag,
                                &info,
                            );
                            solution
                        })
                    }
                }
            } else {
                None
            };
            if pcg_requested {
                log::info!(
                    "[PIRLS/joint-PCG] cycle {:>3} | n={} p={} solved={} elapsed={:.3}s",
                    cycle,
                    total_joint_n,
                    total_p,
                    delta.is_some(),
                    pcg_started.elapsed().as_secs_f64()
                );
            }
            if delta.is_none() {
                if pcg_requested {
                    break;
                }
                let likelihood_hessian = match materialize_joint_hessian_source(
                    &joint_hessian_source,
                    total_p,
                    "joint Newton inner dense fallback Hessian materialization",
                ) {
                    Ok(matrix) => matrix,
                    Err(_) => break,
                };
                // Capture the unpenalized dense `H` for the rest of this
                // cycle (gam#1040): the Cauchy leg and trust-region
                // predicted-reduction matvecs below can then reuse it as a
                // cheap `O(p²)` dense matvec instead of re-applying a
                // matrix-free operator `O(n·p)` per attempt. Only when the
                // source is an `Operator` — a `Dense` source already gives
                // those matvecs the fast path, so cloning would be waste.
                if matches!(&joint_hessian_source, JointHessianSource::Operator { .. }) {
                    materialized_dense_unpenalized = Some(likelihood_hessian.clone());
                }
                let m_dd_for_probe = if true_jeffreys_hessian_required
                    && !jeffreys_true_hessian_probe_logged
                    && log::log_enabled!(log::Level::Info)
                {
                    let mut matrix = likelihood_hessian.clone();
                    add_joint_penalty_to_matrix(
                        &mut matrix,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                    );
                    if let Some((_gradient, hphi)) = inner_jeffreys_term.as_ref() {
                        matrix += hphi;
                    }
                    symmetrize_dense_in_place(&mut matrix);
                    Some(matrix)
                } else {
                    None
                };
                let lhs_true = if true_jeffreys_hessian_required {
                    assemble_true_joint_objective_hessian(
                        likelihood_hessian,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                        inner_jeffreys_term.as_ref().map(|(_, hphi)| hphi),
                        head_jeffreys_completion.as_ref(),
                        "joint Newton unconstrained true-Hessian endgame",
                    )?
                } else {
                    let mut matrix = likelihood_hessian;
                    add_joint_penalty_to_matrix(
                        &mut matrix,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                    );
                    if let Some((_gradient, hphi)) = inner_jeffreys_term.as_ref() {
                        matrix += hphi;
                    }
                    matrix
                };
                // Universal robustness: add the
                // family-general Jeffreys curvature `H_Phi` to the
                // penalized Hessian. This is the Tier-B coupled-Newton form
                // of Firth: the reduced Fisher information `Z_J^T H Z_J`
                // supplies the missing O(n) curvature that bounds a
                // near-separating coefficient to O(1). When the Jeffreys
                // term is unavailable, the step stays unaugmented.
                //
                // `∇Φ` is NOT re-added here: `rhs` (and thus `spectral_rhs`)
                // already carries `+∇Φ` from the single shared computation
                // above, and we REUSE that same `H_Φ` here rather than
                // recomputing the (O(p) directional-derivative) term — the
                // dense fallback and the matrix-free PCG step now solve the
                // SAME Jeffreys-augmented Newton system.
                let spectral_rhs = rhs.clone();
                // ENDGAME EXACTNESS (gam#979). Outside the endgame the
                // matrix above is M_DD = H+S+H_Φ. Once armed it is M_true,
                // assembled by `assemble_true_joint_objective_hessian` from
                // H+S+H_Φ+completion with no component-only PSD authority.
                // Single metric-whitened eigendecomposition drives BOTH the
                // seed step and every trust-region re-solve this cycle
                // (gam#979). The prior code ran a SECOND O(p³)
                // eigendecomposition of the raw Hessian here purely to form
                // the seed step — doubling the dominant per-cycle cost on the
                // ~5 s/cycle ill-conditioned survival marginal-slope inner.
                // The exact trust-region multiplier λ (chosen so ‖δ‖_D = r)
                // subsumes the old self-vanishing Levenberg-μ seed: `decompose`
                // whitens by the trust metric so the penalty (λ~e²⁴) and the
                // likelihood scales are throttled uniformly — the scale
                // invariance the multiplicative μ approximated. `lhs_true`
                // already carries the penalty and the Firth/Jeffreys curvature
                // H_Φ and `spectral_rhs` the augmented stationarity RHS, so the
                // subproblem model matches the predicted-reduction model and the
                // accept/reject gain ratio exactly.
                let spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                    &lhs_true,
                    &spectral_rhs,
                    &joint_trust_metric_diag,
                    KKT_REFUSAL_RANK_TOL,
                )?;
                // A positive-definite M_true owns the exact Newton step and
                // therefore the quadratic endgame. An indefinite M_true is
                // not reflected into a fake local minimum: start directly
                // with the finite-radius Moré–Sorensen solution (including
                // its hard-case negative-eigenspace component), then let the
                // ordinary trust loop accept, shrink, or escape it.
                let spectral_step = if true_jeffreys_hessian_required
                    && spectrum.has_resolvable_negative_curvature()
                {
                    spectrum.trust_region_step(joint_trust_radius)
                } else {
                    spectrum.trust_region_step(f64::INFINITY)
                };
                spectral_nullity_for_step = spectral_step.nullity;
                if let Some(m_dd) = m_dd_for_probe.as_ref() {
                    let component = head_jeffreys_curvature.as_ref().ok_or_else(|| {
                        "true-Hessian diagnostic is missing Jeffreys curvature".to_string()
                    })?;
                    let (component_min, component_max) = symmetric_eigen_extremes(
                        component,
                        "true-Hessian Jeffreys component spectrum",
                    )?;
                    let (m_dd_min, m_dd_max) =
                        symmetric_eigen_extremes(m_dd, "divided-difference Hessian spectrum")?;
                    let (m_true_min, m_true_max) =
                        symmetric_eigen_extremes(&lhs_true, "true objective Hessian spectrum")?;
                    let dd_spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                        m_dd,
                        &spectral_rhs,
                        &joint_trust_metric_diag,
                        KKT_REFUSAL_RANK_TOL,
                    )?;
                    let dd_step = dd_spectrum.trust_region_step(f64::INFINITY);
                    let dd_contraction =
                        linearized_residual_contraction(&lhs_true, &spectral_rhs, &dd_step.delta);
                    let true_contraction = linearized_residual_contraction(
                        &lhs_true,
                        &spectral_rhs,
                        &spectral_step.delta,
                    );
                    log::info!(
                        "[979-TRUE-HESSIAN] cycle={cycle} completion_call={} \
                         eig(H_phi+completion)=[{component_min:.6e},{component_max:.6e}] \
                         eig(M_DD)=[{m_dd_min:.6e},{m_dd_max:.6e}] \
                         eig(M_true)=[{m_true_min:.6e},{m_true_max:.6e}] \
                         current_dd_contraction={dd_contraction:.6e} \
                         true_contraction={true_contraction:.6e} \
                         true_indefinite={} trust_radius={joint_trust_radius:.6e}",
                        jeffreys_completion_calls,
                        spectrum.has_resolvable_negative_curvature(),
                    );
                    jeffreys_true_hessian_probe_logged = true;
                }
                // gam#979: Levenberg shift-to-PD of the SEED search direction on
                // the rigid ill-conditioned path. When the whitened inner Hessian
                // is indefinite the Moré–Sorensen step reflects the negative
                // modes to |λ|; on the near-separable coupled marginal-slope
                // surface those reflected modes then ride a poor gain ratio that
                // shrinks the single scalar trust radius, which clamps the
                // *well-conditioned* modes that DO carry real descent — the
                // measured "reflected-descent crawl" where the residual plateaus
                // (‖g‖~1e-1) while the objective creeps down ~1e-3/cycle and the
                // solve exhausts its budget without ever reaching residual_tol
                // (the binary twin of the survival-MS oversmoothed-ρ endgame).
                // Once the residual has stalled for a few cycles, seed the trust
                // loop from a genuinely convex modified-Newton step: add
                // μ·D_trust (μ just above |λ_min| so the reflected modes become
                // gently positive while the well-conditioned modes, |λ|≫μ, keep
                // their full Newton step) and re-solve once. This is a modified-
                // Newton SEARCH DIRECTION only — the trust-region accept/reject
                // still judges it against the true `lhs_true` model, `joint_spectrum`
                // stays the exact (unshifted) spectrum for the trust re-solves and
                // the Newton-decrement certificate, and μ is applied ONLY while the
                // Hessian is indefinite (it vanishes once the endgame becomes PD),
                // so a healthy convex fit is byte-unchanged and no non-minimum can
                // be certified. Family-gated on `levenberg_on_ill_conditioning()`.
                const JOINT_REFLECTED_CONVEXIFY_STALL_WINDOW: usize = 3;
                const JOINT_REFLECTED_CONVEXIFY_MARGIN: f64 = 1.5;
                let mut seed_delta = spectral_step.delta;
                if !true_jeffreys_hessian_required
                    && family.levenberg_on_ill_conditioning()
                    && spectral_step.reflected_negative_modes > 0
                    && cycles_since_residual_improved >= JOINT_REFLECTED_CONVEXIFY_STALL_WINDOW
                    && spectral_step.most_negative_eigenvalue.is_finite()
                    && spectral_step.most_negative_eigenvalue < 0.0
                {
                    let mu = spectral_step.most_negative_eigenvalue.abs()
                        * JOINT_REFLECTED_CONVEXIFY_MARGIN;
                    if mu.is_finite() && mu > 0.0 {
                        let mut lhs_convex = lhs_true.clone();
                        for d in 0..lhs_convex.nrows() {
                            lhs_convex[[d, d]] += mu * joint_trust_metric_diag[d];
                        }
                        if let Ok(convex_spectrum) =
                            whitened_spectrum::WhitenedHessianSpectrum::decompose(
                                &lhs_convex,
                                &spectral_rhs,
                                &joint_trust_metric_diag,
                                KKT_REFUSAL_RANK_TOL,
                            )
                        {
                            let convex_step = convex_spectrum.trust_region_step(f64::INFINITY);
                            if convex_step.reflected_negative_modes == 0
                                && convex_step.delta.iter().all(|v| v.is_finite())
                            {
                                log::info!(
                                    "[PIRLS/joint-Newton] cycle {cycle:>3} | gam#979 \
                                         Levenberg shift-to-PD seed: μ={mu:.3e}·D convexified \
                                         {} reflected mode(s) (λ_min={:.3e}) after {} stalled \
                                         cycle(s); seeding the trust loop from the convex \
                                         modified-Newton step to break the reflected-descent \
                                         crawl",
                                    spectral_step.reflected_negative_modes,
                                    spectral_step.most_negative_eigenvalue,
                                    cycles_since_residual_improved,
                                );
                                seed_delta = convex_step.delta;
                            }
                        }
                    }
                }
                if spectral_step.reflected_negative_modes > 0 {
                    log::info!(
                        "[PIRLS/joint-Newton] cycle {cycle:>3} | indefinite inner \
                             Hessian: reflected {}/{} negative-curvature modes to |λ| \
                             (λ_min={:.3e}); proceeding with modified-Newton descent step \
                             under trust-region globalization",
                        spectral_step.reflected_negative_modes,
                        total_p,
                        spectral_step.most_negative_eigenvalue,
                    );
                }
                {
                    log::info!(
                        "[979-DIAG] cycle {cycle:>3} spectral solve: nullity@{:.0e}={}/{} \
                         |P0 rhs|∞={:.3e} |P+ rhs|∞={:.3e} λ_min+={:.3e} λ_max={:.3e} reflected={}",
                        spectral_step.rank_tol,
                        spectral_step.nullity,
                        total_p,
                        spectral_step.null_rhs_inf,
                        spectral_step.range_rhs_inf,
                        spectral_step.lambda_min_positive,
                        spectral_step.lambda_max_abs,
                        spectral_step.reflected_negative_modes,
                    );
                }
                delta = Some(seed_delta);
                // The same factorization powers every trust-radius re-solve
                // in the loop below (gam#979) — no second eigendecomposition.
                // `spectrum` is the EXACT (unshifted) Hessian: the gam#979
                // Levenberg shift-to-PD above only reshapes the SEED direction,
                // so the trust re-solves and the Newton-decrement certificate
                // keep judging against the true model.
                joint_spectrum = Some(spectrum);
            }

            let Some(delta) = delta else {
                break; // Fall back to blockwise
            };
            if !delta.iter().all(|v| v.is_finite()) {
                break; // Fall back to blockwise
            }
            (
                beta_joint.clone() + &delta,
                None,
                spectral_nullity_for_step,
                None,
            )
        };
        // Hessian-source build (and any QP solve immediately above) are
        // done by the time we reach `delta`. Capture the wall-clock
        // before the line-search phase so the end-of-cycle summary can
        // attribute time correctly between the Hessian/QP and the
        // backtracking step search.
        let hessian_and_qp_elapsed = hessian_started.elapsed();
        drop(hessian_scope_guard);
        let line_search_started = std::time::Instant::now();
        log::info!(
            "[joint-newton-tr] phase=line_search cycle={} r={:.3e} hessian_qp_elapsed={:.3}s",
            cycle,
            joint_trust_radius,
            hessian_and_qp_elapsed.as_secs_f64(),
        );
        let delta = &candidate_beta - &beta_joint;
        // Effective Hessian source for the remaining per-cycle matvecs
        // (Cauchy leg + trust-region predicted reduction). When the dense
        // fallback above materialized a matrix-free `Operator` to dense, route
        // those matvecs through that `Dense` snapshot so each is an `O(p²)`
        // GEMV rather than an `O(n·p)` operator row-sweep repeated up to
        // `JOINT_TRUST_MAX_ATTEMPTS` times (gam#1040). Byte-identical action
        // (the dense build IS the operator's action by construction); falls
        // back to the original source when no dense snapshot was taken (the
        // already-`Dense` and PCG paths).
        let dense_snapshot_source = materialized_dense_unpenalized.map(JointHessianSource::Dense);
        let effective_hessian_source: &JointHessianSource = dense_snapshot_source
            .as_ref()
            .unwrap_or(&joint_hessian_source);

        // Trust-region globalization for the joint Newton proposal.  The
        // previous implementation used up to eight backtracking likelihood
        // evaluations (each can build the exact joint workspace at large-scale
        // scale).  Here the step is truncated before evaluation and the
        // single trial objective is accepted only when the actual decrease
        // is positive relative to the local quadratic model.
        let step_inf = delta.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);

        let old_beta: Vec<Array1<f64>> = states.iter().map(|s| s.beta.clone()).collect();
        // `‖β‖₁` at the cycle's incumbent, for the penalty term's accumulation
        // scale in the objective-resolution ceiling (gam#2748).
        let old_beta_l1_norm: f64 = old_beta
            .iter()
            .flat_map(|beta| beta.iter())
            .map(|value| value.abs())
            .sum();
        // Firth value Φ at the OLD (start-of-cycle) β, folded under the SAME
        // skippable gate the trial uses below — so `actual_reduction =
        // old_objective − trialobjective` compares two points on one objective
        // `−ℓ + ½βᵀSβ − Φ` (gam#826/#872). `lastobjective` is the pure
        // quadratic-penalized objective; subtract the gated old-β Φ here.
        let old_jeffreys = if !jeffreys_skippable_this_cycle {
            joint_jeffreys_subspace
                .as_ref()
                .map(|z_joint| {
                    custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                })
                .transpose()?
                .unwrap_or_default()
        } else {
            JointJeffreysValue::default()
        };
        let old_phi = old_jeffreys.phi;
        let old_objective = lastobjective - old_phi;
        // Row measure observed by the objective at β. `lastobjective` was
        // set on the previous cycle (or at function entry) under `options`;
        // see top-of-cycle capture for rationale.
        let tr_row_measure_old_objective =
            gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
        let mut accepted = false;
        let mut accepted_joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> = None;
        let mut line_search_attempts = 0usize;

        // Pure Newton must take a full step on the first cycle of an
        // exact quadratic problem (i.e. converge in one cycle when the
        // model is exact). The trust-region globalization above must not
        // truncate the very first proposal merely because the hard-coded
        // initial radius (1.0) is smaller than the natural Newton-step norm.
        //
        // There are two norms in play:
        //   * the constrained-QP / dogleg paths truncate per block against
        //     `joint_block_trust_radii`;
        //   * the exact spectral trust-region path solves one global
        //     Moré–Sorensen problem against `joint_trust_radius`.
        //
        // The old cycle-0 bump only raised the per-block radii and then set the
        // global radius to `max(block_norms)`. For a multiblock exact quadratic
        // with a diagonal metric that leaves a full Newton step like
        // `[0.8, 0.8]` inside every per-block ball (`max = 0.8`) but outside
        // the global spectral ball (`sqrt(0.8² + 0.8²) = 1.13`). Once the
        // constrained branch started populating `joint_spectrum` for the
        // Newton-decrement certificate, the line search correctly used the
        // spectral path and incorrectly clipped that exact feasible Newton
        // step to radius 1.0, preventing one-cycle KKT convergence. Bump the
        // global radius to the full metric norm while still bumping each block
        // radius to its own block norm; this keeps the first exact Newton step
        // untruncated in both globalization modes and leaves the standard
        // adaptive shrink/expand for subsequent cycles.
        if cycle == 0 && joint_step_spectral_nullity == 0 {
            let initial_global_norm =
                joint_trust_region_metric_step_norm(&delta, &joint_trust_metric_diag);
            let initial_block_norms =
                joint_trust_region_block_metric_norms(&delta, &ranges, &joint_trust_metric_diag);
            for (radius, norm) in joint_block_trust_radii.iter_mut().zip(initial_block_norms) {
                if norm.is_finite() && norm > *radius {
                    *radius = norm;
                }
            }
            let block_radius = joint_block_trust_radii
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            joint_trust_radius = if initial_global_norm.is_finite() {
                block_radius.max(initial_global_norm)
            } else {
                block_radius
            };
            if !joint_trust_radius.is_finite() || joint_trust_radius <= 0.0 {
                joint_trust_radius = 1.0;
            }
        }

        let penalty_beta = apply_joint_block_penalty(
            &ranges,
            &s_lambdas,
            &beta_joint,
            joint_mode_diagonal_ridge,
            joint_bundle,
        );
        // Stationarity RHS for the trust-region quadratic model. When the
        // Jeffreys/Firth term is armed the inner objective is `−ℓ+½βᵀSβ+Φ`, so
        // the model RHS is `∇L − Sβ + ∇Φ` — the SAME augmented RHS the Newton
        // step solves and the H_Φ-augmented `hpen_delta` below pairs with. Using
        // the bare `∇L − Sβ` here desyncs `predicted_reduction` from the
        // augmented step + the Φ-augmented `actual_reduction`, which is what
        // froze the coupled K-block line search (gam#729/#715). No-op when the
        // term is condition-gated/unavailable (∇Φ=0).
        let mut rhs = &grad_joint - &penalty_beta;
        if let Some((grad_phi, _hphi)) = head_jeffreys_term.as_ref()
            && grad_phi.len() == rhs.len()
        {
            rhs += grad_phi;
        }
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + old_objective.abs());
        // Scale the KKT residual tolerance against the natural magnitude
        // of ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. The
        // gradient and Sβ scale independently of the likelihood — at
        // large scale with |β|∞ ~ 10²–10³ and non-trivial smoothing,
        // ‖Sβ‖∞ can sit orders of magnitude above |obj| and FP noise
        // alone keeps the residual above any obj-scaled tol, so KKT is
        // never certified even when the iterate is the true optimum.
        let grad_inf = grad_joint
            .iter()
            .map(|x: &f64| x.abs())
            .fold(0.0_f64, f64::max);
        let penalty_inf = penalty_beta
            .iter()
            .map(|x: &f64| x.abs())
            .fold(0.0_f64, f64::max);
        // Name the denominator. `residual_tol` alone is not enough to READ the
        // residual: `R/residual_tol` also divides by `inner_tol`, which takes
        // two values in this code (the `1e-6` default and the derivative
        // lane's `1e-11` floor), so it is not comparable across lanes and is
        // anti-correlated with convergence across part of its range
        // (gam#2713). Every diagnostic that prints `R` here prints
        // `R/(1+scale)` beside it, which is exactly what this gate tests
        // against `inner_tol`.
        let stationarity_scale = grad_inf.max(penalty_inf);
        let residual_tol = inner_tol * (1.0 + stationarity_scale);
        last_residual_tol = residual_tol;
        let current_stationarity_residual = current_kkt_norm;
        // Local-mode certificate: first-order KKT and a small Newton
        // proposal are sufficient only when the exact penalized Hessian
        // has no resolvable negative curvature. CTN's squared SCOP shape
        // chart contains finite strict saddles where both the score and the
        // unconstrained reflected-Newton proposal are exactly zero. Calling
        // those points converged bypasses the finite-radius Moré–Sorensen
        // hard case below and hands an indefinite matrix to the outer LAML
        // determinant. The spectrum uses the same numerical-rank threshold
        // for this certificate and for the hard-case step, so a direction
        // that blocks convergence is guaranteed to be actionable.
        // Conditioning the valid local-minimum exit on additional evidence
        // of objective progress in the previous cycle would refuse to
        // recognize convergence at a starting point that already sits
        // at the optimum (e.g. balanced data with an intercept-only
        // fit, where ∇ℓ vanishes by symmetry from cycle 0 and the
        // Newton step is identically zero so the trust-region search
        // can never produce a strictly negative actual reduction).
        let has_resolvable_negative_curvature = joint_spectrum
            .as_ref()
            .is_some_and(|spectrum| spectrum.has_resolvable_negative_curvature());
        // Record THIS route's decision variables. The blockwise recorder
        // below is on a different loop: `55968a53c` instrumented only that
        // one, and the refusal then read `40 cycle(s) [no terminal
        // convergence state was recorded]` — which is how it became visible
        // that the 40 cycles were spent here, in the joint-Newton loop,
        // rather than there. Placed ahead of every exit this cycle can take
        // (the pre-line-search convergence exit and the strict-saddle
        // refusal immediately below, the returned-mode break above, and
        // running out of `inner_loop_hard_ceiling`) so whatever survives
        // describes the cycle the loop actually left on.
        terminal_convergence_state =
            Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
                cycle,
                stationarity_residual: current_stationarity_residual,
                residual_tol,
                stationarity_scale,
                step_inf,
                step_tol,
                resolvable_negative_curvature: has_resolvable_negative_curvature,
                // The refusal must carry how close this solve ever got, not
                // only where it happened to stand when a guard fired
                // (gam#2600). `min_certified_residual` is the smallest
                // residual the solve computed — the same quantity the
                // terminal WARN line reports as `best_residual_inf`, and
                // deliberately not the stall tracker's `best_residual_seen`,
                // which is written only at the post-step site and so stays
                // at its `inf` sentinel through a head-of-cycle exit (the
                // #1040 truthfulness trap). Folding in this cycle's residual
                // keeps the field finite whenever any residual exists, so a
                // cycle-0 refusal reports its own number rather than `inf`.
                best_stationarity_residual: min_certified_residual
                    .min(current_stationarity_residual),
                cycles_since_best_residual: cycles_since_residual_improved,
                termination_reason: gam_problem::JointNewtonTerminalReason::CycleBudget,
            });
        if returned_mode_curvature_pending {
            returned_mode_curvature_pending = false;
            let returned_spectrum = joint_spectrum.as_ref().ok_or_else(|| {
                "returned-mode curvature cycle did not produce the required exact joint spectrum"
                    .to_string()
            })?;
            let returned_min = returned_spectrum
                .gamma
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let returned_max = returned_spectrum
                .gamma
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            log::info!(
                "[979-MODE-HESSIAN] eig(M_true_tangent_whitened)=[{returned_min:.6e},{returned_max:.6e}] numerical_floor={:.6e} tangent_dim={}",
                returned_spectrum.numerical_floor,
                total_p,
            );
            let returned_decrement = returned_spectrum.newton_decrement();
            let returned_weak_decrement = returned_spectrum.weakly_identified_decrement();
            let returned_null_stationarity = returned_spectrum.numerical_null_stationarity_inf();
            let exact_first_order_certified = current_stationarity_residual <= residual_tol
                || (joint_proposal_at_step_floor(step_inf, step_tol)
                    && joint_newton_decrement_certifies(
                        returned_decrement,
                        returned_weak_decrement,
                        returned_null_stationarity,
                        objective_tol,
                        residual_tol,
                    ));
            if has_resolvable_negative_curvature || !exact_first_order_certified {
                log::info!(
                    "[PIRLS/joint-Newton mode certificate] tentative convergence revoked: \
                     negative_curvature={has_resolvable_negative_curvature}, \
                     residual={current_stationarity_residual:.3e}/{residual_tol:.3e} \
                     (relative_stationarity={:.3e} vs inner_tol={inner_tol:.3e}), \
                     decrement={returned_decrement:.3e}, weak={returned_weak_decrement:.3e}, \
                     null_score={returned_null_stationarity:.3e}, correction={step_inf:.3e}/{step_tol:.3e}",
                    gam_problem::relative_stationarity(
                        current_stationarity_residual,
                        stationarity_scale
                    ),
                );
                if head_jeffreys_term.is_some() {
                    jeffreys_completion_endgame = true;
                }
                converged = false;
                returned_mode_curvature_certified = false;
                last_cycle_residual_below_tol = false;
                last_cycle_obj_change_below_tol = false;
                min_certified_residual = f64::INFINITY;
                best_residual_seen = f64::INFINITY;
                cycles_since_residual_improved = 0;
                residual_descent_history.clear();
                tr_clamped_during_stall = false;
                residual_rate_history.clear();
                merit_window.clear();
                prev_fully_rejected_cycle_signature = None;
                consecutive_identical_rejected_cycles = 0;
                consecutive_all_reject_at_floor_cycles = 0;
                last_joint_math = None;
                last_kkt_refusal_report = None;
                prev_kkt_norm = None;
                geometric_tail_history.clear();
            } else {
                log::info!(
                    "[PIRLS/joint-Newton mode certificate] certified: residual={current_stationarity_residual:.3e}/{residual_tol:.3e}, decrement={returned_decrement:.3e}, weak={returned_weak_decrement:.3e}, null_score={returned_null_stationarity:.3e}, correction={step_inf:.3e}/{step_tol:.3e}"
                );
                returned_mode_curvature_certified = true;
                cached_joint_workspace = hessian_workspace_for_cycle.take();
                cycles_done = cycle;
                break;
            }
        }
        if current_stationarity_residual <= residual_tol
            && step_inf <= step_tol
            && !has_resolvable_negative_curvature
        {
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | pre-line-search converged: proposal_inf={:.3e} (tol={:.3e}) | residual={:.3e} (tol={:.3e}) | relative_stationarity={:.3e} (scale={:.3e}, inner_tol={:.3e})",
                cycle,
                step_inf,
                step_tol,
                current_stationarity_residual,
                residual_tol,
                gam_problem::relative_stationarity(
                    current_stationarity_residual,
                    stationarity_scale
                ),
                stationarity_scale,
                inner_tol,
            );
            // Pre-line-search convergence: β did not move this cycle (the
            // proposal was at the step-tolerance floor), so the cycle
            // workspace is still at the converged β and the post-loop
            // covariance/IFT assembly can reuse it instead of rebuilding the
            // full per-row kernel cache at the same β.
            cached_joint_workspace = hessian_workspace_for_cycle.take();
            cycles_done = cycle;
            finish_post_step_convergence!();
        }
        if current_stationarity_residual <= residual_tol
            && step_inf <= step_tol
            && has_resolvable_negative_curvature
        {
            log::info!(
                "[PIRLS/joint-Newton] cycle {cycle:>3} | first-order stationary strict saddle; refusing convergence and invoking the finite-radius negative-curvature hard case"
            );
        }

        // Trust-region retries preserve the objective-decrease guarantee
        // when the initial radius is too optimistic. If the Newton proposal
        // is not a descent direction for the penalized quadratic model,
        // switch once to a diagonally preconditioned gradient step and keep
        // the same exact full-objective accept/reject test.
        const JOINT_TRUST_MAX_ATTEMPTS: usize = 24;
        let mut search_delta = delta.clone();
        let search_joint_active_set: Option<Vec<usize>> = joint_active_set.clone();
        // A constrained Newton step can discover a different critical cone.
        // Residual/objective-rate samples collected on the previous active
        // face are not evidence about stationarity on the new face: the KKT
        // projection itself changes when the active rows change. Remember the
        // accepted transition so the post-step convergence machinery can
        // start its plateau/descent evidence from this face only (gam#979).
        let active_face_before_step =
            flatten_joint_active_set(&cached_active_sets, &block_constraints);
        let mut accepted_active_face_changed = false;
        let mut tried_preconditioned_descent = false;
        // Dogleg Cauchy leg (gam#826/#808). Compute the unconstrained Cauchy
        // point of the penalized (Firth-augmented) quadratic model ONCE per
        // cycle: the M-metric steepest-descent direction `p_sd = M⁻¹·rhs`
        // and its curvature `p_sd·H·p_sd` (a coupled Hessian-vector product,
        // so it must be hoisted out of the radius-shrink loop). When the
        // Newton step exceeds a block's trust radius the dogleg blends
        // toward this Cauchy leg, guaranteeing at least the Cauchy decrease
        // even when the spectral Newton step is numerically frozen at the
        // oversmoothed seed (the high-curvature log_sigma block's Newton
        // component is `O(g/λ) ≈ 5e-21`). `joint_active_set` is the
        // unconstrained joint Newton path; the constrained-QP path keeps its
        // own globalization, so the dogleg is only built (and used) when no
        // active set is in force.
        // Only the dogleg/box-truncation globalization (no spectrum) ever
        // consumes the Cauchy leg; when the exact Moré–Sorensen spectrum is
        // present the trust loop re-solves from it and `dogleg_cauchy` is dead.
        // Skipping its construction there removes one coupled Hessian-vector
        // product per cycle — an `O(n·p)` operator row-sweep on the matrix-free
        // DENSE_SPECTRAL path that produced no value (gam#1040).
        let dogleg_cauchy: Option<Array1<f64>> =
            if search_joint_active_set.is_none() && joint_spectrum.is_none() {
                let mut p_sd = Array1::<f64>::zeros(total_p);
                for (i, (r, w)) in rhs.iter().zip(joint_trust_metric_diag.iter()).enumerate() {
                    p_sd[i] = r / positive_joint_diagonal_entry(*w);
                }
                let mut h_psd = Array1::<f64>::zeros(total_p);
                let mut cauchy_penalty_scratch = Array1::<f64>::zeros(total_p);
                match apply_joint_penalized_hessian_into_with_workspace(
                    effective_hessian_source,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    &p_sd,
                    &mut h_psd,
                    &mut cauchy_penalty_scratch,
                    joint_bundle,
                ) {
                    Ok(()) => {
                        if let Some((_grad_phi, hphi)) = head_jeffreys_term.as_ref() {
                            h_psd += &hphi.dot(&p_sd);
                        }
                        let cauchy = joint_cauchy_step(&rhs, &p_sd, &h_psd);
                        if cauchy.iter().all(|v| v.is_finite()) {
                            Some(cauchy)
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                }
            } else {
                None
            };
        // SELF-CONCORDANT DAMPING (gam#979). For a family that declares
        // its penalized inner objective self-concordant, compute once per
        // cycle the damped-Newton `α = 1/(1+λ_N)` from the spectrum's
        // Newton decrement. Consumed ONLY by the α-crush rescue arm of the
        // first trust attempt (the measured barrier-overshoot pathology),
        // where `α·δ_N` replaces the radius-clamped step; every other arm,
        // every later attempt, and every non-flagged family are
        // byte-identical. `None` outside the damped phase (λ_N below the
        // quadratic-phase threshold), where plain Newton owns the endgame.
        let self_concordant_damping: Option<f64> =
            if !true_jeffreys_hessian_required && family.inner_objective_is_self_concordant() {
                joint_spectrum.as_ref().and_then(|spectrum| {
                    self_concordant_damped_step_alpha(spectrum.newton_decrement())
                })
            } else {
                None
            };
        let mut model_rejects = 0usize;
        let mut likelihood_rejects = 0usize;
        let mut objective_rejects = 0usize;
        // Feasibility-path rejections (gam#979 survival monotone cone). The two
        // constrained-path `continue`s — the `apply_joint_feasibility_limit`
        // α-crush `Err` (current iterate infeasible / no positive step) and the
        // `project_point_strictly_into_feasible_cone` `None` (degenerate /
        // empty-interior cone at this trial) — consume a trust attempt but were
        // NOT counted by any of model/likelihood/objective. On the survival
        // marginal-slope monotone-cone pathology this is the DOMINANT reject
        // path (the trial step keeps crossing the binding time-derivative cone
        // at slack≈0), so `model + likelihood + objective < MAX_ATTEMPTS`
        // ALWAYS, `all_attempts_rejected` was permanently false, and the
        // fully-rejected stall guard below NEVER armed — the inner joint-Newton
        // spun to `inner_loop_hard_ceiling` every outer ρ-evaluation (the 1322 s
        // hang; #1040). A feasibility rejection IS a "no descent the local model
        // can reconcile at this β" signal exactly like an objective rejection,
        // so counting it restores the partition invariant
        // `model + likelihood + objective + feasibility == MAX_ATTEMPTS` the
        // stall guard relies on. Off the constrained pathology this counter
        // stays 0 (those `continue`s are never taken on a feasible/unconstrained
        // arm), so every converging fit is byte-identical.
        let mut feasibility_rejects = 0usize;
        let mut first_likelihood_reject: Option<String> = None;
        // Watch the trust ratio across the attempt ladder, so a fully-rejected
        // cycle can say WHICH kind of fault it hit. See the type's doc: `rho`
        // tends to 1 under refinement for any model whose rhs is the gradient
        // of the objective the numerator measures, so a ladder that shrinks
        // two decades without `rho` moving toward 1 is a first-order
        // disagreement, and no radius repairs it.
        let mut trust_ratio_witness = TrustRatioRefinementWitness::default();
        // One ladder per cycle: the coarsest and finest ends must come from
        // ONE shrink sequence to be comparable (different cycles sit at
        // different β and different radii). What the witness has already
        // MEASURED survives across cycles; see its doc block.
        objective_resolution_witness.start_ladder();
        let resolution_before_this_ladder = objective_resolution_witness.measured();
        let refused_resolution_before_this_ladder = objective_resolution_witness
            .refused()
            .map_or(0.0, |(claim, _)| claim);
        // Snapshot every mutable input to the trust-region attempt loop
        // before it can shrink a radius. A later fully-rejected cycle is an
        // exact fixed point only if this state and the realized first trial
        // both reproduce bit-for-bit.
        let cycle_start_beta_bits = beta_joint
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let cycle_start_joint_trust_radius_bits = joint_trust_radius.to_bits();
        let cycle_start_block_trust_radius_bits = joint_block_trust_radii
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let mut first_attempt_trial_delta_bits: Option<Vec<u64>> = None;
        let mut first_attempt_trial_objective: Option<f64> = None;
        // Frozen-step line-search short-circuit (n≈3e5 marginal-slope floor
        // stall). Once the joint trust radius is pinned (the shrink rule
        // clamps every block radius at the `1e-12` floor, so a reject can no
        // longer reduce it), the Moré–Sorensen / dogleg step the next attempt
        // builds is a deterministic function of the unchanged radii and the
        // reverted β — byte-identical to this attempt, hence the same trial
        // objective. The floor-stalled cycle therefore logged
        // `JOINT_TRUST_MAX_ATTEMPTS` identical `reject_floor` lines, each a
        // redundant full-data (320k-row) line-search sweep (~0.5 s apiece),
        // every cycle until the cross-cycle stall guard fired — pure waste on
        // the dominant cost of the inner solve. Track the previous rejected
        // attempt's trial objective; when the radius is held AND the current
        // rejected trial reproduces it bit-for-bit the step is provably frozen
        // and the remaining attempts are no-ops, so stop. `frozen_floor_full_reject`
        // records that the cycle was nonetheless fully rejected, preserving the
        // `all_attempts_rejected` partition the cross-cycle stall guard relies
        // on; `first_attempt_trial_objective` is still captured on attempt 0,
        // so the byte-identical cross-cycle detector is unaffected and the
        // converged/non-converged decision is byte-identical to exhausting the
        // loop.
        let mut prev_rejected_attempt_objective: Option<f64> = None;
        let mut frozen_floor_full_reject = false;
        // Coalesce consecutive trust-region attempts whose accept/reject
        // outcome and numeric signature round to the same values, so a long
        // run of identical retries collapses into a single "attempts a..b
        // (×N)" line at flush time instead of spamming one line per try.
        let mut tr_log_sig: Option<String> = None;
        let mut tr_log_first: usize = 0;
        let mut tr_log_last: usize = 0;
        // Hoist the two full-size scratch buffers used in the predicted-
        // reduction computation outside the trust-region attempt loop.
        // The loop runs up to JOINT_TRUST_MAX_ATTEMPTS times per outer
        // Newton step, so allocating these per-attempt would add O(total_p)
        // heap traffic on every radius shrink/expand iteration.
        let mut hpen_delta = Array1::<f64>::zeros(total_p);
        let mut tr_penalty_scratch = Array1::<f64>::zeros(total_p);
        for trust_attempt in 0..JOINT_TRUST_MAX_ATTEMPTS {
            line_search_attempts = trust_attempt + 1;
            accepted_joint_workspace = None;
            // Dogleg globalization (gam#826/#808): when the unconstrained
            // Newton path is in force and a finite Cauchy leg was built,
            // construct the dogleg blend of the Cauchy and Newton points at
            // the current per-block radii. Otherwise (constrained-QP path,
            // or after the preconditioned-descent fallback replaced
            // `search_delta`) fall back to box-truncating the search step.
            let mut trial_delta;
            // gam#979: set when the constrained-QP candidate is taken
            // untruncated because the global α-crush would otherwise collapse
            // a feasible, within-trust QP step (see the gated bypass below).
            let mut qp_feasible_bypass = false;
            // Self-concordant damped first trial (gam#979): live only on
            // the cycle's FIRST attempt; a rejection falls through to the
            // byte-identical trust-region attempts below.
            let sc_first_trial_alpha = if trust_attempt == 0 {
                self_concordant_damping
            } else {
                None
            };
            let mut block_step_norms = if let Some(spectrum) = joint_spectrum.as_ref() {
                // FACE-AWARE CONSTRAINED PATH (gam#979 CTN).
                //
                // `search_delta` is authoritative when it is an exact/reduced
                // positive-curvature Newton step, a reduced spectral
                // completion, or the D-metric projected gradient selected at
                // a fully pinned face. Replacing any of these with an
                // unconstrained Moré-Sorensen step changes the critical cone.
                // That replacement used to happen whenever the QP step
                // exceeded the trust radius. On the 4,800-row Duchon CTN it
                // converted a feasible O(1e-2) proposal into an O(1e-4)
                // projected step, repeatedly changed the active face, and
                // left the KKT residual near 20--30.
                //
                // Global scaling by one alpha is the correct globalization
                // for this case. Both beta and beta + search_delta are cone-
                // feasible, so every point on their convex segment remains
                // feasible. The projected-gradient variational inequality (or
                // the positive-curvature QP optimality condition) certifies
                // `rhs' delta > 0`, so the same segment is first-order descent.
                // Use one alpha across every block (per-block clipping would
                // change direction and can leave the cone). The ordinary
                // family feasibility limiter still runs below for any
                // additional nonlinear constraint not represented by the QP.
                let constrained_search_delta_is_authoritative =
                    constrained_search_delta_owns_trust_step(
                        joint_reduced_face_kind,
                        search_joint_active_set.is_some(),
                        Some(spectrum.has_resolvable_negative_curvature()),
                    );
                // CONSTRAINED-PATH REFLECTED-QP RESCUE (gam#979 n3000 grind).
                //
                // On the constrained path `search_delta` is the *reflected*
                // active-set QP step: the QP convexifies the indefinite
                // survival-marginal-slope penalized Hessian by reflecting its
                // negative-curvature modes to `|γ|` (`symmetric_negative_
                // curvature_reflected`, line ~1490). The reflection changes
                // WHICH monotone-derivative-guard rows bind, so the QP settles
                // to a step whose active set is INCOMPLETE relative to the true
                // KKT (`active_set_incomplete`): the huge time-block stationarity
                // residual is never absorbed by the reported multipliers, and
                // taking that step and then globally α-crushing it (the
                // `apply_joint_feasibility_limit` fraction-to-boundary below)
                // collapses β to ~1e-4/cycle, grinding the whole 30-cycle budget
                // (the measured 478s Weibull-n3000 hang / 600s CI TIMEOUTs).
                //
                // The gam#979 `qp_feasible_bypass` rescue that skips the α-crush
                // lived only in the box-truncation branch below, which became
                // UNREACHABLE once the decrement diagnostic started populating
                // `joint_spectrum` on the constrained path (line ~1488) — routing
                // it into this branch. Resurrect the rescue here, but with the
                // EXACT Moré–Sorensen step from the UNREFLECTED penalized Hessian
                // (`spectrum`, decomposed from the true `lhs` at line ~1482): it
                // handles the indefiniteness rigorously (no reflection, no
                // incomplete active set), and the convex monotone cone projection
                // just below (gam#1108) restores feasibility. Because the cone is
                // convex the projected iterate is feasible, so the α-crush is
                // skipped (`qp_feasible_bypass`). GATED on the α-crush pathology
                // (α below the crush threshold) exactly as the original bypass
                // was, so every healthy constrained arm (α≈1, the binary BMS
                // score-warp monotonicity fit) takes the byte-identical
                // reflected-QP-step path unchanged.
                let constrained_alpha_would_crush = search_joint_active_set.is_some()
                    && match compute_joint_feasibility_alpha(
                        family,
                        &states,
                        &ranges,
                        &search_delta,
                    ) {
                        Ok((alpha, _)) => alpha < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
                        // Err = current iterate infeasible / no positive step;
                        // fall through to the reflected-QP step, which surfaces the
                        // same Err downstream and shrinks the radius.
                        Err(_) => false,
                    };
                if constrained_search_delta_is_authoritative {
                    // A full-space convex QP direction and a reduced-face
                    // direction both own their feasible chord. In the latter
                    // case, ambient negative curvature is inaccessible and
                    // has already been handled inside the face tangent.
                    // Scaling the whole chord to the trust radius preserves
                    // A_active·delta=0 and cone feasibility; replacing it
                    // with an ambient spectrum step destroys both invariants.
                    trial_delta = search_delta.clone();
                    let qp_norms = joint_trust_region_block_metric_norms(
                        &trial_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    );
                    let alpha_trust = qp_norms
                        .iter()
                        .zip(joint_block_trust_radii.iter())
                        .filter(|(norm, _)| norm.is_finite() && **norm > 0.0)
                        .map(|(norm, radius)| (radius / norm).min(1.0))
                        .fold(1.0_f64, f64::min);
                    if alpha_trust.is_finite() && alpha_trust < 1.0 {
                        trial_delta.mapv_inplace(|value| value * alpha_trust);
                        qp_norms.iter().map(|norm| norm * alpha_trust).collect()
                    } else {
                        qp_norms
                    }
                } else if constrained_alpha_would_crush {
                    qp_feasible_bypass = true;
                    // SELF-CONCORDANT DAMPED CRUSH REPLACEMENT (gam#979).
                    // This arm is reached exactly on the barrier-overshoot
                    // pathology: the fraction-to-boundary α the legacy path
                    // would apply is below the crush threshold, i.e. the
                    // Newton proposal steps deep past the −log h' barrier's
                    // region of model validity and the crush would gut it to
                    // ~1e-4·δ_N (the measured ~0.998×/cycle residual crawl).
                    // For a family that declares its inner objective
                    // self-concordant, the damped Newton step `α·δ_N`,
                    // α = 1/(1+λ_N), is the classical largest step with a
                    // guaranteed objective decrease — a principled,
                    // barrier-aware step length where the D-metric radius
                    // carries no barrier information. First attempt only;
                    // a rejection falls back to the byte-identical
                    // radius-clamped rescue below. The authoritative
                    // face-chord arm above is deliberately untouched: an
                    // interior-damped step there pulls every accepted
                    // iterate off the working band, wipes the cached active
                    // face, and forces a from-scratch QP each cycle
                    // (measured: warm_rows=0 on all 244 cycles).
                    if let Some(sc_alpha) = sc_first_trial_alpha {
                        trial_delta = spectrum.trust_region_step(f64::INFINITY).delta;
                        trial_delta.mapv_inplace(|value| value * sc_alpha);
                    } else {
                        trial_delta = spectrum.trust_region_step(joint_trust_radius).delta;
                    }
                    joint_trust_region_block_metric_norms(
                        &trial_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    )
                } else {
                    // Exact Moré–Sorensen trust-region step at the current radius
                    // (gam#979). The step already lies in the `D`-metric ball, so
                    // no dogleg blend or box-truncation is applied: on a shrink the
                    // direction is RE-SOLVED (bending toward the gradient), the
                    // property the dogleg/truncation lacked. Re-solving reuses the
                    // cached factorization at O(p) cost. On the constrained path the
                    // resulting (unconstrained) step is projected back onto the cone
                    // just below (gam#1108), preserving this step's fast convergence
                    // while keeping every accepted iterate feasible.
                    //
                    // If the already-computed Newton/QP step lies inside the
                    // current global trust ball, take it directly instead of asking
                    // the trust-region solver to recover the boundary solution at
                    // `r == ‖δ_N‖`. The boundary multiplier is mathematically zero
                    // in that case, but finite precision can produce a tiny positive
                    // multiplier and perturb an exact quadratic one-step solve by
                    // O(1e-6), which is large relative to the inner KKT floor. The
                    // direct step is the exact unconstrained minimizer of the local
                    // model and is still trust-region feasible.
                    let search_norm = joint_trust_region_metric_step_norm(
                        &search_delta,
                        &joint_trust_metric_diag,
                    );
                    if !spectrum.has_resolvable_negative_curvature()
                        && search_norm.is_finite()
                        && joint_trust_radius.is_finite()
                        && search_norm <= joint_trust_radius * (1.0 + 1e-12)
                    {
                        trial_delta = search_delta.clone();
                    } else {
                        trial_delta = spectrum.trust_region_step(joint_trust_radius).delta;
                    }
                    joint_trust_region_block_metric_norms(
                        &trial_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    )
                }
            } else if let Some(cauchy) = dogleg_cauchy.as_ref()
                && !tried_preconditioned_descent
            {
                trial_delta = Array1::<f64>::zeros(total_p);
                joint_dogleg_step_to_block_metric_radii(
                    &search_delta,
                    cauchy,
                    &ranges,
                    &joint_trust_metric_diag,
                    &joint_block_trust_radii,
                    &mut trial_delta,
                )
            } else {
                // Box-truncation branch — taken on the CONSTRAINED-QP path
                // (search_joint_active_set is Some, so no spectrum / dogleg).
                // `search_delta` is the active-set QP's Newton step, FEASIBLE
                // by construction.
                //
                // gam#979 GATED QP-FEASIBILITY BYPASS. The default behaviour
                // box-truncates this feasible QP step per-block (which can push
                // it off the monotone cone face) and then the global
                // fraction-to-boundary α-crush scales the whole joint step by a
                // single α; on a binding monotonicity row at slack≈0 that α
                // collapses to ~1e-4, freezing β and hanging the inner solve.
                //
                // Bypass that ONLY when the observable pathology is present —
                // never on a healthy step — so every currently-converging arm
                // (binary BMS score-warp monotonicity especially) is byte-
                // identical. The three gate conditions:
                //   (i)   the candidate came from the constrained QP
                //         (search_joint_active_set.is_some()),
                //   (ii)  the QP step is already within the joint trust region
                //         (step_norm ≤ joint_trust_radius), so truncation is
                //         not needed for globalization, AND
                //   (iii) the α the legacy path WOULD apply is below the crush
                //         threshold (it would gut the step).
                // When all hold, take the untruncated QP step and let the
                // magnitude-preserving cone projection below enforce
                // feasibility. Otherwise truncate exactly as before.
                let qp_norms = joint_trust_region_block_metric_norms(
                    &search_delta,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                let qp_step_norm = qp_norms.iter().copied().fold(0.0_f64, f64::max);
                let within_trust = qp_step_norm.is_finite()
                    && joint_trust_radius.is_finite()
                    && qp_step_norm <= joint_trust_radius;
                let alpha_would_crush = if search_joint_active_set.is_some() {
                    match compute_joint_feasibility_alpha(family, &states, &ranges, &search_delta) {
                        Ok((alpha, _)) => alpha < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
                        // Err = current iterate infeasible / no positive step;
                        // fall back to the legacy truncate + α path, which will
                        // surface the same Err and shrink the radius.
                        Err(_) => false,
                    }
                } else {
                    false
                };
                if search_joint_active_set.is_some() && alpha_would_crush {
                    // gam#979 survival n3000 grind. The constrained active-set QP
                    // returns a FEASIBLE point and the monotone cone is convex, so a
                    // step that would trip the fraction-to-boundary α-crush (α below
                    // threshold) must NOT be per-block-truncated: per-block truncation
                    // pushes the joint iterate OFF the cone face, and the α-crush then
                    // collapses it to ~1e-2, freezing β while a huge time-block
                    // gradient persists → the 30-cycle budget grind. Instead skip the
                    // α-crush (feasible by construction) and:
                    //   * within trust      → take the untruncated feasible QP step;
                    //   * exceeds the radius → scale the WHOLE joint step by a SINGLE
                    //     global scalar to the block radii, which stays feasible by
                    //     cone convexity (β and β+δ both feasible ⇒ β+αδ feasible),
                    //     unlike per-block truncation which breaks it.
                    // A feasible boundary step with ρ≈1 then lets the TR GROW the
                    // radius back, curing the collapse-and-grind after one bad
                    // far-field-nonlinear step shrank it. (Reached only on the
                    // observable pathology — constrained candidate whose α WOULD
                    // crush — so every healthy/converging constrained arm is
                    // untouched.)
                    qp_feasible_bypass = true;
                    trial_delta = search_delta.clone();
                    if within_trust {
                        qp_norms
                    } else {
                        let alpha_trust = qp_norms
                            .iter()
                            .zip(joint_block_trust_radii.iter())
                            .filter(|(norm, _)| norm.is_finite() && **norm > 0.0)
                            .map(|(norm, radius)| (radius / norm).min(1.0))
                            .fold(1.0_f64, f64::min);
                        if alpha_trust.is_finite() && alpha_trust < 1.0 {
                            trial_delta.mapv_inplace(|v| v * alpha_trust);
                            qp_norms.iter().map(|n| n * alpha_trust).collect()
                        } else {
                            qp_norms
                        }
                    }
                } else {
                    trial_delta = search_delta.clone();
                    truncate_joint_step_to_block_metric_radii(
                        &mut trial_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                        &joint_block_trust_radii,
                    )
                }
            };
            // FEASIBILITY ENFORCEMENT (gam#979 survival flex non-convergence).
            //
            // The global `apply_joint_feasibility_limit` enforces feasibility
            // by a single fraction-to-boundary scalar `α` applied to the WHOLE
            // joint step. On a binding monotonicity row at slack≈0 with
            // negative drift, `α` collapses to ~1e-4, globally crushing the
            // step so β moves ~1e-4/cycle while a huge time-block gradient
            // (|g|≈720) persists: the objective drifts down ~50/cycle but the
            // KKT residual never clears, the inner joint-Newton grinds the full
            // cycle budget, and the seed is rejected — the survival
            // marginal-slope hang.
            //
            // When the gated bypass above fired (`qp_feasible_bypass`), the
            // pathology is present (constrained-QP candidate, within trust, α
            // below the crush threshold): SKIP the α-crush and let the
            // magnitude-preserving cone projection below enforce feasibility,
            // which keeps the unconstrained step components and only corrects
            // the binding directions. Off the pathology the α-crush runs
            // exactly as before — a no-op when `α = 1` (healthy), the legacy
            // scaling when `α ∈ (0, 1)` — so every converging arm is
            // byte-identical.
            //
            // The alpha-crush refusal carries the ONLY statement of which block
            // refused and why -- which constraint row, and whether the current
            // iterate is infeasible or merely sits on a face with the ray
            // pointing out of it. Discarding it with `.is_err()` made a whole
            // class of stalls unattributable: the survival location-scale
            // linkwiggle fit (gam#2695) spends 379 rejections here, 379 of them
            // from one block, and the refusal message printed none of that.
            //
            // `α = 0` IS NO LONGER A REFUSAL (gam#2719). It says a constraint
            // row is active at β and the ray points out of it — a statement
            // about the FACE, not about the step length. Answering it with a
            // radius shrink cannot converge: `slack / -drift` is invariant
            // under `δ ↦ cδ` once its numerator is zero, so the ladder spent
            // all JOINT_TRUST_MAX_ATTEMPTS re-deriving the same zero, which is
            // exactly why the measured count is 24 per cycle. The correct
            // mechanism is the cone projection immediately below — it keeps the
            // step's magnitude in every direction the face does not block —
            // and it is available precisely when the blocking block's cone is
            // REPRESENTED in `joint_constraints`. When it is not, the family
            // declared a barrier it cannot express as linear rows, no
            // projection exists, and the pre-#2719 shrink-and-retry is still
            // the only tool.
            let alpha_crush_outcome = if qp_feasible_bypass {
                Ok(JointFeasibilityLimit::Unlimited)
            } else {
                apply_joint_feasibility_limit(family, &states, &ranges, &mut trial_delta)
            };
            let joint_feasibility_limit = match alpha_crush_outcome {
                Ok(limit) => limit,
                Err(alpha_crush_reason) => {
                    log::info!(
                        "[PIRLS/joint-Newton feasibility] cycle {} attempt {} rejected at radius {:.6e} (qp_feasible_bypass={}): {}",
                        cycle,
                        trust_attempt,
                        joint_trust_radius,
                        qp_feasible_bypass,
                        alpha_crush_reason
                    );
                    feasibility_rejects += 1;
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                    continue;
                }
            };
            if let JointFeasibilityLimit::BlockedByActiveFace { block } = joint_feasibility_limit {
                let face_is_projectable = block
                    .and_then(|idx| block_constraints.get(idx))
                    .is_some_and(Option::is_some)
                    && joint_constraints.is_some();
                if !face_is_projectable {
                    log::info!(
                        "[PIRLS/joint-Newton feasibility] cycle {} attempt {} rejected at radius {:.6e}: \
                         block {:?} is blocked by an active face it does not represent as linear \
                         constraints, so no projection can restore the step",
                        cycle,
                        trust_attempt,
                        joint_trust_radius,
                        block,
                    );
                    feasibility_rejects += 1;
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                    continue;
                }
                log::debug!(
                    "[PIRLS/joint-Newton feasibility] cycle {} attempt {}: block {:?} is blocked by \
                     an active face; routing the step to the cone projection instead of scaling it \
                     to zero",
                    cycle,
                    trust_attempt,
                    block,
                );
            }
            // CONSTRAINED-PATH FEASIBILITY PROJECTION (gam#1108 / gam#979). The
            // trust-region trial step (Moré–Sorensen / dogleg / box-trunc) is
            // taken in the UNCONSTRAINED D-metric ball, so the step can cross
            // the monotone time-derivative cone `Aβ ≥ b`. The next cycle's
            // `check_linear_feasibility` gate would then reject the accepted
            // iterate — the interval-censored survival warm-start abort.
            // Project the trial iterate back onto the cone with the exact
            // identity-Hessian active-set projection, preserving the trust
            // step's fast convergence while guaranteeing every accepted iterate
            // is feasible. This is the feasibility mechanism for the gated
            // QP-bypass case (gam#979, where the α-crush is skipped) and the
            // safety net for any truncation-induced infeasibility on the
            // constrained path. No-op when the joint design is unconstrained or
            // the trial is already feasible (the common case — including a
            // bypassed QP step, which is feasible by construction);
            // `block_step_norms` is recomputed from the projected step just
            // below so the trust-radius bookkeeping stays consistent.
            let mut projection_moved_the_trial = false;
            if let Some(constraints) = joint_constraints.as_ref() {
                let trial_beta = &beta_joint + &trial_delta;
                if check_linear_feasibility(&trial_beta, constraints).is_err() {
                    // Project ONTO the cone in the trust metric, landing on the
                    // rows that bind, not strictly inside it. The seed-style
                    // projection retreated `ACTIVE_SET_INTERIOR_SEED_MARGIN` off
                    // every face, so the row that clipped this step was never
                    // tight at the accepted point, the accepted face stayed
                    // empty, `certified_reduced_face_candidate` never ran, and
                    // the next ambient trust step was clipped on the same row —
                    // the survival location-scale 1569 deadlock (gam#2695,
                    // gam#2714): scaled slack shuttling between 1e-8 and 1e-6 on
                    // one time-block row for sixty cycles with accepted steps of
                    // 1e-22 while the QP already listed that row as active. The
                    // face the projection lands on is exactly the working face
                    // the next cycle's reduced-face Newton solves on.
                    let warm_face =
                        flatten_joint_active_set(&cached_active_sets, &block_constraints);
                    match gam_solve::active_set::project_point_onto_constraint_set_in_metric(
                        &trial_beta,
                        Some(&joint_trust_metric_diag),
                        constraints,
                        warm_face.as_deref(),
                    ) {
                        Ok((projected, _binding_rows)) => {
                            trial_delta = &projected - &beta_joint;
                            projection_moved_the_trial = true;
                        }
                        Err(_) => {
                            // Projection found no feasible point (degenerate /
                            // empty cone at this trial). Since the global α-crush is gated off on
                            // the constrained path, preserve the old safety
                            // net here: shrink the active block trust radii and
                            // retry, exactly as the α-crush `Err` branch did
                            // (gam#979). Without this an infeasible trial would
                            // reach the next cycle's `check_linear_feasibility`
                            // QP gate and hard-error.
                            log::info!(
                                "[PIRLS/joint-Newton feasibility] cycle {} attempt {} rejected at radius {:.6e}: cone projection found no feasible point",
                                cycle,
                                trust_attempt,
                                joint_trust_radius
                            );
                            feasibility_rejects += 1;
                            joint_trust_radius = shrink_active_joint_block_trust_radii(
                                &mut joint_block_trust_radii,
                                &block_step_norms,
                                0.25,
                            );
                            continue;
                        }
                    }
                }
            }
            // PROJECTION-GUTTED TRUST STEP (gam#2621). The step above was sized
            // in the UNCONSTRAINED `D`-metric ball; when the cone projection had
            // to move it, the part of it that pointed out of the cone is gone, and
            // what survives is no longer the step the trust region sized. Measured
            // on the gaussian location-scale fixture: the Moré–Sorensen step sits
            // exactly on the boundary (`‖δ‖_D = 3.926429e-1 = radius`, to every
            // digit) and the projection returns `‖δ‖_D = 7.170632e-2` — 82% of the
            // norm removed. Two things then went wrong at once. The controller
            // recomputes `block_step_norms` from the PROJECTED step just below, so
            // `step_hit_trust_boundary` compared `5.36e-2` against `3.926e-1` and
            // was false for every block: the grow branch never fired and all three
            // radii stayed byte-identical for 40 consecutive cycles while ρ ≈ 0.95
            // said the model was excellent. And the surviving step bought no
            // stationarity — the residual ROSE from `4.011e1` to `6.207e1` over
            // those 40 cycles while the objective fell monotonically — so the solve
            // exited on the residual-stall guard at cycle 39 of a 1200-cycle
            // budget. A projected step cannot grow the radius that produced it, so
            // the crawl is a fixed point of the controller, not a budget miss.
            //
            // `search_delta` is the active-set QP's chord: `candidate_beta −
            // beta_joint` with BOTH endpoints in the convex cone, so every point on
            // it is feasible and it needs no projection. It is available here and
            // was passed over only because `constrained_search_delta_owns_trust_step`
            // requires the AMBIENT spectrum to be curvature-free, which it is not on
            // this fixture (`γ_min = −1.457e-1`, two modes, fourteen orders above
            // the numerical floor) — even though that ambient negative curvature can
            // be unreachable inside the cone, exactly as the face-authoritative arm
            // above notes of its own case.
            //
            // So do not choose between them by a threshold on how much the
            // projection took: rank them on the quantity the trust region already
            // judges every step by, the predicted decrease of the same true
            // penalized quadratic model that `predicted_reduction` uses below. The
            // winner is by construction no worse than either candidate alone, there
            // is no new constant, and the negative-curvature escape is kept whenever
            // the projected Moré–Sorensen step really is the better step on the
            // model. Scaling the chord uses ONE global scalar to the block radii,
            // never per-block clipping: `β` and `β + search_delta` are both
            // feasible, so cone convexity keeps `β + α·search_delta` feasible, while
            // per-block clipping would leave the face. The two extra Hessian
            // applications are paid only on an attempt where the projection actually
            // moved the step.
            if projection_moved_the_trial && search_joint_active_set.is_some() {
                let mut chord = search_delta.clone();
                let chord_norms = joint_trust_region_block_metric_norms(
                    &chord,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                let chord_alpha = chord_norms
                    .iter()
                    .zip(joint_block_trust_radii.iter())
                    .filter(|(norm, _)| norm.is_finite() && **norm > 0.0)
                    .map(|(norm, radius)| (radius / norm).min(1.0))
                    .fold(1.0_f64, f64::min);
                if chord_alpha.is_finite() && chord_alpha > 0.0 {
                    if chord_alpha < 1.0 {
                        chord.mapv_inplace(|value| value * chord_alpha);
                    }
                    let candidate_model = JointTrustRegionModel {
                        source: effective_hessian_source,
                        ranges: &ranges,
                        s_lambdas: &s_lambdas,
                        diagonal_ridge: joint_mode_diagonal_ridge,
                        joint_bundle,
                        jeffreys_curvature: head_jeffreys_curvature.as_ref(),
                    };
                    let projected_gain = candidate_model.predicted_reduction_at(
                        &rhs,
                        &trial_delta,
                        &mut hpen_delta,
                        &mut tr_penalty_scratch,
                    );
                    let chord_gain = candidate_model.predicted_reduction_at(
                        &rhs,
                        &chord,
                        &mut hpen_delta,
                        &mut tr_penalty_scratch,
                    );
                    if let Some(chord_gain) = chord_gain
                        && chord_gain > projected_gain.unwrap_or(f64::NEG_INFINITY)
                    {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempt={}] gam#2621 cone \
                             projection removed the trust step's descent: projected model gain \
                             {:?} vs feasible QP chord {:.3e} (chord α={:.3e}); taking the \
                             chord, which needs no projection and can still reach its trust \
                             boundary",
                            cycle,
                            trust_attempt,
                            projected_gain,
                            chord_gain,
                            chord_alpha,
                        );
                        trial_delta = chord;
                    }
                }
            }
            if trust_attempt == 0 {
                first_attempt_trial_delta_bits =
                    Some(trial_delta.iter().map(|value| value.to_bits()).collect());
            }
            block_step_norms = joint_trust_region_block_metric_norms(
                &trial_delta,
                &ranges,
                &joint_trust_metric_diag,
            );
            // THE JOINT RADIUS IS JUDGED IN THE JOINT NORM (gam#2612).
            //
            // Two trust constraints act on this step: the D-metric ball on the
            // WHOLE step, which is what `spectrum.trust_region_step(
            // joint_trust_radius)` solves against, and one box per coefficient
            // block. Their norms are related by `‖δ‖² = Σ_b ‖δ_b‖²`, so a step
            // sitting EXACTLY on the joint sphere has `max_b ‖δ_b‖ = ‖δ‖/√K`
            // whenever `K` blocks carry comparable mass — `0.707` of the
            // radius for the two coefficient blocks of a three-class softmax.
            //
            // Handing that per-block maximum to the controller alongside the
            // JOINT radius therefore reports every boundary step as interior,
            // `hit_boundary` is false for the whole solve, and the region can
            // only ever shrink. Measured on this issue's penguins witness
            // before the repair: of 4427 accepted attempts at a held radius,
            // ZERO reached `0.99 · r` and 1295 sat in `[0.70, 0.99)` — the
            // `1/√2` band — with 1149 of those carrying a Newton proposal at
            // least `1.5×` the step actually taken (424 of them `≥ 10×`). The
            // fit then dies with `|prop|∞ = 7.686e-5` against an accepted
            // `|δ|∞ = 5.270e-7` and the residual crawling at `0.9932×/cycle`.
            //
            // So the controller gets the length measured in ITS radius's own
            // norm. The per-block boxes keep their own pairing below.
            let joint_step_norm =
                joint_trust_region_metric_step_norm(&trial_delta, &joint_trust_metric_diag);
            let step_norm = joint_step_norm;
            let trial_step_inf = trial_delta
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            // Which constraint actually bound this step: the joint ball, or
            // some block's box. Either one makes the step a boundary step, and
            // each is asked in its own norm.
            let joint_ball_bound_the_step =
                joint_block_step_hit_trust_boundary(joint_step_norm, joint_trust_radius);
            let step_hit_trust_boundary = joint_ball_bound_the_step
                || block_step_norms.iter().zip(&joint_block_trust_radii).any(
                    |(step_norm, radius)| joint_block_step_hit_trust_boundary(*step_norm, *radius),
                );
            // Predicted reduction must use the TRUE penalized Hessian
            // (the one that appears in `f(β) = -ℓ + ½βᵀSβ + ½·joint_mode_diagonal_ridge·‖β‖²`),
            // NOT the SPD-stabilized version. The stabilizing shift
            // in `joint_solver_diagonal_ridge` is purely a solver-side
            // tool to make the Newton system invertible when H_NLL
            // has negative eigenvalues; it is not part of the true
            // objective the trial-likelihood evaluator computes.
            //
            // If we use `joint_solver_diagonal_ridge` here, then for
            // any Newton step lying in null(H_true) (e.g. the
            // marginal-block cancellation direction in the saturated
            // probit regime — see
            // `marginal_block_hessian_cancels_in_saturated_regime`),
            // predicted = ½·rhs·δ while actual = rhs·δ, giving ρ = 2
            // exactly. The trust-region loop then accepts the step
            // (ρ > 0.75 expands the radius), and the same regime
            // repeats every cycle — exactly the large-scale-saturated
            // failure trace. Pinned by
            // `ridge_stabilization_gap_produces_exact_rho_two_in_null_direction`.
            //
            // `hpen_delta` and `tr_penalty_scratch` are hoisted outside
            // this loop; the workspace variant reuses them without
            // allocating per attempt.
            hpen_delta.fill(0.0);
            if apply_joint_penalized_hessian_into_with_workspace(
                effective_hessian_source,
                &ranges,
                &s_lambdas,
                joint_mode_diagonal_ridge,
                &trial_delta,
                &mut hpen_delta,
                &mut tr_penalty_scratch,
                joint_bundle,
            )
            .is_err()
            {
                break;
            }
            // JEFFREYS/FIRTH CURVATURE IN THE TRUST-REGION MODEL (gam#729/#715).
            // When the Jeffreys term is armed, the inner objective the merit
            // (`trialobjective = −ℓ + ½βᵀSβ + Φ`) measures and the Newton step
            // (`(H+Sλ+H_Φ)δ = ∇L−Sβ+∇Φ`) target both include the Firth term, so
            // the trust-region quadratic model's curvature MUST include `H_Φδ`
            // too. Omitting it (bare `(H+Sλ)δ`) makes `predicted_reduction`
            // inconsistent with the H_Φ-augmented `rhs` and the Φ-augmented
            // `actual_reduction`: for a coupled K-block family near the Firth
            // optimum (residual floored at ‖∇Φ‖) the resulting trust_ratio is
            // wrong, the line search rejects the genuine descent step (accepts
            // ~0), and β freezes with the residual stalled at a constant ≫ tol
            // — the unbounded-cycle non-convergence the inner solve exhibits on
            // the Dirichlet/multinomial fits. Adding `H_Φδ` makes the model
            // curvature match the augmented system the step solves and the
            // merit the accept test uses, so the step is accepted and the
            // residual descends. No-op when the term is condition-gated (∇Φ=0,
            // H_Φ=0) or unavailable.
            if let Some(curvature) = head_jeffreys_curvature.as_ref() {
                let jeffreys_delta = curvature.dot(&trial_delta);
                hpen_delta += &jeffreys_delta;
            }
            let predicted_reduction =
                joint_quadratic_predicted_reduction(&rhs, &hpen_delta, &trial_delta);
            let linearized_next_kkt_inf = hpen_delta
                .iter()
                .zip(rhs.iter())
                .map(|(hpen, rhs)| (hpen - rhs).abs())
                .fold(0.0_f64, f64::max);
            // Reject only non-descent directions on the quadratic model.
            // A small-but-positive predicted reduction is what Newton
            // *should* produce near the optimum of a large-magnitude
            // objective: ½δᵀHδ scales with curvature×step², so it can be
            // far below the (relative) objective_tol = inner_tol·(1+|obj|)
            // while still being a correct Newton step. Trust-region ρ
            // shrink/expand handles small-but-valid Newton steps; the
            // preconditioned branch below is only for model-invalid
            // directions, and preserves linear constraints when present.
            //
            // NEAR-FLOOR CARVE-OUT (gam#787 binary matern centers=12). When
            // the Newton proposal is already at the step-tolerance floor —
            // `step_inf ≤ 4·step_tol`, the same round-off band the cert path
            // uses — the iterate is doing KKT polishing on a flat objective,
            // not global descent: there `predicted_reduction = rhs·δ − ½δᵀHδ`
            // is two near-equal O(step²) quantities and its SIGN is round-off
            // noise (a true Newton step gives +½δᵀHδ but the damped/range-
            // restricted spectral solve leaves rhs·δ a hair below ½δᵀHδ). The
            // `predicted_reduction ≤ 0` branch then mistook this for a model-
            // invalid direction and substituted `joint_preconditioned_descent_delta`,
            // a step sized for OBJECTIVE descent (diagonal-preconditioned
            // gradient, O(900×) larger than the polishing proposal). That step
            // bought a round-off-level objective gain but catapulted the KKT
            // residual off a near-converged iterate (‖∇L−Sβ‖ 1.7e-4 → 4.7e-1),
            // which then never recovered — every later cycle re-triggered the
            // same substitution (proposal stays pred≤0), pinning the residual
            // far above tol until the cycle budget exhausted → seed rejected →
            // hard raise. At the step floor we instead take the tiny proposal
            // as-is and let the trust-region noise-floor guard accept it at
            // ρ=1 (it neither helps nor hurts the objective beyond round-off),
            // so the inner keeps polishing the KKT residual to tol.
            let proposal_at_step_floor = joint_proposal_at_step_floor(step_inf, step_tol);
            if (!predicted_reduction.is_finite() || predicted_reduction <= 0.0)
                && !proposal_at_step_floor
            {
                model_rejects += 1;
                // CONSTRAINED-PATH GUARD (#1108). The preconditioned-descent
                // substitution replaces `search_delta` with an UNCONSTRAINED
                // diagonally-preconditioned gradient step (`δ = M⁻¹·rhs`). That
                // direction respects neither the active set nor the linear
                // inequality cone `Aβ ≥ b`, and nothing downstream re-projects
                // it: a constrained family that maintains feasibility purely
                // through the QP (e.g. `LatentSurvivalFamily`, whose
                // `max_feasible_step_size` is `None` and whose
                // `post_update_block_beta` is the identity) has no barrier clip
                // in `apply_joint_feasibility_limit` to pull the gradient step
                // back onto the monotone time-derivative cone. The trial β then
                // leaves the cone, the objective-descent test ACCEPTS it (the
                // gradient step does lower the unconstrained merit), and the
                // NEXT cycle's `check_linear_feasibility` rejects the accepted
                // iterate as an "infeasible iterate" (raw `Aβ−b` violation
                // ~5.5e-3) — aborting the whole interval-censored warm start.
                // The QP's `search_delta` is a feasible-to-feasible chord
                // (`candidate_beta − beta_joint`, both endpoints in the convex
                // cone), so box-truncating it to a SMALLER trust radius keeps
                // every sub-step feasible. On the constrained path we therefore
                // never swap in the unconstrained descent direction; we only
                // shrink the radius and re-truncate the constrained chord. The
                // comment on the preconditioned branch already promised it
                // "preserves linear constraints when present" — this makes the
                // implementation honor that contract.
                let constrained_path_active = search_joint_active_set.is_some();
                if !tried_preconditioned_descent && !constrained_path_active {
                    match joint_preconditioned_descent_delta(
                        effective_hessian_source,
                        &ranges,
                        &s_lambdas,
                        joint_solver_diagonal_ridge,
                        &rhs,
                        joint_bundle,
                    ) {
                        Ok(descent_delta) => {
                            search_delta = descent_delta;
                        }
                        Err(_) => {
                            joint_trust_radius = shrink_active_joint_block_trust_radii(
                                &mut joint_block_trust_radii,
                                &block_step_norms,
                                0.25,
                            );
                        }
                    }
                    tried_preconditioned_descent = true;
                } else {
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                }
                continue;
            }

            for b in 0..specs.len() {
                let (start, end) = ranges[b];
                let mut trial_beta = old_beta[b].clone();
                trial_beta += &trial_delta.slice(ndarray::s![start..end]);
                let projected =
                    family.post_update_block_beta(&states, b, &specs[b], trial_beta.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    &specs[b],
                    &trial_beta,
                    &projected,
                    block_constraints[b].as_ref(),
                )?;
                states[b].beta.assign(&projected);
            }
            refresh_all_block_etas(family, specs, &mut states)?;
            let mut trial_penalty = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            // Jeffreys objective contribution at the trial point keeps the
            // accept/reject objective consistent with the Jeffreys-modified
            // Newton step. `states` already holds the trial coefficients
            // (assigned + eta-refreshed above). No-op when the Jeffreys term
            // is unavailable or condition-gated to zero. When the cheap pre-
            // check certified this cycle well-conditioned, the step used H_Φ=0
            // / ∇Φ=0, so the consistent accept/reject objective also uses Φ=0:
            // skipping here keeps value and step on the SAME objective (the
            // value/step consistency the term exists to enforce) and avoids the
            // dense H/eigh at the trial point. The 8× conditioning margin makes
            // a single damped Newton step incapable of crossing the gate.
            // SUBTRACT Φ: the inner NLL objective is `−ℓ + ½βᵀSβ − Φ` (Firth
            // adds ½log|I| to the log-likelihood). Must match the cycle-0
            // baseline, the Newton step, and the KKT residual — INCLUDING the
            // `jeffreys_skippable_this_cycle` gate, so that on a well-conditioned
            // cycle the trial, the step (H_Φ=0/∇Φ=0), and the residual all sit
            // on the SAME Φ=0 objective (gam#729/#715 sign fix; the baseline and
            // post-accept folds carry the matching skippable gate).
            // A trial point where the family cannot form its Jeffreys
            // information has no objective to compare; that is a refusal of
            // the trial (folded into `trial_ll_or_refusal` below, ahead of the
            // likelihood sweep it would only waste), never a different Φ.
            let trial_jeffreys: Result<JointJeffreysValue, CustomFamilyError> =
                if !jeffreys_skippable_this_cycle
                    && let Some(z_joint) = joint_jeffreys_subspace.as_ref()
                {
                    custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                } else {
                    Ok(JointJeffreysValue::default())
                };
            let (trial_jeffreys_phi, trial_jeffreys_roundoff) = match &trial_jeffreys {
                Ok(value) => (value.phi, value.roundoff),
                Err(_) => (0.0, 0.0),
            };
            trial_penalty -= trial_jeffreys_phi;
            // Cheap-LL line-search path: rejected backtracking attempts
            // discard the exact-Newton workspace they build, so we evaluate
            // just the scalar full-data log-likelihood for the accept/reject
            // decision and only build the full state once the step is
            // accepted (via the gradient reload below).
            //
            // EARLY-EXIT THRESHOLD MUST BOUND THE NLL, NOT THE FULL OBJECTIVE
            // (was a stall — gam#787/#785, duchon centers≥20). The family's
            // `bernoulli_margslope_line_search_ll_with_early_exit` short-
            // circuits the row sweep when the accumulated `-Σ wᵢ log CDF` (the
            // NLL ALONE — no penalty, no Jeffreys Φ) exceeds the threshold; its
            // monotone-lower-bound proof is valid only for the NLL term. But the
            // accept test is on the FULL augmented objective
            // `F = -ℓ + ½βᵀSβ + Φ_trial`, accepted iff `F ≤ old_objective + slack`,
            // i.e. iff `-ℓ_trial ≤ old_objective + slack − penalty_trial`. Passing
            // the full `old_objective` as the NLL threshold therefore over-rejects
            // by exactly `penalty_trial`: where the trial penalty is NEGATIVE
            // (the Jeffreys term subtracts Φ, and `½βᵀSβ` can be net-negative
            // under the reparam) the NLL threshold sits BELOW the true accept
            // bound, so the early exit kills net-descent steps the trust region
            // would accept — every backtracking attempt false-rejects, the radius
            // collapses, and the inner exits non-converged at cycle ~2 (seed
            // rejected pre-solver → hard raise, β pinned). Subtract the trial
            // penalty so the threshold is the NLL the trial must beat.
            // What ONE evaluation at the incumbent and at this trial sums over,
            // and the rounding the Jeffreys log-determinant carries at each
            // (gam#2748, gam#2718). Built once, before the line search, because
            // TWO decisions read it: the early exit below and the resolution
            // witness after the trial objective is known. The objective
            // magnitudes cover the likelihood accumulation; the penalty's
            // cancellation scale is carried separately because it is precisely
            // the term whose accumulation exceeds its value; the log-determinant
            // is not a sum at all and carries its own certified bound.
            let trial_beta_l1_norm: f64 = states
                .iter()
                .flat_map(|state| state.beta.iter())
                .map(|value| value.abs())
                .sum();
            let penalty_accumulation_scale = penalty_entry_magnitude
                * (old_beta_l1_norm * old_beta_l1_norm + trial_beta_l1_norm * trial_beta_l1_norm);
            let pre_trial_accumulation = ObjectiveAccumulation {
                summed_terms: total_joint_n,
                magnitude: 2.0 * old_objective.abs() + penalty_accumulation_scale,
                logdet_roundoff: old_jeffreys.roundoff + trial_jeffreys_roundoff,
            };
            // The early exit is a CERTIFICATE that the accept test below would
            // refuse this trial, so its slack must be one that no admissible
            // reading of the objective's rounding could overturn: the larger of
            // the accept test's own slack at the incumbent and the ceiling on
            // what one evaluation can round by. A literal `1e-10` here sat three
            // decades below what a Firth objective at the `1e-10·λ_max` floor
            // carries; on the bms 2718 endgame it refused 23 of 24 attempts per
            // cycle before the ratio test or the witness ever saw them, so the
            // witness had no ladder to measure and the radius ratcheted to its
            // floor (gam#2718).
            let early_exit_slack = joint_objective_roundoff_slack(
                old_objective,
                old_objective,
                objective_resolution_witness.measured(),
            )
            .max(pre_trial_accumulation.roundoff_ceiling());
            let line_search_options = coefficient_line_search_options(
                options,
                old_objective + early_exit_slack - trial_penalty,
            );
            // Accept-on-first-attempt fast path (gam#979 `gradient_reload`
            // cost). On the FIRST trust-region attempt of a cycle the step
            // is the undamped (radius-bumped) Newton proposal, which on the
            // common ρ≈1 `hold_inside` large-scale pattern accepts outright.
            // The cheap scalar sweep below would then run a full row stream
            // and immediately discard it, leaving `gradient_reload` to
            // re-stream every row at the SAME β to build the gradient
            // workspace — the ~5s redundant second pass per accepted cycle.
            //
            // Instead, when a workspace gradient source is available, build
            // the joint-Newton workspace ONCE at the trial β and read its
            // `joint_log_likelihood_evaluation()` (the same `Σ wᵢ log Φ` the
            // cheap sweep computes, on the same row measure — both derive
            // from `options`). The materialised per-row cache is threaded
            // forward as `accepted_joint_workspace`, so on accept the reload
            // short-circuits through `joint_gradient_evaluation()` with NO
            // second stream — collapsing the accepted cycle to one row pass.
            //
            // Only the first attempt takes this path: it is the only one
            // expected to accept, so a rejected first attempt pays a single
            // full (non-early-exited) sweep — paid back many-fold on the
            // dominant accept-on-first-attempt cycle. Later backtracking
            // attempts keep the cheap early-exiting sweep (they are expected
            // to reject and the workspace they would build is discarded).
            // Capability absence is the only reason to use the scalar path.
            // Once the family advertises fused likelihood evidence, an error
            // or a missing advertised value is a broken workspace contract,
            // not an infeasible trial: silently replaying the scalar family
            // path could change the row measure and let structurally invalid
            // Hessian/gradient evidence participate in the trust ratio.
            // ── one rejection path for both evaluators (gam#2600) ──────────
            //
            // A likelihood evaluation that fails AT A TRIAL POINT is
            // information about that point, not about the problem: the
            // family refuses because the proposed β is outside its domain
            // (`MonotonicityViolated` when `h' <= 0` on the transformation
            // arm), which is precisely what the trust region exists to back
            // off from. Attempts 1.. already treat it that way — restore β,
            // refresh η, shrink the radius, retry — while attempt 0 went
            // through `?` and ABORTED THE WHOLE INNER SOLVE, because it is
            // the only attempt that can take the fused workspace path
            // (`trust_attempt == 0 && joint_workspace_requested`). Same
            // condition, two verdicts, decided by which attempt index
            // happened to hit it.
            //
            // Today that asymmetry is mostly latent: `TRANSFORMATION_
            // MONOTONICITY_EPS` is added into `h'`, so `h' >= 1e-8` always
            // and the domain refusal cannot fire. It stops being latent the
            // moment that epsilon is removed — which is the actual gam#2600
            // fix, since capping the monotonicity barrier at `log(1e-8)` is
            // what makes collapsing the transformation affordable at
            // `lambda ~ 1370` (measured: `Δobj = +9.987e5` at `ρ = 1.000`).
            // With an uncapped barrier the solver MUST be able to propose a
            // point at or past the cone vertex and be told no. Aborting
            // there would turn a degenerate-but-finite optimum into a hard
            // error on the first attempt of a cycle, so this is a
            // prerequisite for that change rather than a tidy-up.
            //
            // Route both evaluators into one `Result` so there is a single
            // rejection block. Byte-identical while no evaluator errs.
            let trial_ll_or_refusal = match trial_jeffreys.and_then(|_| {
                fused_first_attempt_log_likelihood(
                    family,
                    options,
                    specs,
                    &states,
                    trust_attempt,
                    joint_workspace_requested,
                )
            }) {
                Ok(Some((value, workspace))) => {
                    accepted_joint_workspace = Some(workspace);
                    Ok(value)
                }
                Ok(None) => {
                    match joint_line_search_log_likelihood(family, &line_search_options, &states) {
                        Ok((value, workspace)) => {
                            accepted_joint_workspace = workspace;
                            Ok(value)
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            };
            let trial_ll = match trial_ll_or_refusal {
                Ok(value) => value,
                Err(e) => {
                    likelihood_rejects += 1;
                    if first_likelihood_reject.is_none() {
                        // `first_likelihood_reject` is a log field, so the
                        // render is the point of it (gam#2689).
                        first_likelihood_reject = Some(e.to_string());
                    }
                    for (b, old) in old_beta.iter().enumerate() {
                        states[b].beta.assign(old);
                    }
                    refresh_all_block_etas(family, specs, &mut states)?;
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                    continue;
                }
            };
            let trialobjective = -trial_ll + trial_penalty;
            if trust_attempt == 0 && trialobjective.is_finite() {
                // Deterministic fixed-point signature (see declaration). The
                // first attempt evaluates at the unshrunk pre-cycle β, so this
                // value identifies the iterate exactly.
                first_attempt_trial_objective = Some(trialobjective);
            }
            // Row measure observed by the trial objective at β + δ. The
            // line-search helper above runs under `coefficient_line_search_options`,
            // which now preserves `outer_score_subsample` and disables
            // any further auto-install; if either contract is broken the
            // id will diverge from `tr_row_measure_top` and we Err below.
            let tr_row_measure_trial =
                gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            // Hard invariant: the trust-region ratio numerator (objective
            // at β minus trial at β+δ) and denominator (rhs·δ − ½δᵀH δ)
            // MUST share a row measure with the Hessian/gradient build.
            // Bubble out via `Err` rather than panic; this function
            // already returns `Result<_, CustomFamilyError>`.
            let top_id = tr_row_measure_top.id;
            if tr_row_measure_hessian.id != top_id {
                return Err(CustomFamilyError::trial_point(format!(
                    "trust-region row-measure invariant violated: \
                     Hessian id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                     (cycle {}); the joint Hessian was built against a different \
                     row mask than the trust-region globalization captured at the \
                     top of the cycle. ρ would compare ½δᵀHδ on one measure to \
                     F(β)−F(β+δ) on another.",
                    tr_row_measure_hessian.id, top_id, cycle
                )));
            }
            if tr_row_measure_gradient.id != top_id {
                return Err(CustomFamilyError::trial_point(format!(
                    "trust-region row-measure invariant violated: \
                     gradient id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                     (cycle {}); `cached_joint_gradient` was loaded against a \
                     different row mask than the trust-region globalization \
                     captured at the top of the cycle. rhs·δ in the predicted \
                     reduction would not match the rest of the ρ inputs.",
                    tr_row_measure_gradient.id, top_id, cycle
                )));
            }
            if tr_row_measure_old_objective.id != top_id {
                return Err(CustomFamilyError::trial_point(format!(
                    "trust-region row-measure invariant violated: \
                     objective-at-β id 0x{:016x} differs from top-of-cycle id \
                     0x{:016x} (cycle {}); `lastobjective` was computed against \
                     a different row mask than the trust-region globalization \
                     captured at the top of the cycle.",
                    tr_row_measure_old_objective.id, top_id, cycle
                )));
            }
            if tr_row_measure_trial.id != top_id {
                return Err(CustomFamilyError::trial_point(format!(
                    "trust-region row-measure invariant violated: \
                     trial-objective id 0x{:016x} differs from top-of-cycle id \
                     0x{:016x} (cycle {}, attempt {}); the line-search trial \
                     likelihood evaluated against a different row mask than the \
                     Hessian/gradient/old-objective build. Cf. \
                     `coefficient_line_search_options` and \
                     `install_auto_outer_subsample_options`.",
                    tr_row_measure_trial.id, top_id, cycle, trust_attempt
                )));
            }
            let actual_reduction = old_objective - trialobjective;
            // Read this attempt for what it says about the OBJECTIVE's own
            // resolution before asking the controller what it says about the
            // REGION (gam#2612): `actual − predicted` is `O(‖δ‖³)` for any
            // twice-differentiable objective, so a discrepancy that survives
            // the ladder's shrink is rounding, not remainder — and the
            // controller must know that before it can tell a bad region from
            // an unreadable one.
            // The accumulation the early exit certified against, with the trial
            // objective's own magnitude in place of the incumbent's stand-in.
            let accumulation = ObjectiveAccumulation {
                magnitude: old_objective.abs() + trialobjective.abs() + penalty_accumulation_scale,
                ..pre_trial_accumulation
            };
            objective_resolution_witness.observe(
                step_norm,
                actual_reduction,
                predicted_reduction,
                accumulation,
            );
            let measured_objective_resolution = objective_resolution_witness.measured();
            let trust_update = update_joint_trust_region_radius(
                joint_trust_radius,
                step_norm,
                actual_reduction,
                predicted_reduction,
                old_objective,
                objective_tol,
                measured_objective_resolution,
                current_stationarity_residual > residual_tol,
            );
            trust_ratio_witness.observe(step_norm, trust_update.rho, predicted_reduction);
            let old_radius = joint_trust_radius;
            // Classify the outcome of this attempt so the diagnostic line
            // says *why* the step was taken or rejected rather than just
            // dumping numbers. The four phases partition the post-log
            // branches below; computing them up front lets the log line
            // and the dispatch agree.
            let floor_reached = trust_update.accepted
                && current_stationarity_residual <= residual_tol
                && !has_resolvable_negative_curvature
                && joint_objective_floor_reached(
                    old_objective,
                    trialobjective,
                    actual_reduction,
                    predicted_reduction,
                    objective_tol,
                    measured_objective_resolution,
                );
            let roundoff_slack = joint_objective_roundoff_slack(
                old_objective,
                trialobjective,
                measured_objective_resolution,
            );
            let secondary_ok = !floor_reached
                && trialobjective.is_finite()
                && trust_update.accepted
                && trialobjective <= old_objective + roundoff_slack;
            let phase: &'static str = if floor_reached {
                "converged"
            } else if secondary_ok {
                "accepted"
            } else if trust_update.accepted {
                "stall"
            } else {
                "reject"
            };
            if floor_reached || secondary_ok {
                if joint_ball_bound_the_step {
                    // The JOINT ball is what limited this step, so the joint
                    // decision — taken above on `(joint_trust_radius,
                    // joint_step_norm)`, one norm, one radius — is the one
                    // that owns the outcome. Carry it to every block by the
                    // factor it chose: the Moré–Sorensen solve is handed
                    // `max_b R_b`, so scaling all of them moves that maximum
                    // exactly as the controller decided while preserving
                    // whatever relative sizes the per-block path left behind.
                    //
                    // Running the per-block loop here instead is what made the
                    // region a ratchet (gam#2612): under a joint constraint no
                    // block reaches its OWN radius, so every per-block verdict
                    // is "hold" and a step that was genuinely truncated —
                    // `|prop|∞` an order or more above `|δ|∞` — buys no room.
                    let factor = if old_radius.is_finite() && old_radius > 0.0 {
                        trust_update.radius / old_radius
                    } else {
                        1.0
                    };
                    if factor.is_finite() && factor > 0.0 && factor != 1.0 {
                        for block_radius in joint_block_trust_radii.iter_mut() {
                            *block_radius = (*block_radius * factor)
                                .clamp(JOINT_TRUST_RADIUS_FLOOR, JOINT_TRUST_RADIUS_CEILING);
                        }
                    }
                } else {
                    for (block_radius, block_step_norm) in joint_block_trust_radii
                        .iter_mut()
                        .zip(block_step_norms.iter())
                    {
                        let block_update = update_joint_trust_region_radius(
                            *block_radius,
                            *block_step_norm,
                            actual_reduction,
                            predicted_reduction,
                            old_objective,
                            objective_tol,
                            measured_objective_resolution,
                            current_stationarity_residual > residual_tol,
                        );
                        if block_update.radius >= *block_radius
                            || joint_block_step_hit_trust_boundary(*block_step_norm, *block_radius)
                        {
                            *block_radius = block_update.radius;
                        }
                    }
                }
                joint_trust_radius = joint_block_trust_radii
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
            } else {
                joint_trust_radius = shrink_active_joint_block_trust_radii(
                    &mut joint_block_trust_radii,
                    &block_step_norms,
                    0.25,
                );
            }
            let radius_held =
                (joint_trust_radius - old_radius).abs() <= 1e-12 * old_radius.abs().max(1.0);
            let joint_math = JointNewtonMathDiagnostic {
                old_kkt_inf: current_kkt_norm,
                linearized_next_kkt_inf,
                predicted_reduction,
                actual_reduction,
                trust_ratio: trust_update.rho,
                step_inf: trial_step_inf,
                proposal_inf: step_inf,
            };
            let radius_field = if radius_held {
                format!("r={:.3e} (held)", old_radius)
            } else {
                format!("r={:.3e}->{:.3e}", old_radius, joint_trust_radius)
            };
            // Surface the TR-policy decision so future failures
            // distinguish "TR is throttling Newton" from "TR is not
            // the bottleneck — Newton itself finds short steps".
            // For the large-scale linear-convergence pattern the policy
            // is consistently `hold_inside` (ρ≈1, |δ| ≪ radius),
            // which proves the TR is not what is keeping the step
            // small — that came up before via "(held)" alone but
            // the explicit decision label makes the inference
            // immediate instead of requiring step/radius arithmetic
            // in the reader's head.
            let tr_attempt_sig = format!(
                "{:<9}  ρ={:+.3e}  Δobj={:+.3e}  pred={:+.3e}  {}  decision={:<22}  |δ|={:.3e}  |δ|∞={:.3e}  |prop|∞={:.3e}",
                phase,
                trust_update.rho,
                actual_reduction,
                predicted_reduction,
                radius_field,
                trust_update.decision.label(),
                step_norm,
                trial_step_inf,
                step_inf,
            );
            match tr_log_sig.as_deref() {
                Some(prev) if prev == tr_attempt_sig.as_str() => {
                    tr_log_last = line_search_attempts;
                }
                Some(prev) => {
                    if tr_log_first == tr_log_last {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                            cycle,
                            tr_log_first,
                            prev,
                        );
                    } else {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                            cycle,
                            tr_log_first,
                            tr_log_last,
                            tr_log_last - tr_log_first + 1,
                            prev,
                        );
                    }
                    tr_log_sig = Some(tr_attempt_sig);
                    tr_log_first = line_search_attempts;
                    tr_log_last = line_search_attempts;
                }
                None => {
                    tr_log_sig = Some(tr_attempt_sig);
                    tr_log_first = line_search_attempts;
                    tr_log_last = line_search_attempts;
                }
            }
            if floor_reached {
                if let Some(sig) = tr_log_sig.take() {
                    if tr_log_first == tr_log_last {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                            cycle,
                            tr_log_first,
                            sig,
                        );
                    } else {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                            cycle,
                            tr_log_first,
                            tr_log_last,
                            tr_log_last - tr_log_first + 1,
                            sig,
                        );
                    }
                }
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                last_joint_math = Some(joint_math);
                // A trust-floor accept is a common saddle signature
                // (negative curvature keeps every step rejected), so it
                // routes through the same M_true certificate as every other
                // tentative convergence event.
                finish_post_step_convergence!();
            }
            if secondary_ok {
                if let Some(sig) = tr_log_sig.take() {
                    if tr_log_first == tr_log_last {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                            cycle,
                            tr_log_first,
                            sig,
                        );
                    } else {
                        log::info!(
                            "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                            cycle,
                            tr_log_first,
                            tr_log_last,
                            tr_log_last - tr_log_first + 1,
                            sig,
                        );
                    }
                }
                if let Some(joint_active_set) = search_joint_active_set.as_ref() {
                    // Face provenance belongs to the ACCEPTED estimator state,
                    // not to the path the local QP took to its full endpoint.
                    // Globalization may accept only a strict subsegment of that
                    // chord, so endpoint-only rows can still be slack. More
                    // subtly, a degenerate cone face has many sparse row bases:
                    // filtering only the QP's endpoint basis makes the warm
                    // face depend on active-set history even when two accepted
                    // betas lie on the same geometric face (#979).
                    //
                    // Reclassify the complete tight face at the actual accepted
                    // beta, then retain its deterministic lowest-row independent
                    // representatives. The reduced-face carrier performs this
                    // operator-natively for factored cones, without materializing
                    // a dense n×p constraint matrix. Warm starts and terminal
                    // tangent certificates are therefore functions of beta
                    // alone, while redundant co-tight observation rows never
                    // enter the cached equality system.
                    let accepted_beta = flatten_state_betas(&states, specs);
                    let accepted_joint_active = canonical_accepted_active_rows(
                        joint_constraints.as_ref(),
                        &accepted_beta,
                        joint_active_set,
                    )?;
                    let accepted_face = if accepted_joint_active.is_empty() {
                        None
                    } else {
                        Some(accepted_joint_active.clone())
                    };
                    accepted_active_face_changed = accepted_face != active_face_before_step;
                    cached_active_sets =
                        scatter_joint_active_set(&accepted_joint_active, &block_constraints);
                }
                last_joint_math = Some(joint_math);
                last_accepted_hit_joint_trust_boundary = step_hit_trust_boundary;
                accepted = true;
                break;
            }
            for (b, old) in old_beta.iter().enumerate() {
                states[b].beta.assign(old);
            }
            refresh_all_block_etas(family, specs, &mut states)?;
            objective_rejects += 1;
            // Frozen-step short-circuit (see declaration). `radius_held` here
            // means the post-reject shrink did not change the joint trust
            // radius — i.e. the radii are pinned at the `1e-12` floor. If the
            // trial objective also reproduces the previous rejected attempt's
            // bit-for-bit, the dogleg/Moré–Sorensen step is frozen and every
            // remaining attempt would re-reject the identical step: skip the
            // redundant full-data sweeps and let the cross-cycle stall guard
            // certify the fixed point.
            if radius_held && trialobjective.is_finite() {
                if let Some(prev) = prev_rejected_attempt_objective {
                    if prev.to_bits() == trialobjective.to_bits() {
                        frozen_floor_full_reject = true;
                        break;
                    }
                }
                prev_rejected_attempt_objective = Some(trialobjective);
            } else {
                prev_rejected_attempt_objective = None;
            }
        }
        // A SHRINK TAKEN ON NON-EVIDENCE IS UNDONE (gam#2612).
        //
        // If this ladder is the one that MEASURED the objective's resolution,
        // then by construction its own rejections were decided against a floor
        // the solve now knows was too small: the discrepancies it shrank on
        // were rounding, not model error. The shrinks are therefore not
        // evidence about the region, and leaving them in place is not
        // conservative — it is the ratchet. Each shrink makes the next
        // attempt's true reduction smaller against the SAME rounding, so once
        // a cycle starts rejecting on noise it can only reject harder; the
        // #2612 banded witness went from radius `1.166e2` to the `1e-12` floor
        // inside two cycles and then re-rejected one frozen step for 1,195.
        //
        // Restore the radii this cycle started with. The measurement itself is
        // kept (it is a fact about the arithmetic, not about the region), so
        // the next cycle re-proposes the same Newton step against a floor that
        // can now recognise its realized change as rounding and accept it.
        // Fires at most once per solve: `measured` is monotone, so a later
        // ladder that re-measures the SAME resolution does not restore again.
        // A LADDER WHOSE NON-DECAY IS NOT ARITHMETIC SAYS SO (gam#2748). This is
        // the counterpart of the undo below: the same ladder shape, refused
        // because no evaluation of this objective can carry that much rounding.
        // Reported at `warn`, not `info`: it is the signature of a Newton system
        // proposing steps far outside its own model's validity, which is a fact
        // about the fit and not a tuning detail. The shrinks the ladder took
        // therefore STAND, which is the whole point.
        if let Some((claim, ceiling)) = objective_resolution_witness.refused()
            && claim > refused_resolution_before_this_ladder
        {
            log::warn!(
                "[joint-newton objective-resolution gam#2748] cycle={cycle} REFUSED a \
                 resolution claim of {claim:.6e} against an arithmetic ceiling of \
                 {ceiling:.6e} (= m*eps*(1+sum|terms|) with m={total_joint_n}): the ladder's \
                 discrepancy did not fall at the model remainder's rate, but no evaluation of \
                 this objective can round by that much, so what failed to decay is MODEL \
                 error from steps outside the quadratic model's validity. The {} shrink(s) \
                 this ladder took STAND.",
                line_search_attempts,
            );
        }
        if objective_resolution_witness.measured() > resolution_before_this_ladder {
            log::info!(
                "[joint-newton objective-resolution gam#2612] cycle={} MEASURED resolution={:.6e} \
                 (was {:.6e}); one evaluation of the inner objective carries this much rounding, \
                 so the {} shrink(s) this ladder took were decided on it and are undone: \
                 r={:.3e} -> {:.3e}",
                cycle,
                objective_resolution_witness.measured(),
                resolution_before_this_ladder,
                line_search_attempts,
                joint_trust_radius,
                f64::from_bits(cycle_start_joint_trust_radius_bits),
            );
            joint_trust_radius = f64::from_bits(cycle_start_joint_trust_radius_bits);
            for (radius, bits) in joint_block_trust_radii
                .iter_mut()
                .zip(cycle_start_block_trust_radius_bits.iter())
            {
                *radius = f64::from_bits(*bits);
            }
        }
        if let Some(sig) = tr_log_sig.take() {
            if tr_log_first == tr_log_last {
                log::info!(
                    "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                    cycle,
                    tr_log_first,
                    sig,
                );
            } else {
                log::info!(
                    "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                    cycle,
                    tr_log_first,
                    tr_log_last,
                    tr_log_last - tr_log_first + 1,
                    sig,
                );
            }
        }
        let line_search_elapsed = line_search_started.elapsed();
        if accepted && converged {
            log::info!(
                "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                cycle,
                hessian_and_qp_elapsed.as_secs_f64(),
                line_search_elapsed.as_secs_f64(),
                line_search_attempts,
                model_rejects,
                likelihood_rejects,
                objective_rejects,
                feasibility_rejects,
                first_likelihood_reject.as_deref().unwrap_or("none"),
                cycle_started.elapsed().as_secs_f64(),
            );
            // Accepted step moved β; the cycle workspace is at the OLD
            // (pre-step) β, so it must NOT be carried into the post-loop
            // covariance/IFT assembly (which needs the converged β). Drop it.
            cached_joint_workspace = None;
            cycles_done = cycle + 1;
            break;
        }
        if !accepted {
            // Retry the joint Newton loop from the same state after a
            // failed trust-region search. Falling through into blockwise
            // would switch a coupled exact-Hessian problem onto a
            // principal-block surrogate, which is the ridge-drift failure
            // mode this path is meant to avoid. The trust-region radius
            // already collapsed via the attempt loop's shrink rules, so
            // the next cycle's Newton proposal will be evaluated under
            // a tighter L2 bound without any parallel adaptation here.
            log::info!(
                "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=false hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                cycle,
                hessian_and_qp_elapsed.as_secs_f64(),
                line_search_elapsed.as_secs_f64(),
                line_search_attempts,
                model_rejects,
                likelihood_rejects,
                objective_rejects,
                feasibility_rejects,
                first_likelihood_reject.as_deref().unwrap_or("none"),
                cycle_started.elapsed().as_secs_f64(),
            );
            // WHICH KIND OF FAULT (gam#2695). The four reject counters say
            // which GATE refused; they do not say whether refusing was the
            // right answer. The trust ratio's behaviour under refinement does:
            // it tends to 1 as the step shrinks for any model whose rhs is the
            // gradient of the objective the numerator measures. A ladder that
            // shrank two decades without rho moving toward 1 is reporting a
            // first-order disagreement between the model and the objective,
            // and no radius can repair that -- which is exactly why such a
            // cycle spends its whole attempt budget and lands on the floor.
            if let Some(inconsistency) = trust_ratio_witness.model_inconsistency() {
                trust_ratio_model_inconsistency = Some(inconsistency.clone());
                log::info!(
                    "[PIRLS/joint-Newton/model-consistency] cycle={} {}",
                    cycle,
                    inconsistency,
                );
            }
            // Restore original betas
            for (b, old) in old_beta.iter().enumerate() {
                states[b].beta.assign(old);
            }
            refresh_all_block_etas(family, specs, &mut states)?;
            // β is now back at `old_beta`, the exact β this cycle's
            // exact-Newton workspace was built at. A rejected cycle does NOT
            // run the post-accept gradient reload (which is what otherwise
            // re-stashes a workspace), so without this the next cycle's
            // `cached_joint_workspace.take()` is `None` and re-streams all n
            // rows through the per-row kernel cache build — pure redundancy
            // at the identical β. Hand the still-valid workspace back so the
            // next cycle hits the cache. Bit-identical: same family, same β,
            // so the rebuilt cache would be byte-for-byte this one; the inner
            // solve still re-derives the Newton step and runs to its KKT
            // certificate unchanged. On the loop-exit `break`s below this is
            // a harmless assignment to a value that is then dropped.
            cached_joint_workspace = hessian_workspace_for_cycle.take();
            // If the previous cycle's bookkeeping certified KKT
            // stationarity (residual ≤ tol and objective change ≤
            // tol), the line-search failure here is round-off on a
            // rank-deficient null mode rather than non-convergence:
            // the proposed `H⁻¹ g` step stays O(1) along the null
            // direction at the optimum, every trial moves β along
            // it without changing the objective, and round-off
            // flips the sign of `actual − predicted` so the
            // sufficient-decrease check rejects every trial. The
            // iterate ALREADY satisfies the first-order optimality
            // conditions; we accept that as convergence rather
            // than fail the outer "inner solve did not converge"
            // panic on a fully resolved fit.
            if last_cycle_residual_below_tol && last_cycle_obj_change_below_tol {
                finish_post_step_convergence!();
            }
            // Fully-rejected stall guard. See the constant declaration
            // at the top of this function for the full rationale. The
            // condition is: every trust attempt this cycle was rejected by
            // SOME path (model OR likelihood OR objective OR feasibility; the
            // four reject counters partition the JOINT_TRUST_MAX_ATTEMPTS
            // attempts) AND
            // the joint trust radius did not shrink relative to the previous
            // fully-rejected cycle. Both together prove the next cycle's
            // Newton system, trust radius, and trust-region search are
            // bytewise identical to this cycle's — there is no descent
            // direction the local quadratic model can reconcile at this β.
            //
            // The earlier form required objective_rejects ==
            // JOINT_TRUST_MAX_ATTEMPTS && likelihood_rejects == 0, so it
            // NEVER fired on the biobank gauge-flat marginal/slope fit:
            // there the objective is flat to f64 precision along the
            // residual direction and the BMS line search rejects every
            // trial on the *likelihood* early-exit path
            // (likelihood_rejects == 24), so the stall guard's increment
            // condition was unreachable and the loop spun to its cap. A
            // full rejection by the likelihood path at a collapsed trust
            // radius is the same numerically-flat-no-descent stall as a
            // full objective rejection; counting either lets the guard fire.
            let all_attempts_rejected = frozen_floor_full_reject
                || model_rejects + likelihood_rejects + objective_rejects + feasibility_rejects
                    == JOINT_TRUST_MAX_ATTEMPTS;
            if all_attempts_rejected {
                let signature = FullyRejectedCycleSignature {
                    beta_bits: cycle_start_beta_bits,
                    joint_trust_radius_bits: cycle_start_joint_trust_radius_bits,
                    block_trust_radius_bits: cycle_start_block_trust_radius_bits,
                    first_trial_delta_bits: first_attempt_trial_delta_bits,
                    first_trial_objective_bits: first_attempt_trial_objective
                        .map(|value| value.to_bits()),
                    rejection_counts: [
                        model_rejects,
                        likelihood_rejects,
                        objective_rejects,
                        feasibility_rejects,
                    ],
                };
                consecutive_identical_rejected_cycles =
                    if prev_fully_rejected_cycle_signature.as_ref() == Some(&signature) {
                        consecutive_identical_rejected_cycles.saturating_add(1)
                    } else {
                        1
                    };
                prev_fully_rejected_cycle_signature = Some(signature);
            } else {
                prev_fully_rejected_cycle_signature = None;
                consecutive_identical_rejected_cycles = 0;
            }
            // Collapsed-trust-region all-reject-at-floor detector (gam#979).
            // Increment only when EVERY attempt this cycle was rejected AND
            // the joint trust radius has reached its absolute `1e-12` floor:
            // the radius cannot shrink further and the step makes no
            // progress, so the next cycle is forced to repeat this one. Any
            // accepted cycle (handled below via the post-grad-reload reset)
            // or any cycle whose radius is still above the floor breaks the
            // streak, so a progressing fit never accumulates it.
            let all_attempts_rejected_at_floor_this_cycle =
                all_attempts_rejected && joint_trust_radius_at_absolute_floor(joint_trust_radius);
            if all_attempts_rejected_at_floor_this_cycle {
                consecutive_all_reject_at_floor_cycles =
                    consecutive_all_reject_at_floor_cycles.saturating_add(1);
            } else {
                consecutive_all_reject_at_floor_cycles = 0;
            }
            let collapsed_floor_exit = joint_collapsed_floor_all_reject_exit(
                consecutive_all_reject_at_floor_cycles,
                all_attempts_rejected_at_floor_this_cycle,
            );
            if consecutive_identical_rejected_cycles >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                || collapsed_floor_exit
            {
                // #2485. "Every trust-region attempt was rejected" and "no
                // model-resolvable step lowers the objective by more than
                // tolerance" are the SAME observation, and the second one is a
                // convergence CERTIFICATE (Conn–Gould–Toint 6.4.6) — the very
                // certificate this loop already issues ~600 lines below. This
                // guard used to `break` with `converged = false` before that
                // code could run, so on an iterate that has descended to the
                // floating-point noise floor the certificate never fired: β is
                // reverted every attempt precisely BECAUSE there is nothing
                // left to descend. Measured on the by-group location-scale fit
                // that motivated this: predicted decrease 1e-23, objective
                // change ±1e-13 with an oscillating sign, |∇Φ|∞ = 0, zero
                // nullity, and a strict residual 1.56× over a tolerance that
                // no representable step could close.
                //
                // So ask the shared stopping rule BEFORE conceding. Where the
                // decrement is genuinely large — a fit one resolvable step from
                // the optimum, or one whose local model disagrees with the
                // objective — the predicate is false and this guard returns
                // non-converged exactly as before, so a stuck solve cannot be
                // laundered into a certificate by stalling.
                //
                // `objective_tol` is rebuilt from `lastobjective` with the same
                // formula the certificate site uses. On a fully-rejected cycle β
                // is reverted and `lastobjective` is unchanged by construction,
                // so both sites are evaluating the tolerance at the same iterate.
                let stall_objective_tol = inner_tol * (1.0 + lastobjective.abs());
                let stall_decrement = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.newton_decrement());
                let stall_weak_decrement = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.weakly_identified_decrement());
                let stall_numerical_null_stationarity = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.numerical_null_stationarity_inf());
                // Why the certificate did NOT fire is as diagnosable as why it
                // did: without a spectrum there is no decrement to test, and a
                // decrement above tolerance is a genuinely reducible iterate.
                // Both cases fall through to the refusal below, whose message
                // carries `last_newton_math`; this line names the missing input.
                log::debug!(
                    "[PIRLS/joint-Newton convergence] cycle {cycle:>3} | #2485 stall-certificate \
                     spectrum={} decrement={stall_decrement:?} weak={stall_weak_decrement:?} \
                     null_score={stall_numerical_null_stationarity:?} objective_tol={stall_objective_tol:.3e}",
                    joint_spectrum.is_some()
                );
                if head_jeffreys_term.is_some()
                    && head_jeffreys_completion.is_none()
                    && stall_numerical_null_stationarity.is_some_and(|v| v > residual_tol)
                {
                    jeffreys_completion_endgame = true;
                    continue 'joint_newton_cycles;
                }
                if let (Some(decrement), Some(weak_decrement), Some(null_score)) = (
                    stall_decrement,
                    stall_weak_decrement,
                    stall_numerical_null_stationarity,
                ) && joint_newton_decrement_certifies(
                    decrement,
                    weak_decrement,
                    null_score,
                    stall_objective_tol,
                    residual_tol,
                ) {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {cycle:>3} | #2485 fully-rejected \
                         stall certified: radius={joint_trust_radius:.3e}, decrement={decrement:.3e}, \
                         weak={weak_decrement:.3e}, null_score={null_score:.3e}, \
                         objective_tol={stall_objective_tol:.3e}; no resolvable descent remains"
                    );
                    // Same exit every other certificate takes — so this one is
                    // ALSO tentative until the next cycle head has certified the
                    // returned mode's curvature. On a strict saddle that head
                    // revokes it and resets every stall counter, so certifying
                    // here cannot short-circuit the gam#979 escape and cannot
                    // spin: the guard's streak starts over.
                    finish_post_step_convergence!();
                }
                let last_math_summary = last_joint_math
                    .as_ref()
                    .map(|math| {
                        format!(
                            "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                            math.old_kkt_inf,
                            math.linearized_next_kkt_inf,
                            math.actual_reduction,
                            math.predicted_reduction,
                            math.trust_ratio,
                            math.scalar_model_relative_error(),
                            math.step_inf,
                            math.proposal_inf,
                        )
                    })
                    .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                let stall_trigger = if consecutive_identical_rejected_cycles
                    >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                {
                    format!(
                        "{} consecutive fully-rejected cycles reproduced the complete \
                         pre-cycle iterate/radius state, first proposal, trial objective, \
                         and rejection partition bit-for-bit (exact fixed point)",
                        consecutive_identical_rejected_cycles
                    )
                } else {
                    format!(
                        "{} consecutive fully-rejected cycles with the joint trust radius \
                         collapsed to its absolute 1e-12 floor (no smaller step \
                         representable, step makes no progress)",
                        consecutive_all_reject_at_floor_cycles
                    )
                };
                let rejection_counts = [
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    feasibility_rejects,
                ];
                let typed_termination_reason = if consecutive_identical_rejected_cycles
                    >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                {
                    gam_problem::JointNewtonTerminalReason::FullyRejectedExactFixedPoint {
                        consecutive_cycles: consecutive_identical_rejected_cycles,
                        joint_trust_radius,
                        rejection_counts,
                    }
                } else {
                    gam_problem::JointNewtonTerminalReason::FullyRejectedAtTrustRegionFloor {
                        consecutive_cycles: consecutive_all_reject_at_floor_cycles,
                        joint_trust_radius,
                        rejection_counts,
                    }
                };
                if let Some(gam_problem::InnerConvergenceTerminalState::JointNewton {
                    termination_reason,
                    ..
                }) = terminal_convergence_state.as_mut()
                {
                    *termination_reason = typed_termination_reason;
                }
                // WHY no step was reachable, when the ladder can tell us
                // (gam#2695). "No accepted descent step is reachable under the
                // current local model" is true either way, but it reads as a
                // hard problem when the model may simply not be the objective's
                // model at all. If rho refused to approach 1 while the step
                // shrank two decades, say so here: that is a first-order
                // disagreement, it is a defect rather than a difficulty, and it
                // sends the reader to the gradient instead of to the radius.
                let model_consistency_note = trust_ratio_model_inconsistency
                    .as_deref()
                    .map(|reason| format!(" MODEL-CONSISTENCY FAULT: {reason}."))
                    .unwrap_or_default();
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | fully-rejected stall \
                     early-exit: every trust-region attempt rejected (by any of the model / \
                     likelihood / objective paths) — {} at joint trust radius {:.3e}.{} Reverted β \
                     + identical Newton system mean the next cycle's step is byte-identical to \
                     this one's; no accepted descent step is reachable from this iterate under the \
                     current local model. {}. The strict KKT residual has not converged, so \
                     returning non-converged.",
                    cycle,
                    stall_trigger,
                    joint_trust_radius,
                    model_consistency_note,
                    last_math_summary,
                );
                converged = false;
                break;
            }
            // CONTINUE rather than break (gam#826/#872/#715). The comment
            // above documents the intent — "retry the joint Newton loop from
            // the same state after a failed trust-region search" — but the old
            // code BROKE instead, giving up after a SINGLE cycle of failed line
            // search. On a severely near-separating coupled fit (matern
            // binomial location-scale, quasi-separating multinomial, flexible
            // linkwiggle) the cycle-0 Newton proposal is huge (the separation
            // gradient ÷ the Firth-bounded curvature), the trust region clamps
            // it, and the clamped step does not yet reduce the merit — so the
            // FIRST cycle's backtracking exhausts without acceptance. The
            // attempt loop already shrank `joint_trust_radius` /
            // `joint_block_trust_radii` (carried across cycles), so the NEXT
            // cycle re-proposes under the tighter radius and eventually accepts
            // a productive step — standard trust-region globalization. Breaking
            // at cycle 0 aborted the coupled solve ("exited the joint Newton
            // path before convergence — no math snapshot") before the trust
            // region could adapt. The inner cycle cap and the residual-stall /
            // trust-region-floor guards above still bound the loop, so a
            // genuinely stuck fit exits with a diagnosed non-convergence rather
            // than spinning. Falling through to blockwise (the old `break`)
            // would switch the coupled exact-Hessian problem onto a
            // principal-block surrogate (the ridge-drift mode this path avoids).
            if joint_workspace_requested {
                cached_joint_hessian_source = Some(joint_hessian_source);
            }
            continue;
        }

        let grad_reload_started = std::time::Instant::now();
        log::info!(
            "[joint-newton-tr] phase=gradient_reload cycle={} attempts={} r={:.3e}",
            cycle,
            line_search_attempts,
            joint_trust_radius,
        );
        let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
            family,
            specs,
            options,
            &states,
            joint_workspace_requested,
            accepted_joint_workspace.take(),
        )?;
        let grad_reload_elapsed = grad_reload_started.elapsed();
        // Reset the fully-rejected stall guard's bookkeeping: an accepted
        // cycle moved β and may have grown the trust radius, so the next
        // rejected-cycle comparison must start fresh rather than carry
        // forward a stale radius snapshot from the previous reject streak.
        prev_fully_rejected_cycle_signature = None;
        // An accepted step moved β, so the fixed-point signature is stale;
        // reset it so a later reject streak compares only consecutive
        // fully-rejected cycles at the SAME iterate.
        consecutive_identical_rejected_cycles = 0;
        // An accepted step moved β and (via the trust-region grow rules)
        // lifts the radius off its floor, so the collapsed-floor all-reject
        // streak no longer holds; reset it (gam#979).
        consecutive_all_reject_at_floor_cycles = 0;
        // Accepted-cycle timing breakdown is debug-only. The per-cycle
        // info line below already includes total cycle time; emitting a
        // four-phase split on every verbose cycle adds a redundant info
        // line. Rejected cycles still keep the detailed phase log since
        // the reject reason and per-phase split is the diagnostic.
        log::debug!(
            "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} grad_reload={:.3}s total={:.3}s",
            cycle,
            hessian_and_qp_elapsed.as_secs_f64(),
            line_search_elapsed.as_secs_f64(),
            line_search_attempts,
            grad_reload_elapsed.as_secs_f64(),
            cycle_started.elapsed().as_secs_f64(),
        );
        current_log_likelihood = log_likelihood;
        cached_joint_gradient = gradient;
        cached_eval = eval;
        cached_joint_workspace = workspace;
        current_penalty = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        // `current_penalty` / `lastobjective` stay the pure quadratic-penalized
        // objective (NO Φ folded in) — the Firth value is applied per cycle at
        // each β (see `old_objective` above and `trialobjective` below). The
        // gated Φ at the accepted β is captured separately so the convergence
        // `objective_change` compares the augmented objective at the new vs old
        // β consistently (gam#826/#872).
        lastobjective = -current_log_likelihood + current_penalty;
        let new_phi = if !jeffreys_skippable_this_cycle {
            joint_jeffreys_subspace
                .as_ref()
                .map(|z_joint| {
                    custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                        .map(|value| value.phi)
                })
                .transpose()?
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let accepted_step_inf = states
            .iter()
            .zip(old_beta.iter())
            .flat_map(|(state, old)| {
                state
                    .beta
                    .iter()
                    .zip(old.iter())
                    .map(|(new, old)| (new - old).abs())
            })
            .fold(0.0_f64, f64::max);
        cycles_done = cycle + 1;

        // Check convergence via joint stationarity. When the family-general
        // Firth/Jeffreys term is armed, the penalized objective the inner
        // Newton actually optimizes is `−ℓ + ½βᵀSβ − Φ`, so its KKT
        // stationarity is `∇L − Sβ + ∇Φ = 0`. The Newton STEP already folds
        // `∇Φ` into its RHS (`spectral_rhs += grad_phi`), but the bare
        // `exact_newton_joint_stationarity_*` residual omits it — at the
        // Firth fixed point `∇L − Sβ = −∇Φ`, so the certificate floors at
        // `‖∇Φ‖∞` and never certifies, stalling the inner solve on exactly
        // the near-separating span Firth is meant to bound (the residual the
        // outer REML then rejects). Fold `∇Φ` into the gradient used for the
        // KKT residual so the convergence criterion matches the augmented
        // objective the step descends. No-op when the Jeffreys term is
        // unavailable or condition-gated to zero.
        let Some(gradient) = cached_joint_gradient.as_ref() else {
            break;
        };
        let jeffreys_augmented_gradient: Option<Array1<f64>> = if jeffreys_skippable_this_cycle {
            // Well-conditioned ⇒ ∇Φ = 0, so the KKT residual is the bare
            // stationarity (and floors at 0, not ‖∇Φ‖) — matching the step,
            // which folded H_Φ=0/∇Φ=0 this cycle. Avoids the dense H/eigh.
            None
        } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
            let workspace_term = match cached_joint_workspace.as_ref() {
                Some(workspace) => custom_family_joint_jeffreys_term_from_workspace(
                    workspace.as_ref(),
                    total_p,
                    z_joint,
                    family.joint_jeffreys_term_strength(),
                )?,
                None => None,
            };
            // Workspace evidence is authoritative when available. Families
            // whose workspace does not expose a batched all-axes derivative
            // keep the existing exact family assembly.
            let jeffreys_term = match workspace_term {
                Some(term) => Some(term),
                None => {
                    custom_family_joint_jeffreys_term(family, &states, specs, &ranges, z_joint)?
                }
            };
            match jeffreys_term {
                Some((_phi, grad_phi, hphi))
                    if grad_phi.len() == gradient.len()
                        && hphi.nrows() == total_p
                        && hphi.ncols() == total_p =>
                {
                    let augmented = gradient + &grad_phi;
                    // Cache the exact triple at the just-accepted β so the next
                    // cycle's head reuses it instead of recomputing the
                    // O(p)-directional-derivative + GEMM term (gam#729).
                    let post_beta_key = flatten_state_betas(&states, specs);
                    jeffreys_triple_cache = Some((post_beta_key, grad_phi, hphi));
                    Some(augmented)
                }
                _ => None,
            }
        } else {
            None
        };
        let residual_gradient = jeffreys_augmented_gradient.as_ref().unwrap_or(gradient);
        if accepted_active_face_changed {
            // The accepted QP step changed the critical cone. In particular,
            // the Duchon CTN separator can enlarge the face substantially
            // (the issue-979 4,800-row replay changed 94 -> 274 rows). The
            // projected residual then jumps because it is a different KKT
            // system, while the preceding objective/residual samples describe
            // the old face. Feeding those samples to the constrained-
            // stationary or slow-rate guards creates a false plateau and
            // refuses the very first iterate on the newly discovered face.
            //
            // Reset every cross-cycle progress statistic whose inference
            // assumes a fixed stationarity system. This does not accept an
            // iterate or loosen a tolerance: the current residual is computed
            // below on the new face and must build fresh evidence and satisfy
            // the unchanged KKT certificate.
            min_certified_residual = f64::INFINITY;
            best_residual_seen = f64::INFINITY;
            cycles_since_residual_improved = 0;
            residual_descent_history.clear();
            tr_clamped_during_stall = false;
            residual_rate_history.clear();
            merit_window.clear();
            geometric_tail_history.clear();
            last_kkt_refusal_report = None;
            log::info!(
                "[PIRLS/joint-Newton active-face] cycle {} | accepted critical-cone transition; reset fixed-face convergence histories and require a fresh KKT certificate",
                cycle,
            );
        }
        let residual = exact_newton_joint_stationarity_inf_norm_from_gradient(
            residual_gradient,
            &states,
            specs,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            &block_constraints,
            Some(cached_active_sets.as_slice()),
            joint_lower_bounds.as_ref(),
            joint_penalty_stationarity_score(options, specs, &states).as_ref(),
        )?;
        prev_kkt_norm = Some(residual);
        // Record this cycle's KKT residual for the steady-geometric-descent
        // test at the certificate-refusal gate below (gam#787 centers≥20).
        if residual.is_finite() {
            min_certified_residual = min_certified_residual.min(residual);
            residual_descent_history.push_back(residual);
            while residual_descent_history.len() > RESIDUAL_DESCENT_WINDOW {
                residual_descent_history.pop_front();
            }
        }

        // Scale-aware tolerances. The objective check was already
        // relative (`inner_tol * (1 + |obj|)`), but the step and
        // residual checks were absolute against the bare `inner_tol`
        // — at large scale (n ≈ 320k), β iterates can keep moving
        // by ~1e-5 per cycle along the monotonicity-feasible
        // manifold even after the likelihood has gone flat, and the
        // joint gradient ‖·‖_∞ is O(|obj|), not O(1). Running
        // 50-100 cycles past objective convergence is the
        // dominant inner-PIRLS cost at large scale. Switching to
        // relative scaling (`inner_tol * (1 + ‖β‖_∞)` for steps,
        // `inner_tol * (1 + |obj|)` for the gradient residual)
        // exits PIRLS as soon as the optimum is statistically
        // resolved, without loosening behavior at small n where
        // ‖β‖_∞ ≈ 1 and |obj| ≈ 1 give tolerances within 2× of
        // the historical absolute 1e-6.
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + lastobjective.abs());
        // KKT residual tolerance must scale with the natural magnitude of
        // ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. At
        // large scale with |β|∞ in the 10²–10³ range the gradient and
        // penalty norms can sit orders of magnitude above |obj| and FP
        // noise alone keeps the residual above any obj-scaled tol. The
        // pre-line-search check at the head of the cycle already uses
        // `inner_tol * (1 + max(grad_inf, pen_inf))`; using only grad_inf
        // here created an asymmetry where the same convergence criterion
        // would accept at one site and reject at the other, and on
        // marginal-slope models where Sβ is the larger term it shrank
        // the post-accept tolerance below the achievable FP floor.
        let mut block_gradient_norms = Vec::with_capacity(states.len());
        let mut block_penalty_norms = Vec::with_capacity(states.len());
        for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
            block_gradient_norms.push(
                gradient
                    .slice(s![start..end])
                    .iter()
                    .map(|x: &f64| x.abs())
                    .fold(0.0_f64, f64::max),
            );
            let mut penalty_block = s_lambdas[block_idx].dot(&states[block_idx].beta);
            if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                penalty_block += &states[block_idx].beta.mapv(|v| ridge * v);
            }
            block_penalty_norms.push(
                penalty_block
                    .iter()
                    .map(|x: &f64| x.abs())
                    .fold(0.0_f64, f64::max),
            );
        }
        let grad_inf = block_gradient_norms.iter().copied().fold(0.0_f64, f64::max);
        let pen_inf = block_penalty_norms.iter().copied().fold(0.0_f64, f64::max);
        // Firth/Jeffreys score magnitude. The convergence residual is the
        // AUGMENTED stationarity `∇L − Sβ + ∇Φ`, so `∇Φ` is a first-class term
        // whose own numerical scale sets the achievable KKT floor: `∇Φ` is a
        // trace `½ tr(H_id⁻¹ Z_Jᵀ Ḣ Z_J)` formed from a FLOORED reduced-info
        // pseudo-inverse, so its components carry O(‖∇Φ‖·ε_floor) round-off
        // that the augmented residual cannot polish below. Scaling the KKT
        // tolerance by `max(grad, pen, ‖∇Φ‖)` (not just grad/pen) makes the
        // certificate reachable for coupled K-block Firth fits whose data
        // gradient is small but whose Firth score is O(1): otherwise the
        // augmented residual plateaus a few × above an unattainably tight
        // `inner_tol·(1+grad)` tol and the solve refuses just short of
        // convergence (gam#729/#715 — the residual stalled at ~8.8e-6 against a
        // ~1e-6 tol). No-op when the term is condition-gated (∇Φ=0).
        let firth_score_inf = head_jeffreys_term
            .as_ref()
            .map(|(grad_phi, _hphi)| grad_phi.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
            .unwrap_or(0.0);
        // See the head-of-cycle site for why the denominator is named rather
        // than folded straight into the tolerance (gam#2713).
        let stationarity_scale = grad_inf.max(pen_inf).max(firth_score_inf);
        let residual_tol = inner_tol * (1.0 + stationarity_scale);
        // Arm the Jeffreys second-order endgame completion (gam#979) once
        // the residual enters the convergence band; latched (never
        // un-armed) so the endgame model cannot oscillate between the
        // divided-difference and exact Hessians across cycles.
        if residual.is_finite() && residual <= JEFFREYS_COMPLETION_RESIDUAL_BAND * residual_tol {
            jeffreys_completion_endgame = true;
        }
        // Active-set-projected stationarity residual vector (multiplier
        // mass of every pinned bound row already subtracted). Keep the full
        // vector so the constrained-stationary certificate can distinguish
        // represented active-set multipliers from unresolved KKT mass.
        let projected_residual_vec =
            exact_newton_joint_projected_stationarity_vector_from_gradient(
                gradient,
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
                Some(cached_active_sets.as_slice()),
                joint_penalty_stationarity_score(options, specs, &states).as_ref(),
            )?;
        let block_stationarity_norms = {
            let mut offset = 0usize;
            states
                .iter()
                .map(|state| {
                    let start = offset;
                    let end = start + state.beta.len();
                    offset = end;
                    projected_residual_vec
                        .slice(ndarray::s![start..end])
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max)
                })
                .collect::<Vec<_>>()
        };
        // gam#1082 perf: a per-cycle #979 divergence-trace logging block
        // lived here and computed — EVERY inner cycle for the first 40
        // cycles, purely to feed two `log::info!` lines — a FULL O((P·M)³)
        // eigendecomposition of the penalized-Hessian range, a
        // penalty-matrix min-eigenvalue, and per-penalty quadratic forms.
        // On any penalized family with a penalty null space (every
        // `select=TRUE` double-penalty tp-smooth model, including the
        // multinomial smooth-by-factor fit) the eigh's `nullity > 0` branch
        // actually ran, so each outer REML evaluation paid up to 40
        // redundant O(p³) eigendecompositions inside its inner joint-Newton.
        // That diagnostic instrumentation — not the outer iteration count —
        // was the dominant wall-clock cost (the #1082 overrun the outer
        // rel-cost decouple could not touch, because the cost is
        // per-inner-cycle, not per-outer-iteration). The trace has served
        // its #979 purpose and is removed from the production hot path; the
        // strict residual and per-block diagnostics remain available without
        // introducing a second numerical rank decision.
        let near_convergence = residual <= 10.0 * residual_tol;
        // Augmented-objective change: `(quad(new) − Φ_gated(new)) −
        // (quad(old) − Φ_gated(old))`. `lastobjective` is quadratic-only and
        // `old_objective` already carries `−old_phi`, so subtract the accepted
        // β's `new_phi` here to keep both endpoints on the Φ-augmented merit
        // (gam#826/#872). On a skippable cycle both phis are 0 ⇒ identical to
        // the bare quadratic change.
        let signed_obj_change = (lastobjective - new_phi) - old_objective;
        let objective_change = signed_obj_change.abs();

        // Per-cycle observability for the convergence test. Surfaces
        // WHICH criterion is binding (proposed step, accepted step,
        // residual, objective change) at every iteration so CI logs
        // distinguish "Newton hasn't proposed a small step yet"
        // (algorithm still working) from "step is small but residual
        // won't drop below tol" (tolerance scaling problem). Without
        // this, the only visible signal is the objective itself,
        // which is insufficient to choose the right algorithmic
        // remedy.
        //
        // gam#979 discriminator: the PER-BLOCK projected stationarity
        // breakdown. The aggregate `residual` alone cannot distinguish a
        // genuinely-coupled stall from one block dragging the others — for
        // the survival marginal↔slope grind the question "is the total
        // residual dominated by a single block (the multiplicative
        // z·exp(slope) coupling channel), or spread evenly (global
        // conditioning)?" is answerable only from the split. `block_resid`
        // is already computed above for the convergence test, so surfacing
        // it per cycle is free; reading it across a 75 s repro under
        // RUST_LOG=info tells whether the slowdown is a single stuck block
        // (curvature/coupling channel) or an evenly slow descent
        // (conditioning) — without it the four #979 candidates are not
        // separable from the timeline.
        let block_resid_sig = block_stationarity_norms
            .iter()
            .map(|n| format!("{n:.3e}"))
            .collect::<Vec<_>>()
            .join(",");
        // gam#2647 discriminator: the per-block ‖β‖∞ beside the per-block
        // residual. The aggregate `beta_inf` says a coefficient is growing;
        // it cannot say WHICH block is growing, and that is the whole
        // question when two blocks share a gauge direction. On the binomial
        // location-scale-wiggle arm the aggregate climbs monotonically while
        // ½βᵀSβ FALLS like ‖β‖⁻², which is only readable as "the growth is
        // in a penalty-null direction of one specific block" once the split
        // is visible. Same cost as the residual split beside it: the block
        // states are already in hand.
        let block_beta_sig = states
            .iter()
            .map(|s| {
                format!(
                    "{:.3e}",
                    s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        log::info!(
            "[PIRLS/joint-Newton convergence] cycle {:>3} | step_inf={:.3e} (tol={:.3e}) | accepted_step_inf={:.3e} | residual={:.3e} (tol={:.3e}) | relative_stationarity={:.3e} (scale={:.3e}, inner_tol={:.3e}) | per_block_resid=[{}] | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e} | per_block_beta_inf=[{}]",
            cycle,
            step_inf,
            step_tol,
            accepted_step_inf,
            residual,
            residual_tol,
            gam_problem::relative_stationarity(residual, stationarity_scale),
            stationarity_scale,
            inner_tol,
            block_resid_sig,
            objective_change,
            objective_tol,
            beta_inf,
            block_beta_sig,
        );

        // gam#1082 perf: a tightly-gated `#1040 inner-conditioning probe`
        // lived here. Once the inner joint-Newton stalled (residual stuck
        // above tol for `RESIDUAL_STALL_NO_IMPROVE_CYCLES` cycles), it
        // eigendecomposed the FULL P·M penalized Hessian (O((P·M)³)) plus an
        // O(p²) Rayleigh-quotient loop EVERY cycle thereafter, purely to feed
        // one `log::info!`. The gate's whole point is "the solve is
        // grinding" — exactly the regime where it then fires on EVERY one of
        // the remaining (up to `inner_max_cycles`) cycles, turning a stall
        // into an O(p³)-per-cycle crawl (a dominant face of the #1082
        // multinomial wall-clock overrun: the cost is per-stalled-cycle, not
        // per-outer-iteration). The diagnostic is removed from the hot path;
        // the inner solve's own stall handling (trust-region clamp and
        // Newton-decrement certificate) governs
        // termination, and the cheap per-cycle convergence line above already
        // surfaces residual/step/per-block-residual for observability.

        if verbose_cycle || near_convergence {
            log::info!(
                "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} -loglik={:.6e} pen={:.3e} Δobj={:+.3e} |δ|∞={:.3e} accepted_|δ|∞={:.3e} resid={:.3e} (tol={:.3e}) rel_stat={:.3e} (scale={:.3e}) obj_tol={:.3e} step_tol={:.3e} |β|∞={:.3e} attempts={} t={:.3}s",
                cycle,
                inner_max_cycles,
                lastobjective,
                -current_log_likelihood,
                current_penalty,
                signed_obj_change,
                step_inf,
                accepted_step_inf,
                residual,
                residual_tol,
                gam_problem::relative_stationarity(residual, stationarity_scale),
                stationarity_scale,
                objective_tol,
                step_tol,
                beta_inf,
                line_search_attempts,
                cycle_started.elapsed().as_secs_f64(),
            );
        } else {
            log::info!(
                "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} Δobj={:+.3e} |δ|∞={:.3e} resid={:.3e} attempts={} t={:.3}s",
                cycle,
                inner_max_cycles,
                lastobjective,
                signed_obj_change,
                accepted_step_inf,
                residual,
                line_search_attempts,
                cycle_started.elapsed().as_secs_f64(),
            );
        }

        // Divergence guard: a non-finite KKT residual, objective, or
        // log-likelihood means the inner joint Newton has diverged (NaN
        // mass propagating from a near-unidentified penalized block — the
        // binomial location-scale shared-basis log-σ deviation channel is
        // the canonical trigger, gam#554). Every convergence and
        // residual-stall exit below is gated on finite `<=` comparisons,
        // which a NaN residual silently defeats; left unguarded the loop
        // then grinds the full `inner_loop_hard_ceiling` on every outer
        // ρ-eval and every startup seed, which is the multi-hour "hang".
        // Treat it as immediate non-convergence so the outer optimizer
        // rejects this point cleanly instead of burning the budget.
        if !residual.is_finite()
            || !lastobjective.is_finite()
            || !current_log_likelihood.is_finite()
        {
            log::warn!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | divergence guard: non-finite inner state (residual={:.3e}, objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                cycle,
                residual,
                lastobjective,
                -current_log_likelihood,
            );
            converged = false;
            break;
        }

        // KKT convergence: a small post-step residual is the
        // canonical optimality certificate for the penalized
        // objective. ‖∇L(β) − Sβ‖∞ ≤ residual_tol means the
        // iterate is at a KKT point to numerical precision and
        // further iteration cannot reduce it; the step magnitude
        // is irrelevant once the residual signal has fired.
        //
        // Tying convergence to a small step instead would refuse
        // to recognise quadratic-rate single-shot convergence:
        // exact Newton on an exact quadratic produces one full
        // step that lands at the optimum, so ‖delta‖∞ equals the
        // initial distance ‖β* − β₀‖∞ no matter how exact the
        // model is. Pairing a residual check with a step-size
        // requirement structurally rejects this entirely-correct
        // cycle-0 termination, leaving inner_max_cycles=1 callers
        // unable to certify convergence on a problem that was
        // solved exactly in one Newton step.
        if joint_inner_kkt_converged(residual, residual_tol) {
            finish_post_step_convergence!();
        }
        // Newton-decrement convergence certificate (gam#1040 / gam#1088).
        //
        // The strict / identified-subspace / constrained certificates all
        // gate on the penalized stationarity residual ‖∇L − Sβ‖∞ reaching
        // `residual_tol`. On a weakly-identified (near-flat) carrying block
        // — the survival marginal↔slope alias, the binomial link-wiggle
        // block, the gaussian/binomial location-scale μ block — that residual
        // can stall ORDERS above tol (`g` is O(1e2) along a direction whose
        // penalized curvature `γ` is tiny) while every step the trust region
        // admits is clamped, so neither the residual nor the step-norm gate
        // ever closes and the loop grinds to the cycle ceiling, the outer
        // REML rejects ρ after ρ, and the fit times out (the #1040/#1088
        // benchmark hangs). Yet the ACHIEVABLE objective improvement is
        // `g²/(2γ)` — the Newton decrement — and on such a direction it is
        // far below `objective_tol`: no step the local quadratic model can
        // resolve lowers the penalized objective by more than `objective_tol`.
        // By the Conn–Gould–Toint stopping criterion (*Trust-Region Methods*,
        // Thm 6.4.6) the iterate is then the penalized optimum to within
        // tolerance, on the entire identifiable subspace — the residual's
        // un-resolvable mass lives on near-null directions the outer IFT
        // pseudo-inverse projects out (gam#553). The decrement is read off
        // the SAME D-whitened seed spectrum the step is built from (range
        // modes only; the null space contributes none), so it is exactly the
        // model decrease of the unconstrained modified-Newton step. A genuine
        // defect (real curvature AND large gradient) yields a LARGE decrement,
        // so this never certifies a non-converged iterate.
        //
        // Precondition (gam#1082): the original gate required the LAST cycle's
        // `objective_change ≤ objective_tol` to "confirm we are AT the plateau,
        // not one big step away." That precondition is the multinomial
        // smooth-by-factor blocker: the coupled-softmax select=TRUE gauge mode
        // is a NEAR-null (weak-but-above-`KKT_REFUSAL_RANK_TOL` curvature), so
        // the iterate keeps DRIFTING along it with a small but nonzero
        // `objective_change` every cycle (exactly the gam#979 survival
        // signature) — `objective_change ≤ objective_tol` never holds, the
        // decrement certificate never fires, and the solve crawls to
        // `inner_max_cycles` paying one ~p³ Newton-step eigh per cycle (the
        // eu-stack-profiled #1082 blow-up). But the decrement bound is itself
        // the correct, curvature-aware stopping test: by Conn–Gould–Toint Thm
        // 6.4.6 `decrement ≤ objective_tol` ALONE certifies the iterate is the
        // penalized optimum to tolerance — no model-resolvable step (gauge
        // drift included) lowers the objective by more than tol. So the
        // objective-flat precondition is replaced by the RESIDUAL-STALL window
        // (`cycles_since_residual_improved ≥ DECREMENT_STALL_WINDOW`): the
        // certificate fires once the raw residual has stopped descending and
        // the decrement confirms no resolvable improvement remains. This reuses
        // the EXACT degeneracy classification the Newton step uses (the
        // decrement skips every `|γ_k| ≤ null_cutoff` mode), so it catches the
        // near-null gauge direction the raw-`H_pen` range projection's absolute
        // `1e-10·λ_max` cutoff misses — without ever accepting a genuinely
        // curved (large-decrement) unconverged iterate. A still-progressing
        // solve never reaches the stall window (its residual keeps improving,
        // resetting the counter).
        //
        // Plateau disjunct (gam#1607 gaussian/binomial homoscedastic location-
        // scale). The residual-stall window alone has a complementary blind
        // spot to the multinomial drift it was built for: a near-flat scale
        // ridge (homoscedastic data → the log_σ block is weakly identified, the
        // μ block's penalized residual floors a few ×10⁻² above `residual_tol`
        // with a tiny `decrement`) keeps the raw residual JITTERING by >10% per
        // cycle around its plateau (0.031 → 0.024 → 0.028), so the 10%-drop test
        // resets `cycles_since_residual_improved` to 0 every cycle and the stall
        // window NEVER reaches DECREMENT_STALL_WINDOW within the (outer-capped,
        // ~12-cycle) refit budget. The OBJECTIVE, however, is genuinely flat
        // there (`objective_change` ~10⁻⁵ ≪ `objective_tol`) — that is the very
        // signal the original gam#1082 precondition used before it was narrowed
        // to the stall window for the multinomial gauge-drift case (where the
        // objective keeps changing). Restoring it as a DISJUNCTIVE alternative
        // recovers the homoscedastic case without touching multinomial (which
        // still fires via the stall window): both disjuncts gate the SAME
        // rigorous `decrement ≤ objective_tol` Conn–Gould–Toint stopping test
        // below, so neither can certify a genuinely reducible (large-decrement)
        // iterate — a fit one resolvable step from the optimum has a large
        // decrement (fails the bound) regardless of which precondition admits
        // it, and a fit still making real objective progress has
        // `objective_change > objective_tol` (fails this disjunct) AND a
        // descending residual that resets the stall window (fails the other).
        const DECREMENT_STALL_WINDOW: usize = 3;
        let decrement_precondition = cycles_since_residual_improved >= DECREMENT_STALL_WINDOW
            || objective_change <= objective_tol;
        let numerical_null_stationarity = joint_spectrum
            .as_ref()
            .map(|spectrum| spectrum.numerical_null_stationarity_inf());
        if decrement_precondition
            && head_jeffreys_term.is_some()
            && head_jeffreys_completion.is_none()
            && numerical_null_stationarity.is_some_and(|v| v > residual_tol)
        {
            jeffreys_completion_endgame = true;
            continue 'joint_newton_cycles;
        }
        // Conditioning-robust safety (gam#1449) and the raw decrement bound are
        // BOTH in `joint_newton_decrement_certifies`, which is also what the
        // fully-rejected stall guard consults before conceding (#2485) — one
        // stopping rule, one place, so the two sites cannot disagree again.
        if decrement_precondition
            && let Some(decrement) = joint_spectrum
                .as_ref()
                .map(|spectrum| spectrum.newton_decrement())
            && let Some(weak_decrement) = joint_spectrum
                .as_ref()
                .map(|spectrum| spectrum.weakly_identified_decrement())
            && let Some(null_score) = numerical_null_stationarity
            && joint_newton_decrement_certifies(
                decrement,
                weak_decrement,
                null_score,
                objective_tol,
                residual_tol,
            )
        {
            // Audit witness (#1082): the residual mass this certificate
            // EXCLUDES as gauge-null. The decrement bound is sound only when
            // that excluded mass truly lies on penalty-null directions; if it
            // is large the certificate may have discarded a weakly-identified
            // real mode (the `null_cutoff = rank_tol·λ_max` ill-conditioning
            // edge), so emit it at WARN to keep the decision auditable.
            let excluded_null_residual = joint_spectrum
                .as_ref()
                .map(|spectrum| spectrum.null_residual_inf())
                .unwrap_or(0.0);
            if excluded_null_residual > residual_tol.max(1e-6) {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {cycle:>3} | Newton-decrement \
                     certificate fired with LARGE excluded near-null residual \
                     ={excluded_null_residual:.3e} (> tol={residual_tol:.3e}); the stopping \
                     rule treated this mass as free gauge. Sound iff it lies on genuine \
                     penalty-null directions — flagged for joint-stationarity audit (#1082)."
                );
            }
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {} | decrement certificate: \
                 residual={:.3e}/{:.3e}, stalled_cycles={}, |Δobj|={:.3e}, \
                 decrement={:.3e}/{:.3e}, null_score={:.3e}/{:.3e}; \
                 no model-resolvable descent remains",
                cycle,
                residual,
                residual_tol,
                cycles_since_residual_improved,
                objective_change,
                decrement,
                objective_tol,
                null_score,
                residual_tol,
            );
            // Record the residual this exit certified on so the terminal
            // line reports a finite certified residual (#1040 truthfulness):
            // the converged status is earned by the decrement bound, and the
            // finite stationarity residual at this iterate is the honest
            // certificate witness.
            if residual.is_finite() {
                min_certified_residual = min_certified_residual.min(residual);
            }
            finish_post_step_convergence!();
        }

        // Noise-floor KKT certificate.
        //
        // Reading the joint stationarity residual ‖∇L(β) − Sβ‖_∞ at finite
        // precision picks up rounding mass from the X'WX assembly and the
        // per-block penalty contraction. For well-conditioned problems
        // that floor sits well below `residual_tol`, so the strict path
        // fires and this branch is dormant. For tightly converged inner
        // states where the Newton iterate is already at the analytic
        // optimum but every additional step changes the objective by less
        // than `objective_tol` and the recomputed residual lands just
        // above `residual_tol` due to arithmetic noise, the strict path
        // alone refuses to certify convergence — even though no further
        // useful descent direction exists. Burning hundreds of identical
        // descent cycles past that point neither tightens the inner
        // optimum (the noise floor sets a hard lower bound on ‖rhs‖) nor
        // gives the outer optimizer more hyperparameter information; it
        // just causes the outer wrapper to reject every seed as
        // "inner did not converge" and downstream callers to mark the
        // analytic outer Hessian as unavailable.
        //
        // Combining two independent post-step signals — objective change
        // within scale-aware tolerance AND residual within the same KKT
        // tolerance — supplies the missing certificate without weakening
        // the envelope-theorem requirement. A residual above tolerance
        // can be a free Hessian-null gradient component, not an active
        // multiplier, so it must not be accepted by an objective-flatness
        // rule.
        //
        // Distinct from the strict path because the strict path is silent
        // on objective change;
        // distinct from the trust-region floor certificate at the head
        // of the cycle because that one fires only when the trust radius
        // has collapsed to its 1e-12 floor with all attempts rejected,
        // whereas this branch fires when the trust region is still open
        // but each accepted step is no longer producing detectable
        // objective progress.
        let objective_change = signed_obj_change.abs();
        if objective_change.is_finite() {
            geometric_tail_history.push_back(objective_change);
            while geometric_tail_history.len() > GEOMETRIC_TAIL_WINDOW {
                geometric_tail_history.pop_front();
            }
        }
        if objective_change <= objective_tol && residual <= residual_tol {
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | noise-floor KKT certificate: residual={:.3e} <= tol={:.3e}, |Δobjective|={:.3e} <= obj_tol={:.3e}",
                cycle,
                residual,
                residual_tol,
                objective_change,
                objective_tol,
            );
            finish_post_step_convergence!();
        }

        // Constrained-stationary certificate.
        //
        // The inner Newton system is `Hδ = -g`, solved over the
        // active-constraint-aware subspace (the QP step path).  When
        // the *unprojected* gradient `g` carries a large Lagrange-
        // multiplier component pointing into the constraint —
        // i.e. some β coordinates are pinned at the bound or against
        // the family's structural constraint surface — the linear
        // solve correctly DOES NOT try to eliminate that component,
        // because doing so would push β infeasibly.  The signature of
        // this state is precise and entirely local to the most recent
        // accepted step:
        //
        //   • `‖g + Hδ‖∞ / ‖g‖∞ ≥ 0.5` — the linear solve neutralised
        //     ≤ 50 % of g; the remainder is structurally outside the
        //     solver's range, i.e. it's a Lagrange multiplier of the
        //     active constraints, not a defect of the linear solve.
        //   • `|actual − pred| / max(|pred|, …) ≤ 1e-3` — the local
        //     quadratic Newton model agrees with the actual objective
        //     change to roundoff, so the Hessian and gradient are
        //     correct AT this β.  The "stuck" residual is not noise
        //     in the linearisation; it's a real multiplier.
        //   • `|Δobjective| ≤ objective_tol` — the objective has
        //     ceased moving meaningfully.
        //   • `|δ|∞ ≤ step_tol` — the accepted feasible Newton step is
        //     exhausted. Objective flatness alone is not a terminal
        //     signal on large survival fits: a step of O(1e-2..1e-1)
        //     can still continue reducing the KKT residual after the
        //     objective first crosses tolerance.
        //
        // Together these four are the rigorous certificate that
        // Newton has reached a constrained-stationary point: further
        // cycles would reproduce the same plateau (the diagnostic in
        // PIRLS/JN/math shows `‖g+Hδ‖/‖g‖` constant near 1 cycle
        // after cycle, the very signature this certificate names).
        //
        // The 0.5 threshold on `linearized_rel` is conservative —
        // an unconstrained Newton step has `linearized_rel ≈ 1e-12`;
        // a step deliberately constrained to a (k-1)-dim subspace
        // leaves the orthogonal Lagrange direction in the residual
        // and `linearized_rel ≈ |λ|/|g| > 0`, typically 0.9+ in
        // practice when the multiplier dominates.  Anything ≥ 0.5
        // is unambiguously in the constrained-stationary regime;
        // unconstrained Newton with `linearized_rel ≥ 0.5` would
        // have already failed the trust-region's scalar model test
        // and been rejected upstream.
        if let Some(math) = last_joint_math.as_ref() {
            let linearized_rel = math.linearized_rel();
            let scalar_model_relerr = math.scalar_model_relative_error();
            let geometric_tail_bound = if geometric_tail_history.len() == GEOMETRIC_TAIL_WINDOW {
                let values = geometric_tail_history.iter().copied().collect::<Vec<_>>();
                let mut max_ratio = 0.0_f64;
                let mut valid = true;
                for pair in values.windows(2) {
                    let prev = pair[0];
                    let next = pair[1];
                    if prev <= 0.0 || next < 0.0 || !prev.is_finite() || !next.is_finite() {
                        valid = false;
                        break;
                    }
                    let ratio = next / prev;
                    if !ratio.is_finite() || ratio >= 1.0 {
                        valid = false;
                        break;
                    }
                    max_ratio = max_ratio.max(ratio);
                }
                if valid {
                    Some(objective_change / (1.0 - max_ratio).max(1.0e-12))
                } else {
                    None
                }
            } else {
                None
            };
            let certificate_decision = constrained_stationary_certificate_decision(
                math,
                objective_change,
                objective_tol,
                step_tol,
                geometric_tail_bound,
                residual,
                residual_tol,
            );
            if !matches!(
                certificate_decision,
                ConstrainedStationaryCertificate::NotCandidate
            ) {
                // A multiplier/null-plateau diagnosis requires a
                // positive-semidefinite local mode. When this cycle's exact
                // returned-mode spectrum has resolvable negative curvature,
                // the residual is not trapped multiplier mass: the
                // Moré–Sorensen hard case has a certified descent direction.
                // Its accepted step moves beta after this spectrum was
                // assembled, so only the next cycle's fresh spectrum may
                // decide whether the new point is a local mode. Refusing here
                // after the short descent-history window mislabeled every
                // survival Matérn startup as a four-cycle "budget" exit while
                // the terminal evidence itself said strict saddle. Continue
                // the hard-case escape and re-certify curvature at the moved
                // coefficient state.
                if has_resolvable_negative_curvature {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {cycle:>3} |                              constrained-stationary plateau decision deferred: the exact                              pre-step spectrum has resolvable negative curvature, so the                              accepted hard-case escape must receive a fresh returned-mode                              curvature certificate next cycle"
                    );
                    continue;
                }
                // The `linearized_rel >= 0.5` signal is necessary but not
                // sufficient. It proves either (a) g carries a Lagrange
                // multiplier of an active constraint that the QP's active
                // set already represents — in which case the *projected*
                // residual is at tolerance — or (b) H is rank-deficient
                // in the direction of g, so Hδ ≈ 0 along the null
                // direction regardless of whether g is a multiplier or a
                // real defect. Case (b) is the survival marginal-slope
                // pathology at large scale: H σ_min ≈ 1e-12 and Newton
                // genuinely cannot move g, but the residual is NOT a
                // captured multiplier — it's an unresolved KKT defect in
                // the H-null subspace.
                //
                // The projected residual computed at the top of this
                // block (line ~12055) already subtracts the multiplier
                // mass of every row in `cached_active_sets`. If that
                // residual is at tolerance, case (a) holds and the
                // certificate is honest. If it's still orders of
                // magnitude above tolerance, case (b) holds: certifying
                // here would hand the unified evaluator a
                // `kkt_residual` with norm ≈ ‖g‖ which then gets
                // amplified by H⁻¹_proj in the cost/gradient IFT
                // corrections, contaminating the envelope formula and
                // triggering the "envelope-gradient consistency"
                // tripwire downstream. Bail with `converged = false` so
                // the outer optimizer rejects this ρ cleanly, exactly
                // as it would on any other non-converged inner exit.
                let cert_residual_factor = 1.0;
                if matches!(
                    certificate_decision,
                    ConstrainedStationaryCertificate::Accept
                ) {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary certificate: \
                         linear-solve neutralised {:.1}% of g (the remaining {:.1}% is a Lagrange multiplier \
                         of the active constraint set, not an unresolved gradient); \
                         scalar Newton model agrees with reality to relerr={:.3e} (Hessian+gradient are correct \
                         at this β); projected residual={:.3e} ≤ {:.1}×tol={:.3e} (multipliers captured by active set); \
                         |Δobjective|={:.3e}, geometric_tail_bound={:.3e}, obj_tol={:.3e}; further cycles cannot reduce the \
                         multiplier mass and would reproduce this plateau indefinitely; \
                         active-set multiplier mass will be projected out of the KKT residual \
                         before the outer IFT correction is assembled",
                        cycle,
                        (1.0 - linearized_rel) * 100.0,
                        linearized_rel * 100.0,
                        scalar_model_relerr,
                        residual,
                        cert_residual_factor,
                        cert_residual_factor * residual_tol,
                        objective_change,
                        geometric_tail_bound.unwrap_or(objective_change),
                        objective_tol,
                    );
                    finish_post_step_convergence!();
                }
                // Constrained exact-fixed-point acceptance (gam#797).
                //
                // We reach here only with the iterate ALREADY proven stationary
                // (objective + step exhausted, `linearized_rel >= 0.5` so the
                // residual is multiplier/null mass, `scalar_relerr <= 1e-3` so
                // the quadratic model is exact), the strict/range-space/noise
                // certificates having declined. For a CONSTRAINED block the
                // remaining residual can be a genuine active-constraint Lagrange
                // multiplier that the active-set QP under-identified (it reports
                // only rows it drove tight during a non-degenerate step, so a
                // monotone derivative-guard row tight at the optimum but never
                // explicitly stepped is missing), leaving the cone projection
                // unable to decompose `r = A_activeᵀ λ` and the residual stuck
                // far above tol on an iterate that is EXACTLY the constrained
                // optimum (the `active_set_incomplete` refusal; gam#797 survival
                // marginal/slope/time blocks).
                //
                // When (a) the joint Newton has reached a numerical FIXED POINT
                // — the accepted step and objective change are both at the
                // machine-epsilon floor relative to the iterate, so no further
                // progress is mathematically possible — (b) the local quadratic
                // model is exact (`scalar_relerr` tiny), and (c) the design
                // carries linear inequality constraints AND `H_pen` has NO
                // numerical null space (so the residual is an active-constraint
                // multiplier, NOT an H-null/rank-deficient defect, which the
                // range-space certificate above already handles), the iterate is
                // a bona fide constrained KKT point. The active-constraint
                // multiplier mass is projected out of the KKT residual by the
                // unified evaluator's active-constraint-aware IFT correction
                // before the envelope gradient, exactly as for an explicitly
                // captured multiplier, so certifying here is correct. Gated
                // strictly on a fixed point with no H-null, so a genuinely
                // non-converged or rank-deficient iterate is never accepted.
                let any_block_constrained = block_constraints.iter().any(|c| c.is_some());
                let beta_scale = states
                    .iter()
                    .flat_map(|s| s.beta.iter().copied())
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                let fixed_point_floor = 64.0 * f64::EPSILON * beta_scale;
                let objective_floor = 64.0 * f64::EPSILON * (1.0 + lastobjective.abs());
                // `step_at_eps_floor` records whether the accepted step also reached
                // its OWN machine-eps floor, used only to label the log line with
                // which stationarity witness fired (strict eps step vs the gam#2358
                // objective-eps / gauge-drift-step path below).
                let step_at_eps_floor =
                    accepted_step_inf.is_finite() && accepted_step_inf <= fixed_point_floor;
                // gam#2358: the constrained fixed-point certificate must not
                // additionally require the accepted STEP to reach its own machine-eps
                // floor `64·eps·|β|`. On a FLAT / gauge-drift constrained surface (the
                // coupled mean/log-σ/wiggle location-scale seed: `y ~ x`, noise `1`,
                // degree-3 monotone link wiggle) the joint Newton reaches a genuine
                // constrained KKT point — 4 active I-spline `γ≥0` rows, H_pen full
                // rank (nullity 0), the residual (≈10) a real active-constraint
                // Lagrange multiplier — with the OBJECTIVE already at its eps floor
                // (|Δobj|=1.4e-13 ≤ objective_floor=1.1e-12) and an exact model, but
                // the accepted step floors a HAIR above `64·eps·|β|` (7.99e-14 vs
                // 4.73e-14, ×1.69) EVERY cycle and `step_at_eps_floor` never latches,
                // so the iterate is refused as a phantom multiplier — fatally at the
                // seed. The step floors above `64·eps·|β|` precisely because it is
                // objective-flat gauge drift through the active-set QP + line search
                // (extra rounding beyond a single β update); it is irrelevant to
                // convergence once the objective is at its eps floor. Gate on
                // `objective_at_numerical_floor` and require the accepted step merely
                // small (`≤ step_tol`, the scale-aware stationarity gate the candidate
                // conditions above already established), NOT at its eps floor. This is
                // the exact treatment the UNCONSTRAINED model-stationary certificate
                // below already applies. Safety is unchanged: a still-DESCENDING
                // iterate has `|Δobj| > objective_floor` (fails the eps objective
                // floor) and never reaches here; an H-null/rank-deficient defect fails
                // the `hpen_nullity == 0` gate below (deferred to the range-space
                // certificate); and `linearized_rel ≥ 0.5` (candidacy) proves the
                // residual is constraint-normal multiplier mass, not resolvable
                // descent — so nothing genuinely non-converged is certified.
                let constrained_numerical_fixed_point =
                    crate::joint_newton::constrained_numerical_fixed_point_reached(
                        objective_change,
                        objective_floor,
                        scalar_model_relerr,
                        accepted_step_inf,
                        step_tol,
                    );
                if any_block_constrained && constrained_numerical_fixed_point {
                    // Materialize H_pen = H + S(λ) (+ model ridge) and count its
                    // numerical null space at the shared rank tolerance: nullity == 0
                    // ⇒ the stuck residual is NOT an H-null/rank-deficient defect
                    // (that case is handled by the range-space certificate above) but
                    // a genuine active-constraint multiplier.
                    let hpen_nullity = materialize_joint_hessian_source(
                        &joint_hessian_source,
                        total_p,
                        "constrained fixed-point nullity check",
                    )
                    .ok()
                    .map(|mut h_pen| {
                        let model_diagonal_ridge =
                            if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                                ridge
                            } else {
                                0.0
                            };
                        add_joint_penalty_to_matrix(
                            &mut h_pen,
                            &ranges,
                            &s_lambdas,
                            model_diagonal_ridge,
                            None,
                        );
                        symmetrize_dense_in_place(&mut h_pen);
                        symmetric_penalized_hessian_nullity(&h_pen)
                    })
                    .unwrap_or(None);
                    if hpen_nullity == Some(0) {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained fixed-point certificate ({}): \
                             |Δobjective|={:.3e} ≤ objective_floor={:.3e} (objective at machine-eps floor), accepted_step_inf={:.3e} (eps_floor={:.3e}, step_tol={:.3e}), \
                             scalar_relerr={:.3e}, linearized_rel={:.3e}; H_pen has no numerical null space so the \
                             residual={:.3e} is an active-constraint Lagrange multiplier (the QP under-identified the \
                             binding rows), projected out of the KKT residual by the active-constraint-aware IFT \
                             correction before the envelope gradient — the iterate is a constrained KKT point",
                            cycle,
                            if step_at_eps_floor {
                                "machine-eps step + objective fixed point"
                            } else {
                                "objective-eps fixed point, gauge-drift step (gam#2358)"
                            },
                            objective_change,
                            objective_floor,
                            accepted_step_inf,
                            fixed_point_floor,
                            step_tol,
                            scalar_model_relerr,
                            linearized_rel,
                            residual,
                        );
                        finish_post_step_convergence!();
                    }
                }
                // Still-converging guard (gam#787 duchon centers≥20). The
                // certificates above all declined, so the iterate would be
                // refused as a multiplier/null plateau. But the
                // `linearized_rel ≥ 0.5` + flat-objective signature that
                // routed us here ALSO holds for a slope block whose
                // objective is already at its Φ-bounded floor while the KKT
                // residual is still polishing by a STEADY geometric factor
                // each cycle. Refusing there rejects the seed a few cycles
                // short of `residual_tol` (→ outer seed-rejection → raise).
                // If the residual is in steady geometric descent over the
                // recent window, the direction is genuinely converging, not
                // plateaued: keep iterating (bounded by the inner cycle cap)
                // rather than refuse. The genuine plateau (flat/oscillating
                // residual above tol) fails this test and refuses as before.
                if residual_in_steady_geometric_descent(&residual_descent_history) {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | certificate declined but residual in steady geometric descent (history={:?}, residual={:.3e}, tol={:.3e}); continuing to convergence rather than refusing as a plateau",
                        cycle,
                        residual_descent_history,
                        residual,
                        residual_tol,
                    );
                    continue;
                }
                // EARLY-CYCLE CARVE-OUT (gam#826/#872). The phantom-multiplier
                // refusal asserts that the residual is a captured Lagrange
                // multiplier / H-null mass that Newton genuinely cannot move —
                // a claim that requires EVIDENCE of a plateau. The candidate
                // conditions above (objective + step exhausted, linearized_rel ≥
                // 0.5) are ALSO satisfied transiently when a single Newton step
                // is small because the augmented (Firth) curvature `H_Φ` is
                // legitimately large in the `∇Φ` direction at an oversmoothed
                // cycle-0 seed: the step `(H+Sλ+H_Φ)⁻¹(∇L−Sβ+∇Φ)` is tiny (high
                // curvature ⇒ short step) and ONE step undershoots the
                // nonquadratic Firth optimum, so `step_inf` and `|Δobj|` look
                // exhausted while the residual is still O(‖∇Φ‖) ≫ tol. Refusing
                // there at cycle 0 (no descent history yet) aborts the coupled
                // binomial location-scale / flexible-linkwiggle fit before the
                // inner has taken the handful of cycles it needs to walk the
                // curved Firth basin to its optimum. When the residual is still
                // ORDERS above tol and we lack a full descent window to prove a
                // genuine plateau, keep iterating — the inner cycle cap and the
                // residual-stall / trust-region-floor guards still bound the
                // loop and diagnose a true non-convergence. A genuine multiplier
                // plateau (residual flat across the window) is caught once the
                // history fills, exactly as before. The threshold is the same
                // `RESIDUAL_DESCENT_WINDOW` the descent test uses, so this only
                // defers the refusal until there is enough history to make it,
                // never weakens it.
                let residual_far_above_tol = residual.is_finite()
                    && residual_tol.is_finite()
                    && residual > cert_residual_factor * residual_tol;
                if residual_far_above_tol
                    && residual_descent_history.len() < RESIDUAL_DESCENT_WINDOW
                {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary refusal DEFERRED: residual={:.3e} ≫ tol={:.3e} but only {} descent samples (< {} window) — too early to prove a multiplier/null plateau vs a high-curvature Firth-basin transient; continuing",
                        cycle,
                        residual,
                        residual_tol,
                        residual_descent_history.len(),
                        RESIDUAL_DESCENT_WINDOW,
                    );
                    continue;
                }
                // UNCONSTRAINED MODEL-STATIONARY ACCEPTANCE (gam#826/#808/#715).
                //
                // The phantom-multiplier refusal asserts the residual is a
                // captured Lagrange multiplier of an active constraint that
                // the QP could not decompose. That diagnosis is categorically
                // IMPOSSIBLE when there is no active constraint at all: a
                // residual cannot be a phantom multiplier of a constraint that
                // does not exist. For a fully UNCONSTRAINED coupled fit
                // (multinomial softmax; the location-scale flat blocks) on a
                // near-flat Fisher surface (`diag(p)−ppᵀ → 0`, or the
                // high-curvature/low-curvature `log_sigma` block) the
                // Firth-augmented stationarity residual `‖∇L−Sβ+∇Φ‖` floors
                // LEGITIMATELY above `4·residual_tol`: the absolute curvature
                // is tiny so `residual_tol = inner_tol·(1+grad/pen/firth)` is
                // tiny too, yet the Newton/dogleg step exhausts before the
                // residual drops below that band — `residual_tol` is scaled by
                // the gradient magnitude and does not see the flat-Fisher
                // absolute-curvature floor. The well-conditioned spectrum keeps
                // the conditioning-keyed Levenberg gate (`COND_NEWTON_SAFETY`)
                // off, so neither LM nor the cond-armed dogleg engages, and
                // every seed is refused as `phantom_multiplier_with_well_
                // conditioned_H`.
                //
                // When the model itself certifies stationarity — the standard
                // trust-region "predicted decrease ≈ 0" criterion, here the
                // `at_numerical_fixed_point` flag (accepted step at the
                // machine-eps floor, |Δobj| at the eps floor, scalar model
                // exact to relerr ≤ 1e-3) — AND no further progress is being
                // made (the steady-geometric-descent test above declined) AND
                // we have a full descent window (the early-cycle deferral above
                // passed, so this is a proven plateau not a Firth-basin
                // transient), an unconstrained iterate is a bona fide
                // first-order optimum: the quadratic model says no step can
                // reduce the residual further, and there is no constraint whose
                // multiplier the residual could otherwise represent. The
                // residual that remains lives where the model is flat
                // (vanishing curvature), so it carries no `gᵀ∂β/∂ρ` envelope
                // contribution the outer IFT could not already neutralise
                // through its penalty-projected pseudo-inverse. Accept.
                //
                // This does NOT regress #729 (coupled Dirichlet): that fit
                // converges to a genuine `residual < residual_tol` and exits
                // via the strict KKT certificate long before this branch, and
                // even if reached it has a curved (non-flat) Fisher surface so
                // its model is not at a fixed point with a residual stuck above
                // tol. It does NOT mask a real non-convergence: a still-moving
                // iterate fails `at_numerical_fixed_point` (its step / |Δobj|
                // are above the eps floor), and a rank-deficient H-null defect
                // is the CONSTRAINED concern the fixed-point certificate above
                // already handles via its nullity check.
                // The certificate-candidate conditions that routed us into
                // this block already PROVE model stationarity for the
                // unconstrained case: `objective_exhausted` + `step_inf ≤
                // step_tol` (the model's minimizer is at this β), `scalar_relerr
                // ≤ 1e-3` (the quadratic model is exact), and `linearized_rel ≥
                // 0.5` (‖g+Hδ‖ ≈ ‖g‖, so `Hδ ≈ 0` — the residual lives in the
                // flat/near-null subspace of H, exactly a flat-Fisher direction
                // for an unconstrained fit). We do NOT additionally require the
                // far stricter machine-eps `at_numerical_fixed_point` here: on a
                // flat Fisher surface the dogleg keeps taking a small step at
                // the `step_tol` floor every cycle, so `accepted_step_inf` floors
                // a hair above `64·eps·|β|` and the eps-fixed-point flag never
                // sets even though the model is stationary. The `step_tol` floor
                // (`inner_tol·(1+|β|∞)`) is the principled stationarity gate; the
                // eps floor is for the constrained-multiplier certificate, where
                // a tighter proof is warranted because a wrong accept biases the
                // constraint-aware IFT kernel.
                let any_active_set_rows = cached_active_sets
                    .iter()
                    .any(|maybe| maybe.as_ref().is_some_and(|rows| !rows.is_empty()));
                let unconstrained_fit = !any_block_constrained && !any_active_set_rows;
                if unconstrained_fit {
                    log::info!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | unconstrained model-stationary certificate (gam#826/#808/#715): \
                         no active constraint (active_set_rows_total=0) so the residual={:.3e} cannot be a phantom multiplier; \
                         the iterate is a numerical fixed point (accepted_step_inf={:.3e}, |Δobjective|={:.3e}, scalar_relerr={:.3e}) \
                         on a flat Fisher surface where residual_tol={:.3e} sits below the absolute-curvature floor; \
                         linearized_rel={:.3e}, |Δobjective| exhausted and residual not in steady descent → genuine first-order optimum, accepting",
                        cycle,
                        residual,
                        accepted_step_inf,
                        objective_change,
                        scalar_model_relerr,
                        residual_tol,
                        linearized_rel,
                    );
                    finish_post_step_convergence!();
                }
                // Structured per-block + per-spectrum refusal report.
                // The legacy one-line refusal log printed only aggregate
                // numbers (linearized_rel, scalar_relerr, residual,
                // |Δobj|) and was not actionable on models with many
                // blocks: it could not identify WHICH smooth carried
                // the unresolved mass, nor whether H_pen was genuinely
                // rank-deficient (the "polynomial null space slipped
                // past absorption" pathology). Cost: one dense
                // materialize + symmetric eigh on H_pen at this β,
                // sub-millisecond for typical p, executed once per
                // refusal (the loop breaks immediately after).
                let report = compute_kkt_refusal_report(
                    cycle,
                    &states,
                    specs,
                    &s_lambdas,
                    &ranges,
                    cached_joint_gradient.as_ref(),
                    &cached_active_sets,
                    &block_constraints,
                    Some(&joint_hessian_source),
                    total_p,
                    ridge,
                    options.ridge_policy,
                    accepted_step_inf,
                    step_inf,
                    joint_trust_radius,
                    residual_tol,
                    objective_tol,
                    step_tol,
                    objective_change,
                    residual,
                    Some(&math),
                );
                log::warn!(
                    "{}",
                    report.format_structured_log(cert_residual_factor * residual_tol)
                );
                last_kkt_refusal_report = Some(report);
                converged = false;
                break;
            }
        }

        // INVESTIGATION NOTE — do NOT soft-accept here.
        //
        // The outer objective is V(ρ) = f(β*(ρ), ρ), where β*(ρ)
        // satisfies g(β*,ρ)=∇_β f=0.  The envelope/IFT gradient used
        // by the outer optimizer is
        //
        //   dV/dρ_j = ∂f/∂ρ_j
        //
        // only at g=0.  At a non-stationary β, the actual chain rule is
        //
        //   d f(β(ρ),ρ)/dρ_j = ∂f/∂ρ_j + gᵀ ∂β/∂ρ_j.
        //
        // A soft certificate based only on small Δf discards the second
        // term without proving it is small.  The projected pseudo-inverse
        // in the outer trace path removes null-space components of g, but
        // any range-space component still contributes gᵀ∂β/∂ρ and gives
        // ARC/BFGS a biased outer gradient.  The `[PIRLS/JN/math]` line
        // above now prints the actual Newton identity:
        //
        //   old_kkt = ‖g‖∞,
        //   linearized_next = ‖g + Hδ‖∞ = ‖Hδ-rhs‖∞,
        //   new_kkt = ‖g(β+δ)‖∞,
        //   scalar_model relerr = |actual-pred|/max(1,|pred|).
        //
        // That is the proof surface. The diagnostic reports the measured
        // linear solve residual, post-step KKT residual, scalar model
        // error, and step sizes directly; downstream analysis should use
        // those numbers rather than this solver attaching labels.

        // Residual-stall early-exit. The strict and noise-floor
        // certificates above require the KKT residual to land within
        // a small multiple of residual_tol. On survival marginal-slope
        // at large scale the residual oscillates in a band that is
        // orders of magnitude above tol without trending down while
        // the unconstrained proposal has |prop|∞ in the 10³–10⁶ range,
        // the TR clamps it, and each clamped step moves β by O(1)
        // without driving ‖∇L − Sβ‖∞ closer to KKT.
        //
        // Spending the remaining cycle budget on this pattern hits
        // inner_max_cycles "non-converged", which then routes the
        // outer optimizer through the first-order bridge with a stale
        // same-ρ inner mode and a gradient of magnitude 10⁷ that kills
        // BFGS line search at iter 0 (the failure mode pinned in the
        // commit messages of 6578e884 and 1c181d1f).
        //
        // Track the best residual seen so far and the number of
        // cycles since any meaningful improvement (≥ 10 % drop). Once
        // the inner has burned at least RESIDUAL_STALL_MIN_CYCLES
        // without progress and the accepted step kept hitting the
        // trust-region clamp, return `converged = false` with the current
        // finite β. A stalled residual above the strict KKT tolerance is
        // not converted into convergence by a pointwise Hessian rank test.
        if residual.is_finite() {
            if residual < RESIDUAL_STALL_IMPROVEMENT_FACTOR * best_residual_seen {
                best_residual_seen = residual;
                cycles_since_residual_improved = 0;
                tr_clamped_during_stall = false;
            } else {
                cycles_since_residual_improved = cycles_since_residual_improved.saturating_add(1);
                if last_accepted_hit_joint_trust_boundary {
                    tr_clamped_during_stall = true;
                }
            }
            // Trailing window of post-step residuals for the deterministic
            // slow-geometric-rate stall projection (gam#979 survival). Kept
            // at length ≤ LINEAR_RATE_WINDOW+1 so the front is the residual
            // exactly LINEAR_RATE_WINDOW cycles back.
            if residual_rate_history.len() > LINEAR_RATE_WINDOW {
                residual_rate_history.pop_front();
            }
            residual_rate_history.push_back(residual);
        }
        // THE SURROGATE MODEL HAS A BUDGET, AND IT IS THIS SOLVE'S (gam#2612).
        //
        // The stall guards below defer their projection cap to
        // `max(inner_max_cycles − cycle, LINEAR_RATE_PROJECTION_CAP)`, i.e. they
        // are allowed to say "reachable" against the historic floor of 100 even
        // when this evaluation only has 24 cycles left. That is right for a
        // STALL verdict — the floor exists so a solve is never killed earlier
        // than it historically was — and wrong for the question asked here,
        // which is not "should I give up" but "can the model I am using finish
        // inside the budget I actually have". Measured on the penguins screening
        // evaluations, whose budget is 64: 36 of them ground to `64/64` with
        // `jeffreys_completion_calls = 0` and the residual still improving on
        // the last cycle — never stalled, never certified, simply out of cycles
        // under a model that was not the objective's Hessian.
        //
        // So the same trailing-window projection is asked against the REAL
        // remaining budget, and when the answer is "no" the model is upgraded
        // rather than the solve abandoned. This costs nothing on a solve that
        // is converging: a quadratic endgame projects a handful of cycles and
        // never fills the window in the first place.
        if head_jeffreys_term.is_some()
            && !jeffreys_completion_endgame
            && residual.is_finite()
            && residual > residual_tol
            && residual_rate_history.len() > LINEAR_RATE_WINDOW
            && let Some(&oldest) = residual_rate_history.front()
            && gam_solve::loop_guard::slow_geometric_rate_exceeds_projection_cap(
                residual,
                oldest,
                LINEAR_RATE_WINDOW,
                residual_tol,
                inner_max_cycles.saturating_sub(cycle),
            )
        {
            jeffreys_completion_endgame = true;
            clear_stall_evidence_collected_under_the_previous_model(
                &mut best_residual_seen,
                &mut cycles_since_residual_improved,
                &mut tr_clamped_during_stall,
                &mut residual_descent_history,
                &mut residual_rate_history,
                &mut merit_window,
                &mut geometric_tail_history,
            );
            log::info!(
                "[PIRLS/joint-Newton model] cycle {:>3} | the divided-difference Jeffreys \
                 surrogate cannot reach tol inside this solve's own remaining budget \
                 ({} cycle(s) of {inner_max_cycles}): residual={:.3e} (tol={:.3e}) at \
                 ~{:.4}×/cycle over the last {} cycles. Arming the exact second-order \
                 completion and clearing the cross-cycle evidence collected under the surrogate",
                cycle,
                inner_max_cycles.saturating_sub(cycle),
                residual,
                residual_tol,
                (residual / oldest).powf(1.0 / (LINEAR_RATE_WINDOW as f64)),
                LINEAR_RATE_WINDOW,
            );
            continue 'joint_newton_cycles;
        }
        // Trailing window of the Φ-augmented merit, kept in lockstep with
        // `residual_rate_history` so its front is the merit exactly
        // LINEAR_RATE_WINDOW cycles back. Powers the merit-descent veto on
        // the two residual-trend stall guards below (gam#1607 wiggle).
        if lastobjective.is_finite() {
            if merit_window.len() > LINEAR_RATE_WINDOW {
                merit_window.pop_front();
            }
            merit_window.push_back(lastobjective);
        }
        // Is the merit still descending robustly across the trailing
        // window? A residual-trend stall verdict is premature while it is:
        // the line search is making real progress on the actual objective,
        // and the KKT residual's transient non-monotonicity (the wiggle
        // gauge-null re-anchoring, gam#1607) is not evidence of being
        // stuck. "Robustly" = the merit dropped by more than the
        // accumulated objective tolerance over the window — i.e. by more
        // than the convergence machinery would call flat — so a merit that
        // has genuinely plateaued (the #979 survival stall, ~1e-5 steps ⇒
        // merit flat to f64) does NOT clear the bar and the guard fires as
        // before. The per-cycle `objective_tol` (relative, scale-aware) is
        // the natural unit; require the window drop to exceed
        // LINEAR_RATE_WINDOW × objective_tol so a window of merely
        // tolerance-scale dithering counts as flat.
        let merit_still_descending_over_window = || -> bool {
            if merit_window.len() <= LINEAR_RATE_WINDOW {
                // Not enough history yet to judge a window-scale trend;
                // don't veto on partial information (the guards have their
                // own RESIDUAL_STALL_MIN_CYCLES floor anyway).
                return false;
            }
            let (Some(&oldest), Some(&newest)) = (merit_window.front(), merit_window.back()) else {
                return false;
            };
            if !oldest.is_finite() || !newest.is_finite() {
                return false;
            }
            let drop = oldest - newest;
            drop > (LINEAR_RATE_WINDOW as f64) * objective_tol
        };
        // Deterministic tol-reachability exemption for the two counter-based
        // stall guards below (gam#979 CTN endgame). The ≥10%-drop
        // "improvement" test cannot distinguish a genuinely flat residual
        // from a slow monotone endgame descent closing on tol: a residual
        // walking 1.06×tol → tol over ~a dozen cycles never produces a
        // single 10% drop, so `cycles_since_residual_improved` climbs and
        // the guards (or the merit-veto cap expiring) kill a solve that is
        // cycles away from certifying — measured on the #979 CTN smoke,
        // where one run certifies at cycle ~122 (r=9.05e-3 ≤ tol=9.56e-3)
        // and another is killed at cycle 123 by the veto-cap expiry with
        // the residual 6% above tol and still descending. The trailing
        // -window geometric projection already trusted by the slow-rate
        // guard is the honest discriminator: if it projects reaching
        // `residual_tol` within LINEAR_RATE_PROJECTION_CAP cycles the
        // residual IS trending to KKT and a stall verdict is false, so the
        // guards defer to the projection guard (which fires precisely when
        // reachability fails). The genuine stall shapes keep exiting: a
        // flat or rising window projects `unreachable`, and the historic
        // #979 hang (~0.99×/cycle orders above tol) projects far past the
        // cap. Deterministic: cycle indices and residual ratios only; the
        // `inner_max_cycles` ceiling remains the hard backstop.
        // Effective cap defers to the caller's remaining `inner_max_cycles`
        // budget (floored at the historic 100) — see the const doc.
        let effective_projection_cap = inner_max_cycles
            .saturating_sub(cycle)
            .max(LINEAR_RATE_PROJECTION_CAP);
        let residual_tol_reachable_within_cap = residual_rate_history.len() > LINEAR_RATE_WINDOW
            && residual_rate_history.front().is_some_and(|&oldest| {
                !gam_solve::loop_guard::slow_geometric_rate_exceeds_projection_cap(
                    residual,
                    oldest,
                    LINEAR_RATE_WINDOW,
                    residual_tol,
                    effective_projection_cap,
                )
            });
        if cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
            && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
            && tr_clamped_during_stall
            && !residual_tol_reachable_within_cap
        {
            // gam#2612: a stall verdict taken on the divided-difference
            // Jeffreys surrogate is a statement about the MODEL, not about the
            // problem. See the invariant at `jeffreys_completion_endgame`.
            if head_jeffreys_term.is_some() && !jeffreys_completion_endgame {
                jeffreys_completion_endgame = true;
                clear_stall_evidence_collected_under_the_previous_model(
                    &mut best_residual_seen,
                    &mut cycles_since_residual_improved,
                    &mut tr_clamped_during_stall,
                    &mut residual_descent_history,
                    &mut residual_rate_history,
                    &mut merit_window,
                    &mut geometric_tail_history,
                );
                log::info!(
                    "[PIRLS/joint-Newton model] cycle {:>3} | the residual-stall guard would \
                     concede on a step model that is not the objective's Hessian \
                     (jeffreys_completion_calls={}, residual={:.3e} against a completion band of \
                     {:.3e}); arming the exact Jeffreys second-order completion and clearing the \
                     cross-cycle evidence collected under the surrogate",
                    cycle,
                    jeffreys_completion_calls,
                    residual,
                    JEFFREYS_COMPLETION_RESIDUAL_BAND * residual_tol,
                );
                continue 'joint_newton_cycles;
            }
            let last_math_summary = last_joint_math
                .as_ref()
                .map(|math| {
                    format!(
                        "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                        math.old_kkt_inf,
                        math.linearized_next_kkt_inf,
                        math.actual_reduction,
                        math.predicted_reduction,
                        math.trust_ratio,
                        math.scalar_model_relative_error(),
                        math.step_inf,
                        math.proposal_inf,
                    )
                })
                .unwrap_or_else(|| "last_newton_math=<none>".to_string());
            log::warn!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | residual-stall early-exit: residual={:.3e} relative_stationarity={:.3e} (scale={:.3e}, inner_tol={:.3e}) best_seen={:.3e} no_improve_cycles={} accepted_step_inf={:.3e} trust_radius={:.3e} block_stationarity_inf={:?} {}; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation before inner_max_cycles.",
                cycle,
                residual,
                gam_problem::relative_stationarity(residual, stationarity_scale),
                stationarity_scale,
                inner_tol,
                best_residual_seen,
                cycles_since_residual_improved,
                accepted_step_inf,
                joint_trust_radius,
                block_stationarity_norms,
                last_math_summary,
            );
            // Record a structured KKT-refusal report at the stall iterate so
            // the bubbled IntegrationFailed error carries the per-block
            // residual breakdown + H_pen spectrum instead of the opaque
            // "no joint Newton math snapshot" string (gam#979/#1040). This is
            // the dominant non-convergence exit for the survival
            // marginal-slope monotone-cone DGP; without a report the cause of
            // the abort is invisible past serialization.
            cycles_done = cycle + 1;
            let report = compute_kkt_refusal_report(
                cycle,
                &states,
                specs,
                &s_lambdas,
                &ranges,
                cached_joint_gradient.as_ref(),
                &cached_active_sets,
                &block_constraints,
                Some(&joint_hessian_source),
                total_p,
                ridge,
                options.ridge_policy,
                accepted_step_inf,
                step_inf,
                joint_trust_radius,
                residual_tol,
                objective_tol,
                step_tol,
                objective_change,
                residual,
                last_joint_math.as_ref(),
            );
            last_kkt_refusal_report = Some(report);
            converged = false;
            break;
        }

        // KKT convergence: small residual plus EITHER a small
        // Newton step (tight quadratic-rate convergence, lets β
        // polish to machine precision), confirmed stagnation
        // (`accepted_step_inf <= step_tol` AND `objective_change
        // <= objective_tol`, the rank-deficient null-mode case),
        // OR a stricter stationarity certificate where both the
        // residual and objective change are an additional factor of
        // `inner_tol` below their scale-aware tolerances. The last
        // branch is deliberately stricter than the public tolerance:
        // it handles machine-precision null directions where β can
        // still move by about `step_tol` but the KKT residual and
        // objective are already over-polished. Using objective
        // stagnation alone is not sufficient; the residual guard is
        // what preserves first-order correctness.
        let superconverged_residual_tol = inner_tol * residual_tol;
        let superconverged_objective_tol = inner_tol * objective_tol;
        let superconverged_stationarity = residual <= superconverged_residual_tol
            && objective_change <= superconverged_objective_tol;
        if residual <= residual_tol
            && (step_inf <= step_tol
                || (accepted_step_inf <= step_tol && objective_change <= objective_tol)
                || superconverged_stationarity)
        {
            log::info!(
                "[JN-EXIT] cycle={cycle} reason=strict_kkt residual={residual:.3e} residual_tol={residual_tol:.3e} obj_change={objective_change:.3e} objective_tol={objective_tol:.3e} accepted_step_inf={accepted_step_inf:.3e} step_tol={step_tol:.3e}",
            );
            // This branch certifies on `residual ≤ residual_tol`; record it
            // so the terminal line reports the finite certified residual
            // rather than the `inf` stall sentinel (#1040 truthfulness).
            if residual.is_finite() {
                min_certified_residual = min_certified_residual.min(residual);
            }
            finish_post_step_convergence!();
        }
        // Carry the KKT-stationarity / objective-stagnation signals
        // into the next cycle so the line-search-failure path above
        // can recognise a true KKT optimum on a rank-deficient null
        // mode. See that path for the full rationale.
        last_cycle_residual_below_tol = residual <= residual_tol;
        last_cycle_obj_change_below_tol = objective_change <= objective_tol;

        // Flat-residual stall early-exit (gam#1040/#979/#370/#859).
        //
        // The `tr_clamped_during_stall` residual-stall exit above only fires
        // when the accepted step kept hitting the trust-region boundary. A
        // distinct but equally terminal stall reaches neither it nor any
        // acceptance certificate: the KKT residual stops improving (no ≥10%
        // drop for the full `RESIDUAL_STALL_NO_IMPROVE_CYCLES` window) while
        // the accepted steps stay strictly INSIDE the trust region (so
        // `tr_clamped_during_stall` never latches) and the objective keeps
        // drifting just above `objective_tol` (so the relative-objective
        // plateau exit's flat streak never completes). This is the measured
        // "[joint-newton-tr] cycles 1000+" wall on the binomial location-scale
        // / bms-flex / CTN inner solves: without an exit the loop grinds the
        // remaining budget to `inner_loop_hard_ceiling` on every outer
        // ρ-evaluation, then hands the outer optimizer a non-converged result
        // anyway.
        //
        // Reaching this point means every acceptance certificate above already
        // DECLINED this cycle — the residual is above `residual_tol`, its
        // range-space component is above tolerance (so the iterate is NOT
        // stationary on the identifiable subspace), and there is no
        // constrained-multiplier signature. The honest action mirrors the
        // `tr_clamped` `converged=false` exit: stop and return the current
        // finite β as NON-converged so the outer optimizer rejects this ρ
        // cleanly. This is purely a termination/perf guard — it certifies
        // nothing (`converged=false`) and so cannot bias the envelope
        // gradient; it only rejects the same non-optimum sooner. The `≥10%
        // drop` reset of `cycles_since_residual_improved` keeps a
        // geometrically-descending solve (residual dropping by a steady factor
        // each cycle) from ever reaching the window — only a genuinely flat
        // residual does.
        if residual.is_finite()
            && residual > residual_tol
            && cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
            && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
            && !residual_tol_reachable_within_cap
            && (!merit_still_descending_over_window()
                || cycles_since_residual_improved >= RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES)
        {
            log::warn!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | flat-residual stall early-exit (gam#1040/#979): residual={:.3e} (tol={:.3e}) best_seen={:.3e} stalled {} cycles with steps inside the trust region (tr_clamped={}) and no acceptance certificate satisfied; the residual is neither trending toward KKT nor stationary on the identifiable subspace, so returning unconverged with finite β instead of grinding to inner_max_cycles={}.",
                cycle,
                residual,
                residual_tol,
                best_residual_seen,
                cycles_since_residual_improved,
                tr_clamped_during_stall,
                inner_max_cycles,
            );
            cycles_done = cycle + 1;
            converged = false;
            break;
        }

        // Slow-geometric-rate stall early-exit (gam#979 survival marginal-slope).
        //
        // Distinct from the flat-residual exit above (residual NOT improving
        // for the no-improve window) and the Newton-decrement certificate
        // (decrement ≤ objective_tol). Here the residual IS descending, just
        // geometrically and far too slowly to reach tol in a practical cycle
        // count — the survival marginal-slope oversmoothed-ρ endgame (stiff
        // penalized Hessian → ~1e-5 Newton steps far inside a large trust
        // radius → residual ~0.99×/cycle). Project, from the trailing
        // window's geometric rate, the additional cycles to reach
        // `residual_tol`; if that exceeds LINEAR_RATE_PROJECTION_CAP the
        // ρ-evaluation cannot finish in a practical budget, so return the
        // finite β as NON-converged and let the outer optimizer reject this
        // ρ cleanly instead of grinding ~10³ cycles to inner_max_cycles (the
        // #979 "hang"). DETERMINISTIC: cycle indices and residual ratios
        // only, no wall-clock (cf. the no-wall-clock note below). Certifies
        // nothing (`converged=false`) so it cannot bias the envelope
        // gradient; it only rejects an impractical-to-finish iterate sooner.
        // A still-progressing (quadratic / fast-geometric) solve reaches tol
        // in a handful of cycles and never fills the window, so this never
        // fires on a healthy fit.
        if residual.is_finite()
            && residual > residual_tol
            && cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
            && residual_rate_history.len() > LINEAR_RATE_WINDOW
            && (!merit_still_descending_over_window()
                || cycle + 1 >= RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES)
        {
            let oldest = *residual_rate_history.front().expect(
                "the guard above requires len > LINEAR_RATE_WINDOW, so the history is non-empty",
            );
            // Single source of truth for the slow-geometric-rate projection
            // (gam#979): deterministic cycle-count projection, no wall-clock.
            // The cap defers to the caller's remaining budget (floored at the
            // historic 100) so a solve that can reach tol within its own
            // `inner_max_cycles` is not cut off — see the const doc.
            let effective_projection_cap = inner_max_cycles
                .saturating_sub(cycle)
                .max(LINEAR_RATE_PROJECTION_CAP);
            let too_slow = gam_solve::loop_guard::slow_geometric_rate_exceeds_projection_cap(
                residual,
                oldest,
                LINEAR_RATE_WINDOW,
                residual_tol,
                effective_projection_cap,
            );
            if too_slow {
                // gam#2612: same invariant as the residual-stall guard — a
                // rate measured under the surrogate model is a fact about the
                // surrogate.
                if head_jeffreys_term.is_some() && !jeffreys_completion_endgame {
                    jeffreys_completion_endgame = true;
                    clear_stall_evidence_collected_under_the_previous_model(
                        &mut best_residual_seen,
                        &mut cycles_since_residual_improved,
                        &mut tr_clamped_during_stall,
                        &mut residual_descent_history,
                        &mut residual_rate_history,
                        &mut merit_window,
                        &mut geometric_tail_history,
                    );
                    log::info!(
                        "[PIRLS/joint-Newton model] cycle {:>3} | the slow-geometric-rate \
                         projection would concede on a step model that is not the objective's \
                         Hessian (jeffreys_completion_calls={}, residual={:.3e} against a \
                         completion band of {:.3e}); arming the exact Jeffreys second-order \
                         completion and clearing the cross-cycle evidence collected under the \
                         surrogate",
                        cycle,
                        jeffreys_completion_calls,
                        residual,
                        JEFFREYS_COMPLETION_RESIDUAL_BAND * residual_tol,
                    );
                    continue 'joint_newton_cycles;
                }
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | slow-geometric-rate stall early-exit (gam#979): residual={:.3e} (tol={:.3e}) descending at ~{:.4}×/cycle over the last {} cycles — projected >{} more cycles to reach tol; the residual is converging but far too slowly to finish in a practical budget (the survival marginal-slope oversmoothed-ρ endgame), so returning unconverged with finite β instead of grinding to inner_max_cycles={}.",
                    cycle,
                    residual,
                    residual_tol,
                    (residual / oldest).powf(1.0 / (LINEAR_RATE_WINDOW as f64)),
                    LINEAR_RATE_WINDOW,
                    effective_projection_cap,
                    inner_max_cycles,
                );
                cycles_done = cycle + 1;
                converged = false;
                break;
            }
        }

        // NOTE: there is deliberately NO wall-clock-driven "adaptive
        // early-exit" here. A convergence verdict that fires when a cycle's
        // wall-clock happens to fall below a fraction of a running EMA is
        // non-deterministic — under CPU contention (a parallel sweep) the
        // same fit accepts at a different iterate than it does run alone,
        // which cascades into a different accepted outer state (gam#979's
        // sequential-versus-parallel instability). It also
        // accepts iterates up to 10× outside the real KKT/objective
        // tolerance, biasing the REML/LAML criterion the inner residual
        // feeds. Convergence is certified ONLY by the mathematical tests
        // above (KKT residual / Newton step / objective change at their
        // scale-aware tolerances); whether convergence is *reachable within
        // the cycle budget* is judged by the deterministic descent-rate
        // guard alongside the residual-stall detector above.
    }

    if converged {
        let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
        let joint_constraints =
            assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
        // A full-space PSD test is exact for the unconstrained CTN mode.
        // Constrained modes certify on the active-face tangent (the
        // critical-cone surrogate under strict complementarity) — the same
        // Z the terminal determinant uses, so a mode this certificate
        // accepts can never fail the downstream SPD logdet on curvature
        // grounds. The same M_true certificate includes Jeffreys curvature;
        // Jeffreys families are never exempt from second-order stationarity.
        if !returned_mode_curvature_certified {
            let mode_active_block = if joint_constraints.is_some() {
                // Certify on the full numerically-tight face, not only the
                // QP-recorded rows — see widen_active_sets_to_tight_face.
                let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
                    &block_constraints,
                    &states,
                    &cached_active_sets,
                )?;
                crate::blockwise_solve::assemble_active_constraint_block(
                    &block_constraints,
                    &tight_sets,
                    &ranges,
                    total_p,
                )
            } else {
                None
            };
            let certificate = exact_joint_mode_curvature_certificate(
                family,
                &states,
                specs,
                options,
                &ranges,
                &s_lambdas,
                joint_mode_diagonal_ridge,
                joint_bundle,
                total_p,
                mode_active_block.as_ref(),
            )?;
            if certificate.jeffreys_completion_assembled {
                jeffreys_completion_calls += 1;
            }
            let has_negative_curvature = certificate.has_resolvable_negative_curvature();
            let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
            let numerical_floor = certificate.numerical_floor;
            cached_joint_workspace = certificate.workspace;
            if has_negative_curvature {
                return Err(CustomFamilyError::trial_point(format!(
                    "joint Newton tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={:.6e} < -floor={:.6e}; an indefinite coefficient point cannot define a Laplace mode",
                    minimum_whitened_eigenvalue, numerical_floor,
                )));
            } else {
                log::info!(
                    "[PIRLS/joint-Newton mode certificate] returned beta certified from fresh exact curvature: lambda_min={:.6e}, floor={:.6e}",
                    minimum_whitened_eigenvalue,
                    numerical_floor,
                );
            }
        }
    }

    // Explicit terminal verdict for the joint-Newton inner solve.
    //
    // The per-cycle `[PIRLS/JN] cyc=N/MAX … resid=… (tol=…)` line prints
    // the KKT/step/objective gaps at every cycle but never states which
    // criterion *terminated* the loop, so the final visible line on a
    // budget-exhausted solve looks identical to an ordinary mid-run cycle
    // (gam#744). A reader scanning a sweep log cannot tell a fit that
    // reached a stationary point from one that simply ran out of cycles
    // with the residual still orders of magnitude above tolerance and only
    // the objective stalled. Emit one authoritative line, on every exit
    // path, naming the terminating condition: `converged` is the honest
    // status the result carries downstream, `budget_exhausted` distinguishes
    // "ran the full cap" from an early certificate/divergence exit, and the
    // residual/step/objective stall flags say *why*. A budget-exhausted,
    // non-converged exit is logged at WARN so it is impossible to miss even
    // when per-cycle INFO is filtered out; a clean convergence is INFO.
    {
        let budget_exhausted = cycles_done >= inner_max_cycles;
        // Hard convergence-truthfulness invariant (#1040): a converged exit
        // is, by construction, certified on a finite stationarity residual
        // ≤ tol (every `converged = true` path above is gated on a finite
        // residual / range-space check and records it into
        // `min_certified_residual`). If — through any path — `converged` is
        // set without a finite certified residual on record, the solve has
        // NOT actually certified convergence; reporting `converged=true …
        // best_residual_inf=inf` is the self-contradicting status #1040
        // flags. The honest status is then non-converged: downgrade it so
        // the outer REML/LAML evaluation rejects this ρ rather than
        // consuming a phantom optimum certified on no finite residual.
        if !gam_solve::loop_guard::inner_convergence_is_truthful(converged, min_certified_residual)
        {
            log::warn!(
                "[PIRLS/joint-Newton terminal] cycle {cycles_done}/{inner_max_cycles}: a converged \
                 exit fired without any finite certified stationarity residual on record \
                 (min_certified_residual is non-finite) — this would report \
                 converged=true with best_residual_inf=inf, a convergence-truthfulness \
                 violation (#1040). Downgrading to non-converged so the outer optimizer \
                 rejects this evaluation."
            );
            converged = false;
        }
        let terminator = if converged {
            "KKT/certificate-converged"
        } else if budget_exhausted {
            "budget-exhausted (max cycles reached)"
        } else {
            "early-exit non-converged (divergence/stall guard)"
        };
        // `solve_wall` (whole inner-solve elapsed) + `cycles` make the
        // per-solve cost explicit on ONE line: gam#979's "outer
        // multiplication" candidate is read off by counting these terminal
        // lines across a repro and summing their wall-times, and the
        // overhead candidate by comparing `solve_wall / cycles` against the
        // [joint-newton-tr] phase splits. Together with the per-cycle
        // `per_block_resid` (which block stalls) and the existing TR line
        // (ρ gain-ratio + decision: model infidelity vs TR throttling), a
        // single RUST_LOG=info run separates all four #979 candidates.
        //
        // Report `min_certified_residual` (the smallest stationarity residual
        // the solve actually computed) rather than the stall-tracker
        // `best_residual_seen`: the latter is only written at the post-step
        // residual site, so a head-of-cycle / pre-line-search certificate exit
        // (cycle-0 KKT exit on already-stationary data) left it at the sentinel
        // `inf` and the line read `converged=true … best_residual_inf=inf`, a
        // self-contradicting status (#1040 inner-report truthfulness). A
        // converged exit always certified on a finite residual ≤ tol, so the
        // reported residual is finite whenever `converged` (every converged=true
        // path is gated on a `≤ tol` check of a residual recorded above).
        let reported_residual_below_tol = last_cycle_residual_below_tol
            || (converged && min_certified_residual <= last_residual_tol);
        let verdict = format!(
            "[PIRLS/joint-Newton terminal] converged={} terminator={} cycles={}/{} \
             jeffreys_completion_calls={} \
             solve_wall={:.3}s best_residual_inf={:.3e} (tol={:.3e}) last_residual_below_tol={} \
             last_obj_change_below_tol={} objective={:.6e}; this is the status the inner \
             solve reports to the outer REML/LAML evaluation — a non-converged exit \
             (residual ≫ tol with only the objective stalled) is rejected, not accepted",
            converged,
            terminator,
            cycles_done,
            inner_max_cycles,
            jeffreys_completion_calls,
            inner_started.elapsed().as_secs_f64(),
            min_certified_residual,
            last_residual_tol,
            reported_residual_below_tol,
            last_cycle_obj_change_below_tol,
            lastobjective,
        );
        if converged {
            log::info!("{verdict}");
        } else {
            log::warn!("{verdict}");
        }
    }

    // If joint Newton converged, skip the blockwise loop entirely.
    if converged {
        // The accepted-step cache is keyed by the exact coefficient bits.
        // Nothing between the accepted step and this terminal branch mutates
        // beta, so a hit is the authoritative Jeffreys derivative artifact at
        // the returned mode. A miss deliberately falls through to the normal
        // computation; approximate/stale reuse is never allowed here.
        let final_beta_key = flatten_state_betas(&states, specs);
        let final_jeffreys_cache = jeffreys_triple_cache
            .as_ref()
            .filter(|(beta_key, _, _)| beta_cache_keys_match_bitwise(beta_key, &final_beta_key));
        let penalty_value = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let active_constraints = {
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            // The LAML logdet must project onto the tangent of the FULL
            // numerically-tight face at the returned mode — the same face
            // the curvature certificate used. Projecting only past the
            // QP-recorded rows leaves near-tight rows' normals inside the
            // tangent, and the genuine curvature normal to them reads as a
            // phantom indefiniteness that aborts the ρ-evaluation (the
            // measured survival marginal-slope "no Laplace mode" terminal).
            let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
                &block_constraints,
                &states,
                &cached_active_sets,
            )?;
            assemble_active_constraint_block(&block_constraints, &tight_sets, &ranges, total_p)
                .map(std::sync::Arc::new)
        };
        let (block_logdet_h, block_logdet_s) = if product.requires_laplace_artifacts() {
            let (h, s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
                final_jeffreys_cache.map(|(_, _, hphi)| hphi),
                active_constraints.as_deref(),
            )?;
            (Some(h), Some(s))
        } else {
            (None, None)
        };
        // The IFT/outer KKT residual must be the AUGMENTED stationarity
        // `∇L − Sβ + ∇Φ` the inner Newton actually drove to zero — NOT the bare
        // `∇L − Sβ`. With the Firth term armed, `∇L − Sβ = −∇Φ` at the
        // converged β, so the bare residual's null-space component equals ∇Φ
        // (O(‖∇Φ‖), e.g. 2.49 for the coupled Dirichlet). The outer evaluator's
        // range-projected IFT validity gate (`projected_into_reduced_range`)
        // then sees that ‖∇Φ‖ of "unresolved mass outside the reduced range"
        // and rejects EVERY seed at outer startup validation ("no candidate
        // seeds passed", gam#729/#715). Folding ∇Φ into the gradient makes the
        // residual the genuinely-near-zero augmented stationarity the inner
        // certified, so the gate passes. No-op when the term is
        // condition-gated/unavailable (∇Φ=0).
        let augmented_joint_gradient: Option<Array1<f64>> = match cached_joint_gradient.as_ref() {
            Some(gradient) => match final_jeffreys_cache {
                Some((_, grad_phi, _)) if grad_phi.len() == gradient.len() => {
                    Some(gradient + grad_phi)
                }
                _ => match joint_jeffreys_subspace.as_ref() {
                    Some(z_joint) => match custom_family_joint_jeffreys_term(
                        family, &states, specs, &ranges, z_joint,
                    )? {
                        Some((_phi, grad_phi, _hphi)) if grad_phi.len() == gradient.len() => {
                            Some(gradient + &grad_phi)
                        }
                        _ => None,
                    },
                    None => None,
                },
            },
            None => None,
        };
        let ift_gradient = augmented_joint_gradient
            .as_ref()
            .or(cached_joint_gradient.as_ref());
        let joint_penalty_score = joint_penalty_stationarity_score(options, specs, &states);
        let kkt_residual = exact_newton_joint_kkt_residual_for_ift_from_cached_gradient(
            family,
            specs,
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            Some(cached_active_sets.as_slice()),
            ift_gradient,
            joint_penalty_score.as_ref(),
        )?;
        let kkt_residual =
            require_projected_kkt_residual(kkt_residual, "joint-Newton converged exit")?;
        // Thread the cert tolerance + free subspace rank through to
        // the unified evaluator's certificate so the outer
        // optimiser's InnerStatus carrier sees honest numbers
        // instead of NaN / None.
        let active_set_rows_total: usize = cached_active_sets
            .iter()
            .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
            .sum();
        let free_rank_at_cert = total_p.saturating_sub(active_set_rows_total);
        let kkt_residual = kkt_residual.with_metadata(last_residual_tol, free_rank_at_cert);
        // Build the joint active-constraint block for the unified
        // evaluator's constraint-aware kernel
        // `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. Returns `None` when
        // the family has no declared inequality constraints, or when
        // no rows are currently active at the cert point; in either
        // case the consumer-side `with_active_constraints` helper
        // degrades back to the bare penalty-projected pseudo-inverse.
        // The joint score is reloaded immediately after every accepted
        // step and beta is restored before every rejected one, so the
        // vector held here belongs to the states being returned. Bind
        // the operating point with it so the consumer checks that
        // rather than trusting the loop ordering (gam#2474).
        let retained_likelihood_score =
            cached_joint_gradient
                .as_ref()
                .map(|score| TerminalLikelihoodScore {
                    beta: TerminalLikelihoodScore::joint_beta(&states),
                    score: score.clone(),
                });
        return Ok(BlockwiseInnerResult {
            block_states: states,
            terminal_working_sets: cached_eval
                .as_ref()
                .map(|eval| eval.blockworking_sets.clone()),
            terminal_likelihood_score: retained_likelihood_score,
            active_sets: normalize_active_sets(cached_active_sets),
            log_likelihood: current_log_likelihood,
            penalty_value,
            cycles: cycles_done,
            converged,
            terminal_convergence_state,
            block_logdet_h,
            block_logdet_s,
            s_lambdas,
            joint_workspace: cached_joint_workspace.clone(),
            kkt_residual: Some(kkt_residual),
            active_constraints,
            objective_state,
        });
    }
    if cycles_done >= inner_max_cycles {
        if !converged {
            // Engine-level diagnostic. Emit measured quantities only:
            // objective movement, coefficient scale, per-block dimensions,
            // per-block β and gradient scales, the unprojected stationarity
            // norm at exit, the Hessian source shape, and the last accepted
            // Newton identity diagnostics. The outer error path has no
            // access to these internals, so this line is the complete
            // numerical record needed to decide the next fix.
            let block_grad_norms: Vec<f64> = match cached_joint_gradient.as_ref() {
                Some(joint_grad) => {
                    let mut acc = 0usize;
                    states
                        .iter()
                        .map(|s| {
                            let n = s.beta.len();
                            let end = (acc + n).min(joint_grad.len());
                            let nrm = if acc < end {
                                joint_grad
                                    .slice(ndarray::s![acc..end])
                                    .iter()
                                    .map(|x: &f64| x.abs())
                                    .fold(0.0_f64, f64::max)
                            } else {
                                f64::NAN
                            };
                            acc += n;
                            nrm
                        })
                        .collect()
                }
                None => vec![f64::NAN; states.len()],
            };
            let block_widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
            let block_beta_inf: Vec<f64> = states
                .iter()
                .map(|s| s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max))
                .collect();
            let descent_total = initial_joint_objective - lastobjective;
            let beta_inf_final = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let block_diag_default =
                !family.exact_newton_joint_hessian_beta_dependent() && specs.len() >= 2;
            let exit_unprojected_kkt_inf = cached_joint_gradient
                .as_ref()
                .and_then(|joint_grad| {
                    exact_newton_joint_stationarity_vector_from_gradient(
                        joint_grad,
                        &states,
                        specs,
                        &s_lambdas,
                        ridge,
                        options.ridge_policy,
                    )
                    .ok()
                })
                .map(|residual| {
                    residual
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max)
                })
                .unwrap_or(f64::NAN);
            let last_math_summary = last_joint_math
                .as_ref()
                .map(|math| {
                    format!(
                        "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                        math.old_kkt_inf,
                        math.linearized_next_kkt_inf,
                        math.actual_reduction,
                        math.predicted_reduction,
                        math.trust_ratio,
                        math.scalar_model_relative_error(),
                        math.step_inf,
                        math.proposal_inf,
                    )
                })
                .unwrap_or_else(|| "last_newton_math=<none>".to_string());
            log::warn!(
                "[PIRLS/joint-Newton] cycle={} budget-exhausted without KKT: objective_start={:.6e} objective_end={:.6e} objective_drop={:+.3e} beta_inf={:.3e} exit_unprojected_kkt_inf={:.3e} total_p={} total_n={} block_widths={:?} block_beta_inf={:?} block_grad_inf={:?} block_diag_hessian_default={} {}; rejecting this outer REML/LAML evaluation",
                cycles_done,
                initial_joint_objective,
                lastobjective,
                descent_total,
                beta_inf_final,
                exit_unprojected_kkt_inf,
                total_p,
                total_joint_n,
                block_widths,
                block_beta_inf,
                block_grad_norms,
                block_diag_default,
                last_math_summary,
            );
            {
                // Budget exhaustion is a failed *inner mode at this rho*, not
                // malformed user input.  Propagate it as a finite
                // `converged=false` inner result so the outer objective can
                // reject/back off this smoothing point (the same contract used
                // by non-exact families) instead of bubbling an
                // `InvalidInput` through the custom-family string boundary.
                // This matters on the survival/location-scale flat baseline
                // valley: some startup rho candidates are numerically
                // non-certifying, but neighbouring rho values are perfectly
                // fit-able, so aborting the whole fit prevents the optimizer
                // from ever leaving the valley.
                let block_diag = if let Some(report) = last_kkt_refusal_report.as_ref() {
                    report.format_bubbled_error()
                } else {
                    let block_constraints =
                        collect_block_linear_constraints(family, &states, specs)?;
                    let report = compute_kkt_refusal_report(
                        cycles_done,
                        &states,
                        specs,
                        &s_lambdas,
                        &ranges,
                        cached_joint_gradient.as_ref(),
                        &cached_active_sets,
                        &block_constraints,
                        None,
                        total_p,
                        ridge,
                        options.ridge_policy,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        last_residual_tol,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        exit_unprojected_kkt_inf,
                        last_joint_math.as_ref(),
                    );
                    report.format_bubbled_error()
                };
                log::warn!(
                    "coupled exact-joint inner solve exhausted the joint Newton budget without KKT convergence after {cycles_done} cycle(s) — {block_diag}; returning a non-converged inner mode for outer-rho rejection"
                );
            }
        }
        let penalty_value = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let active_constraints = {
            let local_ranges = block_param_ranges(specs);
            let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            // Full numerically-tight face, not only the QP-recorded rows —
            // see widen_active_sets_to_tight_face (gam#979).
            let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
                &block_constraints,
                &states,
                &cached_active_sets,
            )?;
            assemble_active_constraint_block(
                &block_constraints,
                &tight_sets,
                &local_ranges,
                local_total_p,
            )
            .map(std::sync::Arc::new)
        };
        let (block_logdet_h, block_logdet_s) =
            if converged && product.requires_laplace_artifacts() {
                let (h, s) = blockwise_logdet_terms_with_workspace(
                    family,
                    specs,
                    &mut states,
                    block_log_lambdas,
                    options,
                    cached_joint_workspace.clone(),
                    None,
                    active_constraints.as_deref(),
                )?;
                (Some(h), Some(s))
            } else {
                (None, None)
            };
        // The joint score is reloaded immediately after every accepted
        // step and beta is restored before every rejected one, so the
        // vector held here belongs to the states being returned. Bind
        // the operating point with it so the consumer checks that
        // rather than trusting the loop ordering (gam#2474).
        let retained_likelihood_score =
            cached_joint_gradient
                .as_ref()
                .map(|score| TerminalLikelihoodScore {
                    beta: TerminalLikelihoodScore::joint_beta(&states),
                    score: score.clone(),
                });
        return Ok(BlockwiseInnerResult {
            block_states: states,
            terminal_working_sets: cached_eval
                .as_ref()
                .map(|eval| eval.blockworking_sets.clone()),
            terminal_likelihood_score: retained_likelihood_score,
            active_sets: normalize_active_sets(cached_active_sets),
            log_likelihood: current_log_likelihood,
            penalty_value,
            cycles: cycles_done,
            converged,
            terminal_convergence_state,
            block_logdet_h,
            block_logdet_s,
            s_lambdas,
            joint_workspace: cached_joint_workspace.clone(),
            kkt_residual: None,
            active_constraints,
            objective_state,
        });
    }
    {
        // An early exit from an exact joint path is a non-certifying inner
        // mode at the current rho, not invalid input. The selected joint
        // solver is authoritative even for a single tensor block: falling
        // through to coordinate iteration would silently switch algorithms
        // after a failed certificate, discard its active-set provenance,
        // and grind a second cycle budget before reaching the same outer-rho
        // rejection. Families whose objective is deliberately separable are
        // routed to blockwise before this joint path starts. Return the
        // current finite iterate with `converged=false` so the outer
        // optimizer can reject this rho and continue.
        let block_diag = last_kkt_refusal_report
            .as_ref()
            .map(KktRefusalReport::format_bubbled_error)
            .unwrap_or_else(|| {
                "structured KKT refusal report unavailable: no joint Newton math snapshot"
                    .to_string()
            });
        log::warn!(
            "coupled exact-joint inner solve exited the joint Newton path before convergence — {block_diag}; returning a non-converged inner mode for outer-rho rejection"
        );
        let penalty_value = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let active_constraints = {
            let local_ranges = block_param_ranges(specs);
            let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            // Full numerically-tight face, not only the QP-recorded rows —
            // see widen_active_sets_to_tight_face (gam#979).
            let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
                &block_constraints,
                &states,
                &cached_active_sets,
            )?;
            assemble_active_constraint_block(
                &block_constraints,
                &tight_sets,
                &local_ranges,
                local_total_p,
            )
            .map(std::sync::Arc::new)
        };
        // The joint score is reloaded immediately after every accepted
        // step and beta is restored before every rejected one, so the
        // vector held here belongs to the states being returned. Bind
        // the operating point with it so the consumer checks that
        // rather than trusting the loop ordering (gam#2474).
        let retained_likelihood_score =
            cached_joint_gradient
                .as_ref()
                .map(|score| TerminalLikelihoodScore {
                    beta: TerminalLikelihoodScore::joint_beta(&states),
                    score: score.clone(),
                });
        return Ok(BlockwiseInnerResult {
            block_states: states,
            terminal_working_sets: cached_eval
                .as_ref()
                .map(|eval| eval.blockworking_sets.clone()),
            terminal_likelihood_score: retained_likelihood_score,
            active_sets: normalize_active_sets(cached_active_sets),
            log_likelihood: current_log_likelihood,
            penalty_value,
            cycles: cycles_done,
            converged: false,
            terminal_convergence_state,
            block_logdet_h: None,
            block_logdet_s: None,
            s_lambdas,
            joint_workspace: cached_joint_workspace.clone(),
            kkt_residual: None,
            active_constraints,
            objective_state,
        });
    }
}
