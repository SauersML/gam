//! Certified termination (#968): ONE exhaustion/stagnation policy for
//! every damped inner loop.
//!
//! # The bug genus this kills
//!
//! Every hang in the tracker's history (#874, #789, #683, #744, the
//! survival-AFT cluster, #826's 42-minute frozen-residual stall) traces to
//! the same structural flaw: termination safety was a per-branch,
//! hand-replicated convention. The #874 postmortem is the canonical
//! specimen — the LM *gain-reject* branch lacked the exhaustion guard its
//! sibling *screening-reject* branch in the SAME file already had. Guard
//! drift between sibling branches is the control-flow twin of the
//! objective↔gradient desync class, and the cure is the same: a single
//! source of truth that branches consume and cannot locally re-derive.
//!
//! # The policy pieces
//!
//! [`madsen_can_retry`] / [`madsen_retry_exhausted`] own the damped-retry
//! exhaustion question for Madsen-style Levenberg–Marquardt loops: a retry
//! is alive while the damping is finite and below [`MADSEN_DAMPING_CAP`],
//! and dead once attempts run out or damping leaves that window. Both
//! engines (reweight.rs Madsen-LM and the custom_family.rs spectral
//! Newton) must answer this question through these functions — never
//! through a local predicate.
//!
//! [`IterationBound`] and [`RejectEscalator`] are the two *distinct*
//! safety mechanisms of an unbounded damped-retry loop, kept as two types
//! on purpose. The bound owns the per-iteration hard count: it ticks once
//! at the top of EVERY pass — including `continue` paths that neither
//! accept a step nor reach a reject ritual (Fisher fallback, special
//! cases) — and is the net that makes an unbounded `loop {}` safe. The
//! escalator owns the geometric damping discipline applied on REJECTS
//! only. A single type coupling "count++" to "reject" would either
//! double-count iterations or silently assume every non-accepting pass
//! reaches a reject ritual — the exact unbounded-loop hole the guard
//! exists to close (see the #968 thread's design note).
//!
//! [`FlatStreak`] owns the consecutive-window discipline every stagnation
//! detector shares: a streak that grows on "flat" readings, resets on
//! recovery, and fires once it spans the window. Loops that own a
//! scale-aware flatness predicate of their own (the custom_family
//! joint-Newton objective-flat counter, the blockwise frozen-loglik
//! divergence detector) consume it directly — they answer the question
//! attempt caps cannot see: a loop that still "makes progress" every
//! iteration but whose MERIT is frozen. #744 ran to cycle 1199/1200 at a
//! flat residual; #826 burned a CI timeout on a frozen joint residual. The
//! caller feeds its descent quantity (penalized NLL, residual norm, |g|)
//! through its own flatness predicate once per iteration; the streak
//! reports a plateau once flat readings span a consecutive window — long
//! before any iteration cap.
//!
//! # Verdicts, not panics
//!
//! Exhaustion is an escalation event: the consuming loop converts
//! [`LoopVerdict::Plateaued`] / [`LoopVerdict::Exhausted`] into its
//! honest terminal status (`StalledAtValidMinimum`,
//! `LmStepSearchExhausted`, …) and unwinds. Never a hang, never a panic,
//! never a silent wrong answer.
//!
//! # Migration map (each step deleted a hand-rolled guard)
//!
//! 1. (done) reweight.rs `lm_can_retry`/`lm_retry_exhausted` local fns +
//!    the local `LM_MAX_LAMBDA` const deleted; call sites consume this
//!    module's policy.
//! 2. (done) The 7 copies of the reweight.rs reject ritual
//!    (`loop_lambda *= factor; factor *= 2.0; continue`) collapsed onto
//!    [`RejectEscalator::escalate`], and the per-iteration hard count
//!    moved into [`IterationBound`], so neither discipline can drift
//!    per-branch.
//! 3. (done) custom_family.rs: the joint-Newton objective-flat counter
//!    and the blockwise frozen-loglik divergence streak both ride
//!    [`FlatStreak`] — the #826-class exit discipline now lives here, not
//!    in per-loop counters. The richer certificate machinery those loops
//!    layer on top (geometric-tail bound, clamped-step side condition)
//!    stays local: it is *policy about what counts as flat*, which the
//!    loops rightly own; the streak/window discipline is what must not
//!    fork.
//! 4. (dropped) Terminal-verdict reporting into heartbeat scopes: the
//!    `[JN-EXIT]`/`[PIRLS]` per-exit log lines already name why a loop
//!    ended; a parallel verdict channel in the process monitor would be
//!    redundant global state.

/// Damping ceiling for Madsen-style LM retries. Beyond this the proposed
/// step is numerically a zero step — retrying cannot make progress, so the
/// retry chain is declared dead. (Moved verbatim from reweight.rs, where it
/// was a file-local convention; see module docs for why it must be shared.)
pub const MADSEN_DAMPING_CAP: f64 = 1e12;

/// Default consecutive-window length for a [`FlatStreak`] stagnation
/// detector: how many successive flat readings must accumulate before the
/// loop is declared plateaued. Two is the established in-tree streak
/// convention (reweight.rs soft-acceptance) — one noisy reading can fake a
/// plateau, two consecutive cannot — plus one for the headroom a merit that
/// is genuinely creeping (not frozen) needs to escape.
pub const PLATEAU_DEFAULT_WINDOW: usize = 3;

/// Is a damped retry still alive at this damping level?
#[inline]
pub fn madsen_can_retry(damping: f64) -> bool {
    damping.is_finite() && damping < MADSEN_DAMPING_CAP
}

/// Has the retry chain exhausted its budget — by attempt count or by the
/// damping leaving the productive window?
#[inline]
pub fn madsen_retry_exhausted(damping: f64, attempts: usize, max_attempts: usize) -> bool {
    attempts >= max_attempts || !damping.is_finite() || damping > MADSEN_DAMPING_CAP
}

/// Terminal verdict of a guarded loop. `Continue` is the only
/// non-terminal answer; the two terminal verdicts are ESCALATION events
/// the consumer must convert into an honest status, never swallow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopVerdict {
    Continue,
    /// The merit stream is frozen: stop and report the current iterate as
    /// the honest answer (StalledAtValidMinimum if KKT-near, else a named
    /// stall) instead of grinding out the remaining budget (#744, #826).
    Plateaued,
    /// Attempts or damping window exhausted (#874's missing branch guard).
    Exhausted,
}

/// Consecutive-flatness streak: the window discipline shared by every
/// stagnation detector in the tree. The caller owns the flatness
/// predicate (scale-aware objective tolerance, frozen log-likelihood,
/// sub-tolerance relative improvement, …); this type owns the part that
/// historically forked per loop — grow on flat, reset on recovery, fire
/// once the streak spans the window, and keep firing while it persists.
#[derive(Clone, Debug)]
pub struct FlatStreak {
    window: usize,
    streak: usize,
}

impl FlatStreak {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            streak: 0,
        }
    }

    /// Record one pre-judged flatness reading; returns the current verdict.
    pub fn note(&mut self, flat: bool) -> LoopVerdict {
        if flat {
            self.streak += 1;
            if self.streak >= self.window {
                return LoopVerdict::Plateaued;
            }
        } else {
            self.streak = 0;
        }
        LoopVerdict::Continue
    }

    /// Hard reset, e.g. after a non-finite merit re-baselines the stream.
    pub fn reset(&mut self) {
        self.streak = 0;
    }

    /// Current consecutive-flat count (diagnostic; the verdict is the
    /// contract).
    pub fn streak(&self) -> usize {
        self.streak
    }
}

/// Per-iteration hard bound for a damped retry loop: the net that makes
/// an unbounded `loop {}` safe. Tick it once at the top of EVERY pass —
/// accepted, rejected, or any `continue` path that reaches neither — and
/// ask [`IterationBound::exhausted_at`] wherever the loop's exhaustion
/// question is posed. Created fresh per outer iteration.
#[derive(Clone, Debug)]
pub struct IterationBound {
    used: usize,
    max: usize,
}

impl IterationBound {
    pub fn new(max: usize) -> Self {
        Self {
            used: 0,
            max: max.max(1),
        }
    }

    /// Count one loop pass. Top-of-loop, unconditionally.
    pub fn tick(&mut self) {
        self.used += 1;
    }

    /// Passes counted so far (diagnostics: `last_step_halving`, logs).
    pub fn used(&self) -> usize {
        self.used
    }

    /// The configured cap (diagnostics).
    pub fn max(&self) -> usize {
        self.max
    }

    /// Has the pass count alone exhausted the budget?
    pub fn count_exhausted(&self) -> bool {
        self.used >= self.max
    }

    /// The single exhaustion question: count OR damping window
    /// ([`madsen_retry_exhausted`], answered from owned state).
    pub fn exhausted_at(&self, damping: f64) -> bool {
        madsen_retry_exhausted(damping, self.used, self.max)
    }

    /// [`IterationBound::exhausted_at`] as a verdict, for consumers that
    /// speak [`LoopVerdict`].
    pub fn verdict_at(&self, damping: f64) -> LoopVerdict {
        if self.exhausted_at(damping) {
            LoopVerdict::Exhausted
        } else {
            LoopVerdict::Continue
        }
    }
}

/// Initial damping multiplier on the first rejection of an iteration.
/// Doubles on every further rejection (geometric escalation), reaching
/// [`MADSEN_DAMPING_CAP`] from λ = 1 in ~12 rejections — the established
/// reweight.rs schedule, now owned here.
pub const MADSEN_INITIAL_REJECT_FACTOR: f64 = 2.0;

/// Geometric damping escalator for one reject chain
/// (Madsen–Nielsen–Tingleff eq 3.16: the multiplier starts at 2 and
/// doubles on every rejection, so successive bumps are ×2, ×4, ×8, …).
/// Owns the factor and the reject count as one indivisible discipline —
/// no branch can bump the damping without advancing the schedule, the
/// drift mode behind #874. Deliberately does NOT own the per-iteration
/// count; that is [`IterationBound`]'s job (see module docs for why the
/// two must not be one type).
#[derive(Clone, Debug)]
pub struct RejectEscalator {
    factor: f64,
    rejects: usize,
}

impl Default for RejectEscalator {
    fn default() -> Self {
        Self::new()
    }
}

impl RejectEscalator {
    pub fn new() -> Self {
        Self {
            factor: MADSEN_INITIAL_REJECT_FACTOR,
            rejects: 0,
        }
    }

    /// Record a rejection: bumps the damping and advances the geometric
    /// schedule in one indivisible step.
    pub fn escalate(&mut self, damping: &mut f64) {
        *damping *= self.factor;
        self.factor *= 2.0;
        self.rejects += 1;
    }

    /// Restart the schedule — the problem changed under the chain (e.g. a
    /// Fisher fallback swapped the Hessian curvature), so the trajectory
    /// begins anew. Pairs with the caller resetting its damping baseline.
    pub fn restart(&mut self) {
        self.factor = MADSEN_INITIAL_REJECT_FACTOR;
        self.rejects = 0;
    }

    /// Rejections recorded since construction/restart (diagnostics).
    pub fn rejects(&self) -> usize {
        self.rejects
    }
}

/// Convergence-truthfulness invariant for an inner-solve terminal verdict
/// (gam#1040).
///
/// An inner Newton/PIRLS solve may only report `converged = true` if it
/// actually certified a stationarity point on a FINITE residual. A
/// certificate exit that fires on a cycle where the head-of-cycle KKT norm was
/// non-finite (so the running `min_certified_residual` is left at its `inf`
/// sentinel) would otherwise emit `converged=true … best_residual_inf=inf` — a
/// self-contradicting status: a convergence claim with no finite residual
/// behind it. This predicate is the single source of truth for that gate:
/// `converged` survives iff a finite certified residual is on record. When it
/// returns `false` while the solver believed it converged, the caller must
/// downgrade to non-converged so the outer optimizer rejects the evaluation
/// rather than consuming a phantom optimum.
#[inline]
pub fn inner_convergence_is_truthful(converged: bool, min_certified_residual: f64) -> bool {
    !converged || min_certified_residual.is_finite()
}

/// Deterministic slow-geometric-rate stall predicate (gam#979 survival
/// marginal-slope hang).
///
/// The survival marginal-slope oversmoothed-ρ endgame produces a stiff
/// penalized Hessian (penalty dominates, eigenvalues ~1e6) whose Newton steps
/// are ~1e-5 far INSIDE a large trust radius, so the inner KKT residual
/// descends geometrically but very slowly (~0.99×/cycle, halving only every
/// ~80 cycles). That is neither divergence nor a flat stall: the residual is
/// genuinely shrinking, just far too slowly to reach `residual_tol` in a
/// practical cycle count — so the flat-residual no-improve guard never latches
/// (the residual clears its 10% bar every ~12 cycles) and the loop grinds ~10³
/// cycles at ~p³ each, the measured #979 "hang".
///
/// Given the residual `window_oldest` cycles `window_cycles` ago and the
/// `current` residual, this projects — from the per-cycle geometric rate
/// `(current/window_oldest)^(1/window_cycles)` — how many additional cycles
/// reaching `residual_tol` would take, and returns `true` when that exceeds
/// `projection_cap` (i.e. the ρ-evaluation cannot finish in a practical
/// budget). It is FULLY DETERMINISTIC: cycle indices and residual ratios only,
/// no wall-clock. It also returns `true` when the window shows no net
/// geometric progress at all (rate ≥ 1, or the window did not shrink), which
/// likewise cannot reach tol.
///
/// A healthy (quadratically / fast-geometrically converging) solve reaches tol
/// in a handful of cycles and never fills the window, and even when it does the
/// projected remaining cycles are tiny, so this never fires on it. The caller
/// uses a `true` verdict to stop with the current finite β as `converged=false`
/// so the outer optimizer rejects this ρ and moves on; it certifies nothing and
/// so cannot bias the envelope gradient.
#[inline]
pub fn slow_geometric_rate_exceeds_projection_cap(
    current: f64,
    window_oldest: f64,
    window_cycles: usize,
    residual_tol: f64,
    projection_cap: usize,
) -> bool {
    if window_cycles == 0 {
        return false;
    }
    if !current.is_finite() || current <= residual_tol {
        // Either non-finite (a different guard owns that) or already at/under
        // tol (the convergence certificate owns that): not a slow-rate stall.
        return false;
    }
    if !window_oldest.is_finite() || window_oldest <= 0.0 || current >= window_oldest {
        // No net geometric progress across the whole window: cannot reach tol.
        return true;
    }
    let rate = (current / window_oldest).powf(1.0 / (window_cycles as f64));
    if !rate.is_finite() || rate >= 1.0 {
        return true;
    }
    let projected_cycles = (residual_tol / current).ln() / rate.ln();
    projected_cycles.is_finite() && projected_cycles > projection_cap as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #874 regression shape: a reject storm must reach the damping
    /// ceiling in a bounded number of escalations no matter which branch
    /// asks — the escalator owns the schedule, the predicates own the
    /// window.
    #[test]
    fn reject_storm_exhausts_in_bounded_steps() {
        let mut esc = RejectEscalator::new();
        let mut damping = 1.0;
        let mut steps = 0usize;
        while madsen_can_retry(damping) {
            esc.escalate(&mut damping);
            steps += 1;
            assert!(steps <= 64, "escalation must reach the damping cap");
        }
        // Geometric doubling of the factor reaches 1e12 in ~9 escalations.
        assert!(steps <= 16, "escalation took {steps} steps");
        assert_eq!(esc.rejects(), steps);
        assert!(madsen_retry_exhausted(damping, 0, usize::MAX));
    }

    /// The design split the #968 thread demanded: a loop pass that never
    /// reaches a reject ritual (Fisher fallback / special-case `continue`)
    /// still burns the iteration budget, because the bound ticks at the
    /// top of every pass — independent of the escalator.
    #[test]
    fn continue_paths_without_rejects_still_exhaust_the_bound() {
        let mut bound = IterationBound::new(5);
        let esc = RejectEscalator::new();
        let damping = 1e-6; // benign forever: only the count can kill it
        let mut passes = 0usize;
        while !bound.exhausted_at(damping) {
            bound.tick();
            passes += 1;
            assert!(passes <= 5, "bound must stop a reject-free spin");
            // No escalate(): this pass `continue`d past every reject site.
        }
        assert_eq!(passes, 5);
        assert_eq!(esc.rejects(), 0, "no reject was ever recorded");
        assert_eq!(bound.verdict_at(damping), LoopVerdict::Exhausted);
    }

    /// And the dual: escalations do NOT advance the iteration bound on
    /// their own — collapsing the rituals onto the escalator must not
    /// double-count attempts against the per-iteration budget.
    #[test]
    fn escalations_do_not_double_count_iterations() {
        let mut bound = IterationBound::new(10);
        let mut esc = RejectEscalator::new();
        let mut damping = 1.0;
        bound.tick();
        for _ in 0..3 {
            esc.escalate(&mut damping);
        }
        assert_eq!(bound.used(), 1);
        assert_eq!(esc.rejects(), 3);
        assert!(!bound.count_exhausted());
    }

    #[test]
    fn restart_rewinds_the_geometric_schedule() {
        let mut esc = RejectEscalator::new();
        let mut damping = 1.0;
        esc.escalate(&mut damping); // ×2
        esc.escalate(&mut damping); // ×4
        assert_eq!(damping, 8.0);
        esc.restart();
        assert_eq!(esc.rejects(), 0);
        let mut fresh = 1.0;
        esc.escalate(&mut fresh);
        assert_eq!(
            fresh, MADSEN_INITIAL_REJECT_FACTOR,
            "schedule restarts at ×2"
        );
    }

    /// The streak discipline alone (caller-owned flatness predicate, the
    /// custom_family consumption shape): grows on flat, resets on
    /// recovery, fires at the window and keeps firing while flat.
    #[test]
    fn flat_streak_pins_the_window_discipline() {
        let mut streak = FlatStreak::new(3);
        assert_eq!(streak.note(true), LoopVerdict::Continue); // 1
        assert_eq!(streak.note(true), LoopVerdict::Continue); // 2
        assert_eq!(streak.note(false), LoopVerdict::Continue); // reset
        assert_eq!(streak.streak(), 0);
        assert_eq!(streak.note(true), LoopVerdict::Continue); // 1
        assert_eq!(streak.note(true), LoopVerdict::Continue); // 2
        assert_eq!(streak.note(true), LoopVerdict::Plateaued); // 3 fires
        assert_eq!(streak.note(true), LoopVerdict::Plateaued); // persists
        assert_eq!(streak.streak(), 4);
    }

    /// A certificate exit must never report `converged=true` while the only
    /// residual on record is the non-finite `inf` sentinel — the gam#1040
    /// inner-report truthfulness violation. The predicate downgrades exactly
    /// that case and leaves every genuinely-certified exit untouched.
    #[test]
    fn inner_convergence_truthfulness_rejects_converged_with_nonfinite_residual() {
        // converged with a finite certified residual: honest, survives.
        assert!(inner_convergence_is_truthful(true, 8.0e-6));
        assert!(inner_convergence_is_truthful(true, 0.0));
        // converged with NO finite certified residual (the cycle-1 certificate
        // exit symptom: best_residual_inf=inf): a truthfulness violation.
        assert!(!inner_convergence_is_truthful(true, f64::INFINITY));
        assert!(!inner_convergence_is_truthful(true, f64::NAN));
        // non-converged exits are always truthful regardless of the residual
        // sentinel — the report says "not converged", no contradiction.
        assert!(inner_convergence_is_truthful(false, f64::INFINITY));
        assert!(inner_convergence_is_truthful(false, 1.0e-3));
    }

    /// The shared predicates pin the exact reweight.rs semantics they
    /// replaced (finite + strictly-below cap to retry; count OR window
    /// exit to exhaust).
    #[test]
    fn policy_predicates_pin_the_reweight_semantics() {
        assert!(madsen_can_retry(1e11));
        assert!(!madsen_can_retry(MADSEN_DAMPING_CAP));
        assert!(!madsen_can_retry(f64::INFINITY));
        assert!(madsen_retry_exhausted(1.0, 5, 5));
        assert!(madsen_retry_exhausted(f64::NAN, 0, 5));
        assert!(madsen_retry_exhausted(1e13, 0, 5));
        assert!(!madsen_retry_exhausted(1.0, 4, 5));
    }

    /// gam#979 survival marginal-slope: the slow-geometric-rate stall guard must
    /// TERMINATE the inner joint-Newton in a bounded number of cycles on the
    /// oversmoothed-ρ endgame (a residual crawling down by a fixed small factor
    /// ~0.99×/cycle that would otherwise grind ~10³ cycles to the budget — the
    /// measured hang) WITHOUT firing on a healthy fast-geometric solve. This
    /// replays the production loop's window bookkeeping (the trailing window of
    /// the last `LINEAR_RATE_WINDOW` post-step residuals, the guard armed only
    /// after `MIN_CYCLES`) over a deterministic residual stream and asserts a
    /// finite, bounded exit cycle — an iteration-count assertion, not a
    /// wall-clock threshold.
    #[test]
    fn slow_geometric_stall_guard_terminates_in_bounded_cycles_979() {
        // Mirror the production constants in inner_blockwise_fit.rs.
        const LINEAR_RATE_WINDOW: usize = 16;
        const LINEAR_RATE_PROJECTION_CAP: usize = 100;
        const RESIDUAL_STALL_MIN_CYCLES: usize = 40;
        // A representative inner cycle budget; the guard must exit FAR below it.
        const INNER_BUDGET: usize = 1000;
        let residual_tol = 1e-6_f64;

        // Replay the production window: a VecDeque holding at most
        // LINEAR_RATE_WINDOW+1 residuals (front = residual LINEAR_RATE_WINDOW
        // cycles back), the guard armed only at/after MIN_CYCLES and once the
        // window is full.
        fn run_stream(
            per_cycle_rate: f64,
            residual_tol: f64,
            window: usize,
            min_cycles: usize,
            cap: usize,
            budget: usize,
        ) -> (Option<usize>, bool) {
            let mut history: std::collections::VecDeque<f64> =
                std::collections::VecDeque::with_capacity(window + 1);
            let mut residual = 1.0_f64; // start well above tol
            let mut reached_tol = false;
            for cycle in 0..budget {
                // A genuine convergence certificate would have exited already.
                if residual <= residual_tol {
                    reached_tol = true;
                    return (None, reached_tol);
                }
                if history.len() > window {
                    history.pop_front();
                }
                history.push_back(residual);
                if cycle + 1 >= min_cycles && history.len() > window {
                    let oldest = *history.front().unwrap();
                    if slow_geometric_rate_exceeds_projection_cap(
                        residual,
                        oldest,
                        window,
                        residual_tol,
                        cap,
                    ) {
                        return (Some(cycle + 1), reached_tol);
                    }
                }
                residual *= per_cycle_rate;
            }
            (None, reached_tol)
        }

        // 1) The #979 hang signature: ~0.99×/cycle. Reaching 1e-6 from 1.0 at
        //    0.99×/cycle would take ~1375 cycles (> the 1000 budget) — a hang.
        //    The guard must fire, and bounded: just past MIN_CYCLES once the
        //    window first fills, NOT at the budget.
        let (slow_exit, slow_reached) = run_stream(
            0.99,
            residual_tol,
            LINEAR_RATE_WINDOW,
            RESIDUAL_STALL_MIN_CYCLES,
            LINEAR_RATE_PROJECTION_CAP,
            INNER_BUDGET,
        );
        assert!(
            !slow_reached,
            "the slow-geometric stream must not reach tol within budget (it is the hang)"
        );
        let slow_exit =
            slow_exit.expect("slow-geometric stall guard must fire, not grind to the budget");
        assert!(
            slow_exit < INNER_BUDGET / 4,
            "guard must terminate well below the {INNER_BUDGET}-cycle budget, fired at {slow_exit}"
        );
        // It can only arm at MIN_CYCLES, so the exit is bounded from both sides.
        assert!(
            slow_exit >= RESIDUAL_STALL_MIN_CYCLES,
            "guard must not fire before it is armed (MIN_CYCLES={RESIDUAL_STALL_MIN_CYCLES})"
        );

        // 2) A healthy fast-geometric solve (~0.3×/cycle) reaches tol in a
        //    handful of cycles and NEVER reaches the armed window — the guard
        //    must never fire on it.
        let (fast_exit, fast_reached) = run_stream(
            0.3,
            residual_tol,
            LINEAR_RATE_WINDOW,
            RESIDUAL_STALL_MIN_CYCLES,
            LINEAR_RATE_PROJECTION_CAP,
            INNER_BUDGET,
        );
        assert!(
            fast_reached,
            "a fast-geometric solve must reach tol (healthy convergence)"
        );
        assert!(
            fast_exit.is_none(),
            "the slow-rate guard must NEVER fire on a healthy fast-geometric solve"
        );

        // 3) Direct predicate properties at the boundary.
        // No net progress across the window => fire (cannot reach tol).
        assert!(slow_geometric_rate_exceeds_projection_cap(
            1.0,
            1.0,
            LINEAR_RATE_WINDOW,
            residual_tol,
            LINEAR_RATE_PROJECTION_CAP
        ));
        // Residual already at/under tol => never fire (certificate owns it).
        assert!(!slow_geometric_rate_exceeds_projection_cap(
            1e-7,
            1.0,
            LINEAR_RATE_WINDOW,
            residual_tol,
            LINEAR_RATE_PROJECTION_CAP
        ));
        // A brisk window (0.5×/cycle over 16 cycles, residual 1e-3) projects to
        // only ~10 more cycles to tol => never fire.
        let brisk_oldest = 1e-3 / 0.5_f64.powi(LINEAR_RATE_WINDOW as i32);
        assert!(!slow_geometric_rate_exceeds_projection_cap(
            1e-3,
            brisk_oldest,
            LINEAR_RATE_WINDOW,
            residual_tol,
            LINEAR_RATE_PROJECTION_CAP
        ));
    }
}
