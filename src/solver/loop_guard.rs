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
//! divergence detector) consume it directly; [`PlateauDetector`] composes
//! it with the default relative-improvement predicate for loops that just
//! hand over their merit stream.
//!
//! [`PlateauDetector`] answers the question attempt caps cannot see: a
//! loop that still "makes progress" every iteration but whose MERIT is
//! frozen. #744 ran to cycle 1199/1200 at a flat residual; #826 burned a
//! CI timeout on a frozen joint residual. Feed it the loop's descent
//! quantity (penalized NLL, residual norm, |g|) once per iteration; it
//! reports a plateau once the relative improvement stays below a
//! tolerance for a consecutive window — long before any iteration cap.
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

/// Default consecutive-window length for [`PlateauDetector`]: how many
/// successive merit readings must show sub-tolerance relative improvement
/// before the loop is declared plateaued. Two is the established in-tree
/// streak convention (reweight.rs soft-acceptance) — one noisy reading can
/// fake a plateau, two consecutive cannot — plus one for the headroom a
/// merit that is genuinely creeping (not frozen) needs to escape.
pub const PLATEAU_DEFAULT_WINDOW: usize = 3;

/// Default relative-improvement tolerance for [`PlateauDetector`]. 1e-8 of
/// the merit magnitude per iteration is far below any improvement a
/// convergent inner Newton produces mid-run, yet orders of magnitude above
/// roundoff jitter on a frozen merit (#744's flat residual, #826's frozen
/// joint residual both sit at exactly 0 relative change).
pub const PLATEAU_DEFAULT_REL_TOL: f64 = 1e-8;

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

/// Stagnation detector on a loop's merit stream: [`FlatStreak`] composed
/// with the default relative-improvement flatness predicate.
///
/// Feed it the loop's descent quantity once per iteration; it answers
/// "has the relative improvement stayed below `rel_tol` for `window`
/// consecutive readings?". Direction-agnostic: it watches |Δmerit|
/// relative to the merit's magnitude, so minimized NLLs, residual norms,
/// and gradient norms all work unaltered.
#[derive(Clone, Debug)]
pub struct PlateauDetector {
    rel_tol: f64,
    streak: FlatStreak,
    last_merit: Option<f64>,
}

impl PlateauDetector {
    pub fn new(window: usize, rel_tol: f64) -> Self {
        Self {
            rel_tol,
            streak: FlatStreak::new(window),
            last_merit: None,
        }
    }

    /// In-tree default tuning; see the constant docs.
    pub fn standard() -> Self {
        Self::new(PLATEAU_DEFAULT_WINDOW, PLATEAU_DEFAULT_REL_TOL)
    }

    /// Record one merit reading; returns the current verdict. Non-finite
    /// merits never count toward a plateau (a NaN merit is a different
    /// failure, owned by the loop's own error handling) and reset the
    /// streak so recovery is observed from scratch.
    pub fn note(&mut self, merit: f64) -> LoopVerdict {
        if !merit.is_finite() {
            self.streak.reset();
            self.last_merit = None;
            return LoopVerdict::Continue;
        }
        let verdict = match self.last_merit {
            Some(prev) => {
                let scale = prev.abs().max(merit.abs()).max(1.0);
                self.streak
                    .note((prev - merit).abs() <= self.rel_tol * scale)
            }
            None => LoopVerdict::Continue,
        };
        self.last_merit = Some(merit);
        verdict
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

    /// #744 trace shape: merit descends, then freezes. The detector must
    /// fire within `window` readings of the freeze and NOT fire while
    /// genuine descent is happening.
    #[test]
    fn plateau_fires_on_frozen_merit_and_not_during_descent() {
        let mut det = PlateauDetector::new(3, 1e-8);
        let mut merit = 100.0;
        for _ in 0..50 {
            assert_eq!(det.note(merit), LoopVerdict::Continue, "descent phase");
            merit *= 0.9;
        }
        let mut fired_after = 0usize;
        loop {
            fired_after += 1;
            assert!(fired_after <= 4, "plateau must fire within the window");
            if det.note(merit) == LoopVerdict::Plateaued {
                break;
            }
        }
        assert!(fired_after >= 3, "must not fire before the streak window");
    }

    #[test]
    fn plateau_resets_on_recovery_and_ignores_non_finite() {
        let mut det = PlateauDetector::new(2, 1e-8);
        det.note(10.0);
        assert_eq!(det.note(10.0), LoopVerdict::Continue); // streak 1
        assert_eq!(det.note(9.0), LoopVerdict::Continue); // recovery resets
        assert_eq!(det.note(9.0), LoopVerdict::Continue); // streak 1 again
        assert_eq!(det.note(f64::NAN), LoopVerdict::Continue); // hard reset
        assert_eq!(det.note(9.0), LoopVerdict::Continue); // re-baseline
        assert_eq!(det.note(9.0), LoopVerdict::Continue); // streak 1
        assert_eq!(det.note(9.0), LoopVerdict::Plateaued); // streak 2 fires
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

    /// Suspect #1 of gam#1040: the plateau detector MUST be scale-invariant.
    /// A survival marginal-slope NLL objective sits at O(1e4); an absolute
    /// flat-threshold (|ΔV| < ε_abs) never fires there because a single cycle
    /// can move the objective by O(1) in absolute terms while the relative
    /// change |ΔV|/|V| is already at roundoff — the iterate IS on a flat
    /// plateau yet an absolute test calls it "still moving" forever, which is
    /// the never-terminating hang. The relative test (|ΔV| ≤ rel_tol·|V|)
    /// fires identically at O(1) and O(1e4), so the loop terminates as
    /// converged on the flat valley regardless of objective magnitude.
    #[test]
    fn plateau_detector_is_scale_invariant_on_large_magnitude_objective() {
        let rel_tol = 1e-8;
        let window = 3;

        // A flat plateau at O(1e4): the objective creeps by ~6e-5 per cycle,
        // which is |ΔV|/|V| ≈ 1e-9 < rel_tol — flat — yet would dwarf any
        // sane absolute ε if one keyed off |ΔV| directly (6e-5 ≫ 1e-8).
        let big = 6.0e4_f64;
        let mut det_big = PlateauDetector::new(window, rel_tol);
        det_big.note(big);
        let mut big_fired_at = None;
        for k in 0..window {
            let v = big - 6.0e-5 * (k as f64 + 1.0);
            if det_big.note(v) == LoopVerdict::Plateaued {
                big_fired_at = Some(k);
                break;
            }
        }
        assert!(
            big_fired_at.is_some(),
            "relative plateau detector must terminate on an O(1e4) flat valley \
             within the window; instead it ran past it (the #1040 hang)"
        );

        // The SAME relative stream scaled down to O(1) fires at the SAME
        // streak position — scale invariance, the whole point.
        let small = 1.0_f64;
        let mut det_small = PlateauDetector::new(window, rel_tol);
        det_small.note(small);
        let mut small_fired_at = None;
        for k in 0..window {
            let v = small - 1.0e-9 * (k as f64 + 1.0);
            if det_small.note(v) == LoopVerdict::Plateaued {
                small_fired_at = Some(k);
                break;
            }
        }
        assert_eq!(
            big_fired_at, small_fired_at,
            "plateau detection must be invariant to objective magnitude"
        );

        // A genuinely non-flat large-magnitude descent (|ΔV|/|V| ≈ 1e-2 ≫
        // rel_tol) must NOT be called a plateau — the relative test does not
        // weaken a real convergence check, it only rescales it.
        let mut det_moving = PlateauDetector::new(window, rel_tol);
        det_moving.note(big);
        let mut moving = big;
        for _ in 0..window {
            moving -= 0.01 * moving;
            assert_eq!(
                det_moving.note(moving),
                LoopVerdict::Continue,
                "a 1%-per-cycle relative descent must never register as flat"
            );
        }
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
}
