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
//! # The two policies
//!
//! [`madsen_can_retry`] / [`madsen_retry_exhausted`] own the damped-retry
//! exhaustion question for Madsen-style Levenberg–Marquardt loops: a retry
//! is alive while the damping is finite and below [`MADSEN_DAMPING_CAP`],
//! and dead once attempts run out or damping leaves that window. Both
//! engines (reweight.rs Madsen-LM and the custom_family.rs spectral
//! Newton) must answer this question through these functions — never
//! through a local predicate.
//!
//! [`PlateauDetector`] owns the question attempt caps cannot see: a loop
//! that still "makes progress" every iteration but whose MERIT is frozen.
//! #744 ran to cycle 1199/1200 at a flat residual; #826 burned a CI
//! timeout on a frozen joint residual. The detector watches the loop's
//! descent quantity (penalized NLL, residual norm, |g|) and reports a
//! plateau once the relative improvement stays below a tolerance for a
//! consecutive window — long before any iteration cap.
//!
//! # Verdicts, not panics
//!
//! Exhaustion is an escalation event: the consuming loop converts
//! [`LoopVerdict::Plateaued`] / [`LoopVerdict::Exhausted`] into its
//! honest terminal status (`StalledAtValidMinimum`,
//! `LmStepSearchExhausted`, …) and unwinds. Never a hang, never a panic,
//! never a silent wrong answer.
//!
//! # Migration map (each step deletes a hand-rolled guard)
//!
//! 1. (this commit) reweight.rs `lm_can_retry`/`lm_retry_exhausted` local
//!    fns + the local `LM_MAX_LAMBDA` const are deleted; the 10 call
//!    sites consume this module's policy.
//! 2. The 7 copies of the reject ritual
//!    (`loop_lambda *= factor; factor *= 2.0; continue`) collapse onto
//!    [`MadsenGuard::escalate`] so the doubling discipline cannot drift
//!    per-branch either.
//! 3. custom_family.rs Newton cycle: `outer_inner_max_iterations`
//!    (the shared `Arc<AtomicUsize>` countdown) becomes a `MadsenGuard`
//!    field, and the cycle gains a `PlateauDetector` on its joint
//!    residual — the guard that would have caught #826 in seconds.
//! 4. Terminal verdicts report into the heartbeat scope (heartbeat.rs
//!    stays a monitor; the 60s liveness log then shows WHY a loop ended,
//!    not just where it was).

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

/// Stagnation detector on a loop's merit stream.
///
/// Feed it the loop's descent quantity once per iteration; it answers
/// "has the relative improvement stayed below `rel_tol` for `window`
/// consecutive readings?". Direction-agnostic: it watches |Δmerit|
/// relative to the merit's magnitude, so minimized NLLs, residual norms,
/// and gradient norms all work unaltered.
#[derive(Clone, Debug)]
pub struct PlateauDetector {
    window: usize,
    rel_tol: f64,
    streak: usize,
    last_merit: Option<f64>,
}

impl PlateauDetector {
    pub fn new(window: usize, rel_tol: f64) -> Self {
        Self {
            window: window.max(1),
            rel_tol,
            streak: 0,
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
            self.streak = 0;
            self.last_merit = None;
            return LoopVerdict::Continue;
        }
        let verdict = match self.last_merit {
            Some(prev) => {
                let scale = prev.abs().max(merit.abs()).max(1.0);
                if (prev - merit).abs() <= self.rel_tol * scale {
                    self.streak += 1;
                    if self.streak >= self.window {
                        LoopVerdict::Plateaued
                    } else {
                        LoopVerdict::Continue
                    }
                } else {
                    self.streak = 0;
                    LoopVerdict::Continue
                }
            }
            None => LoopVerdict::Continue,
        };
        self.last_merit = Some(merit);
        verdict
    }
}

/// Stateful retry guard for one damped reject chain: owns the attempt
/// count AND the geometric escalation discipline, so no branch can apply
/// one without the other (the #874 drift mode). Created fresh per
/// outer iteration (`attempts` resets); the damping itself stays a solver
/// local (it enters the linear system) and is escalated through the guard.
#[derive(Clone, Debug)]
pub struct MadsenGuard {
    attempts: usize,
    max_attempts: usize,
    reject_factor: f64,
}

/// Initial damping multiplier on the first rejection of an iteration.
/// Doubles on every further rejection (geometric escalation), reaching
/// [`MADSEN_DAMPING_CAP`] from λ = 1 in ~12 rejections — the established
/// reweight.rs schedule, now owned here.
pub const MADSEN_INITIAL_REJECT_FACTOR: f64 = 2.0;

impl MadsenGuard {
    pub fn new(max_attempts: usize) -> Self {
        Self {
            attempts: 0,
            max_attempts: max_attempts.max(1),
            reject_factor: MADSEN_INITIAL_REJECT_FACTOR,
        }
    }

    /// Record a rejection: bumps the attempt count and applies the
    /// geometric damping escalation in one indivisible step.
    pub fn escalate(&mut self, damping: &mut f64) {
        self.attempts += 1;
        *damping *= self.reject_factor;
        self.reject_factor *= 2.0;
    }

    /// The single exhaustion question, answered from owned state.
    pub fn verdict(&self, damping: f64) -> LoopVerdict {
        if madsen_retry_exhausted(damping, self.attempts, self.max_attempts) {
            LoopVerdict::Exhausted
        } else {
            LoopVerdict::Continue
        }
    }

    pub fn attempts(&self) -> usize {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #874 regression shape: a reject storm must reach Exhausted in a
    /// bounded number of escalations no matter which branch asks — the
    /// guard owns both the count and the damping window.
    #[test]
    fn reject_storm_exhausts_in_bounded_steps() {
        let mut guard = MadsenGuard::new(usize::MAX); // no attempt cap:
        let mut damping = 1.0;
        let mut steps = 0usize;
        while guard.verdict(damping) == LoopVerdict::Continue {
            guard.escalate(&mut damping);
            steps += 1;
            assert!(steps <= 64, "escalation must reach the damping cap");
        }
        // Geometric doubling of the factor reaches 1e12 in ~9 escalations.
        assert!(steps <= 16, "escalation took {steps} steps");
        assert!(!madsen_can_retry(damping));
    }

    #[test]
    fn attempt_cap_exhausts_even_at_benign_damping() {
        let mut guard = MadsenGuard::new(3);
        let mut damping = 1e-6;
        for _ in 0..3 {
            assert_eq!(guard.verdict(1e-6), LoopVerdict::Continue);
            // Keep damping benign: only the attempt count should kill it.
            let mut local = damping;
            guard.escalate(&mut local);
            damping = 1e-6;
        }
        assert_eq!(guard.verdict(damping), LoopVerdict::Exhausted);
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
