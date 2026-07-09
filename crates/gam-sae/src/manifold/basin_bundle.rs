//! Basin-bundle lower envelope for the outer REML criterion (#2230/#2087 genus).
//!
//! The outer criterion as historically implemented is `V_{b(warm,ρ)}(ρ)` — the
//! value of whichever inner basin the warm-started inner solve lands in. The
//! object the outer search should descend is the lower envelope
//! `V*(ρ) = min_b V_b(ρ)`, which is continuous and piecewise smooth; the
//! trajectory-dependent version JUMPS at basin-boundary crossings and is
//! hysteretic, which is exactly the measured #2230 pathology (hours of
//! `[#1026] restoring inner-fit reconstruction incumbent` churn = the outer
//! line search oscillating across a boundary it cannot represent) and the
//! #2087 audit's −338-vs−628 one-Δρ-away teleport.
//!
//! This module is the bundle bookkeeping: a small set of saved inner states,
//! one per distinct basin encountered during the outer walk. Each criterion
//! evaluation re-converges EVERY member from its own saved state (members near
//! their basin optimum re-converge in a round or two — warm), takes the min,
//! and returns the argmin member's state as the eval's state. The envelope
//! gradient is the argmin basin's gradient (envelope theorem, exact wherever
//! the argmin is unique — a.e.; at a crossing the envelope is continuous with
//! a subgradient kink, which the outer BFGS/ARC tolerate because the VALUE no
//! longer jumps). Adding a member can only LOWER the envelope, so discovery
//! (escape hatches, reseeds, multi-starts) strictly improves the criterion
//! surface instead of teleporting it.
//!
//! The module is deliberately generic over the inner state `S` and the solver
//! closure so the envelope algebra is unit-testable without a fit.

/// One saved basin: its converged inner state and eval-history bookkeeping.
pub struct BasinMember<S> {
    /// The basin's converged inner state at the most recent evaluation.
    pub state: S,
    /// The basin's criterion value at the most recent evaluation.
    pub last_value: f64,
    /// Eval counter value the last time this member was the envelope argmin.
    pub last_win_eval: u64,
    /// Eval counter value when the member was admitted.
    pub born_eval: u64,
}

/// A small bundle of inner-basin states whose pointwise minimum is the outer
/// criterion envelope.
pub struct BasinBundle<S> {
    members: Vec<BasinMember<S>>,
    eval_counter: u64,
    /// Hard cap on bundle size; admission beyond it evicts the worst
    /// non-argmin member. The caller derives this from problem structure
    /// (number of plausible coexisting basins), not tuning.
    max_members: usize,
    /// A member that has not won the envelope within this many evaluations
    /// (and is not the newest) is pruned as dominated. Caller-derived.
    dominance_window: u64,
}

impl<S> BasinBundle<S> {
    pub fn new(max_members: usize, dominance_window: u64) -> Self {
        Self {
            members: Vec::new(),
            eval_counter: 0,
            max_members: max_members.max(1),
            dominance_window: dominance_window.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Drop every saved basin and reset the eval clock. Called by the outer
    /// objective at any seam that invalidates the saved states wholesale — a
    /// multi-start reset, a fresh β seed, a row-support swap (subsample
    /// engage/restore), or a homotopy basin mutation — the same seams that
    /// discard the single-shot probe→accept handoff. The bundle re-seeds itself
    /// from the new accepted basin on the next envelope evaluation.
    pub fn clear(&mut self) {
        self.members.clear();
        self.eval_counter = 0;
    }

    /// Current envelope argmin member, if any evaluation has happened.
    pub fn argmin(&self) -> Option<&BasinMember<S>> {
        self.members
            .iter()
            .min_by(|a, b| a.last_value.total_cmp(&b.last_value))
    }

    /// Admit a newly-discovered basin state. `is_same_basin` is the caller's
    /// basin-identity test (e.g. reconstruction distance below the noise
    /// floor, or criterion gap below the stall tolerance at equal ρ): on a
    /// duplicate the better-valued state replaces the stored one instead of
    /// growing the bundle. Over the cap, the worst non-argmin member is
    /// evicted — admission can therefore never RAISE the envelope.
    pub fn admit(&mut self, state: S, value: f64, mut is_same_basin: impl FnMut(&S, &S) -> bool) {
        if let Some(existing) = self
            .members
            .iter_mut()
            .find(|m| is_same_basin(&m.state, &state))
        {
            if value < existing.last_value {
                existing.state = state;
                existing.last_value = value;
            }
            return;
        }
        self.members.push(BasinMember {
            state,
            last_value: value,
            last_win_eval: self.eval_counter,
            born_eval: self.eval_counter,
        });
        if self.members.len() > self.max_members {
            // Evict the worst member that is neither the current argmin NOR the
            // newest (the member just admitted may carry a placeholder value —
            // e.g. +∞ before its first evaluation — and evicting it here would
            // undo the admission, contradicting the "admission can only lower
            // the envelope" contract the evaluate() prune already honors).
            let argmin_idx = self.argmin_index();
            let newest_born = self.members.iter().map(|m| m.born_eval).max().unwrap_or(0);
            let candidate = self
                .members
                .iter()
                .enumerate()
                .filter(|(i, m)| Some(*i) != argmin_idx && m.born_eval != newest_born)
                .max_by(|(_, a), (_, b)| a.last_value.total_cmp(&b.last_value))
                .map(|(i, _)| i)
                // Same-eval admissions can all share `newest_born` (e.g. bundle
                // seeding before the first evaluation); the cap is a hard
                // memory bound, so fall back to evicting the worst non-argmin.
                .or_else(|| {
                    self.members
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| Some(*i) != argmin_idx)
                        .max_by(|(_, a), (_, b)| a.last_value.total_cmp(&b.last_value))
                        .map(|(i, _)| i)
                });
            if let Some(worst) = candidate {
                self.members.swap_remove(worst);
            }
        }
    }

    fn argmin_index(&self) -> Option<usize> {
        self.members
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.last_value.total_cmp(&b.last_value))
            .map(|(i, _)| i)
    }

    /// One envelope evaluation at the caller's current ρ: re-converge every
    /// member from its own saved state via `solve` (state in, converged
    /// (state, value) out), take the min, prune members dominated for longer
    /// than the window, and return `(argmin_index, envelope_value)`.
    ///
    /// A member whose solve errors is treated as infeasible AT THIS ρ (value
    /// = +∞) but retained — an infeasible basin at one ρ can be the winner at
    /// another, and dropping it would re-open the hysteresis this exists to
    /// close. The call errors only when EVERY member fails, since then there
    /// is no envelope value to return.
    pub fn evaluate<E>(
        &mut self,
        mut solve: impl FnMut(&S) -> Result<(S, f64), E>,
    ) -> Result<(usize, f64), E> {
        self.eval_counter += 1;
        let mut last_err: Option<E> = None;
        for member in &mut self.members {
            match solve(&member.state) {
                Ok((state, value)) => {
                    member.state = state;
                    member.last_value = value;
                }
                Err(err) => {
                    member.last_value = f64::INFINITY;
                    last_err = Some(err);
                }
            }
        }
        let argmin = match self.argmin_index() {
            Some(idx) if self.members[idx].last_value.is_finite() => idx,
            _ => return Err(last_err.expect("empty or all-failed bundle must carry an error")),
        };
        self.members[argmin].last_win_eval = self.eval_counter;
        let newest_born = self.members.iter().map(|m| m.born_eval).max().unwrap_or(0);
        let window = self.dominance_window;
        let counter = self.eval_counter;
        let argmin_value = self.members[argmin].last_value;
        // Prune long-dominated members — but never the argmin and never the
        // newest (it has not had a fair window yet).
        self.members.retain(|m| {
            m.last_value <= argmin_value
                || m.born_eval == newest_born
                || counter.saturating_sub(m.last_win_eval) < window
        });
        // Recompute the argmin index post-retain (indices may have shifted).
        let idx = self
            .argmin_index()
            .expect("argmin member is never pruned");
        Ok((idx, self.members[idx].last_value))
    }

    /// The argmin member's state (the state the outer eval should return).
    pub fn argmin_state(&self) -> Option<&S> {
        self.argmin().map(|m| &m.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock basin: V_b(ρ) = a·(ρ−c)² + d — a smooth per-basin criterion whose
    /// converged "state" is just its parameters (re-converging is a no-op).
    #[derive(Clone, PartialEq, Debug)]
    struct Basin {
        a: f64,
        c: f64,
        d: f64,
    }
    fn value(b: &Basin, rho: f64) -> f64 {
        b.a * (rho - b.c) * (rho - b.c) + b.d
    }

    #[test]
    fn envelope_is_continuous_across_the_basin_crossing() {
        // Two basins crossing at ρ=0: hysteretic single-state tracking jumps
        // there; the bundle envelope must be continuous.
        let b1 = Basin { a: 1.0, c: -1.0, d: 0.0 };
        let b2 = Basin { a: 1.0, c: 1.0, d: 0.0 };
        let mut bundle = BasinBundle::new(4, 100);
        bundle.admit(b1.clone(), f64::INFINITY, |x, y| x == y);
        bundle.admit(b2.clone(), f64::INFINITY, |x, y| x == y);
        let mut prev: Option<f64> = None;
        let mut max_jump = 0.0_f64;
        let mut rho = -0.5;
        while rho <= 0.5 {
            let (_, v) = bundle
                .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, rho))))
                .unwrap();
            if let Some(p) = prev {
                max_jump = max_jump.max((v - p).abs());
            }
            prev = Some(v);
            rho += 0.01;
        }
        // Envelope slope is bounded by max|V'| ≈ 3 on this window, so with
        // step 0.01 any jump beyond ~0.05 would betray a discontinuity.
        assert!(
            max_jump < 0.05,
            "envelope jumped by {max_jump} across the crossing"
        );
    }

    #[test]
    fn admitting_a_better_basin_lowers_the_envelope_and_switches_argmin() {
        let shallow = Basin { a: 1.0, c: 0.0, d: 5.0 };
        let deep = Basin { a: 1.0, c: 0.0, d: -10.0 };
        let mut bundle = BasinBundle::new(4, 100);
        bundle.admit(shallow, f64::INFINITY, |x, y| x == y);
        let (_, v1) = bundle
            .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.3))))
            .unwrap();
        bundle.admit(deep.clone(), f64::INFINITY, |x, y| x == y);
        let (idx, v2) = bundle
            .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.3))))
            .unwrap();
        assert!(v2 < v1, "admission must lower the envelope: {v2} vs {v1}");
        assert_eq!(bundle.members[idx].state, deep);
    }

    #[test]
    fn duplicate_admission_replaces_instead_of_growing() {
        let b = Basin { a: 1.0, c: 0.0, d: 1.0 };
        let mut bundle = BasinBundle::new(4, 100);
        bundle.admit(b.clone(), 3.0, |x, y| x == y);
        bundle.admit(b.clone(), 2.0, |x, y| x == y);
        assert_eq!(bundle.len(), 1);
        assert_eq!(bundle.members[0].last_value, 2.0);
        // Worse duplicate does not overwrite the better stored value.
        bundle.admit(b, 4.0, |x, y| x == y);
        assert_eq!(bundle.members[0].last_value, 2.0);
    }

    #[test]
    fn dominated_members_are_pruned_after_the_window_but_argmin_survives() {
        let winner = Basin { a: 1.0, c: 0.0, d: -1.0 };
        let loser = Basin { a: 1.0, c: 0.0, d: 10.0 };
        let mut bundle = BasinBundle::new(4, 3);
        bundle.admit(winner, f64::INFINITY, |x, y| x == y);
        bundle.admit(loser, f64::INFINITY, |x, y| x == y);
        for _ in 0..2 {
            bundle
                .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.0))))
                .unwrap();
            assert_eq!(bundle.len(), 2, "inside the window both survive");
        }
        for _ in 0..4 {
            bundle
                .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.0))))
                .unwrap();
        }
        assert_eq!(bundle.len(), 1, "dominated member pruned after window");
        assert_eq!(bundle.members[0].last_value, -1.0);
    }

    #[test]
    fn infeasible_member_is_retained_and_can_win_later() {
        // Member 2 errors (infeasible) at the first ρ but is the winner at the
        // second — dropping it on failure would re-open the hysteresis.
        let b1 = Basin { a: 1.0, c: 0.0, d: 0.0 };
        let b2 = Basin { a: 1.0, c: 2.0, d: -5.0 };
        let mut bundle = BasinBundle::new(4, 100);
        bundle.admit(b1, f64::INFINITY, |x, y| x == y);
        bundle.admit(b2.clone(), f64::INFINITY, |x, y| x == y);
        let (_, v1) = bundle
            .evaluate(|s: &Basin| {
                if *s == b2 {
                    Err("infeasible here")
                } else {
                    Ok((s.clone(), value(s, 0.0)))
                }
            })
            .unwrap();
        assert_eq!(v1, 0.0);
        assert_eq!(bundle.len(), 2, "infeasible member retained");
        let (idx, v2) = bundle
            .evaluate(|s: &Basin| Ok::<_, &str>((s.clone(), value(s, 2.0))))
            .unwrap();
        assert_eq!(v2, -5.0, "previously-infeasible basin wins where it should");
        assert_eq!(bundle.members[idx].state, b2);
    }

    #[test]
    fn all_failed_bundle_surfaces_the_error() {
        let b = Basin { a: 1.0, c: 0.0, d: 0.0 };
        let mut bundle = BasinBundle::new(2, 10);
        bundle.admit(b, f64::INFINITY, |x, y| x == y);
        let out = bundle.evaluate(|_s: &Basin| Err::<(Basin, f64), _>("boom"));
        assert_eq!(out.unwrap_err(), "boom");
    }
}
