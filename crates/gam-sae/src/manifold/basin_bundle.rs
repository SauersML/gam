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
//! This module is the bundle bookkeeping: a memory-admitted set of saved inner states,
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

/// One saved basin and its most recent converged criterion value.
pub struct BasinMember<S> {
    /// The basin's converged inner state at the most recent evaluation.
    pub state: S,
    /// The basin's criterion value at the most recent evaluation.
    pub last_value: f64,
}

/// Exact-envelope admission refused because retaining one more basin would
/// exceed the caller's memory-derived state capacity. The bundle is unchanged:
/// it never evicts a previously admitted branch and never returns an inexact
/// envelope while claiming success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasinAdmissionError {
    pub member_capacity: usize,
    pub retained_members: usize,
}

impl std::fmt::Display for BasinAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "exact basin envelope cannot retain another distinct state: memory-derived capacity is {} and {} members are already retained",
            self.member_capacity, self.retained_members
        )
    }
}

/// A memory-admitted bundle of inner-basin states whose pointwise minimum is
/// the outer criterion envelope.
pub struct BasinBundle<S> {
    members: Vec<BasinMember<S>>,
    /// Maximum number of full states that fit the caller's retained-state
    /// memory budget. Reaching it is an explicit feasibility refusal, never an
    /// eviction heuristic: a currently dominated state can win at another rho.
    member_capacity: usize,
}

impl<S> BasinBundle<S> {
    pub fn new(member_capacity: usize) -> Self {
        Self {
            members: Vec::new(),
            member_capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn member_capacity(&self) -> usize {
        self.member_capacity
    }

    /// Drop every saved basin. Called by the outer
    /// objective at any seam that invalidates the saved states wholesale — a
    /// multi-start reset, a fresh β seed, a row-support swap (subsample
    /// engage/restore), or a homotopy basin mutation — the same seams that
    /// discard the single-shot probe→accept handoff. The bundle re-seeds itself
    /// from the new accepted basin on the next envelope evaluation.
    pub fn clear(&mut self) {
        self.members.clear();
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
    /// growing the bundle. A distinct state beyond `member_capacity` is refused
    /// explicitly and leaves every prior branch intact. Silent eviction cannot
    /// preserve the exact lower envelope because present value does not prove
    /// dominance at another rho.
    pub fn admit(
        &mut self,
        state: S,
        value: f64,
        mut is_same_basin: impl FnMut(&S, &S) -> bool,
    ) -> Result<(), BasinAdmissionError> {
        if let Some(existing) = self
            .members
            .iter_mut()
            .find(|m| is_same_basin(&m.state, &state))
        {
            if value < existing.last_value {
                existing.state = state;
                existing.last_value = value;
            }
            return Ok(());
        }
        if self.members.len() >= self.member_capacity {
            return Err(BasinAdmissionError {
                member_capacity: self.member_capacity,
                retained_members: self.members.len(),
            });
        }
        self.members.push(BasinMember {
            state,
            last_value: value,
        });
        Ok(())
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
    /// (state, value) out), take the min, and return
    /// `(argmin_index, envelope_value)`.
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
        let idx = match self.argmin_index() {
            Some(idx) if self.members[idx].last_value.is_finite() => idx,
            _ => return Err(last_err.expect("empty or all-failed bundle must carry an error")),
        };
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
        let b1 = Basin {
            a: 1.0,
            c: -1.0,
            d: 0.0,
        };
        let b2 = Basin {
            a: 1.0,
            c: 1.0,
            d: 0.0,
        };
        let mut bundle = BasinBundle::new(4);
        bundle
            .admit(b1.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
        bundle
            .admit(b2.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
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
        let shallow = Basin {
            a: 1.0,
            c: 0.0,
            d: 5.0,
        };
        let deep = Basin {
            a: 1.0,
            c: 0.0,
            d: -10.0,
        };
        let mut bundle = BasinBundle::new(4);
        bundle
            .admit(shallow, f64::INFINITY, |x, y| x == y)
            .unwrap();
        let (_, v1) = bundle
            .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.3))))
            .unwrap();
        bundle
            .admit(deep.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
        let (idx, v2) = bundle
            .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.3))))
            .unwrap();
        assert!(v2 < v1, "admission must lower the envelope: {v2} vs {v1}");
        assert_eq!(bundle.members[idx].state, deep);
    }

    #[test]
    fn duplicate_admission_replaces_instead_of_growing() {
        let b = Basin {
            a: 1.0,
            c: 0.0,
            d: 1.0,
        };
        let mut bundle = BasinBundle::new(4);
        bundle.admit(b.clone(), 3.0, |x, y| x == y).unwrap();
        bundle.admit(b.clone(), 2.0, |x, y| x == y).unwrap();
        assert_eq!(bundle.len(), 1);
        assert_eq!(bundle.members[0].last_value, 2.0);
        // Worse duplicate does not overwrite the better stored value.
        bundle.admit(b, 4.0, |x, y| x == y).unwrap();
        assert_eq!(bundle.members[0].last_value, 2.0);
    }

    #[test]
    fn dominated_member_is_retained_and_can_win_at_a_later_rho() {
        let winner = Basin {
            a: 1.0,
            c: 0.0,
            d: -1.0,
        };
        let loser = Basin {
            a: 1.0,
            c: 0.0,
            d: 10.0,
        };
        let mut bundle = BasinBundle::new(4);
        bundle
            .admit(winner.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
        bundle
            .admit(loser.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
        for _ in 0..8 {
            bundle
                .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.0))))
                .unwrap();
            assert_eq!(bundle.len(), 2, "evaluation count must not prune a basin");
        }
        // Make the formerly dominated branch the true lower-envelope winner.
        // A workload-count pruning deadline would have lost it and reopened
        // trajectory hysteresis.
        let (idx, _) = bundle
            .evaluate(|s: &Basin| {
                let branch_value = if *s == loser { -2.0 } else { value(s, 0.0) };
                Ok::<_, ()>((s.clone(), branch_value))
            })
            .unwrap();
        assert_eq!(bundle.members[idx].state, loser);
    }

    #[test]
    fn fifth_dominated_member_is_retained_and_can_win_later() {
        let mut bundle = BasinBundle::new(5);
        let basins = (0..5)
            .map(|i| Basin {
                a: 1.0,
                c: i as f64,
                d: i as f64,
            })
            .collect::<Vec<_>>();
        for basin in &basins {
            bundle
                .admit(basin.clone(), f64::INFINITY, |x, y| x == y)
                .unwrap();
        }
        let (initial_idx, _) = bundle
            .evaluate(|s: &Basin| Ok::<_, ()>((s.clone(), value(s, 0.0))))
            .unwrap();
        assert_eq!(bundle.members[initial_idx].state, basins[0]);

        // The fifth branch is worst at the first rho but becomes the exact
        // lower-envelope winner later. The former four-member eviction deleted
        // this branch and returned the wrong envelope at the second point.
        let fifth = basins[4].clone();
        let (later_idx, later_value) = bundle
            .evaluate(|s: &Basin| {
                let branch_value = if *s == fifth { -20.0 } else { value(s, 0.0) };
                Ok::<_, ()>((s.clone(), branch_value))
            })
            .unwrap();
        assert_eq!(later_value, -20.0);
        assert_eq!(bundle.members[later_idx].state, fifth);
        assert_eq!(bundle.len(), 5, "no admitted branch may be evicted");
    }

    #[test]
    fn memory_capacity_refuses_without_evicting_history() {
        let mut bundle = BasinBundle::new(2);
        let b0 = Basin {
            a: 1.0,
            c: 0.0,
            d: 0.0,
        };
        let b1 = Basin {
            a: 1.0,
            c: 1.0,
            d: 1.0,
        };
        let b2 = Basin {
            a: 1.0,
            c: 2.0,
            d: 2.0,
        };
        bundle.admit(b0.clone(), 0.0, |x, y| x == y).unwrap();
        bundle.admit(b1.clone(), 1.0, |x, y| x == y).unwrap();
        let error = bundle
            .admit(b2, 2.0, |x, y| x == y)
            .expect_err("a distinct state beyond the memory capacity must be refused");
        assert_eq!(
            error,
            BasinAdmissionError {
                member_capacity: 2,
                retained_members: 2,
            }
        );
        assert_eq!(bundle.len(), 2);
        assert_eq!(bundle.members[0].state, b0);
        assert_eq!(bundle.members[1].state, b1);
    }

    #[test]
    fn infeasible_member_is_retained_and_can_win_later() {
        // Member 2 errors (infeasible) at the first ρ but is the winner at the
        // second — dropping it on failure would re-open the hysteresis.
        let b1 = Basin {
            a: 1.0,
            c: 0.0,
            d: 0.0,
        };
        let b2 = Basin {
            a: 1.0,
            c: 2.0,
            d: -5.0,
        };
        let mut bundle = BasinBundle::new(4);
        bundle
            .admit(b1, f64::INFINITY, |x, y| x == y)
            .unwrap();
        bundle
            .admit(b2.clone(), f64::INFINITY, |x, y| x == y)
            .unwrap();
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
        let b = Basin {
            a: 1.0,
            c: 0.0,
            d: 0.0,
        };
        let mut bundle = BasinBundle::new(2);
        bundle
            .admit(b, f64::INFINITY, |x, y| x == y)
            .unwrap();
        let out = bundle.evaluate(|_s: &Basin| Err::<(Basin, f64), _>("boom"));
        assert_eq!(out.unwrap_err(), "boom");
    }
}
