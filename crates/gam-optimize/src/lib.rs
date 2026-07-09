//! Shared numerical-optimization primitives for the gam engine.
//!
//! This crate is the ONE canonical home for two small control-flow patterns
//! that were, before the #1521-lineage reuse audits, re-typed verbatim across a
//! dozen production modules (`gam-geometry`, `gam-custom-family`, `gam-models`,
//! `gam-sae`, …):
//!
//!  1. an **Armijo backtracking line search** — evaluate a trial step, test a
//!     sufficient-decrease predicate, halve on rejection until a floor;
//!  2. a **geometric ridge escalation** — try an operation at ridge `δ`, and on
//!     failure grow `δ` by a fixed factor until a cap.
//!
//! The primitives deliberately own only the *loop skeleton, the shared tuning
//! constants, and the iteration policy* (what a rejected/invalid trial does,
//! how the step or ridge advances, when to give up). The numerically load-
//! bearing arithmetic — the exact sufficient-decrease threshold, the retraction,
//! the objective evaluation, the factorization — stays in the caller's closures,
//! so migrating a site preserves its numerics bit-for-bit. Nothing here is a
//! math change; it is a *deduplication* home.
//!
//! It is a leaf crate: it depends on nothing but `core`, so every consumer crate
//! can adopt it without introducing a dependency cycle.

#![no_std]

/// Named tuning constants shared across the migrated call sites. Each site used
/// to re-declare these as local `const`s with identical values; they now live
/// here so a single edit updates every optimizer in lockstep.
pub mod constants {
    /// Armijo sufficient-decrease parameter `c₁` (the textbook `1e-4`). Used by
    /// every backtracking line search: a trial is accepted when the objective
    /// drops by at least `c₁ · t · |directional slope|`.
    pub const ARMIJO_C1: f64 = 1.0e-4;

    /// Multiplicative step factor applied to a *rejected* backtracking trial.
    /// The classic halving schedule (`t ← t/2`).
    pub const BACKTRACK_CONTRACTION: f64 = 0.5;

    /// Default cap on backtracking halvings. Starting from a unit step and
    /// halving, `2⁻⁶⁰ ≈ 1e-18` sits below `f64` resolution, so exhausting the
    /// budget means no positive step decreases the objective within precision.
    pub const MAX_BACKTRACK_HALVINGS: usize = 60;

    /// Round-off cushion on the Armijo test, in units of `f64::EPSILON`. Near a
    /// flat optimum the sufficient-decrease term vanishes and this cushion lets
    /// the test admit a step that only increases the objective by round-off.
    /// See [`armijo_roundoff_cushion`].
    pub const ARMIJO_ROUNDOFF_EPS_MULTIPLE: f64 = 8.0;

    /// Geometric growth factor for a ridge / Levenberg–Marquardt δ escalation
    /// (`δ ← 10·δ` per step). Shared by every ridge-retry loop.
    pub const RIDGE_GROWTH: f64 = 10.0;
}

/// Round-off cushion `f_tol = ARMIJO_ROUNDOFF_EPS_MULTIPLE · ε · (1 + |f_cur|)`
/// added to the right-hand side of the Armijo test. Sourced here so every site
/// computes it identically. The arithmetic reproduces the expression the
/// migrated sites inlined verbatim, so the cushion is bit-for-bit unchanged.
#[inline]
#[must_use]
pub fn armijo_roundoff_cushion(f_cur: f64) -> f64 {
    constants::ARMIJO_ROUNDOFF_EPS_MULTIPLE * f64::EPSILON * (1.0 + f_cur.abs())
}

/// A step accepted by [`backtracking_line_search`], carrying the accepted step
/// scale, the objective value there, and whatever trial payload the caller's
/// evaluation closure produced (typically the retracted point, so the caller
/// need not recompute it).
#[derive(Clone, Copy, Debug)]
pub struct AcceptedStep<P> {
    /// The step scale `t` at which the trial was accepted.
    pub step: f64,
    /// The objective value at the accepted trial.
    pub value: f64,
    /// The caller-produced payload for the accepted trial (e.g. the new point).
    pub payload: P,
}

/// Configuration for [`backtracking_line_search`].
#[derive(Clone, Copy, Debug)]
pub struct BacktrackConfig {
    /// Initial step scale `t₀` (the migrated geodesic-descent sites start at the
    /// unit Karcher step `t = 1`).
    pub initial_step: f64,
    /// Multiplicative factor applied to a rejected trial (`< 1`).
    pub contraction: f64,
    /// Maximum number of trials before giving up.
    pub max_steps: usize,
}

impl Default for BacktrackConfig {
    /// The schedule the geometry Fréchet-mean drivers used: `t₀ = 1`, halving,
    /// [`MAX_BACKTRACK_HALVINGS`](constants::MAX_BACKTRACK_HALVINGS) trials.
    fn default() -> Self {
        Self {
            initial_step: 1.0,
            contraction: constants::BACKTRACK_CONTRACTION,
            max_steps: constants::MAX_BACKTRACK_HALVINGS,
        }
    }
}

/// Armijo backtracking line search.
///
/// Starting from `config.initial_step`, repeatedly evaluate the trial step and
/// test it against the caller's `accept` predicate, contracting the step by
/// `config.contraction` after each rejection, for at most `config.max_steps`
/// trials.
///
/// The two closures split responsibility so the caller keeps full control of its
/// numerics:
///
///  * `trial(t)` evaluates the step of scale `t`. It returns
///    - `Ok(Some((f, payload)))` — the trial is well defined with objective `f`;
///      `payload` is threaded into [`AcceptedStep`] if this trial is accepted;
///    - `Ok(None)` — the step is *invalid* (e.g. it left the manifold domain);
///      the search shrinks and retries without consulting `accept`;
///    - `Err(e)` — an unrecoverable evaluation error; it aborts the search and
///      is propagated to the caller (matching sites that used `?` inside the
///      loop body).
///  * `accept(t, f)` is the sufficient-decrease predicate. It receives the step
///    scale and the trial objective and returns whether to accept. The caller
///    inlines its exact Armijo arithmetic here (using [`constants::ARMIJO_C1`]
///    and [`armijo_roundoff_cushion`]), so the acceptance threshold is bit-for-
///    bit identical to the pre-migration code.
///
/// Returns `Ok(Some(accepted))` on the first accepted trial, `Ok(None)` if the
/// budget is exhausted with no acceptance (the caller then falls back to its
/// best iterate), or `Err(e)` if a trial evaluation failed unrecoverably.
pub fn backtracking_line_search<P, E>(
    config: BacktrackConfig,
    mut trial: impl FnMut(f64) -> Result<Option<(f64, P)>, E>,
    accept: impl Fn(f64, f64) -> bool,
) -> Result<Option<AcceptedStep<P>>, E> {
    let mut t = config.initial_step;
    for _ in 0..config.max_steps {
        if let Some((value, payload)) = trial(t)? {
            if accept(t, value) {
                return Ok(Some(AcceptedStep {
                    step: t,
                    value,
                    payload,
                }));
            }
        }
        t *= config.contraction;
    }
    Ok(None)
}

/// Geometric schedule for [`escalate_ridge`]: try `δ = initial`, then
/// `initial·growth`, `initial·growth²`, … for at most `max_escalations` trials.
#[derive(Clone, Copy, Debug)]
pub struct RidgeSchedule {
    /// First ridge `δ₀` attempted.
    pub initial: f64,
    /// Geometric growth factor applied to a failed ridge (`> 1`).
    pub growth: f64,
    /// Maximum number of ridge values attempted before reporting exhaustion.
    pub max_escalations: usize,
}

impl RidgeSchedule {
    /// Construct a schedule with the shared [`RIDGE_GROWTH`](constants::RIDGE_GROWTH)
    /// factor, supplying only the initial ridge and the escalation cap.
    #[must_use]
    pub fn geometric(initial: f64, max_escalations: usize) -> Self {
        Self {
            initial,
            growth: constants::RIDGE_GROWTH,
            max_escalations,
        }
    }
}

/// A ridge escalation that succeeded: the operation's result plus the ridge and
/// the (1-based) escalation index at which it succeeded (`escalations == 1`
/// means the first ridge `δ₀` worked).
#[derive(Clone, Copy, Debug)]
pub struct RidgeSuccess<T> {
    /// The successful operation result.
    pub value: T,
    /// The ridge `δ` at which the attempt succeeded.
    pub ridge: f64,
    /// 1-based index of the successful escalation.
    pub escalations: usize,
}

/// A ridge escalation that exhausted its schedule without a success. Carries the
/// ridge the schedule would advance to next, so a caller can log it or route to
/// a different fallback (e.g. an eigen-floor solve).
#[derive(Clone, Copy, Debug)]
pub struct RidgeExhausted {
    /// The ridge value reached after the final (failed) escalation was grown.
    pub next_ridge: f64,
    /// Number of escalations attempted (`== schedule.max_escalations`).
    pub escalations: usize,
}

/// Geometric ridge / Levenberg–Marquardt escalation.
///
/// Repeatedly invoke `attempt(δ)` with `δ` following the geometric
/// [`RidgeSchedule`]; return the first `Some` as a [`RidgeSuccess`]. If every
/// scheduled ridge fails (`attempt` returns `None`), return [`RidgeExhausted`]
/// with the count and the next ridge the schedule would have grown to.
///
/// This owns only the *escalation control flow*; the caller's `attempt` closure
/// performs the actual ridged operation (add `δ` to a diagonal, factor, solve,
/// …) and decides success via `Some`/`None`. A bare (unridged) attempt and any
/// terminal fallback beyond the schedule are the caller's responsibility, kept
/// out of the primitive so it stays operation-agnostic.
pub fn escalate_ridge<T>(
    schedule: RidgeSchedule,
    mut attempt: impl FnMut(f64) -> Option<T>,
) -> Result<RidgeSuccess<T>, RidgeExhausted> {
    let mut ridge = schedule.initial;
    for escalation in 1..=schedule.max_escalations {
        if let Some(value) = attempt(ridge) {
            return Ok(RidgeSuccess {
                value,
                ridge,
                escalations: escalation,
            });
        }
        ridge *= schedule.growth;
    }
    Err(RidgeExhausted {
        next_ridge: ridge,
        escalations: schedule.max_escalations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_search_accepts_first_trial_when_predicate_holds() {
        let out = backtracking_line_search::<f64, ()>(
            BacktrackConfig::default(),
            |t| Ok(Some((10.0 - t, t))),
            |_t, f| f <= 9.5,
        )
        .unwrap()
        .expect("should accept");
        assert_eq!(out.step, 1.0);
        assert_eq!(out.value, 9.0);
        assert_eq!(out.payload, 1.0);
    }

    #[test]
    fn line_search_contracts_until_accept() {
        // Reject until the step has halved twice (t = 0.25).
        let out = backtracking_line_search::<f64, ()>(
            BacktrackConfig::default(),
            |t| Ok(Some((t, t))),
            |t, _f| t <= 0.3,
        )
        .unwrap()
        .expect("should accept after contraction");
        assert_eq!(out.step, 0.25);
    }

    #[test]
    fn line_search_exhaustion_returns_none() {
        let out = backtracking_line_search::<f64, ()>(
            BacktrackConfig {
                initial_step: 1.0,
                contraction: 0.5,
                max_steps: 4,
            },
            |t| Ok(Some((t, t))),
            |_t, _f| false,
        )
        .unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn line_search_invalid_trial_shrinks_without_consulting_accept() {
        // First two trials are invalid (None); third (t = 0.25) is valid and
        // accepted. `accept` must never see the invalid steps.
        let out = backtracking_line_search::<f64, ()>(
            BacktrackConfig::default(),
            |t| {
                if t > 0.3 {
                    Ok(None)
                } else {
                    Ok(Some((t, t)))
                }
            },
            |t, _f| {
                assert!(t <= 0.3, "accept saw an invalid step scale {t}");
                true
            },
        )
        .unwrap()
        .expect("should accept the first valid trial");
        assert_eq!(out.step, 0.25);
    }

    #[test]
    fn line_search_propagates_trial_error() {
        let out: Result<Option<AcceptedStep<f64>>, &str> = backtracking_line_search(
            BacktrackConfig::default(),
            |_t| Err("boom"),
            |_t, _f| true,
        );
        assert_eq!(out, Err("boom"));
    }

    #[test]
    fn ridge_succeeds_at_first_that_works() {
        // Fail below δ = 5, succeed at or above.
        let out = escalate_ridge(RidgeSchedule::geometric(1.0, 16), |d| {
            if d >= 5.0 { Some(d) } else { None }
        })
        .expect("should succeed");
        // δ: 1, 10 → 10 is the first ≥ 5, at escalation 2.
        assert_eq!(out.ridge, 10.0);
        assert_eq!(out.escalations, 2);
    }

    #[test]
    fn ridge_exhaustion_reports_next_ridge_and_count() {
        let err = escalate_ridge::<f64>(
            RidgeSchedule {
                initial: 1.0,
                growth: 10.0,
                max_escalations: 3,
            },
            |_d| None,
        )
        .unwrap_err();
        // Tried 1, 10, 100; grown once more → 1000.
        assert_eq!(err.escalations, 3);
        assert_eq!(err.next_ridge, 1000.0);
    }
}
