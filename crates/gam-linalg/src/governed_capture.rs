//! Thread-local record of every memory-governor densification decision.
//!
//! # Why this exists
//!
//! Several hot kernels pick between two numerically different routes based on
//! whether the process-wide [`MemoryGovernor`](gam_runtime::resource::MemoryGovernor)
//! admits a dense buffer — `xt_diag_x_symmetric`'s "reserve-or-stream" arm is
//! the canonical one: a granted reservation runs a dense BLAS crossproduct, a
//! refusal runs a streaming CSC accumulation, and the two do not agree
//! bit-for-bit because they sum in different orders.
//!
//! That makes the *decision* an input to the answer, and gh#2486 is the report
//! that identical fits on one host returned different results. Investigating it
//! ran into a specific epistemic problem worth naming here, because it is what
//! this module is for: when a run shows no divergence, that is only evidence of
//! "the governed branch is not the carrier" **if the branch was actually
//! reached**. A fixture that never crosses the branch is silent about it, not
//! exculpatory — and three of that issue's refuted candidates died on
//! reachability rather than on behaviour. A null result without reachability
//! evidence cannot be distinguished from a fixture that missed the mechanism.
//!
//! So the capture records, per decision: the caller's context string, the
//! footprint it asked for, and which arm it got. "Branch reached 9 times, took
//! the dense arm every time, answers identical" is an interpretable null.
//! "Answers identical" on its own is not.
//!
//! # Why thread-local, and why not an environment variable
//!
//! Ambient process-wide state is exactly the thing under suspicion here — an
//! env-gated instrument for this same issue was reverted from main for tripping
//! the repo's ban on `std::env::var` conditionals in non-test code, and the
//! deeper objection stands on its own: a diagnostic that changes behaviour
//! through the environment cannot be used to study a bug about environment
//! sensitivity. The capture is therefore opt-in per thread, off by default, and
//! costs one thread-local borrow of a `None` when nobody is listening.

use std::cell::RefCell;

/// Which route a governed densification request actually took.
///
/// The distinction that matters to a caller is binary — dense or not — but the
/// refusal *reason* is what tells an investigator whether the governor was
/// genuinely under pressure or the request was structurally impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernedArm {
    /// The reservation was granted and the dense buffer was materialized.
    Admitted,
    /// A dense copy was already cached, so the bytes were charged elsewhere.
    /// Structurally the dense arm, but it exercises no budget.
    CacheHit,
    /// The governor refused: the dense footprint does not fit the live ledger.
    /// This is the arm whose alternate route can differ numerically.
    Refused,
    /// The request never reached the governor — the footprint was not
    /// representable, or policy forbade materialization outright.
    Ineligible,
}

/// One governed densification decision.
#[derive(Clone, Debug)]
pub struct GovernedDecision {
    /// The caller-supplied site label, e.g. `"xt_diag_x_symmetric dense sparse route"`.
    pub context: String,
    pub nrows: usize,
    pub ncols: usize,
    /// Dense footprint requested, or `None` when it was not computable.
    pub bytes: Option<usize>,
    pub arm: GovernedArm,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<GovernedDecision>>> = const { RefCell::new(None) };
}

/// Record one decision. Cheap and silent when capture is off.
pub(crate) fn record_governed_decision(
    context: &str,
    nrows: usize,
    ncols: usize,
    bytes: Option<usize>,
    arm: GovernedArm,
) {
    CAPTURE.with(|slot| {
        if let Ok(mut borrowed) = slot.try_borrow_mut()
            && let Some(decisions) = borrowed.as_mut()
        {
            decisions.push(GovernedDecision {
                context: context.to_string(),
                nrows,
                ncols,
                bytes,
                arm,
            });
        }
    });
}

