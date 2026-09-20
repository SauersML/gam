use super::*;

/// Inner-PIRLS progress channel between the outer loop and the inner solver.
///
/// Every inner solve runs to its residual-based forcing tolerance under the
/// full `pirls_max_iterations` budget (#3536): the outer loop never truncates
/// the inner Newton by an iteration count. The inner solver
/// (`execute_pirls_if_needed`) writes back `last_iters` / `last_converged`
/// after each solve so the outer certificate can refuse a stationarity claim
/// taken at a non-converged inner mode.
///
/// All atomics are owned by `RemlObjectiveState`; the bridges hold `Arc`
/// clones. `last_iters == 0` means "no inner solve has reported yet".
#[derive(Clone, Debug)]
pub(crate) struct InnerProgressFeedback {
    /// Count of accepted outer steps observed via the `OuterAcceptObserver`
    /// plugged into `opt`'s solver. Routes that see trial-and-rejection probing
    /// (ARC dense, matrix-free TR) count only accepted steps here; the
    /// custom-family joint path reads it to key its warm start (#2668).
    pub accepted_iter: Arc<AtomicUsize>,
    pub last_iters: Arc<AtomicUsize>,
    pub last_converged: Arc<AtomicBool>,
    /// #2349 — one-shot "re-evaluate COLD" pulse raised by the outer
    /// cost-stall guard when it grants a STUCK-stall escape.
    ///
    /// A stuck stall means the outer objective has flatlined over the
    /// no-improvement window while the projected gradient is still far above
    /// the certified-stationary band — i.e. genuine feasible descent remains,
    /// but the optimizer cannot see it. On a near-separating profiled fit the
    /// cause is warm-start value HYSTERESIS: successive trial-ρ inner solves
    /// are warm-started from the previous iterate's coefficient mode, and on a
    /// near-flat inner ridge (vanishing softmax Fisher curvature at the simplex
    /// boundary) two warm starts converge to different ridge points whose
    /// Laplace `½log|H(β)|` — hence the profiled objective — differ by more
    /// than the outer descent resolution. The optimizer's step-acceptance then
    /// cannot distinguish real descent from that hysteresis and the loop grinds
    /// to `max_iter` at a non-stationary point.
    ///
    /// A fully converged warm solve still lands on the warm-biased ridge point,
    /// so the escape asks the next outer evaluation(s) to re-solve the inner
    /// problem COLD — trajectory-independent — restoring a consistent
    /// objective surface the optimizer can descend. `false` = no pending
    /// request. Only the custom-family joint path consumes it today; every
    /// other path leaves it inert.
    pub force_cold: Arc<AtomicBool>,
}
