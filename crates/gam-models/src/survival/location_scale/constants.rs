use super::*;

pub(crate) const SURVIVAL_ROW_PARALLEL_THRESHOLD: usize = 256;

pub(crate) const SURVIVAL_ROW_PARALLEL_CHUNK: usize = 64;

/// Relative slack tolerating round-off when checking a represented
/// nonnegativity / linear-inequality constraint (`value < -tol·max(1, |scale|)`
/// rejects). A coefficient or slack that is negative only by this much is
/// floating-point noise about an active constraint, not a real violation.
pub(crate) const CONSTRAINT_NONNEGATIVITY_REL_TOL: f64 = 1e-10;

/// Absolute feasibility tolerance of the monotone time-derivative cone that the
/// DOWNSTREAM consumers actually enforce — the active-set QP entry gate
/// (`check_linear_feasibility`) and the cone projection
/// (`project_onto_linear_constraints`) both certify feasibility to this `1e-8`
/// (gam#1108/#797). The strictly-interior projection lands with ~1e-6 of margin,
/// so a converged iterate clears this gate with room, but accumulated round-off
/// over an inner solve can leave a binding guard row at slack ~-1e-9..-1e-8 —
/// numerically AT the boundary, not a real violation.
///
/// The post-update sanity check ([`validate_linear_constraints`]) must therefore
/// accept any β the consumer gate accepts: its tolerance is floored at this
/// value so it never rejects a round-off-feasible iterate the rest of the
/// pipeline treats as feasible. Before #1569 it used only the much stricter
/// `CONSTRAINT_NONNEGATIVITY_REL_TOL` (1e-10·scale); once the spectrum-branch
/// α-crush bypass (gam#1569) let the aggressive heteroscedastic survival-LS solve
/// run to its final inner refit, that refit's cone-projected β routinely landed
/// at slack ~-6.6e-9 — feasible to the 1e-8 gate but a hard error at 1e-10 — so
/// the otherwise-converged fit failed on a pure numerical-precision mismatch.
///
/// It is the primal-feasibility contract itself, referenced rather than
/// re-spelled (gam#2719) — a local name for where it is used, not a second
/// number that can drift away from the one the solver actually honours.
///
/// The consumers here apply it to a RAW slack while the contract is stated on
/// unit-normalized rows. That direction is the safe one: the guard rows are
/// pre-normalized by `max(‖row‖, |rhs|, 1)`, so `‖a‖ ≤ 1` and `raw ≤ scaled`,
/// and a raw floor at the contract is therefore never TIGHTER than the gate it
/// exists to agree with. It cannot reject a β the pipeline calls feasible,
/// which is the whole property #1569 needed.
pub(crate) const MONOTONE_CONE_FEASIBILITY_GATE_TOL: f64 =
    gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;

/// Target byte budget for one row-chunk when streaming a design matrix's
/// trailing columns into a dense buffer. The per-chunk row count is derived as
/// `BUDGET / (p · sizeof(f64))`, so wide designs use proportionally fewer rows
/// per chunk and the working set stays near this size regardless of `p`.
/// Imported, not transcribed (#2704): the doc above describes exactly the
/// library row-chunk target, so it takes that value by name.
pub(crate) const ROW_CHUNK_BYTE_BUDGET: usize =
    gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;

/// Relative floor on the monotonicity-guard round-off slack: the compensated
/// subtraction's low-part residual is the primary slack estimate, but this
/// `1e-12 × (1 + ‖state‖∞)` term remains as a floor for moderate-magnitude
/// inputs where the residual underestimates accumulated error.
pub(crate) const MONOTONICITY_GUARD_SLACK_REL: f64 = 1e-12;

/// Location-scale guard policy: a degenerate `guard == 0` (a bare
/// non-negativity request on `q'(t)`) is admissible here.
pub(crate) const LOCATION_SCALE_GUARD_POLICY: GuardPolicy = GuardPolicy::NonNegative;

pub(crate) const DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD: u64 = 200_000;

pub(crate) const DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD: usize = 100_000;

pub(crate) const DENSE_ROW_CHUNKS_PER_THREAD: usize = 4;
