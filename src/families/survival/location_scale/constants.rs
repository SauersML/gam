use super::*;

pub(crate) const SURVIVAL_ROW_PARALLEL_THRESHOLD: usize = 256;

pub(crate) const SURVIVAL_ROW_PARALLEL_CHUNK: usize = 64;

/// Relative slack tolerating round-off when checking a represented
/// nonnegativity / linear-inequality constraint (`value < -tol·max(1, |scale|)`
/// rejects). A coefficient or slack that is negative only by this much is
/// floating-point noise about an active constraint, not a real violation.
pub(crate) const CONSTRAINT_NONNEGATIVITY_REL_TOL: f64 = 1e-10;

/// Maximum number of Dykstra alternating-projection sweeps when projecting an
/// initial coefficient guess onto the represented linear inequality
/// constraints. The projection converges geometrically; this caps the rare
/// near-degenerate constraint set and keeps the warm-start best-effort.
pub(crate) const DYKSTRA_PROJECTION_MAX_SWEEPS: usize = 100;

/// Absolute feasibility tolerance at which the Dykstra projection sweep is
/// declared converged (max constraint violation below this stops the loop).
pub(crate) const DYKSTRA_PROJECTION_TOL: f64 = 1e-10;

/// Squared-row-norm floor below which a constraint row is treated as
/// structurally empty and skipped during Dykstra projection (avoids dividing
/// the projection step by a vanishing normal).
pub(crate) const DYKSTRA_ROW_DEGENERACY_FLOOR: f64 = 1e-18;

/// Relative tolerance (× the largest |eigenvalue|) for accepting a covariance
/// block as positive semidefinite, floored by an absolute value so an
/// all-tiny-eigenvalue block is not rejected on pure round-off. Eigenvalues
/// below `-tol` flag a genuine indefinite block.
pub(crate) const PSD_EIGENVALUE_REL_TOL: f64 = 1e-12;

pub(crate) const PSD_EIGENVALUE_ABS_FLOOR: f64 = 1e-14;

/// Levenberg damping schedule for the direct parametric-AFT Newton solve. When
/// the Hessian is not Cholesky-factorizable, damping starts at
/// `INITIAL × max(1, ‖diag H‖∞)`, grows by `GROWTH` per failed factorization,
/// and the solve aborts once it would exceed `MAX × max(1, ‖diag H‖∞)` (the
/// Hessian is then numerically unsalvageable). All three scale with the
/// Hessian's diagonal magnitude so the schedule is units-invariant.
pub(crate) const LEVENBERG_INITIAL_DAMPING_REL: f64 = 1e-8;

pub(crate) const LEVENBERG_DAMPING_GROWTH: f64 = 10.0;

pub(crate) const LEVENBERG_MAX_DAMPING_REL: f64 = 1e8;

/// Outer (smoothing-parameter) loop budget for the blockwise location-scale
/// fit: at most this many outer iterations, stopping once the outer relative
/// change falls below the tolerance. The dead-flat time-smoothing ridge of the
/// constant-scale case is what makes a finite cap necessary.
pub(crate) const BLOCKWISE_OUTER_MAX_ITER: usize = 60;

pub(crate) const BLOCKWISE_OUTER_TOL: f64 = 1e-5;

/// Lower bound on the gradient tolerance handed to the reduced parametric-AFT
/// direct MLE. The inner-solve tolerance can be configured arbitrarily small;
/// flooring it here keeps the Newton stopping test above the noise of the
/// log-likelihood gradient evaluation.
pub(crate) const REDUCED_AFT_GRAD_TOL_FLOOR: f64 = 1e-8;

/// Relative ridge added to the normal-equations diagonal of the structural
/// time-coefficient warm-start least squares (× the largest diagonal of XᵀX,
/// floored at 1). Stabilizes the best-effort guess against a rank-deficient
/// derivative design without materially biasing it.
pub(crate) const STRUCTURAL_GUESS_RIDGE_REL: f64 = 1e-6;

/// Floor on the exit age when forming the `1/age` structural-derivative target
/// for the time warm-start, guarding against a divide-by-zero at age 0.
pub(crate) const STRUCTURAL_GUESS_AGE_FLOOR: f64 = 1e-9;

/// Target byte budget for one row-chunk when streaming a design matrix's
/// trailing columns into a dense buffer. The per-chunk row count is derived as
/// `BUDGET / (p · sizeof(f64))`, so wide designs use proportionally fewer rows
/// per chunk and the working set stays near this size regardless of `p`.
pub(crate) const ROW_CHUNK_BYTE_BUDGET: usize = 8 * 1024 * 1024;

/// Relative floor on the monotonicity-guard round-off slack: the compensated
/// subtraction's low-part residual is the primary slack estimate, but this
/// `1e-12 × (1 + ‖state‖∞)` term remains as a floor for moderate-magnitude
/// inputs where the residual underestimates accumulated error.
pub(crate) const MONOTONICITY_GUARD_SLACK_REL: f64 = 1e-12;

/// Location-scale guard policy: a degenerate `guard == 0` (a bare
/// non-negativity request on `q'(t)`) is admissible here, and feasibility of
/// coefficient-free rows uses the family's historical absolute slack.
pub(crate) const LOCATION_SCALE_GUARD_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
    guard_policy: GuardPolicy::NonNegative,
    feasibility: FeasibilityTolerance::LegacyAbsolute,
};

pub(crate) const DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD: u64 = 200_000;

pub(crate) const DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD: usize = 100_000;

pub(crate) const DENSE_ROW_CHUNKS_PER_THREAD: usize = 4;
