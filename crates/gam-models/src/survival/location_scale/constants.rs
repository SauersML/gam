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
pub(crate) const MONOTONE_CONE_FEASIBILITY_GATE_TOL: f64 = 1e-8;

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

/// Relative floor for the coupled-survival SCALE-COUPLED block trust-region
/// metric (issue #1569). The free scale predictor `η_σ` enters the likelihood
/// through the standardized index `u = inv_sigma·(h − η_t)` with
/// `inv_sigma = exp(−η_σ)`, so `∂u/∂η_t = −inv_sigma`: the LOCATION (threshold)
/// and LOG-σ channels carry an `exp(−η_σ)` factor in their gradient and an
/// `exp(−2 η_σ)` factor in their likelihood-Hessian diagonal
/// `Σ_r exp(−2 η_σ,r) X_{rj}²` (the flexible time baseline `h` does NOT — its
/// `∂u/∂h = 1` is scale-free). When `exp(−η_σ)` is large on a few rows the metric
/// is dominated by them for the coefficients they load on, while a coefficient
/// loading mostly on large-σ rows is metric-STARVED. The affine-covariant
/// Moré–Sorensen step then over-reaches on the starved coordinates and the inner
/// trust loop grinds. We floor every scale-coupled metric entry at this fraction
/// of the block's MAXIMUM metric entry, capping the `exp(−η_σ)`-induced condition
/// number so no coordinate is starved. The floor is derived ENTIRELY from the
/// scale-coupled diagonal (no knob, no flag): it scales with `exp(−2 η_σ)`
/// automatically and self-vanishes at the KKT fixed point (it shapes the step
/// norm only, never the converged β). `1e-6` keeps the floor six orders below the
/// block's dominant curvature so a well-conditioned block (constant or mild
/// scale) sees a byte-identical metric, while a sharply heteroscedastic block has
/// its metric dynamic range capped at `1e6`.
pub(crate) const SCALE_COUPLED_TRUST_METRIC_FLOOR_REL: f64 = 1e-6;

/// Outer (smoothing-parameter) loop budget for the blockwise location-scale
/// fit: at most this many outer iterations, stopping once the outer relative
/// change falls below the tolerance. The dead-flat time-smoothing ridge of the
/// constant-scale case is what makes a finite cap necessary.
pub(crate) const BLOCKWISE_OUTER_MAX_ITER: usize = 60;

pub(crate) const BLOCKWISE_OUTER_TOL: f64 = 1e-5;

/// Objective-suboptimality floor handed to the reduced parametric-AFT direct
/// MLE as its Newton stopping tolerance. The inner-solve tolerance can be
/// configured arbitrarily small; flooring it here keeps the stopping test — the
/// half-Newton-decrement `½·gᵀH⁻¹g`, an estimate of the log-likelihood gap
/// `ℓ(θ*) − ℓ(θ)` — above the round-off noise of the objective evaluation.
///
/// This is deliberately an OBJECTIVE tolerance, not a gradient tolerance: the
/// log-likelihood gradient is a SUM over the `n` observations, so its attainable
/// sup-norm floor grows like `n·ε`, and an absolute gradient tolerance therefore
/// spuriously fails to converge on perfectly benign data as `n` grows (gam#2112).
/// The Newton decrement `gᵀH⁻¹g` divides the n-scaled gradient by the n-scaled
/// curvature and is affine-invariant, so a single fixed tolerance certifies
/// stationarity uniformly across `n`. See `fit_parametric_aft_direct_mle`.
pub(crate) const REDUCED_AFT_OBJ_TOL_FLOOR: f64 = 1e-8;

/// Near-stationary acceptance tolerance for a stalled line search in the reduced
/// parametric-AFT direct MLE. When the damped-Newton ascent direction admits no
/// Armijo-sufficient step — i.e. `ℓ` can no longer be increased to numerical
/// precision — AND the half-Newton-decrement `½·gᵀH⁻¹g` is below this bound, the
/// iterate IS the numerical MLE and is accepted rather than reported as a
/// convergence failure (gam#2112). A decrement above this bound at a stalled line
/// search signals a genuinely wrong curvature model (not an MLE) and stays a hard
/// error. The bound is generous relative to the primary objective tolerance
/// (≈`1e-7`): a remaining gap of `1e-4` nats is a Mahalanobis distance of only
/// `√(2·1e-4) ≈ 0.014` standard errors from the optimum, so it accepts a fully
/// converged fit while still separating it from a real optimizer breakdown, whose
/// decrement is orders of magnitude larger.
pub(crate) const REDUCED_AFT_NEWTON_STALL_TOL: f64 = 1e-4;

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
    feasibility: FeasibilityTolerance::AbsoluteScaled,
};

pub(crate) const DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD: u64 = 200_000;

pub(crate) const DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD: usize = 100_000;

pub(crate) const DENSE_ROW_CHUNKS_PER_THREAD: usize = 4;
