//! The blockwise inner-fit driver (`inner_blockwise_fit`), the joint
//! Newton polish step, and inner-result assembly, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via
//! `custom_family` so existing paths stay stable.

use super::blockwise_solve::BlockWorkingSetUpdaterExt;
use super::*;
use gam_solve::row_measure::RowSubsampleMaskExt;

mod exact_joint_fit;

use exact_joint_fit::fit_exact_joint;

/// The certified product requested from one coefficient solve.
///
/// A continuation waypoint needs the exact constrained coefficient mode (and
/// its KKT/returned-curvature certificate) to warm-start the next waypoint. It
/// does not need a Laplace scalar. Keeping that distinction typed prevents an
/// interior corrector from paying for determinant artifacts that no caller can
/// consume, without weakening any fact that makes the returned coefficient
/// state a mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InnerFitProduct {
    LaplaceReady,
    CoefficientMode,
}

impl InnerFitProduct {
    fn requires_laplace_artifacts(self) -> bool {
        self == Self::LaplaceReady
    }
}

/// Ownership boundary for the authoritative coupled exact-joint engine.
///
/// The outer driver owns route selection and common setup. Once it selects the
/// exact-joint route, this carrier transfers the complete initialized inner
/// state to that engine; the engine always returns a complete inner result and
/// never falls through to the distinct coordinate/blockwise algorithm.
struct ExactJointFitContext<'a, F> {
    family: &'a F,
    specs: &'a [ParameterBlockSpec],
    block_log_lambdas: &'a [Array1<f64>],
    options: &'a BlockwiseFitOptions,
    states: Vec<ParameterBlockState>,
    s_lambdas: Vec<Array2<f64>>,
    ridge: f64,
    joint_bundle: Option<&'a gam_problem::JointPenaltyBundle>,
    lastobjective: f64,
    converged: bool,
    cycles_done: usize,
    terminal_convergence_state: Option<gam_problem::InnerConvergenceTerminalState>,
    inner_tol: f64,
    inner_max_cycles: usize,
    cached_active_sets: Vec<Option<Vec<usize>>>,
    current_log_likelihood: f64,
    cached_eval: Option<FamilyEvaluation>,
    cached_joint_gradient: Option<Array1<f64>>,
    cached_joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    cached_joint_hessian_source: Option<JointHessianSource>,
    objective_state: crate::assembly::InnerObjectiveState,
    joint_workspace_requested: bool,
    matrix_free_joint_requested: bool,
    total_joint_n: usize,
    prelude_log: bool,
    inner_started: &'a std::time::Instant,
    last_residual_tol: f64,
    product: InnerFitProduct,
}

pub(crate) fn beta_cache_keys_match_bitwise(lhs: &Array1<f64>, rhs: &Array1<f64>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) struct ExactJointModeCurvatureCertificate {
    pub(crate) workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    pub(crate) minimum_whitened_eigenvalue: f64,
    pub(crate) numerical_floor: f64,
    /// Whether this certificate actually assembled the active Jeffreys
    /// second-order completion. This is causal accounting, not a proxy based
    /// only on the family's capability flag.
    pub(crate) jeffreys_completion_assembled: bool,
    /// Coefficient-space direction of the minimum-curvature mode, expressed in
    /// the FULL joint layout (mapped through the active-face tangent when the
    /// mode was certified on a reduced face). Populated only when the mode has
    /// resolvable negative curvature — the direction a saddle-escape steps
    /// along. `None` otherwise (PSD mode, fully pinned face, or no free
    /// direction). Its exact curvature is `minimum_whitened_eigenvalue`: for the
    /// whitened unit eigenvector `v` with eigenvalue `γ_min`, the raw
    /// coefficient direction `δ = D^{-1/2} v` satisfies `δᵀ H_pen δ = γ_min`, so
    /// a step `s·δ` lowers the quadratic model by `½ s² |γ_min|`.
    pub(crate) negative_curvature_direction: Option<Array1<f64>>,
}

impl ExactJointModeCurvatureCertificate {
    pub(crate) fn has_resolvable_negative_curvature(&self) -> bool {
        self.minimum_whitened_eigenvalue < -self.numerical_floor
    }
}

/// Whether the constrained candidate, rather than an ambient unconstrained
/// spectrum step, owns trust-region globalization.
///
/// Every reduced-face variant is already the physical-H solution on the
/// feasible face. Replacing it with an ambient unconstrained spectrum step
/// moves off that face while incorrectly retaining its endpoint row ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReducedFaceCandidateKind {
    /// The original, unmodified reduced Hessian and final constraint cone
    /// certify the constrained Newton equation.
    ExactNewton,
    /// The physical reduced Hessian is positive-semidefinite and its
    /// Moore--Penrose interior solution certifies, but it is not strictly
    /// positive definite.
    ReducedNewton,
    /// The exact physical Moré--Sorensen solution has a positive trust
    /// multiplier or a hard-case minimum-eigenvector fill.
    RegularizedNewton,
}

fn constrained_search_delta_owns_trust_step(
    reduced_face_kind: Option<ReducedFaceCandidateKind>,
    has_active_set: bool,
    ambient_spectrum_has_negative_curvature: Option<bool>,
) -> bool {
    reduced_face_kind.is_some()
        || (has_active_set && ambient_spectrum_has_negative_curvature == Some(false))
}

/// The per-cycle fixed inputs of the trust-region quadratic model, so a candidate
/// step's predicted decrease can be evaluated more than once per attempt without
/// re-threading six unchanging arguments through every call.
struct JointTrustRegionModel<'a> {
    source: &'a JointHessianSource,
    ranges: &'a [(usize, usize)],
    s_lambdas: &'a [Array2<f64>],
    diagonal_ridge: f64,
    joint_bundle: Option<&'a gam_problem::JointPenaltyBundle>,
    jeffreys_curvature: Option<&'a Array2<f64>>,
}

impl JointTrustRegionModel<'_> {
    /// Predicted decrease of the model at `delta`, on the TRUE penalized
    /// (Firth-augmented) Hessian — bit-identically the model the accept/reject
    /// `predicted_reduction` is computed on.
    ///
    /// Exists so two feasible candidate steps can be ranked on the quantity the
    /// trust region already judges every step by, rather than on a threshold
    /// (gam#2621). `None` when the Hessian application or the model value is not
    /// finite, so a caller ranks against a real number or not at all.
    fn predicted_reduction_at(
        &self,
        rhs: &Array1<f64>,
        delta: &Array1<f64>,
        hpen_delta: &mut Array1<f64>,
        penalty_scratch: &mut Array1<f64>,
    ) -> Option<f64> {
        hpen_delta.fill(0.0);
        apply_joint_penalized_hessian_into_with_workspace(
            self.source,
            self.ranges,
            self.s_lambdas,
            self.diagonal_ridge,
            delta,
            hpen_delta,
            penalty_scratch,
            self.joint_bundle,
        )
        .ok()?;
        if let Some(curvature) = self.jeffreys_curvature {
            let jeffreys_delta = curvature.dot(delta);
            *hpen_delta += &jeffreys_delta;
        }
        let predicted = joint_quadratic_predicted_reduction(rhs, hpen_delta, delta);
        predicted.is_finite().then_some(predicted)
    }
}

/// Damping `α` of the self-concordant damped-Newton phase (gam#979 CTN/
/// marginal-slope barrier crawl), or `None` once the undamped machinery owns
/// the step.
///
/// For a family whose penalized inner objective is self-concordant
/// (`CustomFamily::inner_objective_is_self_concordant` — the change-of-
/// variables `−log h'` barrier with `h'` affine in β), the classical damped
/// Newton step `α·δ_N` with `α = 1/(1+λ_N)` guarantees an objective decrease
/// of at least `λ_N − log(1+λ_N)` per step while `λ_N` is large, and plain
/// Newton is quadratically convergent once `λ_N < (3−√5)/2` (Nesterov,
/// *Introductory Lectures on Convex Optimization*, Thm 4.1.12). `λ_N` is the
/// Newton decrement `sqrt(gᵀH⁻¹g)`, recovered from the spectrum's model
/// decrease `newton_decrement() = ½λ_N²` of the full modified-Newton step.
///
/// The trust-region ratio search lacks this guarantee on the barrier's
/// `1/h'²` curvature: a full Newton step overshoots the barrier's region of
/// model validity (the Dikin ellipsoid), the ρ-gate/feasibility limiter
/// rejects it, the radius collapses, and the solve accepts only
/// `O(1e-3)·δ_N` fragments — the measured ~0.998×/cycle KKT-residual crawl
/// that exhausts the inner budget (the #979 survival "hang"). `α = 1/(1+λ_N)`
/// is precisely the largest step with a guaranteed decrease, so substituting
/// it for the first trial of each cycle replaces the crawl with the textbook
/// bounded-decrease phase while every rejection still falls back to the
/// unchanged trust-region machinery.
///
/// Returns `None` in the quadratic phase (`λ_N` below the threshold) and on a
/// non-finite or non-positive decrement, where no self-concordance statement
/// is available — the caller's step policy is byte-identical there.
fn self_concordant_damped_step_alpha(newton_decrement: f64) -> Option<f64> {
    if !(newton_decrement.is_finite() && newton_decrement > 0.0) {
        return None;
    }
    let lambda_n = (2.0 * newton_decrement).sqrt();
    // (3 − √5)/2: the classical bound on the decrement below which the FULL
    // Newton step keeps the iterate inside the quadratic-convergence region
    // of a standard self-concordant function.
    const SC_QUADRATIC_PHASE_THRESHOLD: f64 = 0.381_966_011_250_105;
    if !lambda_n.is_finite() || lambda_n < SC_QUADRATIC_PHASE_THRESHOLD {
        return None;
    }
    Some(1.0 / (1.0 + lambda_n))
}

/// Exact reduced Moré--Sorensen step from the generalized eigensystem `(H, D)`.
///
/// `D` is the existing affine-covariant trust/preconditioner metric. Whitening
/// by `D` makes the curvature classification and radius invariant to coefficient
/// units. Positive-definite interior Newton uses `λ=0` and leaves `H` unchanged.
/// Otherwise the shared Moré–Sorensen secular solver chooses the unique
/// `λ >= max(0, -gamma_min)` for the supplied trust radius:
///
/// ```text
/// B = D^(-1/2) H D^(-1/2) = V diag(gamma) V'
/// c = V' D^(-1/2) rhs
/// eta(lambda) = c / (gamma + lambda)
/// ||eta(lambda)||₂ <= trust_radius
/// delta = D^(-1/2) eta(lambda).
/// ```
///
/// In the hard case `eta(lambda)` is the Moore--Penrose base and the exact
/// solution additionally contains `tau * v_min`, where `tau` fills the trust
/// radius. The scalar shift therefore does *not* determine the step. Returning
/// the full primal step (and its opposite hard-case sign) is the semantic
/// boundary that prevents a convexified surrogate metric from silently deleting
/// all negative-curvature motion (#2656).
struct ReducedMoreSorensenStep {
    delta: Array1<f64>,
    alternate_hard_case_delta: Option<Array1<f64>>,
    trust_shift: f64,
    hard_case: bool,
    exact_positive_curvature: bool,
    minimum_shifted_curvature: f64,
    numerical_floor: f64,
}

fn generalized_trust_region_reduced_step(
    reduced_hessian: &Array2<f64>,
    reduced_trust_metric: &Array2<f64>,
    reduced_rhs: &Array1<f64>,
    trust_radius: f64,
) -> Result<ReducedMoreSorensenStep, CustomFamilyError> {
    let dimension = reduced_hessian.nrows();
    if dimension == 0
        || reduced_hessian.ncols() != dimension
        || reduced_trust_metric.nrows() != dimension
        || reduced_trust_metric.ncols() != dimension
        || reduced_rhs.len() != dimension
    {
        return Err(CustomFamilyError::trial_point(
            "generalized reduced trust region requires equal matrix/vector dimensions",
        ));
    }
    if !(trust_radius.is_finite() && trust_radius > 0.0) {
        return Err(CustomFamilyError::trial_point(
            "generalized reduced trust region requires a finite positive radius".to_string(),
        ));
    }

    let mut trust_metric = reduced_trust_metric.clone();
    symmetrize_dense_in_place(&mut trust_metric);
    let (metric_eigenvalues, metric_eigenvectors) =
        FaerEigh::eigh(&trust_metric, faer::Side::Lower).map_err(|error| {
            format!("reduced trust-metric eigendecomposition failed: {error:?}")
        })?;
    if metric_eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(CustomFamilyError::trial_point(
            "reduced trust metric is not positive definite".to_string(),
        ));
    }

    let mut metric_inv_sqrt_columns = metric_eigenvectors.clone();
    for (column, eigenvalue) in metric_eigenvalues.iter().enumerate() {
        let sqrt = eigenvalue.sqrt();
        let inverse_sqrt = sqrt.recip();
        for row in 0..dimension {
            metric_inv_sqrt_columns[[row, column]] *= inverse_sqrt;
        }
    }
    let metric_inv_sqrt = metric_inv_sqrt_columns.dot(&metric_eigenvectors.t());

    let mut whitened_hessian = metric_inv_sqrt.dot(reduced_hessian).dot(&metric_inv_sqrt);
    symmetrize_dense_in_place(&mut whitened_hessian);
    let (generalized_eigenvalues, generalized_eigenvectors) =
        FaerEigh::eigh(&whitened_hessian, faer::Side::Lower).map_err(|error| {
            format!("whitened reduced-Hessian eigendecomposition failed: {error:?}")
        })?;
    if generalized_eigenvalues
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CustomFamilyError::trial_point(
            "whitened reduced Hessian contains non-finite curvature".to_string(),
        ));
    }
    let whitened_rhs = metric_inv_sqrt.dot(reduced_rhs);
    let spectral_rhs = generalized_eigenvectors.t().dot(&whitened_rhs);
    if spectral_rhs.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(
            "whitened reduced trust-region rhs is non-finite".to_string(),
        ));
    }
    let lambda_max_abs = generalized_eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let numerical_floor = joint_hessian_numerical_eigenvalue_floor(lambda_max_abs, dimension);
    let null_cutoff = (KKT_REFUSAL_RANK_TOL * lambda_max_abs).max(numerical_floor);
    let spectrum = whitened_spectrum::WhitenedHessianSpectrum {
        gamma: generalized_eigenvalues.clone(),
        evecs: generalized_eigenvectors.clone(),
        c: spectral_rhs,
        d_inv_sqrt: Array1::ones(dimension),
        lambda_max_abs,
        null_cutoff,
        numerical_floor,
    };
    let trust_step = spectrum.trust_region_step(trust_radius);
    let Some(trust_shift) = trust_step.trust_region_shift else {
        return Err(CustomFamilyError::trial_point(
            "generalized reduced trust region unexpectedly selected reflected fallback",
        ));
    };
    if !(trust_shift.is_finite() && trust_shift >= 0.0) {
        return Err(CustomFamilyError::trial_point(
            "generalized reduced trust-region shift is invalid".to_string(),
        ));
    }
    let exact_positive_curvature = trust_shift == 0.0
        && !trust_step.trust_region_hard_case
        && generalized_eigenvalues
            .iter()
            .all(|value| *value > numerical_floor);
    let minimum_shifted_curvature = generalized_eigenvalues
        .iter()
        .map(|value| *value + trust_shift)
        .fold(f64::INFINITY, f64::min);
    let delta = metric_inv_sqrt.dot(&trust_step.delta);
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(
            "generalized reduced trust-region step is non-finite".to_string(),
        ));
    }

    // `WhitenedHessianSpectrum::assemble` stores the hard-case fill directly
    // in the returned whitened step. Its projection onto the minimum
    // eigenspace is exactly `tau`; reflecting that component produces the
    // second, model-equivalent sign. Constraints can make only one sign
    // feasible, so both must survive until the face/blocker solver sees them.
    let alternate_hard_case_delta = if trust_step.trust_region_hard_case {
        let gamma_min = generalized_eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let min_index = generalized_eigenvalues
            .iter()
            .position(|value| *value == gamma_min)
            .ok_or_else(|| "hard-case reduced spectrum has no minimum mode".to_string())?;
        let min_mode = generalized_eigenvectors.column(min_index);
        let tau = min_mode.dot(&trust_step.delta);
        if !tau.is_finite() {
            return Err(CustomFamilyError::trial_point(
                "hard-case reduced trust-region fill is non-finite".to_string(),
            ));
        }
        let mut alternate_whitened = trust_step.delta.clone();
        alternate_whitened.scaled_add(-2.0 * tau, &min_mode);
        let alternate = metric_inv_sqrt.dot(&alternate_whitened);
        if alternate.iter().any(|value| !value.is_finite()) {
            return Err(CustomFamilyError::trial_point(
                "alternate hard-case reduced trust-region step is non-finite".to_string(),
            ));
        }
        Some(alternate)
    } else {
        None
    };

    Ok(ReducedMoreSorensenStep {
        delta,
        alternate_hard_case_delta,
        trust_shift,
        hard_case: trust_step.trust_region_hard_case,
        exact_positive_curvature,
        minimum_shifted_curvature,
        numerical_floor,
    })
}

/// Upgrade a tolerance-feasible active-set result to a mathematically feasible
/// point without changing either the quadratic objective or its feasible
/// reference.
///
/// The generic QP solver intentionally classifies violations up to
/// `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` as numerically feasible. That contract is
/// appropriate for active-set iteration, but it is weaker than the exact
/// feasibility premise used by the reduced-face descent theorem. When the
/// solver endpoint is on the wrong side of a constraint, intersect the chord
/// from the already-certified feasible reference with the first blocking
/// hyperplane. The ratio-test endpoint is reclassified through the constraint
/// carrier. In the exceptional case where floating-point reconstruction rounds
/// it outward, an ordered-bit bisection selects the largest representable chord
/// step whose *evaluated* endpoint is feasible. This is a finite exact search
/// over binary64 values, not a geometric tolerance or objective fallback.
fn clip_infeasible_candidate_to_certified_feasible_chord(
    constraints: &ConstraintSet,
    reference: &Array1<f64>,
    candidate: &Array1<f64>,
) -> Result<(Array1<f64>, usize, f64), CustomFamilyError> {
    if reference.len() != candidate.len()
        || reference.iter().any(|value| !value.is_finite())
        || candidate.iter().any(|value| !value.is_finite())
    {
        return Err(CustomFamilyError::trial_point(format!(
            "feasible-chord finite/dimension contract failed \
             (reference={}, candidate={})",
            reference.len(),
            candidate.len(),
        )));
    }
    let (reference_violation, reference_worst_row) = constraints
        .max_scaled_violation(reference.view())
        .map_err(|error| {
            CustomFamilyError::trial_point(format!(
                "feasible-chord reference classification failed: {error}"
            ))
        })?;
    let (candidate_violation, candidate_worst_row) = constraints
        .max_scaled_violation(candidate.view())
        .map_err(|error| {
            CustomFamilyError::trial_point(format!(
                "feasible-chord candidate classification failed: {error}"
            ))
        })?;
    if !reference_violation.is_finite()
        || reference_violation > 0.0
        || !candidate_violation.is_finite()
        || candidate_violation <= 0.0
    {
        return Err(CustomFamilyError::trial_point(format!(
            "feasible-chord clipping premise failed \
             (reference_scaled_violation={reference_violation:.6e}@{reference_worst_row:?}, \
             candidate_scaled_violation={candidate_violation:.6e}@{candidate_worst_row:?})"
        )));
    }

    let direction = candidate - reference;
    if direction.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(
            "feasible-chord direction overflowed binary64".to_string(),
        ));
    }
    // The ratio test is an ACCELERATOR here, not the authority.
    //
    // It answers a different question from `max_scaled_violation` — "where
    // does this chord cross a hyperplane, in exact arithmetic" versus "is this
    // EVALUATED point on the wrong side of a row" — and at the magnitudes that
    // reach this function the two disagree. Measured (#2590): a candidate
    // infeasible by `4.701406e-32` on row 34 for which the ratio test reported
    // no blocking row at all. The cheap predicate then vetoed the precise one
    // and the whole fit died, at a function whose entire job is to repair a
    // violation that small.
    //
    // The bisection below needs no ratio test to be correct. Step 0 is the
    // reference, already certified feasible; step 1 is the candidate, whose
    // evaluated violation is positive by the premise checked above. An
    // ordered-bit search on the SAME predicate that declared the infeasibility
    // is therefore always bracketed and always terminates. Use the ratio test
    // only to start that search closer to the answer, and fall back to the
    // whole chord when it declines to answer.
    let (raw_boundary_step, ratio_test_row) = constraints
        .max_feasible_step(reference.view(), direction.view(), &[])
        .map_err(|error| {
            CustomFamilyError::trial_point(format!(
                "feasible-chord boundary ratio test failed: {error}"
            ))
        })?;
    let boundary_step = if raw_boundary_step.is_finite() && (0.0..=1.0).contains(&raw_boundary_step)
    {
        raw_boundary_step
    } else {
        1.0
    };
    // Which row the clip is ATTRIBUTED to. The ratio test's answer when it has
    // one; otherwise the row whose evaluated violation is what forced the clip,
    // which is the row the chord actually ran into.
    let Some(blocking_row) = ratio_test_row.or(candidate_worst_row) else {
        return Err(CustomFamilyError::trial_point(format!(
            "infeasible QP endpoint names no violated row \
             (candidate_scaled_violation={candidate_violation:.6e}@{candidate_worst_row:?}, \
             ratio_test_step={raw_boundary_step:.6e})"
        )));
    };

    let point_at = |step: f64| reference + &(&direction * step);
    let mut certified_step = boundary_step;
    let mut clipped = point_at(certified_step);
    let (mut clipped_violation, mut clipped_worst_row) = constraints
        .max_scaled_violation(clipped.view())
        .map_err(|error| {
            CustomFamilyError::trial_point(format!(
                "feasible-chord boundary classification failed: {error}"
            ))
        })?;
    if !clipped_violation.is_finite() {
        return Err(CustomFamilyError::trial_point(format!(
            "feasible-chord boundary classification is non-finite \
             (step={certified_step:.6e}, blocking_row={blocking_row}, \
             scaled_violation={clipped_violation:.6e}@{clipped_worst_row:?})"
        )));
    }

    if clipped_violation > 0.0 {
        let mut feasible_bits = 0_u64;
        let mut infeasible_bits = certified_step.to_bits();
        if infeasible_bits == 0 {
            return Err(CustomFamilyError::trial_point(format!(
                "certified feasible reference became infeasible at zero chord step \
                 (blocking_row={blocking_row}, \
                 scaled_violation={clipped_violation:.6e}@{clipped_worst_row:?})"
            )));
        }
        while infeasible_bits - feasible_bits > 1 {
            let middle_bits = feasible_bits + (infeasible_bits - feasible_bits) / 2;
            let middle_step = f64::from_bits(middle_bits);
            let middle = point_at(middle_step);
            let (middle_violation, middle_worst_row) = constraints
                .max_scaled_violation(middle.view())
                .map_err(|error| {
                    format!("feasible-chord representable-step classification failed: {error}")
                })?;
            if !middle_violation.is_finite() {
                return Err(CustomFamilyError::trial_point(format!(
                    "feasible-chord representable-step classification is non-finite \
                     (step={middle_step:.6e}, blocking_row={blocking_row}, \
                     scaled_violation={middle_violation:.6e}@{middle_worst_row:?})"
                )));
            }
            if middle_violation <= 0.0 {
                feasible_bits = middle_bits;
            } else {
                infeasible_bits = middle_bits;
            }
        }
        certified_step = f64::from_bits(feasible_bits);
        clipped = point_at(certified_step);
        (clipped_violation, clipped_worst_row) = constraints
            .max_scaled_violation(clipped.view())
            .map_err(|error| {
            format!("feasible-chord final endpoint classification failed: {error}")
        })?;
    }
    if !clipped_violation.is_finite() || clipped_violation > 0.0 {
        return Err(CustomFamilyError::trial_point(format!(
            "feasible-chord endpoint failed mathematical feasibility certification \
             (step={certified_step:.6e}, blocking_row={blocking_row}, \
             scaled_violation={clipped_violation:.6e}@{clipped_worst_row:?})"
        )));
    }
    Ok((clipped, blocking_row, certified_step))
}

/// Solve the physical-H constrained trust-region subproblem on an inequality
/// face, exchanging blockers and invalid multiplier rows to closure.
///
/// For an equality working face `A(beta + delta) = b`, write
/// `delta = delta_p + Zz`, where `delta_p` is the minimum-`D`-norm affine
/// particular and `Z` spans `null(A)`. Then `Z'D delta_p = 0`, so both the
/// objective and trust ball reduce without approximation:
///
/// ```text
/// min_z  1/2 (delta_p + Zz)' H (delta_p + Zz)
///              - rhs' (delta_p + Zz)
/// s.t.   z'(Z'DZ)z <= r² - delta_p'D delta_p.
/// ```
///
/// The reduced Moré--Sorensen solve returns the full primal step, including
/// both signs of a hard-case minimum-eigenvector fill. An infeasible sign adds
/// its first exact chord blocker and recomputes the affine reduction and
/// spectrum from the original `H`; no shifted-SPD surrogate is ever handed to
/// a different QP. At a feasible endpoint, operator NNLS supplies the complete
/// positive-multiplier support. Any entry or release triggers another physical
/// solve on that critical face.
///
/// A result is returned only after mathematical primal feasibility, trust-ball
/// complementarity, physical shifted stationarity
/// `(H delta - rhs + lambda D delta) - A' mu = 0`, and positive semidefiniteness
/// of `Z'(H + lambda D)Z` all certify. Thus the step, predicted gain, blockers,
/// and first-/second-order KKT certificate describe one subproblem (#2656).
///
/// The exchange itself is a heuristic and can cycle on a degenerate face (a row
/// at exactly zero scaled slack whose multiplier is negative on the face and
/// which is re-added as the first blocker once released). That is not a
/// contract violation, so a revisited face returns `Ok(None)` — no certified
/// reduced-face candidate — exactly like an empty warm face, and the caller's
/// general constrained QP owns the subproblem (gam#2600).
///
/// The same applies to a warm face whose equalities are mutually inconsistent:
/// the affine system has no solution, so there is nothing here to certify, and
/// the subproblem is handed over rather than ending the fit. What is NOT
/// handled that way is a face that is merely rank-deficient — redundant rows
/// are solved through, on the minimum-D-norm solution their common row space
/// determines, because a redundant equality is still an equality.
fn certified_reduced_face_candidate(
    exact_hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &ConstraintSet,
    active_rows: &[usize],
    trust_metric_diag: &Array1<f64>,
    trust_radius: f64,
) -> Result<Option<(Array1<f64>, Vec<usize>, ReducedFaceCandidateKind)>, CustomFamilyError> {
    let p = beta.len();
    if active_rows.is_empty() {
        return Ok(None);
    }
    if rhs.len() != p
        || exact_hessian.nrows() != p
        || exact_hessian.ncols() != p
        || constraints.ncols() != p
        || trust_metric_diag.len() != p
        || !(trust_radius.is_finite() && trust_radius > 0.0)
        || rhs.iter().any(|value| !value.is_finite())
        || exact_hessian.iter().any(|value| !value.is_finite())
        || trust_metric_diag
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(CustomFamilyError::trial_point(format!(
            "reduced-face candidate dimension/metric contract failed \
             (p={p}, rhs={}, hessian={}x{}, constraints={}x{}, trust_metric={}, \
             trust_radius={trust_radius:.6e})",
            rhs.len(),
            exact_hessian.nrows(),
            exact_hessian.ncols(),
            constraints.nrows(),
            constraints.ncols(),
            trust_metric_diag.len(),
        )));
    }
    let mut trust_metric = Array2::<f64>::zeros((p, p));
    for index in 0..p {
        trust_metric[[index, index]] = trust_metric_diag[index];
    }
    let (original_base_violation, original_base_worst_row) = constraints
        .max_scaled_violation(beta.view())
        .map_err(|error| {
            CustomFamilyError::trial_point(format!(
                "reduced-face base feasibility classification failed: {error}"
            ))
        })?;
    if !original_base_violation.is_finite() {
        return Err(CustomFamilyError::trial_point(format!(
            "reduced-face base feasibility classification is non-finite \
             (base_scaled_violation={original_base_violation:.6e}@{original_base_worst_row:?})"
        )));
    }
    // Mathematical feasibility is the theorem premise, so any positive
    // violation triggers repair. This decision deliberately has no tolerance.
    let feasible_base = if original_base_violation > 0.0 {
        gam_solve::active_set::project_point_strictly_into_feasible_constraint_set(
            beta,
            constraints,
        )
        .map_err(|error| {
            format!(
                "reduced-face infeasible base has no certified feasible reference \
                 (base_scaled_violation={original_base_violation:.6e}@{original_base_worst_row:?}): \
                 {error}"
            )
        })?
    } else {
        beta.clone()
    };
    let (reference_violation, reference_worst_row) = constraints
        .max_scaled_violation(feasible_base.view())
        .map_err(|error| {
            format!("reduced-face feasible-reference certification failed: {error}")
        })?;
    if !reference_violation.is_finite() || reference_violation > 0.0 {
        return Err(CustomFamilyError::trial_point(format!(
            "reduced-face solver reference is not mathematically feasible \
             (base_scaled_violation={original_base_violation:.6e}@{original_base_worst_row:?}, \
             reference_scaled_violation={reference_violation:.6e}@{reference_worst_row:?})"
        )));
    }
    let reduce_face =
        |rows: &[usize]| -> Result<(Array2<f64>, Array1<f64>, Vec<usize>), CustomFamilyError> {
            let mut unique = rows.to_vec();
            unique.sort_unstable();
            unique.dedup();
            if unique.is_empty() {
                return Ok((
                    Array2::<f64>::zeros((0, p)),
                    Array1::<f64>::zeros(0),
                    unique,
                ));
            }
            let gathered = constraints.gather_rows(&unique).map_err(|error| {
                CustomFamilyError::trial_point(format!(
                    "physical reduced-face row gather failed: {error}"
                ))
            })?;
            let mut normalized_a = gathered.a;
            let mut normalized_b = gathered.b;
            for row in 0..normalized_a.nrows() {
                let norm = normalized_a.row(row).dot(&normalized_a.row(row)).sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(CustomFamilyError::trial_point(format!(
                        "physical reduced face contains a zero/non-finite row \
                     (constraint_row={})",
                        unique[row],
                    )));
                }
                normalized_a.row_mut(row).mapv_inplace(|value| value / norm);
                normalized_b[row] /= norm;
            }
            let groups = unique.iter().copied().map(|row| vec![row]).collect();
            let (a, b, groups, _dependence) =
                gam_solve::active_set::rank_reduce_rows_pivoted_qr_with_dependence(
                    normalized_a,
                    normalized_b,
                    groups,
                );
            let representatives = groups
                .into_iter()
                .filter_map(|group| group.into_iter().min())
                .collect();
            Ok((a, b, representatives))
        };
    let model_gain = |delta: &Array1<f64>| {
        let h_delta = exact_hessian.dot(delta);
        rhs.dot(delta) - 0.5 * delta.dot(&h_delta)
    };

    struct PhysicalFaceStep {
        deltas: Vec<Array1<f64>>,
        trust_shift: f64,
        hard_case: bool,
        exact_positive_curvature: bool,
        minimum_shifted_curvature: f64,
        numerical_floor: f64,
    }

    let mut working_active = active_rows.to_vec();
    let mut seen_faces = std::collections::HashSet::<Vec<usize>>::new();
    // COST BOUND FOR THE EXCHANGE (gam#2600).
    //
    // Detecting a repeated FACE is the loop's only termination argument, and
    // it is a pigeonhole one: the bound it gives is the number of subsets of
    // the constraint rows. That is not a cost bound, and the difference is
    // measured rather than theoretical. On this issue's pit arm the exchange
    // declined on 273 of 284 cycles, visiting a MEDIAN of 87 faces and a
    // maximum of 270 before the repeat surfaced — 27,731 face solves, each an
    // SVD of the face plus a Moré--Sorensen solve plus a feasibility scan, all
    // of them discarded, with the general constrained QP doing every solve
    // (`path=linear` on all 284 cycles).
    //
    // The bound is `p + |warm face|`, and it is what the exchange's own moves
    // are worth: its only legitimate corrections are adding one blocking row
    // and adopting the operator NNLS support, so REBUILDING the warm face
    // completely — releasing every row it was handed and pinning every ambient
    // direction — takes `|warm face| + p` single-row corrections. Past that it
    // is no longer correcting the warm face; it is searching the face lattice,
    // where branch (2) has no monotone measure and therefore no termination
    // argument at all.
    //
    // The measurement says the same thing with a wide margin. Over 100
    // exchanges that DID certify (wine arm, `p = 6`), the visited-face count
    // was min 1, median 1, max 4 — against a budget of 7 there. An exchange
    // that has not certified within a complete rebuild of its warm face does
    // not certify at all.
    //
    // Exhausting the budget is the same verdict a detected cycle already
    // produces: no certified reduced-face candidate, `Ok(None)`, and the
    // caller's general constrained QP owns the subproblem. Nothing is
    // certified on a budget and no tolerance moves.
    let exchange_budget = p.saturating_add(active_rows.len());
    loop {
        let (face_a, face_b, canonical_active) = reduce_face(&working_active)?;
        working_active = canonical_active;
        if !seen_faces.insert(working_active.clone()) {
            // A revisited face is a proven cycle of the exchange, and it is a
            // failure of this heuristic rather than a violated contract. At a
            // constraint sitting at exactly zero scaled slack the support map
            // is not idempotent: the face solve returns a negative multiplier
            // for that row, releasing it makes the free step re-add the very
            // same row as its first blocker, and the two faces alternate
            // forever. Nothing about the model, the metric, or the geometry is
            // wrong -- there is simply no certified reduced-face candidate
            // here, which is exactly the state this routine already reports by
            // declining an empty warm face. Decline, and let the caller's
            // general constrained QP own the same subproblem; every guard,
            // tolerance, and certificate below is untouched, and no
            // uncertified point is ever returned. Erroring instead ends the
            // entire fit on one trial point (gam#2600).
            log::warn!(
                "[gam#2600 reduced-face] declining a cycled active-set exchange \
                 (face_rows={}, visited_faces={}); the general constrained QP \
                 owns this subproblem",
                working_active.len(),
                seen_faces.len(),
            );
            return Ok(None);
        }
        if seen_faces.len() > exchange_budget {
            log::warn!(
                "[gam#2600 reduced-face] declining an exchange that outran its own \
                 rebuild budget (face_rows={}, visited_faces={}, budget={exchange_budget} \
                 = ambient_dim {p} + warm_rows {}); the general constrained QP owns \
                 this subproblem",
                working_active.len(),
                seen_faces.len(),
                active_rows.len(),
            );
            return Ok(None);
        }

        // The equality face is affine in the step:
        //
        //     A delta = b - A beta.
        //
        // Use its minimum-D-norm particular solution. The tangent basis Z then
        // satisfies Z'D delta_p = 0, so the physical trust ball decomposes
        // exactly into ||delta_p||_D² + ||z||_(Z'DZ)² <= r².
        let (delta_particular, tangent) = if working_active.is_empty() {
            (Array1::<f64>::zeros(p), Some(Array2::<f64>::eye(p)))
        } else {
            // ONE factorization decides the whole face. The rank, the tangent
            // and the affine particular solution all come off the SAME
            // row-normalized block, in the coefficient metric the constraints
            // are themselves stated in.
            //
            // This used to factor the trust-whitened face `A D^{-1/2}` and
            // return `Err` whenever any of its singular values fell under a
            // rank floor — which ends the entire fit at the trial point that
            // produced it. Measured on #2600's pit arm, that refusal fired 90
            // times, and in EVERY one of them the block `reduce_face` had just
            // returned carried exactly one more row than its numerical rank
            // (`face_rows = rank + 1`, e.g. 39 rows at rank 38 with the
            // retained singular values down to `3.86e-5` and the redundant one
            // at `~1e-15`).
            //
            // So the face really was rank-deficient, and the defect is what was
            // done about it: a REDUNDANT EQUALITY IS STILL AN EQUALITY. Its row
            // space determines `delta` exactly as well as a full-rank block
            // would; there is nothing here that cannot be solved. Refusing was
            // the mistake, and refusing FATALLY — from a heuristic accelerator
            // the caller has a general constrained QP to fall back on — was the
            // expensive part.
            //
            // (The redundancy itself is now also gone at its source: the scan
            // in `rank_reduce_rows_pivoted_qr_with_dependence` reorthogonalizes,
            // so it no longer keeps a row whose independence is an artifact of
            // its own accumulated loss of orthogonality. This path does not
            // depend on that — it is total over rank-deficient faces either
            // way, which is what makes it the right place for the guarantee.)
            //
            // Whitening is not the mechanism and should not be read as one: on
            // that fixture `kappa(D)` stayed between 3.1 and 27.2, so `A
            // D^{-1/2}` was at most 5x worse conditioned than `A`. It is still
            // the wrong matrix to ask, because the trust metric has exactly one
            // job on this face and it is not deciding solvability: it picks
            // WHICH solution of an underdetermined face is smallest. That is
            // the D-orthogonal re-anchoring below, and it is the only place `D`
            // now appears.
            let geometry = active_constraint_face_geometry(&face_a).map_err(|error| {
                format!(
                    "physical reduced-face affine geometry failed \
                     (active_rows={}, ambient_dim={p}): {error}",
                    working_active.len(),
                )
            })?;
            if geometry.rank() < face_a.nrows() {
                // `reduce_face` is supposed to hand this solve one
                // representative per independent direction, so a block that
                // still factors rank-deficient means the two disagree about
                // the same face. That is survivable here — the row space is
                // what the affine solve uses — but it is the signature that
                // located gam#2600, so say it rather than absorb it.
                log::debug!(
                    "[gam#2600 reduced-face] the reduced face carries {} redundant row(s) \
                     (face_rows={}, rank={}, sigma_max={:.6e}, sigma_min_retained={:.6e}); \
                     solving on its row space",
                    face_a.nrows() - geometry.rank(),
                    face_a.nrows(),
                    geometry.rank(),
                    geometry.largest_singular_value(),
                    geometry.smallest_retained_singular_value(),
                );
            }
            let affine_rhs = &face_b - &face_a.dot(beta);
            let particular = geometry
                .minimum_norm_particular(&affine_rhs)
                .map_err(|error| {
                    format!(
                        "physical reduced-face affine particular failed \
                     (active_rows={}, ambient_dim={p}): {error}",
                        working_active.len(),
                    )
                })?;
            if particular.residual_inf > particular.residual_tolerance {
                // The working face asks for equalities that no step satisfies
                // simultaneously. That is a failure of the warm face, not a
                // violated contract, so decline exactly as a cycled exchange
                // does and let the caller's general constrained QP own the
                // subproblem instead of ending the fit on this trial point.
                log::warn!(
                    "[gam#2600 reduced-face] declining an inconsistent equality face \
                     (face_rows={}, rank={}, residual_inf={:.6e}, tolerance={:.6e}); \
                     the general constrained QP owns this subproblem",
                    working_active.len(),
                    geometry.rank(),
                    particular.residual_inf,
                    particular.residual_tolerance,
                );
                return Ok(None);
            }
            let tangent = match geometry.into_tangent() {
                ActiveConstraintTangentGeometry::FullyPinned => None,
                ActiveConstraintTangentGeometry::Tangent(basis) => Some(basis),
            };
            // Re-anchor the particular solution D-orthogonally to the tangent.
            // `delta_p = delta_0 - Z (Z'DZ)^{-1} Z'D delta_0` leaves `A
            // delta_p = A delta_0` (because `A Z = 0`) while forcing `Z'D
            // delta_p = 0`, which is exactly the minimum-D-norm point of the
            // face and exactly what makes the physical trust ball split as
            // `||delta_p||_D^2 + ||z||_(Z'DZ)^2 <= r^2` below. A fully pinned
            // face has no freedom left to spend, so the metric never enters.
            let delta_particular = match tangent.as_ref() {
                None => particular.delta,
                Some(basis) => {
                    let mut weighted = basis.clone();
                    for (mut row, weight) in weighted
                        .rows_mut()
                        .into_iter()
                        .zip(trust_metric_diag.iter())
                    {
                        row *= *weight;
                    }
                    let mut reduced_metric = basis.t().dot(&weighted);
                    symmetrize_dense_in_place(&mut reduced_metric);
                    let projection = weighted.t().dot(&particular.delta);
                    let factor = reduced_metric
                        .cholesky(faer::Side::Lower)
                        .map_err(|error| {
                            format!(
                                "physical reduced-face tangent metric is not positive definite \
                                 (active_rows={}, tangent_dim={}): {error:?}",
                                working_active.len(),
                                basis.ncols(),
                            )
                        })?;
                    let shift = factor.solvevec(&projection);
                    &particular.delta - &basis.dot(&shift)
                }
            };
            (delta_particular, tangent)
        };
        if delta_particular.iter().any(|value| !value.is_finite()) {
            return Err(CustomFamilyError::trial_point(
                "physical reduced-face affine particular is non-finite".to_string(),
            ));
        }
        let particular_norm_sq = delta_particular
            .iter()
            .zip(trust_metric_diag.iter())
            .map(|(delta, weight)| weight * delta * delta)
            .sum::<f64>();
        let radius_sq = trust_radius * trust_radius;
        let ball_roundoff = f64::EPSILON.sqrt()
            * (p.max(1) as f64)
            * radius_sq.abs().max(particular_norm_sq.abs()).max(1.0);
        if !particular_norm_sq.is_finite() {
            return Err(CustomFamilyError::trial_point(format!(
                "physical reduced-face minimum-norm point is not finite \
                 (minimum_face_norm_sq={particular_norm_sq:.6e}, active_rows={})",
                working_active.len(),
            )));
        }
        if particular_norm_sq > radius_sq + ball_roundoff {
            // The closest point of this face is outside the current trust
            // region, so no step on it is admissible AT THIS RADIUS. That is a
            // statement about the warm face and the radius, not about the
            // problem: the face came from a previous iterate's active set and
            // the radius has since shrunk. Declining is what the caller's
            // general constrained QP is for; ending the fit on it makes an
            // ordinary conditional-transformation-normal fit with one smooth
            // covariate unfittable (gam#2600, measured
            // `minimum_face_norm_sq = 3.745717e1` against `radius_sq = 1`).
            log::warn!(
                "[gam#2600 reduced-face] declining a face that does not intersect the trust \
                 ball (face_rows={}, minimum_face_norm_sq={:.6e}, radius_sq={:.6e}); \
                 the general constrained QP owns this subproblem",
                working_active.len(),
                particular_norm_sq,
                radius_sq,
            );
            return Ok(None);
        }

        let face_step = if let Some(tangent) = tangent.as_ref() {
            let remaining_radius_sq = (radius_sq - particular_norm_sq).max(0.0);
            if remaining_radius_sq == 0.0 {
                // The face's minimum-norm point sits exactly on the trust
                // sphere, so the reduced ball this tangent would search is
                // empty and no trust multiplier can be certified from it.
                // Again a statement about the face and the radius rather than
                // about the problem, and again the general constrained QP is
                // the designed owner (gam#2600).
                log::warn!(
                    "[gam#2600 reduced-face] declining a face that touches the trust ball with \
                     a nonzero tangent (face_rows={}, tangent_dim={}, \
                     minimum_face_norm_sq={:.6e}, radius_sq={:.6e}); the general constrained QP \
                     owns this subproblem",
                    working_active.len(),
                    tangent.ncols(),
                    particular_norm_sq,
                    radius_sq,
                );
                return Ok(None);
            }
            let mut reduced_hessian = tangent.t().dot(exact_hessian).dot(tangent);
            symmetrize_dense_in_place(&mut reduced_hessian);
            let mut reduced_trust_metric = tangent.t().dot(&trust_metric).dot(tangent);
            symmetrize_dense_in_place(&mut reduced_trust_metric);
            let reduced_rhs = tangent
                .t()
                .dot(&(rhs - &exact_hessian.dot(&delta_particular)));
            let reduced_step = generalized_trust_region_reduced_step(
                &reduced_hessian,
                &reduced_trust_metric,
                &reduced_rhs,
                remaining_radius_sq.sqrt(),
            )
            .map_err(|error| {
                format!(
                    "physical reduced-face Moré--Sorensen solve failed \
                     (ambient_dim={p}, tangent_dim={}, active_rows={}): {error}",
                    tangent.ncols(),
                    working_active.len(),
                )
            })?;
            let mut deltas = vec![&delta_particular + &tangent.dot(&reduced_step.delta)];
            if let Some(alternate) = reduced_step.alternate_hard_case_delta.as_ref() {
                deltas.push(&delta_particular + &tangent.dot(alternate));
            }
            PhysicalFaceStep {
                deltas,
                trust_shift: reduced_step.trust_shift,
                hard_case: reduced_step.hard_case,
                exact_positive_curvature: reduced_step.exact_positive_curvature,
                minimum_shifted_curvature: reduced_step.minimum_shifted_curvature,
                numerical_floor: reduced_step.numerical_floor,
            }
        } else {
            // No nonzero direction survives all equality rows, so second-order
            // KKT on that face is vacuous. Invalid multipliers are still
            // released below by the complete operator normal-cone projection.
            PhysicalFaceStep {
                deltas: vec![delta_particular],
                trust_shift: 0.0,
                hard_case: false,
                exact_positive_curvature: true,
                minimum_shifted_curvature: f64::INFINITY,
                numerical_floor: 0.0,
            }
        };
        if face_step.minimum_shifted_curvature < -face_step.numerical_floor {
            return Err(CustomFamilyError::trial_point(format!(
                "physical reduced-face second-order KKT failed \
                 (lambda_min_shifted={:.6e}, numerical_floor={:.6e}, \
                 trust_shift={:.6e}, active_rows={})",
                face_step.minimum_shifted_curvature,
                face_step.numerical_floor,
                face_step.trust_shift,
                working_active.len(),
            )));
        }

        // Preserve both signs of a hard-case fill until feasibility is known.
        // If neither sign is feasible, the first exact chord blocker becomes a
        // new equality and the reduced spectrum is recomputed from physical H.
        // `chord_repair` records how far the certified-feasible chord had to pull
        // a candidate back, as `1 − t` on the chord from the feasible base. It is
        // `0.0` for a candidate the face solve produced feasible outright, and it
        // is what tells the first-order KKT check below whether it is grading the
        // Moré--Sorensen solve or grading a repair.
        let mut feasible_candidates: Vec<(Array1<f64>, Array1<f64>, f64, f64)> = Vec::new();
        let mut blocker_candidates: Vec<(usize, f64)> = Vec::new();
        for raw_delta in &face_step.deltas {
            let raw_candidate = beta + raw_delta;
            let (violation, worst_row) = constraints
                .max_scaled_violation(raw_candidate.view())
                .map_err(|error| {
                    format!("physical reduced-face feasibility classification failed: {error}")
                })?;
            if !violation.is_finite() {
                return Err(CustomFamilyError::trial_point(format!(
                    "physical reduced-face feasibility classification is non-finite \
                     (scaled_violation={violation:.6e}@{worst_row:?})"
                )));
            }
            if violation <= 0.0 {
                feasible_candidates.push((
                    raw_candidate,
                    raw_delta.clone(),
                    model_gain(raw_delta),
                    0.0,
                ));
                continue;
            }
            let (clipped, blocker, step) = clip_infeasible_candidate_to_certified_feasible_chord(
                constraints,
                &feasible_base,
                &raw_candidate,
            )
            .map_err(|error| {
                format!(
                    "physical reduced-face blocker classification failed \
                         (scaled_violation={violation:.6e}@{worst_row:?}): {error}"
                )
            })?;
            let clipped_delta = &clipped - beta;
            let clipped_gain = model_gain(&clipped_delta);
            if working_active.contains(&blocker) {
                // The equality solve landed outside a wall it is HOLDING AS AN
                // EQUALITY. Keep the exactly feasible adjacent value and let the
                // physical KKT residual decide whether that repair is admissible
                // — but carry how big the repair was, because "one representable
                // value outside its own wall" and "a chord clipped back by a
                // finite fraction" are different events with the same shape, and
                // only the first is a rounding repair (gam#2600).
                feasible_candidates.push((clipped, clipped_delta, clipped_gain, 1.0 - step));
            } else {
                blocker_candidates.push((blocker, clipped_gain));
            }
        }
        if feasible_candidates.is_empty() {
            let Some((blocker, _)) = blocker_candidates.into_iter().max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                return Err(CustomFamilyError::trial_point(
                    "physical reduced-face step produced no candidate or blocker".to_string(),
                ));
            };
            working_active.push(blocker);
            continue;
        }
        let (candidate, delta, predicted_gain, chord_repair) = feasible_candidates
            .into_iter()
            .max_by(|left, right| {
                left.2
                    .partial_cmp(&right.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("nonempty feasible reduced-face candidates");
        if !predicted_gain.is_finite() {
            return Err(CustomFamilyError::trial_point(
                "physical reduced-face predicted gain is non-finite".to_string(),
            ));
        }
        let (candidate_violation, candidate_worst_row) = constraints
            .max_scaled_violation(candidate.view())
            .map_err(|error| {
                format!("physical reduced-face final feasibility check failed: {error}")
            })?;
        if !candidate_violation.is_finite() || candidate_violation > 0.0 {
            return Err(CustomFamilyError::trial_point(format!(
                "physical reduced-face candidate is not mathematically feasible \
                 (scaled_violation={candidate_violation:.6e}@{candidate_worst_row:?})"
            )));
        }

        let h_delta = exact_hessian.dot(&delta);
        let trust_normal = (&delta * trust_metric_diag) * face_step.trust_shift;
        let shifted_gradient = &h_delta - rhs + &trust_normal;
        let Some((projected, support)) =
            gam_solve::active_set::project_stationarity_residual_on_constraint_set(
                &shifted_gradient,
                &candidate,
                constraints,
                &[],
            )
        else {
            return Err(CustomFamilyError::trial_point(
                "physical reduced-face operator KKT projection failed".to_string(),
            ));
        };
        let (_support_a, _support_b, support) = reduce_face(&support)?;
        if support != working_active {
            // This is both release (negative/zero multiplier rows disappear)
            // and entry (other tight positive-multiplier rows appear). Re-solve
            // the Moré--Sorensen spectrum on the resulting critical face.
            working_active = support;
            continue;
        }
        let closure_inf = projected
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let stationarity_scale = shifted_gradient
            .iter()
            .chain(rhs.iter())
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        let kkt_tolerance = f64::EPSILON.sqrt() * (p.max(1) as f64) * stationarity_scale;
        if !closure_inf.is_finite() || closure_inf > kkt_tolerance {
            if chord_repair > 0.0 {
                // The candidate that failed here is not the Moré--Sorensen
                // solution: it is that solution pulled back along the certified
                // feasible chord because the equality solve landed outside a row
                // it was holding. A repair that large is not a rounding repair,
                // and a heuristic repair failing its own certificate is not a
                // violated contract — it is the same situation a cycled
                // active-set exchange is in, and it gets the same answer:
                // decline, and let the caller's general constrained QP own the
                // subproblem, instead of ending the fit on this trial point.
                //
                // Grading the repair by the KKT residual and then treating the
                // verdict as a contract violation is what made an ordinary
                // conditional-transformation-normal fit with one smooth covariate
                // unfittable: `projected_residual_inf = 9.736333e-1` against
                // `tolerance = 1.162291e-6` at `active_rows = 1` (gam#2600).
                log::warn!(
                    "[gam#2600 reduced-face] declining a chord-repaired candidate that fails \
                     its own first-order KKT (face_rows={}, chord_repair={:.6e}, \
                     projected_residual_inf={:.6e}, tolerance={:.6e}, trust_shift={:.6e}); \
                     the general constrained QP owns this subproblem",
                    working_active.len(),
                    chord_repair,
                    closure_inf,
                    kkt_tolerance,
                    face_step.trust_shift,
                );
                return Ok(None);
            }
            return Err(CustomFamilyError::trial_point(format!(
                "physical reduced-face first-order KKT failed \
                 (projected_residual_inf={closure_inf:.6e}, \
                 tolerance={kkt_tolerance:.6e}, trust_shift={:.6e}, \
                 active_rows={}, chord_repair=0)",
                face_step.trust_shift,
                working_active.len(),
            )));
        }

        let trust_norm_sq = delta
            .iter()
            .zip(trust_metric_diag.iter())
            .map(|(value, weight)| weight * value * value)
            .sum::<f64>();
        let trust_norm = trust_norm_sq.sqrt();
        let trust_tolerance = f64::EPSILON.sqrt()
            * (p.max(1) as f64)
            * trust_radius.abs().max(trust_norm.abs()).max(1.0);
        let trust_feasible = trust_norm <= trust_radius + trust_tolerance;
        let trust_complementary =
            face_step.trust_shift == 0.0 || (trust_norm - trust_radius).abs() <= trust_tolerance;
        if !trust_norm.is_finite() || !trust_feasible || !trust_complementary {
            return Err(CustomFamilyError::trial_point(format!(
                "physical reduced-face trust-ball KKT failed \
                 (metric_norm={trust_norm:.6e}, radius={trust_radius:.6e}, \
                 trust_shift={:.6e}, tolerance={trust_tolerance:.6e})",
                face_step.trust_shift,
            )));
        }

        if original_base_violation <= 0.0 {
            let linear = rhs.dot(&delta);
            let quadratic = 0.5 * delta.dot(&h_delta);
            let gain_tolerance = f64::EPSILON.sqrt()
                * (p.max(1) as f64)
                * linear.abs().max(quadratic.abs()).max(1.0);
            if predicted_gain < -gain_tolerance {
                return Err(CustomFamilyError::trial_point(format!(
                    "physical reduced-face candidate is inferior to the feasible \
                     zero step (predicted_gain={predicted_gain:.6e}, \
                     tolerance={gain_tolerance:.6e}, active_rows={})",
                    working_active.len(),
                )));
            }
        }

        let kind = if face_step.exact_positive_curvature
            && face_step.trust_shift == 0.0
            && !face_step.hard_case
        {
            ReducedFaceCandidateKind::ExactNewton
        } else if face_step.trust_shift == 0.0 && !face_step.hard_case {
            ReducedFaceCandidateKind::ReducedNewton
        } else {
            ReducedFaceCandidateKind::RegularizedNewton
        };
        // How many faces this exchange had to visit before it certified. The
        // decline path already reports its own count; without the same number
        // on the SUCCESS path the two cannot be compared, and a cost bound for
        // the exchange cannot be sized from anything but a guess (gam#2600).
        log::debug!(
            "[gam#2600 reduced-face] certified after {} visited face(s) \
             (face_rows={}, ambient_dim={p}, kind={kind:?})",
            seen_faces.len(),
            working_active.len(),
        );
        return Ok(Some((candidate, working_active, kind)));
    }
}

/// Canonical constraint face at an accepted nonlinear iterate.
///
/// `endpoint_active` is the sparse working-set provenance returned by the local
/// QP. It is authoritative only when there is no constraint carrier. With a
/// carrier, the accepted point itself determines the face: reduce its complete
/// tight set to deterministic independent representatives so equivalent QP
/// paths cannot seed different warm faces.
fn canonical_accepted_active_rows(
    constraints: Option<&ConstraintSet>,
    accepted_beta: &Array1<f64>,
    endpoint_active: &[usize],
) -> Result<Vec<usize>, CustomFamilyError> {
    let Some(constraints) = constraints else {
        return Ok(endpoint_active.to_vec());
    };
    gam_solve::active_set::ConstraintSetReducedFace::reduced_face(
        constraints,
        accepted_beta.view(),
        gam_solve::active_set::ACTIVE_SET_WORKING_FACE_TOL,
    )
    .map_err(|error| CustomFamilyError::trial_point(error.to_string()))
    .map(|face| {
        face.representatives
            .into_iter()
            .map(|row| row.index())
            .collect()
    })
    .map_err(|error| {
        CustomFamilyError::trial_point(format!(
            "accepted constrained-Newton canonical face reduction failed: {error}"
        ))
    })
}

#[cfg(test)]
mod exact_face_newton_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_face_the_trust_metric_makes_singular_is_still_solved_2600() {
        // `x >= 0` and `z >= 0`: two active rows that are unit-norm and
        // EXACTLY ORTHOGONAL. Their face has singular values `{1, 1}` — there
        // is no degeneracy here in any sense, and no rank decision taken on
        // the constraints themselves can call it singular.
        //
        // The removed gate did not take its rank decision on the constraints.
        // It factored `A D^{-1/2}`, and with `D = diag(1e-18, 1, 1e18)` those
        // same two rows become `(1e9, 0, 0)` and `(0, 0, 1e-9)`: singular
        // values `{1e9, 1e-9}` against a floor of `100·eps·3·1e9 = 6.66e-5`,
        // so the smaller one is under the floor by a factor of 6.7e4 and the
        // whole fit ends on this trial point with
        // `physical reduced-face rank reduction left a singular affine system`.
        // That is #2600's pit-arm refusal (`active_rows=39,
        // singular_min=4.838920e-18, rank_floor=1.067916e-13`) in three
        // dimensions: the trust metric's dynamic range, not the geometry.
        //
        // Both rows are tight at beta, the face pins the step to zero, and the
        // shifted gradient sits exactly in the normal cone with unit
        // multipliers on both rows — so there IS a certified answer here.
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0, 0.0], [0.0_f64, 0.0, 1.0]],
                array![0.0_f64, 0.0],
            )
            .expect("two orthogonal half-spaces"),
        );
        let hessian = Array2::<f64>::eye(3);
        // -rhs = 1*(1,0,0) + 1*(0,0,1): both multipliers strictly positive, so
        // the operator KKT projection reproduces exactly this working face.
        let rhs = array![-1.0_f64, 0.0, -1.0];
        let beta = array![0.0_f64, 5.0, 0.0];
        let metric = array![1.0e-18_f64, 1.0, 1.0e18];

        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0, 1],
            &metric,
            1.0,
        )
        .expect("orthogonal unit rows are not a singular affine system")
        .expect("the pinned face certifies its own zero step");
        assert_eq!(active, vec![0, 1]);
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
        for (value, expected) in candidate.iter().zip(beta.iter()) {
            assert!(
                (value - expected).abs() < 1e-12,
                "the certified step on this face is zero, got {candidate:?}"
            );
        }
    }

    #[test]
    fn the_affine_particular_is_the_minimum_trust_metric_norm_point_2600() {
        // One warm-active row `x + y >= 0` at a base that is FEASIBLE with
        // slack 1, so the face's affine system is `x + y = -1` and has a whole
        // line of solutions. The trust metric decides which one: the
        // minimum-D-norm point of `x + y = a` is `(a*d2, a*d1)/(d1 + d2)`, NOT
        // the minimum-Euclidean-norm point `(a/2, a/2)`.
        //
        // This is the property the removed whitened factorization DID deliver,
        // and the replacement has to keep it — the metric selects among
        // solutions, which is the only job it has left on this face. `d1 = 3`
        // and `d2 = 1` make the answer `(-1/4, -3/4)` exactly representable,
        // so the candidate lands on `x + y = 0` to the last bit and the
        // certificate is not reading a rounding repair. `H = D` with a
        // right-hand side offset along the constraint normal leaves the
        // tangent coordinate at zero, so the returned step IS the particular
        // solution.
        let (d1, d2) = (3.0_f64, 1.0_f64);
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 1.0]], array![0.0])
                .expect("x+y>=0 half-space"),
        );
        let hessian = array![[d1, 0.0], [0.0, d2]];
        let beta = array![1.0_f64, 0.0];
        let metric = array![d1, d2];
        // `rhs = H*delta_p - 1*(1,1)`: the tangent gradient vanishes and row 0
        // keeps a strictly positive multiplier, so the operator KKT projection
        // reproduces exactly this working face.
        let rhs = array![-1.75_f64, -1.75];

        let (candidate, active, _kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            10.0,
        )
        .expect("an underdetermined face has a minimum-D-norm solution")
        .expect("the face certifies a candidate");
        assert_eq!(active, vec![0]);
        let expected_y = -d1 / (d1 + d2);
        assert!(
            (candidate[1] - expected_y).abs() < 1e-12,
            "the affine step must be the minimum-D-norm point of x+y=-1 \
             (expected y={expected_y:.12e}, got {:.12e}); the minimum-EUCLIDEAN-norm \
             point would put y at -0.5",
            candidate[1]
        );
        let closure = candidate[0] + candidate[1];
        assert!(
            closure.abs() < 1e-12,
            "the certified candidate must sit on its own equality face, got x+y={closure:.6e}"
        );
    }

    #[test]
    fn reduced_face_contract_violation_is_an_error_not_a_fallback() {
        let hessian = Array2::<f64>::eye(2);
        let rhs = array![1.0_f64];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![0.0])
                .expect("x>=0 half-space"),
        );
        let metric = Array1::<f64>::ones(2);

        let error = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            1.0,
        )
        .expect_err("an attempted reduced-face solve must never silently change models");

        assert!(
            error
                .to_string()
                .contains("dimension/metric contract failed"),
            "unexpected reduced-face diagnostic: {error}"
        );
    }

    #[test]
    fn physical_face_repairs_an_infeasible_base_without_reanchoring_2525() {
        // The physical subproblem stays anchored at beta=-9e-9. Its active
        // affine equation is delta=9e-9, which reaches x=0 exactly; the trust
        // metric affects the ball, never a surrogate objective.
        let exact_hessian = array![[-1.0_f64]];
        let rhs = array![-1.0_f64];
        let beta = array![-9.0e-9_f64];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64]], array![0.0])
                .expect("x>=0 half-space"),
        );
        let (base_violation, base_worst_row) = constraints
            .max_scaled_violation(beta.view())
            .expect("base violation");
        assert_eq!(base_violation, 9.0e-9);
        assert_eq!(base_worst_row, Some(0));

        let (candidate, active, kind) = certified_reduced_face_candidate(
            &exact_hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &array![2.0e11_f64],
            1.0,
        )
        .expect("physical affine-face certificate")
        .expect("fully pinned physical candidate");
        let (candidate_violation, candidate_worst_row) = constraints
            .max_scaled_violation(candidate.view())
            .expect("candidate violation");
        assert_eq!(candidate_violation, 0.0);
        assert_eq!(candidate_worst_row, None);
        assert_eq!(candidate, array![0.0_f64]);
        assert_eq!(active, vec![0]);
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
    }

    #[test]
    fn accepted_face_is_canonical_across_degenerate_qp_row_bases() {
        // Every row is tight at the vertex. Rows 0/1 are parallel, while row 3
        // is a general-position dependent of independent rows 0/2. Different
        // QP paths can therefore report different sparse bases for the same
        // geometric face; the accepted handoff must not preserve that history.
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![
                    [1.0_f64, 0.0],
                    [2.0_f64, 0.0],
                    [0.0_f64, 1.0],
                    [1.0_f64, 1.0]
                ],
                array![0.0, 0.0, 0.0, 0.0],
            )
            .expect("degenerate vertex"),
        );
        let beta = array![0.0_f64, 0.0];

        let from_parallel_and_oblique =
            canonical_accepted_active_rows(Some(&constraints), &beta, &[1, 3])
                .expect("canonical face");
        let from_coordinate_rows =
            canonical_accepted_active_rows(Some(&constraints), &beta, &[0, 2])
                .expect("canonical face");

        assert_eq!(from_parallel_and_oblique, vec![0, 2]);
        assert_eq!(from_coordinate_rows, vec![0, 2]);
    }

    #[test]
    fn exact_face_newton_uses_tangent_curvature_not_ambient_reflection() {
        // H is indefinite in ambient space (det(H) = -3), but the active
        // half-space x>=0 pins its inaccessible negative-curvature direction.
        // On the tangent x=0 the exact curvature is +2, so the certified Newton
        // solution is y=1. Ambient eigenvalue reflection changes that tangent
        // equation and therefore cannot own the fixed-face endgame.
        let hessian = array![[-1.0_f64, 1.0], [1.0, 2.0]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![0.0])
                .expect("x>=0 half-space"),
        );
        let metric = array![1.0_f64, 1.0];
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            1.0e6,
        )
        .expect("exact face classification")
        .expect("positive reduced curvature and nonnegative multiplier");
        assert_eq!(active, vec![0]);
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
        assert!(candidate[0].abs() <= 1e-12);
        assert!((candidate[1] - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn physical_reduced_face_enters_the_blocker_without_crossing_it() {
        // x>=0 is the current face. The reduced tangent direction meets the
        // inactive y<=0.5 row at half its length. The face exchange must add
        // the blocker, release the zero-multiplier x face, recompute the
        // spectrum, and never cross the boundary.
        let hessian = array![[1.0_f64, 0.0], [0.0, -2.0]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, -1.0]],
                array![0.0, -0.5],
            )
            .expect("x>=0 and y<=0.5"),
        );
        let metric = array![1.0_f64, 2.0];
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            1.0,
        )
        .expect("reduced face classification")
        .expect("physical reduced solve resolves the blocker");
        assert!(candidate[0].abs() <= 1e-12);
        assert!(
            candidate[1] <= 0.5,
            "candidate must never overstep the blocker boundary (y={})",
            candidate[1]
        );
        // Lands in the working-face band (scaled |slack| ≤ WORKING_FACE_TOL),
        // not merely inside the looser feasibility band.
        assert!(
            (0.5 - candidate[1]).abs() <= gam_solve::active_set::ACTIVE_SET_WORKING_FACE_TOL,
            "candidate must land inside the working-face band (y={})",
            candidate[1]
        );
        assert_eq!(active, vec![1]);
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
    }

    #[test]
    fn reduced_face_blocker_carry_survives_the_accepted_face_filter_at_large_slack() {
        // #2298/#2301 exchange stall. The blocker is approached from a LARGE
        // scaled slack (here 1e5, the magnitude the competing-risks derivative-
        // guard cone rows reach). The physical face solve owns the boundary
        // exactly and releases the orthogonal zero-multiplier seed row, so the
        // accepted-face filter and returned active provenance must agree
        // independently of the original slack magnitude.
        let hessian = array![[1.0_f64, 0.0], [0.0, 1.0]];
        // Tangent (y-axis) Newton step is g/h = 2e5, overshooting the y<=1e5
        // blocker at half its length.
        let rhs = array![0.0_f64, 2.0e5];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, -1.0]],
                array![0.0, -1.0e5],
            )
            .expect("x>=0 and y<=1e5"),
        );
        let metric = array![1.0_f64, 1.0];
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            1.0e6,
        )
        .expect("reduced face classification")
        .expect("physical face solve resolves the far blocker");
        // The solve carries the positive-multiplier blocker (row 1) into the
        // returned face and omits the released zero-multiplier seed row.
        assert_eq!(active, vec![1]);
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
        // The accepted-face filter, applied at the candidate exactly as
        // the caller applies it at the accepted β, must AGREE — row 1 stays
        // tight.
        let tight = gam_solve::active_set::constraint_set_rows_tight_at_point(
            &constraints,
            &candidate,
            &active,
        )
        .expect("accepted-face classification");
        assert!(
            tight.contains(&1),
            "the carried blocker (row 1) must survive the accepted-face filter, \
             else the exchange never records (candidate y={}, scaled slack={:.3e})",
            candidate[1],
            1.0e5 - candidate[1],
        );
        assert!(candidate[1] <= 1.0e5 + gam_solve::active_set::ACTIVE_SET_WORKING_FACE_TOL);
    }

    #[test]
    fn physical_reduced_face_batches_independent_blockers_in_one_solve_979() {
        // The old reduced-face chord returned after the FIRST blocker (x1=.25),
        // so these three independent walls required three nonlinear cycles.
        // The physical face metric is the identity here; one solve must
        // discover the complete positive-multiplier vertex, release the
        // zero-multiplier seed row, and certify against the original Hessian.
        let hessian = Array2::<f64>::eye(4);
        let rhs = array![0.0_f64, 1.0, 1.0, 1.0];
        let beta = array![0.0_f64, 0.0, 0.0, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![
                    [1.0_f64, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0]
                ],
                array![0.0, -0.25, -0.5, -0.75],
            )
            .expect("one current face and three independent upper blockers"),
        );
        let metric = Array1::<f64>::ones(4);
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            1.0e6,
        )
        .expect("face classification")
        .expect("batched physical face solve");
        assert_eq!(
            kind,
            ReducedFaceCandidateKind::ExactNewton,
            "positive original curvature must certify"
        );
        assert_eq!(active, vec![1, 2, 3]);
        assert!(candidate[0].abs() <= 1e-12);
        assert!((candidate[1] - 0.25).abs() <= 1e-10);
        assert!((candidate[2] - 0.5).abs() <= 1e-10);
        assert!((candidate[3] - 0.75).abs() <= 1e-10);
    }

    #[test]
    fn indefinite_reduced_face_uses_trust_curvature_instead_of_reflecting() {
        // Negative accessible curvature has no Newton minimizer. Its
        // generalized mode receives trust curvature rather than |gamma| or an
        // almost-singular numerical floor. Both Hessians therefore head
        // toward the same constrained endpoint instead of turning strong
        // negative curvature into a tiny surrogate-Newton step.
        let hessian = array![[1.0_f64, 0.0], [0.0, -2.0]];
        let strongly_indefinite_hessian = array![[1.0_f64, 0.0], [0.0, -2.0e6]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let metric = array![1.0_f64, 2.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, -1.0]],
                array![0.0, -1.0],
            )
            .expect("x>=0 and y<=1"),
        );
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            2.0,
        )
        .expect("reduced face classification")
        .expect("regularized reduced step is feasible");
        let (strong_candidate, strong_active, strong_kind) = certified_reduced_face_candidate(
            &strongly_indefinite_hessian,
            &rhs,
            &beta,
            &constraints,
            &[0],
            &metric,
            2.0,
        )
        .expect("strongly indefinite face classification")
        .expect("regularized reduced step reaches the same endpoint");
        assert!(candidate[0].abs() <= 1e-12);
        assert!((candidate[1] - 1.0).abs() <= 1e-12);
        assert_eq!(active, vec![1]);
        assert_eq!(strong_active, active);
        assert!(
            (&strong_candidate - &candidate)
                .iter()
                .all(|difference| difference.abs() <= 1e-12)
        );
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
        assert_eq!(strong_kind, ReducedFaceCandidateKind::ExactNewton);
    }

    #[test]
    fn indefinite_reduced_step_uses_more_sorensen_radius_not_unit_curvature() {
        // In one dimension D=2, H=-2, rhs=2 and radius=1. The exact generalized
        // trust solution has whitened step eta=1, hence delta=1/sqrt(2).
        let reduced_hessian = array![[-2.0_f64]];
        let reduced_trust = array![[2.0_f64]];
        let reduced_rhs = array![2.0_f64];
        let step = generalized_trust_region_reduced_step(
            &reduced_hessian,
            &reduced_trust,
            &reduced_rhs,
            1.0,
        )
        .expect("generalized Moré--Sorensen step");

        assert!(!step.exact_positive_curvature);
        assert!(!step.hard_case);
        let delta = step.delta[0];
        assert!(((2.0 * delta * delta).sqrt() - 1.0).abs() <= 1e-10);
        assert!((delta - 1.0 / 2.0_f64.sqrt()).abs() <= 1e-10);
        assert!(((-2.0 + 2.0 * step.trust_shift) * delta - 2.0).abs() <= 1e-10);
    }

    #[test]
    fn mixed_hard_case_retains_both_physical_minimum_mode_signs_2656() {
        // Static witness for the #2656 root cause. For H=diag(-2,1), rhs=(0,1)
        // and r=2, lambda=2. The Moore--Penrose base is (0,1/3), and the
        // missing hard-case component has |x|=sqrt(4-1/9). A surrogate built
        // only from H+lambda I returns (0,1/3) and silently deletes almost the
        // entire step.
        let step = generalized_trust_region_reduced_step(
            &array![[-2.0_f64, 0.0], [0.0, 1.0]],
            &Array2::<f64>::eye(2),
            &array![0.0_f64, 1.0],
            2.0,
        )
        .expect("mixed hard-case physical step");
        let alternate = step
            .alternate_hard_case_delta
            .as_ref()
            .expect("opposite minimum-mode sign");
        let expected_x = (4.0_f64 - 1.0 / 9.0).sqrt();

        assert!(step.hard_case);
        assert!((step.trust_shift - 2.0).abs() <= 1e-12);
        assert!((step.delta.dot(&step.delta).sqrt() - 2.0).abs() <= 1e-12);
        assert!((alternate.dot(alternate).sqrt() - 2.0).abs() <= 1e-12);
        assert!((step.delta[0].abs() - expected_x).abs() <= 1e-12);
        assert!((alternate[0] + step.delta[0]).abs() <= 1e-12);
        assert!((step.delta[1] - 1.0 / 3.0).abs() <= 1e-12);
        assert!((alternate[1] - 1.0 / 3.0).abs() <= 1e-12);
    }

    #[test]
    fn hard_case_blocker_recomputes_physical_spectrum_and_releases_old_face_2656() {
        // The initial z=0 face leaves the mixed hard case in (x,y). Both
        // minimum-mode signs overstep one of -1<=x<=1, so a blocker must enter.
        // On the new x-bound face H is positive on the (y,z) critical tangent;
        // the obsolete zero-multiplier z wall must release.
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[0.0_f64, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],],
                array![0.0_f64, -1.0, -1.0],
            )
            .expect("z>=0 and -1<=x<=1"),
        );
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &array![[-2.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
            &array![0.0_f64, 1.0, 0.0],
            &array![0.0_f64, 0.0, 0.0],
            &constraints,
            &[0],
            &Array1::<f64>::ones(3),
            2.0,
        )
        .expect("physical hard-case blocker exchange")
        .expect("certified endpoint");

        assert!((candidate[0].abs() - 1.0).abs() <= 1e-12);
        assert!((candidate[1] - 1.0).abs() <= 1e-12);
        assert!(candidate[2].abs() <= 1e-12);
        assert!(
            active == vec![1] || active == vec![2],
            "exactly the reached x blocker must remain: {active:?}"
        );
        assert_eq!(kind, ReducedFaceCandidateKind::ExactNewton);
    }

    #[test]
    fn fully_pinned_vertex_releases_then_recomputes_physical_trust_step_979() {
        // x>=0 and y>=0 make the origin's active tangent zero-dimensional.
        // "Fully pinned" is geometry at the current point, not proof that the
        // active rows have valid KKT multipliers. A positive rhs releases both
        // walls, after which the spectrum is recomputed on physical H. For
        // H=-I and r=1 the exact boundary direction is rhs/||rhs||.
        let hessian = array![[-1.0_f64, 0.0], [0.0, -1.0]];
        let rhs = array![1.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let metric = array![1.0_f64, 1.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 0.0], [0.0, 1.0]], array![0.0, 0.0])
                .expect("nonnegative quadrant"),
        );
        let (candidate, active, kind) = certified_reduced_face_candidate(
            &hessian,
            &rhs,
            &beta,
            &constraints,
            &[0, 1],
            &metric,
            1.0,
        )
        .expect("fully pinned face classification")
        .expect("physical trust solve releases invalid active walls");
        assert!((candidate[0] - 1.0 / 5.0_f64.sqrt()).abs() <= 1e-12);
        assert!((candidate[1] - 2.0 / 5.0_f64.sqrt()).abs() <= 1e-12);
        assert!((candidate.dot(&candidate).sqrt() - 1.0).abs() <= 1e-12);
        assert!(active.is_empty());
        assert_eq!(kind, ReducedFaceCandidateKind::RegularizedNewton);
    }

    #[test]
    fn ambient_negative_curvature_cannot_replace_a_reduced_face_direction() {
        assert!(constrained_search_delta_owns_trust_step(
            Some(ReducedFaceCandidateKind::RegularizedNewton),
            true,
            Some(true),
        ));
        assert!(constrained_search_delta_owns_trust_step(
            Some(ReducedFaceCandidateKind::ExactNewton),
            true,
            Some(true),
        ));
        assert!(constrained_search_delta_owns_trust_step(
            None,
            true,
            Some(false),
        ));
        assert!(!constrained_search_delta_owns_trust_step(
            None,
            true,
            Some(true),
        ));
    }

    #[test]
    fn self_concordant_damping_is_inactive_in_the_quadratic_phase() {
        // λ_N below (3−√5)/2 — plain Newton owns the endgame, so the damped
        // trial must decline and the step policy stays byte-identical. This is
        // the STEP-POLICY-ONLY invariant: a converged/converging fit whose
        // decrement has entered the quadratic phase never sees a damped step.
        let lambda_n = 0.3_f64;
        assert!(self_concordant_damped_step_alpha(0.5 * lambda_n * lambda_n).is_none());
        // Degenerate decrements carry no self-concordance statement.
        assert!(self_concordant_damped_step_alpha(0.0).is_none());
        assert!(self_concordant_damped_step_alpha(-1.0).is_none());
        assert!(self_concordant_damped_step_alpha(f64::NAN).is_none());
        assert!(self_concordant_damped_step_alpha(f64::INFINITY).is_none());
    }

    #[test]
    fn self_concordant_damping_matches_the_damped_newton_alpha() {
        // decrement = ½λ_N² ⇒ α = 1/(1+λ_N). λ_N = 2 ⇒ α = 1/3.
        let alpha = self_concordant_damped_step_alpha(2.0).expect("damped phase");
        assert!((alpha - 1.0 / 3.0).abs() <= 1e-15);
        // Just above the threshold the damping engages continuously.
        let lambda_n = 0.4_f64;
        let alpha = self_concordant_damped_step_alpha(0.5 * lambda_n * lambda_n)
            .expect("damped phase just above threshold");
        assert!((alpha - 1.0 / 1.4).abs() <= 1e-15);
    }
}

pub(crate) fn fused_first_attempt_log_likelihood<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    options: &BlockwiseFitOptions,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    trust_attempt: usize,
    joint_workspace_requested: bool,
) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, CustomFamilyError> {
    if trust_attempt == 0 && joint_workspace_requested {
        joint_line_search_log_likelihood_with_workspace(family, options, specs, states)
    } else {
        Ok(None)
    }
}

/// The exact Jeffreys second-order remainder at one coefficient vector.
///
/// Once `H_Φ` is active, omitting this term changes the Hessian of the inner
/// objective. Returned-mode certification therefore requires an exact
/// completion: a fused contracted implementation when the family supplies one,
/// otherwise the mathematically identical pairwise directional assembly. An
/// unavailable completion is a derivative-contract error, never permission to
/// certify a different objective.
fn exact_joint_jeffreys_completion_at<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    z_joint: &Array2<f64>,
    total_p: usize,
    context: &str,
) -> Result<Array2<f64>, CustomFamilyError> {
    let h_information = family
        .joint_jeffreys_information_with_specs(states, specs)?
        .ok_or_else(|| {
            CustomFamilyError::trial_point(format!(
                "{context}: active Jeffreys term has no information matrix"
            ))
        })?;
    if h_information.dim() != (total_p, total_p) {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: Jeffreys information shape {:?}, expected ({total_p}, {total_p})",
            h_information.dim(),
        )));
    }
    let completion = custom_family_joint_jeffreys_second_order_completion(
        family,
        states,
        specs,
        &h_information,
        z_joint,
        JeffreysCompletionAssembly::Exact,
    )?
    .ok_or_else(|| {
        format!("{context}: active Jeffreys term did not supply its exact second-order completion")
    })?;
    if completion.dim() != (total_p, total_p) || completion.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: Jeffreys completion is non-finite or has shape {:?}, expected ({total_p}, {total_p})",
            completion.dim(),
        )));
    }
    Ok(completion)
}

/// Assemble the one authoritative Hessian of the coefficient objective:
///
/// `M_true = H_likelihood + S_lambda + H_Φ + H_completion`.
///
/// `H_Φ` and its completion are an atomic pair. A caller may assemble the
/// unaugmented objective by passing neither, but it may not certify a
/// Jeffreys-augmented objective with only the divided-difference component.
fn assemble_true_joint_objective_hessian(
    mut likelihood_hessian: Array2<f64>,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    hphi: Option<&Array2<f64>>,
    completion: Option<&Array2<f64>>,
    context: &str,
) -> Result<Array2<f64>, CustomFamilyError> {
    let total_p = likelihood_hessian.nrows();
    if likelihood_hessian.ncols() != total_p {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: likelihood Hessian is not square: {:?}",
            likelihood_hessian.dim(),
        )));
    }
    add_joint_penalty_to_matrix(
        &mut likelihood_hessian,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
    );
    match (hphi, completion) {
        (None, None) => {}
        (Some(hphi), Some(completion))
            if hphi.dim() == (total_p, total_p) && completion.dim() == (total_p, total_p) =>
        {
            likelihood_hessian += hphi;
            likelihood_hessian += completion;
        }
        (Some(hphi), Some(completion)) => {
            return Err(CustomFamilyError::trial_point(format!(
                "{context}: Jeffreys curvature shape mismatch: H_phi={:?}, completion={:?}, expected ({total_p}, {total_p})",
                hphi.dim(),
                completion.dim(),
            )));
        }
        _ => {
            return Err(CustomFamilyError::trial_point(format!(
                "{context}: H_phi and its exact second-order completion must be supplied together"
            )));
        }
    }
    symmetrize_dense_in_place(&mut likelihood_hessian);
    if likelihood_hessian.iter().any(|value| !value.is_finite()) {
        return Err(CustomFamilyError::trial_point(format!(
            "{context}: exact joint objective Hessian contains a non-finite value"
        )));
    }
    Ok(likelihood_hessian)
}

fn symmetric_eigen_extremes(
    matrix: &Array2<f64>,
    context: &str,
) -> Result<(f64, f64), CustomFamilyError> {
    let (eigenvalues, _) = matrix.eigh(Side::Lower).map_err(|error| {
        CustomFamilyError::trial_point(format!(
            "{context}: symmetric eigendecomposition failed: {error}"
        ))
    })?;
    Ok((
        eigenvalues.iter().copied().fold(f64::INFINITY, f64::min),
        eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    ))
}

fn linearized_residual_contraction(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    delta: &Array1<f64>,
) -> f64 {
    let denominator = rhs
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let residual = rhs - &matrix.dot(delta);
    residual
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        / denominator
}

/// Rebuild the exact penalized coefficient Hessian at the coefficient vector
/// that an inner solve is about to return.
///
/// A first-order/stall exit inside the Newton cycle is only tentative for a
/// nonconvex family: the cycle's spectrum belongs to its head β, while an
/// accepted step changes β before several later exits can fire. This fresh
/// certificate uses the same structural dense materialization required by the
/// Laplace log-determinant, assembles the complete coefficient-objective
/// Hessian `H + S + H_Φ + completion`, and tests its inertia in the scale-aware
/// trust metric. Solver stabilization, reflected curvature, trace-only ridge,
/// and component-wise PSD projections are deliberately excluded.
pub(crate) fn exact_joint_mode_curvature_certificate<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    total_p: usize,
    active_constraints: Option<&ActiveLinearConstraintBlock>,
) -> Result<ExactJointModeCurvatureCertificate, CustomFamilyError> {
    let workspace =
        family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?;
    let source = match workspace.as_ref() {
        Some(workspace) => exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total_p,
            MaterializationIntent::LogdetFactorization,
            "fresh exact joint-mode curvature certificate",
        )?,
        None => None,
    };
    let likelihood_hessian = match source {
        Some(source) => materialize_joint_hessian_source(
            &source,
            total_p,
            "fresh exact joint-mode curvature certificate",
        )?,
        None => exact_newton_joint_hessian_symmetrized(
            family,
            states,
            specs,
            total_p,
            "fresh exact joint-mode curvature certificate",
        )?
        .ok_or_else(|| {
            "fresh exact joint-mode curvature certificate requires a joint Hessian".to_string()
        })?,
    };
    let mut metric = joint_penalty_preconditioner_diag(
        &likelihood_hessian.diag().to_owned(),
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
    );
    if let Some(floor) = family.joint_trust_metric_block_floor(states, specs)?
        && floor.len() == metric.len()
    {
        for (value, floor_value) in metric.iter_mut().zip(floor.iter()) {
            if floor_value.is_finite() && *floor_value > *value {
                *value = *floor_value;
            }
        }
    }
    let jeffreys_curvature = if family.joint_jeffreys_term_required() {
        let z_joint = build_joint_jeffreys_subspace(family, specs, ranges)?.ok_or_else(|| {
            "fresh exact joint-mode curvature certificate: Jeffreys family has no coefficient subspace"
                .to_string()
        })?;
        match custom_family_joint_jeffreys_term(family, states, specs, ranges, &z_joint)? {
            Some((_phi, _gradient, hphi)) => {
                let completion = exact_joint_jeffreys_completion_at(
                    family,
                    states,
                    specs,
                    &z_joint,
                    total_p,
                    "fresh exact joint-mode curvature certificate",
                )?;
                Some((hphi, completion))
            }
            None => None,
        }
    } else {
        None
    };
    let jeffreys_completion_assembled = jeffreys_curvature.is_some();
    let hessian = assemble_true_joint_objective_hessian(
        likelihood_hessian,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
        jeffreys_curvature.as_ref().map(|(hphi, _)| hphi),
        jeffreys_curvature
            .as_ref()
            .map(|(_, completion)| completion),
        "fresh exact joint-mode curvature certificate",
    )?;
    // Constrained modes are certified on the active-face TANGENT null(A_act) —
    // the same geometry the terminal determinant integrates over
    // (`active_face_logdet_with_ridge_policy`). Curvature normal to the face
    // is neither integrated by the Laplace approximation nor differentiated by
    // the constrained outer kernel, so full-space indefiniteness there is not
    // evidence against the mode; conversely a saddle WITHIN the face is
    // exactly the point a first-order KKT certificate cannot see (the #979
    // CTN cycle-97 witness: KKT-certified with tangent min_eig = -7.9, which
    // then killed the downstream SPD determinant). A fully pinned mode has no
    // free directions and is trivially certified.
    // Retain the active-face tangent `Z` so a resolved negative-curvature mode
    // (certified in the reduced tangent space) can be mapped back into the full
    // joint coefficient layout as a saddle-escape direction.
    let (certificate_matrix, certificate_metric, tangent) = match active_constraints {
        Some(active) => match active_constraint_tangent_geometry(&active.a)? {
            ActiveConstraintTangentGeometry::FullyPinned => {
                return Ok(ExactJointModeCurvatureCertificate {
                    workspace,
                    minimum_whitened_eigenvalue: f64::INFINITY,
                    numerical_floor: 0.0,
                    jeffreys_completion_assembled,
                    negative_curvature_direction: None,
                });
            }
            ActiveConstraintTangentGeometry::Tangent(z) => {
                let reduced = z.t().dot(&hessian).dot(&z);
                // Diagonal of Zᵀ·diag(metric)·Z: the exact positive scaling of
                // the whitening metric expressed on the tangent basis.
                let mut reduced_metric = Array1::<f64>::zeros(z.ncols());
                for j in 0..z.ncols() {
                    let mut projected = 0.0;
                    for k in 0..z.nrows() {
                        projected += metric[k] * z[[k, j]] * z[[k, j]];
                    }
                    reduced_metric[j] = projected;
                }
                (reduced, reduced_metric, Some(z))
            }
        },
        None => (hessian, metric, None),
    };
    let zero_rhs = Array1::<f64>::zeros(certificate_matrix.nrows());
    let spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
        &certificate_matrix,
        &zero_rhs,
        &certificate_metric,
        KKT_REFUSAL_RANK_TOL,
    )?;
    let minimum_whitened_eigenvalue = spectrum.gamma.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_whitened_eigenvalue = spectrum
        .gamma
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    log::info!(
        "[979-MODE-HESSIAN] eig(M_true_tangent_whitened)=[{minimum_whitened_eigenvalue:.6e},{maximum_whitened_eigenvalue:.6e}] numerical_floor={:.6e} tangent_dim={}",
        spectrum.numerical_floor,
        certificate_matrix.nrows(),
    );
    // A strict saddle exposes a resolvable negative-curvature eigenvector. Map
    // that whitened mode back to the raw coefficient direction `δ = D^{-1/2} v`
    // (curvature `δᵀ H_pen δ = γ_min` for the unit eigenvector `v`), then lift it
    // through the tangent `Z` when the mode was certified on a reduced face. The
    // resulting full-space direction is the one an inner saddle-escape steps
    // along; a PSD mode carries no such direction.
    let negative_curvature_direction = if minimum_whitened_eigenvalue < -spectrum.numerical_floor {
        let mut argmin = 0usize;
        let mut best = f64::INFINITY;
        for (index, &value) in spectrum.gamma.iter().enumerate() {
            if value < best {
                best = value;
                argmin = index;
            }
        }
        let eigenvector = spectrum.evecs.column(argmin);
        let reduced_direction = Array1::from_iter(
            spectrum
                .d_inv_sqrt
                .iter()
                .zip(eigenvector.iter())
                .map(|(scale, component)| scale * component),
        );
        let full_direction = match tangent.as_ref() {
            Some(z) => z.dot(&reduced_direction),
            None => reduced_direction,
        };
        if full_direction.iter().all(|value| value.is_finite()) {
            Some(full_direction)
        } else {
            None
        }
    } else {
        None
    };
    Ok(ExactJointModeCurvatureCertificate {
        workspace,
        minimum_whitened_eigenvalue,
        numerical_floor: spectrum.numerical_floor,
        jeffreys_completion_assembled,
        negative_curvature_direction,
    })
}

/// Maximum number of times the constrained joint-Newton inner solve steps off a
/// first-order KKT point that certifies as a strict saddle on the active-face
/// tangent before it refuses the fit.
///
/// A first-order-stationary point with a resolvable negative face-tangent
/// eigenvalue is NOT a Laplace mode: the penalized objective strictly decreases
/// along that eigenvector, so the standard second-order response is to step
/// along it and continue the solve, not to refuse. Two escapes clear any
/// isolated saddle the fixed-penalty coefficient objective exposes on the way to
/// a mode; a point that still certifies as a saddle after two feasible escapes
/// is evidence of a genuinely non-modal ρ the outer optimizer must reject, so
/// the honest typed refusal is kept on the final attempt.
const MAX_SADDLE_ESCAPES: usize = 2;

/// How many times one saddle resolution may move a blocking row onto the
/// certified face before it gives the honest refusal instead.
///
/// Each exchange costs a full exact curvature certificate — the expensive
/// object in this function — and strictly grows the face, so this is an
/// operational budget rather than the termination proof (the face cannot grow
/// past the constraint count, which is the mathematical bound and far too large
/// to spend). A genuine constrained mode is expected to certify within one or
/// two exchanges; needing more says the face is degenerate in a way the refusal
/// should report rather than grind at.
const MAX_ESCAPE_FACE_EXCHANGES: usize = 3;

/// Verdict of second-order certification at a constrained first-order KKT point.
enum ConstrainedModeResolution {
    /// The active-face-tangent curvature is PSD: a genuine Laplace mode.
    Certified {
        workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    },
    /// The point is a strict face-tangent saddle. Step `alpha · direction` (in
    /// the full joint layout, `alpha` carrying the feasible sign) strictly
    /// lowers the objective and stays feasible; the caller applies it and
    /// resumes the inner solve.
    Escape {
        direction: Array1<f64>,
        alpha: f64,
        lambda_min: f64,
    },
}

/// Second-order certification of a constrained first-order KKT point, with a
/// bounded feasible saddle-escape when the active-face tangent is indefinite.
///
/// `saddle_escapes_used` is the number of escapes already spent this solve; once
/// it reaches [`MAX_SADDLE_ESCAPES`] a still-indefinite point yields the honest
/// typed refusal instead of another escape.
fn resolve_constrained_converged_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    total_p: usize,
    block_constraints: &[Option<ConstraintSet>],
    cached_active_sets: &[Option<Vec<usize>>],
    saddle_escapes_used: usize,
    previous_escape_lambda_min: Option<f64>,
    objective_tol: f64,
    jeffreys_completion_calls: &mut usize,
) -> Result<ConstrainedModeResolution, CustomFamilyError> {
    // Certify on the tangent of every NUMERICALLY-TIGHT constraint, not only the
    // QP's recorded active set. At a degenerate binding vertex the QP can leave a
    // row with slack below the primal-feasibility tolerance OUT of
    // `cached_active_sets` (a phantom-dual / zero-multiplier omission). Such a row
    // is on the active face all the same: the Laplace tangent must null it, or
    // the certificate over-counts free directions and manufactures a PHANTOM
    // saddle whose negative curvature lives almost entirely normal to that
    // near-tight row — the escape direction then points straight into it and the
    // feasible step collapses to ~1e-11 (the measured CTN witness: lambda_min=
    // -7.1e-1 with alpha=-1e-11, a no-op). Building the face from all tight rows
    // resolves that phantom to a genuine constrained mode, and any REAL saddle
    // keeps a feasible escape (its direction lies in the tight-face tangent, so
    // it has zero rate on every tight row and a meaningful feasible length).
    let tight_active_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
        block_constraints,
        states,
        cached_active_sets,
    )?;
    resolve_constrained_converged_mode_on_face(
        family,
        states,
        specs,
        options,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
        total_p,
        block_constraints,
        tight_active_sets,
        saddle_escapes_used,
        previous_escape_lambda_min,
        objective_tol,
        jeffreys_completion_calls,
        0,
    )
}

/// The body of [`resolve_constrained_converged_mode`] with the certified face
/// supplied rather than derived, so a blocked escape can retry on a wider one.
///
/// `face_exchanges` counts the rows this resolution has already moved onto the
/// face; it bounds the recursion at [`MAX_ESCAPE_FACE_EXCHANGES`].
fn resolve_constrained_converged_mode_on_face<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    total_p: usize,
    block_constraints: &[Option<ConstraintSet>],
    tight_active_sets: Vec<Option<Vec<usize>>>,
    saddle_escapes_used: usize,
    previous_escape_lambda_min: Option<f64>,
    objective_tol: f64,
    jeffreys_completion_calls: &mut usize,
    face_exchanges: usize,
) -> Result<ConstrainedModeResolution, CustomFamilyError> {
    let mode_active_block =
        assemble_active_constraint_block(block_constraints, &tight_active_sets, ranges, total_p);
    let certificate = exact_joint_mode_curvature_certificate(
        family,
        states,
        specs,
        options,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
        total_p,
        mode_active_block.as_ref(),
    )?;
    if certificate.jeffreys_completion_assembled {
        *jeffreys_completion_calls += 1;
    }
    let lambda_min = certificate.minimum_whitened_eigenvalue;
    let numerical_floor = certificate.numerical_floor;
    if !certificate.has_resolvable_negative_curvature() {
        log::info!(
            "[PIRLS/joint-Newton mode certificate] constrained returned beta certified from fresh exact curvature: lambda_min={lambda_min:.6e}, floor={numerical_floor:.6e}",
        );
        return Ok(ConstrainedModeResolution::Certified {
            workspace: certificate.workspace,
        });
    }
    // ── did the last escape actually escape? ────────────────────────────────
    //
    // The escape takes a fixed-length step along a direction of negative
    // curvature and keeps it unconditionally. Nothing checked whether the step
    // improved the quantity the escape exists to improve, and #2587 measured
    // what that costs: with the budget raised to 12 the curvature WANDERS
    // rather than converging --
    //
    //     attempt 1  -1.894e0     attempt 5  -2.580e0
    //     attempt 2  -3.803e-1    attempt 6  -1.784e-1   <- best seen
    //     attempt 3  -1.912e0     attempt 7  -4.945e-1
    //     attempt 4  -2.239e0     attempt 8  -3.806e0    <- worse than the start
    //
    // -- ending twice as indefinite as it began, having walked away from two
    // perfectly good points on the way. Five of those eight steps made
    // `lambda_min` worse and every one was kept.
    //
    // A step taken because the model says it should help, with nothing checking
    // whether it did, is the same defect as the sizing bug this issue opened
    // with. `lambda_min` is negative, so improvement means CLOSER TO ZERO: a
    // fresh certificate no better than the one the escape left is proof that
    // this direction is not a way out of this saddle, and spending the rest of
    // the budget on it only travels further. Stop and refuse honestly, naming
    // both values so the refusal states what was tried.
    if let Some(previous) = previous_escape_lambda_min
        && lambda_min <= previous
    {
        return Err(CustomFamilyError::trial_point(format!(
            "joint Newton returned-mode curvature is still a strict saddle after a negative-curvature escape that did not improve it: lambda_min={lambda_min:.6e} is no closer to zero than the pre-escape {previous:.6e} (floor={numerical_floor:.6e}, escapes spent={saddle_escapes_used}); the escape direction is not a way out of this saddle, so the remaining budget would only travel further",
        )));
    }
    if saddle_escapes_used >= MAX_SADDLE_ESCAPES {
        return Err(CustomFamilyError::trial_point(format!(
            "joint Newton tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}; an indefinite coefficient point cannot define a Laplace mode (after {saddle_escapes_used} negative-curvature escapes)",
        )));
    }
    let Some(direction) = certificate.negative_curvature_direction else {
        return Err(CustomFamilyError::trial_point(format!(
            "joint Newton returned-mode curvature is a strict saddle (lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}) but the certificate exposed no finite escape direction",
        )));
    };
    // Along the raw coefficient direction `δ` the exact curvature is `γ_min`
    // (`δᵀ H_pen δ = γ_min`), so a step `s·δ` lowers the quadratic model by
    // `½ s² |γ_min|` — monotonically in `s` and WITHOUT BOUND. A direction of
    // negative curvature has no interior optimum to aim at, so nothing about
    // the model itself bounds `s`. The only legitimate ceilings are the two
    // that say where the model stops being usable: locality (the quadratic is
    // trusted only near β) and feasibility (the step must stay in the
    // polytope). Both are applied below.
    //
    // `decrease_floor` is the `s` at which the guaranteed decrease `½s²|γ_min|`
    // first clears solver noise. That is a FLOOR — a shorter escape cannot be
    // distinguished from not moving — and it was previously combined with the
    // locality cap by `min`, which made every escape exactly as short as noise
    // permitted. Newton is attracted to stationary points, so from that
    // distance it walked straight back down to the same saddle: #2587 measured
    // `lambda_min` unchanged to seven significant figures across both escapes.
    let gamma_min = lambda_min.abs();
    if !(gamma_min.is_finite() && gamma_min > 0.0) {
        return Err(CustomFamilyError::trial_point(format!(
            "saddle-escape could not size a step: non-finite curvature magnitude lambda_min={lambda_min:.6e}",
        )));
    }
    let direction_norm = direction.dot(&direction).sqrt();
    if !(direction_norm.is_finite() && direction_norm > 0.0) {
        return Err(CustomFamilyError::trial_point(
            "saddle-escape direction is degenerate (zero or non-finite norm)".to_string(),
        ));
    }
    let beta = flatten_state_betas(states, specs);
    let beta_norm = beta.dot(&beta).sqrt();
    let decrease_floor = (2.0 * objective_tol.max(1e-8) / gamma_min).sqrt();
    // A failed escape carries exactly one piece of information: the length it
    // used was inside the basin the solve then fell back into. Recomputed from
    // the returned saddle with the same inputs, the next escape would be
    // bit-identical to the last, so the `MAX_SADDLE_ESCAPES` budget could never
    // do anything the first attempt did not — spending it was structurally a
    // repetition, not a retry. Widen the trusted neighbourhood once per spent
    // escape so the budget is a ladder. The bound is `MAX_SADDLE_ESCAPES`, so
    // the widest escape stays within `0.1·2^(MAX-1)` of the coefficient scale.
    let escape_widening = f64::from(1u32 << saddle_escapes_used.min(16) as u32);
    let locality_cap = escape_widening * 0.1 * (1.0 + beta_norm) / direction_norm;
    let base_magnitude = locality_cap.max(decrease_floor);
    // Feasibility. The tangent-projected direction satisfies the ACTIVE rows to
    // first order; truncate strictly inside the first INACTIVE blocker, in the
    // scaled-slack terms the active-set solvers use (row norm cancels in the
    // ratio). Both signs of `δ` give the same second-order decrease, so choose
    // the sign that admits the longer feasible step.
    let joint_active =
        flatten_joint_active_set(&tight_active_sets, block_constraints).unwrap_or_default();
    let mut feasible_positive = f64::INFINITY;
    let mut feasible_negative = f64::INFINITY;
    // WHICH row truncates the chord, per sign. A chord shorter than
    // `decrease_floor` is not a length problem to be solved by stepping
    // further: it says this row pins the direction, and the active-set response
    // to that is to put the row on the face (#2587).
    let mut blocker_positive: Option<usize> = None;
    let mut blocker_negative: Option<usize> = None;
    if let Some(joint_constraints) =
        assemble_joint_linear_constraints(block_constraints, ranges, total_p)?
    {
        let values_beta = joint_constraints.values(beta.view())?;
        let values_direction = joint_constraints.values(direction.view())?;
        for row in 0..joint_constraints.nrows() {
            if joint_active.contains(&row) {
                continue;
            }
            let norm = joint_constraints.row_norm(row)?;
            if !(norm.is_finite() && norm > 0.0) {
                continue;
            }
            let bound = joint_constraints.bound(row)?;
            if bound == f64::NEG_INFINITY {
                continue;
            }
            let scaled_slack = (values_beta[row] - bound) / norm;
            let scaled_rate = values_direction[row] / norm;
            // Decide the row only if it CAN be decided (gam#2721). A `NaN`
            // fails `scaled_slack < 0.0`, `scaled_rate < 0.0` AND
            // `scaled_rate > 0.0`, so the row would drop out of BOTH sign
            // folds and leave the feasible cap at `+INFINITY` — the escape
            // chord would be truncated by nothing at all. Refuse instead; the
            // predicate is the constraint carrier's own.
            if !gam_problem::feasibility_quantities_are_finite(&[scaled_slack, scaled_rate]) {
                return Err(CustomFamilyError::trial_point(format!(
                    "saddle-escape feasibility row {row} cannot be decided: \
                     scaled slack={scaled_slack:.3e}, scaled rate={scaled_rate:.3e}; \
                     every comparison in the chord truncation is false for NaN, so \
                     skipping the row would leave the chord untruncated (gam#2721)"
                )));
            }
            if scaled_slack < 0.0 {
                continue;
            }
            if scaled_rate < 0.0 {
                let step = scaled_slack / -scaled_rate;
                if step < feasible_positive {
                    feasible_positive = step;
                    blocker_positive = Some(row);
                }
            } else if scaled_rate > 0.0 {
                let step = scaled_slack / scaled_rate;
                if step < feasible_negative {
                    feasible_negative = step;
                    blocker_negative = Some(row);
                }
            }
        }
    }
    let (sign, feasible_cap, blocking_row) = if feasible_positive >= feasible_negative {
        (1.0_f64, feasible_positive, blocker_positive)
    } else {
        (-1.0_f64, feasible_negative, blocker_negative)
    };
    let feasible_cap = if feasible_cap.is_finite() {
        // Land strictly inside the blocker (matching the reduced-face solver's
        // 1e-12 inward shave), never on or past it.
        feasible_cap * (1.0 - 1e-12)
    } else {
        feasible_cap
    };
    let magnitude = base_magnitude.min(feasible_cap);
    // ── an escape that cannot escape ────────────────────────────────────────
    //
    // `decrease_floor` is the length at which the guaranteed decrease
    // `½s²|γ_min|` first clears solver noise. When FEASIBILITY truncates the
    // chord below it, the step is — by this function's own definition —
    // indistinguishable from not moving, and taking it spends one of
    // `MAX_SADDLE_ESCAPES` on a provable no-op.
    //
    // Measured (#2587, instrumented transformation-normal arm): `binding=
    // feasible` on both escapes, `feasible_cap=3.706317e-4` against
    // `decrease_floor=6.671092e-3` (18x below) and `1.457053e-3` against
    // `8.098159e-3` (5.6x below). Neither the locality cap nor the floor ever
    // bound. So the DIRECTION fails, not the length: it lies almost entirely in
    // the normal cone of `blocking_row` — exactly the phantom-saddle geometry
    // this function's header describes.
    //
    // The active-set answer to "this direction is pinned by that row" is to put
    // the row ON the face and ask the curvature question again there. Either the
    // narrower tangent certifies — the point was a constrained mode all along
    // and the negative curvature was normal to a row the face omitted — or it
    // yields a direction with a feasible length to travel. An exchange, not a
    // retry: the face strictly grows each time.
    if let Some(row) = blocking_row
        && magnitude < decrease_floor
        && face_exchanges < MAX_ESCAPE_FACE_EXCHANGES
    {
        log::info!(
            "[PIRLS/joint-Newton saddle-escape exchange] lambda_min={lambda_min:.6e} \
             feasible_cap={feasible_cap:.6e} < decrease_floor={decrease_floor:.6e}; \
             moving blocking row {row} onto the certified face (exchange {})",
            face_exchanges + 1,
        );
        let mut widened = joint_active;
        widened.push(row);
        widened.sort_unstable();
        widened.dedup();
        let widened_face =
            crate::blockwise_solve::scatter_joint_active_set(&widened, block_constraints);
        return resolve_constrained_converged_mode_on_face(
            family,
            states,
            specs,
            options,
            ranges,
            s_lambdas,
            joint_mode_diagonal_ridge,
            joint_bundle,
            total_p,
            block_constraints,
            widened_face,
            saddle_escapes_used,
            previous_escape_lambda_min,
            objective_tol,
            jeffreys_completion_calls,
            face_exchanges + 1,
        );
    }
    if !(magnitude.is_finite() && magnitude > 0.0) {
        return Err(CustomFamilyError::trial_point(format!(
            "joint Newton returned-mode curvature is a strict saddle (lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}) with no feasible escape length along the negative-curvature tangent (locality_cap={locality_cap:.6e}, decrease_floor={decrease_floor:.6e}, feasible_cap={feasible_cap:.6e})",
        )));
    }
    // Which of the three terms BINDS is the whole diagnosis when an escape
    // fails to escape, and it is not recoverable from `alpha` alone (#2587 cost
    // a measurement cycle to establish that `decrease_floor` was binding).
    log::info!(
        "[PIRLS/joint-Newton saddle-escape sizing] lambda_min={lambda_min:.6e} \
         locality_cap={locality_cap:.6e} decrease_floor={decrease_floor:.6e} \
         feasible_cap={feasible_cap:.6e} widening={escape_widening:.1} \
         magnitude={magnitude:.6e} binding={}",
        if magnitude >= feasible_cap {
            "feasible"
        } else if base_magnitude <= decrease_floor {
            "decrease_floor"
        } else {
            "locality"
        },
    );
    Ok(ConstrainedModeResolution::Escape {
        direction,
        alpha: sign * magnitude,
        lambda_min,
    })
}

pub(crate) fn inner_blockwise_fit<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    inner_blockwise_fit_for_product(
        family,
        specs,
        block_log_lambdas,
        options,
        warm_start,
        InnerFitProduct::LaplaceReady,
    )
}

/// Correct a coefficient mode without constructing a Laplace product.
///
/// This is deliberately a distinct operation rather than an option bit: the
/// returned state is still required to pass the identical convergence, KKT,
/// active-face, and fresh returned-mode curvature certificates. Only the two
/// determinant artifacts are absent, because a continuation interior consumes
/// only the coefficient state as the next corrector's predictor.
pub(crate) fn inner_blockwise_coefficient_mode<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    inner_blockwise_fit_for_product(
        family,
        specs,
        block_log_lambdas,
        options,
        warm_start,
        InnerFitProduct::CoefficientMode,
    )
}

fn inner_blockwise_fit_for_product<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
    product: InnerFitProduct,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    // Inner-blockwise prelude waypoints. At large-scale n the cold-start
    // path between function entry and the first PIRLS/JN cycle-summary
    // log can run for many minutes (sometimes hours) silently while
    // row-kernel workspace builds run. Emit a `[STAGE] PIRLS/inner`
    // line at each transition so the next failed run pinpoints which
    // named step holds time. Gated on large-scale n so small-fit
    // tests stay quiet.
    let inner_started = std::time::Instant::now();
    let mut states = buildblock_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let total_joint_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let total_joint_n = joint_observation_count(&states);
    const INNER_PRELUDE_LOG_MIN_N: usize = 100_000;
    let prelude_log = total_joint_n >= INNER_PRELUDE_LOG_MIN_N;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=buildblock_states+refresh_etas elapsed={:.3}s n={} p={} blocks={}",
            inner_started.elapsed().as_secs_f64(),
            total_joint_n,
            total_joint_p,
            specs.len(),
        );
    }
    let matrix_free_joint_requested = use_joint_matrix_free_path(total_joint_p, total_joint_n)
        || family.prefers_matrix_free_inner_joint(specs, &states);
    let has_workspace_source = family.inner_coefficient_hessian_hvp_available(specs);
    // Probe the *spec-aware* joint Hessian: it is the canonical source of the
    // coupled joint curvature. A family may override only
    // `exact_newton_joint_hessian_with_specs` (the variant that has access to
    // the realized block designs needed to assemble the cross-block
    // `X_aᵀ diag(w_ab) X_b` blocks — e.g. the Dirichlet common-parameterization
    // family, whose `evaluate` emits diagonal working sets so the spec-less
    // default block assembler returns `None`). Routing the inner joint-Newton
    // availability gate through the spec-less `exact_newton_joint_hessian`
    // would then mis-classify such a family as "no joint Hessian" and drop it
    // onto pure block-diagonal backfitting, which fails to reach KKT on small,
    // concentrated coupled likelihoods. The `_with_specs` path subsumes the
    // spec-less one for every family (single-block / uncoupled delegate
    // identically), so it is the correct probe here.
    // The DECLARED dense joint curvature, when the family commits to one.
    //
    // gam#1088's loud arm is a statement about what the family declares as its
    // analytic second derivative at the starting β — not about whichever
    // representation the solver happens to consume. Those two differed, and the
    // gap had no guard on either side of it. A family supplying BOTH a dense
    // `exact_newton_joint_hessian` and an HVP sets `has_workspace_source`, so
    // this probe short-circuited to `true` without looking at the dense
    // curvature at all; the workspace path's own check then probes the
    // `JointHessianSource::Operator` variant, whose finiteness test is its
    // ASSEMBLED DIAGONAL (the full operator is never materialised there). A
    // non-finite entry present in the declared dense curvature and absent from
    // the HVP was therefore examined by neither. Measured on
    // `TwoBlockNonFiniteCurvatureFamily`, which declares `[[NaN, 0.25], [0.25,
    // 1.0]]`: `inner_blockwise_fit` returned `Ok` — a fit minted from a
    // curvature that does not exist, where the contract is a typed failure.
    //
    // `has_explicit_joint_hessian()` is the family's own statement that it
    // materialises a dense p×p, so consulting the declaration here costs an
    // HVP-only family nothing: such a family answers `false` and never
    // materialises anything.
    let declared_dense_joint = if has_workspace_source && !family.has_explicit_joint_hessian() {
        None
    } else {
        family.exact_newton_joint_hessian_with_specs(&states, specs)?
    };
    let declares_dense_joint = declared_dense_joint.is_some();
    if let Some(joint_hessian) = declared_dense_joint {
        crate::joint_newton::joint_hessian_source_finite_check(
            &crate::joint_newton::JointHessianSource::Dense(joint_hessian),
        )?;
    }
    // A family reaches the joint-exact route either through its HVP workspace
    // or through a declared dense joint curvature; both were previously
    // answered here, but only the second had its finiteness examined, and the
    // check above now covers both. The materialisation the old branch performed
    // is the same one `declared_dense_joint` performs, so nothing is evaluated
    // twice.
    //
    // gam#1088 scopes the loud arm to exactly this point: "a non-finite entry
    // in the family's analytic joint curvature at the starting beta is a
    // contract violation against the family's second derivative -- the solve
    // cannot even begin". A non-finite entry that only emerges after the
    // coupled loop has driven beta to an overflowing operating point is a
    // genuine rho-degeneracy and still exits gracefully through the in-loop
    // guard.
    let has_joint_exacthessian = has_workspace_source || declares_dense_joint;
    // When the family declares its likelihood blocks UNCOUPLED
    // (`∂²L/∂β_a∂β_b = 0` for every a ≠ b) the joint penalized objective is
    // fully separable across blocks: the joint Hessian is exactly
    // block-diagonal and each block carries only its own penalty. On a
    // separable objective block-coordinate descent solves each block's
    // (possibly inequality-constrained) subproblem to its own exact optimum —
    // it IS the joint solve, and each block gets its OWN trust radius, its OWN
    // active-set QP, and its OWN KKT certificate.
    //
    // Forcing the coupled joint-Newton onto such a problem instead couples two
    // independent blocks under ONE shared trust radius and ONE concatenated
    // KKT residual. That is actively harmful when the blocks differ sharply in
    // conditioning — the competing-risks twin time-basis fit (#1025) is the
    // canonical case: two cause-specific baselines share the same I-spline
    // evaluated at the same event times, but one cause sits near its
    // monotonicity-constraint boundary with an O(1e5) hazard-derivative
    // gradient while the other is interior. The shared globalization cannot
    // satisfy both blocks' KKT conditions at once; the joint residual stalls
    // far above tolerance, the inner solve burns its whole cycle budget on
    // every outer ρ-eval, and the fit only survives by falling through to the
    // block-coordinate path anyway (which then converges in a handful of
    // cycles). Route uncoupled multi-block specs straight to that exact
    // separable path. Uncoupled families are routed to blockwise before a joint
    // solve starts, so this stops the engine from attempting — and grinding on
    // — a joint solve it was never required to run.
    //
    // Single-block families and genuinely coupled multi-block families are
    // unaffected: the former never had cross-block coupling to begin with, the
    // latter still take the joint path (their objective is NOT separable, so
    // block-coordinate descent would drop the cross-block ∂²L/∂β_a∂β_b
    // curvature).
    let blocks_separable = specs.len() >= 2 && family.likelihood_blocks_uncoupled();
    let use_joint_newton =
        has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source) && !blocks_separable;
    let joint_workspace_requested = use_joint_newton && has_workspace_source;
    // Row-measure consistency for the outer-score subsample (gam#1135 HT path).
    //
    // `BlockwiseFitOptions::outer_score_subsample` carries a per-row
    // Horvitz–Thompson reweighting. The inner β-solve has several likelihood
    // evaluators that must all agree on ONE row measure for the trust-region
    // ratio `ρ = [F(β) − F(β+δ)] / [−g·δ − ½δᵀHδ]` to be valid:
    //
    //   * the coefficient line search, which ALWAYS evaluates
    //     `log_likelihood_only_with_options` and so applies the subsample
    //     whenever it is present in `options`;
    //   * the joint Hessian, built via
    //     `exact_newton_joint_hessian_workspace_with_options` (HT) when the
    //     workspace path is engaged;
    //   * the entry/reload base objective + gradient from
    //     `load_joint_gradient_evaluation`, which only honours the subsample
    //     through its workspace branch — guarded by
    //     `inner_joint_workspace_gradient_available`. A family that does NOT
    //     advertise that capability (e.g. GaussianLocationScale) falls through
    //     to `family.evaluate` / `exact_newton_joint_gradient_evaluation`, which
    //     ignore `options` and score the FULL data.
    //
    // When the base objective is full-data but the line search is HT, the
    // trust-region numerator compares `F_full(β)` against `F_HT(β+δ)`. The two
    // differ by a β-independent constant (the HT-vs-full log-likelihood gap), so
    // `actual_reduction` stays pinned at that constant even as the step shrinks
    // to machine ε — every attempt rejects, the radius collapses, and the inner
    // solve exits non-converged. That cascades to "no candidate seeds passed
    // outer startup validation" and the whole fit fails — the manifestation is a
    // GaussianLocationScale fit with an outer-score subsample installed (manual
    // or auto-installed at scale) that cannot complete its final inner refit.
    //
    // The subsample is an OUTER-score variance-reduction device, consumed by the
    // outer ψ/ρ derivative path (`psi_hyper`); β̂(ρ) itself must stay the
    // unbiased full-data optimum unless the family can run a FULLY HT-consistent
    // inner solve. It can do so exactly when its entry ll+gradient also honour
    // the subsample, i.e. `inner_joint_workspace_gradient_available` (BMS,
    // survival marginal-slope) — there the line search, Hessian, and base
    // objective are all HT and the contract is preserved. Otherwise (the
    // GaussianLocationScale contract: "inner PIRLS never installs the option, so
    // the inner solve continues to consume the exact full-data log-likelihood")
    // the inner solve must run on full data; strip the subsample so the entry
    // objective, gradient/Hessian, line search, and the trust-region
    // row-measure bookkeeping all agree on the full-data measure.
    let inner_consumes_subsample =
        joint_workspace_requested && family.inner_joint_workspace_gradient_available(specs);
    let stripped_subsample_options;
    let options = if !inner_consumes_subsample && options.outer_score_subsample.is_some() {
        let mut cleaned = options.clone();
        cleaned.outer_score_subsample = None;
        stripped_subsample_options = cleaned;
        &stripped_subsample_options
    } else {
        options
    };
    let inner_tol = options.inner_tol;
    let inner_max_cycles_base = options.inner_max_cycles;
    // Per-outer-call inner-cycle cap. The earlier "adaptive inner cycle
    // cap" doubled this mid-loop on plateaus, but that turned out to be
    // the wrong response to stalled descent (descent ratios pinned at
    // ~0.999 paired with a sub-tolerance objective change is the
    // no-descent signal, not a "give Newton more cycles" signal). The
    // plateau-flat-objective convergence certificate in the inner-cycle
    // body now handles that case directly, so the cap stays fixed at the
    // baseline for the lifetime of this outer call.
    let inner_max_cycles = capped_inner_max_cycles(options, inner_max_cycles_base);
    // Each block's assembled penalty matrix depends only on that block's
    // penalties and smoothing parameters. Build these setup matrices in
    // parallel, but keep the coordinate-descent and line-search loops below
    // strictly serial because each accepted block update changes the state seen
    // by later blocks.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let s_lambdas_launch_started = std::time::Instant::now();
    let s_lambdas_par_iter = (0..specs.len()).into_par_iter().map(|b| {
        let spec = &specs[b];
        let Some(block_log_lambda) = block_log_lambdas.get(b) else {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!("missing log-smoothing parameter vector for block {b}"),
            });
        };
        if block_log_lambda.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} log-smoothing parameter length {} does not match penalties {}",
                    block_log_lambda.len(),
                    spec.penalties.len()
                ),
            });
        }

        let p = spec.design.ncols();
        let lambdas = exact_lambdas_from_log_strengths(
            block_log_lambda,
            &format!("inner block {b} log strength"),
        )?;
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        Ok(s_lambda)
    });
    let s_lambdas_collect_started = std::time::Instant::now();
    let s_lambdas_launch_elapsed = s_lambdas_launch_started.elapsed();
    let s_lambdas = s_lambdas_par_iter.collect::<Result<Vec<_>, CustomFamilyError>>()?;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=s_lambdas par_iter launch={:.3}s collect={:.3}s blocks={} (since inner-start={:.3}s)",
            s_lambdas_launch_elapsed.as_secs_f64(),
            s_lambdas_collect_started.elapsed().as_secs_f64(),
            specs.len(),
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let joint_bundle: Option<&gam_problem::JointPenaltyBundle> = options.joint_penalties.as_deref();
    if let Some(bundle) = joint_bundle {
        for (i, spec) in bundle.specs().iter().enumerate() {
            if spec.dim() != total_joint_p {
                return Err(CustomFamilyError::trial_point(format!(
                    "joint penalty {i}: dim {} != total compiled p {}",
                    spec.dim(),
                    total_joint_p,
                )));
            }
        }
        assert_eq!(bundle.specs().len(), bundle.log_lambdas().len());
    }
    let objective_state =
        crate::assembly::InnerObjectiveState::new(family, block_log_lambdas, joint_bundle);
    // A cached mode that is refused ONLY because the augmentation strength moved
    // is the coefficient-objective homotopy doing exactly what it exists to do,
    // and it used to be the silent case that broke it (#2612): the ρ half is
    // constant along that path, so before the strength joined the state the
    // whole equality answered `true` at every waypoint and the corrector was
    // skipped whenever the incoming mode happened to still read PSD. Saying so
    // makes a homotopy that is tracking its branch distinguishable, in the run
    // record, from one that is not.
    if let Some(cached) = warm_start.and_then(|seed| seed.cached_inner.as_ref())
        && cached.objective_state.jeffreys_strength() != objective_state.jeffreys_strength()
    {
        log::info!(
            "[PIRLS/joint-Newton warm-start] cached inner mode is not this objective's: Jeffreys \
             augmentation strength {:.17e} -> {:.17e} at an unchanged smoothing state; correcting \
             rather than reusing (#2612)",
            cached.objective_state.jeffreys_strength(),
            objective_state.jeffreys_strength(),
        );
    }
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        // The cached mode is reusable only when it was solved for THIS inner
        // coefficient objective — the per-block penalties, the joint bundle
        // (#2615), and the family's Jeffreys augmentation strength (#2612).
        if let Some(cached) = seed.cached_inner.as_ref()
            && cached.objective_state == objective_state
            && cached.converged
            && (!product.requires_laplace_artifacts()
                || (cached.block_logdet_h.is_some_and(f64::is_finite)
                    && cached.block_logdet_s.is_some_and(f64::is_finite)))
            && seed
                .block_beta
                .iter()
                .zip(&states)
                .all(|(beta_seed, state)| beta_seed.len() == state.beta.len())
        {
            for (state, beta_seed) in states.iter_mut().zip(&seed.block_beta) {
                state.beta.assign(beta_seed);
            }
            cached_active_sets = seed.active_sets.clone();
            refresh_all_block_etas(family, specs, &mut states)?;
            let local_ranges = block_param_ranges(specs);
            let local_joint_mode_diagonal_ridge =
                if ridge > 0.0 && options.ridge_policy.accounts_for_objective() {
                    ridge
                } else {
                    0.0
                };
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints = assemble_joint_linear_constraints(
                &block_constraints,
                &local_ranges,
                total_joint_p,
            )?;
            let mode_active_block = if joint_constraints.is_some() {
                let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
                    &block_constraints,
                    &states,
                    &cached_active_sets,
                )?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &tight_sets,
                    &local_ranges,
                    total_joint_p,
                )
            } else {
                None
            };
            let mut cached_mode_acceptable = true;
            let mut certified_workspace = cached.joint_workspace.clone();
            if has_joint_exacthessian {
                match exact_joint_mode_curvature_certificate(
                    family,
                    &states,
                    specs,
                    options,
                    &local_ranges,
                    &s_lambdas,
                    local_joint_mode_diagonal_ridge,
                    joint_bundle,
                    total_joint_p,
                    mode_active_block.as_ref(),
                ) {
                    Ok(certificate) => {
                        cached_mode_acceptable = !certificate.has_resolvable_negative_curvature();
                        let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
                        let numerical_floor = certificate.numerical_floor;
                        certified_workspace = certificate.workspace;
                        if !cached_mode_acceptable {
                            log::warn!(
                                "[PIRLS/joint-Newton warm-start] refused cached same-rho inner mode: fresh returned-mode curvature lambda_min={:.6e} < -floor={:.6e}; retaining beta only as an uncertified solver seed",
                                minimum_whitened_eigenvalue,
                                numerical_floor,
                            );
                        }
                    }
                    Err(error) => {
                        cached_mode_acceptable = false;
                        certified_workspace = None;
                        log::warn!(
                            "[PIRLS/joint-Newton warm-start] refused cached same-rho inner mode because fresh returned-mode curvature could not be certified ({error}); retaining beta only as an uncertified solver seed"
                        );
                    }
                }
            }
            if cached_mode_acceptable {
                log::info!(
                    "[PIRLS/joint-Newton warm-start] reused cached same-rho inner mode | cycles={} product={product:?} logdet_h={:?} logdet_s={:?}",
                    cached.cycles,
                    cached.block_logdet_h,
                    cached.block_logdet_s,
                );
                return Ok(BlockwiseInnerResult {
                    block_states: states,
                    terminal_working_sets: cached.terminal_working_sets.clone(),
                    terminal_likelihood_score: cached.terminal_likelihood_score.clone(),
                    active_sets: normalize_active_sets(cached_active_sets),
                    log_likelihood: cached.log_likelihood,
                    penalty_value: cached.penalty_value,
                    cycles: cached.cycles,
                    converged: cached.converged,
                    terminal_convergence_state: None,
                    block_logdet_h: if product.requires_laplace_artifacts() {
                        cached.block_logdet_h
                    } else {
                        None
                    },
                    block_logdet_s: if product.requires_laplace_artifacts() {
                        cached.block_logdet_s
                    } else {
                        None
                    },
                    s_lambdas,
                    joint_workspace: certified_workspace,
                    kkt_residual: cached.kkt_residual.clone(),
                    active_constraints: cached.active_constraints.clone(),
                    // Equal to `cached.objective_state` by the guard above.
                    objective_state,
                });
            }
        }
        // Cold-start path: copy prior β where dimensions match
        // (best-effort; mismatched blocks keep the freshly-built
        // initial state).
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                let beta_projected =
                    family.post_update_block_beta(&states, b, &specs[b], beta_seed.clone())?;
                states[b].beta.assign(&beta_projected);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let load_joint_started = std::time::Instant::now();
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation begin use_joint_newton={} joint_workspace_requested={} (since inner-start={:.3}s)",
            use_joint_newton,
            joint_workspace_requested,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let (current_log_likelihood, mut cached_eval, cached_joint_gradient, cached_joint_workspace) =
        if use_joint_newton {
            let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
                family,
                specs,
                options,
                &states,
                joint_workspace_requested,
                None,
            )?;
            (log_likelihood, eval, gradient, workspace)
        } else {
            let eval = family.evaluate(&states)?;
            let log_likelihood = eval.log_likelihood;
            (log_likelihood, Some(eval), None, None)
        };
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation end elapsed={:.3}s log_likelihood={:.6e} has_gradient={} has_workspace={}",
            load_joint_started.elapsed().as_secs_f64(),
            current_log_likelihood,
            cached_joint_gradient.is_some(),
            cached_joint_workspace.is_some(),
        );
    }
    // Validate the one authoritative curvature source at the inner-solve
    // boundary. Workspace families must use that exact source here and in
    // cycle 0; asking `family.evaluate` for block Hessians would assemble the
    // same CTN rowwise-Kronecker Gram a second time at the same beta. Families
    // without a workspace retain the generic block-Hessian guard.
    let validate_started = std::time::Instant::now();
    let cached_joint_hessian_source = if joint_workspace_requested {
        let workspace = cached_joint_workspace.as_ref().ok_or_else(|| {
            "joint Newton requested an exact Hessian workspace, but gradient loading retained none"
                .to_string()
        })?;
        Some(
            exact_newton_joint_hessian_source_from_workspace(
                workspace,
                total_joint_p,
                MaterializationIntent::InnerSolve,
                "joint Newton inner prevalidation Hessian source",
            )?
            .ok_or_else(|| {
                "joint Newton exact Hessian workspace supplied no inner-solve curvature source"
                    .to_string()
            })?,
        )
    } else {
        // Gradient-override families (e.g. Gaussian/Binomial location-scale,
        // whose `exact_newton_joint_gradient_evaluation` serves the exact joint
        // score) return no cached evaluation. Materialize it once so the
        // non-workspace block-Hessian guard cannot be skipped (#2108 / #1820).
        if cached_eval.is_none() {
            cached_eval = Some(family.evaluate(&states)?);
        }
        if let Some(eval) = cached_eval.as_ref() {
            validate_block_hessians_finite(eval)?;
        }
        None
    };
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=validate_block_hessians_finite elapsed={:.3}s checked={}",
            validate_started.elapsed().as_secs_f64(),
            cached_eval.is_some() || cached_joint_hessian_source.is_some(),
        );
    }
    let penalty_started = std::time::Instant::now();
    let mut current_penalty = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=total_quadratic_penalty elapsed={:.3}s penalty={:.6e} (prelude_total={:.3}s)",
            penalty_started.elapsed().as_secs_f64(),
            current_penalty,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let mut lastobjective = -current_log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;
    // The decision variables of the most recent completed cycle. Whatever cycle
    // the loop exits on, this holds the numbers the convergence verdict was
    // taken against, so a non-convergence refusal can name them instead of
    // reporting only how many cycles ran.
    let mut terminal_convergence_state: Option<gam_problem::InnerConvergenceTerminalState> = None;
    // Pre-allocate per-block eta backup buffers to avoid O(n) allocation
    // per block per cycle in the backtracking line search.
    let mut eta_backups: Vec<Array1<f64>> =
        states.iter().map(|s| Array1::zeros(s.eta.len())).collect();

    // ── Joint Newton fast path ──
    //
    // When the family provides an exact joint Hessian (GAMLSS location-scale),
    // solve the full (p_mu + p_ls) × (p_mu + p_ls) system in one Newton step
    // per cycle instead of iterating between blocks. This converges quadratically
    // (5-10 steps) instead of linearly (20-100+ blockwise cycles).
    //
    // Generic block-diagonal surrogate families may still fall back to
    // blockwise iteration if the joint surrogate is unavailable. Families that
    // advertise a real coupled joint Hessian must not: the blockwise loop only
    // sees principal blocks, so it drops the cross-block curvature that makes
    // the joint problem well conditioned near saturated optima.

    // `last_residual_tol` mirrors the per-cycle KKT tolerance computed inside
    // the joint-Newton loop (`inner_tol · (1 + max(‖∇L‖∞, ‖Sβ‖∞))`). It must
    // live at function scope so both the post-converged exit block inside
    // `if use_joint_newton` AND the post-block-fit IFT residual builder
    // outside that branch can thread the same tolerance into the
    // `ProjectedKktResidual::with_metadata(...)` builder. Seed at `inner_tol`
    // so a path that skips the loop entirely (no joint-Newton, or zero
    // cycles) still records a finite, non-NaN tolerance on the residual
    // carrier rather than NaN.
    let last_residual_tol: f64 = inner_tol;

    if use_joint_newton {
        return fit_exact_joint(ExactJointFitContext {
            family,
            specs,
            block_log_lambdas,
            options,
            states,
            s_lambdas,
            ridge,
            joint_bundle,
            lastobjective,
            converged,
            cycles_done,
            terminal_convergence_state,
            inner_tol,
            inner_max_cycles,
            cached_active_sets,
            current_log_likelihood,
            cached_eval,
            cached_joint_gradient,
            cached_joint_workspace,
            cached_joint_hessian_source,
            objective_state,
            joint_workspace_requested,
            matrix_free_joint_requested,
            total_joint_n,
            prelude_log,
            inner_started: &inner_started,
            last_residual_tol,
            product,
        });
    }

    let mut cached_eval = match cached_eval {
        Some(eval) => eval,
        None => family.evaluate(&states)?,
    };
    lastobjective = -cached_eval.log_likelihood + current_penalty;

    // Divergence-detection state for the blockwise loop.
    //
    // Some family parameterizations (e.g. BernoulliMarginalSlopeFamily with
    // linkwiggle + scorewarp) carry a near-null direction in the joint
    // Hessian when the link-deviation basis's empirical anchor — fixed at
    // the rigid-pilot η₀ when the basis is constructed — drifts during
    // PIRLS as the location/spatial blocks update η₀. The Newton step
    // becomes dominated by that null direction and is clamped at
    // MAX_NEWTON_STEP every cycle while β grows linearly along it; the
    // log-likelihood stays frozen, only the penalty changes (slowly).
    // Without an early-exit the loop runs to inner_max_cycles producing
    // the same -loglik over and over, which at large scale (each cycle
    // ~0.5s) burns ~50s per ρ-cost call and stacks up to a 2400s timeout.
    //
    // Detect the pattern and bail with `converged = false` so the cost
    // call returns Err / +∞, BFGS κ-optim backs off the divergent ρ
    // region, and the outer loop progresses instead of grinding.

    // Per-block trust-region radius in the block's penalized-Hessian metric.
    // Updated each cycle by `update_joint_trust_region_radius` (the same
    // function the joint-Newton path uses) on a real model-vs-truth rho
    // computed from each block's penalized quadratic. Using the curvature
    // metric here avoids the same starvation mechanism fixed in the joint
    // path: one near-null coordinate in a block must not raw-rescale every
    // other coordinate in that block. The η-overflow safety half of the
    // previous static `MAX_NEWTON_STEP = 20.0` is owned by the family's
    // `max_feasible_step_size` barrier check, called by the line search below;
    // this variable handles only the algorithmic trust-region half. The
    // initial seed value is the family-declared safe step for a fresh fit; the
    // function then adapts it freely (clamped to [1e-12, 1e6] by the function
    // itself, same as the joint path).
    const BLOCK_NEWTON_STEP_INITIAL: f64 = 20.0;
    let mut block_max_step: Vec<f64> = vec![BLOCK_NEWTON_STEP_INITIAL; specs.len()];

    let mut prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
    // Frozen-loglik streak rides the shared window discipline
    // (loop_guard::FlatStreak, #968); the frozen-loglik predicate and the
    // clamped-step side condition below stay local — they are policy about
    // what counts as flat, which this loop rightly owns.
    let mut frozen_loglik_streak =
        gam_solve::loop_guard::FlatStreak::new(DIVERGENCE_FROZEN_LOGLIK_CYCLES);
    // Coordinate descent visits each block in turn, so `max_proposed_step`
    // (the per-cycle max across blocks) only fires the cap on cycles where
    // the divergent block is the active one. On a near-null direction this
    // produces an alternation pattern (e.g. cap, cap, small, cap, small,
    // cap, …) and a strict "consecutive cycles where step is clamped"
    // requirement resets the counter every time another block's smaller
    // step dominates the per-cycle maximum. The frozen-loglik signal,
    // however, is a property of the joint state — it stays true across
    // every cycle of the alternation. Track frozen-loglik consecutively
    // and require that `step_clamped` was observed AT LEAST ONCE inside
    // the frozen run (rather than EVERY cycle).
    let mut clamped_step_in_frozen_run: bool = false;
    const DIVERGENCE_FROZEN_LOGLIK_CYCLES: usize = 8;

    let is_dynamic = family.block_geometry_is_dynamic();
    for cycle in 0..inner_max_cycles {
        // Fires at the top of each blockwise coordinate cycle so we can count
        // iterations from CI logs when a benchmark hangs inside the first
        // outer-eval. Emitted at info-level: same rationale as the joint-Newton
        // sibling above — silent-grind diagnosis without debug logs.
        log::info!(
            "[PIRLS/blockwise coord] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
            cycle,
            inner_max_cycles,
            -cached_eval.log_likelihood,
            current_penalty,
            lastobjective,
        );
        let mut max_proposed_beta_step = 0.0_f64;
        let mut max_accepted_beta_step = 0.0_f64;
        let mut trust_boundary_hit_in_cycle = false;

        let mut objective_cycle_prev = lastobjective;
        // Reuse cached evaluation from end of previous cycle (or initial eval).
        // For dynamic families, the end-of-cycle evaluation is also reused here
        // instead of re-evaluating redundantly — the state hasn't changed since
        // the last cycle's final evaluate.
        let mut cycle_eval = std::mem::replace(
            &mut cached_eval,
            FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: Vec::new(),
            },
        );
        if cycle_eval.blockworking_sets.len() != specs.len() {
            return Err(CustomFamilyError::trial_point(format!(
                "family returned {} block working sets, expected {}",
                cycle_eval.blockworking_sets.len(),
                specs.len()
            )));
        }
        // Track whether any block was modified this cycle (for dynamic families,
        // we only need to re-evaluate before block b if a previous block changed).
        let mut any_block_modified = false;
        for b in 0..specs.len() {
            if is_dynamic && any_block_modified {
                // Only re-evaluate if a previous block in this cycle actually
                // modified coefficients. Skips the redundant evaluate for the
                // first block (b=0) since cached_eval is still valid.
                refresh_all_block_etas(family, specs, &mut states)?;
                cycle_eval = family.evaluate(&states)?;
                if cycle_eval.blockworking_sets.len() != specs.len() {
                    return Err(CustomFamilyError::trial_point(format!(
                        "family returned {} block working sets, expected {}",
                        cycle_eval.blockworking_sets.len(),
                        specs.len()
                    )));
                }
            }

            let spec = &specs[b];
            let work = &cycle_eval.blockworking_sets[b];
            let linear_constraints = family.block_linear_constraints(&states, b, spec)?;
            let s_lambda = &s_lambdas[b];
            let updater = work.updater();
            let update = updater.compute_update_step(&BlockUpdateContext {
                family,
                states: &states,
                spec,
                block_idx: b,
                s_lambda,
                options,
                linear_constraints: linear_constraints.as_ref(),
                cached_active_set: cached_active_sets[b].as_deref(),
            })?;
            if let Some(active_set) = update.active_set {
                cached_active_sets[b] = Some(active_set);
            }
            let beta_new_raw = update.beta_new_raw;
            let beta_new = family.post_update_block_beta(&states, b, spec, beta_new_raw.clone())?;
            reject_constrained_post_update_repair(
                b,
                spec,
                &beta_new_raw,
                &beta_new,
                linear_constraints.as_ref(),
            )?;
            let beta_old = states[b].beta.clone();
            let raw_delta = &beta_new - &beta_old;
            // Per-block trust-region radius in the block's local
            // penalized-Hessian metric. The cap is the current value of
            // `block_max_step[b]`, updated below via
            // `update_joint_trust_region_radius` once we know rho.
            let block_cap = block_max_step[b];
            let (delta, step_metric_norm) = truncate_block_step_to_metric_radius(
                spec,
                work,
                s_lambda,
                raw_delta,
                block_cap,
                ridge,
                options.ridge_policy,
            )?;
            let step_hit_trust_boundary =
                joint_block_step_hit_trust_boundary(step_metric_norm, block_cap);
            trust_boundary_hit_in_cycle |= step_hit_trust_boundary;
            // Capture the objective at the start of this block update so
            // we can compute the true `actual_reduction` once the line
            // search has finished. `objective_cycle_prev` is the running
            // total: it advances inside the line search whenever a trial
            // is accepted, so we must snapshot it here.
            let obj_before_block = objective_cycle_prev;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step_beta_inf = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_proposed_beta_step = max_proposed_beta_step.max(step_beta_inf);
            log::debug!(
                "[PIRLS/blockwise step] block={b} |delta|inf={step_beta_inf:.6e} \
                 metric_norm={step_metric_norm:.6e} cap={block_cap:.6e} \
                 hit_boundary={step_hit_trust_boundary} \
                 block_s_lambda_frob={:.6e} joint_bundle={} obj_before={obj_before_block:.9e}",
                s_lambda.iter().map(|v| v * v).sum::<f64>().sqrt(),
                joint_bundle.map(|bundle| bundle.specs().len()).unwrap_or(0),
            );
            if step_beta_inf <= inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            // Precompute X * delta once so line-search eta updates are O(n) not O(np).
            // Reuse pre-allocated eta backup to avoid O(n) allocation per block per cycle.
            let eta_checkpoint = BlockEtaCheckpoint::capture_reuse(&states[b], &mut eta_backups[b]);
            let x_delta = if !is_dynamic {
                Some(spec.solver_design().matrixvectormultiply(&delta))
            } else {
                None
            };
            let mut accepted = false;
            // Barrier-aware step ceiling: families with natural log-barrier
            // terms (e.g. log(h') in transformation-normal) report the maximum
            // feasible step fraction so the line search never evaluates the
            // likelihood outside its domain.
            let barrier_ceiling = family
                .max_feasible_step_size(&states, b, &delta)?
                .unwrap_or(1.0);
            // Reuse trial_beta_buf to avoid allocation per backtracking trial.
            let mut trial_beta_buf = beta_old.clone();
            let mut accepted_bt: usize = usize::MAX;
            for bt in 0..8 {
                let alpha = (0.5f64.powi(bt)).min(barrier_ceiling);
                trial_beta_buf.assign(&beta_old);
                trial_beta_buf.scaled_add(alpha, &delta);
                let trial_beta =
                    family.post_update_block_beta(&states, b, spec, trial_beta_buf.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    spec,
                    &trial_beta_buf,
                    &trial_beta,
                    linear_constraints.as_ref(),
                )?;
                states[b].beta = trial_beta;
                // Use precomputed X*delta when geometry is static and beta wasn't modified.
                if let Some(ref xd) = x_delta {
                    if states[b].beta == trial_beta_buf {
                        eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                    } else {
                        refresh_single_block_eta(family, specs, &mut states, b)?;
                    }
                } else {
                    refresh_single_block_eta(family, specs, &mut states, b)?;
                }
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                // The early exit certifies that the accept test below would
                // refuse the trial, so its slack is the accept test's own
                // round-off slack at the incumbent, not an absolute number
                // (the joint loop made the same change in `e2b49a23f`).
                let blockwise_slack =
                    joint_objective_roundoff_slack(objective_cycle_prev, objective_cycle_prev, 0.0);
                let line_search_options = coefficient_line_search_options(
                    options,
                    objective_cycle_prev - trial_penalty + blockwise_slack,
                );
                let trial_ll =
                    match family.log_likelihood_only_with_options(&states, &line_search_options) {
                        Ok(value) => value,
                        Err(reason) => {
                            log::debug!(
                                "[PIRLS/blockwise trial] block={b} bt={bt} alpha={alpha:.6e} \
                                 LIKELIHOOD REFUSED: {reason}"
                            );
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                            continue;
                        }
                    };
                let trialobjective = -trial_ll + trial_penalty;
                log::debug!(
                    "[PIRLS/blockwise trial] block={b} bt={bt} alpha={alpha:.6e} \
                     -trial_ll={:.9e} trial_penalty={:.9e} trialobjective={:.9e} \
                     prev={:.9e} margin={:.3e}",
                    -trial_ll,
                    trial_penalty,
                    trialobjective,
                    objective_cycle_prev,
                    objective_cycle_prev + blockwise_slack - trialobjective,
                );
                if trialobjective.is_finite()
                    && trialobjective
                        <= objective_cycle_prev
                            + joint_objective_roundoff_slack(
                                objective_cycle_prev,
                                trialobjective,
                                0.0,
                            )
                {
                    objective_cycle_prev = trialobjective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    accepted_bt = bt as usize;
                    break;
                }
            }
            // Trust-region update for this block, using the same
            // `update_joint_trust_region_radius` strategy the
            // joint-Newton path uses. Predicted reduction is computed
            // from the per-block penalized quadratic model:
            //
            //   Q(β + αδ) ≈ Q(β) − α·rhs·δ + 0.5·α²·δ·H_pen·δ
            //   predicted_reduction(α) = α·(rhs·δ) − 0.5·α²·(δ·H_pen·δ)
            //
            // where `rhs = score − S·β (− ridge·β)` is the penalized
            // gradient (in maximize-direction) and `H_pen = H + S
            // (+ ridge·I)` is the penalized observed information.
            // Actual reduction is the true penalized objective change
            // measured by the line search; rho = actual / predicted is
            // the standard model-vs-truth ratio that drives the same
            // 0.25 / 0.75 grow-shrink rules `update_joint_trust_region_radius`
            // already implements for the joint path.
            let alpha_accepted = if accepted {
                0.5_f64.powi(accepted_bt as i32)
            } else {
                0.0
            };
            let (rhs_block, hpen_delta_full): (Array1<f64>, Array1<f64>) = match work {
                BlockWorkingSet::ExactNewton { gradient, .. } => {
                    let mut rhs = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    // IRLS local-quadratic gradient and Hessian:
                    //   rhs = X^T W (z − Xβ) − Sβ
                    //   H_pen δ = X^T W X δ + Sδ
                    let solver_design = spec.solver_design();
                    let xb = solver_design.matrixvectormultiply(&beta_old);
                    let resid = working_response - &xb;
                    let w_resid = &resid * working_weights;
                    let mut rhs = solver_design.transpose_vector_multiply(&w_resid);
                    rhs -= &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
                BlockWorkingSet::NaturalDiagonal { score, .. } => {
                    let mut rhs = spec.solver_design().transpose_vector_multiply(score);
                    rhs -= &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
            };
            let rhs_dot_delta = rhs_block.dot(&delta);
            let delta_dot_hpen = delta.dot(&hpen_delta_full);
            let predicted_reduction = alpha_accepted * rhs_dot_delta
                - 0.5 * alpha_accepted * alpha_accepted * delta_dot_hpen;
            let actual_reduction = obj_before_block - objective_cycle_prev;
            let trust_update = update_joint_trust_region_radius(
                block_max_step[b],
                alpha_accepted * step_metric_norm,
                actual_reduction,
                predicted_reduction,
                obj_before_block,
                inner_tol * (1.0 + obj_before_block.abs()),
                // The blockwise path takes ONE step per block per cycle rather
                // than a backtracking ladder, so it has no shrink sequence to
                // read a resolution off (gam#2612). `0.0` means "nothing
                // measured", which leaves this site byte-identical — and with
                // nothing measured the residual flag cannot be consulted.
                0.0,
                false,
            );
            block_max_step[b] = trust_update.radius;
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
                if let BlockWorkingSet::ExactNewton { gradient, .. } = work {
                    let mut raw_descent = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        raw_descent -= &beta_old.mapv(|v| ridge * v);
                    }
                    let (descent_dir, descent_metric_norm) = truncate_block_step_to_metric_radius(
                        spec,
                        work,
                        s_lambda,
                        raw_descent,
                        block_cap,
                        ridge,
                        options.ridge_policy,
                    )?;
                    trust_boundary_hit_in_cycle |=
                        joint_block_step_hit_trust_boundary(descent_metric_norm, block_cap);
                    let dir_norm = descent_dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if dir_norm > inner_tol {
                        // Precompute X * descent_dir once for incremental eta updates.
                        let x_descent = if !is_dynamic {
                            Some(spec.solver_design().matrixvectormultiply(&descent_dir))
                        } else {
                            None
                        };
                        let descent_barrier_ceiling = family
                            .max_feasible_step_size(&states, b, &descent_dir)?
                            .unwrap_or(1.0);
                        for bt in 0..12 {
                            let alpha = (0.5f64.powi(bt)).min(descent_barrier_ceiling);
                            trial_beta_buf.assign(&beta_old);
                            trial_beta_buf.scaled_add(alpha, &descent_dir);
                            let trial_beta = family.post_update_block_beta(
                                &states,
                                b,
                                spec,
                                trial_beta_buf.clone(),
                            )?;
                            reject_constrained_post_update_repair(
                                b,
                                spec,
                                &trial_beta_buf,
                                &trial_beta,
                                linear_constraints.as_ref(),
                            )?;
                            states[b].beta = trial_beta;
                            if let Some(ref xd) = x_descent {
                                if states[b].beta == trial_beta_buf {
                                    eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                                } else {
                                    refresh_single_block_eta(family, specs, &mut states, b)?;
                                }
                            } else {
                                refresh_single_block_eta(family, specs, &mut states, b)?;
                            }
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let blockwise_slack = joint_objective_roundoff_slack(
                                objective_cycle_prev,
                                objective_cycle_prev,
                                0.0,
                            );
                            let line_search_options = coefficient_line_search_options(
                                options,
                                objective_cycle_prev - trial_penalty + blockwise_slack,
                            );
                            let trial_ll = match family
                                .log_likelihood_only_with_options(&states, &line_search_options)
                            {
                                Ok(value) => value,
                                Err(_) => {
                                    states[b].beta.assign(&beta_old);
                                    eta_checkpoint.restore_eta(&mut states[b]);
                                    continue;
                                }
                            };
                            let trialobjective = -trial_ll + trial_penalty;
                            if trialobjective.is_finite()
                                && trialobjective
                                    <= objective_cycle_prev
                                        + joint_objective_roundoff_slack(
                                            objective_cycle_prev,
                                            trialobjective,
                                            0.0,
                                        )
                            {
                                objective_cycle_prev = trialobjective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                        }
                    }
                }
            }
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
            } else {
                let accepted_step = states[b]
                    .beta
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(new, old)| (new - old).abs())
                    .fold(0.0_f64, f64::max);
                max_accepted_beta_step = max_accepted_beta_step.max(accepted_step);
                any_block_modified = true;
            }
            // Recycle the checkpoint's buffer back into the pre-allocated pool.
            eta_backups[b] = eta_checkpoint.into_buffer();
        }

        // For non-dynamic families, incremental eta updates within the block loop
        // maintain correct etas. Only refresh from scratch for dynamic-geometry families
        // where block interactions may require recomputation.
        if is_dynamic {
            refresh_all_block_etas(family, specs, &mut states)?;
        }
        cached_eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let objective = -cached_eval.log_likelihood + current_penalty;
        let objective_change = (objective - lastobjective).abs();
        lastobjective = objective;
        cycles_done = cycle + 1;

        // Divergence guard (mirrors the joint-Newton sibling, gam#554): a
        // non-finite objective / log-likelihood means a near-unidentified
        // penalized block has propagated NaN mass through the coordinate
        // descent. Every convergence and divergence-frozen exit below is a
        // finite `<=` comparison that NaN silently defeats, so without this
        // the loop grinds the full `inner_max_cycles` on every outer ρ-eval
        // and startup seed. Break unconverged so the outer optimizer rejects
        // this point immediately instead of burning the budget.
        if !objective.is_finite() || !cached_eval.log_likelihood.is_finite() {
            log::warn!(
                "[PIRLS/blockwise convergence] cycle {:>3} | divergence guard: non-finite inner state (objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                cycle,
                objective,
                -cached_eval.log_likelihood,
            );
            converged = false;
            break;
        }

        // Scale-aware tolerances — see the matching joint-Newton path
        // above for the rationale. At large scale absolute step/residual
        // tolerances against `inner_tol = 1e-6` keep this loop spinning
        // long after the objective has gone flat.
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + objective.abs());
        let residual_tol = objective_tol;
        // The premise this used to skip the measurement on is true and the
        // conclusion drawn from it was not (gam#2612).
        //
        // TRUE: for a single-block model the blockwise iteration IS the joint
        // iteration, so block-conditional convergence implies joint
        // convergence. The stall this guard was written against — the
        // block-conditional and joint gradient formulations disagreeing by
        // ~10× the tolerance and burning cycles on an already-converged
        // solution — is a MULTI-block phenomenon: with one block the two
        // formulations are the same function and cannot disagree.
        //
        // FALSE: that "block-conditional convergence implies joint
        // convergence" licenses ASSUMING joint stationarity. `specs.len() >= 2`
        // made this `true` unconditionally for one block, and the surrounding
        // test is `max_accepted_step <= tol && objective_change <= tol` — both
        // of which are EXACTLY ZERO when the line search accepts nothing. A
        // solve that proposed a step of `1.038e1` against a step tolerance of
        // `1.0e-11` and accepted none of it was certified as converged at cycle
        // 0, and its zero iterate was published as the mode. "Nothing moved"
        // and "nothing needed to move" are the same two numbers; only the
        // residual tells them apart.
        //
        // So measure it for one block too. The premise says the answer must
        // agree with the block-conditional verdict when that verdict is real,
        // which is exactly why measuring costs nothing here.
        let exact_joint_stationarity_ok = if has_joint_exacthessian {
            exact_newton_joint_stationarity_inf_norm(
                family,
                specs,
                &cached_eval,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                None,
            )?
            .map(|residual| residual <= residual_tol)
            .unwrap_or(true)
        } else {
            true
        };
        log::info!(
            "[PIRLS/blockwise convergence] cycle {:>3} | max_proposed_step={:.3e} (tol={:.3e}) | max_accepted_step={:.3e} | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e} | joint_stationarity_ok={}",
            cycle,
            max_proposed_beta_step,
            step_tol,
            max_accepted_beta_step,
            objective_change,
            objective_tol,
            beta_inf,
            exact_joint_stationarity_ok,
        );
        // Record the verdict's inputs here, ahead of EVERY exit this cycle can
        // take — the divergence early-exits below, the certified break, and
        // running out of `inner_max_cycles` — so whatever survives the loop
        // describes the cycle the loop actually left on rather than the one
        // before it.
        terminal_convergence_state = Some(gam_problem::InnerConvergenceTerminalState::Blockwise {
            cycle,
            max_accepted_step: max_accepted_beta_step,
            max_proposed_step: max_proposed_beta_step,
            step_tol,
            objective_change,
            objective_tol,
            joint_stationarity_ok: exact_joint_stationarity_ok,
        });

        // Divergence early-exit. See the rationale block at the top of
        // this loop. We treat "log-likelihood unchanged + Newton step
        // pinned at the trust-region cap" as a near-null direction
        // signature and break out unconverged once it persists for
        // DIVERGENCE_FROZEN_LOGLIK_CYCLES consecutive iterations. Tracking
        // log-likelihood (not objective) is essential: when the null mode
        // dominates, only the penalty drifts cycle-to-cycle, so
        // `objective_change` stays above tol while -loglik is genuinely
        // frozen.
        let loglik_change_for_divergence_check =
            (cached_eval.log_likelihood - prev_log_likelihood_for_divergence_check).abs();
        let loglik_frozen_tol_for_divergence_check =
            inner_tol * (1.0 + cached_eval.log_likelihood.abs());
        let step_clamped_for_divergence_check = trust_boundary_hit_in_cycle;
        let loglik_frozen =
            loglik_change_for_divergence_check <= loglik_frozen_tol_for_divergence_check;
        let frozen_verdict = frozen_loglik_streak.note(loglik_frozen);
        if loglik_frozen {
            if step_clamped_for_divergence_check {
                clamped_step_in_frozen_run = true;
            }
        } else {
            clamped_step_in_frozen_run = false;
        }
        prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
        if frozen_verdict == gam_solve::loop_guard::LoopVerdict::Plateaued
            && clamped_step_in_frozen_run
        {
            log::warn!(
                "[PIRLS/blockwise convergence] divergence early-exit at cycle {} | -loglik={:.6e} frozen for {} consecutive cycles | max_proposed_step={:.3e} (trust-boundary hit observed in frozen run) | step_tol={:.3e}; near-null Hessian direction detected — returning unconverged so the outer optimizer backs off this region instead of running to inner_max_cycles.",
                cycle,
                -cached_eval.log_likelihood,
                frozen_loglik_streak.streak(),
                max_proposed_beta_step,
                step_tol,
            );
            converged = false;
            break;
        }

        // NOTE: there is deliberately NO wall-clock-driven "adaptive
        // early-exit" here — the same discipline the joint-Newton sibling loop
        // documents above. A verdict that fires when a cycle's wall-clock falls
        // below a fraction of a running EMA is non-deterministic: under CPU
        // contention (a parallel sweep) the same fit accepts at a different
        // iterate than it does run alone, and it accepts iterates up to 10×
        // outside the real KKT/objective tolerance, biasing the REML/LAML
        // criterion the inner residual feeds. Convergence is certified ONLY by
        // the exact stationarity gate below.
        if max_accepted_beta_step <= step_tol && objective_change <= objective_tol {
            if exact_joint_stationarity_ok || max_proposed_beta_step <= step_tol {
                converged = true;
            }
            break;
        }
    }

    // ── Polishing joint Newton step ──
    //
    // For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
    // blockwise iteration can reach step_inf < inner_tol while the joint KKT
    // residual (||Sβ − grad_ℓ||_∞) remains at ~10× inner_tol. This is because
    // each block is solved conditionally on other blocks' current values —
    // block-conditional stationarity does not imply joint stationarity when
    // the likelihood couples blocks off-diagonally.
    //
    // Once blockwise has placed β near the true joint optimum, a single (or
    // a few) damped joint Newton steps can tighten the joint residual to the
    // floor set by β magnitudes. This polishing phase is essential for the
    // outer REML gradient formula (which assumes exact β̂ stationarity); a
    // non-converged β̂ produces large envelope-theorem violations in the
    // analytic outer gradient.
    if use_joint_newton && !converged {
        polish_joint_newton_step(
            family,
            specs,
            options,
            &s_lambdas,
            ridge,
            joint_bundle,
            inner_tol,
            &cached_active_sets,
            &mut states,
            &mut cached_eval,
            &mut current_penalty,
            &mut converged,
        )?;
    }

    assemble_inner_blockwise_result(
        family,
        specs,
        states,
        block_log_lambdas,
        options,
        s_lambdas,
        ridge,
        joint_bundle,
        cached_active_sets,
        &cached_eval,
        converged,
        cycles_done,
        last_residual_tol,
        terminal_convergence_state,
        has_joint_exacthessian,
        product,
    )
}

/// Polishing joint-Newton step for the blockwise fall-through path of
/// [`inner_blockwise_fit`].
///
/// For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
/// blockwise iteration can reach `step_inf < inner_tol` while the joint KKT
/// residual (`||Sβ − grad_ℓ||_∞`) remains at ~10× `inner_tol`. Once blockwise
/// has placed β near the joint optimum, a few damped joint-Newton steps tighten
/// the joint residual to the floor set by β magnitudes; this is essential for the
/// outer REML gradient formula (which assumes exact β̂ stationarity).
///
/// Behavior is identical to the inline loop it replaced: the `?`-propagation, the
/// per-iteration `break` exits (gradient/Hessian unavailable, non-finite delta,
/// solver failure, residual-tolerance reached, line-search failure) and the
/// inner backtracking-search `continue` are preserved verbatim. Mutates `states`,
/// `cached_eval`, `current_penalty`, and `converged` in place exactly as before.
pub(crate) fn polish_joint_newton_step<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    inner_tol: f64,
    cached_active_sets: &[Option<Vec<usize>>],
    states: &mut Vec<ParameterBlockState>,
    cached_eval: &mut FamilyEvaluation,
    current_penalty: &mut f64,
    converged: &mut bool,
) -> Result<(), CustomFamilyError> {
    let ranges_joint: Vec<(usize, usize)> = {
        let mut offset = 0;
        specs
            .iter()
            .map(|s| {
                let start = offset;
                offset += s.design.ncols();
                (start, offset)
            })
            .collect()
    };
    let total_p_joint: usize = ranges_joint.last().map_or(0, |r| r.1);
    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.accounts_for_objective()
    {
        ridge
    } else {
        0.0
    };
    let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;

    // Allow up to a few polishing steps. The blockwise endpoint is close
    // to optimum, so step sizes should be small and line search should
    // accept full steps quickly.
    const POLISH_MAX_ITER: usize = 16;
    for _polish_iter in 0..POLISH_MAX_ITER {
        // Re-evaluate at current β to get the joint gradient and Hessian.
        refresh_all_block_etas(family, specs, states)?;
        let eval_for_polish = family.evaluate(states)?;
        let grad_full =
            match exact_newton_joint_gradient_from_eval(&eval_for_polish, specs, states)? {
                Some(g) => g,
                None => break,
            };
        // Spec-aware joint Hessian: canonical coupled-curvature source
        // (see the joint-Newton availability gate). Families overriding
        // only `_with_specs` return `None` from the spec-less default.
        let h_joint_opt = family.exact_newton_joint_hessian_with_specs(states, specs)?;
        let Some(h_joint) = h_joint_opt else { break };
        let mut h_dense = match symmetrized_square_matrix(
            h_joint,
            total_p_joint,
            "joint polish Hessian shape mismatch",
        ) {
            Ok(matrix) => matrix,
            Err(_) => break,
        };
        let h_unpenalized_dense = h_dense.clone();
        add_joint_penalty_to_matrix(
            &mut h_dense,
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            joint_bundle,
        );
        let joint_polish_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
            family,
            &JointHessianSource::Dense(h_unpenalized_dense),
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            options.ridge_floor,
            joint_bundle,
        );
        if joint_polish_diagonal_ridge != trace_diagonal_ridge {
            for d in 0..h_dense.nrows() {
                h_dense[[d, d]] += joint_polish_diagonal_ridge - trace_diagonal_ridge;
            }
        }

        let mut beta_joint = Array1::<f64>::zeros(total_p_joint);
        for b in 0..specs.len() {
            let (start, end) = ranges_joint[b];
            beta_joint
                .slice_mut(ndarray::s![start..end])
                .assign(&states[b].beta);
        }
        let penalty_beta = apply_joint_block_penalty(
            &ranges_joint,
            s_lambdas,
            &beta_joint,
            joint_mode_diagonal_ridge,
            joint_bundle,
        );
        let rhs = &grad_full - &penalty_beta;

        // Respect constraints that block line search on the boundary.
        // Gauss-Seidel blockwise leaves the joint KKT residual at a floor
        // around |λ_k S_k β̂| for boundary-active components. The residual
        // magnitude on FREE components is a better measure of whether we
        // should keep polishing: if β_i is clipped at the boundary and
        // KKT multiplier μ_i > 0, then rhs[i] is the multiplier, not a
        // free-space gradient violation.
        let block_constraints_now = collect_block_linear_constraints(family, states, specs)?;
        let joint_constraints_now = assemble_joint_linear_constraints(
            &block_constraints_now,
            &ranges_joint,
            total_p_joint,
        )?;
        let mut active_mask: Vec<bool> = vec![false; total_p_joint];
        if let Some(ref constraints) = joint_constraints_now
            && let Ok(Some(bounds)) = extract_simple_lower_bounds(constraints, total_p_joint)
        {
            for (idx, (bound, beta_val)) in bounds
                .lower_bounds
                .iter()
                .zip(beta_joint.iter())
                .enumerate()
            {
                if *bound > f64::NEG_INFINITY && (*beta_val - *bound).abs() < 1e-12 {
                    active_mask[idx] = true;
                }
            }
        }
        let res_inf_free = rhs
            .iter()
            .zip(active_mask.iter())
            .filter(|(_, active)| !**active)
            .map(|(v, _)| v.abs())
            .fold(0.0_f64, f64::max);
        // Scale-aware residual tolerance — the joint stationarity
        // residual ‖∇ℓ − Sβ‖_∞ scales with |obj| (≈ O(n) at large-scale
        // scale), so the historical absolute `inner_tol = 1e-6` is
        // unachievable here even at the true minimum. Same rationale
        // as the joint-Newton convergence test above.
        let polish_obj = -cached_eval.log_likelihood + *current_penalty;
        let polish_residual_tol = inner_tol * (1.0 + polish_obj.abs());
        if res_inf_free <= polish_residual_tol {
            *converged = true;
            break;
        }

        // Solve constrained Newton system if simple bounds are present,
        // else unconstrained.
        let delta = if let Some(ref constraints) = joint_constraints_now {
            let warm = flatten_joint_active_set(cached_active_sets, &block_constraints_now);
            let lower_bounds_opt = extract_simple_lower_bounds(constraints, total_p_joint)
                .ok()
                .flatten();
            if let Some(bounds) = lower_bounds_opt.as_ref() {
                match solve_quadratic_with_simple_lower_bounds(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    bounds,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            } else {
                match gam_solve::active_set::solve_quadratic_with_constraint_set(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    constraints,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            }
        } else {
            let solver = gam_linalg::utils::StableSolver::new();
            let factor = match solver.factorize(&h_dense) {
                Ok(factor) => factor,
                Err(_) => break,
            };
            let mut direction = rhs.clone();
            let mut direction_matrix =
                gam_linalg::faer_ndarray::array1_to_col_matmut(&mut direction);
            factor.solve_in_place(direction_matrix.as_mut());
            if !direction.iter().all(|value| value.is_finite()) {
                break;
            }
            direction
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }
        // Keep polishing until the free-space joint residual is small; a
        // tiny delta alone is not a certificate of stationarity.
        // Damped line search with projection.
        let old_states: Vec<ParameterBlockState> = states.clone();
        let old_obj = -eval_for_polish.log_likelihood + *current_penalty;
        let mut accepted_polish = false;
        for bt in 0..10 {
            let alpha = 0.5f64.powi(bt);
            for b in 0..specs.len() {
                let (start, end) = ranges_joint[b];
                let mut trial_beta = old_states[b].beta.clone();
                trial_beta.scaled_add(alpha, &delta.slice(ndarray::s![start..end]));
                let projected =
                    family.post_update_block_beta(&old_states, b, &specs[b], trial_beta.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    &specs[b],
                    &trial_beta,
                    &projected,
                    block_constraints_now[b].as_ref(),
                )?;
                states[b].beta.assign(&projected);
            }
            refresh_all_block_etas(family, specs, states)?;
            let trial_ll = match family.log_likelihood_only(states) {
                Ok(v) => v,
                Err(_) => {
                    for (b, s) in old_states.iter().enumerate() {
                        states[b] = s.clone();
                    }
                    refresh_all_block_etas(family, specs, states)?;
                    continue;
                }
            };
            let trial_penalty = total_quadratic_penalty(
                states,
                s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let trial_obj = -trial_ll + trial_penalty;
            if trial_obj.is_finite() && trial_obj <= old_obj + 1e-12 {
                *current_penalty = trial_penalty;
                *cached_eval = family.evaluate(states)?;
                accepted_polish = true;
                break;
            }
        }
        if !accepted_polish {
            // Restore and stop polishing.
            for (b, s) in old_states.iter().enumerate() {
                states[b] = s.clone();
            }
            refresh_all_block_etas(family, specs, states)?;
            break;
        }
    }
    Ok(())
}

/// Final result assembly for the blockwise / polish fall-through path of
/// [`inner_blockwise_fit`]. Computes the penalty value, the (converged-only)
/// projected KKT residual for the IFT, the active-constraint block, and — only
/// for a Laplace-ready product — the block log-dets, then moves `states`,
/// `s_lambdas`, and `cached_active_sets` into the returned
/// [`BlockwiseInnerResult`]. Every unconstrained converged result with exact
/// joint curvature is re-certified at the coefficient vector being returned,
/// independently of which product was requested; this includes modes minted by
/// the blockwise fall-through and joint-polish paths.
fn assemble_inner_blockwise_result<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    mut states: Vec<ParameterBlockState>,
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    s_lambdas: Vec<Array2<f64>>,
    ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    cached_active_sets: Vec<Option<Vec<usize>>>,
    cached_eval: &FamilyEvaluation,
    converged: bool,
    cycles_done: usize,
    last_residual_tol: f64,
    terminal_convergence_state: Option<gam_problem::InnerConvergenceTerminalState>,
    exact_joint_curvature_available: bool,
    product: InnerFitProduct,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    let local_ranges = block_param_ranges(specs);
    let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
    let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
    let joint_constraints =
        assemble_joint_linear_constraints(&block_constraints, &local_ranges, local_total_p)?;
    let active_constraints = if joint_constraints.is_some() {
        // Full numerically-tight face, not only the QP-recorded rows — see
        // widen_active_sets_to_tight_face (gam#979).
        let tight_sets = crate::blockwise_solve::widen_active_sets_to_tight_face(
            &block_constraints,
            &states,
            &cached_active_sets,
        )?;
        assemble_active_constraint_block(
            &block_constraints,
            &tight_sets,
            &local_ranges,
            local_total_p,
        )
        .map(std::sync::Arc::new)
    } else {
        None
    };
    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.accounts_for_objective()
    {
        ridge
    } else {
        0.0
    };
    let mut certified_workspace = None;
    if converged && exact_joint_curvature_available {
        let certificate = exact_joint_mode_curvature_certificate(
            family,
            &states,
            specs,
            options,
            &local_ranges,
            &s_lambdas,
            joint_mode_diagonal_ridge,
            joint_bundle,
            local_total_p,
            active_constraints.as_deref(),
        )?;
        let has_negative_curvature = certificate.has_resolvable_negative_curvature();
        let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
        let numerical_floor = certificate.numerical_floor;
        certified_workspace = certificate.workspace;
        if has_negative_curvature {
            return Err(CustomFamilyError::trial_point(format!(
                "blockwise/joint-polish tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={:.6e} < -floor={:.6e}; an indefinite coefficient point cannot define a Laplace mode",
                minimum_whitened_eigenvalue, numerical_floor,
            )));
        }
        log::info!(
            "[PIRLS/blockwise mode certificate] returned beta certified from fresh exact curvature: lambda_min={:.6e}, floor={:.6e}",
            minimum_whitened_eigenvalue,
            numerical_floor,
        );
    }

    // Reuse cached evaluation from the last cycle's end (or the initial eval if 0 cycles ran).
    let penalty_value = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );

    let (block_logdet_h, block_logdet_s) = if converged && product.requires_laplace_artifacts() {
        let (h, s) = blockwise_logdet_terms_with_workspace(
            family,
            specs,
            &mut states,
            block_log_lambdas,
            options,
            certified_workspace.clone(),
            None,
            active_constraints.as_deref(),
        )?;
        (Some(h), Some(s))
    } else {
        (None, None)
    };
    let kkt_residual = if converged {
        match exact_newton_joint_gradient_from_eval(cached_eval, specs, &states)? {
            Some(gradient) => {
                let active_set_rows_total: usize = cached_active_sets
                    .iter()
                    .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
                    .sum();
                let free_rank_at_cert = local_total_p.saturating_sub(active_set_rows_total);
                exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
                    &gradient,
                    specs,
                    &states,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                    Some(cached_active_sets.as_slice()),
                    joint_penalty_stationarity_score(options, specs, &states).as_ref(),
                )?
                .map(|r| r.with_metadata(last_residual_tol, free_rank_at_cert))
            }
            None => None,
        }
    } else {
        // Inner did not converge; no caller should trust an IFT correction
        // at a non-KKT iterate.
        None
    };

    Ok(BlockwiseInnerResult {
        block_states: states,
        terminal_working_sets: Some(cached_eval.blockworking_sets.clone()),
        terminal_likelihood_score: None,
        active_sets: normalize_active_sets(cached_active_sets),
        log_likelihood: cached_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        terminal_convergence_state,
        block_logdet_h,
        block_logdet_s,
        s_lambdas,
        joint_workspace: certified_workspace,
        kkt_residual,
        active_constraints,
        objective_state: crate::assembly::InnerObjectiveState::new(
            family,
            block_log_lambdas,
            joint_bundle,
        ),
    })
}

/// Borrowed derivative provider for joint models that wraps closures with
/// non-`'static` lifetimes.
///
/// The closures borrow data from the calling stack frame (family, synced states,
/// specs), so we use borrowed closures with a non-`'static` lifetime.
/// Instead we borrow the closures and implement `HessianDerivativeProvider` directly.
///
/// # Sign convention
///
/// The unified evaluator passes `v_k = H⁻¹(A_k β̂)` to `hessian_derivative_correction`.
/// By the implicit function theorem, `dβ̂/dρ_k = −v_k`. The stored `compute_dh`
/// expects the actual perturbation direction `δβ`, so we negate `v_k` before calling it.
pub(crate) struct BorrowedJointDerivProvider<'a> {
    pub(crate) compute_dh: &'a DriftDerivFn<'a>,
    pub(crate) compute_dh_many: Option<&'a DriftDerivManyFn<'a>>,
    pub(crate) compute_d2h: &'a DriftSecondDerivFn<'a>,
    /// Optional batched second-derivative callback. The unified evaluator's
    /// outer-Hessian ρ-ρ pair loop precomputes all K(K+1)/2 (v_k, v_l, u_kl)
    /// triples and calls this once per outer Hessian assembly when set, so
    /// families that fuse the per-row D²H walk across pairs (e.g. survival
    /// marginal-slope which scans n rows once per outer eval) replace
    /// K(K+1)/2 separate row-walks with one. The default `None` falls back
    /// to the per-pair `compute_d2h` dispatch and preserves the historical
    /// dispatch cost.
    pub(crate) compute_d2h_many: Option<&'a DriftSecondDerivManyFn<'a>>,
    pub(crate) family_outer_hessian_operator: Option<Arc<dyn gam_problem::HessianOperator>>,
}
