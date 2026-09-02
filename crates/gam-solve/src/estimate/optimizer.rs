use super::*;
use crate::estimate::evaluation::{
    materialize_link_outer_hessian, sas_effective_epsilon, sas_effective_epsilon_second,
    sas_log_delta_edge_barriercostgrad, sas_log_delta_edge_barriercostgradhess,
    sas_log_deltaridgeweight,
};
use crate::estimate::edf_accounting::penalized_edf_bundle;
use crate::estimate::penalty::{REML_SEED_SCREENING_RHO_CAP, scaled_covariance};
use crate::estimate::prefit::{
    reject_prefit_binomial_separation, reject_prefit_unpenalized_rank_deficiency,
};
use crate::estimate::smoothing_correction::AUTO_CUBATURE_MAX_EIGENVECTORS;
use gam_linalg::matrix::FactorizedSystem;
use gam_linalg::utils::KahanSum;
use gam_problem::dispersion_cov::se_from_covariance;
use gam_problem::{OrderedRhoBounds, SeedConfig, SeedRiskProfile};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

fn certify_factorized_inference_solve(
    hessian: &gam_linalg::matrix::SymmetricMatrix,
    rhs: &Array2<f64>,
    solution: &Array2<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    let residual = hessian.dot_matrix(solution) - rhs;
    gam_linalg::utils::certify_linear_system_residual(
        hessian.nrows(),
        hessian.max_abs_entry(),
        rhs,
        solution,
        &residual,
        label,
    )
    .map_err(|error| {
        EstimationError::RemlOptimizationFailed(format!(
            "exact factorized inference solve did not certify: {error}"
        ))
    })?;
    Ok(())
}

fn certify_factorized_inference_vector_solve(
    hessian: &gam_linalg::matrix::SymmetricMatrix,
    rhs: &Array1<f64>,
    solution: &Array1<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    let rhs_matrix = rhs.view().insert_axis(Axis(1)).to_owned();
    let solution_matrix = solution.view().insert_axis(Axis(1)).to_owned();
    certify_factorized_inference_solve(hessian, &rhs_matrix, &solution_matrix, label)
}

/// Scale-free KKT residual for the Negative-Binomial conditional ML problem in
/// `tau = log(theta)`. The score is `d log L / d theta`; therefore the
/// minimization gradient in `tau` is `-theta * score`. At either admissible
/// theta boundary, the outward component is a valid KKT multiplier and is
/// projected away. Interior residuals are normalized by the observed
/// log-theta curvature, so this is the Newton displacement still required for
/// theta stationarity rather than an arbitrary percent drift.
fn negbin_theta_stationarity_residual(theta: f64, score: f64, info: f64) -> f64 {
    if !theta.is_finite() || theta <= 0.0 || !score.is_finite() || !info.is_finite() {
        return f64::INFINITY;
    }
    let active_margin = f64::EPSILON.sqrt() * theta.max(1.0);
    let at_lower = theta <= pirls::NEGBIN_THETA_MIN + active_margin;
    let at_upper = theta >= pirls::NEGBIN_THETA_MAX - active_margin;
    // For minimizing -log L: lower-bound KKT requires score <= 0; upper-bound
    // KKT requires score >= 0. Those are exact one-sided optima.
    if (at_lower && score <= 0.0) || (at_upper && score >= 0.0) {
        return 0.0;
    }
    let log_theta_gradient = -theta * score;
    let log_theta_curvature = theta * theta * info - theta * score;
    if !log_theta_curvature.is_finite() || log_theta_curvature <= 0.0 {
        return f64::INFINITY;
    }
    // Both numerator and curvature scale linearly with case weights, so their
    // ratio is invariant to objective rescaling. An absolute denominator floor
    // would instead certify flat theta coordinates whenever raw weights happen
    // to be small.
    (log_theta_gradient / log_theta_curvature).abs()
}

/// Whether the point a fit is about to ship IS the point the outer certificate
/// was minted at, compared over EVERY optimized coordinate.
///
/// The outer optimizer for a flexible-link fit searches the joint coordinate
/// `theta = [rho (k entries), link-shape coordinates]`, so `certified` is that
/// whole joint vector while the shipped `rho` is only its leading block — the
/// link coordinates are shipped separately, inside the link state. Comparing
/// the two vectors directly therefore compares a `K`-vector against a
/// `K + link_dim` one and can never agree (#2727): it refused fits that had
/// converged, whose certificate certified, at `|Pg|` as low as `1.446e-6`.
///
/// Comparing only the rho prefix would turn that over-strict gate into an
/// under-strict one — it would stop checking the link coordinates entirely,
/// and those are exactly the coordinates the flexible-link lane exists to
/// optimize. So the shipped point is reassembled in the optimizer's OWN
/// coordinate system and compared whole. It has to be the raw `theta`
/// coordinates rather than the shipped link state, because the state stores
/// values that have been through the smooth-bound maps
/// (`sas_effective_epsilon`) while the certificate holds the pre-image.
///
/// The comparison stays BITWISE for the reason the rho-only one did: point
/// identity is decided exactly by bit equality, and re-judging a gradient here
/// would refuse honest noise-band certificates with coin-flip probability.
///
/// Reassembling one vector only to compare it against the vector it was sliced
/// from looks circular, and is not: the check is TEMPORAL. `final_rho`,
/// `final_link_coords` and `outer_result` are `let mut` bindings reassigned
/// inside the alternation `loop`, so this asks whether the point being shipped
/// at the END is still the certificate being held at the END. Seeds and
/// nuisance refinements may initialize work between those two moments; they
/// must never promote a different point under the old certificate. Comparing
/// only the rho block would leave the link coordinates free to move across
/// exactly that window.
fn shipped_joint_point_is_certified(
    rho: &Array1<f64>,
    link_coords: &Array1<f64>,
    certified: &Array1<f64>,
) -> bool {
    if certified.len() != rho.len() + link_coords.len() {
        return false;
    }
    rho.iter()
        .chain(link_coords.iter())
        .zip(certified.iter())
        .all(|(shipped, certified)| shipped.to_bits() == certified.to_bits())
}

#[derive(Clone)]
struct NegbinJointCheckpoint {
    merit: f64,
    theta: f64,
    rho: Array1<f64>,
    rho_residual: f64,
    rho_bound: f64,
    theta_residual: f64,
    theta_bound: f64,
}

/// Reserve the complete peak live set of the optional dense inference path.
///
/// The count is assembled from named algorithmic owners rather than a
/// dimension cliff: ten square matrices can survive into/alongside the fit
/// payload, six are base factorization/GEMM workspaces, and eight belong to the
/// first-order smoothing correction. Cubature can retain one inverse Hessian
/// for each positive/negative sigma point while every concurrently evaluated
/// point holds its Hessian and inverse workspace. Charging the whole set
/// atomically prevents several individually acceptable p×p allocations from
/// jointly exceeding the process-wide memory ledger.
fn reserve_dense_covariance_bundle(p: usize) -> Option<gam_runtime::resource::MemoryReservation> {
    const STORED_SQUARE_MATRICES: usize = 10;
    const BASE_FACTORIZATION_AND_GEMM_WORKSPACES: usize = 6;
    const FIRST_ORDER_SMOOTHING_WORKSPACES: usize = 8;
    const CUBATURE_SIGMA_POINTS: usize = 2 * AUTO_CUBATURE_MAX_EIGENVECTORS;
    const RETAINED_CUBATURE_INVERSES: usize = CUBATURE_SIGMA_POINTS;
    const IN_FLIGHT_CUBATURE_HESSIAN_AND_INVERSE: usize = 2 * CUBATURE_SIGMA_POINTS;
    /// A constrained fit assembles its truncated covariance as a sum of Grams
    /// (`ConstrainedPosteriorCorrection::truncated_covariance_psd`, #2705 group
    /// A), which holds three `p × p` blocks live at once: the Cholesky factor of
    /// `Σ`, the projected factor `P L`, and the Gram it accumulates into. Plus
    /// the untruncated conditional covariance the marginal composition keeps so
    /// the corrected estimand can be built from `Vb`, not from `Σ_π`.
    ///
    /// Priced here rather than left to slack, because the whole point of this
    /// reservation is that several individually acceptable `p × p` allocations
    /// must not jointly exceed the ledger — and #2724 is on record for what an
    /// allocating route costs when the pricing does not follow it.
    const CONSTRAINED_TRUNCATION_WORKSPACES: usize = 4;
    const PEAK_SQUARE_MATRIX_EQUIVALENTS: usize = STORED_SQUARE_MATRICES
        + BASE_FACTORIZATION_AND_GEMM_WORKSPACES
        + FIRST_ORDER_SMOOTHING_WORKSPACES
        + CONSTRAINED_TRUNCATION_WORKSPACES
        + RETAINED_CUBATURE_INVERSES
        + IN_FLIGHT_CUBATURE_HESSIAN_AND_INVERSE;

    let policy = gam_runtime::resource::ResourcePolicy::for_problem(
        gam_runtime::resource::ProblemHints::default(),
    );
    if !policy.material_policy().allow_operator_materialization {
        return None;
    }
    match gam_runtime::resource::MemoryGovernor::global().try_reserve_dense_f64_copies(
        p,
        p,
        PEAK_SQUARE_MATRIX_EQUIVALENTS,
        "standard GAM dense covariance/influence bundle",
    ) {
        Ok(reservation) => Some(reservation),
        Err(error) => {
            log::info!(
                "Dense covariance/influence bundle not reserved; using factorized inference: {error}"
            );
            None
        }
    }
}

/// Truncate the ρ-MARGINAL posterior covariance to the fit's feasible set,
/// in place (#2705 group A).
///
/// `covariance` arrives as `Vp = Vb + J·V_ρ·Jᵀ` built from the UNTRUNCATED
/// conditional `Vb`, and leaves as the covariance of `N(β_unc, Vp)` restricted
/// to `{β : Aβ ≥ b}`. The truncation is rebuilt AT `Vp` — its own lift
/// `G_p = Vp·Aᵀ·W_p⁻¹` and its own orthant moments at `W_p = A·Vp·Aᵀ` — because
/// a lift derived from a different covariance is not a projector for this one,
/// and subtracting it is not a truncation of anything.
///
/// A geometry whose moments were DECLINED never truncated the conditional
/// covariance either, so there is nothing here to keep consistent with and the
/// marginal is published untruncated, exactly as the conditional one is.
///
/// A MOMENT failure declines rather than propagating. The corrected covariance
/// is a refinement of an already-published conditional one, and #2601 is on
/// record for what happens when a failure to refine the uncertainty is allowed
/// to destroy a converged point estimate; the honest degradation is the typed
/// absence every consumer of `beta_covariance_corrected` already handles. A
/// STRUCTURAL failure — a geometry whose constraint width disagrees with the
/// covariance it is supposed to constrain — is still fatal, because that is a
/// wiring defect and no absence describes it.
pub(crate) fn apply_marginal_constraint_truncation(
    geometry: &crate::constrained_posterior::ConstrainedPosteriorGeometry,
    covariance: &mut Array2<f64>,
) -> Result<Result<(), String>, EstimationError> {
    if geometry.decline().is_some() {
        return Ok(Ok(()));
    }
    let p = covariance.nrows();
    if geometry.constraints.a.ncols() != p {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "constrained posterior geometry has {} constraint columns against a {p}x{p} \
             corrected covariance",
            geometry.constraints.a.ncols(),
        )));
    }
    let center = match geometry.unconstrained_center() {
        Ok(center) if center.len() == p => center,
        Ok(center) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "constrained posterior geometry carries a length-{} centre against a {p}x{p} \
                 corrected covariance",
                center.len(),
            )));
        }
        Err(reason) => return Ok(Err(reason)),
    };
    let marginal_correction =
        match crate::constrained_posterior::constrained_posterior_correction_from_covariance(
            covariance,
            center,
            &geometry.constraints,
        ) {
            Ok(correction) => correction,
            Err(reason) => return Ok(Err(reason)),
        };
    if let Some(correction) = marginal_correction {
        // Same positive-semidefinite assembly the conditional covariance uses:
        // the marginal is read for standard errors too, and a pinned coordinate
        // cancels there for exactly the same reason.
        match correction.truncated_covariance_psd(covariance, &geometry.constraints) {
            Ok(truncated) => *covariance = truncated,
            Err(reason) => return Ok(Err(reason)),
        }
    }
    Ok(Ok(()))
}

/// Reserve the square matrices that remain live even when inference stays
/// factorized: the two PIRLS Hessian surfaces, the fitted reparameterization,
/// its exported copy, the reusable factor, the exported original-basis
/// precision, and the transformed penalty surface retained by the fit.
fn reserve_factorized_inference_state(
    p: usize,
) -> Option<gam_runtime::resource::MemoryReservation> {
    const RETAINED_FACTOR_AND_PRECISION_MATRICES: usize = 7;
    let policy = gam_runtime::resource::ResourcePolicy::for_problem(
        gam_runtime::resource::ProblemHints::default(),
    );
    if !policy.material_policy().allow_operator_materialization {
        return None;
    }
    match gam_runtime::resource::MemoryGovernor::global().try_reserve_dense_f64_copies(
        p,
        p,
        RETAINED_FACTOR_AND_PRECISION_MATRICES,
        "standard GAM factorized inference state",
    ) {
        Ok(reservation) => Some(reservation),
        Err(error) => {
            log::info!("Factorized inference state could not be fully reserved: {error}");
            None
        }
    }
}

/// Same as `optimize_external_design`, but allows heuristic λ warm-start seeds
/// for the outer smoothing search.
pub fn optimize_external_designwith_heuristic_lambdas<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    heuristic_lambdas: Option<&[f64]>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list
        .into_iter()
        .map(PenaltySpec::from_blockwise)
        .collect();
    optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y,
        w,
        x,
        offset,
        specs,
        heuristic_lambdas,
        None,
        opts,
    )
}

pub(crate) fn external_reml_seed_config(k: usize, link: LinkFunction) -> SeedConfig {
    let gaussian = matches!(link, LinkFunction::Identity);
    if gaussian {
        // Profiled Gaussian REML already constructs and scores two
        // data-derived starts below: the commensurate-curvature `initial.sp`
        // point and the certified summed-penalty diagonal profile.  Sending
        // their winner into the generic generated-seed lattice repeats the
        // same basin decision with arbitrary global shifts and an absolute
        // `rho=8` probe.  Besides violating SPEC's no-grid-search contract,
        // that screen dominates small fits: a converged zero-iteration
        // `y ~ s(x)` fit paid 68 outer cost evaluations / 104 inner solves,
        // while a ten-penalty saturated fold paid 415 inner solves.
        //
        // The analytic candidates are evaluated against the true coupled REML
        // criterion and adopted only on strict improvement, so they retain the
        // #1074 over-smoothing escape without a second heuristic search.  A
        // single generated seed is still required when neither analytic point
        // beats the invariant neutral anchor.  The ordinary optimizer and its
        // analytic terminal certificate remain mandatory either way.
        return SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: 1,
            // The generic lattice remains disabled. The three budget slots are
            // for the unique analytic candidates assembled below: base,
            // initial.sp, and the summed-penalty diagonal restriction.
            seed_budget: 3,
            risk_profile: SeedRiskProfile::Gaussian,
            screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
            num_auxiliary_trailing: 0,
            over_smoothing_probe_rho: None,
        };
    }
    if k >= REML_SEED_SCREENING_RHO_CAP {
        return SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: 2,
            seed_budget: 2,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
            num_auxiliary_trailing: 0,
            over_smoothing_probe_rho: None,
        };
    }
    SeedConfig {
        bounds: (-12.0, 12.0),
        max_seeds: if k <= 4 {
            6
        } else if k <= 12 {
            8
        } else {
            10
        },
        seed_budget: 2,
        risk_profile: SeedRiskProfile::GeneralizedLinear,
        screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
        num_auxiliary_trailing: 0,
        over_smoothing_probe_rho: None,
    }
}

pub(crate) fn standard_reml_search_prefers_gradient_only(link: LinkFunction) -> bool {
    // The optimize-3 / certify-4 split exists to keep the generic
    // non-Gaussian family derivative tower through order four out of the
    // smoothing-parameter search (#2359).  A profiled Gaussian identity
    // likelihood is quadratic in eta: its family derivatives above order two
    // vanish, so reserving the already-available exact outer Hessian for mint
    // buys nothing.  Worse, it routes a small exact-curvature problem through
    // first-order BFGS: the saturated wine_gamair fold took 86 accepted
    // iterations, then failed the first analytic stationarity audit.
    //
    // Let the capability planner choose analytic-Hessian ARC for that exact
    // quadratic objective.  Every non-identity family retains the order-three
    // search ceiling and pays order four only at mint.
    !matches!(link, LinkFunction::Identity)
}

fn reml_inner_progress_feedback(
    state: &crate::estimate::reml::RemlState<'_>,
) -> crate::rho_optimizer::InnerProgressFeedback {
    crate::rho_optimizer::InnerProgressFeedback {
        cap: Arc::clone(&state.outer_inner_cap),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::clone(&state.last_inner_iters),
        last_converged: Arc::clone(&state.last_inner_converged),
        ift_residual: Arc::clone(&state.last_ift_prediction_residual),
        accept_rho: Arc::clone(&state.last_pirls_accept_rho),
        // The standard REML path does not consume the cold-reeval pulse
        // (#2349); give it an inert, unshared flag so the guard's writes go
        // nowhere and behavior is unchanged.
        force_cold: Arc::new(AtomicBool::new(false)),
    }
}

fn with_reml_beta_seed_hook<'state, 'data>() -> impl FnMut(
    &mut &'state mut crate::estimate::reml::RemlState<'data>,
    &Array1<f64>,
) -> Result<
    crate::rho_optimizer::SeedOutcome,
    EstimationError,
> {
    |state, beta| {
        // The REML state stores β as a starting-iterate HINT and validates
        // its width against the design (`self.p`) at store time, silently
        // dropping a mismatched or non-finite hint rather than faulting
        // (see `setwarm_start_original_beta`). A wrong-length seed is
        // therefore never an error: a row-relaxed cross-fold prefix seed
        // degrades to a ρ-only resume, exactly the desired warm-start
        // behaviour. The slot's post-call state (the supplied β if it fit,
        // else the prior state) is what the next eval warm-starts from, so
        // `Installed` is the correct contract reply.
        state.setwarm_start_original_beta(Some(beta.view()));
        Ok(crate::rho_optimizer::SeedOutcome::Installed)
    }
}

/// The weighted-mean response level an unpenalized intercept would absorb, used
/// to center the response during outer REML λ-selection (issue #1000).
///
/// For an identity-link Gaussian fit, adding a constant to the response only
/// shifts the intercept, so λ̂ and the smooth shape must be invariant to the
/// response mean. The outer score/gradient nonetheless accumulate
/// `yᵀy`-magnitude sufficient statistics, so a large response mean costs
/// precision and drifts λ̂. Returns `Some(m)` with
/// `m = Σ wᵢ (yᵢ − offsetᵢ) / Σ wᵢ` — the constant a pure offset relabeling
/// moves into the intercept — so the caller can subtract it and keep the working
/// response `O(σ)` regardless of the mean.
///
/// Returns `None` (do not center, exact previous behaviour) unless the fit is
/// identity-link Gaussian and carries an unpenalized intercept column to absorb
/// the shift, and has no linear constraints that could pin the intercept. A zero
/// or non-finite mean also returns `None` — there is nothing to gain.
///
/// # It is a CORRECTNESS requirement, not only a precision one (#2671)
///
/// `PirlsPenalty` charges `FIXED_STABILIZATION_RIDGE * ||beta||^2` against a
/// target that is `Array1::zeros(p)` at every construction site, so the outer
/// criterion is a function of WHERE THE ORIGIN OF `y` SITS: shifting `y` by `m`
/// moves the intercept by `m` and moves the criterion by
/// `(n/2)/D_p * delta * ((beta0 + m)^2 - beta0^2)`. Centering here pins that
/// origin at the weighted response mean, which is what makes λ̂ invariant to a
/// constant added to the response — which it must be for an identity-link
/// Gaussian fit with an estimated intercept.
///
/// MEASURED at `517b6303f` on `mk_1d(15, t^2, 0.05, 7)`, `y ~ matern(x,nu=5/2)`,
/// three arms of one run: the route that DOES center moved `4.085e-14` under a
/// `+10` shift; the route that did not moved `5.047e-5` — a separation of
/// `1.24e9`, and the un-centered route's fit went from ACCEPTED (pre-centered
/// response) to REFUSED (`+10`). Any outer λ-search over this family must
/// therefore condition through this gate and
/// [`conditioned_outer_response`]; a route that skips it selects λ̂/ψ̂ from the
/// user's choice of response units.
pub(crate) fn gaussian_identity_response_center(
    cfg: &RemlConfig,
    conditioning: &ParametricColumnConditioning,
    has_linear_constraints: bool,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Option<f64> {
    if has_linear_constraints
        || conditioning.intercept_idx.is_none()
        || !matches!(cfg.likelihood.spec.response, ResponseFamily::Gaussian)
        || !matches!(cfg.link_function(), LinkFunction::Identity)
    {
        return None;
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted = KahanSum::default();
    for ((&yi, &wi), &oi) in y.iter().zip(w.iter()).zip(offset.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            weighted.add(wi * (yi - oi));
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let m = weighted.sum() / weight_sum;
    (m.is_finite() && m != 0.0).then_some(m)
}

/// The multiplicative scale an identity-link Gaussian outer REML λ-search should
/// divide the (already centered) response by so its magnitude is `O(1)` for the
/// duration of the search (issue #1127).
///
/// Replacing the response `y` by `a·y` (`a > 0`) for an identity-link Gaussian
/// fit must rescale the entire fit by `a` and leave `λ̂` / EDF unchanged: the
/// penalized normal equations are exactly linear in `y`, so `β̂(a·y)=a·β̂(y)`
/// at any fixed `λ`, and the profiled REML criterion is `a`-invariant up to the
/// additive constant `−(n−p)·ln a` (the dispersion `σ̂²` absorbs the `a²`).
/// Numerically, though, the outer λ-selection's convergence band is keyed to an
/// *absolute* objective scale (the inner-solve `objective_scale.max(1.0)` floor
/// and the outer `1e-6` gradient floor): when the whole Gaussian objective is
/// `O(a²) ≪ 1` those floors swamp the real signal and the optimizer declares
/// premature convergence at an over-smoothed `λ` — silently over-smoothing
/// small-magnitude responses (strains, volts, mole fractions, returns;
/// `a ≈ 1e-6`). Normalizing the working response to `O(1)` makes the absolute
/// floors track the true signal, restoring scale equivariance.
///
/// Returns `Some(s)` with `s = √(Σ wᵢ (yᵢ − mean)² / Σ wᵢ)` — the weighted RMS
/// of the centered response — so the caller can divide by it and keep the outer
/// working response `O(1)` regardless of magnitude. The same gate as
/// [`gaussian_identity_response_center`] applies (identity-link Gaussian with an
/// unpenalized intercept and no linear constraints); a non-finite, zero, or
/// already-`O(1)` RMS returns `None` (do not scale, exact previous behaviour) —
/// scaling near unity buys nothing and only risks a needless allocation.
pub(crate) fn gaussian_identity_response_scale(
    cfg: &RemlConfig,
    conditioning: &ParametricColumnConditioning,
    has_linear_constraints: bool,
    center: f64,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Option<f64> {
    if has_linear_constraints
        || conditioning.intercept_idx.is_none()
        || !matches!(cfg.likelihood.spec.response, ResponseFamily::Gaussian)
        || !matches!(cfg.link_function(), LinkFunction::Identity)
    {
        return None;
    }
    // A multiplicative response rescale `y → y/s` must be matched by `η → η/s`
    // for the residual to scale cleanly. The intercept and smooth coefficients
    // scale freely, but a *fixed* offset column does not — scaling the working
    // response while leaving the offset on its original scale would change the
    // residual geometry, not just its magnitude. The offset is shared verbatim
    // into the outer state and reused by the accept-fit, so rather than thread a
    // separately scaled copy everywhere, restrict the (rare) offset case to the
    // exact previous path: only normalize when there is no nonzero offset.
    if offset.iter().any(|&o| o != 0.0) {
        return None;
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted_sq = KahanSum::default();
    for ((&yi, &wi), &oi) in y.iter().zip(w.iter()).zip(offset.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            let centered = (yi - oi) - center;
            weighted_sq.add(wi * centered * centered);
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let rms = (weighted_sq.sum() / weight_sum).sqrt();
    // Only normalize when the magnitude is far enough from `O(1)` to matter; a
    // factor within ~one order of magnitude of unity cannot push the objective
    // through the absolute floors, so leave the exact previous path untouched.
    (rms.is_finite() && rms > 0.0 && !(0.1..=10.0).contains(&rms)).then_some(rms)
}

/// Apply the outer-λ-search response conditioning `(y − center)/scale`.
///
/// The ONLY place this arithmetic is written. Both routes that run an outer
/// λ-search over an identity-link Gaussian response must condition through this
/// function, or they minimize two different penalized problems and their
/// criteria are not comparable — see the module note on
/// [`gaussian_identity_response_center`] and #2671. `(None, None)` returns
/// `None` so the caller keeps borrowing the original response with no
/// allocation and no behavioural change.
pub(crate) fn conditioned_outer_response(
    center: Option<f64>,
    scale: Option<f64>,
    y: ArrayView1<'_, f64>,
) -> Option<Array1<f64>> {
    match (center, scale) {
        (None, None) => None,
        (center, scale) => {
            let c = center.unwrap_or(0.0);
            let s = scale.unwrap_or(1.0);
            Some(y.mapv(|value| (value - c) / s))
        }
    }
}

/// Pin the λ-search nuisance freeze to a canonical, cache-independent anchor
/// (#2363).
///
/// The λ-search optimizes `F(ρ) = REML(ρ, ψ)` with the estimated nuisance ψ
/// (Gamma shape, Tweedie φ, Beta precision) held FIXED across ρ. Without that
/// freeze ψ is re-profiled from every trial's warm-start η, the analytic outer
/// gradient — which holds ψ fixed — can never match the cost's ψ(ρ) motion, the
/// projected gradient floors above tolerance and the search stalls or rails
/// (#1074 / #1477 / #2369). The freeze is therefore load-bearing; what was not
/// load-bearing, and was wrong, is WHERE ψ got captured.
///
/// Until this call existed, ψ was captured opportunistically at the first
/// non-screening solve that happened to converge — and the persistent warm-start
/// cache decides which solve that is. It donates `initial_rho`, which
/// `run_outer_with_plan` inserts as seed 0, and it donates the warm β that
/// decides whether the pre-search reference solve at ρ = 0 converges at all. So
/// a cold machine and a warm machine froze ψ at different fits, i.e. they
/// minimized DIFFERENT criteria and legitimately landed at different optima:
/// measured on the #2363 fixture, the same Beta fit reported REML −6.382e2 cold
/// and −7.408e2 warm. That is the invariant violation at its root — the
/// objective was a function of the search path, not of the problem.
///
/// The repair is to define ψ, once, by a computation that depends on nothing but
/// `(data, model spec)`: solve P-IRLS at the symmetric reference ρ = 0 (every
/// λ = 1 — the same order-invariant reference point `canonical_rho_keys` uses to
/// label ρ-coordinates) on a state that has no warm start attached and no
/// persistent session open, and let the ordinary capture path freeze ψ at THAT
/// solve's converged η. If the reference point's inner solve does not converge,
/// the deterministic seed candidates are walked in their generated order, so the
/// anchor stays a pure function of the problem in every branch.
///
/// The warm-start slots are emptied on the way IN, so the anchor solve cannot
/// inherit a caller-supplied β. They are deliberately NOT emptied on the way out:
/// the β the anchor leaves behind is itself a function of (data, model spec), so
/// letting the search start from it keeps both a cold and a warm run entering the
/// search from the same predictor — and it is the same β the ρ = 0 reference solve
/// used to leave there before this function existed, so the cold path is
/// unchanged. It does mean `load_persistent_warm_start_once` finds an occupied
/// slot and skips its restore, which is the point: that restore would hand the
/// warm run a different starting β and a different set of adaptive LM/tolerance
/// signals. Nothing is lost — the cached β still reaches the inner solve at the
/// cached ρ through `OuterConfig::initial_inner_seed`, which installs it at
/// exactly the outer coordinate that owns it.
///
/// The on-disk session must not be active while this computation runs (the inner
/// solve would reload the cached β mid-anchor), so a direct call on an attached
/// state is refused. The design-moving evaluator satisfies this precondition
/// with `RemlState::without_persistent_warm_start_store`; the standard evaluator
/// calls before attaching persistence. The ρ-keyed eval/P-IRLS memos are kept:
/// they cache a deterministic computation at a ρ the caller is about to evaluate
/// again.
///
/// Negative-Binomial θ is deliberately not handled here. It is seeded from the
/// resolved family spec before the search and then driven to a certified joint
/// (θ, ρ) fixed point by the alternation loop below, which is already a function
/// of the data alone.
pub(crate) fn freeze_lambda_search_nuisance_at_canonical_anchor(
    reml_state: &RemlState<'_>,
    resolved_likelihood_scale: &gam_problem::ResolvedLikelihoodScale,
    k: usize,
    heuristic_lambdas: Option<&[f64]>,
    seed_config: &SeedConfig,
) -> Result<(), EstimationError> {
    freeze_lambda_search_nuisance_at_canonical_anchor_with_ext_count(
        reml_state,
        resolved_likelihood_scale,
        k,
        heuristic_lambdas,
        seed_config,
        0,
    )
}

/// Joint-design counterpart of
/// [`freeze_lambda_search_nuisance_at_canonical_anchor`].
///
/// `external_hyper_count` preserves the objective-completeness policy of the
/// joint surface while the anchor evaluates `rho = 0`. In particular, an
/// anchor must not memoize a value under the fixed-design correction policy and
/// then let a joint `[rho, psi]` evaluation reuse it.
pub(crate) fn freeze_lambda_search_nuisance_at_canonical_anchor_with_ext_count(
    reml_state: &RemlState<'_>,
    resolved_likelihood_scale: &gam_problem::ResolvedLikelihoodScale,
    k: usize,
    heuristic_lambdas: Option<&[f64]>,
    seed_config: &SeedConfig,
    external_hyper_count: usize,
) -> Result<(), EstimationError> {
    let (frozen, family) = match resolved_likelihood_scale {
        gam_problem::ResolvedLikelihoodScale::Gamma {
            estimated: true, ..
        } => (&reml_state.frozen_gamma_shape, "gamma shape"),
        gam_problem::ResolvedLikelihoodScale::Tweedie {
            estimated: true, ..
        } => (&reml_state.frozen_tweedie_phi, "tweedie dispersion"),
        gam_problem::ResolvedLikelihoodScale::BetaPrecision {
            estimated: true, ..
        } => (&reml_state.frozen_beta_phi, "beta precision"),
        _ => return Ok(()),
    };
    if k == 0 || frozen.load(Ordering::Relaxed) != 0 {
        return Ok(());
    }
    if reml_state.persistent_warm_start_store().is_some() {
        crate::bail_invalid_estim!(
            "the {family} λ-search freeze must be anchored before the persistent warm-start \
             layer is attached, or while it is scoped off (#2363/#2426); with the on-disk \
             session open the anchor solve would reload the cached β and the outer criterion \
             would again depend on cache state"
        );
    }
    // The anchor must see the same starting predictor on every machine, so an
    // externally supplied seed is not admissible input to it.
    reml_state.clear_warm_start_predictor_state();
    reml_state.clear_warm_start_adaptive_signals();

    let mut anchors = vec![Array1::<f64>::zeros(k)];
    anchors.extend(
        crate::seeding::generate_rho_candidates(k, heuristic_lambdas, seed_config)?
            .into_iter()
            .filter(|candidate| candidate.iter().any(|value| *value != 0.0)),
    );
    for anchor in &anchors {
        if let Err(error) =
            reml_state.compute_cost_with_ext_count(anchor, external_hyper_count)
        {
            log::debug!("[OUTER] nuisance anchor candidate rejected: {error:?}");
            continue;
        }
        let bits = frozen.load(Ordering::Relaxed);
        if bits != 0 {
            log::info!(
                "[OUTER] {family} λ-search freeze anchored at ρ=[{}] before any warm start (#2363): \
                 value {:.6e}; the outer criterion is now a function of the data and the model spec alone",
                anchor
                    .iter()
                    .take(4)
                    .map(|value| format!("{value:.3}"))
                    .collect::<Vec<_>>()
                    .join(","),
                f64::from_bits(bits),
            );
            break;
        }
    }
    if frozen.load(Ordering::Relaxed) == 0 {
        // No deterministic anchor produced a converged inner solve. The search
        // is about to try the same points and will report its own refusal;
        // leaving the freeze unset keeps the pre-existing capture path rather
        // than converting a seed-cascade failure into a different error here.
        log::warn!(
            "[OUTER] no deterministic anchor converged for the {family} λ-search freeze; \
             the outer criterion cannot be pinned before the seed cascade"
        );
    }
    Ok(())
}

pub(crate) fn optimize_external_designwith_heuristic_lambdas_andwarm_start<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<PenaltySpec>,
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    if opts.family.is_binomial_mixture() && opts.mixture_link.is_none() {
        crate::bail_invalid_estim!("BinomialMixture requires mixture_link specification");
    }
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&s_list, p, "optimize_external_design")?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &s_list,
        &opts.nullspace_dims,
        p,
        "optimize_external_design",
    )?;
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &s_list);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
    let k = canonical.len();
    if active_nullspace_dims.len() != k {
        crate::bail_invalid_estim!(
            "nullspace_dims length mismatch: expected {k} entries for active penalties, got {}",
            active_nullspace_dims.len()
        );
    }
    let (cfg, effective_sas_link) = resolved_external_config(opts)?;
    reject_prefit_unpenalized_rank_deficiency(w, &x_fit, &canonical)?;
    reject_prefit_binomial_separation(&cfg, y, w, &x_fit, &canonical)?;

    let design_kind = match &x {
        DesignMatrix::Dense(_) => "dense",
        DesignMatrix::Sparse(_) => "sparse",
    };
    log::info!(
        "[GAM fit] n={} p={} k={} fam={:?} link={:?} X={} reml_iter={} firth={}",
        y.len(),
        p,
        k,
        opts.family,
        cfg.link_function(),
        design_kind,
        opts.max_iter,
        cfg.firth_bias_reduction
    );

    // Own the external arrays once; the conditioned design is shared through `reml_state`.
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x;
    let offset_o = offset.to_owned();
    let canonical_shared = Arc::new(canonical);
    let cfg_shared = Arc::new(cfg.clone());

    // Issue #1000: for an identity-link Gaussian fit with an unpenalized
    // intercept, adding a constant `c` to the response is a *pure relabeling of
    // the intercept* — the hat matrix annihilates the constant column, so the
    // residuals, the profiled REML criterion, λ̂, and the smooth shape are all
    // invariant to `c`. Numerically, though, the outer REML score/gradient
    // accumulate `yᵀy`-magnitude sufficient statistics (e.g. the cached
    // `XᵀW(y−offset)`), so an uncentered large-mean response injects a `c²`
    // term that loses precision and drifts λ̂ — silently over-smoothing
    // large-mean responses (Kelvin temperatures, financial levels, calendar
    // years). Center the response by the (weighted) mean the intercept would
    // absorb for the duration of the outer λ-search only: the constant lands in
    // the intercept, which the final accept-fit below recovers *exactly* by
    // re-fitting the original (uncentered) response at the REML-selected λ̂.
    // This mirrors the existing column conditioning, which centers the design
    // columns into the intercept for the same numerical reason.
    let response_center = gaussian_identity_response_center(
        &cfg,
        &conditioning,
        opts.linear_constraints.is_some(),
        y_o.view(),
        w_o.view(),
        offset_o.view(),
    );
    // Issue #1127 (down-scale sibling of #1000): replacing the response `y` by
    // `a·y` must rescale the whole fit by `a` and leave `λ̂`/EDF unchanged (the
    // normal equations are exactly linear in `y`; the profiled REML criterion is
    // `a`-invariant up to the additive `−(n−p)·ln a` the dispersion absorbs).
    // But the outer λ-selection's convergence band is keyed to an *absolute*
    // objective scale (an inner `objective_scale.max(1.0)` floor and a `1e-6`
    // outer gradient floor); when the Gaussian objective is `O(a²) ≪ 1` those
    // floors swamp the signal and the optimizer stops early at an over-smoothed
    // `λ`. Normalize the (centered) working response to `O(1)` for the outer
    // λ-search only, mirroring the #1000 centering: the final accept-fit below
    // re-fits the *original* response at the REML-selected λ̂, so β, μ̂, σ̂² and
    // every reported quantity stay exactly on the user's scale. `center` here is
    // the constant the intercept already absorbs (so the scale is measured on the
    // residual signal, not on the offset).
    let response_scale = gaussian_identity_response_scale(
        &cfg,
        &conditioning,
        opts.linear_constraints.is_some(),
        response_center.unwrap_or(0.0),
        y_o.view(),
        w_o.view(),
        offset_o.view(),
    );
    // The outer loop borrows the response for the lifetime of `reml_state`;
    // the conditioned copy (when any) is owned at function scope so the borrow
    // outlives the state. Off the Gaussian-identity path both `response_center`
    // and `response_scale` are `None` and the outer loop borrows the original
    // response verbatim — no allocation, no behavioural change. When only one is
    // active we still apply just that transform. Both are exactly invertible by
    // the accept-fit, which re-fits the original `y_o` at the selected λ̂.
    let reml_y_conditioned: Option<Array1<f64>> =
        conditioned_outer_response(response_center, response_scale, y_o.view());
    let reml_y_view = reml_y_conditioned
        .as_ref()
        .map_or_else(|| y_o.view(), |conditioned| conditioned.view());

    let mut reml_state = RemlState::newwith_offset_shared(
        reml_y_view,
        x_fit,
        w_o.view(),
        offset_o.view(),
        Arc::clone(&canonical_shared),
        p,
        Arc::clone(&cfg_shared),
        Some(active_nullspace_dims.clone()),
        None,
        fit_linear_constraints.clone(),
    )?;
    reml_state.set_rho_prior(opts.rho_prior.clone());
    if let Some(kron) = opts.kronecker_penalty_system.clone() {
        reml_state.set_kronecker_penalty_system(kron);
    }
    if let Some(kf) = opts.kronecker_factored.clone() {
        reml_state.set_kronecker_factored(kf);
    }
    let resolved_likelihood_scale = cfg
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let estimates_negbin_theta = matches!(
        resolved_likelihood_scale,
        gam_problem::ResolvedLikelihoodScale::NegativeBinomial {
            estimated: true,
            ..
        }
    );
    if let gam_problem::ResolvedLikelihoodScale::NegativeBinomial {
        theta,
        estimated: true,
    } = resolved_likelihood_scale
    {
        let theta_seed = theta
            .value()
            .clamp(pirls::NEGBIN_THETA_MIN, pirls::NEGBIN_THETA_MAX);
        // Treat the estimated family value as a warm-start coordinate. This
        // makes an exhaustion checkpoint resumable by reconstructing the same
        // estimated-NB family with the carried theta and passing the carried rho
        // through the ordinary smoothing warm-start input.
        reml_state
            .frozen_negbin_theta
            .store(theta_seed.to_bits(), Ordering::Relaxed);
    }

    let reml_seed_config = external_reml_seed_config(k, cfg.link_function());
    // #2363: pin the λ-search nuisance BEFORE any warm start — external, in
    // memory, or on disk — can reach this state. `freeze_lambda_search_nuisance_at_canonical_anchor`
    // documents why the criterion is otherwise a function of the search path.
    freeze_lambda_search_nuisance_at_canonical_anchor(
        &reml_state,
        &resolved_likelihood_scale,
        k,
        heuristic_lambdas,
        &reml_seed_config,
    )?;
    if let Some(store) = opts.persistent_warm_start_store.clone() {
        // Attach only after the canonical nuisance anchor so cache history
        // cannot influence the criterion frame.
        reml_state.attach_persistent_warm_start_store(store);
    }
    reml_state.setwarm_start_original_beta(warm_start_beta);

    // Term/margin-order invariance (#1538/#1539). The per-ρ-coordinate canonical
    // keys label each coordinate by its placement-independent (penalty + data)
    // content, letting the outer optimizer operate in an identical canonical
    // coordinate layout for every term order (attached via
    // `with_rho_canonical_keys` below). `None` when the coordinate count does not
    // match the ρ-dimension (legacy native-order path, unchanged).
    let canon_keys = reml_state.canonical_rho_keys(k);

    let reml_tol = cfg.reml_convergence_tolerance;
    let reml_max_iter = opts.max_iter;
    let outer_eval_idx = AtomicUsize::new(0usize);
    let mixture_optspec = if opts.optimize_mixture {
        opts.mixture_link.clone()
    } else {
        None
    };
    let sas_optspec = if opts.optimize_sas {
        effective_sas_link
    } else {
        None
    };
    let mixture_dim = mixture_optspec
        .as_ref()
        .map(|s| s.initial_rho.len())
        .unwrap_or(0);
    let sas_dim = if sas_optspec.is_some() { 2 } else { 0 };
    let sasridgeweight = if sas_dim > 0 {
        sas_log_deltaridgeweight()
    } else {
        0.0
    };
    // Estimated Negative-Binomial theta and smoothing rho are solved by block
    // coordinate optimization, but acceptance is JOINT: both analytic partials
    // are measured at the identical fixed-theta PIRLS solution. The outer
    // iteration budget is also the alternation budget, so exhaustion is not a
    // second hidden tuning parameter; it returns a typed error carrying the best
    // measured checkpoint instead of minting the final iterate as a fit.
    let mut final_rho;
    // #2727: the link-shape coordinates of the shipped point, in the OUTER
    // optimizer's own coordinate system (raw `theta`, before the link state's
    // smooth-bound maps). Empty on every rho-only arm. Carried separately
    // because the shipped link STATE stores transformed values
    // (`sas_effective_epsilon`), so it cannot be compared against the
    // certificate's raw coordinates.
    let mut final_link_coords: Array1<f64>;
    let mut final_mixture_state;
    let mut final_sas_state;
    let mut final_mixture_param_covariance;
    let mut final_sas_param_covariance;
    let mut outer_result;
    let mut pirls_res;
    let mut negbin_alternation_round: usize = 0;
    let mut negbin_rho_seed: Option<Array1<f64>> = None;
    let mut negbin_best_checkpoint: Option<NegbinJointCheckpoint> = None;
    loop {
        (
            final_rho,
            final_link_coords,
            final_mixture_state,
            final_sas_state,
            final_mixture_param_covariance,
            final_sas_param_covariance,
            outer_result,
        ) = if mixture_dim > 0 && sas_dim > 0 {
            crate::bail_invalid_estim!(
                "simultaneous mixture and SAS optimization is not supported"
            );
        } else if mixture_dim == 0 && sas_dim == 0 {
            use crate::rho_optimizer::{OuterEvalOrder, OuterProblem};
            use gam_problem::{DeclaredHessianForm, Derivative};

            let rho_warm_start = negbin_rho_seed
                .as_ref()
                .and_then(|rho| rho.as_slice())
                .or(heuristic_lambdas);
            let analytic_outer_hessian_available = reml_state.analytic_outer_hessian_enabled();
            // #2359: non-Gaussian search consumes the analytic outer gradient
            // (the family derivative ladder through order three), reserving
            // exact curvature for the terminal mint audit.  Profiled Gaussian
            // identity is quadratic in eta and has no expensive order-four
            // family tower, so it uses the declared exact outer Hessian during
            // search instead of approximating it with first-order BFGS.
            let n_obs = y_o.len();
            let problem = OuterProblem::new(k)
                .with_gradient(Derivative::Analytic)
                .with_hessian(if analytic_outer_hessian_available {
                    DeclaredHessianForm::Either
                } else {
                    DeclaredHessianForm::Unavailable
                })
                .with_prefer_gradient_only(standard_reml_search_prefers_gradient_only(
                    cfg.link_function(),
                ))
                .with_barrier(
                    crate::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
                        fit_linear_constraints.as_ref(),
                    ),
                )
                .with_tolerance(reml_tol)
                .with_max_iter(reml_max_iter)
                .with_seed_config(reml_seed_config)
                .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
                .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
                // n-scaled absolute gradient floor for EVERY family (#1082).
                //
                // The REML/LAML profiled criterion is a sum over n rows
                // (deviance / −2·loglik + the penalty/logdet terms), so it and its
                // ∂/∂logλ gradient inherit an O(n) scale for Poisson, NB, binomial,
                // Tweedie, beta — exactly as for Gaussian-identity. The previous gate
                // restricted `with_objective_scale` to the Gaussian-identity arm on
                // the (incorrect) premise that only that criterion is O(n). For a
                // non-Gaussian tensor/cyclic/CI/badhealth fit at n≈1.5k–5k the fixed
                // `abs = tol ≈ 1e-6` gradient floor is then orders of magnitude below
                // the n-scaled gradient's converged residual: the relative-from-seed
                // test declares convergence iters earlier, but the binding abs floor
                // keeps the outer optimizer chasing sub-floor log-λ changes, paying a
                // full inner convergence per phantom iteration until it exhausts
                // the iteration budget — the #1082 outer-loop "cycling"
                // timeout. Lifting the floor to ~n·1e-9 (the same calibration the
                // spatial/custom-family outer already uses via `with_problem_size`,
                // #1053/#1066/#1069) lets the loop terminate as soon as the relative
                // reduction is met, for every family, while the relative-to-cost
                // component still owns the actual convergence decision.
                .with_objective_scale(Some(n_obs as f64))
                .with_problem_size(n_obs, x_o.ncols())
                .with_rho_bound(crate::estimate::RHO_BOUND)
                // Make the outer smoothing-parameter search invariant to the order
                // the smooth terms / tensor margins were written (#1538/#1539). The
                // structural keys label each ρ-coordinate by its placement-
                // independent penalty content, so the optimizer canonicalizes the
                // coordinate layout and resolves the flat double-penalty REML valley
                // identically for `s(x)+s(z)` vs `s(z)+s(x)` and `te(x,z)` vs
                // `te(z,x)`. `None` (coordinate count not matching ρ-dim) leaves the
                // native-order path unchanged.
                .with_rho_canonical_keys(canon_keys.clone());
            let problem = if let Some(h) = rho_warm_start {
                problem.with_heuristic_lambdas(h.to_vec())
            } else {
                problem
            };
            let problem = if let Some(h) = rho_warm_start.filter(|h| h.len() == k) {
                problem.with_initial_rho(Array1::from_iter(h.iter().copied()))
            } else {
                problem
            };

            // Geometric-mean log prior-weight anchor `log g(w) = (1/n₊)·Σ log wᵢ`
            // over the positive-weight rows. The pure-REML optimum for a *profiled*
            // (Gaussian-identity) fit drifts by `ρ̂ → ρ̂ + log c` under a global
            // prior-weight rescale `w → c·w` (`H = XᵀWX + λS`, so λ → c·λ keeps the
            // penalised curvature proportional to the data curvature, β̂ / EDF /
            // predictions fixed). The outer ρ-search seed and the relative-from-seed
            // convergence test would otherwise be referenced to a weight-independent
            // origin (0), so a heavily up-weighted fit starts `log c` further from
            // its (shifted) optimum and the optimiser stops short — exactly the
            // weight-scale non-invariance of λ̂ reported in issue #877. Anchoring the
            // seed at `log g(w)` makes the search start the SAME relative distance
            // from the optimum regardless of the weight magnitude.
            //
            // This is the SAME gated anchor the outer ρ-prior uses
            // ([`RemlState::rho_weight_anchor`]): it is the geometric-mean
            // log-weight for a profiled-dispersion family and *exactly 0* for a
            // fixed-dispersion family (Poisson, binomial, …). For fixed dispersion
            // `w = c` is exact `c`-fold replication: the two encodings share an
            // identical LAML objective and optimum, so anchoring the seed by their
            // (differing) per-row log-weight mean would seed the weighted encoding
            // `log c` above its true optimum and the relative-convergence test would
            // stop it short — over-smoothing vs replication (issue #893). With all
            // weights 1 (or any fixed-dispersion family) the anchor is exactly 0, so
            // those fits stay byte-identical.
            let weight_log_geom_mean: f64 = reml_state.rho_weight_anchor();
            let gaussian_risk = matches!(
                reml_seed_config.risk_profile,
                SeedRiskProfile::Gaussian | SeedRiskProfile::GaussianLocationScale
            );
            // Score a small set of analytic, data-derived starts before the outer
            // solve. These are initial conditions only: the optimizer must converge
            // from the selected start, and no seed is promoted directly to a fit.
            let run_gaussian_anchored_prepass = gaussian_risk && weight_log_geom_mean.abs() > 1e-12;
            // A caller-supplied rho seed (`init_rhos`/`heuristic_lambdas`, now in
            // rho-space) is an explicit warm-start installed via `with_initial_rho`
            // above. It still ANCHORS the initial.sp prepass below rather than
            // short-circuiting it: the prepass only adopts its analytic candidate
            // when that STRICTLY lowers the true REML/LAML cost, so a healthy warm
            // seed is returned unchanged (the candidate never beats it → byte-
            // identical behaviour). What the anchor-and-adopt rescues is a warm seed
            // TRAPPED in a shallow under-smoothing local basin: when the design's
            // kernel collapses (e.g. the constant-curvature `curv()` smooth fitted
            // at a trial κ on the +chart side — the geodesic-exponential kernel's
            // off-diagonals → 1, so its global REML optimum is a LARGE λ that the
            // local outer optimizer, warm-started from the previous-κ λ̂, slides away
            // from into the spurious low-λ optimum). The shallow optimum's
            // spuriously-low deviance made the κ outer objective monotone toward the
            // +chart bound for any curved data (gam#1464 — hyperbolic truth recovered
            // as spherical); the analytic high-λ `initial.sp` candidate lets the
            // prepass jump into the correct high-λ basin so the per-κ REML cost
            // matches the textbook profiled-REML and the curvature SIGN is
            // identifiable. Same machinery as the gam#1266 double-penalty rescue.
            let caller_seeded_rho = rho_warm_start.is_some_and(|h| h.len() == k);
            let prepass_candidates: Vec<Array1<f64>> = {
                // Validate the seed ρ-box ONCE, at the boundary where it enters the
                // seed machinery, and REFUSE an inverted/non-finite interval rather
                // than silently reordering it (#2379). An inverted `[lo, hi]` means
                // the ρ lower wall and the over-smoothing ceiling — two
                // independently-owned constants — have drifted apart (the #2370
                // disease); a swap here would make the optimizer solve a different,
                // silently substituted box and still return a fitted model.
                // `run_outer_uncertified` already enforces this contract at the outer
                // entry; the prepass must not undercut it. Crucially the raw pair is
                // validated BEFORE the `RHO_BOUND` widening below, so the widening
                // can never silently un-invert a drifted box.
                let bnds = reml_seed_config.bounds;
                let raw_bounds = OrderedRhoBounds::new(bnds.0, bnds.1)?;
                // The criterion-ranked prepass evaluates the TRUE REML/LAML cost, so
                // it is safe — and necessary — to let it explore the full
                // over-smoothing range the outer optimizer itself can reach
                // (`RHO_BOUND`), not just the narrower default seed-placement band.
                // A double-penalty (null-space-shrinkage) smooth on data living in
                // one penalty's null space has its global REML optimum at a LARGE
                // wiggliness λ (range block fully smoothed), often beyond the seed
                // band; the cost surface also has a shallower local optimum at a
                // moderate λ that leaves wiggle under-penalized (EDF inflated,
                // gam#1266). If the prepass cannot seed past that local optimum, the
                // outer EFS — which only takes cost-improving steps — relaxes back
                // into it. The collapsing-kernel spatial smooth (gam#1464) has the
                // same shape: the high-λ basin sits beyond a shallow low-λ trap.
                // Widening only the upper (over-smoothing) bound lets the prepass
                // place the seed in the correct high-λ basin; the lower
                // (under-smoothing) bound stays at the default so we never seed an
                // overfit origin. The seed is still only adopted when it strictly
                // lowers the REML cost, so well-balanced and single-penalty fits are
                // unaffected.
                // Widen only the upper (over-smoothing) bound to the full range the
                // outer optimizer can reach. `with_upper_at_least` only ever *raises*
                // `hi`, so the box stays ordered by construction (`RHO_BOUND` is a
                // finite constant) — no re-validation needed.
                let seed_bounds = raw_bounds.with_upper_at_least(crate::estimate::RHO_BOUND);
                // risk_shift is the default seed bias when no caller warm-start is given;
                // it is NOT applied on top of a caller-supplied rho seed.
                let risk_shift: f64 = match reml_seed_config.risk_profile {
                    SeedRiskProfile::Gaussian | SeedRiskProfile::GaussianLocationScale => 0.0,
                    SeedRiskProfile::GeneralizedLinear => 1.0,
                    SeedRiskProfile::Survival => 2.0,
                };
                // Anchor the prepass at the caller-supplied `heuristic_lambdas` when
                // one is present (it is already in rho-space, used as-is) — the
                // analytic candidate is scored relative to the warm start and keeps
                // it unless it is strictly better. Otherwise anchor the default
                // risk-shift origin to the weight scale (issue #877).
                let base = if let Some(h) = rho_warm_start.filter(|h| h.len() == k) {
                    Array1::from_iter(h.iter().map(|&v| seed_bounds.clamp(v)))
                } else {
                    Array1::from_elem(k, seed_bounds.clamp(risk_shift + weight_log_geom_mean))
                };
                // #2069 / #1575: the analytic mgcv-style `initial.sp` seed
                // replaces the banned log-λ grid prepass. One commensurate-
                // curvature estimate — `ρ_j = ln(tr(XᵀWX_j)/tr(S_j))` — proposes a
                // single candidate relative to `base` (the caller warm-start or the
                // weight-anchored origin). A smooth whose penalized subspace carries
                // little data support gets a large `λ_j` by construction, so the
                // #1266/#1464 high-λ basin is reached analytically without a lattice
                // search; `(lo, hi)` already widens `hi` to `RHO_BOUND` so a
                // genuinely large `λ_j` is not clipped to the seed band. The seed is
                // order-independent, so no canonical permutation is needed.
                // Two principled, data-derived candidates are scored against the
                // anchor, each adopted only when it STRICTLY lowers the true
                // REML/LAML cost — exactly the criterion the old grid used, but
                // scoring a handful of hand-derived candidates instead of a
                // lattice. A healthy warm start stays byte-identical (no candidate
                // beats it → none adopted).
                //
                //   1. The mgcv-style analytic `initial.sp` seed
                //      `ρ_j = ln(tr(XᵀWX_j)/tr(S_j))` (#2069/#1575) — a
                //      commensurate-curvature start that jumps a warm seed
                //      trapped in a shallow UNDER-smoothing basin into the
                //      analytic high-λ basin (#1266 double-penalty null-space,
                //      #1464 collapsing-kernel spatial).
                //
                // The generated-seed screen (`generate_rho_candidates` +
                // `rank_seeds_with_screening`) remains the multi-basin backstop.
                let initial_sp = reml_state.analytic_initial_sp_rho(&base, seed_bounds);
                //   2. The certified single-λ (diagonal) profiled optimum on the
                //      SUMMED penalty `Σ_j S_j`, broadcast to a uniform per-block
                //      ρ. This is an honest one-dimensional restriction of the
                //      coupled multi-λ objective: overlapping penalty blocks make
                //      the penalty pseudo-determinant nonseparable, so there is no
                //      per-block "exact" cyclic closed form. The candidate is
                //      admitted only after the true coupled REML cost scores it.
                // A FAILED seed heuristic must never be fatal. The summed-penalty
                // profiled-diagonal candidate solves a closed-form REML on the
                // collapsed 1-D restriction; on a tiny / near-degenerate design
                // (e.g. `n ≈ nullity` of the summed penalty, the `p ≥ n` corner
                // reached by `y ~ s(x)` on very few rows, #2355) that closed form
                // can honestly refuse. That refusal only means "this ONE seed is
                // unavailable" — the generated-seed screen and the neutral/base
                // anchors remain, and the outer optimizer is the sole authority on
                // whether the fit certifies. Propagating the seed error with `?`
                // instead killed the entire fit for a mere unavailable candidate.
                // Treat an errored candidate as absent (`None`) so the search still
                // runs from the surviving seeds.
                let summed_diagonal = reml_state
                    .analytic_gaussian_profiled_diagonal_rho(seed_bounds)
                    .ok()
                    .flatten()
                    .map(|rho_blocks| {
                        let mut seed = base.clone();
                        for (coord, &r) in seed.iter_mut().zip(rho_blocks.iter()) {
                            *coord = seed_bounds.clamp(r);
                        }
                        seed
                    });
                let base_cost = reml_state
                    .compute_cost(&base)
                    .ok()
                    .filter(|c| c.is_finite());
                let mut ranked_candidates: Vec<(f64, Array1<f64>)> = base_cost
                    .map(|cost| vec![(cost, base.clone())])
                    .unwrap_or_default();
                // Keep the strictly-cheapest certified/scored candidate.
                //
                // #2607: this choice is a COMPARISON OF NUMBERS, and until now the
                // log recorded only its outcome (`base -> refined`). That is not
                // enough to read a seed that lands on a wall. On `hifreq_tensor_k10`
                // the selected seed is the ρ ceiling in every coordinate and the fit
                // converges in ONE outer iteration at `edf = 1.294` of `p = 576` —
                // the intercept — and from the old line alone there is no way to
                // tell whether the heuristic malfunctioned or whether the criterion
                // genuinely scores the wall below the origin. Those two call for
                // completely different fixes, so the costs that decided it are now
                // reported next to the point they scored.
                let mut refined = base.clone();
                let mut best_cost = base_cost;
                let mut scored: Vec<(&str, Option<f64>)> = vec![("base", base_cost)];
                for (name, candidate) in [
                    ("initial_sp", initial_sp),
                    ("summed_diagonal", summed_diagonal),
                ] {
                    let Some(candidate) = candidate else {
                        scored.push((name, None));
                        continue;
                    };
                    let candidate_cost = reml_state
                        .compute_cost(&candidate)
                        .ok()
                        .filter(|c| c.is_finite());
                    scored.push((name, candidate_cost));
                    if let Some(cost) = candidate_cost {
                        ranked_candidates.push((cost, candidate.clone()));
                    }
                    let candidate_beats_best = match (candidate_cost, best_cost) {
                        (Some(cc), Some(bc)) => cc < bc,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    if candidate_beats_best {
                        refined = candidate;
                        best_cost = candidate_cost;
                    }
                }
                let scored_report = scored
                    .iter()
                    .map(|(name, cost)| match cost {
                        Some(value) => format!("{name}={value:.9e}"),
                        None => format!("{name}=unavailable"),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let seed_moved = refined
                    .iter()
                    .zip(base.iter())
                    .any(|(&a, &b)| (a - b).abs() > 1e-12);
                // For a caller-seeded fit, adopt the analytic result only when it
                // strictly moved the warm seed (found a strictly-cheaper basin); an
                // unmoved result leaves the warm start exactly as installed above, so
                // healthy warm-started fits stay byte-identical. The Gaussian
                // weight-anchored emit only applies on the non-caller-seeded origin.
                if seed_moved || (run_gaussian_anchored_prepass && !caller_seeded_rho) {
                    // Report the start-point comparison, but do not confuse its
                    // cheapest point with the eventual fit: Gaussian sends every
                    // unique finite analytic candidate through a certified full
                    // solve below and keeps the best converged REML value.
                    log::info!(
                        "[OUTER] standard REML analytic-start ranking: {:?} -> {:?} \
                         (scored: {scored_report}; bounds {:.3}..{:.3})",
                        base.as_slice().unwrap_or(&[]),
                        refined.as_slice().unwrap_or(&[]),
                        seed_bounds.lower(),
                        seed_bounds.upper(),
                    );
                }

                if gaussian_risk {
                    // A start-point cost is not a basin certificate. Preserve
                    // every unique finite analytic candidate, ordered only to
                    // put the cheapest start first; the outer runner performs a
                    // full certified solve from each and keeps the lowest
                    // converged REML value. This restores multimodal robustness
                    // without restoring the arbitrary log-lambda lattice.
                    ranked_candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let mut candidates = Vec::with_capacity(ranked_candidates.len());
                    for (_, candidate) in ranked_candidates {
                        if !candidates.iter().any(|existing| existing == &candidate) {
                            candidates.push(candidate);
                        }
                    }
                    candidates
                } else if seed_moved
                    || (run_gaussian_anchored_prepass && !caller_seeded_rho)
                {
                    vec![refined]
                } else {
                    Vec::new()
                }
            };
            let problem = if let Some((first, remaining)) = prepass_candidates.split_first() {
                problem
                    .with_initial_rho(first.clone())
                    .with_initial_rho_candidates(remaining.to_vec())
            } else {
                problem
            };
            // Attach the outer-loop cache session. The session shares its
            // realized-fit-context key with the inner beta record (different
            // payload namespace), so a SIGKILL mid-outer-iter leaves both the
            // last accepted β (inner record) and the best rho seen so far
            // (outer iterate) on disk for the next run.
            let problem = match reml_state.outer_cache_session() {
                Some(session) => problem.with_cache_session(session),
                None => problem,
            };

            let obj = problem.build_objective_with_screening_proxy(
                &mut reml_state,
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.compute_cost(rho)
                },
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                    state.compute_outer_eval_with_order(
                        rho,
                        if analytic_outer_hessian_available {
                            OuterEvalOrder::ValueGradientHessian
                        } else {
                            OuterEvalOrder::ValueAndGradient
                        },
                    )
                },
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>,
                 order: OuterEvalOrder| {
                    outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                    state.compute_outer_eval_with_order(rho, order)
                },
                Some(|state: &mut &mut crate::estimate::reml::RemlState<'_>| {
                    state.reset_outer_seed_state()
                }),
                Some(
                    |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                        state.compute_efs_steps(rho)
                    },
                ),
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.compute_screening_proxy(rho)
                },
            );
            // #2348 Inc 5: standard REML can form its own λ→∞ face limit
            // exactly (the null-space-restricted fit plus the analytic
            // first-order form of the logdet/trace terms there), so the outer
            // certificate can PROVE an infinite-smoothing face instead of
            // measuring a tail beside the box.
            let obj = obj.with_rail_face_limit(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>,
                 face: &[usize]| { state.rail_face_limit(rho, face) },
            );
            // #2545: publish the soft rho-guard BARRIER's own ρ-gradient so the
            // certificate can subtract it where the barrier is not part of the
            // optimality condition (a railed coordinate, and the tail probes).
            // The `log cosh` barrier's gradient saturates at `w·a = 1.3333e-7`
            // rather than decaying, and `project_gradient_vector` keeps exactly
            // that positive part at an upper rail — so before this hook existed,
            // |Pg| ≥ 1.3333e-7 at every upper rail and a λ=∞ face could never
            // certify however clean the fit. It reads the SAME atom `build_prior`
            // reads, at the SAME weight-anchored coordinate, so the subtraction
            // cannot drift from the addition.
            let obj = obj.with_soft_rho_guard_gradient(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.soft_rho_guard_gradient(rho)
                },
            );
            // #2676: publish the criterion's EXACT invariance — the directions
            // of rho along which the penalty map, and therefore the criterion,
            // does not move at all. The outer certificate deflates them instead
            // of judging a chain-rule term against its own absolute value. Same
            // seam as the barrier hook above: the closure speaks rho, and
            // `ClosureObjective` applies the theta embedding from the declared
            // layout.
            let obj = obj.with_criterion_invariance(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.criterion_invariant_directions(rho)
                },
            );
            // Standard REML publishes its current original-basis coefficients
            // and consumes a cached coefficient vector through the symmetric
            // hook below. The runner calls it only after reset and only for the
            // bitwise-matching outer seed that owns the cached vector.
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());

            let strategy_result = problem.run(&mut obj, "standard REML")?;
            drop(obj);
            let accepted_rho = strategy_result.rho.clone();
            (
                accepted_rho,
                // Rho-only arm: the outer coordinate IS rho, so there are no
                // link coordinates to carry and the joint point is the rho one.
                Array1::zeros(0),
                cfg.link_kind.mixture_state().cloned(),
                cfg.link_kind.sas_state().copied(),
                None,
                None,
                strategy_result,
            )
        } else {
            let use_mixture = mixture_dim > 0;
            let use_sas = sas_dim > 0;
            let use_beta_logistic =
                use_sas && matches!(cfg.link_function(), LinkFunction::BetaLogistic);
            let theta_dim = k + mixture_dim + sas_dim;
            let sasspec = sas_optspec;
            let mixspec = mixture_optspec
                .clone()
                .or_else(|| {
                    if use_mixture {
                        None
                    } else {
                        Some(MixtureLinkSpec {
                            components: Vec::new(),
                            initial_rho: Array1::zeros(0),
                        })
                    }
                })
                .ok_or_else(|| EstimationError::InvalidInput("missing mixture spec".to_string()))?;
            let mut heuristic_theta = Vec::new();
            if let Some(hvals) = heuristic_lambdas
                && hvals.len() == k
            {
                heuristic_theta.extend_from_slice(hvals);
                if use_mixture {
                    heuristic_theta
                        .extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
                }
                if let Some(spec) = sasspec {
                    heuristic_theta.push(spec.initial_epsilon);
                    heuristic_theta.push(spec.initial_log_delta);
                }
            }
            let heuristic_theta_ref = if heuristic_theta.len() == theta_dim {
                Some(heuristic_theta.as_slice())
            } else {
                None
            };
            let aux_dim_outer = if use_mixture { mixture_dim } else { sas_dim };
            let mut reml_seed_config_mix = reml_seed_config;
            reml_seed_config_mix.num_auxiliary_trailing = aux_dim_outer;
            if theta_dim >= REML_SEED_SCREENING_RHO_CAP {
                reml_seed_config_mix.max_seeds = 1;
                reml_seed_config_mix.seed_budget = 1;
            }
            use crate::rho_optimizer::OuterProblem;
            use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
            let initial_link_kind = cfg.link_kind.clone();
            // Same criterion, same declaration as the profiled-REML arm above
            // (#1082): this is the location-scale / SAS-mixture LAML score, a
            // sum over the same n rows, so its d/d-theta inherits the same O(n)
            // scale. Declaring it is what keeps the outer stationarity band a
            // property of the data since #2613 -- an undeclared route falls
            // back to the bare absolute tolerance, which at large n is orders
            // below the residual a converged fit floors at.
            let n_obs = y_o.len();
            let problem = OuterProblem::new(theta_dim)
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Either)
                .with_prefer_gradient_only(true)
                .with_objective_scale(Some(n_obs as f64))
                .with_problem_size(n_obs, x_o.ncols())
                .with_psi_dim(mixture_dim + sas_dim)
                .with_barrier(
                    crate::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
                        fit_linear_constraints.as_ref(),
                    ),
                )
                .with_tolerance(reml_tol)
                .with_max_iter(reml_max_iter)
                .with_seed_config(reml_seed_config_mix)
                .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
                .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
                .with_rho_bound(crate::estimate::RHO_BOUND);
            let problem = if let Some(h) = heuristic_theta_ref {
                problem.with_heuristic_lambdas(h.to_vec())
            } else {
                problem
            };
            let problem = if let Some(h) = heuristic_theta_ref {
                problem.with_initial_rho(Array1::from_iter(h.iter().copied()))
            } else {
                problem
            };
            let problem = match reml_state.outer_cache_session() {
                Some(session) => problem.with_cache_session(session),
                None => problem,
            };
            // Shared helper: parse theta into rho + link params, update link state.
            let apply_link_theta = |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                                    theta: &Array1<f64>|
             -> Result<Array1<f64>, EstimationError> {
                let rho = theta.slice(s![..k]).to_owned();
                let mut cfg_eval = cfg.clone();
                if use_mixture {
                    let mix_rho = theta.slice(s![k..(k + mixture_dim)]).to_owned();
                    cfg_eval.link_kind = InverseLink::Mixture(
                        state_fromspec(&MixtureLinkSpec {
                            components: mixspec.components.clone(),
                            initial_rho: mix_rho,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "invalid blended inverse link: {e}"
                            ))
                        })?,
                    );
                }
                if use_sas {
                    let epsilon = if use_beta_logistic {
                        theta[k]
                    } else {
                        let (v, _) = sas_effective_epsilon(theta[k]);
                        v
                    };
                    let delta_like = theta[k + 1];
                    cfg_eval.link_kind = if use_beta_logistic {
                        InverseLink::BetaLogistic(
                            state_from_beta_logisticspec(SasLinkSpec {
                                initial_epsilon: epsilon,
                                initial_log_delta: delta_like,
                            })
                            .map_err(|e| {
                                EstimationError::InvalidInput(format!(
                                    "invalid Beta-Logistic link: {e}"
                                ))
                            })?,
                        )
                    } else {
                        InverseLink::Sas(
                            state_from_sasspec(SasLinkSpec {
                                initial_epsilon: epsilon,
                                initial_log_delta: delta_like,
                            })
                            .map_err(|e| {
                                EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                            })?,
                        )
                    };
                }
                state.set_link_states(
                    cfg_eval.link_kind.mixture_state().cloned(),
                    cfg_eval.link_kind.sas_state().copied(),
                );
                Ok(rho)
            };

            // SAS ridge/barrier cost correction (shared between cost_fn, eval_fn, efs_fn).
            // #2685: the beta-logistic block used to be excluded from this
            // entirely, leaving `[ε, log δ]` with no counter-term at all — and
            // with k = 0 penalty blocks the criterion carries no `log|S|` term
            // either, so nothing opposed the measured monotone drift of `log δ`
            // toward −∞. It now carries the same weak ridge as the SAS block, on
            // BOTH coordinates (for beta-logistic they are symmetric: the shapes
            // are `exp(log δ ∓ ε)`, so ε is a log-shape too, not the bounded
            // skew SAS reparameterizes). The tanh edge barrier stays SAS-only:
            // it is denominated in `sas_log_delta_bound()`, which is not the
            // beta-logistic shape bound.
            let sas_ridge_cost = |theta: &Array1<f64>| -> f64 {
                if use_sas && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    let mut extra = 0.5 * sasridgeweight * log_delta * log_delta;
                    if use_beta_logistic {
                        let eps = theta[k];
                        extra += 0.5 * sasridgeweight * eps * eps;
                    } else {
                        let (barriercost, _) = sas_log_delta_edge_barriercostgrad(log_delta);
                        extra += barriercost;
                    }
                    extra
                } else {
                    0.0
                }
            };

            let obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut crate::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                // Route the cost through the SAME link-ext evaluator the gradient
                // closure uses (value-only), so both see the #1876 inner-KKT
                // envelope correction `Ṽ = V − ½·rᵀH⁻¹r`. Using the plain
                // `compute_cost` here would report the raw capped-β̂ value `V`
                // while the gradient closure reports `∇Ṽ`, desyncing the outer
                // trust-region ratio test on any first-order-capped inner solve.
                let value_mode =
                    crate::estimate::reml::reml_outer_engine::EvalMode::ValueOnly;
                let result = state.evaluate_unified_with_link_ext(&rho, value_mode)?;
                let cost = result.cost + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut crate::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode =
                    crate::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian;
                let result = state.evaluate_unified_with_link_ext(&rho, eval_mode)?;

                let cost = result.cost + sas_ridge_cost(theta);
                let mut grad = result.gradient.ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "unified evaluator returned no gradient in ValueGradientHessian mode"
                            .to_string(),
                    )
                })?;

                assert_eq!(
                    grad.len(),
                    theta_dim,
                    "unified evaluator gradient length {} != theta_dim {}",
                    grad.len(),
                    theta_dim
                );

                let grad_effective = grad.clone();
                let mut hessian = materialize_link_outer_hessian(result.hessian, theta_dim)?;

                // SAS epsilon reparameterization chain rule.
                if use_sas && !use_beta_logistic {
                    let (_, d_eps_d_raw, d2_eps_d_raw2) = sas_effective_epsilon_second(theta[k]);
                    for j in 0..theta_dim {
                        hessian[[k, j]] *= d_eps_d_raw;
                        hessian[[j, k]] *= d_eps_d_raw;
                    }
                    hessian[[k, k]] += grad_effective[k] * d2_eps_d_raw2;
                    grad[k] *= d_eps_d_raw;
                }
                // Link-block ridge (+ the SAS-only edge barrier) gradient and
                // Hessian, matching `sas_ridge_cost` term for term (#2685).
                if use_sas && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    grad[k + 1] += sasridgeweight * log_delta;
                    hessian[[k + 1, k + 1]] += sasridgeweight;
                    if use_beta_logistic {
                        grad[k] += sasridgeweight * theta[k];
                        hessian[[k, k]] += sasridgeweight;
                    } else {
                        let (_, barriergrad, barrierhess) =
                            sas_log_delta_edge_barriercostgradhess(log_delta);
                        grad[k + 1] += barriergrad;
                        hessian[[k + 1, k + 1]] += barrierhess;
                    }
                }

                let cost_sec = tcost.elapsed().as_secs_f64();
                let aux_dim = if use_mixture { mixture_dim } else { sas_dim };
                log::debug!(
                    "[outer-eval {eval_idx}] theta_dim={} aux_dim={} unified_link_ext time_sec={:.3}",
                    theta_dim,
                    aux_dim,
                    cost_sec,
                );
                // Diagnostic capture of the opening outer evals (no-op unless a
                // measurement test enabled it): records the ANALYTIC θ-gradient
                // the optimizer received, so the ε/log_δ component at the init
                // can be read directly (#1876).
                crate::estimate::outer_eval_capture::record_outer_eval(theta, cost, &grad);
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianValue::Dense(hessian),
                    inner_beta_hint: state.current_original_basis_beta(),
                })
            },
            Some(|state: &mut &mut crate::estimate::reml::RemlState<'_>| {
                state.reset_outer_seed_state();
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
            }),
            Some(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 theta: &Array1<f64>| {
                    let rho = apply_link_theta(state, theta)?;
                    let mut efs_eval = state.compute_efs_steps_with_link_ext(&rho)?;

                    // SAS reparameterization chain rule on ψ steps.
                    if use_sas && !use_beta_logistic {
                        let (_, d_eps_d_raw) = sas_effective_epsilon(theta[k]);
                        if efs_eval.steps.len() > k {
                            efs_eval.steps[k] *= d_eps_d_raw;
                        }
                        if let Some(ref mut pg) = efs_eval.psi_gradient
                            && !pg.is_empty() {
                                pg[0] *= d_eps_d_raw;
                            }
                    }

                    // SAS log-δ ridge + edge barrier: their gradients enter
                    // `result.gradient` from the unified evaluator (estimate.rs
                    // 2170+), and `compute_efs_steps_with_link_ext` runs the
                    // universal-form EFS step `Δρ = log(1 − 2·g_full/q_eff)`
                    // which absorbs them automatically. We only need to
                    // mirror that contribution into the *cost* slot here so
                    // the outer fixed-point bridge's line search compares
                    // augmented-cost trial points consistently.
                    efs_eval.cost += sas_ridge_cost(theta);
                    Ok(efs_eval)
                },
            ),
        );
            // #2629: this objective is built on the SAME `&mut RemlState` as the
            // standard-REML arm above and evaluates through
            // `evaluate_unified_with_link_ext` → `assemble_and_evaluate` →
            // `build_prior`, so it carries the IDENTICAL `log cosh` barrier —
            // measured on the SAS ladder, the barrier is 99.998% of the outer
            // ρ-gradient at ρ=30 (`1.332439e-7` of `1.332418e-7`) with a clean
            // `−22.82·e^{−ρ}` face tail underneath it, and the total is positive,
            // which is exactly the sign `project_gradient_vector` retains at an
            // upper bound. Publishing nothing left a standing `|Pg| ≥ w·a` on
            // every railed coordinate of every flexible-link fit.
            //
            // The closure is BYTE-IDENTICAL to the standard arm's, which is the
            // point of #2629's seam change: this objective's outer coordinate is
            // `θ = [ρ (k entries), mixture/SAS link coordinates]`, and the
            // θ-embedding (barrier in the leading `k`, exact zeros in the link
            // slots) is applied once by `ClosureObjective` from the declared
            // `psi_dim` rather than hand-written here. A hand-written version is
            // the failure this issue exists to prevent: every coordinate's
            // barrier is the same order of magnitude, so a misalignment is
            // invisible in the norm.
            let obj = obj.with_soft_rho_guard_gradient(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.soft_rho_guard_gradient(rho)
                },
            );
            // #2676: publish the criterion's EXACT invariance — the directions
            // of rho along which the penalty map, and therefore the criterion,
            // does not move at all. The outer certificate deflates them instead
            // of judging a chain-rule term against its own absolute value. Same
            // seam as the barrier hook above: the closure speaks rho, and
            // `ClosureObjective` applies the theta embedding from the declared
            // layout.
            let obj = obj.with_criterion_invariance(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.criterion_invariant_directions(rho)
                },
            );
            // Same exact-seed cache publish/consume symmetry as the standard
            // REML arm above (issue #236).
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());
            let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
            drop(obj);
            let final_rho = outer_result.rho.slice(s![..k]).to_owned();
            // #2727: the remainder of the joint outer coordinate. `final_rho`
            // above is only its leading rho block; these are the link-shape
            // coordinates the same certificate covers, kept raw so the shipped
            // point can be reassembled and compared whole.
            let final_link_coords = outer_result.rho.slice(s![k..]).to_owned();
            let final_mix_state = if use_mixture {
                let final_mix_rho = outer_result.rho.slice(s![k..(k + mixture_dim)]).to_owned();
                Some(
                    state_fromspec(&MixtureLinkSpec {
                        components: mixspec.components.clone(),
                        initial_rho: final_mix_rho,
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                    })?,
                )
            } else {
                None
            };
            let final_sas_state = if use_sas {
                let epsilon_eff = if use_beta_logistic {
                    outer_result.rho[k]
                } else {
                    let (v, _) = sas_effective_epsilon(outer_result.rho[k]);
                    v
                };
                Some(if use_beta_logistic {
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: epsilon_eff,
                        initial_log_delta: outer_result.rho[k + 1],
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                    })?
                } else {
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: epsilon_eff,
                        initial_log_delta: outer_result.rho[k + 1],
                    })
                    .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?
                })
            } else {
                cfg.link_kind.sas_state().copied()
            };
            let aux_param_covariance = None;
            let (mix_cov, sas_cov) = if use_mixture {
                (aux_param_covariance, None)
            } else if use_sas {
                (None, aux_param_covariance)
            } else {
                (None, None)
            };
            (
                final_rho,
                final_link_coords,
                final_mix_state,
                final_sas_state,
                mix_cov,
                sas_cov,
                outer_result,
            )
        };
        if estimates_negbin_theta {
            let frozen_bits = reml_state.frozen_negbin_theta.load(Ordering::Relaxed);
            if frozen_bits == 0 {
                return Err(EstimationError::InvalidInput(
                    "estimated Negative-Binomial joint solve lost its frozen theta state"
                        .to_string(),
                ));
            }
            let theta = f64::from_bits(frozen_bits);
            if !theta.is_finite()
                || !(pirls::NEGBIN_THETA_MIN..=pirls::NEGBIN_THETA_MAX).contains(&theta)
            {
                return Err(EstimationError::InvalidInput(format!(
                    "estimated Negative-Binomial joint solve has invalid theta checkpoint {theta}"
                )));
            }

            // Re-evaluate value, rho gradient, and the fixed-theta PIRLS mode
            // through one cache generation. Both partial stationarity checks below
            // therefore refer to the identical (rho, theta, beta) point.
            reml_state.reset_outer_seed_state();
            let (joint_cost, rho_gradient) = reml_state.compute_cost_and_gradient(&final_rho)?;
            let joint_bundle = reml_state.obtain_eval_bundle(&final_rho)?;
            pirls_res = joint_bundle.pirls_result.as_ref().clone();
            pirls_res.likelihood = cfg.likelihood.clone().with_negbin_theta(theta);

            let final_eta = pirls_res.final_eta.to_owned();
            let (theta_score, theta_info) =
                pirls::negbin_theta_score_and_info(y_o.view(), &final_eta, w_o.view(), theta)?;
            let theta_residual = negbin_theta_stationarity_residual(theta, theta_score, theta_info);
            // This residual is a Newton displacement in the outer log-theta
            // coordinate, so it shares the outer REML tolerance. The beta
            // PIRLS tolerance certifies a different coordinate system and must
            // not silently set the theta fixed-point threshold.
            let theta_bound = reml_tol;

            let rho_lower = Array1::from_elem(final_rho.len(), -crate::estimate::RHO_BOUND);
            let rho_upper = Array1::from_elem(final_rho.len(), crate::estimate::RHO_BOUND);
            // Judged against `certificate.stationarity.bound()` just below, so
            // it must be projected against the box that certificate used
            // (#2412) — otherwise a railed coordinate's outward pull is scored
            // against a bound derived without it.
            // #2545: this residual is weighed against the CERTIFICATE's own
            // bound two lines down, so it must be the quantity the certificate
            // judged. The criterion's `log cosh` barrier leaves a saturated
            // `+w*a = 1.3333e-7` at every rail that the KKT projection retains,
            // and the certificate now removes it on exactly the railed
            // coordinates — leaving it in here would refuse a fit the
            // certificate just certified, which is the same two-spellings-of-one-
            // tolerance failure in a different place.
            let rho_barrier = reml_state.soft_rho_guard_gradient(&final_rho);
            let rail_bounds = (rho_lower, rho_upper);
            let rho_residual = crate::rho_optimizer::rail_projected_gradient_norm(
                &final_rho,
                &crate::rho_optimizer::gradient_with_rail_barrier_removed(
                    &final_rho,
                    &rho_gradient,
                    &crate::rho_optimizer::rail_relaxed_bounds(&rail_bounds),
                    Some(&rho_barrier),
                ),
                Some(&rail_bounds),
            );
            let rho_bound = outer_result
                .criterion_certificate
                .as_ref()
                .map(|certificate| certificate.stationarity.bound())
                .unwrap_or(reml_tol)
                .max(f64::EPSILON);
            let rho_certificate_ok = final_rho.is_empty()
                || (outer_result.converged()
                    && outer_result
                        .criterion_certificate
                        .as_ref()
                        .is_some_and(|certificate| certificate.certifies())
                    && rho_residual.is_finite()
                    && rho_residual <= rho_bound);
            // The three coordinates of the joint (θ, ρ, β) optimum are certified
            // independently. The β coordinate must be strictly converged; a
            // near-stationary stalled checkpoint is not a completed joint fit.
            let pirls_certificate_ok = pirls_res.status.is_converged();
            let theta_certificate_ok = theta_residual.is_finite() && theta_residual <= theta_bound;

            let merit = (rho_residual / rho_bound)
                .max(theta_residual / theta_bound)
                .max(if pirls_certificate_ok {
                    0.0
                } else {
                    f64::INFINITY
                });
            let checkpoint = NegbinJointCheckpoint {
                merit,
                theta,
                rho: final_rho.clone(),
                rho_residual,
                rho_bound,
                theta_residual,
                theta_bound,
            };
            if negbin_best_checkpoint
                .as_ref()
                .is_none_or(|best| checkpoint.merit <= best.merit)
            {
                negbin_best_checkpoint = Some(checkpoint);
            }

            if rho_certificate_ok && theta_certificate_ok && pirls_certificate_ok {
                outer_result.final_value = joint_cost;
                outer_result.final_gradient = Some(rho_gradient);
                outer_result.final_grad_norm = Some(rho_residual);
                log::debug!(
                    "[OUTER] negative-binomial joint optimum certified after {} round(s): \
                     rho KKT residual {:.3e} <= {:.3e}, theta residual {:.3e} <= {:.3e}",
                    negbin_alternation_round + 1,
                    rho_residual,
                    rho_bound,
                    theta_residual,
                    theta_bound,
                );
                break;
            }

            if negbin_alternation_round + 1 >= reml_max_iter.max(1) {
                let best = negbin_best_checkpoint
                    .as_ref()
                    .expect("the current joint checkpoint was just recorded");
                return Err(EstimationError::NegativeBinomialAlternationDidNotConverge {
                    rounds: negbin_alternation_round + 1,
                    theta_checkpoint: best.theta,
                    rho_projected_grad_norm: best.rho_residual,
                    rho_stationarity_bound: best.rho_bound,
                    theta_score_residual: best.theta_residual,
                    theta_stationarity_bound: best.theta_bound,
                    rho_checkpoint: best.rho.to_vec(),
                });
            }

            // Exact block update: maximize the conditional NB likelihood in
            // theta at the current converged eta, then re-optimize rho with theta
            // fixed. No secant/grid extrapolation and no unreported answer cap.
            let theta_next =
                pirls::estimate_negbin_theta_from_eta(y_o.view(), &final_eta, w_o.view())?;
            log::info!(
                "[OUTER] negative-binomial joint round {} not yet certified: \
                 rho residual {:.3e}/{:.3e}, theta residual {:.3e}/{:.3e}; \
                 updating theta {:.6e} -> {:.6e} and resuming from rho checkpoint",
                negbin_alternation_round + 1,
                rho_residual,
                rho_bound,
                theta_residual,
                theta_bound,
                theta,
                theta_next,
            );
            reml_state
                .frozen_negbin_theta
                .store(theta_next.to_bits(), Ordering::Relaxed);
            negbin_rho_seed = Some(final_rho.clone());
            reml_state.reset_outer_seed_state();
            negbin_alternation_round += 1;
            continue;
        }

        // Reuse the Gaussian-Identity XᵀWX cache the outer loop already populated,
        // so the final accept-fit skips the streaming GEMM as well.
        //
        // When the outer loop conditioned the response (centering for #1000, scaling
        // for #1127), that cache holds `XᵀW((y−center)/scale)`; the accept-fit runs
        // on the *original* response `y_o`, so reusing the conditioned `XᵀWy` would
        // solve on the shifted/rescaled scale and report every fitted value, residual
        // and dispersion off the user's scale. Rebuild the cross-product from the
        // original response in that case — the constant `XᵀWX` block is the only part
        // the cache would have saved, a one-off cost paid only on the rare
        // large-mean / small-magnitude responses that trigger conditioning.
        let final_cache_handle = if response_center.is_some() || response_scale.is_some() {
            None
        } else {
            reml_state.gaussian_fixed_cache_if_eligible()
        };
        let pirls_res_pair = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
            LogSmoothingParamsView::new(final_rho.view())?,
            pirls::PirlsProblem {
                x: reml_state.x(),
                offset: offset_o.view(),
                y: y_o.view(),
                priorweights: w_o.view(),
                covariate_se: None,
                gaussian_fixed_cache: final_cache_handle.as_deref(),
                // The final reported fit must be exact at the converged ρ/ψ — never
                // serve the frozen-W first-step approximation here.
                glm_first_step_gram: None,
            },
            pirls::PenaltyConfig {
                canonical_penalties: reml_state.canonical_penalties(),
                balanced_penalty_root: Some(reml_state.balanced_penalty_root()),
                reparam_invariant: None,
                p,
                coefficient_lower_bounds: None,
                linear_constraints_original: fit_linear_constraints.as_ref(),
                kronecker_factored: None,
            },
            &pirls::PirlsConfig {
                link_kind: if let Some(state) = final_mixture_state.clone() {
                    InverseLink::Mixture(state)
                } else if let Some(state) = final_sas_state {
                    if matches!(cfg.link_function(), LinkFunction::BetaLogistic) {
                        InverseLink::BetaLogistic(state)
                    } else {
                        InverseLink::Sas(state)
                    }
                } else {
                    cfg.link_kind.clone()
                },
                ..cfg.as_pirls_config()
            },
            None,
            None,
            // Final, reported fit at the REML-selected λ: refine the family's
            // estimated dispersion nuisance at the converged η. For Gamma this
            // re-estimates the shape so `dispersion_phi()` and every SE / interval
            // reflect the conditional noise, not the spread of μ (#678); for Beta
            // it drives the precision φ and the mean β̂ to their joint fixed point,
            // undoing the slope attenuation from a φ frozen at the null predictor
            // (#769). λ is fixed here, so there is no scale↔λ feedback.
            true,
            None,
        )?;
        pirls_res = pirls_res_pair.0;

        break;
    } // negative-binomial joint-coordinate loop
    // Report the outer iteration count that was MEASURED, including a genuine
    // zero. A seed that is a prior fit's terminal certificate and is still
    // stationary here is accepted without iterating
    // (`certified_resume_is_already_stationary`), so zero is a reachable,
    // meaningful outcome; flooring it to one made the reported count a claim no
    // measurement supports, and every consumer asking "did a fit happen" then
    // read a fabricated pass (#2622).
    //
    // Dropping the floor cannot let a zero-iteration fit slip past the #934
    // certificate obligation from this entry point. The `certificate_valid`
    // gate below refuses to ship at all unless the outer result converged AND
    // carries a certifying analytic certificate, for every fit with a smoothing
    // coordinate; assembly then takes its `Analytic` arm on the certificate's
    // presence, never the iteration count, and its `outer_iterations == 0`
    // fixed-λ arm stays unreachable from here.
    let iters = outer_result.iterations;

    // Map beta back to original basis
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());
    let beta_orig = conditioning.backtransform_beta(&beta_orig_internal);

    // Effective sample size for dispersion/REML accounting.
    //
    // A prior weight of exactly 0 makes a row contribute nothing to any weighted
    // cross-product (XᵀWX, XᵀWy) or to the weighted RSS (w_i·r_i² = 0), so such a
    // row is statistically equivalent to an absent row. The *only* channel left by
    // which it could still perturb the fit is an explicit observation count. To
    // keep zero-weight rows exactly equivalent to absent rows (R's `n.ok =
    // nobs − Σ[w==0]`, mgcv's dropped zero-weight observations), the dispersion
    // sample size must be the count of positive-weight rows, not the raw row
    // count. Otherwise the Gaussian scale φ̂ = weighted_rss / (n − edf) puts a
    // numerator that already excludes zero-weight rows over a denominator that
    // counts them, biasing φ̂ low and shrinking every SE (#584). The REML
    // criterion's own observation count (which drives λ selection) lives in the
    // inner-solution assembly and must apply the same positive-weight count.
    let n = w_o.iter().filter(|&&wi| wi > 0.0).count() as f64;
    let mut identity_fit_is_exact = false;
    let weighted_rss = if matches!(cfg.link_function(), LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrixvectormultiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        let raw: f64 = w_o
            .iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum();
        // An identity-link fit whose residual sits at the arithmetic's own
        // resolution has NOT estimated a small residual variance — it has
        // reproduced the response, and the number left over is the rounding in
        // `η = Σ_j x_ij β_j`. Reporting `σ̂ ≈ 4e-16` there hands the caller
        // standard errors and a criterion that move by orders of magnitude if
        // the rows are permuted.
        //
        // Snapping it to an exact zero is the SAME decision the formula path's
        // deterministic-Gaussian dispatch already makes one level up; making it
        // here, where the dispersion is actually estimated, is what stops the
        // two entry points from reporting different inference for identical
        // data (#2595). `weighted_residual_is_at_roundoff_floor` is the shared
        // certificate, and `|y_i| + |η_i|` is a LOWER bound on the operand scale
        // that formed each residual, so this fires conservatively — never on a
        // fit that genuinely misses.
        //
        // The term count is the full design width, matching what
        // `exact_unpenalized_gaussian_beta` counts. On a sparse row that
        // overstates the operations actually summed, widening the bound by at
        // most a factor of `p` — still `p·ε` RELATIVE to the row's own scale
        // (≈2e-13 at p = 1000), orders below any signal a fit could be missing.
        let at_floor = gam_problem::weighted_residual_is_at_roundoff_floor(
            raw,
            w_o.iter().copied(),
            y_o.iter()
                .zip(fitted.iter())
                .map(|(&yi, &fi)| yi.abs() + fi.abs()),
            beta_orig.len() + 1,
        );
        identity_fit_is_exact = at_floor;
        if at_floor { 0.0 } else { raw }
    } else {
        0.0
    };

    // Default solver policy stays on the REML/Laplace path. Joint HMC remains
    // available through explicit sampling flows, but fitting does not
    // automatically densify the Hessian or escalate into NUTS.
    let (final_rho, pirls_res) = (final_rho, pirls_res);

    // Recompute beta in the finalized basis/parameterization.
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    let log_lambdas = final_rho.clone();
    let lambdas = LogSmoothingParamsView::new(log_lambdas.view())?.exact_exp();
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank_total = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank_total as f64).max(0.0);
    let mut edf_by_block = vec![0.0; k];
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk), retained so per-term
    // EDF can be assembled as |coeff_range| − Σ tr_kk (issue #1219).
    let mut penalty_block_trace = vec![0.0; k];
    let mut edf_total = 0.0;
    let mut smoothing_correction = None;
    let mut smoothing_correction_method = None;
    // The exact first-order IFT correction, RETAINED even when the primary
    // pair above escalates to a cubature upgrade (#946): the corrected-EDF/AIC
    // channel reads these instead of the primary pair so it does not go dark
    // exactly when smoothing-parameter uncertainty is large enough to matter.
    let mut smoothing_correction_first_order = None;
    let mut smoothing_correction_method_first_order = None;
    let mut rho_covariance = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut beta_standard_errors = None;
    let mut beta_covariance_corrected = None;
    let mut beta_standard_errors_corrected = None;
    // #2705 group A: carried from where the constrained-posterior correction is
    // APPLIED to where the corrected covariance is READ, so the refusal below
    // can say which producer's budget the negative diagonal is inside.
    let mut constrained_diagonal_uncertainty: Option<Array1<f64>> = None;
    let mut constrained_removed_variance: Option<Array1<f64>> = None;
    // The ρ̂-conditional covariance `Vb = φ·H⁻¹` BEFORE the feasible set
    // truncates it. Present only when a truncation was actually applied, i.e.
    // exactly when `beta_covariance` is no longer that matrix.
    //
    // The corrected covariance is a different estimand from the conditional one
    // and needs the untruncated matrix to build: see the composition argument at
    // the `beta_covariance_corrected` assembly below (#2705 group A).
    let mut untruncated_conditional_covariance: Option<Array2<f64>> = None;
    let mut beta_covariance_frequentist = None;
    let mut coefficient_influence = None;
    let mut weighted_gram = None;
    let mut bias_correction_jacobian = None;
    // Factorization of stabilized Hessian in transformed basis, reused for
    // SE computation via solve-on-demand after dispersion is determined.
    let mut edf_factor: Option<Box<dyn FactorizedSystem>> = None;
    let mut bias_correction_beta = None;
    let mut rho_posterior_certificate = None;
    let mut rho_posterior_escalation = None;
    // Hold the governor charge across every dense inference allocation in this
    // fit. A refusal selects the factorized/diagonal path before any optional
    // covariance, influence, or smoothing-correction matrix is built.
    let dense_covariance_reservation = opts
        .compute_inference
        .then(|| reserve_dense_covariance_bundle(pirls_res.reparam_result.qs.nrows()))
        .flatten();
    let factorized_inference_reservation =
        if opts.compute_inference && dense_covariance_reservation.is_none() {
            reserve_factorized_inference_state(pirls_res.reparam_result.qs.nrows())
        } else {
            None
        };

    let needs_constrained_posterior = fit_linear_constraints.is_some();
    if opts.compute_inference || needs_constrained_posterior {
        // EDF by block using stabilized H and penalty roots in transformed basis.
        let h = &pirls_res.stabilizedhessian_transformed;
        let p_dim = h.nrows();
        // Factor the exact Hessian already minted by PIRLS. Any objective-level
        // ridge is already present in this matrix and its RidgePassport; this
        // inference layer is not allowed to add another unaccounted diagonal.
        let factor = h.factorize_spd().map_err(|reason| {
            EstimationError::RemlOptimizationFailed(format!(
                "exact inference Hessian factorization failed: {reason}"
            ))
        })?;
        let mut traces = vec![0.0f64; k];
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            // Build the p × rank RHS with nonzeros only in [start..end] rows.
            let r = &cp.col_range;
            let rank = cp.rank();
            let mut rhs = Array2::<f64>::zeros((p_dim, rank));
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    rhs[[r.start + row, col]] = cp.root[[col, row]];
                }
            }
            let sol =
                factor
                    .solvemulti(&rhs)
                    .map_err(|_| EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    })?;
            certify_factorized_inference_solve(h, &rhs, &sol, "penalty-block EDF trace")?;
            // Frobenius inner product: only the block rows of rhs are nonzero.
            let mut frob = 0.0f64;
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    frob += sol[[r.start + row, col]] * rhs[[r.start + row, col]];
                }
            }
            // The per-block penalty trace `tr_kk = λ_kk·tr(H⁻¹ S_kk)` is the
            // penalized effective d.f. of block `kk`, mathematically confined to
            // `[0, rank_kk]` (a PSD penalty can absorb at most its own rank). When
            // the outer REML / spatial-κ optimizer drives a redundant block's
            // `λ_kk = exp(ρ_kk)` to the finite ceiling (gam#1379: the Matérn kernel
            // already controls the smoothness a redundant operator block also
            // penalizes, so REML wants `λ → ∞`), the raw product `λ_kk · frob`
            // can overflow to `+∞` on the ridge-stabilized inference Hessian even
            // though the true value is just `rank_kk` — poisoning
            // `penalty_block_trace[kk]` and tripping the fit-result finiteness
            // validator (`fit_result.penalty_block_trace[kk] must be finite, got
            // inf`). Clamp to the valid `[0, rank]` interval so a fully-penalized
            // direction reads its exact saturated trace `rank_kk` instead of `+∞`.
            // Ordinary finite traces are inside `[0, rank]` and pass through
            // unchanged, so non-degenerate fits and their recorded EDF accounting
            // are bit-identical (the `edf_by_block` channel already clamps the
            // complementary `rank − trace` to `[0, rank]`).
            // Raw product: the `[0, rank]` admission, the non-finite
            // resolution and the `[mp, p]` floor are all owned by
            // `penalized_edf_bundle`, which every fitting route shares (#2470).
            // A `+inf` overflow of a ceiling-λ block saturates at `rank`
            // (gam#1379); a NaN (e.g. inf*0 from a poisoned solve) is NOT
            // saturation and is deliberately left to trip the
            // penalty_block_trace finiteness validator rather than being
            // resolved to a plausible number.
            traces[kk] = lambdas[kk] * frob;
        }
        let block_ranks: Vec<usize> = pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .map(|cp| cp.rank())
            .collect();
        let bundle = penalized_edf_bundle(&traces, &block_ranks, p_dim, mp);
        edf_total = bundle.edf_total;
        penalty_block_trace.clone_from(&bundle.penalty_block_trace);
        edf_by_block.clone_from(&bundle.edf_by_block);
        traces.clone_from(&bundle.penalty_block_trace);

        // Reconcile the EDF accounting with the influence matrix F = H⁻¹X'WX.
        //
        // The authoritative model definition of EDF is the influence-matrix
        // trace; the per-term EDF (`FitResult::per_term_edf`) reads `tr(F)` over
        // each block. Recompute the per-block penalty traces from the SAME exact
        // inverse `F` uses, so
        // `edf_total = p − Σ tr_kk = tr(F)`, `Σ edf_by_block = edf_total`, and the
        // total can never fall below a single term's own EDF. Done before the
        // dispersion `σ̂² = RSS/(n − edf_total)` is formed so it, too, uses the
        // honest effective d.f. (the trace-channel collapse otherwise biased
        // σ̂² high → inflated SEs on the same seeds).
        //
        // Per-block traces `tr_kk = λ_kk·tr(H⁻¹ S_kk)` are basis-invariant; map
        // each canonical block's penalty root into the original coefficient basis
        // (`root_orig = Qs · root_t`) and contract against the original-basis
        // inverse. Gated by the SAME resource-policy check as the dense
        // covariance bundle below, so this reconciliation and the influence
        // matrix `F` are formed in exactly the same regime, from the same
        // `map_hessian_to_original_basis(&pirls_res)` matrix, through the same
        // strict-Cholesky solve route; beyond the policy budget both switch off
        // together and the trace-channel value stands.
        {
            let p_orig = pirls_res.reparam_result.qs.nrows();
            if dense_covariance_reservation.is_some() {
                let h_orig = map_hessian_to_original_basis(&pirls_res)?;
                // Solve against the strict Cholesky rather than contracting
                // against a materialized `H⁻¹`. Both carry the same *backward*
                // error certificate, but a product `H⁻¹·R` inherits a FORWARD
                // error of order `cond(H)` times it, while `H·sol = R` does not
                // (#2668 measured `cond(H) = 2.099e8` on an ordinary
                // `y ~ s(x)` fit, which amplified the sibling influence matrix
                // by 3.9%). Both consumers of `H⁻¹` — these block traces and
                // the influence matrix below — now take the solve route, so
                // neither is amplified and the two stay consistent. That
                // consistency is what `influence_trace_matches_conditional_edf`
                // pins: `edf_total` comes from THESE traces while `tr(F)` comes
                // from the influence matrix, so moving only one would make them
                // disagree. Failure is not a request to silently change rank or
                // add a diagonal perturbation.
                let h_factor = gam_linalg::utils::certified_spd_factorize(
                    &h_orig,
                    "edf reconciliation",
                )
                .map_err(|error| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "EDF reconciliation requires an exact SPD Hessian factorization: {error}"
                    ))
                })?;
                {
                    let qs = &pirls_res.reparam_result.qs;
                    let p_t = qs.ncols();
                    let mut traces_f = vec![0.0f64; k];
                    for (kk, cp) in pirls_res
                        .reparam_result
                        .canonical_transformed
                        .iter()
                        .enumerate()
                    {
                        if kk >= lambdas.len() {
                            continue;
                        }
                        let r = &cp.col_range;
                        let rank = cp.rank();
                        let mut root_t = Array2::<f64>::zeros((p_t, rank));
                        for col in 0..rank {
                            for row in 0..cp.block_dim() {
                                root_t[[r.start + row, col]] = cp.root[[col, row]];
                            }
                        }
                        // S_kk = Rᵀ R; λ_kk·tr(H⁻¹ S_kk) = λ_kk·Σ_col (R_col)ᵀ H⁻¹ R_col.
                        let root_orig = qs.dot(&root_t); // p_orig × rank
                        let (sol, _certificate) =
                            h_factor.solve_matrix(&root_orig).map_err(|error| {
                                EstimationError::RemlOptimizationFailed(format!(
                                    "EDF reconciliation block solve did not certify: {error}"
                                ))
                            })?; // H⁻¹ R
                        let mut frob = 0.0f64;
                        for col in 0..rank {
                            for row in 0..p_orig {
                                frob += sol[[row, col]] * root_orig[[row, col]];
                            }
                        }
                        // Same `[0, rank]` clamp as the trace-channel path above
                        // (gam#1379): a ceiling-`λ` redundant block's
                        // `λ_kk·tr(H⁻¹ S_kk)` can overflow to `+∞` here too; the
                        // penalized trace is bounded by the block rank, so clamp to
                        // keep `penalty_block_trace` finite and the EDF accounting
                        // consistent. Finite in-range traces are untouched.
                        // NaN-safe (gam#1379): f64::clamp leaves NaN as NaN, so
                        // map any non-finite product to the saturated `rank`.
                        // Raw product; admitted by the shared accounting below,
                        // exactly as the trace-channel path above (#2470).
                        traces_f[kk] = lambdas[kk] * frob;
                    }
                    let block_ranks_f: Vec<usize> = pirls_res
                        .reparam_result
                        .canonical_transformed
                        .iter()
                        .map(|cp| cp.rank())
                        .collect();
                    let bundle_f =
                        penalized_edf_bundle(&traces_f, &block_ranks_f, p_orig, mp);
                    edf_total = bundle_f.edf_total;
                    penalty_block_trace.clone_from(&bundle_f.penalty_block_trace);
                    edf_by_block.clone_from(&bundle_f.edf_by_block);
                }
            }
        }

        if opts.compute_inference {
            // O(n⁻¹) frequentist bias correction vector b̂ =
            // H⁻¹ S(λ̂)(β̂ - μ). This is an inference product, unlike the
            // posterior-identity solves that constrained fits need regardless
            // of whether standard errors were requested.
            let beta_t = pirls_res.beta_transformed.as_ref();
            let mut s_beta_t = Array1::<f64>::zeros(p_dim);
            for (kk, cp) in pirls_res
                .reparam_result
                .canonical_transformed
                .iter()
                .enumerate()
            {
                let r = &cp.col_range;
                let local = cp.local_ref();
                let beta_block = beta_t.slice(ndarray::s![r.clone()]);
                let centered = &beta_block - &cp.prior_mean;
                let local_beta = local.dot(&centered);
                let lam_k = lambdas[kk];
                let mut acc = s_beta_t.slice_mut(ndarray::s![r.clone()]);
                acc.scaled_add(lam_k, &local_beta);
            }
            let b_t = factor.solve(&s_beta_t).map_err(|reason| {
                EstimationError::RemlOptimizationFailed(format!(
                    "exact bias-correction solve failed: {reason}"
                ))
            })?;
            certify_factorized_inference_vector_solve(h, &s_beta_t, &b_t, "bias correction")?;
            let qs = &pirls_res.reparam_result.qs;
            let b_orig = qs.dot(&b_t);
            if b_orig.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "bias-correction basis map produced non-finite coefficients".to_string(),
                ));
            }
            bias_correction_beta = Some(b_orig);
        }
        // Preserve the factorization for solve-on-demand SE and covariance
        // computation below, after dispersion has been determined.
        edf_factor = Some(factor);
    }

    // Persist residual-based scale for Gaussian identity models.
    // Contract: residual standard deviation sigma, not variance.
    //
    // Gaussian REML scale: σ̂² = RSS / (n − edf_total), matching mgcv's gam.scale.
    // Using the null-space dim (mp = p − rank(Σ_k S_k)) here was wrong: mp is the
    // minimum possible edf (all smooths fully penalized to their null space), so
    // n − mp ≥ n − edf_total, and σ̂² was systematically biased low whenever any
    // smooth/random-effect spent real edf. edf_total ∈ [mp, p_dim] is the effective
    // df computed just above from tr(λ_k · H⁻¹ S_k), and is exactly the residual
    // df mgcv uses. An inference-off unconstrained fit keeps the MLE RSS/n path;
    // a constrained fit computes EDF regardless because its posterior mean and
    // covariance scale are part of the fitted estimand, not optional inference.
    let resolved_likelihood_scale = pirls_res
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let profiled_gaussian_standard_deviation = match resolved_likelihood_scale {
        gam_problem::ResolvedLikelihoodScale::ProfiledGaussian => {
            let denom = if opts.compute_inference || needs_constrained_posterior {
                n - edf_total
            } else {
                n
            };
            if !(denom.is_finite() && denom > 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian residual degrees of freedom must be finite and positive, got {denom:?}"
                )));
            }
            if !(weighted_rss.is_finite() && weighted_rss >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian weighted RSS must be finite and non-negative, got {weighted_rss:?}"
                )));
            }
            let variance = weighted_rss / denom;
            if !variance.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian residual variance is not representable: {weighted_rss:?}/{denom:?}"
                )));
            }
            Some(variance.sqrt())
        }
        _ => None,
    };
    let dispersion =
        dispersion_from_likelihood(&pirls_res.likelihood, profiled_gaussian_standard_deviation)?;
    // Persist the square root of the resolved response dispersion for every
    // scalar-scale family. It is never an overloaded Gamma shape or an inert
    // unit placeholder; family-specific inference consumes the typed metadata.
    let standard_deviation = dispersion.phi().sqrt();

    // Explicit dispersion contract for coefficient covariance matrices:
    // Vb = H⁻¹ · cov_scale, where the stored penalized Hessian is always
    // H = XᵀWX + S_λ with the penalty added UNSCALED. The multiplier therefore
    // restores ONLY the dispersion the working weight W does not already carry:
    //
    //   * Profiled Gaussian keeps W scale-free (W = priorweights), so the data
    //     term has unit implicit scale and Vb = H⁻¹·σ̂².
    //   * Every other family folds its reciprocal dispersion / full Fisher
    //     information into W (Gamma W = prior/φ, Tweedie W = prior·μ^{2−p}/φ,
    //     Beta/NB the complete fixed-scale Fisher info, Poisson/Binomial φ ≡ 1),
    //     so H already equals the true penalized Hessian (identical to mgcv's
    //     XᵀW_sfX/φ + S_λ) and Vb = H⁻¹ with NO extra dispersion factor. A
    //     post-hoc ×φ here would double-count the dispersion and shrink every SE
    //     by √φ (= 1/√shape for Gamma); see #679.
    //
    // The single source of truth for this invariant is
    // `GlmLikelihoodSpec::coefficient_covariance_scale`; the response-level
    // observation noise used by predictive intervals stays in `dispersion`
    // above (a deliberately distinct quantity, e.g. 1/shape for Gamma).
    let cov_scale = pirls_res
        .likelihood
        .coefficient_covariance_scale(standard_deviation * standard_deviation)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let zero_covariance_boundary = dispersion.is_zero_estimate()
        && matches!(
            &pirls_res.likelihood.spec.response,
            ResponseFamily::Gaussian
        )
        && matches!(
            &pirls_res.likelihood.scale,
            LikelihoodScaleMetadata::ProfiledGaussian
        );
    if !cov_scale.is_finite()
        || cov_scale < 0.0
        || (cov_scale == 0.0 && !zero_covariance_boundary)
        || (zero_covariance_boundary && cov_scale != 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "coefficient covariance scale {cov_scale:?} is inconsistent with dispersion {dispersion:?}"
        )));
    }

    // A fit carrying inequality constraints reports the mean of its truncated
    // Laplace posterior, never the boundary MAP. Build the posterior identity
    // in the transformed PIRLS frame, where the accepted Hessian factor,
    // constraint rows, score, and mode are exactly aligned, then lift its two
    // locations and low-rank covariance factor through Qs together.
    //
    // This work is independent of `compute_inference`: requesting standard
    // errors cannot change the fitted coefficient vector. It needs q+1 solves,
    // not a dense p×p inverse.
    let constrained_posterior = match pirls_res.linear_constraints_transformed.as_ref() {
        Some(constraints) => {
            let factor = edf_factor.as_ref().ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "constrained posterior geometry requires the accepted Hessian factor"
                        .to_string(),
                )
            })?;
            let h = &pirls_res.stabilizedhessian_transformed;
            let constraint_rhs = constraints.a.t().to_owned();
            let sigma_at_unscaled = factor.solvemulti(&constraint_rhs).map_err(|reason| {
                EstimationError::RemlOptimizationFailed(format!(
                    "constrained posterior normal solve failed: {reason}"
                ))
            })?;
            certify_factorized_inference_solve(
                h,
                &constraint_rhs,
                &sigma_at_unscaled,
                "constrained posterior normal geometry",
            )?;
            let sigma_at = sigma_at_unscaled * cov_scale;

            let score_t = &pirls_res.penalized_gradient_transformed;
            let center_step_unscaled = factor.solve(score_t).map_err(|reason| {
                EstimationError::RemlOptimizationFailed(format!(
                    "constrained posterior score solve failed: {reason}"
                ))
            })?;
            certify_factorized_inference_vector_solve(
                h,
                score_t,
                &center_step_unscaled,
                "constrained posterior unconstrained centre",
            )?;
            // `penalized_gradient_transformed` and H share the solver's
            // objective scale. For profiled Gaussian both omit the common
            // 1/φ factor, which cancels in H⁻¹g; multiplying this displacement
            // by `cov_scale=φ` would move the Gaussian centre by an extra φ.
            let center_t = pirls_res.beta_transformed.as_ref() - &center_step_unscaled;
            let mut correction =
                crate::constrained_posterior::constrained_posterior_correction(
                    sigma_at.view(),
                    &center_t,
                    constraints,
                )
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior moments failed: {reason}"
                    ))
                })?;
            let qs = &pirls_res.reparam_result.qs;
            if let Some(value) = correction.as_mut() {
                value.lift = qs.dot(&value.lift);
            }
            let constraints_internal = fit_linear_constraints.as_ref().ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "PIRLS exported transformed inequalities without their pre-reparameterization geometry"
                        .to_string(),
                )
            })?;
            Some(crate::constrained_posterior::ConstrainedPosteriorGeometry::with_moments(
                constraints_internal.clone(),
                beta_orig_internal.clone(),
                qs.dot(&center_t),
                correction,
            ))
        }
        None if needs_constrained_posterior => {
            return Err(EstimationError::RemlOptimizationFailed(
                "fit accepted linear inequalities but PIRLS did not export their transformed geometry"
                    .to_string(),
            ));
        }
        None => None,
    };
    let reported_beta_orig_internal = match constrained_posterior.as_ref() {
        Some(posterior) => posterior.posterior_mean().map_err(|reason| {
            EstimationError::RemlOptimizationFailed(format!(
                "constrained posterior mean is unavailable: {reason}"
            ))
        })?,
        None => beta_orig_internal.clone(),
    };

    // Re-install the exact rho point and inner state that will be shipped, and
    // verify it IS the certified optimum. Seeds and nuisance refinements may
    // initialize work, but they can never promote a different point under the
    // optimizer's old certificate.
    //
    // The identity check is BITWISE on ρ, not a re-judged gradient norm: the
    // retained certificate is the analytic stationarity authority minted at
    // `outer_result.rho` by the full certification machinery (noise-floor
    // widenings, flatness probes, asymptote rails). In the deep-smoothing
    // regime the analytic gradient is a noise instrument (|Pg| redraws across
    // evaluations of the SAME point — the reproducibility floor exists because
    // of it), so re-drawing it once here and comparing against the certified
    // band refuses honest noise-band certificates with coin-flip probability
    // while adding nothing to point-identity (which bit equality decides
    // exactly). The evaluation itself is kept: it installs the inner state at
    // the shipped point and supplies the shipped value/gradient fields.
    let (final_value, finalgrad, finalgrad_norm) = if final_rho.is_empty() {
        (outer_result.final_value, Array1::zeros(0), 0.0)
    } else {
        let (value, gradient) = reml_state.compute_cost_and_gradient(&final_rho)?;
        let lower = Array1::from_elem(final_rho.len(), -crate::estimate::RHO_BOUND);
        let upper = Array1::from_elem(final_rho.len(), crate::estimate::RHO_BOUND);
        // Shipped as the result's `final_grad_norm` and reported in the
        // refusal below, so it uses the certificate's rail-relaxed box (#2412)
        // -- the same projection the certified |Pg| was measured with, even
        // though this gate never weighs one against the other.
        // #2545: "the same projection the certified |Pg| was measured with" now
        // includes the certificate's removal of the soft rho-guard barrier on
        // railed coordinates, so the shipped number keeps meaning the same thing
        // as the certified one. `finalgrad` itself stays the raw criterion
        // gradient — this is the projected residual, not the gradient.
        let bounds = (lower, upper);
        let barrier = reml_state.soft_rho_guard_gradient(&final_rho);
        let projected = crate::rho_optimizer::rail_projected_gradient_norm(
            &final_rho,
            &crate::rho_optimizer::gradient_with_rail_barrier_removed(
                &final_rho,
                &gradient,
                &crate::rho_optimizer::rail_relaxed_bounds(&bounds),
                Some(&barrier),
            ),
            Some(&bounds),
        );
        (value, gradient, projected)
    };
    let shipped_point_is_certified = shipped_joint_point_is_certified(
        &final_rho,
        &final_link_coords,
        &outer_result.rho,
    );
    let certificate_valid = final_rho.is_empty()
        || (outer_result.converged()
            && outer_result
                .criterion_certificate
                .as_ref()
                .is_some_and(|certificate| certificate.certifies())
            && shipped_point_is_certified
            && finalgrad_norm.is_finite());
    if !certificate_valid {
        return Err(EstimationError::RemlDidNotConverge {
            context: "standard REML final shipped point".to_string(),
            reason: format!(
                "post-fit certificate identity check failed: shipped point {:?} vs \
                 certified point {:?} (converged={}, certifies={}, |Pg| at shipped point {:.3e})",
                final_rho
                    .iter()
                    .chain(final_link_coords.iter())
                    .copied()
                    .collect::<Vec<_>>(),
                outer_result.rho.to_vec(),
                outer_result.converged(),
                outer_result
                    .criterion_certificate
                    .as_ref()
                    .is_some_and(|certificate| certificate.certifies()),
                finalgrad_norm,
            ),
            iterations: outer_result.iterations,
            final_value,
            projected_grad_norm: finalgrad_norm.is_finite().then_some(finalgrad_norm),
            // This gate deliberately does NOT weigh the re-drawn gradient
            // against the certified band — see the comment above the
            // re-evaluation: in the deep-smoothing regime that comparison
            // refuses honest noise-band certificates by coin flip. What it
            // checks is bitwise point identity plus the certificate's own
            // verdict. Printing a bound beside "against stationarity bound"
            // therefore named a comparison this route does not make
            // (#2458/#2465).
            stationarity_standard: gam_problem::StationarityStandard::NoComparison,
            rho_checkpoint: final_rho.to_vec(),
        });
    }
    outer_result.final_value = final_value;
    outer_result.final_gradient = Some(finalgrad);
    outer_result.final_grad_norm = Some(finalgrad_norm);
    let outer_converged = true;

    if opts.compute_inference || needs_constrained_posterior {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
    }
    if opts.compute_inference {
        let p_cov = penalized_hessian.nrows();
        let qs = &pirls_res.reparam_result.qs;

        // Auto-select covariance strategy from the runtime resource policy.
        //
        // When the WHOLE simultaneous dense bundle fits the policy's
        // process-wide reservation (`reserve_dense_covariance_bundle`) we can
        // afford the full p×p inverse: O(p³) compute, O(p²) memory. The full
        // matrix is needed for the frequentist covariance Ve = H⁻¹ X'WX H⁻¹ φ,
        // the influence matrix F = H⁻¹ X'WX, and the smoothing-parameter
        // correction.
        //
        // For large models we use solve-on-demand against the Cholesky factor
        // already computed for EDF traces above. We solve H_t Z_t = Qs^T in
        // policy-sized column chunks, then extract the diagonal of
        // Qs · Z_t = H_orig⁻¹ to get exact posterior SEs without ever
        // materialising the p×p inverse. Prediction bands continue to work via
        // the factorised-Hessian path in PredictionCovarianceBackend::Factorized.

        // Attempt the full inverse when the bundle fits the policy budget.
        //
        // ONE strict Cholesky serves the whole bundle. `H⁻¹` itself is still
        // materialized — it IS the posterior covariance `Vb = φ·H⁻¹`, an
        // estimand rather than an intermediate — but every DERIVED quantity
        // (`F`, `Ve`, the bias-correction Jacobian) is obtained by solving
        // against this factor instead of by multiplying against `H⁻¹`. The
        // certificate on `H⁻¹` is a *backward* error bound, so a product
        // `H⁻¹·M` carries a forward error of order `cond(H)` times it; a solve
        // `H·X = M` carries the backward error only. #2668 measured
        // `cond(H) = 2.099e8` on an ordinary Gaussian `y ~ s(x)` fit, where
        // that amplification put `H·F` a measured 3.9% away from the `X'WX`
        // it is definitionally equal to.
        let posterior_factor = if dense_covariance_reservation.is_some() {
            Some(
                gam_linalg::utils::certified_spd_factorize(
                    &penalized_hessian,
                    "posterior covariance",
                )
                .map_err(|error| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "posterior covariance requires an exact SPD Hessian factorization: {error}"
                    ))
                })?,
            )
        } else {
            None
        };
        let beta_covariance_unscaled: Option<Array2<f64>> = match posterior_factor.as_ref() {
            Some(factor) => Some(
                factor
                    .inverse()
                    .map(gam_linalg::utils::CertifiedSpdInverse::into_inverse)
                    .map_err(|error| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "posterior covariance requires an exact SPD Hessian inverse: {error}"
                        ))
                    })?,
            ),
            None => None,
        };

        if let (Some(h_inv), Some(posterior_factor)) =
            (beta_covariance_unscaled.as_ref(), posterior_factor.as_ref())
        {
            // Full inverse available: wrap as phi-scaled covariance, compute
            // frequentist quantities, and pass to smoothing-correction cubature.
            let mut posterior_covariance = scaled_covariance(h_inv.clone(), cov_scale);
            let constrained_correction = constrained_posterior
                .as_ref()
                .map(crate::constrained_posterior::ConstrainedPosteriorGeometry::correction)
                .transpose()
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior covariance correction is unavailable: {reason}"
                    ))
                })?
                .flatten();
            if let Some(correction) = constrained_correction {
                constrained_removed_variance = Some(correction.removed_variance_diagonal());
                constrained_diagonal_uncertainty = Some(correction.diagonal_uncertainty());
                // `Σ_π` is read for its DIAGONAL immediately below, and on a
                // pinned coordinate the subtractive form `Σ − GΔGᵀ` is a
                // cancellation whose residue carries a sign — measured at
                // `−3.09e-15` on `y ~ s(x, shape=convex)`, which the strict
                // `variance > 0` gate then refuses. Assemble the identical
                // quantity as `(P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ`, where the
                // diagonal is a sum of squares (#2705 group A).
                let constraints = constrained_posterior
                    .as_ref()
                    .map(|geometry| &geometry.constraints)
                    .ok_or_else(|| {
                        EstimationError::RemlOptimizationFailed(
                            "a constrained posterior correction exists without the geometry that \
                             owns its constraint system"
                                .to_string(),
                        )
                    })?;
                let truncated = correction
                    .truncated_covariance_psd(&posterior_covariance, constraints)
                    .map_err(|reason| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "constrained posterior covariance could not be assembled in its \
                             positive-semidefinite form: {reason}"
                        ))
                    })?;
                untruncated_conditional_covariance = Some(posterior_covariance);
                posterior_covariance = truncated;
            }
            beta_covariance = Some(gam_problem::dispersion_cov::PhiScaledCovariance::wrap(
                posterior_covariance,
            ));

            // Frequentist covariance Ve = F H⁻¹ φ and influence matrix F = H⁻¹ X'WX.
            // Both require the full unscaled inverse; computed in original basis.
            //
            // The canonical penalties live in the TRANSFORMED frame, while
            // `h_inv` is the ORIGINAL-basis inverse — assemble S(λ) in the
            // transformed frame and map it through the same congruence as the
            // Hessian (`S_orig = Qs·S_t·Qsᵀ`, issue #1027). Pairing the
            // transformed-frame S directly with the original-frame inverse made
            // `F` (and everything reconstructed from it) frame-inconsistent.
            let p_t = qs.ncols();
            let mut s_t = Array2::<f64>::zeros((p_t, p_t));
            for (kk, cp) in pirls_res
                .reparam_result
                .canonical_transformed
                .iter()
                .enumerate()
            {
                if kk >= lambdas.len() {
                    continue;
                }
                let r = &cp.col_range;
                let local = cp.local_ref();
                let lam = lambdas[kk];
                for i in 0..cp.block_dim() {
                    for j in 0..cp.block_dim() {
                        s_t[[r.start + i, r.start + j]] += lam * local[[i, j]];
                    }
                }
            }
            let mut s_mat = qs.dot(&s_t).dot(&qs.t());
            gam_linalg::matrix::symmetrize_in_place(&mut s_mat);

            // The frequentist bias-corrected coefficient used by prediction is
            // β_BC = β̂ + b̂ with b̂ = H⁻¹S(β̂ - μ) at fixed smoothing
            // parameters. Its fixed-ρ linearization with respect to β̂ is
            // A = I + H⁻¹S. Credible bands centered at β_BC must use the
            // covariance of that same estimator, A V Aᵀ; otherwise the center is
            // debiased but the reported uncertainty remains for the shrunken
            // penalized mode β̂, producing severely over-narrow bands on heavily
            // smoothed large-scale Duchon fits (#1870).
            // X'WX = H − S(λ) in the original basis — the genuine PSD weighted
            // Gram, reconstructed from the same `penalized_hessian` and `s_mat`
            // that define `F = H⁻¹X'WX` (issue #1027). Stored directly so the
            // WPS corrected-EDF correction never has to recover it from an
            // inconsistent `H·F` product.
            let mut xwx = &penalized_hessian - &s_mat;
            // `H·F = H(I − H⁻¹S) = H − S` is the RAW difference, but the gram
            // stored below is `sym(H − S)`. When `H` and `S` are both symmetric
            // those coincide and `H·F = X'WX` exactly; when they are not, this
            // `symmetrize_in_place` silently absorbs the difference and the
            // identity fails downstream with no way to see which operand caused
            // it (#2668 measures a 3.9% gap on `y ~ s(x)` and could only rule
            // out `H`: `max|H − Hᵀ| = 0.000e0` there). Report both asymmetries
            // at the one place that holds `s_mat`. `debug!` so it costs nothing
            // without a backend installed, and O(p²) beside the O(p³) work above.
            if log::log_enabled!(log::Level::Debug) {
                let asym = |m: &ndarray::Array2<f64>| {
                    let mut worst = 0.0_f64;
                    for i in 0..m.nrows() {
                        for j in 0..m.ncols() {
                            worst = worst.max((m[[i, j]] - m[[j, i]]).abs());
                        }
                    }
                    worst
                };
                let scale = xwx.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
                log::debug!(
                    "[WPS-GRAM #2668] max|H-H^T|={:.3e} max|S-S^T|={:.3e} \
                     max|H-S|={:.3e} (the stored gram is symmetrize(H-S); a \
                     non-zero S asymmetry is absorbed here and surfaces as \
                     H*F != X'WX)",
                    asym(&penalized_hessian),
                    asym(&s_mat),
                    scale
                );
            }
            gam_linalg::matrix::symmetrize_in_place(&mut xwx);

            // Influence matrix F = H⁻¹·X'WX, obtained by SOLVING `H·F = X'WX`
            // against the factor above rather than by forming `I − H⁻¹·S`.
            // The two are equal in real arithmetic; in floating point the solve
            // makes `H·F = X'WX` hold to the factorization's backward error
            // *by construction*, which is exactly the identity
            // `penalized_hessian_times_influence_equals_weighted_gram` asserts
            // and which the explicit-inverse form violated by 3.9% at
            // `cond(H) = 2.099e8` (#2668). Note the right-hand side is the
            // stored, symmetrized `xwx`, so the identity is asserted against
            // the matrix that is actually persisted.
            //
            // `F` is a product of two symmetric matrices and is therefore
            // generally NOT symmetric; it must not be symmetrized —
            // `gam_linalg::matrix::symmetrize_in_place(F)` both breaks the
            // H·F = X'WX consistency identity (so any downstream code that
            // reconstructs X'WX from H·F lands on an asymmetric/indefinite
            // matrix) AND corrupts the frequentist covariance `Ve = F·H⁻¹·φ`
            // (since (F_sym)·H⁻¹ ≠ H⁻¹·X'WX·H⁻¹) AND distorts the
            // Wood-corrected reference d.f. `tr(F_jj)² / tr(F_jj²)` consumed
            // by `smooth_test::reference_df` (tr(F²) ≠ tr(F_sym²) in general).
            // See issue #1027.
            let (f_mat, _f_certificate) = posterior_factor.solve_matrix(&xwx).map_err(|error| {
                EstimationError::RemlOptimizationFailed(format!(
                    "influence matrix solve H·F = X'WX did not certify: {error}"
                ))
            })?;

            // The frequentist bias-corrected coefficient used by prediction is
            // β_BC = β̂ + b̂ with b̂ = H⁻¹S(β̂ - μ) at fixed smoothing
            // parameters. Its fixed-ρ linearization with respect to β̂ is
            // A = I + H⁻¹S. Credible bands centered at β_BC must use the
            // covariance of that same estimator, A V Aᵀ; otherwise the center is
            // debiased but the reported uncertainty remains for the shrunken
            // penalized mode β̂, producing severely over-narrow bands on heavily
            // smoothed large-scale Duchon fits (#1870).
            //
            // `A = I + H⁻¹S = 2I − F` exactly, since `F = I − H⁻¹S`. Deriving
            // it from `F` rather than from a second product keeps the Jacobian
            // and the influence matrix on the identical `H⁻¹S`, and inherits
            // the solve's accuracy instead of the inverse's amplification.
            let mut bc_jac = f_mat.clone();
            bc_jac *= -1.0;
            for diagonal in 0..p_cov {
                bc_jac[[diagonal, diagonal]] += 2.0;
            }
            bias_correction_jacobian = Some(bc_jac);

            // Frequentist covariance Ve = H⁻¹·X'WX·H⁻¹·φ = φ·H⁻¹·Fᵀ (the
            // sandwich is symmetric, so `F·H⁻¹ = (H⁻¹·Fᵀ)`). Solving
            // `H·Z = Fᵀ` instead of multiplying `F·H⁻¹` gives the companion
            // identity `H·Ve·H = φ·X'WX` to backward error: `H·Z·H = Fᵀ·H =
            // (H·F)ᵀ = X'WX`. The explicit-inverse form carried the same
            // `cond(H)` amplification as `F` did, with no identity anywhere
            // that looked at it.
            let f_transpose = f_mat.t().to_owned();
            let (mut ve, _ve_certificate) =
                posterior_factor
                    .solve_matrix(&f_transpose)
                    .map_err(|error| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "frequentist covariance solve H·Ve/φ = Fᵀ did not certify: {error}"
                        ))
                    })?;
            ve *= cov_scale;
            gam_linalg::matrix::symmetrize_in_place(&mut ve);

            weighted_gram = Some(xwx);
            coefficient_influence = Some(f_mat);
            beta_covariance_frequentist = Some(ve);
        }

        // Smoothing-parameter correction (first-order delta + optional cubature).
        // The dense branch can return the complete matrix and optionally
        // upgrade it by cubature. On governor refusal the factorized branch
        // computes only diag(J V_rho J') from cached p×k mode responses; calling
        // `compute_smoothing_correction_auto(..., None, ...)` is not sufficient
        // because that routine constructs the full p×p first-order product
        // before it notices that the base covariance is absent.
        // `cov_scale` is the coefficient-covariance multiplier at the optimum
        // (σ̂² for profiled Gaussian, 1 for every weight-carries-dispersion
        // family). The cubature path multiplies its dispersion-free curvature
        // block `E_ρ[H(ρ)⁻¹] − H_opt⁻¹` by this scale so the FULL cubature
        // correction lands on the same c² variance scale as `Vb = cov_scale·H_opt⁻¹`
        // (#582); the var_beta = Cov_ρ[β̂] block is already on that scale and
        // stays unscaled.
        if beta_covariance_unscaled.is_some() {
            let no_outer_gradient = Array1::<f64>::zeros(0);
            let measured_hessian_error: Vec<
                gam_linalg::curvature_resolution::MeasuredHessianError,
            > = outer_result
                .criterion_hessian_error
                .as_ref()
                .filter(|measured| measured.restricted_to_leading(final_rho.len()).is_some())
                .map(|measured| measured.hessian_error_2norm())
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| {
                    vec![gam_linalg::curvature_resolution::MeasuredHessianError::new(
                        "outer-certificate criterion-vs-analytic curvature disagreement                          |v'Hv - d2V/dalpha2| along the disputed eigenvector",
                        value,
                    )]
                })
                .unwrap_or_default();
            let smoothing_outcome = reml_state.compute_smoothing_correction_auto(
                &final_rho,
                &lambdas,
                &pirls_res,
                beta_covariance_unscaled.as_ref(),
                cov_scale,
                finalgrad_norm,
                // #2428: the residual gradient the outer certificate itself
                // used to accept this ρ̂ is the resolution floor the ρ-Hessian's
                // definiteness must be judged against. Without it the
                // correction applies a strictly stronger standard than the
                // certificate did and can reject a fit the outer loop passed.
                outer_result
                    .final_gradient
                    .as_ref()
                    .unwrap_or(&no_outer_gradient),
                // #2748: the outer certificate does not only accept or refuse
                // this point -- when it disputes a negative curvature it
                // EVALUATES the criterion along that direction, and the
                // disagreement between what the criterion says the curvature is
                // and what the analytic Hessian claimed is a measured `||dH||_2`
                // for the assembly. `invert_identified_rho_hessian` is about to
                // judge the SAME matrix at the SAME point; without this it does
                // so against an eigensolver's backward error, which bounds the
                // decomposition and says nothing about the assembly, and refuses
                // fits this certificate accepted (#2428).
                //
                // The direction must lie inside the rho block for the bound to
                // transfer: `|v'(dH)v| <= ||dH||_2` is about the sub-block's own
                // error only when `v` has no component outside it. A wider theta
                // whose disputed direction reaches into psi yields no component
                // here, which is an absent measurement, not a zero.
                &measured_hessian_error,
            )?;
            match smoothing_outcome {
                super::reml::eval::SmoothingCorrectionOutcome::Unavailable { reason, .. } => {
                    // A fit certified at an infinite-smoothing rail (typed
                    // AsymptoteRail, or box-railed coordinates) has NO finite
                    // ρ-variance along the rail direction — the outer Hessian
                    // is legitimately non-PD there, so the first-order
                    // smoothing correction is TYPED-unavailable rather than a
                    // defect. Ship the certified fit with the plug-in
                    // covariance and no correction; the downstream corrected
                    // EDF/AIC channels report the typed absence (#946/#1027)
                    // instead of the whole fit dying over an enhancement. A
                    // fit WITHOUT rail evidence keeps the fail-loud error: an
                    // unexpectedly uninvertible outer Hessian on a
                    // well-conditioned interior optimum is a real defect.
                    let rail_certified =
                        outer_result
                            .criterion_certificate
                            .as_ref()
                            .is_some_and(|certificate| {
                                matches!(
                            certificate.stationarity,
                            crate::model_types::OuterStationarityCertificate::AsymptoteRail { .. }
                        ) || !certificate.lambdas_railed.is_empty()
                            });
                    // The same typed absence applies when the outer Hessian has
                    // no analytic form for this fit at all (a non-canonical
                    // Firth link, routed to BFGS): nothing about the optimum is
                    // suspect, the correction simply cannot be formed, and the
                    // fit was accepted with that link on purpose (#2158).
                    let structurally_not_analytic = matches!(
                        reason,
                        crate::estimate::smoothing_correction::SmoothingCorrectionUnavailable::OuterHessianNotAnalytic { .. }
                    );
                    if !rail_certified && !structurally_not_analytic {
                        return Err(EstimationError::InvalidInput(format!(
                            "exact smoothing-corrected covariance unavailable: {reason:?}"
                        )));
                    }
                    log::info!(
                        "[SMOOTHING-CORRECTION] typed-unavailable on a {} fit ({reason:?}); \
                         shipping the plug-in covariance without a smoothing correction",
                        if rail_certified { "rail-certified" } else { "non-analytic-outer-Hessian" }
                    );
                    rho_covariance = None;
                    smoothing_correction = None;
                    smoothing_correction_method = None;
                    smoothing_correction_first_order = None;
                    smoothing_correction_method_first_order = None;
                }
                outcome => {
                    rho_covariance = outcome.rho_covariance().cloned();
                    (
                        smoothing_correction,
                        smoothing_correction_method,
                        smoothing_correction_first_order,
                        smoothing_correction_method_first_order,
                    ) = outcome.into_correction_with_method();
                }
            }
        }

        // Tier-0 marginal-smoothing certificate (#938): while the REML objective
        // is still live, sample the outer criterion around the converged ρ̂ to
        // read the PSIS k̂ that says whether the plug-in + first-order V_ρ
        // correction is adequate. This is the objective-lifecycle seam — the
        // certificate runs against the SAME objective the fit converged on, so
        // its criterion is the fit's own bit-for-bit (no retain/rebuild). Absent
        // when there are no smoothing parameters or the outer Hessian is
        // unavailable; never fatal.
        //
        // The Tier-0 certificate is CHEAP (a handful of outer-criterion
        // evaluations) so it is emitted regardless of `skip_rho_posterior_inference`
        // whenever it is available (#1810) — the standard formula/CLI fit surfaces
        // its ρ-posterior certificate by default. Only the EXPENSIVE escalation
        // tiers (Tier-1 quadrature / Tier-2 NUTS over ρ) are gated by the flag:
        // interactive formula/CLI fits keep `skip_rho_posterior_inference = true`
        // so a fit that fails to certify plug-in never turns into a sampler
        // benchmark, while lower-level callers that opt in (`skip = false`) get
        // the auto-selected escalation tier (quadrature for K≤4, NUTS over ρ for
        // K≤16, honest Unavailable beyond) at this same live seam.
        (rho_posterior_certificate, rho_posterior_escalation) = reml_state.rho_posterior_inference(
            &final_rho,
            !opts.skip_rho_posterior_inference,
            None,
        );

        // Standard errors: prefer the diagonal of the full inverse when
        // available; otherwise use the factorised Hessian from the EDF pass
        // (in transformed basis) to compute exact diagonal of H_orig⁻¹ =
        // Qs H_t⁻¹ Qs' via chunked solve-on-demand. The chunk width comes
        // from the runtime resource policy's per-chunk byte target: each
        // chunk keeps ~2 dense p×chunk workspaces (the RHS slice and the
        // solved block) live at once.
        let resource_policy = gam_runtime::resource::ResourcePolicy::for_problem(
            gam_runtime::resource::ProblemHints::default(),
        );
        let governor = gam_runtime::resource::MemoryGovernor::global();
        let se_chunk_target_bytes = resource_policy
            .row_chunk_target_bytes
            .min(governor.remaining_bytes());
        let se_chunk_cols = gam_runtime::resource::rows_for_target_bytes(
            se_chunk_target_bytes,
            qs.ncols().saturating_mul(2),
        );
        beta_standard_errors = if beta_covariance_unscaled.is_some() {
            // The dense covariance already includes the inequality-truncation
            // correction. Derive SEs from that same matrix so the dense and
            // factorized representations cannot disagree.
            let covariance = beta_covariance.as_ref().ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "dense posterior covariance was not retained for standard errors".to_string(),
                )
            })?;
            let mut raw_se = Array1::<f64>::zeros(p_cov);
            // Why an inequality-truncated covariance may show an exactly-zero
            // diagonal, and why that is a measurement rather than a defect
            // (#2705 group A).
            //
            // For an UNCONSTRAINED fit `Σ = φ·H⁻¹` with `H` SPD, so every
            // diagonal entry is strictly positive and a zero would mean the
            // Hessian is singular — which this gate exists to catch, and still
            // does. A TRUNCATED one is a different object: the constraint
            // removes the coordinate's variance along its own normal, and the
            // λ → ∞ limit of that removal is exactly zero. The Gram assembly
            // above computes that limit as a sum of squares, so it reports the
            // clean `0.0` instead of the `±ε·Σ_ii` rounding residue the
            // subtraction used to leave — and a strict `> 0` test would then
            // refuse the fit for producing the right answer.
            let truncation_applied = constrained_removed_variance.is_some();
            for (index, &variance) in covariance.as_array().diag().iter().enumerate() {
                let valid = if zero_covariance_boundary {
                    variance == 0.0
                } else if truncation_applied {
                    variance.is_finite() && variance >= 0.0
                } else {
                    variance.is_finite() && variance > 0.0
                };
                if !valid {
                    let removed = constrained_removed_variance
                        .as_ref()
                        .and_then(|d| d.get(index).copied())
                        .map_or("n/a".to_string(), |v| format!("{v:.6e}"));
                    let allowance = constrained_diagonal_uncertainty
                        .as_ref()
                        .and_then(|d| d.get(index).copied())
                        .map_or("n/a".to_string(), |v| format!("{v:.6e}"));
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "posterior covariance diagonal {index} is not positive and representable: \
                         {variance:?} [#2705 attribution: removed_variance_diag={removed} \
                         cubature_allowance={allowance} truncation_applied={truncation_applied}]"
                    )));
                }
                raw_se[index] = variance.sqrt();
            }
            Some(raw_se)
        } else if let Some(ref factor_t) = edf_factor {
            // Solve-on-demand: process columns of Qs^T in chunks.
            // Qs is (p_cov × p_t) orthogonal. H_orig⁻¹ = Qs H_t⁻¹ Qs'.
            // (H_orig⁻¹)_{ii} = Qs[i,:] · H_t⁻¹ · Qs[i,:]'
            // Batch: column i of Qs^T is row i of Qs. Solve H_t Z = Qs^T[:,chunk]
            // then dot each solution column back with the corresponding Qs row.
            if se_chunk_cols == 0 {
                return Err(EstimationError::RemlOptimizationFailed(
                    "resource policy cannot admit even one exact factorized coefficient-SE column"
                        .to_string(),
                ));
            }
            let mut diag_inv = Array1::<f64>::zeros(p_cov);
            let mut col_start = 0usize;
            while col_start < p_cov {
                let col_end = (col_start + se_chunk_cols).min(p_cov);
                let chunk = col_end - col_start;
                let chunk_reservation = governor
                    .try_reserve_dense_f64_copies(
                        qs.ncols(),
                        chunk,
                        2,
                        "factorized coefficient-SE solve chunk",
                    )
                    // The typed refusal carries the budget, what was already
                    // reserved, and the availability observation the budget was
                    // derived from. Discarding it left two runs that refused for
                    // different reasons indistinguishable in the log, which is
                    // half of why #2702 took a filed issue to diagnose: state the
                    // measured quantities, not just the verdict.
                    .map_err(|refusal| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "resource policy refused exact coefficient-SE columns \
                             {col_start}..{col_end} ({chunk} of {p_cov} columns, \
                             {p_t} transformed rows): {refusal}",
                            p_t = qs.ncols(),
                        ))
                    })?;
                // qs.t() has shape (p_t, p_cov); slice to (p_t, chunk). The
                // reservation covers this buffer and its `solvemulti` output
                // jointly, so it is bound to whichever one outlives the other
                // (both are dropped together at the end of this iteration).
                let rhs = chunk_reservation
                    .bind(qs.t().slice(ndarray::s![.., col_start..col_end]).to_owned());
                let z_chunk = factor_t.solvemulti(&rhs).map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "exact coefficient-SE solve failed at columns {col_start}..{col_end}: {reason}"
                    ))
                })?;
                certify_factorized_inference_solve(
                    &pirls_res.stabilizedhessian_transformed,
                    &rhs,
                    &z_chunk,
                    "factorized coefficient standard errors",
                )?;
                // z_chunk is (p_t × chunk).
                // (H_orig⁻¹)_{ii} = qs.row(i) · z_chunk.column(i - col_start)
                for local_i in 0..chunk {
                    let global_i = col_start + local_i;
                    let qs_row = qs.row(global_i);
                    let z_col = z_chunk.column(local_i);
                    diag_inv[global_i] = qs_row.dot(&z_col);
                }
                col_start = col_end;
            }
            let removed_variance = constrained_posterior
                .as_ref()
                .map(crate::constrained_posterior::ConstrainedPosteriorGeometry::correction)
                .transpose()
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior variance correction is unavailable: {reason}"
                    ))
                })?
                .flatten()
                .map(|correction| correction.removed_variance_diagonal())
                .unwrap_or_else(|| Array1::<f64>::zeros(p_cov));
            let mut se = Array1::<f64>::zeros(p_cov);
            for (index, &variance_unscaled) in diag_inv.iter().enumerate() {
                if !(variance_unscaled.is_finite() && variance_unscaled > 0.0) {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "exact factorized SPD inverse has invalid diagonal {index}: {variance_unscaled:?}"
                    )));
                }
                let base = cov_scale * variance_unscaled;
                let removed = removed_variance[index];
                let variance = base - removed;
                // #2705 group A. The dense branch assembles this quantity as a
                // sum of squares and cannot produce a negative variance; here
                // there is no dense `Σ` to factor, so the subtraction stands —
                // and on a coordinate the constraint pins, `removed` cancels
                // `base` to the last digit and the residue carries a sign.
                //
                // The resolution of that residue is a MEASURED quantity, not a
                // chosen one: `base` and `removed` are each accurate to a
                // relative rounding error, so their difference is accurate to
                // `~ε·max(base, removed)` in ABSOLUTE terms — which is the whole
                // of the answer once the removal is complete. A residue inside
                // that band is the zero it is approximating (the λ → ∞ limit of
                // the truncation, the only value it can be). A residue outside
                // it is a real negative variance and is refused, with the
                // decomposition attached so the next reader does not have to
                // re-derive which producer overran.
                let subtraction_resolution =
                    16.0 * f64::EPSILON * base.abs().max(removed.abs());
                let variance = if variance < 0.0 && -variance <= subtraction_resolution {
                    0.0
                } else {
                    variance
                };
                let valid = if zero_covariance_boundary {
                    variance == 0.0
                } else {
                    variance.is_finite() && variance >= 0.0
                };
                if !valid {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "factorized posterior variance {index} is not positive and \
                         representable: {variance:?} [#2705 attribution: base={base:.6e} \
                         removed_variance_diag={removed:.6e} \
                         subtraction_resolution={subtraction_resolution:.6e}]"
                    )));
                }
                se[index] = variance.sqrt();
            }
            Some(se)
        } else {
            None
        };

        // Vp = Vb + J·V_ρ·Jᵀ, both terms on the SAME dispersion (variance) scale.
        //
        // The smoothing correction is built from the coefficient sensitivities
        // J = dβ̂/dρ = −H⁻¹(λ_k S_k(β̂ − μ_k)), which are linear in β̂, and from
        // V_ρ = (∇²_ρρ V)⁻¹. Under a Gaussian rescaling y → c·y the fit is exactly
        // equivariant: β̂ → c·β̂ (so J → c·J), H is response-scale-invariant, the
        // REML/LAML cost gains only a ρ-independent (n/2)·log(c²) offset (so its
        // ρ-gradient and ρ-Hessian — hence V_ρ — are dispersion-free), and φ̂ → c²·φ̂.
        // Therefore J·V_ρ·Jᵀ ∝ c · c⁰ · c = c², i.e. the correction is already on
        // the c² variance scale, exactly like Vb = φ̂·H⁻¹ ∝ c². It must be added
        // directly to Vb. Multiplying it by cov_scale
        // (≈ c²) again would make the correction scale as c⁴, inflating every
        // predict() interval for large-magnitude responses (#582). cov_scale is
        // applied once, where it belongs: in Vb = scaled_covariance(H⁻¹, cov_scale).
        //
        // #2705 group A — WHICH `Vb` the sum starts from, when the fit carries
        // inequality constraints.
        //
        // `beta_covariance` is the ρ̂-CONDITIONAL posterior covariance and, for a
        // constrained fit, it has already been truncated to the feasible set:
        // `Σ_π = Σ − GΔGᵀ`. Adding `J·V_ρ·Jᵀ` to THAT produced a matrix that is
        // the truncation of neither covariance:
        //
        //     (Σ − GΔGᵀ) + (Vp − Σ)  =  Vp − GΔGᵀ,
        //
        // with `G` and `Δ` derived from `Σ`, not from `Vp`. Along a coordinate
        // the constraint pins, `(GΔGᵀ)_ii` cancels `Σ_ii` to eleven digits, so
        // whatever `(Vp − Σ)_ii` happens to be becomes the WHOLE reported
        // variance — and `Vp − Σ` is a legitimately sign-indefinite second-order
        // increment (the cubature branch is `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂] − φ̂·H_opt⁻¹`,
        // a difference of two averages, PSD only as a SUM with `Vb`). On
        // `y ~ s(x, shape=convex)` that left `Σ_ii = 2.30e-2` truncated to
        // `6.23e-13` with a `−3.03e-9` smoothing increment on top, i.e. a
        // materially negative published variance, and `se_from_covariance`
        // refused the fit.
        //
        // The correct composition follows from the estimand. The feasible set
        // constrains β and says nothing about ρ, so the indicator `1_C(β)`
        // factors straight out of the ρ-integral:
        //
        //     ∫ π(β,ρ|y)·1_C(β) dρ  =  1_C(β)·∫ π(β,ρ|y) dρ,
        //
        // i.e. the β-marginal of the TRUNCATED joint posterior is exactly the
        // truncation of the β-marginal of the untruncated one. So the truncation
        // belongs on `Vp`, applied last, with its own `G_p = Vp·Aᵀ·W_p⁻¹` and its
        // own orthant moments at `W_p = A·Vp·Aᵀ` — not inherited from `Σ`.
        //
        // Two properties come with it, both of which the old order lacked: the
        // published matrix is a genuine truncated-Gaussian covariance, so it sits
        // between `P·Vp·Pᵀ ⪰ 0` and `Vp` instead of below both; and the
        // constraint's effect on the reported interval is measured at the width
        // the interval actually has, rather than at the conditional width.
        //
        // The ρ̂-CONDITIONAL `beta_covariance` keeps its own truncation at `Σ` —
        // that one is right, because that estimand really is conditional on ρ̂.
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.as_array().dim() == corr.dim() => {
                let mut corrected = untruncated_conditional_covariance
                    .as_ref()
                    .unwrap_or_else(|| base_cov.as_array())
                    .clone();
                corrected += corr;
                let truncation = match constrained_posterior.as_ref() {
                    Some(geometry) => {
                        apply_marginal_constraint_truncation(geometry, &mut corrected)?
                    }
                    None => Ok(()),
                };
                match truncation {
                    Ok(()) => {
                        if let Some(a_bc) = bias_correction_jacobian.as_ref() {
                            if a_bc.dim() != corrected.dim() {
                                return Err(EstimationError::RemlOptimizationFailed(format!(
                                    "bias-correction Jacobian shape {:?} does not match corrected covariance {:?}",
                                    a_bc.dim(),
                                    corrected.dim()
                                )));
                            }
                            // The truncation is a statement about the law of β;
                            // `A_bc` then maps that law to the bias-corrected
                            // estimator `β_BC = A_bc·β` (#1870). Truncating
                            // first and mapping second is the only order in
                            // which the feasible set is applied in the frame it
                            // was declared in.
                            corrected = a_bc.dot(&corrected).dot(&a_bc.t());
                        }
                        gam_linalg::matrix::symmetrize_in_place(&mut corrected);
                        Some(corrected)
                    }
                    Err(reason) => {
                        log::warn!(
                            "[CONSTRAINED-Vp] the smoothing-corrected covariance could not be \
                             truncated to the feasible set ({reason}); publishing the typed \
                             absence rather than an untruncated marginal, which would over-state \
                             every constrained interval. The rho-hat-conditional covariance is \
                             unaffected."
                        );
                        None
                    }
                }
            }
            (Some(base), Some(corr)) => {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "base covariance shape {:?} does not match smoothing correction {:?}",
                    base.as_array().dim(),
                    corr.dim()
                )));
            }
            _ => None,
        };
        beta_standard_errors_corrected = beta_covariance_corrected
            .as_ref()
            .map(se_from_covariance)
            .transpose()
            .map_err(|error| {
                // #2705 group A. Three shape-constrained fits die here with one
                // byte-identical message that names the CONSUMER's budget and
                // nothing else, so the refusal cannot say which of the three
                // producers summed into this diagonal overran, or by how much.
                // The matrix being read is
                //     Σ = φH⁻¹  −  GΔGᵀ  +  J V_ρ Jᵀ   (then optionally A·Σ·Aᵀ)
                // and only the first term is accurate to floating point. Print
                // the decomposition of the offending entry against each
                // producer's OWN declared resolution.
                let detail = match &error {
                    gam_problem::CovarianceStandardErrorError::NegativeDiagonal {
                        index,
                        value,
                        tolerance,
                    } => {
                        let removed = constrained_removed_variance
                            .as_ref()
                            .and_then(|d| d.get(*index).copied());
                        let allowance = constrained_diagonal_uncertainty
                            .as_ref()
                            .and_then(|d| d.get(*index).copied());
                        let base = beta_covariance
                            .as_ref()
                            .and_then(|c| c.as_array().diag().get(*index).copied());
                        let smoothing = smoothing_correction
                            .as_ref()
                            .and_then(|c| c.diag().get(*index).copied());
                        format!(
                            " [#2705 attribution: index={index} value={value:.17e} arithmetic_tolerance={tolerance:.6e} post_constrained_diag={} smoothing_correction_diag={} removed_variance_diag={} cubature_allowance={} inside_cubature_allowance={} congruence_applied={}]",
                            base.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            smoothing.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            removed.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            allowance.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            allowance.map_or("unknown".to_string(), |a| (-value <= a).to_string()),
                            bias_correction_jacobian.is_some(),
                        )
                    }
                    _ => String::new(),
                };
                EstimationError::RemlOptimizationFailed(format!(
                    "corrected coefficient covariance is not a valid standard-error source: {error}{detail}"
                ))
            })?;
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        penalty_block_trace,
        edf_total,
        smoothing_correction,
        smoothing_correction_method,
        smoothing_correction_first_order,
        smoothing_correction_method_first_order,
        penalized_hessian: penalized_hessian.clone().into(),
        reparam_qs: Some(pirls_res.reparam_result.qs.clone()),
        dispersion,
        beta_covariance,
        beta_standard_errors,
        beta_covariance_corrected,
        beta_standard_errors_corrected,
        beta_covariance_frequentist,
        coefficient_influence,
        weighted_gram,
        bias_correction_beta,
        bias_correction_jacobian,
    });

    let pirls_status = pirls_res.status;
    let likelihood_scale_field = pirls_res.likelihood.scale;

    // Report the fitted dispersion parameter on the family variant for the two
    // families whose *reporting log-likelihood kernel* reads it from the family
    // enum rather than from `likelihood_scale`: Negative-Binomial `theta` (issue
    // #802) and Beta `phi` (issue #1608). For both, `ResponseFamily` carries the
    // parameter directly (`NegativeBinomial { theta }`, `Beta { phi }`), the
    // PIRLS deviance/log-likelihood arms read it off that variant, and the inner
    // solve updated the family variant in lock-step with the scale metadata via
    // `with_negbin_theta` / `with_beta_phi`. But `opts.family` is the *seed* spec
    // (θ/φ at their construction defaults), so cloning it and stopping there would
    // ship the seed dispersion in the saved model while `likelihood_scale` carries
    // the fitted value — the two views diverge and the kernel reads the seed.
    // Threading the fitted dispersion back onto the reported family restores the
    // `with_negbin_theta` / `with_beta_phi` invariant (family variant ⇔ scale
    // metadata are two synchronized views of one estimated parameter) in the
    // terminal output, so every consumer — the diagnose AIC/PSIS-LOO kernel
    // included — sees the data's dispersion instead of the seed.
    //
    // Gamma shape and Tweedie φ are deliberately NOT threaded here: their family
    // variants carry no dispersion (`Gamma` is parameterless, `Tweedie { p }`
    // carries only the power), so their kernels read the fitted scale from
    // `likelihood_scale` directly and there is nothing on the family to sync.
    let mut reported_family = opts.family.clone();
    match likelihood_scale_field {
        LikelihoodScaleMetadata::EstimatedNegBinTheta {
            theta: fitted_theta,
        } => {
            if let ResponseFamily::NegativeBinomial { theta, .. } = &mut reported_family.response {
                *theta = fitted_theta;
            }
        }
        LikelihoodScaleMetadata::EstimatedBetaPhi { phi: fitted_phi } => {
            if let ResponseFamily::Beta { phi } = &mut reported_family.response {
                *phi = fitted_phi;
            }
        }
        // Every other scale metadata is either fixed (nothing was estimated to
        // thread back), or belongs to a family whose variant carries no
        // dispersion at all — Gamma shape and Tweedie φ live only on
        // `likelihood_scale`, as the comment above records. Enumerated so a new
        // estimated-dispersion metadata has to declare here whether its family
        // variant needs syncing.
        LikelihoodScaleMetadata::ProfiledGaussian
        | LikelihoodScaleMetadata::FixedDispersion { .. }
        | LikelihoodScaleMetadata::FixedGammaShape { .. }
        | LikelihoodScaleMetadata::EstimatedGammaShape { .. }
        | LikelihoodScaleMetadata::FixedBetaPhi { .. }
        | LikelihoodScaleMetadata::EstimatedTweediePhi { .. }
        | LikelihoodScaleMetadata::FixedNegBinTheta { .. }
        | LikelihoodScaleMetadata::Unspecified => {}
    }
    // The fully-normalized reporting kernel (#2096) reads a CONCRETE dispersion
    // `φ = σ̂²` for Gaussian off `likelihood.scale`. A profiled Gaussian carries
    // only the `ProfiledGaussian` marker (`fixed_phi() == None`), which the
    // kernel maps to NaN by contract (the #1583 no-silent-`φ=1` rule) — so the
    // reported `log_likelihood` (and the AIC built from it) came out NaN for
    // every non-degenerate Gaussian fit. Resolve a positive profiled residual
    // scale `σ̂²` into the reporting spec exactly. The validated boundary
    // estimate `σ̂² = 0` deliberately stays `ProfiledGaussian`: an ordinary
    // normalized Lebesgue density does not exist there, and relabeling it as a
    // positive fixed dispersion would falsify both provenance and density. This
    // is a REPORTING-only substitution: the persisted `likelihood_scale` field
    // below stays `ProfiledGaussian` so downstream consumers still see that the
    // scale was profiled, not user-fixed.
    let reporting_scale = match (&reported_family.response, likelihood_scale_field) {
        (ResponseFamily::Gaussian, LikelihoodScaleMetadata::ProfiledGaussian)
            if !zero_covariance_boundary =>
        {
            LikelihoodScaleMetadata::FixedDispersion {
                phi: standard_deviation * standard_deviation,
            }
        }
        _ => likelihood_scale_field,
    };
    let reported_likelihood = GlmLikelihoodSpec {
        spec: reported_family.clone(),
        scale: reporting_scale,
    };
    // At the validated boundary `σ̂² = 0` the fit reproduces the adjusted
    // response exactly, and no ordinary normalized Lebesgue density exists
    // there — so there is no finite FULL log-likelihood to evaluate, and the
    // kernel below must not be asked for one.
    //
    // Report it the way the deterministic-Gaussian route already reports this
    // same boundary: the value `0` under `UserProvided`, which DECLINES to
    // claim a normalized density rather than fabricating one. That is the
    // established convention for this state, not a new one.
    //
    // This replaces a hard refusal that told the caller to "use the dedicated
    // deterministic-Gaussian shortcut". That shortcut is dispatched by a
    // predicate living at ONE entry point (the formula path), so every other
    // entry — the term-collection entries reach this solver directly — had no
    // way to take the advice and died here instead. The condition is DETECTED
    // here, where the dispersion has actually been estimated, so no entry can
    // miss it; an entry-level predicate can only ever PREDICT this state, and
    // the widening history of `exact_unpenalized_gaussian_beta` (which had to
    // grow from the intercept subspace to any exact affine fit) shows how that
    // prediction keeps coming up short.
    //
    // The SAME reasoning applies to the smoothing criterion, which until #2595
    // had no way to say it: `V_r` profiles the scale, so at `φ̂ = 0` its data
    // term is `½ν·log(D_p/ν) → −∞`. The number the outer optimizer happened to
    // stop at there is a function of the last ulp of β, not a criterion — so it
    // is declined here rather than reported, and `UnifiedFitResult::reml_score`
    // carries the absence all the way to `Summary.raw_reml_score`,
    // `compare_models` and the Bayes-factor path, which now refuse it by name
    // instead of ranking a fabricated value.
    let (log_likelihood, log_likelihood_normalization) = if zero_covariance_boundary {
        (0.0, LogLikelihoodNormalization::UserProvided)
    } else {
        (
            crate::pirls::evaluate_full_log_likelihood_from_eta(
                y_o.view(),
                pirls_res.final_eta.view(),
                &reported_likelihood,
                w_o.view(),
            )?
            .total(),
            LogLikelihoodNormalization::Full,
        )
    };

    let result = ExternalOptimResult {
        beta: reported_beta_orig_internal,
        log_lambdas,
        lambdas: lambdas.to_owned(),
        likelihood_family: reported_family,
        likelihood_scale: likelihood_scale_field,
        log_likelihood_normalization,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        outer_converged,
        pirls_status,
        // The Gaussian identity deviance IS this weighted RSS, so it follows the
        // same snap: reporting `1.1e-29` next to `σ̂ = 0` would leave
        // `deviance/(n − edf) ≠ σ̂²` in the same record, and the formula path's
        // exact-fit route already reports an exact zero here.
        deviance: if identity_fit_is_exact {
            0.0
        } else {
            pirls_res.deviance
        },
        stable_penalty_term: pirls_res.stable_penalty_term,
        used_device: pirls_res.used_device,
        max_abs_eta: pirls_res.max_abs_eta,
        constraint_kkt: pirls_res.constraint_kkt.clone(),
        geometry: (opts.compute_inference || needs_constrained_posterior).then(|| FitGeometry {
            coefficient_gauge: gam_problem::Gauge::identity(&[beta_orig_internal.len()]),
            penalized_hessian: penalized_hessian.into(),
            constrained_posterior,
            working: Some(WorkingGeometry {
                weights: pirls_res.solveweights.to_owned(),
                response: pirls_res.solveworking_response.to_owned(),
            }),
        }),
        artifacts: FitArtifacts {
            pirls: Some(pirls_res),
            criterion_certificate: outer_result.criterion_certificate.clone(),
            rho_posterior_certificate,
            rho_posterior_escalation,
            rho_covariance,
            // Persist the optimized target's Firth state so saved-model
            // sampling reconstructs the same posterior (#2245 finding 16).
            firth_bias_reduction: cfg.firth_bias_reduction,
            ..Default::default()
        },
        inference,
        reml_score: (!zero_covariance_boundary).then_some(outer_result.final_value),
        outer_cost_evals: usize::try_from(
            // A panic elsewhere can poison this lock, but the count it guards is
            // a diagnostic that is still perfectly readable; recover it rather
            // than turn a reporting field into a second panic.
            *reml_state
                .arena
                .cost_eval_count
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
        .unwrap_or(usize::MAX),
        inner_pirls_solves: usize::try_from(
            reml_state
                .arena
                .inner_pirls_solve_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )
        .unwrap_or(usize::MAX),
        fitted_link: if let Some(state) = final_mixture_state {
            FittedLinkState::Mixture {
                state,
                covariance: final_mixture_param_covariance,
            }
        } else if let Some(state) = opts.latent_cloglog {
            FittedLinkState::LatentCLogLog { state }
        } else if let Some(state) = final_sas_state {
            if opts.family.is_binomial_sas() {
                FittedLinkState::Sas {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else if opts.family.is_binomial_beta_logistic() {
                FittedLinkState::BetaLogistic {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else {
                FittedLinkState::Standard(None)
            }
        } else {
            FittedLinkState::Standard(None)
        },
    };
    // Every inference allocation the governor charges is behind us; release
    // both holds explicitly before handing the assembled result back.
    drop(dense_covariance_reservation);
    drop(factorized_inference_reservation);
    conditioning.backtransform_external_result(result)
}

#[cfg(test)]
mod shipped_joint_point_identity_2727_tests {
    //! #2727 — the post-fit certificate identity check must compare the shipped
    //! point against the certified one over EVERY optimized coordinate.
    //!
    //! The outer optimizer for a flexible-link fit searches
    //! `theta = [rho, link-shape coords]`, so the certificate is minted at a
    //! `K + link_dim` vector while the shipped rho block is `K`. The old check
    //! compared those two directly and so could never agree: six SAS/mixture
    //! fixtures were refused with `converged=true`, `certifies=true` and `|Pg|`
    //! down to `1.446e-6`, on a `Vec::len()` mismatch alone.
    //!
    //! Two arms, and the second is the one that matters. Arm 1 is what the
    //! defect broke — a faithfully shipped joint point must be ACCEPTED. Arm 2
    //! is what the obvious wrong repair breaks: comparing only the rho prefix
    //! turns this over-strict gate into an under-strict one, silently ceasing
    //! to check the link coordinates, which are exactly the coordinates this
    //! lane exists to optimize. Arm 2 fails under that repair and passes here,
    //! so the two arms cannot both be satisfied by a prefix comparison.

    use super::shipped_joint_point_is_certified;
    use ndarray::Array1;

    /// The SAS shape of the reproducer in #2727: one rho, two link coordinates
    /// `(epsilon, log delta)`, i.e. the `1 vs 3` that the old check refused.
    fn sas_reproducer() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let rho = Array1::from_vec(vec![-4.753038138161757]);
        let link = Array1::from_vec(vec![0.6483514447757568, -1.2814780614332404]);
        let certified = Array1::from_vec(vec![
            -4.753038138161757,
            0.6483514447757568,
            -1.2814780614332404,
        ]);
        (rho, link, certified)
    }

    /// ARM 1 — a faithfully shipped joint point is accepted.
    ///
    /// This is the arm the defect broke. On the pre-fix code the operands are a
    /// 1-vector and a 3-vector, so this returns false and the fit is refused.
    #[test]
    fn a_faithfully_shipped_joint_point_is_certified() {
        let (rho, link, certified) = sas_reproducer();
        assert!(
            shipped_joint_point_is_certified(&rho, &link, &certified),
            "the shipped point IS the certified point in every coordinate              (rho={rho:?}, link={link:?}, certified={certified:?}); refusing it              is #2727"
        );
    }

    /// ARM 2 — a link coordinate that does not match must still be refused,
    /// even though the rho prefix matches bitwise.
    ///
    /// This is the discriminating arm. A repair that compares only
    /// `certified[..rho.len()]` passes arm 1 and FAILS this one, which is what
    /// stops the over-strict gate from being repaired into an under-strict one.
    /// Note the rho block here is bitwise identical to the certificate, so a
    /// prefix comparison has nothing to catch.
    #[test]
    fn a_mismatched_link_coordinate_is_refused_though_the_rho_prefix_matches() {
        let (rho, link, certified) = sas_reproducer();
        let mut perturbed = link.clone();
        // One ULP. The check is bitwise by design, so the smallest
        // representable disagreement must already refuse — a tolerance here
        // would be a second, unstated stationarity comparison.
        perturbed[1] = f64::from_bits(perturbed[1].to_bits() + 1);
        assert_ne!(perturbed[1].to_bits(), link[1].to_bits());

        assert!(
            rho.iter()
                .zip(certified.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "precondition: the rho prefix must match bitwise, or this arm would              be refused for arm 1's reason instead of its own"
        );
        assert!(
            !shipped_joint_point_is_certified(&rho, &perturbed, &certified),
            "a shipped link coordinate differing from the certified one must be              refused; accepting it is the under-strict repair of #2727"
        );
    }

    /// A missing link coordinate is a different point, not a shorter one — the
    /// length conjunct the original check got right, kept.
    #[test]
    fn a_dropped_link_coordinate_is_refused() {
        let (rho, _link, certified) = sas_reproducer();
        let truncated = Array1::from_vec(vec![0.6483514447757568]);
        assert!(
            !shipped_joint_point_is_certified(&rho, &truncated, &certified),
            "shipping 2 coordinates against a 3-coordinate certificate must be              refused"
        );
    }

    /// The rho-only arm is unchanged: no link coordinates, and the shipped rho
    /// vector is the whole certified point.
    #[test]
    fn the_rho_only_arm_still_compares_rho_against_the_whole_certificate() {
        let rho = Array1::from_vec(vec![0.25, -1.5]);
        let none = Array1::<f64>::zeros(0);
        assert!(shipped_joint_point_is_certified(&rho, &none, &rho.clone()));

        let mut moved = rho.clone();
        moved[0] = f64::from_bits(moved[0].to_bits() + 1);
        assert!(
            !shipped_joint_point_is_certified(&moved, &none, &rho),
            "a moved rho must still be refused on the rho-only arm"
        );
    }
}

#[cfg(test)]
mod negative_binomial_joint_certificate_tests {
    use super::negbin_theta_stationarity_residual;
    use crate::pirls::{NEGBIN_THETA_MAX, NEGBIN_THETA_MIN};

    #[test]
    fn theta_residual_is_the_log_scale_newton_displacement() {
        let theta: f64 = 2.0;
        let score: f64 = 3.0;
        let info: f64 = 5.0;
        let expected = (theta * score).abs() / (theta * theta * info - theta * score);
        assert_eq!(
            negbin_theta_stationarity_residual(theta, score, info),
            expected
        );
        let weight_scale = 1.0e-9;
        let scaled =
            negbin_theta_stationarity_residual(theta, weight_scale * score, weight_scale * info);
        assert!(
            (scaled - expected).abs() <= 8.0 * f64::EPSILON * expected.max(1.0),
            "the theta certificate must be invariant to uniform case-weight scaling: {scaled} vs {expected}"
        );
    }

    #[test]
    fn theta_residual_projects_only_outward_boundary_gradients() {
        assert_eq!(
            negbin_theta_stationarity_residual(NEGBIN_THETA_MIN, -1.0, 1.0),
            0.0
        );
        assert_eq!(
            negbin_theta_stationarity_residual(NEGBIN_THETA_MAX, 1.0, 1.0),
            0.0
        );
        assert!(negbin_theta_stationarity_residual(NEGBIN_THETA_MIN, 1.0, 1.0) > 0.0);
        assert!(negbin_theta_stationarity_residual(NEGBIN_THETA_MAX, -1.0, 1.0) > 0.0);
    }

    #[test]
    fn theta_residual_rejects_invalid_curvature_or_coordinates() {
        assert!(negbin_theta_stationarity_residual(f64::NAN, 0.0, 1.0).is_infinite());
        assert!(negbin_theta_stationarity_residual(1.0, 1.0, 0.0).is_infinite());
        assert!(negbin_theta_stationarity_residual(1.0, 2.0, 1.0).is_infinite());
    }

}

