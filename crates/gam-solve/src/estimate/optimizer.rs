use super::*;
use crate::estimate::evaluation::{
    materialize_link_outer_hessian, sas_effective_epsilon, sas_effective_epsilon_second,
    sas_log_delta_edge_barriercostgrad, sas_log_delta_edge_barriercostgradhess,
    sas_log_deltaridgeweight,
};
use crate::estimate::penalty::{
    REML_SECOND_ORDER_RHO_CAP, REML_SEED_SCREENING_RHO_CAP, scaled_covariance,
};
use crate::estimate::prefit::{
    reject_prefit_binomial_separation, reject_prefit_unpenalized_rank_deficiency,
};
use crate::estimate::smoothing_correction::AUTO_CUBATURE_MAX_EIGENVECTORS;
use gam_linalg::matrix::FactorizedSystem;
use gam_linalg::utils::KahanSum;
use gam_problem::dispersion_cov::se_from_covariance;
use gam_problem::{SeedConfig, SeedRiskProfile};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Unscaled posterior covariance `H⁻¹` of the supplied Hessian.
///
/// This is deliberately neither an additive-ridge inverse nor a truncated
/// pseudoinverse: both would change the covariance estimand.  A singular or
/// numerically unrepresentable Hessian is a typed failure, while success has
/// already passed the shared scale-aware residual certificate in `gam-linalg`.
fn posterior_covariance_inverse(
    h: &Array2<f64>,
    label: &str,
) -> Result<Array2<f64>, gam_linalg::utils::CertifiedSymmetricSolveError> {
    certified_spd_inverse(h, label).map(gam_linalg::utils::CertifiedSpdInverse::into_inverse)
}

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
    .map(|_| ())
    .map_err(|error| {
        EstimationError::RemlOptimizationFailed(format!(
            "exact factorized inference solve did not certify: {error}"
        ))
    })
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
    const PEAK_SQUARE_MATRIX_EQUIVALENTS: usize = STORED_SQUARE_MATRICES
        + BASE_FACTORIZATION_AND_GEMM_WORKSPACES
        + FIRST_ORDER_SMOOTHING_WORKSPACES
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

/// Optimize smoothing parameters for an external design using the same REML/LAML machinery.
pub fn optimize_external_design<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    optimize_external_designwith_heuristic_lambdas(y, w, x, offset, s_list, None, opts)
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
    if k >= REML_SEED_SCREENING_RHO_CAP {
        if gaussian {
            // #1074: the over-smoothing safety net (heavy probe + budget-2
            // lowest-cost keep-best) must remain reachable for MULTI-TERM
            // Gaussian fits, not just the single-smooth k < CAP case. A textbook
            // geostatistical model `mag ~ s(long,lat,bs="tp") + s(depth)` carries
            // FOUR penalty blocks (two double-penalized smooths), so it lands in
            // this k >= CAP branch — where the old single-anchor `seed_budget = 1`
            // path descends from the heuristic anchor straight into the flexible
            // (low-λ) basin and over-fits the weak earthquake-magnitude signal
            // (edf ≈ 104 vs mgcv ≈ 15, held-out R² ≈ 0.02). The single-smooth arm
            // (`s(long,lat)` alone, k = 2) does NOT over-fit precisely because it
            // already gets this net.
            //
            // Cost: the probe is ONE extra seed and the heavily-penalized basin
            // solves cheaply (its inner P-IRLS collapses into the penalty null
            // space — few effective dof — so it converges in a handful of
            // iterations), so the added work is ~one cheap solve, NOT a 2× of the
            // expensive flexible solve. The seed lattice stays minimal (anchor +
            // 4 global shifts + the probe = 6 candidates, `max_seeds = 4` so no
            // exploratory lattice is appended), honouring the perf-guard intent of
            // the cap (no large seed lattice for high-rho fits) while restoring the
            // basin coverage. Lowest-cost keep-best adopts the heavy basin only
            // when it scores a strictly lower REML, so a genuinely flexible
            // multi-term Gaussian surface is never worsened — only a weak-signal
            // over-rich fit escapes the over-fit basin it currently rails into.
            return SeedConfig {
                bounds: (-12.0, 12.0),
                max_seeds: 4,
                seed_budget: 2,
                risk_profile: SeedRiskProfile::Gaussian,
                screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
                num_auxiliary_trailing: 0,
                over_smoothing_probe_rho: Some(8.0),
            };
        }
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
        max_seeds: if gaussian && k <= 4 {
            // #1074: widen the small-k Gaussian candidate pool from 2 to 4 so the
            // flexible anchor shifts AND the absolute over-smoothing probe (set
            // below) both survive into the screened pool instead of one being
            // truncated. With promote-extreme seeding (now enabled for Gaussian)
            // and seed_budget 2, the flexible basin is solved at slot 0 and the
            // heavy basin at slot 1.
            4
        } else if gaussian && k <= 12 {
            4
        } else if gaussian {
            6
        } else if k <= 4 {
            6
        } else if k <= 12 {
            8
        } else {
            10
        },
        // #1074: Gaussian small-k fits get TWO full-budget solves (was 1) so the
        // heavily-penalized basin is actually solved alongside the flexible one;
        // lowest-cost keep-best then returns whichever has the lower REML, so a
        // genuinely flexible fit (tp_2d/te_3d) is never worsened while a
        // weak-signal over-rich spatial fit (quakes) can escape the over-fit
        // basin it currently rails into (edf≈104 → mgcv-like).
        seed_budget: 2,
        risk_profile: if gaussian {
            SeedRiskProfile::Gaussian
        } else {
            SeedRiskProfile::GeneralizedLinear
        },
        screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
        num_auxiliary_trailing: 0,
        // #1074: an ABSOLUTE high-λ probe (interior, below the 11.5 over-smoothing
        // boundary so it is promoted as the heaviest interior seed rather than
        // parked at the tail) seeds the over-smoothed basin the Gaussian global
        // shifts (±4) and baseline centers (±6) never reach. None for non-Gaussian
        // (their symmetric shifts + promote-extreme already span both basins).
        over_smoothing_probe_rho: if gaussian { Some(8.0) } else { None },
    }
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
fn gaussian_identity_response_center(
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
fn gaussian_identity_response_scale(
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
    let reml_y_conditioned: Option<Array1<f64>> = match (response_center, response_scale) {
        (None, None) => None,
        (center, scale) => {
            let c = center.unwrap_or(0.0);
            let s = scale.unwrap_or(1.0);
            Some((&y_o - c) / s)
        }
    };
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
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    if let Some(kron) = opts.kronecker_penalty_system.clone() {
        reml_state.set_kronecker_penalty_system(kron);
    }
    if let Some(kf) = opts.kronecker_factored.clone() {
        reml_state.set_kronecker_factored(kf);
    }
    if opts.persist_warm_start_disk {
        // Caller opted into cross-process resume (#1082): engage the on-disk
        // warm-start layer. Default-false keeps replicate/CI loops disk-silent.
        reml_state.enable_persistent_warm_start_disk();
    }
    reml_state.setwarm_start_original_beta(warm_start_beta);
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

    // Term/margin-order invariance (#1538/#1539). The per-ρ-coordinate canonical
    // keys label each coordinate by its placement-independent (penalty + data)
    // content, letting the outer optimizer operate in an identical canonical
    // coordinate layout for every term order (attached via
    // `with_rho_canonical_keys` below). `None` when the coordinate count does not
    // match the ρ-dimension (legacy native-order path, unchanged).
    let canon_keys = reml_state.canonical_rho_keys(k);

    let reml_seed_config = external_reml_seed_config(k, cfg.link_function());
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
            // Standard-GAM dense problem dimensions configure both cost models
            // the planner uses to decide whether ARC+Hessian or BFGS+gradient
            // is faster end-to-end at large scale:
            //
            //   - per-inner-solve cost (n · p²) gates the single-Hessian-
            //     assembly downgrade,
            //   - per-outer-eval cost (k² · n · p²) gates the LAML-Hessian
            //     pairwise-assembly downgrade — independent of (1) and
            //     necessary because the LAML outer Hessian's k² pairwise
            //     inner-derived terms can dominate per-outer work even when
            //     each individual inner solve is moderate.
            //
            // Sparse designs short-circuit the policy because the n · p²
            // model does not apply to sparse linear algebra; ARC stays in
            // place and the sparse path's iteration-count advantage holds.
            // Gaussian-identity REML has two well-conditioned features that
            // the outer optimizer can exploit:
            //
            //   1. The REML cost is dominated by an O(n) likelihood constant,
            //      so ∂/∂logλ inherits the same scale. A unit-magnitude
            //      `abs` gradient floor (1e-6) becomes binding at large-scale n
            //      even after the relative-from-seed component declared
            //      convergence iters earlier. `with_objective_scale(n)`
            //      lifts the floor to ~n·1e-9 so the loop terminates once
            //      the relative reduction is met.
            //
            //   2. The Gaussian profile likelihood is quadratic-like in
            //      log-λ near the optimum, so the analytic Hessian is
            //      trustworthy and the cubic regularization can start
            //      smaller than opt's default sigma=1.0. Setting
            //      sigma=0.25 allows the first ARC step to be ~4× the
            //      default — matching the 2–4 unit log-λ moves typical of
            //      Gaussian-identity REML cold starts on tensor smooths.
            //
            // Other families (logit, log, survival) keep the conservative
            // defaults because their objective is non-quadratic in log-λ
            // and their gradient is not on an O(n) scale.
            let gaussian_identity = matches!(cfg.link_function(), LinkFunction::Identity);
            let n_obs = y_o.len();
            let prefer_gradient_only = k >= REML_SECOND_ORDER_RHO_CAP;
            if prefer_gradient_only {
                log::info!(
                    "[OUTER] rho_dim {k} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
                );
            }
            let problem = OuterProblem::new(k)
                .with_gradient(Derivative::Analytic)
                .with_hessian(if analytic_outer_hessian_available {
                    DeclaredHessianForm::Either
                } else {
                    DeclaredHessianForm::Unavailable
                })
                .with_prefer_gradient_only(prefer_gradient_only)
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
                // full k²·n·p² LAML-Hessian assembly per phantom iteration until it
                // exhausts the iteration budget — the #1082 outer-loop "cycling"
                // timeout. Lifting the floor to ~n·1e-9 (the same calibration the
                // spatial/custom-family outer already uses via `with_problem_size`,
                // #1053/#1066/#1069) lets the loop terminate as soon as the relative
                // reduction is met, for every family, while the relative-to-cost
                // component still owns the actual convergence decision. ARC σ and the
                // initial trust radius stay Gaussian-gated: those exploit the
                // Gaussian profile being quadratic-in-log-λ, which is family-specific.
                .with_objective_scale(Some(n_obs as f64))
                .with_problem_size(n_obs, x_o.ncols())
                .with_arc_initial_regularization(if gaussian_identity { Some(0.25) } else { None })
                .with_operator_initial_trust_radius(if gaussian_identity {
                    Some(4.0)
                } else {
                    None
                })
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
            let prepass_seed: Option<Array1<f64>> = {
                let bnds = reml_seed_config.bounds;
                let (lo, hi_seed) = if bnds.0 <= bnds.1 {
                    bnds
                } else {
                    (bnds.1, bnds.0)
                };
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
                let hi = hi_seed.max(crate::estimate::RHO_BOUND);
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
                    Array1::from_iter(h.iter().map(|&v| v.clamp(lo, hi)))
                } else {
                    Array1::from_elem(k, (risk_shift + weight_log_geom_mean).clamp(lo, hi))
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
                let initial_sp = reml_state.analytic_initial_sp_rho(&base, (lo, hi));
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
                    .analytic_gaussian_profiled_diagonal_rho((lo, hi))
                    .ok()
                    .flatten()
                    .map(|rho_blocks| {
                        let mut seed = base.clone();
                        for (coord, &r) in seed.iter_mut().zip(rho_blocks.iter()) {
                            *coord = r.clamp(lo, hi);
                        }
                        seed
                    });
                let base_cost = reml_state
                    .compute_cost(&base)
                    .ok()
                    .filter(|c| c.is_finite());
                // Keep the strictly-cheapest certified/scored candidate.
                let mut refined = base.clone();
                let mut best_cost = base_cost;
                for candidate in [initial_sp, summed_diagonal].into_iter().flatten() {
                    let candidate_cost = reml_state
                        .compute_cost(&candidate)
                        .ok()
                        .filter(|c| c.is_finite());
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
                    log::info!(
                        "[OUTER] standard REML initial.sp selected seed: {:?} -> {:?}",
                        base.as_slice().unwrap_or(&[]),
                        refined.as_slice().unwrap_or(&[])
                    );
                    Some(refined)
                } else {
                    None
                }
            };
            let problem = if let Some(seed) = prepass_seed {
                problem.with_initial_rho(seed)
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
            let prefer_gradient_only = theta_dim >= REML_SECOND_ORDER_RHO_CAP;
            if prefer_gradient_only {
                log::info!(
                    "[OUTER] theta_dim {theta_dim} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
                );
            }
            let problem = OuterProblem::new(theta_dim)
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Either)
                .with_prefer_gradient_only(prefer_gradient_only)
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
            let sas_ridge_cost = |theta: &Array1<f64>| -> f64 {
                let sasridge = if use_sas && !use_beta_logistic {
                    sasridgeweight
                } else {
                    0.0
                };
                if use_sas && sasridge > 0.0 {
                    let log_delta = theta[k + 1];
                    let mut extra = 0.5 * sasridge * log_delta * log_delta;
                    if !use_beta_logistic {
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
                // SAS log_delta ridge + barrier gradient/Hessian.
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    grad[k + 1] += sasridgeweight * log_delta;
                    hessian[[k + 1, k + 1]] += sasridgeweight;
                    let (_, barriergrad, barrierhess) =
                        sas_log_delta_edge_barriercostgradhess(log_delta);
                    grad[k + 1] += barriergrad;
                    hessian[[k + 1, k + 1]] += barrierhess;
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
            // Same exact-seed cache publish/consume symmetry as the standard
            // REML arm above (issue #236).
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());
            let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
            drop(obj);
            let final_rho = outer_result.rho.slice(s![..k]).to_owned();
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
            let rho_residual = crate::rho_optimizer::projected_gradient_norm(
                &final_rho,
                &rho_gradient,
                Some(&(rho_lower, rho_upper)),
            );
            let rho_bound = outer_result
                .criterion_certificate
                .as_ref()
                .map(|certificate| certificate.stationarity.bound())
                .unwrap_or(reml_tol)
                .max(f64::EPSILON);
            let rho_certificate_ok = final_rho.is_empty()
                || (outer_result.converged
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
                penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
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
        )?;
        pirls_res = pirls_res_pair.0;

        break;
    } // negative-binomial joint-coordinate loop
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, outer_result.iterations);

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
    let weighted_rss = if matches!(cfg.link_function(), LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrixvectormultiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        w_o.iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum()
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
    let mut rho_covariance = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut beta_standard_errors = None;
    let mut beta_covariance_corrected = None;
    let mut beta_standard_errors_corrected = None;
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

    if opts.compute_inference {
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
            // f64::clamp does NOT fix NaN (only ±inf): a NaN product (e.g.
            // inf*0 from an overflowed solve) would slip through and trip the
            // penalty_block_trace finiteness validator. Map any non-finite
            // product to the saturated `rank` bound, exactly as the inf case
            // already resolves (gam#1379).
            let trace_val = lambdas[kk] * frob;
            traces[kk] = if trace_val.is_finite() {
                trace_val.clamp(0.0, rank as f64)
            } else {
                rank as f64
            };
        }
        edf_total = (p_dim as f64 - kahan_sum(traces.iter().copied())).clamp(mp, p_dim as f64);
        penalty_block_trace.clone_from(&traces);
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            let p_k = cp.rank() as f64;
            let edf_k = (p_k - traces[kk]).clamp(0.0, p_k);
            edf_by_block[kk] = edf_k;
        }

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
        // matrix `F` are formed (and share one `posterior_covariance_inverse`)
        // in exactly the same regime; beyond the policy budget both switch off
        // together and the trace-channel value stands.
        {
            let p_orig = pirls_res.reparam_result.qs.nrows();
            if dense_covariance_reservation.is_some() {
                let h_orig = map_hessian_to_original_basis(&pirls_res)?;
                // Sharing one certified SPD inverse makes the block traces and
                // influence matrix read the identical `H⁻¹`. Failure is not a
                // request to silently change rank or add a diagonal perturbation.
                let h_inv = posterior_covariance_inverse(&h_orig, "edf reconciliation").map_err(
                    |error| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "EDF reconciliation requires an exact SPD Hessian inverse: {error}"
                        ))
                    },
                )?;
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
                        let sol = h_inv.dot(&root_orig); // H⁻¹ R
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
                        let trace_val = lambdas[kk] * frob;
                        traces_f[kk] = if trace_val.is_finite() {
                            trace_val.clamp(0.0, rank as f64)
                        } else {
                            rank as f64
                        };
                    }
                    edf_total = (p_orig as f64 - kahan_sum(traces_f.iter().copied()))
                        .clamp(mp, p_orig as f64);
                    penalty_block_trace.clone_from(&traces_f);
                    for (kk, cp) in pirls_res
                        .reparam_result
                        .canonical_transformed
                        .iter()
                        .enumerate()
                    {
                        let p_k = cp.rank() as f64;
                        edf_by_block[kk] = (p_k - traces_f[kk]).clamp(0.0, p_k);
                    }
                }
            }
        }

        // O(n⁻¹) frequentist bias correction vector b̂ = H⁻¹ S(λ̂)(β̂ - μ).
        // Computed in transformed PIRLS basis (where the factorization above lives)
        // and then mapped to the original coefficient basis via Qs.
        // Frequentist bias of the linear predictor at x is -s_*(x)^T b̂; the
        // corrected predictor is η̂_BC(x) = η̂(x) + s_*(x)^T b̂.
        let beta_t = pirls_res.beta_transformed.as_ref();
        let mut s_beta_t = Array1::<f64>::zeros(p_dim);
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            // S_k(β - μ): only the col_range of beta couples through local penalty.
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
    // df mgcv uses. When inference is off, edf_total is unavailable, so the MLE
    // RSS/n is returned instead.
    let resolved_likelihood_scale = pirls_res
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let profiled_gaussian_standard_deviation = match resolved_likelihood_scale {
        gam_problem::ResolvedLikelihoodScale::ProfiledGaussian => {
            let denom = if opts.compute_inference {
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
    let (final_value, finalgrad, finalgrad_norm, stationarity_bound) = if final_rho.is_empty() {
        (outer_result.final_value, Array1::zeros(0), 0.0, reml_tol)
    } else {
        let (value, gradient) = reml_state.compute_cost_and_gradient(&final_rho)?;
        let lower = Array1::from_elem(final_rho.len(), -crate::estimate::RHO_BOUND);
        let upper = Array1::from_elem(final_rho.len(), crate::estimate::RHO_BOUND);
        let projected = crate::rho_optimizer::projected_gradient_norm(
            &final_rho,
            &gradient,
            Some(&(lower, upper)),
        );
        let bound = outer_result
            .criterion_certificate
            .as_ref()
            .map(|certificate| certificate.stationarity.bound())
            .unwrap_or(reml_tol)
            .max(f64::EPSILON);
        (value, gradient, projected, bound)
    };
    let shipped_point_is_certified = final_rho.len() == outer_result.rho.len()
        && final_rho
            .iter()
            .zip(outer_result.rho.iter())
            .all(|(shipped, certified)| shipped.to_bits() == certified.to_bits());
    let certificate_valid = final_rho.is_empty()
        || (outer_result.converged
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
                "post-fit certificate identity check failed: shipped rho {:?} vs \
                 certified rho {:?} (converged={}, certifies={}, |Pg| at shipped point {:.3e})",
                final_rho.to_vec(),
                outer_result.rho.to_vec(),
                outer_result.converged,
                outer_result
                    .criterion_certificate
                    .as_ref()
                    .is_some_and(|certificate| certificate.certifies()),
                finalgrad_norm,
            ),
            iterations: outer_result.iterations,
            final_value,
            projected_grad_norm: finalgrad_norm.is_finite().then_some(finalgrad_norm),
            stationarity_bound,
            rho_checkpoint: final_rho.to_vec(),
        });
    }
    outer_result.final_value = final_value;
    outer_result.final_gradient = Some(finalgrad);
    outer_result.final_grad_norm = Some(finalgrad_norm);
    let outer_converged = true;

    if opts.compute_inference {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
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
        let beta_covariance_unscaled: Option<Array2<f64>> =
            if dense_covariance_reservation.is_some() {
                Some(
                posterior_covariance_inverse(&penalized_hessian, "posterior covariance").map_err(
                    |error| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "posterior covariance requires an exact SPD Hessian inverse: {error}"
                        ))
                    },
                )?,
            )
            } else {
                None
            };

        if let Some(ref h_inv) = beta_covariance_unscaled {
            // Full inverse available: wrap as phi-scaled covariance, compute
            // frequentist quantities, and pass to smoothing-correction cubature.
            beta_covariance = Some(gam_problem::dispersion_cov::PhiScaledCovariance::wrap(
                scaled_covariance(h_inv.clone(), cov_scale),
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
            let mut bc_jac = Array2::<f64>::eye(p_cov);
            bc_jac += &h_inv.dot(&s_mat);
            bias_correction_jacobian = Some(bc_jac);
            // Influence matrix F = I − H⁻¹·S(λ) = H⁻¹·X'WX. This is a product
            // of two symmetric matrices and is therefore generally NOT
            // symmetric; it must not be symmetrized — `gam_linalg::matrix::symmetrize_in_place(F)`
            // both breaks the H·F = X'WX consistency identity (so any
            // downstream code that reconstructs X'WX from H·F lands on an
            // asymmetric/indefinite matrix) AND corrupts the frequentist
            // covariance `Ve = F·H⁻¹·φ` (since (F_sym)·H⁻¹ ≠ H⁻¹·X'WX·H⁻¹)
            // AND distorts the Wood-corrected reference d.f.
            // `tr(F_jj)² / tr(F_jj²)` consumed by `smooth_test::reference_df`
            // (tr(F²) ≠ tr(F_sym²) in general). See issue #1027.
            let mut f_mat = Array2::<f64>::eye(p_cov);
            f_mat -= &h_inv.dot(&s_mat);
            let mut ve = f_mat.dot(h_inv);
            ve *= cov_scale;
            gam_linalg::matrix::symmetrize_in_place(&mut ve);
            // X'WX = H − S(λ) in the original basis — the genuine PSD weighted
            // Gram, reconstructed from the same `penalized_hessian` and `s_mat`
            // that define `F = H⁻¹X'WX` (issue #1027). Stored directly so the
            // WPS corrected-EDF correction never has to recover it from an
            // inconsistent `H·F` product.
            let mut xwx = &penalized_hessian - &s_mat;
            gam_linalg::matrix::symmetrize_in_place(&mut xwx);
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
            let smoothing_outcome = reml_state.compute_smoothing_correction_auto(
                &final_rho,
                &lambdas,
                &pirls_res,
                beta_covariance_unscaled.as_ref(),
                cov_scale,
                finalgrad_norm,
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
                    if !rail_certified {
                        return Err(EstimationError::InvalidInput(format!(
                            "exact smoothing-corrected covariance unavailable: {reason:?}"
                        )));
                    }
                    log::info!(
                        "[SMOOTHING-CORRECTION] typed-unavailable on a rail-certified \
                         fit ({reason:?}); shipping the plug-in covariance without a \
                         smoothing correction"
                    );
                    rho_covariance = None;
                    smoothing_correction = None;
                    smoothing_correction_method = None;
                }
                outcome => {
                    rho_covariance = outcome.rho_covariance().cloned();
                    (smoothing_correction, smoothing_correction_method) =
                        outcome.into_correction_with_method();
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
        beta_standard_errors = if let Some(ref h_inv) = beta_covariance_unscaled {
            // Fast path: SE from stored full inverse (already phi-scaled via
            // beta_covariance, but we need the unscaled diagonal here).
            let mut raw_se = Array1::<f64>::zeros(p_cov);
            for (index, &variance_unscaled) in h_inv.diag().iter().enumerate() {
                if !(variance_unscaled.is_finite() && variance_unscaled > 0.0) {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "exact SPD inverse has invalid diagonal {index}: {variance_unscaled:?}"
                    )));
                }
                let variance = cov_scale * variance_unscaled;
                let valid = if zero_covariance_boundary {
                    variance == 0.0
                } else {
                    variance.is_finite() && variance > 0.0
                };
                if !valid {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "posterior covariance diagonal {index} is not positive and representable: {variance:?}"
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
                    .map_err(|_| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "resource policy refused exact coefficient-SE columns {col_start}..{col_end}"
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
            let mut se = Array1::<f64>::zeros(p_cov);
            for (index, &variance_unscaled) in diag_inv.iter().enumerate() {
                if !(variance_unscaled.is_finite() && variance_unscaled > 0.0) {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "exact factorized SPD inverse has invalid diagonal {index}: {variance_unscaled:?}"
                    )));
                }
                let variance = cov_scale * variance_unscaled;
                let valid = if zero_covariance_boundary {
                    variance == 0.0
                } else {
                    variance.is_finite() && variance > 0.0
                };
                if !valid {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "factorized posterior variance {index} is not positive and representable: {variance:?}"
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
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.as_array().dim() == corr.dim() => {
                let mut corrected = base_cov.as_array().clone();
                corrected += corr;
                if let Some(a_bc) = bias_correction_jacobian.as_ref() {
                    if a_bc.dim() != corrected.dim() {
                        return Err(EstimationError::RemlOptimizationFailed(format!(
                            "bias-correction Jacobian shape {:?} does not match corrected covariance {:?}",
                            a_bc.dim(),
                            corrected.dim()
                        )));
                    }
                    corrected = a_bc.dot(&corrected).dot(&a_bc.t());
                }
                gam_linalg::matrix::symmetrize_in_place(&mut corrected);
                Some(corrected)
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
                EstimationError::RemlOptimizationFailed(format!(
                    "corrected coefficient covariance is not a valid standard-error source: {error}"
                ))
            })?;
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        penalty_block_trace,
        edf_total,
        smoothing_correction,
        smoothing_correction_method,
        penalized_hessian: penalized_hessian.into(),
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
    match (&mut reported_family.response, likelihood_scale_field) {
        (
            ResponseFamily::NegativeBinomial { theta, .. },
            LikelihoodScaleMetadata::EstimatedNegBinTheta {
                theta: fitted_theta,
            },
        ) => {
            *theta = fitted_theta;
        }
        (
            ResponseFamily::Beta { phi },
            LikelihoodScaleMetadata::EstimatedBetaPhi { phi: fitted_phi },
        ) => {
            *phi = fitted_phi;
        }
        _ => {}
    }
    if zero_covariance_boundary {
        return Err(EstimationError::InvalidInput(
            "the general REML reporting path reached the degenerate profiled-Gaussian boundary sigma^2 = 0, where no finite normalized Gaussian density exists; exact constant-response fits must use the dedicated deterministic-Gaussian shortcut"
                .to_string(),
        ));
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
    let log_likelihood = crate::pirls::evaluate_full_log_likelihood_from_eta(
        y_o.view(),
        pirls_res.final_eta.view(),
        &reported_likelihood,
        w_o.view(),
    )?
    .total();

    let result = ExternalOptimResult {
        beta: beta_orig_internal,
        log_lambdas,
        lambdas: lambdas.to_owned(),
        likelihood_family: reported_family,
        likelihood_scale: likelihood_scale_field,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        outer_converged,
        pirls_status,
        deviance: pirls_res.deviance,
        stable_penalty_term: pirls_res.stable_penalty_term,
        used_device: pirls_res.used_device,
        max_abs_eta: pirls_res.max_abs_eta,
        constraint_kkt: pirls_res.constraint_kkt.clone(),
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
        reml_score: outer_result.final_value,
        outer_cost_evals: usize::try_from(*reml_state.arena.cost_eval_count.read().unwrap())
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
mod blended_mixture_link_solve_tests {
    //! Regression coverage for issue #1598.
    //!
    //! A `blended(logit, probit)` learnable link NESTS each pure component
    //! (mixing weight 1 on one inverse link). On data generated from a plain
    //! logit link, the joint (smoothing + mixing-weight ρ) solve must therefore
    //! converge and recover a near-logit fit — at the ρ=0 seed the mixture
    //! collapses to its well-conditioned first component (logit).
    //!
    //! Before the fix the inner P-IRLS observed-Hessian curvature build aborted
    //! with "observed Hessian curvature is not positive finite at row N" the
    //! moment a single row's residual-dependent observed weight went
    //! non-positive (which the non-canonical mixture path produces on finite
    //! data even at the logit-collapsed seed). That abort propagated up as
    //! "no candidate seeds passed outer startup validation (mixture/SAS flexible
    //! link)" and the whole fit failed. The downstream solver weight floor
    //! (`solver_hessian_weights_into`) is designed to clamp exactly those rows
    //! to keep the Newton system SPD, so the array build must tolerate a finite
    //! non-positive observed weight rather than hard-bail on it.

    use super::optimize_external_design;
    use crate::estimate::external_options::ExternalOptimOptions;
    use gam_problem::{
        InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, ResponseFamily, StandardLink,
    };
    use gam_terms::smooth::BlockwisePenalty;
    use ndarray::{Array1, Array2};

    #[test]
    fn blended_logit_probit_fits_clean_logit_data() {
        // --- Synthetic clean logit data ---------------------------------
        // y_i ~ Bernoulli(logistic(eta_i)), eta_i = b0 + b1 * x_i.
        let n = 120usize;
        let b0 = -0.3_f64;
        let b1 = 1.7_f64;
        // Deterministic pseudo-random x and Bernoulli draws so the test is
        // reproducible without an RNG dependency.
        let mut x = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        let mut true_p = Array1::<f64>::zeros(n);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next_unit = || {
            // SplitMix64 → uniform in [0, 1).
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        for i in 0..n {
            let xi = -2.5 + 5.0 * next_unit();
            let eta = b0 + b1 * xi;
            let p = 1.0 / (1.0 + (-eta).exp());
            x[i] = xi;
            true_p[i] = p;
            y[i] = if next_unit() < p { 1.0 } else { 0.0 };
        }

        // Design with intercept + slope, both unpenalized (zero penalty block).
        let mut design = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = x[i];
        }
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let penalty = BlockwisePenalty::new(0..2, Array2::<f64>::zeros((2, 2)));

        // blended(logit, probit) with the documented ρ=0 (pure first-component)
        // seed: mixing weight 1 on logit, 0 on probit.
        let mix_spec = MixtureLinkSpec {
            components: vec![LinkComponent::Logit, LinkComponent::Probit],
            initial_rho: Array1::zeros(1),
        };

        let opts = ExternalOptimOptions {
            family: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            latent_cloglog: None,
            mixture_link: Some(mix_spec),
            optimize_mixture: true,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: true,
            max_iter: 200,
            tol: 1e-9,
            nullspace_dims: vec![2],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };

        // The joint mixture solve must converge (no error) on data its own pure
        // logit component fits trivially.
        let result = optimize_external_design(
            y.view(),
            w.view(),
            design.clone(),
            offset.view(),
            vec![penalty],
            &opts,
        )
        .expect("blended(logit, probit) joint solve must converge on clean logit data");

        // Reconstruct the fitted linear predictor η̂ = X·β̂ and correlate it
        // with the true probabilities. The mixture inverse link is monotone, so
        // a faithful fit makes corr(η̂, true_p) ≈ 1.
        assert_eq!(result.beta.len(), 2, "fitted β has intercept + slope");
        let mut eta_hat = Array1::<f64>::zeros(n);
        for i in 0..n {
            eta_hat[i] = design[[i, 0]] * result.beta[0] + design[[i, 1]] * result.beta[1];
        }
        let corr = pearson(&eta_hat, &true_p);
        assert!(
            corr > 0.95,
            "blended(logit, probit) on clean logit data must recover a near-logit \
             fit: corr(η̂, true_p)={corr:.4} (β̂={:?}, λ̂={:?})",
            result.beta.as_slice().unwrap(),
            result.lambdas.as_slice().unwrap(),
        );
        // The recovered slope must keep the sign and order of magnitude of the
        // generating coefficient (it is not penalized).
        assert!(
            result.beta[1] > 0.5,
            "recovered slope must stay strongly positive: β̂₁={}",
            result.beta[1]
        );
    }

    fn pearson(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let n = a.len() as f64;
        let ma = a.sum() / n;
        let mb = b.sum() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for i in 0..a.len() {
            let da = a[i] - ma;
            let db = b[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[cfg(test)]
mod reported_loglikelihood_normalization_tests {
    //! End-to-end regression coverage for issue #2096.
    //!
    //! `optimize_external_design` populates the user-facing
    //! `ExternalOptimResult::log_likelihood` — the number that surfaces as
    //! `Model.summary()["log_likelihood"]` and feeds the conditional/corrected
    //! AIC. Before #2096 that field was wired to the REML building-block kernel
    //! `calculate_loglikelihood_omitting_constants_from_eta`, which drops the Poisson
    //! count normalizer `−Σ lnΓ(y+1)`. On count data that makes the reported
    //! log-likelihood POSITIVE — impossible for a probability mass — and, because
    //! different families drop different normalizers, non-comparable across
    //! families (breaking any AIC built from it).
    //!
    //! The fix routes the reporting field through the fully-normalized
    //! full eta-space likelihood evaluation and tags it
    //! `LogLikelihoodNormalization::Full`. This test fits a small Poisson GAM
    //! surface through the real optimizer and asserts the reported field is a
    //! proper (negative) log-mass — the direct, field-level symptom of #2096,
    //! distinct from the kernel-level checks in
    //! `crate::pirls::tests::reporting_loglikelihood_tests`.

    use super::optimize_external_design;
    use crate::estimate::external_options::ExternalOptimOptions;
    use gam_problem::{
        InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LogLikelihoodNormalization,
        ResponseFamily, StandardLink,
    };
    use gam_terms::smooth::BlockwisePenalty;
    use ndarray::{Array1, Array2};

    fn poisson_opts() -> ExternalOptimOptions {
        ExternalOptimOptions {
            family: LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: true,
            max_iter: 200,
            tol: 1e-9,
            nullspace_dims: vec![2],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        }
    }

    #[test]
    fn poisson_reported_loglikelihood_is_a_negative_log_mass() {
        // Deterministic log-linear count data: μᵢ = exp(1.5 + 0.4·xᵢ), yᵢ =
        // round(μᵢ). Means run from ≈3 to ≈30, so the omitting-constants kernel
        // Σ(y·ln μ − μ) the buggy field used is strongly POSITIVE, while the
        // fully-normalized kernel (which subtracts Σ lnΓ(y+1)) is negative. A
        // near-perfect fit keeps the two far apart, making the sign check sharp.
        let n = 40usize;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            let mu = (1.5 + 0.4 * xi).exp();
            design[[i, 0]] = 1.0;
            design[[i, 1]] = xi;
            y[i] = mu.round().max(1.0);
        }
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let penalty = BlockwisePenalty::new(0..2, Array2::<f64>::zeros((2, 2)));

        let result = optimize_external_design(
            y.view(),
            w.view(),
            design,
            offset.view(),
            vec![penalty],
            &poisson_opts(),
        )
        .expect("Poisson solve on clean log-linear count data must converge");

        assert!(
            result.log_likelihood.is_finite(),
            "reported Poisson log-likelihood must be finite, got {}",
            result.log_likelihood
        );
        // #2096: a count model's reported log-likelihood is a log probability
        // mass and MUST be ≤ 0. The pre-fix omitting-constants field was large
        // and positive (≈ +1100 on the issue's data) because it dropped
        // −Σ lnΓ(y+1).
        assert!(
            result.log_likelihood <= 0.0,
            "reported Poisson log-likelihood must be a negative log-mass (#2096), \
             got {}",
            result.log_likelihood
        );
        // The paired normalization tag must advertise the fully-normalized kernel,
        // so downstream AIC/elpd consumers know the count normalizer is included.
        assert_eq!(
            result.log_likelihood_normalization,
            LogLikelihoodNormalization::Full,
            "reporting field must be tagged fully-normalized (#2096)"
        );
    }

    fn gaussian_opts() -> ExternalOptimOptions {
        ExternalOptimOptions {
            family: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            ..poisson_opts()
        }
    }

    #[test]
    fn gaussian_reported_loglikelihood_is_finite() {
        // #2096 follow-through: the reporting field uses the
        // fully-normalized Gaussian kernel, which requires a CONCRETE
        // dispersion `φ = σ̂²`. The reporting site must resolve that temporary
        // likelihood without rewriting the fit's canonical scale-ownership
        // contract: the returned metadata remains `ProfiledGaussian`, while
        // the reported full log-likelihood is finite.
        let n = 40usize;
        let mut design = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
            design[[i, 0]] = 1.0;
            design[[i, 1]] = xi;
            // Deterministic near-linear signal with a small deterministic wiggle
            // so RSS > 0 and σ̂² is strictly positive.
            y[i] = 0.7 + 1.3 * xi + 0.05 * (3.0 * xi).sin();
        }
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let penalty = BlockwisePenalty::new(0..2, Array2::<f64>::zeros((2, 2)));

        let result = optimize_external_design(
            y.view(),
            w.view(),
            design,
            offset.view(),
            vec![penalty],
            &gaussian_opts(),
        )
        .expect("Gaussian solve on clean near-linear data must converge");

        assert!(matches!(
            result.likelihood_scale,
            LikelihoodScaleMetadata::ProfiledGaussian
        ));
        assert!(
            result.log_likelihood.is_finite(),
            "reported Gaussian log-likelihood must be finite (the profiled σ̂² \
             must be resolved into the reported scale), got {}",
            result.log_likelihood
        );
        assert_eq!(
            result.log_likelihood_normalization,
            LogLikelihoodNormalization::Full,
            "Gaussian reporting field must be tagged fully-normalized"
        );
    }
}

#[cfg(test)]
mod negative_binomial_joint_certificate_tests {
    use super::{negbin_theta_stationarity_residual, optimize_external_design};
    use crate::estimate::external_options::ExternalOptimOptions;
    use crate::pirls::{NEGBIN_THETA_MAX, NEGBIN_THETA_MIN};
    use gam_problem::{EstimationError, LikelihoodSpec};
    use ndarray::{Array1, Array2, array};

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

    #[test]
    fn exhausted_joint_solve_returns_typed_checkpoint_instead_of_a_fit() {
        // Intercept-only overdispersed counts make the large theta seed
        // decisively non-stationary. With one joint round available the rho
        // block is already vacuous, so exhaustion must be attributed to the
        // theta partial and returned through the resumable typed error.
        let y = array![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 40.0, 0.0, 75.0, 0.0, 3.0, 0.0];
        let n = y.len();
        let design = Array2::<f64>::ones((n, 1));
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let opts = ExternalOptimOptions {
            family: LikelihoodSpec::negative_binomial_log(1_000.0),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: true,
            max_iter: 1,
            tol: 1.0e-8,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };
        let error = match optimize_external_design(
            y.view(),
            weights.view(),
            design,
            offset.view(),
            Vec::new(),
            &opts,
        ) {
            Ok(_) => panic!("one round cannot certify this deliberately displaced theta seed"),
            Err(error) => error,
        };
        match error {
            EstimationError::NegativeBinomialAlternationDidNotConverge {
                rounds,
                theta_checkpoint,
                theta_score_residual,
                rho_checkpoint,
                ..
            } => {
                assert_eq!(rounds, 1);
                assert!(theta_checkpoint.is_finite() && theta_checkpoint > 0.0);
                assert!(!theta_score_residual.is_nan() && theta_score_residual > opts.tol);
                assert!(rho_checkpoint.is_empty());
            }
            other => panic!("expected typed negative-binomial joint exhaustion, got {other}"),
        }
    }
}
