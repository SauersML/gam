use super::*;
use gam_problem::dispersion_cov::se_from_covariance;
use std::sync::atomic::{AtomicUsize, Ordering};

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

enum RemlInnerCapGuardArm {
    Standard,
    MixtureSas,
}

/// Re-run one full-inner-tolerance `compute_cost` at the converged operating
/// point so the cached warm-start β is no longer pinned to whatever coarse cap
/// the outer-aware inner-PIRLS schedule last set (path #3; see the call-site
/// comments).
///
/// `rho` MUST be the smoothing-only penalty-block log-λ — its length equals the
/// number of penalty blocks, because `compute_cost` exponentiates the whole
/// vector into the penalty λ vector. For the parameterized-link arms
/// (`MixtureSas`: SAS / Beta-Logistic / mixture / blended) the outer optimizer
/// works in an augmented θ that trails the link parameters after the smoothing
/// block; the caller must slice that block out (and install the link state on
/// `state`) via `apply_link_theta` BEFORE calling this guard. Passing the raw
/// augmented θ here over-counts the lambdas and faults the reparameterizer
/// ("Lambda count mismatch", #1571). The `arm` only selects the log label.
fn run_outer_inner_cap_guard(
    state: &mut crate::estimate::reml::RemlState<'_>,
    rho: &Array1<f64>,
    arm: RemlInnerCapGuardArm,
) -> Result<(), EstimationError> {
    let prev_cap = state.outer_inner_cap.swap(0, Ordering::Relaxed);
    if prev_cap != 0 {
        let guard_start = std::time::Instant::now();
        state.compute_cost(rho)?;
        match arm {
            RemlInnerCapGuardArm::Standard => log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (prev_cap={prev_cap}, elapsed={:.3}s)",
                guard_start.elapsed().as_secs_f64()
            ),
            RemlInnerCapGuardArm::MixtureSas => log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (mixture/SAS arm; prev_cap={prev_cap}, elapsed={:.3}s)",
                guard_start.elapsed().as_secs_f64()
            ),
        }
    } else if matches!(arm, RemlInnerCapGuardArm::Standard) {
        log::debug!("[OUTER guard] schedule never lifted (prev_cap=0); skipping refit");
    }
    Ok(())
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
    // Raw design row count, captured before `x` is moved (line ~339); used by the
    // #1266 null-space shrink-out escape's `n ≥ 2·p` determinacy gate, which must
    // match `relax_smoothing_rho_prior`'s well-determined gate exactly.
    let n_design_rows = x.nrows();
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

    // Term/margin-order invariance (#1538/#1539). The per-ρ-coordinate canonical
    // keys label each coordinate by its placement-independent (penalty + data)
    // content; the induced canonical permutation lets BOTH the objective-grid
    // seed prepass AND the outer optimizer operate in an identical canonical
    // coordinate layout for every term order. `None` when the coordinate count
    // does not match the ρ-dimension (legacy native-order path, unchanged).
    let canon_keys = reml_state.canonical_rho_keys(k);
    let canon_perm: Option<Vec<usize>> = canon_keys
        .as_ref()
        .and_then(|keys| crate::rho_optimizer::canonical_permutation(keys));

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
    // Negative-Binomial outer θ↔λ alternation (#1448). With θ estimated, the
    // λ-search freezes θ (see `frozen_negbin_theta`, #1082) so the REML criterion
    // `F(ρ) = REML(ρ, θ_frozen)` is stationary in ρ; the final accept-fit then
    // ML-refreshes θ at the converged η. A *single* freeze→refresh leaves the
    // selected ρ optimal only for `θ_frozen`, not for the refreshed `θ_final`.
    // mgcv `nb()` instead alternates θ-estimation and λ-selection to a joint
    // fixed point. Wrap the ρ-search + accept-fit in a bounded loop: after each
    // refit, if the NB θ drifted beyond tolerance, re-freeze the search θ at the
    // refreshed value, reset the surface caches that depend on it, and re-run the
    // outer ρ search. The cap bounds the work; for every non-NB / user-fixed-θ
    // fit the loop runs exactly once (the break condition is met immediately), so
    // those fits are byte-identical to the pre-#1448 single-pass behaviour.
    //
    // 5% relative θ drift is the same band the diagnostic (#1082) flagged as the
    // point beyond which the ρ-optimum for `θ_frozen` and `θ_final` can differ
    // enough to matter; below it the one-refresh approximation is already joint-
    // stationary to the criterion's tolerance.
    const NEGBIN_THETA_JOINT_DRIFT_TOL: f64 = 5.0e-2;
    const NEGBIN_OUTER_ALTERNATION_MAX_ROUNDS: usize = 8;
    let mut final_rho;
    let mut final_mixture_state;
    let mut final_sas_state;
    let mut final_mixture_param_covariance;
    let mut final_sas_param_covariance;
    let mut outer_result;
    let mut pirls_res;
    let mut negbin_alternation_round: usize = 0;
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
            let continuation_prewarm = k < REML_CONTINUATION_PREWARM_RHO_CAP;
            if prefer_gradient_only {
                log::info!(
                    "[OUTER] rho_dim {k} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
                );
            }
            if !continuation_prewarm {
                log::info!(
                    "[OUTER] rho_dim {k} reaches continuation-prewarm budget \
                   ({REML_CONTINUATION_PREWARM_RHO_CAP}); starting optimizer directly from seeds"
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
            .with_continuation_prewarm(continuation_prewarm)
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
            .with_operator_initial_trust_radius(if gaussian_identity { Some(4.0) } else { None })
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
            let problem = if let Some(h) = heuristic_lambdas {
                problem.with_heuristic_lambdas(h.to_vec())
            } else {
                problem
            };
            let problem = if let Some(h) = heuristic_lambdas.filter(|h| h.len() == k) {
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
            // The prepass evaluates the *actual* REML/LAML objective on a tiny,
            // deterministic log-λ grid and only changes startup when that same
            // criterion improves.  It is therefore part of initialization, not a
            // compatibility fallback.  Gaussian fits used to skip this when the
            // weights were on the unit scale, leaving single-start BFGS/ARC tied to
            // the arbitrary λ=1 origin; flat or multi-penalty REML surfaces could
            // then spend the finite outer budget getting into the right basin rather
            // than resolving the optimum that controls EDF and truth recovery.  Run
            // the same criterion-ranked startup for Gaussian as for GLM/survival,
            // while retaining the weight-scale anchor from issue #877.
            let run_gaussian_anchored_prepass = gaussian_risk && weight_log_geom_mean.abs() > 1e-12;
            // A caller-supplied rho seed (`init_rhos`/`heuristic_lambdas`, now in
            // rho-space) is an explicit warm-start installed via `with_initial_rho`
            // above. It still ANCHORS the objective-grid prepass below rather than
            // short-circuiting it: the grid is criterion-ranked and only adopts a
            // candidate that STRICTLY lowers the true REML/LAML cost, so a healthy
            // warm seed is returned unchanged (the grid never beats it → byte-
            // identical behaviour). What the anchor-and-rank rescues is a warm seed
            // TRAPPED in a shallow under-smoothing local basin: when the design's
            // kernel collapses (e.g. the constant-curvature `curv()` smooth fitted
            // at a trial κ on the +chart side — the geodesic-exponential kernel's
            // off-diagonals → 1, so its global REML optimum is a LARGE λ that the
            // local outer optimizer, warm-started from the previous-κ λ̂, slides away
            // from into the spurious low-λ optimum). The shallow optimum's
            // spuriously-low deviance made the κ outer objective monotone toward the
            // +chart bound for any curved data (gam#1464 — hyperbolic truth recovered
            // as spherical); anchoring the global grid at the warm seed lets the
            // prepass jump into the correct high-λ basin so the per-κ REML cost
            // matches the textbook profiled-REML and the curvature SIGN is
            // identifiable. Same machinery as the gam#1266 double-penalty rescue.
            let caller_seeded_rho = heuristic_lambdas.is_some_and(|h| h.len() == k);
            // The grid prepass's lowest-cost sample, kept for the #1371
            // release-and-rerank guard even when it is not adopted as the initial
            // seed (i.e. the grid did not strictly move). It is a known-good lower
            // bound on the achievable REML cost, scored with the SAME functional.
            // Unconditionally assigned inside the prepass block below (before its
            // first read by the #1371 guard), so it carries no dead initializer.
            let release_rerank_seed: Option<Array1<f64>>;
            // #1548: the well-determined Marra-Wood double-penalty null-space
            // selection coordinates, recognised exactly as the #1266 shrink-out
            // escape recognises them (the relaxed `Normal(0, sd=15)` degeneracy
            // prior, gated by `n ≥ 2·p`). A SUPPORTED such coordinate has a deep
            // low-λ_null "keep" basin AND a flat high-λ_null annihilation shelf; the
            // objective-grid prepass below probes the keep corner for exactly these
            // coordinates so it can seed the well-conditioned keep basin directly
            // rather than the shelf — see the keep-saturation probe in
            // `select_objective_seed_on_log_lambda_grid`.
            let nullspace_seed_coords: Vec<usize> = if n_design_rows >= 2 * p {
                match reml_state.effective_rho_prior().as_ref() {
                    gam_problem::RhoPrior::Independent(per_coord) => (0..k)
                        .filter(|&i| {
                            per_coord
                                .get(i)
                                .is_some_and(gam_terms::smooth::is_nullspace_degeneracy_prior)
                        })
                        .collect(),
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            };
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
                // Anchor the grid at the caller-supplied `heuristic_lambdas` when one
                // is present (it is already in rho-space, used as-is) — the grid then
                // searches relative to the warm start and keeps it unless a candidate
                // is strictly better. Otherwise anchor the default risk-shift origin
                // to the weight scale (issue #877).
                let base = if let Some(h) = heuristic_lambdas.as_ref().filter(|h| h.len() == k) {
                    Array1::from_iter(h.iter().map(|&v| v.clamp(lo, hi)))
                } else {
                    Array1::from_elem(k, (risk_shift + weight_log_geom_mean).clamp(lo, hi))
                };
                // Run the objective-grid seed search in CANONICAL coordinate order
                // (#1538/#1539) so its greedy per-axis / pairwise-saturation
                // refinement — which is order-dependent in native layout — explores
                // the SAME axes for every term order. The grid builds canonical
                // candidates; the eval closure maps each back to native order before
                // scoring with the true `compute_cost`, and the refined seed is
                // mapped native again for `with_initial_rho`. Without a permutation
                // this is the identity, so the native-order path is byte-for-byte
                // unchanged.
                let refined = if let Some(perm) = canon_perm.as_ref() {
                    let to_native = |canon: &Array1<f64>| -> Array1<f64> {
                        let mut out = Array1::zeros(canon.len());
                        for (c, &i) in perm.iter().enumerate() {
                            out[i] = canon[c];
                        }
                        out
                    };
                    let base_canon = Array1::from_iter(perm.iter().map(|&i| base[i]));
                    // Canonical slot `c` carries native coordinate `perm[c]`; a
                    // null-space coordinate must be probed in whichever slot it
                    // occupies in the canonical layout the grid refines.
                    let nullspace_canon: Vec<usize> = (0..k)
                        .filter(|&c| nullspace_seed_coords.contains(&perm[c]))
                        .collect();
                    let refined_canon = crate::seeding::select_objective_seed_on_log_lambda_grid(
                        &base_canon,
                        (lo, hi),
                        k,
                        &nullspace_canon,
                        |canon_rho| {
                            let native = to_native(canon_rho);
                            reml_state
                                .compute_cost(&native)
                                .ok()
                                .filter(|c| c.is_finite())
                        },
                    );
                    to_native(&refined_canon)
                } else {
                    crate::seeding::select_objective_seed_on_log_lambda_grid(
                        &base,
                        (lo, hi),
                        k,
                        &nullspace_seed_coords,
                        |rho| reml_state.compute_cost(rho).ok().filter(|c| c.is_finite()),
                    )
                };
                // Emit the seed when the grid moved it, or — on the Gaussian
                // weight-anchored path — whenever the anchored `base` is itself
                // offset from the unanchored origin (so the shifted optimum is
                // actually seeded even if the coarse grid leaves `base` unchanged).
                // Record the grid's best sample for the release-and-rerank guard
                // unconditionally — whether or not it is strong enough to override
                // the optimizer's own cold start, it is still a scored lower bound
                // the certified optimum must not be worse than (#1371).
                release_rerank_seed = Some(refined.clone());
                let grid_moved = refined
                    .iter()
                    .zip(base.iter())
                    .any(|(&a, &b)| (a - b).abs() > 1e-12);
                // For a caller-seeded fit, adopt the grid result only when it
                // STRICTLY moved the warm seed (i.e. found a strictly-cheaper basin);
                // an unmoved grid leaves the warm start exactly as installed above, so
                // healthy warm-started fits stay byte-identical. The Gaussian
                // weight-anchored emit only applies on the non-caller-seeded origin.
                if grid_moved || (run_gaussian_anchored_prepass && !caller_seeded_rho) {
                    log::info!(
                        "[OUTER] standard REML objective-grid selected seed: {:?} -> {:?}",
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
            // #1074 DIAGNOSTIC (log-gated, no behavior change unless the crate
            // logger is installed): sweep each outer log-λ coordinate over a grid
            // while holding the others at the baseline, logging the REML cost. Used
            // to decide whether the spatial range railing is an interior optimum the
            // optimizer misses (optimizer bug) or a genuine criterion preference for
            // λ→∞ (criterion). Placed BEFORE the objective takes its `&mut
            // reml_state` borrow so the immutable `compute_cost` reads are valid.
            // Emitted at warn level so the default-installed crate logger (Info)
            // prints it without a level change (the ban-scanner forbids direct
            // stderr printing and process-env reads).
            if log::log_enabled!(log::Level::Warn) {
                let grid = [
                    -5.0_f64, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0,
                ];
                let mut baselines: Vec<(&str, Array1<f64>)> =
                    vec![("zeros", Array1::<f64>::zeros(k))];
                if k == 4 {
                    baselines.push(("conv", Array1::from(vec![9.0_f64, 30.0, 12.0, 30.0])));
                }
                if k == 2 {
                    baselines.push(("conv6", Array1::from(vec![6.0_f64, 6.0])));
                }
                for (label, baseline) in &baselines {
                    log::warn!("[#1074-sweep] k={k} baseline={label}={baseline:?}");
                    for coord in 0..k {
                        let mut line = format!("[#1074-sweep:{label}] coord={coord}:");
                        for &rho in &grid {
                            let mut p = baseline.clone();
                            p[coord] = rho;
                            let cg = reml_state.compute_cost_and_gradient(&p).ok();
                            let cell = match cg {
                                Some((c, g)) => {
                                    format!(
                                        "{c:.4}(g{}={:.3e})",
                                        coord,
                                        g.get(coord).copied().unwrap_or(f64::NAN)
                                    )
                                }
                                None => "ERR".to_string(),
                            };
                            line.push_str(&format!(" {rho:.0}->{cell}"));
                        }
                        log::warn!("{line}");
                    }
                }
            }

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
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>| { state.compute_cost(rho) },
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>| {
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
                Some(
                    |state: &mut &mut crate::estimate::reml::RemlState<'_>| {
                        state.reset_outer_seed_state()
                    },
                ),
                Some(
                    |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                     rho: &Array1<f64>| { state.compute_efs_steps(rho) },
                ),
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>| { state.compute_screening_proxy(rho) },
            );
            // Standard REML's eval closure publishes
            // `inner_beta_hint = state.current_original_basis_beta()` on
            // every accepted eval. The continuation pre-warm carries that
            // hint forward and calls `seed_inner_state(beta)` before the
            // next eval — see src/solver/reml/continuation.rs:209-212,
            // 434-438. Without a hook here, `ClosureObjective::seed_inner_state`
            // (src/solver/rho_optimizer.rs:2097-2107) rejected any
            // non-empty β fatally, dropping every seed before the inner
            // solver started (issue #236). Wire the symmetric consumer:
            // when the pre-warm forwards the cached β, install it into the
            // same `warm_start_beta` slot the publisher reads from.
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());

            let mut strategy_result = problem.run(&mut obj, "standard REML")?;
            drop(obj);
            // #1371 release-and-rerank guard. The continuation oversmoothing
            // warm-start can deliver the inner β on the high-λ null-space
            // "annihilation" shelf of a double-penalty smooth: there the
            // null-space coefficients are already shrunk to ~0, so the deviance
            // ρ-gradient vanishes (∂dev/∂ρ_null → 0) AND the Occam terms
            // (½ tr(H⁻¹ ∂H/∂ρ) − ½ λ tr(S⁺ S_k)) cancel, leaving the analytic
            // outer gradient ≈ 0. ARC then certifies that point as a stationary
            // optimum even though its REML cost is FAR ABOVE a point the seed
            // prepass already evaluated — driving a genuinely-supported null-space
            // direction (a real linear trend, gam#1371) to EDF → 0. The seed
            // prepass's grid-refined seed is a known-good lower bound on the cost
            // (it was scored with the SAME `compute_cost`), so if the certified
            // optimum is strictly worse than it, re-rank to the seed: re-running
            // the inner solve there installs the correct β̂. This cannot regress a
            // fit whose optimum genuinely IS the high-λ corner (gam#1266: an
            // unsupported term shrinking out) — there the corner is the
            // lowest-cost point, no cheaper seed exists, and the guard is a no-op.
            if let Some(seed) = release_rerank_seed.as_ref() {
                // The certified cost is the optimizer's OWN authoritative
                // `final_value`, NOT a fresh `compute_cost(strategy_result.rho)`
                // re-evaluation (load-bearing, #1426/#1477). The REML/LAML objective
                // for a non-Gaussian family is NOT a pure function of ρ: it carries a
                // profiled dispersion / nuisance that is established by the inner
                // solve at the operating ρ, and `compute_cost` warm-starts the inner
                // PIRLS from whatever β̂/φ the previous eval left behind. Probing the
                // under-penalized prepass seed FIRST (necessary so the no-op path can
                // leave β̂ at the seed for a clean re-install) pollutes that nuisance
                // state, so a subsequent re-eval of the cleanly-converged ρ comes back
                // a few REML units ABOVE its true certified cost — e.g. a Gamma/log
                // optimum certified at 829.857 re-evaluates at 834.90 right after the
                // seed probe, which is then (wrongly) above the seed's own 833.47 and
                // the guard "escapes" to the under-penalized seed, shipping a
                // near-full-basis overfit (EDF ≈ k, falsely tagged converged) that the
                // seed loop's keep-best had already rejected (#1426 silent overfit;
                // #1477 Tweedie boundary/EDF blow-up). `final_value` was scored at the
                // converged ρ with ITS own inner solve, so it is immune to that probe
                // pollution and is the honest cost to compare the seed against.
                let cost_converged = strategy_result.final_value;
                // The seed probe is a non-fatal measurement; a seed that fails to
                // evaluate simply skips the comparison. It leaves β̂ at the seed, so
                // the no-op branch below relies on the unified β̂ re-install after the
                // guard to restore it at `strategy_result.rho`.
                //
                // Probe the seed WITH its outer gradient (not cost alone): the grid
                // prepass scored the seed by `compute_cost`, which runs the inner
                // P-IRLS — at an under-penalized (λ→0) ρ the inner solve hits its
                // iteration cap and reports a spuriously LOW cost (an invalid REML
                // value the line search could not improve) while the analytic outer
                // gradient still points strongly toward more penalization. The #1371
                // false high-λ shelf this guard exists to escape is, by contrast, a
                // GENUINE cheaper optimum: its seed is stationary. Even with the
                // pollution-free `final_value` comparison above, a stuck stall can
                // still under-cut it on raw cost, so only a seed whose cost is
                // trustworthy (small residual gradient) may override the certified ρ.
                let seed_eval = reml_state.compute_cost_and_gradient(seed).ok();
                // Strict relative improvement so a numerically-equal seed (the common
                // case where the optimizer reached the seed's basin) is left untouched
                // and the fit stays byte-identical.
                let floor = 1e-6 * (1.0 + cost_converged.abs());
                if let Some((cost_seed, grad_seed)) = seed_eval.filter(|(c, _)| c.is_finite())
                    && cost_converged.is_finite()
                    && cost_seed < cost_converged - floor
                {
                    // Bound-projected residual gradient at the seed (same criterion
                    // `nonconverged_cost_is_trustworthy` / the flat-valley stall guard
                    // use): a component pinned at a bound by a gradient pushing past
                    // it is feasible-stationary and drops out of the norm.
                    let (blo, bhi) = {
                        let (a, b) = reml_seed_config.bounds;
                        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                        (lo, hi.max(crate::estimate::RHO_BOUND))
                    };
                    let seed_grad_norm = {
                        let mut sumsq = 0.0;
                        for (i, &g) in grad_seed.iter().enumerate() {
                            let s = seed.get(i).copied().unwrap_or(0.0);
                            let pinned_lo = s <= blo + 1e-9 && g > 0.0;
                            let pinned_hi = s >= bhi - 1e-9 && g < 0.0;
                            if !(pinned_lo || pinned_hi) {
                                sumsq += g * g;
                            }
                        }
                        sumsq.sqrt()
                    };
                    let seed_cost_trustworthy = seed_grad_norm.is_finite()
                        && seed_grad_norm
                            <= crate::rho_optimizer::FLAT_VALLEY_STALL_GRAD_CEILING;
                    if seed_cost_trustworthy {
                        log::info!(
                            "[OUTER] #1371 release-and-rerank: certified ρ cost {cost_converged:.6e} \
                         exceeds the prepass seed cost {cost_seed:.6e} (seed |g|={seed_grad_norm:.3e} \
                         ≤ ceiling); adopting the seed (false high-λ stationary shelf escaped)"
                        );
                        strategy_result.rho = seed.clone();
                        strategy_result.converged = true;
                    } else {
                        // #1426 leak: the cheaper seed is a stuck under-penalized
                        // (λ→0) stall, not a genuine optimum — its low cost is an
                        // inner-cap artifact. Adopting it would ship the near-full-
                        // basis overfit (EDF ≈ k) and, worse, certify it converged.
                        // Keep the honest certified ρ. β̂ is restored by the unified
                        // re-install after the guard.
                        log::info!(
                            "[OUTER] #1371 release-and-rerank: prepass seed cost {cost_seed:.6e} is \
                         cheaper than certified ρ {cost_converged:.6e} but UNTRUSTWORTHY \
                         (seed |g|={seed_grad_norm:.3e} > ceiling — stuck under-penalized stall, \
                         #1426); keeping the certified ρ"
                        );
                    }
                }
                // Re-install β̂ at the (possibly newly-adopted) reported ρ so the
                // cached inner state matches `strategy_result.rho` for the downstream
                // cap-guard / final assembly — whether the guard fired (β̂ → seed) or
                // was a no-op (β̂ → the certified ρ, undoing the seed probe).
                reml_state.compute_cost(&strategy_result.rho)?;
            }
            // #1074 UPPER-BOUND INWARD-DESCENT ESCAPE. The outer cost-stall /
            // convergence check projects out a coordinate sitting on the ρ upper
            // bound, so a coordinate that was driven to the over-smoothing rail can
            // be certified "converged" even when the REML criterion is strictly
            // LOWER at an interior ρ (a feasible inward descent the projection
            // masks). On `s(long,lat,bs="tp") + s(depth)` the spatial/depth
            // NULL-SPACE (affine-trend) coordinates rail to ρ=30 while
            // `compute_cost` is ~23/~5 units lower at ρ≈2, annihilating a SUPPORTED
            // spatial trend (#1074). This guard runs a bounded, keep-best
            // coordinate-descent polish on EXACTLY the coordinates pinned at the
            // upper bound: for each, it line-searches `compute_cost` over a coarse
            // inward grid and adopts the strictly-best ρ. It uses the same
            // authoritative `compute_cost` the optimizer minimizes, so it can only
            // LOWER the certified cost — it never raises it and is a no-op when the
            // rail genuinely is the optimum (an unsupported term shrinking out,
            // #1266/#1271: no interior point is cheaper, so nothing is adopted).
            {
                let rho_upper = crate::estimate::RHO_BOUND;
                // Coordinates eligible for the inward (less-smoothing) descent. Two
                // kinds qualify:
                //   (1) any coordinate pinned at the ρ upper rail — the original
                //       #1074 case: the outer convergence check projects out a
                //       rail-pinned coordinate, masking a feasible interior descent;
                //   (2) the well-determined double-penalty NULL-SPACE selection
                //       coordinates (`is_nullspace_degeneracy_prior`). These sit on a
                //       near-FLAT REML ridge in λ_null, so the outer optimizer can
                //       certify convergence at ANY high-but-not-railed ρ_null
                //       depending on its (floating-point) iterate path. Reflecting the
                //       covariate `x → −x` reverses the basis column order and flips
                //       that landing shoulder: a SUPPORTED affine trend is kept at
                //       ρ_null ≈ 0 in one orientation but over-penalized to e.g.
                //       ρ_null ≈ 25 in the mirror (#1548), even though neither is at
                //       the exact rail. Descending these to the cheaper interior
                //       optimum lands BOTH orientations on the same shoulder.
                // The descent below probes ONLY strictly-lower ρ, so it never
                // over-smooths (raising λ_null is the #1266 escape's job, with its
                // EDF parsimony guard against the #1476 concurvity transfer). It is
                // keep-best + cold-confirmed against the authoritative penalized cost,
                // so it is an exact no-op wherever the current ρ already is the
                // optimum — e.g. an unsupported trend correctly shrunk out at the rail
                // (#1266/#1271), where no interior point is cheaper.
                let mut descent_coords: Vec<usize> = (0..strategy_result.rho.len())
                    .filter(|&i| strategy_result.rho[i] >= rho_upper - 1e-9)
                    .collect();
                if n_design_rows >= 2 * p
                    && let gam_problem::RhoPrior::Independent(per_coord) =
                        reml_state.effective_rho_prior().as_ref()
                {
                    for i in 0..strategy_result.rho.len() {
                        if strategy_result.rho[i] < rho_upper - 1e-9
                            && per_coord
                                .get(i)
                                .is_some_and(gam_terms::smooth::is_nullspace_degeneracy_prior)
                        {
                            descent_coords.push(i);
                        }
                    }
                }
                // Baseline to beat = the optimizer's OWN authoritative converged
                // cost (`final_value`), which was scored at the converged ρ with its
                // own inner solve and is immune to warm-start pollution from the
                // probes below (the #1371 lesson). A probe only wins if it is
                // strictly cheaper than this honest cost.
                let base_cost = strategy_result.final_value;
                if !descent_coords.is_empty() && base_cost.is_finite() {
                    // Inward probe grid (descending from the rail). Bounded and
                    // cheap: at most 2 · |railed| · 8 inner solves, and only when a
                    // coord is actually pinned at the upper rail. Two coordinate-
                    // descent passes pick up cross-coordinate coupling between the
                    // railed axes.
                    const INWARD_GRID: [f64; 8] = [25.0, 20.0, 15.0, 10.0, 5.0, 2.0, 0.0, -2.0];
                    let mut best_rho = strategy_result.rho.clone();
                    let mut best_cost = base_cost;
                    let mut improved = false;
                    for _pass in 0..2 {
                        let mut pass_improved = false;
                        for &coord in &descent_coords {
                            let mut local_best = best_rho.clone();
                            let mut local_cost = best_cost;
                            for &cand in &INWARD_GRID {
                                // Inward escape only ever DESCENDS (less smoothing):
                                // skip any grid point at or above this coordinate's
                                // current ρ. Over-smoothing a null-space coordinate is
                                // the #1266 escape's job (it carries the EDF parsimony
                                // guard that prevents the #1476 concurvity transfer);
                                // this guard must never raise λ without it.
                                if cand >= best_rho[coord] - 1e-9 {
                                    continue;
                                }
                                let mut probe = best_rho.clone();
                                probe[coord] = cand;
                                if let Ok(c) = reml_state.compute_cost(&probe)
                                    && c.is_finite()
                                    && c < local_cost - 1e-6 * (1.0 + local_cost.abs())
                                {
                                    local_cost = c;
                                    local_best = probe;
                                }
                            }
                            if local_cost < best_cost - 1e-6 * (1.0 + best_cost.abs()) {
                                best_rho = local_best;
                                best_cost = local_cost;
                                improved = true;
                                pass_improved = true;
                            }
                        }
                        if !pass_improved {
                            break;
                        }
                    }
                    // CONTINUOUS REFINEMENT of each descended coordinate. The coarse
                    // INWARD_GRID snaps λ to a grid node (e.g. ρ_null = 0), but in the
                    // OTHER covariate orientation the outer optimizer reports the
                    // continuous interior minimizer (e.g. ρ_null = −0.37). Leaving one
                    // orientation on the grid node while the other keeps the continuous
                    // optimum leaves a small residual reflection asymmetry (#1548:
                    // ~1.7e-3 mirror drift survives the grid descent alone). Golden-
                    // section the SAME authoritative penalized cost on each moved
                    // coordinate so both orientations converge to the identical
                    // continuous minimum. It can only lower the cost from the grid node
                    // (the bracket straddles it), and the cold confirmation below still
                    // gates adoption, so this never raises the certified cost.
                    if improved {
                        const GS_R: f64 = 0.618_033_988_749_894_8; // (√5 − 1) / 2
                        for coord in descent_coords.clone() {
                            if (best_rho[coord] - strategy_result.rho[coord]).abs() <= 1e-9 {
                                continue; // coordinate did not descend
                            }
                            // Bracket straddling the adopted grid node, never re-entering
                            // the over-smoothing region above the coordinate's start ρ.
                            let node = best_rho[coord];
                            let mut a = node - 3.0;
                            let mut b = (node + 3.0).min(strategy_result.rho[coord]);
                            if b <= a + 1e-6 {
                                continue;
                            }
                            let cost_at =
                                |st: &mut RemlState, base: &Array1<f64>, x: f64| -> Option<f64> {
                                    let mut p = base.clone();
                                    p[coord] = x;
                                    st.compute_cost(&p).ok().filter(|c| c.is_finite())
                                };
                            let mut c = b - GS_R * (b - a);
                            let mut d = a + GS_R * (b - a);
                            let mut fc = cost_at(&mut reml_state, &best_rho, c);
                            let mut fd = cost_at(&mut reml_state, &best_rho, d);
                            let mut refine_ok = fc.is_some() && fd.is_some();
                            for _ in 0..40 {
                                if (b - a).abs() < 1e-4 {
                                    break;
                                }
                                match (fc, fd) {
                                    (Some(vc), Some(vd)) if vc <= vd => {
                                        b = d;
                                        d = c;
                                        fd = fc;
                                        c = b - GS_R * (b - a);
                                        fc = cost_at(&mut reml_state, &best_rho, c);
                                    }
                                    (Some(_), Some(_)) => {
                                        a = c;
                                        c = d;
                                        fc = fd;
                                        d = a + GS_R * (b - a);
                                        fd = cost_at(&mut reml_state, &best_rho, d);
                                    }
                                    _ => {
                                        refine_ok = false;
                                        break;
                                    }
                                }
                            }
                            if refine_ok {
                                let xm = 0.5 * (a + b);
                                if let Some(fm) = cost_at(&mut reml_state, &best_rho, xm)
                                    && fm < best_cost
                                {
                                    best_rho[coord] = xm;
                                    best_cost = fm;
                                }
                            }
                        }
                    }
                    if improved {
                        // COLD CONFIRMATION (guards against adopting a warm-start /
                        // inner-cap artifact, the #1426/#1371 trap). The grid probes
                        // ran warm-started off each other; a λ→0-ish interior point
                        // can report a spuriously low cost from a capped inner solve.
                        // Clear the inner cache and re-score the candidate cold; only
                        // adopt if it STILL strictly beats the authoritative
                        // `final_value`.
                        reml_state.reset_outer_seed_state();
                        let cold = reml_state.compute_cost(&best_rho);
                        let cold_ok = matches!(cold, Ok(c)
                        if c.is_finite() && c < base_cost - 1e-6 * (1.0 + base_cost.abs()));
                        if cold_ok {
                            let cold_cost = cold.unwrap_or(best_cost);
                            log::info!(
                                "[OUTER] #1074/#1548 upper-bound escape: certified ρ cost \
                             {base_cost:.6e} lowered to {cold_cost:.6e} (cold-confirmed) by \
                             descending {} over-smoothed coord(s) inward; adopting the cheaper \
                             interior ρ",
                                descent_coords.len()
                            );
                            strategy_result.rho = best_rho;
                            strategy_result.final_value = cold_cost;
                            // β̂ already installed at `best_rho` by the cold eval above.
                        } else {
                            // The improvement did not survive a cold re-score — it was
                            // a warm-start artifact. Keep the certified ρ and restore
                            // its inner state for downstream assembly.
                            reml_state.reset_outer_seed_state();
                            reml_state.compute_cost(&strategy_result.rho)?;
                        }
                    }
                }
            }
            // #1266 NULL-SPACE SHRINK-OUT ESCAPE (pure-REML; the OUTWARD-direction
            // dual of the #1074 inward escape above).
            //
            // A default double-penalty smooth (mgcv `select = TRUE`) carries a
            // `DoublePenaltyNullspace` shrinkage ridge on the term's penalty null
            // space ({1, x} for a 1-D bend) whose only job is SELECTION: drive its
            // λ_null UP to shrink an UNSUPPORTED term's constant+linear component out
            // (EDF → 0). On a well-determined Gaussian fit the relaxed ρ-prior places
            // a WIDE, symmetric `Normal(0, sd=15)` on that coordinate — NOT as a
            // selection criterion but purely as a degeneracy-breaker: the #1476
            // concurvity flat-ridge needs strictly-positive outer curvature to
            // certify an interior allocation. That symmetric prior's `ρ/sd²` gradient
            // also OPPOSES the (genuinely shallow) REML shrink-out tail, so the outer
            // optimizer certifies a stationary point at a MODERATE λ_null
            // (ρ_null ≈ 3.5, EDF ≈ 1.6) instead of following pure REML to the
            // shrink-out corner — the residual #1266 "Half B" contract violation. The
            // prior cannot be made one-sided: its high-ρ curvature is exactly what
            // stops a SUPPORTED concurvity null space from railing out (#1476), so a
            // data-INDEPENDENT prior cannot separate "shrink the unsupported term"
            // (#1266) from "keep the supported one" (#1476/#1371) — they overlap in ρ.
            //
            // The data-DEPENDENT discriminator is pure data-REML PLUS a parsimony
            // check. For each well-determined null-space selection coordinate,
            // line-search the OVER-SMOOTHING (high-ρ) direction on the PURE REML cost
            // (`compute_cost − configured_ρ_prior`; the prior is a conditioning
            // device, not a selection criterion), then adopt the strictly-best
            // COLD-confirmed point ONLY if it also does not increase the model's total
            // EDF:
            //   * UNSUPPORTED, uncorrelated null space (#1266 `s(z)`): pure REML
            //     descends toward shrink-out AND total EDF drops (z carries no signal,
            //     so nothing absorbs it) → the escape fires, EDF → 0.
            //   * SUPPORTED null space (#1371 genuine slope): pure REML strictly RISES
            //     under over-smoothing (killing a real linear trend dumps its variance
            //     into σ̂²) → no strict improvement → exact no-op.
            //   * CONCURVITY null space (#1476 `s(x1)+s(x2)`, corr ≈ 0.9): pure REML
            //     *marginally* prefers over-smoothing one coordinate because the inner
            //     β re-solve lets the CORRELATED partner absorb the shared signal — the
            //     "signal transfer" the degeneracy prior exists to forbid. That
            //     transfer keeps the deviance flat but INFLATES total EDF (the partner
            //     spends extra basis), so the EDF-non-increase guard vetoes it →
            //     no-op, the interior allocation is kept. (Pure REML alone cannot see
            //     this: the concurvity ridge is flat, so the transfer reads as a tiny
            //     improvement; the parsimony guard is what distinguishes a genuine
            //     simplification from a lateral reallocation.)
            //
            // Unlike #1074 (where the OPTIMIZER's bound projection masks the descent),
            // here it is the PRIOR that masks it, so the search runs on the pure
            // (prior-stripped) criterion. SCOPE: eligible coordinates are exactly the
            // well-determined relaxed null-space degeneracy coordinates
            // (`is_nullspace_degeneracy_prior`, gated by `n ≥ 2·p`). This deliberately
            // EXCLUDES the under-determined regime (`n < 2·p`, #1392 wine `p > n`),
            // where the null-space prior is the AGGRESSIVE PC select-out — a
            // deliberate, load-bearing selection push onto a genuinely-flat REML
            // score that stripping would undo.
            {
                let well_determined = n_design_rows >= 2 * p;
                // ELIGIBLE COORDINATES: every RELAXED smoothing coordinate in the
                // well-determined `Independent` prior — BOTH the null-space
                // shrinkage selection coordinate (`is_nullspace_degeneracy_prior`,
                // the wide `Normal(0,15)` degeneracy breaker) AND the BENDING
                // (wiggliness) coordinate (relaxed to `Flat`).
                //
                // The bending coordinate is the load-bearing addition (#1266
                // "Half A"). `relax_smoothing_rho_prior` frees a well-determined
                // B-spline-family smooth's bending log-λ to `Flat` so that pure
                // REML — not the prior — picks it (matching mgcv). On linear /
                // weakly-curved data pure REML wants λ_bend → ∞ (the bending
                // 2nd-difference penalty annihilates {1, x}, so a huge λ_bend
                // removes only spurious wiggle and leaves the linear null space
                // at EDF 2). MEASURED: a flat-prior-everywhere fit rails λ_bend to
                // 1e9–1e13 and lands EDF ≈ 2.00 on every seed. But the production
                // `Independent` prior also places the `Normal(0,15)` degeneracy
                // breaker on the SAME term's null-space coordinate; that prior's
                // curvature perturbs the JOINT outer BFGS so it certifies
                // termination with λ_bend still at a moderate ≈2e5 (EDF ≈ 3–5) —
                // the bending coordinate stalls short of its own pure-REML optimum.
                // The escape line-searches it the rest of the way OUT on the pure
                // (prior-stripped) REML cost, with the SAME self-validating guards
                // used for the null coordinate (strict pure-REML improvement +
                // total-EDF-non-increase + cold confirmation):
                //   * LINEAR / weakly-curved smooth (#1266 Half A, #1271): pure
                //     REML strictly descends toward λ_bend → ∞ and total EDF drops
                //     (only spurious wiggle is removed) → the escape fires, EDF → 2.
                //   * GENUINELY WIGGLY smooth (the supported `s(x)` on `sin(6x)`):
                //     pure REML has an interior optimum in λ_bend, so over-smoothing
                //     strictly RAISES it (real curvature dumped into σ̂²) → no strict
                //     improvement → exact no-op. The supported signal is preserved.
                // Within an `Independent` prior a per-coordinate `Flat` is, by
                // construction, exactly a relaxed bending coordinate: a fully-`Flat`
                // BASE prior is returned un-wrapped (never `Independent`), so the
                // only way a coordinate reads `Flat` inside `Independent` is the
                // relaxation having freed it.
                let select_coords: Vec<usize> = if well_determined {
                    // The BENDING (range-space wiggliness) coordinates are those the
                    // orchestration relaxed to `Flat` — recovered from the CONFIGURED
                    // prior's firth-default mask (in the post-resolution
                    // `effective_rho_prior` a `Flat` coordinate has been rewritten to
                    // the firth-default `PenalizedComplexity` and is no longer
                    // distinguishable as relaxed). The NULL-SPACE shrinkage selection
                    // coordinates carry the wide `Normal(0,15)` degeneracy breaker,
                    // detected on the effective prior.
                    let bend_mask = reml_state.firth_default_coord_mask(strategy_result.rho.len());
                    let null_is_degeneracy: Vec<bool> =
                        match reml_state.effective_rho_prior().as_ref() {
                            gam_problem::RhoPrior::Independent(per_coord) => (0..strategy_result
                                .rho
                                .len())
                                .map(|i| {
                                    per_coord.get(i).is_some_and(|p| {
                                        gam_terms::smooth::is_nullspace_degeneracy_prior(p)
                                    })
                                })
                                .collect(),
                            _ => vec![false; strategy_result.rho.len()],
                        };
                    (0..strategy_result.rho.len())
                        .filter(|&i| {
                            bend_mask.get(i).copied().unwrap_or(false)
                                || null_is_degeneracy.get(i).copied().unwrap_or(false)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                // Authoritative pure-REML baseline at the converged ρ: the optimizer's
                // own `final_value` (immune to warm-start pollution, the #1371 lesson)
                // minus the configured ρ-prior + soft λ→0 guard it carried. A probe
                // wins only if it strictly beats THIS pure cost.
                let conv_prior = reml_state
                    .configured_rho_prior_atom(&strategy_result.rho)
                    .cost()
                    + reml_state
                        .soft_rho_guard_prior_atom(&strategy_result.rho)
                        .cost();
                let base_pure = strategy_result.final_value - conv_prior;
                if !select_coords.is_empty() && base_pure.is_finite() && conv_prior.is_finite() {
                    // Drop the coarse inner-PIRLS cap the outer-aware schedule left
                    // behind (path #3: a fast-converging outer loop ends on a 5/10/20
                    // inner-iteration cap, not the full budget). Every escape
                    // evaluation below — `edf_conv`, the cold over-smoothing probes,
                    // and `edf_best` — MUST run the inner β to full tolerance, else
                    // the EDF read at a high-λ shrink-out point is a coarse-cap
                    // artifact (β not converged → EDF spuriously railed near the
                    // interpolation limit), which would make the parsimony guard
                    // veto a genuine shrink-out. Resetting the cap here makes
                    // `compute_cost` / `obtain_eval_bundle` solve to `tol`.
                    reml_state.outer_inner_cap.store(0, Ordering::Relaxed);
                    reml_state.reset_outer_seed_state();
                    // Converged-point total inner EDF, for the PARSIMONY guard below.
                    // The inner P-IRLS solve at the converged ρ is cached, so this is
                    // free. A genuine #1266 shrink-out (an UNSUPPORTED, uncorrelated
                    // term selected out) strictly LOWERS the model's total EDF; a
                    // concurvity TRANSFER (#1476: one null-space shrinks but its
                    // correlated partner absorbs the signal via the inner β re-solve)
                    // INFLATES it. Pure REML alone marginally prefers the transfer on
                    // a flat concurvity ridge — exactly the allocation the degeneracy
                    // prior exists to forbid — so the escape must additionally refuse
                    // any adoption that does not reduce total EDF.
                    let edf_conv = reml_state
                        .obtain_eval_bundle(&strategy_result.rho)
                        .ok()
                        .map(|b| b.pirls_result.edf);
                    // Pure data-REML at ρ: penalized `compute_cost` minus the configured
                    // ρ-prior and the soft λ→0 guard (both `O(K)` functions of ρ alone).
                    // Subtracting them recovers the mgcv-parity criterion selection
                    // must follow; the prior bias on λ_null is removed exactly.
                    let pure_reml = |rho: &Array1<f64>| -> Option<f64> {
                        let c = reml_state.compute_cost(rho).ok()?;
                        if !c.is_finite() {
                            return None;
                        }
                        let prior = reml_state.configured_rho_prior_atom(rho).cost()
                            + reml_state.soft_rho_guard_prior_atom(rho).cost();
                        if !prior.is_finite() {
                            return None;
                        }
                        Some(c - prior)
                    };
                    // Re-establish the pure-REML baseline at FULL inner tolerance (the
                    // cap was just dropped above). The pre-cap-reset `base_pure` from
                    // `final_value` may carry a coarse-cap inner solve; re-scoring the
                    // converged ρ here makes the strict-improvement test below compare
                    // converged-vs-converged costs. Falls back to the prior `base_pure`
                    // if the re-score is unavailable.
                    let base_pure = pure_reml(&strategy_result.rho).unwrap_or(base_pure);
                    // Ascending over-smoothing grid in ABSOLUTE ρ (toward the
                    // shrink-out rail at `RHO_BOUND`); only values strictly above a
                    // coordinate's current ρ are over-smoothing candidates. Bounded:
                    // at most 2 · |select| · 6 inner solves, and only fires when a
                    // null-space coordinate is actually held below the rail.
                    let rho_upper = crate::estimate::RHO_BOUND;
                    const OUTWARD_GRID: [f64; 6] = [6.0, 9.0, 12.0, 18.0, 24.0, 30.0];
                    // INDEPENDENT COLD per-coordinate over-smoothing search.
                    //
                    // Each candidate is scored from a FRESHLY-RESET inner state with
                    // ONLY ONE selection coordinate raised above its converged ρ. This
                    // is essential for a MULTI-TERM fit (#1266 Half B: `s(x)+s(z)` with
                    // a SUPPORTED wiggly `s(x)` beside an UNSUPPORTED `s(z)`):
                    //   * over-smoothing a coordinate of the SUPPORTED term strictly
                    //     RAISES pure REML (real signal dumped into σ̂²) → rejected;
                    //   * over-smoothing the UNSUPPORTED term's null-space coordinate
                    //     strictly LOWERS pure REML (its penalty-determinant cost was
                    //     not buying any fit) → accepted, shrinking that term out.
                    // The earlier WARM coupled coordinate-descent bundled the two: its
                    // warm-start pollution manufactured a spurious joint cliff that
                    // raised the supported term's λ too, producing an interpolating
                    // (EDF-inflated) point the parsimony guard then vetoed — so an
                    // unsupported term that pure REML genuinely wants shrunk was left
                    // under-shrunk. Scoring each coordinate independently AND cold
                    // isolates exactly the coordinates pure REML wants over-smoothed.
                    //
                    // Cost: at most |select|·|grid| cold inner solves, only on a
                    // well-determined relaxed double-penalty fit after convergence.
                    let pure_reml_cold = |rho: &Array1<f64>| -> Option<f64> {
                        reml_state.reset_outer_seed_state();
                        pure_reml(rho)
                    };
                    let mut best_rho = strategy_result.rho.clone();
                    let mut improved = false;
                    for &coord in &select_coords {
                        let mut coord_target = strategy_result.rho[coord];
                        // A coordinate's over-smoothing is accepted only if, scored
                        // COLD with that single coordinate raised from the converged ρ:
                        //   (1) pure REML strictly beats the converged pure baseline
                        //       (distinguishes a SUPPORTED term — over-smoothing raises
                        //       its cost — from an UNSUPPORTED one), AND
                        //   (2) the total inner EDF does NOT exceed the converged EDF.
                        // Guard (2) is BOTH the #1476 parsimony rule (over-smoothing a
                        // coordinate adds shrinkage, so it can only LOWER total EDF; a
                        // reported INCREASE is a concurvity transfer that must be
                        // refused) AND a numerical-sanity filter: at the ρ rail
                        // (λ ≳ 1e10) the hat-matrix EDF trace loses precision and
                        // reports a spurious EDF far above the design's column count.
                        // Rejecting any EDF-increasing probe discards exactly those
                        // numerically-broken rail rungs while keeping the stable ones
                        // (λ ≲ 1e8 already annihilates any unsupported direction), so
                        // the deepest numerically-sound shrink is selected.
                        let mut coord_pure = base_pure;
                        for &cand in &OUTWARD_GRID {
                            let target = cand.min(rho_upper);
                            if target <= strategy_result.rho[coord] + 1e-9 {
                                continue;
                            }
                            let mut probe = strategy_result.rho.clone();
                            probe[coord] = target;
                            let c = pure_reml_cold(&probe);
                            let edf_here = reml_state
                                .obtain_eval_bundle(&probe)
                                .ok()
                                .map(|b| b.pirls_result.edf);
                            let edf_sane = match (edf_here, edf_conv) {
                                (Some(e), Some(ec)) => e.is_finite() && e <= ec + 1e-6,
                                _ => false,
                            };
                            if let Some(c) = c
                                && edf_sane
                                && c < coord_pure - 1e-6 * (1.0 + coord_pure.abs())
                            {
                                coord_pure = c;
                                coord_target = target;
                            }
                        }
                        if coord_target > strategy_result.rho[coord] + 1e-9 {
                            best_rho[coord] = coord_target;
                            improved = true;
                        }
                    }
                    if improved {
                        // COLD confirmation (mirror of #1074): the warm grid probes
                        // ran off each other's inner warm starts and can report a
                        // spuriously-low cost. Clear the inner cache and re-score the
                        // candidate cold; adopt only if its PURE REML STILL strictly
                        // beats the authoritative converged baseline.
                        reml_state.reset_outer_seed_state();
                        let cold_penalized = reml_state.compute_cost(&best_rho);
                        let cold_pure = cold_penalized.as_ref().ok().and_then(|&c| {
                            c.is_finite().then(|| {
                                c - reml_state.configured_rho_prior_atom(&best_rho).cost()
                                    - reml_state.soft_rho_guard_prior_atom(&best_rho).cost()
                            })
                        });
                        // Total inner EDF at the candidate (cached from the cold eval).
                        // The PARSIMONY guard: a genuine shrink-out must not INCREASE
                        // the model's effective dimension (see `edf_conv`). When either
                        // EDF is unavailable, refuse the adoption — a shrink that can't
                        // be certified parsimonious is not worth the #1476 risk.
                        let edf_best = reml_state
                            .obtain_eval_bundle(&best_rho)
                            .ok()
                            .map(|b| b.pirls_result.edf);
                        let edf_non_increasing = match (edf_best, edf_conv) {
                            (Some(eb), Some(ec)) => eb <= ec + 1e-6,
                            _ => false,
                        };
                        if let (Ok(penalized), Some(cold_pure)) = (cold_penalized, cold_pure)
                            && cold_pure.is_finite()
                            && cold_pure < base_pure - 1e-6 * (1.0 + base_pure.abs())
                            && edf_non_increasing
                        {
                            // β̂ already installed at `best_rho` by the cold eval above.
                            // Report the PENALIZED cost there as the objective so the
                            // cached inner state and `final_value` agree with the
                            // adopted ρ for the downstream cap-guard / assembly.
                            log::info!(
                                "[OUTER] #1266 null-space shrink-out escape: pure REML \
                             {base_pure:.6e} → {cold_pure:.6e} (cold-confirmed), total \
                             EDF {edf_conv:?} → {edf_best:?} (parsimonious) by \
                             over-smoothing {} selection coord(s); adopting the \
                             shrink-out ρ (penalized cost {penalized:.6e})",
                                select_coords.len()
                            );
                            strategy_result.rho = best_rho;
                            strategy_result.final_value = penalized;
                        } else {
                            // The improvement did not survive a cold re-score (or the
                            // re-score failed) — a warm-start artifact. Keep the
                            // certified ρ and restore its inner state.
                            reml_state.reset_outer_seed_state();
                            reml_state.compute_cost(&strategy_result.rho)?;
                        }
                    }
                }
            }
            // Convergence guard for the outer-aware inner-PIRLS schedule
            // (path #3): the BFGS bridge stores a coarsen-then-tighten cap
            // into `reml_state.outer_inner_cap` on every accepted gradient
            // eval. After the outer optimizer returns, the cached warm-start
            // β was computed at whatever cap the schedule last set — which
            // for fast-converging fits (≤5 BFGS iters) is a coarse cap of
            // 5/10/20 rather than the full inner budget. Reset the cap to 0
            // and run one final cost eval at the converged ρ so the cached
            // β is at full inner tolerance.
            run_outer_inner_cap_guard(
                &mut reml_state,
                &strategy_result.rho,
                RemlInnerCapGuardArm::Standard,
            )?;
            // Honour an explicit caller rho seed as the accepted log-λ: when the
            // caller pins `init_rhos`, the outer search is warm-started there and
            // the seed is the requested operating point, so report it verbatim
            // rather than the optimizer's (possibly clamped) returned rho.
            //
            // EXCEPTION (gam#1464): a caller seed that arrives as a warm-start hint
            // (the spatial-κ sweep reuses the previous-κ λ̂ as `heuristic_lambdas`)
            // must NOT pin the fit at a seed the optimizer has just been able to
            // strictly improve on. At a collapsing kernel (the constant-curvature
            // `curv()` smooth on the +κ side) the warm seed sits in a shallow
            // under-smoothing basin whose spuriously-low deviance, if reported
            // verbatim, makes the κ outer objective rail to the +chart bound for any
            // curved data. The objective-grid prepass and the #1371 release-and-
            // rerank guard above redirect `strategy_result.rho` into the correct
            // high-λ basin; defer to that converged ρ whenever it is STRICTLY cheaper
            // than the caller seed under the same REML cost. A genuine user pin (or a
            // healthy warm start) converges at the seed, so the seed stays cheapest
            // and is honoured verbatim, byte-for-byte as before.
            let accepted_rho = match heuristic_lambdas.filter(|h| h.len() == k) {
                Some(h) => {
                    let seed = Array1::from_iter(h.iter().copied());
                    let prefer_converged = {
                        let cost_seed = reml_state.compute_cost(&seed).ok();
                        let cost_converged = reml_state.compute_cost(&strategy_result.rho).ok();
                        // Restore the cached β̂ to the converged operating point after
                        // the seed probe (the no-op path below expects β̂ at
                        // `strategy_result.rho`). Propagate any failure rather than
                        // swallowing it: proceeding with β̂ at the wrong operating
                        // point would silently corrupt the reported fit.
                        reml_state.compute_cost(&strategy_result.rho)?;
                        match (cost_seed, cost_converged) {
                            (Some(cs), Some(cc)) if cs.is_finite() && cc.is_finite() => {
                                cc < cs - 1e-6 * (1.0 + cs.abs())
                            }
                            _ => false,
                        }
                    };
                    if prefer_converged {
                        log::info!(
                            "[OUTER] #1464 warm-seed override: converged ρ is strictly cheaper than \
                         the caller warm seed; reporting the optimizer's ρ instead of the seed"
                        );
                        strategy_result.rho.clone()
                    } else {
                        seed
                    }
                }
                None => strategy_result.rho.clone(),
            };
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
            use gam_problem::{DeclaredHessianForm, Derivative, HessianResult, OuterEval};
            let initial_link_kind = cfg.link_kind.clone();
            let prefer_gradient_only = theta_dim >= REML_SECOND_ORDER_RHO_CAP;
            let continuation_prewarm = theta_dim < REML_CONTINUATION_PREWARM_RHO_CAP;
            if prefer_gradient_only {
                log::info!(
                    "[OUTER] theta_dim {theta_dim} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
                );
            }
            if !continuation_prewarm {
                log::info!(
                    "[OUTER] theta_dim {theta_dim} reaches continuation-prewarm budget \
                   ({REML_CONTINUATION_PREWARM_RHO_CAP}); starting optimizer directly from seeds"
                );
            }
            let problem = OuterProblem::new(theta_dim)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_prefer_gradient_only(prefer_gradient_only)
            .with_continuation_prewarm(continuation_prewarm)
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
            let apply_link_theta =
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
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
                let cost = state.compute_cost(&rho)? + sas_ridge_cost(theta);
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
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianResult::Analytic(hessian),
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
            // Same publish/consume symmetry as the standard REML arm above
            // (issue #236). The mixture/SAS eval closure also surfaces
            // `inner_beta_hint = state.current_original_basis_beta()` (see
            // src/solver/estimate.rs:3275), so continuation pre-warm needs
            // a real seed hook to install it.
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());
            let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
            drop(obj);
            // Convergence guard for the outer-aware inner-PIRLS schedule
            // (path #3) — see the matching comment in the standard REML arm
            // above. Reset the cap and run one final compute_cost at the
            // converged θ so the cached warm-start β is at full inner
            // tolerance regardless of where the BFGS schedule was when the
            // optimizer terminated.
            //
            // The outer vector here is the AUGMENTED θ = [ρ_smooth (k) | link
            // params (mixture_dim and/or sas_dim)], not a smoothing-only ρ.
            // `compute_cost` exponentiates its argument wholesale into the
            // penalty λ vector (loop_driver.rs `rho.mapv(exp)`), so the guard
            // must receive exactly the same smoothing-only ρ — and the same
            // installed link state — the outer evaluator operated on, never the
            // raw augmented θ. Feeding the full θ in made the guard hand `k +
            // mixture_dim + sas_dim` "lambdas" to a `k`-penalty reparameterizer,
            // which faults with "Lambda count mismatch" (#1571). Route θ through
            // the same `apply_link_theta` the eval closure (optimizer.rs:1759)
            // and the accept-fit slice (the `final_rho` line just below) use: it
            // installs the converged mixture/SAS link state onto `reml_state`
            // and returns the smoothing-only ρ block.
            let guard_rho = {
                let mut state_ref: &mut crate::estimate::reml::RemlState<'_> =
                    &mut reml_state;
                apply_link_theta(&mut state_ref, &outer_result.rho)?
            };
            run_outer_inner_cap_guard(
                &mut reml_state,
                &guard_rho,
                RemlInnerCapGuardArm::MixtureSas,
            )?;
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
            LogSmoothingParamsView::new(final_rho.view()),
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

        // Negative-Binomial outer θ↔λ alternation decision (#1448, supersedes the
        // #1082 drift diagnostic).
        //
        // θ was frozen at the λ-search value (`frozen_negbin_theta`) so `F(ρ)` is
        // stationary in ρ; the accept-fit above ML-refreshed θ at the converged η.
        // If that refreshed θ_final drifted from the search θ_frozen by more than the
        // joint-stationarity tolerance, the ρ we just selected was optimal for the
        // OLD θ, not θ_final: re-freeze the search at θ_final, reset the outer seed
        // state (eval bundle, PIRLS cache, warm-start signals, inner caps — all keyed
        // to the old θ), and run the ρ search again. Iterate to the (ρ, θ) joint
        // fixed point or until the round cap, after which we accept the last fit and
        // log the residual drift. For non-NB / user-fixed-θ fits the criterion below
        // is never met (θ is not estimated), so the loop breaks on round 0 and the
        // fit is byte-identical to the pre-#1448 single pass.
        let mut should_alternate = false;
        if pirls_res.likelihood.negbin_theta_is_estimated() {
            let frozen_bits = reml_state.frozen_negbin_theta.load(Ordering::Relaxed);
            if frozen_bits != 0
                && let Some(theta_final) = pirls_res.likelihood.negbin_theta()
            {
                let theta_frozen = f64::from_bits(frozen_bits);
                if theta_frozen.is_finite() && theta_frozen > 0.0 && theta_final.is_finite() {
                    let rel_drift =
                        (theta_final - theta_frozen).abs() / theta_frozen.max(f64::MIN_POSITIVE);
                    let drift_pct = rel_drift * 100.0;
                    if rel_drift > NEGBIN_THETA_JOINT_DRIFT_TOL {
                        if negbin_alternation_round + 1 < NEGBIN_OUTER_ALTERNATION_MAX_ROUNDS {
                            log::info!(
                                "[OUTER] negative-binomial θ↔λ alternation round {}: θ drifted \
                             {drift_pct:.1}% (θ_frozen={theta_frozen:.6e} → θ_final={theta_final:.6e}); \
                             re-freezing at θ_final and re-running the ρ search (#1448).",
                                negbin_alternation_round + 1
                            );
                            // Re-freeze the λ-search θ at the refreshed value. The
                            // capture in `solve_for_unified_rho` only writes when the
                            // frozen slot is 0, so a non-zero value here pins every
                            // subsequent λ-search inner solve to θ_final rather than
                            // re-deriving it from the seed η.
                            reml_state
                                .frozen_negbin_theta
                                .store(theta_final.to_bits(), Ordering::Relaxed);
                            // The cached criterion / factor bundle and warm-start
                            // signals were all computed at θ_frozen; drop them so the
                            // next round's ρ search recomputes `F(ρ) = REML(ρ, θ_final)`.
                            reml_state.reset_outer_seed_state();
                            should_alternate = true;
                        } else {
                            log::warn!(
                                "[OUTER] negative-binomial θ↔λ alternation hit the round cap \
                             ({NEGBIN_OUTER_ALTERNATION_MAX_ROUNDS}) with residual θ drift \
                             {drift_pct:.1}% (θ_frozen={theta_frozen:.6e} → θ_final={theta_final:.6e}); \
                             accepting the last fit (#1448)."
                            );
                        }
                    } else {
                        log::debug!(
                            "[OUTER] negative-binomial (ρ, θ) jointly stationary after {} \
                         alternation round(s): drift {drift_pct:.2}% \
                         (θ_frozen={theta_frozen:.6e} → θ_final={theta_final:.6e}).",
                            negbin_alternation_round + 1
                        );
                    }
                }
            }
        }
        if should_alternate {
            negbin_alternation_round += 1;
            continue;
        }
        break;
    } // negbin θ↔λ alternation loop (#1448)
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

    let lambdas = final_rho.mapv(f64::exp);
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank_total = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank_total as f64).max(0.0);
    let mut edf_by_block = vec![0.0; k];
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk), retained so per-term
    // EDF can be assembled as |coeff_range| − Σ tr_kk (issue #1219).
    let mut penalty_block_trace = vec![0.0; k];
    let mut edf_total = 0.0;
    let mut smoothing_correction = None;
    let mut rho_covariance = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut beta_standard_errors = None;
    let mut beta_covariance_corrected = None;
    let mut beta_standard_errors_corrected = None;
    let mut beta_covariance_frequentist = None;
    let mut coefficient_influence = None;
    let mut weighted_gram = None;
    // Factorization of stabilized Hessian in transformed basis, reused for
    // SE computation via solve-on-demand after dispersion is determined.
    let mut edf_factor: Option<Box<dyn FactorizedSystem>> = None;
    let mut bias_correction_beta = None;
    let mut rho_posterior_certificate = None;
    let mut rho_posterior_escalation = None;

    if opts.compute_inference {
        // EDF by block using stabilized H and penalty roots in transformed basis.
        let h = &pirls_res.stabilizedhessian_transformed;
        let p_dim = h.nrows();
        // Sparse-aware factorization with ridge retry — no densification.
        // Uses SymmetricMatrix::factorize() -> sparse Cholesky for sparse,
        // dense Cholesky for dense.
        let factor = {
            let scale = h.max_abs_diag();
            let min_step = scale * 1e-10;
            let mut ridge = 0.0_f64;
            let mut attempts = 0_usize;
            loop {
                let candidate = if ridge > 0.0 {
                    match h.addridge(ridge) {
                        Ok(c) => c,
                        Err(_) => h.clone(),
                    }
                } else {
                    h.clone()
                };
                if let Ok(f) = candidate.factorize() {
                    if ridge > 0.0 {
                        // This ridged factor is reused for the reported standard
                        // errors, covariance, and bias correction below, so those
                        // quantities are stabilized approximations, not the exact
                        // (unridged) Hessian-based values.
                        log::warn!(
                            "Inference Hessian was rank-deficient and required a stabilizing \
                             ridge {:.3e}; reported standard errors, covariance, and bias \
                             correction are computed from the ridge-stabilized factor and are \
                             approximations, not exact unridged values",
                            ridge,
                        );
                    }
                    break f;
                }
                attempts += 1;
                if attempts >= MAX_FACTORIZATION_ATTEMPTS {
                    return Err(EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    });
                }
                ridge = if ridge <= 0.0 { min_step } else { ridge * 10.0 };
            }
        };
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
        // The block-trace channel above factorizes the TRANSFORMED stabilized
        // Hessian with a bespoke 10×-escalation ridge loop. On rank-deficient
        // spatial-smooth corners (degenerate-Hessian thin-plate fits) that loop
        // can take an enormous ridge, inflating Σ tr_kk toward `p` and collapsing
        // `edf_total = p − Σ tr_kk` onto its floor `mp` (e.g. 1.0 for a single
        // smooth) even though the fitted surface — and the influence matrix `F`
        // that the prediction, dispersion, and per-term EDF all consume — has
        // legitimately spent ~70 EDF (issue #1356). The authoritative model
        // definition of EDF is the influence-matrix trace; the per-term EDF
        // (`FitResult::per_term_edf`) reads `tr(F)` over each block. Recompute the
        // per-block penalty traces from the SAME rank-revealing inverse `F` uses
        // (`matrix_inversewith_regularization` of the original-basis Hessian), so
        // `edf_total = p − Σ tr_kk = tr(F)`, `Σ edf_by_block = edf_total`, and the
        // total can never fall below a single term's own EDF. Done before the
        // dispersion `σ̂² = RSS/(n − edf_total)` is formed so it, too, uses the
        // honest effective d.f. (the trace-channel collapse otherwise biased
        // σ̂² high → inflated SEs on the same seeds).
        //
        // Per-block traces `tr_kk = λ_kk·tr(H⁻¹ S_kk)` are basis-invariant; map
        // each canonical block's penalty root into the original coefficient basis
        // (`root_orig = Qs · root_t`) and contract against the original-basis
        // inverse. Restricted to small models (where the dense inverse `F` itself
        // is formed); large models keep the trace-channel value.
        {
            let p_orig = pirls_res.reparam_result.qs.nrows();
            const COV_FULL_INVERSE_MAX_P: usize = 10_000;
            if p_orig <= COV_FULL_INVERSE_MAX_P {
                let h_orig = map_hessian_to_original_basis(&pirls_res)?;
                if let Some(h_inv) =
                    matrix_inversewith_regularization(&h_orig, "edf reconciliation")
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
        match factor.solve(&s_beta_t) {
            Ok(b_t) => {
                let qs = &pirls_res.reparam_result.qs;
                let b_orig = qs.dot(&b_t);
                if b_orig.iter().all(|v| v.is_finite()) {
                    bias_correction_beta = Some(b_orig);
                } else {
                    log::warn!("bias-correction vector contained non-finite entries; skipping");
                }
            }
            Err(e) => {
                log::warn!("bias-correction solve failed: {e}");
            }
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
    // df mgcv uses. When inference is off, edf_total is unavailable, so the MLE
    // RSS/n is returned instead.
    let standard_deviation = match &pirls_res.likelihood.spec.response {
        ResponseFamily::Gaussian => {
            let denom = if opts.compute_inference {
                (n - edf_total).max(1.0)
            } else {
                n.max(1.0)
            };
            (weighted_rss / denom).sqrt()
        }
        ResponseFamily::Gamma => pirls_res.likelihood.gamma_shape().unwrap_or(1.0),
        ResponseFamily::Binomial
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Poisson
        | ResponseFamily::RoystonParmar => 1.0,
    };
    let dispersion = dispersion_from_likelihood(&pirls_res.likelihood, standard_deviation);

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
        .max(f64::MIN_POSITIVE);

    // Compute gradient norm at final rho for reporting
    let finalgrad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    let finalgrad_norm_rho = finalgrad.dot(&finalgrad).sqrt();
    let finalgrad_norm = if finalgrad_norm_rho.is_finite() {
        finalgrad_norm_rho
    } else {
        outer_result.final_grad_norm.unwrap_or(0.0)
    };

    if opts.compute_inference {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
        let p_cov = penalized_hessian.nrows();
        let qs = &pirls_res.reparam_result.qs;

        // Auto-select covariance strategy based on model size.
        //
        // For small-to-medium models (p ≤ COV_FULL_INVERSE_MAX_P) we can afford
        // the full p×p inverse: O(p³) compute, O(p²) memory. The full matrix is
        // needed for the frequentist covariance Ve = H⁻¹ X'WX H⁻¹ φ, the
        // influence matrix F = H⁻¹ X'WX, and the smoothing-parameter correction.
        //
        // For large models we use solve-on-demand against the Cholesky factor
        // already computed for EDF traces above. We solve H_t Z_t = Qs^T in
        // column chunks of size COV_SE_CHUNK, then extract the diagonal of
        // Qs · Z_t = H_orig⁻¹ to get exact posterior SEs without ever
        // materialising the p×p inverse. Prediction bands continue to work via
        // the factorised-Hessian path in PredictionCovarianceBackend::Factorized.
        const COV_FULL_INVERSE_MAX_P: usize = 10_000;
        const COV_SE_CHUNK: usize = 512;

        // Attempt the full inverse when the model is small enough.
        let beta_covariance_unscaled: Option<Array2<f64>> = if p_cov <= COV_FULL_INVERSE_MAX_P {
            match matrix_inversewith_regularization(&penalized_hessian, "posterior covariance") {
                Some(h_inv) => Some(h_inv),
                None => {
                    log::warn!(
                        "posterior covariance inversion failed (p={p_cov}): \
                         falling back to solve-on-demand standard errors"
                    );
                    None
                }
            }
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
        // Passes None for large models; compute_smoothing_correction_auto falls
        // back to first-order correction when no base covariance is supplied.
        // `cov_scale` is the coefficient-covariance multiplier at the optimum
        // (σ̂² for profiled Gaussian, 1 for every weight-carries-dispersion
        // family). The cubature path multiplies its dispersion-free curvature
        // block `E_ρ[H(ρ)⁻¹] − H_opt⁻¹` by this scale so the FULL cubature
        // correction lands on the same c² variance scale as `Vb = cov_scale·H_opt⁻¹`
        // (#582); the var_beta = Cov_ρ[β̂] block is already on that scale and
        // stays unscaled.
        let smoothing_outcome = reml_state.compute_smoothing_correction_auto(
            &final_rho,
            &pirls_res,
            beta_covariance_unscaled.as_ref(),
            cov_scale,
            finalgrad_norm,
        );
        rho_covariance = smoothing_outcome.rho_covariance().cloned();
        smoothing_correction = smoothing_outcome.into_correction();

        // Tier-0 marginal-smoothing certificate (#938): while the REML objective
        // is still live, sample the outer criterion around the converged ρ̂ to
        // read the PSIS k̂ that says whether the plug-in + first-order V_ρ
        // correction is adequate. This is the objective-lifecycle seam — the
        // certificate runs against the SAME objective the fit converged on, so
        // its criterion is the fit's own bit-for-bit (no retain/rebuild). Absent
        // when there are no smoothing parameters or the outer Hessian is
        // unavailable; never fatal. Superseded intermediate fits skip this block
        // and the caller must refit with a live objective before returning that
        // model. When the certificate reads Escalate, the auto-selected escalation
        // tier (quadrature for K≤4, NUTS over ρ for K≤16, honest Unavailable
        // beyond) runs at this same live seam.
        if !opts.skip_rho_posterior_inference {
            (rho_posterior_certificate, rho_posterior_escalation) =
                reml_state.rho_posterior_inference(&final_rho, None);
        }

        // Standard errors: prefer the diagonal of the full inverse when
        // available; otherwise use the factorised Hessian from the EDF pass
        // (in transformed basis) to compute exact diagonal of H_orig⁻¹ =
        // Qs H_t⁻¹ Qs' via chunked solve-on-demand. Memory per chunk:
        // 2 × p × COV_SE_CHUNK × 8 bytes.
        beta_standard_errors = if let Some(ref h_inv) = beta_covariance_unscaled {
            // Fast path: SE from stored full inverse (already phi-scaled via
            // beta_covariance, but we need the unscaled diagonal here).
            let raw_se = Array1::from_iter(
                h_inv
                    .diag()
                    .iter()
                    .map(|&v| (cov_scale * v.max(0.0)).sqrt()),
            );
            Some(raw_se)
        } else if let Some(ref factor_t) = edf_factor {
            // Solve-on-demand: process columns of Qs^T in chunks.
            // Qs is (p_cov × p_t) orthogonal. H_orig⁻¹ = Qs H_t⁻¹ Qs'.
            // (H_orig⁻¹)_{ii} = Qs[i,:] · H_t⁻¹ · Qs[i,:]'
            // Batch: column i of Qs^T is row i of Qs. Solve H_t Z = Qs^T[:,chunk]
            // then dot each solution column back with the corresponding Qs row.
            let mut diag_inv = Array1::<f64>::zeros(p_cov);
            let mut col_start = 0usize;
            while col_start < p_cov {
                let col_end = (col_start + COV_SE_CHUNK).min(p_cov);
                let chunk = col_end - col_start;
                // qs.t() has shape (p_t, p_cov); slice to (p_t, chunk).
                let rhs = qs.t().slice(ndarray::s![.., col_start..col_end]).to_owned();
                match factor_t.solvemulti(&rhs) {
                    Ok(z_chunk) => {
                        // z_chunk is (p_t × chunk).
                        // (H_orig⁻¹)_{ii} = qs.row(i) · z_chunk.column(i - col_start)
                        for local_i in 0..chunk {
                            let global_i = col_start + local_i;
                            let qs_row = qs.row(global_i);
                            let z_col = z_chunk.column(local_i);
                            diag_inv[global_i] = qs_row.dot(&z_col);
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "SE solve-on-demand failed at chunk {col_start}..{col_end}: {e}"
                        );
                        // Leave remaining entries as 0 (no SE).
                        break;
                    }
                }
                col_start = col_end;
            }
            let se = diag_inv.mapv(|v| (cov_scale * v.max(0.0)).sqrt());
            if se.iter().all(|v| v.is_finite()) {
                Some(se)
            } else {
                log::warn!("SE solve-on-demand produced non-finite entries; discarding");
                None
            }
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
                gam_linalg::matrix::symmetrize_in_place(&mut corrected);
                Some(corrected)
            }
            (Some(_), Some(corr)) => {
                log::warn!(
                    "Skipping corrected covariance: dimension mismatch (base {:?}, corr {:?})",
                    beta_covariance.as_ref().map(|c| c.as_array().dim()),
                    Some(corr.dim())
                );
                None
            }
            _ => None,
        };
        beta_standard_errors_corrected = beta_covariance_corrected.as_ref().map(se_from_covariance);
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        penalty_block_trace,
        edf_total,
        smoothing_correction,
        penalized_hessian: penalized_hessian.into(),
        working_weights: pirls_res.solveweights.clone(),
        working_response: pirls_res.solveworking_response.clone(),
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
    });

    let pirls_status = pirls_res.status;
    let likelihood_scale_field = pirls_res.likelihood.scale;
    let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
        y_o.view(),
        &pirls_res.finalmu,
        &pirls_res.likelihood,
        w_o.view(),
    );

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

    let result = ExternalOptimResult {
        beta: beta_orig_internal,
        lambdas: lambdas.to_owned(),
        likelihood_family: reported_family,
        likelihood_scale: likelihood_scale_field,
        log_likelihood_normalization: LogLikelihoodNormalization::OmittingResponseConstants,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        outer_converged: outer_result.converged,
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
    Ok(conditioning.backtransform_external_result(result))
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
