use super::*;
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
        let glm_seed_budget = if gaussian { 1 } else { 2 };
        return SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: glm_seed_budget,
            seed_budget: glm_seed_budget,
            risk_profile: if gaussian {
                SeedRiskProfile::Gaussian
            } else {
                SeedRiskProfile::GeneralizedLinear
            },
            screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
            num_auxiliary_trailing: 0,
        };
    }
    SeedConfig {
        bounds: (-12.0, 12.0),
        max_seeds: if gaussian && k <= 4 {
            2
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
        seed_budget: if k <= 6 { 1 } else { 2 },
        risk_profile: if gaussian {
            SeedRiskProfile::Gaussian
        } else {
            SeedRiskProfile::GeneralizedLinear
        },
        screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
        num_auxiliary_trailing: 0,
    }
}

fn reml_inner_progress_feedback(
    state: &crate::solver::estimate::reml::RemlState<'_>,
) -> crate::solver::rho_optimizer::InnerProgressFeedback {
    crate::solver::rho_optimizer::InnerProgressFeedback {
        cap: Arc::clone(&state.outer_inner_cap),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::clone(&state.last_inner_iters),
        last_converged: Arc::clone(&state.last_inner_converged),
        ift_residual: Arc::clone(&state.last_ift_prediction_residual),
        accept_rho: Arc::clone(&state.last_pirls_accept_rho),
    }
}

fn with_reml_beta_seed_hook<'state, 'data>() -> impl FnMut(
    &mut &'state mut crate::solver::estimate::reml::RemlState<'data>,
    &Array1<f64>,
) -> Result<
    crate::solver::rho_optimizer::SeedOutcome,
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
        Ok(crate::solver::rho_optimizer::SeedOutcome::Installed)
    }
}

enum RemlInnerCapGuardArm {
    Standard,
    MixtureSas,
}

fn run_outer_inner_cap_guard(
    state: &mut crate::solver::estimate::reml::RemlState<'_>,
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
    validate_penalty_specs(&s_list, p, "optimize_external_design")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
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
    let (
        final_rho,
        final_mixture_state,
        final_sas_state,
        final_mixture_param_covariance,
        final_sas_param_covariance,
        outer_result,
    ) = if mixture_dim > 0 && sas_dim > 0 {
        crate::bail_invalid_estim!("simultaneous mixture and SAS optimization is not supported");
    } else if mixture_dim == 0 && sas_dim == 0 {
        use crate::solver::rho_optimizer::{
            DeclaredHessianForm, Derivative, OuterEvalOrder, OuterProblem,
        };

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
                crate::solver::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
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
            .with_rho_bound(crate::estimate::RHO_BOUND);
        let problem = if let Some(h) = heuristic_lambdas {
            problem.with_heuristic_lambdas(h.to_vec())
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
        let gaussian_risk = matches!(reml_seed_config.risk_profile, SeedRiskProfile::Gaussian);
        // The Gaussian path historically skipped the objective-grid prepass and
        // seeded the outer search from the weight-independent origin 0. That is
        // exactly correct for an UNWEIGHTED fit (anchor 0), but breaks the
        // weight-scale invariance of λ̂ the moment a global rescale shifts the
        // optimum off 0 (issue #877). Run the anchored prepass for Gaussian ONLY
        // when the weight scale is non-trivial, so unweighted Gaussian fits stay
        // byte-identical while up-/down-weighted fits seed at the shifted optimum.
        let run_gaussian_anchored_prepass = gaussian_risk && weight_log_geom_mean.abs() > 1e-12;
        let prepass_seed: Option<Array1<f64>> = if gaussian_risk && !run_gaussian_anchored_prepass {
            None
        } else {
            let bnds = reml_seed_config.bounds;
            let (lo, hi) = if bnds.0 <= bnds.1 {
                bnds
            } else {
                (bnds.1, bnds.0)
            };
            // risk_shift is the default seed bias when no caller warm-start is given;
            // it is NOT applied on top of a caller-supplied heuristic_lambdas.
            let risk_shift: f64 = match reml_seed_config.risk_profile {
                SeedRiskProfile::Gaussian => 0.0,
                SeedRiskProfile::GeneralizedLinear => 1.0,
                SeedRiskProfile::Survival => 2.0,
            };
            // Anchor the default seed origin to the weight scale (issue #877). A
            // caller-supplied `heuristic_lambdas` already carries the absolute λ
            // scale, so it is used as-is; only the default risk-shift origin is
            // weight-anchored.
            let base = if let Some(h) = heuristic_lambdas.as_ref().filter(|h| h.len() == k) {
                Array1::from_iter(h.iter().map(|&v| v.max(1e-12).ln().clamp(lo, hi)))
            } else {
                Array1::from_elem(k, (risk_shift + weight_log_geom_mean).clamp(lo, hi))
            };
            let refined = crate::seeding::select_objective_seed_on_log_lambda_grid(
                &base,
                (lo, hi),
                k,
                |rho| reml_state.compute_cost(rho).ok().filter(|c| c.is_finite()),
            );
            // Emit the seed when the grid moved it, or — on the Gaussian
            // weight-anchored path — whenever the anchored `base` is itself
            // offset from the unanchored origin (so the shifted optimum is
            // actually seeded even if the coarse grid leaves `base` unchanged).
            let grid_moved = refined
                .iter()
                .zip(base.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-12);
            if grid_moved || run_gaussian_anchored_prepass {
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
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                state.compute_cost(rho)
            },
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
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
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             rho: &Array1<f64>,
             order: OuterEvalOrder| {
                outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                state.compute_outer_eval_with_order(rho, order)
            },
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>| {
                    state.reset_outer_seed_state()
                },
            ),
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>| { state.compute_efs_steps(rho) },
            ),
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                state.compute_screening_proxy(rho)
            },
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

        let strategy_result = problem.run(&mut obj, "standard REML")?;
        drop(obj);
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
        (
            strategy_result.rho.clone(),
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
                heuristic_theta.extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
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
        use crate::solver::rho_optimizer::{
            DeclaredHessianForm, Derivative, HessianResult, OuterEval, OuterProblem,
        };
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
                crate::solver::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
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
        let problem = match reml_state.outer_cache_session() {
            Some(session) => problem.with_cache_session(session),
            None => problem,
        };
        // Shared helper: parse theta into rho + link params, update link state.
        let apply_link_theta = |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
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
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
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
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                let cost = state.compute_cost(&rho)? + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode =
                    crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian;
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
            Some(|state: &mut &mut crate::solver::estimate::reml::RemlState<'_>| {
                state.reset_outer_seed_state();
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
            }),
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
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
        // converged ρ so the cached warm-start β is at full inner
        // tolerance regardless of where the BFGS schedule was when the
        // optimizer terminated.
        run_outer_inner_cap_guard(
            &mut reml_state,
            &outer_result.rho,
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
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, outer_result.iterations);
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
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
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
            traces[kk] = lambdas[kk] * frob;
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
            beta_covariance = Some(crate::inference::dispersion_cov::PhiScaledCovariance::wrap(
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
            crate::matrix::symmetrize_in_place(&mut s_mat);
            // Influence matrix F = I − H⁻¹·S(λ) = H⁻¹·X'WX. This is a product
            // of two symmetric matrices and is therefore generally NOT
            // symmetric; it must not be symmetrized — `crate::matrix::symmetrize_in_place(F)`
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
            crate::matrix::symmetrize_in_place(&mut ve);
            // X'WX = H − S(λ) in the original basis — the genuine PSD weighted
            // Gram, reconstructed from the same `penalized_hessian` and `s_mat`
            // that define `F = H⁻¹X'WX` (issue #1027). Stored directly so the
            // WPS corrected-EDF correction never has to recover it from an
            // inconsistent `H·F` product.
            let mut xwx = &penalized_hessian - &s_mat;
            crate::matrix::symmetrize_in_place(&mut xwx);
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
                crate::matrix::symmetrize_in_place(&mut corrected);
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

    // Report the fitted Negative-Binomial overdispersion `theta` on the family
    // variant (issue #802). Unlike the Gamma shape / Tweedie φ (which live only
    // in `likelihood_scale`) and the Beta φ (whose estimate downstream consumers
    // read from `likelihood_scale` via a separate override), NB `theta` is the
    // *canonical* parameter on `ResponseFamily::NegativeBinomial { theta }` that
    // every NB predictive consumer (prediction-interval variance, quadrature,
    // sampling, `generate` draws) reads directly off the saved family. The fit
    // updated it in lock-step with the `EstimatedNegBinTheta` scale metadata via
    // `with_negbin_theta`, so threading that fitted `theta` back onto the reported
    // family is what makes those consumers see the data's overdispersion instead
    // of the seed. Non-NB families keep `opts.family` (their estimates live in the
    // scale metadata), preserving the existing seed-in-family convention.
    let mut reported_family = opts.family.clone();
    if let (
        ResponseFamily::NegativeBinomial { theta, .. },
        LikelihoodScaleMetadata::EstimatedNegBinTheta {
            theta: fitted_theta,
        },
    ) = (&mut reported_family.response, likelihood_scale_field)
    {
        *theta = fitted_theta;
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
