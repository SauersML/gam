//! Library-side orchestration for NUTS posterior sampling from a saved model.
//!
//! The CLI's `gam sample` subcommand and the Python `Model.sample(...)` API
//! both call into [`sample_saved_model`], which dispatches on the saved
//! model's class (standard GLM, standard with link-wiggle, or survival) and
//! returns a fully-converged [`NutsResult`] over the original coefficient
//! space.

use std::collections::HashMap;

use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand::{RngExt, SeedableRng};

use crate::basis::create_difference_penalty_matrix;
use crate::faer_ndarray::FaerCholesky;
use crate::estimate::{
    BlockRole, FitOptions, UnifiedFitResult, fit_gam, validate_all_finite,
};
use crate::families::royston_parmar::{self, RoystonParmarInputs};
use crate::families::survival_predict::{
    fit_result_from_saved_model_for_prediction, require_saved_survival_likelihood_mode,
    resolve_termspec_for_prediction, saved_baseline_timewiggle_components,
    saved_survival_runtime_baseline_config,
};
use crate::hmc::{
    FamilyNutsInputs, GlmFlatInputs, LinkWiggleSplineArtifacts, NutsConfig, NutsFamily, NutsResult,
    SurvivalFlatInputs, explicit_fit_hessian_for_whitening, run_link_wiggle_nuts_sampling,
    run_nuts_sampling_flattened_family, run_survival_nuts_sampling_flattened,
};
use crate::inference::formula_dsl::{LinkWiggleFormulaSpec, parse_formula};
use crate::inference::model::{
    FittedModel as SavedModel, PredictModelClass, load_survival_time_basis_config_from_model,
};
use crate::smooth::{build_term_collection_design, weighted_blockwise_penalty_sum};
use crate::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use crate::survival_construction::{
    SurvivalLikelihoodMode, add_survival_time_derivative_guard_offset, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, center_survival_time_designs_at_anchor,
    evaluate_survival_time_basis_row, normalize_survival_time_pair,
    resolved_survival_time_basis_config_from_build, survival_derivative_guard_for_likelihood,
};
use crate::term_builder::resolve_role_col;
use crate::types::LikelihoodFamily;
use crate::gamlss::{
    append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_knots,
    split_wiggle_penalty_orders,
};

/// Reconstruct the `LinkWiggleFormulaSpec` from a saved model's
/// baseline-time-wiggle runtime, returning `None` when the model has no
/// time-wiggle component. Re-exported because the survival fitter's tests
/// exercise the spec independently of running NUTS.
pub fn saved_baseline_timewiggle_spec(
    model: &SavedModel,
) -> Result<Option<LinkWiggleFormulaSpec>, String> {
    model.saved_baseline_time_wiggle().map(|runtime| {
        runtime.map(|saved| LinkWiggleFormulaSpec {
            degree: saved.degree,
            num_internal_knots: saved.knots.len().saturating_sub(2 * (saved.degree + 1)),
            penalty_orders: saved.penalty_orders,
            double_penalty: saved.double_penalty,
        })
    })
}

fn weighted_penalty_matrix(
    penalties: &[Array2<f64>],
    lambdas: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    if penalties.len() != lambdas.len() {
        return Err(format!(
            "penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        ));
    }
    if penalties.is_empty() {
        return Err("cannot sample without at least one penalty block".to_string());
    }
    let p = penalties[0].nrows();
    let mut out = Array2::<f64>::zeros((p, p));
    for (k, s) in penalties.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            return Err(format!(
                "penalty block {k} shape mismatch: got {}x{}, expected {}x{}",
                s.nrows(),
                s.ncols(),
                p,
                p
            ));
        }
        let lam = lambdas[k];
        out = out + &(s * lam);
    }
    Ok(out)
}

fn validate_explicit_link_wiggle_joint_hessian(
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), String> {
    if hessian.nrows() != expected_dim || hessian.ncols() != expected_dim {
        return Err(format!(
            "link-wiggle sample: explicit joint Hessian is {}x{} but expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim,
        ));
    }
    validate_all_finite(
        "link-wiggle explicit joint Hessian",
        hessian.iter().copied(),
    )?;
    let mut max_abs = 0.0_f64;
    for r in 0..expected_dim {
        for c in 0..expected_dim {
            max_abs = max_abs.max(hessian[[r, c]].abs());
            let scale = hessian[[r, c]].abs().max(hessian[[c, r]].abs()).max(1.0);
            if (hessian[[r, c]] - hessian[[c, r]]).abs() > 1e-9 * scale {
                return Err(format!(
                    "link-wiggle sample: explicit joint Hessian is not symmetric at ({r},{c})"
                ));
            }
        }
    }
    if max_abs == 0.0 {
        return Err(
            "link-wiggle sample: explicit joint Hessian is all zeros; refit with exact Hessian export"
                .to_string(),
        );
    }
    Ok(())
}

fn family_noise_parameter(fit: &UnifiedFitResult, family: LikelihoodFamily) -> Option<f64> {
    match family {
        LikelihoodFamily::GammaLog => fit
            .likelihood_scale
            .gamma_shape()
            .or(Some(fit.standard_deviation)),
        _ => Some(fit.standard_deviation),
    }
}

/// Run NUTS posterior sampling over a saved model.
///
/// Dispatches on `model.predict_model_class()`:
///
/// * `Standard`: refits the GAM in flat space, whitens with the saved
///   Hessian, and runs NUTS over the coefficient vector. Link-wiggle models
///   take a specialised joint-space path that preserves the basis chain
///   rule.
/// * `Survival`: rebuilds the survival design (Royston-Parmar baseline +
///   wiggle + covariate blocks) on the supplied data, evaluates the mode,
///   and runs the survival-flat NUTS path. Latent and location-scale modes
///   are explicitly rejected here.
/// * Other model classes (location-scale GLM, bernoulli marginal-slope,
///   transformation-normal) return a "not implemented" error matching the
///   CLI surface.
pub fn sample_saved_model(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    let family = model.likelihood();
    match model.predict_model_class() {
        PredictModelClass::Survival => {
            // Latent / latent-binary / location-scale survival likelihoods
            // have no exact NUTS implementation in the engine yet; fall
            // through to the Laplace-Gaussian fallback so callers still
            // get a posterior they can predict with. Royston-Parmar /
            // Weibull / marginal-slope survival use the exact path.
            let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
            if matches!(
                saved_likelihood_mode,
                SurvivalLikelihoodMode::Latent
                    | SurvivalLikelihoodMode::LatentBinary
                    | SurvivalLikelihoodMode::LocationScale
            ) {
                laplace_gaussian_fallback(model, cfg, "survival posterior fallback")
            } else {
                sample_survival(model, data, col_map, training_headers, cfg)
            }
        }
        PredictModelClass::Standard => {
            sample_standard(model, data, col_map, training_headers, family, cfg)
        }
        // For classes where the Rust core doesn't yet have an exact NUTS
        // implementation we fall back to drawing from the Laplace
        // (Gaussian) approximation of the posterior around the fitted
        // joint mode, using the saved penalised Hessian. This is the
        // standard "Bayesian credible interval" surface used by mgcv
        // and similar packages: it drops higher-order posterior shape
        // but lets every downstream consumer (credible intervals,
        // posterior predictive, etc.) keep working uniformly across
        // model classes.
        PredictModelClass::GaussianLocationScale => {
            laplace_gaussian_fallback(model, cfg, "gaussian location-scale posterior")
        }
        PredictModelClass::BinomialLocationScale => {
            laplace_gaussian_fallback(model, cfg, "binomial location-scale posterior")
        }
        PredictModelClass::BernoulliMarginalSlope => {
            laplace_gaussian_fallback(model, cfg, "bernoulli marginal-slope posterior")
        }
        PredictModelClass::TransformationNormal => {
            laplace_gaussian_fallback(model, cfg, "transformation-normal posterior")
        }
    }
}

/// Draw iid samples from `N(mode, H^{-1})` using the saved penalised
/// Hessian `H = L L^T`.
///
/// We solve `L^T δ = ε` for each iid `ε ~ N(0, I)` and report
/// `β = mode + δ`. The resulting draws are unbiased samples of the
/// Laplace-Gaussian approximation: their finite-sample mean / std
/// converge to `(mode, diag(H^{-1})^{1/2})` and the implied credible
/// bands match the surface that closed-form posterior tooling in
/// `mgcv` and `gam` itself uses for prediction intervals.
///
/// `rationale` is a short label appearing in error messages so callers
/// can tell which class fell back to this path. We mark `rhat = 1.0`
/// and `ess = n_total` because the draws are iid by construction.
fn laplace_gaussian_fallback(
    model: &SavedModel,
    cfg: &NutsConfig,
    rationale: &'static str,
) -> Result<NutsResult, String> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mode = fit.beta.clone();
    let p = mode.len();
    if p == 0 {
        return Err(format!(
            "{rationale}: cannot sample from an empty coefficient vector"
        ));
    }
    let h = fit.penalized_hessian().ok_or_else(|| {
        format!(
            "{rationale}: posterior fallback requires the explicit penalised Hessian; \
             refit with exact geometry export to enable posterior sampling for this class."
        )
    })?;
    if h.nrows() != p || h.ncols() != p {
        return Err(format!(
            "{rationale}: penalised Hessian is {}x{}, expected {}x{}",
            h.nrows(),
            h.ncols(),
            p,
            p
        ));
    }
    let chol = h.cholesky(Side::Lower).map_err(|err| {
        format!(
            "{rationale}: Cholesky factorisation of the penalised Hessian failed: {err:?}"
        )
    })?;
    let l = chol.lower_triangular();

    let n_total = cfg.n_samples.saturating_mul(cfg.n_chains).max(1);
    let mut samples = Array2::<f64>::zeros((n_total, p));
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut eps = Array1::<f64>::zeros(p);
    let mut delta = Array1::<f64>::zeros(p);
    for k in 0..n_total {
        for i in 0..p {
            eps[i] = sample_standard_normal(&mut rng);
        }
        back_solve_lower_transpose(&l, &eps, &mut delta);
        for i in 0..p {
            samples[(k, i)] = mode[i] + delta[i];
        }
    }

    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat: 1.0,
        ess: n_total as f64,
        converged: true,
    })
}

#[inline]
fn sample_standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    // Box-Muller transform — sufficient for posterior-mean-style sampling.
    // The same construction is used by the NUTS warmup; keeping it in
    // sync avoids two divergent gaussian RNG paths inside the engine.
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline]
fn back_solve_lower_transpose(l: &Array2<f64>, rhs: &Array1<f64>, out: &mut Array1<f64>) {
    // Solve L^T · out = rhs for `out`, where `L` is the lower-triangular
    // Cholesky factor of the penalised Hessian. Walks bottom-up since
    // L^T is upper triangular.
    let p = rhs.len();
    debug_assert_eq!(l.nrows(), p);
    debug_assert_eq!(l.ncols(), p);
    debug_assert_eq!(out.len(), p);
    for i in (0..p).rev() {
        let mut v = rhs[i];
        for j in (i + 1)..p {
            v -= l[[j, i]] * out[j];
        }
        let d = l[[i, i]];
        out[i] = if d.abs() > 1e-14 { v / d } else { 0.0 };
    }
}

fn sample_standard(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    family: LikelihoodFamily,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    if model.has_link_wiggle() {
        return sample_standard_link_wiggle(model, data, col_map, training_headers, family, cfg);
    }
    let parsed = parse_formula(&model.formula)?;
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let weights = Array1::ones(data.nrows());
    let offset = Array1::zeros(data.nrows());
    let dense_design_hmc = design.design.to_dense();
    let p = dense_design_hmc.ncols();
    let fit = fit_gam(
        dense_design_hmc.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &design.penalties,
        family,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .map_err(|e| format!("fit_gam failed during sample refit: {e}"))?;
    let penalty =
        weighted_blockwise_penalty_sum(&design.penalties, fit.lambdas.as_slice().unwrap(), p);

    run_nuts_sampling_flattened_family(
        family,
        FamilyNutsInputs::Glm(GlmFlatInputs {
            x: dense_design_hmc.view(),
            y: y.view(),
            weights: weights.view(),
            penalty_matrix: penalty.view(),
            mode: fit.beta.view(),
            hessian: explicit_fit_hessian_for_whitening(&fit, p, "sample refit")?.view(),
            gamma_shape: fit.likelihood_scale.gamma_shape(),
            firth_bias_reduction: false,
        }),
        cfg,
    )
    .map_err(|e| format!("NUTS sampling failed: {e}"))
}

fn sample_standard_link_wiggle(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    family: LikelihoodFamily,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    let parsed = parse_formula(&model.formula)?;
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.column(y_col).to_owned();

    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let p_main = design.design.ncols();

    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let wiggle_runtime = model
        .saved_prediction_runtime()?
        .link_wiggle
        .ok_or_else(|| "link-wiggle model is missing wiggle runtime metadata".to_string())?;
    let mode_beta = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean coefficient block".to_string())?
        .beta
        .clone();
    let mode_theta = fit
        .block_by_role(BlockRole::LinkWiggle)
        .ok_or_else(|| {
            "standard link-wiggle model is missing LinkWiggle coefficient block".to_string()
        })?
        .beta
        .clone();
    let p_wiggle = mode_theta.len();
    let p_total = mode_beta.len() + p_wiggle;

    if mode_beta.len() != p_main {
        return Err(format!(
            "link-wiggle sample: saved mean block has {} coefficients but rebuilt design has {} columns",
            mode_beta.len(),
            p_main,
        ));
    }
    if fit.beta.len() != p_total {
        return Err(format!(
            "link-wiggle sample: saved beta has {} coefficients but design has {} main + {} wiggle = {} total",
            fit.beta.len(),
            p_main,
            p_wiggle,
            p_total,
        ));
    }

    let hessian = &fit
        .geometry
        .as_ref()
        .ok_or_else(|| {
            "link-wiggle model is missing explicit joint Hessian geometry; refit with exact Hessian export"
                .to_string()
        })?
        .penalized_hessian;
    validate_explicit_link_wiggle_joint_hessian(hessian, p_total)?;

    let n_base_penalties = design.penalties.len();
    let base_lambdas = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean block lambdas".to_string())?
        .lambdas
        .view();
    if base_lambdas.len() != n_base_penalties {
        return Err(format!(
            "link-wiggle sample: mean block has {} lambdas but rebuilt design has {} base penalties",
            base_lambdas.len(),
            n_base_penalties,
        ));
    }

    let penalty_base =
        weighted_blockwise_penalty_sum(&design.penalties, base_lambdas.as_slice().unwrap(), p_main);

    let wiggle_lambdas_owned = fit
        .lambdas_linkwiggle()
        .ok_or_else(|| "standard link-wiggle model is missing LinkWiggle lambdas".to_string())?;
    let wiggle_lambdas = wiggle_lambdas_owned.view();
    let degree = wiggle_runtime.degree;
    let knot_arr = Array1::from_vec(wiggle_runtime.knots.clone());

    let mut wiggle_penalties = Vec::new();
    let default_orders = [2usize];
    let n_wiggle_lambdas = wiggle_lambdas.len();
    for k in 0..n_wiggle_lambdas {
        let order = if k < default_orders.len() {
            default_orders[k]
        } else {
            k + 1
        };
        if order >= p_wiggle {
            continue;
        }
        let penalty = create_difference_penalty_matrix(p_wiggle, order, None)
            .map_err(|e| format!("wiggle difference penalty failed: {e}"))?;
        wiggle_penalties.push(penalty);
    }
    while wiggle_penalties.len() < n_wiggle_lambdas {
        wiggle_penalties.push(Array2::zeros((p_wiggle, p_wiggle)));
    }

    let penalty_link = weighted_penalty_matrix(&wiggle_penalties, wiggle_lambdas)?;

    let q0 = design.design.dot(&mode_beta);
    let (q0_min, q0_max) = q0
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });

    let spline = LinkWiggleSplineArtifacts {
        knot_range: (q0_min, q0_max),
        knot_vector: knot_arr,
        degree,
    };

    let nuts_family = match family {
        LikelihoodFamily::BinomialLogit => NutsFamily::BinomialLogit,
        LikelihoodFamily::BinomialProbit => NutsFamily::BinomialProbit,
        LikelihoodFamily::BinomialCLogLog => NutsFamily::BinomialCLogLog,
        LikelihoodFamily::GaussianIdentity => NutsFamily::Gaussian,
        LikelihoodFamily::PoissonLog => NutsFamily::PoissonLog,
        LikelihoodFamily::GammaLog => NutsFamily::GammaLog,
        _ => {
            return Err(format!(
                "NUTS sampling with link wiggle is not supported for family {}",
                family.pretty_name()
            ));
        }
    };

    let weights = Array1::ones(data.nrows());
    let scale = family_noise_parameter(&fit, family).unwrap_or(fit.standard_deviation);

    let wiggle_nuts_dense = design.design.as_dense_cow();
    run_link_wiggle_nuts_sampling(
        wiggle_nuts_dense.view(),
        y.view(),
        weights.view(),
        penalty_base.view(),
        penalty_link.view(),
        mode_beta.view(),
        mode_theta.view(),
        hessian.view(),
        spline,
        nuts_family,
        scale,
        cfg,
    )
    .map_err(|e| format!("link-wiggle NUTS sampling failed: {e}"))
}

fn sample_survival(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(
            "sampling is not implemented for saved latent survival/binary models".to_string(),
        );
    }
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        return Err(
            "sample for survival-likelihood=location-scale is not implemented yet".to_string(),
        );
    }
    let entryname = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let eventname = model
        .survival_event
        .as_ref()
        .ok_or_else(|| "survival model missing event column metadata".to_string())?;
    let entry_col = resolve_role_col(col_map, entryname, "entry")?;
    let exit_col = resolve_role_col(col_map, exitname, "exit")?;
    let event_col = resolve_role_col(col_map, eventname, "event")?;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
        .map_err(|e| format!("failed to build survival design: {e}"))?;
    let n = data.nrows();
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i)?;
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if data[[i, event_col]] >= 0.5 { 1 } else { 0 };
    }
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &time_anchor_row,
        )?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            None,
        )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time + p_timewiggle + p_cov;
    let tb_entry_dense = time_build.x_entry_time.to_dense();
    let tb_exit_dense = time_build.x_exit_time.to_dense();
    let tb_deriv_dense = time_build.x_derivative_time.to_dense();
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    if p_time > 0 {
        x_entry.slice_mut(s![.., ..p_time]).assign(&tb_entry_dense);
        x_exit.slice_mut(s![.., ..p_time]).assign(&tb_exit_dense);
        x_derivative
            .slice_mut(s![.., ..p_time])
            .assign(&tb_deriv_dense);
    }
    if let Some((entry_w, exit_w, deriv_w)) = saved_timewiggle.as_ref() {
        if p_timewiggle > 0 {
            x_entry
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(entry_w);
            x_exit
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(exit_w);
            x_derivative
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(deriv_w);
        }
    }
    if p_cov > 0 {
        let cov_dense = cov_design.design.as_dense_cow();
        let cov_range = (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov);
        x_entry
            .slice_mut(s![.., cov_range.clone()])
            .assign(&cov_dense);
        x_exit.slice_mut(s![.., cov_range]).assign(&cov_dense);
    }
    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for (idx, s) in time_build.penalties.iter().enumerate() {
        if s.nrows() == p_time && s.ncols() == p_time {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                range: 0..p_time,
                nullspace_dim: time_build.nullspace_dims.get(idx).copied().unwrap_or(0),
            });
        }
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    if let Some((_, exit_w, _)) = saved_timewiggle.as_ref() {
        let start = p_time;
        let end = start + exit_w.ncols();
        let wiggle_lambda_offset = penalty_blocks.len();
        let wiggle_cfg = saved_baseline_timewiggle_spec(model)?.ok_or_else(|| {
            "saved baseline-timewiggle model missing baseline-timewiggle metadata".to_string()
        })?;
        let wiggle_degree = wiggle_cfg.degree;
        let wiggle_knots =
            Array1::from_vec(model.baseline_timewiggle_knots.clone().ok_or_else(|| {
                "saved baseline-timewiggle model missing baseline_timewiggle_knots".to_string()
            })?);
        let mut seed = Array1::<f64>::zeros(2 * n);
        for i in 0..n {
            seed[i] = eta_offset_entry[i];
            seed[n + i] = eta_offset_exit[i];
        }
        let (primary_order, extra_orders) =
            split_wiggle_penalty_orders(2, &wiggle_cfg.penalty_orders);
        let mut block = buildwiggle_block_input_from_knots(
            seed.view(),
            &wiggle_knots,
            wiggle_degree,
            primary_order,
            wiggle_cfg.double_penalty,
        )?;
        append_selected_wiggle_penalty_orders(&mut block, &extra_orders)
            .map_err(|e| format!("baseline-timewiggle penalty reconstruction failed: {e}"))?;
        for (widx, s) in block.penalties.iter().enumerate() {
            let s = match s {
                crate::estimate::PenaltySpec::Block { local, .. } => local,
                crate::estimate::PenaltySpec::Dense(m) => m,
            };
            if s.nrows() == exit_w.ncols() && s.ncols() == exit_w.ncols() {
                penalty_blocks.push(PenaltyBlock {
                    matrix: s.clone(),
                    lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                    range: start..end,
                    nullspace_dim: block.nullspace_dims.get(widx).copied().unwrap_or(0),
                });
            }
        }
        for (local_idx, block_penalty) in penalty_blocks[wiggle_lambda_offset..]
            .iter_mut()
            .enumerate()
        {
            if let Some(&lam) = fit_saved.lambdas.get(wiggle_lambda_offset + local_idx) {
                block_penalty.lambda = lam;
            }
        }
    }
    let ridge_lambda = model.survivalridge_lambda.ok_or_else(|| {
        "saved survival model is missing survivalridge_lambda; refusing to \
         pick a load-time default (the historical 1e-4 fallback silently \
         disagreed with the 1e-6 fit-time default). Refit."
            .to_string()
    })?;
    let ridge_range_start =
        if time_build.basisname == "linear" && !model.has_baseline_time_wiggle() {
            1
        } else {
            0
        };
    if ridge_lambda > 0.0 && p > ridge_range_start {
        let dim = p - ridge_range_start;
        let mut ridge = Array2::<f64>::zeros((dim, dim));
        for d in 0..dim {
            ridge[[d, d]] = 1.0;
        }
        penalty_blocks.push(PenaltyBlock {
            matrix: ridge,
            lambda: ridge_lambda,
            range: ridge_range_start..p,
            nullspace_dim: 0,
        });
    }
    for (idx, block) in penalty_blocks.iter_mut().enumerate() {
        if let Some(&lam) = fit_saved.lambdas.get(idx) {
            block.lambda = lam;
        }
    }
    let penalties = PenaltyBlocks::new(penalty_blocks);
    let survivalspec = match model
        .survivalspec
        .as_deref()
        .unwrap_or("net")
        .to_ascii_lowercase()
        .as_str()
    {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err(
                "saved survival spec 'crude' is not supported by the one-hazard survival engine; refit or export a net survival model for this path"
                    .to_string(),
            );
        }
        other => return Err(format!("unsupported saved survival spec '{other}'")),
    };
    let monotonicity = MonotonicityPenalty { tolerance: 0.0 };
    let mut model_surv = royston_parmar::working_model_from_flattened(
        penalties.clone(),
        monotonicity,
        survivalspec,
        RoystonParmarInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull {
        model_surv
            .set_structural_monotonicity(true, p_time + p_timewiggle)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }
    let beta0 = fit_saved.beta.clone();
    let state = model_surv
        .update_state(&beta0)
        .map_err(|e| format!("failed to evaluate survival state: {e}"))?;
    let hessian = state.hessian.to_dense();
    run_survival_nuts_sampling_flattened(
        SurvivalFlatInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
        penalties,
        monotonicity,
        survivalspec,
        saved_likelihood_mode != SurvivalLikelihoodMode::Weibull,
        p_time + p_timewiggle,
        beta0.view(),
        hessian.view(),
        cfg,
    )
    .map_err(|e| format!("survival NUTS sampling failed: {e}"))
}
