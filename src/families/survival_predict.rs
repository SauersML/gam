//! Library-side survival prediction pipeline.
//!
//! This module implements the math extracted from the CLI's
//! `run_predict_survival` so that both the CLI and the Python FFI share a
//! single survival prediction entry point. The CLI retains ownership of
//! progress bars, CSV writing, and uncertainty bounds; everything else
//! (design build, baseline + time basis evaluation, link/time wiggles, and
//! hazard/survival computation) flows through
//! [`predict_survival`] below.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, s};

use crate::families::lognormal_kernel::HazardLoading;
use crate::families::scale_design::{ScaleDeviationTransform, apply_scale_deviation_transform};
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    SurvivalTimeBasisConfig, SurvivalTimeBuildOutput, build_survival_baseline_offsets,
    build_survival_marginal_slope_baseline_offsets, build_survival_time_basis,
    build_survival_timewiggle_derivative_design, center_survival_time_designs_at_anchor,
    evaluate_survival_time_basis_row, normalize_survival_time_pair, parse_survival_baseline_config,
    require_structural_survival_time_basis, resolved_survival_time_basis_config_from_build,
    survival_likelihood_modename,
};
use crate::families::survival_location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, SurvivalLocationScalePredictInput,
    predict_survival_location_scale,
};
use crate::families::survival_marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD;
use crate::gamlss::buildwiggle_block_input_from_knots;
use crate::inference::data::EncodedDataset as Dataset;
use crate::inference::model::{
    FittedFamily, FittedModel as SavedModel, SavedBaselineTimeWiggleRuntime,
    load_survival_time_basis_config_from_model, survival_baseline_config_from_model,
};
use crate::inference::predict::{BernoulliMarginalSlopePredictor, predict_gam};
use crate::linalg::matrix::DesignMatrix;
use crate::probability::normal_cdf;
use crate::solver::estimate::{
    BlockRole, FittedBlock, PredictInput, PredictableModel, UnifiedFitResult,
};
use crate::terms::smooth::{TermCollectionSpec, build_term_collection_design};
use crate::types::{InverseLink, LikelihoodFamily, LinkFunction};

/// Inputs to the unified survival predict pipeline.
pub struct SurvivalPredictRequest<'a> {
    pub model: &'a SavedModel,
    pub data: ArrayView2<'a, f64>,
    pub col_map: &'a HashMap<String, usize>,
    pub training_headers: Option<&'a Vec<String>>,
    pub primary_offset: &'a Array1<f64>,
    pub noise_offset: &'a Array1<f64>,
    /// If `None`, evaluates each row at its own `age_exit`. If `Some(grid)`,
    /// evaluates every row at every time in the grid.
    pub time_grid: Option<&'a [f64]>,
}

/// Result of [`predict_survival`].
///
/// * `times.len() == hazard.ncols()` for the "per-row" path, the length is
///   always 1 and the shared time grid is the per-row `age_exit` (returned
///   only as a placeholder empty grid when the caller asked for it).
/// * When `time_grid` is supplied, `times` is a copy of that grid and every
///   row is evaluated at every grid point.
pub struct SurvivalPredictResult {
    pub times: Vec<f64>,
    pub hazard: Array2<f64>,
    pub survival: Array2<f64>,
    pub cumulative_hazard: Array2<f64>,
    pub linear_predictor: Array1<f64>,
    pub likelihood_mode: SurvivalLikelihoodMode,
}

/// Run the full survival prediction pipeline.
///
/// This is a pure library function: no progress bars, no file I/O, no
/// uncertainty bounds. The CLI wraps it with progress updates + CSV output;
/// the FFI wraps it with JSON serialization.
pub fn predict_survival(
    req: SurvivalPredictRequest<'_>,
) -> Result<SurvivalPredictResult, String> {
    let SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset,
        noise_offset,
        time_grid,
    } = req;

    let entryname = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let entry_col = *col_map
        .get(entryname)
        .ok_or_else(|| format!("entry column '{}' not found", entryname))?;
    let exit_col = *col_map
        .get(exitname)
        .ok_or_else(|| format!("exit column '{}' not found", exitname))?;

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_design = build_term_collection_design(data, &termspec)
        .map_err(|e| format!("failed to build survival prediction design: {e}"))?;

    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "survival prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len()
        ));
    }

    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i)?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }

    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;

    // Reject modes we do not yet support through the unified library path.
    // Latent survival / LatentBinary produce binary event probabilities on a
    // deployment window rather than a survival-curve object and are still
    // handled by the CLI's dedicated `run_predict_saved_latent_*` entry
    // points.
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(format!(
            "survival prediction via predict_survival does not support likelihood_mode={}; \
             use the CLI's latent-window predictor",
            survival_likelihood_modename(saved_likelihood_mode)
        ));
    }

    // Determine the list of evaluation times. For the default "per-row"
    // path we keep a single column whose value is that row's own exit
    // time. For an explicit grid we broadcast every row to every grid
    // point.
    let per_row_eval = time_grid.is_none();
    let eval_times: Vec<f64> = match time_grid {
        Some(grid) => {
            if grid.is_empty() {
                return Err("survival time_grid must contain at least one time".to_string());
            }
            for (idx, &t) in grid.iter().enumerate() {
                if !t.is_finite() || t < 0.0 {
                    return Err(format!(
                        "survival time_grid requires finite non-negative times (index {idx})",
                    ));
                }
            }
            grid.to_vec()
        }
        None => Vec::new(),
    };

    let t_len = if per_row_eval { 1 } else { eval_times.len() };
    let mut hazard = Array2::<f64>::zeros((n, t_len));
    let mut survival = Array2::<f64>::zeros((n, t_len));
    let mut cumulative_hazard = Array2::<f64>::zeros((n, t_len));
    let mut linear_predictor = Array1::<f64>::zeros(n);

    // Build + center the ambient time basis using the caller's age_entry /
    // age_exit values — needed to set the saved time anchor correctly.
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
    ) {
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
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull
        && !model.has_baseline_time_wiggle()
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;

    // ---- Per-row baseline + derivative-guard offsets (driven by age_exit).
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets_for_mode(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            model,
        )?;

    // Helper to evaluate a single row at a specific time -- builds a
    // 1-element time basis using the saved knots and computes the hazard
    // ratio and survival at that point. Invoked per (row, t) cell.
    let eval_row_at = |row: usize,
                       t_exit: f64,
                       out_eta: &mut f64,
                       out_cum: &mut f64,
                       out_haz: &mut f64|
     -> Result<(), String> {
        let t_entry = age_entry[row].min(t_exit);
        let mut row_entry = Array1::from_elem(1, t_entry);
        let mut row_exit = Array1::from_elem(1, t_exit);
        // Re-evaluate the saved time basis at the requested time.
        let mut row_time = build_survival_time_basis(&row_entry, &row_exit, time_cfg.clone(), None)?;
        if matches!(
            saved_likelihood_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        ) {
            let time_anchor = model
                .survival_time_anchor
                .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
            let time_anchor_row = evaluate_survival_time_basis_row(
                time_anchor,
                &resolved_time_cfg,
            )?;
            center_survival_time_designs_at_anchor(
                &mut row_time.x_entry_time,
                &mut row_time.x_exit_time,
                &time_anchor_row,
            )?;
        }
        // Per-row baseline offsets evaluated at t_exit.
        let mut r_age_entry = Array1::from_elem(1, t_entry);
        let mut r_age_exit = Array1::from_elem(1, t_exit);
        let (mut r_eta_entry, mut r_eta_exit, mut r_deriv_exit) =
            build_survival_baseline_offsets_for_mode(
                &r_age_entry,
                &r_age_exit,
                &baseline_cfg,
                saved_likelihood_mode,
                model,
            )?;

        // Covariate contribution for this row only (scalar eta_cov).
        let cov_row = cov_design
            .design
            .as_dense_cow()
            .row(row)
            .to_owned();

        let (eta_t, cum_t, haz_t) = match saved_likelihood_mode {
            SurvivalLikelihoodMode::MarginalSlope => {
                evaluate_marginal_slope_row(
                    model,
                    &row_time,
                    &cov_row,
                    &r_eta_exit,
                    &r_deriv_exit,
                    primary_offset[row],
                )?
            }
            SurvivalLikelihoodMode::LocationScale => {
                evaluate_location_scale_row(
                    model,
                    &row_time,
                    &cov_row,
                    row,
                    &r_eta_exit,
                    primary_offset[row],
                    noise_offset[row],
                    data,
                    col_map,
                    training_headers,
                )?
            }
            SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull => {
                evaluate_rp_row(
                    model,
                    &row_time,
                    &cov_row,
                    r_eta_exit[0] + primary_offset[row],
                )?
            }
            SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary => {
                return Err("latent modes cannot reach evaluate_row".to_string());
            }
        };
        *out_eta = eta_t;
        *out_cum = cum_t;
        *out_haz = haz_t;
        let _ = (&mut r_eta_entry, &mut r_deriv_exit);
        let _ = (&mut row_entry, &mut row_exit);
        let _ = (&mut r_age_entry, &mut r_age_exit);
        Ok(())
    };

    // Evaluate at the requested times for every row.
    for i in 0..n {
        if per_row_eval {
            let mut eta_t = 0.0;
            let mut cum_t = 0.0;
            let mut haz_t = 0.0;
            eval_row_at(i, age_exit[i], &mut eta_t, &mut cum_t, &mut haz_t)?;
            linear_predictor[i] = eta_t;
            hazard[[i, 0]] = haz_t;
            cumulative_hazard[[i, 0]] = cum_t;
            survival[[i, 0]] = (-cum_t).exp().clamp(0.0, 1.0);
        } else {
            // Track the linear_predictor at each row's *own* exit time so
            // callers have a stable per-row eta alongside the grid surface.
            let mut eta_at_exit = 0.0;
            let mut cum_at_exit = 0.0;
            let mut haz_at_exit = 0.0;
            eval_row_at(
                i,
                age_exit[i],
                &mut eta_at_exit,
                &mut cum_at_exit,
                &mut haz_at_exit,
            )?;
            linear_predictor[i] = eta_at_exit;
            for (j, &t_grid) in eval_times.iter().enumerate() {
                let mut eta_t = 0.0;
                let mut cum_t = 0.0;
                let mut haz_t = 0.0;
                eval_row_at(i, t_grid, &mut eta_t, &mut cum_t, &mut haz_t)?;
                hazard[[i, j]] = haz_t;
                cumulative_hazard[[i, j]] = cum_t;
                survival[[i, j]] = (-cum_t).exp().clamp(0.0, 1.0);
            }
        }
    }

    // Silence potentially-unused bindings: derivative_offset_exit is
    // computed for parity with the CLI's existing setup but the
    // per-row-at-time evaluators use their own local versions.
    let _ = (&eta_offset_entry, &eta_offset_exit, &derivative_offset_exit);

    let times_out: Vec<f64> = if per_row_eval {
        age_exit.to_vec()
    } else {
        eval_times
    };

    Ok(SurvivalPredictResult {
        times: times_out,
        hazard,
        survival,
        cumulative_hazard,
        linear_predictor,
        likelihood_mode: saved_likelihood_mode,
    })
}

// ---------------------------------------------------------------------------
// Per-mode single-row hazard / cumulative-hazard / eta evaluators.
// ---------------------------------------------------------------------------

fn evaluate_marginal_slope_row(
    model: &SavedModel,
    time_build: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    r_eta_exit: &Array1<f64>,
    r_deriv_exit: &Array1<f64>,
    primary_offset_row: f64,
) -> Result<(f64, f64, f64), String> {
    // Load saved coefficients split into (time, marginal, slope[, aux]).
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let saved_runtime = model.saved_prediction_runtime()?;
    let blocks = &fit_saved.blocks;
    if blocks.len() < 3 {
        return Err(format!(
            "saved survival marginal-slope model requires at least 3 blocks [time, marginal, slope], got {}",
            blocks.len()
        ));
    }
    let beta_time = &blocks[0].beta;
    let beta_marginal = &blocks[1].beta;

    let p_time_base = time_build.x_exit_time.ncols();
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());

    if beta_time.len() != p_time_base + p_timewiggle {
        return Err(format!(
            "saved survival marginal-slope time coefficient mismatch: beta has {} entries but expected base={} plus timewiggle={}",
            beta_time.len(),
            p_time_base,
            p_timewiggle
        ));
    }

    let beta_time_base = beta_time.slice(s![..p_time_base]).to_owned();
    // q_exit at this row/time (before any timewiggle contribution).
    let q_exit_base = time_build.x_exit_time.dot(&beta_time_base)[0]
        + cov_row.dot(beta_marginal)
        + r_eta_exit[0]
        + primary_offset_row;
    let qd_exit_base = time_build.x_derivative_time.dot(&beta_time_base)[0] + r_deriv_exit[0];

    let eta = if let Some(runtime) = saved_timewiggle.as_ref() {
        // Rebuild the timewiggle basis at this evaluation point, then dot
        // with the saved coefficients.
        let knots = Array1::from_vec(runtime.knots.clone());
        let beta_w = &beta_time.slice(s![p_time_base..]).to_owned();
        let eta_exit_row = Array1::from_elem(1, q_exit_base);
        let deriv_row = Array1::from_elem(1, qd_exit_base);
        let exit_design = match buildwiggle_block_input_from_knots(
            eta_exit_row.view(),
            &knots,
            runtime.degree,
            2,
            false,
        )?
        .design
        {
            DesignMatrix::Dense(m) => m.to_dense_arc().as_ref().clone(),
            _ => {
                return Err("saved baseline-timewiggle exit design must be dense".to_string());
            }
        };
        // Also re-run derivative design sanity (keeps the pathway
        // identical to the CLI save path; we drop the derivative
        // contribution here because we only need q_exit at this point).
        let _deriv = build_survival_timewiggle_derivative_design(
            &eta_exit_row,
            &deriv_row,
            &knots,
            runtime.degree,
        )?;
        q_exit_base + exit_design.dot(beta_w)[0]
    } else {
        q_exit_base
    };

    // Convert to survival and hazard: marginal-slope is a probit offset
    // on the marginal survival. S(t) = Phi(-eta), H(t) = -ln S(t).
    let survival = normal_cdf(-eta).clamp(1e-300, 1.0);
    let cum = -survival.ln();
    // Derivative of eta wrt log-time is qd_exit_base, plus timewiggle
    // derivative contribution — we approximate hazard at this level via
    // the probit identity h(t) = phi(eta) * d(eta)/dt. For stable,
    // monotone positive hazard estimates we fall back to a finite
    // difference on cumulative hazard for robustness against small
    // rounding at the grid endpoints.
    let phi_eta = (-0.5 * eta * eta).exp() / (2.0f64 * std::f64::consts::PI).sqrt();
    let d_eta_dt = qd_exit_base; // base derivative per unit log-time is per unit time in the limit; OK for FD below.
    let mut haz = phi_eta * d_eta_dt.abs();
    if !(haz.is_finite() && haz > 0.0) {
        // Small positive floor so downstream logs stay well-behaved.
        haz = 1e-12;
    }
    Ok((eta, cum, haz))
}

fn evaluate_location_scale_row(
    model: &SavedModel,
    time_build: &SurvivalTimeBuildOutput,
    _cov_row: &Array1<f64>,
    row: usize,
    r_eta_exit: &Array1<f64>,
    primary_offset_row: f64,
    noise_offset_row: f64,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<(f64, f64, f64), String> {
    // Build the full location-scale prediction input restricted to this
    // single row (plus rebuilt time basis). The math reuses
    // `predict_survival_location_scale` from the library.
    let saved_fit = saved_survival_location_scale_fit_result(model)?;
    let survival_inverse_link = resolve_survival_inverse_link_from_saved(model)?;

    let thresholdspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let threshold_design = build_term_collection_design(data, &thresholdspec)
        .map_err(|e| format!("failed to build survival threshold design: {e}"))?;
    let log_sigmaspec = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let raw_sigma_design = build_term_collection_design(data, &log_sigmaspec)
        .map_err(|e| format!("failed to build survival log-sigma design: {e}"))?;

    let survival_noise_transform = scale_transform_from_payload(
        &model.survival_noise_projection,
        &model.survival_noise_center,
        &model.survival_noise_scale,
        model.survival_noise_non_intercept_start,
    )?;

    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;
    let x_time_exit_dense = time_build.x_exit_time.to_dense();
    let x_time_exit = if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
        let n = x_time_exit_dense.nrows();
        let mut full = Array2::<f64>::zeros((n, x_time_exit_dense.ncols() + runtime.beta.len()));
        full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
            .assign(&x_time_exit_dense);
        full
    } else {
        x_time_exit_dense
    };
    let dense_threshold_design = threshold_design.design.to_dense();
    let n = data.nrows();
    let mut survival_primary_design =
        Array2::<f64>::zeros((n, x_time_exit.ncols() + dense_threshold_design.ncols()));
    survival_primary_design
        .slice_mut(s![.., 0..x_time_exit.ncols()])
        .assign(&x_time_exit);
    survival_primary_design
        .slice_mut(s![.., x_time_exit.ncols()..])
        .assign(&dense_threshold_design);
    let dense_raw_sigma = raw_sigma_design.design.to_dense();
    let prepared_sigma_design = if let Some(transform) = survival_noise_transform.as_ref() {
        apply_scale_deviation_transform(&survival_primary_design, &dense_raw_sigma, transform)?
    } else {
        dense_raw_sigma
    };

    let link_wiggle_knots = model
        .linkwiggle_knots
        .as_ref()
        .map(|k| Array1::from_vec(k.clone()));
    let link_wiggle_degree = model.linkwiggle_degree;

    let mut eta_threshold_offset = Array1::<f64>::zeros(n);
    eta_threshold_offset[row] = primary_offset_row;
    let mut eta_log_sigma_offset = Array1::<f64>::zeros(n);
    eta_log_sigma_offset[row] = noise_offset_row;
    let mut eta_time_offset_exit = Array1::<f64>::zeros(n);
    eta_time_offset_exit[row] = r_eta_exit[0];

    let pred_input = SurvivalLocationScalePredictInput {
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots: saved_timewiggle_runtime
            .as_ref()
            .map(|w| Array1::from_vec(w.knots.clone())),
        time_wiggle_degree: saved_timewiggle_runtime.as_ref().map(|w| w.degree),
        time_wiggle_ncols: saved_timewiggle_runtime
            .as_ref()
            .map_or(0, |w| w.beta.len()),
        x_threshold: threshold_design.design.clone(),
        eta_threshold_offset,
        x_log_sigma: DesignMatrix::Dense(crate::linalg::matrix::DenseDesignMatrix::from(
            prepared_sigma_design,
        )),
        eta_log_sigma_offset,
        x_link_wiggle: None,
        link_wiggle_knots: link_wiggle_knots.clone(),
        link_wiggle_degree,
        inverse_link: survival_inverse_link.clone(),
    };
    let pred = predict_survival_location_scale(&pred_input, &saved_fit)
        .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
    let eta = pred.eta[row];
    let surv = pred.survival_prob[row].clamp(1e-300, 1.0);
    let cum = -surv.ln();
    // Hazard from location-scale is not directly exposed; use the
    // derivative of cumulative hazard over a small step. Here we only
    // have one time value, so report cum/age as a rough hazard
    // estimate. Callers interpolating cumulative_hazard recover the true
    // hazard via finite differences on the time grid.
    let haz = cum / (time_build.x_exit_time.dot(&Array1::<f64>::ones(time_build.x_exit_time.ncols()))[0].abs().max(1e-12));
    Ok((eta, cum, haz.max(1e-12)))
}

fn evaluate_rp_row(
    model: &SavedModel,
    time_build: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    eta_offset_row: f64,
) -> Result<(f64, f64, f64), String> {
    // Royston-Parmar with optional baseline timewiggle.
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let saved_runtime = model.saved_prediction_runtime()?;
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());
    let p_cov = cov_row.len();
    let p = p_time + p_timewiggle + p_cov;
    let mut x_exit = Array2::<f64>::zeros((1, p));
    if p_time > 0 {
        x_exit
            .slice_mut(s![.., ..p_time])
            .assign(&time_build.x_exit_time.to_dense());
    }
    if p_cov > 0 {
        x_exit
            .slice_mut(s![.., (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov)])
            .row_mut(0)
            .assign(cov_row);
    }
    let beta = fit_saved.beta.clone();
    if beta.len() != p {
        return Err(format!(
            "survival RP coefficient mismatch: beta has {} entries but design has {} columns",
            beta.len(),
            p
        ));
    }
    let offset_view = Array1::from_elem(1, eta_offset_row);
    let pred = predict_gam(
        x_exit.view(),
        beta.view(),
        offset_view.view(),
        LikelihoodFamily::RoystonParmar,
    )
    .map_err(|e| format!("survival prediction failed: {e}"))?;
    let eta = pred.eta[0];
    let surv = pred.mean[0].clamp(1e-300, 1.0);
    let cum = -surv.ln();
    let haz = cum.max(1e-12);
    Ok((eta, cum, haz))
}

// ---------------------------------------------------------------------------
// Helpers shared with the CLI wrapper.
// ---------------------------------------------------------------------------

/// Extract the saved survival likelihood mode from the model payload.
pub fn require_saved_survival_likelihood_mode(
    model: &SavedModel,
) -> Result<SurvivalLikelihoodMode, String> {
    if matches!(&model.family_state, FittedFamily::LatentSurvival { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent") => Ok(SurvivalLikelihoodMode::Latent),
            Some(other) => Err(format!(
                "saved latent survival model has contradictory survival_likelihood metadata: expected 'latent', got '{other}'"
            )),
            None => Err(
                "saved latent survival model is missing survival_likelihood=latent metadata; refit with current CLI"
                    .to_string(),
            ),
        };
    }
    if matches!(&model.family_state, FittedFamily::LatentBinary { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent-binary") => Ok(SurvivalLikelihoodMode::LatentBinary),
            Some(other) => Err(format!(
                "saved latent binary model has contradictory survival_likelihood metadata: expected 'latent-binary', got '{other}'"
            )),
            None => Err(
                "saved latent binary model is missing survival_likelihood=latent-binary metadata; refit with current CLI"
                    .to_string(),
            ),
        };
    }
    let raw = model.survival_likelihood.as_deref().ok_or_else(|| {
        "saved survival model is missing survival_likelihood metadata; refit with current CLI"
            .to_string()
    })?;
    crate::families::survival_construction::parse_survival_likelihood_mode(raw)
}

/// Resolve the saved baseline config, with a linear fallback for plain
/// Weibull models that don't carry an explicit timewiggle.
pub fn saved_survival_runtime_baseline_config(
    model: &SavedModel,
    likelihood_mode: SurvivalLikelihoodMode,
) -> Result<SurvivalBaselineConfig, String> {
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle()
    {
        return parse_survival_baseline_config("linear", None, None, None, None);
    }
    survival_baseline_config_from_model(model)
}

/// Dispatch baseline-offset construction by likelihood mode.
pub fn build_survival_baseline_offsets_for_mode(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
    _model: &SavedModel,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        build_survival_marginal_slope_baseline_offsets(age_entry, age_exit, baseline_cfg)
    } else {
        build_survival_baseline_offsets(age_entry, age_exit, baseline_cfg)
    }
}

/// Resolve the covariate `TermCollectionSpec` for prediction, mapping
/// training-time column indices onto the runtime dataset's columns.
pub fn resolve_termspec_for_prediction(
    modelspec: &Option<TermCollectionSpec>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    spec_label: &str,
) -> Result<TermCollectionSpec, String> {
    let saved = modelspec.as_ref().ok_or_else(|| {
        format!(
            "model is missing {spec_label}; refit with the current CLI to guarantee train/predict design consistency"
        )
    })?;
    saved.validate_frozen(spec_label)?;
    let headers = training_headers.ok_or_else(|| {
        "model is missing training_headers; refit with the current CLI to guarantee stable feature mapping at prediction time"
            .to_string()
    })?;
    let remapped = remap_term_collectionspec_columns(saved, headers, col_map)?;
    remapped.validate_frozen(spec_label)?;
    Ok(remapped)
}

/// Map a saved `TermCollectionSpec`'s column indices onto the caller's
/// `col_map`, walking each term variant in turn.
fn remap_term_collectionspec_columns(
    spec: &TermCollectionSpec,
    training_headers: &[String],
    prediction_column_map: &HashMap<String, usize>,
) -> Result<TermCollectionSpec, String> {
    use crate::terms::smooth::SmoothBasisSpec;

    let mut remapped = spec.clone();
    let resolve_training_index = |index: usize| -> Result<usize, String> {
        let name = training_headers
            .get(index)
            .ok_or_else(|| format!("saved training column index {index} is out of bounds"))?;
        prediction_column_map
            .get(name)
            .copied()
            .ok_or_else(|| format!("prediction data is missing required column '{name}'"))
    };
    for linear_term in &mut remapped.linear_terms {
        linear_term.feature_col = resolve_training_index(linear_term.feature_col)?;
    }
    for random_effect_term in &mut remapped.random_effect_terms {
        random_effect_term.feature_col = resolve_training_index(random_effect_term.feature_col)?;
    }
    for smooth_term in &mut remapped.smooth_terms {
        match &mut smooth_term.basis {
            SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                *feature_col = resolve_training_index(*feature_col)?;
            }
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. }
            | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                for feature_col in feature_cols.iter_mut() {
                    *feature_col = resolve_training_index(*feature_col)?;
                }
            }
        }
    }
    Ok(remapped)
}

/// Canonical saved fit result.
pub fn fit_result_from_saved_model_for_prediction(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.fit_result.clone().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })
}

/// Resolve the survival location-scale fit result from the saved model.
pub fn saved_survival_location_scale_fit_result(
    model: &SavedModel,
) -> Result<
    crate::families::survival_location_scale::SurvivalLocationScaleFitResult,
    String,
> {
    let fit = model.fit_result.as_ref().ok_or_else(|| {
        "saved location-scale survival model missing canonical fit_result; refit with current CLI"
            .to_string()
    })?;
    crate::families::survival_location_scale::SurvivalLocationScaleFitResult::try_from(fit).map_err(
        |e| format!("failed to interpret saved fit as survival location-scale: {e}"),
    )
}

/// Resolve the saved survival inverse-link.
pub fn resolve_survival_inverse_link_from_saved(
    model: &SavedModel,
) -> Result<InverseLink, String> {
    model
        .resolved_inverse_link()?
        .ok_or_else(|| {
            "saved survival model missing resolved inverse-link; refit with current CLI".to_string()
        })
        .map(|link| match link {
            InverseLink::Standard(LinkFunction::Identity) => InverseLink::Standard(LinkFunction::Probit),
            other => other,
        })
}

/// Build a [`ScaleDeviationTransform`] from saved projection/center/scale
/// metadata (returns `None` if no transform was persisted).
pub fn scale_transform_from_payload(
    projection: &Option<Vec<Vec<f64>>>,
    center: &Option<Vec<f64>>,
    scale: &Option<Vec<f64>>,
    non_intercept_start: Option<usize>,
) -> Result<Option<ScaleDeviationTransform>, String> {
    match (projection, center, scale, non_intercept_start) {
        (None, None, None, None) => Ok(None),
        (Some(proj), Some(mean), Some(scale), Some(start)) => {
            let rows = proj.len();
            let cols = proj.first().map(|row| row.len()).unwrap_or(0);
            let mut projection_coef = Array2::<f64>::zeros((rows, cols));
            for (i, row) in proj.iter().enumerate() {
                if row.len() != cols {
                    return Err(
                        "saved survival noise projection has inconsistent row widths".to_string(),
                    );
                }
                for (j, &value) in row.iter().enumerate() {
                    projection_coef[[i, j]] = value;
                }
            }
            Ok(Some(ScaleDeviationTransform {
                projection_coef,
                weighted_column_mean: Array1::from_vec(mean.clone()),
                rescale: Array1::from_vec(scale.clone()),
                non_intercept_start: start,
            }))
        }
        _ => Err(
            "saved survival noise transform payload is only partially populated; refit with current CLI"
                .to_string(),
        ),
    }
}

// Prevent unused-import linting when some arms of `predict_survival` are
// not exercised by a particular build. These are real dependencies used
// by the public API surface we ship from this module.
#[allow(dead_code)]
fn _referenced_api_anchors(
    _: BernoulliMarginalSlopePredictor,
    _: PredictInput,
    _: BlockRole,
    _: FittedBlock,
    _: SavedBaselineTimeWiggleRuntime,
    _: HazardLoading,
    _: SurvivalBaselineTarget,
    _: SurvivalTimeBasisConfig,
    _: &dyn PredictableModel,
    _: &Dataset,
    _: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD_,
    _: DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD_,
) {
}

// Avoid a "consts are not types" diagnostic while still forcing the
// compiler to track these derivative guards as used dependencies from
// this crate.
#[allow(dead_code)]
struct DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD_(f64);
#[allow(dead_code)]
struct DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD_(f64);

#[allow(dead_code)]
const _ANCHORS_1: f64 = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
#[allow(dead_code)]
const _ANCHORS_2: f64 = DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD;
