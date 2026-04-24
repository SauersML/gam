//! Library-side survival prediction pipeline.
//!
//! Extracts the hazard/survival/cumulative-hazard math from the CLI's
//! `run_predict_survival` so that both the CLI and the Python FFI can
//! share a single entry point. The CLI retains ownership of progress
//! bars, CSV writing, and uncertainty bounds; everything else (design
//! build, baseline + time basis evaluation, link/time wiggles, and
//! hazard/survival conversion) flows through [`predict_survival`].

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, s};

use crate::families::scale_design::{ScaleDeviationTransform, apply_scale_deviation_transform};
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalLikelihoodMode, SurvivalTimeBuildOutput,
    build_survival_baseline_offsets, build_survival_marginal_slope_baseline_offsets,
    build_survival_time_basis, build_survival_timewiggle_derivative_design,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    normalize_survival_time_pair, parse_survival_baseline_config,
    parse_survival_likelihood_mode, require_structural_survival_time_basis,
    resolved_survival_time_basis_config_from_build, survival_likelihood_modename,
};
use crate::families::survival_location_scale::{
    SurvivalLocationScalePredictInput, predict_survival_location_scale,
};
use crate::gamlss::buildwiggle_block_input_from_knots;
use crate::inference::model::{
    FittedFamily, FittedModel as SavedModel, load_survival_time_basis_config_from_model,
    survival_baseline_config_from_model,
};
use crate::inference::predict::predict_gam;
use crate::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use crate::probability::normal_cdf;
use crate::solver::estimate::UnifiedFitResult;
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
    /// If `None`, every row is evaluated at its own `age_exit`. If
    /// `Some(grid)`, every row is evaluated at every time in the grid.
    pub time_grid: Option<&'a [f64]>,
}

/// Result of [`predict_survival`].
pub struct SurvivalPredictResult {
    pub times: Vec<f64>,
    pub hazard: Array2<f64>,
    pub survival: Array2<f64>,
    pub cumulative_hazard: Array2<f64>,
    pub linear_predictor: Array1<f64>,
    pub likelihood_mode: SurvivalLikelihoodMode,
}

/// Run the survival prediction pipeline.
///
/// Pure library function: no progress bars, no file I/O, no uncertainty
/// bounds. The CLI wraps this with progress updates + CSV writes; the
/// FFI wraps it with JSON serialization.
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

    // Latent modes emit binary event-window probabilities, not survival
    // curves. They stay in the CLI's dedicated `run_predict_saved_latent_*`
    // helpers for now.
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

    // Ambient time basis: built once with (age_entry, age_exit) so that
    // the saved anchor / monotonicity checks fire at construction time.
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let mut time_anchor_row_cached: Option<Array1<f64>> = None;
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
        time_anchor_row_cached = Some(time_anchor_row);
    }
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull
        && !model.has_baseline_time_wiggle()
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;

    // Resolve the time-grid: either the explicit grid (same for every
    // row) or per-row exit times (one column per row).
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

    let t_cols = if per_row_eval { 1 } else { eval_times.len() };
    let mut hazard = Array2::<f64>::zeros((n, t_cols));
    let mut survival = Array2::<f64>::zeros((n, t_cols));
    let mut cumulative_hazard = Array2::<f64>::zeros((n, t_cols));
    let mut linear_predictor = Array1::<f64>::zeros(n);

    // Evaluate each (row, t) cell.
    for i in 0..n {
        let row_eta_exit_input = if per_row_eval {
            vec![age_exit[i]]
        } else {
            eval_times.clone()
        };
        for (j, &t_query) in row_eta_exit_input.iter().enumerate() {
            let t_entry = age_entry[i].min(t_query);
            let single_entry = Array1::from_elem(1, t_entry);
            let single_exit = Array1::from_elem(1, t_query);
            let mut row_time = build_survival_time_basis(
                &single_entry,
                &single_exit,
                time_cfg.clone(),
                None,
            )?;
            if let Some(anchor_row) = time_anchor_row_cached.as_ref() {
                center_survival_time_designs_at_anchor(
                    &mut row_time.x_entry_time,
                    &mut row_time.x_exit_time,
                    anchor_row,
                )?;
            }
            let (r_eta_entry, r_eta_exit, r_deriv_exit) =
                build_baseline_offsets_by_mode(
                    &single_entry,
                    &single_exit,
                    &baseline_cfg,
                    saved_likelihood_mode,
                )?;

            let cov_row = cov_design.design.as_dense_cow().row(i).to_owned();

            let (eta_t, cum_t, haz_t) = match saved_likelihood_mode {
                SurvivalLikelihoodMode::MarginalSlope => evaluate_marginal_slope_row(
                    model,
                    &row_time,
                    &cov_row,
                    &r_eta_exit,
                    &r_deriv_exit,
                    primary_offset[i],
                )?,
                SurvivalLikelihoodMode::LocationScale => evaluate_location_scale_row(
                    model,
                    &row_time,
                    &cov_row,
                    &r_eta_exit,
                    primary_offset[i],
                    noise_offset[i],
                    data,
                    col_map,
                    training_headers,
                    i,
                )?,
                SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull => evaluate_rp_row(
                    model,
                    &row_time,
                    &cov_row,
                    r_eta_exit[0] + primary_offset[i],
                )?,
                SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary => {
                    return Err("latent modes cannot reach evaluate_row".to_string());
                }
            };
            if per_row_eval {
                linear_predictor[i] = eta_t;
                hazard[[i, 0]] = haz_t;
                cumulative_hazard[[i, 0]] = cum_t;
                survival[[i, 0]] = (-cum_t).exp().clamp(0.0, 1.0);
            } else {
                hazard[[i, j]] = haz_t;
                cumulative_hazard[[i, j]] = cum_t;
                survival[[i, j]] = (-cum_t).exp().clamp(0.0, 1.0);
            }
            // Unused-binding parity (matches ambient construction path).
            let _ = &r_eta_entry;
        }
        if !per_row_eval {
            // Track the linear predictor at each row's own exit time.
            let t_exit = age_exit[i];
            let t_entry = age_entry[i].min(t_exit);
            let single_entry = Array1::from_elem(1, t_entry);
            let single_exit = Array1::from_elem(1, t_exit);
            let mut row_time = build_survival_time_basis(
                &single_entry,
                &single_exit,
                time_cfg.clone(),
                None,
            )?;
            if let Some(anchor_row) = time_anchor_row_cached.as_ref() {
                center_survival_time_designs_at_anchor(
                    &mut row_time.x_entry_time,
                    &mut row_time.x_exit_time,
                    anchor_row,
                )?;
            }
            let (_, r_eta_exit, r_deriv_exit) = build_baseline_offsets_by_mode(
                &single_entry,
                &single_exit,
                &baseline_cfg,
                saved_likelihood_mode,
            )?;
            let cov_row = cov_design.design.as_dense_cow().row(i).to_owned();
            let (eta_t, _, _) = match saved_likelihood_mode {
                SurvivalLikelihoodMode::MarginalSlope => evaluate_marginal_slope_row(
                    model,
                    &row_time,
                    &cov_row,
                    &r_eta_exit,
                    &r_deriv_exit,
                    primary_offset[i],
                )?,
                SurvivalLikelihoodMode::LocationScale => evaluate_location_scale_row(
                    model,
                    &row_time,
                    &cov_row,
                    &r_eta_exit,
                    primary_offset[i],
                    noise_offset[i],
                    data,
                    col_map,
                    training_headers,
                    i,
                )?,
                SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull => evaluate_rp_row(
                    model,
                    &row_time,
                    &cov_row,
                    r_eta_exit[0] + primary_offset[i],
                )?,
                SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary => {
                    return Err("latent modes cannot reach evaluate_row".to_string());
                }
            };
            linear_predictor[i] = eta_t;
        }
    }

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
// Per-mode single-row evaluators.
// ---------------------------------------------------------------------------

fn evaluate_marginal_slope_row(
    model: &SavedModel,
    row_time: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    r_eta_exit: &Array1<f64>,
    r_deriv_exit: &Array1<f64>,
    primary_offset_row: f64,
) -> Result<(f64, f64, f64), String> {
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
    let p_time_base = row_time.x_exit_time.ncols();
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
    let q_exit_base = row_time.x_exit_time.dot(&beta_time_base)[0]
        + cov_row.dot(beta_marginal)
        + r_eta_exit[0]
        + primary_offset_row;
    let qd_exit_base = row_time.x_derivative_time.dot(&beta_time_base)[0] + r_deriv_exit[0];

    let eta = if let Some(runtime) = saved_timewiggle.as_ref() {
        let knots = Array1::from_vec(runtime.knots.clone());
        let beta_w = beta_time.slice(s![p_time_base..]).to_owned();
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
        let _ = build_survival_timewiggle_derivative_design(
            &eta_exit_row,
            &deriv_row,
            &knots,
            runtime.degree,
        )?;
        q_exit_base + exit_design.dot(&beta_w)[0]
    } else {
        q_exit_base
    };

    let surv = normal_cdf(-eta).clamp(1e-300, 1.0);
    let cum = -surv.ln();
    let phi_eta = (-0.5 * eta * eta).exp() / (2.0f64 * std::f64::consts::PI).sqrt();
    let mut haz = phi_eta * qd_exit_base.abs();
    if !(haz.is_finite() && haz > 0.0) {
        haz = 1e-12;
    }
    Ok((eta, cum, haz))
}

fn evaluate_location_scale_row(
    model: &SavedModel,
    row_time: &SurvivalTimeBuildOutput,
    _cov_row: &Array1<f64>,
    r_eta_exit: &Array1<f64>,
    primary_offset_row: f64,
    noise_offset_row: f64,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    row: usize,
) -> Result<(f64, f64, f64), String> {
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
    let x_time_exit_dense = row_time.x_exit_time.to_dense();
    let x_time_exit = if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
        let mut full =
            Array2::<f64>::zeros((1, x_time_exit_dense.ncols() + runtime.beta.len()));
        full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
            .assign(&x_time_exit_dense);
        full
    } else {
        x_time_exit_dense
    };
    let n = data.nrows();
    let dense_threshold_design = threshold_design.design.to_dense();
    let single_threshold_row = dense_threshold_design.row(row).to_owned();
    let mut survival_primary_design =
        Array2::<f64>::zeros((1, x_time_exit.ncols() + dense_threshold_design.ncols()));
    survival_primary_design
        .slice_mut(s![.., 0..x_time_exit.ncols()])
        .assign(&x_time_exit);
    survival_primary_design
        .slice_mut(s![.., x_time_exit.ncols()..])
        .row_mut(0)
        .assign(&single_threshold_row);
    let dense_raw_sigma = raw_sigma_design.design.to_dense();
    let single_raw_sigma = dense_raw_sigma.row(row).to_owned();
    let single_raw_sigma_mat =
        Array2::from_shape_vec((1, dense_raw_sigma.ncols()), single_raw_sigma.to_vec())
            .map_err(|e| format!("single-row sigma reshape failed: {e}"))?;
    let prepared_sigma_design = if let Some(transform) = survival_noise_transform.as_ref() {
        apply_scale_deviation_transform(
            &survival_primary_design,
            &single_raw_sigma_mat,
            transform,
        )?
    } else {
        single_raw_sigma_mat
    };

    let link_wiggle_knots = model
        .linkwiggle_knots
        .as_ref()
        .map(|k| Array1::from_vec(k.clone()));
    let link_wiggle_degree = model.linkwiggle_degree;

    let single_threshold_design = Array2::from_shape_vec(
        (1, dense_threshold_design.ncols()),
        single_threshold_row.to_vec(),
    )
    .map_err(|e| format!("single-row threshold reshape failed: {e}"))?;

    let pred_input = SurvivalLocationScalePredictInput {
        x_time_exit,
        eta_time_offset_exit: Array1::from_elem(1, r_eta_exit[0]),
        time_wiggle_knots: saved_timewiggle_runtime
            .as_ref()
            .map(|w| Array1::from_vec(w.knots.clone())),
        time_wiggle_degree: saved_timewiggle_runtime.as_ref().map(|w| w.degree),
        time_wiggle_ncols: saved_timewiggle_runtime
            .as_ref()
            .map_or(0, |w| w.beta.len()),
        x_threshold: DesignMatrix::Dense(DenseDesignMatrix::from(single_threshold_design)),
        eta_threshold_offset: Array1::from_elem(1, primary_offset_row),
        x_log_sigma: DesignMatrix::Dense(DenseDesignMatrix::from(prepared_sigma_design)),
        eta_log_sigma_offset: Array1::from_elem(1, noise_offset_row),
        x_link_wiggle: None,
        link_wiggle_knots,
        link_wiggle_degree,
        inverse_link: survival_inverse_link,
    };
    let pred = predict_survival_location_scale(&pred_input, &saved_fit)
        .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
    let eta = pred.eta[0];
    let surv = pred.survival_prob[0].clamp(1e-300, 1.0);
    let cum = -surv.ln();
    let _ = n;
    Ok((eta, cum, cum.max(1e-12)))
}

fn evaluate_rp_row(
    model: &SavedModel,
    row_time: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    eta_offset_row: f64,
) -> Result<(f64, f64, f64), String> {
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let saved_runtime = model.saved_prediction_runtime()?;
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();
    let p_time = row_time.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());
    let p_cov = cov_row.len();
    let p = p_time + p_timewiggle + p_cov;
    let mut x_exit = Array2::<f64>::zeros((1, p));
    if p_time > 0 {
        x_exit
            .slice_mut(s![.., ..p_time])
            .assign(&row_time.x_exit_time.to_dense());
    }
    if p_cov > 0 {
        x_exit
            .slice_mut(s![
                ..,
                (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov)
            ])
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
    Ok((eta, cum, cum.max(1e-12)))
}

// ---------------------------------------------------------------------------
// Shared library helpers (used by the CLI wrapper too).
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
    parse_survival_likelihood_mode(raw)
}

/// Baseline config with a linear fallback for plain Weibull models that
/// don't carry an explicit timewiggle.
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
pub fn build_baseline_offsets_by_mode(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        build_survival_marginal_slope_baseline_offsets(age_entry, age_exit, baseline_cfg)
    } else {
        build_survival_baseline_offsets(age_entry, age_exit, baseline_cfg)
    }
}

/// Resolve the covariate `TermCollectionSpec` for prediction, remapping
/// saved training-column indices onto the runtime dataset's layout.
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

/// Canonical saved fit result for prediction.
pub fn fit_result_from_saved_model_for_prediction(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.fit_result.clone().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })
}

/// Resolve the saved survival location-scale fit result.
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
    crate::families::survival_location_scale::SurvivalLocationScaleFitResult::try_from(fit)
        .map_err(|e| format!("failed to interpret saved fit as survival location-scale: {e}"))
}

/// Resolve the saved survival inverse-link, with a probit fallback for
/// payloads that stored an identity link by mistake.
pub fn resolve_survival_inverse_link_from_saved(
    model: &SavedModel,
) -> Result<InverseLink, String> {
    model
        .resolved_inverse_link()?
        .ok_or_else(|| {
            "saved survival model missing resolved inverse-link; refit with current CLI".to_string()
        })
        .map(|link| match link {
            InverseLink::Standard(LinkFunction::Identity) => {
                InverseLink::Standard(LinkFunction::Probit)
            }
            other => other,
        })
}

/// Build a [`ScaleDeviationTransform`] from saved projection metadata
/// (returns `None` if no transform was persisted).
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
