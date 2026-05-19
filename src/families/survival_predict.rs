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

use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::scale_design::scale_transform_from_payload;
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalLikelihoodMode, SurvivalTimeBuildOutput,
    add_survival_time_derivative_guard_offset, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, build_survival_timewiggle_derivative_design,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    normalize_survival_time_pair, parse_survival_baseline_config, parse_survival_distribution,
    parse_survival_likelihood_mode, require_structural_survival_time_basis,
    resolved_survival_time_basis_config_from_build, survival_derivative_guard_for_likelihood,
    survival_likelihood_modename,
};
use crate::families::survival_location_scale::residual_distribution_inverse_link;
use crate::gamlss::buildwiggle_block_input_from_knots;
use crate::inference::formula_dsl::parse_link_choice;
use crate::inference::model::{
    FittedFamily, FittedModel as SavedModel, SavedBaselineTimeWiggleRuntime,
    load_survival_time_basis_config_from_model, survival_baseline_config_from_model,
};
use crate::inference::predict::{BernoulliMarginalSlopePredictor, PredictInput, predict_gam};
use crate::linalg::matrix::DesignMatrix;
use crate::mixture_link::{
    inverse_link_jet_for_inverse_link, state_from_beta_logisticspec, state_from_sasspec,
    state_fromspec,
};
use crate::probability::signed_probit_logcdf_and_mills_ratio;
use crate::solver::estimate::{BlockRole, FittedBlock, FittedLinkState, UnifiedFitResult};
use crate::term_builder::resolve_role_col;
use crate::terms::smooth::{TermCollectionSpec, build_term_collection_design};
use crate::types::{
    InverseLink, LikelihoodFamily, LinkComponent, LinkFunction, MixtureLinkSpec, SasLinkSpec,
};

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
    /// When true, the result also carries delta-method standard errors
    /// for the survival surface (response scale) and the linear
    /// predictor.  Currently honored for `LocationScale` only; other
    /// likelihood modes return `Err` rather than silently dropping
    /// the request.
    pub with_uncertainty: bool,
}

/// Result of [`predict_survival`].
pub struct SurvivalPredictResult {
    pub times: Vec<f64>,
    pub hazard: Array2<f64>,
    pub survival: Array2<f64>,
    pub cumulative_hazard: Array2<f64>,
    pub linear_predictor: Array1<f64>,
    pub likelihood_mode: SurvivalLikelihoodMode,
    /// Per-cell delta-method SE on the survival surface.  Same shape as
    /// `survival`.  Populated only when the request set
    /// `with_uncertainty = true` and the model class supports it.
    pub survival_se: Option<Array2<f64>>,
    /// Per-row delta-method SE on the linear predictor at the row's own
    /// exit time.  Length `n`.  Populated under the same conditions as
    /// `survival_se`.
    pub eta_se: Option<Array1<f64>>,
}

/// Run the survival prediction pipeline.
///
/// Pure library function: no progress bars, no file I/O, no uncertainty
/// bounds. The CLI wraps this with progress updates + CSV writes; the
/// FFI wraps it with JSON serialization.
pub fn predict_survival(req: SurvivalPredictRequest<'_>) -> Result<SurvivalPredictResult, String> {
    let SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset,
        noise_offset,
        time_grid,
        with_uncertainty,
    } = req;

    let entryname = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let entry_col = resolve_role_col(col_map, entryname, "entry")?;
    let exit_col = resolve_role_col(col_map, exitname, "exit")?;

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    // Clip continuous covariate columns to the training range before basis
    // assembly so polyharmonic / spline terms cannot extrapolate outside the
    // data envelope. Times (`entry_col` / `exit_col`) are read from the
    // original `data` view further down so the hazard integration stays on
    // the raw timestamps the user supplied.
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
        .map_err(|e| format!("failed to build survival prediction design: {e}"))?;

    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "survival prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len()
        ));
    }

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let pairs: Result<Vec<(f64, f64)>, _> = (0..n)
        .into_par_iter()
        .map(|i| normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i))
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (t0, t1)) in pairs.into_iter().enumerate() {
        age_entry[i] = t0;
        age_exit[i] = t1;
    }

    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;

    // Latent modes emit binary event-window probabilities, not survival
    // curves. The CLI's `run_predict_saved_latent_*` helpers wrap them with
    // window quadrature + uncertainty pipelines that aren't ported yet.
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(format!(
            "survival prediction via predict_survival does not support likelihood_mode={} yet; \
             latent window prediction lives in the CLI's run_predict_saved_latent_window_impl \
             pipeline and has not yet been ported to the library. Use the CLI predict command.",
            survival_likelihood_modename(saved_likelihood_mode)
        ));
    }
    // Location-scale: handled via a dedicated batch path that calls
    // `predict_survival_location_scale` directly.
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        return predict_survival_location_scale_batch(
            model,
            &age_entry,
            &age_exit,
            &cov_design,
            primary_offset,
            noise_offset,
            training_headers,
            col_map,
            data,
            time_grid,
            with_uncertainty,
        );
    }
    if with_uncertainty {
        return Err(format!(
            "predict_survival: with_uncertainty is currently supported only for the \
             location-scale likelihood mode; got {}",
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
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle()
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

    // For marginal-slope, build the saved predictor (with link-deviation +
    // score-warp blocks plumbed in) once. The per-(row, t) loop reuses this
    // predictor and only assembles the per-cell q-design slice. Without this,
    // the library skipped link-deviation and score-warp replay entirely and
    // disagreed with the CLI's `gam predict` on every flex model.
    let marginal_slope_ctx = if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        // Baseline offsets at the predict-data's age_entry / age_exit. Used to
        // build the predictor's `pred_input` (which we discard) — the actual
        // per-(row, t) offset is rebuilt inside `evaluate_marginal_slope_row`.
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                &age_entry,
                &age_exit,
                &baseline_cfg,
                saved_likelihood_mode,
                None,
            )?;
        Some(build_marginal_slope_predict_context(
            model,
            data,
            col_map,
            training_headers,
            &cov_design.design,
            primary_offset,
            noise_offset,
            &time_build,
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
        )?)
    } else {
        None
    };

    // Evaluate each row independently.  For an explicit time grid, each worker
    // reuses the row's covariate slice across all grid times and returns a
    // complete row, avoiding synchronized writes into the output matrices.
    struct SurvivalPredictionRow {
        hazard: Vec<f64>,
        survival: Vec<f64>,
        cumulative_hazard: Vec<f64>,
        linear_predictor: f64,
    }

    let row_results: Result<Vec<SurvivalPredictionRow>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let cov_row = if matches!(
                saved_likelihood_mode,
                SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
            ) {
                Some(design_row_owned(
                    &cov_design.design,
                    i,
                    "survival predict covariate row",
                )?)
            } else {
                None
            };
            let evaluate_at = |t_query: f64| -> Result<(f64, f64, f64), String> {
                let t_entry = age_entry[i].min(t_query);
                let single_entry = Array1::from_elem(1, t_entry);
                let single_exit = Array1::from_elem(1, t_query);
                let mut row_time =
                    build_survival_time_basis(&single_entry, &single_exit, time_cfg.clone(), None)?;
                if let Some(anchor_row) = time_anchor_row_cached.as_ref() {
                    center_survival_time_designs_at_anchor(
                        &mut row_time.x_entry_time,
                        &mut row_time.x_exit_time,
                        anchor_row,
                    )?;
                }
                let (_r_eta_entry, r_eta_exit, r_deriv_exit) =
                    build_survival_time_offsets_for_likelihood(
                        &single_entry,
                        &single_exit,
                        &baseline_cfg,
                        saved_likelihood_mode,
                        None,
                    )?;

                match saved_likelihood_mode {
                    SurvivalLikelihoodMode::MarginalSlope => {
                        let ctx = marginal_slope_ctx.as_ref().ok_or_else(|| {
                            "internal error: marginal-slope context missing for marginal-slope mode"
                                .to_string()
                        })?;
                        evaluate_marginal_slope_row(
                            i,
                            ctx,
                            &row_time,
                            &r_eta_exit,
                            &r_deriv_exit,
                            primary_offset[i],
                        )
                    }
                    SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                        let cov_row = cov_row.as_ref().ok_or_else(|| {
                            "internal error: covariate row missing for Royston-Parmar prediction"
                                .to_string()
                        })?;
                        evaluate_rp_row(
                            model,
                            &row_time,
                            cov_row,
                            r_eta_exit[0],
                            r_deriv_exit[0],
                            primary_offset[i],
                        )
                    }
                    SurvivalLikelihoodMode::Latent
                    | SurvivalLikelihoodMode::LatentBinary
                    | SurvivalLikelihoodMode::LocationScale => {
                        Err("unreachable: unsupported likelihood_mode filtered earlier".to_string())
                    }
                }
            };

            let mut row = SurvivalPredictionRow {
                hazard: vec![0.0; t_cols],
                survival: vec![0.0; t_cols],
                cumulative_hazard: vec![0.0; t_cols],
                linear_predictor: 0.0,
            };
            if per_row_eval {
                let (eta_t, cum_t, haz_t) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
                row.hazard[0] = haz_t;
                row.cumulative_hazard[0] = cum_t;
                row.survival[0] = (-cum_t).exp().clamp(0.0, 1.0);
            } else {
                for (j, &t_query) in eval_times.iter().enumerate() {
                    let (_eta_t, cum_t, haz_t) = evaluate_at(t_query)?;
                    row.hazard[j] = haz_t;
                    row.cumulative_hazard[j] = cum_t;
                    row.survival[j] = (-cum_t).exp().clamp(0.0, 1.0);
                }
                let (eta_t, _, _) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
            }
            Ok(row)
        })
        .collect();

    for (i, row) in row_results?.into_iter().enumerate() {
        linear_predictor[i] = row.linear_predictor;
        for j in 0..t_cols {
            hazard[[i, j]] = row.hazard[j];
            cumulative_hazard[[i, j]] = row.cumulative_hazard[j];
            survival[[i, j]] = row.survival[j];
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
        survival_se: None,
        eta_se: None,
    })
}

// ---------------------------------------------------------------------------
// Per-mode single-row evaluators.
// ---------------------------------------------------------------------------

/// Precomputed context for evaluating the saved survival marginal-slope
/// predictor row-by-row. Built once per call to `predict_survival` so the
/// per-(row, t) loop only assembles the per-time q-design slice.
struct MarginalSlopePredictContext {
    predictor: BernoulliMarginalSlopePredictor,
    /// Time-block coefficients (length `p_time_base + p_timewiggle`).
    beta_time: Array1<f64>,
    /// Covariate (marginal) coefficients.
    beta_marginal: Array1<f64>,
    saved_timewiggle: Option<SavedBaselineTimeWiggleRuntime>,
    /// Covariate design (n × p_marginal), kept operator-backed when possible.
    cov_design: DesignMatrix,
    /// Logslope design (n × p_logslope), kept operator-backed when possible.
    logslope_design: DesignMatrix,
    /// Per-row covariate eta = `cov_design[i] · beta_marginal`. Used to
    /// pre-compute `q_exit_base`.
    cov_eta: Array1<f64>,
    /// Per-row latent z (raw, un-normalized — the predictor's
    /// `latent_z_normalization` is applied internally).
    z_raw: Array1<f64>,
    /// Per-row noise offset, mirroring the `pred_input.offset_noise` slice
    /// used by the CLI.
    noise_offset: Array1<f64>,
}

fn design_row_owned(
    design: &DesignMatrix,
    row: usize,
    context: &str,
) -> Result<Array1<f64>, String> {
    let chunk = design
        .try_row_chunk(row..row + 1)
        .map_err(|e| format!("{context}: {e}"))?;
    Ok(chunk.row(0).to_owned())
}

fn build_marginal_slope_predict_context(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cov_design: &DesignMatrix,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    time_build: &SurvivalTimeBuildOutput,
    eta_offset_entry: &Array1<f64>,
    eta_offset_exit: &Array1<f64>,
    derivative_offset_exit: &Array1<f64>,
) -> Result<MarginalSlopePredictContext, String> {
    let z_name = model
        .z_column
        .as_ref()
        .ok_or_else(|| "saved survival marginal-slope model missing z_column".to_string())?;
    let z_col = resolve_role_col(col_map, z_name, "z")?;
    let z_raw = data.column(z_col).to_owned();

    let logslopespec = resolve_termspec_for_prediction(
        &model.resolved_termspec_logslope.as_ref().cloned(),
        training_headers,
        col_map,
        "resolved_termspec_logslope",
    )?;
    let logslope_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let logslope_input = logslope_clipped.as_ref().map_or(data, |arr| arr.view());
    let logslope_design = build_term_collection_design(logslope_input, &logslopespec)
        .map_err(|e| format!("failed to build survival marginal-slope logslope design: {e}"))?;

    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let (predictor, _pred_input, _predictor_fit) = build_saved_survival_marginal_slope_predictor(
        model,
        &fit_saved,
        z_name,
        &z_raw,
        cov_design,
        &logslope_design.design,
        time_build,
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        primary_offset,
        noise_offset,
    )?;

    let blocks = &fit_saved.blocks;
    if blocks.len() < 3 {
        return Err(format!(
            "saved survival marginal-slope model requires at least 3 blocks [time, marginal, slope], got {}",
            blocks.len()
        ));
    }
    let beta_time = blocks[0].beta.clone();
    let beta_marginal = blocks[1].beta.clone();
    let saved_runtime = model.saved_prediction_runtime()?;
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();

    // cov_eta is time-independent so doing it here avoids `O(n × T)`
    // re-multiplications inside the per-cell loop.
    let cov_eta = cov_design.dot(&beta_marginal);

    Ok(MarginalSlopePredictContext {
        predictor,
        beta_time,
        beta_marginal,
        saved_timewiggle,
        cov_design: cov_design.clone(),
        logslope_design: logslope_design.design.clone(),
        cov_eta,
        z_raw,
        noise_offset: noise_offset.clone(),
    })
}

/// Evaluate one (row, t) cell for the saved survival marginal-slope kernel.
///
/// Calls the saved [`BernoulliMarginalSlopePredictor`]
/// (`predict_eta_and_q_chain`) to obtain both the linear predictor `eta` and
/// the exact IFT-pullback factor `∂eta/∂q`. The survival-index time derivative
/// is then `(∂eta/∂q) · qd_with_wiggle`. In rigid mode this collapses to
/// `c · qd` (the closed-form probit-frailty composition); under score-warp /
/// link-deviation it picks up the exact implicit-function pull-back through the
/// per-row calibration intercept, mirroring `compute_survival_timepoint_exact`
/// in `survival_marginal_slope.rs`.
fn evaluate_marginal_slope_row(
    row_index: usize,
    ctx: &MarginalSlopePredictContext,
    row_time: &SurvivalTimeBuildOutput,
    r_eta_exit: &Array1<f64>,
    r_deriv_exit: &Array1<f64>,
    primary_offset_row: f64,
) -> Result<(f64, f64, f64), String> {
    let beta_time = &ctx.beta_time;
    let p_time_base = row_time.x_exit_time.ncols();
    let p_timewiggle = ctx
        .saved_timewiggle
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

    // Pre-wiggle q-eta for this (row, t) cell. Mirrors the CLI's `q_exit_base`
    // construction in `build_saved_survival_marginal_slope_predictor`:
    //   q = time_basis(t) · beta_time_base + cov[row] · beta_marginal
    //       + r_eta_exit + primary_offset_row.
    let q_exit_base = row_time.x_exit_time.dot(&beta_time_base)[0]
        + ctx.cov_eta[row_index]
        + r_eta_exit[0]
        + primary_offset_row;
    let qd_exit_base = row_time.x_derivative_time.dot(&beta_time_base)[0] + r_deriv_exit[0];

    // For timewiggle the `exit_design` row enters the predictor's q-design;
    // the `derivative_design` row enters the time-derivative used to build the
    // hazard. Both are evaluated at the wiggle anchor `q_exit_base`.
    let (qd_with_wiggle, exit_wiggle_design) = if let Some(runtime) = ctx.saved_timewiggle.as_ref()
    {
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
        let derivative_design = build_survival_timewiggle_derivative_design(
            &eta_exit_row,
            &deriv_row,
            &knots,
            runtime.degree,
        )?;
        (
            qd_exit_base + derivative_design.dot(&beta_w)[0],
            Some(exit_design),
        )
    } else {
        (qd_exit_base, None)
    };

    // Build a 1-row PredictInput for this (row, t) cell and call the saved
    // predictor. The predictor's `marginal_eta` formula is
    //   marginal_eta = q_design · combined_q_beta + baseline_marginal + offset
    // with `combined_q_beta = [beta_time | beta_marginal]` and the survival
    // predictor sets `baseline_marginal = 0`. We supply the full per-row
    // q_design = [time_basis(t) | timewiggle | cov_design[row]] so the
    // predictor reproduces `q_with_wiggle` exactly with `offset = r_eta_exit[0]
    // + primary_offset_row`.
    let cov_dim = ctx.beta_marginal.len();
    let q_design_ncols = p_time_base + p_timewiggle + cov_dim;
    let mut q_design_full = Array2::<f64>::zeros((1, q_design_ncols));
    q_design_full
        .slice_mut(s![.., ..p_time_base])
        .assign(&row_time.x_exit_time.to_dense());
    if let Some(exit_w) = exit_wiggle_design.as_ref() {
        q_design_full
            .slice_mut(s![.., p_time_base..p_time_base + p_timewiggle])
            .assign(exit_w);
    }
    if cov_dim > 0 {
        let cov_row = design_row_owned(
            &ctx.cov_design,
            row_index,
            "survival marginal covariate row",
        )?;
        q_design_full
            .slice_mut(s![.., p_time_base + p_timewiggle..])
            .row_mut(0)
            .assign(&cov_row);
    }

    // Logslope design row + offset chosen so that the predictor's logslope_eta
    // equals our precomputed `slope_eta[row]`.  The predictor computes:
    //   logslope_eta = design_noise · beta_logslope + baseline_logslope
    //                  + offset_noise.
    // We feed the actual saved logslope row + the row's noise offset, matching
    // exactly the CLI's `pred_input.design_noise` / `offset_noise` slice.
    let logslope_row = design_row_owned(
        &ctx.logslope_design,
        row_index,
        "survival marginal logslope row",
    )?;
    let mut logslope_design_2d = Array2::<f64>::zeros((1, logslope_row.len()));
    logslope_design_2d.row_mut(0).assign(&logslope_row);

    let pred_input = PredictInput {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(q_design_full)),
        offset: Array1::from_elem(1, r_eta_exit[0] + primary_offset_row),
        design_noise: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            logslope_design_2d,
        ))),
        offset_noise: Some(Array1::from_elem(1, ctx.noise_offset[row_index])),
        auxiliary_scalar: Some(Array1::from_elem(1, ctx.z_raw[row_index])),
        auxiliary_matrix: None,
    };

    // Exact IFT pull-back: the predictor returns both `eta` and the analytic
    // factor `∂eta/∂q` for this (row, t). This gives d eta(t) / dt; the hazard
    // conversion below divides the event density by S(t).
    let (eta_arr, deta_dq_arr) = ctx
        .predictor
        .predict_eta_and_q_chain(&pred_input)
        .map_err(|e| format!("saved survival marginal-slope predictor eta failed: {e}"))?;
    let eta = eta_arr[0];
    let eta_derivative = deta_dq_arr[0] * qd_with_wiggle;
    let (cum, haz) = probit_survival_hazard_components(eta, eta_derivative)?;
    Ok((eta, cum, haz))
}

#[inline]
fn probit_survival_hazard_components(eta: f64, eta_derivative: f64) -> Result<(f64, f64), String> {
    if !(eta.is_finite() && eta_derivative.is_finite() && eta_derivative > 0.0) {
        return Err(format!(
            "saved survival marginal-slope prediction produced invalid survival index derivative: eta={eta}, eta_t={eta_derivative}"
        ));
    }

    // Survival marginal-slope defines S(t) = Phi(-eta(t)). The event density
    // is f(t) = phi(eta(t)) * eta'(t), while the hazard rate exposed by the
    // prediction API is h(t) = f(t) / S(t). The signed-probit helper returns
    // both log Phi(-eta) and the stable Mills ratio phi(eta) / Phi(-eta).
    let (log_survival, mills_ratio) = signed_probit_logcdf_and_mills_ratio(-eta);
    let cumulative_hazard = -log_survival;
    let hazard = mills_ratio * eta_derivative;
    // `>= 0.0` rejects NaN (a programming-bug signal) and accepts the full
    // mathematical range [0, +∞]. Saturated probit fits where the model
    // genuinely says S(t)→0 produce a +∞ cumulative hazard — that is the
    // truthful answer, and the consumer's `survival = exp(-cum).clamp(0,1)`
    // handles it cleanly. Rejecting +∞ would force the predictor to fail on
    // models that the inner solver has already certified as a valid fit.
    if !(cumulative_hazard >= 0.0 && hazard >= 0.0) {
        return Err(format!(
            "saved survival marginal-slope prediction produced invalid survival components: eta={eta}, eta_t={eta_derivative}, log_survival={log_survival}, hazard={hazard}"
        ));
    }
    Ok((cumulative_hazard, hazard))
}

fn evaluate_rp_row(
    model: &SavedModel,
    row_time: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    eta_time_offset_row: f64,
    derivative_time_offset_row: f64,
    primary_offset_row: f64,
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
    let beta = fit_saved.beta.clone();
    if beta.len() != p {
        return Err(format!(
            "survival RP coefficient mismatch: beta has {} entries but design has {} columns",
            beta.len(),
            p
        ));
    }
    let mut x_exit = Array2::<f64>::zeros((1, p));
    if p_time > 0 {
        x_exit
            .slice_mut(s![.., ..p_time])
            .assign(&row_time.x_exit_time.to_dense());
    }
    let mut eta_derivative = derivative_time_offset_row;
    if p_time > 0 {
        eta_derivative += row_time
            .x_derivative_time
            .dot(&beta.slice(s![..p_time]).to_owned())[0];
    }
    if let Some(runtime) = saved_timewiggle.as_ref() {
        let knots = Array1::from_vec(runtime.knots.clone());
        let beta_w = beta.slice(s![p_time..p_time + p_timewiggle]).to_owned();
        let eta_exit_row = Array1::from_elem(1, eta_time_offset_row);
        let derivative_exit_row = Array1::from_elem(1, derivative_time_offset_row);
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
            _ => return Err("saved baseline-timewiggle exit design must be dense".to_string()),
        };
        if exit_design.ncols() != p_timewiggle {
            return Err(format!(
                "survival RP timewiggle design mismatch: rebuilt {} columns but runtime expects {}",
                exit_design.ncols(),
                p_timewiggle
            ));
        }
        x_exit
            .slice_mut(s![.., p_time..p_time + p_timewiggle])
            .assign(&exit_design);
        let derivative_design = build_survival_timewiggle_derivative_design(
            &eta_exit_row,
            &derivative_exit_row,
            &knots,
            runtime.degree,
        )?;
        eta_derivative += derivative_design.dot(&beta_w)[0];
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
    let offset_view = Array1::from_elem(1, eta_time_offset_row + primary_offset_row);
    let pred = predict_gam(
        x_exit.view(),
        beta.view(),
        offset_view.view(),
        LikelihoodFamily::RoystonParmar,
    )
    .map_err(|e| format!("survival prediction failed: {e}"))?;
    let eta = pred.eta[0];
    let (cum, haz) = royston_parmar_survival_hazard_components(eta, eta_derivative)?;
    Ok((eta, cum, haz))
}

#[inline]
fn royston_parmar_survival_hazard_components(
    eta: f64,
    eta_derivative: f64,
) -> Result<(f64, f64), String> {
    if !(eta.is_finite() && eta_derivative.is_finite() && eta_derivative > 0.0) {
        return Err(format!(
            "saved Royston-Parmar survival prediction produced invalid log-cumulative-hazard derivative: eta={eta}, eta_t={eta_derivative}"
        ));
    }
    let cumulative_hazard = eta.exp();
    let hazard = cumulative_hazard * eta_derivative;
    // Royston-Parmar parameterizes `eta = log Lambda(t)`, so `Lambda = exp(eta)`
    // is unbounded above and `exp(eta)` saturates to `+∞` in f64 once
    // `eta >~ 709.78` — exactly the regime a saturated RP fit produces in the
    // right tail. The math is well-defined (`S(t) → 0`, `h(t) → ∞`); rejecting
    // `+∞` here would crash predict on a fit the inner solver already accepted.
    // `>= 0.0` rejects NaN (the only true bug signal) while allowing the full
    // [0, +∞] range. The consumer materializes survival via
    // `survival = exp(-cum).clamp(0, 1)`, which collapses cleanly at saturation.
    if !(cumulative_hazard >= 0.0 && hazard >= 0.0) {
        return Err(format!(
            "saved Royston-Parmar survival prediction produced invalid survival components: eta={eta}, eta_t={eta_derivative}, cumulative_hazard={cumulative_hazard}, hazard={hazard}"
        ));
    }
    Ok((cumulative_hazard, hazard))
}

/// Batch evaluator for the location-scale survival likelihood mode.
///
/// Mirrors the CLI's LocationScale predict path (main.rs::run_predict_survival
/// LocationScale arm) but stays library-only: builds the threshold/log_sigma
/// designs from the saved frozen specs, replays the saved scale-deviation
/// transform on the noise design, applies the survival time-derivative guard,
/// and calls `predict_survival_location_scale`.
///
/// Plugin survival only — uncertainty paths still live in the CLI.
fn predict_survival_location_scale_batch(
    model: &SavedModel,
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cov_design: &crate::terms::smooth::TermCollectionDesign,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    data: ArrayView2<'_, f64>,
    time_grid: Option<&[f64]>,
    with_uncertainty: bool,
) -> Result<SurvivalPredictResult, String> {
    use crate::families::scale_design::build_scale_deviation_operator;
    use crate::families::survival_construction::evaluate_survival_time_basis_row;
    use crate::families::survival_location_scale::{
        SurvivalLocationScalePredictInput, predict_survival_location_scale,
        predict_survival_location_scale_from_linear_components,
        predict_survival_location_scalewith_uncertainty,
    };
    use crate::matrix::DesignMatrix;

    let n = age_entry.len();
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
    let eval_width = if per_row_eval { 1 } else { t_cols + 1 };
    let saved_likelihood_mode = SurvivalLikelihoodMode::LocationScale;
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(age_entry, age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor = model
        .survival_time_anchor
        .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    if !model.has_baseline_time_wiggle() {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let saved_inverse_link = resolve_survival_inverse_link_from_saved(model)?;
    let (eval_entry, eval_exit) = if per_row_eval {
        (age_entry.clone(), age_exit.clone())
    } else {
        let total = n * eval_width;
        let mut entry = Array1::<f64>::zeros(total);
        let mut exit = Array1::<f64>::zeros(total);
        {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let pairs: Vec<(f64, f64)> = (0..total)
                .into_par_iter()
                .map(|k| {
                    let i = k / eval_width;
                    let col = k % eval_width;
                    let t = if col < t_cols {
                        eval_times[col]
                    } else {
                        age_exit[i]
                    };
                    (age_entry[i].min(t), t)
                })
                .collect();
            for (k, (t0, t1)) in pairs.into_iter().enumerate() {
                entry[k] = t0;
                exit[k] = t1;
            }
        }
        (entry, exit)
    };
    let mut time_build =
        build_survival_time_basis(&eval_entry, &eval_exit, time_cfg.clone(), None)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &eval_entry,
            &eval_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            Some(&saved_inverse_link),
        )?;
    add_survival_time_derivative_guard_offset(
        &eval_entry,
        &eval_exit,
        time_anchor,
        survival_derivative_guard_for_likelihood(saved_likelihood_mode),
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;

    let saved_fit = saved_survival_location_scale_fit_result(model)?;
    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;

    // Build threshold + log-sigma designs from the frozen saved specs. Re-using
    // resolve_termspec_for_prediction guarantees we honor the predict-data's
    // column layout via the model's training_headers.
    // The threshold design uses the same frozen spec as the covariate design
    // already built for predict_survival; reuse it instead of rebuilding.
    let threshold_design = cov_design;
    let log_sigmaspec = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let sigma_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let sigma_input = sigma_clipped.as_ref().map_or(data, |arr| arr.view());
    let raw_sigma_design =
        crate::terms::smooth::build_term_collection_design(sigma_input, &log_sigmaspec)
            .map_err(|err| format!("failed to build survival log-sigma design: {err}"))?;
    let survival_noise_transform = scale_transform_from_payload(
        &model.survival_noise_projection,
        &model.survival_noise_center,
        &model.survival_noise_scale,
        model.survival_noise_non_intercept_start,
        model.survival_noise_projection_ridge_alpha,
    )?;

    let x_time_exit_dense = time_build
        .x_exit_time
        .try_to_dense_by_chunks("survival location-scale prediction time-exit design")?;
    let total_rows = eval_exit.len();
    let x_time_exit = if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
        let mut full =
            Array2::<f64>::zeros((total_rows, x_time_exit_dense.ncols() + runtime.beta.len()));
        full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
            .assign(&x_time_exit_dense);
        full
    } else {
        x_time_exit_dense
    };

    let repeat_rows = |matrix: &DesignMatrix, label: &str| -> Result<DesignMatrix, String> {
        if per_row_eval {
            return Ok(matrix.clone());
        }
        let dense = matrix.try_to_dense_by_chunks(label)?;
        let mut repeated = Array2::<f64>::zeros((total_rows, dense.ncols()));
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let rows: Vec<Vec<f64>> = (0..total_rows)
            .into_par_iter()
            .map(|k| dense.row(k / eval_width).to_vec())
            .collect();
        for (k, row) in rows.into_iter().enumerate() {
            for (j, value) in row.into_iter().enumerate() {
                repeated[[k, j]] = value;
            }
        }
        Ok(DesignMatrix::from(repeated))
    };
    let threshold_matrix = repeat_rows(
        &threshold_design.design,
        "survival location-scale prediction threshold design",
    )?;
    let raw_sigma_matrix = repeat_rows(
        &raw_sigma_design.design,
        "survival location-scale prediction log-sigma design",
    )?;

    let time_design = DesignMatrix::from(x_time_exit.clone());
    let survival_primary_design =
        DesignMatrix::hstack(vec![time_design, threshold_matrix.clone()])?;
    let prepared_sigma_design = if let Some(transform) = survival_noise_transform.as_ref() {
        build_scale_deviation_operator(survival_primary_design, raw_sigma_matrix, transform)?
    } else {
        raw_sigma_matrix
    };
    let link_wiggle_knots = model
        .linkwiggle_knots
        .as_ref()
        .map(|k| Array1::from_vec(k.clone()));
    let link_wiggle_degree = model.linkwiggle_degree;
    let time_wiggle_knots = saved_timewiggle_runtime
        .as_ref()
        .map(|w| Array1::from_vec(w.knots.clone()));
    let time_wiggle_degree = saved_timewiggle_runtime.as_ref().map(|w| w.degree);
    let time_wiggle_ncols = saved_timewiggle_runtime
        .as_ref()
        .map_or(0, |w| w.beta.len());

    let expand_vector = |values: &Array1<f64>| -> Array1<f64> {
        if per_row_eval {
            values.clone()
        } else {
            Array1::from_shape_fn(total_rows, |k| values[k / eval_width])
        }
    };
    // Build the SurvivalLocationScalePredictInput once, with replicated /
    // expanded designs and offsets, regardless of `per_row_eval`.  This
    // unifies the mean-only and uncertainty paths and lets the
    // uncertainty branch reuse the same input.
    let pred_input = SurvivalLocationScalePredictInput {
        x_time_exit,
        eta_time_offset_exit: eta_offset_exit.clone(),
        time_wiggle_knots: time_wiggle_knots.clone(),
        time_wiggle_degree,
        time_wiggle_ncols,
        x_threshold: threshold_matrix,
        eta_threshold_offset: expand_vector(primary_offset),
        x_log_sigma: prepared_sigma_design,
        eta_log_sigma_offset: expand_vector(noise_offset),
        x_link_wiggle: None,
        link_wiggle_knots: link_wiggle_knots.clone(),
        link_wiggle_degree,
        inverse_link: saved_inverse_link.clone(),
    };

    // Mean / SE computation.  The uncertainty path also computes the
    // survival mean and eta, so we use whichever output we have.
    let (eta_full, survival_prob_full, response_se_full, eta_se_full): (
        Array1<f64>,
        Array1<f64>,
        Option<Array1<f64>>,
        Option<Array1<f64>>,
    ) = if with_uncertainty {
        let cov = saved_fit.beta_covariance().ok_or_else(|| {
            "survival location-scale uncertainty: saved fit is missing the \
             posterior covariance; refit / library to \
             populate beta_covariance"
                .to_string()
        })?;
        let unc = predict_survival_location_scalewith_uncertainty(
            &pred_input,
            &saved_fit,
            cov,
            false,
            true,
        )
        .map_err(|err| format!("survival location-scale uncertainty predict failed: {err}"))?;
        let response_se = unc.response_standard_error.ok_or_else(|| {
            "survival location-scale uncertainty: response_standard_error \
             missing despite include_response_sd=true"
                .to_string()
        })?;
        (
            unc.eta,
            unc.survival_prob,
            Some(response_se),
            Some(unc.eta_standard_error),
        )
    } else if per_row_eval {
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|err| format!("survival location-scale predict failed: {err}"))?;
        (pred.eta, pred.survival_prob, None, None)
    } else {
        let beta_threshold = saved_fit.beta_threshold();
        let beta_log_sigma = saved_fit.beta_log_sigma();
        let eta_t_subject =
            cov_design.design.matrixvectormultiply(&beta_threshold) + primary_offset;
        let eta_ls_subject = prepared_sigma_design_view(&pred_input)
            .matrixvectormultiply(&beta_log_sigma)
            + &expand_vector(noise_offset);
        let eta_t = expand_vector(&eta_t_subject);
        let pred = predict_survival_location_scale_from_linear_components(
            &pred_input.x_time_exit,
            &eta_offset_exit,
            time_wiggle_knots.as_ref(),
            time_wiggle_degree,
            time_wiggle_ncols,
            &eta_t,
            &eta_ls_subject,
            link_wiggle_knots.as_ref(),
            link_wiggle_degree,
            &saved_inverse_link,
            &saved_fit,
        )
        .map_err(|err| format!("survival location-scale predict failed: {err}"))?;
        (pred.eta, pred.survival_prob, None, None)
    };
    let x_time_derivative = time_build
        .x_derivative_time
        .try_to_dense_by_chunks("survival location-scale prediction time-derivative design")?;
    let eta_derivative_full = location_scale_eta_derivative_components(
        &x_time_derivative,
        &derivative_offset_exit,
        &pred_input.x_time_exit,
        &pred_input.eta_time_offset_exit,
        time_wiggle_knots.as_ref(),
        time_wiggle_degree,
        time_wiggle_ncols,
        &saved_fit,
    )?;
    let hazard_full = location_scale_hazard_from_eta_derivative(
        &eta_full,
        &eta_derivative_full,
        &saved_inverse_link,
    )?;

    let mut survival = Array2::<f64>::zeros((n, t_cols));
    let mut cumulative_hazard = Array2::<f64>::zeros((n, t_cols));
    let mut hazard = Array2::<f64>::zeros((n, t_cols));
    ndarray::Zip::indexed(&mut survival)
        .and(&mut cumulative_hazard)
        .and(&mut hazard)
        .par_for_each(|(i, j), s, ch, h| {
            let k = if per_row_eval { i } else { i * eval_width + j };
            let surv = survival_prob_full[k].clamp(1e-300, 1.0);
            *s = surv;
            *ch = -surv.ln();
            *h = hazard_full[k];
        });

    let linear_predictor = if per_row_eval {
        eta_full.clone()
    } else {
        Array1::from_shape_fn(n, |i| eta_full[i * eval_width + t_cols])
    };
    let times = if per_row_eval {
        age_exit.to_vec()
    } else {
        eval_times
    };

    let survival_se = response_se_full.as_ref().map(|response_se| {
        let mut out = Array2::<f64>::zeros((n, t_cols));
        ndarray::Zip::indexed(&mut out).par_for_each(|(i, j), slot| {
            let k = if per_row_eval { i } else { i * eval_width + j };
            *slot = response_se[k].max(0.0);
        });
        out
    });
    let eta_se_per_row = eta_se_full.as_ref().map(|eta_se| {
        if per_row_eval {
            eta_se.clone()
        } else {
            Array1::from_shape_fn(n, |i| eta_se[i * eval_width + t_cols])
        }
    });

    Ok(SurvivalPredictResult {
        times,
        hazard,
        survival,
        cumulative_hazard,
        linear_predictor,
        likelihood_mode: saved_likelihood_mode,
        survival_se,
        eta_se: eta_se_per_row,
    })
}

/// Helper: borrow the prepared sigma design back from the pred_input
/// without consuming it.  Used so the mean-only fast path can reuse the
/// log-sigma design without an extra clone.
fn prepared_sigma_design_view(
    input: &crate::families::survival_location_scale::SurvivalLocationScalePredictInput,
) -> &crate::matrix::DesignMatrix {
    &input.x_log_sigma
}

fn location_scale_eta_derivative_components(
    x_time_derivative: &Array2<f64>,
    derivative_offset_exit: &Array1<f64>,
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    fit: &UnifiedFitResult,
) -> Result<Array1<f64>, String> {
    let n = x_time_exit.nrows();
    if x_time_derivative.nrows() != n
        || derivative_offset_exit.len() != n
        || eta_time_offset_exit.len() != n
    {
        return Err(
            "survival location-scale hazard derivative row mismatch across inputs".to_string(),
        );
    }
    let beta_time = fit.beta_time();
    let p_time_total = beta_time.len();
    let p_wiggle = time_wiggle_ncols.min(p_time_total);
    let p_base = p_time_total - p_wiggle;
    if x_time_exit.ncols() != p_time_total || x_time_derivative.ncols() != p_base {
        return Err(format!(
            "survival location-scale hazard derivative design mismatch: x_exit={} beta_time={} x_derivative={} base={}",
            x_time_exit.ncols(),
            p_time_total,
            x_time_derivative.ncols(),
            p_base
        ));
    }

    let beta_base = beta_time.slice(s![..p_base]).to_owned();
    let mut eta_derivative = if p_base > 0 {
        x_time_derivative.dot(&beta_base) + derivative_offset_exit
    } else {
        derivative_offset_exit.clone()
    };
    if p_wiggle > 0 {
        let knots = time_wiggle_knots.ok_or_else(|| {
            "survival location-scale hazard derivative: timewiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = time_wiggle_degree.ok_or_else(|| {
            "survival location-scale hazard derivative: timewiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let beta_w = beta_time.slice(s![p_base..p_time_total]).to_owned();
        let h_base = if p_base > 0 {
            x_time_exit.slice(s![.., ..p_base]).dot(&beta_base) + eta_time_offset_exit
        } else {
            eta_time_offset_exit.clone()
        };
        let basis_d1 = crate::families::gamlss::monotone_wiggle_basis_with_derivative_order(
            h_base.view(),
            knots,
            degree,
            1,
        )?;
        if basis_d1.ncols() != p_wiggle {
            return Err(format!(
                "survival location-scale hazard derivative timewiggle mismatch: derivative basis has {} columns but beta has {}",
                basis_d1.ncols(),
                p_wiggle
            ));
        }
        eta_derivative *= &(basis_d1.dot(&beta_w) + 1.0);
    }
    if eta_derivative
        .iter()
        .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(
            "survival location-scale hazard derivative must be finite and positive".to_string(),
        );
    }
    Ok(eta_derivative)
}

fn location_scale_hazard_from_eta_derivative(
    eta: &Array1<f64>,
    eta_derivative: &Array1<f64>,
    inverse_link: &InverseLink,
) -> Result<Array1<f64>, String> {
    if eta.len() != eta_derivative.len() {
        return Err(format!(
            "survival location-scale hazard row mismatch: eta={} eta_derivative={}",
            eta.len(),
            eta_derivative.len()
        ));
    }
    let values = eta
        .iter()
        .zip(eta_derivative.iter())
        .map(|(&q, &q_t)| location_scale_hazard_component(q, q_t, inverse_link))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Array1::from_vec(values))
}

fn location_scale_hazard_component(
    eta: f64,
    eta_derivative: f64,
    inverse_link: &InverseLink,
) -> Result<f64, String> {
    if !(eta.is_finite() && eta_derivative.is_finite() && eta_derivative > 0.0) {
        return Err(format!(
            "survival location-scale hazard requires finite eta and positive eta_t, got eta={eta}, eta_t={eta_derivative}"
        ));
    }
    match inverse_link {
        InverseLink::Standard(LinkFunction::Probit) => {
            let (_, hazard) = probit_survival_hazard_components(eta, eta_derivative)?;
            Ok(hazard)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            let (_, hazard) = royston_parmar_survival_hazard_components(eta, eta_derivative)?;
            Ok(hazard)
        }
        InverseLink::Standard(LinkFunction::Logit) => {
            let failure = if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let exp_eta = eta.exp();
                exp_eta / (1.0 + exp_eta)
            };
            Ok(failure * eta_derivative)
        }
        InverseLink::Standard(LinkFunction::Identity) => {
            let survival = 1.0 - eta;
            if !(survival.is_finite() && survival > 0.0) {
                return Err(format!(
                    "survival location-scale identity link produced invalid survival={survival} at eta={eta}"
                ));
            }
            Ok(eta_derivative / survival)
        }
        _ => {
            let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                .map_err(|err| format!("survival location-scale inverse-link jet failed: {err}"))?;
            let survival = 1.0 - jet.mu;
            let hazard = jet.d1 * eta_derivative / survival;
            if !(survival.is_finite() && survival > 0.0 && hazard.is_finite() && hazard >= 0.0) {
                return Err(format!(
                    "survival location-scale inverse link produced invalid hazard components: eta={eta}, eta_t={eta_derivative}, failure={}, d_failure={}, survival={survival}, hazard={hazard}",
                    jet.mu, jet.d1
                ));
            }
            Ok(hazard)
        }
    }
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
                "saved latent survival model is missing survival_likelihood=latent metadata; refit"
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
                "saved latent binary model is missing survival_likelihood=latent-binary metadata; refit"
                    .to_string(),
            ),
        };
    }
    let raw = model.survival_likelihood.as_deref().ok_or_else(|| {
        "saved survival model is missing survival_likelihood metadata; refit".to_string()
    })?;
    parse_survival_likelihood_mode(raw)
}

/// Baseline config with a linear fallback for plain Weibull models that
/// don't carry an explicit timewiggle.
pub fn saved_survival_runtime_baseline_config(
    model: &SavedModel,
    likelihood_mode: SurvivalLikelihoodMode,
) -> Result<SurvivalBaselineConfig, String> {
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle() {
        return parse_survival_baseline_config("linear", None, None, None, None);
    }
    survival_baseline_config_from_model(model)
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
            "model is missing {spec_label}; refit to guarantee train/predict design consistency"
        )
    })?;
    saved.validate_frozen(spec_label)?;
    let headers = training_headers.ok_or_else(|| {
        "model is missing training_headers; refit to guarantee stable feature mapping at prediction time"
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
        resolve_role_col(prediction_column_map, name, "prediction")
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
            | SmoothBasisSpec::Sphere { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. }
            | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                for feature_col in feature_cols.iter_mut() {
                    *feature_col = resolve_training_index(*feature_col)?;
                }
            }
            SmoothBasisSpec::FactorSmooth { spec } => {
                for feature_col in spec.continuous_cols.iter_mut() {
                    *feature_col = resolve_training_index(*feature_col)?;
                }
                spec.group_col = resolve_training_index(spec.group_col)?;
            }
            SmoothBasisSpec::BySmooth { smooth, by_kind } => {
                match smooth.as_mut() {
                    SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                        *feature_col = resolve_training_index(*feature_col)?
                    }
                    SmoothBasisSpec::ThinPlate { feature_cols, .. }
                    | SmoothBasisSpec::Sphere { feature_cols, .. }
                    | SmoothBasisSpec::Matern { feature_cols, .. }
                    | SmoothBasisSpec::Duchon { feature_cols, .. }
                    | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                        for feature_col in feature_cols.iter_mut() {
                            *feature_col = resolve_training_index(*feature_col)?;
                        }
                    }
                    SmoothBasisSpec::FactorSmooth { spec } => {
                        for feature_col in spec.continuous_cols.iter_mut() {
                            *feature_col = resolve_training_index(*feature_col)?;
                        }
                        spec.group_col = resolve_training_index(spec.group_col)?;
                    }
                    SmoothBasisSpec::BySmooth { .. } => {}
                }
                match by_kind {
                    crate::smooth::ByVarKind::Numeric { feature_col }
                    | crate::smooth::ByVarKind::Factor { feature_col, .. } => {
                        *feature_col = resolve_training_index(*feature_col)?;
                    }
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
    model
        .fit_result
        .clone()
        .ok_or_else(|| "model is missing canonical fit_result payload; refit".to_string())
}

/// Resolve the saved survival location-scale fit result.
///
/// Returns a `UnifiedFitResult` with the fitted inverse-link state
/// re-applied -- matching the CLI's behaviour in
/// `main.rs::saved_survival_location_scale_fit_result`.
pub fn saved_survival_location_scale_fit_result(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.saved_prediction_runtime()?;
    let mut fit = model.fit_result.clone().ok_or_else(|| {
        "saved location-scale survival model missing canonical fit_result; refit".to_string()
    })?;
    let inverse_link = resolve_survival_inverse_link_from_saved(model)?;
    apply_inverse_link_state_to_fit_result(&mut fit, &inverse_link);
    Ok(fit)
}

pub fn apply_inverse_link_state_to_fit_result(
    fit_result: &mut UnifiedFitResult,
    inverse_link: &InverseLink,
) {
    fit_result.fitted_link = match inverse_link {
        InverseLink::LatentCLogLog(state) => FittedLinkState::LatentCLogLog { state: *state },
        InverseLink::Sas(state) => FittedLinkState::Sas {
            state: *state,
            covariance: None,
        },
        InverseLink::BetaLogistic(state) => FittedLinkState::BetaLogistic {
            state: *state,
            covariance: None,
        },
        InverseLink::Mixture(state) => FittedLinkState::Mixture {
            state: state.clone(),
            covariance: None,
        },
        InverseLink::Standard(_) => FittedLinkState::Standard(None),
    };
}

/// Resolve the saved survival inverse-link from saved link metadata and fitted
/// state.
pub fn resolve_survival_inverse_link_from_saved(model: &SavedModel) -> Result<InverseLink, String> {
    let raw = model
        .link
        .as_deref()
        .or(model.survival_distribution.as_deref())
        .ok_or_else(|| "saved survival model is missing link/distribution metadata".to_string())?;
    let name = raw.trim().to_ascii_lowercase();
    if name == "loglog" || name == "cauchit" {
        let component = if name == "loglog" {
            LinkComponent::LogLog
        } else {
            LinkComponent::Cauchit
        };
        return state_fromspec(&MixtureLinkSpec {
            components: vec![component],
            initial_rho: Array1::zeros(0),
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid saved survival {name} link state: {e}"));
    }
    let choice = match parse_link_choice(Some(raw), false) {
        Ok(v) => v,
        Err(_) => {
            let dist = parse_survival_distribution(raw)?;
            return Ok(residual_distribution_inverse_link(dist));
        }
    };
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "saved survival model is missing fit_result".to_string())?;
    let Some(choice) = choice else {
        let dist = parse_survival_distribution(raw)?;
        return Ok(residual_distribution_inverse_link(dist));
    };
    if let Some(components) = choice.mixture_components {
        let rho = match &fit.fitted_link {
            FittedLinkState::Mixture { state, .. } => state.rho.clone(),
            _ => {
                return Err(
                    "saved survival blended-link model missing fitted mixture link parameters"
                        .to_string(),
                );
            }
        };
        return state_fromspec(&MixtureLinkSpec {
            components,
            initial_rho: rho,
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid saved survival blended link state: {e}"));
    }
    match choice.link {
        crate::types::LinkFunction::Sas => {
            let (epsilon, log_delta) = match &fit.fitted_link {
                FittedLinkState::Sas { state, .. } => (state.epsilon, state.log_delta),
                _ => {
                    return Err(
                        "saved survival SAS model missing fitted SAS link parameters".to_string(),
                    );
                }
            };
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .map(InverseLink::Sas)
            .map_err(|e| format!("invalid saved survival SAS state: {e}"))
        }
        crate::types::LinkFunction::BetaLogistic => {
            let (epsilon, delta) = match &fit.fitted_link {
                FittedLinkState::BetaLogistic { state, .. } => {
                    (state.epsilon, state.log_delta)
                }
                _ => {
                    return Err(
                        "saved survival beta-logistic model missing fitted beta-logistic link parameters"
                            .to_string(),
                    )
                }
            };
            state_from_beta_logisticspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: delta,
            })
            .map(InverseLink::BetaLogistic)
            .map_err(|e| format!("invalid saved survival beta-logistic state: {e}"))
        }
        other => Ok(InverseLink::Standard(other)),
    }
}

/// Concatenate referenced 1-D arrays into a single owned `Array1<f64>`.
pub fn concat_array1_refs(parts: &[&Array1<f64>]) -> Array1<f64> {
    let total: usize = parts.iter().map(|part| part.len()).sum();
    let mut out = Array1::<f64>::zeros(total);
    let mut offset = 0usize;
    for part in parts {
        let width = part.len();
        out.slice_mut(s![offset..offset + width]).assign(part);
        offset += width;
    }
    out
}

/// Rebuild the saved baseline-timewiggle entry/exit/derivative design blocks
/// from the saved runtime metadata. Returns `None` when the saved model has no
/// baseline-timewiggle.
pub fn saved_baseline_timewiggle_components(
    eta_entry: &Array1<f64>,
    eta_exit: &Array1<f64>,
    derivative_exit: &Array1<f64>,
    model: &SavedModel,
) -> Result<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    match model.saved_baseline_time_wiggle()? {
        None => Ok(None),
        Some(runtime) => {
            runtime.validate_global_monotonicity()?;
            let SavedBaselineTimeWiggleRuntime {
                knots,
                degree,
                beta,
                ..
            } = runtime;
            let knots = Array1::from_vec(knots);
            let entry = match buildwiggle_block_input_from_knots(
                eta_entry.view(),
                &knots,
                degree,
                2,
                false,
            )?
            .design
            {
                DesignMatrix::Dense(m) => m.to_dense_arc().as_ref().clone(),
                _ => return Err("saved baseline-timewiggle entry design must be dense".to_string()),
            };
            let exit = match buildwiggle_block_input_from_knots(
                eta_exit.view(),
                &knots,
                degree,
                2,
                false,
            )?
            .design
            {
                DesignMatrix::Dense(m) => m.to_dense_arc().as_ref().clone(),
                _ => return Err("saved baseline-timewiggle exit design must be dense".to_string()),
            };
            let betaw = beta;
            if entry.ncols() != betaw.len() || exit.ncols() != betaw.len() {
                return Err(format!(
                    "saved baseline-timewiggle dimension mismatch: coefficients have {} entries but basis has entry={} exit={}",
                    betaw.len(),
                    entry.ncols(),
                    exit.ncols()
                ));
            }
            let derivative = build_survival_timewiggle_derivative_design(
                eta_exit,
                derivative_exit,
                &knots,
                degree,
            )
            .map_err(|e| {
                e.replace(
                    "build baseline-timewiggle",
                    "evaluate saved baseline-timewiggle",
                )
            })?;
            if derivative.ncols() != betaw.len() {
                return Err(format!(
                    "saved baseline-timewiggle derivative dimension mismatch: coefficients have {} entries but derivative basis has {} columns",
                    betaw.len(),
                    derivative.ncols()
                ));
            }
            Ok(Some((entry, exit, derivative)))
        }
    }
}

/// Build the saved survival marginal-slope predictor along with the matching
/// `PredictInput` and a `UnifiedFitResult` repackaged into the layout
/// `BernoulliMarginalSlopePredictor::from_unified` expects.
///
/// This is the single source of truth for assembling the marginal-slope
/// predictor at predict time. The CLI's `gam predict` flow and the
/// library-side `predict_survival` both call into this helper so they share
/// bit-identical eta math (link-deviation + score-warp replay included).
pub fn build_saved_survival_marginal_slope_predictor(
    model: &SavedModel,
    fit_saved: &UnifiedFitResult,
    z_name: &str,
    z: &Array1<f64>,
    cov_design: &DesignMatrix,
    logslope_design: &DesignMatrix,
    time_build: &SurvivalTimeBuildOutput,
    eta_offset_entry: &Array1<f64>,
    eta_offset_exit: &Array1<f64>,
    derivative_offset_exit: &Array1<f64>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<
    (
        BernoulliMarginalSlopePredictor,
        PredictInput,
        UnifiedFitResult,
    ),
    String,
> {
    let saved_runtime = model.saved_prediction_runtime()?;
    if saved_runtime.link_wiggle.is_some() {
        return Err(
            "saved survival marginal-slope model contains legacy linkwiggle metadata; refit with the anchored link-deviation runtime"
                .to_string(),
        );
    }

    let saved_score_runtime = saved_runtime.score_warp;
    let saved_link_runtime = saved_runtime.link_deviation;
    let blocks = &fit_saved.blocks;
    let expected_blocks =
        3 + usize::from(saved_score_runtime.is_some()) + usize::from(saved_link_runtime.is_some());
    if blocks.len() != expected_blocks {
        return Err(format!(
            "saved survival marginal-slope model requires {} blocks [time, marginal, slope{}{}], got {}",
            expected_blocks,
            if saved_score_runtime.is_some() {
                ", score-warp"
            } else {
                ""
            },
            if saved_link_runtime.is_some() {
                ", link-deviation"
            } else {
                ""
            },
            blocks.len(),
        ));
    }

    let beta_time = &blocks[0].beta;
    let beta_marginal = &blocks[1].beta;
    let beta_logslope = &blocks[2].beta;
    if let Some(runtime) = saved_score_runtime.as_ref() {
        let beta = &blocks[3].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "saved survival marginal-slope score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    if let Some(runtime) = saved_link_runtime.as_ref() {
        let idx = 3 + usize::from(saved_score_runtime.is_some());
        let beta = &blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "saved survival marginal-slope link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }

    if beta_marginal.len() != cov_design.ncols() {
        return Err(format!(
            "saved survival marginal-slope marginal coefficient mismatch: beta has {} entries but baseline design has {} columns",
            beta_marginal.len(),
            cov_design.ncols()
        ));
    }
    if beta_logslope.len() != logslope_design.ncols() {
        return Err(format!(
            "saved survival marginal-slope slope coefficient mismatch: beta has {} entries but slope design has {} columns",
            beta_logslope.len(),
            logslope_design.ncols()
        ));
    }

    let p_time_base = time_build.x_exit_time.ncols();
    let saved_timewiggle = saved_runtime.baseline_time_wiggle;
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
    let q_entry_base = time_build.x_entry_time.dot(&beta_time_base)
        + cov_design.dot(beta_marginal)
        + eta_offset_entry
        + primary_offset;
    let q_exit_base = time_build.x_exit_time.dot(&beta_time_base)
        + cov_design.dot(beta_marginal)
        + eta_offset_exit
        + primary_offset;
    let qd_exit_base = time_build.x_derivative_time.dot(&beta_time_base) + derivative_offset_exit;

    let mut q_design_parts = vec![time_build.x_exit_time.clone()];
    if saved_timewiggle.is_some() {
        let (_, exit_w, _) = saved_baseline_timewiggle_components(
            &q_entry_base,
            &q_exit_base,
            &qd_exit_base,
            model,
        )?
        .ok_or_else(|| {
            "saved survival marginal-slope model is missing baseline-timewiggle runtime metadata"
                .to_string()
        })?;
        if exit_w.ncols() != p_timewiggle {
            return Err(format!(
                "saved survival marginal-slope timewiggle design mismatch: rebuilt {} columns but runtime expects {}",
                exit_w.ncols(),
                p_timewiggle
            ));
        }
        q_design_parts.push(DesignMatrix::from(exit_w));
    }
    q_design_parts.push(cov_design.clone());
    let q_design = DesignMatrix::hstack(q_design_parts)?;

    let combined_q_beta = concat_array1_refs(&[beta_time, beta_marginal]);
    let combined_q_lambdas = concat_array1_refs(&[&blocks[0].lambdas, &blocks[1].lambdas]);
    let mut predictor_blocks = Vec::with_capacity(
        2 + usize::from(saved_score_runtime.is_some()) + usize::from(saved_link_runtime.is_some()),
    );
    predictor_blocks.push(FittedBlock {
        beta: combined_q_beta.clone(),
        role: BlockRole::Mean,
        edf: blocks[0].edf + blocks[1].edf,
        lambdas: combined_q_lambdas,
    });
    predictor_blocks.push(FittedBlock {
        beta: beta_logslope.clone(),
        role: BlockRole::Scale,
        edf: blocks[2].edf,
        lambdas: blocks[2].lambdas.clone(),
    });
    if saved_score_runtime.is_some() {
        let mut block = blocks[3].clone();
        block.role = BlockRole::Mean;
        predictor_blocks.push(block);
    }
    if saved_link_runtime.is_some() {
        let idx = 3 + usize::from(saved_score_runtime.is_some());
        let mut block = blocks[idx].clone();
        block.role = BlockRole::LinkWiggle;
        predictor_blocks.push(block);
    }

    let mut predictor_fit = fit_saved.clone();
    predictor_fit.blocks = predictor_blocks;
    predictor_fit.beta = concat_array1_refs(
        &predictor_fit
            .blocks
            .iter()
            .map(|block| &block.beta)
            .collect::<Vec<_>>(),
    );
    predictor_fit.block_states.clear();

    let predictor = BernoulliMarginalSlopePredictor::from_unified(
        &predictor_fit,
        z_name.to_string(),
        model.latent_z_normalization.ok_or_else(|| {
            "saved survival marginal-slope model missing latent_z_normalization".to_string()
        })?,
        model.latent_measure.clone().ok_or_else(|| {
            "saved survival marginal-slope model missing latent_measure".to_string()
        })?,
        0.0,
        model.logslope_baseline.ok_or_else(|| {
            "saved survival marginal-slope model missing logslope_baseline".to_string()
        })?,
        model
            .resolved_inverse_link()?
            .unwrap_or(InverseLink::Standard(LinkFunction::Probit)),
        model
            .family_state
            .frailty()
            .cloned()
            .unwrap_or(FrailtySpec::None),
        saved_score_runtime,
        saved_link_runtime,
        model.latent_z_rank_int_calibration.clone(),
    )?;

    let pred_input = PredictInput {
        design: q_design,
        offset: eta_offset_exit + primary_offset,
        design_noise: Some(logslope_design.clone()),
        offset_noise: Some(noise_offset.clone()),
        auxiliary_scalar: Some(z.clone()),
        auxiliary_matrix: None,
    };

    Ok((predictor, pred_input, predictor_fit))
}

/// Extract the fixed Gaussian-shift sigma (if any) from a frailty spec. Used
/// to compute the rigid-path probit frailty scale that mirrors the predictor's
/// internal `probit_frailty_scale()`.
pub fn gaussian_frailty_sigma_from_frailty(frailty: Option<&FrailtySpec>) -> Option<f64> {
    match frailty {
        Some(FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        }) => Some(*sigma),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::{normal_cdf, normal_pdf};

    #[test]
    fn probit_survival_hazard_uses_density_over_survival() {
        let eta = 2.0;
        let eta_t = 0.3;

        let (cum, hazard) =
            probit_survival_hazard_components(eta, eta_t).expect("valid components");

        let survival = normal_cdf(-eta);
        let expected_cum = -survival.ln();
        let expected_hazard = normal_pdf(eta) * eta_t / survival;
        assert!((cum - expected_cum).abs() <= 1e-14);
        assert!((hazard - expected_hazard).abs() <= 1e-14);
    }

    #[test]
    fn probit_survival_hazard_stays_finite_in_right_tail() {
        let eta = 40.0;
        let eta_t = 9.694_340_360_912_401e-5;

        let event_density =
            (-0.5_f64 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt() * eta_t;
        assert_eq!(event_density, 0.0);

        let (cum, hazard) =
            probit_survival_hazard_components(eta, eta_t).expect("valid tail components");
        assert!(cum > 800.0, "right-tail cumulative hazard was {cum}");
        assert!(
            (3.87e-3..3.89e-3).contains(&hazard),
            "right-tail hazard was {hazard}"
        );
    }

    #[test]
    fn probit_survival_hazard_rejects_nonpositive_time_derivative() {
        let err = probit_survival_hazard_components(1.0, 0.0)
            .expect_err("zero derivative should be invalid");
        assert!(err.contains("invalid survival index derivative"));
    }

    #[test]
    fn royston_parmar_hazard_is_cumulative_hazard_derivative() {
        let eta = 2.0_f64.ln();
        let eta_t = 0.25;

        let (cum, hazard) =
            royston_parmar_survival_hazard_components(eta, eta_t).expect("valid components");

        assert!((cum - 2.0).abs() <= 1e-14);
        assert!((hazard - 0.5).abs() <= 1e-14);
        assert_ne!(hazard, cum);
    }

    #[test]
    fn royston_parmar_hazard_rejects_nonpositive_log_hazard_derivative() {
        let err = royston_parmar_survival_hazard_components(0.0, 0.0)
            .expect_err("zero derivative should be invalid");
        assert!(err.contains("invalid log-cumulative-hazard derivative"));
    }

    #[test]
    fn royston_parmar_hazard_propagates_saturation_as_infinity() {
        // η = log Λ(t); a saturated RP fit can drive η well past the
        // exp(709.78)≈f64::MAX boundary in the right tail. The math is
        // S(t)→0, h(t)→∞; the helper must not reject this regime, because the
        // inner solver has already accepted the underlying fit.
        let eta = 1000.0_f64;
        let eta_t = 0.5_f64;
        assert!(eta.exp().is_infinite(), "test premise: exp(1000) overflows");

        let (cum, hazard) = royston_parmar_survival_hazard_components(eta, eta_t)
            .expect("saturated RP fit must yield a result, not an error");
        assert!(cum.is_infinite() && cum > 0.0, "expected +∞ cum, got {cum}");
        assert!(hazard.is_infinite() && hazard > 0.0, "expected +∞ hazard, got {hazard}");

        // Consumer materializes survival via exp(-cum).clamp(0,1).
        let survival = (-cum).exp().clamp(0.0, 1.0);
        assert_eq!(survival, 0.0, "saturated cum_hazard must give survival 0");
    }

    #[test]
    fn royston_parmar_hazard_rejects_nan_eta() {
        let err = royston_parmar_survival_hazard_components(f64::NAN, 0.5)
            .expect_err("NaN eta should be invalid");
        assert!(err.contains("invalid log-cumulative-hazard derivative"));
    }

    #[test]
    fn probit_survival_hazard_left_tail_collapses_to_zero() {
        // η→-∞ mirror of the right-tail test: survival → 1, hazard → 0.
        // Asymptote: Mills(η) = φ(η)/Φ(-η) → 0 as η → -∞ (φ underflows,
        // Φ(-η) → 1).  No error, no NaN, no spurious negativity.
        let eta = -40.0_f64;
        let eta_t = 1.5_f64;

        let (cum, hazard) = probit_survival_hazard_components(eta, eta_t)
            .expect("left tail must remain valid");
        assert!(cum >= 0.0 && cum < 1e-300, "left-tail cum should be ~0, got {cum}");
        assert_eq!(hazard, 0.0, "left-tail hazard should underflow to 0, got {hazard}");
    }

    #[test]
    fn location_scale_logit_hazard_is_failure_slope_over_survival() {
        let eta = 0.7;
        let eta_t = 0.4;

        let hazard = location_scale_hazard_component(
            eta,
            eta_t,
            &InverseLink::Standard(LinkFunction::Logit),
        )
        .expect("valid logit hazard");

        let failure = 1.0 / (1.0 + (-eta).exp());
        assert!((hazard - failure * eta_t).abs() <= 1e-14);
    }

    #[test]
    fn location_scale_cloglog_hazard_matches_log_cumulative_hazard_derivative() {
        let eta = 1.5;
        let eta_t = 0.2;

        let hazard = location_scale_hazard_component(
            eta,
            eta_t,
            &InverseLink::Standard(LinkFunction::CLogLog),
        )
        .expect("valid cloglog hazard");

        assert!((hazard - eta.exp() * eta_t).abs() <= 1e-14);
    }
}
