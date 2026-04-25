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

use crate::families::lognormal_kernel::{FrailtySpec, ProbitFrailtyScale};
use crate::families::scale_design::ScaleDeviationTransform;
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalLikelihoodMode, SurvivalTimeBuildOutput,
    build_survival_baseline_offsets, build_survival_marginal_slope_baseline_offsets,
    build_survival_time_basis, build_survival_timewiggle_derivative_design,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    normalize_survival_time_pair, parse_survival_baseline_config, parse_survival_distribution,
    parse_survival_likelihood_mode, require_structural_survival_time_basis,
    resolved_survival_time_basis_config_from_build, survival_likelihood_modename,
};
use crate::families::survival_location_scale::residual_distribution_inverse_link;
use crate::gamlss::buildwiggle_block_input_from_knots;
use crate::inference::formula_dsl::parse_link_choice;
use crate::inference::model::{
    FittedFamily, FittedModel as SavedModel, SavedBaselineTimeWiggleRuntime,
    load_survival_time_basis_config_from_model, survival_baseline_config_from_model,
};
use crate::inference::predict::{
    BernoulliMarginalSlopePredictor, PredictInput, PredictableModel, predict_gam,
};
use crate::linalg::matrix::DesignMatrix;
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::probability::normal_cdf;
use crate::solver::estimate::{BlockRole, FittedBlock, FittedLinkState, UnifiedFitResult};
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
pub fn predict_survival(req: SurvivalPredictRequest<'_>) -> Result<SurvivalPredictResult, String> {
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
    // helpers for now. Location-scale survival requires a link resolver
    // that is still CLI-specific (parses CLI args like `--sas-init`);
    // expose through the library as soon as a library-first link
    // resolver ships.
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent
            | SurvivalLikelihoodMode::LatentBinary
            | SurvivalLikelihoodMode::LocationScale
    ) {
        return Err(format!(
            "survival prediction via predict_survival does not support likelihood_mode={} yet; \
             use the CLI predict command",
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
            build_baseline_offsets_by_mode(
                &age_entry,
                &age_exit,
                &baseline_cfg,
                saved_likelihood_mode,
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
            let mut row_time =
                build_survival_time_basis(&single_entry, &single_exit, time_cfg.clone(), None)?;
            if let Some(anchor_row) = time_anchor_row_cached.as_ref() {
                center_survival_time_designs_at_anchor(
                    &mut row_time.x_entry_time,
                    &mut row_time.x_exit_time,
                    anchor_row,
                )?;
            }
            let (_r_eta_entry, r_eta_exit, r_deriv_exit) = build_baseline_offsets_by_mode(
                &single_entry,
                &single_exit,
                &baseline_cfg,
                saved_likelihood_mode,
            )?;

            let cov_row = cov_design.design.as_dense_cow().row(i).to_owned();

            let (eta_t, cum_t, haz_t) = match saved_likelihood_mode {
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
                    )?
                }
                SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                    evaluate_rp_row(
                        model,
                        &row_time,
                        &cov_row,
                        r_eta_exit[0] + primary_offset[i],
                    )?
                }
                SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
                | SurvivalLikelihoodMode::LocationScale => {
                    return Err(
                        "unreachable: unsupported likelihood_mode filtered earlier".to_string()
                    );
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
        }
        if !per_row_eval {
            // Track the linear predictor at each row's own exit time.
            let t_exit = age_exit[i];
            let t_entry = age_entry[i].min(t_exit);
            let single_entry = Array1::from_elem(1, t_entry);
            let single_exit = Array1::from_elem(1, t_exit);
            let mut row_time =
                build_survival_time_basis(&single_entry, &single_exit, time_cfg.clone(), None)?;
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
                    )?
                }
                SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                    evaluate_rp_row(
                        model,
                        &row_time,
                        &cov_row,
                        r_eta_exit[0] + primary_offset[i],
                    )?
                }
                SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
                | SurvivalLikelihoodMode::LocationScale => {
                    return Err(
                        "unreachable: unsupported likelihood_mode filtered earlier".to_string()
                    );
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
    /// Dense covariate design (n × p_marginal). Stored once so per-row eta
    /// assembly can copy the row without re-building the design.
    cov_design_dense: Array2<f64>,
    /// Dense logslope design (n × p_logslope), built once from the saved
    /// `resolved_termspec_logslope`.
    logslope_design_dense: Array2<f64>,
    /// Per-row covariate eta = `cov_design[i] · beta_marginal`. Used to
    /// pre-compute `q_exit_base`.
    cov_eta: Array1<f64>,
    /// Per-row logslope eta = `logslope_design[i] · beta_logslope +
    /// baseline_logslope + noise_offset[i]`. Used to compute the rigid-path
    /// hazard scale `c = sqrt(1 + (s · b)^2)`.
    slope_eta: Array1<f64>,
    /// Per-row latent z (raw, un-normalized — the predictor's
    /// `latent_z_normalization` is applied internally).
    z_raw: Array1<f64>,
    /// Per-row noise offset, mirroring the `pred_input.offset_noise` slice
    /// used by the CLI.
    noise_offset: Array1<f64>,
    /// Probit frailty scale `s = 1/sqrt(1 + sigma_frailty^2)`. Used for the
    /// rigid-path hazard derivative.
    probit_scale: f64,
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
    let z_col = *col_map
        .get(z_name)
        .ok_or_else(|| format!("prediction data is missing z column '{}'", z_name))?;
    let z_raw = data.column(z_col).to_owned();

    let logslopespec = resolve_termspec_for_prediction(
        &model.resolved_termspec_logslope.as_ref().cloned(),
        training_headers,
        col_map,
        "resolved_termspec_logslope",
    )?;
    let logslope_design = build_term_collection_design(data, &logslopespec)
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
    let beta_logslope = blocks[2].beta.clone();
    let saved_runtime = model.saved_prediction_runtime()?;
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();

    let cov_design_dense = cov_design.to_dense();
    let logslope_design_dense = logslope_design.design.to_dense();

    // Per-row precomputed accumulations. cov_eta and slope_eta are time-
    // independent so doing them here avoids `O(n × T)` re-multiplications
    // inside the per-cell loop.
    let cov_eta = cov_design.dot(&beta_marginal);
    let baseline_logslope = model.logslope_baseline.ok_or_else(|| {
        "saved survival marginal-slope model missing logslope_baseline".to_string()
    })?;
    let slope_eta = logslope_design.design.dot(&beta_logslope) + noise_offset + baseline_logslope;

    let frailty = model.family_state.frailty();
    let sigma = gaussian_frailty_sigma_from_frailty(frailty);
    let probit_scale = ProbitFrailtyScale::new(sigma.unwrap_or(0.0)).s;

    Ok(MarginalSlopePredictContext {
        predictor,
        beta_time,
        beta_marginal,
        saved_timewiggle,
        cov_design_dense,
        logslope_design_dense,
        cov_eta,
        slope_eta,
        z_raw,
        noise_offset: noise_offset.clone(),
        probit_scale,
    })
}

/// Evaluate one (row, t) cell for the saved survival marginal-slope kernel.
///
/// Calls the saved [`BernoulliMarginalSlopePredictor`] (built once and held in
/// `ctx`) to obtain `eta` — this is the same code path the CLI's `gam predict`
/// uses, so the link-deviation and score-warp blocks are replayed at predict
/// time. The hazard time-derivative `c · qd` is the rigid-path probit-frailty
/// correction; in flex mode (score-warp / link-deviation active) it is the
/// leading-order rigid approximation, since the exact ∂eta/∂t requires the
/// IFT pull-back through the per-row implicit-function intercept (see
/// `compute_survival_timepoint_exact` in `survival_marginal_slope.rs`).
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

    let (_q_with_wiggle, qd_with_wiggle, exit_wiggle_design) =
        if let Some(runtime) = ctx.saved_timewiggle.as_ref() {
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
                q_exit_base + exit_design.dot(&beta_w)[0],
                qd_exit_base + derivative_design.dot(&beta_w)[0],
                Some(exit_design),
            )
        } else {
            (q_exit_base, qd_exit_base, None)
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
        q_design_full
            .slice_mut(s![.., p_time_base + p_timewiggle..])
            .row_mut(0)
            .assign(&ctx.cov_design_dense.row(row_index));
    }

    // Logslope design row + offset chosen so that the predictor's logslope_eta
    // equals our precomputed `slope_eta[row]`.  The predictor computes:
    //   logslope_eta = design_noise · beta_logslope + baseline_logslope
    //                  + offset_noise.
    // We feed the actual saved logslope row + the row's noise offset, matching
    // exactly the CLI's `pred_input.design_noise` / `offset_noise` slice.
    let logslope_row = ctx.logslope_design_dense.row(row_index).to_owned();
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
    };

    let pred = ctx
        .predictor
        .predict_plugin_response(&pred_input)
        .map_err(|e| format!("saved survival marginal-slope predictor eta failed: {e}"))?;
    let eta = pred.eta[0];

    // Rigid-path time derivative: d(eta)/dt = c · qd, with
    //   c = sqrt(1 + (s · b)^2),  s = 1/sqrt(1 + sigma_frailty^2),
    //   b = slope_eta[row].
    // In flex mode (score-warp / link-deviation active) this is the leading-order
    // rigid approximation; the exact ∂eta/∂t additionally picks up the implicit-
    // function pull-back (∂eta/∂a)·(∂a/∂q)·qd. Flagged for review — see the
    // training kernel's `compute_survival_timepoint_exact` for the full IFT path.
    let s_b = ctx.probit_scale * ctx.slope_eta[row_index];
    let c = (1.0 + s_b * s_b).sqrt();
    let eta_derivative = c * qd_with_wiggle;

    let surv = normal_cdf(-eta).clamp(1e-300, 1.0);
    let cum = -surv.ln();
    let phi_eta = (-0.5f64 * eta * eta).exp() / (2.0f64 * std::f64::consts::PI).sqrt();
    let haz = phi_eta * eta_derivative;
    if !(eta_derivative.is_finite() && eta_derivative > 0.0 && haz.is_finite() && haz > 0.0) {
        return Err(format!(
            "saved survival marginal-slope prediction produced non-positive time derivative: eta_t={eta_derivative}, hazard={haz}"
        ));
    }
    Ok((eta, cum, haz))
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
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle() {
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
///
/// Returns a `UnifiedFitResult` with the fitted inverse-link state
/// re-applied -- matching the CLI's behaviour in
/// `main.rs::saved_survival_location_scale_fit_result`.
pub fn saved_survival_location_scale_fit_result(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.saved_prediction_runtime()?;
    let mut fit = model.fit_result.clone().ok_or_else(|| {
        "saved location-scale survival model missing canonical fit_result; refit with current CLI"
            .to_string()
    })?;
    let inverse_link = resolve_survival_inverse_link_from_saved(model)?;
    apply_inverse_link_state_to_fit_result(&mut fit, &inverse_link);
    Ok(fit)
}

fn apply_inverse_link_state_to_fit_result(
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
    )?;

    let pred_input = PredictInput {
        design: q_design,
        offset: eta_offset_exit + primary_offset,
        design_noise: Some(logslope_design.clone()),
        offset_noise: Some(noise_offset.clone()),
        auxiliary_scalar: Some(z.clone()),
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
