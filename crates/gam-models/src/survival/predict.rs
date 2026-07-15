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

use crate::fit_orchestration::prepare_survival_time_stack;
use crate::inference::model::{
    FittedFamily, FittedModel as SavedModel, SavedBaselineTimeWiggleRuntime,
    load_survival_time_basis_config_from_model, survival_baseline_config_from_model,
};
use crate::inference::predict_io::{BernoulliMarginalSlopePredictor, PredictInput};
use crate::model_types::{BlockRole, FittedBlock, FittedLinkState, UnifiedFitResult};
use crate::probability::signed_probit_logcdf_and_mills_ratio;
use crate::survival::construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    SurvivalTimeBuildOutput, add_survival_time_derivative_guard_offset, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, build_survival_timewiggle_derivative_design,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    normalize_survival_time_pair, parse_survival_likelihood_mode,
    require_structural_survival_time_basis, resolved_survival_time_basis_config_from_build,
    survival_derivative_guard_for_likelihood, survival_likelihood_modename,
};
use crate::survival::latent::fixed_latent_hazard_frailty;
use crate::survival::lognormal_kernel::FrailtySpec;
use crate::survival::{CompetingRisksCifResult, assemble_competing_risks_cif_from_endpoints};
use crate::wiggle::buildwiggle_block_input_from_knots;
use gam_linalg::matrix::DesignMatrix;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::mixture_link::inverse_link_jet_for_inverse_link;
use gam_terms::smooth::TermCollectionSpec;
use gam_terms::smooth::build_term_collection_design;
use gam_terms::term_builder::resolve_role_col;

/// Resolved survival entry/exit column indices for a saved survival model.
///
/// `entry_col` is `None` when the model was trained with the right-censored
/// shorthand `Surv(time, event)`; callers synthesize a zero entry time per
/// row in that case via [`SurvivalTimeColumns::row_entry_time`]. Mirrors
/// the CLI predict path so every site that consumes saved survival
/// metadata applies the same fallback contract.
pub struct SurvivalTimeColumns {
    pub entry_col: Option<usize>,
    pub exit_col: usize,
}

impl SurvivalTimeColumns {
    /// Entry time for row `i`, defaulting to `0.0` when the saved model has
    /// no `survival_entry` column (right-censored shorthand).
    #[inline]
    pub fn row_entry_time(&self, data: ArrayView2<'_, f64>, i: usize) -> f64 {
        self.entry_col.map_or(0.0, |idx| data[[i, idx]])
    }
}

/// Resolve saved survival entry/exit column names against the runtime
/// `col_map`, treating an absent `survival_entry` as the right-censored
/// shorthand (entry times synthesized as zero downstream).
pub fn resolve_saved_survival_time_columns(
    model: &SavedModel,
    col_map: &HashMap<String, usize>,
) -> Result<SurvivalTimeColumns, String> {
    let entry_col: Option<usize> = model
        .survival_entry
        .as_deref()
        .map(|name| resolve_role_col(col_map, name, "entry"))
        .transpose()?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let exit_col = resolve_role_col(col_map, exitname, "exit")?;
    Ok(SurvivalTimeColumns {
        entry_col,
        exit_col,
    })
}

/// Smallest positive survival probability we admit before taking
/// `-ln(S)` for the cumulative hazard. Using `f64::MIN_POSITIVE` (≈ 2.2e-308)
/// would let `-ln(S)` reach ~709 and risk downstream `exp(-cum)` underflow
/// patterns that don't round-trip through `clamp(0,1)`. `1e-300` keeps
/// `-ln(S) ≤ ~691` and matches the location-scale predict contract upstream.
const SURVIVAL_PROB_MIN_FOR_LOG: f64 = 1e-300;

/// Typed errors emitted by the survival prediction pipeline.
///
/// Each variant carries a pre-formatted `reason` string so `Display` is
/// byte-equivalent to the original `format!(...)` outputs the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind without dragging the string apart.
#[derive(Debug, Clone)]
pub enum SurvivalPredictError {
    /// Request-level input did not satisfy the predict contract: bad offset
    /// lengths, malformed time grids, empty grids, non-finite times.
    InvalidInput { reason: String },
    /// The saved model is missing metadata required to drive the prediction
    /// (anchor, link/distribution tags, likelihood-mode marker, etc.) or
    /// carries legacy metadata that the current runtime refuses to consume.
    MissingFitMetadata { reason: String },
    /// Saved coefficient blocks, design columns, or baseline-timewiggle
    /// runtime dimensions disagree with the rebuilt prediction designs.
    IncompatibleSchema { reason: String },
    /// The requested combination of saved-model mode and predict-time
    /// options is not implemented in this library entry point yet (e.g.
    /// uncertainty for a plug-in non-location-scale prediction or latent
    /// window prediction).
    UnsupportedConfiguration { reason: String },
    /// Posterior-mean prediction requires the fitted joint coefficient
    /// covariance in exactly the same block-concatenated coordinate system as
    /// the saved coefficient vector. Missing, malformed, or dimensionally
    /// incompatible covariance is an error; it must never change the requested
    /// estimand by falling back to a plug-in surface.
    PosteriorCovariance { reason: String },
    /// A numerical step (hazard / derivative / survival reconstruction)
    /// produced a non-finite or out-of-domain value that downstream code
    /// cannot consume.
    NumericalFailure { reason: String },
    /// Saved-model validation failed below this prediction layer; the model
    /// source error keeps its own payload/schema category.
    ModelPayload {
        context: &'static str,
        source: crate::inference::model::FittedModelError,
    },
}

impl std::fmt::Display for SurvivalPredictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurvivalPredictError::InvalidInput { reason }
            | SurvivalPredictError::MissingFitMetadata { reason }
            | SurvivalPredictError::IncompatibleSchema { reason }
            | SurvivalPredictError::UnsupportedConfiguration { reason }
            | SurvivalPredictError::PosteriorCovariance { reason }
            | SurvivalPredictError::NumericalFailure { reason } => f.write_str(reason),
            SurvivalPredictError::ModelPayload { context, source } => {
                write!(f, "{context}: {source}")
            }
        }
    }
}

impl std::error::Error for SurvivalPredictError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SurvivalPredictError::ModelPayload { source, .. } => Some(source),
            SurvivalPredictError::InvalidInput { .. }
            | SurvivalPredictError::MissingFitMetadata { .. }
            | SurvivalPredictError::IncompatibleSchema { .. }
            | SurvivalPredictError::UnsupportedConfiguration { .. }
            | SurvivalPredictError::PosteriorCovariance { .. }
            | SurvivalPredictError::NumericalFailure { .. } => None,
        }
    }
}

impl From<SurvivalPredictError> for String {
    fn from(err: SurvivalPredictError) -> String {
        err.to_string()
    }
}

impl From<String> for SurvivalPredictError {
    /// Inbound conversion from the many `Result<_, String>` helpers this
    /// module still calls into (basis builders, fit deserializers,
    /// term-collection assembly). The text is preserved verbatim; we only
    /// pick a category so external messages flow through `?` without
    /// per-callsite `.map_err`.
    fn from(reason: String) -> SurvivalPredictError {
        SurvivalPredictError::InvalidInput { reason }
    }
}

impl From<gam_data::DataError> for SurvivalPredictError {
    /// Column-resolution failures from `resolve_role_col` / `resolve_col`
    /// land as `InvalidInput` since they reflect a mismatch between the
    /// caller-supplied predict frame and the model's expected schema.
    fn from(err: gam_data::DataError) -> SurvivalPredictError {
        SurvivalPredictError::InvalidInput {
            reason: err.to_string(),
        }
    }
}

/// Statistical target returned by the survival prediction API.
///
/// Survival, cumulative hazard, and hazard are nonlinear in the fitted
/// coefficients, so evaluating them at the posterior centre is not the same
/// estimand as integrating the coefficient posterior. The default is the
/// posterior-predictive surface. Callers that specifically need the historical
/// coefficient-mode surface must opt in to [`Self::Plugin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurvivalPredictEstimand {
    #[default]
    PosteriorMean,
    Plugin,
}

/// Exact coefficient-covariance definition used for competing-risks
/// uncertainty.
///
/// Selection is strict: requesting [`Self::SmoothingCorrected`] requires a
/// saved smoothing-corrected covariance and never substitutes the conditional
/// covariance.  The resolved value is carried on
/// [`CompetingRisksPredictResult`] so public frontends report what they used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvivalPredictionCovarianceMode {
    Conditional,
    SmoothingCorrected,
}

impl SurvivalPredictionCovarianceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conditional => "conditional",
            Self::SmoothingCorrected => "smoothing-corrected",
        }
    }
}

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
    /// When true, the result also carries posterior standard errors for the
    /// reported surfaces and linear predictors. Posterior-mean prediction uses
    /// the same joint coefficient quadrature as the point estimand; explicit
    /// plug-in single-event prediction retains its model-specific uncertainty
    /// implementation.
    pub with_uncertainty: bool,
    /// Response-scale estimand. [`SurvivalPredictEstimand::PosteriorMean`] is
    /// the default; plug-in prediction is available only as an explicit opt-in.
    pub estimand: SurvivalPredictEstimand,
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
    /// Exact coefficient-covariance definition behind `survival_se`/`eta_se`.
    /// Result-owned provenance (#2296): presenters must serialize this, never
    /// the requested mode. `None` iff the result carries no uncertainty
    /// surfaces.
    pub covariance_source: Option<SurvivalPredictionCovarianceMode>,
}

/// Exact plug-in survival probability over each requested latent-hazard window.
///
/// The latent survival and latent-binary fits share the same persisted hazard
/// law; they differ only in which response functional is presented to users.
/// This result deliberately exposes the common probability
/// `P(T > exit | T > entry, x)`. Observation generation can therefore sample
/// the fitted window event indicator as Bernoulli with probability
/// `1 - window_survival` without reconstructing a censoring or inspection law.
pub struct LatentWindowSurvivalResult {
    pub window_survival: Array1<f64>,
    pub likelihood_mode: SurvivalLikelihoodMode,
}

/// Evaluate the saved latent hazard-multiplier law over the rows' own windows.
///
/// This is the library authority for both `latent` and `latent-binary` saved
/// models. It replays the persisted covariate design, anchored time basis,
/// loaded/unloaded baseline decomposition, fitted mean/time coefficients, and
/// fixed lognormal hazard multiplier. No response column, refit, or surrogate
/// family participates in the calculation.
pub fn predict_latent_window_survival(
    req: SurvivalPredictRequest<'_>,
) -> Result<LatentWindowSurvivalResult, SurvivalPredictError> {
    let SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset,
        noise_offset,
        time_grid,
        with_uncertainty,
        estimand,
    } = req;
    if time_grid.is_some() {
        return Err(SurvivalPredictError::InvalidInput {
            reason: "latent-window prediction consumes each row's saved entry/exit columns; an independent time_grid is not a window law".to_string(),
        });
    }
    if with_uncertainty || estimand != SurvivalPredictEstimand::Plugin {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "latent-window observation generation requires the fitted plug-in hazard law; posterior coefficient integration is a different sampling target".to_string(),
        });
    }

    let likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if !matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "latent-window prediction requires latent or latent-binary likelihood mode, got {}",
                survival_likelihood_modename(likelihood_mode)
            ),
        });
    }
    if model.has_baseline_time_wiggle() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason:
                "saved latent survival/binary model contains forbidden baseline timewiggle metadata"
                    .to_string(),
        });
    }

    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(SurvivalPredictError::InvalidInput {
            reason: format!(
                "latent-window offset length mismatch: rows={n}, primary={}, noise={}",
                primary_offset.len(),
                noise_offset.len()
            ),
        });
    }
    if noise_offset.iter().any(|value| *value != 0.0) {
        return Err(SurvivalPredictError::InvalidInput {
            reason: "latent-window survival has no secondary offset coordinate".to_string(),
        });
    }

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let covariate_input = clipped.as_ref().map_or(data, |array| array.view());
    let covariate_design = build_term_collection_design(covariate_input, &termspec)
        .map_err(|error| format!("failed to build latent-window covariate design: {error}"))?;
    let effective_primary_offset = covariate_design
        .compose_offset(primary_offset.view(), "latent-window covariate block")
        .map_err(|error| error.to_string())?;

    let time_columns = resolve_saved_survival_time_columns(model, col_map)?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (entry, exit) = normalize_survival_time_pair(
            time_columns.row_entry_time(data, row),
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }

    let time_config = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_config, None)?;
    let resolved_time_config = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor =
        model
            .survival_time_anchor
            .ok_or_else(|| SurvivalPredictError::MissingFitMetadata {
                reason: "saved latent-window model is missing survival_time_anchor".to_string(),
            })?;
    let anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_config)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &anchor_row,
    )?;
    require_structural_survival_time_basis(
        &time_build.basisname,
        "saved latent-window prediction",
    )?;

    let frailty =
        model
            .family_state
            .frailty()
            .ok_or_else(|| SurvivalPredictError::MissingFitMetadata {
                reason: "saved latent-window model is missing its hazard-multiplier frailty"
                    .to_string(),
            })?;
    let (sigma, loading) = fixed_latent_hazard_frailty(frailty, "saved latent-window prediction")
        .map_err(|reason| SurvivalPredictError::MissingFitMetadata { reason })?;
    let baseline_config = saved_survival_runtime_baseline_config(model)?;
    let prepared = prepare_survival_time_stack(
        &age_entry,
        &age_exit,
        &baseline_config,
        likelihood_mode,
        None,
        time_anchor,
        survival_derivative_guard_for_likelihood(likelihood_mode),
        &time_build,
        None,
        Some(loading),
    )?;

    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mean_block = fit.block_by_role(BlockRole::Mean).ok_or_else(|| {
        SurvivalPredictError::MissingFitMetadata {
            reason: "saved latent-window model is missing its mean coefficient block".to_string(),
        }
    })?;
    let time_block = fit.block_by_role(BlockRole::Time).ok_or_else(|| {
        SurvivalPredictError::MissingFitMetadata {
            reason: "saved latent-window model is missing its time coefficient block".to_string(),
        }
    })?;
    if mean_block.beta.len() != covariate_design.design.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "latent-window mean/design mismatch: beta has {} coefficients but design has {} columns",
                mean_block.beta.len(),
                covariate_design.design.ncols()
            ),
        });
    }
    if time_block.beta.len() != prepared.time_design_exit.ncols() {
        let hint = stale_weibull_time_basis_hint(
            &time_build.basisname,
            time_block.beta.len() == prepared.time_design_exit.ncols() + 1,
        );
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "latent-window time/design mismatch: beta has {} coefficients but design has {} columns{hint}",
                time_block.beta.len(),
                prepared.time_design_exit.ncols()
            ),
        });
    }

    let eta = covariate_design.design.dot(&mean_block.beta) + &effective_primary_offset;
    let q_entry = prepared.time_design_entry.dot(&time_block.beta) + &prepared.eta_offset_entry;
    let q_exit = prepared.time_design_exit.dot(&time_block.beta) + &prepared.eta_offset_exit;
    let quadrature = gam_solve::quadrature::QuadratureContext::new();
    let mut window_survival = Array1::<f64>::zeros(n);
    for row in 0..n {
        let latent_row = crate::survival::lognormal_kernel::LatentSurvivalRow::right_censored(
            q_entry[row].exp(),
            q_exit[row].exp(),
            prepared.unloaded_mass_entry[row],
            prepared.unloaded_mass_exit[row],
        );
        let jet = crate::survival::lognormal_kernel::LatentSurvivalRowJet::evaluate(
            &quadrature,
            &latent_row,
            eta[row],
            sigma,
        )
        .map_err(|error| SurvivalPredictError::NumericalFailure {
            reason: format!("latent-window row {row} evaluation failed: {error}"),
        })?;
        let survival = jet.log_lik.exp();
        if !(survival.is_finite() && (0.0..=1.0).contains(&survival)) {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "latent-window row {row} produced invalid conditional survival {survival}"
                ),
            });
        }
        window_survival[row] = survival;
    }

    Ok(LatentWindowSurvivalResult {
        window_survival,
        likelihood_mode,
    })
}

fn select_survival_prediction_covariance<'a>(
    conditional: Option<&'a Array2<f64>>,
    smoothing_corrected: Option<&'a Array2<f64>>,
    mode: SurvivalPredictionCovarianceMode,
) -> Result<&'a Array2<f64>, SurvivalPredictError> {
    match mode {
        SurvivalPredictionCovarianceMode::Conditional => {
            conditional.ok_or_else(|| SurvivalPredictError::PosteriorCovariance {
                reason: "fit result does not contain conditional covariance".to_string(),
            })
        }
        SurvivalPredictionCovarianceMode::SmoothingCorrected => {
            smoothing_corrected.ok_or_else(|| SurvivalPredictError::PosteriorCovariance {
                reason: "fit result does not contain smoothing-corrected covariance".to_string(),
            })
        }
    }
}

/// Exact selected posterior covariance projected onto coefficients that can
/// affect a survival prediction. The absorbed stage-one influence block in a
/// marginal-slope fit is persisted for inference provenance but deliberately
/// drops out of deployment, so its trailing coordinates are not quadrature
/// dimensions.
fn survival_prediction_posterior_factor(
    model: &SavedModel,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<(Array1<f64>, Array2<f64>), SurvivalPredictError> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let inactive_tail = if require_saved_survival_likelihood_mode(model)?
        == SurvivalLikelihoodMode::MarginalSlope
    {
        model
            .saved_prediction_runtime()?
            .influence_absorber_width
            .unwrap_or(0)
    } else {
        0
    };
    let active_len = fit.beta.len().checked_sub(inactive_tail).ok_or_else(|| {
        SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival influence-absorber width {inactive_tail} exceeds the {} fitted coefficients",
                fit.beta.len()
            ),
        }
    })?;
    let covariance = select_survival_prediction_covariance(
        fit.beta_covariance(),
        fit.beta_covariance_corrected(),
        covariance_mode,
    )?;
    if covariance.nrows() != fit.beta.len() || covariance.ncols() != fit.beta.len() {
        return Err(SurvivalPredictError::PosteriorCovariance {
            reason: format!(
                "saved survival {} covariance has shape {}x{}, expected {}x{} in fitted block order",
                covariance_mode.as_str(),
                covariance.nrows(),
                covariance.ncols(),
                fit.beta.len(),
                fit.beta.len(),
            ),
        });
    }
    Ok((
        fit.beta.clone(),
        covariance.slice(s![..active_len, ..active_len]).to_owned(),
    ))
}

fn saved_model_with_survival_coefficients(
    model: &SavedModel,
    coefficients: &Array1<f64>,
) -> Result<SavedModel, SurvivalPredictError> {
    let mut draw_model = model.clone();
    let payload = match &mut draw_model {
        SavedModel::Standard { payload }
        | SavedModel::LocationScale { payload }
        | SavedModel::MarginalSlope { payload }
        | SavedModel::Survival { payload }
        | SavedModel::TransformationNormal { payload } => payload,
    };

    let (beta_time, beta_threshold, beta_log_sigma, beta_link_wiggle, beta_time_blocks) = {
        let fit = payload.fit_result.as_mut().ok_or_else(|| {
            SurvivalPredictError::MissingFitMetadata {
                reason: "saved survival model is missing canonical fit_result".to_string(),
            }
        })?;
        if coefficients.len() != fit.beta.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "posterior survival coefficient draw has length {}, expected {}",
                    coefficients.len(),
                    fit.beta.len()
                ),
            });
        }
        fit.beta.assign(coefficients);
        let mut cursor = 0usize;
        for block in &mut fit.blocks {
            let end = cursor + block.beta.len();
            block.beta.assign(&coefficients.slice(s![cursor..end]));
            cursor = end;
        }
        if cursor != coefficients.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved survival coefficient blocks total {cursor} entries, but the joint vector has {}",
                    coefficients.len()
                ),
            });
        }
        (
            fit.block_by_role(BlockRole::Time)
                .map(|block| block.beta.to_vec()),
            fit.block_by_role(BlockRole::Threshold)
                .map(|block| block.beta.to_vec()),
            fit.block_by_role(BlockRole::Scale)
                .map(|block| block.beta.to_vec()),
            fit.block_by_role(BlockRole::LinkWiggle)
                .map(|block| block.beta.to_vec()),
            fit.blocks
                .iter()
                .map(|block| block.beta.to_vec())
                .collect::<Vec<_>>(),
        )
    };

    if payload.survival_beta_time.is_some() {
        payload.survival_beta_time = beta_time.clone();
    }
    if payload.survival_beta_threshold.is_some() {
        payload.survival_beta_threshold = beta_threshold;
    }
    if payload.survival_beta_log_sigma.is_some() {
        payload.survival_beta_log_sigma = beta_log_sigma;
    }
    if payload.beta_link_wiggle.is_some() {
        payload.beta_link_wiggle = beta_link_wiggle;
    }
    if let (Some(saved), Some(time_beta)) = (
        payload.beta_baseline_timewiggle.as_mut(),
        beta_time.as_ref(),
    ) {
        if saved.len() > time_beta.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved baseline-timewiggle has {} coefficients, but the time block has {}",
                    saved.len(),
                    time_beta.len()
                ),
            });
        }
        *saved = time_beta[time_beta.len() - saved.len()..].to_vec();
    }
    if let Some(saved_by_cause) = payload.beta_baseline_timewiggle_by_cause.as_mut() {
        if saved_by_cause.len() != beta_time_blocks.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved cause-specific timewiggles have {} blocks, but the fit has {} cause blocks",
                    saved_by_cause.len(),
                    beta_time_blocks.len()
                ),
            });
        }
        for (saved, block) in saved_by_cause.iter_mut().zip(&beta_time_blocks) {
            if saved.len() > block.len() {
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: format!(
                        "saved cause-specific timewiggle has {} coefficients, but its endpoint block has {}",
                        saved.len(),
                        block.len()
                    ),
                });
            }
            *saved = block[block.len() - saved.len()..].to_vec();
        }
    }
    Ok(draw_model)
}

fn conditional_event_density(
    survival: f64,
    cumulative_hazard: f64,
    hazard: f64,
) -> Result<f64, SurvivalPredictError> {
    if hazard == 0.0 {
        return Ok(0.0);
    }
    if survival > 0.0 && hazard.is_finite() {
        return Ok(survival * hazard);
    }
    if cumulative_hazard.is_finite() && hazard > 0.0 {
        return Ok((hazard.ln() - cumulative_hazard).exp());
    }
    if cumulative_hazard == f64::INFINITY && hazard.is_finite() && hazard >= 0.0 {
        return Ok(0.0);
    }
    Err(SurvivalPredictError::NumericalFailure {
        reason: format!(
            "posterior survival quadrature could not resolve conditional density from S={survival}, H={cumulative_hazard}, h={hazard}"
        ),
    })
}

/// Third-degree spherical-radial quadrature for a possibly singular Gaussian
/// coefficient posterior.  The `2r` equal-weight nodes are exact for every
/// polynomial through total degree three in the active rank-`r` subspace, use
/// the full covariance (including cross-block/cross-cause terms), and require
/// no sampling seed or dimension-specific tuning constant.
fn for_each_survival_posterior_node<F>(
    posterior_mean: &Array1<f64>,
    active_covariance: &Array2<f64>,
    mut consume: F,
) -> Result<(), SurvivalPredictError>
where
    F: FnMut(&Array1<f64>, f64) -> Result<(), SurvivalPredictError>,
{
    let active_len = active_covariance.nrows();
    if active_covariance.ncols() != active_len || active_len > posterior_mean.len() {
        return Err(SurvivalPredictError::PosteriorCovariance {
            reason: format!(
                "survival posterior quadrature received mean length {} and active covariance {}x{}",
                posterior_mean.len(),
                active_covariance.nrows(),
                active_covariance.ncols(),
            ),
        });
    }
    let factorization = crate::survival::location_scale::factorize_psd_covariance(
        active_covariance,
        "survival posterior coefficient covariance",
    )
    .map_err(|reason| SurvivalPredictError::PosteriorCovariance { reason })?;
    let rank = factorization.factor.ncols();
    if rank == 0 {
        return consume(posterior_mean, 1.0);
    }
    let scale = (rank as f64).sqrt();
    let weight = 1.0 / (2 * rank) as f64;
    for column in 0..rank {
        for sign in [-1.0_f64, 1.0_f64] {
            let mut node = posterior_mean.clone();
            for row in 0..active_len {
                node[row] += sign * scale * factorization.factor[[row, column]];
            }
            consume(&node, weight)?;
        }
    }
    Ok(())
}

fn posterior_standard_error_matrix(
    mean: &Array2<f64>,
    second_moment: &Array2<f64>,
    label: &str,
) -> Result<Array2<f64>, SurvivalPredictError> {
    if second_moment.dim() != mean.dim() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "posterior {label} moment shape mismatch: mean={:?}, second={:?}",
                mean.dim(),
                second_moment.dim(),
            ),
        });
    }
    let mut standard_error = Array2::<f64>::zeros(mean.raw_dim());
    for ((row, column), slot) in standard_error.indexed_iter_mut() {
        let first = mean[[row, column]];
        let second = second_moment[[row, column]];
        if !(first.is_finite() && second.is_finite()) {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "posterior {label} moments must be finite at row {row}, time column {column}: mean={first}, second={second}"
                ),
            });
        }
        let variance = second - first * first;
        let roundoff_tolerance =
            128.0 * f64::EPSILON * second.abs().max((first * first).abs()).max(1.0);
        if variance < -roundoff_tolerance {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "posterior {label} variance is negative beyond roundoff at row {row}, time column {column}: {variance}"
                ),
            });
        }
        *slot = variance.max(0.0).sqrt();
    }
    Ok(standard_error)
}

fn posterior_standard_error_vector(
    mean: &Array1<f64>,
    second_moment: &Array1<f64>,
    label: &str,
) -> Result<Array1<f64>, SurvivalPredictError> {
    if second_moment.len() != mean.len() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "posterior {label} moment length mismatch: mean={}, second={}",
                mean.len(),
                second_moment.len(),
            ),
        });
    }
    let mut standard_error = Array1::<f64>::zeros(mean.len());
    for row in 0..mean.len() {
        let first = mean[row];
        let second = second_moment[row];
        if !(first.is_finite() && second.is_finite()) {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "posterior {label} moments must be finite at row {row}: mean={first}, second={second}"
                ),
            });
        }
        let variance = second - first * first;
        let roundoff_tolerance =
            128.0 * f64::EPSILON * second.abs().max((first * first).abs()).max(1.0);
        if variance < -roundoff_tolerance {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "posterior {label} variance is negative beyond roundoff at row {row}: {variance}"
                ),
            });
        }
        standard_error[row] = variance.max(0.0).sqrt();
    }
    Ok(standard_error)
}

fn posterior_standard_error_surfaces(
    mean: &[Array2<f64>],
    second_moment: &[Array2<f64>],
    label: &str,
) -> Result<Vec<Array2<f64>>, SurvivalPredictError> {
    if second_moment.len() != mean.len() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "posterior {label} cause count mismatch: mean={}, second={}",
                mean.len(),
                second_moment.len(),
            ),
        });
    }
    mean.iter()
        .zip(second_moment)
        .enumerate()
        .map(|(cause, (first, second))| {
            posterior_standard_error_matrix(first, second, &format!("{label} cause {}", cause + 1))
        })
        .collect()
}

fn posterior_standard_error_vectors(
    mean: &[Array1<f64>],
    second_moment: &[Array1<f64>],
    label: &str,
) -> Result<Vec<Array1<f64>>, SurvivalPredictError> {
    if second_moment.len() != mean.len() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "posterior {label} cause count mismatch: mean={}, second={}",
                mean.len(),
                second_moment.len(),
            ),
        });
    }
    mean.iter()
        .zip(second_moment)
        .enumerate()
        .map(|(cause, (first, second))| {
            posterior_standard_error_vector(first, second, &format!("{label} cause {}", cause + 1))
        })
        .collect()
}

fn predict_survival_posterior_mean(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    let (posterior_mean, active_covariance) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    let mut result = predict_survival(
        SurvivalPredictRequest {
            model: req.model,
            data: req.data,
            col_map: req.col_map,
            training_headers: req.training_headers,
            primary_offset: req.primary_offset,
            noise_offset: req.noise_offset,
            time_grid: req.time_grid,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        covariance_mode,
    )?;
    let (n_rows, n_times) = result.survival.dim();
    let mut survival_mean = Array2::<f64>::zeros((n_rows, n_times));
    let mut survival_second = Array2::<f64>::zeros((n_rows, n_times));
    let mut density_mean = Array2::<f64>::zeros((n_rows, n_times));
    let mut hazard_mean = Array2::<f64>::zeros((n_rows, n_times));
    let mut eta_mean = Array1::<f64>::zeros(n_rows);
    let mut eta_second = Array1::<f64>::zeros(n_rows);

    for_each_survival_posterior_node(&posterior_mean, &active_covariance, |node, weight| {
        let draw_model = saved_model_with_survival_coefficients(req.model, node)?;
        let draw = predict_survival(
            SurvivalPredictRequest {
                model: &draw_model,
                data: req.data,
                col_map: req.col_map,
                training_headers: req.training_headers,
                primary_offset: req.primary_offset,
                noise_offset: req.noise_offset,
                time_grid: req.time_grid,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::Plugin,
            },
            covariance_mode,
        )?;
        if draw.survival.dim() != (n_rows, n_times)
            || draw.hazard.dim() != (n_rows, n_times)
            || draw.cumulative_hazard.dim() != (n_rows, n_times)
            || draw.linear_predictor.len() != n_rows
            || draw.times != result.times
            || draw.likelihood_mode != result.likelihood_mode
        {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: "posterior survival quadrature node changed the prediction schema"
                    .to_string(),
            });
        }
        for row in 0..n_rows {
            let eta = draw.linear_predictor[row];
            eta_mean[row] += weight * eta;
            eta_second[row] += weight * eta * eta;
            for time in 0..n_times {
                let survival = draw.survival[[row, time]];
                let hazard = draw.hazard[[row, time]];
                let density = conditional_event_density(
                    survival,
                    draw.cumulative_hazard[[row, time]],
                    hazard,
                )?;
                survival_mean[[row, time]] += weight * survival;
                survival_second[[row, time]] += weight * survival * survival;
                density_mean[[row, time]] += weight * density;
                hazard_mean[[row, time]] += weight * hazard;
            }
        }
        Ok(())
    })?;

    for row in 0..n_rows {
        for time in 0..n_times {
            let survival = survival_mean[[row, time]].clamp(0.0, 1.0);
            let density = density_mean[[row, time]];
            if !(density.is_finite() && density >= 0.0) {
                return Err(SurvivalPredictError::NumericalFailure {
                    reason: format!(
                        "posterior survival density is invalid at row {row}, time column {time}: {density}"
                    ),
                });
            }
            result.survival[[row, time]] = survival;
            result.cumulative_hazard[[row, time]] = -survival.ln();
            result.hazard[[row, time]] = if survival > 0.0 {
                density / survival
            } else if hazard_mean[[row, time]] == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
        }
    }
    result.survival_se = req.with_uncertainty.then(|| {
        Array2::from_shape_fn((n_rows, n_times), |(row, time)| {
            (survival_second[[row, time]] - survival_mean[[row, time]] * survival_mean[[row, time]])
                .max(0.0)
                .sqrt()
        })
    });
    result.eta_se = req.with_uncertainty.then(|| {
        Array1::from_shape_fn(n_rows, |row| {
            (eta_second[row] - eta_mean[row] * eta_mean[row])
                .max(0.0)
                .sqrt()
        })
    });
    result.covariance_source = req.with_uncertainty.then_some(covariance_mode);
    Ok(result)
}

fn predict_competing_risks_with_posterior(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<CompetingRisksPredictResult, SurvivalPredictError> {
    let posterior_mean_estimand = req.estimand == SurvivalPredictEstimand::PosteriorMean;
    let (posterior_mean, active_covariance) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    // The public posterior-mean point is always the conditional-posterior
    // estimand. A smoothing-corrected interval changes only its reported
    // uncertainty, exactly as on the standard-family path. If corrected
    // covariance becomes available for this model class, compute the
    // conditional point once and the corrected second moments separately;
    // never silently change the point estimand with the interval mode.
    let separate_conditional_point = posterior_mean_estimand
        && req.with_uncertainty
        && covariance_mode == SurvivalPredictionCovarianceMode::SmoothingCorrected;
    let mut result = if separate_conditional_point {
        predict_competing_risks_with_posterior(
            SurvivalPredictRequest {
                model: req.model,
                data: req.data,
                col_map: req.col_map,
                training_headers: req.training_headers,
                primary_offset: req.primary_offset,
                noise_offset: req.noise_offset,
                time_grid: req.time_grid,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::PosteriorMean,
            },
            SurvivalPredictionCovarianceMode::Conditional,
        )?
    } else {
        predict_competing_risks_survival(
            SurvivalPredictRequest {
                model: req.model,
                data: req.data,
                col_map: req.col_map,
                training_headers: req.training_headers,
                primary_offset: req.primary_offset,
                noise_offset: req.noise_offset,
                time_grid: req.time_grid,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::Plugin,
            },
            SurvivalPredictionCovarianceMode::Conditional,
        )?
    };
    let cause_count = result.cif.len();
    let (n_rows, n_times) = result.overall_survival.dim();
    let mut survival_mean = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut survival_second = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut hazard_mean = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut hazard_second = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut cumulative_hazard_mean = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut cumulative_hazard_second = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut cif_mean = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut cif_second = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect::<Vec<_>>();
    let mut overall_mean = Array2::<f64>::zeros((n_rows, n_times));
    let mut overall_second = Array2::<f64>::zeros((n_rows, n_times));
    let mut eta_mean = (0..cause_count)
        .map(|_| Array1::<f64>::zeros(n_rows))
        .collect::<Vec<_>>();
    let mut eta_second = (0..cause_count)
        .map(|_| Array1::<f64>::zeros(n_rows))
        .collect::<Vec<_>>();

    for_each_survival_posterior_node(&posterior_mean, &active_covariance, |node, weight| {
        let draw_model = saved_model_with_survival_coefficients(req.model, node)?;
        let draw = predict_competing_risks_survival(
            SurvivalPredictRequest {
                model: &draw_model,
                data: req.data,
                col_map: req.col_map,
                training_headers: req.training_headers,
                primary_offset: req.primary_offset,
                noise_offset: req.noise_offset,
                time_grid: req.time_grid,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::Plugin,
            },
            SurvivalPredictionCovarianceMode::Conditional,
        )?;
        if draw.cif.len() != cause_count
            || draw.survival.len() != cause_count
            || draw.hazard.len() != cause_count
            || draw.cumulative_hazard.len() != cause_count
            || draw.linear_predictor.len() != cause_count
            || draw.overall_survival.dim() != (n_rows, n_times)
            || draw.times != result.times
            || draw.endpoint_names != result.endpoint_names
            || draw.likelihood_mode != result.likelihood_mode
        {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: "posterior competing-risks quadrature node changed the prediction schema"
                    .to_string(),
            });
        }
        for cause in 0..cause_count {
            if draw.survival[cause].dim() != (n_rows, n_times)
                || draw.hazard[cause].dim() != (n_rows, n_times)
                || draw.cumulative_hazard[cause].dim() != (n_rows, n_times)
                || draw.cif[cause].dim() != (n_rows, n_times)
                || draw.linear_predictor[cause].len() != n_rows
            {
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: format!(
                        "posterior competing-risks quadrature node changed cause {} surface dimensions",
                        cause + 1
                    ),
                });
            }
            for row in 0..n_rows {
                let eta = draw.linear_predictor[cause][row];
                eta_mean[cause][row] += weight * eta;
                eta_second[cause][row] += weight * eta * eta;
                for time in 0..n_times {
                    let survival = draw.survival[cause][[row, time]];
                    let hazard = draw.hazard[cause][[row, time]];
                    let cumulative_hazard = draw.cumulative_hazard[cause][[row, time]];
                    let cif = draw.cif[cause][[row, time]];
                    survival_mean[cause][[row, time]] += weight * survival;
                    survival_second[cause][[row, time]] += weight * survival * survival;
                    hazard_mean[cause][[row, time]] += weight * hazard;
                    hazard_second[cause][[row, time]] += weight * hazard * hazard;
                    cumulative_hazard_mean[cause][[row, time]] += weight * cumulative_hazard;
                    cumulative_hazard_second[cause][[row, time]] +=
                        weight * cumulative_hazard * cumulative_hazard;
                    cif_mean[cause][[row, time]] += weight * cif;
                    cif_second[cause][[row, time]] += weight * cif * cif;
                }
            }
        }
        for row in 0..n_rows {
            for time in 0..n_times {
                let overall_survival = draw.overall_survival[[row, time]];
                overall_mean[[row, time]] += weight * overall_survival;
                overall_second[[row, time]] += weight * overall_survival * overall_survival;
            }
        }
        Ok(())
    })?;

    let (hazard_se, survival_se, cumulative_hazard_se, cif_se, overall_survival_se, eta_se) =
        if req.with_uncertainty {
            (
                Some(posterior_standard_error_surfaces(
                    &hazard_mean,
                    &hazard_second,
                    "competing-risks hazard",
                )?),
                Some(posterior_standard_error_surfaces(
                    &survival_mean,
                    &survival_second,
                    "competing-risks survival",
                )?),
                Some(posterior_standard_error_surfaces(
                    &cumulative_hazard_mean,
                    &cumulative_hazard_second,
                    "competing-risks cumulative hazard",
                )?),
                Some(posterior_standard_error_surfaces(
                    &cif_mean,
                    &cif_second,
                    "competing-risks cumulative incidence",
                )?),
                Some(posterior_standard_error_matrix(
                    &overall_mean,
                    &overall_second,
                    "competing-risks overall survival",
                )?),
                Some(posterior_standard_error_vectors(
                    &eta_mean,
                    &eta_second,
                    "competing-risks linear predictor",
                )?),
            )
        } else {
            (None, None, None, None, None, None)
        };

    if posterior_mean_estimand && !separate_conditional_point {
        result.hazard = hazard_mean;
        result.survival = survival_mean
            .into_iter()
            .map(|surface| surface.mapv(|value| value.clamp(0.0, 1.0)))
            .collect();
        result.cumulative_hazard = cumulative_hazard_mean;
        result.cif = cif_mean
            .into_iter()
            .map(|surface| surface.mapv(|value| value.clamp(0.0, 1.0)))
            .collect();
        result.overall_survival = overall_mean.mapv(|value| value.clamp(0.0, 1.0));
        result.linear_predictor = eta_mean;
    }
    result.hazard_se = hazard_se;
    result.survival_se = survival_se;
    result.cumulative_hazard_se = cumulative_hazard_se;
    result.cif_se = cif_se;
    result.overall_survival_se = overall_survival_se;
    result.eta_se = eta_se;
    result.covariance_source = req.with_uncertainty.then_some(covariance_mode);
    Ok(result)
}

/// Trapezoidal integral of a per-row survival curve `s(t)` sampled at the shared
/// increasing `times` grid, restricted to `[0, tau]` — the restricted mean
/// survival time (RMST) at horizon `tau`.
///
/// `RMST_i(tau) = \int_0^{tau} S_i(t) dt`. This is the standard clinical-trial
/// survival summary (`survRM2`, lifelines `restricted_mean_survival_time`,
/// flexsurv `rmst_*`): the area under the survival curve up to `tau`, equal to
/// the mean of `min(T_i, tau)`. The curve is integrated with the trapezoid rule
/// over the prediction grid; the head segment `[0, times[0]]` uses `S(0) = 1`
/// (every subject is alive at the time origin), and when `tau` falls strictly
/// inside a grid cell the survival value at `tau` is linearly interpolated so the
/// partial cell contributes exactly. Grid points beyond `tau` are dropped.
///
/// Returns `None` when the grid is empty or `tau <= 0` (no area to accumulate),
/// or when any sampled survival value on the integrated span is non-finite.
fn restricted_mean_survival_time_from_curve(
    times: &[f64],
    survival_row: ndarray::ArrayView1<'_, f64>,
    tau: f64,
) -> Option<f64> {
    if times.is_empty() || !(tau > 0.0) || !tau.is_finite() {
        return None;
    }
    if times.len() != survival_row.len() {
        return None;
    }

    // Survival at the cell boundaries we sweep through, starting from S(0) = 1.
    let mut prev_t = 0.0_f64;
    let mut prev_s = 1.0_f64;
    let mut area = 0.0_f64;

    for (idx, &t) in times.iter().enumerate() {
        if !t.is_finite() || t < prev_t {
            return None;
        }
        let s = survival_row[idx];
        if !s.is_finite() {
            return None;
        }
        if t >= tau {
            // tau lands in (prev_t, t]; interpolate S(tau) and add the partial cell.
            let span = t - prev_t;
            let s_tau = if span > 0.0 {
                let w = (tau - prev_t) / span;
                prev_s + w * (s - prev_s)
            } else {
                prev_s
            };
            area += 0.5 * (prev_s + s_tau) * (tau - prev_t);
            return Some(area);
        }
        area += 0.5 * (prev_s + s) * (t - prev_t);
        prev_t = t;
        prev_s = s;
    }

    // tau is beyond the last grid point: extend the last survival value flat to
    // tau (conservative, matches survRM2's tau-at-or-before-last-event contract;
    // callers wanting a strict horizon pass a tau within the grid).
    area += prev_s * (tau - prev_t);
    Some(area)
}

impl SurvivalPredictResult {
    /// Per-row restricted mean survival time `\int_0^{tau} S_i(t) dt` from the
    /// predicted survival surface. `tau` is the restriction horizon (e.g. the
    /// study follow-up bound). Length-`n` vector, one RMST per predicted row.
    ///
    /// Returns `None` if the prediction grid is empty, `tau <= 0`, or any row's
    /// survival curve carries a non-finite value on `[0, tau]`.
    pub fn restricted_mean_survival_time(&self, tau: f64) -> Option<Array1<f64>> {
        let n = self.survival.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let rmst =
                restricted_mean_survival_time_from_curve(&self.times, self.survival.row(i), tau)?;
            out[i] = rmst;
        }
        Some(out)
    }
}

impl CompetingRisksPredictResult {
    /// Per-row restricted mean survival time of the OVERALL (all-cause) survival
    /// curve, `\int_0^{tau} S_overall_i(t) dt`. For competing risks the relevant
    /// restricted-mean summary is taken on the all-cause survival
    /// `exp(-sum_k H_k(t))`; cause-specific restricted-mean-time-lost is
    /// `tau - RMST` partitioned by CIF and is left to the CIF surface directly.
    pub fn restricted_mean_overall_survival_time(&self, tau: f64) -> Option<Array1<f64>> {
        let n = self.overall_survival.nrows();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let rmst = restricted_mean_survival_time_from_curve(
                &self.times,
                self.overall_survival.row(i),
                tau,
            )?;
            out[i] = rmst;
        }
        Some(out)
    }
}

/// Harrell's concordance index (C-index) of a survival risk score against
/// held-out outcomes. A larger `risk[i]` must predict a SHORTER survival time
/// (higher hazard). Over every orderable pair — pairs whose earlier observed
/// time is a genuine event, so the failure ordering is observed — a pair is
/// concordant when the earlier-failing subject carries the larger risk; equal
/// risks score half credit. `C = (concordant + 0.5·tied) / comparable`.
/// `C = 0.5` is random ranking, `C = 1.0` a perfect ordering.
///
/// This is the standard discrimination metric (`survival::concordance`,
/// `lifelines.utils.concordance_index`, scikit-survival `concordance_index_censored`).
/// `time`, `event` (1 = event, 0 = censored), and `risk` must share length `n`.
/// Returns `None` if there are no comparable pairs (e.g. all rows censored).
pub fn harrell_concordance(time: &[f64], event: &[f64], risk: &[f64]) -> Option<f64> {
    let n = time.len();
    if n != event.len() || n != risk.len() {
        return None;
    }
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                // Tied times are comparable only if both failed; such a pair is a
                // pure tie (no strict outcome ordering).
                if event[i] > 0.5 && event[j] > 0.5 {
                    comparable += 1.0;
                    concordant += 0.5;
                }
                continue;
            };
            if event[early] < 0.5 {
                // The earlier subject was censored: the true ordering is unknown.
                continue;
            }
            comparable += 1.0;
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        return None;
    }
    Some(concordant / comparable)
}

/// IPCW (inverse-probability-of-censoring-weighted) Brier score of a predicted
/// survival probability at a fixed horizon `tau` against held-out outcomes — the
/// Graf et al. (1999) estimator used by scikit-survival `brier_score`, `pec`, and
/// `survival::brier`.
///
/// `s_pred[i]` is the model's predicted survival probability `S(tau | x_i)`.
/// `time`/`event` are the held-out observed time and event indicator. `g_cens`
/// is the censoring survival distribution `G(t) = P(C > t)` evaluated at the two
/// weighting times the estimator needs per subject — supplied as a callable so
/// the caller can pass a Kaplan–Meier fit of the censoring process. Each
/// subject's squared residual `(target − Ŝ_i(τ))²` is reweighted by the inverse
/// censoring probability:
///   * event at/before `τ` (`T_i ≤ τ, δ_i = 1`) → target `0` (dead), weight `1/G(T_i)`;
///   * still alive past `τ` (`T_i > τ`)         → target `1` (alive), weight `1/G(τ)`;
///   * censored at/before `τ`                    → target undefined, contributes `0`.
///
/// The score is the **sample mean over all valid subjects** (Graf normalization,
/// dividing by `n`, not by the sum of weights):
///   `BS(τ) = (1/n) Σ_i w_i·(target_i − Ŝ_i(τ))²`.
/// This is the convention scikit-survival / pec / `survival::brier` report, so
/// the value is directly comparable to those packages. Lower is better; `0` is
/// perfect. Returns `None` on length mismatch or when no subject is valid.
///
/// Subjects with non-finite or non-positive `time`/`event` are dropped from both
/// numerator and denominator. When `G` collapses to `0` at a weighting time the
/// IPCW weight is undefined; such a subject contributes `0` (rather than `∞`),
/// which keeps the estimator finite at the extreme tail where the censoring KM
/// runs out of support.
pub fn ipcw_brier_score(
    s_pred: &[f64],
    time: &[f64],
    event: &[f64],
    tau: f64,
    g_cens: impl Fn(f64) -> f64,
) -> Option<f64> {
    let n = s_pred.len();
    if n != time.len() || n != event.len() {
        return None;
    }
    let mut n_valid = 0.0_f64;
    let mut acc = 0.0_f64;
    for i in 0..n {
        if !time[i].is_finite() || !event[i].is_finite() || time[i] <= 0.0 {
            continue;
        }
        // Every valid subject counts toward the Graf denominator, even when its
        // IPCW contribution is zero (censored before τ, or G undefined).
        n_valid += 1.0;
        let (target, weight) = if time[i] <= tau && event[i] > 0.5 {
            // Failed at or before the horizon: contributes via 1/G(T_i).
            let g = g_cens(time[i]);
            if !(g > 0.0) {
                continue;
            }
            (0.0, 1.0 / g)
        } else if time[i] > tau {
            // Survived past the horizon: contributes via 1/G(τ).
            let g = g_cens(tau);
            if !(g > 0.0) {
                continue;
            }
            (1.0, 1.0 / g)
        } else {
            // Censored at or before τ (and not an event past τ): no info.
            continue;
        };
        let resid = target - s_pred[i];
        acc += weight * resid * resid;
    }
    if n_valid == 0.0 {
        return None;
    }
    Some(acc / n_valid)
}

/// Integrated IPCW Brier score (IBS) — the time-integrated [`ipcw_brier_score`],
/// matching scikit-survival's `integrated_brier_score` and `pec`'s integrated
/// prediction-error curve.
///
/// `s_pred` is the `n × m` matrix of predicted survival probabilities whose
/// column `k` is `Ŝ_i(grid[k])`; `grid` is the strictly-increasing set of
/// evaluation times. The per-time Graf Brier `BS(grid[k])` is integrated by the
/// trapezoidal rule over the grid and normalized by the integration span:
///   `IBS = (1 / (t_max − t_min)) ∫_{t_min}^{t_max} BS(t) dt`.
///
/// `g_cens` is the censoring survival `G(t) = P(C > t)` (see [`KaplanMeier`]).
/// Integration is restricted to grid points within `[grid[0], horizon]`; pass
/// `horizon = f64::INFINITY` to integrate the full grid. Restricting to the
/// observed support is the standard guard against the extrapolation tail where
/// no subject remains at risk and the IPCW weights become unstable.
///
/// Returns `None` if the grid is malformed (fewer than two usable points, wrong
/// width, non-increasing) or every per-time Brier is undefined.
pub fn integrated_ipcw_brier_score(
    s_pred: ArrayView2<f64>,
    time: &[f64],
    event: &[f64],
    grid: &[f64],
    horizon: f64,
    g_cens: impl Fn(f64) -> f64,
) -> Option<f64> {
    let m = grid.len();
    if m < 2 || s_pred.ncols() != m || s_pred.nrows() != time.len() {
        return None;
    }
    if grid.windows(2).any(|pair| !(pair[1] > pair[0])) {
        return None;
    }
    // Collect (time, Brier) at every grid point inside the integration window.
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(m);
    for k in 0..m {
        if grid[k] > horizon {
            break;
        }
        let col = s_pred.column(k);
        let col_slice: Vec<f64> = col.to_vec();
        if let Some(bs) = ipcw_brier_score(&col_slice, time, event, grid[k], &g_cens) {
            pts.push((grid[k], bs));
        }
    }
    if pts.len() < 2 {
        return None;
    }
    let span = pts[pts.len() - 1].0 - pts[0].0;
    if !(span > 0.0) {
        return None;
    }
    let mut integral = 0.0_f64;
    for w in pts.windows(2) {
        integral += 0.5 * (w[1].1 + w[0].1) * (w[1].0 - w[0].0);
    }
    Some(integral / span)
}

/// Right-continuous Kaplan–Meier survival estimator `Ŝ(t) = ∏_{t_j ≤ t}(1 − d_j/n_j)`.
///
/// Built from observed `(time, event)` pairs. To estimate the **censoring**
/// survival `G(t) = P(C > t)` required by the IPCW Brier score, fit with the
/// event indicator flipped (`1 − event`) so that censorings are the "events"
/// of the reversed process — see [`KaplanMeier::fit_censoring`].
#[derive(Clone, Debug, Default)]
pub struct KaplanMeier {
    /// `(event_time, survival_after_that_time)`, strictly increasing in time.
    steps: Vec<(f64, f64)>,
}

impl KaplanMeier {
    /// Fit the survival of the process whose event indicator is `event > 0.5`.
    pub fn fit(time: &[f64], event: &[f64]) -> Self {
        let mut rows: Vec<(f64, bool)> = time
            .iter()
            .zip(event.iter())
            .filter_map(|(&t, &e)| {
                (t.is_finite() && e.is_finite() && t > 0.0).then_some((t, e > 0.5))
            })
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut steps = Vec::new();
        let mut at_risk = rows.len() as f64;
        let mut survival = 1.0_f64;
        let mut i = 0usize;
        while i < rows.len() {
            let t = rows[i].0;
            let mut j = i;
            let mut deaths = 0usize;
            while j < rows.len() && rows[j].0 == t {
                deaths += usize::from(rows[j].1);
                j += 1;
            }
            if deaths > 0 && at_risk > 0.0 {
                survival *= ((at_risk - deaths as f64) / at_risk).max(0.0);
                steps.push((t, survival));
            }
            at_risk -= (j - i) as f64;
            i = j;
        }
        Self { steps }
    }

    /// Fit the censoring survival `G(t) = P(C > t)` by reversing the event role:
    /// a censored observation (`event ≤ 0.5`) is an "event" of the censoring
    /// process and a death (`event > 0.5`) is a censoring of it.
    pub fn fit_censoring(time: &[f64], event: &[f64]) -> Self {
        let flipped: Vec<f64> = event
            .iter()
            .map(|&e| if e > 0.5 { 0.0 } else { 1.0 })
            .collect();
        Self::fit(time, &flipped)
    }

    /// Right-continuous step lookup: `Ŝ(t)` = survival at the last event time
    /// `≤ t` (and `1.0` before the first event).
    pub fn at(&self, t: f64) -> f64 {
        let mut s = 1.0_f64;
        for &(time, surv) in &self.steps {
            if time <= t {
                s = surv;
            } else {
                break;
            }
        }
        s
    }
}

/// Joint cause-specific competing-risks prediction result.
pub struct CompetingRisksPredictResult {
    pub times: Vec<f64>,
    pub endpoint_names: Vec<String>,
    /// Cause-specific instantaneous hazards, shaped endpoint x row x time.
    pub hazard: Vec<Array2<f64>>,
    /// Endpoint-specific survival surfaces exp(-H_k(t)), endpoint x row x time.
    pub survival: Vec<Array2<f64>>,
    /// Cause-specific cumulative hazards, endpoint x row x time.
    pub cumulative_hazard: Vec<Array2<f64>>,
    /// Aalen-Johansen cumulative incidence, endpoint x row x time.
    pub cif: Vec<Array2<f64>>,
    /// Overall survival exp(-sum_k H_k(t)), row x time.
    pub overall_survival: Array2<f64>,
    /// Per-endpoint linear predictor at each row's own exit time, endpoint x row.
    pub linear_predictor: Vec<Array1<f64>>,
    pub likelihood_mode: SurvivalLikelihoodMode,
    /// Exact covariance definition used for posterior standard errors.
    /// `None` means no uncertainty was requested.
    pub covariance_source: Option<SurvivalPredictionCovarianceMode>,
    /// Posterior standard deviation of each cause-specific hazard surface.
    pub hazard_se: Option<Vec<Array2<f64>>>,
    /// Posterior standard deviation of each endpoint-specific survival surface.
    pub survival_se: Option<Vec<Array2<f64>>>,
    /// Posterior standard deviation of each cause-specific cumulative hazard.
    pub cumulative_hazard_se: Option<Vec<Array2<f64>>>,
    /// Posterior standard deviation of each cause-specific cumulative incidence.
    pub cif_se: Option<Vec<Array2<f64>>>,
    /// Posterior standard deviation of the all-cause survival surface.
    pub overall_survival_se: Option<Array2<f64>>,
    /// Posterior standard deviation of each cause-specific linear predictor.
    pub eta_se: Option<Vec<Array1<f64>>>,
}

/// Run the survival prediction pipeline.
///
/// Pure library function: no progress bars, no file I/O, no uncertainty
/// bounds. The CLI wraps this with progress updates + CSV writes; the
/// FFI wraps it with JSON serialization.
pub fn predict_survival(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    if req.estimand == SurvivalPredictEstimand::PosteriorMean {
        return predict_survival_posterior_mean(req, covariance_mode);
    }
    let SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset,
        noise_offset,
        time_grid,
        with_uncertainty,
        estimand: _,
    } = req;

    // `survival_entry == None` is the right-censored shorthand
    // `Surv(time, event)` produced by `gam fit` / `gamfit.fit`: no entry
    // column was supplied at training time, so entry ages default to
    // zero at prediction time too. The CLI's `run_predict_survival`
    // applies the same fallback; mirroring it here keeps `gam predict`,
    // `gam sample`, and the Python `model.predict` FFI symmetric across
    // every likelihood that lands in this code path (weibull,
    // transformation, ...).
    let time_cols = resolve_saved_survival_time_columns(model, col_map)?;
    let exit_col = time_cols.exit_col;

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
        return Err(SurvivalPredictError::InvalidInput {
            reason: format!(
                "survival prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
                primary_offset.len(),
                noise_offset.len()
            ),
        });
    }
    let effective_primary_offset = cov_design
        .compose_offset(primary_offset.view(), "survival prediction covariate block")
        .map_err(|error| error.to_string())?;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            normalize_survival_time_pair(time_cols.row_entry_time(data, i), data[[i, exit_col]], i)
        })
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
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "survival prediction via predict_survival does not support likelihood_mode={} yet; \
             latent window prediction lives in the CLI's run_predict_saved_latent_window_impl \
             pipeline and has not yet been ported to the library. Use the CLI predict command.",
                survival_likelihood_modename(saved_likelihood_mode)
            ),
        });
    }
    // Location-scale: handled via a dedicated batch path that calls
    // `predict_survival_location_scale` directly.
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        return predict_survival_location_scale_batch(
            model,
            &age_entry,
            &age_exit,
            &cov_design,
            &effective_primary_offset,
            noise_offset,
            training_headers,
            col_map,
            data,
            time_grid,
            with_uncertainty,
            covariance_mode,
        )
        .map_err(SurvivalPredictError::from);
    }
    if with_uncertainty {
        return Err(SurvivalPredictError::from(format!(
            "predict_survival: with_uncertainty is currently supported only for the \
             location-scale likelihood mode; got {}",
            survival_likelihood_modename(saved_likelihood_mode)
        )));
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
    // Single-cause Weibull without a learned baseline timewiggle carries its
    // ENTIRE log-cumulative-hazard baseline in the fitted `[1, log t]` linear
    // time-basis coefficients, not in a parametric offset. The fit centers that
    // basis at the survival time anchor (`center_survival_time_designs_at_anchor`
    // in the workflow), which zeroes the constant column so `beta[0]` is
    // unidentified and the fitted baseline is exactly
    // `beta[1] * (log t - log anchor)`. The model still SAVES a `Weibull`
    // baseline target (recovered scale/shape) for CIF/reporting, but that
    // metadata must NOT re-enter prediction as a parametric offset: doing so
    // double-counts the baseline (offset + beta) and, combined with predicting
    // against the UN-centered basis, collapses the survival surface to the
    // degenerate `S(t) == 1` (issue #897). Mirror the fit here: center the basis
    // at the anchor and carry a zero baseline offset, so predict reproduces the
    // fitted `beta[1] * (log t - log anchor)`. Weibull-WITH-timewiggle is a
    // different regime (the parametric offset is the baseline and beta carries
    // only the wiggle deviation), so it is excluded.
    let weibull_baseline_in_beta = saved_likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !model.has_baseline_time_wiggle();
    let mut time_anchor: Option<f64> = None;
    let mut time_anchor_row_cached: Option<Array1<f64>> = None;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
    ) || weibull_baseline_in_beta
    {
        let anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let time_anchor_row = evaluate_survival_time_basis_row(anchor, &resolved_time_cfg)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &time_anchor_row,
        )?;
        time_anchor = Some(anchor);
        time_anchor_row_cached = Some(time_anchor_row);
    }
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle()
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let mut baseline_cfg = saved_survival_runtime_baseline_config(model)?;
    if weibull_baseline_in_beta {
        baseline_cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Linear,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        };
    }

    // Resolve the time-grid: either the explicit grid (same for every
    // row) or per-row exit times (one column per row).
    let per_row_eval = time_grid.is_none();
    let eval_times: Vec<f64> = match time_grid {
        Some(grid) => {
            if grid.is_empty() {
                return Err(SurvivalPredictError::InvalidInput {
                    reason: "survival time_grid must contain at least one time".to_string(),
                });
            }
            for (idx, &t) in grid.iter().enumerate() {
                if !t.is_finite() || t < 0.0 {
                    return Err(SurvivalPredictError::InvalidInput {
                        reason: format!(
                            "survival time_grid requires finite non-negative times (index {idx})",
                        ),
                    });
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
        let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                &age_entry,
                &age_exit,
                &baseline_cfg,
                saved_likelihood_mode,
                None,
            )?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor.ok_or_else(|| {
                "saved survival marginal-slope model missing survival_time_anchor".to_string()
            })?,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
        Some(build_marginal_slope_predict_context(
            model,
            data,
            col_map,
            training_headers,
            &cov_design.design,
            &effective_primary_offset,
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

    let row_results: Result<Vec<SurvivalPredictionRow>, SurvivalPredictError> = (0..n)
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
            let evaluate_at = |t_query: f64| -> Result<(f64, f64, f64), SurvivalPredictError> {
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
                let (mut r_eta_entry, mut r_eta_exit, mut r_deriv_exit) =
                    build_survival_time_offsets_for_likelihood(
                        &single_entry,
                        &single_exit,
                        &baseline_cfg,
                        saved_likelihood_mode,
                        None,
                    )?;
                if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
                    add_survival_time_derivative_guard_offset(
                        &single_entry,
                        &single_exit,
                        time_anchor.ok_or_else(|| {
                            "saved survival marginal-slope model missing survival_time_anchor"
                                .to_string()
                        })?,
                        survival_derivative_guard_for_likelihood(saved_likelihood_mode),
                        &mut r_eta_entry,
                        &mut r_eta_exit,
                        &mut r_deriv_exit,
                    )?;
                }

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
                            effective_primary_offset[i],
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
                            effective_primary_offset[i],
                        )
                    }
                    SurvivalLikelihoodMode::Latent
                    | SurvivalLikelihoodMode::LatentBinary
                    | SurvivalLikelihoodMode::LocationScale => {
                        Err(SurvivalPredictError::NumericalFailure {
                            reason: "unreachable: unsupported likelihood_mode filtered earlier"
                                .to_string(),
                        })
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
                    if t_query <= 0.0 {
                        row.hazard[j] = 0.0;
                        row.cumulative_hazard[j] = 0.0;
                        row.survival[j] = 1.0;
                    } else {
                        let (_eta_t, cum_t, haz_t) = evaluate_at(t_query)?;
                        row.hazard[j] = haz_t;
                        row.cumulative_hazard[j] = cum_t;
                        row.survival[j] = (-cum_t).exp().clamp(0.0, 1.0);
                    }
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
        covariance_source: None,
    })
}

pub fn predict_competing_risks_survival(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<CompetingRisksPredictResult, SurvivalPredictError> {
    if req.estimand == SurvivalPredictEstimand::PosteriorMean || req.with_uncertainty {
        return predict_competing_risks_with_posterior(req, covariance_mode);
    }
    let SurvivalPredictRequest {
        model,
        data,
        col_map,
        training_headers,
        primary_offset,
        noise_offset,
        time_grid,
        with_uncertainty: _,
        estimand: _,
    } = req;

    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if !matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "joint cause-specific prediction supports transformation/weibull survival only; got {}",
                survival_likelihood_modename(saved_likelihood_mode)
            ),
        });
    }

    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let cause_count = model
        .survival_cause_count
        .unwrap_or(fit.blocks.len())
        .max(1);
    if cause_count <= 1 {
        return Err(SurvivalPredictError::MissingFitMetadata {
            reason: "competing-risks survival prediction requires a saved model with at least two causes"
                .to_string(),
        });
    }
    if fit.blocks.len() != cause_count {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved competing-risks survival fit has {} coefficient blocks but metadata says {cause_count} causes",
                fit.blocks.len()
            ),
        });
    }
    let endpoint_names = model.survival_endpoint_names.clone().unwrap_or_else(|| {
        (1..=cause_count)
            .map(|idx| format!("cause_{idx}"))
            .collect()
    });
    if endpoint_names.len() != cause_count {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved competing-risks survival endpoint_names has length {}, expected {cause_count}",
                endpoint_names.len()
            ),
        });
    }

    // Right-censored shorthand: same fallback as the single-cause path
    // above — entry ages default to zero when the model was fit without
    // an explicit entry column.
    let time_cols = resolve_saved_survival_time_columns(model, col_map)?;
    let exit_col = time_cols.exit_col;

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
        .map_err(|e| format!("failed to build competing-risks prediction design: {e}"))?;

    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(SurvivalPredictError::InvalidInput {
            reason: format!(
                "competing-risks prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
                primary_offset.len(),
                noise_offset.len()
            ),
        });
    }
    let effective_primary_offset = cov_design
        .compose_offset(
            primary_offset.view(),
            "competing-risks prediction covariate block",
        )
        .map_err(|error| error.to_string())?;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            normalize_survival_time_pair(time_cols.row_entry_time(data, i), data[[i, exit_col]], i)
        })
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (t0, t1)) in pairs.into_iter().enumerate() {
        age_entry[i] = t0;
        age_exit[i] = t1;
    }

    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    // See the single-cause `predict_survival` note: per-cause Weibull baselines
    // (no learned timewiggle) live in the anchor-centered linear time-basis
    // coefficients, so prediction must center the basis at the saved anchor and
    // carry a zero parametric baseline offset rather than re-adding the saved
    // (reporting-only) `Weibull` target as an offset (issues #897 / #689 / #690).
    // The ambient `time_build` is consumed only for the structural-basis check;
    // the per-(cause, row) loop rebuilds and centers its own `row_time`, so the
    // anchor row is all that needs threading through.
    let weibull_baseline_in_beta = saved_likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !model.has_baseline_time_wiggle();
    let cr_time_anchor_row: Option<Array1<f64>> = if weibull_baseline_in_beta {
        let anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        Some(evaluate_survival_time_basis_row(
            anchor,
            &resolved_time_cfg,
        )?)
    } else {
        None
    };
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull && !model.has_baseline_time_wiggle()
    {
        require_structural_survival_time_basis(
            &time_build.basisname,
            "saved competing-risks survival prediction",
        )?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model)?;

    let per_row_eval = time_grid.is_none();
    let eval_times: Vec<f64> = match time_grid {
        Some(grid) => {
            if grid.is_empty() {
                return Err(SurvivalPredictError::InvalidInput {
                    reason: "survival time_grid must contain at least one time".to_string(),
                });
            }
            for (idx, &t) in grid.iter().enumerate() {
                if !t.is_finite() || t < 0.0 {
                    return Err(SurvivalPredictError::InvalidInput {
                        reason: format!(
                            "survival time_grid requires finite non-negative times (index {idx})",
                        ),
                    });
                }
            }
            grid.to_vec()
        }
        None => Vec::new(),
    };
    let t_cols = if per_row_eval { 1 } else { eval_times.len() };

    // Refined internal grid for the Aalen-Johansen CIF assembly (gam#1385).
    //
    // The discrete AJ increment ΔF_k = S(t_{j-1})·(1−exp(−ΔH_total))·ΔH_k/ΔH_total
    // assumes the cause-specific hazard *ratio* h_k/h_total is constant within
    // each interval. On a coarse user grid with differently-shaped competing
    // hazards that assumption is violated, making the returned CIF a function of
    // the requested grid resolution (up to ~22% pointwise error) rather than a
    // pure function of the query time. We assemble AJ on a refined grid (extra
    // points inserted from 0 to the first user time and between consecutive user
    // times — cause-specific cumulative hazards are cheap closed-form
    // evaluate_at calls) and then read CIF/overall-survival back at the user's
    // requested times. The per-cause hazard/survival/cumulative_hazard returned
    // to the caller stay on the user grid (those are pointwise and already
    // grid-independent); only the AJ assembly uses the refinement.
    //
    // `refined_times` is strictly increasing and is a superset of `eval_times`;
    // `user_time_to_refined_index[j]` is the position of the j-th user time
    // inside `refined_times`. Per-row eval keeps its single-time anchor path.
    const CIF_REFINE_SUBINTERVALS: usize = 32;
    let (refined_times, user_time_to_refined_index): (Vec<f64>, Vec<usize>) = if per_row_eval {
        (Vec::new(), Vec::new())
    } else {
        // The user grid may arrive in any order (and contain duplicates); the
        // AJ recurrence is a time-ordered prefix integral, so the refinement
        // walks the SORTED times and maps every user position back to its
        // refined index. Walking an unsorted grid directly is not merely
        // inaccurate: a decreasing grid produces negative gaps, skips the
        // fill, and silently maps later user times onto the wrong refined
        // column (e.g. grid [2, 1] returned the t=2 CIF for both queries).
        let mut order: Vec<usize> = (0..eval_times.len()).collect();
        order.sort_by(|&a, &b| {
            eval_times[a]
                .partial_cmp(&eval_times[b])
                .expect("survival time_grid entries are validated finite above")
        });
        let mut refined: Vec<f64> = Vec::new();
        let mut user_index: Vec<usize> = vec![0; eval_times.len()];
        let mut prev = 0.0_f64;
        for &j_user in &order {
            let t_user = eval_times[j_user];
            // Insert CIF_REFINE_SUBINTERVALS-1 strictly-interior points in
            // (prev, t_user], landing exactly on t_user as the last point. Skip
            // the interior fill for a zero-length gap (duplicate / origin user
            // time) so `refined` stays strictly increasing.
            let gap = t_user - prev;
            if gap > 0.0 {
                for s in 1..CIF_REFINE_SUBINTERVALS {
                    let t_mid = prev + gap * (s as f64) / (CIF_REFINE_SUBINTERVALS as f64);
                    // Guard against ties from floating-point rounding.
                    if refined.last().is_none_or(|&last| t_mid > last) {
                        refined.push(t_mid);
                    }
                }
            }
            if refined.last().is_none_or(|&last| t_user > last) {
                refined.push(t_user);
            }
            user_index[j_user] = refined.len() - 1;
            prev = t_user;
        }
        (refined, user_index)
    };
    // Per-row eval integrates each row's CIF on its own refined [0, age_exit]
    // subdivision (normalized-fraction grid; see the assembly step below).
    let refined_cols = if per_row_eval {
        CIF_REFINE_SUBINTERVALS
    } else {
        refined_times.len()
    };

    let saved_timewiggle_by_cause = saved_cause_specific_timewiggles(model, &fit, cause_count)?;
    let cov_rows = (0..n)
        .map(|i| design_row_owned(&cov_design.design, i, "competing-risks covariate row"))
        .collect::<Result<Vec<_>, _>>()?;

    let mut hazard = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n, t_cols)))
        .collect::<Vec<_>>();
    let mut survival = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n, t_cols)))
        .collect::<Vec<_>>();
    let mut cumulative_hazard = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n, t_cols)))
        .collect::<Vec<_>>();
    // Cause-specific cumulative hazards on the refined AJ grid (gam#1385);
    // unused (zero-width) on the per-row-eval path.
    let mut cumulative_hazard_refined = (0..cause_count)
        .map(|_| Array2::<f64>::zeros((n, refined_cols)))
        .collect::<Vec<_>>();
    let mut linear_predictor = (0..cause_count)
        .map(|_| Array1::<f64>::zeros(n))
        .collect::<Vec<_>>();

    struct CauseRow {
        cause: usize,
        row: usize,
        hazard: Vec<f64>,
        survival: Vec<f64>,
        cumulative: Vec<f64>,
        /// Cumulative hazard on the refined AJ grid (gam#1385); empty on the
        /// per-row-eval path.
        cumulative_refined: Vec<f64>,
        eta_exit: f64,
    }

    let rows: Result<Vec<CauseRow>, SurvivalPredictError> = (0..cause_count * n)
        .into_par_iter()
        .map(|flat| {
            let cause = flat / n;
            let i = flat % n;
            let block = &fit.blocks[cause];
            let timewiggle = saved_timewiggle_by_cause[cause].as_ref();
            let evaluate_at = |t_query: f64| -> Result<(f64, f64, f64), SurvivalPredictError> {
                let t_entry = age_entry[i].min(t_query);
                let single_entry = Array1::from_elem(1, t_entry);
                let single_exit = Array1::from_elem(1, t_query);
                let mut row_time =
                    build_survival_time_basis(&single_entry, &single_exit, time_cfg.clone(), None)?;
                if let Some(anchor_row) = cr_time_anchor_row.as_ref() {
                    center_survival_time_designs_at_anchor(
                        &mut row_time.x_entry_time,
                        &mut row_time.x_exit_time,
                        anchor_row,
                    )?;
                }
                let (r_eta_exit, r_deriv_exit) = if weibull_baseline_in_beta {
                    (0.0, 0.0)
                } else {
                    let (_, eta_exit, deriv_exit) = build_survival_time_offsets_for_likelihood(
                        &single_entry,
                        &single_exit,
                        &baseline_cfg,
                        saved_likelihood_mode,
                        None,
                    )?;
                    (eta_exit[0], deriv_exit[0])
                };
                evaluate_rp_row_with_beta(
                    &block.beta,
                    timewiggle,
                    &row_time,
                    &cov_rows[i],
                    r_eta_exit,
                    r_deriv_exit,
                    effective_primary_offset[i],
                )
            };

            let mut out = CauseRow {
                cause,
                row: i,
                hazard: vec![0.0; t_cols],
                survival: vec![0.0; t_cols],
                cumulative: vec![0.0; t_cols],
                cumulative_refined: vec![0.0; refined_cols],
                eta_exit: 0.0,
            };
            if per_row_eval {
                let (eta_t, cum_t, haz_t) = evaluate_at(age_exit[i])?;
                out.eta_exit = eta_t;
                out.hazard[0] = haz_t;
                out.cumulative[0] = cum_t;
                out.survival[0] = (-cum_t).exp().clamp(0.0, 1.0);
                // Cause-specific cumulative hazards on this row's refined
                // [0, age_exit] subdivision for the time-ordered AJ assembly.
                // A single-interval assembly splits the CIF by ENDPOINT
                // cumulative-hazard proportions, which is exact only when the
                // cause-specific hazard ratio is constant in time; the CIF is
                // the time-ordered integral ∫ S(u−) dH_k(u) (gam#1385).
                for s in 1..=CIF_REFINE_SUBINTERVALS {
                    let frac = (s as f64) / (CIF_REFINE_SUBINTERVALS as f64);
                    let t_query = age_exit[i] * frac;
                    out.cumulative_refined[s - 1] = if t_query <= 0.0 {
                        0.0
                    } else if s == CIF_REFINE_SUBINTERVALS {
                        // frac == 1 exactly: reuse the exit evaluation so the
                        // assembled CIF and the reported cumulative hazard
                        // agree to the bit.
                        cum_t
                    } else {
                        evaluate_at(t_query)?.1
                    };
                }
            } else {
                for (j, &t_query) in eval_times.iter().enumerate() {
                    // Mirror the single-cause origin guard: every subject is
                    // alive at the time origin, so S(0)=1, H(0)=0, h(0)=0.
                    // Without this, the time basis floors t=0 to
                    // SURVIVAL_TIME_FLOOR and returns a nonzero hazard, which
                    // would anchor the Aalen-Johansen CIF assembly on a
                    // non-unit S(0) and bias every downstream value.
                    if t_query <= 0.0 {
                        out.hazard[j] = 0.0;
                        out.cumulative[j] = 0.0;
                        out.survival[j] = 1.0;
                    } else {
                        let (_eta_t, cum_t, haz_t) = evaluate_at(t_query)?;
                        out.hazard[j] = haz_t;
                        out.cumulative[j] = cum_t;
                        out.survival[j] = (-cum_t).exp().clamp(0.0, 1.0);
                    }
                }
                // Refined-grid cumulative hazards for the AJ CIF assembly
                // (gam#1385). Same closed-form evaluate_at; reuse the exact
                // user-grid values at the points that coincide so the returned
                // per-cause cumulative_hazard and the assembly agree at the user
                // times to the bit.
                for (jr, &t_query) in refined_times.iter().enumerate() {
                    out.cumulative_refined[jr] = if t_query <= 0.0 {
                        0.0
                    } else {
                        evaluate_at(t_query)?.1
                    };
                }
                let (eta_t, _, _) = evaluate_at(age_exit[i])?;
                out.eta_exit = eta_t;
            }
            Ok(out)
        })
        .collect();

    for row in rows? {
        linear_predictor[row.cause][row.row] = row.eta_exit;
        for j in 0..t_cols {
            hazard[row.cause][[row.row, j]] = row.hazard[j];
            survival[row.cause][[row.row, j]] = row.survival[j];
            cumulative_hazard[row.cause][[row.row, j]] = row.cumulative[j];
        }
        for jr in 0..refined_cols {
            cumulative_hazard_refined[row.cause][[row.row, jr]] = row.cumulative_refined[jr];
        }
    }

    // Assemble the Aalen-Johansen CIF on the refined grid (gam#1385), then read
    // the result back at the user-requested times so the CIF is grid-resolution
    // independent.
    let assembled = if per_row_eval {
        // Each row was integrated on its own normalized subdivision
        // t = age_exit·s/K. The AJ recurrence consumes only the time-ORDERED
        // cumulative-hazard values (the time stamps enter validation, never
        // the arithmetic), so a shared fraction grid s/K is an exact
        // parameterization of every row's [0, age_exit]; the row's CIF at its
        // exit time is the final column.
        let assembly_times = Array1::from_shape_fn(CIF_REFINE_SUBINTERVALS, |s| {
            ((s + 1) as f64) / (CIF_REFINE_SUBINTERVALS as f64)
        });
        let refined_assembled = assemble_competing_risks_cif_from_endpoints(
            assembly_times.view(),
            &cumulative_hazard_refined,
        )
        .map_err(|err| err.to_string())?;
        let last = CIF_REFINE_SUBINTERVALS - 1;
        let mut cif_user = (0..cause_count)
            .map(|_| Array2::<f64>::zeros((n, 1)))
            .collect::<Vec<_>>();
        let mut overall_user = Array2::<f64>::zeros((n, 1));
        for cause in 0..cause_count {
            for row in 0..n {
                cif_user[cause][[row, 0]] = refined_assembled.cif[cause][[row, last]];
            }
        }
        for row in 0..n {
            overall_user[[row, 0]] = refined_assembled.overall_survival[[row, last]];
        }
        CompetingRisksCifResult {
            cif: cif_user,
            overall_survival: overall_user,
        }
    } else {
        let assembly_times = Array1::from_vec(refined_times.clone());
        let refined_assembled = assemble_competing_risks_cif_from_endpoints(
            assembly_times.view(),
            &cumulative_hazard_refined,
        )
        .map_err(|err| err.to_string())?;
        // Project refined CIF / overall-survival columns onto the user grid.
        let mut cif_user = (0..cause_count)
            .map(|_| Array2::<f64>::zeros((n, t_cols)))
            .collect::<Vec<_>>();
        let mut overall_user = Array2::<f64>::zeros((n, t_cols));
        for (j_user, &jr) in user_time_to_refined_index.iter().enumerate() {
            for cause in 0..cause_count {
                for row in 0..n {
                    cif_user[cause][[row, j_user]] = refined_assembled.cif[cause][[row, jr]];
                }
            }
            for row in 0..n {
                overall_user[[row, j_user]] = refined_assembled.overall_survival[[row, jr]];
            }
        }
        CompetingRisksCifResult {
            cif: cif_user,
            overall_survival: overall_user,
        }
    };
    if assembled.cif.len() != cause_count {
        return Err(format!(
            "competing-risks CIF assembly produced {} endpoint matrices, expected {cause_count}",
            assembled.cif.len()
        )
        .into());
    }
    let cif = assembled.cif;
    let overall_survival = assembled.overall_survival;
    let times_out = if per_row_eval {
        age_exit.to_vec()
    } else {
        eval_times
    };
    Ok(CompetingRisksPredictResult {
        times: times_out,
        endpoint_names,
        hazard,
        survival,
        cumulative_hazard,
        cif,
        overall_survival,
        linear_predictor,
        likelihood_mode: saved_likelihood_mode,
        covariance_source: None,
        hazard_se: None,
        survival_se: None,
        cumulative_hazard_se: None,
        cif_se: None,
        overall_survival_se: None,
        eta_se: None,
    })
}

fn saved_cause_specific_timewiggles(
    model: &SavedModel,
    fit: &UnifiedFitResult,
    cause_count: usize,
) -> Result<Vec<Option<SavedBaselineTimeWiggleRuntime>>, SurvivalPredictError> {
    let has_metadata = model.baseline_timewiggle_knots.is_some()
        || model.baseline_timewiggle_degree.is_some()
        || model.baseline_timewiggle_penalty_orders.is_some()
        || model.baseline_timewiggle_double_penalty.is_some()
        || model.beta_baseline_timewiggle_by_cause.is_some();
    if !has_metadata {
        return Ok(vec![None; cause_count]);
    }
    let knots = model.baseline_timewiggle_knots.clone().ok_or_else(|| {
        "joint cause-specific survival missing baseline_timewiggle_knots".to_string()
    })?;
    let degree = model.baseline_timewiggle_degree.ok_or_else(|| {
        "joint cause-specific survival missing baseline_timewiggle_degree".to_string()
    })?;
    let penalty_orders = model
        .baseline_timewiggle_penalty_orders
        .clone()
        .ok_or_else(|| {
            "joint cause-specific survival missing baseline_timewiggle_penalty_orders".to_string()
        })?;
    let double_penalty = model.baseline_timewiggle_double_penalty.ok_or_else(|| {
        "joint cause-specific survival missing baseline_timewiggle_double_penalty".to_string()
    })?;
    let by_cause = model
        .beta_baseline_timewiggle_by_cause
        .as_ref()
        .ok_or_else(|| {
            "joint cause-specific survival missing beta_baseline_timewiggle_by_cause".to_string()
        })?;
    if by_cause.len() != cause_count {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "joint cause-specific survival has {} timewiggle coefficient blocks, expected {cause_count}",
                by_cause.len()
            ),
        });
    }
    for (cause, (block, beta_w)) in fit.blocks.iter().zip(by_cause).enumerate() {
        if beta_w.len() > block.beta.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "joint cause-specific survival cause {} timewiggle beta has length {}, but endpoint beta has {} coefficients",
                    cause + 1,
                    beta_w.len(),
                    block.beta.len()
                ),
            });
        }
    }
    Ok(by_cause
        .iter()
        .map(|beta| {
            Some(SavedBaselineTimeWiggleRuntime {
                knots: knots.clone(),
                degree,
                penalty_orders: penalty_orders.clone(),
                double_penalty,
                beta: beta.clone(),
            })
        })
        .collect())
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
) -> Result<Array1<f64>, SurvivalPredictError> {
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
) -> Result<MarginalSlopePredictContext, SurvivalPredictError> {
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
    let effective_noise_offset = logslope_design
        .compose_offset(
            noise_offset.view(),
            "survival marginal-slope logslope block",
        )
        .map_err(|error| error.to_string())?;

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
        &effective_noise_offset,
    )?;

    let blocks = &fit_saved.blocks;
    if blocks.len() < 3 {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope model requires at least 3 blocks [time, marginal, slope], got {}",
                blocks.len()
            ),
        });
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
        noise_offset: effective_noise_offset,
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
) -> Result<(f64, f64, f64), SurvivalPredictError> {
    let beta_time = &ctx.beta_time;
    let p_time_base = row_time.x_exit_time.ncols();
    let p_timewiggle = ctx
        .saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());
    if beta_time.len() != p_time_base + p_timewiggle {
        let hint = stale_weibull_time_basis_hint(
            &row_time.basisname,
            beta_time.len() == p_time_base + p_timewiggle + 1,
        );
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope time coefficient mismatch: beta has {} entries but expected base={} plus timewiggle={}{hint}",
                beta_time.len(),
                p_time_base,
                p_timewiggle
            ),
        });
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
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: "saved baseline-timewiggle exit design must be dense".to_string(),
                });
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
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(q_design_full)),
        offset: Array1::from_elem(1, r_eta_exit[0] + primary_offset_row),
        design_noise: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(logslope_design_2d),
        )),
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
    // `qd_with_wiggle` is the base survival-index time derivative q'(t), built
    // identically to fit-time `qd1 = dq_dq0·d_raw` (the wiggle chain and the
    // `+derivative_guard` offset are both already folded into `qd_exit_base`),
    // so there is no predict-vs-fit desync in the derivative reconstruction.
    //
    // Fit enforces the monotonicity floor `q'(t) >= derivative_guard` ONLY at
    // each training row's own exit time (one `t` per row), via the active-set
    // guard constraints. A prediction horizon is an arbitrary `t` — typically a
    // single CIF horizon evaluated for every row — which generally is NOT one of
    // the constrained training exit times. Where that horizon lands in a region
    // of sparse/no training exits, the penalized baseline spline can extrapolate
    // to a locally decreasing survival index, so `q'(t) < 0` is a legitimate
    // model statement ("no instantaneous hazard accrues here"), not a numerical
    // bug. The instantaneous hazard rate is physically non-negative, so the
    // truthful response is to clamp the index time-derivative at its floor 0
    // (flat hazard, survival locally constant) rather than reject the whole
    // prediction — clamping keeps the CIF well-posed and monotone. Only a
    // non-finite derivative (a real numerical failure) is surfaced to the strict
    // validator below.
    let eta_derivative = marginal_slope_index_derivative_at_horizon(deta_dq_arr[0], qd_with_wiggle);
    let (cum, haz) = probit_survival_hazard_components(eta, eta_derivative)?;
    Ok((eta, cum, haz))
}

/// Reconstruct the marginal-slope survival index time-derivative `eta'(t)` at a
/// prediction horizon and clamp it to its physical floor.
///
/// `deta_dq = ∂eta/∂q ≥ 1` is the rigid probit-frailty chain factor and
/// `qd_with_wiggle = q'(t)` is the base survival-index time derivative built
/// identically to fit-time `qd1`. The instantaneous hazard rate `h(t) = mills ·
/// eta'(t)` is physically non-negative, so a finite negative `eta'(t)` — which a
/// penalized baseline spline can legitimately produce when the prediction
/// horizon lands outside the training exit times the monotonicity guard
/// constrains — is clamped to its floor 0 (flat hazard, locally constant
/// survival), keeping the CIF well-posed. Non-finite values pass through
/// unchanged so the strict validator rejects them as genuine numerical failures.
#[inline]
fn marginal_slope_index_derivative_at_horizon(deta_dq: f64, qd_with_wiggle: f64) -> f64 {
    let eta_derivative = deta_dq * qd_with_wiggle;
    if eta_derivative.is_finite() {
        eta_derivative.max(0.0)
    } else {
        eta_derivative
    }
}

#[inline]
fn probit_survival_hazard_components(
    eta: f64,
    eta_derivative: f64,
) -> Result<(f64, f64), SurvivalPredictError> {
    if !(eta.is_finite() && eta_derivative.is_finite() && eta_derivative >= 0.0) {
        return Err(SurvivalPredictError::NumericalFailure {
            reason: format!(
                "saved survival marginal-slope prediction produced invalid survival index derivative: eta={eta}, eta_t={eta_derivative}"
            ),
        });
    }

    // Survival marginal-slope defines S(t) = Phi(-eta(t)). The event density
    // is f(t) = phi(eta(t)) * eta'(t), while the hazard rate exposed by the
    // prediction API is h(t) = f(t) / S(t). The signed-probit helper returns
    // both log Phi(-eta) and the stable Mills ratio phi(eta) / Phi(-eta).
    let (log_survival, mills_ratio) = signed_probit_logcdf_and_mills_ratio(-eta);
    let cumulative_hazard = -log_survival;
    let hazard = if eta_derivative == 0.0 {
        0.0
    } else {
        mills_ratio * eta_derivative
    };
    // `>= 0.0` rejects NaN (a programming-bug signal) and accepts the full
    // mathematical range [0, +∞]. Saturated probit fits where the model
    // genuinely says S(t)→0 produce a +∞ cumulative hazard — that is the
    // truthful answer, and the consumer's `survival = exp(-cum).clamp(0,1)`
    // handles it cleanly. Rejecting +∞ would force the predictor to fail on
    // models that the inner solver has already certified as a valid fit.
    if !(cumulative_hazard >= 0.0 && hazard >= 0.0) {
        return Err(SurvivalPredictError::NumericalFailure {
            reason: format!(
                "saved survival marginal-slope prediction produced invalid survival components: eta={eta}, eta_t={eta_derivative}, log_survival={log_survival}, hazard={hazard}"
            ),
        });
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
) -> Result<(f64, f64, f64), SurvivalPredictError> {
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let saved_runtime = model.saved_prediction_runtime()?;
    evaluate_rp_row_with_beta(
        &fit_saved.beta,
        saved_runtime.baseline_time_wiggle.as_ref(),
        row_time,
        cov_row,
        eta_time_offset_row,
        derivative_time_offset_row,
        primary_offset_row,
    )
}

fn evaluate_rp_row_with_beta(
    beta: &Array1<f64>,
    saved_timewiggle: Option<&SavedBaselineTimeWiggleRuntime>,
    row_time: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    eta_time_offset_row: f64,
    derivative_time_offset_row: f64,
    primary_offset_row: f64,
) -> Result<(f64, f64, f64), SurvivalPredictError> {
    let p_time = row_time.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle.map_or(0, |runtime| runtime.beta.len());
    let p_cov = cov_row.len();
    let p = p_time + p_timewiggle + p_cov;
    if beta.len() != p {
        let hint = stale_weibull_time_basis_hint(&row_time.basisname, beta.len() == p + 1);
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "survival RP coefficient mismatch: beta has {} entries but design has {} columns{hint}",
                beta.len(),
                p
            ),
        });
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
    if let Some(runtime) = saved_timewiggle {
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
            _ => {
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: "saved baseline-timewiggle exit design must be dense".to_string(),
                });
            }
        };
        if exit_design.ncols() != p_timewiggle {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "survival RP timewiggle design mismatch: rebuilt {} columns but runtime expects {}",
                    exit_design.ncols(),
                    p_timewiggle
                ),
            });
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
    let likelihood = LikelihoodSpec::new(
        ResponseFamily::RoystonParmar,
        InverseLink::Standard(StandardLink::Identity),
    );
    let eta =
        predict_royston_parmar_eta(x_exit.view(), beta.view(), offset_view.view(), &likelihood)?[0];
    let (cum, haz) = royston_parmar_survival_hazard_components(eta, eta_derivative)?;
    Ok((eta, cum, haz))
}

fn predict_royston_parmar_eta<X>(
    x: X,
    beta: ndarray::ArrayView1<'_, f64>,
    offset: ndarray::ArrayView1<'_, f64>,
    likelihood: &LikelihoodSpec,
) -> Result<Array1<f64>, SurvivalPredictError>
where
    X: Into<DesignMatrix>,
{
    if !matches!(likelihood.response, ResponseFamily::RoystonParmar)
        || !matches!(
            likelihood.link,
            InverseLink::Standard(StandardLink::Identity)
        )
    {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "survival prediction requires RoystonParmar with identity link".to_string(),
        });
    }
    let x = x.into();
    if x.nrows() != offset.len() || x.ncols() != beta.len() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "survival prediction design dimensions disagree: design is {}x{}, beta has length {}, offset has length {}",
                x.nrows(),
                x.ncols(),
                beta.len(),
                offset.len()
            ),
        });
    }
    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    Ok(eta)
}

#[inline]
fn royston_parmar_survival_hazard_components(
    eta: f64,
    eta_derivative: f64,
) -> Result<(f64, f64), SurvivalPredictError> {
    // `eta = log Λ(t)` and `eta_derivative = d(log Λ)/dt`, so the instantaneous
    // hazard is `h(t) = Λ(t) · eta_derivative = dΛ/dt`. Reject only the true bug
    // signals: a non-finite `eta`, and a derivative that is NaN or genuinely
    // negative.
    //
    // `eta_derivative == 0` is a VALID boundary value, not a failure. The RP
    // baseline `log Λ(t)` is an I-spline (monotone non-decreasing cumulative
    // hazard): beyond its last interior knot every I-spline basis is flat, so
    // its time-derivative is exactly 0 and the instantaneous hazard there is 0
    // (`S(t)` locally constant). Any RP model predicted on a grid that extends
    // past its training support hits this regime on the tail nodes. The earlier
    // strict `> 0.0` gate spuriously failed those predictions (#1564). The
    // probit / marginal-slope sibling guard
    // (`probit_survival_hazard_components`) already accepts the full `[0, ∞)`
    // range and maps a zero derivative to a zero hazard; the RP guard must match.
    if !(eta.is_finite() && eta_derivative.is_finite() && eta_derivative >= 0.0) {
        return Err(SurvivalPredictError::NumericalFailure {
            reason: format!(
                "saved Royston-Parmar survival prediction produced invalid log-cumulative-hazard derivative: eta={eta}, eta_t={eta_derivative}"
            ),
        });
    }
    let cumulative_hazard = eta.exp();
    // `h(t) = Λ(t) · d(log Λ)/dt`. Compute the zero-derivative boundary FIRST so
    // the `Λ = +∞` (saturated tail, `eta >~ 709.78`) × `0` (flat I-spline)
    // indeterminate form resolves to the mathematically correct `0`, not the
    // `NaN` that `f64::INFINITY * 0.0` produces. A flat cumulative hazard has
    // zero instantaneous hazard regardless of its (possibly saturated) level.
    let hazard = if eta_derivative == 0.0 {
        0.0
    } else {
        cumulative_hazard * eta_derivative
    };
    // Royston-Parmar parameterizes `eta = log Lambda(t)`, so `Lambda = exp(eta)`
    // is unbounded above and `exp(eta)` saturates to `+∞` in f64 once
    // `eta >~ 709.78` — exactly the regime a saturated RP fit produces in the
    // right tail. The math is well-defined (`S(t) → 0`, `h(t) → ∞`); rejecting
    // `+∞` here would crash predict on a fit the inner solver already accepted.
    // `>= 0.0` rejects NaN (the only true bug signal) while allowing the full
    // [0, +∞] range. The consumer materializes survival via
    // `survival = exp(-cum).clamp(0, 1)`, which collapses cleanly at saturation.
    if !(cumulative_hazard >= 0.0 && hazard >= 0.0) {
        return Err(SurvivalPredictError::NumericalFailure {
            reason: format!(
                "saved Royston-Parmar survival prediction produced invalid survival components: eta={eta}, eta_t={eta_derivative}, cumulative_hazard={cumulative_hazard}, hazard={hazard}"
            ),
        });
    }
    Ok((cumulative_hazard, hazard))
}

/// Batch evaluator for the location-scale survival likelihood mode.
///
/// Mirrors the CLI's LocationScale predict path (main.rs::run_predict_survival
/// LocationScale arm) but stays library-only: builds the threshold/log_sigma
/// designs from the saved frozen specs and resolved time margins, applies the
/// survival time-derivative guard, and calls `predict_survival_location_scale`.
///
/// Plugin survival only — uncertainty paths still live in the CLI.
fn predict_survival_location_scale_batch(
    model: &SavedModel,
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cov_design: &gam_terms::smooth::TermCollectionDesign,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    data: ArrayView2<'_, f64>,
    time_grid: Option<&[f64]>,
    with_uncertainty: bool,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, String> {
    use crate::survival::construction::evaluate_survival_time_basis_row;
    use crate::survival::location_scale::{
        SurvivalLocationScalePredictInput, predict_survival_location_scale,
        predict_survival_location_scalewith_uncertainty, replay_survival_covariate_channels,
    };
    use gam_linalg::matrix::DesignMatrix;

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
    let baseline_cfg = saved_survival_runtime_baseline_config(model)?;
    let saved_fit = saved_survival_location_scale_fit_result(model)?;
    // Reduced AFT changes the likelihood program (`h ≡ 0` and `-log(t)` moves
    // to the location channel), so it is persisted as topology. Coefficient
    // values are never interpreted as a model-class discriminator.
    let saved_structure = model
        .survival_location_scale_structure
        .as_ref()
        .ok_or_else(|| {
            "saved location-scale survival model is missing exact replay structure".to_string()
        })?;
    let reduced_parametric_aft = matches!(
        saved_structure.time_parameterization,
        crate::survival::location_scale::SurvivalLocationScaleTimeParameterization::ReducedParametricAft
    );
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
    // The reduced-AFT regime has no structural time warp (the monotone baseline
    // rides the location channel), so the structural-basis requirement does not
    // apply to it.
    if !model.has_baseline_time_wiggle() && !reduced_parametric_aft {
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
    if reduced_parametric_aft {
        // The warp is removed in this regime (`h ≡ 0`); the σ-scaled log-t baseline
        // rides the location channel via the `−log t` threshold shift applied
        // below. The saved `beta_time` is an all-zero length-`p` vector (the
        // reduced time block has zero free columns and a zero affine lift), so the
        // time-warp contribution `x_exit_time · beta_time` is identically zero for
        // ANY design — we therefore KEEP the full-width centered basis (so the
        // hazard's `beta.len() == x_exit_time.ncols()` check holds and the
        // scale-deviation primary keeps its full column count to match the saved
        // transform) and only zero the value OFFSET so `h_base = 0`. The derivative
        // is handled separately from `inv_sigma / t` in the hazard computation, so
        // the entry/derivative designs and offsets are left as built.
        eta_offset_exit = Array1::<f64>::zeros(eval_exit.len());
    }

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
        gam_terms::smooth::build_term_collection_design(sigma_input, &log_sigmaspec)
            .map_err(|err| format!("failed to build survival log-sigma design: {err}"))?;
    let effective_noise_offset = raw_sigma_design
        .compose_offset(
            noise_offset.view(),
            "survival location-scale log-sigma block",
        )
        .map_err(|error| error.to_string())?;

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

    let repeat_rows =
        |matrix: &DesignMatrix, label: &str| -> Result<DesignMatrix, SurvivalPredictError> {
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
    let expand_vector = |values: &Array1<f64>| -> Array1<f64> {
        if per_row_eval {
            values.clone()
        } else {
            Array1::from_shape_fn(total_rows, |k| values[k / eval_width])
        }
    };
    if saved_structure.threshold_time_basis.is_some()
        && threshold_design
            .affine_offset
            .iter()
            .any(|value| *value != 0.0)
    {
        return Err(
            "saved time-varying survival threshold cannot carry a non-zero smooth anchor"
                .to_string(),
        );
    }
    if saved_structure.log_sigma_time_basis.is_some()
        && raw_sigma_design
            .affine_offset
            .iter()
            .any(|value| *value != 0.0)
    {
        return Err(
            "saved time-varying survival log-sigma cannot carry a non-zero smooth anchor"
                .to_string(),
        );
    }
    let threshold_base_matrix = repeat_rows(
        &threshold_design.design,
        "survival location-scale prediction threshold design",
    )?;
    let raw_sigma_base_matrix = repeat_rows(
        &raw_sigma_design.design,
        "survival location-scale prediction log-sigma design",
    )?;
    let mut threshold_replay = replay_survival_covariate_channels(
        &threshold_base_matrix,
        &expand_vector(primary_offset),
        &eval_entry,
        &eval_exit,
        saved_structure.threshold_time_basis.as_ref(),
        "survival location-scale threshold",
    )?;
    let sigma_replay = replay_survival_covariate_channels(
        &raw_sigma_base_matrix,
        &expand_vector(&effective_noise_offset),
        &eval_entry,
        &eval_exit,
        saved_structure.log_sigma_time_basis.as_ref(),
        "survival location-scale log-sigma",
    )?;
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

    // Threshold (location) offset. In the reduced parametric-AFT regime the
    // σ-scaled `log t` baseline rides the location channel: shift the effective
    // location `η_t → η_t − log t` per query time so the predicted standardized
    // residual reproduces `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`, exactly
    // as the fit's `LocationLogTimeOffset` does. `eval_exit` already carries the
    // per-(row, time) query exit times in the same flattened layout as the
    // expanded offsets; `−log t` uses the same `SURVIVAL_TIME_FLOOR` floor as the
    // fit's `checked_log_survival_times` (issue #892).
    if reduced_parametric_aft {
        for (slot, &t) in threshold_replay.offset.iter_mut().zip(eval_exit.iter()) {
            *slot -= t
                .max(crate::survival::construction::SURVIVAL_TIME_FLOOR)
                .ln();
        }
    }
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
        x_threshold: threshold_replay.design_exit.clone(),
        eta_threshold_offset: threshold_replay.offset.clone(),
        x_log_sigma: sigma_replay.design_exit.clone(),
        eta_log_sigma_offset: sigma_replay.offset.clone(),
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
        // #2296: resolve the requested covariance definition exactly. A
        // smoothing-corrected request must never be satisfied with the
        // conditional matrix; location-scale fits do not persist a corrected
        // covariance today, so that request is a typed refusal, not a
        // silently narrower band.
        let cov = match select_survival_prediction_covariance(
            saved_fit.beta_covariance(),
            saved_fit.beta_covariance_corrected(),
            covariance_mode,
        ) {
            Ok(cov) => cov,
            Err(SurvivalPredictError::PosteriorCovariance { reason })
                if covariance_mode == SurvivalPredictionCovarianceMode::Conditional =>
            {
                return Err(format!(
                    "survival location-scale uncertainty: {reason}; refit with the \
                     current CLI / library to populate beta_covariance"
                ));
            }
            Err(err) => return Err(String::from(err)),
        };
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
    } else {
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|err| format!("survival location-scale predict failed: {err}"))?;
        (pred.eta, pred.survival_prob, None, None)
    };

    let beta_threshold = saved_fit.beta_threshold();
    let beta_log_sigma = saved_fit.beta_log_sigma();
    let eta_threshold = threshold_replay
        .design_exit
        .matrixvectormultiply(&beta_threshold)
        + &threshold_replay.offset;
    let mut eta_threshold_derivative = threshold_replay
        .design_derivative_exit
        .as_ref()
        .map(|design| design.matrixvectormultiply(&beta_threshold))
        .unwrap_or_else(|| Array1::zeros(total_rows));
    if reduced_parametric_aft {
        for (slot, &time) in eta_threshold_derivative.iter_mut().zip(eval_exit.iter()) {
            *slot -= 1.0 / time.max(crate::survival::construction::SURVIVAL_TIME_FLOOR);
        }
    }
    let eta_log_sigma = sigma_replay
        .design_exit
        .matrixvectormultiply(&beta_log_sigma)
        + &sigma_replay.offset;
    let eta_log_sigma_derivative = sigma_replay
        .design_derivative_exit
        .as_ref()
        .map(|design| design.matrixvectormultiply(&beta_log_sigma))
        .unwrap_or_else(|| Array1::zeros(total_rows));
    let hdot = if reduced_parametric_aft {
        Array1::zeros(total_rows)
    } else {
        let x_time_derivative = time_build
            .x_derivative_time
            .try_to_dense_by_chunks("survival location-scale prediction time-derivative design")?;
        location_scale_eta_derivative_components(
            &x_time_derivative,
            &derivative_offset_exit,
            &pred_input.x_time_exit,
            &pred_input.eta_time_offset_exit,
            time_wiggle_knots.as_ref(),
            time_wiggle_degree,
            time_wiggle_ncols,
            &saved_fit,
        )?
    };
    let inv_sigma = eta_log_sigma.mapv(crate::sigma_link::exp_sigma_inverse_from_eta_scalar);
    let q_base = -&eta_threshold * &inv_sigma;
    let mut qdot =
        &inv_sigma * &(&eta_threshold * &eta_log_sigma_derivative - &eta_threshold_derivative);
    if let Some(beta_wiggle) = saved_fit.beta_link_wiggle() {
        let knots = link_wiggle_knots.as_ref().ok_or_else(|| {
            "saved location-scale link-wiggle coefficients are missing knots".to_string()
        })?;
        let degree = link_wiggle_degree.ok_or_else(|| {
            "saved location-scale link-wiggle coefficients are missing degree".to_string()
        })?;
        let derivative_basis = crate::wiggle::monotone_wiggle_basis_with_derivative_order(
            q_base.view(),
            knots,
            degree,
            1,
        )?;
        if derivative_basis.ncols() != beta_wiggle.len() {
            return Err(format!(
                "saved location-scale link-wiggle derivative width mismatch: design={}, beta={}",
                derivative_basis.ncols(),
                beta_wiggle.len()
            ));
        }
        qdot *= &(derivative_basis.dot(&beta_wiggle) + 1.0);
    }
    let eta_derivative_full = hdot + qdot;
    if eta_derivative_full
        .iter()
        .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(
            "saved location-scale survival event-rate derivative must be finite and positive"
                .to_string(),
        );
    }
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
            // Survival-curve origin: at t = 0 everyone is still at risk, so
            // S(0) = 1, H(0) = 0 and h(0) = 0 exactly, independent of the
            // fitted baseline. Anchor the origin column directly instead of
            // routing it through the (probit-survival) baseline, whose index is
            // -inf at S0(0) = 1. This matches the transformation / marginal-slope
            // predict path's `t <= 0` handling and keeps the default surface grid
            // — whose first node is the origin for the `Surv(time, event)`
            // right-censored shorthand — evaluable end to end (#1024).
            let query_time = if per_row_eval {
                age_exit[i]
            } else {
                eval_times[j]
            };
            if query_time <= 0.0 {
                *s = 1.0;
                *ch = 0.0;
                *h = 0.0;
                return;
            }
            let k = if per_row_eval { i } else { i * eval_width + j };
            let surv = survival_prob_full[k].clamp(SURVIVAL_PROB_MIN_FOR_LOG, 1.0);
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
        // Cloned (not moved) so the origin-column anchor below can still read the
        // per-column query times when assembling the survival standard errors.
        eval_times.clone()
    };

    let survival_se = response_se_full.as_ref().map(|response_se| {
        let mut out = Array2::<f64>::zeros((n, t_cols));
        ndarray::Zip::indexed(&mut out).par_for_each(|(i, j), slot| {
            // S(0) = 1 is a deterministic identity, so its standard error is 0
            // at the origin column (consistent with the anchored survival above).
            let query_time = if per_row_eval {
                age_exit[i]
            } else {
                eval_times[j]
            };
            if query_time <= 0.0 {
                *slot = 0.0;
                return;
            }
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
        covariance_source: with_uncertainty.then_some(covariance_mode),
    })
}

pub(crate) struct LocationScaleEtaComponents {
    pub h: Array1<f64>,
    pub time_jac: Array2<f64>,
    pub eta_t: Array1<f64>,
    pub eta_ls: Array1<f64>,
    pub inv_sigma: Array1<f64>,
}

pub(crate) struct LocationScaleTimeWarpComponents {
    pub(crate) h: Array1<f64>,
    pub(crate) time_jac: Array2<f64>,
    pub(crate) time_wiggle_dq: Option<Array1<f64>>,
}

pub(crate) fn location_scale_time_warp_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    fit: &UnifiedFitResult,
) -> Result<LocationScaleTimeWarpComponents, String> {
    let n = x_time_exit.nrows();
    if eta_time_offset_exit.len() != n {
        return Err("survival location-scale time-warp row mismatch across inputs".to_string());
    }
    let beta_time = fit.beta_time();
    if x_time_exit.ncols() != beta_time.len() {
        return Err(format!(
            "survival location-scale time-warp design mismatch: x_exit={} beta_time={}",
            x_time_exit.ncols(),
            beta_time.len()
        ));
    }

    let p_time_total = beta_time.len();
    let p_wiggle = time_wiggle_ncols.min(p_time_total);
    let p_base = p_time_total - p_wiggle;
    let beta_base = beta_time.slice(s![..p_base]).to_owned();
    let h_base = if p_base > 0 {
        x_time_exit.slice(s![.., ..p_base]).dot(&beta_base) + eta_time_offset_exit
    } else {
        eta_time_offset_exit.clone()
    };
    let mut h = h_base.clone();
    let mut time_jac = x_time_exit.clone();
    let mut time_wiggle_dq = None;
    if p_wiggle > 0 {
        if x_time_exit
            .slice(s![.., p_base..p_time_total])
            .iter()
            .any(|&value| value != 0.0)
        {
            return Err(
                "survival location-scale timewiggle prediction requires zero placeholder tail columns"
                    .to_string(),
            );
        }
        let knots = time_wiggle_knots.ok_or_else(|| {
            "survival location-scale time-warp: timewiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = time_wiggle_degree.ok_or_else(|| {
            "survival location-scale time-warp: timewiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let beta_w = beta_time.slice(s![p_base..p_time_total]).to_owned();
        let time_basis = crate::wiggle::monotone_wiggle_basis_with_derivative_order(
            h_base.view(),
            knots,
            degree,
            0,
        )?;
        let time_basis_d1 = crate::wiggle::monotone_wiggle_basis_with_derivative_order(
            h_base.view(),
            knots,
            degree,
            1,
        )?;
        if time_basis.ncols() != p_wiggle || time_basis_d1.ncols() != p_wiggle {
            return Err(format!(
                "survival location-scale time-warp timewiggle mismatch: value basis has {} columns, derivative basis has {}, beta has {}",
                time_basis.ncols(),
                time_basis_d1.ncols(),
                p_wiggle
            ));
        }
        let dq = time_basis_d1.dot(&beta_w) + 1.0;
        h = &h_base + &time_basis.dot(&beta_w);
        time_jac = Array2::<f64>::zeros((n, p_time_total));
        if p_base > 0 {
            let scaled_base = crate::survival::location_scale::scale_dense_rows(
                &x_time_exit.slice(s![.., ..p_base]).to_owned(),
                &dq,
            )?;
            time_jac.slice_mut(s![.., ..p_base]).assign(&scaled_base);
        }
        time_jac
            .slice_mut(s![.., p_base..p_time_total])
            .assign(&time_basis);
        time_wiggle_dq = Some(dq);
    }

    Ok(LocationScaleTimeWarpComponents {
        h,
        time_jac,
        time_wiggle_dq,
    })
}

pub(crate) fn location_scale_eta_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    x_threshold: &gam_linalg::matrix::DesignMatrix,
    eta_threshold_offset: &Array1<f64>,
    x_log_sigma: &gam_linalg::matrix::DesignMatrix,
    eta_log_sigma_offset: &Array1<f64>,
    fit: &UnifiedFitResult,
) -> Result<LocationScaleEtaComponents, String> {
    let n = x_time_exit.nrows();
    if x_threshold.nrows() != n
        || eta_threshold_offset.len() != n
        || x_log_sigma.nrows() != n
        || eta_log_sigma_offset.len() != n
    {
        return Err("survival location-scale eta component row mismatch across inputs".to_string());
    }
    let time_components = location_scale_time_warp_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        fit,
    )?;
    let beta_threshold = fit.beta_threshold();
    let beta_log_sigma = fit.beta_log_sigma();
    let eta_t = x_threshold.matrixvectormultiply(&beta_threshold) + eta_threshold_offset;
    let eta_ls = x_log_sigma.matrixvectormultiply(&beta_log_sigma) + eta_log_sigma_offset;
    let inv_sigma = eta_ls.mapv(crate::sigma_link::exp_sigma_inverse_from_eta_scalar);
    Ok(LocationScaleEtaComponents {
        h: time_components.h,
        time_jac: time_components.time_jac,
        eta_t,
        eta_ls,
        inv_sigma,
    })
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

    let time_components = location_scale_time_warp_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        fit,
    )?;
    let beta_base = beta_time.slice(s![..p_base]).to_owned();
    let mut eta_derivative = if p_base > 0 {
        x_time_derivative.dot(&beta_base) + derivative_offset_exit
    } else {
        derivative_offset_exit.clone()
    };
    if let Some(dq) = time_components.time_wiggle_dq.as_ref() {
        eta_derivative *= dq;
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
        InverseLink::Standard(StandardLink::Probit) => {
            let (_, hazard) = probit_survival_hazard_components(eta, eta_derivative)?;
            Ok(hazard)
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            let (_, hazard) = royston_parmar_survival_hazard_components(eta, eta_derivative)?;
            Ok(hazard)
        }
        InverseLink::Standard(StandardLink::Logit) => {
            let failure = if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let exp_eta = eta.exp();
                exp_eta / (1.0 + exp_eta)
            };
            Ok(failure * eta_derivative)
        }
        InverseLink::Standard(StandardLink::Identity) => {
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
) -> Result<SurvivalLikelihoodMode, SurvivalPredictError> {
    if matches!(&model.family_state, FittedFamily::LatentSurvival { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent") => Ok(SurvivalLikelihoodMode::Latent),
            Some(other) => Err(SurvivalPredictError::MissingFitMetadata { reason: format!(
                "saved latent survival model has contradictory survival_likelihood metadata: expected 'latent', got '{other}'"
            ) }),
            None => Err(SurvivalPredictError::MissingFitMetadata {
                reason:
                    "saved latent survival model is missing survival_likelihood=latent metadata; refit"
                        .to_string(),
            }),
        };
    }
    if matches!(&model.family_state, FittedFamily::LatentBinary { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent-binary") => Ok(SurvivalLikelihoodMode::LatentBinary),
            Some(other) => Err(SurvivalPredictError::MissingFitMetadata { reason: format!(
                "saved latent binary model has contradictory survival_likelihood metadata: expected 'latent-binary', got '{other}'"
            ) }),
            None => Err(SurvivalPredictError::MissingFitMetadata {
                reason:
                    "saved latent binary model is missing survival_likelihood=latent-binary metadata; refit"
                        .to_string(),
            }),
        };
    }
    let raw = model.survival_likelihood.as_deref().ok_or_else(|| {
        "saved survival model is missing survival_likelihood metadata; refit".to_string()
    })?;
    parse_survival_likelihood_mode(raw).map_err(SurvivalPredictError::from)
}

/// Baseline config persisted by the saved survival model.
pub fn saved_survival_runtime_baseline_config(
    model: &SavedModel,
) -> Result<SurvivalBaselineConfig, SurvivalPredictError> {
    survival_baseline_config_from_model(model).map_err(SurvivalPredictError::from)
}

/// Resolve the covariate `TermCollectionSpec` for prediction, remapping
/// saved training-column indices onto the runtime dataset's layout.
pub fn resolve_termspec_for_prediction(
    modelspec: &Option<TermCollectionSpec>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    spec_label: &str,
) -> Result<TermCollectionSpec, SurvivalPredictError> {
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
) -> Result<TermCollectionSpec, SurvivalPredictError> {
    // Delegate the (variant-exhaustive, easy-to-miss-a-field) walk to the
    // single shared authority on TermCollectionSpec; supply the survival
    // train→predict resolution as the per-index remap closure.
    spec.remap_feature_columns(|index| -> Result<usize, SurvivalPredictError> {
        let name = training_headers
            .get(index)
            .ok_or_else(|| format!("saved training column index {index} is out of bounds"))?;
        resolve_role_col(prediction_column_map, name, "prediction")
            .map_err(SurvivalPredictError::from)
    })
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
) -> Result<UnifiedFitResult, SurvivalPredictError> {
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
pub fn resolve_survival_inverse_link_from_saved(
    model: &SavedModel,
) -> Result<InverseLink, SurvivalPredictError> {
    if let Some(link) = model.link.as_ref() {
        return Ok(link.clone());
    }
    Err(SurvivalPredictError::MissingFitMetadata {
        reason: "saved survival model is missing link metadata; refit".to_string(),
    })
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
) -> Result<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>, SurvivalPredictError> {
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
                _ => {
                    return Err(SurvivalPredictError::IncompatibleSchema {
                        reason: "saved baseline-timewiggle entry design must be dense".to_string(),
                    });
                }
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
                _ => {
                    return Err(SurvivalPredictError::IncompatibleSchema {
                        reason: "saved baseline-timewiggle exit design must be dense".to_string(),
                    });
                }
            };
            let betaw = beta;
            if entry.ncols() != betaw.len() || exit.ncols() != betaw.len() {
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: format!(
                        "saved baseline-timewiggle dimension mismatch: coefficients have {} entries but basis has entry={} exit={}",
                        betaw.len(),
                        entry.ncols(),
                        exit.ncols()
                    ),
                });
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
                return Err(SurvivalPredictError::IncompatibleSchema {
                    reason: format!(
                        "saved baseline-timewiggle derivative dimension mismatch: coefficients have {} entries but derivative basis has {} columns",
                        betaw.len(),
                        derivative.ncols()
                    ),
                });
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
    SurvivalPredictError,
> {
    let saved_runtime = model.saved_prediction_runtime()?;
    if saved_runtime.link_wiggle.is_some() {
        return Err(SurvivalPredictError::MissingFitMetadata {
            reason:
                "saved survival marginal-slope model contains legacy linkwiggle metadata; refit with the anchored link-deviation runtime"
                    .to_string(),
        });
    }

    let saved_score_runtime = saved_runtime.score_warp;
    let saved_link_runtime = saved_runtime.link_deviation;
    // #461: the absorbed Stage-1 influence block (when present) is the trailing
    // block. Its `γ` is DROPPED at predict (the orthogonalized β̂ is a
    // training-fit property), so it is NOT read below — but it IS persisted, so
    // the saved block count includes it.
    let influence_absorber_width = saved_runtime.influence_absorber_width;
    let blocks = &fit_saved.blocks;
    let expected_blocks = 3
        + usize::from(saved_score_runtime.is_some())
        + usize::from(saved_link_runtime.is_some())
        + usize::from(influence_absorber_width.is_some());
    if blocks.len() != expected_blocks {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope model requires {} blocks [time, marginal, slope{}{}{}], got {}",
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
                if influence_absorber_width.is_some() {
                    ", influence-absorber(dropped)"
                } else {
                    ""
                },
                blocks.len(),
            ),
        });
    }

    let beta_time = &blocks[0].beta;
    let beta_marginal = &blocks[1].beta;
    let beta_logslope = &blocks[2].beta;
    if let Some(runtime) = saved_score_runtime.as_ref() {
        let beta = &blocks[3].beta;
        if beta.len() != runtime.basis_dim {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved survival marginal-slope score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                    beta.len(),
                    runtime.basis_dim
                ),
            });
        }
    }
    if let Some(runtime) = saved_link_runtime.as_ref() {
        let idx = 3 + usize::from(saved_score_runtime.is_some());
        let beta = &blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved survival marginal-slope link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                    beta.len(),
                    runtime.basis_dim
                ),
            });
        }
    }

    if beta_marginal.len() != cov_design.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope marginal coefficient mismatch: beta has {} entries but baseline design has {} columns",
                beta_marginal.len(),
                cov_design.ncols()
            ),
        });
    }
    if beta_logslope.len() != logslope_design.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope slope coefficient mismatch: beta has {} entries but slope design has {} columns",
                beta_logslope.len(),
                logslope_design.ncols()
            ),
        });
    }

    let p_time_base = time_build.x_exit_time.ncols();
    let saved_timewiggle = saved_runtime.baseline_time_wiggle;
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());
    if beta_time.len() != p_time_base + p_timewiggle {
        let hint = stale_weibull_time_basis_hint(
            &row_time.basisname,
            beta_time.len() == p_time_base + p_timewiggle + 1,
        );
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope time coefficient mismatch: beta has {} entries but expected base={} plus timewiggle={}{hint}",
                beta_time.len(),
                p_time_base,
                p_timewiggle
            ),
        });
    }

    let beta_time_base = beta_time.slice(s![..p_time_base]).to_owned();
    // `cov_design · beta_marginal` is row-only (no time dependence); hoist it
    // once so both the entry- and exit-time baselines share the single
    // matrix-vector multiply instead of recomputing it.
    let cov_eta_marginal = cov_design.dot(beta_marginal);
    let q_entry_base = time_build.x_entry_time.dot(&beta_time_base)
        + &cov_eta_marginal
        + eta_offset_entry
        + primary_offset;
    let q_exit_base = time_build.x_exit_time.dot(&beta_time_base)
        + &cov_eta_marginal
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
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved survival marginal-slope timewiggle design mismatch: rebuilt {} columns but runtime expects {}",
                    exit_w.ncols(),
                    p_timewiggle
                ),
            });
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
            .unwrap_or(InverseLink::Standard(StandardLink::Probit)),
        model
            .family_state
            .frailty()
            .cloned()
            .unwrap_or(FrailtySpec::None),
        saved_score_runtime,
        saved_link_runtime,
        model.latent_z_rank_int_calibration.clone(),
        // Survival marginal-slope never engages the BMS-only conditional Auto
        // gate (#905); the field is always `None` for survival fits.
        model.latent_z_conditional_calibration.clone(),
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

/// Typed hint appended to a survival time-coefficient / design mismatch when the
/// saved model looks like a pre-#2301 linear Weibull fit. The built-in Weibull
/// linear time basis dropped its redundant constant column (2 → 1 columns), so a
/// model saved before that change carries exactly one extra time coefficient
/// against the rebuilt 1-column basis. Naming it keeps the load path from
/// silently misindexing the stale constant coefficient as the shape.
fn stale_weibull_time_basis_hint(basisname: &str, extra_time_coefficient: bool) -> &'static str {
    if basisname == "linear" && extra_time_coefficient {
        " (this looks like a model saved before the #2301 Weibull time-basis \
         change, which removed the redundant constant column; refit the model)"
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::{normal_cdf, normal_pdf};

    #[test]
    fn competing_risks_covariance_mode_selects_exact_requested_matrix() {
        let conditional = ndarray::array![[1.0, 0.2], [0.2, 2.0]];
        let corrected = ndarray::array![[1.5, 0.4], [0.4, 3.0]];

        let selected_conditional = select_survival_prediction_covariance(
            Some(&conditional),
            Some(&corrected),
            SurvivalPredictionCovarianceMode::Conditional,
        )
        .expect("conditional covariance");
        let selected_corrected = select_survival_prediction_covariance(
            Some(&conditional),
            Some(&corrected),
            SurvivalPredictionCovarianceMode::SmoothingCorrected,
        )
        .expect("smoothing-corrected covariance");

        assert_eq!(selected_conditional, &conditional);
        assert_eq!(selected_corrected, &corrected);
        assert_eq!(
            SurvivalPredictionCovarianceMode::Conditional.as_str(),
            "conditional"
        );
        assert_eq!(
            SurvivalPredictionCovarianceMode::SmoothingCorrected.as_str(),
            "smoothing-corrected"
        );
    }

    #[test]
    fn competing_risks_smoothing_covariance_never_falls_back() {
        let conditional = ndarray::array![[1.0]];
        let error = select_survival_prediction_covariance(
            Some(&conditional),
            None,
            SurvivalPredictionCovarianceMode::SmoothingCorrected,
        )
        .expect_err("a corrected request must not substitute conditional covariance");
        assert_eq!(
            error.to_string(),
            "fit result does not contain smoothing-corrected covariance"
        );
    }

    #[test]
    fn posterior_quadrature_second_moment_honors_cross_coordinate_covariance() {
        let posterior_mean = ndarray::array![0.4, -0.2];
        let covariance = ndarray::array![[0.9, 0.35], [0.35, 0.6]];
        let mut functional_mean = 0.0_f64;
        let mut functional_second = 0.0_f64;
        let mut recovered_cross_covariance = 0.0_f64;

        for_each_survival_posterior_node(&posterior_mean, &covariance, |node, weight| {
            let functional = node[0] + 2.0 * node[1];
            functional_mean += weight * functional;
            functional_second += weight * functional * functional;
            recovered_cross_covariance +=
                weight * (node[0] - posterior_mean[0]) * (node[1] - posterior_mean[1]);
            Ok(())
        })
        .expect("joint posterior quadrature");

        let expected_mean = posterior_mean[0] + 2.0 * posterior_mean[1];
        let expected_variance =
            covariance[[0, 0]] + 4.0 * covariance[[1, 1]] + 4.0 * covariance[[0, 1]];
        assert!((functional_mean - expected_mean).abs() <= 1e-12);
        assert!((recovered_cross_covariance - covariance[[0, 1]]).abs() <= 1e-12);

        let mean_surface = Array2::from_elem((1, 1), functional_mean);
        let second_surface = Array2::from_elem((1, 1), functional_second);
        let standard_error = posterior_standard_error_matrix(
            &mean_surface,
            &second_surface,
            "joint-covariance witness",
        )
        .expect("posterior standard error");
        assert!((standard_error[[0, 0]].powi(2) - expected_variance).abs() <= 1e-11);
    }

    #[test]
    fn posterior_quadrature_zero_covariance_has_zero_standard_error() {
        let posterior_mean = ndarray::array![0.25, -0.75];
        let covariance = Array2::<f64>::zeros((2, 2));
        let mut functional_mean = 0.0_f64;
        let mut functional_second = 0.0_f64;
        let mut node_count = 0usize;

        for_each_survival_posterior_node(&posterior_mean, &covariance, |node, weight| {
            let functional = node[0].exp() + node[1].sin();
            functional_mean += weight * functional;
            functional_second += weight * functional * functional;
            node_count += 1;
            Ok(())
        })
        .expect("rank-zero posterior quadrature");

        assert_eq!(node_count, 1, "rank-zero covariance has one exact node");
        let standard_error = posterior_standard_error_matrix(
            &Array2::from_elem((1, 1), functional_mean),
            &Array2::from_elem((1, 1), functional_second),
            "rank-zero witness",
        )
        .expect("rank-zero posterior standard error");
        assert_eq!(standard_error[[0, 0]], 0.0);
    }

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
    fn probit_survival_hazard_accepts_zero_time_derivative_as_flat_hazard() {
        let (cum, hazard) =
            probit_survival_hazard_components(1.0, 0.0).expect("zero derivative is flat hazard");
        assert!(cum > 0.0);
        assert_eq!(hazard, 0.0);
    }

    #[test]
    fn marginal_slope_index_derivative_clamps_extrapolation_negative_to_flat_hazard() {
        // The #1040 end-to-end blocker: at a prediction horizon outside the
        // training exit times, the penalized baseline derivative q'(t) can dip
        // negative (e.g. the reported eta_t=-0.00135), producing a negative
        // index time-derivative the strict validator used to reject. The
        // physical hazard floor is 0, so the clamp must turn it into a flat
        // hazard the validator accepts — keeping predict/CIF runnable.
        let deta_dq = (1.0_f64 + 0.4 * 0.4).sqrt(); // rigid c = sqrt(1+sb^2) >= 1
        let qd_with_wiggle = -1.35e-3;
        let eta_t = marginal_slope_index_derivative_at_horizon(deta_dq, qd_with_wiggle);
        assert_eq!(
            eta_t, 0.0,
            "negative extrapolation derivative must clamp to 0"
        );
        // Downstream validator now accepts it as a flat-hazard point.
        let (cum, hazard) = probit_survival_hazard_components(-0.563, eta_t)
            .expect("clamped flat-hazard prediction must validate");
        assert!(
            cum >= 0.0,
            "cumulative hazard must be well-posed, got {cum}"
        );
        assert_eq!(
            hazard, 0.0,
            "clamped derivative gives zero instantaneous hazard"
        );
    }

    #[test]
    fn marginal_slope_index_derivative_preserves_positive_and_nonfinite() {
        // A genuinely positive derivative passes through unchanged (scaled by
        // the chain factor), and a non-finite value is left for the strict
        // validator to reject as a real numerical failure rather than masked.
        let positive = marginal_slope_index_derivative_at_horizon(1.25, 0.8);
        assert!(
            (positive - 1.0).abs() <= 1e-15,
            "positive derivative scaled by chain factor"
        );
        let nonfinite = marginal_slope_index_derivative_at_horizon(1.25, f64::NAN);
        assert!(
            nonfinite.is_nan(),
            "non-finite derivative passes through unclamped"
        );
        assert!(
            probit_survival_hazard_components(0.5, nonfinite).is_err(),
            "non-finite derivative must still be rejected by the validator"
        );
    }

    #[test]
    fn probit_survival_hazard_rejects_infinite_time_derivative() {
        let err = probit_survival_hazard_components(1.0, f64::INFINITY)
            .expect_err("infinite derivative should be invalid");
        assert!(
            err.to_string()
                .contains("invalid survival index derivative")
        );
    }

    #[test]
    fn probit_survival_hazard_rejects_nan_inputs() {
        // The upstream input gate is the only line that rejects NaN — the
        // output gate (`>= 0.0`) is dead-code for finite input because
        // `signed_probit_logcdf_and_mills_ratio` is provably NaN-free on the
        // finite domain (every internal branch clamps `erfcx`/`cdf` away from
        // zero). Pin both NaN slots so the input gate cannot regress.
        let err_eta =
            probit_survival_hazard_components(f64::NAN, 0.5).expect_err("NaN eta must be rejected");
        assert!(
            err_eta
                .to_string()
                .contains("invalid survival index derivative")
        );
        let err_dt = probit_survival_hazard_components(1.0, f64::NAN)
            .expect_err("NaN eta_derivative must be rejected");
        assert!(
            err_dt
                .to_string()
                .contains("invalid survival index derivative")
        );
    }

    #[test]
    fn probit_survival_hazard_rejects_negative_time_derivative() {
        // The CDF S(t) = Phi(-eta(t)) is monotone in t iff eta'(t) > 0. A
        // negative slope would give a non-monotone survival curve, which is
        // not a valid survival function.
        let err = probit_survival_hazard_components(1.0, -0.5)
            .expect_err("negative derivative should be invalid");
        assert!(
            err.to_string()
                .contains("invalid survival index derivative")
        );
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
    fn royston_parmar_hazard_rejects_negative_log_hazard_derivative() {
        // A negative time-derivative of log Λ(t) means a *decreasing* cumulative
        // hazard — not a valid survival model. Only the genuinely-negative slope
        // is rejected; the zero boundary is valid (see the sibling test below).
        let err = royston_parmar_survival_hazard_components(0.0, -0.5)
            .expect_err("negative derivative should be invalid");
        assert!(
            err.to_string()
                .contains("invalid log-cumulative-hazard derivative")
        );
    }

    #[test]
    fn royston_parmar_hazard_accepts_zero_derivative_as_flat_boundary() {
        // #1564: a monotone I-spline cumulative hazard is flat beyond its last
        // interior knot, so `d(log Λ)/dt == 0` exactly on any grid node past the
        // training support. That is a *valid* prediction (zero instantaneous
        // hazard, locally constant survival), not a numerical failure. The old
        // strict `> 0.0` gate rejected it and crashed saved-model RP predict.
        let eta = 1.9909019457445971_f64; // the exact η from the #1564 report
        let (cum, hazard) = royston_parmar_survival_hazard_components(eta, 0.0)
            .expect("zero derivative is a valid flat boundary, not an error");
        assert!((cum - eta.exp()).abs() <= 1e-12, "cum = Λ(t) = exp(η)");
        assert_eq!(
            hazard, 0.0,
            "flat cumulative hazard ⇒ zero instantaneous hazard"
        );
        // Survival is finite and well-defined at the boundary.
        let survival = (-cum).exp().clamp(0.0, 1.0);
        assert!(survival.is_finite() && (0.0..=1.0).contains(&survival));
    }

    #[test]
    fn royston_parmar_hazard_zero_derivative_in_saturated_tail_is_zero_not_nan() {
        // The dangerous corner: a saturated tail (η large ⇒ Λ = exp(η) = +∞) that
        // also lands past the I-spline support (derivative == 0). The naive
        // product `+∞ * 0.0` is `NaN`, which would (a) trip the components guard
        // and (b) serialize to JSON `null` and break the Python parse (#1564,
        // bug 1). The hazard must resolve to the mathematically correct `0`.
        let eta = 1000.0_f64;
        assert!(
            eta.exp().is_infinite(),
            "test premise: exp(1000) overflows to +∞"
        );
        assert!(
            (f64::INFINITY * 0.0).is_nan(),
            "test premise: the naive product is NaN"
        );
        let (cum, hazard) = royston_parmar_survival_hazard_components(eta, 0.0)
            .expect("saturated + flat boundary must be valid");
        assert!(cum.is_infinite() && cum > 0.0, "cum saturates to +∞");
        assert_eq!(hazard, 0.0, "hazard at a flat boundary is 0, never NaN");
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
        assert!(
            hazard.is_infinite() && hazard > 0.0,
            "expected +∞ hazard, got {hazard}"
        );

        // Consumer materializes survival via exp(-cum).clamp(0,1).
        let survival = (-cum).exp().clamp(0.0, 1.0);
        assert_eq!(survival, 0.0, "saturated cum_hazard must give survival 0");
    }

    #[test]
    fn royston_parmar_hazard_rejects_nan_eta() {
        let err = royston_parmar_survival_hazard_components(f64::NAN, 0.5)
            .expect_err("NaN eta should be invalid");
        assert!(
            err.to_string()
                .contains("invalid log-cumulative-hazard derivative")
        );
    }

    #[test]
    fn royston_parmar_hazard_left_tail_collapses_to_zero() {
        // η = log Λ(t); η → -∞ means Λ(t) → 0, so cum_hazard underflows to 0
        // and hazard rate underflows to 0. Survival → 1. No error.
        let eta = -1000.0_f64;
        let eta_t = 2.0_f64;
        assert_eq!(eta.exp(), 0.0, "test premise: exp(-1000) underflows to 0");

        let (cum, hazard) = royston_parmar_survival_hazard_components(eta, eta_t)
            .expect("RP left tail must remain valid");
        assert_eq!(
            cum, 0.0,
            "left-tail cum_hazard should underflow to 0, got {cum}"
        );
        assert_eq!(
            hazard, 0.0,
            "left-tail hazard should underflow to 0, got {hazard}"
        );

        // Consumer: survival = exp(-0) = 1.
        let survival = (-cum).exp().clamp(0.0, 1.0);
        assert_eq!(survival, 1.0);
    }

    #[test]
    fn probit_survival_hazard_left_tail_collapses_to_zero() {
        // η→-∞ mirror of the right-tail test: survival → 1, hazard → 0.
        // Asymptote: Mills(η) = φ(η)/Φ(-η) → 0 as η → -∞ (φ underflows,
        // Φ(-η) → 1).  No error, no NaN, no spurious negativity.
        let eta = -40.0_f64;
        let eta_t = 1.5_f64;

        let (cum, hazard) =
            probit_survival_hazard_components(eta, eta_t).expect("left tail must remain valid");
        assert!(
            (0.0..1e-300).contains(&cum),
            "left-tail cum should be ~0, got {cum}"
        );
        assert_eq!(
            hazard, 0.0,
            "left-tail hazard should underflow to 0, got {hazard}"
        );
    }

    #[test]
    fn location_scale_logit_hazard_is_failure_slope_over_survival() {
        let eta = 0.7;
        let eta_t = 0.4;

        let hazard = location_scale_hazard_component(
            eta,
            eta_t,
            &InverseLink::Standard(StandardLink::Logit),
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
            &InverseLink::Standard(StandardLink::CLogLog),
        )
        .expect("valid cloglog hazard");

        assert!((hazard - eta.exp() * eta_t).abs() <= 1e-14);
    }

    // ---- IPCW Brier score (Graf et al. 1999) -------------------------------

    #[test]
    fn kaplan_meier_censoring_is_right_continuous_step() {
        // Two censorings (events flipped) at t=4 and t=8; deaths at t=2,6.
        let time = [2.0, 4.0, 6.0, 8.0];
        let event = [1.0, 0.0, 1.0, 0.0];
        let g = KaplanMeier::fit_censoring(&time, &event);
        // Before the first censoring the censoring-survival is 1.
        assert!((g.at(0.0) - 1.0).abs() <= 1e-15);
        assert!((g.at(2.0) - 1.0).abs() <= 1e-15);
        assert!((g.at(3.999) - 1.0).abs() <= 1e-15);
        // At t=4 the at-risk set {4,6,8} loses one to censoring: G = 2/3.
        assert!((g.at(4.0) - 2.0 / 3.0).abs() <= 1e-12);
        assert!((g.at(5.0) - 2.0 / 3.0).abs() <= 1e-12);
        // A death at t=6 does not move the censoring KM.
        assert!((g.at(6.0) - 2.0 / 3.0).abs() <= 1e-12);
        // At t=8 the last (sole) at-risk subject is censored: G collapses to 0.
        assert!(g.at(8.0).abs() <= 1e-15);
    }

    #[test]
    fn ipcw_brier_no_censoring_reduces_to_plain_brier() {
        // With no censoring G(t) ≡ 1, so the IPCW Brier is the ordinary Brier of
        // the predicted survival against the alive-indicator I(T_i > tau).
        let s_pred = [0.3, 0.7, 0.6, 0.2];
        let time = [2.0, 8.0, 10.0, 3.0];
        let event = [1.0, 1.0, 0.0, 1.0];
        let tau = 5.0;
        let g = KaplanMeier::fit_censoring(&time, &event);
        let bs = ipcw_brier_score(&s_pred, &time, &event, tau, |t| g.at(t)).unwrap();
        // targets: dead→0 (subj1,4), alive→1 (subj2,3).
        let expected =
            (0.3f64.powi(2) + (1.0 - 0.7f64).powi(2) + (1.0 - 0.6f64).powi(2) + 0.2f64.powi(2))
                / 4.0;
        assert!(
            (bs - expected).abs() <= 1e-12,
            "bs={bs} expected={expected}"
        );
    }

    #[test]
    fn ipcw_brier_reweights_by_inverse_censoring_probability() {
        // Hand-computed Graf estimator with real censoring weights.
        // times/events: death@2, cens@4, death@6, cens@8; tau=5.
        // Censoring KM: G(5)=2/3 (one censoring at t=4 among {4,6,8}); G(2)=1.
        let s_pred = [0.4, 0.5, 0.7, 0.8];
        let time = [2.0, 4.0, 6.0, 8.0];
        let event = [1.0, 0.0, 1.0, 0.0];
        let tau = 5.0;
        let g = KaplanMeier::fit_censoring(&time, &event);
        let bs = ipcw_brier_score(&s_pred, &time, &event, tau, |t| g.at(t)).unwrap();
        // subj1 dead by 5: weight 1/G(2)=1, contrib 0.4²=0.16.
        // subj2 censored before 5: contributes 0.
        // subj3 alive: weight 1/G(5)=1.5, contrib 1.5·0.3²=0.135.
        // subj4 alive: weight 1/G(5)=1.5, contrib 1.5·0.2²=0.06.
        let expected = (0.16 + 0.0 + 0.135 + 0.06) / 4.0;
        assert!(
            (bs - expected).abs() <= 1e-12,
            "bs={bs} expected={expected}"
        );
    }

    #[test]
    fn ipcw_brier_drops_invalid_rows_from_both_numerator_and_denominator() {
        // A NaN-time row and a non-positive-time row must not be counted at all.
        let s_pred = [0.3, 0.7, 0.5, 0.5];
        let time = [2.0, 8.0, f64::NAN, -1.0];
        let event = [1.0, 1.0, 1.0, 0.0];
        let g = KaplanMeier::fit_censoring(&time, &event);
        let bs = ipcw_brier_score(&s_pred, &time, &event, 5.0, |t| g.at(t)).unwrap();
        // Only subj1 (dead, contrib 0.3²) and subj2 (alive, contrib 0.3²) count;
        // censoring KM has no censorings so G≡1.
        let expected = (0.3f64.powi(2) + (1.0 - 0.7f64).powi(2)) / 2.0;
        assert!(
            (bs - expected).abs() <= 1e-12,
            "bs={bs} expected={expected}"
        );
    }

    #[test]
    fn integrated_ipcw_brier_of_constant_brier_is_that_constant() {
        // A survival matrix whose every column equals a perfect classifier yields
        // BS(t)=0 at every grid point, so the integral is 0.
        let time = [2.0, 8.0, 10.0, 3.0];
        let event = [1.0, 1.0, 0.0, 1.0];
        let grid = [0.0, 1.0, 2.5, 4.0, 6.0];
        // Perfect prediction at every grid time given the (no-censoring) data is
        // not generally achievable, so instead test the integral of a literally
        // constant-in-time Brier: replicate one column across the grid.
        let col = [0.3, 0.7, 0.6, 0.2];
        let mut surv = Array2::<f64>::zeros((4, grid.len()));
        for k in 0..grid.len() {
            for i in 0..4 {
                surv[[i, k]] = col[i];
            }
        }
        let g = KaplanMeier::fit_censoring(&time, &event);
        let per_time = ipcw_brier_score(&col, &time, &event, grid[2], |t| g.at(t)).unwrap();
        // Because the predicted survival is identical at every grid time, BS(t)
        // is *not* constant (tau changes which subjects are "alive"), so use a
        // direct trapezoid as the oracle.
        let mut oracle_pts = Vec::new();
        for k in 0..grid.len() {
            oracle_pts.push((
                grid[k],
                ipcw_brier_score(&col, &time, &event, grid[k], |t| g.at(t)).unwrap(),
            ));
        }
        let mut integral = 0.0;
        for w in oracle_pts.windows(2) {
            integral += 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0);
        }
        let oracle = integral / (grid[grid.len() - 1] - grid[0]);
        let ibs =
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, |t| {
                g.at(t)
            })
            .unwrap();
        assert!((ibs - oracle).abs() <= 1e-12, "ibs={ibs} oracle={oracle}");
        // Sanity: per-time value is in a sensible [0,1]-ish range.
        assert!(per_time >= 0.0);
    }

    #[test]
    fn integrated_ipcw_brier_respects_the_horizon_cutoff() {
        let time = [2.0, 8.0, 10.0, 3.0];
        let event = [1.0, 1.0, 0.0, 1.0];
        let grid = [0.0, 2.0, 4.0, 100.0];
        let col = [0.3, 0.7, 0.6, 0.2];
        let mut surv = Array2::<f64>::zeros((4, grid.len()));
        for k in 0..grid.len() {
            for i in 0..4 {
                surv[[i, k]] = col[i];
            }
        }
        let g = KaplanMeier::fit_censoring(&time, &event);
        // Horizon 5 drops the extrapolation point at t=100: integral runs [0,4].
        let restricted =
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, 5.0, |t| g.at(t))
                .unwrap();
        let full =
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, |t| {
                g.at(t)
            })
            .unwrap();
        // The huge [4,100] tail interval dominates the full integral, so the two
        // must differ substantially — the horizon guard is doing real work.
        assert!(
            (restricted - full).abs() > 1e-3,
            "horizon cutoff had no effect: restricted={restricted} full={full}"
        );
    }

    #[test]
    fn integrated_ipcw_brier_rejects_malformed_grids() {
        let time = [2.0, 8.0];
        let event = [1.0, 0.0];
        let surv = Array2::<f64>::from_elem((2, 3), 0.5);
        let g = KaplanMeier::fit_censoring(&time, &event);
        // Non-increasing grid.
        let bad = [0.0, 2.0, 1.0];
        assert!(
            integrated_ipcw_brier_score(surv.view(), &time, &event, &bad, f64::INFINITY, |t| g
                .at(t))
            .is_none()
        );
        // Grid width mismatched to the survival matrix.
        let short = [0.0, 1.0];
        assert!(
            integrated_ipcw_brier_score(surv.view(), &time, &event, &short, f64::INFINITY, |t| g
                .at(t))
            .is_none()
        );
    }
}
