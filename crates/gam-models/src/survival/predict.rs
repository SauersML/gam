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
    FittedFamily, FittedModel as SavedModel, FittedModelPayload,
    SavedBaselineTimeWiggleRuntime, load_survival_time_basis_config_from_model,
    survival_baseline_config_from_model,
};
use gam_data::EncodedDataset;
use crate::inference::predict_io::{
    BernoulliMarginalSlopePredictor, LatentConditioningSpan, PredictInput,
};
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
use crate::survival::location_scale::{
    SurvivalCovariateTimeBasis, TruncatedCoefficientDraws, build_truncated_coefficient_draws,
    replicate_standard_error,
};
use crate::survival::latent::fixed_latent_hazard_frailty;
use crate::survival::lognormal_kernel::FrailtySpec;
use crate::survival::{
    CompetingRisksCifResult, assemble_competing_risks_cif_from_endpoints_with_rounding,
};
use crate::wiggle::monotone_wiggle_basis_with_derivative_order;
use gam_linalg::matrix::DesignMatrix;
use gam_math::probability::normal_pdf;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::mixture_link::inverse_link_jet_for_inverse_link;
use gam_terms::smooth::TermCollectionSpec;
use gam_terms::smooth::build_term_collection_design;
use gam_terms::term_builder::resolve_role_col;

mod bands;
mod truncated_law;
use bands::*;
use truncated_law::*;
pub use bands::{BandRefusal, CompetingRisksBands};

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
    /// The reported survival curve would increase at a requested cell,
    /// `dH/dt < 0`, so no non-negative hazard is the derivative of the
    /// cumulative hazard reported beside it (gam#3026). The fit's likelihood is
    /// defined only where the survival index increases, so the cell is refused
    /// rather than published with a hazard that belongs to a different curve.
    DecreasingSurvival { reason: String },
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
            | SurvivalPredictError::NumericalFailure { reason }
            | SurvivalPredictError::DecreasingSurvival { reason } => f.write_str(reason),
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
            | SurvivalPredictError::NumericalFailure { .. }
            | SurvivalPredictError::DecreasingSurvival { .. } => None,
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
#[derive(Clone, Copy)]
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
    /// The plug-in surface `S(eta_hat)` when `survival` carries the posterior
    /// mean `E[S(eta) | data]`; `None` when `survival` IS the plug-in (the
    /// [`SurvivalPredictEstimand::Plugin`] request).
    ///
    /// The posterior-mean path builds the plug-in prediction first and then
    /// replaces its surfaces with the quadrature means, so this costs one
    /// clone per CALL and no extra prediction. Publishing it means a presenter
    /// reports both estimands BY NAME rather than asking for one with a mode:
    /// `gam predict` has published `survival_prob_plugin` beside
    /// `survival_prob` since #2670, and the Python payload could not, because
    /// the surface it integrated was discarded here.
    pub survival_plugin: Option<Array2<f64>>,
    /// Lower and upper ends of the central posterior interval of `survival`,
    /// cell by cell, at the level [`predict_survival_with_band`] was asked for
    /// (gam#3560). They are quantiles of the posterior law of `S` itself — the
    /// law whose mean `survival` publishes — so each end is a value `S`
    /// attains and nothing is clamped; `mean ± z·survival_se` is not an
    /// interval of that law. `None` unless the band was requested.
    pub survival_lower: Option<Array2<f64>>,
    pub survival_upper: Option<Array2<f64>>,
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
    /// `P(T > t | T > entry, x)` for every row at every requested grid time `t`,
    /// one column per time, when the request carries a time grid; `None`
    /// otherwise.
    pub grid_survival: Option<Array2<f64>>,
    pub likelihood_mode: SurvivalLikelihoodMode,
}

/// Evaluate the saved latent hazard-multiplier law over the rows' own windows,
/// and over every requested grid time.
///
/// This is the library authority for both `latent` and `latent-binary` saved
/// models. It replays the persisted covariate design, anchored time basis,
/// loaded/unloaded baseline decomposition, fitted mean/time coefficients, and
/// fixed lognormal hazard multiplier. No response column, refit, or surrogate
/// family participates in the calculation. A grid time `t` closes each row's
/// window at `t` instead of at its exit, so a row with no entry column gives the
/// fitted survival curve `S(t | x)`.
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
    if let Some(grid) = time_grid {
        if grid.is_empty() {
            return Err(SurvivalPredictError::InvalidInput {
                reason: "latent-window time_grid must contain at least one time".to_string(),
            });
        }
        if let Some(index) = grid.iter().position(|time| !(time.is_finite() && *time >= 0.0)) {
            return Err(SurvivalPredictError::InvalidInput {
                reason: format!(
                    "latent-window time_grid requires finite non-negative times (index {index})"
                ),
            });
        }
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
    let mut raw_entry = Array1::<f64>::zeros(n);
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        raw_entry[row] = time_columns.row_entry_time(data, row);
        let (entry, exit) = normalize_survival_time_pair(
            raw_entry[row],
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }

    let time_config = load_survival_time_basis_config_from_model(model)?;
    let time_anchor =
        model
            .survival_time_anchor
            .ok_or_else(|| SurvivalPredictError::MissingFitMetadata {
                reason: "saved latent-window model is missing survival_time_anchor".to_string(),
            })?;
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

    let eta = covariate_design.design.dot(&mean_block.beta) + &effective_primary_offset;
    let quadrature = gam_solve::quadrature::QuadratureContext::new();
    // The saved law over one window per row: the anchored time basis and the
    // loaded/unloaded offsets realized at `(entry_i, exit_i)`, then each row's
    // exact conditional survival.
    let windows = |entry: &Array1<f64>,
                   exit: &Array1<f64>|
     -> Result<Array1<f64>, SurvivalPredictError> {
        let mut time_build = build_survival_time_basis(entry, exit, time_config.clone(), None)?;
        let resolved_time_config = resolved_survival_time_basis_config_from_build(
            &time_build.basisname,
            time_build.degree,
            time_build.knots.as_ref(),
            time_build.keep_cols.as_ref(),
        )?;
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
        let prepared = prepare_survival_time_stack(
            entry,
            exit,
            &baseline_config,
            likelihood_mode,
            None,
            time_anchor,
            survival_derivative_guard_for_likelihood(likelihood_mode),
            &time_build,
            None,
            Some(loading),
        )?;
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
        let q_entry =
            prepared.time_design_entry.dot(&time_block.beta) + &prepared.eta_offset_entry;
        let q_exit = prepared.time_design_exit.dot(&time_block.beta) + &prepared.eta_offset_exit;
        let mut survival = Array1::<f64>::zeros(entry.len());
        for row in 0..entry.len() {
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
            let value = jet.log_lik.exp();
            if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
                return Err(SurvivalPredictError::NumericalFailure {
                    reason: format!(
                        "latent-window row {row} produced invalid conditional survival {value}"
                    ),
                });
            }
            survival[row] = value;
        }
        Ok(survival)
    };
    let window_survival = windows(&age_entry, &age_exit)?;
    // One grid column at a time: each row's window closes at the grid time, so
    // memory stays at one window per row however long the grid.
    let grid_survival = time_grid
        .map(|grid| -> Result<Array2<f64>, SurvivalPredictError> {
            let mut grid_survival = Array2::<f64>::zeros((n, grid.len()));
            let mut grid_entry = Array1::<f64>::zeros(n);
            let mut grid_exit = Array1::<f64>::zeros(n);
            for (column, &time) in grid.iter().enumerate() {
                for row in 0..n {
                    if time < age_entry[row] {
                        return Err(SurvivalPredictError::InvalidInput {
                            reason: format!(
                                "latent-window grid time {time} precedes row {row}'s entry {}; P(T > t | T > entry) needs t >= entry",
                                age_entry[row]
                            ),
                        });
                    }
                    let (entry, exit) = normalize_survival_time_pair(raw_entry[row], time, row)?;
                    grid_entry[row] = entry;
                    grid_exit[row] = exit;
                }
                grid_survival
                    .column_mut(column)
                    .assign(&windows(&grid_entry, &grid_exit)?);
            }
            Ok(grid_survival)
        })
        .transpose()?;

    Ok(LatentWindowSurvivalResult {
        window_survival,
        grid_survival,
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
) -> Result<(Array1<f64>, Array2<f64>, Vec<usize>), SurvivalPredictError> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    // A fit saved as its constrained mode under a typed posterior-moment
    // decline has no posterior to integrate; refuse by the decline's own
    // reason rather than by the covariance it therefore lacks (gam#3008).
    fit.require_posterior_mean("survival posterior-mean prediction")
        .map_err(|error| SurvivalPredictError::PosteriorCovariance {
            reason: error.to_string(),
        })?;
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
    let cone_coords = survival_posterior_cone_coordinates(model, active_len)?;
    Ok((
        fit.beta.clone(),
        covariance.slice(s![..active_len, ..active_len]).to_owned(),
        cone_coords,
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

    let (beta_time, beta_time_blocks) = {
        let fit = payload.fit_result.as_mut().ok_or_else(|| {
            SurvivalPredictError::MissingFitMetadata {
                reason: "saved survival model is missing canonical fit_result".to_string(),
            }
        })?;
        assign_survival_fit_coefficients(fit, coefficients)?;
        (
            fit.block_by_role(BlockRole::Time)
                .map(|block| block.beta.to_vec()),
            fit.blocks
                .iter()
                .map(|block| block.beta.to_vec())
                .collect::<Vec<_>>(),
        )
    };

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

/// Set a fit's joint coefficient vector and every block's slice of it.
fn assign_survival_fit_coefficients(
    fit: &mut UnifiedFitResult,
    coefficients: &Array1<f64>,
) -> Result<(), SurvivalPredictError> {
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
        if end > coefficients.len() {
            break;
        }
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
    Ok(())
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
    // The node's hazard is signed where its law's survival rises
    // ([`predict_survival_coefficient_law`]); its density `S·h` keeps that sign.
    if cumulative_hazard.is_finite() && !hazard.is_nan() {
        return Ok(hazard.signum() * (hazard.abs().ln() - cumulative_hazard).exp());
    }
    if cumulative_hazard == f64::INFINITY && hazard.is_finite() {
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
///
/// `cone_coords` names the coefficient positions (indices into the active
/// subspace `0..active_len`) that were constrained to the nonnegativity cone
/// `β_j ≥ 0` when the fit was certified — the structural monotone-I-spline
/// baseline time columns of a Royston-Parmar survival fit. The parameter space
/// of such a model is the cone `C = {β_j ≥ 0 : j ∈ cone}`, so the Laplace
/// posterior is `N(β̂, Vb)` **truncated to `C`**, not the untruncated Gaussian.
/// Its quadrature nodes must lie in `C`; a node that pokes a structural time
/// coefficient below zero manufactures a non-monotone baseline log-cumulative
/// hazard whose derivative the plugin evaluator then (correctly) refuses
/// (`royston_parmar_survival_hazard_components`, #2375). For each factor
/// direction we therefore shrink the symmetric ±step to the largest value that
/// keeps BOTH nodes feasible (the standard fraction-to-boundary rule):
///
/// ```text
///   α_k = min( √rank,  min_{j ∈ cone, f_{j,k} ≠ 0}  β̂_j / |f_{j,k}| )
///   nodes = β̂ ± α_k · f_{·,k},   weight 1/(2·rank)  (unchanged)
/// ```
///
/// This keeps the rule symmetric about `β̂` (so it stays exact for linear
/// functionals and leaves the posterior mean unbiased), keeps every node inside
/// `C` by construction, and represents `(α_k/√rank)² · Vb` of the spread along
/// a constrained direction — the right direction of travel, since a truncated
/// Gaussian genuinely has smaller variance than its untruncated parent. Passing
/// an empty `cone_coords` recovers the exact unconstrained rule verbatim.
fn for_each_survival_posterior_node<F>(
    posterior_mean: &Array1<f64>,
    active_covariance: &Array2<f64>,
    cone_coords: &[usize],
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
    let nominal_scale = (rank as f64).sqrt();
    let weight = 1.0 / (2 * rank) as f64;
    for column in 0..rank {
        // Fraction-to-boundary step for the cone `{β_j ≥ 0 : j ∈ cone}`. Each
        // symmetric node is `β̂ ± scale · f_{·,column}`; feasibility of both
        // nodes at coordinate `j` requires `|scale · f_{j,column}| ≤ β̂_j`, i.e.
        // `scale ≤ β̂_j / |f_{j,column}|`. `β̂_j` is clamped at 0 so a coordinate
        // already numerically pinned at the wall collapses that direction's
        // step to 0 rather than admitting an infeasible (negative-step) node.
        let mut scale = nominal_scale;
        for &j in cone_coords {
            if j >= active_len {
                continue;
            }
            let load = factorization.factor[[j, column]].abs();
            if load == 0.0 {
                continue;
            }
            let limit = posterior_mean[j].max(0.0) / load;
            if limit < scale {
                scale = limit;
            }
        }
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

/// Structural-monotonicity cone coordinates for a saved survival model's
/// posterior quadrature — the coefficient positions the fit constrained to
/// `β_j ≥ 0` (the leading I-spline baseline time columns, per cause).
///
/// Only the transformation (Royston-Parmar) family carries a *coordinate* cone:
/// the fit realizes structural monotonicity as a per-coordinate lower-bound box
/// `lb[j] = 0` over the leading `p_time_base + p_timewiggle` columns of every
/// cause block (`fit_survival_transformation_model` /
/// `fit_cause_specific_survival_transformation_custom`). Weibull carries a
/// parametric `log t` baseline with no structural cone; marginal-slope enforces
/// monotonicity with row-wise (not coordinate) constraints; location-scale and
/// latent posteriors are not routed through this quadrature. In all those cases
/// this returns an empty cone, recovering the untruncated quadrature verbatim.
fn survival_posterior_cone_coordinates(
    model: &SavedModel,
    active_len: usize,
) -> Result<Vec<usize>, SurvivalPredictError> {
    if require_saved_survival_likelihood_mode(model)? != SurvivalLikelihoodMode::Transformation {
        return Ok(Vec::new());
    }
    // Baseline I-spline width (time-independent: it is fixed by the saved knots
    // / kept columns, not the evaluation times). The timewiggle arm saves the
    // base basis as `None`, in which case the whole learned time block is the
    // monotone wiggle tail counted separately below.
    let time_cfg = load_survival_time_basis_config_from_model(model)
        .map_err(|err| SurvivalPredictError::MissingFitMetadata {
            reason: err.to_string(),
        })?;
    let p_time_base = if matches!(
        time_cfg,
        crate::survival::construction::SurvivalTimeBasisConfig::None
    ) {
        0
    } else {
        let dummy = Array1::from_elem(1, 1.0_f64);
        build_survival_time_basis(&dummy, &dummy, time_cfg, None)
            .map_err(|reason| SurvivalPredictError::MissingFitMetadata { reason })?
            .x_exit_time
            .ncols()
    };

    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let cause_count = model
        .survival_cause_count
        .unwrap_or(fit.blocks.len())
        .max(1);
    // Per-cause monotone timewiggle width (0 when the model carries none). The
    // cone spans the leading `p_time_base + p_timewiggle` coefficients of each
    // cause block — exactly the structural columns the fit lower-bounds at 0.
    let per_cause_wiggle: Vec<usize> = if cause_count > 1 {
        saved_cause_specific_timewiggles(model, &fit, cause_count)?
            .iter()
            .map(|w| w.as_ref().map_or(0, |runtime| runtime.beta.len()))
            .collect()
    } else {
        vec![
            model
                .saved_baseline_time_wiggle()
                .map_err(|err| SurvivalPredictError::MissingFitMetadata {
                    reason: err.to_string(),
                })?
                .map_or(0, |runtime| runtime.beta.len()),
        ]
    };

    let mut cone = Vec::new();
    let mut cursor = 0usize;
    for (cause, block) in fit.blocks.iter().enumerate() {
        let block_len = block.beta.len();
        let width = (p_time_base + per_cause_wiggle.get(cause).copied().unwrap_or(0)).min(block_len);
        for j in cursor..cursor + width {
            if j < active_len {
                cone.push(j);
            }
        }
        cursor += block_len;
    }
    Ok(cone)
}

/// The weight, mean and centred second moment `Σ wᵢ (fᵢ − f̄)²` of one
/// posterior functional over a weighted node rule.
///
/// Nodes are merged by the Chan–Golub–LeVeque pairwise update, so the centred
/// moment is a sum of non-negative terms: the variance is `≥ 0` by
/// construction and exactly `0` for a functional that is constant over the
/// nodes, at any node count. Recovering it as `E f² − (E f)²` instead cancels
/// catastrophically, with a rounding error that grows with the node count and
/// that no fixed multiple of machine epsilon bounds (gam#4086).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PosteriorMoment {
    weight: f64,
    mean: f64,
    centered: f64,
}

impl PosteriorMoment {
    /// No nodes yet.
    pub(crate) const EMPTY: Self = Self {
        weight: 0.0,
        mean: 0.0,
        centered: 0.0,
    };

    /// One node carrying the value `value`.
    pub(crate) fn point(value: f64) -> Self {
        Self {
            weight: 1.0,
            mean: value,
            centered: 0.0,
        }
    }

    /// A unit-weight law with the given mean and (centred, non-negative)
    /// variance, as a closed-form producer reports it.
    fn from_mean_variance(mean: f64, variance: f64) -> Self {
        Self {
            weight: 1.0,
            mean,
            centered: variance,
        }
    }

    /// Add `other` with its weight multiplied by `scale ≥ 0`.
    pub(crate) fn merge(&mut self, scale: f64, other: Self) {
        let added = scale * other.weight;
        if added == 0.0 {
            return;
        }
        if self.weight == 0.0 {
            *self = Self {
                weight: added,
                mean: other.mean,
                centered: scale * other.centered,
            };
            return;
        }
        let total = self.weight + added;
        let difference = other.mean - self.mean;
        self.centered +=
            scale * other.centered + difference * difference * (self.weight * added / total);
        // The incremental form `f̄ + d·w/W` keeps the mean exact for a constant
        // functional; an infinite difference (a node at ±∞) falls back to the
        // convex combination, which carries the infinity without `∞ − ∞`.
        self.mean = if difference.is_finite() {
            self.mean + difference * (added / total)
        } else {
            self.mean * (self.weight / total) + other.mean * (added / total)
        };
        self.weight = total;
    }

    /// The same law with every node weight multiplied by `factor > 0`.
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self {
            weight: self.weight * factor,
            mean: self.mean,
            centered: self.centered * factor,
        }
    }

    pub(crate) fn mean(self) -> f64 {
        self.mean
    }

    pub(crate) fn variance(self) -> f64 {
        self.centered / self.weight
    }
}

impl gam_solve::quadrature::GhqValue for PosteriorMoment {
    fn zero() -> Self {
        Self::EMPTY
    }

    fn addweighted(&mut self, weight: f64, value: Self) {
        self.merge(weight, value);
    }

    fn scale(self, factor: f64) -> Self {
        self.scaled(factor)
    }
}

/// The posterior means of a moment surface.
fn posterior_moment_means<D: ndarray::Dimension>(
    moments: &ndarray::Array<PosteriorMoment, D>,
) -> ndarray::Array<f64, D> {
    moments.map(|moment| moment.mean())
}

/// The posterior standard deviations of a moment surface. The centred variance
/// is non-negative by construction, so the only refusal is a moment that is not
/// finite.
fn posterior_standard_errors<D: ndarray::Dimension>(
    moments: &ndarray::Array<PosteriorMoment, D>,
    label: &str,
) -> Result<ndarray::Array<f64, D>, SurvivalPredictError>
where
    D::Pattern: std::fmt::Debug,
{
    if let Some((index, moment)) = moments
        .indexed_iter()
        .find(|(_, moment)| !(moment.mean().is_finite() && moment.variance().is_finite()))
    {
        return Err(SurvivalPredictError::NumericalFailure {
            reason: format!(
                "posterior {label} moments must be finite at cell {index:?}: mean={}, variance={}",
                moment.mean(),
                moment.variance(),
            ),
        });
    }
    Ok(moments.map(|moment| moment.variance().sqrt()))
}

/// How the coefficient posterior enters the posterior-mean surfaces of a
/// single-event survival prediction: `E_θ[S(t; θ)]`, and beside it the event
/// density and the linear predictor's moments, under `θ ~ N(θ̂, V)`.
///
/// The variants exist so the rule that is not the default stays reachable by
/// name for comparison, not as alternatives a caller should prefer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalPosteriorIntegration {
    /// The `2·rank` symmetric sigma-point rule over the whole coefficient
    /// posterior, every node replayed through the plug-in prediction (anchor,
    /// timewiggle and flexible runtimes included). It is exact for cubic
    /// functionals of `θ` only; on the #2765 survival marginal-slope fixture it
    /// sat 4.1e-4 off a Monte Carlo reference at the fitted covariance and
    /// 2.0e-3 at 9× it. Every model the exact rule does not cover uses it.
    SigmaPoint,
    /// Exact for a survival marginal-slope model on a single latent score
    /// without a timewiggle, score warp or link deviation; a model anchored on
    /// the joint law of several scores reads a slope per score, so it is not a
    /// function of two primaries. `S(t) = Φ(−η(q(t), b(t)))` depends on `θ` only
    /// through the two primaries, both affine in `θ` (the slope's follow-up
    /// margin included), so their joint law is exactly bivariate Gaussian and
    /// the posterior mean is adaptive Gauss–Hermite over it with the anchor
    /// re-solved at every node. The event density `φ(η)·η′` also reads the
    /// tangents `(q′(t), b′(t))`: given the primaries they are Gaussian and
    /// `η′ = η_q·q′ + η_b·b′` is linear in them, so they enter through their
    /// conditional mean alone ([`exact_anchor_node_moments`]), and the published
    /// density is exactly `−dE_θ[S]/dt`.
    ///
    /// Survival and density are integrated separately under this one rule and
    /// the published hazard is their ratio `E_θ[f]/E_θ[S]`, the hazard of the
    /// posterior-predictive law, not the posterior mean `E_θ[f/S]` of the
    /// per-coefficient hazard; the cumulative hazard is `−log E_θ[S]` and the
    /// cumulative incidence `1 − E_θ[S]`.
    ExactAnchor,
    /// The inequality-truncated posterior a location-scale fit (gam#3038) or a
    /// Royston-Parmar fit with an active monotone-baseline bound (gam#3575)
    /// reports: `N(β_unc, Σ)` truncated to the fit's cone, integrated on the
    /// joint constraint-normal × tangent rule the location-scale response
    /// moments use (#2679), every node a feasible coefficient vector replayed
    /// through the plug-in survival surfaces. The moment-matched normal
    /// [`Self::SigmaPoint`] integrates instead puts nodes outside the cone,
    /// where the model has no hazard.
    ///
    /// Survival, event density and the linear predictor are integrated under
    /// the one rule, each cell certified on the spread of the rule's replicate
    /// lattices to the law's own relative accuracy, and published as under
    /// [`Self::ExactAnchor`]: `E_θ[S]`, `−log E_θ[S]` and `E_θ[f]/E_θ[S]`.
    TruncatedLaw,
}

impl SurvivalPosteriorIntegration {
    /// The integration [`predict_survival`] runs for `model` under
    /// `covariance_mode`: exact wherever the saved model is a function of
    /// `(q(t), b(t))`, the truncated law wherever a location-scale or
    /// Royston-Parmar posterior is cone-truncated, sigma-point otherwise.
    pub fn default_for(
        model: &SavedModel,
        covariance_mode: SurvivalPredictionCovarianceMode,
    ) -> Result<Self, SurvivalPredictError> {
        match require_saved_survival_likelihood_mode(model)? {
            SurvivalLikelihoodMode::MarginalSlope => {}
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::Transformation => {
                return Ok(if truncated_survival_posterior_draws(model, covariance_mode)?.is_some() {
                    Self::TruncatedLaw
                } else {
                    Self::SigmaPoint
                });
            }
            _ => return Ok(Self::SigmaPoint),
        }
        let runtime = model.saved_prediction_runtime()?;
        let affine_primaries = runtime.baseline_time_wiggle.is_none()
            && runtime.score_warp.is_none()
            && runtime.link_deviation.is_none()
            && model.survival_marginal_slope_joint_latent_law.is_none();
        Ok(if affine_primaries {
            Self::ExactAnchor
        } else {
            Self::SigmaPoint
        })
    }
}

/// [`predict_survival`] under [`SurvivalPredictEstimand::PosteriorMean`] with a
/// named `integration` (`req.estimand` is not consulted).
/// [`SurvivalPosteriorIntegration::ExactAnchor`] and
/// [`SurvivalPosteriorIntegration::TruncatedLaw`] are refused for a model they do
/// not cover.
///
/// The published point is always the conditional-posterior mean
/// `E[S | D, ρ̂]`, exactly as on the competing-risks and standard-family paths:
/// `covariance_mode` governs only the reported uncertainty, so requesting an
/// interval (or a covariance definition for it) never moves the point
/// (gam#398, gam#3421). A smoothing-corrected band integrates its second
/// moments under the corrected law in a separate pass over the same rule.
pub fn predict_survival_posterior_mean_with(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    integration: SurvivalPosteriorIntegration,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    predict_survival_posterior_mean_integrated(req, covariance_mode, integration, integration, None)
}

/// [`predict_survival_posterior_mean_with`] with the point's conditional law
/// integrated by `point_integration` and a smoothing-corrected band's law by
/// `band_integration`. `band_level` asks the pass whose law the band describes
/// for the central interval of every cell's survival at that level.
fn predict_survival_posterior_mean_integrated(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    point_integration: SurvivalPosteriorIntegration,
    band_integration: SurvivalPosteriorIntegration,
    band_level: Option<f64>,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    let point_request = SurvivalPredictRequest {
        with_uncertainty: false,
        ..req
    };
    // The conditional law carries the band exactly when it is also the law the
    // uncertainty describes; a smoothing-corrected band is read off its own pass.
    let point_band_level = match (req.with_uncertainty, covariance_mode) {
        (true, SurvivalPredictionCovarianceMode::Conditional) => band_level,
        _ => None,
    };
    let (mut result, point) = survival_posterior_moments(
        point_request,
        SurvivalPredictionCovarianceMode::Conditional,
        point_integration,
        point_band_level,
    )?;
    let band = match (req.with_uncertainty, covariance_mode) {
        (false, _) => None,
        (true, SurvivalPredictionCovarianceMode::Conditional) => None,
        (true, SurvivalPredictionCovarianceMode::SmoothingCorrected) => Some(
            survival_posterior_moments(
                point_request,
                covariance_mode,
                band_integration,
                band_level,
            )?
            .1,
        ),
    };
    let uncertainty = req
        .with_uncertainty
        .then(|| (band.as_ref().unwrap_or(&point), covariance_mode));
    publish_survival_posterior_moments(&mut result, &point, uncertainty)?;
    Ok(result)
}

/// The plug-in prediction beside the posterior moments of every cell under
/// the `covariance_mode` coefficient law, integrated by `integration`.
fn survival_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    integration: SurvivalPosteriorIntegration,
    band_level: Option<f64>,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    match integration {
        SurvivalPosteriorIntegration::SigmaPoint => {
            survival_sigma_point_posterior_moments(req, covariance_mode, band_level)
        }
        SurvivalPosteriorIntegration::ExactAnchor => {
            survival_exact_anchor_posterior_moments(req, covariance_mode, band_level)
        }
        SurvivalPosteriorIntegration::TruncatedLaw => {
            survival_truncated_law_posterior_moments(req, covariance_mode, band_level)
        }
    }
}

/// [`predict_survival`]'s posterior-mean prediction with, beside the posterior
/// standard deviations, the central posterior interval of every survival cell
/// at `level` (gam#3560), published as
/// [`SurvivalPredictResult::survival_lower`] and
/// [`SurvivalPredictResult::survival_upper`].
///
/// The interval is a pair of quantiles of the posterior law of `S` that the
/// published mean integrates, not `mean ± z·sd`: `S` is bounded and skewed near
/// either rail, where a symmetric band puts all of its miss mass in one tail and
/// covers values the law never reaches.
///
/// - A marginal-slope model on the exact anchored rule has `S = Φ(−η(q, b))`
///   with `(q, b)` exactly bivariate Gaussian. The band is the central interval
///   of that law ([`gam_solve::quadrature::central_response_interval_on_a_monotone_axis`]),
///   taken on `η` and carried through the decreasing map `S = Φ(−η)`, which
///   preserves quantiles and keeps both ends resolved where `S` is within
///   rounding of a rail.
/// - A model on the sigma-point rule has no low-dimensional law to invert: its
///   `2·rank` nodes are a moment rule, and a quantile read off them is an
///   interval of the nodes. Its survival is a decreasing function of one index
///   per cell — `S = Φ(−η)` for marginal slope, `S = exp(−exp(η))` with
///   `η = log H` for the transformation and Weibull families — so the band is
///   the image under that map of the index's central interval, the index
///   carried at its posterior mean and standard deviation from the same rule.
///   That index is exactly Gaussian where it is linear in the coefficients (the
///   transformation and Weibull `log H`); for a marginal-slope index with a
///   timewiggle, score warp or link deviation it is the index's own normal law.
/// - The truncated-law rule's posterior is a mixture over its nodes of laws
///   each node leaves exactly Gaussian in the coefficients, so the band is the
///   certified central interval of that mixture: a Royston-Parmar cell's
///   `log H` is linear in the coefficients ([`replayed_truncated_surface_band`]),
///   and a location-scale cell's index is resolved in one coordinate per node
///   ([`location_scale_truncated_surface_band`]).
///
/// `req.with_uncertainty` must be set and `level` must lie strictly inside
/// `(0, 1)`.
pub fn predict_survival_with_band(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    level: f64,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "a central survival band needs a level strictly inside (0, 1); got {level}"
            ),
        });
    }
    if !req.with_uncertainty || req.estimand != SurvivalPredictEstimand::PosteriorMean {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "a central survival band describes the posterior law of the posterior-mean \
                     prediction, so it needs with_uncertainty and the posterior-mean estimand"
                .to_string(),
        });
    }
    predict_survival_posterior_mean_banded(req, covariance_mode, Some(level))
}

fn predict_survival_posterior_mean(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    predict_survival_posterior_mean_banded(req, covariance_mode, None)
}

fn predict_survival_posterior_mean_banded(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    band_level: Option<f64>,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    // Each law takes the integration that covers it: the cone truncates only
    // the conditional posterior (gam#3038), so a truncated location-scale
    // point integrates the truncated law while its smoothing-corrected band
    // integrates the corrected normal.
    let point_integration = SurvivalPosteriorIntegration::default_for(
        req.model,
        SurvivalPredictionCovarianceMode::Conditional,
    )?;
    let band_integration = SurvivalPosteriorIntegration::default_for(req.model, covariance_mode)?;
    predict_survival_posterior_mean_integrated(
        req,
        covariance_mode,
        point_integration,
        band_integration,
        band_level,
    )
}

/// Posterior moments of a single-event survival prediction, row × time (row
/// for the linear predictor at each row's own exit time), as each
/// [`SurvivalPosteriorIntegration`] accumulates them: centred moments of the
/// survival and the linear predictor, whose spread is published, and the
/// posterior means of the event density and hazard.
struct SurvivalPosteriorMoments {
    survival: Array2<PosteriorMoment>,
    density_mean: Array2<f64>,
    hazard_mean: Array2<f64>,
    eta: Array1<PosteriorMoment>,
    /// The central interval `(lower, upper)` of each cell's survival under the
    /// same law, present exactly when the pass was given a band level
    /// (gam#3560).
    survival_band: Option<Array2<(f64, f64)>>,
}

impl SurvivalPosteriorMoments {
    fn zeros(n_rows: usize, n_times: usize) -> Self {
        Self {
            survival: Array2::from_elem((n_rows, n_times), PosteriorMoment::EMPTY),
            density_mean: Array2::zeros((n_rows, n_times)),
            hazard_mean: Array2::zeros((n_rows, n_times)),
            eta: Array1::from_elem(n_rows, PosteriorMoment::EMPTY),
            survival_band: None,
        }
    }
}

/// Replace the plug-in surfaces of `result` with those of the posterior-predictive
/// law in `moments`: survival `S̄ = E_θ[S]`, cumulative hazard `−log S̄`, and the
/// hazard `E_θ[f]/E_θ[S]`, both expectations from the same rule. That is the
/// predictive law's own hazard; the posterior mean of the hazard, `E_θ[f/S]`,
/// is a different quantity wherever `S` varies across the posterior and is
/// never published (`hazard_mean` only tells a zero hazard from an infinite
/// one where `S̄ = 0`). When `uncertainty` names the band's moments and their
/// covariance definition, the posterior standard deviations under that law are
/// added; the plug-in survival is kept by name in `survival_plugin`.
fn publish_survival_posterior_moments(
    result: &mut SurvivalPredictResult,
    moments: &SurvivalPosteriorMoments,
    uncertainty: Option<(&SurvivalPosteriorMoments, SurvivalPredictionCovarianceMode)>,
) -> Result<(), SurvivalPredictError> {
    let (n_rows, n_times) = result.survival.dim();
    for published in std::iter::once(moments).chain(uncertainty.map(|(band, _)| band)) {
        if published.survival.dim() != (n_rows, n_times)
            || published.density_mean.dim() != (n_rows, n_times)
            || published.hazard_mean.dim() != (n_rows, n_times)
            || published.eta.len() != n_rows
            || published
                .survival_band
                .as_ref()
                .is_some_and(|band| band.dim() != (n_rows, n_times))
        {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "posterior survival moments have shape {:?}, but the prediction is {n_rows}x{n_times}",
                    published.survival.dim()
                ),
            });
        }
    }
    let SurvivalPosteriorMoments {
        survival: survival_moments,
        density_mean,
        hazard_mean,
        ..
    } = moments;
    // `result` is the plug-in prediction and the loop below overwrites its
    // surfaces with the posterior means, so the plug-in survival is taken
    // here: one clone per call.
    let survival_plugin = result.survival.clone();
    for row in 0..n_rows {
        for time in 0..n_times {
            let survival = survival_moments[[row, time]].mean().clamp(0.0, 1.0);
            let density = density_mean[[row, time]];
            if !density.is_finite() {
                return Err(SurvivalPredictError::NumericalFailure {
                    reason: format!(
                        "posterior survival density is not finite at row {row}, time column {time}: {density}"
                    ),
                });
            }
            if density < 0.0 {
                return Err(decreasing_survival_refusal(format!(
                    "posterior-mean event density {density:.3e} = -dE[S]/dt at row {row}, time column {time}"
                )));
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
    result.survival_se = uncertainty
        .map(|(band, _)| posterior_standard_errors(&band.survival, "survival"))
        .transpose()?;
    result.eta_se = uncertainty
        .map(|(band, _)| posterior_standard_errors(&band.eta, "linear predictor"))
        .transpose()?;
    let band = uncertainty.and_then(|(band, _)| band.survival_band.as_ref());
    result.survival_lower = band.map(|band| band.map(|&(lower, _)| lower));
    result.survival_upper = band.map(|band| band.map(|&(_, upper)| upper));
    result.covariance_source = uncertainty.map(|(_, covariance_mode)| covariance_mode);
    result.survival_plugin = Some(survival_plugin);
    Ok(())
}

/// The posterior moments of one marginal-slope `(row, t)` cell over the
/// coefficient posterior: centred moments of the survival `S = Φ(−η)` and the
/// linear predictor `η`, and the means `E f` of the event density
/// `f = φ(η)·η′` and `E h` of the hazard `h = f/S`. The published hazard is
/// `E f / E S` ([`publish_survival_posterior_moments`]); `E h` only decides
/// between a zero and an infinite hazard where `E S = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactAnchorCellMoments {
    pub(crate) survival: PosteriorMoment,
    pub(crate) density: f64,
    pub(crate) hazard: f64,
    pub(crate) eta: PosteriorMoment,
}

impl gam_solve::quadrature::GhqValue for ExactAnchorCellMoments {
    fn zero() -> Self {
        Self {
            survival: PosteriorMoment::EMPTY,
            density: 0.0,
            hazard: 0.0,
            eta: PosteriorMoment::EMPTY,
        }
    }

    fn addweighted(&mut self, weight: f64, value: Self) {
        self.survival.merge(weight, value.survival);
        self.density += weight * value.density;
        self.hazard += weight * value.hazard;
        self.eta.merge(weight, value.eta);
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            survival: self.survival.scaled(factor),
            density: self.density * factor,
            hazard: self.hazard * factor,
            eta: self.eta.scaled(factor),
        }
    }
}

/// One node of the exact anchored posterior integral at the primaries
/// `(q, b)`: `S`, `f`, `h` and `η` with `S = Φ(−η)`, event density
/// `f = φ(η)·E[η′ | q, b]` and `h = f/S`.
///
/// `conditional_tangent` is `(E[q′ | q, b], E[b′ | q, b])`. The index rate
/// `η′ = η_q·q′ + η_b·b′` is linear in the tangents, so its conditional mean is
/// `η_q·E[q′ | q, b] + η_b·E[b′ | q, b]` and
///
/// ```text
///   E_θ[φ(η)·η′] = E_{q,b}[φ(η)·E(η′ | q, b)] = −d/dt E_θ[Φ(−η(t))]
/// ```
///
/// by differentiating under the integral (`φ ≤ 1/√(2π)`, `η′` Gaussian given
/// the primaries). The published hazard `h̄ = E f / E S` is therefore exactly
/// `dH̄/dt` for the published `H̄ = −log E S`. The density is signed: the mean
/// of a positive part, `E[max(η′, 0)]`, is the expectation of a clamp, and the
/// `h̄` it gives is the derivative of no reported curve (gam#3026). A cell whose
/// integrated density is negative is refused by
/// [`decreasing_survival_refusal`] where the moments are published.
pub(crate) fn exact_anchor_node_moments(
    eta: f64,
    eta_q: f64,
    eta_b: f64,
    conditional_tangent: [f64; 2],
) -> ExactAnchorCellMoments {
    let rate = eta_q * conditional_tangent[0] + eta_b * conditional_tangent[1];
    let (log_survival, mills_ratio) = signed_probit_logcdf_and_mills_ratio(-eta);
    ExactAnchorCellMoments {
        survival: PosteriorMoment::point(log_survival.exp()),
        density: normal_pdf(eta) * rate,
        hazard: mills_ratio * rate,
        eta: PosteriorMoment::point(eta),
    }
}

/// The coefficient posterior [`SurvivalPosteriorIntegration::ExactAnchor`]
/// pushes onto every cell's primaries: the active covariance over the
/// `[time | marginal | slope]` coefficients.
struct ExactAnchorPosterior {
    covariance: Array2<f64>,
    /// The level of the central survival band each cell also reports, or `None`
    /// when no band was asked for (gam#3560).
    band_level: Option<f64>,
}

impl ExactAnchorPosterior {
    /// The exact integration reads `q(t)`, `q′(t)`, `b(t)` and `b′(t)` as affine
    /// functions of the `[time | marginal | slope]` coefficients, which the saved
    /// model is exactly when it carries no timewiggle (whose basis is evaluated
    /// at `q` itself) and no score-warp or link-deviation runtime (which anchor
    /// the intercept on their own coefficients).
    fn require_rigid_coordinates(
        &self,
        ctx: &MarginalSlopePredictContext,
    ) -> Result<(), SurvivalPredictError> {
        if ctx.saved_timewiggle.is_some() || ctx.predictor.has_flexible_runtime() {
            return Err(SurvivalPredictError::UnsupportedConfiguration {
                reason: "the exact anchored survival posterior integration needs q(t) and b(t) \
                         affine in the coefficients and an anchor that depends on them alone; a \
                         baseline timewiggle, score warp or link deviation breaks that, and such a \
                         model integrates with the sigma-point rule"
                    .to_string(),
            });
        }
        let width = ctx.beta_time.len() + ctx.beta_marginal.len() + ctx.beta_slope.len();
        if self.covariance.dim() != (width, width) {
            return Err(SurvivalPredictError::PosteriorCovariance {
                reason: format!(
                    "survival marginal-slope active covariance is {:?}, expected {width}x{width} over [time | marginal | slope]",
                    self.covariance.dim()
                ),
            });
        }
        Ok(())
    }

    /// The exact posterior moments of one assembled cell.
    ///
    /// `(q, b, q′, b′)` are affine in the active coefficients through the rows
    /// `x_q = [time(t) | covariates | 0]`, `x_b = [0 | 0 | slope(t)]`,
    /// `x_q′ = [time′(t) | 0 | 0]` and `x_b′ = [0 | 0 | slope′(t)]`, so they are
    /// jointly Gaussian with covariance `X V Xᵀ`. The primaries are integrated
    /// by the projected bivariate Gauss–Hermite rule; given the primaries `y`
    /// the tangents have mean `t̂ + B(y − ŷ)` with `B = Σ_ty Σ_yy⁻¹`, taken over
    /// the support the rule integrates (only the major axis when `Σ_yy` is
    /// singular in floating point). The density is linear in the tangents, so
    /// that conditional mean is all it reads ([`exact_anchor_node_moments`]).
    ///
    /// With a `band_level`, the cell also reports the central interval of its
    /// survival under the same law of the primaries, the same support the mean
    /// integrates: on the plane the two-coordinate rule
    /// ([`gam_solve::quadrature::central_response_interval_on_a_monotone_axis`]),
    /// along the major axis the one-coordinate one, and at a point mass the
    /// point. The interval is taken on `η(q, b)` and mapped through the
    /// decreasing `S = Φ(−η)`, which carries quantiles to quantiles (gam#3560).
    fn cell_moments(
        &self,
        quadctx: &gam_solve::quadrature::QuadratureContext,
        ctx: &MarginalSlopePredictContext,
        cell: &MarginalSlopeCell,
    ) -> Result<(ExactAnchorCellMoments, Option<(f64, f64)>), SurvivalPredictError> {
        let kernel = ctx
            .predictor
            .anchored_row_kernels(&cell.input)
            .map_err(|e| format!("survival marginal-slope anchored row kernel: {e}"))?
            .pop()
            .ok_or_else(|| "survival marginal-slope cell produced no row kernel".to_string())?;
        let (q_hat, b_hat) = ctx
            .predictor
            .anchored_primaries(&cell.input, &ctx.predictor.theta())
            .map_err(|e| format!("survival marginal-slope primaries: {e}"))?;
        let (q_hat, b_hat) = (q_hat[0], b_hat[0]);

        let p_q = ctx.beta_time.len() + ctx.beta_marginal.len();
        let width = p_q + ctx.beta_slope.len();
        let q_row = cell.input.design.to_dense();
        let slope_row = cell
            .input
            .design_noise
            .as_ref()
            .ok_or_else(|| "survival marginal-slope cell has no slope design".to_string())?
            .to_dense();
        let slope_tangent_width = cell.slope_tangent_row.as_ref().map_or(0, Array1::len);
        if q_row.dim() != (1, p_q)
            || slope_row.dim() != (1, width - p_q)
            || cell.time_derivative_row.len() != ctx.beta_time.len()
            || (cell.slope_tangent_row.is_some() && slope_tangent_width != width - p_q)
        {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "survival marginal-slope cell rows q={:?}, b={:?}, q'={}, b'={slope_tangent_width} \
                     do not match the [time | marginal | slope] widths {}, {}, {}",
                    q_row.dim(),
                    slope_row.dim(),
                    cell.time_derivative_row.len(),
                    ctx.beta_time.len(),
                    ctx.beta_marginal.len(),
                    ctx.beta_slope.len(),
                ),
            });
        }
        let mut rows = Array2::<f64>::zeros((4, width));
        rows.slice_mut(s![0, ..p_q]).assign(&q_row.row(0));
        rows.slice_mut(s![1, p_q..]).assign(&slope_row.row(0));
        rows.slice_mut(s![2, ..ctx.beta_time.len()])
            .assign(&cell.time_derivative_row);
        if let Some(tangent_row) = cell.slope_tangent_row.as_ref() {
            rows.slice_mut(s![3, p_q..]).assign(tangent_row);
        }
        let sigma = rows.dot(&self.covariance).dot(&rows.t());
        if !sigma.iter().all(|entry| entry.is_finite()) {
            return Err(SurvivalPredictError::PosteriorCovariance {
                reason: format!(
                    "survival marginal-slope cell covariance of (q, b, q', b') is not finite: {sigma:?}"
                ),
            });
        }
        let cov_primaries = [
            [sigma[[0, 0]], sigma[[0, 1]]],
            [sigma[[1, 0]], sigma[[1, 1]]],
        ];
        let cov_tangent_primary = [
            [sigma[[2, 0]], sigma[[2, 1]]],
            [sigma[[3, 0]], sigma[[3, 1]]],
        ];
        let regression = match gam_solve::quadrature::BivariateNormalSupport::of(cov_primaries) {
            gam_solve::quadrature::BivariateNormalSupport::Plane => {
                // Σ_yy⁻¹ from the pivots `a` and `b − (c/√a)²` the support test
                // found positive.
                let (a, c) = (cov_primaries[0][0], cov_primaries[1][0]);
                let below = c / a.sqrt();
                let pivot = cov_primaries[1][1] - below * below;
                let inverse = [
                    [1.0 / a + c * c / (a * a * pivot), -c / (a * pivot)],
                    [-c / (a * pivot), 1.0 / pivot],
                ];
                cov_tangent_primary.map(|row| {
                    [
                        row[0] * inverse[0][0] + row[1] * inverse[1][0],
                        row[0] * inverse[0][1] + row[1] * inverse[1][1],
                    ]
                })
            }
            gam_solve::quadrature::BivariateNormalSupport::Axis { axis, variance } => {
                cov_tangent_primary.map(|row| {
                    let along = (row[0] * axis[0] + row[1] * axis[1]) / variance;
                    [along * axis[0], along * axis[1]]
                })
            }
            gam_solve::quadrature::BivariateNormalSupport::Point => [[0.0; 2]; 2],
        };
        let tangent_hat = [cell.q_t, cell.b_t];

        let moments = gam_solve::quadrature::normal_expectation_2d_projected_result(
            quadctx,
            [q_hat, b_hat],
            cov_primaries,
            |q, b| -> Result<ExactAnchorCellMoments, SurvivalPredictError> {
                let (eta, eta_q, eta_b) = kernel
                    .eta_and_partials(q, b)
                    .map_err(|e| format!("survival marginal-slope anchored kernel: {e}"))?;
                if !(eta.is_finite() && eta_q.is_finite() && eta_b.is_finite()) {
                    return Err(SurvivalPredictError::NumericalFailure {
                        reason: format!(
                            "survival marginal-slope posterior node (q={q}, b={b}) produced eta={eta}, eta_q={eta_q}, eta_b={eta_b}"
                        ),
                    });
                }
                let (dq, db) = (q - q_hat, b - b_hat);
                let q_t = tangent_hat[0] + regression[0][0] * dq + regression[0][1] * db;
                let b_t = tangent_hat[1] + regression[1][0] * dq + regression[1][1] * db;
                Ok(exact_anchor_node_moments(eta, eta_q, eta_b, [q_t, b_t]))
            },
        )?;
        let Some(level) = self.band_level else {
            return Ok((moments, None));
        };
        let index = |q: f64, b: f64| -> Result<f64, String> {
            let (eta, _, _) = kernel
                .eta_and_partials(q, b)
                .map_err(|e| format!("survival marginal-slope anchored kernel: {e}"))?;
            if !eta.is_finite() {
                return Err(format!(
                    "survival marginal-slope band node (q={q}, b={b}) produced eta={eta}"
                ));
            }
            Ok(eta)
        };
        let band = probit_survival_band(quadctx, [q_hat, b_hat], cov_primaries, level, index)?;
        Ok((moments, Some(band)))
    }
}

fn survival_exact_anchor_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    band_level: Option<f64>,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    let (_, active_covariance, _) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    let posterior = ExactAnchorPosterior {
        covariance: active_covariance,
        band_level,
    };
    let (result, moments) = predict_survival_surfaces(
        SurvivalPredictRequest {
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
            ..req
        },
        covariance_mode,
        Some(SurvivalSurfacePosterior::ExactAnchor(&posterior)),
    )?;
    let moments = moments.ok_or_else(|| {
        "internal error: the exact anchored survival pass returned no posterior moments".to_string()
    })?;
    // The plug-in survival is published beside the posterior mean
    // (`survival_plugin`), so its curve is held to the same domain as when it
    // is published alone.
    refuse_decreasing_survival(&result)?;
    Ok((result, moments))
}

fn survival_sigma_point_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    band_level: Option<f64>,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    let (posterior_mean, active_covariance, cone_coords) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    let result = predict_survival(
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
    let band_index = band_level
        .map(|level| {
            SurvivalBandIndex::for_mode(result.likelihood_mode).map(|index| (level, index))
        })
        .transpose()?;
    let mut index_cells = Array2::from_elem((n_rows, n_times), SurvivalIndexCell::EMPTY);
    let mut moments = SurvivalPosteriorMoments::zeros(n_rows, n_times);
    let SurvivalPosteriorMoments {
        survival: survival_moments,
        density_mean,
        hazard_mean,
        eta: eta_moments,
        ..
    } = &mut moments;

    for_each_survival_posterior_node(&posterior_mean, &active_covariance, &cone_coords, |node, weight| {
        let draw_model = saved_model_with_survival_coefficients(req.model, node)?;
        let draw = predict_survival_coefficient_law(
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
            eta_moments[row].merge(weight, PosteriorMoment::point(draw.linear_predictor[row]));
            for time in 0..n_times {
                let survival = draw.survival[[row, time]];
                let hazard = draw.hazard[[row, time]];
                let density = conditional_event_density(
                    survival,
                    draw.cumulative_hazard[[row, time]],
                    hazard,
                )?;
                survival_moments[[row, time]].merge(weight, PosteriorMoment::point(survival));
                density_mean[[row, time]] += weight * density;
                hazard_mean[[row, time]] += weight * hazard;
                if let Some((_, index)) = band_index {
                    let cell = &mut index_cells[[row, time]];
                    let cumulative_hazard = draw.cumulative_hazard[[row, time]];
                    match index.index(cumulative_hazard)? {
                        Some(value) => {
                            cell.moment.merge(weight, PosteriorMoment::point(value));
                            cell.resolved = true;
                        }
                        None if cumulative_hazard == 0.0 => cell.at_one = true,
                        None => cell.at_zero = true,
                    }
                }
            }
        }
        Ok(())
    })?;

    if let Some((level, index)) = band_index {
        let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)?;
        let mut band = Array2::from_elem((n_rows, n_times), (0.0, 0.0));
        for ((row, time), cell) in index_cells.indexed_iter() {
            band[[row, time]] = cell.band(index, z, (row, time))?;
        }
        moments.survival_band = Some(band);
    }
    Ok((result, moments))
}

/// [`predict_competing_risks_survival`]'s posterior-mean prediction with, beside
/// the posterior standard deviations, the central posterior bands of the
/// surfaces [`CompetingRisksBands`] describes, at `level` (gam#3560).
/// `req.with_uncertainty` must be set and `level` must lie strictly inside
/// `(0, 1)`.
pub fn predict_competing_risks_with_band(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    level: f64,
) -> Result<CompetingRisksPredictResult, SurvivalPredictError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "a competing-risks band needs a level strictly inside (0, 1); got {level}"
            ),
        });
    }
    if !req.with_uncertainty || req.estimand != SurvivalPredictEstimand::PosteriorMean {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "a competing-risks band describes the posterior law of the posterior-mean \
                     prediction, so it needs with_uncertainty and the posterior-mean estimand"
                .to_string(),
        });
    }
    predict_competing_risks_with_posterior(req, covariance_mode, Some(level))
}

fn predict_competing_risks_with_posterior(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    band_level: Option<f64>,
) -> Result<CompetingRisksPredictResult, SurvivalPredictError> {
    let posterior_mean_estimand = req.estimand == SurvivalPredictEstimand::PosteriorMean;
    // The public posterior-mean point is always the conditional-posterior
    // estimand. `covariance_mode` governs only the reported uncertainty,
    // exactly as on the single-event and standard-family paths, so a request
    // without uncertainty integrates the conditional law whatever mode it
    // names, and a smoothing-corrected interval computes the conditional point
    // once and the corrected second moments separately (gam#3421).
    let covariance_mode = if req.with_uncertainty {
        covariance_mode
    } else {
        SurvivalPredictionCovarianceMode::Conditional
    };
    let (posterior_mean, active_covariance, cone_coords) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    let separate_conditional_point = posterior_mean_estimand
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
            None,
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
    let empty_surfaces = || {
        (0..cause_count)
            .map(|_| Array2::from_elem((n_rows, n_times), PosteriorMoment::EMPTY))
            .collect::<Vec<_>>()
    };
    let mut survival_moments = empty_surfaces();
    let mut hazard_moments = empty_surfaces();
    let mut cumulative_hazard_moments = empty_surfaces();
    let mut cif_moments = empty_surfaces();
    let mut overall_moments = Array2::from_elem((n_rows, n_times), PosteriorMoment::EMPTY);
    let mut eta_moments = (0..cause_count)
        .map(|_| Array1::from_elem(n_rows, PosteriorMoment::EMPTY))
        .collect::<Vec<_>>();
    // The bands read the joint law of every cause's log cumulative hazard and
    // its rate, from the same nodes as the moments (gam#3560).
    let mut band_cells = band_level.is_some().then(|| {
        Array2::from_elem(
            (n_rows, n_times),
            CompetingRisksBandCell {
                coordinates: JointPosteriorMoment::new(2 * cause_count),
                resolved: false,
                at_origin: false,
            },
        )
    });

    for_each_survival_posterior_node(&posterior_mean, &active_covariance, &cone_coords, |node, weight| {
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
                eta_moments[cause][row]
                    .merge(weight, PosteriorMoment::point(draw.linear_predictor[cause][row]));
            }
            for (moments, surface) in [
                (&mut survival_moments[cause], &draw.survival[cause]),
                (&mut hazard_moments[cause], &draw.hazard[cause]),
                (&mut cumulative_hazard_moments[cause], &draw.cumulative_hazard[cause]),
                (&mut cif_moments[cause], &draw.cif[cause]),
            ] {
                ndarray::Zip::from(moments)
                    .and(surface)
                    .for_each(|moment, &value| moment.merge(weight, PosteriorMoment::point(value)));
            }
        }
        ndarray::Zip::from(&mut overall_moments)
            .and(&draw.overall_survival)
            .for_each(|moment, &value| moment.merge(weight, PosteriorMoment::point(value)));
        if let Some(cells) = band_cells.as_mut() {
            let mut coordinates = vec![0.0_f64; 2 * cause_count];
            for ((row, time), cell) in cells.indexed_iter_mut() {
                let mut at_origin = false;
                for cause in 0..cause_count {
                    let cumulative_hazard = draw.cumulative_hazard[cause][[row, time]];
                    let hazard = draw.hazard[cause][[row, time]];
                    if cumulative_hazard == 0.0 {
                        at_origin = true;
                        continue;
                    }
                    if !(cumulative_hazard.is_finite()
                        && cumulative_hazard > 0.0
                        && hazard.is_finite())
                    {
                        return Err(SurvivalPredictError::NumericalFailure {
                            reason: format!(
                                "competing-risks band node at cause {}, row {row}, time column \
                                 {time} has cumulative hazard {cumulative_hazard} and hazard \
                                 {hazard}",
                                cause + 1
                            ),
                        });
                    }
                    coordinates[cause] = cumulative_hazard.ln();
                    coordinates[cause_count + cause] = hazard / cumulative_hazard;
                }
                if at_origin {
                    cell.at_origin = true;
                } else {
                    cell.coordinates.merge_point(weight, &coordinates);
                    cell.resolved = true;
                }
            }
        }
        Ok(())
    })?;

    let surfaces_se = |surfaces: &[Array2<PosteriorMoment>], label: &str| {
        surfaces
            .iter()
            .enumerate()
            .map(|(cause, moments)| {
                posterior_standard_errors(moments, &format!("{label} cause {}", cause + 1))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let (hazard_se, survival_se, cumulative_hazard_se, cif_se, overall_survival_se, eta_se) =
        if req.with_uncertainty {
            (
                Some(surfaces_se(&hazard_moments, "competing-risks hazard")?),
                Some(surfaces_se(&survival_moments, "competing-risks survival")?),
                Some(surfaces_se(
                    &cumulative_hazard_moments,
                    "competing-risks cumulative hazard",
                )?),
                Some(surfaces_se(&cif_moments, "competing-risks cumulative incidence")?),
                Some(posterior_standard_errors(
                    &overall_moments,
                    "competing-risks overall survival",
                )?),
                Some(
                    eta_moments
                        .iter()
                        .enumerate()
                        .map(|(cause, moments)| {
                            posterior_standard_errors(
                                moments,
                                &format!("competing-risks linear predictor cause {}", cause + 1),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            )
        } else {
            (None, None, None, None, None, None)
        };

    if posterior_mean_estimand && !separate_conditional_point {
        let bounded_means = |surfaces: &[Array2<PosteriorMoment>]| {
            surfaces
                .iter()
                .map(|moments| moments.map(|moment| moment.mean().clamp(0.0, 1.0)))
                .collect::<Vec<_>>()
        };
        result.hazard = hazard_moments.iter().map(posterior_moment_means).collect();
        result.survival = bounded_means(&survival_moments);
        result.cumulative_hazard = cumulative_hazard_moments
            .iter()
            .map(posterior_moment_means)
            .collect();
        result.cif = bounded_means(&cif_moments);
        result.overall_survival = overall_moments.map(|moment| moment.mean().clamp(0.0, 1.0));
        result.linear_predictor = eta_moments.iter().map(posterior_moment_means).collect();
    }
    result.hazard_se = hazard_se;
    result.survival_se = survival_se;
    result.cumulative_hazard_se = cumulative_hazard_se;
    result.cif_se = cif_se;
    result.overall_survival_se = overall_survival_se;
    result.eta_se = eta_se;
    result.bands = match (band_level, band_cells.as_ref()) {
        (Some(level), Some(cells)) => Some(competing_risks_bands(cells, cause_count, level)?),
        _ => None,
    };
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

/// A restricted-mean-survival column together with the horizon it was
/// integrated to.
///
/// RMST is meaningless without its `tau` — "8.4 months" answers a different
/// question at a 12-month horizon than at a 24-month one — so the two travel
/// together and every reporting surface emits both.
#[derive(Clone, Debug)]
pub struct RestrictedMeanSurvival {
    /// The restriction horizon the area was accumulated to, in time units.
    pub tau: f64,
    /// Per-row `\int_0^{tau} S_i(t) dt`, one entry per predicted row.
    pub values: Array1<f64>,
}

/// The horizon a prediction grid supports: its last time point.
///
/// `None` when the grid is empty or its last point is not a positive finite
/// time, which are exactly the cases the RMST integral rejects.
fn prediction_horizon(times: &[f64]) -> Option<f64> {
    let tau = *times.last()?;
    (tau.is_finite() && tau > 0.0).then_some(tau)
}

impl SurvivalPredictResult {
    /// Restricted mean survival time over the prediction horizon — the area
    /// under each row's survival curve out to the last time on the grid.
    ///
    /// This is the reporting default: the horizon is the grid the caller
    /// already chose, so no separate knob decides it. Callers wanting a
    /// different `tau` use [`Self::restricted_mean_survival_time`] directly.
    pub fn rmst_over_prediction_horizon(&self) -> Option<RestrictedMeanSurvival> {
        let tau = prediction_horizon(&self.times)?;
        Some(RestrictedMeanSurvival {
            tau,
            values: self.restricted_mean_survival_time(tau)?,
        })
    }

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
    /// All-cause restricted mean survival time over the prediction horizon.
    ///
    /// The competing-risks counterpart of
    /// [`SurvivalPredictResult::rmst_over_prediction_horizon`], taken on the
    /// overall survival `exp(-sum_k H_k(t))`.
    pub fn overall_rmst_over_prediction_horizon(&self) -> Option<RestrictedMeanSurvival> {
        let tau = prediction_horizon(&self.times)?;
        Some(RestrictedMeanSurvival {
            tau,
            values: self.restricted_mean_overall_survival_time(tau)?,
        })
    }

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
/// (higher hazard). A pair is comparable exactly when its failure ordering is
/// observed: subject `a` had an event (`event[a] > 0.5`) and subject `b` was
/// still at risk afterwards — either `time[b] > time[a]`, or `time[b] ==
/// time[a]` with `b` censored (a censoring recorded at a death time happened
/// after the death). Two events at the same time are NOT comparable: neither
/// failed first, so the pair carries no ordering information. A comparable pair
/// is concordant when the earlier-failing subject carries the larger risk;
/// equal risks score half credit. `C = (concordant + 0.5·tied_risk) /
/// comparable`. `C = 0.5` is random ranking, `C = 1.0` a perfect ordering.
///
/// These are the pair rules of `survival::concordance`,
/// `lifelines.utils.concordance_index` and scikit-survival
/// `concordance_index_censored`.
///
/// Evaluated in `O(n log n)`: subjects are swept in descending time order one
/// tie block at a time against a Fenwick tree of the risk ranks of every subject
/// observed strictly later. A block's censored subjects enter the tree before
/// its events are queried (they are comparable partners of those events) and
/// its events enter after (tied events are not partners of each other). Counts
/// are exact integers, so the value equals the pair-loop definition exactly.
///
/// `time`, `event` (1 = event, 0 = censored), and `risk` must share length `n`.
/// Returns `None` on a length mismatch, on any non-finite `time`, `event` or
/// `risk` (the ordering of such a row is undefined), or when there are no
/// comparable pairs
/// (e.g. all rows censored).
pub fn harrell_concordance(time: &[f64], event: &[f64], risk: &[f64]) -> Option<f64> {
    let n = time.len();
    if n != event.len() || n != risk.len() {
        return None;
    }
    if time
        .iter()
        .chain(event)
        .chain(risk)
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let mut levels = risk.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let rank_of = |value: f64| levels.partition_point(|&level| level < value);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut later = RiskRankCounts::new(levels.len());
    let mut comparable: u64 = 0;
    // Twice the concordance numerator: 2 per concordant pair, 1 per risk tie.
    let mut concordant_halves: u64 = 0;
    let mut block_end = n;
    while block_end > 0 {
        let block_time = time[order[block_end - 1]];
        let mut block_start = block_end - 1;
        while block_start > 0 && time[order[block_start - 1]] == block_time {
            block_start -= 1;
        }
        let block = &order[block_start..block_end];
        for &row in block {
            if event[row] <= 0.5 {
                later.insert(rank_of(risk[row]));
            }
        }
        for &row in block {
            if event[row] > 0.5 {
                let rank = rank_of(risk[row]);
                let below = later.count_below(rank);
                let tied = later.count_below(rank + 1) - below;
                comparable += later.total;
                concordant_halves += 2 * below + tied;
            }
        }
        for &row in block {
            if event[row] > 0.5 {
                later.insert(rank_of(risk[row]));
            }
        }
        block_end = block_start;
    }
    if comparable == 0 {
        return None;
    }
    Some(concordant_halves as f64 / (2.0 * comparable as f64))
}

/// Fenwick (binary indexed) tree counting inserted risk ranks, for
/// [`harrell_concordance`].
struct RiskRankCounts {
    tree: Vec<u64>,
    total: u64,
}

impl RiskRankCounts {
    fn new(levels: usize) -> Self {
        Self {
            tree: vec![0; levels + 1],
            total: 0,
        }
    }

    fn insert(&mut self, rank: usize) {
        self.total += 1;
        let mut node = rank + 1;
        while node < self.tree.len() {
            self.tree[node] += 1;
            node += node & node.wrapping_neg();
        }
    }

    /// Number of inserted entries whose rank is strictly below `rank`.
    fn count_below(&self, rank: usize) -> u64 {
        let mut node = rank;
        let mut sum = 0;
        while node > 0 {
            sum += self.tree[node];
            node &= node - 1;
        }
        sum
    }
}

/// IPCW (inverse-probability-of-censoring-weighted) Brier score of a predicted
/// survival probability at a fixed horizon `tau` against held-out outcomes — the
/// Graf et al. (1999) estimator used by scikit-survival `brier_score`, `pec`, and
/// `survival::brier`.
///
/// `s_pred[i]` is the model's predicted survival probability `S(tau | x_i)`.
/// `time`/`event` are the held-out observed time and event indicator.
/// `censoring` is the Kaplan–Meier fit of the censoring survival
/// `G(t) = P(C > t)` ([`KaplanMeier::fit_censoring`]). Each subject's squared
/// residual `(target − Ŝ_i(τ))²` is reweighted by the inverse probability that
/// the subject's outcome at `τ` was observed:
///   * event at/before `τ` (`T_i ≤ τ, δ_i = 1`) → target `0` (dead), weight `1/G(T_i−)`;
///   * still alive past `τ` (`T_i > τ`)         → target `1` (alive), weight `1/G(τ)`;
///   * censored at/before `τ`                    → target undefined, contributes `0`.
///
/// The event weight is the left limit `G(T_i−) = P(C ≥ T_i)`, not `G(T_i)`: an
/// event tied with a censoring is recorded as an event, so the probability that
/// an event at `T_i` is observed is `P(C ≥ T_i)`, and
/// `E[δ·1{T ≤ τ}/G(T−)] = P(T ≤ τ)` is the identity that makes the estimator
/// unbiased (Gerds & Schumacher 2006). The right-continuous `G(T_i)` also removes
/// the censorings tied at `T_i`, so on tied (discretised) times it overweights
/// every such event by `1/(1 − c_j/n_j)`. A survivor past `τ` is observed when
/// `C > τ`, so its weight is the right-continuous `1/G(τ)`.
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
/// runs out of support. (A censoring fit on the scored sample itself never has
/// `G(T_i−) = 0` at an observed `T_i`: subject `i` is in every earlier risk set.)
pub fn ipcw_brier_score(
    s_pred: &[f64],
    time: &[f64],
    event: &[f64],
    tau: f64,
    censoring: &KaplanMeier,
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
            // Failed at or before the horizon: contributes via 1/G(T_i−).
            let g = censoring.before(time[i]);
            if !(g > 0.0) {
                continue;
            }
            (0.0, 1.0 / g)
        } else if time[i] > tau {
            // Survived past the horizon: contributes via 1/G(τ).
            let g = censoring.at(tau);
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

/// Per-subject scores read off a predicted survival path, from
/// [`monotone_survival_and_hazard_scores`].
#[derive(Clone, Debug, Default)]
pub struct HazardPathScores {
    /// `H(T_i) − δ_i·ln h(T_i)`: the negative log-likelihood of subject `i`
    /// under the piecewise-constant hazard implied by its survival path.
    pub log_losses: Vec<f64>,
    /// `½∫₀^{T_i} h² − δ_i·h(T_i)`: a proper score for the hazard model, and
    /// **not** a Brier score — see [`integrated_ipcw_brier_score`] for that.
    pub hazard_quadratic_losses: Vec<f64>,
}

/// Repair a predicted survival matrix into a valid survival path, then read the
/// per-subject hazard scores off it.
///
/// Each row of `raw` is clamped into `[0, 1]`, forced non-increasing, and
/// pinned to `1.0` in the first grid column. The piecewise-constant hazard on
/// interval `k` is then `(H(t_{k+1}) − H(t_k)) / Δt_k` with `H = −ln S`, and
/// each subject is scored at its own event time `T_i` — exactly on a grid point
/// when one coincides, otherwise by linear accumulation inside the containing
/// interval.
///
/// A prediction the observation contradicts with certainty scores `+∞`, not a
/// clipped finite number: a subject whose repaired survival reaches `0` by its
/// time `T_i` has infinite cumulative hazard there, and an event at a time of
/// zero predicted hazard has an infinite log-loss.
///
/// Returns the repaired matrix alongside the scores, because callers need the
/// same repaired matrix for [`integrated_ipcw_brier_score`]; scoring a
/// differently-repaired matrix would make the two metrics disagree about which
/// prediction they scored.
///
/// `grid` must start at the time origin `0` (the only time where `S = 1`, which
/// the pinned first column asserts) and be strictly increasing with at least two
/// points, `observed[i]` is `δ_i`, and every `event_times[i]` must be finite and positive — callers
/// validate that, since what to do about a malformed input is theirs to decide.
pub fn monotone_survival_and_hazard_scores(
    raw: ArrayView2<f64>,
    event_times: &[f64],
    observed: &[bool],
    grid: &[f64],
) -> (Array2<f64>, HazardPathScores) {
    let mut surv = raw.to_owned();
    for mut row in surv.rows_mut() {
        row[0] = 1.0;
        let mut prev = 1.0;
        for value in row.iter_mut() {
            *value = value.clamp(0.0, 1.0).min(prev);
            prev = *value;
        }
    }
    let dt: Vec<f64> = grid.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let cumhaz = surv.mapv(|value| -value.ln());
    let mut haz = Array2::<f64>::zeros((surv.nrows(), surv.ncols() - 1));
    for row in 0..surv.nrows() {
        for col in 0..surv.ncols() - 1 {
            // Once the repaired survival has reached zero the cumulative hazard is
            // infinite, and so is every later hazard; `∞ − ∞` has no other value.
            haz[[row, col]] = if cumhaz[[row, col]].is_infinite() {
                f64::INFINITY
            } else {
                ((cumhaz[[row, col + 1]] - cumhaz[[row, col]]) / dt[col]).max(0.0)
            };
        }
    }
    let mut haz_sq_prefix = Array2::<f64>::zeros((surv.nrows(), surv.ncols()));
    for row in 0..surv.nrows() {
        for col in 0..haz.ncols() {
            haz_sq_prefix[[row, col + 1]] =
                haz_sq_prefix[[row, col]] + haz[[row, col]] * haz[[row, col]] * dt[col];
        }
    }
    let mut log_losses = vec![0.0; event_times.len()];
    let mut hazard_quadratic_losses = vec![0.0; event_times.len()];
    for (row, &time) in event_times.iter().enumerate() {
        let mut j = grid.partition_point(|value| *value < time);
        if j >= grid.len() {
            j = grid.len() - 1;
        }
        let interval_idx = j.saturating_sub(1);
        // An event time on a grid point reads the prefix sums there; any other time
        // adds its interval's correction, which reaches the same values continuously.
        let (h_z, h2_int, hcum_z) = if grid[j] == time {
            (
                haz[[row, interval_idx]],
                haz_sq_prefix[[row, j]],
                cumhaz[[row, j]],
            )
        } else {
            let elapsed = time - grid[interval_idx];
            let h = haz[[row, interval_idx]];
            (
                h,
                haz_sq_prefix[[row, interval_idx]] + h * h * elapsed,
                cumhaz[[row, interval_idx]] + h * elapsed,
            )
        };
        // A subject whose predicted survival is already zero at its own time is
        // contradicted with certainty by being observed there: both scores are
        // `+∞`, and the event terms `ln h` and `h` would otherwise form `∞ − ∞`.
        if hcum_z.is_infinite() {
            log_losses[row] = f64::INFINITY;
            hazard_quadratic_losses[row] = f64::INFINITY;
            continue;
        }
        log_losses[row] = hcum_z - if observed[row] { h_z.ln() } else { 0.0 };
        hazard_quadratic_losses[row] = 0.5 * h2_int - if observed[row] { h_z } else { 0.0 };
    }
    (
        surv,
        HazardPathScores {
            log_losses,
            hazard_quadratic_losses,
        },
    )
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
/// `censoring` is the Kaplan–Meier fit of the censoring survival
/// `G(t) = P(C > t)` ([`KaplanMeier::fit_censoring`]).
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
    censoring: &KaplanMeier,
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
        if let Some(bs) = ipcw_brier_score(&col_slice, time, event, grid[k], censoring) {
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

/// The data-driven grid a held-out survival prediction is scored on: `0`, 24
/// interior knots at the empirical quantiles `j / 25` of the finite positive
/// training times, and the largest such time, strictly increasing.
///
/// A fixed grid such as `{0, 1, 2, 5, 10, median}` suits only O(1)–O(10) survival
/// times; on any other scale it either runs far past the data, so an integrated
/// Brier is dominated by an empty extrapolation tail, or never reaches it.
/// Quantile knots resolve the event-dense region for both the survival-matrix
/// evaluation and the [`integrated_ipcw_brier_score`] integration. With no finite
/// positive time the grid is `[0, 1]`.
pub fn survival_score_grid(train_times: &[f64]) -> Vec<f64> {
    const INTERIOR: usize = 24;
    let mut times: Vec<f64> = train_times
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    let Some(max_t) = times.iter().copied().reduce(f64::max) else {
        return vec![0.0, 1.0];
    };
    times.sort_by(f64::total_cmp);
    let mut grid: Vec<f64> = Vec::with_capacity(INTERIOR + 2);
    grid.push(0.0);
    for j in 1..=INTERIOR {
        let p = j as f64 / (INTERIOR as f64 + 1.0);
        grid.push(gam_math::quantile::quantile_from_sorted(&times, p));
    }
    grid.push(max_t);
    grid.sort_by(f64::total_cmp);
    // Heavy ties pull many quantiles onto one value: drop the points that collapse
    // onto their predecessor, so the grid stays strictly increasing.
    grid.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * a.abs().max(*b).max(1.0));
    grid[0] = 0.0;
    if grid.len() < 2 {
        grid = vec![0.0, max_t.max(1.0)];
    }
    grid
}

/// Held-out scores of a predicted survival matrix, and their skill relative to a
/// null survival matrix when one is given. A field is `None` where the score has
/// no value on this input.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurvivalPredictionScores {
    /// Integrated IPCW Brier score ([`integrated_ipcw_brier_score`]).
    pub brier: Option<f64>,
    /// Mean hazard quadratic score `½∫h² − δ·h(T)` ([`HazardPathScores`]).
    pub hazard_quadratic_score: Option<f64>,
    /// Mean hazard-path negative log-likelihood ([`HazardPathScores`]).
    pub logloss: Option<f64>,
    /// `(null − model) / |null|` of the integrated IPCW Brier score.
    pub lifted_brier: Option<f64>,
    /// `(null − model) / |null|` of the mean hazard quadratic score.
    pub lifted_hazard_quadratic_score: Option<f64>,
    /// `(null − model) / |null|` of the mean hazard-path log-loss.
    pub lifted_logloss: Option<f64>,
    /// Nagelkerke R² of the hazard-path log-likelihood over the null matrix's.
    pub nagelkerke_r2: Option<f64>,
}

/// Score a predicted survival matrix against the observed `(event_times, events)`
/// on `grid`, and against `null_survival` when it is given with the same shape.
///
/// Both matrices are repaired into survival paths by
/// [`monotone_survival_and_hazard_scores`] before scoring. The censoring
/// distribution is Kaplan–Meier on the evaluation set itself, so every model
/// scored on the same fold gets the same IPCW weights, and integration stops at
/// the largest observed time, before the tail where those weights blow up.
///
/// Every field is `None` when the shapes disagree, the grid does not start at
/// the time origin `0` or is not strictly increasing with at least two points,
/// or an event time is not finite and positive. The hazard-path scores integrate
/// from `t = 0`, where `S = 1`: a grid starting later has no column for the
/// hazard accumulated before its first point, and no interval containing an
/// event before it. What a malformed input means is decided here, once, for every front
/// door.
pub fn survival_prediction_scores(
    event_times: &[f64],
    events: &[f64],
    grid: &[f64],
    survival: ArrayView2<f64>,
    null_survival: Option<ArrayView2<f64>>,
) -> SurvivalPredictionScores {
    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }
    if event_times.len() != events.len()
        || survival.nrows() != event_times.len()
        || survival.ncols() != grid.len()
        || grid.len() < 2
        || grid[0] != 0.0
        || grid.windows(2).any(|pair| pair[1] <= pair[0])
        || event_times.iter().any(|time| !time.is_finite() || *time <= 0.0)
    {
        return SurvivalPredictionScores::default();
    }
    let observed: Vec<bool> = events.iter().map(|value| *value > 0.5).collect();
    let censoring = KaplanMeier::fit_censoring(event_times, events);
    let horizon = event_times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score = |matrix: ArrayView2<f64>| {
        let (repaired, path) =
            monotone_survival_and_hazard_scores(matrix, event_times, &observed, grid);
        let brier = integrated_ipcw_brier_score(
            repaired.view(),
            event_times,
            events,
            grid,
            horizon,
            &censoring,
        );
        (brier, path)
    };
    let (brier, path) = score(survival);
    let hazard_quadratic_score = mean(&path.hazard_quadratic_losses);
    let logloss = mean(&path.log_losses);
    let mut scores = SurvivalPredictionScores {
        brier,
        hazard_quadratic_score: Some(hazard_quadratic_score),
        logloss: Some(logloss),
        ..SurvivalPredictionScores::default()
    };
    if let Some(null_matrix) = null_survival.filter(|matrix| matrix.dim() == survival.dim()) {
        let (null_brier, null_path) = score(null_matrix);
        // A relative skill over a null score of exactly zero, or over an infinite
        // null score, has no value; it is not divided by a clipped denominator.
        let relative_skill = |null: f64, model: f64| {
            (null != 0.0 && null.is_finite()).then(|| (null - model) / null.abs())
        };
        scores.lifted_brier = brier
            .zip(null_brier)
            .and_then(|(model, null)| relative_skill(null, model));
        scores.lifted_hazard_quadratic_score = relative_skill(
            mean(&null_path.hazard_quadratic_losses),
            hazard_quadratic_score,
        );
        scores.lifted_logloss = relative_skill(mean(&null_path.log_losses), logloss);
        scores.nagelkerke_r2 = gam_problem::diagnostics::nagelkerke_r_squared_from_log_likelihoods(
            -path.log_losses.iter().sum::<f64>(),
            -null_path.log_losses.iter().sum::<f64>(),
            event_times.len(),
        );
    }
    scores
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

    /// Left limit `Ŝ(t−)`: survival at the last event time strictly before `t`
    /// (and `1.0` at or before the first event). This is `P(T ≥ t)` where
    /// [`Self::at`] is `P(T > t)`; the two differ exactly at an event time.
    pub fn before(&self, t: f64) -> f64 {
        let idx = self.steps.partition_point(|&(time, _)| time < t);
        if idx == 0 { 1.0 } else { self.steps[idx - 1].1 }
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

    /// [`Self::at`] evaluated across a whole grid.
    pub fn on_grid(&self, grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|&t| self.at(t)).collect()
    }

    /// Right-continuous step lookup: `Ŝ(t)` = survival at the last event time
    /// `≤ t` (and `1.0` before the first event). `steps` is sorted by
    /// construction, so the lookup is a binary search: IPCW scoring evaluates
    /// the censoring curve once per subject per grid point against one step per
    /// distinct censoring time. A NaN `t` precedes no step and reads `1.0`.
    pub fn at(&self, t: f64) -> f64 {
        let idx = self.steps.partition_point(|&(time, _)| time <= t);
        if idx == 0 { 1.0 } else { self.steps[idx - 1].1 }
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
    /// Central posterior bands of the surfaces whose law this prediction can
    /// invert, at the level [`predict_competing_risks_with_band`] was asked for
    /// (gam#3560); `None` unless a band was requested.
    pub bands: Option<CompetingRisksBands>,
}

/// Run the survival prediction pipeline.
///
/// Pure library function: no progress bars, no file I/O, no uncertainty
/// bounds. The CLI wraps this with progress updates + CSV writes; the
/// FFI wraps it with JSON serialization. A surface whose survival curve
/// increases at a requested cell is refused by name (gam#3026).
pub fn predict_survival(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    if req.estimand == SurvivalPredictEstimand::PosteriorMean {
        return predict_survival_posterior_mean(req, covariance_mode);
    }
    let result = predict_survival_coefficient_law(req, covariance_mode)?;
    refuse_decreasing_survival(&result)?;
    Ok(result)
}

/// The survival law at the coefficients `req.model` carries, for an integrator
/// that sums it over coefficient draws or quadrature nodes: the sigma-point
/// posterior rule's nodes and a Monte Carlo reference both read it. It is the
/// plug-in pass of [`predict_survival`] without the refusal of a decreasing
/// survival curve: where the law's survival rises, its hazard is the negative
/// `dH/dt` it is, so a sum of `S·h` over draws is exactly `−d/dt` of the same
/// sum of `S`, and the integrator refuses only the curve it publishes
/// (gam#3026). It is not a survival prediction; publish through
/// [`predict_survival`].
pub fn predict_survival_coefficient_law(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    predict_survival_surfaces(req, covariance_mode, None).map(|(result, _)| result)
}

/// A coefficient posterior [`predict_survival_surfaces`] integrates beside the
/// plug-in pass, from the same assembled designs.
#[derive(Clone, Copy)]
enum SurvivalSurfacePosterior<'a> {
    /// [`SurvivalPosteriorIntegration::ExactAnchor`], marginal-slope only.
    ExactAnchor(&'a ExactAnchorPosterior),
    /// [`SurvivalPosteriorIntegration::TruncatedLaw`], location-scale only.
    /// The level of the central survival band the pass also reports, if any
    /// (gam#3560).
    TruncatedLaw(&'a TruncatedCoefficientDraws, Option<f64>),
}

/// The plug-in pass of [`predict_survival`]. With `posterior` it also
/// integrates every cell over the coefficient posterior — a marginal-slope cell
/// with the anchor re-solved at each node, a location-scale cell at every node
/// of the truncated law — from the same assembled designs the plug-in kernel
/// evaluates, and returns those moments beside the plug-in surfaces.
fn predict_survival_surfaces(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    posterior: Option<SurvivalSurfacePosterior<'_>>,
) -> Result<(SurvivalPredictResult, Option<SurvivalPosteriorMoments>), SurvivalPredictError> {
    let (exact_posterior, truncated_posterior) = match posterior {
        None => (None, None),
        Some(SurvivalSurfacePosterior::ExactAnchor(posterior)) => (Some(posterior), None),
        Some(SurvivalSurfacePosterior::TruncatedLaw(draws, band_level)) => {
            (None, Some((draws, band_level)))
        }
    };
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
        if exact_posterior.is_some() {
            return Err(SurvivalPredictError::UnsupportedConfiguration {
                reason: "the exact anchored survival posterior covers marginal-slope models only"
                    .to_string(),
            });
        }
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
            truncated_posterior,
        )
        .map_err(SurvivalPredictError::from);
    }
    if truncated_posterior.is_some() {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "the truncated-law survival posterior covers location-scale models only"
                .to_string(),
        });
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
    // The FIT centers the time design at the anchor UNCONDITIONALLY — every
    // front end routes through `center_survival_time_designs_at_anchor`
    // immediately after building the basis, for every likelihood mode
    // (`fit_orchestration::materialize::survival`, `gam-cli`'s survival path) —
    // so the saved `beta_time` are the coefficients of the CENTERED design.
    // Predict has to ask the same question, and it asks it of the same object:
    // the anchor the fit used is saved on the model.
    //
    // This used to be gated on an enumerated mode list (`LocationScale |
    // MarginalSlope`, plus bare Weibull), and the list silently omitted
    // `Transformation` — the Royston-Parmar default. Evaluating centered
    // coefficients against an uncentered basis shifts every reported
    // `log Λ(t)` by the CONSTANT `X(anchor)ᵀγ`, which is why the defect is
    // invisible on ordinary right-censored data: the anchor there is the
    // earliest entry, i.e. the time origin, where `I_k(left) = 0` exactly and
    // the shift is zero. It appears the moment the anchor moves — on any
    // genuinely left-truncated dataset, which takes the robust interior anchor
    // by rule (#751/#1790/#2631), and on any explicit `survival_time_anchor`.
    //
    // Measured on a 1200-row `Surv(entry, exit, event) ~ s(x)` fit with hazard
    // `0.4·exp(0.9x)` (gam#2705): the same data fitted at anchor `1e-7` and at
    // anchor `1.19` reaches the SAME maximised log-likelihood to seven digits
    // (`-1.364394e3`, so the fit is invariant exactly as the anchor rule
    // documents), while the predicted `η` differs by `+2.859821842` at every
    // one of six (time, covariate) pairs — a constant, to eight digits, and a
    // factor `e^2.86 = 17.5` on the reported cumulative hazard.
    let mut time_anchor: Option<f64> = None;
    let mut time_anchor_row_cached: Option<Array1<f64>> = None;
    if time_build.x_exit_time.ncols() > 0 {
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
    // gam#2929: a model anchored on the joint latent law of K ≥ 2 scores replays
    // that law's anchor per row; the single-score predictor below refuses it.
    let joint_marginal_slope_ctx = if saved_likelihood_mode
        == SurvivalLikelihoodMode::MarginalSlope
        && model.survival_marginal_slope_joint_latent_law.is_some()
    {
        Some(build_joint_marginal_slope_predict_context(
            model,
            data,
            col_map,
            training_headers,
            &cov_design.design,
            noise_offset,
        )?)
    } else {
        None
    };
    let marginal_slope_ctx = if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope
        && joint_marginal_slope_ctx.is_none()
    {
        // Baseline offsets at the predict-data's age_entry / age_exit. Used to
        // build the predictor's `pred_input` (which we discard) — the actual
        // per-(row, t) offset is rebuilt inside `marginal_slope_cell`.
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
            &age_exit,
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
        /// The exact posterior moments of every time column and of the row's
        /// exit-time cell; empty and `None` unless `exact_posterior` was given.
        posterior_cells: Vec<ExactAnchorCellMoments>,
        posterior_exit: Option<ExactAnchorCellMoments>,
        /// The central survival interval of every time column, beside
        /// `posterior_cells` when the exact posterior was given a band level.
        posterior_bands: Vec<(f64, f64)>,
    }

    if let (Some(posterior), Some(ctx)) = (exact_posterior, marginal_slope_ctx.as_ref()) {
        posterior.require_rigid_coordinates(ctx)?;
    } else if exact_posterior.is_some() {
        let got = if joint_marginal_slope_ctx.is_some() {
            "a marginal-slope model anchored on the joint law of several scores".to_string()
        } else {
            survival_likelihood_modename(saved_likelihood_mode).to_string()
        };
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "the exact anchored survival posterior integration is defined for the \
                 single-score marginal-slope likelihood only; got {got}"
            ),
        });
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
            let quadctx = gam_solve::quadrature::QuadratureContext::new();
            type CellOutput = (
                (f64, f64, f64),
                Option<(ExactAnchorCellMoments, Option<(f64, f64)>)>,
            );
            let evaluate_at = |t_query: f64| -> Result<CellOutput, SurvivalPredictError> {
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
                        if let Some(joint) = joint_marginal_slope_ctx.as_ref() {
                            return evaluate_joint_marginal_slope_row(
                                i,
                                joint,
                                &row_time,
                                &r_eta_exit,
                                &r_deriv_exit,
                                effective_primary_offset[i],
                            )
                            .map(|plugin| (plugin, None));
                        }
                        let ctx = marginal_slope_ctx.as_ref().ok_or_else(|| {
                            "internal error: marginal-slope context missing for marginal-slope mode"
                                .to_string()
                        })?;
                        let cell = marginal_slope_cell(
                            i,
                            ctx,
                            &row_time,
                            &r_eta_exit,
                            &r_deriv_exit,
                            effective_primary_offset[i],
                            t_query,
                        )?;
                        let plugin = evaluate_marginal_slope_cell(ctx, &cell)?;
                        let posterior = exact_posterior
                            .map(|posterior| posterior.cell_moments(&quadctx, ctx, &cell))
                            .transpose()?;
                        Ok((plugin, posterior))
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
                        .map(|plugin| (plugin, None))
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
                posterior_cells: Vec::new(),
                posterior_exit: None,
                posterior_bands: Vec::new(),
            };
            if per_row_eval {
                let ((eta_t, cum_t, haz_t), posterior) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
                row.hazard[0] = haz_t;
                row.cumulative_hazard[0] = cum_t;
                row.survival[0] = (-cum_t).exp().clamp(0.0, 1.0);
                row.posterior_cells
                    .extend(posterior.map(|(moments, _)| moments));
                row.posterior_bands
                    .extend(posterior.and_then(|(_, band)| band));
                row.posterior_exit = posterior.map(|(moments, _)| moments);
            } else {
                for (j, &t_query) in eval_times.iter().enumerate() {
                    if t_query <= 0.0 {
                        row.hazard[j] = 0.0;
                        row.cumulative_hazard[j] = 0.0;
                        row.survival[j] = 1.0;
                        // At the time origin every coefficient draw has S = 1 and
                        // no hazard, so the moments carry no posterior spread.
                        if let Some(posterior) = exact_posterior {
                            row.posterior_cells.push(ExactAnchorCellMoments {
                                survival: PosteriorMoment::point(1.0),
                                density: 0.0,
                                hazard: 0.0,
                                eta: PosteriorMoment::point(0.0),
                            });
                            // Every draw has `S = 1` here, so the band is that point.
                            if posterior.band_level.is_some() {
                                row.posterior_bands.push((1.0, 1.0));
                            }
                        }
                    } else {
                        let ((_eta_t, cum_t, haz_t), posterior) = evaluate_at(t_query)?;
                        row.hazard[j] = haz_t;
                        row.cumulative_hazard[j] = cum_t;
                        row.survival[j] = (-cum_t).exp().clamp(0.0, 1.0);
                        row.posterior_cells
                            .extend(posterior.map(|(moments, _)| moments));
                        row.posterior_bands
                            .extend(posterior.and_then(|(_, band)| band));
                    }
                }
                let ((eta_t, _, _), posterior) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
                row.posterior_exit = posterior.map(|(moments, _)| moments);
            }
            Ok(row)
        })
        .collect();
    let row_results = row_results?;

    for (i, row) in row_results.iter().enumerate() {
        linear_predictor[i] = row.linear_predictor;
        for j in 0..t_cols {
            hazard[[i, j]] = row.hazard[j];
            cumulative_hazard[[i, j]] = row.cumulative_hazard[j];
            survival[[i, j]] = row.survival[j];
        }
    }
    let posterior_moments = exact_posterior
        .map(|posterior| {
            let mut moments = SurvivalPosteriorMoments::zeros(n, t_cols);
            if posterior.band_level.is_some() {
                let mut band = Array2::from_elem((n, t_cols), (0.0, 0.0));
                for (i, row) in row_results.iter().enumerate() {
                    if row.posterior_bands.len() != t_cols {
                        return Err(SurvivalPredictError::from(format!(
                            "internal error: exact survival bands cover {} of {t_cols} time columns at row {i}",
                            row.posterior_bands.len()
                        )));
                    }
                    for (j, &cell) in row.posterior_bands.iter().enumerate() {
                        band[[i, j]] = cell;
                    }
                }
                moments.survival_band = Some(band);
            }
            for (i, row) in row_results.iter().enumerate() {
                let exit = row.posterior_exit.ok_or_else(|| {
                    "internal error: exact posterior moments missing at a row's exit time"
                        .to_string()
                })?;
                if row.posterior_cells.len() != t_cols {
                    return Err(SurvivalPredictError::from(format!(
                        "internal error: exact posterior moments cover {} of {t_cols} time columns at row {i}",
                        row.posterior_cells.len()
                    )));
                }
                moments.eta[i] = exit.eta;
                for (j, cell) in row.posterior_cells.iter().enumerate() {
                    moments.survival[[i, j]] = cell.survival;
                    moments.density_mean[[i, j]] = cell.density;
                    moments.hazard_mean[[i, j]] = cell.hazard;
                }
            }
            Ok(moments)
        })
        .transpose()?;

    let times_out: Vec<f64> = if per_row_eval {
        age_exit.to_vec()
    } else {
        eval_times
    };

    Ok((
        SurvivalPredictResult {
            times: times_out,
            hazard,
            survival,
            cumulative_hazard,
            linear_predictor,
            likelihood_mode: saved_likelihood_mode,
            survival_se: None,
            eta_se: None,
            covariance_source: None,
            // This IS the plug-in prediction; `survival` carries it.
            survival_plugin: None,
            survival_lower: None,
            survival_upper: None,
        },
        posterior_moments,
    ))
}

pub fn predict_competing_risks_survival(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<CompetingRisksPredictResult, SurvivalPredictError> {
    if req.estimand == SurvivalPredictEstimand::PosteriorMean || req.with_uncertainty {
        return predict_competing_risks_with_posterior(req, covariance_mode, None);
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
    // Unconditional, for the reason given at the single-cause site: the fit
    // centers every time design at the anchor, so the saved coefficients are
    // the centered ones and a per-cause Royston-Parmar baseline predicted
    // against an uncentered basis carries the constant `X(anchor)ᵀγ` shift
    // (gam#2705).
    let cr_time_anchor_row: Option<Array1<f64>> = if time_build.x_exit_time.ncols() > 0 {
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
    // Rounding band of each refined cumulative hazard, so the AJ assembly
    // clamps only decreases that are rounding of these evaluations (#3529).
    let mut cumulative_hazard_refined_band = (0..cause_count)
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
        /// Rounding band of each `cumulative_refined` entry (#3529).
        cumulative_refined_band: Vec<f64>,
        eta_exit: f64,
    }

    let rows: Result<Vec<CauseRow>, SurvivalPredictError> = (0..cause_count * n)
        .into_par_iter()
        .map(|flat| {
            let cause = flat / n;
            let i = flat % n;
            let block = &fit.blocks[cause];
            let timewiggle = saved_timewiggle_by_cause[cause].as_ref();
            let evaluate_at = |t_query: f64| -> Result<RpRowEvaluation, SurvivalPredictError> {
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
                cumulative_refined_band: vec![0.0; refined_cols],
                eta_exit: 0.0,
            };
            if per_row_eval {
                let (eta_t, cum_t, haz_t, cum_band_t) = evaluate_at(age_exit[i])?;
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
                    let (cum, cum_band) = if t_query <= 0.0 {
                        (0.0, 0.0)
                    } else if s == CIF_REFINE_SUBINTERVALS {
                        // frac == 1 exactly: reuse the exit evaluation so the
                        // assembled CIF and the reported cumulative hazard
                        // agree to the bit.
                        (cum_t, cum_band_t)
                    } else {
                        let (_, cum, _, cum_band) = evaluate_at(t_query)?;
                        (cum, cum_band)
                    };
                    out.cumulative_refined[s - 1] = cum;
                    out.cumulative_refined_band[s - 1] = cum_band;
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
                        let (_eta_t, cum_t, haz_t, _) = evaluate_at(t_query)?;
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
                    let (cum, cum_band) = if t_query <= 0.0 {
                        (0.0, 0.0)
                    } else {
                        let (_, cum, _, cum_band) = evaluate_at(t_query)?;
                        (cum, cum_band)
                    };
                    out.cumulative_refined[jr] = cum;
                    out.cumulative_refined_band[jr] = cum_band;
                }
                let (eta_t, _, _, _) = evaluate_at(age_exit[i])?;
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
            cumulative_hazard_refined_band[row.cause][[row.row, jr]] =
                row.cumulative_refined_band[jr];
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
        let refined_assembled = assemble_competing_risks_cif_from_endpoints_with_rounding(
            assembly_times.view(),
            &cumulative_hazard_refined,
            &cumulative_hazard_refined_band,
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
        let refined_assembled = assemble_competing_risks_cif_from_endpoints_with_rounding(
            assembly_times.view(),
            &cumulative_hazard_refined,
            &cumulative_hazard_refined_band,
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
        bands: None,
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
    /// Slope coefficients, used with the saved time-margin tangent to recover
    /// `b_t` at each curve point.
    beta_slope: Array1<f64>,
    saved_timewiggle: Option<SavedBaselineTimeWiggleRuntime>,
    /// Covariate design (n × p_marginal), kept operator-backed when possible.
    cov_design: DesignMatrix,
    /// Slope design (n × p_slope), kept operator-backed when possible.
    /// With a follow-up margin present this is the tensor product at each row's
    /// own exit time; the per-`(row, t)` curve replay rebuilds it at `t` from
    /// `slope_cov_design` instead (gam#2765, gam#2767).
    slope_design: DesignMatrix,
    /// The COVARIATE factor of the slope design (n × p_cov), which is what
    /// the saved term spec describes. Equal to `slope_design` when the slope
    /// is time-constant.
    slope_cov_design: DesignMatrix,
    /// The fit's resolved slope follow-up margin, when it has one. `None`
    /// is a slope that is constant within a person, and then the row's design
    /// does not depend on the evaluation time at all.
    slope_time_basis: Option<SurvivalCovariateTimeBasis>,
    /// Per-row covariate eta = `cov_design[i] · beta_marginal`. Used to
    /// pre-compute `q_exit_base`.
    cov_eta: Array1<f64>,
    /// Per-row latent z (raw, un-normalized — the predictor's
    /// `latent_z_normalization` is applied internally).
    z_raw: Array1<f64>,
    /// Per-row noise offset, mirroring the `pred_input.offset_noise` slice
    /// used by the CLI.
    noise_offset: Array1<f64>,
    /// Per-row scaled context covariates a local latent law is replayed from
    /// (gam#2926); `None` for every other law.
    local_law_conditioning: Option<Array2<f64>>,
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
    age_exit: &Array1<f64>,
) -> Result<MarginalSlopePredictContext, SurvivalPredictError> {
    let z_name = model.z_column.as_deref()
        .ok_or_else(|| "saved marginal-slope model lacks score column".to_string())?;
    let z_raw = crate::inference::ctn::latent_scores(model, data, col_map)?;

    let slopespec = resolve_termspec_for_prediction(
        &model.resolved_slopespec.as_ref().cloned(),
        training_headers,
        col_map,
        "resolved_slopespec",
    )?;
    let slope_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let slope_input = slope_clipped.as_ref().map_or(data, |arr| arr.view());
    let slope_design = build_term_collection_design(slope_input, &slopespec)
        .map_err(|e| format!("failed to build survival marginal-slope slope design: {e}"))?;
    // The slope offset is a slope on the score as given; the model reads the
    // score `(z − mean)/sd`, on which the same slope is `sd` times it, the factor
    // the fit applied (gam#3477).
    let score_sd = model
        .latent_z_normalization
        .as_ref()
        .ok_or_else(|| {
            "saved survival marginal-slope model missing latent_z_normalization".to_string()
        })?
        .sd;
    let effective_noise_offset = slope_design
        .compose_offset(
            noise_offset.view(),
            "survival marginal-slope slope block",
        )
        .map_err(|error| error.to_string())?
        * score_sd;
    // gam#2765 / gam#2767: the term spec names the covariate factor only. With a
    // follow-up margin the fitted coefficients live against `X_cov ⊗ᵣ B(log t)`,
    // so keep BOTH — the factor, which the per-`(row, t)` replay re-tensors at
    // the time being predicted, and the product at each row's own exit time,
    // which is what the predictor's width contract is stated against.
    let slope_time_basis = model.slope_time_basis.clone();
    let slope_cov_design = slope_design.design.clone();
    let slope_exit_design = match slope_time_basis.as_ref() {
        None => slope_cov_design.clone(),
        Some(time_basis) => {
            crate::survival::construction::replay_slope_time_margin_value_tangent_design(
                age_exit.view(),
                time_basis,
                &slope_cov_design,
            )?
            .value
        }
    };

    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let local_law_conditioning =
        crate::inference::predict_input::build_marginal_slope_local_auxiliary_matrix(
            model, data, col_map,
        )
        .map_err(|error| SurvivalPredictError::InvalidInput {
            reason: error.to_string(),
        })?;
    let (predictor, _pred_input, _predictor_fit) = build_saved_survival_marginal_slope_predictor(
        model,
        &fit_saved,
        z_name,
        &z_raw,
        cov_design,
        &slope_exit_design,
        time_build,
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        primary_offset,
        &effective_noise_offset,
        local_law_conditioning.clone(),
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
    let beta_slope = blocks[2].beta.clone();
    let saved_runtime = model.saved_prediction_runtime()?;
    let saved_timewiggle = saved_runtime.baseline_time_wiggle.clone();

    // cov_eta is time-independent so doing it here avoids `O(n × T)`
    // re-multiplications inside the per-cell loop.
    let cov_eta = cov_design.dot(&beta_marginal);

    Ok(MarginalSlopePredictContext {
        predictor,
        beta_time,
        beta_marginal,
        beta_slope,
        saved_timewiggle,
        cov_design: cov_design.clone(),
        slope_design: slope_exit_design,
        slope_cov_design,
        slope_time_basis,
        cov_eta,
        z_raw,
        noise_offset: effective_noise_offset,
        local_law_conditioning,
    })
}

/// Precomputed context for a saved survival marginal-slope model anchored on
/// the joint latent law of `K ≥ 2` scores (gam#2929): the per-row slope vector,
/// the per-row score vector, and the law transported to every row's context.
struct JointMarginalSlopePredictContext {
    /// Time-block coefficients; a joint-law model carries no time wiggle.
    beta_time: Array1<f64>,
    /// Per-row covariate eta `cov_design[i] · beta_marginal`.
    cov_eta: Array1<f64>,
    /// `n × K` raw per-score slopes `g_k`, offsets and baseline included.
    slopes: Array2<f64>,
    /// `n × K` scores as the fit consumed them.
    scores: Array2<f64>,
    probit_scale: f64,
    law: crate::survival::marginal_slope::JointLatentLawRuntime,
}

fn build_joint_marginal_slope_predict_context(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cov_design: &DesignMatrix,
    noise_offset: &Array1<f64>,
) -> Result<JointMarginalSlopePredictContext, SurvivalPredictError> {
    let law = model
        .survival_marginal_slope_joint_latent_law
        .as_ref()
        .ok_or_else(|| "saved survival marginal-slope model lacks its joint latent law".to_string())?;
    let k = law.score_dim;
    let saved_runtime = model.saved_prediction_runtime()?;
    if saved_runtime.score_warp.is_some()
        || saved_runtime.link_deviation.is_some()
        || saved_runtime.baseline_time_wiggle.is_some()
        || saved_runtime.influence_absorber_width.is_some()
        || model.slope_time_basis.is_some()
        || model.score_transform.is_some()
    {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "saved survival marginal-slope joint latent law serves rigid time-constant \
                     per-score slopes on external scores only; this model also names a flex \
                     block, time wiggle, influence absorber, follow-up margin or score transform"
                .to_string(),
        });
    }
    if model.latent_z_rank_int_calibration.is_some()
        || model.latent_z_conditional_calibration.is_some()
    {
        // gam#2949: a `K ≥ 2` model's conditional maps travel inside the joint
        // law, one per coordinate, and are applied below. A scalar map in the
        // payload beside them would map one coordinate twice, and a rank-INT
        // map has no per-coordinate form at all, so both are refused here as
        // well as at save.
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "saved survival marginal-slope joint latent law carries its scores' \
                     conditional maps per coordinate; this model also carries a scalar \
                     latent-z calibration, which would map a coordinate twice"
                .to_string(),
        });
    }
    // The joint law carries every score's unit map z̃_j = (z_j − m_j)/s_j
    // (gam#4331); the replay validator already checked that the scalar
    // `latent_z_normalization` equals the law's map for score 0.
    let z_columns = model
        .z_columns
        .as_ref()
        .filter(|names| names.len() == k)
        .ok_or_else(|| format!("saved K={k} survival marginal-slope model must name {k} score columns"))?;
    let surface_specs = model
        .resolved_slopespecs
        .as_ref()
        .filter(|specs| specs.len() == k)
        .ok_or_else(|| format!("saved K={k} survival marginal-slope model must carry {k} slope surfaces"))?;
    let n = data.nrows();
    // gam#2949: a coordinate whose conditional law moves was read by the row
    // program as `ζ = (z − m(a))/√v(a)`, so a new score is read on that same
    // axis before the law's unit map. The span `a(C)` is the marginal design,
    // rebuilt here from the resolved marginal spec — which is exactly the
    // reproducibility the save refusal requires, so a model that reaches this
    // point has one. The composition order is the fit's: raw score, conditional
    // map, then the unit map the law's nodes live on (gam#4331).
    let conditioning = if law.calibrates_any_score() {
        Some(
            cov_design
                .try_to_dense_arc("saved survival marginal-slope joint-law conditioning span")
                .map_err(|reason| SurvivalPredictError::InvalidInput { reason })?,
        )
    } else {
        None
    };
    let mut scores = Array2::<f64>::zeros((n, k));
    for (column, name) in z_columns.iter().enumerate() {
        let index = *col_map
            .get(name)
            .ok_or_else(|| format!("missing score column '{name}'"))?;
        let raw = data.column(index).to_owned();
        let fitted_axis = match (law.score_calibration(column), conditioning.as_ref()) {
            (None, _) => raw,
            (Some(calibration), Some(span)) => {
                crate::inference::predict_io::FittedLatentScoreMap::conditional_only(calibration)
                    .calibrate(raw.view(), Some(span.view()))
                    .map_err(|reason| SurvivalPredictError::InvalidInput { reason })?
            }
            (Some(_), None) => {
                return Err(SurvivalPredictError::InvalidInput {
                    reason: format!(
                        "saved survival marginal-slope joint latent law calibrates score column \
                         {column} but built no conditioning span to read it on"
                    ),
                });
            }
        };
        scores
            .column_mut(column)
            .assign(&fitted_axis.mapv(|z| law.standardized_score(column, z)));
    }

    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    if fit_saved.blocks.len() != 3 {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope joint-law model requires 3 blocks [time, marginal, slope], got {}",
                fit_saved.blocks.len()
            ),
        });
    }
    let beta_time = fit_saved.blocks[0].beta.clone();
    let beta_marginal = &fit_saved.blocks[1].beta;
    let beta_slope = &fit_saved.blocks[2].beta;
    if beta_marginal.len() != cov_design.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope marginal coefficient mismatch: beta has {} entries but baseline design has {} columns",
                beta_marginal.len(),
                cov_design.ncols()
            ),
        });
    }
    let baseline_slope = model
        .baseline_slope
        .ok_or_else(|| "saved survival marginal-slope model missing baseline_slope".to_string())?;
    let slope_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let slope_input = slope_clipped.as_ref().map_or(data, |arr| arr.view());
    let mut slopes = Array2::<f64>::zeros((n, k));
    let mut cursor = 0usize;
    for (surface, saved_spec) in surface_specs.iter().enumerate() {
        let spec = resolve_termspec_for_prediction(
            &Some(saved_spec.clone()),
            training_headers,
            col_map,
            "resolved_slopespecs",
        )?;
        let design = build_term_collection_design(slope_input, &spec).map_err(|e| {
            format!("failed to build survival marginal-slope slope surface {surface} design: {e}")
        })?;
        let width = design.design.ncols();
        if cursor + width > beta_slope.len() {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "saved survival marginal-slope slope surfaces are wider than the {} slope coefficients",
                    beta_slope.len()
                ),
            });
        }
        let offset = design
            .compose_offset(noise_offset.view(), "survival marginal-slope slope surface")
            .map_err(|error| error.to_string())?;
        let values = design
            .design
            .dot(&beta_slope.slice(s![cursor..cursor + width]).to_owned());
        // The slope on the standardized score z̃_j is s_j times the slope on
        // the raw score, so the raw slope offset enters channel j as s_j·o
        // (gam#4331), exactly as the fit materialised it.
        let scale = law.score_scale[surface];
        slopes
            .column_mut(surface)
            .assign(&(values + &offset.mapv(|o| scale * o) + baseline_slope));
        cursor += width;
    }
    if cursor != beta_slope.len() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope slope surfaces span {cursor} columns but the slope block has {} coefficients",
                beta_slope.len()
            ),
        });
    }
    let cov_eta = cov_design.dot(beta_marginal);
    let conditioning = law.conditional.is_some().then(|| cov_design.to_dense());
    let runtime = law.runtime(conditioning.as_ref().map(|block| block.view()), n)?;
    let sigma = match model.family_state.frailty() {
        None | Some(FrailtySpec::None) => None,
        Some(FrailtySpec::GaussianShift {
            scale: crate::survival::lognormal_kernel::FrailtyScale::Fixed { sigma },
        }) => Some(*sigma),
        Some(other) => {
            return Err(SurvivalPredictError::UnsupportedConfiguration {
                reason: format!(
                    "saved survival marginal-slope joint-law model has a frailty state the marginal-slope schema forbids: {}",
                    match other {
                        FrailtySpec::HazardMultiplier { .. } => "hazard multiplier",
                        _ => "learned Gaussian shift",
                    }
                ),
            });
        }
    };
    Ok(JointMarginalSlopePredictContext {
        beta_time,
        cov_eta,
        slopes,
        scores,
        probit_scale: crate::marginal_slope_shared::probit_frailty_scale(sigma),
        law: runtime,
    })
}

/// One `(row, t)` cell of a saved joint-law model: `η = α(q(t), r) + rᵀz` on
/// the row's transported law, `η′ = α_q·q′(t)`, and the probit survival and
/// hazard they define.
fn evaluate_joint_marginal_slope_row(
    row_index: usize,
    ctx: &JointMarginalSlopePredictContext,
    row_time: &SurvivalTimeBuildOutput,
    r_eta_exit: &Array1<f64>,
    r_deriv_exit: &Array1<f64>,
    primary_offset_row: f64,
) -> Result<(f64, f64, f64), SurvivalPredictError> {
    if ctx.beta_time.len() != row_time.x_exit_time.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope time coefficient mismatch: beta has {} entries but the time basis has {}",
                ctx.beta_time.len(),
                row_time.x_exit_time.ncols()
            ),
        });
    }
    let q = row_time.x_exit_time.dot(&ctx.beta_time)[0]
        + ctx.cov_eta[row_index]
        + r_eta_exit[0]
        + primary_offset_row;
    let qd = row_time.x_derivative_time.dot(&ctx.beta_time)[0] + r_deriv_exit[0];
    let slopes = ctx.slopes.row(row_index).to_vec();
    let scores = ctx.scores.row(row_index).to_vec();
    let mut workspace = crate::survival::marginal_slope::JointAnchorRowWorkspace::new(&ctx.law);
    let (eta, eta_t) = crate::survival::marginal_slope::joint_anchored_index_and_rate(
        row_index,
        q,
        qd,
        &slopes,
        &scores,
        ctx.probit_scale,
        &ctx.law,
        &mut workspace,
    )?;
    let (cum, haz) = probit_survival_hazard_components(eta, eta_t)?;
    Ok((eta, cum, haz))
}

/// One `(row, t)` cell of a saved survival marginal-slope prediction, assembled
/// but not yet evaluated: the 1-row predictor input at `t`, both primaries' time
/// tangents at the saved coefficients, and the design rows those tangents are
/// linear in.
struct MarginalSlopeCell {
    /// The q-design row `[time(t) | timewiggle | covariates]`, the slope row
    /// `b(t)`, both offsets and the row's latent score.
    input: PredictInput,
    /// `q′(t)`, the timewiggle chain included.
    q_t: f64,
    /// `b′(t)`; `0` for a slope that is constant within a person.
    b_t: f64,
    /// `∂q′(t)/∂β_time` over the base time columns: all of `q′(t)`'s dependence
    /// on the coefficients when the model carries no timewiggle.
    time_derivative_row: Array1<f64>,
    /// `∂b′(t)/∂β_slope`; `None` for a slope that is constant within a person.
    slope_tangent_row: Option<Array1<f64>>,
}

/// Assemble one (row, t) cell for the saved survival marginal-slope kernel.
fn marginal_slope_cell(
    row_index: usize,
    ctx: &MarginalSlopePredictContext,
    row_time: &SurvivalTimeBuildOutput,
    r_eta_exit: &Array1<f64>,
    r_deriv_exit: &Array1<f64>,
    primary_offset_row: f64,
    evaluation_time: f64,
) -> Result<MarginalSlopeCell, SurvivalPredictError> {
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
        // Only the VALUE basis is wanted here, so the design is built without a
        // penalty set: since gam#2647 assembling one costs a function Gram and a
        // generalized eigendecomposition per predicted row.
        let exit_design = monotone_wiggle_basis_with_derivative_order(
            eta_exit_row.view(),
            &knots,
            runtime.degree,
            0,
        )?;
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

    // Slope design row + offset chosen so that the predictor's slope_eta
    // equals our precomputed `slope_eta[row]`.  The predictor computes:
    //   slope_eta = design_noise · beta_slope + baseline_slope
    //                  + offset_noise.
    // We feed the actual saved slope row + the row's noise offset, matching
    // exactly the CLI's `pred_input.design_noise` / `offset_noise` slice.
    // gam#2765 / gam#2767: with a follow-up margin the slope is `b(t)`, so the
    // row's design has to be re-tensored at the time the curve is being
    // evaluated at rather than frozen at the row's own exit time. Reading the
    // exit-time row here would return `S(t)` computed with `b(t_exit)` — a
    // different model at every point of the curve except one.
    let (slope_row, slope_tangent, slope_tangent_row) = match ctx.slope_time_basis.as_ref() {
        None => (
            design_row_owned(
                &ctx.slope_design,
                row_index,
                "survival marginal slope row",
            )?,
            0.0,
            None,
        ),
        Some(time_basis) => {
            let cov_row = design_row_owned(
                &ctx.slope_cov_design,
                row_index,
                "survival marginal slope covariate row",
            )?;
            let cov_row_design = DesignMatrix::from(
                cov_row
                    .into_shape_with_order((1, ctx.slope_cov_design.ncols()))
                    .map_err(|e| format!("survival marginal slope covariate row shape: {e}"))?,
            );
            let replay =
                crate::survival::construction::replay_slope_time_margin_value_tangent_design(
                    Array1::from_elem(1, evaluation_time).view(),
                    time_basis,
                    &cov_row_design,
                )?;
            let value = design_row_owned(&replay.value, 0, "survival marginal slope row at t")?;
            let derivative = design_row_owned(
                &replay.derivative,
                0,
                "survival marginal slope tangent row at t",
            )?;
            let tangent = derivative.dot(&ctx.beta_slope);
            (value, tangent, Some(derivative))
        }
    };
    let mut slope_design_2d = Array2::<f64>::zeros((1, slope_row.len()));
    slope_design_2d.row_mut(0).assign(&slope_row);

    let pred_input = PredictInput {
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(q_design_full)),
        offset: Array1::from_elem(1, r_eta_exit[0] + primary_offset_row),
        design_noise: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(slope_design_2d),
        )),
        offset_noise: Some(Array1::from_elem(1, ctx.noise_offset[row_index])),
        auxiliary_scalar: Some(Array1::from_elem(1, ctx.z_raw[row_index])),
        // gam#2926: the row's context covariates, so a local latent law is
        // replayed for this cell exactly as for the whole table.
        auxiliary_matrix: ctx
            .local_law_conditioning
            .as_ref()
            .map(|conditioning| conditioning.slice(s![row_index..row_index + 1, ..]).to_owned()),
    };
    Ok(MarginalSlopeCell {
        input: pred_input,
        q_t: qd_with_wiggle,
        b_t: slope_tangent,
        time_derivative_row: design_row_owned(
            &row_time.x_derivative_time,
            0,
            "survival marginal time derivative row",
        )?,
        slope_tangent_row,
    })
}

/// Evaluate one (row, t) cell for the saved survival marginal-slope kernel.
///
/// Calls the saved [`BernoulliMarginalSlopePredictor`]
/// (`predict_eta_and_time_tangent`) to obtain both the linear predictor `eta`
/// and its complete time tangent
/// `eta_t = (∂eta/∂q) q_t + (∂eta/∂b) b_t`. In rigid mode both partials have
/// closed forms; empirical and flexible latent laws carry their exact implicit
/// calibration pull-backs. This mirrors `compute_survival_timepoint_exact` in
/// `survival_marginal_slope.rs`.
fn evaluate_marginal_slope_cell(
    ctx: &MarginalSlopePredictContext,
    cell: &MarginalSlopeCell,
) -> Result<(f64, f64, f64), SurvivalPredictError> {
    // Exact IFT pull-back: the predictor consumes both moving primary
    // coordinates. The slope margin contributes even when q is locally flat:
    // `eta_t = eta_q q_t + eta_b b_t`.
    let (eta_arr, eta_t_arr) = ctx
        .predictor
        .predict_eta_and_time_tangent(
            &cell.input,
            &Array1::from_elem(1, cell.q_t),
            &Array1::from_elem(1, cell.b_t),
        )
        .map_err(|e| format!("saved survival marginal-slope predictor replay failed: {e}"))?;
    let eta = eta_arr[0];
    // `qd_with_wiggle` is the base survival-index time derivative q'(t), built
    // identically to fit-time `qd1 = dq_dq0·d_raw` (the wiggle chain and the
    // `+derivative_guard` offset are both already folded into `qd_exit_base`),
    // so there is no predict-vs-fit desync in the derivative reconstruction.
    //
    // The complete rate `η′ = η_q·q′ + η_b·b′` goes to the hazard unchanged: the
    // hazard reported is the derivative of the cumulative hazard reported beside
    // it, and a published surface whose index decreases at a cell is refused by
    // name ([`refuse_decreasing_survival`]).
    let (cum, haz) = probit_survival_hazard_components(eta, eta_t_arr[0])?;
    Ok((eta, cum, haz))
}

/// The named refusal of a published survival surface whose curve increases,
/// `dH/dt < 0` (gam#3026). Every survival surface that is published, the
/// plug-in ([`refuse_decreasing_survival`]) and the posterior mean
/// ([`publish_survival_posterior_moments`]), declines such a cell through this
/// one route. `detail` names the quantity that went negative and where.
#[cold]
#[inline(never)]
pub(crate) fn decreasing_survival_refusal(detail: String) -> SurvivalPredictError {
    SurvivalPredictError::DecreasingSurvival {
        reason: format!(
            "survival prediction refused: the reported survival curve increases here ({detail}), \
             so no hazard h >= 0 is the derivative dH/dt of the reported cumulative hazard; the \
             fitted model is not a survival model at this cell"
        ),
    }
}

/// Refuse a plug-in survival surface that reports a negative hazard anywhere.
/// The hazard of every family is the derivative of the cumulative hazard
/// reported beside it, sign included, so `h < 0` is a cell where the reported
/// survival rises: no survival function reports it, and publishing `h = 0`
/// there instead would pair the curve with the hazard of a different one.
fn refuse_decreasing_survival(result: &SurvivalPredictResult) -> Result<(), SurvivalPredictError> {
    match result
        .hazard
        .indexed_iter()
        .find(|(_, hazard)| **hazard < 0.0)
    {
        Some(((row, time), hazard)) => Err(decreasing_survival_refusal(format!(
            "hazard dH/dt={hazard:.6e} < 0 at row {row}, time column {time}"
        ))),
        None => Ok(()),
    }
}

/// Cumulative hazard and hazard of the probit survival law `S(t) = Φ(−η(t))`
/// at one cell, from the index and its complete time derivative:
///
/// ```text
///   H = −log Φ(−η),   h = dH/dt = φ(η)/Φ(−η) · η′
/// ```
///
/// `h` is the exact derivative of the `H` returned beside it for every finite
/// rate, its sign included, so a consumer composing the two (the cumulative
/// incidence `∫ exp(−Σ ΔH) h`) reads one curve, and an integrator summing
/// `S·h` over coefficient draws or quadrature nodes obtains `−d/dt` of its sum
/// of `S`. The survival domain is `η′ ≥ 0`, and it is the fit's own: its
/// likelihood carries `log η′` at every event and holds
/// `q′ ≥ derivative_guard ≥ 0` at every row. A time-constant slope has
/// `η′ = α_q·q′ ≥ α_q·derivative_guard ≥ 0` at every `t` by construction (the
/// I-spline time block has `M_k ≥ 0` on all of `ℝ` and is coned to `β ≥ 0`, the
/// baseline offset's rate is `S₀h₀/φ ≥ 0`, and `α_q > 0`), so its fitted law
/// never leaves the domain; `η′ = 0` is a flat stretch with `h = dH/dt = 0`
/// exactly. A slope that varies along follow-up carries no such guarantee:
/// `η′ = α_q·q′ + (α_b + z)·b′` is affine in `z` with an unsigned `b′(t)`
/// (gam#2767). Where `η′ < 0` this returns the negative `h` it is; it is never
/// clamped, and the surfaces that publish a hazard refuse it
/// ([`decreasing_survival_refusal`]).
#[inline]
pub(crate) fn probit_survival_hazard_components(
    eta: f64,
    eta_derivative: f64,
) -> Result<(f64, f64), SurvivalPredictError> {
    if !(eta.is_finite() && eta_derivative.is_finite()) {
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
    // mathematical range [0, +∞] of the cumulative hazard. Saturated probit
    // fits where the model genuinely says S(t)→0 produce a +∞ cumulative
    // hazard — that is the truthful answer, and the consumer's
    // `survival = exp(-cum).clamp(0,1)` handles it cleanly. Rejecting +∞ would
    // force the predictor to fail on models that the inner solver has already
    // certified as a valid fit.
    if !(cumulative_hazard >= 0.0 && !hazard.is_nan()) {
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
    .map(|(eta, cumulative_hazard, hazard, _)| (eta, cumulative_hazard, hazard))
}

/// `(eta, H, h, band_H)` of one Royston-Parmar row; see [`evaluate_rp_row_with_beta`].
type RpRowEvaluation = (f64, f64, f64, f64);

/// Evaluate one Royston-Parmar row: `(eta, H, h, band_H)`.
///
/// `band_H` bounds the rounding of `H = exp(eta)` as evaluated here, for the
/// design row as built. `eta = Σ_j x_j β_j + (eta_time_offset + primary_offset)`
/// is a sum of `p + 2` terms, so its forward error is at most
/// `δ = γ_{p+2} · (Σ_j |x_j β_j| + |eta_time_offset| + |primary_offset|)`
/// (Higham, Lemma 3.1 / inner-product bound). `exp` is faithfully rounded, a
/// relative error below `ε`, so
/// `|Ĥ − H| ≤ Ĥ · (expm1(δ) + ε) / (1 − ε)`.
/// The competing-risks Aalen-Johansen assembly uses it to tell a rounding-level
/// decrease of `H` between two evaluations from a real one (#3529).
fn evaluate_rp_row_with_beta(
    beta: &Array1<f64>,
    saved_timewiggle: Option<&SavedBaselineTimeWiggleRuntime>,
    row_time: &SurvivalTimeBuildOutput,
    cov_row: &Array1<f64>,
    eta_time_offset_row: f64,
    derivative_time_offset_row: f64,
    primary_offset_row: f64,
) -> Result<RpRowEvaluation, SurvivalPredictError> {
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
    let offset_derivative_component = derivative_time_offset_row;
    let mut eta_derivative = offset_derivative_component;
    let mut time_derivative_component = 0.0_f64;
    if p_time > 0 {
        time_derivative_component = row_time
            .x_derivative_time
            .dot(&beta.slice(s![..p_time]).to_owned())[0];
        eta_derivative += time_derivative_component;
    }
    let mut wiggle_derivative_component = 0.0_f64;
    if let Some(runtime) = saved_timewiggle {
        let knots = Array1::from_vec(runtime.knots.clone());
        let beta_w = beta.slice(s![p_time..p_time + p_timewiggle]).to_owned();
        let eta_exit_row = Array1::from_elem(1, eta_time_offset_row);
        let derivative_exit_row = Array1::from_elem(1, derivative_time_offset_row);
        let exit_design = monotone_wiggle_basis_with_derivative_order(
            eta_exit_row.view(),
            &knots,
            runtime.degree,
            0,
        )?;
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
        wiggle_derivative_component = derivative_design.dot(&beta_w)[0];
        eta_derivative += wiggle_derivative_component;
    }
    // Cold-path diagnostic (fires only when the assembled log-cumulative-hazard
    // derivative is about to be refused): decompose `eta_t` into its additive
    // components and report the time-coefficient / derivative-basis extrema so a
    // refused prediction is traceable to the specific negative term instead of
    // only surfacing the aggregate. Never fires on the accepted path.
    if !(eta_derivative.is_finite() && eta_derivative >= 0.0) {
        let time_beta = beta.slice(s![..p_time]);
        let beta_min = time_beta.iter().copied().fold(f64::INFINITY, f64::min);
        let beta_max = time_beta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dtime = row_time.x_derivative_time.to_dense();
        let dmin = dtime.iter().copied().fold(f64::INFINITY, f64::min);
        let dmax = dtime.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        log::debug!(
            "[rp-predict/eta_t-refusal] eta_t={eta_derivative:.12e} = offset({offset_derivative_component:.12e}) + time({time_derivative_component:.12e}) + wiggle({wiggle_derivative_component:.12e}); p_time={p_time} p_timewiggle={p_timewiggle} p_cov={p_cov} time_beta=[{beta_min:.6e},{beta_max:.6e}] x_derivative_time=[{dmin:.6e},{dmax:.6e}] has_wiggle={}",
            saved_timewiggle.is_some(),
        );
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
    let eta_magnitude = x_exit
        .row(0)
        .iter()
        .zip(beta.iter())
        .map(|(x, b)| (x * b).abs())
        .sum::<f64>()
        + eta_time_offset_row.abs()
        + primary_offset_row.abs();
    let eta_band = gam_linalg::roundoff::accumulation_growth(p + 2) * eta_magnitude;
    // `cum = exp(eta)` is accurate to one ulp, `2u`, so the exact cumulative
    // hazard lies within `(1 + γ₂)·exp(±eta_band)` of it.
    let exp_growth = gam_linalg::roundoff::accumulation_growth(2);
    let cum_band = cum * (eta_band.exp_m1() * (1.0 + exp_growth) + exp_growth);
    Ok((eta, cum, haz, cum_band))
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
    // probit / marginal-slope sibling (`probit_survival_hazard_components`)
    // maps a zero derivative to a zero hazard; the RP guard must match.
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
    truncated_posterior: Option<(&TruncatedCoefficientDraws, Option<f64>)>,
) -> Result<(SurvivalPredictResult, Option<SurvivalPosteriorMoments>), String> {
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
    let (eta_full, survival_prob_full, log_survival_prob_full, response_se_full, eta_se_full): (
        Array1<f64>,
        Array1<f64>,
        Option<Array1<f64>>,
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
        // The posterior-mean survival is a closed-form response moment with no
        // log-space form; its cumulative hazard is `−ln` of the value it has.
        (
            unc.eta,
            unc.survival_prob,
            None,
            Some(response_se),
            Some(unc.eta_standard_error),
        )
    } else {
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|err| format!("survival location-scale predict failed: {err}"))?;
        (
            pred.eta,
            pred.survival_prob,
            Some(pred.log_survival_prob),
            None,
            None,
        )
    };

    let x_time_derivative = if reduced_parametric_aft {
        None
    } else {
        Some(
            time_build
                .x_derivative_time
                .try_to_dense_by_chunks("survival location-scale prediction time-derivative design")?,
        )
    };
    // The rate `η′ = dη/dt` of the location-scale index at every cell, under the
    // coefficients `fit` carries. The designs are fixed; only the coefficients
    // vary between the plug-in and the posterior nodes.
    let index_rate = |fit: &UnifiedFitResult| -> Result<Array1<f64>, String> {
        let beta_threshold = fit.beta_threshold();
        let beta_log_sigma = fit.beta_log_sigma();
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
        let hdot = match x_time_derivative.as_ref() {
            None => Array1::zeros(total_rows),
            Some(x_time_derivative) => location_scale_eta_derivative_components(
                &eval_entry,
                &eval_exit,
                x_time_derivative,
                &derivative_offset_exit,
                &pred_input.x_time_exit,
                &pred_input.eta_time_offset_exit,
                time_wiggle_knots.as_ref(),
                time_wiggle_degree,
                time_wiggle_ncols,
                fit,
            )?,
        };
        let inv_sigma = eta_log_sigma.mapv(crate::sigma_link::exp_sigma_inverse_from_eta_scalar);
        let q_base = -&eta_threshold * &inv_sigma;
        let mut qdot =
            &inv_sigma * &(&eta_threshold * &eta_log_sigma_derivative - &eta_threshold_derivative);
        if let Some(beta_wiggle) = fit.beta_link_wiggle() {
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
        // The scale divides the time transform too (#2695):
        // `g = e^{−η_σ}·(ḣ − h·η_σ') + qdot`, with `h` the same exit-time channel
        // the predicted residual reads.
        let h_exit = location_scale_time_warp_components(
            &pred_input.x_time_exit,
            &pred_input.eta_time_offset_exit,
            time_wiggle_knots.as_ref(),
            time_wiggle_degree,
            time_wiggle_ncols,
            fit,
        )?
        .h;
        Ok(&inv_sigma * &(&hdot - &(&h_exit * &eta_log_sigma_derivative)) + qdot)
    };
    let eta_derivative_full = index_rate(&saved_fit)?;
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

    let posterior_moments = match truncated_posterior {
        None => None,
        Some((draws, band_level)) => {
            // One posterior node's cells: the plug-in surfaces replayed at the
            // node's coefficients. The hazard keeps the sign of the node's rate,
            // so the node's density `S·h` is exactly `−dS/dt` there and the
            // integrated density is `−dE[S]/dt` ([`conditional_event_density`]).
            let node_cells = |fit: &UnifiedFitResult| -> Result<SurvivalNodeCells, String> {
                let pred = predict_survival_location_scale(&pred_input, fit)
                    .map_err(|err| format!("survival location-scale predict failed: {err}"))?;
                let rate = index_rate(fit)?;
                let hazard = pred
                    .eta
                    .iter()
                    .zip(rate.iter())
                    .map(|(&eta, &rate)| {
                        if rate == 0.0 {
                            Ok(0.0)
                        } else {
                            location_scale_hazard_component(eta, rate.abs(), &saved_inverse_link)
                                .map(|hazard| rate.signum() * hazard)
                        }
                    })
                    .collect::<Result<Array1<f64>, String>>()?;
                Ok(SurvivalNodeCells {
                    eta: pred.eta,
                    log_survival: pred.log_survival_prob,
                    hazard,
                })
            };
            let surface_cells: Vec<(usize, usize, usize)> = (0..n)
                .flat_map(|i| (0..t_cols).map(move |j| (i, j)))
                .filter(|&(i, j)| {
                    let query_time = if per_row_eval {
                        age_exit[i]
                    } else {
                        eval_times[j]
                    };
                    query_time > 0.0
                })
                .map(|(i, j)| (i, j, if per_row_eval { i } else { i * eval_width + j }))
                .collect();
            let eta_cells: Vec<usize> = (0..n)
                .map(|i| if per_row_eval { i } else { i * eval_width + t_cols })
                .collect();
            let mut moments = truncated_survival_surface_moments(
                draws,
                &saved_fit,
                &node_cells,
                &surface_cells,
                &eta_cells,
                t_cols,
            )?;
            if let Some(level) = band_level {
                moments.survival_band = Some(location_scale_truncated_surface_band(
                    &pred_input,
                    &saved_fit,
                    covariance_mode,
                    level,
                    &surface_cells,
                    n,
                    t_cols,
                )?);
            }
            Some(moments)
        }
    };

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
            // The cumulative hazard is `−ln S` in log space where the fit
            // provides it, so it stays finite wherever the linear predictor is;
            // a posterior-mean survival that has underflowed to exactly 0 has
            // the infinite cumulative hazard the model assigns it, not a floored
            // one (#2469).
            match log_survival_prob_full.as_ref() {
                Some(log_s) => {
                    *s = log_s[k].exp();
                    *ch = -log_s[k];
                }
                None => {
                    let surv = survival_prob_full[k];
                    *s = surv;
                    *ch = if surv > 0.0 { -surv.ln() } else { f64::INFINITY };
                }
            }
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

    Ok((
        SurvivalPredictResult {
            times,
            hazard,
            survival,
            cumulative_hazard,
            linear_predictor,
            likelihood_mode: saved_likelihood_mode,
            survival_se,
            eta_se: eta_se_per_row,
            covariance_source: with_uncertainty.then_some(covariance_mode),
            // This IS the plug-in prediction; `survival` carries it.
            survival_plugin: None,
            survival_lower: None,
            survival_upper: None,
        },
        posterior_moments,
    ))
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
    eval_entry: &Array1<f64>,
    eval_exit: &Array1<f64>,
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
        || eval_entry.len() != n
        || eval_exit.len() != n
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
    if let Some(row) = eta_derivative
        .iter()
        .position(|value| !(value.is_finite() && *value > 0.0))
    {
        let basis_part = if p_base > 0 {
            x_time_derivative.row(row).dot(&beta_base)
        } else {
            0.0
        };
        return Err(format!(
            "survival location-scale hazard derivative must be finite and positive: row {row} \
             (entry={:.6}, exit={:.6}) has dη/dt={:.6e} (basis part {basis_part:.6e}, offset part \
             {:.6e}, time-wiggle dq {:?})",
            eval_entry[row],
            eval_exit[row],
            eta_derivative[row],
            derivative_offset_exit[row],
            time_components
                .time_wiggle_dq
                .as_ref()
                .map(|dq| dq[row]),
        ));
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
    saved_fit_result(model).cloned()
}

/// Borrow the saved canonical fit result, for readers that need no owned copy.
pub fn saved_fit_result(model: &SavedModel) -> Result<&UnifiedFitResult, String> {
    model
        .fit_result
        .as_ref()
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
            let entry =
                monotone_wiggle_basis_with_derivative_order(eta_entry.view(), &knots, degree, 0)?;
            let exit =
                monotone_wiggle_basis_with_derivative_order(eta_exit.view(), &knots, degree, 0)?;
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
    slope_design: &DesignMatrix,
    time_build: &SurvivalTimeBuildOutput,
    eta_offset_entry: &Array1<f64>,
    eta_offset_exit: &Array1<f64>,
    derivative_offset_exit: &Array1<f64>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    local_law_conditioning: Option<Array2<f64>>,
) -> Result<
    (
        BernoulliMarginalSlopePredictor,
        PredictInput,
        UnifiedFitResult,
    ),
    SurvivalPredictError,
> {
    if model.survival_marginal_slope_joint_latent_law.is_some() {
        // gam#2929: the model's latent object is a joint law of K ≥ 2 scores.
        // This predictor carries one score and one scalar law, so serving it
        // would replay a different model; the joint replay lives in
        // `build_joint_marginal_slope_predict_context`.
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "saved survival marginal-slope model is anchored on the joint latent law of \
                     K ≥ 2 scores; the single-score predictor cannot replay it"
                .to_string(),
        });
    }
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
    let beta_slope = &blocks[2].beta;
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
    if beta_slope.len() != slope_design.ncols() {
        return Err(SurvivalPredictError::IncompatibleSchema {
            reason: format!(
                "saved survival marginal-slope slope coefficient mismatch: beta has {} entries but slope design has {} columns",
                beta_slope.len(),
                slope_design.ncols()
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
            &time_build.basisname,
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
        beta: beta_slope.clone(),
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
        model.baseline_slope.ok_or_else(|| {
            "saved survival marginal-slope model missing baseline_slope".to_string()
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
        model.latent_z_conditional_calibration.clone(),
        // gam#2768: the survival marginal-slope now runs the same automatic
        // conditional gate BMS does, so this field is no longer always `None` —
        // and the survival predictor's primary design is the q-design
        // `[time | timewiggle | marginal]`, NOT the marginal design. The span the
        // fit conditioned on is its trailing covariate block; naming anything
        // else here would rebuild `a(C)` from time columns and apply a different
        // map from the one the fit applied.
        LatentConditioningSpan::PrimaryDesignTail {
            ncols: cov_design.ncols(),
        },
        // The residual repair block (gam#2924) is a Bernoulli-only block until
        // the survival kernel takes it (gam#2923).
        None,
    )?;

    let pred_input = PredictInput {
        design: q_design,
        offset: eta_offset_exit + primary_offset,
        design_noise: Some(slope_design.clone()),
        offset_noise: Some(noise_offset.clone()),
        auxiliary_scalar: Some(z.clone()),
        // gam#2926: a local latent law is replayed from the context covariates
        // of the prediction rows, exactly as the Bernoulli predictor replays it.
        auxiliary_matrix: local_law_conditioning,
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
mod tests;

/// Multiplier applied to the Weibull baseline scale when no time-basis knots are
/// available, so the default surface grid reaches into the right tail of the
/// fitted distribution rather than stopping at the characteristic time.
const SURVIVAL_DEFAULT_GRID_SCALE_MARGIN: f64 = 5.0;

/// Training-time upper bound for the default survival surface grid, read from
/// the saved model rather than the prediction frame.
///
/// The default surface grid must be a property of the FITTED model, not of the
/// `exit` placeholder a caller happens to put in the prediction frame. A small
/// placeholder `exit` previously shrank the grid to `[entry, exit]` and silently
/// truncated the surface (#896); a large one stretched the fixed grid past the
/// fitted range and coarsened every in-range cell (#1717).
///
/// Returns `None` when the model carries neither a time-basis knot vector nor a
/// usable training range / baseline scale (the caller then falls back to the
/// prediction-frame range alone, preserving the prior behavior).
pub fn survival_training_time_upper_bound(payload: &FittedModelPayload) -> Option<f64> {
    if let Some(knots) = payload.survival_time_knots.as_ref() {
        let max_log_knot = knots
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        if max_log_knot.is_finite() {
            let hi = max_log_knot.exp();
            if hi.is_finite() && hi > 0.0 {
                return Some(hi);
            }
        }
    }

    // No time-basis knots (the linear Weibull baseline). Two candidate anchors:
    //
    //   * the model's recorded TRAINING time support — the upper end of the
    //     survival exit column's training range — which is the true observed
    //     upper bound of the fitted distribution; and
    //   * a margin over the parametric Weibull `survival_baseline_scale`, to
    //     reach into the right tail.
    //
    // Take the LARGER so the surface grid always covers the observed time range.
    // The scale field alone is unreliable here: on the covariate-driven Weibull
    // parameterization the baseline scale is absorbed into the linear predictor
    // and `survival_baseline_scale` is left as a degenerate floor sentinel
    // (≈ SURVIVAL_TIME_FLOOR ≈ 1e-9). Anchoring on it alone collapsed the grid
    // to ~0 and truncated every `survival_at` query past the prediction frame's
    // `exit` placeholder, even for times well inside the fitted range (#896).
    let mut upper = f64::NEG_INFINITY;
    if let Some(training_hi) = survival_training_exit_upper_bound(payload) {
        upper = upper.max(training_hi);
    }
    if let Some(scale) = payload.survival_baseline_scale
        && scale.is_finite()
        && scale > 0.0
    {
        upper = upper.max(scale * SURVIVAL_DEFAULT_GRID_SCALE_MARGIN);
    }
    (upper.is_finite() && upper > 0.0).then_some(upper)
}

/// Upper end of the survival exit column's recorded training range.
///
/// Used to anchor the default survival-surface grid to the fitted model's
/// observed time support when the parametric baseline carries no usable time
/// signal (the linear Weibull basis: no knots, and a degenerate
/// `survival_baseline_scale` sentinel — #896). Returns `None` when the model
/// carries no training-range metadata or the exit column cannot be located.
fn survival_training_exit_upper_bound(payload: &FittedModelPayload) -> Option<f64> {
    let exit_name = payload.survival_exit.as_deref()?;
    let headers = payload.training_headers.as_ref()?;
    let ranges = payload.training_feature_ranges.as_ref()?;
    let idx = headers.iter().position(|h| h == exit_name)?;
    let (_, hi) = *ranges.get(idx)?;
    (hi.is_finite() && hi > 0.0).then_some(hi)
}

/// The default survival-surface time grid: 64 uniform points spanning the
/// prediction frame's `[min entry, max exit]`, with the upper edge anchored to
/// the fitted model's training time support when one is supplied.
///
/// This is the SINGLE owner of the default-grid policy for every frontend
/// (#2470); it used to live only in the Python bindings, so `gam predict` and
/// `model.predict()` produced different survival surfaces by default.
///
/// The `entry_name == None` case is the right-censored shorthand
/// `Surv(time, event)`: every subject enters at time zero, so the grid lower
/// bound is zero and there is no entry column to read per row. The anchor CAPS
/// (and floors) the frame's `hi` — the frame's `exit` placeholder is a
/// semantically meaningless response value that `survival_at` ignores, so it
/// must neither truncate the surface below the fitted range (#896) nor stretch
/// the fixed 64-point grid past it (#1717); query times legitimately beyond
/// the support are handled by `survival_at`'s extrapolation (#1595).
pub fn default_survival_time_grid(
    formula: &str,
    dataset: &EncodedDataset,
    training_time_upper: Option<f64>,
) -> Result<Option<Vec<f64>>, String> {
    let parsed = gam_terms::inference::formula_dsl::parse_formula(formula)
        .map_err(|err| format!("failed to parse survival formula: {err}"))?;
    let Some((entry_name, exit_name, _event_name)) =
        gam_terms::inference::formula_dsl::parse_surv_response(&parsed.response)
            .map_err(|err| format!("failed to parse Surv(...) response: {err}"))?
    else {
        return Ok(None);
    };

    let header_to_index: HashMap<&str, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let entry_idx = match entry_name.as_deref() {
        Some(name) => match header_to_index.get(name).copied() {
            Some(idx) => Some(idx),
            None => {
                return Err(format!(
                    "survival prediction data is missing required time column(s): {name}"
                ));
            }
        },
        None => None,
    };
    let exit_idx = match header_to_index.get(exit_name.as_str()).copied() {
        Some(idx) => idx,
        None => {
            return Err(format!(
                "survival prediction data is missing required time column(s): {exit_name}"
            ));
        }
    };

    if let Some(index) = entry_idx
        && matches!(
            dataset.schema.columns[index].kind,
            gam_data::ColumnKindTag::Categorical
        )
    {
        return Err(format!(
            "survival entry column '{}' is categorical, expected numeric times",
            entry_name.as_deref().unwrap_or_default()
        ));
    }
    if matches!(
        dataset.schema.columns[exit_idx].kind,
        gam_data::ColumnKindTag::Categorical
    ) {
        return Err(format!(
            "survival exit column '{exit_name}' is categorical, expected numeric times"
        ));
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for row_index in 0..dataset.values.nrows() {
        let entry_value = match entry_idx {
            None => 0.0,
            Some(index) => dataset.values[[row_index, index]],
        };
        let exit_value = dataset.values[[row_index, exit_idx]];
        if !entry_value.is_finite() || !exit_value.is_finite() {
            return Err("survival time columns must contain only finite values".to_string());
        }
        lo = lo.min(entry_value);
        hi = hi.max(exit_value);
    }
    if dataset.values.nrows() == 0 {
        return Ok(None);
    }
    if let Some(training_hi) = training_time_upper
        && training_hi.is_finite()
    {
        hi = training_hi;
    }
    if hi <= lo {
        return Err(format!(
            "survival exit times must extend beyond entry times; got min entry {lo:?} and max exit {hi:?}"
        ));
    }
    // 64 uniform points with BOTH endpoints exact. The last point is `hi`
    // itself, not the accumulation `lo + 63·step`, which lands off the frame's
    // upper edge by the step's own rounding; the padding that used to hide that
    // pushed the grid PAST `hi`, which is the stretch #1717 rules out and which
    // this function's own contract ("spanning the prediction frame's
    // `[min entry, max exit]`") forbids. Every interior point is
    // `lo + (hi − lo)·t` with `t < 1`, so it lies strictly inside the frame and
    // the sequence is monotone.
    const GRID_POINTS: usize = 64;
    let last = GRID_POINTS - 1;
    Ok(Some(
        (0..GRID_POINTS)
            .map(|index| {
                if index == last {
                    hi
                } else {
                    lo + (hi - lo) * (index as f64 / last as f64)
                }
            })
            .collect(),
    ))
}
