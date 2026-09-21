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

    let (beta_time, beta_threshold, beta_log_sigma, beta_link_wiggle, beta_time_blocks) = {
        let fit = payload.fit_result.as_mut().ok_or_else(|| {
            SurvivalPredictError::MissingFitMetadata {
                reason: "saved survival model is missing canonical fit_result".to_string(),
            }
        })?;
        assign_survival_fit_coefficients(fit, coefficients)?;
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
    predict_survival_posterior_mean_integrated(req, covariance_mode, integration, integration)
}

/// [`predict_survival_posterior_mean_with`] with the point's conditional law
/// integrated by `point_integration` and a smoothing-corrected band's law by
/// `band_integration`.
fn predict_survival_posterior_mean_integrated(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    point_integration: SurvivalPosteriorIntegration,
    band_integration: SurvivalPosteriorIntegration,
) -> Result<SurvivalPredictResult, SurvivalPredictError> {
    let point_request = SurvivalPredictRequest {
        with_uncertainty: false,
        ..req
    };
    let (mut result, point) = survival_posterior_moments(
        point_request,
        SurvivalPredictionCovarianceMode::Conditional,
        point_integration,
    )?;
    let band = match (req.with_uncertainty, covariance_mode) {
        (false, _) => None,
        (true, SurvivalPredictionCovarianceMode::Conditional) => None,
        (true, SurvivalPredictionCovarianceMode::SmoothingCorrected) => Some(
            survival_posterior_moments(point_request, covariance_mode, band_integration)?.1,
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
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    match integration {
        SurvivalPosteriorIntegration::SigmaPoint => {
            survival_sigma_point_posterior_moments(req, covariance_mode)
        }
        SurvivalPosteriorIntegration::ExactAnchor => {
            survival_exact_anchor_posterior_moments(req, covariance_mode)
        }
        SurvivalPosteriorIntegration::TruncatedLaw => {
            survival_truncated_law_posterior_moments(req, covariance_mode)
        }
    }
}

fn predict_survival_posterior_mean(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
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
    )
}

/// First and second posterior moments of a single-event survival prediction,
/// row × time (row for the linear predictor at each row's own exit time), as
/// either [`SurvivalPosteriorIntegration`] accumulates them.
struct SurvivalPosteriorMoments {
    survival_mean: Array2<f64>,
    survival_second: Array2<f64>,
    density_mean: Array2<f64>,
    hazard_mean: Array2<f64>,
    eta_mean: Array1<f64>,
    eta_second: Array1<f64>,
}

impl SurvivalPosteriorMoments {
    fn zeros(n_rows: usize, n_times: usize) -> Self {
        Self {
            survival_mean: Array2::zeros((n_rows, n_times)),
            survival_second: Array2::zeros((n_rows, n_times)),
            density_mean: Array2::zeros((n_rows, n_times)),
            hazard_mean: Array2::zeros((n_rows, n_times)),
            eta_mean: Array1::zeros(n_rows),
            eta_second: Array1::zeros(n_rows),
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
        if published.survival_mean.dim() != (n_rows, n_times)
            || published.eta_mean.len() != n_rows
        {
            return Err(SurvivalPredictError::IncompatibleSchema {
                reason: format!(
                    "posterior survival moments have shape {:?}, but the prediction is {n_rows}x{n_times}",
                    published.survival_mean.dim()
                ),
            });
        }
    }
    let SurvivalPosteriorMoments {
        survival_mean,
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
            let survival = survival_mean[[row, time]].clamp(0.0, 1.0);
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
    result.survival_se = uncertainty.map(|(band, _)| {
        Array2::from_shape_fn((n_rows, n_times), |(row, time)| {
            let mean = band.survival_mean[[row, time]];
            (band.survival_second[[row, time]] - mean * mean)
                .max(0.0)
                .sqrt()
        })
    });
    result.eta_se = uncertainty.map(|(band, _)| {
        Array1::from_shape_fn(n_rows, |row| {
            let mean = band.eta_mean[row];
            (band.eta_second[row] - mean * mean).max(0.0).sqrt()
        })
    });
    result.covariance_source = uncertainty.map(|(_, covariance_mode)| covariance_mode);
    result.survival_plugin = Some(survival_plugin);
    Ok(())
}

/// `(E S, E S², E f, E h, E η, E η²)` of one marginal-slope `(row, t)` cell over
/// the coefficient posterior: survival `S = Φ(−η)`, event density
/// `f = φ(η)·η′`, hazard `h = f/S`, and the linear predictor. The published
/// hazard is `E f / E S` ([`publish_survival_posterior_moments`]); `E h` only
/// decides between a zero and an infinite hazard where `E S = 0`.
pub(crate) type ExactAnchorCellMoments = (f64, f64, f64, f64, f64, f64);

/// One node of the exact anchored posterior integral at the primaries
/// `(q, b)`: `(S, S², f, h, η, η²)` with `S = Φ(−η)`, event density
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
    let survival = log_survival.exp();
    (
        survival,
        survival * survival,
        normal_pdf(eta) * rate,
        mills_ratio * rate,
        eta,
        eta * eta,
    )
}

/// The coefficient posterior [`SurvivalPosteriorIntegration::ExactAnchor`]
/// pushes onto every cell's primaries: the active covariance over the
/// `[time | marginal | slope]` coefficients.
struct ExactAnchorPosterior {
    covariance: Array2<f64>,
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
    fn cell_moments(
        &self,
        quadctx: &gam_solve::quadrature::QuadratureContext,
        ctx: &MarginalSlopePredictContext,
        cell: &MarginalSlopeCell,
    ) -> Result<ExactAnchorCellMoments, SurvivalPredictError> {
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

        gam_solve::quadrature::normal_expectation_2d_projected_result(
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
        )
    }
}

fn survival_exact_anchor_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    let (_, active_covariance, _) =
        survival_prediction_posterior_factor(req.model, covariance_mode)?;
    let posterior = ExactAnchorPosterior {
        covariance: active_covariance,
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

/// The cone-truncated coefficient posterior of a location-scale or
/// Royston-Parmar fit under `covariance_mode`, as a rule over whole coefficient
/// vectors, or `None` when the selected law retains no constraint row.
fn truncated_survival_posterior_draws(
    model: &SavedModel,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<Option<TruncatedCoefficientDraws>, SurvivalPredictError> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    fit.require_posterior_mean("survival posterior-mean prediction")
        .map_err(|error| SurvivalPredictError::PosteriorCovariance {
            reason: error.to_string(),
        })?;
    let covariance = select_survival_prediction_covariance(
        fit.beta_covariance(),
        fit.beta_covariance_corrected(),
        covariance_mode,
    )?;
    build_truncated_coefficient_draws(&fit, covariance)
        .map_err(|reason| SurvivalPredictError::PosteriorCovariance { reason })
}

fn survival_truncated_law_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    let draws = truncated_survival_posterior_draws(req.model, covariance_mode)?.ok_or_else(|| {
        SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "the truncated-law survival posterior needs a location-scale or Royston-Parmar \
                 fit whose {} posterior carries an active inequality cone",
                covariance_mode.as_str()
            ),
        }
    })?;
    if require_saved_survival_likelihood_mode(req.model)? != SurvivalLikelihoodMode::LocationScale {
        return survival_replayed_truncated_law_posterior_moments(req, covariance_mode, &draws);
    }
    let (result, moments) = predict_survival_surfaces(
        SurvivalPredictRequest {
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
            ..req
        },
        covariance_mode,
        Some(SurvivalSurfacePosterior::TruncatedLaw(&draws)),
    )?;
    let moments = moments.ok_or_else(|| {
        "internal error: the truncated-law survival pass returned no posterior moments".to_string()
    })?;
    refuse_decreasing_survival(&result)?;
    Ok((result, moments))
}

/// [`SurvivalPosteriorIntegration::TruncatedLaw`] for a fit whose plug-in
/// survival law is replayed whole at each coefficient vector rather than from
/// a location-scale batch — the Royston-Parmar fit, whose monotone I-spline
/// baseline is the cone `β_j ≥ 0` (gam#3575).
///
/// Every node of the law's joint rule is a feasible coefficient vector, so its
/// baseline is monotone and [`predict_survival_coefficient_law`] evaluates it
/// as it evaluates the fit's own coefficients. The cells are the plug-in
/// surfaces flattened row-major, `k = row·n_times + time`, followed by one
/// linear-predictor cell per row, and are certified exactly as the
/// location-scale surfaces are ([`truncated_survival_surface_moments`]).
fn survival_replayed_truncated_law_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    draws: &TruncatedCoefficientDraws,
) -> Result<(SurvivalPredictResult, SurvivalPosteriorMoments), SurvivalPredictError> {
    fn plugin_request<'a: 'b, 'b>(
        req: &SurvivalPredictRequest<'a>,
        model: &'b SavedModel,
    ) -> SurvivalPredictRequest<'b> {
        SurvivalPredictRequest {
            model,
            data: req.data,
            col_map: req.col_map,
            training_headers: req.training_headers,
            primary_offset: req.primary_offset,
            noise_offset: req.noise_offset,
            time_grid: req.time_grid,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        }
    }
    let fit = fit_result_from_saved_model_for_prediction(req.model)?;
    let result = predict_survival(plugin_request(&req, req.model), covariance_mode)?;
    let (n_rows, n_times) = result.survival.dim();
    let surface_width = n_rows * n_times;
    let times = result.times.clone();
    let likelihood_mode = result.likelihood_mode;
    let node_cells = |node_fit: &UnifiedFitResult| -> Result<SurvivalNodeCells, String> {
        let draw_model = saved_model_with_survival_coefficients(req.model, &node_fit.beta)?;
        let draw =
            predict_survival_coefficient_law(plugin_request(&req, &draw_model), covariance_mode)?;
        if draw.survival.dim() != (n_rows, n_times)
            || draw.hazard.dim() != (n_rows, n_times)
            || draw.cumulative_hazard.dim() != (n_rows, n_times)
            || draw.linear_predictor.len() != n_rows
            || draw.times != times
            || draw.likelihood_mode != likelihood_mode
        {
            return Err(
                "truncated posterior survival node changed the prediction schema".to_string(),
            );
        }
        let mut eta = Array1::<f64>::zeros(surface_width + n_rows);
        let mut log_survival = Array1::<f64>::zeros(surface_width + n_rows);
        let mut hazard = Array1::<f64>::zeros(surface_width + n_rows);
        for row in 0..n_rows {
            let linear_predictor = draw.linear_predictor[row];
            eta[surface_width + row] = linear_predictor;
            for time in 0..n_times {
                let k = row * n_times + time;
                eta[k] = linear_predictor;
                log_survival[k] = -draw.cumulative_hazard[[row, time]];
                hazard[k] = draw.hazard[[row, time]];
            }
        }
        Ok(SurvivalNodeCells {
            eta,
            log_survival,
            hazard,
        })
    };
    let surface_cells: Vec<(usize, usize, usize)> = (0..n_rows)
        .flat_map(|row| (0..n_times).map(move |time| (row, time, row * n_times + time)))
        .collect();
    let eta_cells: Vec<usize> = (0..n_rows).map(|row| surface_width + row).collect();
    let moments = truncated_survival_surface_moments(
        draws,
        &fit,
        &node_cells,
        &surface_cells,
        &eta_cells,
        n_times,
    )?;
    // The plug-in survival is published beside the posterior mean
    // (`survival_plugin`), so its curve is held to the same domain as when it
    // is published alone.
    refuse_decreasing_survival(&result)?;
    Ok((result, moments))
}

fn survival_sigma_point_posterior_moments(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
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
    let mut moments = SurvivalPosteriorMoments::zeros(n_rows, n_times);
    let SurvivalPosteriorMoments {
        survival_mean,
        survival_second,
        density_mean,
        hazard_mean,
        eta_mean,
        eta_second,
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

    Ok((result, moments))
}

fn predict_competing_risks_with_posterior(
    req: SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
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
    TruncatedLaw(&'a TruncatedCoefficientDraws),
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
        Some(SurvivalSurfacePosterior::TruncatedLaw(draws)) => (None, Some(draws)),
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
            type CellOutput = ((f64, f64, f64), Option<ExactAnchorCellMoments>);
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
            };
            if per_row_eval {
                let ((eta_t, cum_t, haz_t), posterior) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
                row.hazard[0] = haz_t;
                row.cumulative_hazard[0] = cum_t;
                row.survival[0] = (-cum_t).exp().clamp(0.0, 1.0);
                row.posterior_cells.extend(posterior);
                row.posterior_exit = posterior;
            } else {
                for (j, &t_query) in eval_times.iter().enumerate() {
                    if t_query <= 0.0 {
                        row.hazard[j] = 0.0;
                        row.cumulative_hazard[j] = 0.0;
                        row.survival[j] = 1.0;
                        // At the time origin every coefficient draw has S = 1 and
                        // no hazard, so the moments carry no posterior spread.
                        if exact_posterior.is_some() {
                            row.posterior_cells.push((1.0, 1.0, 0.0, 0.0, 0.0, 0.0));
                        }
                    } else {
                        let ((_eta_t, cum_t, haz_t), posterior) = evaluate_at(t_query)?;
                        row.hazard[j] = haz_t;
                        row.cumulative_hazard[j] = cum_t;
                        row.survival[j] = (-cum_t).exp().clamp(0.0, 1.0);
                        row.posterior_cells.extend(posterior);
                    }
                }
                let ((eta_t, _, _), posterior) = evaluate_at(age_exit[i])?;
                row.linear_predictor = eta_t;
                row.posterior_exit = posterior;
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
        .map(|_| {
            let mut moments = SurvivalPosteriorMoments::zeros(n, t_cols);
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
                moments.eta_mean[i] = exit.4;
                moments.eta_second[i] = exit.5;
                for (j, cell) in row.posterior_cells.iter().enumerate() {
                    moments.survival_mean[[i, j]] = cell.0;
                    moments.survival_second[[i, j]] = cell.1;
                    moments.density_mean[[i, j]] = cell.2;
                    moments.hazard_mean[[i, j]] = cell.3;
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
        },
        posterior_moments,
    ))
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
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: "saved survival marginal-slope joint latent law was fitted on uncalibrated \
                     scores; this model also carries a per-score calibration"
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
    let mut scores = Array2::<f64>::zeros((n, k));
    for (column, name) in z_columns.iter().enumerate() {
        let index = *col_map
            .get(name)
            .ok_or_else(|| format!("missing score column '{name}'"))?;
        scores
            .column_mut(column)
            .assign(&data.column(index).mapv(|z| law.standardized_score(column, z)));
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
    truncated_posterior: Option<&TruncatedCoefficientDraws>,
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
        Some(draws) => {
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
            Some(truncated_survival_surface_moments(
                draws,
                &saved_fit,
                &node_cells,
                &surface_cells,
                &eta_cells,
                t_cols,
            )?)
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
        },
        posterior_moments,
    ))
}

/// One coefficient vector's survival cells, in the caller's flattened
/// `(row, time)` layout: the linear predictor, `log S`, and the hazard, signed
/// where the node's survival rises (the location-scale index's rate, the
/// Royston-Parmar `dH/dt`).
struct SurvivalNodeCells {
    eta: Array1<f64>,
    log_survival: Array1<f64>,
    hazard: Array1<f64>,
}

impl SurvivalNodeCells {
    /// `(S, f)` at flattened cell `k`, `f = S·h` the node's event density.
    fn survival_and_density(&self, k: usize) -> Result<(f64, f64), String> {
        let log_survival = self.log_survival[k];
        let survival = log_survival.exp();
        let density = conditional_event_density(survival, -log_survival, self.hazard[k])
            .map_err(String::from)?;
        Ok((survival, density))
    }
}

/// The posterior moments of the survival surfaces over the cone-truncated
/// coefficient law — location-scale (gam#3038) and Royston-Parmar (gam#3575) —
/// certified per cell.
///
/// Every node of the law's joint rule is a feasible coefficient vector, and the
/// surfaces are replayed at it through `node_cells`. Each replicate lattice keeps
/// its own sums; a cell is retired once the replicate standard error of its
/// moments, less the integrand's rounding, is within the law's certified
/// relative accuracy — survival as a fraction of `sqrt(E[S](1 − E[S]))` (the
/// largest standard deviation a variable in `[0, 1]` with that mean can have),
/// the event density as a fraction of the larger of `|E[f]|` and its posterior
/// standard deviation (the published hazard is `E[f]/E[S]`, a ratio, so the
/// density is resolved relative to itself; where the posterior spread of `f`
/// exceeds its mean — deep in a tail, where the survival certificate already
/// resolves `E[S]` only to a fraction of `sqrt(E[S])` — resolving it to a
/// fraction of that spread keeps the Monte Carlo error a fraction `tol` of the
/// posterior uncertainty the surface carries), the linear predictor as a
/// fraction of its posterior standard deviation. Past the rule's maximum node count the moments are
/// refused, naming the worst cell, never reported.
///
/// `surface_cells` lists `(row, time column, flattened cell)` for every
/// published cell after the origin; origin cells are `S = 1`, `f = h = 0`
/// exactly. `eta_cells` is each row's linear-predictor cell.
fn truncated_survival_surface_moments(
    draws: &TruncatedCoefficientDraws,
    fit: &UnifiedFitResult,
    node_cells: &(dyn Fn(&UnifiedFitResult) -> Result<SurvivalNodeCells, String> + Sync),
    surface_cells: &[(usize, usize, usize)],
    eta_cells: &[usize],
    t_cols: usize,
) -> Result<SurvivalPosteriorMoments, String> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let rule = draws.rule();
    let replicates = rule.replicates();
    if replicates < 2 {
        return Err(format!(
            "survival truncated posterior surfaces need at least two replicate \
             lattices to certify on; the joint rule carries {replicates}"
        ));
    }
    let tolerance = rule.relative_tolerance();
    // The sums run about the plug-in cells so their rounding is relative to the
    // posterior spread, not to the values (see `ResponseMomentAccumulator`).
    let reference = node_cells(fit)?;
    let total_cells = reference.eta.len();
    let reference_surface = surface_cells
        .iter()
        .map(|&(_, _, k)| reference.survival_and_density(k))
        .collect::<Result<Vec<_>, String>>()?;
    let mut accumulators: Vec<SurfaceMomentAccumulator> =
        std::iter::repeat_with(|| SurfaceMomentAccumulator::new(total_cells))
            .take(replicates)
            .collect();
    let n_rows = eta_cells.len();
    let mut moments = SurvivalPosteriorMoments::zeros(n_rows, t_cols);
    moments.survival_mean.fill(1.0);
    moments.survival_second.fill(1.0);
    let mut active_surface: Vec<usize> = (0..surface_cells.len()).collect();
    let mut active_eta: Vec<usize> = (0..n_rows).collect();
    let mut evaluated = 0usize;
    while !(active_surface.is_empty() && active_eta.is_empty()) {
        let target = if evaluated == 0 {
            rule.initial_points()
        } else {
            2 * evaluated
        };
        let cells = SurfaceMomentCells {
            surface_cells,
            eta_cells,
            reference: &reference,
            reference_surface: &reference_surface,
            active_surface: &active_surface,
            active_eta: &active_eta,
        };
        accumulators
            .as_mut_slice()
            .into_par_iter()
            .enumerate()
            .try_for_each(|(replicate, accumulator)| {
                let mut node_fit = fit.clone();
                rule.visit_nodes(
                    replicate,
                    evaluated,
                    target,
                    |log_weight, normal_coordinates, tangent| {
                        let coefficients = draws.coefficients(normal_coordinates, tangent)?;
                        assign_survival_fit_coefficients(&mut node_fit, &coefficients)
                            .map_err(String::from)?;
                        accumulator.push(&cells, log_weight, &node_cells(&node_fit)?)
                    },
                )
            })?;
        evaluated = target;
        let pooled = PooledSurfaceMoments::new(&accumulators)?;
        let mut worst: Option<(String, f64)> = None;
        let mut note_uncertified = |label: String, error: f64| {
            if worst.as_ref().is_none_or(|(_, current)| error > *current) {
                worst = Some((label, error));
            }
        };
        let mut uncertified_surface = Vec::with_capacity(active_surface.len());
        for &cell in &active_surface {
            let (row, time, k) = surface_cells[cell];
            let (reference_survival, reference_density) = reference_surface[cell];
            let survival =
                pooled.moments(|a| (a.survival[k], a.survival_square[k]), reference_survival);
            let density = pooled.moments(|a| (a.density[k], a.density_square[k]), reference_density);
            let hazard = pooled.mean(|a| a.hazard[k]);
            let first = survival.mean.clamp(0.0, 1.0);
            let second = (survival.variance + first * first).clamp(0.0, 1.0);
            // `S = exp(log S)` resolves a probability to `f64::EPSILON` absolute
            // and `f` to one part in `f64::EPSILON` of itself: spread within that
            // is the integrand's rounding, which no node count removes.
            let survival_error = certified_fraction(
                survival.mean_spread.max(survival.sd_spread) - f64::EPSILON,
                (first * (1.0 - first)).sqrt(),
            );
            let density_error = certified_fraction(
                density.mean_spread
                    - f64::EPSILON * reference_density.abs().max(density.mean.abs()),
                density.mean.abs().max(density.variance.sqrt()),
            );
            let error = survival_error.max(density_error);
            if error <= tolerance {
                moments.survival_mean[[row, time]] = first;
                moments.survival_second[[row, time]] = second;
                moments.density_mean[[row, time]] = density.mean;
                moments.hazard_mean[[row, time]] = hazard;
            } else {
                uncertified_surface.push(cell);
                note_uncertified(format!("row {row}, time column {time}"), error);
            }
        }
        let mut uncertified_eta = Vec::with_capacity(active_eta.len());
        for &row in &active_eta {
            let k = eta_cells[row];
            let reference_eta = reference.eta[k];
            let eta = pooled.moments(|a| (a.eta[k], a.eta_square[k]), reference_eta);
            let error = certified_fraction(
                eta.mean_spread.max(eta.sd_spread)
                    - f64::EPSILON * reference_eta.abs().max(eta.mean.abs()),
                eta.variance.sqrt(),
            );
            if error <= tolerance {
                moments.eta_mean[row] = eta.mean;
                moments.eta_second[row] = eta.variance + eta.mean * eta.mean;
            } else {
                uncertified_eta.push(row);
                note_uncertified(format!("row {row}'s linear predictor"), error);
            }
        }
        active_surface = uncertified_surface;
        active_eta = uncertified_eta;
        let nodes = evaluated * replicates;
        if let Some((cell, error)) = worst
            && nodes >= rule.maximum_points()
        {
            return Err(format!(
                "survival truncated posterior surfaces did not certify: after \
                 {nodes} joint cubature nodes over {replicates} replicate lattices, the replicate \
                 standard error at {cell}, less the integrand's rounding, is {error:.3e} of its \
                 scale, above the law's certified relative accuracy {tolerance:.1e}"
            ));
        }
    }
    Ok(moments)
}

/// `excess / scale`, with an excess within rounding certified outright and a
/// positive excess on a zero scale never certified.
fn certified_fraction(excess: f64, scale: f64) -> f64 {
    if excess <= 0.0 {
        0.0
    } else if scale > 0.0 {
        excess / scale
    } else {
        f64::INFINITY
    }
}

/// What one node's accumulation reads, shared by every replicate.
struct SurfaceMomentCells<'a> {
    surface_cells: &'a [(usize, usize, usize)],
    eta_cells: &'a [usize],
    reference: &'a SurvivalNodeCells,
    reference_surface: &'a [(f64, f64)],
    active_surface: &'a [usize],
    active_eta: &'a [usize],
}

/// One replicate lattice's weighted sums, on one log scale, over the flattened
/// cells: deviations of `S`, `f` and `η` from the plug-in cell and their
/// squares, and the raw hazard, which only decides between a
/// zero and an infinite published hazard where `E[S] = 0`.
struct SurfaceMomentAccumulator {
    log_scale: f64,
    weight_sum: f64,
    survival: Array1<f64>,
    survival_square: Array1<f64>,
    density: Array1<f64>,
    density_square: Array1<f64>,
    hazard: Array1<f64>,
    eta: Array1<f64>,
    eta_square: Array1<f64>,
}

impl SurfaceMomentAccumulator {
    fn new(cells: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            weight_sum: 0.0,
            survival: Array1::zeros(cells),
            survival_square: Array1::zeros(cells),
            density: Array1::zeros(cells),
            density_square: Array1::zeros(cells),
            hazard: Array1::zeros(cells),
            eta: Array1::zeros(cells),
            eta_square: Array1::zeros(cells),
        }
    }

    fn push(
        &mut self,
        cells: &SurfaceMomentCells<'_>,
        log_weight: f64,
        node: &SurvivalNodeCells,
    ) -> Result<(), String> {
        if log_weight > self.log_scale {
            let rescale = (self.log_scale - log_weight).exp();
            self.weight_sum *= rescale;
            for sums in [
                &mut self.survival,
                &mut self.survival_square,
                &mut self.density,
                &mut self.density_square,
                &mut self.hazard,
                &mut self.eta,
                &mut self.eta_square,
            ] {
                *sums *= rescale;
            }
            self.log_scale = log_weight;
        }
        let weight = (log_weight - self.log_scale).exp();
        // A node whose weight underflowed on this scale contributes nothing, and
        // must not turn an infinite hazard into `0·∞`.
        if weight == 0.0 {
            return Ok(());
        }
        self.weight_sum += weight;
        for &cell in cells.active_surface {
            let (_, _, k) = cells.surface_cells[cell];
            let (reference_survival, reference_density) = cells.reference_surface[cell];
            let (survival, density) = node.survival_and_density(k)?;
            let survival_deviation = survival - reference_survival;
            self.survival[k] += weight * survival_deviation;
            self.survival_square[k] += weight * survival_deviation * survival_deviation;
            let density_deviation = density - reference_density;
            self.density[k] += weight * density_deviation;
            self.density_square[k] += weight * density_deviation * density_deviation;
            self.hazard[k] += weight * node.hazard[k];
        }
        for &row in cells.active_eta {
            let k = cells.eta_cells[row];
            let deviation = node.eta[k] - cells.reference.eta[k];
            self.eta[k] += weight * deviation;
            self.eta_square[k] += weight * deviation * deviation;
        }
        Ok(())
    }
}

/// The replicate lattices pooled on the heaviest replicate's log scale.
struct PooledSurfaceMoments<'a> {
    accumulators: &'a [SurfaceMomentAccumulator],
    pooling_scales: Vec<f64>,
    pooled_weight: f64,
}

/// One quantity's pooled mean and variance, and the replicate standard errors
/// of its mean and of its standard deviation.
struct CellMoments {
    mean: f64,
    variance: f64,
    mean_spread: f64,
    sd_spread: f64,
}

impl<'a> PooledSurfaceMoments<'a> {
    fn new(accumulators: &'a [SurfaceMomentAccumulator]) -> Result<Self, String> {
        if let Some(accumulator) = accumulators
            .iter()
            .find(|accumulator| !(accumulator.weight_sum.is_finite() && accumulator.weight_sum > 0.0))
        {
            return Err(format!(
                "survival truncated posterior surfaces: a replicate lattice \
                 accumulated no finite node weight (weight sum {})",
                accumulator.weight_sum
            ));
        }
        let top = accumulators
            .iter()
            .map(|accumulator| accumulator.log_scale)
            .fold(f64::NEG_INFINITY, f64::max);
        let pooling_scales: Vec<f64> = accumulators
            .iter()
            .map(|accumulator| (accumulator.log_scale - top).exp())
            .collect();
        let pooled_weight = accumulators
            .iter()
            .zip(&pooling_scales)
            .map(|(accumulator, scale)| scale * accumulator.weight_sum)
            .sum();
        Ok(Self {
            accumulators,
            pooling_scales,
            pooled_weight,
        })
    }

    /// The pooled weighted mean of one raw sum.
    fn mean(&self, sum: impl Fn(&SurfaceMomentAccumulator) -> f64) -> f64 {
        self.accumulators
            .iter()
            .zip(&self.pooling_scales)
            .filter(|(_, scale)| **scale > 0.0)
            .map(|(accumulator, scale)| scale * sum(accumulator))
            .sum::<f64>()
            / self.pooled_weight
    }

    /// The moments of a quantity whose sums of deviations from `reference`, and
    /// of their squares, `sums` reads.
    fn moments(
        &self,
        sums: impl Fn(&SurfaceMomentAccumulator) -> (f64, f64),
        reference: f64,
    ) -> CellMoments {
        let mut means = Vec::with_capacity(self.accumulators.len());
        let mut standard_deviations = Vec::with_capacity(self.accumulators.len());
        let mut pooled_deviation = 0.0;
        let mut pooled_square = 0.0;
        for (accumulator, scale) in self.accumulators.iter().zip(&self.pooling_scales) {
            let (deviation, square) = sums(accumulator);
            let mean = deviation / accumulator.weight_sum;
            means.push(mean);
            standard_deviations
                .push((square / accumulator.weight_sum - mean * mean).max(0.0).sqrt());
            pooled_deviation += scale * deviation;
            pooled_square += scale * square;
        }
        let mean_deviation = pooled_deviation / self.pooled_weight;
        CellMoments {
            mean: reference + mean_deviation,
            variance: (pooled_square / self.pooled_weight - mean_deviation * mean_deviation)
                .max(0.0),
            mean_spread: replicate_standard_error(&means),
            sd_spread: replicate_standard_error(&standard_deviations),
        }
    }
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
mod tests {
    use super::*;
    use crate::bms::LatentMeasureKind;
    use crate::inference::model::SavedLatentZNormalization;
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

        for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
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

        for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
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
    fn posterior_quadrature_keeps_cone_coordinates_feasible_and_unbiased() {
        // Coordinate 0 is a structural monotone-I-spline baseline time
        // coefficient the fit constrained to β_0 ≥ 0; coordinate 1 is an
        // unconstrained covariate intercept. The covariance loads coordinate 0
        // with a spread wider than β̂_0, so the untruncated √rank·σ sigma point
        // steps β_0 below zero — the exact #2375 infeasible-node signature that
        // manufactures a non-monotone RP baseline the plugin evaluator refuses.
        let posterior_mean = ndarray::array![0.354, -8.30];
        let covariance = ndarray::array![[0.2304, 0.30], [0.30, 0.9604]];

        // Without the cone, the untruncated rule produces an infeasible node.
        let mut min_cone0_unconstrained = f64::INFINITY;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, _| {
            min_cone0_unconstrained = min_cone0_unconstrained.min(node[0]);
            Ok(())
        })
        .expect("unconstrained quadrature");
        assert!(
            min_cone0_unconstrained < 0.0,
            "fixture must reproduce the infeasible-node bug (min β_0 = {min_cone0_unconstrained})"
        );

        // With the cone, every node stays feasible AND the rule stays unbiased
        // (symmetric ± steps, unchanged weights → the posterior mean is exact).
        let mut mean0 = 0.0_f64;
        let mut mean1 = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        let mut min_cone0 = f64::INFINITY;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
            assert!(
                node[0] >= -1e-12,
                "cone coordinate stepped below its β_0 ≥ 0 wall: {}",
                node[0]
            );
            min_cone0 = min_cone0.min(node[0]);
            mean0 += weight * node[0];
            mean1 += weight * node[1];
            weight_sum += weight;
            Ok(())
        })
        .expect("cone-truncated quadrature");
        assert!((weight_sum - 1.0).abs() <= 1e-12, "weights must sum to one");
        assert!(
            (mean0 - posterior_mean[0]).abs() <= 1e-12
                && (mean1 - posterior_mean[1]).abs() <= 1e-12,
            "cone truncation must leave the posterior mean unbiased (got [{mean0}, {mean1}])"
        );

        // Truncation shrinks — never inflates — the represented spread along the
        // constrained coordinate (a truncated Gaussian has smaller variance).
        let mut var0_unconstrained = 0.0_f64;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
            var0_unconstrained += weight * (node[0] - posterior_mean[0]).powi(2);
            Ok(())
        })
        .expect("unconstrained spread");
        let mut var0_cone = 0.0_f64;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
            var0_cone += weight * (node[0] - posterior_mean[0]).powi(2);
            Ok(())
        })
        .expect("cone spread");
        assert!(
            var0_cone <= var0_unconstrained + 1e-12 && var0_cone < var0_unconstrained,
            "cone spread {var0_cone} must be strictly smaller than the untruncated {var0_unconstrained}"
        );
    }

    #[test]
    fn posterior_quadrature_cone_is_a_noop_far_from_the_wall() {
        // When β̂ sits comfortably inside the cone (every √rank·σ node stays
        // feasible), the fraction-to-boundary step never binds, so the cone rule
        // must reproduce the untruncated covariance to full precision — the fix
        // is inert on the healthy fits that dominate production.
        let posterior_mean = ndarray::array![40.0, -0.2];
        let covariance = ndarray::array![[0.9, 0.35], [0.35, 0.6]];
        let mut recovered_var0 = 0.0_f64;
        let mut recovered_cross = 0.0_f64;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
            recovered_var0 += weight * (node[0] - posterior_mean[0]).powi(2);
            recovered_cross +=
                weight * (node[0] - posterior_mean[0]) * (node[1] - posterior_mean[1]);
            Ok(())
        })
        .expect("cone quadrature far from the wall");
        assert!((recovered_var0 - covariance[[0, 0]]).abs() <= 1e-11);
        assert!((recovered_cross - covariance[[0, 1]]).abs() <= 1e-11);
    }

    /// A cone coefficient sitting EXACTLY on its wall (`β̂_j = 0`) is the
    /// ordinary state of an active box face, not an edge case: the fit's
    /// `coefficient_lower_bounds` pins increments there routinely. The
    /// fraction-to-boundary limit is then `0 / |f_j| = 0`, so every direction
    /// that loads that coordinate collapses to the mean and contributes no
    /// spread, while directions that do not load it keep the full `√rank` step.
    ///
    /// This is the one place the rule cannot represent the posterior it is
    /// approximating: a truncated Gaussian at an active bound is ONE-SIDED and
    /// carries real mass, but no symmetric `±` pair can express that. Reporting
    /// zero spread there is the conservative feasible answer rather than a
    /// fabricated one, and pinning it here means a future switch to an
    /// asymmetric rule has to change this test deliberately instead of silently.
    #[test]
    fn posterior_quadrature_radius_collapses_on_an_active_bound() {
        // Distinct eigenvalues so the PSD factor is axis-aligned and "the
        // direction that loads the pinned coordinate" is unambiguous.
        let posterior_mean = ndarray::array![0.0, 0.75];
        let covariance = ndarray::array![[0.5, 0.0], [0.0, 0.2]];

        let mut min_pinned = f64::INFINITY;
        let mut max_pinned = f64::NEG_INFINITY;
        let mut spread_unpinned = 0.0_f64;
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
            min_pinned = min_pinned.min(node[0]);
            max_pinned = max_pinned.max(node[0]);
            spread_unpinned += weight * (node[1] - posterior_mean[1]).powi(2);
            Ok(())
        })
        .expect("active-bound quadrature");

        assert!(
            min_pinned >= 0.0,
            "an active bound must never be crossed, got {min_pinned}"
        );
        assert!(
            max_pinned.abs() <= 1e-12,
            "a direction loading an active-bound coordinate carries zero symmetric spread, \
             but the coordinate reached {max_pinned}"
        );
        // The unconstrained coordinate is untouched: collapsing one direction
        // must not collapse the whole rule.
        assert!(
            (spread_unpinned - covariance[[1, 1]]).abs() <= 1e-11,
            "a coordinate outside the cone keeps its full spread, got {spread_unpinned} want {}",
            covariance[[1, 1]]
        );
    }

    /// Round-off guard for the `β̂_j.max(0.0)` clamp in the fraction-to-boundary
    /// limit. A converged active-set coefficient can land a few ulps BELOW its
    /// wall, and the unclamped ratio `β̂_j / |f_j|` would then be NEGATIVE — a
    /// negative step that silently inverts the `±` geometry of that direction
    /// instead of shrinking it. The clamp sends the limit to zero, so the pair
    /// collapses onto `β̂` and truncation never drives a coordinate further
    /// outside the cone than the fit already left it.
    #[test]
    fn posterior_quadrature_clamps_a_roundoff_negative_cone_coordinate() {
        let roundoff_below_wall = -1e-15_f64;
        let posterior_mean = ndarray::array![roundoff_below_wall, 0.75];
        let covariance = ndarray::array![[0.5, 0.0], [0.0, 0.2]];

        let mut nodes = Vec::new();
        for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, _| {
            nodes.push(node[0]);
            Ok(())
        })
        .expect("round-off-negative cone quadrature");

        for value in &nodes {
            assert!(
                *value >= roundoff_below_wall,
                "truncation must never push a cone coordinate further below the wall than the \
                 fit left it: node {value} < β̂ {roundoff_below_wall}"
            );
            assert!(
                (*value - roundoff_below_wall).abs() <= 1e-12,
                "a coordinate at the wall carries no spread, got {value}"
            );
        }
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
    fn saved_marginal_slope_hazard_replay_differentiates_moving_slope_2767() {
        let predictor = BernoulliMarginalSlopePredictor {
            beta_marginal: ndarray::array![1.0],
            beta_slope: ndarray::array![1.0],
            beta_score_warp: None,
            beta_link_dev: None,
            base_link: InverseLink::Standard(StandardLink::Probit),
            z_column: "z".to_string(),
            latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            latent_measure: LatentMeasureKind::StandardNormal,
            baseline_marginal: 0.0,
            baseline_slope: 0.0,
            covariance: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            gaussian_frailty_sd: None,
            latent_z_calibration: None,
            latent_z_conditional_calibration: None,
            latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
            residual_repair: None,
            beta_residual: None,
        };
        let z = 1.1;
        let q = 0.8;
        let b = 0.65;
        let q_t = 0.31;
        let b_t = 0.22;
        let input_at = |q_value: f64, b_value: f64| PredictInput {
            design: DesignMatrix::from(ndarray::array![[q_value]]),
            offset: ndarray::array![0.0],
            design_noise: Some(DesignMatrix::from(ndarray::array![[b_value]])),
            offset_noise: Some(ndarray::array![0.0]),
            auxiliary_scalar: Some(ndarray::array![z]),
            auxiliary_matrix: None,
        };

        let (eta, eta_t) = predictor
            .predict_eta_and_time_tangent(
                &input_at(q, b),
                &ndarray::array![q_t],
                &ndarray::array![b_t],
            )
            .expect("saved value+tangent replay");
        let step = 1e-6;
        let eta_plus = predictor
            .predict_eta_and_time_tangent(
                &input_at(q + step * q_t, b + step * b_t),
                &ndarray::array![0.0],
                &ndarray::array![0.0],
            )
            .expect("positive saved replay")
            .0[0];
        let eta_minus = predictor
            .predict_eta_and_time_tangent(
                &input_at(q - step * q_t, b - step * b_t),
                &ndarray::array![0.0],
                &ndarray::array![0.0],
            )
            .expect("negative saved replay")
            .0[0];
        let eta_t_fd = (eta_plus - eta_minus) / (2.0 * step);
        assert!(
            (eta_t[0] - eta_t_fd).abs() <= 2e-9 * (1.0 + eta_t_fd.abs()),
            "complete saved eta tangent {} != centered FD {eta_t_fd}",
            eta_t[0]
        );

        let eta_q = (1.0_f64 + b * b).sqrt();
        let omitted_slope_chain = eta_q * q_t;
        assert!(
            (eta_t[0] - omitted_slope_chain).abs() > 0.1,
            "fixture must detect omission of eta_b*b_t"
        );
        let (_, analytic_hazard) = probit_survival_hazard_components(eta[0], eta_t[0])
            .expect("positive analytic hazard");
        let (_, fd_hazard) = probit_survival_hazard_components(eta[0], eta_t_fd)
            .expect("positive finite-difference hazard");
        assert!(
            (analytic_hazard - fd_hazard).abs() <= 2e-9 * (1.0 + fd_hazard.abs()),
            "analytic saved hazard {analytic_hazard} != FD hazard {fd_hazard}"
        );
    }

    /// gam#3026: the hazard is the derivative of the cumulative hazard reported
    /// beside it, on both tails and through the bulk. `η(t) = a + c·log t` with
    /// `c > 0`, `H(t) = −log Φ(−η(t))`, and `dH/dt` by a Richardson-extrapolated
    /// central difference in `log t`, whose truncation error is `O(δ⁴)`.
    #[test]
    fn probit_survival_hazard_is_the_derivative_of_its_cumulative_hazard_3026() {
        let (a, c) = (-1.15, 0.95);
        let cumulative = |t: f64| {
            probit_survival_hazard_components(a + c * t.ln(), c / t)
                .expect("increasing index")
                .0
        };
        let delta = 1e-3;
        for t in [1e-4_f64, 1e-2, 0.3, 1.0, 5.0, 40.0, 1e3] {
            let (_, hazard) =
                probit_survival_hazard_components(a + c * t.ln(), c / t).expect("increasing index");
            let at = |m: f64| cumulative(t * (m * delta).exp());
            let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
            let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
            let derivative = (4.0 * d1 - d2) / 3.0 / t;
            assert!(
                (hazard - derivative).abs() <= 1e-8 * derivative.abs(),
                "at t={t}: hazard {hazard} is not dH/dt {derivative}"
            );
        }
    }

    /// gam#3026: where the index decreases the kernel returns the negative
    /// hazard it is, still `dH/dt`, instead of a flat hazard:
    /// `η(t) = a − c·log t` with `c > 0`, `dH/dt` by the same Richardson
    /// difference in `log t`. The rate `−1.35e-3` is the one the deleted
    /// clamp's test recorded (#1040), where the clamp reported `h = 0` beside
    /// an `H` whose derivative is negative.
    #[test]
    fn probit_survival_hazard_is_signed_where_the_index_decreases_3026() {
        let (a, c) = (0.4, 0.3);
        let cumulative = |t: f64| {
            probit_survival_hazard_components(a - c * t.ln(), -c / t)
                .expect("finite index")
                .0
        };
        let delta = 1e-3;
        for t in [1e-2_f64, 0.3, 1.0, 5.0, 40.0] {
            let (_, hazard) =
                probit_survival_hazard_components(a - c * t.ln(), -c / t).expect("finite index");
            let at = |m: f64| cumulative(t * (m * delta).exp());
            let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
            let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
            let derivative = (4.0 * d1 - d2) / 3.0 / t;
            assert!(
                hazard < 0.0 && (hazard - derivative).abs() <= 1e-8 * derivative.abs(),
                "at t={t}: hazard {hazard} is not dH/dt {derivative}"
            );
        }
        let (cumulative, hazard) =
            probit_survival_hazard_components(-0.563, -1.35e-3).expect("finite index");
        let (log_survival, mills_ratio) = signed_probit_logcdf_and_mills_ratio(0.563);
        assert_eq!((cumulative, hazard), (-log_survival, mills_ratio * -1.35e-3));
    }

    /// gam#3026: a plug-in surface that reports a negative hazard at any cell
    /// is refused by name, naming the cell, and one whose hazards are all
    /// non-negative (a flat stretch's `0` included) publishes.
    #[test]
    fn a_surface_whose_survival_increases_is_refused_by_name_3026() {
        let surface = |hazard: Array2<f64>| SurvivalPredictResult {
            times: vec![1.0, 2.0],
            survival: Array2::from_elem((2, 2), 0.5),
            cumulative_hazard: Array2::from_elem((2, 2), 2.0_f64.ln()),
            hazard,
            linear_predictor: Array1::zeros(2),
            likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
            survival_se: None,
            eta_se: None,
            covariance_source: None,
            survival_plugin: None,
        };
        refuse_decreasing_survival(&surface(ndarray::array![[0.3, 0.0], [0.1, 0.2]]))
            .expect("non-negative hazards publish");
        match refuse_decreasing_survival(&surface(ndarray::array![[0.3, 0.2], [0.1, -4.0e-3]])) {
            Err(SurvivalPredictError::DecreasingSurvival { reason }) => assert!(
                reason.contains("row 1, time column 1"),
                "the refusal must name the cell: {reason}"
            ),
            other => panic!("a negative hazard must be refused by name, got {other:?}"),
        }
    }

    /// gam#3026: the exact anchored posterior's density is `−d/dt E[S]`, so the
    /// published hazard `E f / E S` is `dH̄/dt`. The law makes the rate negative
    /// with visible probability: `q(t) = u + v·log t` with `(u, v)` bivariate
    /// normal and `P(v < 0) = Φ(−0.5/0.6) ≈ 0.20`. On the rigid frame with
    /// `b = 0`, `η = q`, `η_q = 1` and `η_b = 0`. Given `q(t)`, `v` has mean
    /// `m_v + β(q − m_q)`, and `E[S]` and `E[f]` are integrated by the same
    /// trapezoid over `q`, which is spectrally accurate for a Gaussian weight.
    /// The positive-part rule the node replaced is measured too, so the fixture
    /// can tell the two apart.
    #[test]
    fn exact_anchor_node_density_is_the_derivative_of_the_posterior_survival_3026() {
        let (m_u, m_v) = (-0.4, 0.5);
        let (s_u, s_v, rho) = (0.7_f64, 0.6_f64, -0.3_f64);
        // Moments of (q(t), v) at log t = x.
        let law = |x: f64| {
            let mean_q = m_u + m_v * x;
            let var_q = s_u * s_u + 2.0 * rho * s_u * s_v * x + s_v * s_v * x * x;
            let cov_vq = rho * s_u * s_v + s_v * s_v * x;
            let var_v_given_q = s_v * s_v - cov_vq * cov_vq / var_q;
            (mean_q, var_q, cov_vq / var_q, var_v_given_q)
        };
        let nodes = 4001;
        let integrate = |t: f64, signed: bool| -> (f64, f64) {
            let x = t.ln();
            let (mean_q, var_q, beta, var_v_given_q) = law(x);
            let sd_q = var_q.sqrt();
            let (mut survival, mut density) = (0.0, 0.0);
            for k in 0..nodes {
                let w = -12.0 + 24.0 * k as f64 / (nodes - 1) as f64;
                let q = mean_q + sd_q * w;
                let weight = normal_pdf(w) * 24.0 / (nodes - 1) as f64;
                // d q / d t = v / t, so the tangent's conditional mean is E[v | q] / t.
                let v_mean = m_v + beta * (q - mean_q);
                let rate = if signed {
                    v_mean / t
                } else {
                    let sd = var_v_given_q.max(0.0).sqrt();
                    (v_mean * normal_cdf(v_mean / sd) + sd * normal_pdf(v_mean / sd)) / t
                };
                let moments = exact_anchor_node_moments(q, 1.0, 0.0, [rate, 0.0]);
                survival += weight * moments.0;
                density += weight * moments.2;
            }
            (survival, density)
        };
        let delta = 1e-3;
        let mut positive_part_gap = 0.0_f64;
        for t in [0.2_f64, 1.0, 3.0, 12.0] {
            let (survival, density) = integrate(t, true);
            let at = |m: f64| -(integrate(t * (m * delta).exp(), true).0).ln();
            let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
            let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
            let derivative = (4.0 * d1 - d2) / 3.0 / t;
            let hazard = density / survival;
            assert!(
                (hazard - derivative).abs() <= 1e-7 * derivative.abs(),
                "at t={t}: posterior hazard E f / E S = {hazard} is not dH̄/dt = {derivative}"
            );
            let (_, positive_density) = integrate(t, false);
            positive_part_gap =
                positive_part_gap.max((positive_density / survival - derivative).abs() / derivative);
        }
        assert!(
            positive_part_gap > 1e-3,
            "the fixture must separate the signed density from the positive-part rule; the gap \
             was {positive_part_gap:.3e}"
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

    // ---- Held-out survival scoring, one owner for every front door (#2899) --

    #[test]
    fn survival_score_grid_spans_zero_to_the_largest_time_strictly_increasing() {
        let times = [0.3, 0.1, 0.1, 0.1, 0.45, f64::NAN, -2.0, 0.0, 0.2];
        let grid = survival_score_grid(&times);
        assert_eq!(grid[0], 0.0);
        assert_eq!(grid[grid.len() - 1], 0.45);
        assert!(grid.windows(2).all(|pair| pair[1] > pair[0]), "grid={grid:?}");
        // Half the finite positive times tie at 0.1, so the low quantiles land
        // there exactly and collapse onto one knot.
        assert_eq!(grid.iter().filter(|knot| **knot == 0.1).count(), 1);
        assert_eq!(survival_score_grid(&[f64::NAN, -1.0]), vec![0.0, 1.0]);
    }

    #[test]
    fn survival_prediction_scores_integrate_the_ipcw_brier_and_lift_it_over_the_null() {
        let time = [2.0, 8.0, 10.0, 3.0, 6.0, 1.5];
        let event = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let grid = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0];
        // Rows that are already survival paths, so the repair leaves them as they are.
        let model = Array2::from_shape_fn((6, 6), |(row, col)| {
            1.0 - col as f64 * (0.05 + 0.02 * row as f64)
        });
        let null = Array2::from_shape_fn((6, 6), |(_, col)| 1.0 - 0.1 * col as f64);
        let scores =
            survival_prediction_scores(&time, &event, &grid, model.view(), Some(null.view()));
        let censoring = KaplanMeier::fit_censoring(&time, &event);
        let model_ibs =
            integrated_ipcw_brier_score(model.view(), &time, &event, &grid, 10.0, &censoring)
                .expect("model integrated Brier");
        let null_ibs =
            integrated_ipcw_brier_score(null.view(), &time, &event, &grid, 10.0, &censoring)
                .expect("null integrated Brier");
        assert_eq!(scores.brier, Some(model_ibs));
        let lifted = scores.lifted_brier.expect("lifted Brier");
        assert!((lifted - (null_ibs - model_ibs) / null_ibs.abs()).abs() <= 1e-15);
        assert!(scores.hazard_quadratic_score.is_some() && scores.logloss.is_some());
        assert!(scores.lifted_hazard_quadratic_score.is_some());
        assert!(scores.lifted_logloss.is_some());
        let alone = survival_prediction_scores(&time, &event, &grid, model.view(), None);
        assert_eq!(alone.brier, scores.brier);
        assert_eq!(alone.logloss, scores.logloss);
        assert_eq!(alone.lifted_brier, None);
        assert_eq!(alone.nagelkerke_r2, None);
        let repeated_knot = [0.0, 1.0, 1.0, 3.0, 5.0, 7.0];
        assert_eq!(
            survival_prediction_scores(&time, &event, &repeated_knot, model.view(), None),
            SurvivalPredictionScores::default()
        );
    }

    #[test]
    fn a_scoring_grid_that_does_not_start_at_the_origin_is_refused_3609() {
        // S = 1 holds only at t = 0. On a grid starting at 1.0 the repair would
        // pin the model's S(1.0) < 1 to 1, dropping −ln S(1.0) from every H(T_i),
        // and the event at T = 0.5 would sit before the first interval and read a
        // negative cumulative hazard.
        let time = [0.5, 2.0, 8.0, 3.0];
        let event = [1.0, 1.0, 0.0, 1.0];
        let origin = [0.0, 1.0, 2.0, 3.0, 5.0];
        let model = Array2::from_shape_fn((4, 5), |(row, col)| {
            1.0 - col as f64 * (0.05 + 0.02 * row as f64)
        });
        let scored = survival_prediction_scores(&time, &event, &origin, model.view(), None);
        assert!(scored.brier.is_some() && scored.logloss.is_some());
        for late in [[1.0, 2.0, 3.0, 5.0, 7.0], [-1.0, 1.0, 2.0, 3.0, 5.0]] {
            assert_eq!(
                survival_prediction_scores(&time, &event, &late, model.view(), None),
                SurvivalPredictionScores::default(),
                "grid {late:?}"
            );
        }
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
        // `on_grid` is `at` mapped over the grid, NaN included (it precedes no step).
        let probe = [f64::NAN, -1.0, 0.0, 3.999, 4.0, 7.5, 8.0, 9.0, f64::INFINITY];
        let mapped: Vec<f64> = probe.iter().map(|&t| g.at(t)).collect();
        assert_eq!(g.on_grid(&probe), mapped);
        assert_eq!(g.at(f64::NAN), 1.0);
        assert_eq!(g.at(f64::INFINITY), 0.0);
    }

    /// The pair-loop definition of Harrell's C, written independently of the
    /// Fenwick sweep: a pair is comparable when one subject had an event and the
    /// other was observed strictly later, or at the same time but censored.
    fn harrell_concordance_by_pairs(time: &[f64], event: &[f64], risk: &[f64]) -> Option<f64> {
        let mut comparable = 0.0_f64;
        let mut concordant = 0.0_f64;
        for i in 0..time.len() {
            for j in 0..time.len() {
                let i_fails_first = event[i] > 0.5
                    && (time[j] > time[i] || (time[j] == time[i] && event[j] <= 0.5));
                if !i_fails_first {
                    continue;
                }
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if risk[i] == risk[j] {
                    concordant += 0.5;
                }
            }
        }
        if comparable > 0.0 {
            Some(concordant / comparable)
        } else {
            None
        }
    }

    #[test]
    fn harrell_concordance_tie_rules_match_hand_count() {
        // Tied block at t=2: two events (rows 1, 2) and one censoring (row 3).
        // Comparable pairs (early, late): row 0 with all four later rows
        // (risk 5 beats 4, 1, 3, 2: 4 concordant); rows 1 and 2 with the tied
        // censoring row 3 (4>3 concordant, 1<3 discordant) and with row 4
        // (4>2 concordant, 1<2 discordant). The tied event pair (1, 2) is not
        // comparable, and row 3 (censored) orders nothing after it.
        // C = (4 + 1 + 1) / (4 + 2 + 2) = 0.75.
        let time = [1.0, 2.0, 2.0, 2.0, 3.0];
        let event = [1.0, 1.0, 1.0, 0.0, 0.0];
        let risk = [5.0, 4.0, 1.0, 3.0, 2.0];
        assert_eq!(harrell_concordance(&time, &event, &risk), Some(0.75));

        // Two events at the same time are not comparable: nothing is orderable.
        assert_eq!(harrell_concordance(&[2.0, 2.0], &[1.0, 1.0], &[1.0, 0.0]), None);
        // An event tied with a censoring is comparable; the censored subject
        // outlived the death, so the larger risk on the event is concordant.
        assert_eq!(harrell_concordance(&[2.0, 2.0], &[0.0, 1.0], &[1.0, 3.0]), Some(1.0));
        // Equal risks score half credit.
        assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0, 1.0], &[7.0, 7.0]), Some(0.5));
        // All censored: no comparable pair.
        assert_eq!(harrell_concordance(&[1.0, 2.0], &[0.0, 0.0], &[1.0, 2.0]), None);
        // Non-finite inputs have no defined ordering.
        assert_eq!(harrell_concordance(&[1.0, f64::NAN], &[1.0, 1.0], &[1.0, 2.0]), None);
        assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0, 1.0], &[f64::NAN, 2.0]), None);
        // Length mismatch.
        assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn harrell_concordance_sweep_equals_pair_definition_under_heavy_ties() {
        // Coarse integer times and risks force many tied time blocks and tied
        // risks; the O(n log n) sweep must reproduce the pair loop exactly.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |modulus: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % modulus
        };
        for trial in 0..200 {
            let n = (next(40) + 1) as usize;
            let time: Vec<f64> = (0..n).map(|_| (next(6) + 1) as f64).collect();
            let event: Vec<f64> = (0..n)
                .map(|_| if next(5) < 3 { 1.0 } else { 0.0 })
                .collect();
            let risk: Vec<f64> = (0..n).map(|_| next(5) as f64 - 2.0).collect();
            let sweep = harrell_concordance(&time, &event, &risk);
            let pairs = harrell_concordance_by_pairs(&time, &event, &risk);
            // Both are one correctly rounded division of the same exact rational
            // (the numerators are integer and half-integer counts), so they are
            // bitwise equal, not merely close.
            assert_eq!(sweep, pairs, "trial {trial}");
        }
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
        let bs = ipcw_brier_score(&s_pred, &time, &event, tau, &g).unwrap();
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
        let bs = ipcw_brier_score(&s_pred, &time, &event, tau, &g).unwrap();
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
    fn ipcw_brier_weights_an_event_tied_with_a_censoring_by_the_left_limit() {
        // Discretised follow-up: a death and a censoring tie at t = 2, and again
        // at t = 5. The censoring KM steps at each tie (G(2) = 4/5, G(5) = 8/15),
        // but a death at T was observed because C ≥ T, whose probability is the
        // left limit G(T−) (G(2−) = 1, G(5−) = 4/5). Weighting by the
        // right-continuous G(T) would count each tied death 1/(1 − c/n) times.
        let s_pred = [0.4, 0.6, 0.7, 0.8, 0.9];
        let time = [2.0, 2.0, 5.0, 5.0, 7.0];
        let event = [1.0, 0.0, 1.0, 0.0, 1.0];
        let g = KaplanMeier::fit_censoring(&time, &event);
        assert_eq!(g.before(2.0), 1.0);
        assert!((g.at(2.0) - 0.8).abs() <= 1e-15);
        assert!((g.before(5.0) - 0.8).abs() <= 1e-15);
        assert!((g.at(5.0) - 8.0 / 15.0).abs() <= 1e-15);
        // tau = 3: subj1 dead, weight 1/G(2−) = 1, contrib 0.4²; subj2 censored
        // at 2 contributes 0; subj3..5 alive, weight 1/G(3) = 5/4.
        let bs = ipcw_brier_score(&s_pred, &time, &event, 3.0, &g).unwrap();
        let expected = (0.16 + 1.25 * (0.09 + 0.04 + 0.01)) / 5.0;
        assert!((bs - expected).abs() <= 1e-12, "bs={bs} expected={expected}");
        // tau = 5: subj3 dead at the tie, weight 1/G(5−) = 5/4; subj5 alive,
        // weight 1/G(5) = 15/8; the two censored subjects contribute 0.
        let bs = ipcw_brier_score(&s_pred, &time, &event, 5.0, &g).unwrap();
        let expected = (0.16 + 1.25 * 0.49 + 1.875 * 0.01) / 5.0;
        assert!((bs - expected).abs() <= 1e-12, "bs={bs} expected={expected}");
    }

    #[test]
    fn ipcw_brier_drops_invalid_rows_from_both_numerator_and_denominator() {
        // A NaN-time row and a non-positive-time row must not be counted at all.
        let s_pred = [0.3, 0.7, 0.5, 0.5];
        let time = [2.0, 8.0, f64::NAN, -1.0];
        let event = [1.0, 1.0, 1.0, 0.0];
        let g = KaplanMeier::fit_censoring(&time, &event);
        let bs = ipcw_brier_score(&s_pred, &time, &event, 5.0, &g).unwrap();
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
        let per_time = ipcw_brier_score(&col, &time, &event, grid[2], &g).unwrap();
        // Because the predicted survival is identical at every grid time, BS(t)
        // is *not* constant (tau changes which subjects are "alive"), so use a
        // direct trapezoid as the oracle.
        let mut oracle_pts = Vec::new();
        for k in 0..grid.len() {
            oracle_pts.push((
                grid[k],
                ipcw_brier_score(&col, &time, &event, grid[k], &g).unwrap(),
            ));
        }
        let mut integral = 0.0;
        for w in oracle_pts.windows(2) {
            integral += 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0);
        }
        let oracle = integral / (grid[grid.len() - 1] - grid[0]);
        let ibs =
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, &g)
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
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, 5.0, &g).unwrap();
        let full =
            integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, &g)
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
            integrated_ipcw_brier_score(surv.view(), &time, &event, &bad, f64::INFINITY, &g)
                .is_none()
        );
        // Grid width mismatched to the survival matrix.
        let short = [0.0, 1.0];
        assert!(
            integrated_ipcw_brier_score(surv.view(), &time, &event, &short, f64::INFINITY, &g)
                .is_none()
        );
    }

    /// `S(t) = exp(-rate*t)` sampled on `times`, one identical row per subject.
    fn exponential_survival_result(times: Vec<f64>, rate: f64, rows: usize) -> SurvivalPredictResult {
        let t = times.len();
        let mut survival = Array2::<f64>::zeros((rows, t));
        let mut hazard = Array2::<f64>::zeros((rows, t));
        let mut cumulative_hazard = Array2::<f64>::zeros((rows, t));
        for i in 0..rows {
            for (j, &time) in times.iter().enumerate() {
                survival[[i, j]] = (-rate * time).exp();
                hazard[[i, j]] = rate;
                cumulative_hazard[[i, j]] = rate * time;
            }
        }
        SurvivalPredictResult {
            times,
            hazard,
            survival,
            cumulative_hazard,
            linear_predictor: Array1::zeros(rows),
            likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
            survival_se: None,
            eta_se: None,
            covariance_source: None,
            survival_plugin: None,
        }
    }

    #[test]
    fn rmst_over_prediction_horizon_matches_the_exponential_closed_form() {
        // For S(t) = exp(-rate*t) the restricted mean has a closed form,
        // RMST(tau) = (1 - exp(-rate*tau)) / rate, so the trapezoid sum is
        // checked against an exact value rather than against itself.
        let rate = 0.35_f64;
        let tau = 4.0_f64;
        let steps = 4000_usize;
        let times: Vec<f64> = (1..=steps)
            .map(|k| tau * (k as f64) / (steps as f64))
            .collect();
        let result = exponential_survival_result(times, rate, 3);

        let rmst = result
            .rmst_over_prediction_horizon()
            .expect("a positive finite grid yields an RMST column");
        let expected = (1.0 - (-rate * tau).exp()) / rate;

        assert!((rmst.tau - tau).abs() < 1e-12, "tau = {}", rmst.tau);
        assert_eq!(rmst.values.len(), 3);
        for value in rmst.values.iter() {
            // Trapezoid error on a convex curve is O(h^2); h = tau/steps here,
            // so 1e-6 is several orders above the discretization floor and
            // still far below any difference that would matter clinically.
            assert!(
                (value - expected).abs() < 1e-6,
                "rmst {value} vs closed form {expected}"
            );
        }
    }

    #[test]
    fn rmst_over_prediction_horizon_integrates_to_the_last_grid_time() {
        let times = vec![0.5, 1.0, 2.5, 6.0];
        let result = exponential_survival_result(times.clone(), 0.2, 2);

        let horizon = result
            .rmst_over_prediction_horizon()
            .expect("non-empty grid");
        let explicit = result
            .restricted_mean_survival_time(6.0)
            .expect("explicit tau at the same horizon");

        assert_eq!(horizon.tau, 6.0, "tau is the last grid point");
        assert_eq!(horizon.values, explicit, "no second integration policy");
    }

    #[test]
    fn rmst_over_prediction_horizon_declines_an_empty_or_degenerate_grid() {
        let empty = exponential_survival_result(Vec::new(), 0.2, 2);
        assert!(empty.rmst_over_prediction_horizon().is_none(), "empty grid");

        // A grid whose only point is the time origin encloses no area.
        let origin_only = exponential_survival_result(vec![0.0], 0.2, 2);
        assert!(
            origin_only.rmst_over_prediction_horizon().is_none(),
            "tau = 0 encloses no area"
        );
    }

    #[test]
    fn rmst_over_prediction_horizon_declines_a_non_finite_curve() {
        let mut result = exponential_survival_result(vec![1.0, 2.0], 0.2, 2);
        result.survival[[1, 0]] = f64::NAN;
        assert!(
            result.rmst_over_prediction_horizon().is_none(),
            "a NaN anywhere on the integrated span refuses the whole column"
        );
    }

    #[test]
    fn overall_rmst_over_prediction_horizon_reads_the_all_cause_surface() {
        // Two endpoints, each with constant hazard `rate`, so the all-cause
        // survival is exp(-2*rate*t) and the closed form applies to it.
        let rate = 0.25_f64;
        let tau = 3.0_f64;
        let steps = 3000_usize;
        let times: Vec<f64> = (1..=steps)
            .map(|k| tau * (k as f64) / (steps as f64))
            .collect();
        let rows = 2_usize;
        let t = times.len();
        let mut overall_survival = Array2::<f64>::zeros((rows, t));
        for i in 0..rows {
            for (j, &time) in times.iter().enumerate() {
                overall_survival[[i, j]] = (-2.0 * rate * time).exp();
            }
        }
        let per_cause = Array2::<f64>::zeros((rows, t));
        let result = CompetingRisksPredictResult {
            times,
            endpoint_names: vec!["a".to_string(), "b".to_string()],
            hazard: vec![per_cause.clone(), per_cause.clone()],
            survival: vec![per_cause.clone(), per_cause.clone()],
            cumulative_hazard: vec![per_cause.clone(), per_cause.clone()],
            cif: vec![per_cause.clone(), per_cause],
            overall_survival,
            linear_predictor: vec![Array1::zeros(rows), Array1::zeros(rows)],
            likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
            covariance_source: None,
            hazard_se: None,
            survival_se: None,
            cumulative_hazard_se: None,
            cif_se: None,
            overall_survival_se: None,
            eta_se: None,
        };

        let rmst = result
            .overall_rmst_over_prediction_horizon()
            .expect("all-cause RMST over a positive grid");
        let expected = (1.0 - (-2.0 * rate * tau).exp()) / (2.0 * rate);

        assert!((rmst.tau - tau).abs() < 1e-12);
        for value in rmst.values.iter() {
            assert!(
                (value - expected).abs() < 1e-6,
                "all-cause rmst {value} vs closed form {expected}"
            );
        }
    }

    /// Deterministic uniforms and normals for the gam#3038 fixture.
    struct Lcg3038(u64);

    impl Lcg3038 {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let radius = (-2.0 * self.unit().ln()).sqrt();
            radius * (2.0 * std::f64::consts::PI * self.unit()).cos()
        }
    }

    /// gam#3038: a location-scale survival fit whose time channel sits against
    /// its cone publishes its posterior-mean surfaces by integrating the
    /// cone-truncated posterior `π = N(β_unc, Σ) | C` it reports, not the
    /// moment-matched normal `N(E_π[β], Σ_π)`, whose sigma points leave `C`
    /// where the model has no hazard.
    ///
    /// The reference is independent of the rule: rejection draws from the
    /// ambient `N(β_unc, Σ)` kept only inside `C`, every draw replayed through
    /// the per-coefficient survival law. The published survival must match the
    /// Monte Carlo `E_π[S]` and the published density `h·S` its `E_π[f]`, each
    /// within four Monte Carlo standard errors plus the rule's certified
    /// relative accuracy.
    #[test]
    fn location_scale_posterior_mean_integrates_the_cone_truncated_law_3038() {
        use crate::fit_orchestration::FitConfig;
        use crate::inference::model::FittedModel;
        use crate::inference::model_payload_builders::fit_formula_to_payload;
        use crate::survival::location_scale::factorize_psd_covariance;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        // `log T = 0.4x + e^{−0.3 + 0.4x}·ε` with independent censoring: a
        // linear log-scale, as in the report.
        let n = 400;
        let mut rng = Lcg3038(0x3038);
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.normal();
            let log_t = 0.4 * x + (-0.3 + 0.4 * x).exp() * rng.normal();
            let censor = (-0.5 + 2.5 * rng.unit()).exp();
            let event_time = log_t.exp();
            let (time, event) = if event_time <= censor {
                (event_time, 1)
            } else {
                (censor, 0)
            };
            records.push(csv::StringRecord::from(vec![
                format!("{time:.17e}"),
                event.to_string(),
                format!("{x:.17e}"),
            ]));
        }
        let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
        let data = gam_data::encode_recordswith_inferred_schema(headers, records)
            .expect("encode the #3038 fixture");
        let config = FitConfig {
            survival_likelihood: Some("location-scale".to_string()),
            survival_distribution: "gaussian".to_string(),
            noise_formula: Some("x".to_string()),
            ..FitConfig::default()
        };
        let payload =
            fit_formula_to_payload("Surv(time, event) ~ x".to_string(), &data, &config)
                .expect("survival location-scale fit");
        let model = FittedModel::from_payload(payload);
        let mode = SurvivalPredictionCovarianceMode::Conditional;

        assert_eq!(
            SurvivalPosteriorIntegration::default_for(&model, mode).expect("default integration"),
            SurvivalPosteriorIntegration::TruncatedLaw,
            "the fixture's posterior must carry an active cone"
        );

        let frame = ndarray::array![[1.0, 0.0, -1.5], [1.0, 0.0, 0.0], [1.0, 0.0, 1.5]];
        let col_map = data.column_map();
        let zeros = Array1::<f64>::zeros(frame.nrows());
        let times = [0.25, 0.5, 1.0, 2.0, 4.0];
        let request = |estimand| SurvivalPredictRequest {
            model: &model,
            data: frame.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: Some(&times),
            with_uncertainty: false,
            estimand,
        };
        let published = predict_survival(request(SurvivalPredictEstimand::PosteriorMean), mode)
            .expect("the posterior-mean surfaces integrate the truncated law");
        let sigma_point = predict_survival_posterior_mean_with(
            request(SurvivalPredictEstimand::PosteriorMean),
            mode,
            SurvivalPosteriorIntegration::SigmaPoint,
        );
        eprintln!(
            "[3038] moment-matched normal sigma points: {}",
            match &sigma_point {
                Ok(_) => "published".to_string(),
                Err(error) => format!("refused: {error}"),
            }
        );

        // The ambient law the cone truncates, in raw coefficients.
        let fit = fit_result_from_saved_model_for_prediction(&model).expect("saved fit");
        let geometry = fit.geometry.as_ref().expect("fit geometry");
        let constrained = geometry
            .constrained_posterior
            .as_ref()
            .expect("constrained posterior");
        let correction = constrained
            .correction()
            .expect("available moments")
            .expect("an active cone");
        let gauge = &geometry.coefficient_gauge;
        let unconstrained = constrained.unconstrained_center().expect("ambient centre");
        let lift = gauge.t_full.dot(&correction.lift);
        let ambient = fit.beta_covariance().expect("conditional covariance")
            + &lift
                .dot(&correction.removed_normal_variance)
                .dot(&lift.t());
        let factor = factorize_psd_covariance(&ambient, "#3038 ambient covariance")
            .expect("ambient factor")
            .factor;
        let center = gauge.t_full.dot(unconstrained) + &gauge.affine_shift;
        // Raw to active coordinates: every raw displacement `L·z` lies in the
        // range of the full-column-rank gauge `T`, so `T d_a = d_raw` is solved
        // exactly through its normal equations.
        let normal = gauge.t_full.t().dot(&gauge.t_full);
        let normal_inverse = {
            let eig = factorize_psd_covariance(&normal, "#3038 gauge normal equations")
                .expect("gauge factor");
            let scaled = &eig.eigenvectors * &eig.inv_sqrt_eigenvalues;
            scaled.dot(&scaled.t())
        };

        let chunks = 16usize;
        let per_chunk = 250usize;
        let (n_rows, n_times) = published.survival.dim();
        let sums = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut rng = Lcg3038(0x5eed_3038 ^ ((chunk as u64 + 1) << 32));
                let mut s1 = Array2::<f64>::zeros((n_rows, n_times));
                let mut s2 = Array2::<f64>::zeros((n_rows, n_times));
                let mut f1 = Array2::<f64>::zeros((n_rows, n_times));
                let mut f2 = Array2::<f64>::zeros((n_rows, n_times));
                let mut proposals = 0usize;
                let mut accepted = 0usize;
                while accepted < per_chunk {
                    proposals += 1;
                    let z = Array1::from_shape_fn(factor.ncols(), |_| rng.normal());
                    let displacement = factor.dot(&z);
                    let active = unconstrained
                        + &normal_inverse.dot(&gauge.t_full.t().dot(&displacement));
                    let slack = constrained.constraints.a.dot(&active) - &constrained.constraints.b;
                    if slack.iter().any(|&value| value < 0.0) {
                        continue;
                    }
                    accepted += 1;
                    let draw_model =
                        saved_model_with_survival_coefficients(&model, &(&center + &displacement))
                            .expect("draw model");
                    let draw = predict_survival_coefficient_law(
                        SurvivalPredictRequest {
                            model: &draw_model,
                            estimand: SurvivalPredictEstimand::Plugin,
                            ..request(SurvivalPredictEstimand::Plugin)
                        },
                        mode,
                    )
                    .expect("per-coefficient survival law at a feasible draw");
                    let density = &draw.survival * &draw.hazard;
                    s1 += &draw.survival;
                    s2 += &draw.survival.mapv(|s| s * s);
                    f1 += &density;
                    f2 += &density.mapv(|f| f * f);
                }
                (s1, s2, f1, f2, proposals)
            })
            .reduce(
                || {
                    (
                        Array2::<f64>::zeros((n_rows, n_times)),
                        Array2::<f64>::zeros((n_rows, n_times)),
                        Array2::<f64>::zeros((n_rows, n_times)),
                        Array2::<f64>::zeros((n_rows, n_times)),
                        0usize,
                    )
                },
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
            );
        let draws = (chunks * per_chunk) as f64;
        eprintln!(
            "[3038] Monte Carlo: {} feasible draws of {} ambient proposals",
            chunks * per_chunk,
            sums.4
        );
        let mut worst_survival_gap = 0.0_f64;
        let mut largest_truncation_effect = 0.0_f64;
        let plugin = published
            .survival_plugin
            .as_ref()
            .expect("posterior-mean prediction carries the plug-in survival");
        for row in 0..n_rows {
            for time in 0..n_times {
                let cell = [row, time];
                let mc_survival = sums.0[cell] / draws;
                let mc_survival_se =
                    ((sums.1[cell] / draws - mc_survival * mc_survival).max(0.0) / draws).sqrt();
                let mc_density = sums.2[cell] / draws;
                let mc_density_se =
                    ((sums.3[cell] / draws - mc_density * mc_density).max(0.0) / draws).sqrt();
                let survival = published.survival[cell];
                let density = published.hazard[cell] * survival;
                let rule_accuracy = 2.0e-3 * (mc_survival * (1.0 - mc_survival)).sqrt();
                eprintln!(
                    "[3038] row {row} t={:.2}: E[S] rule {survival:.6} MC {mc_survival:.6} \
                     (se {mc_survival_se:.1e}, plug-in {:.6}); E[f] rule {density:.6} MC \
                     {mc_density:.6} (se {mc_density_se:.1e})",
                    times[time], plugin[cell]
                );
                assert!(
                    (survival - mc_survival).abs() <= 4.0 * mc_survival_se + rule_accuracy,
                    "row {row}, t={}: published E[S] {survival:.6} vs Monte Carlo \
                     {mc_survival:.6} (se {mc_survival_se:.2e})",
                    times[time]
                );
                assert!(
                    (density - mc_density).abs()
                        <= 4.0 * mc_density_se + 2.0e-3 * mc_density.abs(),
                    "row {row}, t={}: published E[f] {density:.6} vs Monte Carlo \
                     {mc_density:.6} (se {mc_density_se:.2e})",
                    times[time]
                );
                worst_survival_gap = worst_survival_gap.max((survival - mc_survival).abs());
                largest_truncation_effect =
                    largest_truncation_effect.max((plugin[cell] - mc_survival).abs());
            }
        }
        eprintln!(
            "[3038] max |E[S] rule − MC| {worst_survival_gap:.2e}; max |plug-in − MC| \
             {largest_truncation_effect:.2e}"
        );
    }

    /// gam#3575: a Royston-Parmar fit whose monotone I-spline baseline has a
    /// flat segment pins baseline coefficients on their bound `β_j = 0`. Its
    /// posterior is `N(β̂ − H⁻¹g, H⁻¹)` restricted to the cone, and the
    /// posterior-mean surfaces must integrate THAT law: a strictly positive
    /// survival SE and moments matching a rejection Monte Carlo of the same
    /// law. The fraction-to-boundary sigma-point rule this replaces collapsed
    /// every node onto the mode, publishing the plug-in with SE ≈ 0.
    ///
    /// The reference is independent of the rule: rejection draws from the
    /// ambient `N(β_unc, Σ)` kept only inside the cone, every draw replayed
    /// through the per-coefficient survival law. The published `E[S]` must
    /// match the Monte Carlo mean within four Monte Carlo standard errors plus
    /// the rule's certified relative accuracy, and the published variance
    /// `SE²` the Monte Carlo variance within four standard errors of the
    /// sample variance (from the fourth central moment) plus the accuracy the
    /// rule certifies for `E[S²] − E[S]²`.
    #[test]
    fn royston_parmar_posterior_mean_integrates_the_cone_truncated_law_3575() {
        use crate::fit_orchestration::FitConfig;
        use crate::inference::model::FittedModel;
        use crate::inference::model_payload_builders::fit_formula_to_payload;
        use crate::survival::location_scale::factorize_psd_covariance;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        // Events in two clusters, early `t ∈ (0.1, 0.3)` and late
        // `t ∈ (8, 12)`, with subjects censored across the gap between them:
        // the gap carries exposure but no events, so the maximum-likelihood
        // baseline hazard is zero there and the monotone baseline presses
        // against its cone.
        let n = 300;
        let mut rng = Lcg3038(0x3575);
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.normal();
            let event_time = if rng.unit() < 0.5 {
                0.1 + 0.2 * rng.unit()
            } else {
                8.0 + 4.0 * rng.unit()
            };
            let censor = if rng.unit() < 0.3 {
                0.3 + 7.7 * rng.unit()
            } else {
                15.0
            };
            let (time, event) = if event_time <= censor {
                (event_time, 1)
            } else {
                (censor, 0)
            };
            records.push(csv::StringRecord::from(vec![
                format!("{time:.17e}"),
                event.to_string(),
                format!("{x:.17e}"),
            ]));
        }
        let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
        let data = gam_data::encode_recordswith_inferred_schema(headers, records)
            .expect("encode the #3575 fixture");
        let config = FitConfig {
            survival_likelihood: Some("transformation".to_string()),
            ..FitConfig::default()
        };
        let payload =
            fit_formula_to_payload("Surv(time, event) ~ x".to_string(), &data, &config)
                .expect("Royston-Parmar survival fit");
        let model = FittedModel::from_payload(payload);
        let mode = SurvivalPredictionCovarianceMode::Conditional;

        let fit = fit_result_from_saved_model_for_prediction(&model).expect("saved fit");
        let geometry = fit.geometry.as_ref().expect("fit geometry");
        let constrained = geometry
            .constrained_posterior
            .as_ref()
            .expect("the Royston-Parmar fit publishes its cone-truncated posterior");
        eprintln!(
            "[3575] mode {:?}; published posterior mean {:?}",
            constrained.mode,
            fit.beta.to_vec()
        );
        assert_eq!(
            SurvivalPosteriorIntegration::default_for(&model, mode).expect("default integration"),
            SurvivalPosteriorIntegration::TruncatedLaw,
            "the fixture's posterior must carry an active cone"
        );

        let frame = ndarray::array![[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]];
        let col_map = data.column_map();
        let zeros = Array1::<f64>::zeros(frame.nrows());
        let times = [0.15, 0.25, 1.0, 4.0, 9.0, 11.0];
        let request = |estimand, with_uncertainty| SurvivalPredictRequest {
            model: &model,
            data: frame.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: Some(&times),
            with_uncertainty,
            estimand,
        };
        let published = predict_survival(request(SurvivalPredictEstimand::PosteriorMean, true), mode)
            .expect("the posterior-mean surfaces integrate the truncated law");
        let published_se = published
            .survival_se
            .as_ref()
            .expect("an uncertainty request publishes the survival SE");
        let sigma_point = predict_survival_posterior_mean_with(
            request(SurvivalPredictEstimand::PosteriorMean, true),
            mode,
            SurvivalPosteriorIntegration::SigmaPoint,
        );
        match &sigma_point {
            Ok(result) => eprintln!(
                "[3575] fraction-to-boundary sigma points: survival SE {:?}",
                result.survival_se.as_ref().map(|se| se.to_vec())
            ),
            Err(error) => eprintln!("[3575] fraction-to-boundary sigma points refused: {error}"),
        }

        // The ambient law the cone truncates. The Royston-Parmar gauge is the
        // identity, so the active coordinates are the raw coefficients.
        let correction = constrained
            .correction()
            .expect("available moments")
            .expect("an active cone");
        let gauge = &geometry.coefficient_gauge;
        let unconstrained = constrained.unconstrained_center().expect("ambient centre");
        let lift = gauge.t_full.dot(&correction.lift);
        let ambient = fit.beta_covariance().expect("conditional covariance")
            + &lift
                .dot(&correction.removed_normal_variance)
                .dot(&lift.t());
        let factor = factorize_psd_covariance(&ambient, "#3575 ambient covariance")
            .expect("ambient factor")
            .factor;
        let center = gauge.t_full.dot(unconstrained) + &gauge.affine_shift;
        let normal = gauge.t_full.t().dot(&gauge.t_full);
        let normal_inverse = {
            let eig = factorize_psd_covariance(&normal, "#3575 gauge normal equations")
                .expect("gauge factor");
            let scaled = &eig.eigenvectors * &eig.inv_sqrt_eigenvalues;
            scaled.dot(&scaled.t())
        };

        let chunks = 16usize;
        let per_chunk = 250usize;
        let (n_rows, n_times) = published.survival.dim();
        let (samples, proposals) = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut rng = Lcg3038(0x5eed_3575 ^ ((chunk as u64 + 1) << 32));
                let mut samples = Vec::with_capacity(per_chunk);
                let mut proposals = 0usize;
                while samples.len() < per_chunk {
                    proposals += 1;
                    let z = Array1::from_shape_fn(factor.ncols(), |_| rng.normal());
                    let displacement = factor.dot(&z);
                    let active = unconstrained
                        + &normal_inverse.dot(&gauge.t_full.t().dot(&displacement));
                    let slack = constrained.constraints.a.dot(&active) - &constrained.constraints.b;
                    if slack.iter().any(|&value| value < 0.0) {
                        continue;
                    }
                    let draw_model =
                        saved_model_with_survival_coefficients(&model, &(&center + &displacement))
                            .expect("draw model");
                    let draw = predict_survival_coefficient_law(
                        SurvivalPredictRequest {
                            model: &draw_model,
                            ..request(SurvivalPredictEstimand::Plugin, false)
                        },
                        mode,
                    )
                    .expect("per-coefficient survival law at a feasible draw");
                    samples.push(draw.survival);
                }
                (samples, proposals)
            })
            .reduce(
                || (Vec::new(), 0usize),
                |mut a, b| {
                    a.0.extend(b.0);
                    (a.0, a.1 + b.1)
                },
            );
        let draws = samples.len() as f64;
        eprintln!(
            "[3575] Monte Carlo: {} feasible draws of {proposals} ambient proposals",
            samples.len()
        );
        let plugin = published
            .survival_plugin
            .as_ref()
            .expect("posterior-mean prediction carries the plug-in survival");
        let mut largest_jensen_gap = 0.0_f64;
        for row in 0..n_rows {
            for time in 0..n_times {
                let cell = [row, time];
                let mc_mean = samples.iter().map(|s| s[cell]).sum::<f64>() / draws;
                let central = |power: i32| {
                    samples
                        .iter()
                        .map(|s| (s[cell] - mc_mean).powi(power))
                        .sum::<f64>()
                        / draws
                };
                let mc_variance = central(2);
                let mc_mean_se = (mc_variance / draws).sqrt();
                let mc_variance_se = ((central(4) - mc_variance * mc_variance).max(0.0) / draws).sqrt();
                let survival = published.survival[cell];
                let se = published_se[cell];
                let rule_accuracy = 2.0e-3 * (mc_mean * (1.0 - mc_mean)).sqrt();
                eprintln!(
                    "[3575] row {row} t={:.2}: E[S] rule {survival:.6} MC {mc_mean:.6} \
                     (se {mc_mean_se:.1e}, plug-in {:.6}); SE rule {se:.3e} MC {:.3e}",
                    times[time],
                    plugin[cell],
                    mc_variance.sqrt()
                );
                assert!(
                    se > 0.0,
                    "row {row}, t={}: the truncated posterior carries mass off the bound, yet the \
                     published survival SE is {se:e}",
                    times[time]
                );
                assert!(
                    (survival - mc_mean).abs() <= 4.0 * mc_mean_se + rule_accuracy,
                    "row {row}, t={}: published E[S] {survival:.6} vs Monte Carlo {mc_mean:.6} \
                     (se {mc_mean_se:.2e})",
                    times[time]
                );
                // `Var = E[S²] − E[S]²`: an error δ in each certified moment
                // moves it by at most `δ + 2·E[S]·δ ≤ 3δ`.
                assert!(
                    (se * se - mc_variance).abs() <= 4.0 * mc_variance_se + 3.0 * rule_accuracy,
                    "row {row}, t={}: published Var[S] {:.3e} vs Monte Carlo {mc_variance:.3e} \
                     (se {mc_variance_se:.2e})",
                    times[time],
                    se * se
                );
                largest_jensen_gap = largest_jensen_gap.max((survival - plugin[cell]).abs());
            }
        }
        eprintln!("[3575] max |E[S] − S(E[β])| {largest_jensen_gap:.2e}");
        assert!(
            largest_jensen_gap > 0.0,
            "the posterior-mean survival must integrate the law, not re-publish the plug-in"
        );
    }
}

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
    let span = hi - lo;
    let hi_padded = hi + (span * 1.0e-6).max(1.0e-9);
    let step = (hi_padded - lo) / 63.0;
    Ok(Some(
        (0..64).map(|index| lo + step * (index as f64)).collect(),
    ))
}
