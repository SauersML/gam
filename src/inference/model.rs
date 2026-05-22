use crate::basis::BasisOptions;
use crate::estimate::{BlockRole, FittedLinkState, UnifiedFitResult};
use crate::families::bernoulli_marginal_slope::{LatentMeasureKind, LatentZRankIntCalibration};
use crate::families::gamlss::{
    monotone_wiggle_basis_with_derivative_order, validate_monotone_wiggle_beta_nonnegative,
};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalTimeBasisConfig, parse_survival_baseline_config,
};
use crate::families::survival_location_scale::ResidualDistribution;
use crate::inference::formula_dsl::{
    inverse_link_supports_joint_wiggle, joint_wiggle_unsupported_link_message,
};
use crate::inference::predict::{
    BernoulliMarginalSlopePredictor, BinomialLocationScalePredictor,
    GaussianLocationScalePredictor, PredictableModel, StandardPredictor, SurvivalPredictor,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use crate::smooth::{AdaptiveRegularizationDiagnostics, TermCollectionSpec};
use crate::span::span_index_for_breakpoints;
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodFamily, LinkFunction, MixtureLinkState, SasLinkSpec,
    SasLinkState,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// Canonical saved-model payload schema version.
///
/// Every `FittedModelPayload` written by any binary (CLI `gam`, gam-pyffi,
/// downstream library users) must set this as its `version` field, and every
/// load path asserts equality via `validate_for_persistence`. Bump this when:
///   - A required field is added to `FittedModelPayload` and the set of
///     Option<T> fields that must be `Some(...)` for a given `family_state`
///     changes (otherwise the `#[serde(default)]` decode would silently fill
///     the new field with `None` when loading an older model and the CLI
///     predict path would run with stale metadata).
///   - The on-wire shape of any `serde`-tagged enum variant changes such that
///     older payloads no longer round-trip losslessly.
///   - The semantics of an existing field change (e.g. sign convention,
///     coordinate frame) in a way that predict output would silently diverge
///     between old and new readers.
///
/// Do NOT bump for purely additive `Option<T>` fields that the save-time
/// invariant (`validate_for_persistence`) does not yet require. Those are
/// forward-compatible.
pub const MODEL_PAYLOAD_VERSION: u32 = 6;

/// Schema-free saved-model metadata keyed by stable group id.
///
/// The values are JSON rather than a typed enum because group provenance is
/// supplied by caller-owned catalogs. `FittedModelPayload::group_metadata`
/// wraps this in `Option` with `#[serde(default)]`, so model files written
/// before the field existed deserialize as `None`.
pub type GroupMetadata = BTreeMap<String, JsonValue>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSchema {
    pub columns: Vec<SchemaColumn>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    pub kind: ColumnKindTag,
    #[serde(default)]
    pub levels: Vec<String>,
}

/// Typed error surface for `src/inference/model.rs` saved-model code.
///
/// Every variant carries a free-form `reason: String` payload; `Display`
/// emits exactly that payload, so converting a `FittedModelError` into
/// `String` (via the `From` impl below) is byte-equivalent to the pre-
/// refactor `Err(format!(...))` / `Err("...".to_string())` strings that
/// the same call sites produced. This lets external callers keep using
/// `?` against `Result<_, String>` without source changes — the typed
/// enum is purely an in-module discipline gain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FittedModelError {
    /// Saved payload structure / shape / version disagrees with what the
    /// current binary expects (e.g. covariance shape, block ordering,
    /// schema version, C2 continuity, out-of-range span/basis indices).
    SchemaMismatch { reason: String },
    /// Saved payload bytes / numeric content are corrupt or unreadable
    /// (non-finite scalars, invalid JSON, IO failure, malformed stateful
    /// link state).
    PayloadCorrupt { reason: String },
    /// A required field that the current code path needs is absent from
    /// the payload (typically `..; refit` errors).
    MissingField { reason: String },
    /// A combination of saved-model options is not supported by the
    /// current binary (unsupported deployment-extension kind, unsupported
    /// kernel marker, unsupported survival_time_basis variant, etc.).
    IncompatibleConfig { reason: String },
    /// An input value rejected by a save-time sanity gate (e.g. negative
    /// ridge alpha).
    InvalidInput { reason: String },
}

impl std::fmt::Display for FittedModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaMismatch { reason }
            | Self::PayloadCorrupt { reason }
            | Self::MissingField { reason }
            | Self::IncompatibleConfig { reason }
            | Self::InvalidInput { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for FittedModelError {}

impl From<FittedModelError> for String {
    fn from(err: FittedModelError) -> String {
        err.to_string()
    }
}

// Boundary conversions so that external `Result<_, EstimationError>` /
// `Result<_, SurvivalPredictError>` call sites continue to propagate via
// `?` without per-callsite `.map_err`. The category mapping below mirrors
// the pre-refactor `String` → `EstimationError::InvalidInput(s)` /
// `SurvivalPredictError::from(s)` defaults: a typed `FittedModelError`
// flowing across the boundary is recategorised with the same Display text.
impl From<FittedModelError> for crate::solver::estimate::EstimationError {
    fn from(err: FittedModelError) -> Self {
        crate::solver::estimate::EstimationError::InvalidInput(err.to_string())
    }
}

impl From<FittedModelError> for crate::families::survival_predict::SurvivalPredictError {
    fn from(err: FittedModelError) -> Self {
        // Route by saved-model category so the boundary preserves the
        // categorical information the load path already discovered while
        // keeping the human-readable Display text byte-equivalent.
        let reason = err.to_string();
        match err {
            FittedModelError::SchemaMismatch { .. } => {
                crate::families::survival_predict::SurvivalPredictError::IncompatibleSchema {
                    reason,
                }
            }
            FittedModelError::PayloadCorrupt { .. } => {
                crate::families::survival_predict::SurvivalPredictError::InvalidInput { reason }
            }
            FittedModelError::MissingField { .. } => {
                crate::families::survival_predict::SurvivalPredictError::MissingFitMetadata {
                    reason,
                }
            }
            FittedModelError::IncompatibleConfig { .. } => {
                crate::families::survival_predict::SurvivalPredictError::UnsupportedConfiguration {
                    reason,
                }
            }
            FittedModelError::InvalidInput { .. } => {
                crate::families::survival_predict::SurvivalPredictError::InvalidInput { reason }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedLatentZNormalization {
    pub mean: f64,
    pub sd: f64,
}

impl SavedLatentZNormalization {
    pub fn validate(&self, context: &str) -> Result<(), FittedModelError> {
        if !self.mean.is_finite() {
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!("{context} latent z mean must be finite"),
            });
        }
        if !(self.sd.is_finite() && self.sd > 1e-12) {
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!(
                    "{context} latent z sd must be finite and > 1e-12; got {}",
                    self.sd
                ),
            });
        }
        Ok(())
    }

    pub fn apply(&self, z: &Array1<f64>, context: &str) -> Result<Array1<f64>, FittedModelError> {
        self.validate(context)?;
        if z.iter().any(|value| !value.is_finite()) {
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!("{context} requires finite z values"),
            });
        }
        Ok(z.mapv(|zi| (zi - self.mean) / self.sd))
    }
}

pub const TRANSFORMATION_SCORE_PIT_CLIP_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum TransformationScoreKind {
    FiniteSupportPit,
}

impl Default for TransformationScoreKind {
    fn default() -> Self {
        Self::FiniteSupportPit
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TransformationScoreCalibration {
    #[serde(default)]
    pub score_kind: TransformationScoreKind,
    #[serde(default = "default_transformation_score_pit_clip_eps")]
    pub clip_eps: f64,
}

fn default_transformation_score_pit_clip_eps() -> f64 {
    TRANSFORMATION_SCORE_PIT_CLIP_EPS
}

impl TransformationScoreCalibration {
    pub fn finite_support_pit() -> Self {
        Self {
            score_kind: TransformationScoreKind::FiniteSupportPit,
            clip_eps: TRANSFORMATION_SCORE_PIT_CLIP_EPS,
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), FittedModelError> {
        if self.score_kind != TransformationScoreKind::FiniteSupportPit {
            return Err(FittedModelError::IncompatibleConfig {
                reason: format!("{context} supports only finite-support CTN PIT score semantics"),
            });
        }
        if !(self.clip_eps.is_finite() && self.clip_eps > 0.0 && self.clip_eps < 0.5) {
            return Err(FittedModelError::IncompatibleConfig {
                reason: format!(
                    "{context} requires PIT clip_eps in (0, 0.5), got {}",
                    self.clip_eps
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ColumnKindTag {
    Continuous,
    Binary,
    Categorical,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FittedModelPayload {
    pub version: u32,
    pub formula: String,
    pub model_kind: ModelKind,
    pub family_state: FittedFamily,
    pub family: String,
    #[serde(default)]
    pub fit_result: Option<UnifiedFitResult>,
    /// Unified (family-agnostic) representation of the fit result.
    #[serde(default)]
    pub unified: Option<UnifiedFitResult>,
    #[serde(default)]
    pub data_schema: Option<DataSchema>,
    pub link: Option<InverseLink>,
    #[serde(default)]
    pub mixture_link_param_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub sas_param_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub formula_noise: Option<String>,
    #[serde(default)]
    pub formula_logslope: Option<String>,
    #[serde(default)]
    pub formula_logslopes: Option<Vec<String>>,
    #[serde(default)]
    pub offset_column: Option<String>,
    #[serde(default)]
    pub noise_offset_column: Option<String>,
    #[serde(default)]
    pub beta_noise: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_projection: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub noise_center: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_non_intercept_start: Option<usize>,
    /// Tikhonov ridge alpha used by `solve_scale_projection` when fitting
    /// `noise_projection`.  Persisted so prediction-time replay is identical
    /// to fit-time projection.  `None` for legacy payloads (interpreted as 0).
    #[serde(default)]
    pub noise_projection_ridge_alpha: Option<f64>,
    #[serde(default)]
    pub gaussian_response_scale: Option<f64>,
    #[serde(default)]
    pub linkwiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub linkwiggle_degree: Option<usize>,
    #[serde(default)]
    pub beta_link_wiggle: Option<Vec<f64>>,
    #[serde(default)]
    pub baseline_timewiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub baseline_timewiggle_degree: Option<usize>,
    #[serde(default)]
    pub baseline_timewiggle_penalty_orders: Option<Vec<usize>>,
    #[serde(default)]
    pub baseline_timewiggle_double_penalty: Option<bool>,
    #[serde(default)]
    pub beta_baseline_timewiggle: Option<Vec<f64>>,
    #[serde(default)]
    pub beta_baseline_timewiggle_by_cause: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub z_column: Option<String>,
    #[serde(default)]
    pub z_columns: Option<Vec<String>>,
    #[serde(default)]
    pub latent_z_normalization: Option<SavedLatentZNormalization>,
    #[serde(default)]
    pub latent_score_contract: Option<SavedLatentScoreContract>,
    #[serde(default)]
    pub latent_measure: Option<LatentMeasureKind>,
    /// Optional rank-INT calibration for the latent score (BMS family).
    /// When `Some`, the marginal-slope predictor routes the input `z`
    /// through [`LatentZRankIntCalibration::apply_at_predict`] before the
    /// closed-form standard-normal kernel, matching fit-time semantics.
    /// `#[serde(default)]` so models persisted before this field existed
    /// continue to deserialize cleanly (interpreted as: no calibration).
    #[serde(default)]
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    #[serde(default)]
    pub marginal_baseline: Option<f64>,
    #[serde(default)]
    pub logslope_baseline: Option<f64>,
    #[serde(default)]
    pub logslope_baselines: Option<Vec<f64>>,
    #[serde(default)]
    pub score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
    #[serde(default)]
    pub link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
    #[serde(default)]
    pub survival_entry: Option<String>,
    #[serde(default)]
    pub survival_exit: Option<String>,
    #[serde(default)]
    pub survival_event: Option<String>,
    #[serde(default)]
    pub survivalspec: Option<String>,
    #[serde(default)]
    pub survival_cause_count: Option<usize>,
    #[serde(default)]
    pub survival_endpoint_names: Option<Vec<String>>,
    #[serde(default)]
    pub survival_baseline_target: Option<String>,
    #[serde(default)]
    pub survival_baseline_scale: Option<f64>,
    #[serde(default)]
    pub survival_baseline_shape: Option<f64>,
    #[serde(default)]
    pub survival_baseline_rate: Option<f64>,
    #[serde(default)]
    pub survival_baseline_makeham: Option<f64>,
    #[serde(default)]
    pub survival_time_basis: Option<String>,
    #[serde(default)]
    pub survival_time_degree: Option<usize>,
    #[serde(default)]
    pub survival_time_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_time_keep_cols: Option<Vec<usize>>,
    #[serde(default)]
    pub survival_time_smooth_lambda: Option<f64>,
    #[serde(default)]
    pub survival_time_anchor: Option<f64>,
    #[serde(default)]
    pub survivalridge_lambda: Option<f64>,
    #[serde(default)]
    pub survival_likelihood: Option<String>,
    #[serde(default)]
    pub survival_beta_time: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_beta_threshold: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_beta_log_sigma: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_projection: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub survival_noise_center: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_non_intercept_start: Option<usize>,
    /// Survival analog of `noise_projection_ridge_alpha`: the Tikhonov ridge
    /// used when fitting the survival log-sigma projection.  See doc comment
    /// on `noise_projection_ridge_alpha`.
    #[serde(default)]
    pub survival_noise_projection_ridge_alpha: Option<f64>,
    #[serde(default)]
    pub survival_distribution: Option<ResidualDistribution>,
    #[serde(default)]
    pub training_headers: Option<Vec<String>>,
    /// Per-column (min, max) of the training input matrix, parallel to
    /// `training_headers`. At predict time, inputs are axis-clipped to these
    /// ranges so that out-of-distribution points evaluate at the nearest face
    /// of the training bounding box rather than extrapolating polynomial
    /// trends from polyharmonic / spline bases beyond the data envelope. Old
    /// model JSONs that pre-date this field load with `None`, in which case
    /// the predict path falls through unchanged (no clipping).
    #[serde(default)]
    pub training_feature_ranges: Option<Vec<(f64, f64)>>,
    /// User-supplied per-group metadata, keyed by stable group identifier.
    ///
    /// This is intentionally schema-free JSON so provenance maps can carry
    /// mixed scalar/list/object values. Missing in older payloads means no
    /// group metadata was persisted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_metadata: Option<GroupMetadata>,
    /// Deployment-time no-refit group extensions applied after fitting.
    ///
    /// Each entry records the requested group coordinate, caller metadata, and
    /// prior used to initialize the inserted coefficient. The active
    /// prediction contract lives in `data_schema` + `resolved_termspec`; this
    /// ledger preserves provenance without requiring a refit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deployment_extensions: Vec<SavedDeploymentExtension>,
    /// Transformation-normal: B-spline knots for the response-direction basis.
    #[serde(default)]
    pub transformation_response_knots: Option<Vec<f64>>,
    /// Transformation-normal: deviation nullspace transform matrix (row-major).
    #[serde(default)]
    pub transformation_response_transform: Option<Vec<Vec<f64>>>,
    /// Transformation-normal: B-spline degree for the response basis.
    #[serde(default)]
    pub transformation_response_degree: Option<usize>,
    /// Transformation-normal: median of the response used for anchoring.
    #[serde(default)]
    pub transformation_response_median: Option<f64>,
    /// Transformation-normal saved score contract. The score is the exact
    /// finite-support PIT:
    /// z = Phi^{-1}((Phi(h) - Phi(h_L)) / (Phi(h_U) - Phi(h_L))).
    #[serde(default)]
    pub transformation_score_calibration: Option<TransformationScoreCalibration>,
    #[serde(default)]
    pub resolved_termspec: Option<TermCollectionSpec>,
    #[serde(default)]
    pub resolved_termspec_noise: Option<TermCollectionSpec>,
    #[serde(default)]
    pub resolved_termspec_logslope: Option<TermCollectionSpec>,
    #[serde(default)]
    pub resolved_termspec_logslopes: Option<Vec<TermCollectionSpec>>,
    #[serde(default)]
    pub adaptive_regularization_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedDeploymentExtension {
    pub name: String,
    pub kind: String,
    pub term: String,
    pub level: JsonValue,
    pub level_bits: u64,
    pub coefficient_index: usize,
    pub coefficient_mean: f64,
    pub coefficient_variance: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior: Option<JsonValue>,
}

/// Append deployment-only extension columns to the fitted design coordinate system.
///
/// No-refit group extension adds a new coefficient block after the fitted
/// coefficient vector:
///
///   beta_ext = [beta_old, beta_new],    beta_new = mu_new.
///
/// For a new random-effect level g, the appended basis is the indicator
/// e_g(x_i) = 1{x_i == g}.  The old fitted basis X_old is not rebuilt or
/// reordered, so rows that do not exercise g have
///
///   eta_ext = X_old beta_old + 0 * beta_new = eta_old.
///
/// Rows at the new level get the exact prior-mean shift e_g beta_new.  This
/// helper enforces the coordinate identity by requiring extension coefficient
/// indices to be the consecutive tail columns of the base design.
pub fn append_deployment_extension_columns(
    model: &FittedModelPayload,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    base_design: Array2<f64>,
) -> Result<Array2<f64>, FittedModelError> {
    if model.deployment_extensions.is_empty() {
        return Ok(base_design);
    }
    if base_design.nrows() != data.nrows() {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "deployment extension design row mismatch: base design has {} rows but data has {}",
                base_design.nrows(),
                data.nrows()
            ),
        });
    }
    let spec = model
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| FittedModelError::MissingField {
            reason: "deployment extension prediction requires saved resolved_termspec; refit"
                .to_string(),
        })?;
    let n = base_design.nrows();
    let p_old = base_design.ncols();
    let mut extensions: Vec<&SavedDeploymentExtension> =
        model.deployment_extensions.iter().collect();
    extensions.sort_by_key(|extension| extension.coefficient_index);
    for (tail_idx, extension) in extensions.iter().enumerate() {
        let expected = p_old + tail_idx;
        if extension.coefficient_index != expected {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "deployment extension '{}' has coefficient index {}, expected append-only index {}",
                    extension.name, extension.coefficient_index, expected
                ),
            });
        }
    }

    let mut out = Array2::<f64>::zeros((n, p_old + extensions.len()));
    out.slice_mut(ndarray::s![.., ..p_old]).assign(&base_design);
    for (tail_idx, extension) in extensions.into_iter().enumerate() {
        if extension.kind != "random-effect-level" {
            return Err(FittedModelError::IncompatibleConfig {
                reason: format!(
                    "unsupported deployment extension kind '{}' for '{}'",
                    extension.kind, extension.name
                ),
            });
        }
        let term = spec
            .random_effect_terms
            .iter()
            .find(|term| term.name == extension.term)
            .ok_or_else(|| FittedModelError::MissingField {
                reason: format!(
                    "deployment extension '{}' references unknown random-effect term '{}'",
                    extension.name, extension.term
                ),
            })?;
        let prediction_col = training_headers
            .and_then(|headers| headers.get(term.feature_col))
            .and_then(|name| col_map.get(name))
            .copied()
            .unwrap_or(term.feature_col);
        if prediction_col >= data.ncols() {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "deployment extension '{}' feature column {} out of bounds for {} prediction columns",
                    extension.name,
                    prediction_col,
                    data.ncols()
                ),
            });
        }
        let col = p_old + tail_idx;
        for row in 0..n {
            if data[[row, prediction_col]].to_bits() == extension.level_bits {
                out[[row, col]] = 1.0;
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedLatentScoreContract {
    pub semantics: String,
    pub source_transform_id: Option<String>,
    pub normalization_mean: f64,
    pub normalization_sd: f64,
    pub clip_eps: Option<f64>,
    pub conditioning_columns: Vec<String>,
}

impl FittedModelPayload {
    pub fn new(
        version: u32,
        formula: String,
        model_kind: ModelKind,
        family_state: FittedFamily,
        family: String,
    ) -> Self {
        Self {
            version,
            formula,
            model_kind,
            family_state,
            family,
            fit_result: None,
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            formula_logslopes: None,
            offset_column: None,
            noise_offset_column: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            noise_projection_ridge_alpha: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            beta_baseline_timewiggle_by_cause: None,
            z_column: None,
            z_columns: None,
            latent_z_normalization: None,
            latent_score_contract: None,
            latent_measure: None,
            latent_z_rank_int_calibration: None,
            marginal_baseline: None,
            logslope_baseline: None,
            logslope_baselines: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_cause_count: None,
            survival_endpoint_names: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survival_time_anchor: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_noise_projection_ridge_alpha: None,
            survival_distribution: None,
            training_headers: None,
            training_feature_ranges: None,
            group_metadata: None,
            deployment_extensions: Vec::new(),
            transformation_response_knots: None,
            transformation_response_transform: None,
            transformation_response_degree: None,
            transformation_response_median: None,
            transformation_score_calibration: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            resolved_termspec_logslope: None,
            resolved_termspec_logslopes: None,
            adaptive_regularization_diagnostics: None,
        }
    }

    pub fn set_training_feature_metadata(
        &mut self,
        headers: Vec<String>,
        feature_ranges: Vec<(f64, f64)>,
    ) {
        self.training_headers = Some(headers);
        self.training_feature_ranges = Some(feature_ranges);
    }

    /// Write the persistable time-basis snapshot for a survival model.
    ///
    /// This is the only path that should populate the `survival_time_*`
    /// fields used by the loader. Routing every FFI builder through this
    /// helper guarantees no builder can silently drop a field — the
    /// gamfit 0.1.69 marginal-slope save→load bug was a builder that
    /// missed `survival_time_basis`.
    pub fn apply_survival_time_basis(
        &mut self,
        snapshot: &crate::families::survival_construction::SavedSurvivalTimeBasis,
    ) {
        self.survival_time_basis = Some(snapshot.basisname.clone());
        self.survival_time_degree = snapshot.degree;
        self.survival_time_knots = snapshot.knots.clone();
        self.survival_time_keep_cols = snapshot.keep_cols.clone();
        self.survival_time_smooth_lambda = snapshot.smooth_lambda;
        self.survival_time_anchor = Some(snapshot.anchor);
    }

    fn validate_payload_version(&self) -> Result<(), FittedModelError> {
        if self.version != MODEL_PAYLOAD_VERSION {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved model payload schema mismatch: file has version={}, \
                 this binary expects MODEL_PAYLOAD_VERSION={}. \
                 Refit with the current CLI, or rebuild the reader at the same \
                 version the model was written with.",
                    self.version, MODEL_PAYLOAD_VERSION
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
pub enum FittedModel {
    Standard { payload: FittedModelPayload },
    LocationScale { payload: FittedModelPayload },
    MarginalSlope { payload: FittedModelPayload },
    Survival { payload: FittedModelPayload },
    TransformationNormal { payload: FittedModelPayload },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Standard,
    LocationScale,
    MarginalSlope,
    Survival,
    TransformationNormal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "family_kind", rename_all = "kebab-case")]
pub enum FittedFamily {
    Standard {
        likelihood: LikelihoodFamily,
        link: Option<LinkFunction>,
        #[serde(default)]
        latent_cloglog_state: Option<LatentCLogLogState>,
        #[serde(default)]
        mixture_state: Option<MixtureLinkState>,
        #[serde(default)]
        sas_state: Option<SasLinkState>,
    },
    LocationScale {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        base_link: Option<InverseLink>,
    },
    MarginalSlope {
        likelihood: LikelihoodFamily,
        base_link: Option<InverseLink>,
        frailty: FrailtySpec,
    },
    Survival {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        survival_likelihood: Option<String>,
        #[serde(default)]
        survival_distribution: Option<ResidualDistribution>,
        frailty: FrailtySpec,
    },
    LatentSurvival {
        frailty: FrailtySpec,
    },
    LatentBinary {
        frailty: FrailtySpec,
    },
    TransformationNormal {
        likelihood: LikelihoodFamily,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictModelClass {
    Standard,
    GaussianLocationScale,
    BinomialLocationScale,
    BernoulliMarginalSlope,
    Survival,
    TransformationNormal,
}

impl PredictModelClass {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::GaussianLocationScale => "gaussian location-scale",
            Self::BinomialLocationScale => "binomial location-scale",
            Self::BernoulliMarginalSlope => "bernoulli marginal-slope",
            Self::Survival => "survival",
            Self::TransformationNormal => "transformation-normal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SavedLinkWiggleRuntime {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub beta: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SavedBaselineTimeWiggleRuntime {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub beta: Vec<f64>,
}

// Re-export so saved-model consumers can refer to the anchor-block tag
// without reaching across module boundaries.
pub use crate::families::bernoulli_marginal_slope::deviation_runtime::ParametricAnchorBlock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedAnchoredDeviationRuntime {
    pub kernel: String,
    pub breakpoints: Vec<f64>,
    pub basis_dim: usize,
    pub span_c0: Vec<Vec<f64>>,
    pub span_c1: Vec<Vec<f64>>,
    pub span_c2: Vec<Vec<f64>>,
    pub span_c3: Vec<Vec<f64>>,
    /// Cross-block anchor-residual coefficient matrix `M` of shape
    /// `d × basis_dim`. When present, predict-time evaluation subtracts
    /// `n_row · M` from each cubic-span row (where `n_row` stacks the
    /// per-row parametric anchor values in the order given by
    /// `anchor_residual_components`).
    #[serde(default)]
    pub anchor_residual_coefficients: Option<Vec<Vec<f64>>>,
    /// Ordered list of parametric anchor components whose stacked row
    /// values combine into `n_row`. Empty unless
    /// `anchor_residual_coefficients` is `Some`.
    #[serde(default)]
    pub anchor_residual_components: Vec<SavedAnchorComponent>,
    /// Optional `d × d` orthonormalising rotation. The current
    /// construction bakes the rotation into
    /// `anchor_residual_coefficients`, so this is always either `None`
    /// or the identity. Reserved for layouts that store the rotation
    /// separately from the coefficient matrix.
    #[serde(default)]
    pub anchor_residual_rotation: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedAnchorComponent {
    pub kind: SavedAnchorKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SavedAnchorKind {
    Parametric {
        block: ParametricAnchorBlock,
        ncols: usize,
    },
}

#[derive(Clone, Debug)]
pub struct SavedPredictionRuntime {
    pub model_class: PredictModelClass,
    pub likelihood: LikelihoodFamily,
    pub inverse_link: Option<InverseLink>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
    pub baseline_time_wiggle: Option<SavedBaselineTimeWiggleRuntime>,
    pub score_warp: Option<SavedAnchoredDeviationRuntime>,
    pub link_deviation: Option<SavedAnchoredDeviationRuntime>,
    /// Rank-INT latent-z calibration carried into the predictor build.
    /// `None` for non-BMS models and for BMS fits whose latent measure
    /// did not require rank-INT calibration.
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
}

fn gaussian_location_scale_mean_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Location)
        .or_else(|| fit.block_by_role(BlockRole::Mean))
        .map(|block| block.beta.clone())
}

fn binomial_location_scale_threshold_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Threshold)
        .or_else(|| fit.block_by_role(BlockRole::Location))
        .or_else(|| fit.block_by_role(BlockRole::Mean))
        .map(|block| block.beta.clone())
}

fn location_scale_noise_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Scale)
        .map(|block| block.beta.clone())
}

fn validate_location_scale_saved_fit(
    fit: &UnifiedFitResult,
    model_class: PredictModelClass,
    link_wiggle: Option<&SavedLinkWiggleRuntime>,
) -> Result<(), FittedModelError> {
    let primary = match model_class {
        PredictModelClass::GaussianLocationScale => gaussian_location_scale_mean_beta(fit),
        PredictModelClass::BinomialLocationScale => binomial_location_scale_threshold_beta(fit),
        _ => None,
    }
    .ok_or_else(|| FittedModelError::MissingField {
        reason: match model_class {
            PredictModelClass::GaussianLocationScale => {
                "gaussian-location-scale saved fit is missing mean/location block".to_string()
            }
            PredictModelClass::BinomialLocationScale => {
                "binomial-location-scale saved fit is missing threshold/location block".to_string()
            }
            _ => "location-scale saved fit is missing primary block".to_string(),
        },
    })?;

    let scale = location_scale_noise_beta(fit).ok_or_else(|| FittedModelError::MissingField {
        reason: "location-scale saved fit is missing scale block".to_string(),
    })?;
    let expected =
        primary.len() + scale.len() + link_wiggle.map_or(0, |runtime| runtime.beta.len());

    if let Some(cov) = fit.beta_covariance()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "location-scale saved conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                expected,
                expected
            ),
        });
    }
    if let Some(cov) = fit.beta_covariance_corrected()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "location-scale saved corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                expected,
                expected
            ),
        });
    }
    Ok(())
}

fn validate_survival_saved_block_matches_payload(
    fit: &UnifiedFitResult,
    role: BlockRole,
    payload_beta: Option<&Vec<f64>>,
    label: &str,
) -> Result<usize, FittedModelError> {
    let block = fit
        .block_by_role(role)
        .ok_or_else(|| FittedModelError::MissingField {
            reason: format!("location-scale survival saved fit is missing {label} block"),
        })?;
    if let Some(saved) = payload_beta
        && block.beta.to_vec() != *saved
    {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "location-scale survival saved {label} coefficients disagree with fit_result"
            ),
        });
    }
    Ok(block.beta.len())
}

fn validate_survival_location_scale_saved_fit(
    payload: &FittedModelPayload,
    link_wiggle: Option<&SavedLinkWiggleRuntime>,
) -> Result<(), FittedModelError> {
    let fit = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| FittedModelError::MissingField {
            reason: "location-scale survival model is missing canonical fit_result payload"
                .to_string(),
        })?;
    let p_time = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Time,
        payload.survival_beta_time.as_ref(),
        "time",
    )?;
    let p_threshold = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Threshold,
        payload.survival_beta_threshold.as_ref(),
        "threshold",
    )?;
    let p_log_sigma = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Scale,
        payload.survival_beta_log_sigma.as_ref(),
        "log-sigma",
    )?;
    let p_wiggle = match link_wiggle {
        Some(runtime) => {
            let block = fit.block_by_role(BlockRole::LinkWiggle).ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "location-scale survival saved fit is missing link-wiggle block"
                        .to_string(),
                }
            })?;
            if block.beta.to_vec() != runtime.beta {
                return Err(FittedModelError::SchemaMismatch {
                    reason:
                        "location-scale survival saved link-wiggle coefficients disagree with fit_result"
                            .to_string(),
                });
            }
            runtime.beta.len()
        }
        None => {
            if fit.block_by_role(BlockRole::LinkWiggle).is_some() {
                return Err(FittedModelError::SchemaMismatch {
                    reason:
                        "location-scale survival saved fit has a LinkWiggle block without payload metadata"
                            .to_string(),
                });
            }
            0
        }
    };
    let expected = p_time + p_threshold + p_log_sigma + p_wiggle;

    if let Some(cov) = fit.beta_covariance()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "location-scale survival saved conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                expected,
                expected
            ),
        });
    }
    if let Some(cov) = fit.beta_covariance_corrected()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "location-scale survival saved corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                expected,
                expected
            ),
        });
    }
    Ok(())
}

fn validate_marginal_slope_saved_fit(
    fit: &UnifiedFitResult,
    score_warp: Option<&SavedAnchoredDeviationRuntime>,
    link_deviation: Option<&SavedAnchoredDeviationRuntime>,
    fit_label: &str,
) -> Result<(), FittedModelError> {
    validate_marginal_slope_saved_fit_impl(
        fit,
        score_warp,
        link_deviation,
        fit_label,
        "bernoulli",
        2,
        "marginal, logslope",
    )
}

fn validate_survival_marginal_slope_saved_fit(
    fit: &UnifiedFitResult,
    score_warp: Option<&SavedAnchoredDeviationRuntime>,
    link_deviation: Option<&SavedAnchoredDeviationRuntime>,
    fit_label: &str,
) -> Result<(), FittedModelError> {
    validate_marginal_slope_saved_fit_impl(
        fit,
        score_warp,
        link_deviation,
        fit_label,
        "survival",
        3,
        "time, marginal, slope",
    )
}

/// Shared block-count + coefficient-dimension validation for the bernoulli
/// and survival marginal-slope saved-fit gates. The only family-specific
/// inputs are the family kind string ("bernoulli" / "survival"), the base
/// block count (2 for bernoulli, 3 for survival — the survival path has an
/// extra time block), and the base-block role list rendered in the error
/// message ("marginal, logslope" / "time, marginal, slope"). The score-warp
/// / link-deviation tail follows the same shape in both families.
fn validate_marginal_slope_saved_fit_impl(
    fit: &UnifiedFitResult,
    score_warp: Option<&SavedAnchoredDeviationRuntime>,
    link_deviation: Option<&SavedAnchoredDeviationRuntime>,
    fit_label: &str,
    family_kind: &str,
    base_block_count: usize,
    base_block_role_list: &str,
) -> Result<(), FittedModelError> {
    let expected_blocks = base_block_count
        + usize::from(score_warp.is_some())
        + usize::from(link_deviation.is_some());
    if fit.blocks.len() != expected_blocks {
        let score_warp_suffix = if score_warp.is_some() {
            ", score-warp"
        } else {
            ""
        };
        let link_deviation_suffix = if link_deviation.is_some() {
            ", link-deviation"
        } else {
            ""
        };
        return Err(FittedModelError::SchemaMismatch {
            reason: format!(
                "{family_kind} marginal-slope saved {fit_label} requires {expected_blocks} blocks [{base_block_role_list}{score_warp_suffix}{link_deviation_suffix}], got {}",
                fit.blocks.len(),
            ),
        });
    }
    if let Some(runtime) = score_warp {
        let beta = &fit.blocks[base_block_count].beta;
        if beta.len() != runtime.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "{family_kind} marginal-slope saved {fit_label} score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                    beta.len(),
                    runtime.basis_dim
                ),
            });
        }
    }
    if let Some(runtime) = link_deviation {
        let idx = base_block_count + usize::from(score_warp.is_some());
        let beta = &fit.blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "{family_kind} marginal-slope saved {fit_label} link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                    beta.len(),
                    runtime.basis_dim
                ),
            });
        }
    }
    Ok(())
}

impl SavedLinkWiggleRuntime {
    fn validate_global_monotonicity(&self) -> Result<(), FittedModelError> {
        validate_monotone_wiggle_beta_nonnegative(&self.beta, "saved link-wiggle")
            .map_err(|reason| FittedModelError::PayloadCorrupt { reason })
    }

    fn validate_monotone_derivative(
        &self,
        q0: &Array1<f64>,
    ) -> Result<Array1<f64>, FittedModelError> {
        self.validate_global_monotonicity()?;
        let d_constrained = self.constrained_basis(q0, BasisOptions::first_derivative())?;
        let beta_link_wiggle = Array1::from_vec(self.beta.clone());
        let dq_dq0 = d_constrained.dot(&beta_link_wiggle) + 1.0;
        if let Some((idx, value)) = dq_dq0.iter().copied().enumerate().find(|(_, v)| *v <= 0.0) {
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!(
                    "saved link-wiggle is not monotone at row {idx}: dq/dq0={value:.3e} <= 0"
                ),
            });
        }
        Ok(dq_dq0)
    }

    pub fn constrained_basis(
        &self,
        q0: &Array1<f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, FittedModelError> {
        let knot_arr = Array1::from_vec(self.knots.clone());
        let constrained = monotone_wiggle_basis_with_derivative_order(
            q0.view(),
            &knot_arr,
            self.degree,
            basis_options.derivative_order,
        )
        .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
        if constrained.ncols() != self.beta.len() {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved link-wiggle dimension mismatch: coefficients have {} entries but basis has {} columns",
                    self.beta.len(),
                    constrained.ncols()
                ),
            });
        }
        Ok(constrained)
    }

    pub fn design(&self, q0: &Array1<f64>) -> Result<Array2<f64>, FittedModelError> {
        self.validate_global_monotonicity()?;
        self.constrained_basis(q0, BasisOptions::value())
    }

    pub fn basis_row_scalar(&self, q0: f64) -> Result<Array1<f64>, FittedModelError> {
        let q = Array1::from_vec(vec![q0]);
        let x = self.design(&q)?;
        if x.nrows() != 1 {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved link-wiggle scalar evaluation expected 1 row, got {}",
                    x.nrows()
                ),
            });
        }
        Ok(x.row(0).to_owned())
    }

    pub fn apply(&self, q0: &Array1<f64>) -> Result<Array1<f64>, FittedModelError> {
        self.validate_monotone_derivative(q0)?;
        let xwiggle = self.constrained_basis(q0, BasisOptions::value())?;
        let beta_link_wiggle = Array1::from_vec(self.beta.clone());
        Ok(q0 + &xwiggle.dot(&beta_link_wiggle))
    }

    pub fn derivative_q0(&self, q0: &Array1<f64>) -> Result<Array1<f64>, FittedModelError> {
        self.validate_monotone_derivative(q0)
    }
}

impl SavedBaselineTimeWiggleRuntime {
    pub fn validate_global_monotonicity(&self) -> Result<(), FittedModelError> {
        validate_monotone_wiggle_beta_nonnegative(&self.beta, "saved baseline-timewiggle")
            .map_err(|reason| FittedModelError::PayloadCorrupt { reason })
    }
}

impl SavedAnchoredDeviationRuntime {
    pub(crate) fn validate_exact_replay_contract(&self) -> Result<(), FittedModelError> {
        if self.kernel.is_empty() {
            return Err(FittedModelError::SchemaMismatch {
                reason: "saved anchored deviation runtime is missing the exact kernel marker"
                    .to_string(),
            });
        }
        if self.kernel
            != crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL
        {
            return Err(FittedModelError::IncompatibleConfig {
                reason: format!(
                    "saved anchored deviation runtime uses unsupported kernel '{}'; expected {}",
                    self.kernel,
                    crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL
                ),
            });
        }
        if self.basis_dim == 0 {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation runtime basis_dim must be positive, got {}",
                    self.basis_dim
                ),
            });
        }
        if self.breakpoints.len() < 2 {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation runtime requires at least two breakpoints, got {}",
                    self.breakpoints.len()
                ),
            });
        }
        for window in self.breakpoints.windows(2) {
            let left = window[0];
            let right = window[1];
            if !left.is_finite() || !right.is_finite() || right <= left {
                return Err(FittedModelError::PayloadCorrupt {
                    reason: format!(
                        "saved anchored deviation runtime breakpoints must be finite and strictly increasing, got [{left}, {right}]"
                    ),
                });
            }
        }
        let span_count = self.breakpoints.len() - 1;
        self.validate_coefficient_matrix(&self.span_c0, "c0", span_count)?;
        self.validate_coefficient_matrix(&self.span_c1, "c1", span_count)?;
        self.validate_coefficient_matrix(&self.span_c2, "c2", span_count)?;
        self.validate_coefficient_matrix(&self.span_c3, "c3", span_count)?;
        self.validate_c2_span_continuity()?;
        self.validate_anchor_residual_shape()?;
        Ok(())
    }

    fn validate_anchor_residual_shape(&self) -> Result<(), FittedModelError> {
        let coeffs = match self.anchor_residual_coefficients.as_ref() {
            Some(c) => c,
            None => {
                if !self.anchor_residual_components.is_empty() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason:
                            "saved anchored deviation runtime has anchor_residual_components but no anchor_residual_coefficients"
                                .to_string(),
                    });
                }
                if self.anchor_residual_rotation.is_some() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason:
                            "saved anchored deviation runtime has anchor_residual_rotation but no anchor_residual_coefficients"
                                .to_string(),
                    });
                }
                return Ok(());
            }
        };
        let d: usize = self
            .anchor_residual_components
            .iter()
            .map(|c| match &c.kind {
                SavedAnchorKind::Parametric { ncols, .. } => *ncols,
            })
            .sum();
        if coeffs.len() != d {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation runtime anchor_residual_coefficients has {} rows; expected {} (sum of component ncols)",
                    coeffs.len(),
                    d,
                ),
            });
        }
        for (i, row) in coeffs.iter().enumerate() {
            if row.len() != self.basis_dim {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "saved anchored deviation runtime anchor_residual_coefficients row {} has width {}, expected basis_dim {}",
                        i,
                        row.len(),
                        self.basis_dim,
                    ),
                });
            }
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(FittedModelError::PayloadCorrupt {
                        reason: format!(
                            "saved anchored deviation runtime anchor_residual_coefficients ({i},{j}) is non-finite"
                        ),
                    });
                }
            }
        }
        if let Some(rot) = self.anchor_residual_rotation.as_ref() {
            if rot.len() != d {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "saved anchored deviation runtime anchor_residual_rotation has {} rows; expected {}",
                        rot.len(),
                        d,
                    ),
                });
            }
            for (i, row) in rot.iter().enumerate() {
                if row.len() != d {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: format!(
                            "saved anchored deviation runtime anchor_residual_rotation row {} has width {}; expected {}",
                            i,
                            row.len(),
                            d,
                        ),
                    });
                }
                for (j, &v) in row.iter().enumerate() {
                    if !v.is_finite() {
                        return Err(FittedModelError::PayloadCorrupt {
                            reason: format!(
                                "saved anchored deviation runtime anchor_residual_rotation ({i},{j}) is non-finite"
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_c2_span_continuity(&self) -> Result<(), FittedModelError> {
        const TOL: f64 = 1e-8;
        for span_idx in 1..self.breakpoints.len() - 1 {
            let left_span = span_idx - 1;
            let right_span = span_idx;
            let width = self.breakpoints[span_idx] - self.breakpoints[left_span];
            for basis_idx in 0..self.basis_dim {
                let left_value = self.span_c0[left_span][basis_idx]
                    + self.span_c1[left_span][basis_idx] * width
                    + self.span_c2[left_span][basis_idx] * width * width
                    + self.span_c3[left_span][basis_idx] * width * width * width;
                let left_d1 = self.span_c1[left_span][basis_idx]
                    + 2.0 * self.span_c2[left_span][basis_idx] * width
                    + 3.0 * self.span_c3[left_span][basis_idx] * width * width;
                let left_d2 = 2.0 * self.span_c2[left_span][basis_idx]
                    + 6.0 * self.span_c3[left_span][basis_idx] * width;
                let right_value = self.span_c0[right_span][basis_idx];
                let right_d1 = self.span_c1[right_span][basis_idx];
                let right_d2 = 2.0 * self.span_c2[right_span][basis_idx];
                if (left_value - right_value).abs() > TOL
                    || (left_d1 - right_d1).abs() > TOL
                    || (left_d2 - right_d2).abs() > TOL
                {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: format!(
                            "saved anchored deviation runtime must be C2 cubic at breakpoint {span_idx}, basis {basis_idx}: value jump={:.3e}, d1 jump={:.3e}, d2 jump={:.3e}",
                            left_value - right_value,
                            left_d1 - right_d1,
                            left_d2 - right_d2
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_coefficient_matrix(
        &self,
        matrix: &[Vec<f64>],
        label: &str,
        expected_rows: usize,
    ) -> Result<(), FittedModelError> {
        if matrix.len() != expected_rows {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation runtime {label} row count mismatch: got {}, expected {}",
                    matrix.len(),
                    expected_rows
                ),
            });
        }
        for (row_idx, row) in matrix.iter().enumerate() {
            if row.len() != self.basis_dim {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "saved anchored deviation runtime {label} row {} has width {}, expected {}",
                        row_idx,
                        row.len(),
                        self.basis_dim
                    ),
                });
            }
            for (j, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(FittedModelError::PayloadCorrupt {
                        reason: format!(
                            "saved anchored deviation runtime {label} entry ({row_idx},{j}) is non-finite"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn right_boundary_basis_value(&self, basis_idx: usize) -> f64 {
        let last_span = self.breakpoints.len() - 2;
        let width = self.breakpoints[last_span + 1] - self.breakpoints[last_span];
        self.span_c0[last_span][basis_idx]
            + self.span_c1[last_span][basis_idx] * width
            + self.span_c2[last_span][basis_idx] * width * width
            + self.span_c3[last_span][basis_idx] * width * width * width
    }

    fn evaluate_span_polynomial_design(
        &self,
        values: &Array1<f64>,
        derivative_order: usize,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.validate_exact_replay_contract()?;
        let (left_ep, right_ep) = self.support_interval()?;
        let mut out = Array2::<f64>::zeros((values.len(), self.basis_dim));
        for (row_idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(FittedModelError::PayloadCorrupt {
                    reason: format!(
                        "saved anchored deviation runtime design value at row {row_idx} is non-finite ({value})"
                    ),
                });
            }
            if value < left_ep {
                if derivative_order == 0 {
                    for basis_idx in 0..self.basis_dim {
                        out[[row_idx, basis_idx]] = self.span_c0[0][basis_idx];
                    }
                }
                continue;
            }
            if value > right_ep {
                if derivative_order == 0 {
                    for basis_idx in 0..self.basis_dim {
                        out[[row_idx, basis_idx]] = self.right_boundary_basis_value(basis_idx);
                    }
                }
                continue;
            }
            let span_idx = self.left_biased_span_index_for(value)?;
            let t = value - self.breakpoints[span_idx];
            for basis_idx in 0..self.basis_dim {
                let c0 = self.span_c0[span_idx][basis_idx];
                let c1 = self.span_c1[span_idx][basis_idx];
                let c2 = self.span_c2[span_idx][basis_idx];
                let c3 = self.span_c3[span_idx][basis_idx];
                out[[row_idx, basis_idx]] = match derivative_order {
                    0 => c0 + c1 * t + c2 * t * t + c3 * t * t * t,
                    1 => c1 + 2.0 * c2 * t + 3.0 * c3 * t * t,
                    2 => 2.0 * c2 + 6.0 * c3 * t,
                    3 => 6.0 * c3,
                    4 => 0.0,
                    other => {
                        return Err(FittedModelError::IncompatibleConfig {
                            reason: format!(
                                "saved anchored deviation runtime only supports derivative orders up to 4, got {other}"
                            ),
                        });
                    }
                };
            }
        }
        Ok(out)
    }

    pub fn breakpoints(&self) -> Result<Vec<f64>, FittedModelError> {
        self.validate_exact_replay_contract()?;
        Ok(self.breakpoints.clone())
    }

    pub fn span_count(&self) -> Result<usize, FittedModelError> {
        Ok(self.breakpoints()?.windows(2).count())
    }

    pub fn span_index_for(&self, value: f64) -> Result<usize, FittedModelError> {
        let points = self.breakpoints()?;
        span_index_for_breakpoints(&points, value, "saved anchored deviation span lookup")
            .map_err(|reason| FittedModelError::PayloadCorrupt { reason })
    }

    fn left_biased_span_index_for(&self, value: f64) -> Result<usize, FittedModelError> {
        let mut span_idx = span_index_for_breakpoints(
            &self.breakpoints,
            value,
            "saved anchored deviation span lookup",
        )
        .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
        // LEFT-bias at interior breakpoints mirrors DeviationRuntime. The
        // saved cubic basis is C2, but d3 remains span-local.
        if span_idx > 0 && value == self.breakpoints[span_idx] {
            span_idx -= 1;
        }
        Ok(span_idx)
    }

    pub fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        self.validate_exact_replay_contract()?;
        if beta.len() != self.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation coefficient length mismatch: got {}, expected {}",
                    beta.len(),
                    self.basis_dim
                ),
            });
        }
        self.local_cubic_on_span_validated(beta, span_idx)
    }

    fn local_cubic_on_span_validated(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        let points = &self.breakpoints;
        if span_idx + 1 >= points.len() {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation span index {} out of range for {} spans",
                    span_idx,
                    points.len() - 1
                ),
            });
        }
        let left = points[span_idx];
        let right = points[span_idx + 1];
        Ok(
            crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                left,
                right,
                c0: self.span_c0[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c1: self.span_c1[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c2: self.span_c2[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c3: self.span_c3[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
            },
        )
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        self.validate_exact_replay_contract()?;
        if basis_idx >= self.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation basis index {} out of range for {} coefficients",
                    basis_idx, self.basis_dim
                ),
            });
        }
        self.basis_span_cubic_validated(span_idx, basis_idx)
    }

    fn basis_span_cubic_validated(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        let points = &self.breakpoints;
        if span_idx + 1 >= points.len() {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation span index {} out of range for {} spans",
                    span_idx,
                    points.len() - 1
                ),
            });
        }
        Ok(
            crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                left: points[span_idx],
                right: points[span_idx + 1],
                c0: self.span_c0[span_idx][basis_idx],
                c1: self.span_c1[span_idx][basis_idx],
                c2: self.span_c2[span_idx][basis_idx],
                c3: self.span_c3[span_idx][basis_idx],
            },
        )
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        self.validate_exact_replay_contract()?;
        if basis_idx >= self.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation basis index {} out of range for {} coefficients",
                    basis_idx, self.basis_dim
                ),
            });
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: left_ep,
                    right: left_ep + 1.0,
                    c0: self.span_c0[0][basis_idx],
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        if value > right_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: right_ep,
                    right: right_ep + 1.0,
                    c0: self.right_boundary_basis_value(basis_idx),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.basis_span_cubic_validated(span_idx, basis_idx)
    }

    pub fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<
        crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic,
        FittedModelError,
    > {
        self.validate_exact_replay_contract()?;
        if beta.len() != self.basis_dim {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation coefficient length mismatch: got {}, expected {}",
                    beta.len(),
                    self.basis_dim
                ),
            });
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: left_ep,
                    right: left_ep + 1.0,
                    c0: self.span_c0[0]
                        .iter()
                        .zip(beta.iter())
                        .map(|(coeff, weight)| coeff * weight)
                        .sum(),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        if value > right_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: right_ep,
                    right: right_ep + 1.0,
                    c0: (0..self.basis_dim)
                        .map(|basis_idx| {
                            self.right_boundary_basis_value(basis_idx) * beta[basis_idx]
                        })
                        .sum(),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.local_cubic_on_span_validated(beta, span_idx)
    }

    fn support_interval(&self) -> Result<(f64, f64), FittedModelError> {
        let points = self.breakpoints()?;
        match (points.first(), points.last()) {
            (Some(&left), Some(&right)) => Ok((left, right)),
            _ => Err(FittedModelError::MissingField {
                reason: "saved anchored deviation runtime is missing support breakpoints"
                    .to_string(),
            }),
        }
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, FittedModelError> {
        // Note: when the saved runtime carries an anchor residual
        // (cross-block orthogonalisation), the value `design()` returns
        // is the raw cubic span output *without* the per-row `n_row · M`
        // subtraction. Callers used inside BMS prediction must either
        // switch to `design_with_anchor_rows` (when the per-row anchor
        // rows are available) or call `design_uncorrected` explicitly and
        // apply the subtraction at the call site. For runtimes without a
        // residual the two paths coincide.
        self.evaluate_span_polynomial_design(values, BasisOptions::value().derivative_order)
    }

    /// Raw cubic-span design without any anchor-residual subtraction.
    ///
    /// Exposed for callers that intend to apply the `n_row · M` correction
    /// post-hoc (e.g., BMS `link_terms_value_d1` subtracts a precomputed
    /// `correction.dot(beta)` scalar from the linear-predictor contribution
    /// rather than building a full anchor-row matrix). Equivalent to
    /// `design()` when no residual is present.
    pub fn design_uncorrected(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.evaluate_span_polynomial_design(values, BasisOptions::value().derivative_order)
    }

    /// Evaluate the residual-corrected design at the supplied values.
    ///
    /// `anchor_rows` must be an `n × d` matrix where `n == values.len()`
    /// and `d == sum of anchor_residual_components ncols`. Each row holds
    /// the concatenated parametric anchor design at the same prediction
    /// row as the corresponding `values[i]`. When the runtime has no
    /// anchor residual, `anchor_rows` must have zero columns (or be
    /// `Array2::zeros((n, 0))`).
    pub fn design_with_anchor_rows(
        &self,
        values: &Array1<f64>,
        anchor_rows: ndarray::ArrayView2<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        let mut out =
            self.evaluate_span_polynomial_design(values, BasisOptions::value().derivative_order)?;
        if let Some(m_rows) = self.anchor_residual_coefficients.as_ref() {
            let d = m_rows.len();
            if anchor_rows.nrows() != values.len() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "design_with_anchor_rows: anchor_rows has {} rows, expected {} (matching values)",
                        anchor_rows.nrows(),
                        values.len(),
                    ),
                });
            }
            if anchor_rows.ncols() != d {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "design_with_anchor_rows: anchor_rows has {} cols, expected {} (sum of component ncols)",
                        anchor_rows.ncols(),
                        d,
                    ),
                });
            }
            // Materialise M (d × basis_dim) once.
            let mut m_dense = Array2::<f64>::zeros((d, self.basis_dim));
            for (i, row) in m_rows.iter().enumerate() {
                if row.len() != self.basis_dim {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: format!(
                            "design_with_anchor_rows: anchor_residual_coefficients row {} has length {}, expected basis_dim {}",
                            i,
                            row.len(),
                            self.basis_dim,
                        ),
                    });
                }
                for (j, &v) in row.iter().enumerate() {
                    m_dense[[i, j]] = v;
                }
            }
            // Effective subtraction is `n_anchor_rows · R · M` where R is
            // the orthonormalising rotation. In the current construction R
            // is baked into M (the saved rotation is identity and omitted),
            // so the rotation step is a no-op; keep the multiplication
            // explicit so a future non-identity persisted rotation is
            // honoured automatically.
            let rotated_anchor = self.rotated_anchor_rows(anchor_rows, d)?;
            let subtract = rotated_anchor.dot(&m_dense);
            out = out - subtract;
        } else if anchor_rows.ncols() != 0 {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "design_with_anchor_rows: runtime has no anchor residual but anchor_rows has {} cols",
                    anchor_rows.ncols(),
                ),
            });
        }
        Ok(out)
    }

    /// Build the n × basis_dim per-row, per-basis correction matrix
    /// `N · M` for a batch of predict rows.
    ///
    /// `n_anchor_rows` is the n × d matrix of stacked parametric anchor
    /// rows at the prediction rows (concatenation of the marginal and
    /// logslope design rows in component order). Returns `None` when the
    /// runtime has no anchor residual (zero-cost path).
    pub fn anchor_correction_matrix(
        &self,
        n_anchor_rows: ndarray::ArrayView2<f64>,
    ) -> Result<Option<Array2<f64>>, FittedModelError> {
        let Some(m_rows) = self.anchor_residual_coefficients.as_ref() else {
            return Ok(None);
        };
        let d = m_rows.len();
        if n_anchor_rows.ncols() != d {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "anchor_correction_matrix: anchor_rows has {} cols, expected {} (sum of component ncols)",
                    n_anchor_rows.ncols(),
                    d,
                ),
            });
        }
        let mut m_dense = Array2::<f64>::zeros((d, self.basis_dim));
        for (i, row) in m_rows.iter().enumerate() {
            if row.len() != self.basis_dim {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "anchor_correction_matrix: M row {} has length {}, expected basis_dim {}",
                        i,
                        row.len(),
                        self.basis_dim,
                    ),
                });
            }
            for (j, &v) in row.iter().enumerate() {
                m_dense[[i, j]] = v;
            }
        }
        // The effective correction is `n_anchor_rows · R · M` where R is
        // the orthonormalising rotation. Rotation is identity in the
        // current construction (omitted from the saved payload); we keep
        // the explicit multiplication so a future non-identity rotation
        // works without further plumbing.
        let rotated = self.rotated_anchor_rows(n_anchor_rows, d)?;
        Ok(Some(rotated.dot(&m_dense)))
    }

    /// Apply the orthonormalising rotation `R` (d × d) to a row-major
    /// anchor matrix `N` (n × d). Returns `N` unchanged when no rotation
    /// is persisted (the identity case) and `N · R` otherwise.
    fn rotated_anchor_rows(
        &self,
        n_anchor_rows: ndarray::ArrayView2<f64>,
        d: usize,
    ) -> Result<Array2<f64>, FittedModelError> {
        let Some(rot_rows) = self.anchor_residual_rotation.as_ref() else {
            return Ok(n_anchor_rows.to_owned());
        };
        if rot_rows.len() != d {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "rotated_anchor_rows: rotation has {} rows, expected {}",
                    rot_rows.len(),
                    d,
                ),
            });
        }
        let mut rotation = Array2::<f64>::zeros((d, d));
        for (i, row) in rot_rows.iter().enumerate() {
            if row.len() != d {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "rotated_anchor_rows: rotation row {} has length {}, expected {}",
                        i,
                        row.len(),
                        d,
                    ),
                });
            }
            for (j, &v) in row.iter().enumerate() {
                rotation[[i, j]] = v;
            }
        }
        Ok(n_anchor_rows.dot(&rotation))
    }

    pub fn first_derivative_design(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.evaluate_span_polynomial_design(
            values,
            BasisOptions::first_derivative().derivative_order,
        )
    }

    pub fn second_derivative_design(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.evaluate_span_polynomial_design(
            values,
            BasisOptions::second_derivative().derivative_order,
        )
    }

    pub fn third_derivative_design(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.evaluate_span_polynomial_design(values, 3)
    }

    pub fn fourth_derivative_design(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, FittedModelError> {
        self.evaluate_span_polynomial_design(values, 4)
    }
}

impl FittedFamily {
    #[inline]
    pub fn likelihood(&self) -> LikelihoodFamily {
        match self {
            Self::Standard { likelihood, .. }
            | Self::LocationScale { likelihood, .. }
            | Self::MarginalSlope { likelihood, .. }
            | Self::Survival { likelihood, .. }
            | Self::TransformationNormal { likelihood, .. } => *likelihood,
            Self::LatentSurvival { .. } | Self::LatentBinary { .. } => {
                LikelihoodFamily::RoystonParmar
            }
        }
    }

    #[inline]
    pub fn frailty(&self) -> Option<&FrailtySpec> {
        match self {
            Self::MarginalSlope { frailty, .. }
            | Self::Survival { frailty, .. }
            | Self::LatentSurvival { frailty }
            | Self::LatentBinary { frailty } => Some(frailty),
            _ => None,
        }
    }
}

impl FittedModel {
    /// Axis-clip each continuous new-data column to the (min, max) range
    /// observed in training. Categorical and binary columns are left
    /// untouched so unseen levels surface rather than being silently remapped
    /// onto seen ones. Returns `Some(clipped_copy)` only if at least one
    /// value was actually clipped; otherwise `None` so callers can avoid
    /// owning a redundant copy. Pre-2026-04-29 model JSONs that lack the
    /// `training_feature_ranges` field deserialize to `None` and pass through
    /// unchanged.
    pub fn axis_clip_to_training_ranges(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        col_map: &std::collections::HashMap<String, usize>,
    ) -> Option<ndarray::Array2<f64>> {
        let training_headers = self.training_headers.as_ref()?;
        let ranges = self.training_feature_ranges.as_ref()?;
        if training_headers.len() != ranges.len() {
            return None;
        }
        let mut kind_by_header: std::collections::HashMap<&str, ColumnKindTag> =
            std::collections::HashMap::new();
        if let Some(schema) = self.data_schema.as_ref() {
            for col in &schema.columns {
                kind_by_header.insert(col.name.as_str(), col.kind);
            }
        }
        // Periodic axes (sphere longitude, periodic-B-spline 1D, periodic
        // tensor margins) must never be clipped to the training range:
        // clamping a value just past the seam to the training extreme breaks
        // the cyclic invariant f(x₀) = f(x₀ + period) at predict time and
        // shows up as a visible seam in surface plots.
        let periodic_axes = self.training_periodic_axes(training_headers);
        let mut clipped = data.to_owned();
        let mut any_clipped = false;
        for (col_in_training, (header, &(lo, hi))) in
            training_headers.iter().zip(ranges.iter()).enumerate()
        {
            if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
                continue;
            }
            if !matches!(
                kind_by_header.get(header.as_str()).copied(),
                Some(ColumnKindTag::Continuous)
            ) {
                continue;
            }
            if periodic_axes.contains(&col_in_training) {
                continue;
            }
            let Some(&col_idx) = col_map.get(header) else {
                continue;
            };
            if col_idx >= clipped.ncols() {
                continue;
            }
            let mut col = clipped.column_mut(col_idx);
            for v in col.iter_mut() {
                if v.is_finite() {
                    if *v < lo {
                        *v = lo;
                        any_clipped = true;
                    } else if *v > hi {
                        *v = hi;
                        any_clipped = true;
                    }
                }
            }
        }
        if any_clipped { Some(clipped) } else { None }
    }

    /// Collect the set of training-column indices that are periodic axes —
    /// i.e. features for which a periodic basis (sphere longitude, periodic
    /// B-spline 1D, periodic tensor margin) must be allowed to take any
    /// real value at predict time and not be clamped to the training range.
    /// Returned indices reference `self.training_headers` (training-time
    /// layout), matching the iteration in `axis_clip_to_training_ranges`.
    fn training_periodic_axes(
        &self,
        training_headers: &[String],
    ) -> std::collections::HashSet<usize> {
        use crate::basis::BSplineKnotSpec;
        use crate::smooth::SmoothBasisSpec;
        let mut out: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let Some(spec) = self.resolved_termspec.as_ref() else {
            return out;
        };
        for term in &spec.smooth_terms {
            match &term.basis {
                // Sphere terms: longitude (second feature col) is always
                // periodic. Latitude is bounded and stays clamped.
                SmoothBasisSpec::Sphere { feature_cols, .. } => {
                    if let Some(&lon_col) = feature_cols.get(1) {
                        if lon_col < training_headers.len() {
                            out.insert(lon_col);
                        }
                    }
                }
                // 1D periodic B-spline: the single feature column is periodic.
                SmoothBasisSpec::BSpline1D { feature_col, spec } => {
                    if matches!(spec.knotspec, BSplineKnotSpec::PeriodicUniform { .. })
                        && *feature_col < training_headers.len()
                    {
                        out.insert(*feature_col);
                    }
                }
                // Tensor B-spline: each axis whose marginal knotspec is
                // PeriodicUniform is periodic; mark those columns.
                SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
                    for (i, marginal) in spec.marginalspecs.iter().enumerate() {
                        if matches!(marginal.knotspec, BSplineKnotSpec::PeriodicUniform { .. }) {
                            if let Some(&col) = feature_cols.get(i) {
                                if col < training_headers.len() {
                                    out.insert(col);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    pub fn from_payload(mut payload: FittedModelPayload) -> Self {
        let likelihood = payload.family_state.likelihood();
        let class = match payload.model_kind {
            ModelKind::Survival => PredictModelClass::Survival,
            ModelKind::MarginalSlope => PredictModelClass::BernoulliMarginalSlope,
            ModelKind::TransformationNormal => PredictModelClass::TransformationNormal,
            ModelKind::LocationScale => {
                if likelihood == LikelihoodFamily::GaussianIdentity {
                    PredictModelClass::GaussianLocationScale
                } else {
                    PredictModelClass::BinomialLocationScale
                }
            }
            _ => PredictModelClass::Standard,
        };
        match class {
            PredictModelClass::Survival => {
                payload.model_kind = ModelKind::Survival;
                Self::Survival { payload }
            }
            PredictModelClass::BernoulliMarginalSlope => {
                payload.model_kind = ModelKind::MarginalSlope;
                Self::MarginalSlope { payload }
            }
            PredictModelClass::TransformationNormal => {
                payload.model_kind = ModelKind::TransformationNormal;
                Self::TransformationNormal { payload }
            }
            PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
                payload.model_kind = ModelKind::LocationScale;
                Self::LocationScale { payload }
            }
            PredictModelClass::Standard => {
                payload.model_kind = ModelKind::Standard;
                Self::Standard { payload }
            }
        }
        .with_synchronized_stateful_link_metadata()
    }

    #[inline]
    pub fn payload(&self) -> &FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::MarginalSlope { payload }
            | Self::Survival { payload }
            | Self::TransformationNormal { payload } => payload,
        }
    }

    #[inline]
    fn payload_mut(&mut self) -> &mut FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::MarginalSlope { payload }
            | Self::Survival { payload }
            | Self::TransformationNormal { payload } => payload,
        }
    }

    fn with_synchronized_stateful_link_metadata(mut self) -> Self {
        self.synchronize_stateful_link_metadata();
        self
    }

    fn synchronize_stateful_link_metadata(&mut self) {
        let payload = self.payload_mut();
        let Some(fit) = payload.fit_result.as_ref() else {
            return;
        };
        match (&mut payload.family_state, &fit.fitted_link) {
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                    latent_cloglog_state,
                    ..
                },
                FittedLinkState::LatentCLogLog { state },
            ) => {
                *latent_cloglog_state = Some(*state);
            }
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialSas,
                    sas_state,
                    ..
                },
                FittedLinkState::Sas { state, covariance },
            ) => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialBetaLogistic,
                    sas_state,
                    ..
                },
                FittedLinkState::BetaLogistic { state, covariance },
            ) => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialMixture,
                    mixture_state,
                    ..
                },
                FittedLinkState::Mixture { state, covariance },
            ) => {
                *mixture_state = Some(state.clone());
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nestedvec);
            }
            _ => {}
        }
    }

    #[inline]
    pub fn likelihood(&self) -> LikelihoodFamily {
        self.payload().family_state.likelihood()
    }

    #[inline]
    pub fn predict_model_class(&self) -> PredictModelClass {
        match self.payload().family_state {
            FittedFamily::Survival { .. }
            | FittedFamily::LatentSurvival { .. }
            | FittedFamily::LatentBinary { .. } => PredictModelClass::Survival,
            FittedFamily::MarginalSlope { .. } => PredictModelClass::BernoulliMarginalSlope,
            FittedFamily::TransformationNormal { .. } => PredictModelClass::TransformationNormal,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::GaussianIdentity,
                ..
            } => PredictModelClass::GaussianLocationScale,
            FittedFamily::LocationScale { .. } => PredictModelClass::BinomialLocationScale,
            _ => PredictModelClass::Standard,
        }
    }

    pub fn saved_link_wiggle(&self) -> Result<Option<SavedLinkWiggleRuntime>, FittedModelError> {
        let payload = self.payload();
        let (knots, degree) = match (
            payload.linkwiggle_knots.as_ref(),
            payload.linkwiggle_degree,
        ) {
            (None, None) => return Ok(None),
            (Some(knots), Some(degree)) => (knots.clone(), degree),
            _ => {
                return Err(FittedModelError::SchemaMismatch {
                    reason:
                        "saved model has partial link-wiggle metadata; expected linkwiggle_knots and linkwiggle_degree together"
                            .to_string(),
                })
            }
        };
        let resolved_link = self.resolved_inverse_link()?;
        let saved_link_disallows_wiggle = resolved_link
            .as_ref()
            .is_some_and(|link| !inverse_link_supports_joint_wiggle(link))
            || payload
                .link
                .as_ref()
                .is_some_and(|link| !inverse_link_supports_joint_wiggle(link));
        if saved_link_disallows_wiggle {
            return Err(FittedModelError::IncompatibleConfig {
                reason: joint_wiggle_unsupported_link_message("link wiggle"),
            });
        }
        let beta = match self.predict_model_class() {
            PredictModelClass::Standard => {
                if payload.beta_link_wiggle.is_some() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason:
                            "standard link-wiggle coefficients must be stored in fit_result LinkWiggle block, not payload.beta_link_wiggle"
                                .to_string(),
                    });
                }
                let fit = payload.fit_result.as_ref().ok_or_else(|| {
                    FittedModelError::MissingField {
                        reason:
                            "standard link-wiggle model is missing canonical fit_result payload"
                                .to_string(),
                    }
                })?;
                if fit.blocks.len() != 2
                    || fit.blocks[0].role != BlockRole::Mean
                    || fit.blocks[1].role != BlockRole::LinkWiggle
                {
                    return Err(FittedModelError::SchemaMismatch {
                        reason:
                            "standard link-wiggle models must store blocks in [Mean, LinkWiggle] order"
                                .to_string(),
                    });
                }
                fit.block_by_role(BlockRole::LinkWiggle)
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason:
                            "standard link-wiggle model is missing LinkWiggle coefficient block"
                                .to_string(),
                    })?
                    .beta
                    .to_vec()
            }
            _ => payload
                .beta_link_wiggle
                .clone()
                .ok_or_else(|| FittedModelError::MissingField {
                    reason:
                        "saved model has link-wiggle metadata but is missing payload.beta_link_wiggle"
                            .to_string(),
                })?,
        };
        Ok(Some(SavedLinkWiggleRuntime {
            knots,
            degree,
            beta,
        }))
    }

    pub fn saved_baseline_time_wiggle(
        &self,
    ) -> Result<Option<SavedBaselineTimeWiggleRuntime>, FittedModelError> {
        let payload = self.payload();
        if payload
            .survival_cause_count
            .is_some_and(|cause_count| cause_count > 1)
            && payload.beta_baseline_timewiggle.is_none()
            && payload.beta_baseline_timewiggle_by_cause.is_some()
        {
            return Err(FittedModelError::SchemaMismatch {
                reason:
                    "joint cause-specific survival stores baseline-timewiggle coefficients per cause"
                        .to_string(),
            });
        }
        match (
            payload.baseline_timewiggle_knots.as_ref(),
            payload.baseline_timewiggle_degree,
            payload.baseline_timewiggle_penalty_orders.as_ref(),
            payload.baseline_timewiggle_double_penalty,
            payload.beta_baseline_timewiggle.as_ref(),
        ) {
            (None, None, None, None, None) => Ok(None),
            (Some(knots), Some(degree), Some(penalty_orders), Some(double_penalty), Some(beta)) => {
                Ok(Some(SavedBaselineTimeWiggleRuntime {
                    knots: knots.clone(),
                    degree,
                    penalty_orders: penalty_orders.clone(),
                    double_penalty,
                    beta: beta.clone(),
                }))
            }
            _ => Err(FittedModelError::SchemaMismatch {
                reason:
                    "saved model has partial baseline-timewiggle metadata; expected knots+degree+penalty_order+double_penalty+beta_baseline_timewiggle together"
                        .to_string(),
            }),
        }
    }

    /// Whether this model has a link wiggle component with complete metadata.
    #[inline]
    pub fn has_link_wiggle(&self) -> bool {
        self.saved_link_wiggle()
            .map(|runtime| runtime.is_some())
            .unwrap_or(false)
    }

    /// Whether this model has a baseline-time wiggle component with complete metadata.
    #[inline]
    pub fn has_baseline_time_wiggle(&self) -> bool {
        let payload = self.payload();
        if payload
            .survival_cause_count
            .is_some_and(|cause_count| cause_count > 1)
        {
            return payload.baseline_timewiggle_knots.is_some()
                && payload.baseline_timewiggle_degree.is_some()
                && payload.baseline_timewiggle_penalty_orders.is_some()
                && payload.baseline_timewiggle_double_penalty.is_some()
                && payload.beta_baseline_timewiggle_by_cause.is_some();
        }
        self.saved_baseline_time_wiggle()
            .map(|runtime| runtime.is_some())
            .unwrap_or(false)
    }

    pub fn saved_prediction_runtime(&self) -> Result<SavedPredictionRuntime, FittedModelError> {
        self.payload().validate_payload_version()?;
        if matches!(
            self.predict_model_class(),
            PredictModelClass::BernoulliMarginalSlope | PredictModelClass::Survival
        ) {
            if let Some(runtime) = self.payload().score_warp_runtime.as_ref() {
                runtime.validate_exact_replay_contract().map_err(|err| {
                    FittedModelError::PayloadCorrupt {
                        reason: format!("saved anchored score-warp runtime is invalid: {err}"),
                    }
                })?;
            }
            if let Some(runtime) = self.payload().link_deviation_runtime.as_ref() {
                runtime.validate_exact_replay_contract().map_err(|err| {
                    FittedModelError::PayloadCorrupt {
                        reason: format!("saved anchored link-deviation runtime is invalid: {err}"),
                    }
                })?;
            }
        }
        let runtime = SavedPredictionRuntime {
            model_class: self.predict_model_class(),
            likelihood: self.likelihood(),
            inverse_link: self.resolved_inverse_link()?,
            link_wiggle: self.saved_link_wiggle()?,
            baseline_time_wiggle: self.saved_baseline_time_wiggle()?,
            score_warp: self.payload().score_warp_runtime.clone(),
            link_deviation: self.payload().link_deviation_runtime.clone(),
            latent_z_rank_int_calibration: self.payload().latent_z_rank_int_calibration.clone(),
        };
        if matches!(
            runtime.model_class,
            PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale
        ) {
            let fit = self.payload().fit_result.as_ref().ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "location-scale model is missing canonical fit_result payload"
                        .to_string(),
                }
            })?;
            validate_location_scale_saved_fit(
                fit,
                runtime.model_class,
                runtime.link_wiggle.as_ref(),
            )?;
        } else if matches!(runtime.model_class, PredictModelClass::Survival)
            && self
                .payload()
                .survival_likelihood
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("location-scale"))
        {
            validate_survival_location_scale_saved_fit(
                self.payload(),
                runtime.link_wiggle.as_ref(),
            )?;
        } else if matches!(
            runtime.model_class,
            PredictModelClass::BernoulliMarginalSlope
        ) {
            let unified =
                self.payload()
                    .unified
                    .as_ref()
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason: "marginal-slope model is missing unified fit payload; refit"
                            .to_string(),
                    })?;
            validate_marginal_slope_saved_fit(
                unified,
                runtime.score_warp.as_ref(),
                runtime.link_deviation.as_ref(),
                "unified",
            )?;
        } else if matches!(runtime.model_class, PredictModelClass::Survival)
            && self
                .payload()
                .survival_likelihood
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("marginal-slope"))
        {
            let fit = self.payload().fit_result.as_ref().ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "survival marginal-slope model is missing canonical fit_result payload"
                        .to_string(),
                }
            })?;
            validate_survival_marginal_slope_saved_fit(
                fit,
                runtime.score_warp.as_ref(),
                runtime.link_deviation.as_ref(),
                "fit_result",
            )?;
        }
        Ok(runtime)
    }

    pub fn saved_sas_state(&self) -> Result<Option<SasLinkState>, FittedModelError> {
        let payload = self.payload();
        let raw = match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialSas,
                sas_state,
                ..
            } => (*sas_state).ok_or_else(|| FittedModelError::MissingField {
                reason: "binomial-sas model is missing state in family_state.sas_state".to_string(),
            })?,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialSas,
                base_link,
            } => match base_link {
                Some(InverseLink::Sas(state)) => *state,
                _ => {
                    return Err(FittedModelError::MissingField {
                        reason: "binomial-sas location-scale model is missing SAS base_link state"
                            .to_string(),
                    });
                }
            },
            _ => return Ok(None),
        };
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: raw.epsilon,
            initial_log_delta: raw.log_delta,
        })
        .map(Some)
        .map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("invalid saved SAS link state: {e}"),
        })
    }

    pub fn saved_beta_logistic_state(&self) -> Result<Option<SasLinkState>, FittedModelError> {
        let payload = self.payload();
        let raw = match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                sas_state,
                ..
            } => (*sas_state).ok_or_else(|| FittedModelError::MissingField {
                reason: "binomial-beta-logistic model is missing state in family_state.sas_state"
                    .to_string(),
            })?,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                base_link,
            } => match base_link {
                Some(InverseLink::BetaLogistic(state)) => *state,
                _ => {
                    return Err(FittedModelError::MissingField {
                        reason:
                            "binomial-beta-logistic location-scale model is missing beta-logistic base_link state"
                                .to_string(),
                    });
                }
            },
            _ => return Ok(None),
        };
        state_from_beta_logisticspec(SasLinkSpec {
            initial_epsilon: raw.epsilon,
            initial_log_delta: raw.log_delta,
        })
        .map(Some)
        .map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("invalid saved Beta-Logistic link state: {e}"),
        })
    }

    pub fn saved_mixture_state(&self) -> Result<Option<MixtureLinkState>, FittedModelError> {
        let payload = self.payload();
        match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialMixture,
                mixture_state,
                ..
            } => mixture_state
                .clone()
                .ok_or_else(|| FittedModelError::MissingField {
                    reason: "binomial-mixture model is missing state in family_state.mixture_state"
                        .to_string(),
                })
                .map(Some),
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialMixture,
                base_link,
            } => match base_link {
                Some(InverseLink::Mixture(state)) => Ok(Some(state.clone())),
                _ => Err(FittedModelError::MissingField {
                    reason:
                        "binomial-mixture location-scale model is missing mixture base_link state"
                            .to_string(),
                }),
            },
            _ => Ok(None),
        }
    }

    pub fn saved_latent_cloglog_state(
        &self,
    ) -> Result<Option<LatentCLogLogState>, FittedModelError> {
        let payload = self.payload();
        match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                latent_cloglog_state,
                ..
            } => latent_cloglog_state
                .ok_or_else(|| FittedModelError::MissingField {
                    reason:
                        "latent-cloglog-binomial model is missing state in family_state.latent_cloglog_state"
                            .to_string(),
                })
                .map(Some),
            _ => Ok(None),
        }
    }

    pub fn resolved_inverse_link(&self) -> Result<Option<InverseLink>, FittedModelError> {
        let stateful = if let Some(state) = self.saved_mixture_state()? {
            Some(InverseLink::Mixture(state))
        } else if let Some(state) = self.saved_latent_cloglog_state()? {
            Some(InverseLink::LatentCLogLog(state))
        } else if let Some(state) = self.saved_beta_logistic_state()? {
            Some(InverseLink::BetaLogistic(state))
        } else {
            self.saved_sas_state()?.map(InverseLink::Sas)
        };
        match &self.payload().family_state {
            FittedFamily::LocationScale { base_link, .. } => Ok(base_link.clone().or(stateful)),
            FittedFamily::Standard { link, .. } => {
                Ok(stateful.or_else(|| link.map(InverseLink::Standard)))
            }
            FittedFamily::MarginalSlope { base_link, .. } => Ok(base_link.clone()),
            FittedFamily::Survival { .. }
            | FittedFamily::LatentSurvival { .. }
            | FittedFamily::LatentBinary { .. } => Ok(None),
            FittedFamily::TransformationNormal { .. } => Ok(None),
        }
    }

    /// Build a validated predictor for the saved model shape and runtime.
    ///
    /// Survival callers build the time-basis design from saved metadata before
    /// handing prediction to the same trait-level machinery.
    pub fn predictor(&self) -> Option<Box<dyn PredictableModel>> {
        let runtime = self.saved_prediction_runtime().ok()?;
        match self.predict_model_class() {
            PredictModelClass::GaussianLocationScale => {
                let fit = self.fit_result.as_ref()?;
                let beta_mu = gaussian_location_scale_mean_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                let response_scale = self.gaussian_response_scale.unwrap_or(1.0);
                Some(Box::new(GaussianLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    response_scale,
                    covariance: fit.beta_covariance().cloned(),
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::Standard => {
                let family = self.family_state.likelihood();
                let link_kind = self.resolved_inverse_link().ok().flatten();
                let fit = self.fit_result.as_ref()?;
                let beta = if runtime.link_wiggle.is_some() {
                    fit.block_by_role(BlockRole::Mean)?.beta.clone()
                } else if let Some(unified) = self.unified() {
                    StandardPredictor::from_unified(unified, family, link_kind.clone(), None)
                        .ok()
                        .map(|p| p.beta)
                        .unwrap_or_else(|| fit.beta.clone())
                } else {
                    fit.beta.clone()
                };
                let covariance = fit.beta_covariance().cloned();
                Some(Box::new(StandardPredictor {
                    beta,
                    family,
                    link_kind,
                    covariance,
                    link_wiggle: runtime.link_wiggle,
                }))
            }
            PredictModelClass::Survival => {
                if matches!(
                    self.family_state,
                    FittedFamily::Survival {
                        survival_likelihood: Some(ref survival_likelihood),
                        ..
                    } if survival_likelihood == "marginal-slope"
                ) {
                    return None;
                }
                let unified = self.unified()?;
                // Default to probit inverse link for survival models.
                let inverse_link = self
                    .resolved_inverse_link()
                    .ok()
                    .flatten()
                    .unwrap_or(InverseLink::Standard(LinkFunction::Probit));
                SurvivalPredictor::from_unified(unified, inverse_link)
                    .ok()
                    .map(|p| Box::new(p) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BinomialLocationScale => {
                let inverse_link = self
                    .resolved_inverse_link()
                    .ok()
                    .flatten()
                    .unwrap_or(InverseLink::Standard(LinkFunction::Probit));
                let fit = self.fit_result.as_ref()?;
                let beta_threshold = binomial_location_scale_threshold_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                Some(Box::new(BinomialLocationScalePredictor {
                    beta_threshold,
                    beta_noise,
                    covariance: fit.beta_covariance().cloned(),
                    inverse_link,
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BernoulliMarginalSlope => {
                let unified = self.unified()?;
                let payload = self.payload();
                let z_column = payload.z_column.clone()?;
                let predictor = BernoulliMarginalSlopePredictor::from_unified(
                    unified,
                    z_column,
                    payload.latent_z_normalization?,
                    payload.latent_measure.clone()?,
                    payload.marginal_baseline?,
                    payload.logslope_baseline?,
                    self.resolved_inverse_link()
                        .ok()?
                        .unwrap_or(InverseLink::Standard(LinkFunction::Probit)),
                    self.family_state.frailty()?.clone(),
                    runtime.score_warp,
                    runtime.link_deviation,
                    runtime.latent_z_rank_int_calibration,
                )
                .ok()?;
                Some(Box::new(predictor) as Box<dyn PredictableModel>)
            }
            PredictModelClass::TransformationNormal => {
                // The h values are computed in build_predict_input_for_model
                // and stored in the offset field. The predictor is a simple
                // identity: eta = offset, mean = offset (h is already the
                // PIT-transformed value on the standard normal scale).
                let fit = self.fit_result.as_ref()?;
                Some(Box::new(super::predict::TransformationNormalPredictor {
                    covariance: fit.beta_covariance().cloned(),
                }) as Box<dyn PredictableModel>)
            }
        }
    }

    /// Returns the block roles for this model via the `PredictableModel` trait.
    ///
    /// For standard models this is `[BlockRole::Mean]`.
    pub fn block_roles(&self) -> Option<Vec<BlockRole>> {
        self.predictor().map(|p| p.block_roles())
    }

    /// Access the unified fit result, if stored.
    pub fn unified(&self) -> Option<&UnifiedFitResult> {
        self.payload().unified.as_ref()
    }

    pub fn load_from_path(path: &Path) -> Result<Self, FittedModelError> {
        let payload = fs::read_to_string(path).map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("failed to read model '{}': {e}", path.display()),
        })?;
        let model: Self =
            serde_json::from_str(&payload).map_err(|e| FittedModelError::PayloadCorrupt {
                reason: format!("failed to parse model json: {e}"),
            })?;
        let model = model.with_synchronized_stateful_link_metadata();
        model.validate_for_persistence()?;
        model.validate_numeric_finiteness()?;
        Ok(model)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), FittedModelError> {
        let normalized = self.clone().with_synchronized_stateful_link_metadata();
        normalized.validate_for_persistence()?;
        normalized.validate_numeric_finiteness()?;
        let file = fs::File::create(path).map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("failed to write model '{}': {e}", path.display()),
        })?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &normalized).map_err(|e| {
            FittedModelError::PayloadCorrupt {
                reason: format!("failed to serialize model: {e}"),
            }
        })?;
        std::io::Write::flush(&mut writer).map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("failed to write model '{}': {e}", path.display()),
        })?;
        Ok(())
    }

    pub fn require_data_schema(&self) -> Result<&DataSchema, FittedModelError> {
        self.data_schema
            .as_ref()
            .ok_or_else(|| FittedModelError::MissingField {
                reason: "model is missing data_schema; refit".to_string(),
            })
    }

    pub fn validate_for_persistence(&self) -> Result<(), FittedModelError> {
        // Hard version gate. The struct's ~40 Option<T> fields carry
        // `#[serde(default)]`, which is by design forward-compatible: old
        // payloads missing a new optional field decode with `None`. BUT:
        // when a new CLI release adds a required field for some family_state
        // (enforced below), an older model loaded by the newer CLI would have
        // `None` in that slot and the family-specific branch below would
        // correctly reject it — unless the new field also happens to slot
        // under a branch that hasn't been touched. Conversely, a newer model
        // loaded by an older CLI silently drops fields the older struct
        // doesn't know about. Both directions are silent-drift hazards. We
        // close them with an exact-version check anchored to the canonical
        // MODEL_PAYLOAD_VERSION constant — every payload must round-trip
        // identically between writers and readers running the same schema.
        self.validate_payload_version()?;
        if self.fit_result.is_none() {
            return Err(FittedModelError::MissingField {
                reason: "model is missing canonical fit_result payload; refit".to_string(),
            });
        }
        if self.data_schema.is_none() {
            return Err(FittedModelError::MissingField {
                reason: "model is missing data_schema; refit".to_string(),
            });
        }
        if self.training_headers.is_none() {
            return Err(FittedModelError::MissingField {
                reason: "model is missing training_headers; refit to guarantee stable feature mapping at prediction time"
                    .to_string(),
            });
        }
        let spec = self.resolved_termspec.as_ref().ok_or_else(|| {
            FittedModelError::MissingField {
                reason: "model is missing resolved_termspec; refit to guarantee train/predict design consistency"
                    .to_string(),
            }
        })?;
        validate_frozen_term_collectionspec(spec, "resolved_termspec")?;

        if self.formula_noise.is_some() && self.resolved_termspec_noise.is_none() {
            return Err(FittedModelError::MissingField {
                reason: "model defines formula_noise but is missing resolved_termspec_noise; refit"
                    .to_string(),
            });
        }
        if let Some(spec_noise) = self.resolved_termspec_noise.as_ref() {
            validate_frozen_term_collectionspec(spec_noise, "resolved_termspec_noise")?;
        }
        if matches!(self.family_state, FittedFamily::TransformationNormal { .. }) {
            let score = self.transformation_score_calibration.ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "transformation-normal model is missing transformation_score_calibration; refit"
                        .to_string(),
                }
            })?;
            score.validate("transformation-normal model")?;
        }
        if matches!(self.family_state, FittedFamily::MarginalSlope { .. }) {
            if self.formula_logslope.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "marginal-slope model is missing formula_logslope; refit".to_string(),
                });
            }
            if self.z_column.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "marginal-slope model is missing z_column; refit".to_string(),
                });
            }
            let z_normalization =
                self.latent_z_normalization
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason: "marginal-slope model is missing latent_z_normalization; refit"
                            .to_string(),
                    })?;
            z_normalization.validate("marginal-slope model")?;
            let latent_measure =
                self.latent_measure
                    .as_ref()
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason: "marginal-slope model is missing latent_measure; refit".to_string(),
                    })?;
            latent_measure
                .validate("marginal-slope model latent_measure")
                .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
            if self.marginal_baseline.is_none() || self.logslope_baseline.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "marginal-slope model is missing baseline offsets; refit".to_string(),
                });
            }
            if self.resolved_termspec_logslope.as_ref().is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "marginal-slope model is missing resolved_termspec_logslope for the logslope surface"
                        .to_string(),
                });
            }
            match self.family_state.frailty() {
                Some(FrailtySpec::None)
                | Some(FrailtySpec::GaussianShift {
                    sigma_fixed: Some(_),
                }) => {}
                Some(FrailtySpec::GaussianShift { sigma_fixed: None }) => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "marginal-slope model requires a fixed GaussianShift sigma in family_state.frailty"
                            .to_string(),
                    });
                }
                Some(FrailtySpec::HazardMultiplier { .. }) => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "marginal-slope model does not support HazardMultiplier frailty"
                            .to_string(),
                    });
                }
                None => {
                    return Err(FittedModelError::MissingField {
                        reason: "marginal-slope model is missing family_state.frailty; refit"
                            .to_string(),
                    });
                }
            }
        }

        if let FittedFamily::Survival {
            survival_likelihood,
            frailty,
            ..
        } = &self.family_state
        {
            if matches!(
                survival_likelihood.as_deref(),
                Some("latent") | Some("latent-binary")
            ) {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "latent hazard-window models must persist explicit family_state metadata, not generic survival metadata"
                        .to_string(),
                });
            }
            if survival_likelihood.as_deref() == Some("marginal-slope") {
                if self.formula_logslope.is_none() {
                    return Err(FittedModelError::MissingField {
                        reason: "survival marginal-slope model is missing formula_logslope; refit"
                            .to_string(),
                    });
                }
                if self.z_column.is_none() {
                    return Err(FittedModelError::MissingField {
                        reason: "survival marginal-slope model is missing z_column; refit"
                            .to_string(),
                    });
                }
                let z_normalization =
                    self.latent_z_normalization
                        .ok_or_else(|| {
                            FittedModelError::MissingField {
                        reason:
                            "survival marginal-slope model is missing latent_z_normalization; refit"
                                .to_string(),
                    }
                        })?;
                z_normalization.validate("survival marginal-slope model")?;
                let latent_measure =
                    self.latent_measure
                        .as_ref()
                        .ok_or_else(|| FittedModelError::MissingField {
                            reason:
                                "survival marginal-slope model is missing latent_measure; refit"
                                    .to_string(),
                        })?;
                latent_measure
                    .validate("survival marginal-slope model latent_measure")
                    .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
                if self.logslope_baseline.is_none() {
                    return Err(FittedModelError::MissingField {
                        reason: "survival marginal-slope model is missing logslope_baseline; refit"
                            .to_string(),
                    });
                }
                if self.resolved_termspec_logslope.as_ref().is_none() {
                    return Err(FittedModelError::MissingField {
                        reason: "survival marginal-slope model is missing resolved_termspec_logslope for the logslope surface"
                            .to_string(),
                    });
                }
                match frailty {
                    FrailtySpec::None
                    | FrailtySpec::GaussianShift {
                        sigma_fixed: Some(_),
                    } => {}
                    FrailtySpec::GaussianShift { sigma_fixed: None } => {
                        return Err(FittedModelError::IncompatibleConfig {
                            reason: "survival marginal-slope model requires a fixed GaussianShift sigma in family_state.frailty"
                                .to_string(),
                        });
                    }
                    FrailtySpec::HazardMultiplier { .. } => {
                        return Err(FittedModelError::IncompatibleConfig {
                            reason: "survival marginal-slope model does not support HazardMultiplier frailty"
                                .to_string(),
                        });
                    }
                }
            } else if !matches!(frailty, FrailtySpec::None) {
                return Err(FittedModelError::IncompatibleConfig {
                    reason:
                        "non-marginal survival models do not currently persist a frailty modifier"
                            .to_string(),
                });
            }
        }
        if let FittedFamily::LatentSurvival { frailty } = &self.family_state {
            match frailty {
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(_),
                    ..
                } => {}
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: None, ..
                } => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "latent survival model requires a fixed HazardMultiplier sigma in family_state.frailty"
                            .to_string(),
                    });
                }
                FrailtySpec::GaussianShift { .. } | FrailtySpec::None => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "latent survival model requires a fixed HazardMultiplier frailty specification"
                            .to_string(),
                    });
                }
            }
            if self.survival_likelihood.as_deref() != Some("latent") {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "latent survival model must persist survival_likelihood=latent"
                        .to_string(),
                });
            }
        }
        if let FittedFamily::LatentBinary { frailty } = &self.family_state {
            match frailty {
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(_),
                    ..
                } => {}
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: None, ..
                } => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "latent binary model requires a fixed HazardMultiplier sigma in family_state.frailty"
                            .to_string(),
                    });
                }
                FrailtySpec::GaussianShift { .. } | FrailtySpec::None => {
                    return Err(FittedModelError::IncompatibleConfig {
                        reason: "latent binary model requires a fixed HazardMultiplier frailty specification"
                            .to_string(),
                    });
                }
            }
            if self.survival_likelihood.as_deref() != Some("latent-binary") {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "latent binary model must persist survival_likelihood=latent-binary"
                        .to_string(),
                });
            }
        }

        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialSas,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialSas,
                ..
            }
        ) {
            self.saved_sas_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                ..
            }
        ) {
            self.saved_beta_logistic_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialMixture,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialMixture,
                ..
            }
        ) {
            self.saved_mixture_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                ..
            }
        ) {
            self.saved_latent_cloglog_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                ..
            }
        ) {
            return Err(FittedModelError::IncompatibleConfig {
                reason: "latent-cloglog-binomial is not supported for location-scale saved models"
                    .to_string(),
            });
        }
        if matches!(self.family_state, FittedFamily::Survival { .. })
            && self.survival_likelihood.is_none()
        {
            return Err(FittedModelError::MissingField {
                reason: "saved survival model is missing survival_likelihood metadata; refit"
                    .to_string(),
            });
        }
        if matches!(self.family_state, FittedFamily::Survival { .. })
            && self.survival_time_basis.is_none()
        {
            return Err(FittedModelError::MissingField {
                reason: "saved survival model is missing survival_time_basis metadata; refit"
                    .to_string(),
            });
        }
        if matches!(self.family_state, FittedFamily::Survival { .. })
            && self.survival_time_anchor.is_none()
        {
            // Pairs with the survival_time_basis check above: predict-side
            // code unconditionally requires the anchor too (see
            // `load_survival_time_basis_config_from_model`), so a model
            // serialized without it would fail at the first predict call.
            // Catching it here makes save fail fast — the same defence the
            // basis check provides for the basisname field.
            return Err(FittedModelError::MissingField {
                reason: "saved survival model is missing survival_time_anchor metadata; refit"
                    .to_string(),
            });
        }
        let has_any_saved_link_wiggle = self.linkwiggle_knots.is_some()
            || self.linkwiggle_degree.is_some()
            || self.beta_link_wiggle.is_some()
            || self
                .fit_result
                .as_ref()
                .and_then(|fit| fit.block_by_role(BlockRole::LinkWiggle))
                .is_some();
        let saved_link_wiggle = self.saved_link_wiggle()?;
        if has_any_saved_link_wiggle && saved_link_wiggle.is_none() {
            return Err(FittedModelError::SchemaMismatch {
                reason: "saved model has incomplete link-wiggle state; expected metadata and coefficients"
                    .to_string(),
            });
        }
        let has_any_saved_baseline_time_wiggle = self.baseline_timewiggle_knots.is_some()
            || self.baseline_timewiggle_degree.is_some()
            || self.baseline_timewiggle_penalty_orders.is_some()
            || self.baseline_timewiggle_double_penalty.is_some()
            || self.beta_baseline_timewiggle.is_some()
            || self.beta_baseline_timewiggle_by_cause.is_some();
        let is_joint_cause_specific = self
            .survival_cause_count
            .is_some_and(|cause_count| cause_count > 1);
        if has_any_saved_baseline_time_wiggle {
            if is_joint_cause_specific {
                let complete = self.baseline_timewiggle_knots.is_some()
                    && self.baseline_timewiggle_degree.is_some()
                    && self.baseline_timewiggle_penalty_orders.is_some()
                    && self.baseline_timewiggle_double_penalty.is_some()
                    && self.beta_baseline_timewiggle_by_cause.is_some();
                if !complete {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: "saved joint cause-specific survival model has incomplete baseline-timewiggle state; expected metadata and per-cause coefficients"
                            .to_string(),
                    });
                }
            } else if self.saved_baseline_time_wiggle()?.is_none() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "saved model has incomplete baseline-timewiggle state; expected metadata and coefficients"
                        .to_string(),
                });
            }
        }
        if self
            .survival_likelihood
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("location-scale"))
        {
            validate_survival_location_scale_saved_fit(self.payload(), saved_link_wiggle.as_ref())?;
        }

        // Validate anchored-deviation replay contracts at LOAD/SAVE time rather
        // than waiting for first predict call. Previously these contracts
        // (span table dimensions, coefficient matrices, etc.) were only
        // asserted inside `saved_prediction_runtime`, which runs on the first
        // predict invocation. A corrupted runtime would therefore pass
        // `load_from_path` silently and fail later under a different error
        // surface. Enforcing the same check here makes the model self-
        // diagnostic: `gam fit` catches its own bad output at save, and
        // `gam predict` catches bad input at load rather than mid-pipeline.
        if let Some(runtime) = self.score_warp_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|err| {
                FittedModelError::PayloadCorrupt {
                    reason: format!("saved anchored score-warp runtime is invalid: {err}"),
                }
            })?;
        }
        if let Some(runtime) = self.link_deviation_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|err| {
                FittedModelError::PayloadCorrupt {
                    reason: format!("saved anchored link-deviation runtime is invalid: {err}"),
                }
            })?;
        }
        if matches!(self.family_state, FittedFamily::MarginalSlope { .. }) {
            validate_marginal_slope_saved_fit(
                self.fit_result.as_ref().expect("checked above"),
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "fit_result",
            )?;
            let unified = self
                .unified
                .as_ref()
                .ok_or_else(|| FittedModelError::MissingField {
                    reason: "marginal-slope model is missing unified fit payload; refit"
                        .to_string(),
                })?;
            validate_marginal_slope_saved_fit(
                unified,
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "unified",
            )?;
        }
        if self
            .survival_likelihood
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("marginal-slope"))
        {
            validate_survival_marginal_slope_saved_fit(
                self.fit_result.as_ref().expect("checked above"),
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "fit_result",
            )?;
            if let Some(unified) = self.unified.as_ref() {
                validate_survival_marginal_slope_saved_fit(
                    unified,
                    self.score_warp_runtime.as_ref(),
                    self.link_deviation_runtime.as_ref(),
                    "unified",
                )?;
            }
        }

        // Posterior-mean / uncertainty backends are validated at predict time
        // by `prediction_backend_from_model`, which has access to the actual
        // requested mode and emits the canonical "nonlinear posterior-mean
        // prediction requires either covariance or a saved penalized Hessian"
        // error.  Save-time we deliberately do NOT enforce that gate: a fit
        // produced for MAP / plug-in scoring can be persisted and replayed
        // without ever needing a covariance backend, and gating it here would
        // refuse legitimate MAP-only saves whose `UnifiedFitResult` carries
        // beta + lambdas without a stabilized Hessian.

        Ok(())
    }

    pub fn validate_numeric_finiteness(&self) -> Result<(), FittedModelError> {
        let corrupt = |reason: String| FittedModelError::PayloadCorrupt { reason };
        if let Some(fit) = self.fit_result.as_ref() {
            fit.validate_numeric_finiteness()
                .map_err(|e| corrupt(e.to_string()))?;
        }

        for (name, opt) in [
            ("survival_baseline_scale", self.survival_baseline_scale),
            ("survival_baseline_shape", self.survival_baseline_shape),
            ("survival_baseline_rate", self.survival_baseline_rate),
            ("survival_baseline_makeham", self.survival_baseline_makeham),
            (
                "survival_time_smooth_lambda",
                self.survival_time_smooth_lambda,
            ),
            ("survival_time_anchor", self.survival_time_anchor),
            ("survivalridge_lambda", self.survivalridge_lambda),
        ] {
            if let Some(v) = opt {
                ensure_finite_scalar(name, v).map_err(corrupt)?;
            }
        }

        if let Some(v) = self.beta_noise.as_ref() {
            validate_all_finite("beta_noise", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.noise_projection.as_ref() {
            validate_all_finite("noise_projection", v.iter().flatten().copied())
                .map_err(corrupt)?;
        }
        if let Some(v) = self.noise_center.as_ref() {
            validate_all_finite("noise_center", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.noise_scale.as_ref() {
            validate_all_finite("noise_scale", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.noise_projection_ridge_alpha {
            ensure_finite_scalar("noise_projection_ridge_alpha", v).map_err(corrupt)?;
            if v < 0.0 {
                return Err(FittedModelError::InvalidInput {
                    reason: format!("noise_projection_ridge_alpha must be non-negative, got {v}"),
                });
            }
        }
        if let Some(v) = self.gaussian_response_scale {
            ensure_finite_scalar("gaussian_response_scale", v).map_err(corrupt)?;
        }
        if let Some(v) = self.beta_link_wiggle.as_ref() {
            validate_all_finite("beta_link_wiggle", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.beta_baseline_timewiggle.as_ref() {
            validate_all_finite("beta_baseline_timewiggle", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.beta_baseline_timewiggle_by_cause.as_ref() {
            validate_all_finite(
                "beta_baseline_timewiggle_by_cause",
                v.iter().flatten().copied(),
            )
            .map_err(corrupt)?;
        }
        if let Some(v) = self.latent_z_normalization {
            v.validate("latent_z_normalization")?;
        }
        if let Some(v) = self.latent_measure.as_ref() {
            v.validate("latent_measure").map_err(corrupt)?;
        }
        if let Some(v) = self.survival_beta_time.as_ref() {
            validate_all_finite("survival_beta_time", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.survival_beta_threshold.as_ref() {
            validate_all_finite("survival_beta_threshold", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.survival_beta_log_sigma.as_ref() {
            validate_all_finite("survival_beta_log_sigma", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.survival_noise_projection.as_ref() {
            validate_all_finite("survival_noise_projection", v.iter().flatten().copied())
                .map_err(corrupt)?;
        }
        if let Some(v) = self.survival_noise_center.as_ref() {
            validate_all_finite("survival_noise_center", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.survival_noise_projection_ridge_alpha {
            ensure_finite_scalar("survival_noise_projection_ridge_alpha", v).map_err(corrupt)?;
            if v < 0.0 {
                return Err(FittedModelError::InvalidInput {
                    reason: format!(
                        "survival_noise_projection_ridge_alpha must be non-negative, got {v}"
                    ),
                });
            }
        }
        if let Some(v) = self.survival_noise_scale.as_ref() {
            validate_all_finite("survival_noise_scale", v.iter().copied()).map_err(corrupt)?;
        }
        if let Some(v) = self.mixture_link_param_covariance.as_ref() {
            validate_all_finite("mixture_link_param_covariance", v.iter().flatten().copied())
                .map_err(corrupt)?;
        }
        if let Some(v) = self.sas_param_covariance.as_ref() {
            validate_all_finite("sas_param_covariance", v.iter().flatten().copied())
                .map_err(corrupt)?;
        }
        Ok(())
    }
}

fn array2_to_nestedvec(a: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|row| row.to_vec()).collect()
}

use crate::solver::estimate::{ensure_finite_scalar, validate_all_finite};

fn validate_frozen_term_collectionspec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), FittedModelError> {
    spec.validate_frozen(label)
        .map_err(|reason| FittedModelError::SchemaMismatch { reason })
}

impl Deref for FittedModel {
    type Target = FittedModelPayload;

    fn deref(&self) -> &Self::Target {
        self.payload()
    }
}

impl DerefMut for FittedModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload_mut()
    }
}

// ---------------------------------------------------------------------------
// Reconstruct library types from saved models
// ---------------------------------------------------------------------------

pub fn survival_baseline_config_from_model(
    model: &FittedModel,
) -> Result<SurvivalBaselineConfig, FittedModelError> {
    parse_survival_baseline_config(
        model
            .survival_baseline_target
            .as_deref()
            .unwrap_or("linear"),
        model.survival_baseline_scale,
        model.survival_baseline_shape,
        model.survival_baseline_rate,
        model.survival_baseline_makeham,
    )
    .map_err(|reason| FittedModelError::IncompatibleConfig { reason })
}

pub fn load_survival_time_basis_config_from_model(
    model: &FittedModel,
) -> Result<SurvivalTimeBasisConfig, FittedModelError> {
    match model
        .survival_time_basis
        .as_deref()
        .ok_or_else(|| FittedModelError::MissingField {
            reason: "saved survival model missing survival_time_basis".to_string(),
        })?
        .to_ascii_lowercase()
        .as_str()
    {
        "none" => Ok(SurvivalTimeBasisConfig::None),
        "linear" => Ok(SurvivalTimeBasisConfig::Linear),
        "bspline" => {
            let degree =
                model
                    .survival_time_degree
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason: "saved survival bspline model missing survival_time_degree"
                            .to_string(),
                    })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "saved survival bspline model missing survival_time_knots".to_string(),
                }
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "saved survival bspline time basis metadata is invalid".to_string(),
                });
            }
            Ok(SurvivalTimeBasisConfig::BSpline {
                degree,
                knots: Array1::from_vec(knots),
                smooth_lambda,
            })
        }
        "ispline" => {
            let degree =
                model
                    .survival_time_degree
                    .ok_or_else(|| FittedModelError::MissingField {
                        reason: "saved survival ispline model missing survival_time_degree"
                            .to_string(),
                    })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "saved survival ispline model missing survival_time_knots".to_string(),
                }
            })?;
            let keep_cols = model.survival_time_keep_cols.clone().ok_or_else(|| {
                FittedModelError::MissingField {
                    reason: "saved survival ispline model missing survival_time_keep_cols"
                        .to_string(),
                }
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() || keep_cols.is_empty() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "saved survival ispline time basis metadata is invalid".to_string(),
                });
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: Array1::from_vec(knots),
                keep_cols,
                smooth_lambda,
            })
        }
        other => Err(FittedModelError::IncompatibleConfig {
            reason: format!("unsupported saved survival_time_basis '{other}'"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL;
    use crate::families::lognormal_kernel::FrailtySpec;
    use crate::pirls::PirlsStatus;
    use crate::solver::estimate::{FitArtifacts, FittedBlock, FittedLinkState};
    use crate::types::{LikelihoodScaleMetadata, LogLikelihoodNormalization};
    use ndarray::{Array1, Array2, array};

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn anchored_runtime(basis_dim: usize) -> SavedAnchoredDeviationRuntime {
        SavedAnchoredDeviationRuntime {
            kernel: ANCHORED_DEVIATION_KERNEL.to_string(),
            breakpoints: vec![-1.0, 1.0],
            basis_dim,
            span_c0: vec![vec![0.0; basis_dim]],
            span_c1: vec![vec![0.0; basis_dim]],
            span_c2: vec![vec![0.0; basis_dim]],
            span_c3: vec![vec![0.0; basis_dim]],
            anchor_residual_coefficients: None,
            anchor_residual_components: Vec::new(),
            anchor_residual_rotation: None,
        }
    }

    fn saved_fit(blocks: Vec<FittedBlock>) -> UnifiedFitResult {
        let beta = Array1::from_vec(
            blocks
                .iter()
                .flat_map(|block| block.beta.iter().copied())
                .collect(),
        );
        let p = beta.len();
        UnifiedFitResult {
            blocks,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(LikelihoodFamily::BinomialProbit),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
            standard_deviation: 1.0,
            covariance_conditional: Some(Array2::zeros((p, p))),
            covariance_corrected: Some(Array2::zeros((p, p))),
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: vec![],
            beta,
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                survival_link_wiggle_knots: None,
                survival_link_wiggle_degree: None,
            },
            inner_cycles: 0,
        }
    }

    fn marginal_slope_payload(version: u32, fit: UnifiedFitResult) -> FittedModelPayload {
        let mut payload = FittedModelPayload::new(
            version,
            "y ~ 1".to_string(),
            ModelKind::MarginalSlope,
            FittedFamily::MarginalSlope {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
                frailty: FrailtySpec::None,
            },
            "bernoulli-marginal-slope".to_string(),
        );
        payload.fit_result = Some(fit.clone());
        payload.unified = Some(fit);
        payload.data_schema = Some(DataSchema {
            columns: vec![SchemaColumn {
                name: "z".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        });
        payload.set_training_feature_metadata(vec!["z".to_string()], vec![(0.0, 0.0)]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_logslope = Some(empty_termspec());
        payload.formula_logslope = Some("1".to_string());
        payload.z_column = Some("z".to_string());
        payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
        payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
        payload.marginal_baseline = Some(0.0);
        payload.logslope_baseline = Some(0.0);
        payload.link = Some(InverseLink::Standard(LinkFunction::Probit));
        payload
    }

    fn survival_marginal_slope_payload(version: u32, fit: UnifiedFitResult) -> FittedModelPayload {
        let mut payload = FittedModelPayload::new(
            version,
            "Surv(entry, exit, event) ~ 1".to_string(),
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some(ResidualDistribution::Gaussian),
                frailty: FrailtySpec::None,
            },
            "survival".to_string(),
        );
        payload.fit_result = Some(fit.clone());
        payload.unified = Some(fit);
        payload.survival_likelihood = Some("marginal-slope".to_string());
        payload.survival_distribution = Some(ResidualDistribution::Gaussian);
        payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
        payload.data_schema = Some(DataSchema {
            columns: vec![SchemaColumn {
                name: "z".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        });
        payload.set_training_feature_metadata(vec!["z".to_string()], vec![(0.0, 0.0)]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_logslope = Some(empty_termspec());
        payload.formula_logslope = Some("1".to_string());
        payload.z_column = Some("z".to_string());
        payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
        payload.logslope_baseline = Some(0.0);
        payload.link = Some(InverseLink::Standard(LinkFunction::Probit));
        payload
    }

    #[test]
    fn validate_for_persistence_rejects_marginal_slope_score_warp_basis_mismatch() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);
        payload.score_warp_runtime = Some(anchored_runtime(2));

        let err = FittedModel::from_payload(payload)
            .validate_for_persistence()
            .expect_err("marginal-slope score-warp basis mismatch should fail validation");
        assert!(err.to_string().contains("score-warp coefficient mismatch"));
    }

    #[test]
    fn saved_prediction_runtime_rejects_survival_marginal_slope_link_basis_mismatch() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.4],
                role: BlockRole::LinkWiggle,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = survival_marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);
        payload.link_deviation_runtime = Some(anchored_runtime(2));

        let err = FittedModel::from_payload(payload)
            .saved_prediction_runtime()
            .expect_err(
                "survival marginal-slope link basis mismatch should fail runtime validation",
            );
        assert!(
            err.to_string()
                .contains("link-deviation coefficient mismatch")
        );
    }

    #[test]
    fn apply_survival_time_basis_writes_all_required_fields() {
        use crate::families::survival_construction::SavedSurvivalTimeBasis;

        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = survival_marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);

        // Snapshot writes must match every persisted survival_time_* field —
        // forgetting one is exactly the gamfit 0.1.69 marginal-slope save
        // regression. Routing through `apply_survival_time_basis` is the
        // structural contract that prevents that recurrence.
        let snapshot = SavedSurvivalTimeBasis {
            basisname: "royston-parmar".to_string(),
            degree: Some(3),
            knots: Some(vec![0.0, 1.0, 2.0]),
            keep_cols: Some(vec![0, 2]),
            smooth_lambda: Some(0.5),
            anchor: 0.25,
        };
        payload.apply_survival_time_basis(&snapshot);

        assert_eq!(
            payload.survival_time_basis.as_deref(),
            Some("royston-parmar")
        );
        assert_eq!(payload.survival_time_degree, Some(3));
        assert_eq!(payload.survival_time_knots, Some(vec![0.0, 1.0, 2.0]));
        assert_eq!(payload.survival_time_keep_cols, Some(vec![0, 2]));
        assert_eq!(payload.survival_time_smooth_lambda, Some(0.5));
        assert_eq!(payload.survival_time_anchor, Some(0.25));
    }

    #[test]
    fn validate_for_persistence_rejects_survival_without_time_anchor_metadata() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = survival_marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);
        // Pass the time_basis presence check but deliberately omit the
        // anchor — this is exactly the partial-write shape that the CLI's
        // marginal-slope+time-wiggle save path had before the structural
        // refactor (main.rs previously set basis/degree/knots/keep_cols/
        // smooth_lambda but forgot the anchor).
        payload.survival_time_basis = Some("ispline".to_string());

        let err = FittedModel::from_payload(payload)
            .validate_for_persistence()
            .expect_err("survival model without time-anchor metadata should fail validation");
        assert!(err.to_string().contains("missing survival_time_anchor"));
    }

    #[test]
    fn validate_for_persistence_rejects_survival_without_time_basis_metadata() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let payload = survival_marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);

        let err = FittedModel::from_payload(payload)
            .validate_for_persistence()
            .expect_err("survival model without time-basis metadata should fail validation");
        assert!(err.to_string().contains("missing survival_time_basis"));
    }

    #[test]
    fn saved_prediction_runtime_rejects_stale_payload_version() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let payload = marginal_slope_payload(MODEL_PAYLOAD_VERSION - 1, fit);

        let err = FittedModel::from_payload(payload)
            .saved_prediction_runtime()
            .expect_err("stale payload version should fail before runtime assembly");
        assert!(err.to_string().contains("payload schema mismatch"));
    }
}
