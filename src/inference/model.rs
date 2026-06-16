use crate::basis::BasisOptions;
use crate::estimate::{BlockRole, FittedLinkState, UnifiedFitResult};
use crate::families::bms::{
    LatentMeasureKind, LatentZConditionalCalibration, LatentZRankIntCalibration,
};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalTimeBasisConfig, parse_survival_baseline_config,
};
use crate::families::survival_location_scale::ResidualDistribution;
use crate::families::wiggle::{
    monotone_wiggle_basis_with_derivative_order, validate_monotone_wiggle_beta_nonnegative,
};
use crate::inference::formula_dsl::{
    inverse_link_supports_joint_wiggle, joint_wiggle_unsupported_link_message, parse_formula,
    parse_surv_interval_response, parse_surv_response, parsed_term_column_names,
};
use crate::inference::predict::{
    BernoulliMarginalSlopePredictor, BinomialLocationScalePredictor,
    DispersionLocationScalePredictor, GaussianLocationScalePredictor, PredictableModel,
    StandardPredictor, SurvivalPredictor,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use crate::smooth::{AdaptiveRegularizationDiagnostics, TermCollectionSpec};
use crate::util::span::span_index_for_breakpoints;
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, MixtureLinkState, ResponseFamily, SasLinkSpec,
    SasLinkState, StandardLink,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
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
pub const MODEL_PAYLOAD_VERSION: u32 = 7;

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

/// Saved exact spline-scan fit (#1030/#1034): the predict-time feature column
/// plus the lossless smoother state the Gaussian-bridge `predict` replays.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedSplineScan {
    /// Training column name feeding the single 1-D smooth at predict time.
    pub feature_column: String,
    pub state: crate::solver::spline_scan::SplineScanState,
}

/// Saved multiresolution residual-cascade fit (#1032): the predict-time feature
/// columns (d ∈ {2, 3}) plus the serializable cascade state that `from_state`
/// rebuilds a predict-capable `ResidualCascadeFit` from. The cascade is a
/// DIFFERENT posterior from the dense Duchon/Matérn term — never a silent swap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedResidualCascade {
    /// Training column names for the d ∈ {2, 3} scattered-smooth coordinates.
    pub feature_columns: Vec<String>,
    pub state: crate::solver::residual_cascade::ResidualCascadeState,
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

// Boundary conversions so external `Result<_, EstimationError>` /
// `Result<_, SurvivalPredictError>` call sites can propagate with `?`.
// Survival prediction keeps the model-layer source so the chain identifies
// the payload/schema failure that triggered the prediction error.
impl From<FittedModelError> for crate::solver::estimate::EstimationError {
    fn from(err: FittedModelError) -> Self {
        crate::solver::estimate::EstimationError::InvalidInput(err.to_string())
    }
}

impl From<FittedModelError> for crate::families::survival_predict::SurvivalPredictError {
    fn from(err: FittedModelError) -> Self {
        crate::families::survival_predict::SurvivalPredictError::ModelPayload {
            context: "saved-model survival prediction payload",
            source: err,
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
        Ok::<(), _>(())
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
#[derive(Default)]
pub enum TransformationScoreKind {
    #[default]
    FiniteSupportPit,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TransformationScoreCalibration {
    #[serde(default)]
    pub score_kind: TransformationScoreKind,
    #[serde(default = "default_transformation_score_pit_clip_eps")]
    pub clip_eps: f64,
}

const fn default_transformation_score_pit_clip_eps() -> f64 {
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
        Ok::<(), _>(())
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
    /// Exact O(n) spline-scan fit representation (#1030/#1034): the
    /// state-space smoothing-spline posterior of a single 1-D Gaussian
    /// smooth. When `Some`, this standard Gaussian model's predictions
    /// replay the Gaussian bridge from this state and the model carries no
    /// dense `fit_result` (the representations are mutually exclusive —
    /// enforced by `validate_for_persistence`). `#[serde(default)]` so older
    /// payloads read as: not a scan model.
    #[serde(default)]
    pub spline_scan: Option<SavedSplineScan>,
    /// O(n log n) multiresolution residual-cascade fit (#1032): the persisted
    /// multilevel Wendland-frame state for a single scattered 2–3D Gaussian
    /// smooth past the dense-kernel cliff. When `Some`, predictions replay the
    /// cascade posterior; mutually exclusive with `spline_scan`/`fit_result`.
    /// `#[serde(default)]` keeps forward-compatibility with older payloads.
    #[serde(default)]
    pub residual_cascade: Option<SavedResidualCascade>,
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
    /// to fit-time projection.
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
    /// Optional conditional location-scale calibration of the latent score
    /// (#905, BMS family). When `Some`, the marginal-slope predictor replaces
    /// the (normalized) input `z` by `ζ = (z − m(C))/√v(C)` — rebuilding the
    /// conditioning span `a(C)` from the marginal prediction design — before
    /// the closed-form standard-normal kernel, matching fit-time semantics.
    /// Mutually exclusive with `latent_z_rank_int_calibration`. `#[serde(default)]`
    /// so pre-existing models deserialize cleanly (interpreted as: no
    /// conditional calibration).
    #[serde(default)]
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    #[serde(default)]
    pub marginal_baseline: Option<f64>,
    #[serde(default)]
    pub logslope_baseline: Option<f64>,
    #[serde(default)]
    pub logslope_baselines: Option<Vec<f64>>,
    #[serde(default)]
    pub score_warp_runtime: Option<SavedCompiledFlexBlock>,
    #[serde(default)]
    pub link_deviation_runtime: Option<SavedCompiledFlexBlock>,
    /// Width `p₁` of the survival marginal-slope absorbed Stage-1 influence block
    /// (#461) when present (the dedicated trailing absorber block). Predict drops
    /// its `γ`; this records the column count so the predictor can account for
    /// the extra block and slice `γ` out of the joint covariance.
    #[serde(default)]
    pub influence_absorber_width: Option<usize>,
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
    /// used when fitting the survival log-sigma projection.
    #[serde(default)]
    pub survival_noise_projection_ridge_alpha: Option<f64>,
    #[serde(default)]
    pub survival_distribution: Option<ResidualDistribution>,
    #[serde(default)]
    pub training_headers: Option<Vec<String>>,
    /// Container type of the table the model was fitted on, as detected by the
    /// Python binding (`"pandas"`, `"polars"`, `"pyarrow"`, `"numpy"`, or an
    /// ambiguous tag such as `"unknown"`). This is presentation provenance, not
    /// math: `gamfit.Model.predict` uses it as the output-container fallback
    /// when the *prediction input* is itself container-ambiguous (a `dict` of
    /// columns or a `list` of record dicts). Persisting it in the model payload
    /// makes the fallback survive `save`/`load` and `dumps`/`loads`, so a
    /// reloaded model predicts into the same container as the in-memory one.
    /// `None` for older payloads (and for fits that never recorded a kind), in
    /// which case the fallback degrades to `"dict"`, matching pre-persistence
    /// behaviour for unknown-kind training tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_table_kind: Option<String>,
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
    /// Precomputed exact Gaussian-identity jackknife+ statistics (#942).
    ///
    /// Populated *only* for a standard Gaussian-identity model fit with unit
    /// prior weights, where the closed-form Sherman–Morrison leave-one-out
    /// substrate gives a distribution-free finite-sample (≥ level) prediction
    /// interval with no held-out fold. When `Some`, `predict(interval=level)`
    /// auto-routes through it (the MAGIC default); when `None` — any other
    /// family/link, reweighted rows, or an older payload — predict falls back
    /// to the model-based posterior band and labels the provenance honestly.
    /// `#[serde(default)]` so pre-existing models deserialize as: no jackknife+
    /// substrate available.
    #[serde(default)]
    pub gaussian_jackknife_plus:
        Option<crate::inference::full_conformal::GaussianJackknifePlusStats>,
    /// Precomputed substrate for the EXACT Gaussian-identity full-conformal set
    /// (#942 Layer 1 + the frozen-ρ self-diagnostic).
    ///
    /// Populated under the SAME eligibility as `gaussian_jackknife_plus`
    /// (Gaussian-identity, unit prior weights, offset-free, no link wiggle). It
    /// persists the training design + response + frozen penalty `Sλ` so the
    /// distribution-free EXACT prediction set (a union of intervals, valid for
    /// any penalized smooth) can be replayed per test point — one Cholesky each,
    /// zero refits — and surfaces the frozen-ρ certificate flag. `None` for any
    /// ineligible model or an older payload, in which case the exact-set predict
    /// path errors with a clear message and the caller uses jackknife+ or the
    /// posterior band. `#[serde(default)]` so pre-existing models deserialize as
    /// no exact substrate available.
    #[serde(default)]
    pub full_conformal: Option<crate::inference::full_conformal::ExactFullConformalSubstrate>,
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
            spline_scan: None,
            residual_cascade: None,
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
            latent_z_conditional_calibration: None,
            marginal_baseline: None,
            logslope_baseline: None,
            logslope_baselines: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            influence_absorber_width: None,
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
            training_table_kind: None,
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
            gaussian_jackknife_plus: None,
            full_conformal: None,
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
    /// marginal-slope save→load bug was a builder that
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
        likelihood: LikelihoodSpec,
        #[serde(default)]
        link: Option<StandardLink>,
        #[serde(default)]
        latent_cloglog_state: Option<LatentCLogLogState>,
        #[serde(default)]
        mixture_state: Option<MixtureLinkState>,
        #[serde(default)]
        sas_state: Option<SasLinkState>,
    },
    LocationScale {
        likelihood: LikelihoodSpec,
        #[serde(default)]
        base_link: Option<InverseLink>,
    },
    MarginalSlope {
        likelihood: LikelihoodSpec,
        #[serde(default)]
        base_link: Option<InverseLink>,
        frailty: FrailtySpec,
    },
    Survival {
        likelihood: LikelihoodSpec,
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
        likelihood: LikelihoodSpec,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictModelClass {
    Standard,
    GaussianLocationScale,
    BinomialLocationScale,
    /// Genuine-dispersion location-scale (#913): NegativeBinomial / Gamma / Beta
    /// / Tweedie mean families fitted with a `noise_formula` overdispersion
    /// channel. Predicted through the GLM mean inverse link (not the binomial
    /// threshold-scale predictor).
    DispersionLocationScale,
    BernoulliMarginalSlope,
    Survival,
    TransformationNormal,
}

impl PredictModelClass {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::GaussianLocationScale => "gaussian location-scale",
            Self::BinomialLocationScale => "binomial location-scale",
            Self::DispersionLocationScale => "dispersion location-scale",
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
pub use crate::families::bms::deviation_runtime::ParametricAnchorBlock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedCompiledFlexBlock {
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
    /// `anchor_components`).
    #[serde(default)]
    pub anchor_correction: Option<Vec<Vec<f64>>>,
    /// Ordered list of parametric anchor components whose stacked row
    /// values combine into `n_row`. Empty unless
    /// `anchor_correction` is `Some`.
    #[serde(default)]
    pub anchor_components: Vec<SavedAnchorComponent>,
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
    /// Flex-evaluation anchor (sibling flex block's reparameterised basis,
    /// evaluated at training rows at fit time and at predict rows at
    /// predict time). The predictor stacks `ncols` columns from the
    /// sibling runtime's `design(arg)` into `n_row`.
    FlexEvaluation { ncols: usize },
}

#[derive(Clone, Debug)]
pub struct SavedPredictionRuntime {
    pub model_class: PredictModelClass,
    pub likelihood: LikelihoodSpec,
    pub inverse_link: Option<InverseLink>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
    pub baseline_time_wiggle: Option<SavedBaselineTimeWiggleRuntime>,
    pub score_warp: Option<SavedCompiledFlexBlock>,
    pub link_deviation: Option<SavedCompiledFlexBlock>,
    /// Rank-INT latent-z calibration carried into the predictor build.
    /// `None` for non-BMS models and for BMS fits whose latent measure
    /// did not require rank-INT calibration.
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    /// Conditional location-scale latent-z calibration (#905) carried into the
    /// predictor build. `None` for non-BMS models and for BMS fits whose Auto
    /// path did not detect a conditional `E[z|C]`/`Var(z|C)` shift. When
    /// `Some`, the predictor replaces the normalized `z` by `ζ = (z−m(C))/√v(C)`
    /// using the marginal prediction design as the conditioning span.
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the
    /// survival marginal-slope fit hosted a dedicated additive absorber (the
    /// trailing block). `None` when no CTN Stage-1 chain produced an influence
    /// Jacobian. At predict the absorber's `γ` is DROPPED (the orthogonalized
    /// β̂ is a training-fit property), so the predictor uses this width only to
    /// (a) account for the extra trailing block in the saved block count and
    /// (b) slice `γ`'s rows/cols out of the joint covariance. Survival hosts the
    /// absorber as its own block (unlike the BMS A2 widened-marginal design),
    /// so it never widens any persisted prediction design.
    pub influence_absorber_width: Option<usize>,
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

/// Whether a `ModelKind::LocationScale` likelihood's response is one of the
/// genuine-dispersion mean families (#913) — NegativeBinomial, Gamma, Beta or
/// Tweedie. These carry a `noise_formula` overdispersion channel and must be
/// predicted through the GLM mean inverse link (the
/// [`PredictModelClass::DispersionLocationScale`] path), NOT the binomial
/// threshold-scale predictor. The binomial location-scale (BMS ordinal) path is
/// the only other non-Gaussian location-scale family, with a `Binomial`
/// response.
fn is_dispersion_location_scale_response(response: &crate::types::ResponseFamily) -> bool {
    use crate::types::ResponseFamily;
    matches!(
        response,
        ResponseFamily::NegativeBinomial { .. }
            | ResponseFamily::Gamma
            | ResponseFamily::Beta { .. }
            | ResponseFamily::Tweedie { .. }
    )
}

fn validate_location_scale_saved_fit(
    fit: &UnifiedFitResult,
    model_class: PredictModelClass,
    link_wiggle: Option<&SavedLinkWiggleRuntime>,
) -> Result<(), FittedModelError> {
    let primary = match model_class {
        // Gaussian and dispersion (#913) location-scale both predict the mean
        // through the Location block; the binomial threshold-scale class reads
        // the Threshold block instead.
        PredictModelClass::GaussianLocationScale | PredictModelClass::DispersionLocationScale => {
            gaussian_location_scale_mean_beta(fit)
        }
        PredictModelClass::BinomialLocationScale => binomial_location_scale_threshold_beta(fit),
        _ => None,
    }
    .ok_or_else(|| FittedModelError::MissingField {
        reason: match model_class {
            PredictModelClass::GaussianLocationScale => {
                "gaussian-location-scale saved fit is missing mean/location block".to_string()
            }
            PredictModelClass::DispersionLocationScale => {
                "dispersion-location-scale saved fit is missing mean/location block".to_string()
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
    score_warp: Option<&SavedCompiledFlexBlock>,
    link_deviation: Option<&SavedCompiledFlexBlock>,
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
    score_warp: Option<&SavedCompiledFlexBlock>,
    link_deviation: Option<&SavedCompiledFlexBlock>,
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
    score_warp: Option<&SavedCompiledFlexBlock>,
    link_deviation: Option<&SavedCompiledFlexBlock>,
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

impl SavedCompiledFlexBlock {
    pub(crate) fn validate_exact_replay_contract(&self) -> Result<(), FittedModelError> {
        if self.kernel.is_empty() {
            return Err(FittedModelError::SchemaMismatch {
                reason: "saved anchored deviation runtime is missing the exact kernel marker"
                    .to_string(),
            });
        }
        if self.kernel != crate::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL {
            return Err(FittedModelError::IncompatibleConfig {
                reason: format!(
                    "saved anchored deviation runtime uses unsupported kernel '{}'; expected {}",
                    self.kernel,
                    crate::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL
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
        let coeffs = match self.anchor_correction.as_ref() {
            Some(c) => c,
            None => {
                if !self.anchor_components.is_empty() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason:
                            "saved anchored deviation runtime has anchor_components but no anchor_correction"
                                .to_string(),
                    });
                }
                return Ok(());
            }
        };
        let d: usize = self
            .anchor_components
            .iter()
            .map(|c| match &c.kind {
                SavedAnchorKind::Parametric { ncols, .. } => *ncols,
                SavedAnchorKind::FlexEvaluation { ncols } => *ncols,
            })
            .sum();
        if coeffs.len() != d {
            return Err(FittedModelError::SchemaMismatch {
                reason: format!(
                    "saved anchored deviation runtime anchor_correction has {} rows; expected {} (sum of component ncols)",
                    coeffs.len(),
                    d,
                ),
            });
        }
        for (i, row) in coeffs.iter().enumerate() {
            if row.len() != self.basis_dim {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "saved anchored deviation runtime anchor_correction row {} has width {}, expected basis_dim {}",
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
                            "saved anchored deviation runtime anchor_correction ({i},{j}) is non-finite"
                        ),
                    });
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
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
        Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
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
        })
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
        Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
            left: points[span_idx],
            right: points[span_idx + 1],
            c0: self.span_c0[span_idx][basis_idx],
            c1: self.span_c1[span_idx][basis_idx],
            c2: self.span_c2[span_idx][basis_idx],
            c3: self.span_c3[span_idx][basis_idx],
        })
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
            return Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
                left: left_ep,
                right: left_ep + 1.0,
                c0: self.span_c0[0][basis_idx],
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        if value > right_ep {
            return Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
                left: right_ep,
                right: right_ep + 1.0,
                c0: self.right_boundary_basis_value(basis_idx),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.basis_span_cubic_validated(span_idx, basis_idx)
    }

    pub fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<crate::families::cubic_cell_kernel::LocalSpanCubic, FittedModelError> {
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
            return Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
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
            });
        }
        if value > right_ep {
            return Ok(crate::families::cubic_cell_kernel::LocalSpanCubic {
                left: right_ep,
                right: right_ep + 1.0,
                c0: (0..self.basis_dim)
                    .map(|basis_idx| self.right_boundary_basis_value(basis_idx) * beta[basis_idx])
                    .sum(),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
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
    /// and `d == sum of anchor_components ncols`. Each row holds
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
        if let Some(m_rows) = self.anchor_correction.as_ref() {
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
                            "design_with_anchor_rows: anchor_correction row {} has length {}, expected basis_dim {}",
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
            // The compiler bakes the orthonormalising rotation into M, so
            // the predict-time subtraction is simply `n_anchor_rows · M`.
            let subtract = anchor_rows.dot(&m_dense);
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
        let Some(m_rows) = self.anchor_correction.as_ref() else {
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
        // The compiler bakes the orthonormalising rotation into M, so
        // the predict-time correction is simply `n_anchor_rows · M`.
        Ok(Some(n_anchor_rows.dot(&m_dense)))
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
}

impl FittedFamily {
    #[inline]
    pub fn likelihood(&self) -> LikelihoodSpec {
        let spec = match self {
            Self::Standard { likelihood, .. }
            | Self::LocationScale { likelihood, .. }
            | Self::MarginalSlope { likelihood, .. }
            | Self::Survival { likelihood, .. }
            | Self::TransformationNormal { likelihood, .. } => likelihood,
            Self::LatentSurvival { .. } | Self::LatentBinary { .. } => {
                return LikelihoodSpec::royston_parmar();
            }
        };
        spec.clone()
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

/// Recursively collect the feature columns of a smooth basis whose out-of-hull
/// evaluation is bounded, so they can be exempted from the predict-time axis
/// clip (see [`FittedModel::training_smooth_extrapolation_axes`]). Wrapper bases
/// (`by=`, factor-smooth, sum-to-zero) delegate to their inner smooth; `Sphere`
/// and `Pca` are intentionally not collected.
fn collect_smooth_extrapolation_axes(
    basis: &crate::smooth::SmoothBasisSpec,
    n_training_headers: usize,
    out: &mut std::collections::HashSet<usize>,
) {
    use crate::smooth::SmoothBasisSpec;
    let push = |col: usize, out: &mut std::collections::HashSet<usize>| {
        if col < n_training_headers {
            out.insert(col);
        }
    };
    match basis {
        // 1D B-spline: first-order linear extension off the boundary slope.
        SmoothBasisSpec::BSpline1D { feature_col, .. } => push(*feature_col, out),
        // Tensor B-spline: each margin linear-extends independently. Periodic
        // margins are additionally (and harmlessly) exempted via the periodic set.
        SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            for &c in feature_cols {
                push(c, out);
            }
        }
        // Radial bases with a bounded out-of-hull contract: Duchon / thin-plate
        // are linear outside the data span (natural-spline boundary conditions),
        // Matérn reverts to its mean as the kernel decays. Measure-jet shares
        // the Matérn contract (Gaussian representers decay to the parametric
        // layer off the data support) — and off-web queries are exactly the
        // ones its support diagnostic must see unclipped.
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. } => {
            for &c in feature_cols {
                push(c, out);
            }
        }
        // Factor-smooth: the continuous marginal axes are B-splines that
        // linear-extend; the group column is categorical and is left to the
        // random-effect / level-lookup machinery.
        SmoothBasisSpec::FactorSmooth { spec } => {
            for &c in &spec.continuous_cols {
                push(c, out);
            }
        }
        // Wrappers delegate to the inner smooth they modulate / replicate.
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            collect_smooth_extrapolation_axes(inner, n_training_headers, out)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            collect_smooth_extrapolation_axes(smooth, n_training_headers, out)
        }
        // Sphere: latitude is clipped to manifold bounds, longitude is periodic —
        // both handled elsewhere with non-plateau semantics. Pca: no extrapolation
        // contract, stays clipped. ConstantCurvature: chart coordinates must stay
        // inside the κ-stereographic chart (open ball for κ < 0), so clipping new
        // data to the training range is the safe out-of-hull behavior.
        SmoothBasisSpec::Sphere { .. }
        | SmoothBasisSpec::ConstantCurvature { .. }
        | SmoothBasisSpec::Pca { .. } => {}
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
        // Parametric/linear-term axes must never be clipped either: a linear
        // term's contract is η = β0 + β1·x, i.e. genuine linear extrapolation.
        // Clamping its input to the training extreme turns predict into a
        // piecewise-constant plateau outside the training hull and freezes the
        // prediction SE at the boundary (the clamped x feeds xᵀ Var(β) x), so
        // credible intervals stop widening with distance from the data. This
        // mirrors how periodic axes are exempted just above.
        let linear_axes = self.training_linear_axes(training_headers.len());
        // Random-effect grouping axes are categorical even when their source
        // column is numeric. Clipping them would remap an unseen group label to
        // a boundary training level instead of letting the random-effect block
        // encode it as the prior-mean zero effect.
        let random_effect_axes = self.training_random_effect_axes(training_headers.len());
        // Non-parametric smooth axes whose basis extrapolates boundedly on its
        // own (B-spline linear extension, Duchon/thin-plate natural-spline linear
        // tail, Matérn kernel decay). Clamping their input to the training extreme
        // hands the basis an already-clamped coordinate, so its extrapolation
        // never fires and predict freezes at a boundary plateau — diverging from
        // the raw design path, which does not clip. Exempt them so both paths go
        // through the single basis-layer extrapolation. See the method doc.
        let smooth_extrapolation_axes =
            self.training_smooth_extrapolation_axes(training_headers.len());
        // Sphere latitude is a closed-manifold coordinate: its clip bounds are
        // the manifold's intrinsic domain ([-π/2, π/2] or [-90, 90]), not the
        // sampled range, so a pole prediction reaches the true pole instead of
        // being clamped to a near-pole latitude (see the method doc).
        let sphere_lat_bounds = self.training_sphere_latitude_bounds(training_headers);
        let mut clipped = data.to_owned();
        let mut any_clipped = false;
        for (col_in_training, (header, &(lo, hi))) in
            training_headers.iter().zip(ranges.iter()).enumerate()
        {
            let (lo, hi) = sphere_lat_bounds
                .get(&col_in_training)
                .copied()
                .unwrap_or((lo, hi));
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
            if linear_axes.contains(&col_in_training) {
                continue;
            }
            if random_effect_axes.contains(&col_in_training) {
                continue;
            }
            if smooth_extrapolation_axes.contains(&col_in_training) {
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

    fn saved_term_specs(&self) -> Vec<&TermCollectionSpec> {
        let mut specs: Vec<&TermCollectionSpec> = [
            self.resolved_termspec.as_ref(),
            self.resolved_termspec_noise.as_ref(),
            self.resolved_termspec_logslope.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect();
        if let Some(logslopes) = self.resolved_termspec_logslopes.as_ref() {
            specs.extend(logslopes.iter());
        }
        specs
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
                // periodic and exempt from clipping. Latitude is not periodic
                // but is a closed-manifold coordinate, so it is clipped to the
                // manifold's intrinsic bounds rather than the sampled range —
                // see `training_sphere_latitude_bounds`.
                SmoothBasisSpec::Sphere { feature_cols, .. } => {
                    if let Some(&lon_col) = feature_cols.get(1)
                        && lon_col < training_headers.len()
                    {
                        out.insert(lon_col);
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
                        if matches!(marginal.knotspec, BSplineKnotSpec::PeriodicUniform { .. })
                            && let Some(&col) = feature_cols.get(i)
                            && col < training_headers.len()
                        {
                            out.insert(col);
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Collect the set of training-column indices that feed a parametric/linear
    /// term — on *any* modelled surface (mean, noise/scale, log-slope). A
    /// linear term realises the design column `∏ feature_cols` and contributes
    /// `β·(that product)` to the linear predictor, so its inputs must be allowed
    /// to take any real value at predict time: clamping them to the training
    /// range would replace genuine linear extrapolation with a boundary plateau
    /// (and freeze the prediction SE at the hull edge). Returned indices
    /// reference `self.training_headers` (training-time layout), matching the
    /// iteration in `axis_clip_to_training_ranges`. Wilkinson-Rogers `:`
    /// interactions contribute every column in their `feature_cols` product.
    fn training_linear_axes(&self, n_training_headers: usize) -> std::collections::HashSet<usize> {
        let mut out: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for spec in self.saved_term_specs() {
            for term in &spec.linear_terms {
                for col in term.effective_feature_cols() {
                    if col < n_training_headers {
                        out.insert(col);
                    }
                }
            }
        }
        out
    }

    /// Collect the set of training-column indices that feed random-effect
    /// grouping terms. These columns are categorical model axes regardless of
    /// the ingest schema's scalar storage type, so prediction must leave them
    /// untouched and let the frozen random-effect levels decide whether a row is
    /// a seen level or the zero-effect unseen-level fallback.
    fn training_random_effect_axes(
        &self,
        n_training_headers: usize,
    ) -> std::collections::HashSet<usize> {
        let mut out: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for spec in self.saved_term_specs() {
            for term in &spec.random_effect_terms {
                if term.feature_col < n_training_headers {
                    out.insert(term.feature_col);
                }
            }
        }
        out
    }

    /// Collect the set of training-column indices that feed a non-parametric
    /// smooth whose basis performs its own *bounded* extrapolation outside the
    /// training hull — on any modelled surface (mean, noise/scale, log-slope).
    ///
    /// These columns must be exempt from the predict-time axis clip for the same
    /// reason periodic/linear/random-effect axes are: the clip clamps a new value
    /// to the training extreme *before* the design is built, so the basis is
    /// handed an already-clamped coordinate and its extrapolation machinery never
    /// fires. The result is a piecewise-constant plateau frozen at the boundary
    /// fitted value (with a prediction SE frozen at the hull edge), and — worse —
    /// a model that yields *different* predictions through the `FittedModel`
    /// predict pipeline than through the raw `build_term_collection_design` path,
    /// which does not clip. Exempting these axes routes both entry points through
    /// the single basis-layer extrapolation, restoring internal consistency.
    ///
    /// Only bases with a *bounded* out-of-hull contract are listed, so removing
    /// the clip cannot reintroduce the wild basis blow-up the clip guards against:
    ///   - B-spline 1D / tensor margins: first-order linear extension off the
    ///     boundary slope (`apply_linear_extension_from_first_derivative`) — grows
    ///     at most linearly.
    ///   - Duchon / thin-plate: the natural-spline boundary conditions make the
    ///     fit linear outside the data span — also at most linear growth.
    ///   - Matérn: the kernel decays with distance, so the fit reverts smoothly to
    ///     its (low-order polynomial / constant) mean far from the data — bounded.
    /// `sphere()` axes are deliberately *not* listed: latitude is a closed-manifold
    /// coordinate clipped to its intrinsic bounds (`training_sphere_latitude_bounds`)
    /// and longitude is periodic (`training_periodic_axes`); both already have the
    /// correct, non-plateau handling. `Pca` projections have no extrapolation
    /// contract and stay clipped.
    ///
    /// Returned indices reference `self.training_headers` (training-time layout),
    /// matching the iteration in `axis_clip_to_training_ranges`.
    fn training_smooth_extrapolation_axes(
        &self,
        n_training_headers: usize,
    ) -> std::collections::HashSet<usize> {
        let mut out: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for spec in self.saved_term_specs() {
            for term in &spec.smooth_terms {
                collect_smooth_extrapolation_axes(&term.basis, n_training_headers, &mut out);
            }
        }
        out
    }

    /// Manifold-intrinsic clip bounds for sphere *latitude* columns.
    ///
    /// A `sphere(lat, lon)` smooth charts the closed manifold S²: the poles
    /// (lat = ±π/2, or ±90°) are interior limit points of that manifold, not
    /// endpoints of an unbounded axis to extrapolate along. A finite training
    /// sample never reaches the pole exactly, so clamping a pole prediction to
    /// the *observed* latitude extreme lands at a near-pole latitude where the
    /// Wahba SOS kernel's `cos(lat)·cos(lat_c)·cos(Δlon)` term has not yet
    /// damped — and the predictor then sweeps a spurious `cos(lon)` profile at
    /// what is physically a single point, reintroducing the pole artefact the
    /// SOS basis exists to remove.
    ///
    /// The correct clip bound for this coordinate is therefore the manifold's
    /// intrinsic domain — `[-π/2, π/2]` radians or `[-90, 90]` degrees — not
    /// the sampled range. Clamping to those bounds keeps the pole reachable
    /// (single-valued in longitude) while still mapping any out-of-domain
    /// latitude onto the manifold boundary. Longitude needs no entry here: it
    /// is periodic and already exempted from clipping entirely
    /// (`training_periodic_axes`). Returned indices reference
    /// `self.training_headers`, matching the iteration in
    /// `axis_clip_to_training_ranges`.
    fn training_sphere_latitude_bounds(
        &self,
        training_headers: &[String],
    ) -> std::collections::HashMap<usize, (f64, f64)> {
        use crate::smooth::SmoothBasisSpec;
        let mut out: std::collections::HashMap<usize, (f64, f64)> =
            std::collections::HashMap::new();
        let Some(spec) = self.resolved_termspec.as_ref() else {
            return out;
        };
        for term in &spec.smooth_terms {
            if let SmoothBasisSpec::Sphere { feature_cols, spec } = &term.basis
                && let Some(&lat_col) = feature_cols.first()
                && lat_col < training_headers.len()
            {
                let bound = if spec.radians {
                    std::f64::consts::FRAC_PI_2
                } else {
                    90.0
                };
                out.insert(lat_col, (-bound, bound));
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
                if likelihood == LikelihoodSpec::gaussian_identity() {
                    PredictModelClass::GaussianLocationScale
                } else if is_dispersion_location_scale_response(&likelihood.response) {
                    PredictModelClass::DispersionLocationScale
                } else {
                    PredictModelClass::BinomialLocationScale
                }
            }
            ModelKind::Standard => PredictModelClass::Standard,
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
            PredictModelClass::GaussianLocationScale
            | PredictModelClass::BinomialLocationScale
            | PredictModelClass::DispersionLocationScale => {
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
                    likelihood,
                    latent_cloglog_state,
                    ..
                },
                FittedLinkState::LatentCLogLog { state },
            ) if likelihood.is_latent_cloglog() => {
                *latent_cloglog_state = Some(*state);
            }
            (
                FittedFamily::Standard {
                    likelihood,
                    sas_state,
                    ..
                },
                FittedLinkState::Sas { state, covariance },
            ) if likelihood.is_binomial_sas() => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood,
                    sas_state,
                    ..
                },
                FittedLinkState::BetaLogistic { state, covariance },
            ) if likelihood.is_binomial_beta_logistic() => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood,
                    mixture_state,
                    ..
                },
                FittedLinkState::Mixture { state, covariance },
            ) if likelihood.is_binomial_mixture() => {
                *mixture_state = Some(state.clone());
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nestedvec);
            }
            _ => {}
        }
    }

    #[inline]
    pub fn likelihood(&self) -> LikelihoodSpec {
        self.payload().family_state.likelihood()
    }

    /// Columns this model consumes from a prediction frame — its *input
    /// contract*.
    ///
    /// Every variable named by the main formula (features, interaction margins,
    /// random-effect groups, and a smooth's `by=` column), the survival
    /// entry/exit columns or the transformation-normal response, the auxiliary
    /// noise / logslope formula columns, and the offset / noise-offset /
    /// latent-`z` columns. The event-indicator and the plain response of a
    /// standard model are deliberately excluded: they are not needed to *form*
    /// a prediction (the conformal-calibration fold layers the response back on
    /// separately).
    ///
    /// This is the single authority shared by the CLI and PyFFI predict paths.
    /// A prediction frame column that is *not* in this set is irrelevant to the
    /// model and must be ignored rather than strict-encoded against the
    /// training schema — otherwise an unrelated ID/label column with a held-out
    /// categorical level aborts predict (#840).
    pub fn prediction_required_columns(
        &self,
    ) -> Result<std::collections::BTreeSet<String>, String> {
        let payload = self.payload();
        let parsed = parse_formula(payload.formula.as_str()).map_err(|e| e.to_string())?;
        let mut required = std::collections::BTreeSet::<String>::new();
        parsed_term_column_names(&parsed.terms, &mut required);

        if let Some((entry, exit, _event)) =
            parse_surv_response(parsed.response.as_str()).map_err(|e| e.to_string())?
        {
            if let Some(entry) = entry {
                required.insert(entry);
            }
            required.insert(exit);
        } else if let Some((left, right, _event)) =
            parse_surv_interval_response(parsed.response.as_str()).map_err(|e| e.to_string())?
        {
            required.insert(left);
            required.insert(right);
        } else if matches!(
            self.predict_model_class(),
            PredictModelClass::TransformationNormal
        ) {
            let response = parsed.response.trim();
            if !response.is_empty() && !response.starts_with("Surv(") {
                required.insert(response.to_string());
            }
        }

        if let Some(offset) = payload.offset_column.as_ref() {
            required.insert(offset.clone());
        }
        if let Some(noise_offset) = payload.noise_offset_column.as_ref() {
            required.insert(noise_offset.clone());
        }
        if matches!(
            self.predict_model_class(),
            PredictModelClass::BernoulliMarginalSlope | PredictModelClass::Survival
        ) {
            if let Some(z_column) = payload.z_column.as_ref() {
                required.remove("z");
                required.insert(z_column.clone());
            }
        }
        if let Some(noise_formula) = payload.formula_noise.as_ref() {
            self.add_auxiliary_formula_columns(
                &mut required,
                noise_formula,
                parsed.response.as_str(),
            )?;
        }
        if let Some(logslope_formula) = payload.formula_logslope.as_ref() {
            if logslope_formula != "same-as-main" {
                self.add_auxiliary_formula_columns(
                    &mut required,
                    logslope_formula,
                    parsed.response.as_str(),
                )?;
            }
        }
        Ok(required)
    }

    /// Columns a *post-fit diagnostic* command (diagnose / sample / report)
    /// needs **beyond** [`Self::prediction_required_columns`].
    ///
    /// Prediction deliberately drops a standard GAM's bare response so a
    /// prediction frame may omit it (#840 / #864). Diagnostics are statements
    /// *about* that observed response — residuals, R², posterior likelihoods,
    /// leave-one-out — so the response must be present. This returns the bare
    /// response column when the prediction projection would otherwise drop it,
    /// and nothing when the response is already prediction-required (survival
    /// `Surv(...)` time/event columns, the transformation-normal response) or
    /// is not a plain data column.
    ///
    /// Centralising the intent here is what makes it *structurally impossible*
    /// for a diagnostic command to silently drop the response: callers use
    /// `load_dataset…_for_diagnostics`, which always folds these in, instead of
    /// each remembering to thread an `extra_required` response by hand.
    pub fn diagnostic_extra_columns(&self) -> Result<Vec<String>, String> {
        let payload = self.payload();
        let parsed = parse_formula(payload.formula.as_str()).map_err(|e| e.to_string())?;
        // Survival responses are `Surv(...)` expressions, not bare columns; the
        // underlying entry/exit columns are already prediction-required.
        if parse_surv_response(parsed.response.as_str())
            .map_err(|e| e.to_string())?
            .is_some()
            || parse_surv_interval_response(parsed.response.as_str())
                .map_err(|e| e.to_string())?
                .is_some()
        {
            return Ok(Vec::new());
        }
        let response = parsed.response.trim();
        // A response that is empty, or a function-call expression rather than a
        // plain data column, has no bare column to re-add.
        if response.is_empty() || response.contains('(') {
            return Ok(Vec::new());
        }
        // Already prediction-required (e.g. transformation-normal re-adds it):
        // nothing extra to fold in.
        if self.prediction_required_columns()?.contains(response) {
            return Ok(Vec::new());
        }
        Ok(vec![response.to_string()])
    }

    /// Add the columns referenced by an auxiliary (noise / logslope) formula,
    /// which may be supplied as a full `lhs ~ rhs` formula or as a bare RHS.
    fn add_auxiliary_formula_columns(
        &self,
        required: &mut std::collections::BTreeSet<String>,
        formula_or_rhs: &str,
        response: &str,
    ) -> Result<(), String> {
        let trimmed = formula_or_rhs.trim();
        if trimmed.is_empty() || trimmed == "1" {
            return Ok(());
        }
        let formula = if trimmed.contains('~') {
            trimmed.to_string()
        } else {
            format!("{response} ~ {trimmed}")
        };
        let parsed = parse_formula(formula.as_str()).map_err(|e| e.to_string())?;
        parsed_term_column_names(&parsed.terms, required);
        Ok(())
    }

    #[inline]
    pub fn predict_model_class(&self) -> PredictModelClass {
        match &self.payload().family_state {
            FittedFamily::Survival { .. }
            | FittedFamily::LatentSurvival { .. }
            | FittedFamily::LatentBinary { .. } => PredictModelClass::Survival,
            FittedFamily::MarginalSlope { .. } => PredictModelClass::BernoulliMarginalSlope,
            FittedFamily::TransformationNormal { .. } => PredictModelClass::TransformationNormal,
            FittedFamily::LocationScale { likelihood, .. } if likelihood.is_gaussian_identity() => {
                PredictModelClass::GaussianLocationScale
            }
            FittedFamily::LocationScale { likelihood, .. }
                if is_dispersion_location_scale_response(&likelihood.response) =>
            {
                PredictModelClass::DispersionLocationScale
            }
            FittedFamily::LocationScale { .. } => PredictModelClass::BinomialLocationScale,
            FittedFamily::Standard { .. } => PredictModelClass::Standard,
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

    /// Whether the default point prediction must integrate the inverse link
    /// over the coefficient posterior — reporting the posterior mean
    /// `E[g⁻¹(Xβ)]` — rather than plugging in the posterior mode `g⁻¹(Xβ̂)`.
    ///
    /// SPEC (issue #960): the posterior mean is *always* the default point
    /// estimate (never MAP). It is observably distinct from the plug-in exactly
    /// when the inverse link is *curved* over the posterior's uncertainty, so
    /// `E[g⁻¹(η)] ≠ g⁻¹(E[η])` by Jensen. The curvature-based classification is:
    ///   * all log-link families (Poisson / Gamma / Tweedie / NegativeBinomial):
    ///     `E[exp η] = exp(η + se²/2) ≠ exp(η)` (log-normal MGF);
    ///   * all Binomial links (logit / probit / cloglog / SAS / BetaLogistic /
    ///     Mixture / LatentCLogLog): bounded sigmoidal inverse links;
    ///   * Beta (logit link): `E[σ(η)] ≠ σ(E[η])`;
    ///   * Royston–Parmar (curved survival-probability inverse link).
    /// The integral collapses to the plug-in (so the cheaper plug-in path is
    /// exact and taken instead) only for the effectively-linear identity-link
    /// Gaussian. Any model carrying a link wiggle or baseline-time wiggle is
    /// curved regardless of family. This curvature partition mirrors
    /// `families::family_runtime::posterior_mean`, the compute path that produces the
    /// corrected mean for each of these families.
    ///
    /// This is the single source of truth shared by the CLI (`gam predict`)
    /// and the Python FFI prediction path so the two can never drift on which
    /// models receive the posterior-mean correction.
    #[inline]
    pub fn prediction_uses_posterior_mean(&self) -> bool {
        let family = self.likelihood();
        let curved_family = match &family.response {
            // Identity-link Gaussian: inverse link is linear, so the posterior
            // mean equals the plug-in and the cheaper exact path is taken.
            ResponseFamily::Gaussian => false,
            // Log-link families: E[exp η] = exp(η + se²/2) ≠ exp(η).
            ResponseFamily::Poisson
            | ResponseFamily::Gamma
            | ResponseFamily::Tweedie { .. }
            | ResponseFamily::NegativeBinomial { .. } => true,
            // Beta (logit link): E[σ(η)] ≠ σ(E[η]).
            ResponseFamily::Beta { .. } => true,
            // Royston–Parmar: curved survival-probability inverse link.
            ResponseFamily::RoystonParmar => true,
            // Binomial: every link variant (logit / probit / cloglog / SAS /
            // BetaLogistic / Mixture / LatentCLogLog) is a curved sigmoid.
            ResponseFamily::Binomial => matches!(
                &family.link,
                InverseLink::Standard(_)
                    | InverseLink::Sas(_)
                    | InverseLink::BetaLogistic(_)
                    | InverseLink::Mixture(_)
                    | InverseLink::LatentCLogLog(_)
            ),
        };
        curved_family || self.has_link_wiggle() || self.has_baseline_time_wiggle()
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
            latent_z_conditional_calibration: self
                .payload()
                .latent_z_conditional_calibration
                .clone(),
            influence_absorber_width: self.payload().influence_absorber_width,
        };
        if matches!(
            runtime.model_class,
            PredictModelClass::GaussianLocationScale
                | PredictModelClass::BinomialLocationScale
                | PredictModelClass::DispersionLocationScale
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
                likelihood,
                sas_state,
                ..
            } if likelihood.is_binomial_sas() => {
                (*sas_state).ok_or_else(|| FittedModelError::MissingField {
                    reason: "binomial-sas model is missing state in family_state.sas_state"
                        .to_string(),
                })?
            }
            FittedFamily::LocationScale {
                likelihood,
                base_link,
            } if likelihood.is_binomial_sas() => match base_link {
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
                likelihood,
                sas_state,
                ..
            } if likelihood.is_binomial_beta_logistic() => {
                (*sas_state).ok_or_else(|| FittedModelError::MissingField {
                    reason:
                        "binomial-beta-logistic model is missing state in family_state.sas_state"
                            .to_string(),
                })?
            }
            FittedFamily::LocationScale {
                likelihood,
                base_link,
            } if likelihood.is_binomial_beta_logistic() => match base_link {
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
                likelihood,
                mixture_state,
                ..
            } if likelihood.is_binomial_mixture() => mixture_state
                .clone()
                .ok_or_else(|| FittedModelError::MissingField {
                    reason: "binomial-mixture model is missing state in family_state.mixture_state"
                        .to_string(),
                })
                .map(Some),
            FittedFamily::LocationScale {
                likelihood,
                base_link,
            } if likelihood.is_binomial_mixture() => match base_link {
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
                likelihood,
                latent_cloglog_state,
                ..
            } if likelihood.is_latent_cloglog() => latent_cloglog_state
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
                // The log-σ coefficients were mapped to raw response units by
                // shifting the intercept by `+ln(response_scale)`, which scales
                // only the `exp(η)` term. The σ floor must be scaled separately
                // so reconstructed σ is response-scale-equivariant (#884): use
                // `LOGB_SIGMA_FLOOR · response_scale` (≈ 1 % of the response
                // spread). A model fitted without standardization persists
                // `gaussian_response_scale = 1`, recovering the raw floor.
                let response_scale = self.payload().gaussian_response_scale.unwrap_or(1.0);
                let sigma_floor = crate::families::sigma_link::LOGB_SIGMA_FLOOR * response_scale;
                Some(Box::new(GaussianLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    sigma_floor,
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
                    StandardPredictor::from_unified(
                        unified,
                        family.clone(),
                        link_kind.clone(),
                        None,
                    )
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
                    .unwrap_or(InverseLink::Standard(StandardLink::Probit));
                SurvivalPredictor::from_unified(unified, inverse_link)
                    .ok()
                    .map(|p| Box::new(p) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BinomialLocationScale => {
                let inverse_link = self
                    .resolved_inverse_link()
                    .ok()
                    .flatten()
                    .unwrap_or(InverseLink::Standard(StandardLink::Probit));
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
            PredictModelClass::DispersionLocationScale => {
                let fit = self.fit_result.as_ref()?;
                // The mean prediction routes through the family's GLM inverse
                // link (log for NB/Gamma/Tweedie, logit for Beta); the
                // log-precision block feeds the overdispersion / predictive-SD
                // channel — never the binomial threshold-scale predictor.
                let beta_mu = gaussian_location_scale_mean_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                let inverse_link = self.resolved_inverse_link().ok().flatten();
                Some(Box::new(DispersionLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    likelihood: self.family_state.likelihood(),
                    inverse_link,
                    covariance: fit.beta_covariance().cloned(),
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
                        .unwrap_or(InverseLink::Standard(StandardLink::Probit)),
                    self.family_state.frailty()?.clone(),
                    runtime.score_warp,
                    runtime.link_deviation,
                    runtime.latent_z_rank_int_calibration,
                    runtime.latent_z_conditional_calibration,
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

    /// Concrete bernoulli marginal-slope predictor with explicit error
    /// surfacing. `predictor()` boxes the same object behind the
    /// `PredictableModel` trait and swallows construction failures into
    /// `None`; the posterior predictive path (#1049) needs the concrete type
    /// (for `final_eta_from_theta` / `theta_len`) and propagatable error
    /// messages, so it builds the predictor here instead.
    pub fn bernoulli_marginal_slope_predictor(
        &self,
    ) -> Result<BernoulliMarginalSlopePredictor, String> {
        if !matches!(
            self.predict_model_class(),
            PredictModelClass::BernoulliMarginalSlope
        ) {
            return Err(format!(
                "bernoulli_marginal_slope_predictor: model is not a bernoulli marginal-slope \
                 model (class {:?})",
                self.predict_model_class()
            ));
        }
        let runtime = self
            .saved_prediction_runtime()
            .map_err(|err| format!("bernoulli marginal-slope predictor runtime: {err}"))?;
        let unified = self.unified().ok_or_else(|| {
            "bernoulli marginal-slope predictor requires a unified fit".to_string()
        })?;
        let payload = self.payload();
        let z_column = payload.z_column.clone().ok_or_else(|| {
            "bernoulli marginal-slope predictor requires a saved z column".to_string()
        })?;
        BernoulliMarginalSlopePredictor::from_unified(
            unified,
            z_column,
            payload.latent_z_normalization.ok_or_else(|| {
                "marginal-slope predictor requires saved latent-z normalization".to_string()
            })?,
            payload.latent_measure.clone().ok_or_else(|| {
                "marginal-slope predictor requires a saved latent measure".to_string()
            })?,
            payload.marginal_baseline.ok_or_else(|| {
                "marginal-slope predictor requires a saved marginal baseline".to_string()
            })?,
            payload.logslope_baseline.ok_or_else(|| {
                "marginal-slope predictor requires a saved logslope baseline".to_string()
            })?,
            self.resolved_inverse_link()
                .map_err(|err| format!("marginal-slope predictor inverse link: {err}"))?
                .unwrap_or(InverseLink::Standard(StandardLink::Probit)),
            self.family_state
                .frailty()
                .ok_or_else(|| {
                    "marginal-slope predictor requires a saved frailty spec".to_string()
                })?
                .clone(),
            runtime.score_warp,
            runtime.link_deviation,
            runtime.latent_z_rank_int_calibration,
            runtime.latent_z_conditional_calibration,
        )
    }

    /// V∞ §5 coverage floor for the measure-jet extrapolation variance: a
    /// band level "covers" a query once its kernel mass reaches this fraction
    /// of that level's web-averaged support. Magic-by-default (no dial):
    /// 0.05 keeps the ε★ gate's bounded discontinuity at ≤ 5 % of the
    /// spectrum's total prior ignorance (see the monotonicity theorem in
    /// `terms/basis/measure_jet_predict.rs`) while still refusing credit
    /// for stray sub-floor kernel mass at levels finer than the first
    /// covering scale.
    const MEASURE_JET_COVERAGE_FLOOR: f64 = 0.05;

    /// V∞ §5 producer: per-row measure-jet extrapolation variance on the η
    /// scale for a prediction batch (`docs/measure_jet_v_infinity.md`).
    ///
    /// For every frozen measure-jet term in `resolved_termspec` this prices
    /// the off-support ignorance of the fitted multiscale spectrum at each
    /// query row: support curve from the frozen nodes/masses/band
    /// ([`crate::basis::measure_jet_support_curve`]), fitted per-scale
    /// amplitudes λ̂_ℓ read from the fit's `lambdas` through the replayed
    /// design's penalty layout, folded through
    /// [`crate::basis::measure_jet_extrapolation_variance`] and scaled by
    /// the fit's coefficient-covariance scale φ̂ so the result sits on Vp's
    /// η-variance scale. Terms not yet frozen (no `frozen_quadrature` or
    /// non-`UserProvided` centers) are skipped with a warning. Returns
    /// `Ok(None)` when no measure-jet term contributes, so callers leave
    /// `PredictUncertaintyOptions::extrapolation_variance` untouched.
    ///
    /// `data` must be the RAW (unclipped) prediction matrix in prediction
    /// column order — clipping to the training ranges would freeze the
    /// distance signal at the hull and defeat the honesty contract — and
    /// `col_map` the prediction header → column map (the same map handed to
    /// the design builder). This is the minimal-plumbing producer seam: the
    /// option-building callers (CLI predict, FFI) hold exactly
    /// `(model, data, col_map)` at the point where they assemble
    /// `PredictUncertaintyOptions`, and the fusion in
    /// `predict_gamwith_uncertainty` adds the array AFTER its multiplicative
    /// inflations: `Var_total = Var_Vp·inflation + Var_extrap`.
    pub fn measure_jet_extrapolation_variance(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        col_map: &HashMap<String, usize>,
    ) -> Result<Option<Array1<f64>>, FittedModelError> {
        use crate::basis::{CenterStrategy, MeasureJetExtrapolationSpectrum, PenaltySource};
        use crate::smooth::{SmoothBasisSpec, build_term_collection_design};
        let Some(saved_spec) = self.resolved_termspec.as_ref() else {
            return Ok(None);
        };
        if data.nrows() == 0
            || !saved_spec
                .smooth_terms
                .iter()
                .any(|t| matches!(t.basis, SmoothBasisSpec::MeasureJet { .. }))
        {
            return Ok(None);
        }
        let fit = self
            .fit_result
            .as_ref()
            .ok_or_else(|| FittedModelError::MissingField {
                reason: "measure-jet extrapolation variance requires the canonical \
                    fit_result payload; refit"
                    .to_string(),
            })?;
        let spec = crate::families::survival_predict::resolve_termspec_for_prediction(
            &self.resolved_termspec,
            self.training_headers.as_ref(),
            col_map,
            "resolved_termspec",
        )
        .map_err(|e| FittedModelError::SchemaMismatch {
            reason: format!("measure-jet extrapolation variance: {e}"),
        })?;
        // Penalty layout replay: the global penalty indices (→ `fit.lambdas`)
        // come from the SAME design builder the predict pipeline uses. One
        // probe row suffices — for a frozen spec the penalty layout is
        // row-count-invariant (centers, masses, band, and identifiability
        // transforms all replay verbatim) — keeping this O(centers²) instead
        // of duplicating the full O(rows·centers) prediction design build.
        let probe = data.slice(ndarray::s![0..1, ..]);
        let design = build_term_collection_design(probe, &spec).map_err(|e| {
            FittedModelError::SchemaMismatch {
                reason: format!(
                    "measure-jet extrapolation variance: penalty-layout replay failed: {e}"
                ),
            }
        })?;
        let lambdas = &fit.lambdas;
        // λ̂ are fitted on Frobenius-normalized penalties. The term loop
        // unnormalizes them to physical precisions before pricing; multiplying
        // by the coefficient-covariance scale puts Var_extrap on the same
        // η-variance scale as Vp.
        let phi_scale = fit.coefficient_covariance_scale();
        let mut total = Array1::<f64>::zeros(data.nrows());
        let mut contributed = false;
        for term in &spec.smooth_terms {
            let SmoothBasisSpec::MeasureJet {
                feature_cols,
                spec: mj,
                input_scales,
            } = &term.basis
            else {
                continue;
            };
            let (Some(frozen), CenterStrategy::UserProvided(centers)) =
                (mj.frozen_quadrature.as_ref(), &mj.center_strategy)
            else {
                log::warn!(
                    "measure-jet term '{}' is not frozen (UserProvided centers + frozen \
                    quadrature); skipping its extrapolation variance",
                    term.name
                );
                continue;
            };
            let n_levels = frozen.eps_band.len();
            // λ̂ per level from the replayed layout: per-scale candidates carry
            // `PenaltySource::Other("measure_jet_scale_ℓ")`; fused
            // (pinned-order) mode carries one Primary charged once for the
            // whole band. The DoublePenaltyNullspace ridge is EXCLUDED — it shrinks
            // coefficients, it is not a scale amplitude, and counting it would
            // double-charge the spectrum.
            let read_lambda = |global_index: usize| -> Result<f64, FittedModelError> {
                lambdas
                    .get(global_index)
                    .copied()
                    .ok_or_else(|| FittedModelError::SchemaMismatch {
                        reason: format!(
                            "measure-jet term '{}': penalty global index {global_index} out \
                            of bounds for {} fitted lambdas",
                            term.name,
                            lambdas.len()
                        ),
                    })
            };
            let mut per_scale: Vec<(usize, f64)> = Vec::new();
            let mut fused: Option<f64> = None;
            for info in &design.penaltyinfo {
                if info.termname.as_deref() != Some(term.name.as_str()) {
                    continue;
                }
                match &info.penalty.source {
                    PenaltySource::Other(label) => {
                        if let Some(level_txt) = label.strip_prefix("measure_jet_scale_") {
                            let level: usize = level_txt.parse().map_err(|_| {
                                FittedModelError::SchemaMismatch {
                                    reason: format!(
                                        "measure-jet term '{}': unparseable penalty label \
                                        '{label}'",
                                        term.name
                                    ),
                                }
                            })?;
                            per_scale.push((level, read_lambda(info.global_index)?));
                        }
                    }
                    PenaltySource::Primary => {
                        fused = Some(read_lambda(info.global_index)?);
                    }
                    _ => {}
                }
            }
            let mut lambda_phys = Vec::with_capacity(n_levels);
            let spectrum = if per_scale.is_empty() {
                let Some(lam) = fused else {
                    log::warn!(
                        "measure-jet term '{}' has no fitted amplitude in the penalty \
                        layout; skipping its extrapolation variance",
                        term.name
                    );
                    continue;
                };
                let Some(c) = frozen.fused_penalty_normalization_scale else {
                    log::warn!(
                        "measure-jet term '{}' is missing the fused penalty normalization scale; \
                        skipping its extrapolation variance",
                        term.name
                    );
                    continue;
                };
                MeasureJetExtrapolationSpectrum::Fused(lam / c)
            } else {
                per_scale.sort_by_key(|&(level, _)| level);
                let levels_complete = per_scale.len() == n_levels
                    && per_scale
                        .iter()
                        .enumerate()
                        .all(|(i, &(level, _))| level == i);
                if !levels_complete {
                    log::warn!(
                        "measure-jet term '{}': {} fitted per-scale amplitudes for {} band \
                        scales; skipping its extrapolation variance",
                        term.name,
                        per_scale.len(),
                        n_levels
                    );
                    continue;
                }
                if frozen.penalty_normalization_scales.len() != n_levels {
                    log::warn!(
                        "measure-jet term '{}': {} frozen penalty normalization scales for {} \
                        band scales; skipping its extrapolation variance",
                        term.name,
                        frozen.penalty_normalization_scales.len(),
                        n_levels
                    );
                    continue;
                }
                lambda_phys.extend(
                    per_scale
                        .iter()
                        .map(|&(level, lam)| lam / frozen.penalty_normalization_scales[level]),
                );
                MeasureJetExtrapolationSpectrum::PerLevel(&lambda_phys)
            };
            // Query rows in the frozen geometry's coordinates: select the
            // term's axes and replay the per-axis standardization exactly as
            // the build dispatch does (divide by σ_a when input_scales is
            // Some; the persisted centers are already post-standardization).
            let mut queries = Array2::<f64>::zeros((data.nrows(), feature_cols.len()));
            for (j, &col) in feature_cols.iter().enumerate() {
                if col >= data.ncols() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: format!(
                            "measure-jet term '{}': prediction column {col} out of bounds \
                            for {} data columns",
                            term.name,
                            data.ncols()
                        ),
                    });
                }
                queries.column_mut(j).assign(&data.column(col));
            }
            if let Some(scales) = input_scales {
                if scales.len() != feature_cols.len() {
                    return Err(FittedModelError::SchemaMismatch {
                        reason: format!(
                            "measure-jet term '{}': {} input scales for {} axes",
                            term.name,
                            scales.len(),
                            feature_cols.len()
                        ),
                    });
                }
                for (j, &scale) in scales.iter().enumerate() {
                    queries.column_mut(j).mapv_inplace(|v| v / scale);
                }
            }
            let support = crate::basis::measure_jet_support_curve(
                queries.view(),
                centers.view(),
                frozen.masses.view(),
                &frozen.eps_band,
            )
            .map_err(|e| FittedModelError::SchemaMismatch {
                reason: format!(
                    "measure-jet term '{}': support curve failed: {e}",
                    term.name
                ),
            })?;
            for i in 0..data.nrows() {
                let v = crate::basis::measure_jet_extrapolation_variance(
                    support.row(i),
                    &frozen.eps_band,
                    &frozen.support_means,
                    spectrum,
                    Self::MEASURE_JET_COVERAGE_FLOOR,
                )
                .map_err(|e| FittedModelError::SchemaMismatch {
                    reason: format!(
                        "measure-jet term '{}': extrapolation variance failed: {e}",
                        term.name
                    ),
                })?;
                total[i] += phi_scale * v;
            }
            contributed = true;
        }
        Ok(contributed.then_some(total))
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
        // Write to a sibling temp file, fsync, then rename into place so a
        // crash mid-write never corrupts the user's existing saved fit.
        // Concurrent writers to the same path each have a distinct temp
        // suffix (pid + nanos), so neither stomps the other's in-flight
        // bytes; the rename winner is last-rename-wins, which is the
        // expected last-write-wins semantics for a single canonical path.
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.json");
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = parent.join(format!(".{file_name}.tmp.{pid}.{nanos:x}"));
        let file = fs::File::create(&tmp).map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("failed to write model '{}': {e}", tmp.display()),
        })?;
        let mut writer = std::io::BufWriter::new(file);
        let ser_result = serde_json::to_writer(&mut writer, &normalized);
        if let Err(e) = ser_result {
            // Best-effort temp cleanup on serialization failure. flush
            // returns io::Result<()>; discarding via `.ok()` is enough.
            std::io::Write::flush(&mut writer).ok();
            drop(writer);
            fs::remove_file(&tmp).ok();
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!("failed to serialize model: {e}"),
            });
        }
        std::io::Write::flush(&mut writer).map_err(|e| FittedModelError::PayloadCorrupt {
            reason: format!("failed to write model '{}': {e}", tmp.display()),
        })?;
        // Recover the underlying File to fsync its contents before rename.
        let inner = writer
            .into_inner()
            .map_err(|e| FittedModelError::PayloadCorrupt {
                reason: format!("failed to flush model '{}': {}", tmp.display(), e.error()),
            })?;
        inner.sync_all().ok();
        drop(inner);
        if let Err(e) = fs::rename(&tmp, path) {
            fs::remove_file(&tmp).ok();
            return Err(FittedModelError::PayloadCorrupt {
                reason: format!("failed to publish model '{}': {e}", path.display()),
            });
        }
        // fsync the parent directory so the rename itself is durable
        // across a crash; without this, the rename can be lost even though
        // file contents reached disk. Best-effort on platforms that don't
        // support opening a directory for fsync.
        if let Ok(d) = fs::File::open(parent) {
            d.sync_all().ok();
        }
        Ok(())
    }

    pub fn require_data_schema(&self) -> Result<&DataSchema, FittedModelError> {
        self.data_schema
            .as_ref()
            .ok_or_else(|| FittedModelError::MissingField {
                reason: "model is missing data_schema; refit".to_string(),
            })
    }

    /// Restore the exact in-memory spline-scan fit from a scan-bearing
    /// payload (#1030/#1034). `Ok(None)` for dense models; the returned
    /// `predict` replays the training Gaussian bridge bit-for-bit.
    pub fn saved_spline_scan(
        &self,
    ) -> Result<Option<(&str, crate::solver::spline_scan::SplineScanFit)>, FittedModelError> {
        let Some(saved) = self.spline_scan.as_ref() else {
            return Ok(None);
        };
        let fit = crate::solver::spline_scan::SplineScanFit::from_state(&saved.state)
            .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
        Ok(Some((saved.feature_column.as_str(), fit)))
    }

    /// Restore the in-memory residual-cascade fit from a cascade-bearing
    /// payload (#1032). `Ok(None)` for non-cascade models; the returned fit
    /// replays the multilevel Wendland-frame posterior for the d ∈ {2, 3}
    /// feature columns at each predict point.
    pub fn saved_residual_cascade(
        &self,
    ) -> Result<
        Option<(
            &[String],
            crate::solver::residual_cascade::ResidualCascadeFit,
        )>,
        FittedModelError,
    > {
        let Some(saved) = self.residual_cascade.as_ref() else {
            return Ok(None);
        };
        let fit = crate::solver::residual_cascade::ResidualCascadeFit::from_state(&saved.state)
            .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
        Ok(Some((saved.feature_columns.as_slice(), fit)))
    }

    pub fn random_effect_group_columns(&self) -> HashSet<String> {
        let Some(training_headers) = self.training_headers.as_ref() else {
            return HashSet::new();
        };
        let mut out = HashSet::<String>::new();
        for spec in self.saved_term_specs() {
            for term in &spec.random_effect_terms {
                if let Some(name) = training_headers.get(term.feature_col) {
                    out.insert(name.clone());
                }
            }
        }
        out
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
        if let Some(scan) = self.spline_scan.as_ref() {
            // Spline-scan representation (#1030/#1034): the smoother state IS
            // the fit. It is exclusive with the dense representation, only
            // standard Gaussian-identity models can carry it, and the state
            // must restore cleanly so predict never sees a corrupt snapshot.
            if self.fit_result.is_some() || self.unified.is_some() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "spline-scan model must not also carry a dense fit_result/unified \
                             payload; the representations are mutually exclusive"
                        .to_string(),
                });
            }
            if self.model_kind != ModelKind::Standard
                || self.family_state.likelihood() != LikelihoodSpec::gaussian_identity()
            {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "spline-scan representation requires a standard Gaussian-identity model; \
                         got model_kind={:?}, likelihood={:?}",
                        self.model_kind,
                        self.family_state.likelihood()
                    ),
                });
            }
            if scan.feature_column.is_empty() {
                return Err(FittedModelError::MissingField {
                    reason: "spline-scan model is missing its feature column name; refit"
                        .to_string(),
                });
            }
            crate::solver::spline_scan::SplineScanFit::from_state(&scan.state)
                .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
            // A scan model carries NO dense design, so the dense-path
            // requirements below (resolved_termspec, fit_result finiteness,
            // family-specific blocks) do not apply. Enforce only the metadata
            // predict actually consumes — the feature column resolves against
            // training_headers / data_schema — then accept.
            if self.data_schema.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "spline-scan model is missing data_schema; refit".to_string(),
                });
            }
            if self.training_headers.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "spline-scan model is missing training_headers; refit".to_string(),
                });
            }
            return Ok(());
        } else if let Some(cascade) = self.residual_cascade.as_ref() {
            // Residual-cascade representation (#1032): a multilevel
            // Wendland-frame model for a scattered d ∈ {2,3} Gaussian smooth.
            // Exclusive with the dense representation and with the scan.
            if self.spline_scan.is_some() || self.fit_result.is_some() || self.unified.is_some() {
                return Err(FittedModelError::SchemaMismatch {
                    reason: "residual-cascade model must not also carry spline_scan / \
                             fit_result / unified payloads; the representations are \
                             mutually exclusive"
                        .to_string(),
                });
            }
            if self.model_kind != ModelKind::Standard
                || self.family_state.likelihood() != LikelihoodSpec::gaussian_identity()
            {
                return Err(FittedModelError::SchemaMismatch {
                    reason: format!(
                        "residual-cascade representation requires a standard Gaussian-identity \
                         model; got model_kind={:?}, likelihood={:?}",
                        self.model_kind,
                        self.family_state.likelihood()
                    ),
                });
            }
            if cascade.feature_columns.is_empty()
                || !(2..=3).contains(&cascade.feature_columns.len())
            {
                return Err(FittedModelError::MissingField {
                    reason: format!(
                        "residual-cascade model needs 2 or 3 feature columns; got {}; refit",
                        cascade.feature_columns.len()
                    ),
                });
            }
            crate::solver::residual_cascade::ResidualCascadeFit::from_state(&cascade.state)
                .map_err(|reason| FittedModelError::PayloadCorrupt { reason })?;
            if self.data_schema.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "residual-cascade model is missing data_schema; refit".to_string(),
                });
            }
            if self.training_headers.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "residual-cascade model is missing training_headers; refit".to_string(),
                });
            }
            return Ok(());
        } else if self.fit_result.is_none() {
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
            // Non-latent survival predict reconstructs the baseline-time
            // basis via `load_survival_time_basis_config_from_model` and
            // anchors that basis at `survival_time_anchor`; both are
            // required for the saved model to be loadable. The CLI's
            // marginal-slope+time-wiggle save path previously dropped one or
            // the other on partial-write, producing models that loaded but
            // would panic at the first predict. Enforce both before persisting.
            if self.survival_time_basis.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "survival model is missing survival_time_basis; refit to persist the baseline-time basis configuration".to_string(),
                });
            }
            if self.survival_time_anchor.is_none() {
                return Err(FittedModelError::MissingField {
                    reason: "survival model is missing survival_time_anchor; refit to persist the baseline-time anchor".to_string(),
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

        let family_likelihood = match &self.family_state {
            FittedFamily::Standard { likelihood, .. }
            | FittedFamily::LocationScale { likelihood, .. }
            | FittedFamily::MarginalSlope { likelihood, .. }
            | FittedFamily::Survival { likelihood, .. }
            | FittedFamily::TransformationNormal { likelihood, .. } => Some(likelihood),
            FittedFamily::LatentSurvival { .. } | FittedFamily::LatentBinary { .. } => None,
        };
        let is_standard_or_location_scale = matches!(
            self.family_state,
            FittedFamily::Standard { .. } | FittedFamily::LocationScale { .. }
        );
        if is_standard_or_location_scale
            && family_likelihood.is_some_and(LikelihoodSpec::is_binomial_sas)
        {
            self.saved_sas_state()?;
        }
        if is_standard_or_location_scale
            && family_likelihood.is_some_and(LikelihoodSpec::is_binomial_beta_logistic)
        {
            self.saved_beta_logistic_state()?;
        }
        if is_standard_or_location_scale
            && family_likelihood.is_some_and(LikelihoodSpec::is_binomial_mixture)
        {
            self.saved_mixture_state()?;
        }
        if matches!(self.family_state, FittedFamily::Standard { .. })
            && family_likelihood.is_some_and(LikelihoodSpec::is_latent_cloglog)
        {
            self.saved_latent_cloglog_state()?;
        }
        if matches!(self.family_state, FittedFamily::LocationScale { .. })
            && family_likelihood.is_some_and(LikelihoodSpec::is_latent_cloglog)
        {
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
            if self.noise_projection_ridge_alpha.is_none() {
                return Err(FittedModelError::MissingField {
                    reason:
                        "model has noise_projection but is missing noise_projection_ridge_alpha; refit"
                            .to_string(),
                });
            }
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
            if self.survival_noise_projection_ridge_alpha.is_none() {
                return Err(FittedModelError::MissingField {
                    reason:
                        "model has survival_noise_projection but is missing survival_noise_projection_ridge_alpha; refit"
                            .to_string(),
                });
            }
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
    let target = model.survival_baseline_target.as_deref().ok_or_else(|| {
        FittedModelError::MissingField {
            reason: "saved survival model missing survival_baseline_target; refit".to_string(),
        }
    })?;
    parse_survival_baseline_config(
        target,
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
    use crate::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL;
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

    /// #1030/#1034: a scan-bearing payload must round-trip through JSON +
    /// `validate_for_persistence` and replay the training Gaussian bridge
    /// bit-for-bit; structural corruption must fail loudly at validation.
    #[test]
    fn spline_scan_payload_round_trips_and_validates() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| (4.0 * v).sin() + 0.1 * v).collect();
        let w = vec![1.0_f64; x.len()];
        let fit = crate::solver::spline_scan::fit_spline_scan(&x, &y, &w, 2).expect("scan fit");
        let make_payload = || {
            crate::inference::model_payload_builders::assemble_spline_scan_payload(
                "y ~ s(x)".to_string(),
                "x".to_string(),
                &fit,
                DataSchema {
                    columns: vec![
                        SchemaColumn {
                            name: "y".to_string(),
                            kind: ColumnKindTag::Continuous,
                            levels: vec![],
                        },
                        SchemaColumn {
                            name: "x".to_string(),
                            kind: ColumnKindTag::Continuous,
                            levels: vec![],
                        },
                    ],
                },
                vec!["x".to_string()],
                vec![(0.0, 1.0)],
            )
        };
        // The on-disk form is the FittedModel tagged enum; validation and the
        // scan accessor live on FittedModel (Deref only goes Model -> Payload).
        let model = FittedModel::from_payload(make_payload());
        model
            .validate_for_persistence()
            .expect("scan model validates");
        model
            .validate_numeric_finiteness()
            .expect("scan model is finite");

        let json = serde_json::to_string(&model).expect("serialize model");
        let restored: FittedModel = serde_json::from_str(&json).expect("parse model");
        restored
            .validate_for_persistence()
            .expect("restored scan model validates");
        let (column, replay) = restored
            .saved_spline_scan()
            .expect("restore scan fit")
            .expect("payload carries the scan representation");
        assert_eq!(column, "x");
        for &xq in &[-0.1, 0.0, 0.31, 0.5, 0.77, 1.0, 1.4] {
            let (m0, v0) = fit.predict(xq).expect("predict original");
            let (m1, v1) = replay.predict(xq).expect("predict replayed");
            assert_eq!(m0.to_bits(), m1.to_bits(), "mean drift at x={xq}");
            assert_eq!(v0.to_bits(), v1.to_bits(), "variance drift at x={xq}");
        }

        // A dense model without the scan channel still requires fit_result.
        let mut dense = make_payload();
        dense.spline_scan = None;
        let err = FittedModel::from_payload(dense)
            .validate_for_persistence()
            .expect_err("dense payload without fit_result must be rejected");
        assert!(err.to_string().contains("fit_result"));

        // Structural corruption fails at validation, not inside predict.
        let mut corrupt = make_payload();
        corrupt
            .spline_scan
            .as_mut()
            .expect("scan channel present")
            .state
            .knots
            .truncate(2);
        FittedModel::from_payload(corrupt)
            .validate_for_persistence()
            .expect_err("corrupt scan state must be rejected");
        let mut unnamed = make_payload();
        unnamed
            .spline_scan
            .as_mut()
            .expect("scan channel present")
            .feature_column
            .clear();
        FittedModel::from_payload(unnamed)
            .validate_for_persistence()
            .expect_err("missing feature column must be rejected");
    }

    fn standard_gaussian_payload() -> FittedModelPayload {
        FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            "y ~ 1".to_string(),
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: LikelihoodSpec::gaussian_identity(),
                link: Some(StandardLink::Identity),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            "gaussian".to_string(),
        )
    }

    fn anchored_runtime(basis_dim: usize) -> SavedCompiledFlexBlock {
        SavedCompiledFlexBlock {
            kernel: ANCHORED_DEVIATION_KERNEL.to_string(),
            breakpoints: vec![-1.0, 1.0],
            basis_dim,
            span_c0: vec![vec![0.0; basis_dim]],
            span_c1: vec![vec![0.0; basis_dim]],
            span_c2: vec![vec![0.0; basis_dim]],
            span_c3: vec![vec![0.0; basis_dim]],
            anchor_correction: None,
            anchor_components: Vec::new(),
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
            likelihood_family: Some(LikelihoodSpec::binomial_probit()),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
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
                null_space_logdet: None,
                null_space_dim: None,
                survival_link_wiggle_knots: None,
                survival_link_wiggle_degree: None,
                criterion_certificate: None,
                rho_posterior_certificate: None,
                rho_posterior_escalation: None,
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
                likelihood: LikelihoodSpec::binomial_probit(),
                base_link: Some(InverseLink::Standard(StandardLink::Probit)),
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
        payload.link = Some(InverseLink::Standard(StandardLink::Probit));
        payload
    }

    fn survival_marginal_slope_payload(version: u32, fit: UnifiedFitResult) -> FittedModelPayload {
        let mut payload = FittedModelPayload::new(
            version,
            "Surv(entry, exit, event) ~ 1".to_string(),
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodSpec::royston_parmar(),
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
        payload.link = Some(InverseLink::Standard(StandardLink::Probit));
        payload
    }

    fn gamma_dispersion_location_scale_payload() -> FittedModelPayload {
        // A #913 genuine-dispersion location-scale model: Gamma mean family with
        // a log-precision `noise_formula` channel. Its likelihood response is
        // non-Gaussian and non-Binomial, so the predict-path classifier must
        // route it to `DispersionLocationScale`, NOT the binomial threshold-scale
        // class (issue #1064).
        let mut payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            "y ~ x".to_string(),
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodSpec::gamma_log(),
                base_link: Some(InverseLink::Standard(StandardLink::Log)),
            },
            "gamma-location-scale".to_string(),
        );
        payload.data_schema = Some(DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        });
        payload.set_training_feature_metadata(vec!["x".to_string()], vec![(-1.0, 1.0)]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_noise = Some(empty_termspec());
        payload.formula_noise = Some("x".to_string());
        payload.beta_noise = Some(vec![0.0]);
        payload.link = Some(InverseLink::Standard(StandardLink::Log));
        payload
    }

    /// #1064 regression: a dispersion location-scale (#913) payload must be
    /// classified as `DispersionLocationScale` at every predict-path entry —
    /// both `from_payload` (load) and `predict_model_class` (runtime) — and never
    /// fall through to the binomial threshold-scale class. Before the fix the
    /// non-Gaussian `else` arm mis-routed every dispersion model to
    /// `BinomialLocationScale`, predicting the wrong family/link.
    #[test]
    fn dispersion_location_scale_payload_is_not_classified_binomial() {
        let model = FittedModel::from_payload(gamma_dispersion_location_scale_payload());
        assert_eq!(
            model.predict_model_class(),
            PredictModelClass::DispersionLocationScale,
            "Gamma dispersion location-scale must route through the dispersion \
             predictor, not the binomial threshold-scale class",
        );
        assert!(
            !matches!(
                model.predict_model_class(),
                PredictModelClass::BinomialLocationScale
            ),
            "dispersion location-scale must never be classified as binomial",
        );

        // Each of the four #913 dispersion mean families classifies the same way.
        for likelihood in [
            LikelihoodSpec::gamma_log(),
            LikelihoodSpec::new(
                ResponseFamily::NegativeBinomial {
                    theta: 1.0,
                    theta_fixed: false,
                },
                InverseLink::Standard(StandardLink::Log),
            ),
            LikelihoodSpec::new(
                ResponseFamily::Beta { phi: 1.0 },
                InverseLink::Standard(StandardLink::Logit),
            ),
            LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: 1.5 },
                InverseLink::Standard(StandardLink::Log),
            ),
        ] {
            let mut payload = gamma_dispersion_location_scale_payload();
            payload.family_state = FittedFamily::LocationScale {
                base_link: Some(likelihood.link.clone()),
                likelihood: likelihood.clone(),
            };
            let model = FittedModel::from_payload(payload);
            assert_eq!(
                model.predict_model_class(),
                PredictModelClass::DispersionLocationScale,
                "dispersion family {:?} mis-classified",
                likelihood.response,
            );
        }
    }

    #[test]
    fn axis_clip_leaves_numeric_random_effect_group_axis_unclipped() {
        let data = array![[100.0], [-100.0]];
        let col_map = HashMap::from([("g".to_string(), 0usize)]);

        let mut plain_payload = standard_gaussian_payload();
        plain_payload.data_schema = Some(DataSchema {
            columns: vec![SchemaColumn {
                name: "g".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        });
        plain_payload.set_training_feature_metadata(vec!["g".to_string()], vec![(0.0, 7.0)]);
        plain_payload.resolved_termspec = Some(empty_termspec());
        let plain = FittedModel::from_payload(plain_payload.clone());
        let clipped = plain
            .axis_clip_to_training_ranges(data.view(), &col_map)
            .expect("ordinary continuous axis should clip outside the training range");
        assert_eq!(clipped.column(0).to_vec(), vec![7.0, 0.0]);

        let mut group_payload = plain_payload;
        let mut group_spec = empty_termspec();
        group_spec
            .random_effect_terms
            .push(crate::smooth::RandomEffectTermSpec {
                name: "g".to_string(),
                feature_col: 0,
                drop_first_level: false,
                penalized: true,
                frozen_levels: Some(vec![0.0_f64.to_bits(), 7.0_f64.to_bits()]),
            });
        group_payload.resolved_termspec = Some(group_spec);
        let group_model = FittedModel::from_payload(group_payload);

        assert_eq!(
            group_model.random_effect_group_columns(),
            HashSet::from(["g".to_string()])
        );

        assert_eq!(
            group_model.axis_clip_to_training_ranges(data.view(), &col_map),
            None,
            "numeric group labels must reach RandomEffectOperator as unseen levels, not be clipped to boundary seen levels"
        );
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
        // forgetting one is exactly the marginal-slope save
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
